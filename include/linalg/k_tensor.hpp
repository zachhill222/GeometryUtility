#pragma once

#include "utility/utility.hpp"
#include "math/math.hpp"

#include <vector>
#include <array>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <cstdlib>

namespace gutil {
	
	
	/////////////////////////////////////////////////////////////
	/// A small struct to store tensors.
	/// These are dense and stored equivalent to column major (K=2).
	/// The order of the tensor must be known at compile time, but the
	/// dimensions can be set at runtime, if the tensor owns the storage
	/// resource.
	///
	/// It is the user's responsibility to ensure access is valid.
	/// A wrapper over this base class may be helpful.
	/////////////////////////////////////////////////////////////
	template<typename T, size_t K, bool IsConst>
	class RawTensorBase {
	public:
		//adapt to common interfaces
		using value_type = T;
		using pointer_type = std::conditional_t<IsConst, const T*, T*>;


		//access data as a flat array
		[[nodiscard]] const T* data() const noexcept {return ptr;}
		[[nodiscard]] T* data() noexcept requires(!IsConst) {return ptr;}

		[[nodiscard]] const T* begin() const noexcept {return ptr;}
		[[nodiscard]] const T* end() const noexcept {return ptr ? ptr+count() : nullptr; }
		[[nodiscard]] T* begin() noexcept requires(!IsConst) {return ptr;}
		[[nodiscard]] T* end() noexcept requires(!IsConst) {return ptr ? ptr+count() : nullptr; }


		//constructor and move (delete copy)
		constexpr RawTensorBase() noexcept : ptr{nullptr}, dim_{} {}
		RawTensorBase(pointer_type pointer) : ptr{pointer} {}
		RawTensorBase(std::array<size_t,K> dims) : ptr{nullptr}, dim_{std::move(dims)} {}
		RawTensorBase(pointer_type pointer, std::array<size_t,K> dims) : ptr{pointer}, dim_{std::move(dims)} {}
		RawTensorBase(RawTensorBase&& other) : ptr{other.ptr}, dim_{std::move(other.dim_)} {other.ptr=nullptr;}
		RawTensorBase& operator=(RawTensorBase&& other) noexcept {
			if (this != &other) {
				ptr  = other.ptr;
				dim_ = std::move(other.dim_);
				other.ptr = nullptr;
			}
			return *this;
		}
		RawTensorBase(const RawTensorBase&) = delete; //this is likely to cause bugs
		RawTensorBase& operator=(const RawTensorBase&) = delete;

		//a few utility methods
		[[nodiscard]] constexpr size_t count() const noexcept {
			size_t N=1;
			for (size_t len : dim_) { N*=len; }
			return N;
		}

		[[nodiscard]] constexpr const std::array<size_t,K>& dim() const noexcept {return dim_;}
		[[nodiscard]] constexpr const size_t dim(size_t k) const noexcept {GUTIL_ASSERT(k<K); return dim_[k];}


		////////////////////////////////////////////////////////////////////////////
		/// Core static methods. Users may want to just use these on raw data buffers.
		////////////////////////////////////////////////////////////////////////////
		static size_t FlatIndex(std::array<size_t,K> idx, const std::array<size_t,K>& dims) noexcept {
			//Get the flat "column - major" index
			#ifndef NDEBUG
				for (size_t k=0; k<K; ++k) {GUTIL_ASSERT(idx[k]<dims[k]);}
			#endif
			if constexpr (K==0) {return 0;}
			else if constexpr (K==1) {return idx[0];}
			else if constexpr (K==2) {return idx[0] + dims[0]*idx[1];}
			else if constexpr (K==3) {return idx[0] + dims[0]*(idx[1] + dims[1]*idx[2]);}
			else {
				size_t flat = 0, stride = 1;
				for (size_t k=0; k<K; ++k) {
					flat   += idx[k]*stride;
					stride *= dims[k];
				}
				return flat;
			}
		}

		static void FlatToTensorIndex(size_t flat, std::array<size_t,K>& idx, const std::array<size_t,K>& dims) noexcept {
			GUTIL_ASSERT(flat < gutil::product_reduce(dims));
			if constexpr (K==1) {idx[0]=flat;}
			else if constexpr (K==2) {
				const size_t q = flat / dims[0];
				idx[0] = flat - q*dims[0];
				idx[1] = q;
			}
			else {
				for (size_t axis=0; axis<K; ++axis) {
					const size_t q = flat / dims[axis];
					idx[axis] = flat - q*dims[axis];
					flat = q;
				}
			}
		}

		[[nodiscard]] static constexpr size_t AxisStride(size_t axis, const std::array<size_t,K>& dims) noexcept {
			if constexpr (K==0) {return 0;}
			if constexpr (K==1) {return 1;}
			if constexpr (K==2) {return (axis==0) ? 1 : dims[0];}
			else {
				size_t stride = 1;
				GUTIL_SIMD(reduction(*:stride))
				for (size_t a=0; a<axis; ++a) {stride *= dims[a];}
				return stride;
			}
		}

		//applies op to every entry in the fixed-i slice along the specified axis (the K-1 dim tensor).
		//this can be used to do various tensor operations
		template<bool Simd=false, typename Op>
		static void ApplyAlongAxisIndex(pointer_type data_ptr, size_t axis, size_t i, Op&& op, const std::array<size_t,K>& dims) noexcept {
			// GUTIL_PROFILE_FUNCTION(Simd);
			size_t stride = 1;
			for (size_t i=0; i<axis; ++i) {
				stride *= dims[i];
			}

			size_t total = stride;
			for (size_t i=axis; i<K; ++i) {
				total *= dims[i];
			}

			const size_t dim_axis = dims[axis];
			const size_t outer_count = total / (stride * dim_axis);
			const size_t offset = i*stride;
			const size_t outer_stride = stride*dim_axis;

			//note (axis,i) partions the flat index into a low/fast and a high/slow parts
			// ...fast axes..., specified axis=i, ...slow axes...
			//outer looper over the slow axes while inner loops over the fast.
			GUTIL_SIMD(collapse(2) if(Simd))
			for (size_t outer=0; outer<outer_count; ++outer) {
				for (size_t inner=0; inner<stride; ++inner) {
					op(data_ptr[outer*outer_stride + offset + inner]);
				}
			}
		}

		//conjugate of apply_along_axis_index: fixes a single logical (flat, (K-1)-dim)
		//position "slice" and walks i across the axis, rather than fixing i and walking
		//the logical space. "slice" is also exactly the destination position in a
		//compacted, axis-reduced result -- see contract_axis.
		template<bool Simd=false, typename Op>
		static void ApplyAlongSlice(pointer_type data_ptr, size_t axis, size_t slice, Op&& op, const std::array<size_t,K>& dims) noexcept {
			GUTIL_PROFILE_FUNCTION(Simd);
			const size_t stride = AxisStride(axis, dims);
			const size_t dim_axis = dims[axis];
			const size_t outer = slice / stride;
			const size_t inner = slice % stride;
			const size_t base = outer*(stride*dim_axis) + inner;
			GUTIL_SIMD(if(Simd))
			for (size_t i=0; i<dim_axis; ++i) {
				op(data_ptr[base + i*stride]);
			}
		}


		[[nodiscard]] constexpr size_t flat_index(std::array<size_t,K> idx) const noexcept {
			return FlatIndex(std::move(idx), dim_);
		}

		//invert the index (flat to tensor/array index)
		void flat_to_tensor_index(size_t flat, std::array<size_t,K>& idx) const noexcept {
			FlatToTensorIndex(flat, idx, dim_);
		}

		//a few helper functions to compute strides and starts of axis slices.
		//e.g., fixing all indices but one, tensor(..., i, ...) is a vector with a start and stride.
		//note incrementing the flat index has a fast (left dots) and slow (right dots) blocks of indices.
		//keeping i fixed, the remaining indices form a K-1 tensor.
		[[nodiscard]] constexpr size_t axis_stride(size_t axis) const noexcept {
			return AxisStride(axis, dim_);
		}

		//applies op to every entry in the fixed-i slice along the specified axis (the K-1 dim tensor).
		//this can be used to do various tensor operations
		template<bool Simd=false, typename Op>
		void apply_along_axis_index(size_t axis, size_t i, Op&& op) const noexcept {
			ApplyAlongAxisIndex<false>(ptr, axis, i, std::forward<Op>(op), dim_);
		}

		template<typename Op>
		void apply_along_axis_index_simd(size_t axis, size_t i, Op&& op) const noexcept {
			ApplyAlongAxisIndex<true>(ptr, axis, i, std::forward<Op>(op), dim_);
		}

		//conjugate of apply_along_axis_index: fixes a single logical (flat, (K-1)-dim)
		//position "slice" and walks i across the axis, rather than fixing i and walking
		//the logical space. "slice" is also exactly the destination position in a
		//compacted, axis-reduced result -- see contract_axis.
		template<typename Op>
		void apply_along_slice(size_t axis, size_t slice, Op&& op) const noexcept {
			ApplyAlongSlice<false>(ptr, axis, slice, std::forward<Op>(op), dim_);
		}

		template<typename Op>
		void apply_along_slice_simd(size_t axis, size_t slice, Op&& op) const noexcept {
			ApplyAlongSlice<true>(ptr, axis, slice, std::forward<Op>(op), dim_);
		}

		//tensor-vector product along an axis. mathematically, the result is a K-1 tensor,
		//but we keep the K-tensor and set the dim[axis]=1 instead.
		void contract_axis_vector(size_t axis, const T* v) noexcept requires(!IsConst) {
			const size_t dim_axis = dim_[axis];

			//step 1: multiply every entry by its own v[i] -- fix i, walk the logical space
			for (size_t i=0; i<dim_axis; ++i) {
				const T v_i = v[i];
				apply_along_axis_index_simd(axis, i, [v_i](T& val){val *= v_i;});
			}

			//step 2: reduce (sum) across the axis -- fix each logical position, walk i.
			//"slice" is already the correct, compacted destination position.
			const size_t n_slices = count() / dim_axis;
			
			GUTIL_SIMD()
			for (size_t slice=0; slice<n_slices; ++slice) {
				T sum{0};
				apply_along_slice_simd(axis, slice, [&sum](const T& val){sum += val;});
				ptr[slice] = sum;
			}

			dim_[axis] = 1;
		}

		template<typename ArgT, typename ArgsX> requires (std::same_as<T, std::remove_cvref_t<decltype(std::declval<const ArgsX&>()[size_t{0}])>>)
		static T EvaluateKFormConsume(ArgT&& tensor_data, const std::array<ArgsX,K>& x_ptrs, const std::array<size_t,K>& dims) noexcept {
			GUTIL_PROFILE_FUNCTION();
			if constexpr (K==0) {return tensor_data[0];}
			else if constexpr (K==1) {
				T val{0};
				GUTIL_SIMD(reduction(+:val))
				for (size_t i=0; i<dims[0]; ++i) {val += tensor_data[i]*x_ptrs[0][i];}
				return val;
			}
			else if constexpr (K==2) {
				T val{0};
				GUTIL_SIMD(reduction(+:val) collapse(2))
				for (size_t j=0; j<dims[1]; ++j) {
					for (size_t i=0; i<dims[0]; ++i) {
						val += tensor_data[i + dims[0]*j] * x_ptrs[0][i] * x_ptrs[1][j];
					}
				}
				return val;
			}
			else {

				//args should be containers that are convertible to spans.
				T* data_ptr = ToRawPtr(std::forward<ArgT>(tensor_data));
				size_t total = 1;
				for (size_t d : dims) {total *= d;}

				//multiply every entry by its own, per-axis coefficient, one axis at a time,
				for (size_t axis=0; axis<K; ++axis) {
					for (size_t i=0; i<dims[axis]; ++i) {
						const T x_val = x_ptrs[axis][i];
						//true is simd
						ApplyAlongAxisIndex<true>(data_ptr, axis, i, [x_val](T& val){val *= x_val;}, dims);
					}
				}

				//plain, contiguous reduction
				T result{0};
				GUTIL_SIMD(reduction(+:result))
				for (size_t i=0; i<total; ++i) {result += tensor_data[i];}
				return result;
			}
		}

		template<typename ArgT, typename ArgsX> requires (std::same_as<T,typename ArgsX::value_type>)
		static T EvaluateKFormConsume(ArgT&& tensor_data, const std::array<ArgsX,K>& x_ptrs) noexcept {
			std::array<size_t,K> dims;
			for (size_t k=0; k<K; ++k) {dims[k] = x_ptrs[k].size();}
			return EvaluateKFormConsume(std::forward<ArgT>(tensor_data), x_ptrs, std::move(dims));
		}



		//access data by index (the user/derived class must validate the underlying data)
		[[nodiscard]] T& operator()(std::array<size_t,K> idx) noexcept requires(!IsConst) {return ptr[flat_index(std::move(idx))];}
		[[nodiscard]] const T& operator()(std::array<size_t,K> idx) const noexcept {return ptr[flat_index(std::move(idx))];}

		//convenient accessors
		template<typename... Is> requires (sizeof...(Is)==K)
		[[nodiscard]] T& operator()(Is... idxs) noexcept requires(!IsConst) {
			return ptr[flat_index(std::array<size_t,K>{idxs...})];
		}
		template<typename... Is> requires (sizeof...(Is)==K)
		[[nodiscard]] const T& operator()(Is... idxs) const noexcept {
			return ptr[flat_index(std::array<size_t,K>{idxs...})];
		}

	protected:
		//helper for getting raw pointers from containers
		template<typename Arg>
		[[nodiscard]] static constexpr const T* ToRawPtr(const Arg& a) noexcept {
			if constexpr (requires {a.data();}) {return a.data();}
			else {return a;}
		}

		template<typename Arg>
		[[nodiscard]] static constexpr T* ToRawPtr(Arg& a) noexcept {
			if constexpr (requires {a.data();}) {return a.data();}
			else {return a;}
		}

		//data and dimensions
		pointer_type ptr{nullptr};
		std::array<size_t, K> dim_{};
	};


	/////////////////////////////////////////////////////////////////
	/// Useful Tensor types that inherit from the raw type.
	/////////////////////////////////////////////////////////////////
	template<typename T, size_t K>
	class Tensor : public RawTensorBase<T,K,false> {
	public:
		using BASE = RawTensorBase<T,K,false>;

		/////////////////////////////////////////////////////////////
		/// Manage constructors and copy/moves
		/////////////////////////////////////////////////////////////
		Tensor(std::vector<T> v, std::array<size_t,K> dims) : BASE(std::move(dims)), data_(std::move(v)) {
			GUTIL_ASSERT(this->count() == data_.size());
			repoint();
		}
		
		Tensor(std::array<size_t,K> dims) noexcept : BASE(std::move(dims)) {
			resize(this->dim());
		}

		Tensor(const Tensor& other) noexcept : BASE(other.dim()), data_(other.data_) {
			repoint();
		}

		Tensor(Tensor&& other) noexcept : BASE(other.ptr, std::move(other.dim_)), data_(std::move(other.data_)) {
			GUTIL_ASSERT(this->ptr == data_.data());
			other.ptr = nullptr;
		}

		Tensor& operator=(const Tensor& other) noexcept {
			if (this != &other) {
				this->dim_ = other.dim();
				data_ = other.data_;
				repoint();
			}
			return *this;
		}

		Tensor& operator=(Tensor&& other) noexcept {
			if (this != &other) {
				this->dim_ = other.dim();
				data_ = std::move(other.data_);
				other.ptr = nullptr;
				repoint();
			}
			return *this;
		}


		/////////////////////////////////////////////////////////////
		/// Convenient operations
		/////////////////////////////////////////////////////////////
		[[nodiscard]] size_t size() const noexcept {
			GUTIL_ASSERT(data_.size()==this->count());
			return data_.size();
		}

		void resize(std::array<size_t,K> new_sizes) noexcept {
			this->dim_ = new_sizes;
			data_.assign(this->count(), T{0});
			repoint();
		}

		void fill(T val) noexcept {
			std::fill(data_.begin(), data_.end(), val);
		}

	protected:
		std::vector<T> data_{};
		void repoint() noexcept {this->ptr = data_.data();}
	};

	template<typename T, size_t K>
	class TensorWrapper : public RawTensorBase<T,K,false> {
		public:
		using BASE = RawTensorBase<T,K,false>;

		TensorWrapper(std::vector<T>& v) noexcept : BASE(), data_(v) {
			repoint();
		}

		TensorWrapper(std::vector<T>& v, std::array<size_t,K> dims) noexcept : BASE(std::move(dims)), data_(v) {
			data_.resize(this->count());
			repoint();
		}

		TensorWrapper(TensorWrapper&& other) noexcept : BASE(other.ptr, std::move(other.dim_)), data_(other.data_) {
			GUTIL_ASSERT(this->ptr == data_.data());
			other.ptr = nullptr;
		}

		TensorWrapper& operator=(TensorWrapper&& other) noexcept {
			if (this != &other) {
				this->dim_ = other.dim();
				data_ = std::move(other.data_);
				other.ptr = nullptr;
				repoint();
			}
			return *this;
		}

		TensorWrapper(const TensorWrapper& other) = delete;
		TensorWrapper& operator=(const TensorWrapper& other) = delete;


		/////////////////////////////////////////////////////////////
		/// Convenient operations
		/////////////////////////////////////////////////////////////
		[[nodiscard]] size_t size() const noexcept {
			GUTIL_ASSERT(data_.size()==this->count());
			return data_.size();
		}

		void resize(std::array<size_t,K> new_sizes) noexcept {
			this->dim_ = new_sizes;
			data_.resize(this->count());
			repoint();   // re-point in case resize/assign reallocated
		}

		void fill(T val) noexcept {
			std::fill(data_.begin(), data_.end(), val);
		}

		protected:
		std::vector<T>& data_;
		void repoint() noexcept {this->ptr = data_.data();}
	};

	
}