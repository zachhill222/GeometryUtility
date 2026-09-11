#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <type_traits>

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


		//get the flat index "column major"
		[[nodiscard]] constexpr size_t flat_index(std::array<size_t,K> idx) const noexcept requires(K>3) {
			//K=1: flat = idx[0]
			//K=2: flat = idx[0] + dim[0]*idx[1]
			//K=3: flat = idx[0] + dim[0]*( idx[1] + dim[1]*idx[2])
			//K=4: flat = idx[0] + dim[0]*( idx[1] + dim[1]*( idx[2] + dim[2]*idx[3]))
			size_t flat = 0, stride = 1;
			for (size_t k=0; k<K; ++k) {
				GUTIL_ASSERT(idx[k]<dim_[k]);
				flat   += idx[k]*stride;
				stride *= dim_[k];
			}
			return flat;
		}

		[[nodiscard]] const size_t flat_index(std::array<size_t,0> idx) const noexcept requires(K==0) {
			return 0;
		}
		
		[[nodiscard]] const size_t flat_index(std::array<size_t,1> idx) const noexcept requires(K==1) {
			GUTIL_ASSERT(idx[0]<dim_[0]);
			return idx[0];
		}
		
		[[nodiscard]] const size_t flat_index(std::array<size_t,2> idx) const noexcept requires(K==2) {
			GUTIL_ASSERT(idx[0]<dim_[0] && idx[1]<dim_[1]);
			return idx[0] + dim_[0]*idx[1];
		}

		[[nodiscard]] const size_t flat_index(std::array<size_t,3> idx) const noexcept requires(K==3) {
			GUTIL_ASSERT(idx[0]<dim_[0] && idx[1]<dim_[1] && idx[2]<dim_[2]);
			return idx[0] + dim_[0]*(idx[1] + dim_[1]*idx[2]);
		}

		//invert the index (flat to tensor/array index)
		void flat_to_tensor_index(size_t flat, std::array<size_t,K>& idx) const noexcept {
			for (size_t axis=0; axis<K; ++axis) {
				idx[axis] = flat % dim_[axis];
				flat /= dim_[axis];
			}
		}

		std::array<size_t,K> flat_to_tensor_index(size_t flat) const noexcept {
			std::array<size_t,K> idx{};
			for (size_t axis=0; axis<K; ++axis) {
				idx[axis] = flat % dim_[axis];
				flat /= dim_[axis];
			}
			return idx;
		}

		//a few helper functions to compute strides and starts of axis slices.
		//e.g., fixing all indices but one, tensor(..., i, ...) is a vector with a start and stride.
		//note incrementing the flat index has a fast (left dots) and slow (right dots) blocks of indices.
		//keeping i fixed, the remaining indices form a K-1 tensor.
		[[nodiscard]] constexpr size_t axis_stride(size_t axis) const noexcept {
			size_t stride = 1;
			for (size_t a=0; a<axis; ++a) {stride *= dim_[a];}
			return stride;
		}

		//applies op to every entry in the fixed-i slice along the specified axis (the K-1 dim tensor).
		//this can be used to do various tensor operations
		template<typename Op>
		void apply_along_axis_index(size_t axis, size_t i, Op&& op) const noexcept {
			const size_t stride = axis_stride(axis);
			const size_t dim_axis = dim_[axis];
			const size_t outer_count = count() / (stride * dim_axis);

			//note (axis,i) partions the flat index into a low/fast and a high/slow parts
			// ...fast axes..., specified axis=i, ...slow axes...
			//outer looper over the slow axes while inner loops over the fast.
			for (size_t outer=0; outer<outer_count; ++outer) {
				const size_t base = outer*(stride*dim_axis) + i*stride;
				for (size_t inner=0; inner<stride; ++inner) {
					op(ptr[base + inner]);
				}
			}
		}

		template<typename Op>
		void apply_along_axis_index_simd(size_t axis, size_t i, Op&& op) const noexcept {
			const size_t stride = axis_stride(axis);
			const size_t dim_axis = dim_[axis];
			const size_t outer_count = count() / (stride * dim_axis);

			//note (axis,i) partions the flat index into a low/fast and a high/slow parts
			// ...fast axes..., specified axis=i, ...slow axes...
			//outer looper over the slow axes while inner loops over the fast.
			GUTIL_SIMD(collapse(2) if(Simd))
			for (size_t outer=0; outer<outer_count; ++outer) {
				const size_t base = outer*(stride*dim_axis) + i*stride;
				for (size_t inner=0; inner<stride; ++inner) {
					op(ptr[base + inner]);
				}
			}
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