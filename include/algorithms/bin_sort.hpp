#pragma once

#include "utility/utility.hpp"
#include "threads/threads.hpp"

#include <span>
#include <algorithm>
#include <functional>
#include <bit>


namespace gutil {


	//////////////////////////////////////////////////////////
	/// A class for partitioning data in-place. The class may either
	/// own the data or just a view into the data, depending on the Derived type.
	///
	/// For efficiency, the Derived class should provide BinFunc statically via Derived::BinFun.
	/// However, that may not always be possible. If there is no static member, a runtime
	/// fallback can be supplied. Additionally, a static function can be supplied at runtime to the
	/// sorter. This will then be captured into the fallback function to be used for data lookup.
	///
	/// The Derived class must provide data() and size() methods to point to the beginning of the
	/// data to sort and the size of the data to sort.
	//////////////////////////////////////////////////////////
	template<typename Derived, typename T>
	struct BinSortBase {


		///////////////////////////////////////////////////////////////////////////////
		/// Bin information
		///////////////////////////////////////////////////////////////////////////////

		//check if Derived supplies the BinFun and number of bins
		static constexpr bool HAS_STATIC_BINFUN = requires {Derived::BinFunc(T{});};
		static constexpr bool HAS_STATIC_N_BINS = requires {Derived::N_BINS;};
		
		//initialize primary data using static data if possible.
		int n_bins_ = [](){
			if constexpr (HAS_STATIC_N_BINS) {return Derived::N_BINS;}
			else {return -1;}
		}();

		int n_bits_ = [](){
			if constexpr (HAS_STATIC_N_BINS) {return std::bit_width(static_cast<uint>(Derived::N_BINS-1));}
			else {return -1;}
		}();
		
		std::vector<size_t> bins = [](){
			if constexpr (HAS_STATIC_N_BINS) {
				std::vector<size_t> b(Derived::N_BINS+1, 0);
				return b;
			}
			else {
				return std::vector<size_t>{};
			}
		}();
		
		//hold a fallback bin function if it must be set at runtime
		template<typename BinFun> requires (std::is_invocable_r_v<int,BinFun,T>)
		void set_bin_fun(BinFun&& f) noexcept requires(!HAS_STATIC_BINFUN) {fallback_bin_fun = std::forward<BinFun>(f);}
		std::function<int(const T&)> fallback_bin_fun;

		[[nodiscard]] static int bin(const T& val) noexcept requires (HAS_STATIC_BINFUN) {
			return Derived::BinFunc(val);
		}

		[[nodiscard]] int bin(const T& val) const noexcept requires (!HAS_STATIC_BINFUN) {
			return fallback_bin_fun(val);
		}

		//set and resize the number of bins
		void set_n_bins(int M) noexcept requires (!HAS_STATIC_N_BINS) {
			GUTIL_ASSERT(M>0);
			bins.assign(M+1,0);
			n_bins_ = M;
			n_bits_ = std::bit_width(static_cast<uint>(M-1));
		}

		///////////////////////////////////////////////////////////////////////////////
		/// Data access
		///////////////////////////////////////////////////////////////////////////////
		//cast to the derived class for convenience
		[[nodiscard]]       Derived* derived()       noexcept {return static_cast<Derived*>(this);}
		[[nodiscard]] const Derived* derived() const noexcept {return static_cast<const Derived*>(this);}

		//get a pointer to the beginning of the data (const or mutable) and the size of the data
		[[nodiscard]] T*       data()       noexcept {return derived()->data_impl();}
		[[nodiscard]] const T* data() const noexcept {return derived()->data_impl();}
		[[nodiscard]] size_t   size() const noexcept {return derived()->size_impl();}

		//get pointers to the beginning/end of each bin
		[[nodiscard]] T*       begin(int i)       noexcept {GUTIL_ASSERT(is_valid(i)); return data()+bins[i];}
		[[nodiscard]] T*       end(int i)         noexcept {GUTIL_ASSERT(is_valid(i)); return data()+bins[i+1];}
		[[nodiscard]] const T* begin(int i) const noexcept {GUTIL_ASSERT(is_valid(i)); return data()+bins[i];}
		[[nodiscard]] const T* end(int i)   const noexcept {GUTIL_ASSERT(is_valid(i)); return data()+bins[i+1];}

		//get spans into a bin
		[[nodiscard]] std::span<T>       get_bin(int i)       noexcept {GUTIL_ASSERT(is_valid(i)); return std::span<T>{begin(i), end(i)};}
		[[nodiscard]] std::span<const T> get_bin(int i) const noexcept {GUTIL_ASSERT(is_valid(i)); return std::span<const T>{begin(i), end(i)};}

		//get indices to the start/end of a bin and the bin size
		[[nodiscard]] size_t bin_size(int i)  const noexcept {GUTIL_ASSERT(is_valid(i)); return bins[i+1]-bins[i];}
		[[nodiscard]] size_t bin_start(int i) const noexcept {GUTIL_ASSERT(is_valid(i)); return bins[i];}
		[[nodiscard]] size_t bin_end(int i)   const noexcept {GUTIL_ASSERT(is_valid(i)); return bins[i+1];}

		//a few extra convenience methods
		[[nodiscard]] static constexpr int n_bins() noexcept requires (HAS_STATIC_N_BINS) {return Derived::N_BINS;}
		[[nodiscard]] int n_bins() const noexcept requires (!HAS_STATIC_N_BINS) {return n_bins_;}
		[[nodiscard]] bool empty() const noexcept {return derived()->empty_impl();}
		
		//Forward const element access to the entire data set
		const T& operator[](size_t idx) const noexcept {GUTIL_ASSERT(idx < size()); return data()[idx];}
		std::span<const T> as_span() const noexcept {return std::span<const T>{data(), size()};}


		///////////////////////////////////////////////////////////////////////////////
		/// Manage multithreading
		///////////////////////////////////////////////////////////////////////////////
		ThreadPool* threads{nullptr};
		void set_threadpool(ThreadPool& tp) noexcept {threads = &tp;}
		void clear_threadpool() noexcept {threads = nullptr;}
		
		///////////////////////////////////////////////////////////////////////////////
		/// Look up data index or pointers. Sort each bin for better performance.
		///////////////////////////////////////////////////////////////////////////////
		//get the index to data
		[[nodiscard]] size_t index(const T& val) const noexcept {
			const int i = bin(val);
			auto it = std::find(begin(i), end(i), val);
			if (it==end(i)) {return size_t(-1);}
			else {return bin_start(i) + std::distance(begin(i), it);}
		}

		template<typename Less = std::less<T>>
		[[nodiscard]] size_t index_sorted(const T& val, Less&& less = Less{}) const noexcept {
			const int i = bin(val);
			auto it = std::lower_bound(begin(i), end(i), val, std::forward<Less>(less));
			if (it==end(i) || *it!=val) {return size_t(-1);}
			else {return bin_start(i) + std::distance(begin(i), it);}
		}

		//get an iterator to data
		[[nodiscard]] const T* find(const T& val) const noexcept {
			const size_t idx = index(val);
			return (idx==size_t(-1)) ? data()+size() : data()+idx;
		}

		template<typename Less = std::less<T>>
		[[nodiscard]] const T* find_sorted(const T& val, Less&& less = Less{}) const noexcept {
			const size_t idx = index_sorted(val, std::forward<Less>(less));
			return (idx==size_t(-1)) ? data()+size() : data()+idx;
		}


		///////////////////////////////////////////////////////////////////////////////
		/// Extra utility
		///////////////////////////////////////////////////////////////////////////////
		void clear() noexcept {
			if constexpr (!HAS_STATIC_N_BINS) {
				n_bins_ = -1;
				n_bits_ = -1;
			}
			threads = nullptr;
			derived()->clear_impl();
		}


		///////////////////////////////////////////////////////////////////////////////
		/// Sorting interface
		///////////////////////////////////////////////////////////////////////////////
		template<typename Less = std::less<T>>
		void sort_bins(Less&& less = Less{}) noexcept {
			if (threads) {
				std::vector<ThreadPool::Handle> finished_flags;
				for (int i=0; i<n_bins(); ++i) {
					finished_flags.push_back( threads->submit([less](auto a, auto b){std::sort(a, b, less);}, begin(i), end(i)));
				}
				for (auto h : finished_flags) {
					threads->wait_for(h);
				}
			}
			else {
				for (int i=0; i<n_bins(); ++i) {
					std::sort(begin(i), end(i), less);
				}
			}
		}

		template<typename BinFun> requires(std::is_invocable_r_v<int, BinFun, T> && !HAS_STATIC_BINFUN)
		void sort(BinFun&& bin_fun) {
			GUTIL_ASSERT(is_valid(n_bins_-1));
			fallback_bin_fun = std::function<int(const T&)>{bin_fun};
			bins[n_bins_] = size();

			if (threads) {
				recursive_partition_bit_parallel(0, size(), n_bits_-1, 0, std::forward<BinFun>(bin_fun));
			}
			else {
				recursive_partition_bit(0, size(), n_bits_-1, 0, std::forward<BinFun>(bin_fun));
			}
		}

		void sort() {
			GUTIL_ASSERT(is_valid(n_bins_-1));
			bins[n_bins_] = size();
			if (threads) {
				if constexpr (HAS_STATIC_BINFUN) {
					recursive_partition_bit_parallel(0, size(), n_bits_-1, 0, Derived::BinFunc);
				}
				else {
					recursive_partition_bit_parallel(0, size(), n_bits_-1, 0, fallback_bin_fun);
				}
			}
			else {
				if constexpr (HAS_STATIC_BINFUN) {
					recursive_partition_bit(0, size(), n_bits_-1, 0, Derived::BinFunc);
				}
				else {
					recursive_partition_bit(0, size(), n_bits_-1, 0, fallback_bin_fun);
				}
			}
		}

	protected:
		[[nodiscard]] bool is_valid(int i) const noexcept {
			if (i<0 || i>=n_bins_)                 {return false;}
			if (bins.size() != (size_t) n_bins_+1) {return false;}
			if (bins[i+1]<bins[i])                 {return false;}
			return true;
		}

		template<typename BinFun> requires(std::is_invocable_r_v<int, BinFun, T>)
		void recursive_partition_bit(size_t left, size_t right, int bit, int bin, BinFun&& bin_fun) noexcept;

		template<typename BinFun> requires(std::is_invocable_r_v<int, BinFun, T>)
		void recursive_partition_bit_parallel(size_t left, size_t right, int bit, int bin, BinFun&& bin_fun) noexcept;
		
	public:
		BinSortBase() = default;
		BinSortBase(const BinSortBase&) = default;
		BinSortBase(BinSortBase&&) = default;
		BinSortBase& operator=(const BinSortBase&) = default;
		BinSortBase& operator=(BinSortBase&&) = default;
			
		template<typename BinFun> requires(std::is_invocable_r_v<int, BinFun, T> && !HAS_STATIC_BINFUN)
		BinSortBase(BinFun&& fun) : fallback_bin_fun{std::forward<BinFun>(fun)} {}
	};


	////////////////////////////////////////////////////////////////////////////////////////////////
	/// A non-owning version of BinSort (implementation class)
	////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename Derived, typename T>
	struct BinSortSpanImpl : public BinSortBase<Derived,T> {
		std::span<T> view{};

		BinSortSpanImpl() = default;
		explicit BinSortSpanImpl(std::span<T> data) : view(data) {}
		BinSortSpanImpl(T* data, size_t n) : view(data,n) {}

		[[nodiscard]] T* data_impl() noexcept {return view.data();}
		[[nodiscard]] const T* data_impl() const noexcept {return view.data();}
		[[nodiscard]] size_t size_impl() const noexcept {return view.size();}
		[[nodiscard]] bool empty_impl() const noexcept {return view.empty();}
		void clear_impl() noexcept {view = std::span<T>{};}

		void rebind_to_copy(std::span<T> data_copy) noexcept {
			#ifndef NDEBUG
			GUTIL_ASSERT(data_copy.size() == view.size());
			for (size_t i=0; i<std::min(size_t{100}, view.size()); ++i) {
				GUTIL_ASSERT(data_copy[i]==view[i]);
			}
			#endif
			view = data_copy;
		}
	};


	////////////////////////////////////////////////////////////////////////////////////////////////
	/// An owning version of BinSort (implementation class)
	////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename Derived, typename T>
	struct BinSortVectorImpl : public BinSortBase<Derived,T> {
		std::vector<T> values{};

		BinSortVectorImpl() = default;
		explicit BinSortVectorImpl(std::vector<T> data) : values(std::move(data)) {}
		template<std::random_access_iterator I> requires (std::same_as<T,std::iter_value_t<I>>)
		BinSortVectorImpl(I begin, I end) : values(begin, end) {}

		[[nodiscard]] T* data_impl() noexcept {return values.data();}
		[[nodiscard]] const T* data_impl() const noexcept {return values.data();}
		[[nodiscard]] size_t size_impl() const noexcept {return values.size();}
		[[nodiscard]] bool empty_impl() const noexcept {return values.empty();}
		void clear_impl() noexcept {values.clear();}

		[[nodiscard]] std::vector<T>& vector() noexcept {return values;}
		[[nodiscard]] const std::vector<T>& vector() const noexcept {return values;}
	};


	////////////////////////////////////////////////////////////////////////////////////////////////
	/// Pre-declare final classes for enabling better conversions
	////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename T>
	struct BinSortSpan;

	template<typename T, auto BinFun, int nBins> requires(std::is_invocable_r_v<int,decltype(BinFun),T> && nBins>0)
	struct StaticBinSortSpan;

	template<typename T>
	struct BinSortVector;

	template<typename T, auto BinFun, int nBins> requires(std::is_invocable_r_v<int,decltype(BinFun),T> && nBins>0)
	struct StaticBinSortVector;


	////////////////////////////////////////////////////////////////////////////////////////////////
	/// Final non-owning/view classes
	////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename T>
	struct BinSortSpan : public BinSortSpanImpl<BinSortSpan<T>,T> {
		using BinSortSpanImpl<BinSortSpan<T>,T>::BinSortSpanImpl;

		template<auto BinFun, int nBins> requires(std::is_invocable_r_v<int,decltype(BinFun),T> && nBins>0)
		[[nodiscard]] explicit operator StaticBinSortSpan<T,BinFun,nBins>() noexcept {
			StaticBinSortSpan<T,BinFun,nBins> result{this->data(), this->size()};
			result.bins = this->bins;
			return result;
		}

		explicit BinSortSpan(BinSortVector<T>& bin_sort_vec) noexcept : BinSortSpan(bin_sort_vec.data(), bin_sort_vec.size()) {
			this->bins    = bin_sort_vec.bins;
			this->n_bins_ = bin_sort_vec.n_bins_;
			this->n_bits_ = bin_sort_vec.n_bits_;
			this->fallback_bin_fun = bin_sort_vec.fallback_bin_fun;
		}
	};

	template<typename T, auto BinFun, int nBins> requires(std::is_invocable_r_v<int,decltype(BinFun),T> && nBins>0)
	struct StaticBinSortSpan : public BinSortSpanImpl<StaticBinSortSpan<T, BinFun, nBins>,T> {
		static constexpr int N_BINS = nBins;
		static int BinFunc(const T& val) noexcept {return BinFun(val);}
		using BinSortSpanImpl<StaticBinSortSpan<T, BinFun, nBins>,T>::BinSortSpanImpl;

		[[nodiscard]] explicit operator BinSortSpan<T>() noexcept {
			BinSortSpan<T> result{this->data(), this->size()};
			result.set_n_bins(N_BINS);
			result.set_bin_fun(BinFunc);
			result.bins = this->bins;
			return result;
		}

		explicit StaticBinSortSpan(StaticBinSortVector<T,BinFun,nBins>& bin_sort_vec) noexcept : StaticBinSortSpan(bin_sort_vec.data(), bin_sort_vec.size()) {
			this->bins = bin_sort_vec.bins;
		}
	};


	////////////////////////////////////////////////////////////////////////////////////////////////
	/// Final owning classes
	////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename T>
	struct BinSortVector : public BinSortVectorImpl<BinSortVector<T>,T> {
		using BinSortVectorImpl<BinSortVector<T>,T>::BinSortVectorImpl;

		template<auto BinFun, int nBins> requires(std::is_invocable_r_v<int,decltype(BinFun),T> && nBins>0)
		[[nodiscard]] explicit operator StaticBinSortVector<T,BinFun,nBins>() const noexcept {
			StaticBinSortVector<T,BinFun,nBins> result{this->values};
			result.bins = this->bins;
			return result;
		}

		template<auto BinFun, int nBins> requires(std::is_invocable_r_v<int,decltype(BinFun),T> && nBins>0)
		[[nodiscard]] explicit operator StaticBinSortVector<T,BinFun,nBins>() noexcept {
			StaticBinSortVector<T,BinFun,nBins> result{std::move(this->values)};
			result.bins = this->bins;
			return result;
		}

		explicit BinSortVector(const BinSortSpan<T>& bin_sort_span) noexcept : 
			BinSortVector(bin_sort_span.data(), bin_sort_span.data()+bin_sort_span.size()) {
			
			this->bins    = bin_sort_span.bins;
			this->n_bins_ = bin_sort_span.n_bins_;
			this->n_bits_ = bin_sort_span.n_bits_;
			this->fallback_bin_fun = bin_sort_span.fallback_bin_fun;
		}
	};

	template<typename T, auto BinFun, int nBins> requires(std::is_invocable_r_v<int,decltype(BinFun),T> && nBins>0)
	struct StaticBinSortVector : public BinSortVectorImpl<StaticBinSortVector<T,BinFun,nBins>,T> {
		static constexpr int N_BINS = nBins;
		static int BinFunc(const T& val) noexcept {return BinFun(val);}
		using BinSortVectorImpl<StaticBinSortVector<T,BinFun,nBins>,T>::BinSortVectorImpl;

		[[nodiscard]] explicit operator BinSortVector<T>() const noexcept {
			BinSortVector<T> result{this->values};
			result.set_n_bins(N_BINS);
			result.set_bin_fun(BinFunc);
			result.bins = this->bins;
			return result;
		}

		[[nodiscard]] explicit operator BinSortVector<T>() noexcept {
			BinSortVector<T> result{std::move(this->values)};
			result.set_n_bins(N_BINS);
			result.set_bin_fun(BinFunc);
			result.bins = this->bins;
			return result;
		}

		explicit StaticBinSortVector(const StaticBinSortSpan<T,BinFun,nBins>& bin_sort_span) noexcept : 
			StaticBinSortVector(bin_sort_span.data(), bin_sort_span.data()+bin_sort_span.size()) {
			this->bins = bin_sort_span.bins;
		}
	};


	////////////////////////////////////////////////////////////////////////////////////////////////
	/// Implementation of bin-sort algorithms
	////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename Derived, typename T>
	template<typename BinFun> requires(std::is_invocable_r_v<int, BinFun, T>)
	void BinSortBase<Derived,T>::recursive_partition_bit(size_t left, size_t right, int bit, int bin, BinFun&& bin_fun) noexcept {
		GUTIL_ASSERT(left<=right);
		if (bit<0) {
			GUTIL_ASSERT(is_valid(bin));
			bins[bin]   = left;
			return;
		}

		//construct the bool predicate and partition. note that because we wish to sort by
		//increasing bin number, the bit-check must be negated when passing to std::partition
		const int mask = int{1} << bit;

		T* it;
		if constexpr (HAS_STATIC_BINFUN) {
			it = std::partition(data()+left, data()+right,
				[mask](const T& val){return !static_cast<bool>(Derived::BinFunc(val) & mask);});
		}
		else {
			it = std::partition(data()+left, data()+right, 
				[mask, &bin_fun](const T& val){return !static_cast<bool>(bin_fun(val) & mask);});
		}

		size_t mid = static_cast<size_t>(std::distance(data(), it));

		const int left_bin  = bin;
		const int right_bin = bin | (int{1} << bit);

		if (left_bin < n_bins_) {
			recursive_partition_bit(left, mid, bit-1, left_bin, bin_fun);
		}
		
		if (right_bin < n_bins_) {
			recursive_partition_bit(mid, right, bit-1, right_bin, std::forward<BinFun>(bin_fun));
		}
	}

	template<typename Derived, typename T>
	template<typename BinFun> requires(std::is_invocable_r_v<int, BinFun, T>)
	void BinSortBase<Derived, T>::recursive_partition_bit_parallel(size_t left, size_t right, int bit, int bin, BinFun&& bin_fun) noexcept {
		GUTIL_ASSERT(threads);
		GUTIL_ASSERT(left<=right);
		if (bit<0) {
			GUTIL_ASSERT(is_valid(bin));
			bins[bin]   = left;
			return;
		}

		//construct the bool predicate and partition. note that because we wish to sort by
		//increasing bin number, the bit-check must be negated when passing to std::partition
		const int mask = int{1} << bit;
		
		T* it;
		if constexpr (HAS_STATIC_BINFUN) {
			it = std::partition(data()+left, data()+right,
				[mask](const T& val){return !static_cast<bool>(Derived::BinFunc(val) & mask);});
		}
		else {
			it = std::partition(data()+left, data()+right, 
				[mask, &bin_fun](const T& val){return !static_cast<bool>(bin_fun(val) & mask);});
		}

		size_t mid = static_cast<size_t>(std::distance(data(), it));

		const int left_bin  = bin;
		const int right_bin = bin | (int{1} << bit);
		const bool fork     = ((right-left) > 4096) && (threads->n_active_tasks()<threads->n_threads());

		ThreadPool::Handle h;

		if (left_bin < n_bins_) {
			if (fork) {
				h = threads->submit( [&](size_t l, size_t r, int bt, int bn, std::decay_t<BinFun> pred) noexcept {recursive_partition_bit_parallel(l,r,bt,bn,pred);},
							left, mid, bit-1, left_bin, bin_fun );
			}
			else {
				recursive_partition_bit_parallel(left, mid, bit-1, left_bin, bin_fun);
			}
		}
		
		if (right_bin < n_bins_) {
			recursive_partition_bit_parallel(mid, right, bit-1, right_bin, std::forward<BinFun>(bin_fun));
		}

		if (fork) {threads->wait_for(h);}
	}
}
