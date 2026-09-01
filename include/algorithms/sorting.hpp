#pragma once

#include "utility/utility.hpp"
#include "threads/threads.hpp"

#include <span>
#include <algorithm>
#include <bit>

namespace gutil
{


	//////////////////////////////////////////////////////////
	/// A few single threaded utility methods
	//////////////////////////////////////////////////////////
	template<std::contiguous_iterator I>
	[[nodiscard]] inline constexpr I sort_and_unique(I begin, I end) noexcept {
		std::sort(begin, end);
		return std::unique(begin, end);
	}

	template<typename Container> requires(std::contiguous_iterator<typename Container::iterator>)
	[[nodiscard]] inline constexpr auto sort_and_unique(Container& list) noexcept {
		return sort_and_unique(list.begin(), list.end());
	}

	template<std::contiguous_iterator I, typename Less_t> requires(std::is_invocable_r_v<bool, Less_t, typename std::iter_value_t<I>, typename std::iter_value_t<I>>)
	[[nodiscard]] inline constexpr I sort_and_unique(I begin, I end, Less_t&& less) noexcept {
		std::sort(begin, end, std::forward<Less_t>(less));
		return std::unique(begin, end);
	}

	template<typename Container, typename Less_t> requires(std::contiguous_iterator<typename Container::iterator>
										 && std::is_invocable_r_v<bool, Less_t, typename Container::value_type, typename Container::value_type>)
	[[nodiscard]] inline constexpr auto sort_and_unique(Container& list, Less_t&& less) noexcept {
		return sort_and_unique(list.begin(), list.end(), std::forward<Less_t>(less));
	}



	//////////////////////////////////////////////////////////
	/// A class for partitioning data in-place. A predicate of the type
	/// pred(value) -> int must be supplied with the return value (the 'bin')
	/// a number in [0,N). The algorithm uses divide-and-conquer
	/// by splitting the data into left/right partitions for each bit
	/// in the bin. An internal state is kept to more easily extract subspans.
	//////////////////////////////////////////////////////////
	template<typename T>
	struct BinSort {
	private:
		using iterator_type 		= typename std::span<T>::iterator;
		using const_iterator_type 	= typename std::span<const T>::iterator;

		int 					n_bins_{-1};
		int 					n_bits_{-1};
		std::span<T> 			data{};
		std::vector<size_t>		bins{};
		ThreadPool* 			threads{nullptr};	//allow parallel sorting if another class provides the resource

		template<typename BinFun> requires(std::is_invocable_r_v<int, BinFun, T>)
		void recursive_partition_bit(size_t left, size_t right, int bit, int bin, BinFun&& bin_fun) noexcept;

		template<typename BinFun> requires(std::is_invocable_r_v<int, BinFun, T>)
		void recursive_partition_bit_parallel(size_t left, size_t right, int bit, int bin, BinFun&& bin_fun) noexcept;
	public:
		BinSort() = default;
		BinSort(const BinSort&) = default;
		BinSort(BinSort&&) = default;
		BinSort& operator=(const BinSort&) = default;
		BinSort& operator=(BinSort&&) = default;
		
		template<typename I>
		BinSort(I begin, I end, int N) : BinSort(std::span<T>{begin, end}, N) {}
		
		template<typename I>
		BinSort(I begin, size_t len, int N) : BinSort(std::span<T>{begin, begin+len}, N) {}

		BinSort(std::vector<T>& data, int N) : BinSort(std::span<T>{data.begin(), data.end()}, N) {}

		BinSort(std::span<T> data, int N) : n_bins_(N), data(data), bins(N+1) {
			GUTIL_ASSERT(N>0);
			n_bits_ = std::bit_width(static_cast<uint>(N-1));
			bins[N] = data.size();
		}

		[[nodiscard]] bool empty() const noexcept { return data.empty(); }
		[[nodiscard]] int n_bins() const noexcept { return n_bins_; }

		void rebind_to_copy(std::span<T> data_copy) noexcept {
			GUTIL_ASSERT(data_copy.size() == data.size());
			data = data_copy;
		}

		void clear() noexcept {
			data = std::span<T>{};
			n_bins_ = -1;
			n_bits_ = -1;
			threads = nullptr;
			bins.clear();
		}

		[[nodiscard]] size_t size() const noexcept {
			return data.size();
		}

		/// Primary call (pass the full predicate to bin number)
		template<typename BinFun> requires(std::is_invocable_r_v<int, BinFun, T>)
		void sort(BinFun&& bin_fun) {
			//note std::bit_width requires an unsigned integer
			//if there are N bins, then bins.size() = N+1, the max bin index is N-1,
			//and the max bit index is bit_width(N-1) - 1
			if (threads) {
				//note this thread will be the primary thread and not return until the sort is finished.
				//this way threads.wait_idle() doesn't have to be called from here.
				recursive_partition_bit_parallel(0, data.size(), n_bits_-1, 0, std::forward<BinFun>(bin_fun));
			}
			else {
				recursive_partition_bit(0, data.size(), n_bits_-1, 0, std::forward<BinFun>(bin_fun));
			}
		}

		/// Primary call (pass the full predicate to bin number)
		template<typename BinFun> requires(std::is_invocable_r_v<int, BinFun, T>)
		void dispatch_sort(BinFun&& bin_fun, ThreadPool* tp) {
			threads = tp;
			if (threads) {
				recursive_partition_bit_parallel(0, data.size(), n_bits_-1, 0, std::forward<BinFun>(bin_fun));
			}
			else {
				recursive_partition_bit(0, data.size(), n_bits_-1, 0, std::forward<BinFun>(bin_fun));
			}
		}



		/// Once sorted, get a subspan into the requested bin
		[[nodiscard]] std::span<const T> get_bin(int i) const noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			GUTIL_ASSERT(bins[i+1]>=bins[i]);
			return std::span<const T>{data.begin()+bins[i], data.begin()+bins[i+1]};
		}

		[[nodiscard]] std::span<T> get_bin(int i) noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			GUTIL_ASSERT(bins[i+1]>=bins[i]);
			return std::span<T>{data.begin()+bins[i], data.begin()+bins[i+1]};
		}

		[[nodiscard]] size_t bin_size(int i) const noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			GUTIL_ASSERT(bins[i+1]>=bins[i]);
			return static_cast<size_t>(bins[i+1]-bins[i]);
		}

		[[nodiscard]] size_t bin_start(int i) const noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			return bins[i];
		}

		[[nodiscard]] size_t bin_end(int i) const noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			return bins[i+1];
		}

		[[nodiscard]] iterator_type begin(int i) noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			return data.begin()+bins[i];
		}

		[[nodiscard]] iterator_type end(int i) noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			return data.begin()+bins[i+1];
		}

		[[nodiscard]] const_iterator_type begin(int i) const noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			return data.begin()+bins[i+1];
		}

		[[nodiscard]] const_iterator_type end(int i) const noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			return data.begin()+bins[i+1];
		}
	};

	template<typename T>
	template<typename BinFun> requires(std::is_invocable_r_v<int, BinFun, T>)
	void BinSort<T>::recursive_partition_bit(size_t left, size_t right, int bit, int bin, BinFun&& bin_fun) noexcept {
		GUTIL_ASSERT(left<=right);
		if (bit<0) {
			GUTIL_ASSERT(0<=bin && bin<=n_bins_);
			bins[bin]   = left;
			// bins[bin+1] = right;
			return;
		}

		//construct the bool predicate and partition. note that because we wish to sort by
		//increasing bin number, the bit-check must be negated when passing to std::partition
		const int mask = int{1} << bit;
		auto bool_pred = [&bin_fun, mask](const T& val) {GUTIL_ASSERT(bin_fun(val)>=0); return !(bool)(bin_fun(val) & mask);};
		iterator_type it = std::partition(data.begin()+left, data.begin()+right, bool_pred);
		size_t mid = static_cast<size_t>(std::distance(data.begin(), it));

		const int left_bin  = bin;
		const int right_bin = bin | (int{1} << bit);

		if (left_bin < n_bins_) {
			recursive_partition_bit(left, mid, bit-1, left_bin, bin_fun);
		}
		
		if (right_bin < n_bins_) {
			recursive_partition_bit(mid, right, bit-1, right_bin, std::forward<BinFun>(bin_fun));
		}
	}

	template<typename T>
	template<typename BinFun> requires(std::is_invocable_r_v<int, BinFun, T>)
	void BinSort<T>::recursive_partition_bit_parallel(size_t left, size_t right, int bit, int bin, BinFun&& bin_fun) noexcept {
		GUTIL_ASSERT(threads);
		GUTIL_ASSERT(left<=right);
		if (bit<0) {
			GUTIL_ASSERT(0<=bin && bin<=n_bins_);
			bins[bin]   = left;
			return;
		}

		//construct the bool predicate and partition. note that because we wish to sort by
		//increasing bin number, the bit-check must be negated when passing to std::partition
		const int mask = int{1} << bit;
		auto bool_pred = [&bin_fun, mask](const T& val) {GUTIL_ASSERT(bin_fun(val)>=0); return !(bool)(bin_fun(val) & mask);};
		iterator_type it = std::partition(data.begin()+left, data.begin()+right, bool_pred);
		size_t mid = static_cast<size_t>(std::distance(data.begin(), it));

		const int left_bin  = bin;
		const int right_bin = bin | (int{1} << bit);
		const bool fork     = ((right-left) > 4096) && (threads->n_active_tasks()<threads->n_threads());

		if (left_bin < n_bins_) {
			if (fork) {
				threads->submit( [&](size_t l, size_t r, int bt, int bn, std::decay_t<BinFun> pred) noexcept {recursive_partition_bit_parallel(l,r,bt,bn,pred);},
							left, mid, bit-1, left_bin, bin_fun );
			}
			else {
				recursive_partition_bit_parallel(left, mid, bit-1, left_bin, bin_fun);
			}
		}
		
		if (right_bin < n_bins_) {
			recursive_partition_bit_parallel(mid, right, bit-1, right_bin, std::forward<BinFun>(bin_fun));
		}
	}



	/////////////////////////////////////////////////////////////////////////////////////////////
	/// Multi-threaded sort and unique functions
	/////////////////////////////////////////////////////////////////////////////////////////////
	template<typename T>
	[[maybe_unused]] inline void sort_and_unique_parallel_impl(size_t begin, size_t end, size_t& partition_index, T* const vals, ThreadPool& tp) {
		GUTIL_ASSERT(vals);
		GUTIL_ASSERT(end>=begin);
		
		const bool fork = ((end-begin) > 4096) && (tp.n_active_tasks()<tp.n_threads());

		if (fork) {
			const size_t pivot = begin + (end-begin)/2;
			
			size_t left_idx, right_idx;
			auto h = tp.submit([](size_t b, size_t e, size_t& idx, T* const v, ThreadPool& p){	
				sort_and_unique_parallel_impl<T>(b,e,idx,v,p);
			}, begin, pivot, std::ref(left_idx), vals, std::ref(tp));
			
			sort_and_unique_parallel_impl<T>(pivot,end,right_idx,vals,tp);
			tp.wait_for(h);

			//state of data
			// 	vals	<---   begin -------- left_idx -------- pivot -------- right_idx -------- end
			// 					  |     KEEP      |  MOVE TO END   |     KEEP       |  IN PLACE (DELETE)

			size_t swap_size = right_idx - pivot;
			for (size_t i=0; i<swap_size; ++i) {
				std::swap(vals[pivot+i],vals[left_idx+i]);
			}
			
			size_t reduced_end = left_idx + swap_size;	//right of this we know is bad
			std::inplace_merge(vals+begin, vals+left_idx, vals+reduced_end);
			//state of data
			// begin -------- left_idx ----- reduced_end ----------- end
			//   |     KEEP      |    UNKNOWN    |     IN PLACE (DELETE)

			auto it = std::unique(vals+begin, vals+reduced_end);
			partition_index = begin + static_cast<size_t>(std::distance(vals+begin, it));
			return;
		}
		else {
			std::sort(vals+begin, vals+end);
			auto it = std::unique(vals+begin, vals+end);
			partition_index = begin + static_cast<size_t>(std::distance(vals+begin,it));
			return;
		}
	}


	template<std::contiguous_iterator I>
	[[nodiscard]] I sort_and_unique(I begin, I end, ThreadPool& tp) {
		size_t mid;
		sort_and_unique_parallel_impl(0, std::distance(begin,end), mid, std::to_address(begin), tp);
		return begin + mid;
	}

	template<typename Container> requires (std::contiguous_iterator<typename Container::iterator>)
	[[nodiscard]] typename Container::iterator sort_and_unique(Container& list, ThreadPool& tp) {
		return sort_and_unique(list.begin(), list.end(), tp);
	}
}
