#pragma once

#include "utility/utility.hpp"
#include "memory/thread_pool.hpp"

#include <span>
#include <algorithm>
#include <bit>

namespace gutil
{
	//////////////////////////////////////////////////////////
	/// A class for partitioning data in-place. A predicate of the type
	/// pred(value) -> uint8_t must be supplied with the return value (the 'bin')
	/// a number in [0,N). The algorithm uses divide-and-conquer
	/// by splitting the data into left/right partitions for each bit
	/// in the bin. An internal state is kept to more easily extract subspans.
	//////////////////////////////////////////////////////////
	template<typename T>
	struct BinSort {
	private:
		using iterator_type = typename std::span<T>::iterator;
		
		uint8_t 					n_bins_{-1};
		uint8_t 					n_bits_{-1};
		std::span<T> 				data{};
		std::vector<iterator_type> 	bins{};
		ThreadPool* 				threads{nullptr};	//allow parallel sorting if another class provides the resource

		template<typename BinFun> requires(std::invocable_r_v<uint8_t, BinFun, T>)
		void recursive_partition_bit(std::span<T> data, uint8_t bit, uint8_t bin, BinFun&& bun_fun) noexcept;

		template<typename BinFun> requires(std::invocable_r_v<uint8_t, BinFun, T>)
		void recursive_partition_bit_parallel(std::span<T> data, uint8_t bit, uint8_t bin, BinFun&& bun_fun) noexcept;
	public:
		BinSort() = default;
		BinSort(const BinSort&) = default;
		BinSort(BinSort&&) = default;
		BinSort& operator=(const BinSort&) = default;
		BinSort& operator=(BinSort&&) = default;
		
		template<typename I>
		BinSort(I begin, I end, uint8_t N) : n_bins_{N}, data{begin, end}, bins(N+1) {}
		
		template<typename I>
		BinSort(I begin, size_t len, uint8_t N) : n_bins_{N}, data{begin, len}, bins(N+1) {}

		BinSort(std::vector<T>& data_, uint8_t N) : n_bins_{N}, data{data_.begin(), data_.end()}, bins(N+1) {}

		BinSort(std::span<T> data_, uint8_t N) : n_bins_{N}, data(data_), bins(N+1) {
			GUTIL_ASSERT(N>0);
			n_bits_ = std::bit_width(static_cast<uint>(N-1));
		}

		[[nodiscard]] bool empty() const noexcept { return data.empty(); }
		[[nodiscard]] size_t n_bins() const noexcept { return static_cast<size_t>(n_bins_); }

		void clear() noexcept {
			data = std::span<T>{};
			n_bins_ = -1;
			n_bits_ = -1;
			ThreadPool = nullptr;
			bins.clear();
		}

		/// Primary call (pass the full predicate to bin number)
		template<typename BinFun> requires(std::invocable_r_v<uint8_t, BinFun, T>)
		void sort(BinFun&& bun_fun) {
			//note std::bit_width requires an unsigned integer
			//if there are N bins, then bins.size() = N+1, the max bin index is N-1,
			//and the max bit index is bit_width(N-1) - 1
			if (threads) {
				//note this thread will be the primary thread and not return until the sort is finished.
				//this way threads.wait_idle() doesn't have to be called from here.
				recursive_partition_bit_parallel(data, n_bits_-1, 0, std::forward<BinFun>(bun_fun));
			}
			else {
				recursive_partition_bit(data, n_bits_-1, 0, std::forward<BinFun>(bun_fun));
			}
		}

		/// Primary call (pass the full predicate to bin number)
		template<typename BinFun> requires(std::invocable_r_v<uint8_t, BinFun, T>)
		void dispatch_sort(BinFun&& bun_fun, ThreadPool* tp) {
			threads = tp;
			if (threads) {
				recursive_partition_bit_parallel(data, n_bits_-1, 0, std::forward<BinFun>(bun_fun));
			}
			else {
				recursive_partition_bit(data, n_bits_-1, 0, std::forward<BinFun>(bun_fun));
			}
		}



		/// Once sorted, get a subspan into the requested bin
		[[nodiscard]] std::span<T> get_bin(uint8_t i) const noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			return std::span<T>{bins[i], bins[i+1]};
		}

		[[nodiscard]] size_t bin_size(uint8_t i) const noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			return static_cast<size_t>(std::distance(bins[i],bins[i+1]));
		}

		[[nodiscard]] size_t bin_start(uint8_t i) const noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			return static_cast<size_t>(std::distance(bins[0], bins[i]));
		}

		[[nodiscard]] size_t bin_end(uint8_t i) const noexcept {
			GUTIL_ASSERT(0<=i && i<n_bins_);
			GUTIL_ASSERT( static_cast<size_t>(n_bins_)+1 == bins.size() );
			return static_cast<size_t>(std::distance(bins[0], bins[i+1]));
		}
	};

	template<typename T>
	template<typename BinFun> requires(std::invocable_r_v<uint8_t, BinFun, T>)
	void BinSort<T>::recursive_partition_bit(std::span<T> data, uint8_t bit, uint8_t bin, BinFun&& bun_fun) noexcept {
		if (bit<0) {
			GUTIL_ASSERT(0<=bin && bin<=n_bins_);
			bins[bin] = data.begin();
			bins[bin+1] = data.end();
			return;
		}

		//construct the bool predicate and partition. note that because we wish to sort by
		//increasing bin number, the bit-check must be negated when passing to std::partition
		const uint8_t mask = uint8_t{1} << bit;
		auto bool_pred = [&bun_fun, mask](const T& val) {return !(bool)(bun_fun(val) & mask);};
		iterator_type mid = std::partition(data.begin(), data.end(), bool_pred);

		const uint8_t left_bin = bin;
		const uint8_t right_bin = bin | (uint8_t{1} << bit);

		if (left_bin < n_bins_) {
			recursive_partition_bit(std::span<T>{data.begin(), mid}, bit-1, left_bin, bun_fun);
		}
		
		if (right_bin < n_bins_) {
			recursive_partition_bit(std::span<T>{mid, data.end()}, bit-1, right_bin, std::forward<BinFun>(bun_fun));
		}
	}

	template<typename T>
	template<typename BinFun> requires(std::invocable_r_v<uint8_t, BinFun, T>)
	void BinSort<T>::recursive_partition_bit_parallel(std::span<T> data, uint8_t bit, uint8_t bin, BinFun&& bun_fun) noexcept {
		GUTIL_ASSERT(threads && threads->n_threads()>0);

		if (bit<0) {
			GUTIL_ASSERT(0<=bin && bin<=n_bins_);
			bins[bin] = data.begin();
			bins[bin+1] = data.end();
			return;
		}

		//construct the bool predicate and partition. note that because we wish to sort by
		//increasing bin number, the bit-check must be negated when passing to std::partition
		const uint8_t mask = uint8_t{1} << bit;
		auto bool_pred = [&bun_fun, mask](const T& val) {return !(bool)(bun_fun(val) & mask);};
		iterator_type mid = std::partition(data.begin(), data.end(), bool_pred);

		const uint8_t left_bin = bin;
		const uint8_t right_bin = bin | (uint8_t{1} << bit);

		if (left_bin < n_bins_) {
			threads->submit( [&](std::span<T> d, uint8_t bt, uint8_t bn, std::decay_t<BinFun> pred) noexcept {recursive_partition_bit_parallel(d,bt,bn,pred);},
						std::span<T>{data.begin(), mid}, bit-1, left_bin, bun_fun );
		}
		
		if (right_bin < n_bins_) {
			recursive_partition_bit_parallel(std::span<T>{mid, data.end()}, bit-1, right_bin, std::forward<BinFun>(bun_fun));
		}
	}
}
