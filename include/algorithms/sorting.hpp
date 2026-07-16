#pragma once

#include "utility/utility.hpp"

#include <span>
#include <algorithm>
#include <thread>
#include <atomic>
#include <bit>

namespace gutil
{
	//////////////////////////////////////////////////////////
	/// A class for partitioning data in-place. A predicate of the type
	/// pred(value) -> int must be supplied with the return value (the 'bin')
	/// a number in [0,N). The algorithm uses divide-and-conquer
	/// by splitting the data into left/right partitions for each bit
	/// in the bin. An internal state is kept to more easily extract subspans.
	//////////////////////////////////////////////////////////
	template<typename T>
	struct BinSort
	{
	private:
		using iterator_type = typename std::span<T>::iterator;
		static constexpr int MAX_THREADS = 4;
		static constexpr int SPAWN_THREAD_THRESHOLD = 512;
		
		int n_bins;
		int n_bits;
		std::span<T> data;
		std::atomic<int> n_threads{0};
		std::vector<iterator_type> bins;

		template<typename Predicate>
		void recursive_partition_bit(std::span<T> data, int bit, int bin, Predicate&& int_pred) noexcept;
	public:
		BinSort(std::span<T> data, int N) : n_bins{N}, data(data), bins(N+1) {
			assert(N>0);
			n_bits = std::bit_width(static_cast<uint>(N-1));
		}
		
		/// Primary call (pass the full predicate to bin number)
		template<typename Predicate>
		void sort(Predicate&& int_pred) {
			n_threads = 1;
			//note std::bit_width requires an unsigned integer
			//if there are N bins, then bins.size() = N+1, the max bin index is N-1,
			//and the max bit index is bit_width(N-1) - 1
			recursive_partition_bit(data, n_bits-1, 0, std::forward<Predicate>(int_pred));
		}

		/// Once sorted, get a subspan into the requested bin
		[[nodiscard]] std::span<T> get_bin(int i) const noexcept {
			assert(0<=i && i<n_bins);
			assert( static_cast<size_t>(n_bins)+1 == bins.size() );
			return std::span<T>{bins[i], bins[i+1]};
		}

		[[nodiscard]] size_t bin_size(int i) const noexcept {
			assert(0<=i && i<n_bins);
			assert( static_cast<size_t>(n_bins)+1 == bins.size() );
			return static_cast<size_t>(std::distance(bins[i],bins[i+1]));
		}
	};

	template<typename T>
	template<typename Predicate>
	void BinSort<T>::recursive_partition_bit(std::span<T> data, int bit, int bin, Predicate&& int_pred) noexcept {
		#ifndef NDEBUG
		Logger::log("recursive_partition: size=",data.size()," bit= ",bit," bin= ",bin);
		#endif

		if (bit<0) {
			assert(0<=bin && bin<n_bins);
			bins[bin] = data.begin();
			bins[bin+1] = data.end();
			return;
		}

		//construct the bool predicate and partition. note that because we wish to sort by
		//increasing bin number, the bit-check must be negated when passing to std::partition
		const int mask = int{1} << bit;
		auto bool_pred = [&int_pred, mask](const T& val) {return !(bool)(int_pred(val) & mask);};
		iterator_type mid = std::partition(data.begin(), data.end(), bool_pred);

		//need to update bins here
		if (bit==0) {

		}


		const int left_bin = bin;
		const int right_bin = bin | (int{1} << bit);

		//recurse and spawn new thread if allowed
		const bool allow_spawn = std::min(std::distance(data.begin(),mid), std::distance(mid,data.end())) > SPAWN_THREAD_THRESHOLD;
		if ( allow_spawn && (n_threads < MAX_THREADS) ) {
			++n_threads;
			auto fun = [&]() {
				this->recursive_partition_bit(std::span<T>{data.begin(), mid}, bit-1, left_bin, int_pred);
			};
			auto t = std::thread(fun);
			recursive_partition_bit({mid, data.end()}, bit-1, right_bin, std::forward<Predicate>(int_pred));
			t.join(); --n_threads;
		}
		else {
			recursive_partition_bit(std::span<T>{data.begin(), mid}, bit-1, left_bin, int_pred);
			recursive_partition_bit(std::span<T>{mid, data.end()}, bit-1, right_bin, std::forward<Predicate>(int_pred));
		}
	}
}
