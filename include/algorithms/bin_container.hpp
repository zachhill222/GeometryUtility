#pragma once

#include "utility/utility.hpp"
#include "algorithms/sorting.hpp"

#include <vector>
#include <span>

namespace gutil {


	//////////////////////////////////////////////////////////////////////////
	/// In some instances, it is convenient to tie an array and its inverse
	/// or lookup together. The BinSort class provides an interface for this,
	/// but it only works with spans/view into data. This class owns its data
	/// and the sorting method so that data can be looked up in O(log (N/M)) time
	/// when the bins are sorted or O(N/M) time if the bins are not sorted,
	/// where N is the size of the data and M is the number of bins.
	///
	/// In order for the lookup to behave correctly, we need to pass the bin function.
	///////////////////////////////////////////////////////////////////////////
	template<typename T, auto BinFunc> requires(std::is_invocable_r_v<int,BinFunc,T>)
	struct BinContainer {
		std::vector<T> owned_data;
		BinSort<T> sorter;

		BinContainer() = default;

		BinContainer(std::span<const T> data_to_copy, const BinSort<T>& source_sorter)
			: owned_data(data_to_copy.begin(), data_to_copy.end()),
			  sorter(source_sorter)
		{
			GUTIL_ASSERT(owned_data.size() == source_sorter.size());

			//the BinSort view needs to be pointed towards the copied data
			sorter.rebind_to_copy(std::span<T>{owned_data});
		}

		BinContainer(const BinContainer& other) : owned_data(other.owned_data), sorter(other.sorter) {
			sorter.rebind_to_copy(std::span<T>{owned_data});
		}
		BinContainer(BinContainer&& other) noexcept : owned_data(std::move(other.owned_data)), sorter(std::move(other.sorter)) {
			sorter.rebind_to_copy(std::span<T>{owned_data});
		}
		BinContainer& operator=(const BinContainer& other) {
			owned_data = other.owned_data;
			sorter = other.sorter;
			sorter.rebind_to_copy(std::span<T>{owned_data});
			return *this;
		}
		BinContainer& operator=(BinContainer&& other) noexcept {
			owned_data = std::move(other.owned_data);
			sorter = std::move(other.sorter);
			sorter.rebind_to_copy(std::span<T>{owned_data});
			return *this;
		}

		//forward part of the  vector interface
		[[nodiscard]] size_t size() const noexcept { return owned_data.size(); }
		[[nodiscard]] T& operator[](size_t i) noexcept { return owned_data[i]; }
		[[nodiscard]] const T& operator[](size_t i) const noexcept { return owned_data[i]; }
		[[nodiscard]] auto begin() noexcept { return owned_data.begin(); }
		[[nodiscard]] auto end() noexcept { return owned_data.end(); }
		[[nodiscard]] auto begin() const noexcept { return owned_data.begin(); }
		[[nodiscard]] auto end() const noexcept { return owned_data.end(); }
		[[nodiscard]] T* data() noexcept { return owned_data.data(); }
		[[nodiscard]] const T* data() const noexcept { return owned_data.data(); }
		operator std::span<const T>() const noexcept { return owned_data; }

		//forward part of the BinSort interface
		[[nodiscard]] const BinSort<T>& as_binsort() const noexcept { return sorter; }
		[[nodiscard]] std::span<const T> get_bin(int i) const noexcept { return sorter.get_bin(i); }
		[[nodiscard]] size_t bin_start(int i) const noexcept { return sorter.bin_start(i); }

		//get the index to data
		[[nodiscard]] size_t index(const T& val) const noexcept {
			const int i = BinFunc(val);
			auto it = std::find(sorter.begin(i), sorter.end(i), val);
			if (it==sorter.end(i)) {return size_t(-1);}
			else {return bin_start(i) + std::distance(sorter.begin(i), it);}
		}

		[[nodiscard]] size_t index_sorted(const T& val) const noexcept {
			const int i = BinFunc(val);
			auto it = std::lower_bound(sorter.begin(i), sorter.end(i), val);
			if (it==sorter.end(i) || *it!=val) {return size_t(-1);}
			else {return bin_start(i) + std::distance(sorter.begin(i), it);}
		}

		//get an iterator to data
		[[nodiscard]] auto find(const T& val) const noexcept {
			const size_t idx = index(val);
			return (idx==size_t(-1)) ? owned_data.end() : owned_data.begin()+idx;
		}

		[[nodiscard]] auto find_sorted(const T& val) const noexcept {
			const size_t idx = index_sorted(val);
			return (idx==size_t(-1)) ? owned_data.end() : owned_data.begin()+idx;
		}
	};
}