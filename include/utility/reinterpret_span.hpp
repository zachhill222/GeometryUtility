#pragma once

#include <type_traits>
#include <concepts>
#include <iterator>
#include <span>

namespace gutil {
	//////////////////////////////////////////////////////////
	/// A helper to provide a necessary (but not necessarily sufficient)
	/// condition on when it is ok to reinterpret some contiguous container
	/// as a span of another type. This can be helpful when bitpacking
	/// data into an e.g., uint64_t.
	//////////////////////////////////////////////////////////
	template<typename A, typename B>
	concept LayoutCompatible = (sizeof(A)==sizeof(B)) && (alignof(A)==alignof(B));

	template<typename A, typename B>
	concept LayoutCompatibleIterators = std::contiguous_iterator<A> && std::contiguous_iterator<B> 
					&& LayoutCompatible<typename std::iter_value_t<A>, typename std::iter_value_t<B>>;

	template<typename ContainerA, typename ContainerB>
	concept LayoutCompatibleContainers = LayoutCompatibleIterators<typename ContainerA::iterator, typename ContainerB::iterator>;

	template<typename ContainerA, typename ContainerB>
	[[nodiscard]] inline bool containers_are_same_bytes(const ContainerA& A, const ContainerB& B) noexcept {
		if constexpr (LayoutCompatibleContainers<ContainerA,ContainerB>) {
			if (reinterpret_cast<uintptr_t>(A.data()) != reinterpret_cast<uintptr_t>(B.data())) {return false;}
			if (reinterpret_cast<uintptr_t>(A.data()+A.size()) != reinterpret_cast<uintptr_t>(B.data()+B.size())) {return false;}
			return true;
		}
		else {
			return false;
		}
	}


	// Reinterpret by iterator range
	template<typename T, std::contiguous_iterator I> requires LayoutCompatible<T, typename std::iter_value_t<I>>
														&& (!std::is_const_v<std::remove_reference_t<std::iter_reference_t<I>>>)
	[[nodiscard]] inline std::span<T> reinterpret_as_span(I begin, I end) noexcept {
		#ifndef NDEBUG
			using U = typename std::iter_value_t<I>;
			std::span<T> view{reinterpret_cast<T*>(std::to_address(begin)), reinterpret_cast<T*>(std::to_address(end))};
			GUTIL_ASSERT(containers_are_same_bytes(view, std::span<U>{begin, end}));
			return view;
		#else
		return std::span<T>{reinterpret_cast<T*>(std::to_address(begin)), reinterpret_cast<T*>(std::to_address(end))};
		#endif
	}

	template<typename T, std::contiguous_iterator I> requires LayoutCompatible<T, typename std::iter_value_t<I>>
														&& std::is_const_v<std::remove_reference_t<std::iter_reference_t<I>>>
	[[nodiscard]] inline std::span<const T> reinterpret_as_span(I begin, I end) noexcept {
		#ifndef NDEBUG
			using U = typename std::iter_value_t<I>;
			std::span<const T> view{reinterpret_cast<const T*>(std::to_address(begin)), reinterpret_cast<const T*>(std::to_address(end))};
			GUTIL_ASSERT(containers_are_same_bytes(view, std::span<const U>{begin, end}));
			return view;
		#else
		return std::span<const T>{reinterpret_cast<const T*>(std::to_address(begin)), reinterpret_cast<const T*>(std::to_address(end))};
		#endif
	}


	// Reinterpret by container type (disambiguating data immutability between something like a span<const T> and a const vector<T>& is tricky)
	template<typename T, typename U> requires LayoutCompatible<T, U> 
	[[nodiscard]] inline std::span<const T> reinterpret_as_span( std::span<const U> list) noexcept {
		return reinterpret_as_span<T>(list.begin(), list.end());
	}

	template<typename T, typename U> requires LayoutCompatible<T, U> 
	[[nodiscard]] inline std::span<T> reinterpret_as_span( std::span<U> list) noexcept {
		return reinterpret_as_span<T>(list.begin(), list.end());
	}

	template<typename T, typename Container> requires LayoutCompatible<T, typename Container::value_type> 	//container value type
											    && std::contiguous_iterator<typename Container::iterator>	//container is contiguous
	[[nodiscard]] inline std::span<const T> reinterpret_as_span(const Container& list) noexcept {
		return reinterpret_as_span<T>(list.begin(), list.end());
	}

	template<typename T, typename Container> requires LayoutCompatible<T, typename Container::value_type> 	//container value type
											    && std::contiguous_iterator<typename Container::iterator>	//container is contiguous
	[[nodiscard]] inline std::span<T> reinterpret_as_span(Container& list) noexcept {
		return reinterpret_as_span<T>(list.begin(), list.end());
	}
}