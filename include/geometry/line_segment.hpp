#pragma once

#include "utility/utility.hpp"
#include "math/math.hpp"
#include "geometry/point.hpp"
#include "geometry/line.hpp"

#include <iostream>
#include <cassert>
#include <functional>







namespace gutil
{
	template<typename T>
	concept IsSegment = GeometryObject<T> && std::same_as<T, Segment<T::DIMENSION, typename T::scalar_type>>;

	//////////////////////////////////////////////////////////
	/// A class for line segments in cartesian space
	///
	/// @tparam DIM   The spacial dimension. These points are allocated
	///                   on the stack, so DIM shouldn't be too large.
	/// @tparam T     The underlying numeric type
	///
	/// Note that the type T must implement any comparisons if
	/// floating point rounding is important and unacceptable.
	//////////////////////////////////////////////////////////
	template<int DIM, IsReal T> requires (DIM>0)
	struct Segment
	{
		////////////////////////////////////////////////////////////////
		// Data and aliases
		////////////////////////////////////////////////////////////////
		using scalar_type = T;
		using point_type  = Point<DIM,T>;

		point_type start;
		point_type end;

		static constexpr int dim = DIM;

		////////////////////////////////////////////////////////////////
		// Constructors and move/copy
		////////////////////////////////////////////////////////////////
		constexpr Segment() noexcept {} //note start and end are not initialized
		constexpr Segment(const Segment& other) noexcept = default;
		constexpr Segment(Segment&& other) noexcept = default;
		constexpr Segment& operator=(const Segment& other) noexcept = default;
		constexpr Segment& operator=(Segment&& other) noexcept = default;
		
		constexpr Segment(const point_type& s, const point_type& e) : 
			start{s}, 
			end{e} {}
		
		[[nodiscard]] constexpr point_type at(const T t) const noexcept {
			assert(T{0}<=t && t <= T{1});
			return start + t * (end-start);
		}

		[[nodiscard]] constexpr operator gutil::Line<DIM,T>() const noexcept {
			return {end-start, start};
		}

		[[nodiscard]] constexpr operator gutil::Ray<DIM,T>() const noexcept {
			return {end-start, start};
		}

		[[nodiscard]] constexpr point_type direction() const noexcept {
			return end-start;
		}

		[[nodiscard]] T length() const noexcept {
			return gutil::distance(start, end);
		}
	};

	/////////////////////////////////////////////////////////////////////////////
	//////////////////////////// UTILITY OPERATIONS /////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<IsSegment T>
	std::ostream& operator<<(std::ostream& os, const T& line) {
		return os << "Segment{start= {" << line.start << "}, end={" << line.end << "}}";
	}

	template<IsSegment T>
	[[nodiscard]] inline constexpr bool lexicographic_less(const T& left, const T& right) noexcept {
		if (gutil::lexicographic_less(left.start, right.start)) {return true;}
		if (gutil::lexicographic_less(right.start, left.start)) {return false;}
		return gutil::lexicographic_less(left.end, right.end);
	}
}

namespace std {
	//inject the hash into std
	template<gutil::IsSegment T>
	struct hash<T> {
		[[nodiscard]] size_t operator()(const T& key) const noexcept{
			size_t seed{0};
			gutil::hash_combine(seed, key.start);
			gutil::hash_combine(seed, key.end);
			return seed;
		}
	};
}