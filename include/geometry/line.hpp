#pragma once

#include "utility/utility.hpp"
#include "math/math.hpp"
#include "geometry/point.hpp"

#include <iostream>
#include <cassert>
#include <functional>


namespace gutil
{
	template<typename T>
	concept IsLine = GeometryObject<T> && std::same_as<T, Line<T::DIMENSION, typename T::scalar_type>>;

	template<typename T>
	concept IsRay = GeometryObject<T> && std::same_as<T, Ray<T::DIMENSION, typename T::scalar_type>>;

	//////////////////////////////////////////////////////////
	/// A class for lines in cartesian space
	///
	/// @tparam DIM   The spacial dimension. These points are allocated
	///                   on the stack, so DIM shouldn't be too large.
	/// @tparam T     The underlying numeric type
	///
	/// Note that the type T must implement any comparisons if
	/// floating point rounding is important and unacceptable.
	//////////////////////////////////////////////////////////
	template<int DIM, IsReal T> requires (DIM>0)
	struct Line
	{
		////////////////////////////////////////////////////////////////
		// Data and aliases
		////////////////////////////////////////////////////////////////
		using scalar_type = T;
		using point_type = Point<DIM,T>;

		point_type direction;
		point_type origin;

		static constexpr int DIMENSION = DIM;

		////////////////////////////////////////////////////////////////
		// Constructors and move/copy
		////////////////////////////////////////////////////////////////
		constexpr Line() noexcept {} //note direction and origin are not initialized
		constexpr Line(const Line& other) noexcept = default;
		constexpr Line(Line&& other) noexcept = default;
		constexpr Line& operator=(const Line& other) noexcept = default;
		constexpr Line& operator=(Line&& other) noexcept = default;
		
		constexpr Line(const point_type& dir, const point_type& orig) : 
			direction{dir}, 
			origin{orig} {}
		
		explicit constexpr Line(const point_type& dir) : 
			direction{dir}, 
			origin{point_type::Zeros()} {}
		
		[[nodiscard]] constexpr point_type at(const T t) const noexcept {
			return origin + t*direction;
		}

		Line& normalize() noexcept {
			direction = gutil::normalized(direction);
			return *this;
		}

		Line reciprocal() const {
			return {T{1}/direction, origin};
		}
	};


	/////////////////////////////////////////////////////////////////////////////
	/// Define a Ray class that has a different type but behaves like a Line.
	/// Methods in gutilmath.hpp may make a distinction.
	/////////////////////////////////////////////////////////////////////////////
	template<int DIM, IsReal T> requires (DIM>0)
	struct Ray : Line<DIM,T> {
		using BASE = Line<DIM,T>;
		using point_type = typename BASE::point_type;
		using BASE::DIMENSION;
		using BASE::direction;
		using BASE::origin;
		using BASE::BASE;
		
		[[nodiscard]] constexpr point_type at(const T t) const noexcept {
			assert(t>=T{0});
			return BASE::at(t);
		}

		Ray& normalize() noexcept {
			direction = gutil::normalized(direction);
			return *this;
		}

		Ray reciprocal() const {
			return {T{1}/direction, origin};
		}
	};

	/////////////////////////////////////////////////////////////////////////////
	//////////////////////////// UTILITY OPERATIONS /////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<int DIM, IsReal T> requires (DIM>0)
	std::ostream& operator<<(std::ostream& os, const Line<DIM,T>& line) {
		return os << "Line{direction= {" << line.direction << "}, origin={" << line.origin << "}}";
	}

	template<typename T> requires (gutil::IsLine<T> || gutil::IsRay<T>)
	[[nodiscard]] inline constexpr bool lexicographic_less(const T& left, const T& right) noexcept {
		if (gutil::lexicographic_less(left.direction, right.direction)) {return true;}
		if (gutil::lexicographic_less(right.direction, left.direction)) {return false;}
		return gutil::lexicographic_less(left.origin, right.origin);
	}
}

namespace std {
	//inject the hash into std
	template<typename T> requires (gutil::IsLine<T> || gutil::IsRay<T>)
	struct hash<T> {
		[[nodiscard]] size_t operator()(const T& key) const noexcept{
			size_t seed{0};
			gutil::hash_combine(seed, key.direction);
			gutil::hash_combine(seed, key.origin);
			return seed;
		}
	};
}