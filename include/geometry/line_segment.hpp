#pragma once

#include "utility/utility.hpp"
#include "math/math.hpp"

#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "geometry/line.hpp"

#include <iostream>
#include <cassert>
#include <functional>







namespace gutil
{
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

		static constexpr int DIMENSION = DIM;

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


	///////////////////////////////////////////////////////////////////
	/// Ensure that the concept 'IsSegment' is valid
	///////////////////////////////////////////////////////////////////
	template<typename T>
	concept IsSegment = GeometryObject<T> && std::same_as<T, Segment<T::DIMENSION, typename T::scalar_type>>;

	static_assert(IsSegment<Segment<3,float>>);
	static_assert(IsSegment<Segment<2,float>>);
	static_assert(IsSegment<Segment<3,double>>);
	static_assert(IsSegment<Segment<2,double>>);


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


	/////////////////////////////////////////////////////////////////////////////
	////////////////// SEGMENT/POINT MATH/GEOMETRY OPERATIONS ///////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T closest_t(const Segment<DIM,T>& seg, const Point<DIM,T>& point) noexcept {
		const Point<DIM,T> dir = seg.direction();
		const T dd = gutil::dot(point - seg.start, dir);
		if (dd < T{0}) {return T{0};}
		return gutil::min(T{1}, dd / gutil::squared_norm(dir));
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> closest_point(const Segment<DIM,T>& seg, const Point<DIM,T>& point) noexcept {
		return seg.at(gutil::closest_t(seg, point));
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Segment<DIM,T>& A, const Point<DIM,T>& B) noexcept {
		return gutil::distance_squared(gutil::closest_point(A,B), B);
	}


	/////////////////////////////////////////////////////////////////////////////
	/////////////////// SEGMENT/BOX MATH/GEOMETRY OPERATIONS ////////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool collide(const Box<DIM,T>& box, const Segment<DIM,T>& seg, T& t_enter, T& t_exit) noexcept {
		const bool hits = collide(box, static_cast<Line<DIM,T>>(seg).reciprocal(), t_enter, t_exit);
		if (!hits) {return false;}
		if (t_exit < T{0} || t_enter > T{1}) {return false;}

		t_enter = gutil::max(T{0}, t_enter);
		t_exit = gutil::min(T{1}, t_exit);
		return true;
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool collide(const Box<DIM,T>& box, const Segment<DIM,T>& seg) noexcept {
		T t_enter, t_exit;
		return collide(box, seg, t_enter, t_exit);
	}


	/////////////////////////////////////////////////////////////////////////////
	//////////////// SEGMENT/LINE/RAY MATH/GEOMETRY OPERATIONS //////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<int DIM, IsReal T> requires (DIM>0)
	inline constexpr void closest_st(const Line<DIM,T>& A, const Segment<DIM,T>& B, T& s, T& t) noexcept {
		gutil::closest_st(A, static_cast<Line<DIM,T>>(B), s, t);
		
		//clamp line segment
		if (t < T{0}) {
			t = T{0};
			s = gutil::closest_t(A, B.start);
			return;
		}

		if (t > T{1}) {
			t = T{1};
			s = gutil::closest_t(A, B.end);
			return;
		}
	}

	template<int DIM, IsReal T> requires (DIM>0)
	inline constexpr void closest_st(const Ray<DIM,T>& A, const Segment<DIM,T>& B, T& s, T& t) noexcept {
		gutil::closest_st(static_cast<Line<DIM,T>>(A), static_cast<Line<DIM,T>>(B), s, t);
		
		//clamp line segment
		if (t < T{0}) {
			t = T{0};
			s = gutil::closest_t(A, B.start);
			return;
		}
		if (t > T{1}) {
			t = T{1};
			s = gutil::closest_t(A, B.end);
			return;
		}

		//clamp ray
		if (s < T{0}) {
			s = T{0};
			t = gutil::closest_t(B, A.origin);
			return;
		}
	}

	template<int DIM, IsReal T> requires (DIM>0)
	inline constexpr void closest_st(const Segment<DIM,T>& A, const Segment<DIM,T>& B) noexcept {
		T s, t;
		gutil::closest_st(static_cast<Line<DIM,T>>(A), static_cast<Line<DIM,T>>(B), s, t);
		
		//clamp both line segments
		if (t < T{0}) {
			t = T{0};
			s = gutil::closest_t(A, B.start);
			return;
		}
		if (t > T{1}) {
			t = T{1};
			s = gutil::closest_t(A, B.end);
			return;
		}
		if (s < T{0}) {
			s = T{0};
			t = gutil::closest_t(B, A.start);
			return;
		}
		if (s > T{1}) {
			s = T{1};
			t = gutil::closest_t(B, A.end);
			return;
		}
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Line<DIM,T>& A, const Segment<DIM,T>& B) noexcept {
		T s, t;
		gutil::closest_st(A, static_cast<Line<DIM,T>>(B), s, t);
		
		//clamp line segment
		if (t < T{0}) {return gutil::distance_squared(A, B.start);}
		if (t > T{1}) {return gutil::distance_squared(A, B.end);}

		return gutil::distance_squared(A.at(s), B.at(t));
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Ray<DIM,T>& A, const Segment<DIM,T>& B) noexcept {
		T s, t;
		gutil::closest_st(static_cast<Line<DIM,T>>(A), static_cast<Line<DIM,T>>(B), s, t);
		
		//clamp line segment
		if (t < T{0}) {return gutil::distance_squared(A, B.start);}
		if (t > T{1}) {return gutil::distance_squared(A, B.end);}

		//clamp ray
		if (s < T{0}) {return gutil::distance_squared(B, A.origin);}

		return gutil::distance_squared(A.at(s), B.at(t));
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Segment<DIM,T>& A, const Segment<DIM,T>& B) noexcept {
		T s, t;
		gutil::closest_st(static_cast<Line<DIM,T>>(A), static_cast<Line<DIM,T>>(B), s, t);
		
		//clamp both line segments
		if (t < T{0}) {return gutil::distance_squared(A, B.start);}
		if (t > T{1}) {return gutil::distance_squared(A, B.end);}
		if (s < T{0}) {return gutil::distance_squared(B, A.start);}
		if (s > T{1}) {return gutil::distance_squared(B, A.end);}

		return gutil::distance_squared(A.at(s), B.at(t));
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