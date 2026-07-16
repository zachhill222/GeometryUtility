#pragma once

#include "utility/utility.hpp"
#include "math/math.hpp"

#include "geometry/point.hpp"
#include "geometry/box.hpp"

#include <iostream>
#include <cassert>
#include <functional>


namespace gutil
{
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


	///////////////////////////////////////////////////////////////////
	/// Ensure that the concepts 'IsLine' and 'IsRay' are valid
	///////////////////////////////////////////////////////////////////
	template<typename T>
	concept IsLine = GeometryObject<T> && std::same_as<T, Line<T::DIMENSION, typename T::scalar_type>>;

	template<typename T>
	concept IsRay = GeometryObject<T> && std::same_as<T, Ray<T::DIMENSION, typename T::scalar_type>>;

	static_assert(IsLine<Line<3,float>>);
	static_assert(IsLine<Line<2,float>>);
	static_assert(IsLine<Line<3,double>>);
	static_assert(IsLine<Line<2,double>>);
	static_assert(IsRay<Ray<3,float>>);
	static_assert(IsRay<Ray<2,float>>);
	static_assert(IsRay<Ray<3,double>>);
	static_assert(IsRay<Ray<2,double>>);


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


	/////////////////////////////////////////////////////////////////////////////
	////////////////// LINE/RAY/POINT MATH/GEOMETRY OPERATIONS //////////////////
	/////////////////////////////////////////////////////////////////////////////
	/// Get the input parameter (t) that produces the closest point on the line/ray to the given point
	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T closest_t(const Line<DIM,T>& line, const Point<DIM,T>& point) noexcept {
		const T dd = gutil::dot(point - line.origin, line.direction);
		return dd / gutil::squared_norm(line.direction);
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T closest_t(const Ray<DIM,T>& ray, const Point<DIM,T>& point) noexcept {
		const T dd = gutil::dot(point - ray.origin, ray.direction);
		return dd > T{0} ? dd / gutil::squared_norm(ray.direction) : T{0};
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> closest_point(const Line<DIM,T>& line, const Point<DIM,T>& point) noexcept {
		return line.at(gutil::closest_t(line, point));
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> closest_point(const Ray<DIM,T>& ray, const Point<DIM,T>& point) noexcept {
		return ray.at(gutil::closest_t(ray, point));
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Line<DIM,T>& A, const Point<DIM,T>& B) noexcept {
		return gutil::distance_squared(gutil::closest_point(A,B), B);
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Ray<DIM,T>& A, const Point<DIM,T>& B) noexcept {
		return gutil::distance_squared(gutil::closest_point(A,B), B);
	}


	/////////////////////////////////////////////////////////////////////////////
	/////////////////// LINE/RAY/BOX MATH/GEOMETRY OPERATIONS ///////////////////
	/////////////////////////////////////////////////////////////////////////////
	/// Determine if a line/ray collides a given box and (optionally) return the parameter values for the 
	/// first and last collideion points. It is assumed that many of these operations will be performed,
	/// so the reciprocal of the ray/line direction is needed. You may call something like:
	///		Ray recip = query_ray.reciprocal()
	/// 	collide(boxA, recip)
	/// 	collide(boxB, recip)
	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool collide(const Box<DIM,T>& box, const Line<DIM,T>& inv_line, T& t_enter, T& t_exit) noexcept {
		t_enter = Lowest<T>::value;
		t_exit = Max<T>::value;

		for (int i=0; i<DIM; ++i) {
			//t values for the two faces of 'slab' with normal axis i
			T t0 = (box.low.data[i] - inv_line.origin.data[i]) * inv_line.direction.data[i];
			T t1 = (box.high.data[i] - inv_line.origin.data[i]) * inv_line.direction.data[i];

			if (t1 < t0) {std::swap(t0, t1);}

			t_enter = gutil::max(t_enter, t0);
			t_exit = gutil::min(t_exit, t1);
		}

		return t_enter <= t_exit;
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool collide(const Box<DIM,T>& box, const Ray<DIM,T>& inv_ray, T& t_enter, T& t_exit) noexcept {
		const bool hits = collide(box, static_cast<Line<DIM,T>>(inv_ray), t_enter, t_exit);
		if (!hits) {return false;}
		if (t_exit < T{0}) {return false;}

		t_enter = gutil::max(T{0}, t_enter);
		return true;
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool collide(const Box<DIM,T>& box, const Line<DIM,T>& inv_line) noexcept {
		T t_enter, t_exit;
		return collide(box, inv_line, t_enter, t_exit);
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool collide(const Box<DIM,T>& box, const Ray<DIM,T>& inv_ray) noexcept {
		T t_enter, t_exit;
		return collide(box, inv_ray, t_enter, t_exit);
	}


	/////////////////////////////////////////////////////////////////////////////
	///////////////////// LINE/RAY MATH/GEOMETRY OPERATIONS /////////////////////
	/////////////////////////////////////////////////////////////////////////////
	/// Get the parameter values for the points on each line that are closest
	template<int DIM, IsReal T> requires (DIM>0)
	inline constexpr void closest_st(const Line<DIM,T>& A, const Line<DIM,T>& B, T& s, T& t) noexcept {
		//s is the parameter for line A and t is the parameter for line B

		//Algorithm 2.4.3 from 'Finite Element Mesh Generation' by Daniel S.H. Lo
		T aa{0}, bb{0}, cc{0}, ee{0}, ff{0};
		for (int i=0; i<DIM; ++i) {
			aa = gutil::fma(A.direction.data[i], A.direction.data[i], aa);	//aa = dot(A.dir, A.dir)
			bb = gutil::fma(B.direction.data[i], B.direction.data[i], bb);	//bb = dot(B.dir, B.dir)
			cc = gutil::fma(A.direction.data[i], B.direction.data[i], cc);	//cc = dot(A.dir, B.dir)
			
			const T ba_i = B.origin.data[i] - A.origin.data[i];
			ee = gutil::fma(A.direction.data[i], ba_i, ee);					//ee = dot(A.dir, B.orig-A.orig)
			ff = gutil::fma(B.direction.data[i], ba_i, ff);					//ff = dot(B.dir, B.orig-A.orig)
		}

		const T dd = aa*bb - cc*cc;											//non-negative (Cauchy-Schwarz: |A*B| <= |A|*|B|)

		//check if the lines are parallel (dd==0)
		constexpr T tol = (IsExact<T>) ? T{0} : T{4} * Epsilon<T>::value;
		if (dd <= tol) {
			//lines are parallel
			t = 0;	//origin of B
			s = closest_t(A, B.origin);
			return;
		}
		
		//lines are not parallel
		const T dd_inv = T{1}/dd;
		s = dd_inv * (bb*ee - cc*ff);
		t = dd_inv * (cc*ee - aa*ff);
	}

	template<int DIM, IsReal T> requires (DIM>0)
	inline constexpr void closest_st(const Line<DIM,T>& A, const Ray<DIM,T>& B, T& s, T& t) noexcept {
		gutil::closest_st(A, static_cast<Line<DIM,T>>(B), s, t);
		
		//clamp ray
		if (t < T{0}) {
			t = T{0};
			s = gutil::closest_t(A, B.origin);
			return;
		}
	}

	template<int DIM, IsReal T> requires (DIM>0)
	inline constexpr void closest_st(const Ray<DIM,T>& A, const Ray<DIM,T>& B, T& s, T& t) noexcept {
		gutil::closest_st(static_cast<Line<DIM,T>>(A), static_cast<Line<DIM,T>>(B), s, t);
		
		//clamp both rays
		if (t < T{0}) {
			t = T{0};
			s = gutil::closest_t(A, B.origin);
			return;
		}

		if (s < T{0}) {
			s = T{0};
			t =  gutil::closest_t(B, A.origin);
			return;
		}
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Line<DIM,T>& A, const Line<DIM,T>& B) noexcept {
		T s, t;
		gutil::closest_st(A, B, s, t);
		return gutil::distance_squared(A.at(s), B.at(t));
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Line<DIM,T>& A, const Ray<DIM,T>& B) noexcept {
		T s, t;
		gutil::closest_st(A, B, s, t);
		return gutil::distance_squared(A.at(s), B.at(t));
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Ray<DIM,T>& A, const Ray<DIM,T>& B) noexcept {
		T s, t;
		gutil::closest_st(A, B, s, t);
		return gutil::distance_squared(A.at(s), B.at(t));
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