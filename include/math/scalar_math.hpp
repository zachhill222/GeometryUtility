#pragma once

#include "utility/macros.hpp"
#include "utility/concepts.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>
#include <span>
#include <functional>

namespace gutil
{
	///////////////////////////////////////////////////////////
	/// Forward declare the various classes. Each class must
	/// satisfy the GeometryObject concept.
	///////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	struct Point;	//contains: T data[DIM]

	template<int DIM, IsScalar T> requires (DIM>0)
	struct Box;		//contains: Point<DIM,T> low, high

	template<int DIM, IsReal T> requires (DIM>0)
	struct Line;	//contains: Point<DIM,T> direction, origin

	template<int DIM, IsReal T> requires (DIM>0)
	struct Ray;	//contains: Point<DIM,T> direction, origin

	template<int DIM, IsReal T> requires (DIM>0)
	struct Segment;	//contains: Point<DIM,T> start, end


	///////////////////////////////////////////////////////////
	/////////////// CUSTOMIZATION POINT OBJECTS ///////////////
	///////////////////////////////////////////////////////////
	// use Customization Point Objects to ensure that scalar functions
	// go in the order of preference: user -> std -> fallback
	///////////////////////////////////////////////////////////
	namespace _cpo_
	{
		void sqrt() = delete;	//keep the function name lookup in _cpo_ from seeing gutil::sqrt
		struct sqrt_fn final {
			template<IsReal T>
			[[nodiscard]] GUTIL_STATIC_CALL T operator()(const T x) GUTIL_STATIC_CALL_CONST noexcept {
				if constexpr (requires { sqrt(x); }) {
					//ADL user or std::sqrt
					return sqrt(x);
				}
				else if constexpr (std::is_convertible_v<T,double>) {
					//fallback to convert to double and use std::sqrt
					return static_cast<T>(std::sqrt(static_cast<double>(x)));
				}
				else {
					//throw a compile error
					static_assert(always_false_v<T>, "gutil::sqrt - no function found");
				}
			}
		};

		void cos() = delete;
		struct cos_fn final {
			template<IsReal T>
			[[nodiscard]] GUTIL_STATIC_CALL T operator()(const T x) GUTIL_STATIC_CALL_CONST noexcept {
				if constexpr (requires { cos(x); }) {
					//ADL user or std::cos
					return cos(x);
				}
				else if constexpr (std::is_convertible_v<T,double>) {
					//fallback to convert to double and use std::cos
					return static_cast<T>(std::cos(static_cast<double>(x)));
				}
				else {
					//throw a compile error
					static_assert(always_false_v<T>, "gutil::cos - no function found");
				}
			}
		};

		void sin() = delete;
		struct sin_fn final {
			template<IsReal T>
			[[nodiscard]] GUTIL_STATIC_CALL T operator()(const T x) GUTIL_STATIC_CALL_CONST noexcept {
				if constexpr (requires { sin(x); }) {
					//ADL user or std::sin
					return sin(x);
				}
				else if constexpr (std::is_convertible_v<T,double>) {
					//fallback to convert to double and use std::sin
					return static_cast<T>(std::sin(static_cast<double>(x)));
				}
				else {
					//throw a compile error
					static_assert(always_false_v<T>, "gutil::sin - no function found");
				}
			}
		};

		void tan() = delete;
		struct tan_fn final {
			template<IsReal T>
			[[nodiscard]] GUTIL_STATIC_CALL T operator()(const T x) GUTIL_STATIC_CALL_CONST noexcept {
				if constexpr (requires { tan(x); }) {
					//ADL user or std::tan
					return tan(x);
				}
				else if constexpr (std::is_convertible_v<T,double>) {
					//fallback to convert to double and use std::tan
					return static_cast<T>(std::tan(static_cast<double>(x)));
				}
				else {
					//throw a compile error
					static_assert(always_false_v<T>, "gutil::tan - no function found");
				}
			}
		};

		void atan2() = delete;
		struct atan2_fn final {
			template<IsReal T>
			[[nodiscard]] GUTIL_STATIC_CALL T operator()(const T y, const T x) GUTIL_STATIC_CALL_CONST noexcept {
				if constexpr (requires { atan2(y,x); }) {
					//ADL user or std::atan2
					return atan2(y,x);
				}
				else if constexpr (std::is_convertible_v<T,double>) {
					//fallback to convert to double and use std::atan2
					return static_cast<T>(std::atan2(static_cast<double>(y), static_cast<double>(x)));
				}
				else {
					//throw a compile error
					static_assert(always_false_v<T>, "gutil::atan2 - no function found");
				}
			}
		};

		void fma() = delete;	//keep the function name lookup in _cpo_ from seeing std::fma
		struct fma_fn final {
			template<IsScalar T>
			[[nodiscard]] GUTIL_STATIC_CALL constexpr T operator()(const T x, const T y, const T z) GUTIL_STATIC_CALL_CONST noexcept {
				if consteval {return fallback(x,y,z);}
				else {return cpo_fma(x,y,z);}
			}

			template<IsScalar T>
			[[nodiscard]] static constexpr T fallback(const T x, const T y, const T z) noexcept {
				return (x*y) + z;
			}

			template<IsScalar T>
			[[nodiscard]] static T cpo_fma(const T x, const T y, const T z) noexcept {
				if constexpr (requires { fma(x,y,z); }) {return fma(x,y,z);}
				else {return fallback(x,y,z);}
			}
		};
	}

	inline constexpr _cpo_::sqrt_fn sqrt{};
	inline constexpr _cpo_::fma_fn fma{};
	inline constexpr _cpo_::sin_fn sin{};
	inline constexpr _cpo_::cos_fn cos{};
	inline constexpr _cpo_::tan_fn tan{};
	inline constexpr _cpo_::atan2_fn atan2{};


	///////////////////////////////////////////////////////////
	/// Define a simple method to combine two hash results
	/// This allows us to inject hashes for Point, etc into std easily.
	/// This mimics boost's hash_combine function.
	/// std::hash<T>{}(key) must return a valid hash.
	///////////////////////////////////////////////////////////
	template<typename T>
	void hash_combine(size_t& seed, const T& key) noexcept {
		if constexpr (sizeof(size_t) == 4) {
			seed ^= std::hash<T>{}(key) + 0x9e3779b9U + (seed<<6) + (seed>>2);
		}
		else if constexpr (sizeof(size_t) == 8) {
			seed ^= std::hash<T>{}(key) + 0x9e3779b97f4a7c15ULL + (seed<<6) + (seed>>2);
		}
		else {
			seed ^= std::hash<T>{}(key) + (seed<<6) + (seed>>2);
		}
	}

	///////////////////////////////////////////////////////////
	///////////////// SIMPLE SCALAR ROUTINES //////////////////
	///////////////////////////////////////////////////////////
	GUTIL_DECLARE_SIMD()
	template<IsScalar T>
	[[nodiscard]] inline constexpr T max(T x, T y) noexcept {
		return x > y ? x : y;
	}

	template<IsScalar T, typename... Ts> requires (std::same_as<T,Ts> && ...)
	[[nodiscard]] inline constexpr T max(T x, T y, Ts... rest) noexcept {
		return gutil::max( gutil::max(x,y), rest... );
	}

	GUTIL_DECLARE_SIMD()
	template<IsScalar T>
	[[nodiscard]] inline constexpr T min(T x, T y) noexcept {
		return x < y ? x : y;
	}

	template<IsScalar T, typename... Ts> requires (std::same_as<T,Ts> && ...)
	[[nodiscard]] inline constexpr T min(T x, T y, Ts... rest) noexcept {
		return gutil::min( gutil::min(x,y), rest... );
	}

	GUTIL_DECLARE_SIMD()
	template<IsScalar T>
	[[nodiscard]] inline constexpr T abs(T x) noexcept {
		return x < T{0} ? -x : x;
	}

	GUTIL_DECLARE_SIMD()
	template<IsScalar T>
	[[nodiscard]] inline constexpr T clamp(T x, T lo, T hi) noexcept {
		return x < lo ? lo : (x > hi ? hi : x);
	}


	///////////////////////////////////////////////////////////
	////////// ROUTINES ON SPANS THAT CAN BE REUSED ///////////
	///////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////
	/// Component-wise arithmetic
	///////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	inline constexpr void in_place_sum(std::span<T,DIM> left, std::span<const T,DIM> right) noexcept {
		for (int i=0; i<DIM; ++i) {
			left[i] += right[i];
		}
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	inline constexpr void in_place_subtract(std::span<T,DIM> left, std::span<const T,DIM> right) noexcept {
		for (int i=0; i<DIM; ++i) {
			left[i] -= right[i];
		}
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	inline constexpr void in_place_product(std::span<T,DIM> left, std::span<const T,DIM> right) noexcept {
		for (int i=0; i<DIM; ++i) {
			left[i] *= right[i];
		}
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	inline constexpr void in_place_divide(std::span<T,DIM> left, std::span<const T,DIM> right) noexcept {
		for (int i=0; i<DIM; ++i) {
			assert(right[i] != T{0});
			left[i] /= right[i];
		}
	}

	template<int DIM, IsInteger T> requires (DIM>0)
	inline constexpr void in_place_modulo(std::span<T,DIM> left, std::span<const T,DIM> right) noexcept {
		for (int i=0; i<DIM; ++i) {
			left[i] %= right[i];
		}
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	inline constexpr void in_place_negate(std::span<T,DIM> data) noexcept {
		for (int i=0; i<DIM; ++i) {
			data[i] = -data[i];
		}
	}


	template<int DIM, IsScalar T> requires (DIM>0)
	inline constexpr void in_place_sum(std::span<T,DIM> left, const T right) noexcept {
		for (int i=0; i<DIM; ++i) {
			left[i] += right;
		}
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	inline constexpr void in_place_subtract(std::span<T,DIM> left, const T right) noexcept {
		for (int i=0; i<DIM; ++i) {
			left[i] -= right;
		}
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	inline constexpr void in_place_product(std::span<T,DIM> left, const T right) noexcept {
		for (int i=0; i<DIM; ++i) {
			left[i] *= right;
		}
	}

	template<int DIM, IsReal T> requires (DIM>0)
	inline constexpr void in_place_division(std::span<T,DIM> left, const T right) noexcept {
		assert(right != T{0});
		const T r_inv = T{1}/right;
		gutil::in_place_product(left, r_inv);
	}

	template<int DIM, IsInteger T> requires (DIM>0)
	inline constexpr void in_place_division(std::span<T,DIM> left, const T right) noexcept {
		assert(right != T{0});
		for (int i=0; i<DIM; ++i) {
			left[i] /= right;
		}
	}

	template<int DIM, IsInteger T> requires (DIM>0)
	inline constexpr void in_place_modulo(std::span<T,DIM> left, const T right) noexcept {
		for (int i=0; i<DIM; ++i) {
			left[i] %= right;
		}
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	inline constexpr void in_place_clamp(std::span<T,DIM> data, std::span<const T,DIM> lo, std::span<const T,DIM> hi) noexcept {
		for (int i=0; i<DIM; ++i) {
			data[i] = gutil::clamp(data[i], lo[i], hi[i]);
		}
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	inline constexpr void in_place_clamp(std::span<T,DIM> data, T lo, T hi) noexcept {
		for (int i=0; i<DIM; ++i) {
			data[i] = gutil::clamp(data[i], lo, hi);
		}
	}


	///////////////////////////////////////////////////////////
	/// Comparisons
	///////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool cone_compare_less_than(std::span<const T,DIM> left, std::span<const T,DIM> right) noexcept {
		// return true if left[i] < right[i] for all i
		for (int i=0; i<DIM; ++i) {
			if (left[i] >= right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool cone_compare_less_than_equal(std::span<const T,DIM> left, std::span<const T,DIM> right) noexcept {
		// return true if left[i] <= right[i] for all i
		for (int i=0; i<DIM; ++i) {
			if (left[i] > right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool cone_compare_greater_than(std::span<const T,DIM> left, std::span<const T,DIM> right) noexcept {
		// return true if left[i] > right[i] for all i
		for (int i=0; i<DIM; ++i) {
			if (left[i] <= right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool cone_compare_greater_than_equal(std::span<const T,DIM> left, std::span<const T,DIM> right) noexcept {
		// return true if left[i] >= right[i] for all i
		for (int i=0; i<DIM; ++i) {
			if (left[i] < right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool elements_equal(std::span<const T,DIM> left, std::span<const T,DIM> right) noexcept {
		// return true if left[i] == right[i] for all i
		for (int i=0; i<DIM; ++i) {
			if (left[i] != right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool lexicographic_less_than(std::span<const T,DIM> left, std::span<const T,DIM> right) noexcept {
		// return true if there exists an I such that left[I] < right[I] and left[i] == right[i] for all i<I
		for (int i=0; i<DIM; ++i) {
			if (left[i] < right[i]) {return true;}
			if (right[i] < left[i]) {return false;}
		}
		return false;
	}

	///////////////////////////////////////////////////////////
	/// Reductions and norms
	///////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T sum_reduce(std::span<const T,DIM> data) noexcept {
		T result{data[0]};
		for (int i=1; i<DIM; ++i) {
			result += data[i];
		}
		return result;
	}

	GUTIL_NO_ASSOC_MATH_START
	template<int DIM, IsScalar T, IsScalar U=T> requires (DIM>0)
	[[nodiscard]] inline constexpr T kahan_sum_reduce(std::span<const T,DIM> data) noexcept {
		U aa{0};	//accumulator
		U cc{0};	//compensation
		U yy, tt;	//intermediates

		for (int i=0; i<DIM; ++i) {
			yy = static_cast<U>(data[i]) - cc;
			tt = aa + yy;
			cc = (tt - aa) - yy;
			aa = tt;
		}
		return static_cast<T>(aa);
	}
	GUTIL_NO_ASSOC_MATH_START

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T product_reduce(std::span<const T,DIM> data) noexcept {
		T result{data[0]};
		for (int i=1; i<DIM; ++i) {
			result *= data[i];
		}
		return result;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T max_reduce(std::span<const T,DIM> data) noexcept {
		T result{data[0]};
		for (int i=1; i<DIM; ++i) {
			result = gutil::max(result, data[i]);
		}
		return result;
	}

	/// get the minimum value
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T min_reduce(std::span<const T,DIM> data) noexcept {
		T result{data[0]};
		for (int i=1; i<DIM; ++i) {
			result = gutil::min(result, data[i]);
		}
		return result;
	}

	/// get the minimum absolute value
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T min_abs_reduce(std::span<const T,DIM> data) noexcept {
		T result{gutil::abs(data[0])};
		for (int i=1; i<DIM; ++i) {
			result = gutil::min(result, gutil::abs(data[i]));
		}
		return result;
	}

	/// get the dot product
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T dot_product_reduce(std::span<const T,DIM> left, std::span<const T,DIM> right) noexcept {
		T result{left[0]*right[0]};
		for (int i=1; i<DIM; ++i) {
			result = gutil::fma(left[i], right[i], result);
		}
		return result;
	}

	/// infinity norm
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T norminfty(std::span<const T,DIM> data) noexcept {
		T result{gutil::abs(data[0])};
		for (int i=1; i<DIM; ++i) {
			result = gutil::max(result, gutil::abs(data[i]));
		}
		return result;
	}

	/// 1-norm
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T norm1(std::span<const T,DIM> data) noexcept {
		T result{gutil::abs(data[0])};
		for (int i=1; i<DIM; ++i) {
			result += gutil::abs(data[i]);
		}
		return result;
	}

	/// squared 2-norm
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T squared_norm(std::span<const T,DIM> data) noexcept {
		T result{data[0]*data[0]};
		for (int i=1; i<DIM; ++i) {
			result = gutil::fma(data[i], data[i], result);
		}
		return result;
	}

	/// 2-norm
	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline T norm2(std::span<const T,DIM> data) noexcept {
		return gutil::sqrt(gutil::squared_norm(data));
	}

	/// sum of squared differences/errors
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T sum_square_error(std::span<const T,DIM> left, std::span<const T,DIM> right) noexcept {
		T result{0};
		for (int i=0; i<DIM; ++i) {
			const T dd = left[i] - right[i];
			result = gutil::fma(dd, dd, result);
		}
		return result;
	}

	///////////////////////////////////////////////////////////
	/// Element-wise operations
	///////////////////////////////////////////////////////////	
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> elmin(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result;
		for (int i=0; i<DIM; i++) {result.data[i] = gutil::min(left.data[i], right.data[i]);}
		return result;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> elmax(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result;
		for (int i=0; i<DIM; i++) {result.data[i] = gutil::max(left.data[i], right.data[i]);}
		return result;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> clamp(const Point<DIM,T>& p, const Point<DIM,T>& lo, const Point<DIM,T>& hi) noexcept {
		assert(lo<=hi);
		Point<DIM,T> result;
		for (int i=0; i<DIM; i++) {result.data[i] = gutil::clamp(p.data[i], lo.data[i], hi.data[i]);}
		return result;
	}

	///////////////////////////////////////////////////////////
	/// Geometry operations
	///////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T dot(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		return gutil::dot_product_reduce<DIM,T>(left.data, right.data);
	}

	template<IsScalar T>
	[[nodiscard]] inline constexpr Point<3,T> cross(const Point<3,T>& left, const Point<3,T>& right) noexcept {
		Point<3,T> result;
		result[0] = left.data[1]*right.data[2] - left.data[2]*right.data[1];
		result[1] = left.data[2]*right.data[0] - left.data[0]*right.data[2];
		result[2] = left.data[0]*right.data[1] - left.data[1]*right.data[0];
		return result;
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline Point<DIM,T> normalized(const Point<DIM,T>& vec) noexcept {
		const T nn = gutil::norm2(vec);
		assert(nn>T{0} && "normalized: input was the zero vector");
		return vec/nn;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Point<DIM,T>& A, const Point<DIM,T>& B) noexcept {
		return gutil::sum_square_error<DIM,T>(A.data, B.data);
	}

	///////////////////////////////////////////////////////////
	///////////////////// BOX OPERATIONS //////////////////////
	///////////////////////////////////////////////////////////
	
	/// Return 'union' of two boxes (minimal box that contains both inputs)
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Box<DIM,T> merge(const Box<DIM,T>& A, const Box<DIM,T>& B) noexcept {
		return {gutil::elmin(A.low, B.low), gutil::elmax(A.high, B.high)};
	}

	/// Return the smallest box the contains the given box and point
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Box<DIM,T> expand(const Box<DIM,T>& box, const Point<DIM,T>& point) noexcept {
		return {gutil::elmin(box.low, point), gutil::elmax(box.high, point)};
	}

	/// Check if the boxes intersect
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool intersect(const Box<DIM,T>& A, const Box<DIM,T>& B) noexcept {
		for (int i = 0; i < DIM; i++) {
			if (A.high.data[i] < B.low.data[i] || B.high.data[i] < A.low.data[i]) {
				return false;
			}
		}
		return true;
	}

	/// Return intersection of two boxes (undefined if boxes don't intersect)
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Box<DIM,T> intersection(const Box<DIM,T>& A, const Box<DIM,T>& B) noexcept {
		assert(gutil::intersect(A,B));
		return {gutil::elmax(A.low, B.low), gutil::elmin(A.high, B.high)};
	}

	/// Project/clamp a point to a box
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> clamp(const Point<DIM,T>& point, const Box<DIM,T>& box) noexcept {
		return gutil::clamp(point, box.low, box.high);
	}

	/// Project/clamp a point to a box periodically
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> clamp_periodic(const Point<DIM,T>& point, const Box<DIM,T>& box) noexcept {
		if constexpr (IsInteger<T>) {
			return box.low + (point - box.low)%(box.high - box.low);
		}
		else {
			return box.low + (point - box.low)/(box.high - box.low);
		}
	}

	/// Return the squared distance from a point to the box.
	/// The distance is 0 if the point is contained in the box.
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Box<DIM,T>& A, const Point<DIM,T>& B) noexcept {
		// clamp the point to the box, then get the distance
		return gutil::distance_squared(gutil::clamp(B, A), B);
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Box<DIM,T>& A, const Box<DIM,T>& B) noexcept {
		// record the gap (if any) along each axis and then sum the squares.
		//		[   A - axis[i]    ] <---- gap = B.low - A.high ----> [    B - axis[i]   ]
		T dd{0};
		for (int i=0; i<DIM; ++i) {
			const T gap = gutil::max(T{0}, A.low.data[i] - B.high.data[i], B.low.data[i] - A.high.data[i]);
			dd = gutil::fma(gap, gap, dd);
		}
		return dd;
	}


	///////////////////////////////////////////////////////////
	///////////////// LINE AND RAY OPERATIONS /////////////////
	///////////////////////////////////////////////////////////

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
	[[nodiscard]] inline constexpr T closest_t(const Segment<DIM,T>& seg, const Point<DIM,T>& point) noexcept {
		const Point<DIM,T> dir = seg.direction();
		const T dd = gutil::dot(point - seg.start, dir);
		if (dd < T{0}) {return T{0};}
		return gutil::min(T{1}, dd / gutil::squared_norm(dir));
	}


	/// Get the closest point on the line/ray to the given point
	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> closest_point(const Line<DIM,T>& line, const Point<DIM,T>& point) noexcept {
		return line.at(gutil::closest_t(line, point));
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> closest_point(const Ray<DIM,T>& ray, const Point<DIM,T>& point) noexcept {
		return ray.at(gutil::closest_t(ray, point));
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> closest_point(const Segment<DIM,T>& seg, const Point<DIM,T>& point) noexcept {
		return seg.at(gutil::closest_t(seg, point));
	}


	/// Get the minimal (squared) distance from the point to the line/ray
	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Line<DIM,T>& A, const Point<DIM,T>& B) noexcept {
		return gutil::distance_squared(gutil::closest_point(A,B), B);
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Ray<DIM,T>& A, const Point<DIM,T>& B) noexcept {
		return gutil::distance_squared(gutil::closest_point(A,B), B);
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Segment<DIM,T>& A, const Point<DIM,T>& B) noexcept {
		return gutil::distance_squared(gutil::closest_point(A,B), B);
	}


	/// Get the minimal (squared) distance between lines/rays
	/// As a helper, get the parameter values for the points on each line that are closest
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
	[[nodiscard]] inline constexpr T distance_squared(const Line<DIM,T>& A, const Line<DIM,T>& B) noexcept {
		T s, t;
		gutil::closest_st(A, B, s, t);
		return gutil::distance_squared(A.at(s), B.at(t));
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Line<DIM,T>& A, const Ray<DIM,T>& B) noexcept {
		T s, t;
		gutil::closest_st(A, static_cast<Line<DIM,T>>(B), s, t);
		
		//clamp ray
		if (t < T{0}) {return gutil::distance_squared(A, B.origin);}
		
		return gutil::distance_squared(A.at(s), B.at(t));
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
	[[nodiscard]] inline constexpr T distance_squared(const Ray<DIM,T>& A, const Ray<DIM,T>& B) noexcept {
		T s, t;
		gutil::closest_st(static_cast<Line<DIM,T>>(A), static_cast<Line<DIM,T>>(B), s, t);
		
		//clamp both rays
		if (t < T{0}) {return gutil::distance_squared(A, B.origin);}
		if (s < T{0}) {return gutil::distance_squared(B, A.origin);}

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



	/// Determine if a line/ray intersects a given box and (optionally) return the parameter values for the 
	/// first and last intersection points. It is assumed that many of these operations will be performed,
	/// so the reciprocal of the ray/line direction is needed. You may call something like:
	///		Ray recip = query_ray.reciprocal()
	/// 	intersect(boxA, recip)
	/// 	intersect(boxB, recip)
	
	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool intersect(const Box<DIM,T>& box, const Line<DIM,T>& inv_line, T& t_enter, T& t_exit) noexcept {
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
	[[nodiscard]] inline constexpr bool intersect(const Box<DIM,T>& box, const Ray<DIM,T>& inv_ray, T& t_enter, T& t_exit) noexcept {
		const bool hits = intersect(box, static_cast<Line<DIM,T>>(inv_ray), t_enter, t_exit);
		if (!hits) {return false;}
		if (t_exit < T{0}) {return false;}

		t_enter = gutil::max(T{0}, t_enter);
		return true;
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool intersect(const Box<DIM,T>& box, const Segment<DIM,T>& seg, T& t_enter, T& t_exit) noexcept {
		const bool hits = intersect(box, static_cast<Line<DIM,T>>(seg).reciprocal(), t_enter, t_exit);
		if (!hits) {return false;}
		if (t_exit < T{0} || t_enter > T{1}) {return false;}

		t_enter = gutil::max(T{0}, t_enter);
		t_exit = gutil::min(T{1}, t_exit);
		return true;
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool intersect(const Box<DIM,T>& box, const Line<DIM,T>& inv_line) noexcept {
		T t_enter, t_exit;
		return intersect(box, inv_line, t_enter, t_exit);
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool intersect(const Box<DIM,T>& box, const Ray<DIM,T>& inv_ray) noexcept {
		T t_enter, t_exit;
		return intersect(box, inv_ray, t_enter, t_exit);
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool intersect(const Box<DIM,T>& box, const Segment<DIM,T>& seg) noexcept {
		T t_enter, t_exit;
		return intersect(box, seg, t_enter, t_exit);
	}


	///////////////////////////////////////////////////////////
	////////////// TEMPLATE FUNCTION DEFINITIONS //////////////
	///////////////////////////////////////////////////////////
	/// Use CPO to naturally extend distance/distance_squared methods,
	/// but only if a function with the matching signature does not exist.

	template<typename T, typename U>
	concept HasDistanceSquared = requires(const T& A, const U& B) {
		{ distance_squared(A,B) } -> IsScalar;
	};

	template<typename T, typename U>
	concept HasDistance = requires(const T& A, const U& B) {
		{ distance(A,B) } -> IsScalar;
	};


	namespace _cpo_
	{
		void distance_squared() = delete;
		struct distance_squared_fn final {
			template<GeometryObject T, GeometryObject U> requires CompatibleGeometryObjects<T,U>
			[[nodiscard]] GUTIL_STATIC_CALL constexpr typename T::scalar_type operator()(const T& A, const U& B) GUTIL_STATIC_CALL_CONST noexcept {
				if constexpr (HasDistanceSquared<T,U>) {
					//ADL found a matching function
					return distance_squared(A,B);
				}
				else if constexpr (!std::same_as<T,U> && HasDistanceSquared<U,T>) {
					//ADL found a match when the arguments are swapped
					return distance_squared(B,A);
				}
				else {
					static_assert(always_false_v<T>, "gutil::distance_squared - no function found.");
				}
			}
		};

		void distance() = delete;
		struct distance_fn final {
			template<GeometryObject T, GeometryObject U> requires (CompatibleGeometryObjects<T,U> && IsReal<typename T::scalar_type>)
			[[nodiscard]] GUTIL_STATIC_CALL typename T::scalar_type operator()(const T& A, const U& B) GUTIL_STATIC_CALL_CONST noexcept {
				if constexpr (HasDistance<T,U>) {
					//ADL found a matching function
					return distance(A,B);
				}
				else if constexpr (!std::same_as<T,U> && HasDistance<U,T>) {
					//ADL found a match when the arguments are swapped
					return distance(B,A);
				}
				else if constexpr (HasDistanceSquared<T,U>) {
					//ADL found a match for distance squared
					return gutil::sqrt(distance_squared(A,B));
				}
				else if constexpr (HasDistanceSquared<U,T>) {
					//ADL found a match for distance squared with the args reversed
					return gutil::sqrt(distance_squared(B,A));
				}
				else {
					static_assert(always_false_v<T>, "gutil::distance - no function found.");
				}
			}
		};
	}

	inline constexpr _cpo_::distance_squared_fn distance_squared_extended{};	//don't name distance_squared so HasDistanceSquared<> can't see it
	inline constexpr _cpo_::distance_fn distance{};

	/// Define a lexicographic ordering for use in std::sort and such.
	template<GeometryObject T> requires(requires(const T& a, const T& b) { lexicographic_less(a,b); })
	struct LexicographicLess {
		[[nodiscard]] GUTIL_STATIC_CALL constexpr bool operator()(const T& left, const T& right) GUTIL_STATIC_CALL_CONST noexcept {
			return lexicographic_less(left, right);
		}
	};
}