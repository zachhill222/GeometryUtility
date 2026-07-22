#pragma once

#include "utility/macros.hpp"
#include "utility/concepts.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>
#include <span>
#include <functional>
#include <type_traits>

namespace gutil {
	///////////////////////////////////////////////////////////
	/////////////// CUSTOMIZATION POINT OBJECTS ///////////////
	///////////////////////////////////////////////////////////
	// use Customization Point Objects to ensure that scalar functions
	// go in the order of preference: user -> std -> fallback
	///////////////////////////////////////////////////////////
	namespace _cpo_ {
		void sqrt() = delete;	//keep the function name lookup in _cpo_ from seeing gutil::sqrt
		struct sqrt_fn final {
			template<IsReal T>
			[[nodiscard]] GUTIL_STATIC_CALL T operator()(T x) GUTIL_STATIC_CALL_CONST noexcept {
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
			[[nodiscard]] GUTIL_STATIC_CALL T operator()(T x) GUTIL_STATIC_CALL_CONST noexcept {
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
			[[nodiscard]] GUTIL_STATIC_CALL T operator()(T x) GUTIL_STATIC_CALL_CONST noexcept {
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
			[[nodiscard]] GUTIL_STATIC_CALL T operator()(T x) GUTIL_STATIC_CALL_CONST noexcept {
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
			[[nodiscard]] GUTIL_STATIC_CALL T operator()(T y, T x) GUTIL_STATIC_CALL_CONST noexcept {
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

		/// fused multiply add
		void fma() = delete;	//keep the function name lookup in _cpo_ from seeing std::fma
		struct fma_fn final {
			template<IsScalar T>
			[[nodiscard]] GUTIL_STATIC_CALL constexpr T operator()(T x, T y, T z) GUTIL_STATIC_CALL_CONST noexcept {
				if consteval {return fallback(x,y,z);}
				else {return cpo_fma(x,y,z);}
			}

			template<IsScalar T>
			[[nodiscard]] static constexpr T fallback(T x, T y, T z) noexcept {
				return (x*y) + z;
			}

			template<IsScalar T>
			[[nodiscard]] static T cpo_fma(T x, T y, T z) noexcept {
				if constexpr (requires { fma(x,y,z); }) {return fma(x,y,z);}
				else {return fallback(x,y,z);}
			}
		};

		/// floating point modulo
		void fmod() = delete;
		struct fmod_fn final {
			template<IsReal T>
			[[nodiscard]] GUTIL_STATIC_CALL constexpr T operator()(T num, T den) GUTIL_STATIC_CALL_CONST noexcept {
				if constexpr (requires { fmod(num,den); }) {
					return fmod(num,den);
				}
				else if constexpr (std::is_nothrow_convertible_v<T,double>) {
					return static_cast<T>(std::fmod(static_cast<double>(num), static_cast<double>(den)));
				}
				else {
					static_assert(always_false_v<T>, "gutil::fmod - no function found");
				}
			}
		};

		/// compute x * 2^a where x is a scalar and a is an integer
		void ldexp() = delete;
		struct ldexp_fn final {
			template<IsScalar X, IsInteger A>
			[[nodiscard]] GUTIL_STATIC_CALL constexpr X operator()(X x, A a) GUTIL_STATIC_CALL_CONST noexcept {
				if constexpr (requires { ldexp(x,a); }) {
					return ldexp(x,a);
				}
				else if constexpr (IsReal<X> && std::is_nothrow_convertible_v<X,double> && std::is_nothrow_convertible_v<A,int>) {
					return static_cast<X>( std::ldexp( static_cast<double>(x), static_cast<int>(a) ) );
				}
				else if constexpr (IsInteger<X> && std::is_nothrow_convertible_v<X,int> && std::is_nothrow_convertible_v<A,int>) {
					const int xx = static_cast<int>(x);
					const int aa = static_cast<int>(a);
					assert(aa>-32 && aa<32 && "gutil::ldexp - exponent is too large for cast to int fallback");
					const int result = (aa>0) ? xx*(1<<aa) : xx/(1<<-aa);
					return static_cast<X>(result);
				}
				else {
					static_assert(always_false_v<X>, "gutil::ldexp - no function found");
				}
			}
		};
	}

	inline constexpr _cpo_::sqrt_fn		sqrt{};
	inline constexpr _cpo_::fma_fn 		fma{};
	inline constexpr _cpo_::sin_fn 		sin{};
	inline constexpr _cpo_::cos_fn 		cos{};
	inline constexpr _cpo_::tan_fn 		tan{};
	inline constexpr _cpo_::atan2_fn	atan2{};
	inline constexpr _cpo_::fmod_fn 	fmod{};
	inline constexpr _cpo_::ldexp_fn 	ldexp{};


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

	template<int DIM, IsReal T> requires (DIM>0)
	inline constexpr void in_place_divide(std::span<T,DIM> left, const T right) noexcept {
		assert(right != T{0});
		const T r_inv = T{1}/right;
		gutil::in_place_product(left, r_inv);
	}

	template<int DIM, IsInteger T> requires (DIM>0)
	inline constexpr void in_place_divide(std::span<T,DIM> left, const T right) noexcept {
		assert(right != T{0});
		for (int i=0; i<DIM; ++i) {
			left[i] /= right;
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
	template<int DIM, IsScalar T, IsScalar U=T> requires (DIM>0) && std::is_nothrow_convertible_v<T,U>
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
	GUTIL_NO_ASSOC_MATH_END

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
	[[nodiscard]] inline constexpr T sse_reduce(std::span<const T,DIM> left, std::span<const T,DIM> right) noexcept {
		T result{0};
		for (int i=0; i<DIM; ++i) {
			const T dd = left[i] - right[i];
			result = gutil::fma(dd, dd, result);
		}
		return result;
	}


	///////////////////////////////////////////////////////////
	///////// TRIVIAL/SYMMETRIC DISTANCE DEFINITIONS //////////
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

	///////////////////////////////////////////////////////////
	//// TOTAL ORDERING AND HASHING FOR STL COMPATIBILITY /////
	///////////////////////////////////////////////////////////
	template<GeometryObject T> requires(requires(const T& a, const T& b) { lexicographic_less(a,b); })
	struct LexicographicLess {
		[[nodiscard]] GUTIL_STATIC_CALL constexpr bool operator()(const T& left, const T& right) GUTIL_STATIC_CALL_CONST noexcept {
			return lexicographic_less(left, right);
		}
	};

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
}