#pragma once

#include <cmath>
#include <algorithm>
#include <type_traits>
#include <concepts>
#include <span>
#include <cassert>

//////////////////////////////////////////////////////
/// If OpenMP is present, some operations are provided with 
/// vectorized versions over spans.
//////////////////////////////////////////////////////
#ifdef _OPENMP
	#include <omp.h>
#endif




//////////////////////////////////////////////////////
/// Macro to disable floating point associativity for functions
/// that require exact arithmetic order.
//////////////////////////////////////////////////////
#if defined(__GNUC__)
	//linux (gcc) compiler
	#define GUTIL_NO_ASSOC_MATH_START _Pragma("GCC push_options") _Pragma("GCC optimize(\"no-associative-math\")")
	#define GUTIL_NO_ASSOC_MATH_END _Pragma("GCC pop_options")
#elif defined(_MSC_VER)
	//microsoft compiler
	#define GUTIL_NO_ASSOC_MATH_START __pragma(float_control(precise, on, push))
	#define GUTIL_NO_ASSOC_MATH_END __pragma(float_control(pop))
#else
	//unkown compiler
	#define GUTIL_NO_ASSOC_MATH_START
	#define GUTIL_NO_ASSOC_MATH_END
#endif

//////////////////////////////////////////////////////
/// Macro to use OpenMP pragmas when it is available
/// and supress when it is not.
//////////////////////////////////////////////////////
#ifdef _OPENMP
	#define GUTIL_IF_OPENMP(prag) _Pragma(prag)
#else
	#define GUTIL_IF_OPENMP(prag)
#endif

//////////////////////////////////////////////////////
/// Macro to to help Customization Point Objects have
/// static operator() methods when possible. CPO is used
/// for some functions (e.g., sqrt) so that the call order is
///		user supplied -> std::sqrt -> fallback (cast to double)
/// If operator() can't be static, it should be const, but
/// the two are mutually exclusive, so two macros are needed.
//////////////////////////////////////////////////////
#ifdef __cpp_static_call_operator
	#define GUTIL_STATIC_CALL static
	#define GUTIL_STATIC_CALL_CONST
#else
	#define GUTIL_STATIC_CALL
	#define GUTIL_STATIC_CALL_CONST const
#endif

namespace gutil
{
	///////////////////////////////////////////////////////////
	/// A small math library to provide some scalar operations and
	/// important operations for other data types:
	///		Point (point.hpp)	-- model points/vectors in space
	///		Box	(box.hpp)		-- model axis aligned bounding boxes (aabb)
	/// Some vectorization over spans is availible via OpenMP if supplied.
	///////////////////////////////////////////////////////////



	///////////////////////////////////////////////////////////
	/// Concept to ensure that a type is some form of scalar
	/// that emulates the real line and will work with the Point class.
	///
	/// If a custom scalar type (e.g., a fixed precision type), it
	/// must satisfy this concept. Note it should have a sqrt() function.
	/// If it doesn't, it should be castable to/from a double to use 
	/// std::sqrt as a fallback.
	///////////////////////////////////////////////////////////
	template<typename T>
	concept IsScalar = requires(T a, T b) {
		T{0};
		T{1};
		{ a + b } -> std::same_as<T>;
		{ a - b } -> std::same_as<T>;
		{ a * b } -> std::same_as<T>;
		{ a / b } -> std::same_as<T>;
		{ -a    } -> std::same_as<T>;
		{ a+=b  } -> std::same_as<T&>;
		{ a-=b  } -> std::same_as<T&>;
		{ a*=b  } -> std::same_as<T&>;
		{ a/=b  } -> std::same_as<T&>;
		{ a==b  } -> std::same_as<bool>;
		{ a<b   } -> std::same_as<bool>;
	};

	///////////////////////////////////////////////////////////
	/// Forward declare the various classes
	///////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	struct Point;

	template<int DIM, IsScalar T> requires (DIM>0)
	struct Box;

	///////////////////////////////////////////////////////////
	/// Use a templated type to trigger conditional compile
	/// errors (say no valid constexpr branch of an if statement
	/// was found).
	///////////////////////////////////////////////////////////
	template<typename...>
	inline constexpr bool always_false_v = false;


	///////////////////////////////////////////////////////////
	//////////////////// SCALAR OPERATIONS ////////////////////
	///////////////////////////////////////////////////////////
	// use Customization Point Objects to ensure that scalar functions
	// go in the order of preference: user -> std -> fallback
	///////////////////////////////////////////////////////////
	namespace _cpo_
	{

		void sqrt() = delete;	//keep the function name lookup in _cpo_ from seeing gutil::sqrt
		struct sqrt_fn final {
			template<IsScalar T>
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
	}

	/// Allow use of gutil::sqrt() which then calls gutil::_cpo_::sqrt_fn::operator() and
	/// dispatches (at compile time) to the correct square root function.
	/// If a user defines usrlib::usr_type and a usrlib::sqrt(usrlib::usr_type) -> usrlib::usr_type,
	/// then this will be prioritized.
	inline constexpr _cpo_::sqrt_fn sqrt{};

	GUTIL_IF_OPENMP("omp declare simd")
	template<IsScalar T>
	[[nodiscard]] inline T max(const T x, const T y) noexcept {
		return x > y ? x : y;
	}

	GUTIL_IF_OPENMP("omp declare simd")
	template<IsScalar T>
	[[nodiscard]] inline T min(const T x, const T y) noexcept {
		return x < y ? x : y;
	}

	GUTIL_IF_OPENMP("omp declare simd")
	template<IsScalar T>
	[[nodiscard]] inline T abs(const T x) noexcept {
		return x < T{0} ? -x : x;
	}

	GUTIL_IF_OPENMP("omp declare simd")
	template<IsScalar T>
	[[nodiscard]] inline T clamp(const T x, const T lo, const T hi) noexcept {
		return x < lo ? lo : (x > hi ? hi : x);
	}


	///////////////////////////////////////////////////////////
	//////////////////// POINT OPERATIONS /////////////////////
	///////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////
	/// In-place y += a*x (axpy in BLAS)
	///////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	void axpy(T a, const Point<DIM,T>& x, Point<DIM,T>& y) noexcept {
		for (int i=0; i<DIM; ++i) {
			if constexpr (std::floating_point<T>) {
				y[i] = std::fma(a, x[i], y[i]);
			}
			else {
				y[i] += a * x[i];
			}
		}
	}


	///////////////////////////////////////////////////////////
	/// Sum, product, etc. reductions and norms
	///////////////////////////////////////////////////////////	
	template<int DIM, IsScalar T> requires(DIM>0)
	[[nodiscard]] inline T sum_reduce(const Point<DIM,T>& vec) {
		T result{vec.data[0]};

		GUTIL_IF_OPENMP("omp simd reduction(+:result) if(DIM>16)")
		for (int i=1; i<DIM; ++i) {
			result += vec.data[i];
		}
		return result;
	}

	template<int DIM, IsScalar T> requires(DIM>0)
	[[nodiscard]] inline T product_reduce(const Point<DIM,T>& vec) {
		T result{vec.data[0]};

		GUTIL_IF_OPENMP("omp simd reduction(*:result) if(DIM>16)")
		for (int i=1; i<DIM; ++i) {
			result *= vec.data[i];
		}
		return result;
	}

	template<int DIM, IsScalar T> requires(DIM>0)
	[[nodiscard]] inline T max_reduce(const Point<DIM,T>& vec) {
		T result{vec.data[0]};

		GUTIL_IF_OPENMP("omp simd reduction(max:result) if(DIM>16)")
		for (int i=1; i<DIM; ++i) {
			result = gutil::max(result, vec.data[i]);
		}
		return result;
	}

	/// get the minimum value
	template<int DIM, IsScalar T> requires(DIM>0)
	[[nodiscard]] inline T min_reduce(const Point<DIM,T>& vec) {
		T result{vec.data[0]};

		GUTIL_IF_OPENMP("omp simd reduction(min:result) if(DIM>16)")
		for (int i=1; i<DIM; ++i) {
			result = gutil::min(result, vec.data[i]);
		}
		return result;
	}

	/// get the minimum absolute value
	template<int DIM, IsScalar T> requires(DIM>0)
	[[nodiscard]] inline T min_abs_reduce(const Point<DIM,T>& vec) {
		T result{gutil::abs(vec.data[0])};

		GUTIL_IF_OPENMP("omp simd reduction(min:result) if(DIM>16)")
		for (int i=1; i<DIM; ++i) {
			result = gutil::min(result, gutil::abs(vec.data[i]));
		}
		return result;
	}

	/// infinity norm
	template<int DIM, IsScalar T> requires(DIM>0)
	[[nodiscard]] inline T norminfty(const Point<DIM,T>& vec) {
		T result{gutil::abs(vec.data[0])};

		GUTIL_IF_OPENMP("omp simd reduction(max:result) if(DIM>16)")
		for (int i=1; i<DIM; ++i) {
			result = gutil::max(result, gutil::abs(vec.data[i]));
		}
		return result;
	}

	/// 1-norm
	template<int DIM, IsScalar T> requires(DIM>0)
	[[nodiscard]] inline T norm1(const Point<DIM,T>& vec) {
		T result{gutil::abs(vec.data[0])};

		GUTIL_IF_OPENMP("omp simd reduction(+:result) if(DIM>16)")
		for (int i=1; i<DIM; ++i) {
			result += gutil::abs(vec.data[i]);
		}
		return result;
	}

	/// squared 2-norm
	template<int DIM, IsScalar T> requires(DIM>0)
	[[nodiscard]] inline T squared_norm(const Point<DIM,T>& vec) {
		T result{vec.data[0]*vec.data[0]};

		GUTIL_IF_OPENMP("omp simd reduction(+:result) if(DIM>16)")
		for (int i=1; i<DIM; ++i) {
			result += vec.data[i]*vec.data[i];
		}
		return result;
	}

	/// 2-norm
	template<int DIM, IsScalar T> requires(DIM>0)
	[[nodiscard]] inline T norm2(const Point<DIM,T>& vec) {
		return gutil::sqrt(gutil::squared_norm(vec));
	}

	///////////////////////////////////////////////////////////
	/// Element-wise operations
	///////////////////////////////////////////////////////////	
	template<int DIM, IsScalar T>
	[[nodiscard]] inline constexpr Point<DIM,T> elmin(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result = Point<DIM,T>::Zeros();
		for (int i=0; i<DIM; i++) {result.data[i] = gutil::min(left.data[i], right.data[i]);}
		return result;
	}

	template<int DIM, IsScalar T>
	[[nodiscard]] inline constexpr Point<DIM,T> elmax(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result = Point<DIM,T>::Zeros();
		for (int i=0; i<DIM; i++) {result.data[i] = gutil::max(left.data[i], right.data[i]);}
		return result;
	}

	///////////////////////////////////////////////////////////
	/// Geometry operations
	///////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline T dot(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		T result{left.data[0]*right.data[0]};

		GUTIL_IF_OPENMP("omp simd reduction(+:result) if(DIM>16)")
		for (int i=1; i<DIM; ++i) {
			result += left.data[i]*right.data[i];
		}
		return result;
	}

	template<IsScalar T>
	[[nodiscard]] inline constexpr Point<3,T> cross(const Point<3,T>& left, const Point<3,T>& right) noexcept {
		Point<3,T> result;
		result[0] = left.data[1]*right.data[2] - left.data[2]*right.data[1];
		result[1] = left.data[2]*right.data[0] - left.data[0]*right.data[2];
		result[2] = left.data[0]*right.data[1] - left.data[1]*right.data[0];
		return result;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline Point<DIM,T> normalized(const Point<DIM,T>& vec) {
		const T nn = gutil::norm2(vec);
		assert(nn>T{0} && "normalized: input was the zero vector");
		return vec/nn;
	}

	///////////////////////////////////////////////////////////
	///////////////////// BOX OPERATIONS //////////////////////
	///////////////////////////////////////////////////////////





















	



}