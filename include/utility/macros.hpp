#pragma once

//////////////////////////////////////////////////////
/// If OpenMP is present, some operations are provided with 
/// vectorized versions over spans. Note that if
/// a program is compiled agains OpenMP, then 
/// _OPENMP is defined.
//////////////////////////////////////////////////////


//////////////////////////////////////////////////////
/// Helper macros.
//////////////////////////////////////////////////////
#define GUTIL_STRINGIFY(x) #x
#define GUTIL_PRAGMA(x) _Pragma(#x)

#ifndef NDEBUG
	#define GUTIL_DEBUG(...)	__VA_ARGS__ //inject in debug build
	#define GUTIL_NDEBUG(...)				//supress in release build
#else
	#define GUTIL_DEBUG(...) 				//supress in release build
	#define GUTIL_NDEBUG(...) 	__VA_ARGS__ //inject in release build
#endif


//////////////////////////////////////////////////////
/// Macro to disable floating point associativity for functions
/// that require exact arithmetic order.
//////////////////////////////////////////////////////
#if defined(__GNUC__)
	//linux (gcc) compiler, should work for mac (clang) as well.
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
/// Macros to call OpenMP pragmas if compiled with openmp
//////////////////////////////////////////////////////
#ifdef _OPENMP
	#define GUTIL_OMP(...) GUTIL_PRAGMA(omp __VA_ARGS__)
	#define GUTIL_SIMD(...) GUTIL_PRAGMA(omp simd __VA_ARGS__)
	#define GUTIL_DECLARE_SIMD(...) GUTIL_PRAGMA(omp declare simd __VA_ARGS__)
#else
	#define GUTIL_OMP(...)
	#define GUTIL_SIMD(...)
	#define GUTIL_DECLARE_SIMD(...)
#endif


//////////////////////////////////////////////////////
/// To better use Customization Point Objects,
/// it is nice to have operator() as static.
/// This is only availible in later std versions
/// (i.e., in some implementations of c++23).
//////////////////////////////////////////////////////
#ifdef __cpp_static_call_operator
    #define GUTIL_STATIC_CALL static
    #define GUTIL_STATIC_CALL_CONST
#else
    #define GUTIL_STATIC_CALL
    #define GUTIL_STATIC_CALL_CONST const
#endif