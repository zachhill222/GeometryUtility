#pragma once

#include <concepts>
#include <limits>
#include <type_traits>

namespace gutil
{
	///////////////////////////////////////////////////////////
	/// Concept to ensure that a type is some form of scalar
	/// that emulates the real line and will work with the Point class.
	///
	/// If a custom scalar type (e.g., a fixed precision type), it
	/// must satisfy this concept. Note it should have a sqrt() function.
	/// If it doesn't, it should be castable to/from a double to use 
	/// std::sqrt as a fallback.
	///
	/// Additionally, the std::numeric_limits<T> must be specialized.
	///////////////////////////////////////////////////////////
	template<typename T>
	concept IsScalar = std::numeric_limits<T>::is_specialized && requires(T a, T b) {
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
	/// For some algorithms, we need to know if a scalar type
	/// is exact, integer, etc. Additionally, values like epsilon, max, lower, etc
	/// are helpful. Some geometry operations only make sense if the underlying
	/// scalar emulates the real line.
	///////////////////////////////////////////////////////////
	template<typename T>
	concept IsReal = IsScalar<T> && !std::numeric_limits<T>::is_integer;

	template<typename T>
	concept IsInteger = IsScalar<T> && std::numeric_limits<T>::is_integer;

	template<typename T>
	concept IsExact = IsScalar<T> && std::numeric_limits<T>::is_exact;

	template<typename T>
	concept IsFloatingPoint = IsScalar<T> && !std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_exact;

	template<typename T>
	concept IsFixedPoint = IsScalar<T> && !std::numeric_limits<T>::is_integer && std::numeric_limits<T>::is_exact;

	template<IsScalar T>	//returns the smallest finite value of the given non-floating-point type, or the smallest positive normal value of the given floating-point type 
	struct Min {static constexpr T value = std::numeric_limits<T>::min();};

	template<IsScalar T>
	struct Lowest {static constexpr T value = std::numeric_limits<T>::lowest();};

	template<IsScalar T>
	struct Max {static constexpr T value = std::numeric_limits<T>::max();};

	template<IsScalar T>
	struct Epsilon {static constexpr T value = std::numeric_limits<T>::epsilon();};

	template<typename T>
	concept GeometryObject = requires {
		typename std::integral_constant<int, T::DIMENSION>;
		typename T::scalar_type;
	} && T::DIMENSION>0 && IsScalar<typename T::scalar_type> && std::same_as<std::remove_cv_t<decltype(T::DIMENSION)>, int>;

	template<typename T, typename U>
	concept CompatibleGeometryObjects = GeometryObject<T> && GeometryObject<U>
		&& std::same_as<typename T::scalar_type, typename U::scalar_type>
		&& (T::DIMENSION == U::DIMENSION);

	///////////////////////////////////////////////////////////
	/// Use a templated type to trigger conditional compile
	/// errors (say no valid constexpr branch of an if statement
	/// was found).
	///////////////////////////////////////////////////////////
	template<typename T>
	inline constexpr bool always_false_v = false;
}