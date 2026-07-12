#pragma once

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <concepts>
#include <vector>
#include <span>
#include <iostream>
#include <cassert>
#include <numeric>


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




namespace gutil
{
	///////////////////////////////////////////////////////////
	/// Concept to ensure that a type is some form of scalar
	/// that emulates the real line and will work with the Point class.
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



	template<int DIM, IsScalar T> requires (DIM>0)
	struct Point;

	///////////////////////////////////////////////////////////
	/// Concept to ensure that a type is some form of point
	/// defined in the class below
	///////////////////////////////////////////////////////////
	template<typename T>
	concept IsPointLike = requires {
		typename std::integral_constant<int, T::dim>;
		typename T::value_type;
	} && IsScalar<typename T::value_type> && std::same_as<T, Point<T::dim, typename T::value_type>>;

	
	//////////////////////////////////////////////////////////
	/// Pre define any special operations that are constexpr and used in the class
	/// but should be defined outside the class.
	//////////////////////////////////////////////////////////
	template<int DIM, typename T> requires (DIM>0)
	[[nodiscard]] inline T dot(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		T result{left[0]*right[0]};
		for (int i=1; i<DIM; i++) {
			if constexpr (std::floating_point<T>) {
				//use explicit fused multiply add when possible
				result = std::fma(left[i], right[i], result);
			}
			else {
				//this might be able to be fused with compiler options
				result += left[i]*right[i];
			}
		}
		return result;
	}

	template<typename T>
	[[nodiscard]] inline constexpr Point<3,T> cross(const Point<3,T>& left, const Point<3,T>& right) noexcept {
		//*this x other
		Point<3,T> result;
		result[0] = left[1]*right[2] - left[2]*right[1];
		result[1] = left[2]*right[0] - left[0]*right[2];
		result[2] = left[0]*right[1] - left[1]*right[0];
		return result;
	}

	//////////////////////////////////////////////////////////
	/// A class for cartesian points in space
	///
	/// Can be treated as a vector with no column or row orientation.
	///
	/// @tparam DIM   The spacial dimension. These points are allocated
	///                   on the stack, so DIM shouldn't be too large.
	/// @tparam T     The underlying numeric type
	///
	/// Note that the type T must implement any comparisons if
	/// floating point rounding is important and unacceptable.
	//////////////////////////////////////////////////////////
	template<int DIM, IsScalar T=double> requires (DIM>0)
	struct Point
	{
		//track type information
		static constexpr int dim = DIM;
		using value_type = T;

		//essential constructors
		// note that the default constructor does not initialize data.
		// it has indeterminant values and all values should be assigned before reading.
		// use Zeros() to have data be initialized to all zeros.
		constexpr Point() noexcept {}
		constexpr Point(const Point& other) noexcept = default;
		constexpr Point(Point&& other) noexcept = default;

		//initialize via Point v(1,2,3)
		template<typename... Ts> requires (sizeof...(Ts)==DIM && (std::is_nothrow_convertible<Ts,T>::value && ...))
		constexpr Point(Ts... args) : data{static_cast<T>(args)...} {}

		//factory methods
		[[nodiscard]] static constexpr Point Filled(T val) noexcept {
			Point p;
			std::fill(p.data, p.data+DIM, val);
			return p;
		}

		[[nodiscard]] static constexpr Point Zeros() noexcept {return Filled(T{0});}

		//copy and move assignment
		constexpr Point& operator=(const Point& other) noexcept = default;
		constexpr Point& operator=(Point&& other) noexcept = default;

		//element access
		[[nodiscard]] constexpr T operator[](const int idx) const noexcept {assert(0<=idx and idx<DIM); return data[idx];}
		[[nodiscard]] constexpr T& operator[](const int idx) noexcept {assert(0<=idx and idx<DIM); return data[idx];}

		[[nodiscard]] constexpr T x() const noexcept requires (DIM>0) {return data[0];}
		[[nodiscard]] constexpr T y() const noexcept requires (DIM>1) {return data[1];}
		[[nodiscard]] constexpr T z() const noexcept requires (DIM>2) {return data[2];}
		[[nodiscard]] constexpr T w() const noexcept requires (DIM>3) {return data[3];}
		[[nodiscard]] constexpr T& x() noexcept requires (DIM>0) {return data[0];}
		[[nodiscard]] constexpr T& y() noexcept requires (DIM>1) {return data[1];}
		[[nodiscard]] constexpr T& z() noexcept requires (DIM>2) {return data[2];}
		[[nodiscard]] constexpr T& w() noexcept requires (DIM>3) {return data[3];}

		//standard container access
		[[nodiscard]] static constexpr size_t size() noexcept {return static_cast<size_t>(DIM);}

		constexpr T*       begin()        noexcept {return data;}
		constexpr const T* begin()  const noexcept {return data;}
		constexpr T*       end()          noexcept {return data + DIM;}
		constexpr const T* end()    const noexcept {return data + DIM;}
		constexpr const T* cbegin() const noexcept {return data;}
		constexpr const T* cend()   const noexcept {return data + DIM;}

		//type conversion
		template<int OTHER_DIM, typename OTHER_T> requires std::is_nothrow_convertible<T,OTHER_T>::value
		[[nodiscard]] explicit constexpr operator Point<OTHER_DIM, OTHER_T>() const noexcept {
			Point<OTHER_DIM,OTHER_T> result;
			constexpr int min_dim = DIM < OTHER_DIM ? DIM : OTHER_DIM;
			//copy with cast valid data
			for (int i=0; i<min_dim; i++) {result[i] = static_cast<OTHER_T>(data[i]);}
			//append 0 if necessary
			for (int i=min_dim; i<OTHER_DIM; i++) {result[i] = OTHER_T{0};}
			return result;
		}

		//asymetric/in-place element-wise math operations
		constexpr Point& operator+=(const Point& other) noexcept {
			for (int i=0; i<DIM; i++) {data[i] += other.data[i];}
			return *this;
		}

		constexpr Point& operator-=(const Point& other) noexcept {
			for (int i=0; i<DIM; i++) {data[i] -= other.data[i];}
			return *this;
		}

		constexpr Point& operator*=(const Point& other) noexcept {
			for (int i=0; i<DIM; i++) {data[i] *= other.data[i];}
			return *this;
		}

		constexpr Point& operator/=(const Point& other) noexcept {
			for (int i=0; i<DIM; i++) {data[i] /= other.data[i];}
			return *this;
		}

		constexpr Point& operator%=(const Point& other) noexcept requires(std::integral<T>) {
			for (int i=0; i<DIM; i++) {data[i] %= other.data[i];}
			return *this;
		}

		template<typename U> requires std::is_nothrow_convertible<U,T>::value
		constexpr Point& operator*=(const U scalar) noexcept {
			const T s = static_cast<T>(scalar);
			for (int i=0; i<DIM; i++) {data[i] *= s;}
			return *this;
		}

		template<typename U> requires std::is_nothrow_convertible<U,T>::value
		constexpr Point& operator/=(const U scalar) noexcept {
			const T s = static_cast<T>(scalar);
			assert(s!=T{0} && "Point::operator/=: divide by zero");
			for (int i=0; i<DIM; i++) {data[i] /= s;}
			return *this;
		}

		//negation
		[[nodiscard]] constexpr Point operator-() const noexcept {
			Point result;
			for (int i=0; i<DIM; i++) {result.data[i] = -data[i];}
			return result;
		}

		//norms
		[[nodiscard]] constexpr T norminfty() const noexcept {
			using std::abs;
			T result{abs(data[0])};
			for (int i=1; i<DIM; i++) {
				T val = abs(data[i]);
				result = val > result ? val : result;
			}
			return result;
		}

		[[nodiscard]] constexpr T norm1() const noexcept {
			using std::abs;
			T result{abs(data[0])};
			for (int i=1; i<DIM; i++) {result += abs(data[i]);}
			return result;
		}

		[[nodiscard]] T squaredNorm() const noexcept {
			return gutil::dot(*this,*this);
		}

		[[nodiscard]] T norm2() const noexcept {
			using std::sqrt;
			return static_cast<T>(sqrt(squaredNorm()));
		}

		//accumulators
		[[nodiscard]] constexpr T prod() const noexcept {
			T result{data[0]};
			for (int i=1; i<DIM; i++) {result *= data[i];}
			return result;
		}

		[[nodiscard]] constexpr T sum() const noexcept {
			T result{data[0]};
			for (int i=1; i<DIM; i++) {result += data[i];}
			return result;
		}

		[[nodiscard]] constexpr T max() const noexcept {
			T result{data[0]};
			for (int i=1; i<DIM; i++) {
				result = data[i] > result ? data[i] : result;
			}

			return result;
		}

		[[nodiscard]] constexpr T min() const noexcept {
			T result{data[0]};
			for (int i=1; i<DIM; i++) {
				result = data[i] < result ? data[i] : result;
			}

			return result;
		}

		//standard vector operations
		[[nodiscard]] T dot(const Point& other) const noexcept {
			return gutil::dot(*this, other);
		}

		[[nodiscard]] constexpr Point<3,T> cross(const Point& other) const noexcept requires (DIM==3) {
			return gutil::cross(*this, other);
		}

		T data[DIM];
	};

	///////////////////////////////////////////////////////////////////
	/// Ensure that the concept 'IsPointLike' is valid
	///////////////////////////////////////////////////////////////////
	static_assert(IsPointLike<Point<3,float>>);
	static_assert(IsPointLike<Point<2,float>>);
	static_assert(IsPointLike<Point<3,double>>);
	static_assert(IsPointLike<Point<2,double>>);


	////////////////////////////////////////////////////////////////////
	/////// COMPARISON (CONE/ELEMENT-WISE, NOT A TOTAL ORDERING) ///////
	////////////////////////////////////////////////////////////////////
	template<int DIM, typename T>
	[[nodiscard]] constexpr bool operator==(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		for (int i=0; i<DIM; i++) {
			if (left[i]!=right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, typename T>
	[[nodiscard]] constexpr bool operator<(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		for (int i=0; i<DIM; i++) {
			if (left[i] >= right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, typename T>
	[[nodiscard]] constexpr bool operator<=(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		for (int i=0; i<DIM; i++) {
			if (left[i] > right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, typename T>
	[[nodiscard]] constexpr bool operator>(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		for (int i=0; i<DIM; i++) {
			if (left[i] <= right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, typename T>
	[[nodiscard]] constexpr bool operator>=(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		for (int i=0; i<DIM; i++) {
			if (left[i] < right[i]) {return false;}
		}
		return true;
	}

	/////////////////////////////////////////////////////////////////////////////
	//////////////////////// ARITHMETIC (COMPONENT-WISE) ////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<int DIM, typename T>
	[[nodiscard]] constexpr Point<DIM,T> operator+(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result{left};
		return result+=right;
	}

	template<int DIM, typename T>
	[[nodiscard]] constexpr Point<DIM,T> operator-(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result{left};
		return result-=right;
	}

	template<int DIM, typename T>
	[[nodiscard]] constexpr Point<DIM,T> operator*(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result{left};
		return result*=right;
	}

	template<int DIM, typename T, typename U> requires std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] constexpr Point<DIM,T> operator*(const U left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result{right};
		return result*=static_cast<T>(left);
	}

	template<int DIM, typename T, typename U> requires std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] constexpr Point<DIM,T> operator*(const Point<DIM,T>& left, const U right) noexcept {
		Point<DIM,T> result{left};
		return result*=static_cast<T>(right);
	}

	template<int DIM, typename T, typename U> requires std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] constexpr Point<DIM,T> operator/(const Point<DIM,T>& left, const U right) noexcept {
		Point<DIM,T> result{left};
		return result/=static_cast<T>(right);
	}

	template<int DIM, typename T, typename U> requires std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] constexpr Point<DIM,T> operator/(const U left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result = Point<DIM,T>::Filled(static_cast<T>(left));
		return result/=right;
	}

	template<int DIM, typename T>
	[[nodiscard]] constexpr Point<DIM,T> operator/(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result{left};
		return result/=right;
	}

	template<int DIM, typename T> requires std::integral<T>
	[[nodiscard]] constexpr Point<DIM,T> operator%(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept{
		Point<DIM,T> result{left};
		return result%=right;
	}

	///////////////////////////////////////////////////////////////////////////////
	//////////////////////// TRADITIONAL VECTOR OPERATIONS ////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	template<int DIM, typename T>
	[[nodiscard]] inline T squaredNorm(const Point<DIM,T>& point) noexcept {
		return gutil::dot(point,point);
	}

	template<int DIM, typename T>
	[[nodiscard]] inline T norm2(const Point<DIM,T>& point) noexcept {
		using std::sqrt;
		return static_cast<T>(sqrt(squaredNorm(point)));
	}

	template<int DIM, typename T>
	[[nodiscard]] inline constexpr T norm1(const Point<DIM,T>& point) noexcept {
		return point.norm1();
	}

	template<int DIM, typename T>
	[[nodiscard]] inline constexpr T norminfty(const Point<DIM,T>& point) noexcept {
		return point.norminfty();
	}

	template<int DIM, typename T>
	[[nodiscard]] Point<DIM,T> normalize(const Point<DIM,T>& point) noexcept {
		T scale{norm2(point)};
		assert(scale!=T{0} && "normalize: zero-length vector");
		return point/scale;
	}

	//////////////////////////////////////////////////////////////////////////
	//////////////////////// OTHER UTILITY OPERATIONS ////////////////////////
	//////////////////////////////////////////////////////////////////////////
	template<int DIM, typename T>
	[[nodiscard]] inline constexpr Point<DIM,T> abs(const Point<DIM,T>& point) noexcept {
		using std::abs;
		Point<DIM,T> result;
		for (int i=0; i<DIM; i++) {result[i] = abs(point[i]);}
		return result;
	}

	template<int DIM, typename T>
	[[nodiscard]] inline constexpr Point<DIM,T> elmax(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		using std::max;
		Point<DIM,T> result;
		for (int i=0; i<DIM; i++) {result[i] = max(left[i], right[i]);}
		return result;
	}

	template<int DIM, typename T>
	[[nodiscard]] inline constexpr Point<DIM,T> elmax(std::span<const Point<DIM,T>> container) noexcept {
		assert(container.size()>0);

		Point<DIM,T> result = container[0];
		for (size_t i=1; i<container.size(); ++i) {
			result = elmax(result, container[i]);
		}

		return result;
	}

	template<int DIM, typename T>
	[[nodiscard]] inline constexpr Point<DIM,T> elmin(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		using std::min;
		Point<DIM,T> result;
		for (int i=0; i<DIM; i++) {result[i] = min(left[i], right[i]);}
		return result;
	}

	template<int DIM, typename T>
	[[nodiscard]] inline constexpr Point<DIM,T> elmin(std::span<const Point<DIM,T>> container) noexcept {
		assert(container.size()>0);

		Point<DIM,T> result = container[0];
		for (size_t i=1; i<container.size(); ++i) {
			result = elmin(result, container[i]);
		}

		return result;
	}

	template<int DIM, typename T>
	[[nodiscard]] inline constexpr Point<DIM,T> clamp(const Point<DIM,T>& p, const Point<DIM,T>& low, const Point<DIM,T>& high) noexcept {
		using std::clamp;
		Point<DIM,T> result;
		for (int i=0; i<DIM; ++i) {
			result[i] = clamp(p[i], low[i], high[i]);
		}
		return result;
	}

	template<int DIM, typename T, typename U=T> requires std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] inline constexpr Point<DIM,T> lerp(const Point<DIM,T>& a, const Point<DIM,T>& b, const U t) noexcept {
		return a + static_cast<T>(t) * (b - a);
	}

	template<int DIM, typename T>
	[[nodiscard]] inline constexpr T max(const Point<DIM,T>& point) noexcept {
		return point.max();
	}

	template<int DIM, typename T>
	[[nodiscard]] inline constexpr T min(const Point<DIM,T>& point) noexcept {
		return point.min();
	}

	template<int DIM, typename T>
	[[nodiscard]] inline constexpr T sum(const Point<DIM,T>& point) noexcept {
		return point.sum();
	}

	template<int DIM, typename T>
	[[nodiscard]] inline constexpr T prod(const Point<DIM,T>& point) noexcept {
		return point.prod();
	}

	template<int DIM, typename T>
	std::ostream& operator<<(std::ostream& os, const Point<DIM,T>& point) {
		for (int i = 0; i < DIM-1; i++) {os << point[i] << " ";}
		os << point[DIM-1];
		return os;
	}


	////////////////////////////////////////////////////////////////////////////////
	/// Sum the points by pre-sorting each component and summing from least to greatest.
	///	Extended precision or truncating may be done by specifying the intermediate arithmetic type.
	///
	/// @param points A span of points to add
	///
	/// @tparam U is the type that the arithmetic should be done in
	/// @tparam T is the input/output type
	////////////////////////////////////////////////////////////////////////////////
	template<int DIM, typename T, typename U=T>
	[[nodiscard]] inline Point<DIM,T> sorted_sum(std::span<const Point<DIM,T>> points) noexcept {
		if (points.empty()) {return Point<DIM,T>::Zeros();}

		Point<DIM,T> result;
		std::vector<U> component;
		component.reserve(points.size());

		for (int i = 0; i < DIM; i++) {
			component.clear();
			for ( const Point<DIM,T>& p : points) {
				component.push_back(static_cast<U>(p[i]));
			}

			std::sort(component.begin(), component.end(), 
				[](U a, U b) {
					using std::abs;
					return abs(a) < abs(b);});

			result[i] = static_cast<T>(std::accumulate(component.begin(), component.end(), U{0}));
		}
		return result;
	}

	/// Convenient ways to call the sorted sum.
	template<int DIM, typename T, typename U=T>
	[[nodiscard]] inline Point<DIM,T> sorted_sum(std::initializer_list<Point<DIM,T>> points) noexcept {
		std::vector<Point<DIM,T>> intermediate{points};
		return sorted_sum<DIM,T,U>(intermediate);
	}


	////////////////////////////////////////////////////////////////////////////////
	/// Sum the points using Kahan or 2sum to reduce floating point errors (not pre-sorted)
	///
	/// @param points A span of points to add
	///
	/// @tparam U is the type that the arithmetic should be done in
	/// @tparam T is the input/output type
	////////////////////////////////////////////////////////////////////////////////
	GUTIL_NO_ASSOC_MATH_START
	template<int DIM, typename T, typename U=T> requires (std::floating_point<T> && std::is_nothrow_convertible<T,U>::value)
	[[nodiscard]] inline Point<DIM,T> kahan_sum(std::span<const Point<DIM,T>> points) noexcept {
		if (points.empty()) {return Point<DIM,T>::Zeros();}
		
		Point<DIM,U> result = Point<DIM,U>::Zeros();	//accumulator
		Point<DIM,U> c      = Point<DIM,U>::Zeros();	//compensation
		Point<DIM,U> y,t;								//intermediates

		for (size_t i=0; i<points.size(); ++i) {
			y = static_cast<Point<DIM,U>>(points[i]) - c;
			t = result + y;
			c = (t - result) - y;
			result = t;
		}

		return static_cast<Point<DIM,T>>(result);
	}
	GUTIL_NO_ASSOC_MATH_START



	/// In-place y += a*x (axpy in BLAS)
	template<int DIM, typename T>
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

}