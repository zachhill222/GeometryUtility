#pragma once

#include "math/gutilmath.hpp"

#include <algorithm>
#include <initializer_list>
#include <concepts>
#include <vector>
#include <span>
#include <iostream>
#include <cassert>
#include <numeric>







namespace gutil
{
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
		Point() noexcept {}
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

		//////////////////////////////////////////////////////////
		/// IN-PLACE BINARY OPERATIONS
		//////////////////////////////////////////////////////////
		Point& operator+=(const Point& other) noexcept {
			GUTIL_IF_OPENMP("omp simd if(DIM>16)")
			for (int i=0; i<DIM; i++) {data[i] += other.data[i];}
			return *this;
		}

		Point& operator-=(const Point& other) noexcept {
			GUTIL_IF_OPENMP("omp simd if(DIM>16)")
			for (int i=0; i<DIM; i++) {data[i] -= other.data[i];}
			return *this;
		}

		Point& operator*=(const Point& other) noexcept {
			GUTIL_IF_OPENMP("omp simd if(DIM>16)")
			for (int i=0; i<DIM; i++) {data[i] *= other.data[i];}
			return *this;
		}

		Point& operator/=(const Point& other) noexcept {
			GUTIL_IF_OPENMP("omp simd if(DIM>16)")
			for (int i=0; i<DIM; i++) {data[i] /= other.data[i];}
			return *this;
		}

		Point& operator%=(const Point& other) noexcept requires(std::integral<T>) {
			GUTIL_IF_OPENMP("omp simd if(DIM>16)")
			for (int i=0; i<DIM; i++) {data[i] %= other.data[i];}
			return *this;
		}

		template<typename U> requires std::is_nothrow_convertible<U,T>::value
		Point& operator*=(const U scalar) noexcept {
			const T s = static_cast<T>(scalar);
			GUTIL_IF_OPENMP("omp simd if(DIM>16)")
			for (int i=0; i<DIM; i++) {data[i] *= s;}
			return *this;
		}

		template<typename U> requires std::is_nothrow_convertible<U,T>::value
		Point& operator/=(const U scalar) noexcept {
			assert(scalar!=U{0} && "Point::operator/=: divide by zero");
			const T s_inv = static_cast<T>(U{1}/scalar);
			
			GUTIL_IF_OPENMP("omp simd if(DIM>16)")
			for (int i=0; i<DIM; i++) {data[i] *= s_inv;}
			return *this;
		}

		//////////////////////////////////////////////////////////
		/// UNARY OPERATIONS
		//////////////////////////////////////////////////////////
		[[nodiscard]] Point operator-() const noexcept {
			Point result;
			GUTIL_IF_OPENMP("omp simd if(DIM>16)")
			for (int i=0; i<DIM; i++) {result.data[i] = -data[i];}
			return result;
		}

		Point& normalize() {
			*this = gutil::normalized(*this);
			return *this;
		}

		Point normalized() const {
			return gutil::normalized(*this);
		}

		//////////////////////////////////////////////////////////
		/// REDUCTIONS AND NORM OPERATIONS
		//////////////////////////////////////////////////////////
		[[nodiscard]] T norminfty() const noexcept {
			return gutil::norminfty(*this);
		}

		[[nodiscard]] T norm1() const noexcept {
			return gutil::norm1(*this);
		}

		[[nodiscard]] T squared_norm() const noexcept {
			return gutil::squared_norm(*this);
		}

		[[nodiscard]] T norm2() const noexcept {
			return gutil::norm2(*this);
		}

		[[nodiscard]] T prod() const noexcept {
			return gutil::product_reduce(*this);
		}

		[[nodiscard]] T sum() const noexcept {
			return gutil::sum_reduce(*this);
		}

		[[nodiscard]] T max() const noexcept {
			return gutil::max_reduce(*this);
		}

		[[nodiscard]] T min() const noexcept {
			return gutil::min_reduce(*this);
		}

		//////////////////////////////////////////////////////////
		/// BINARY VECTOR OPERATIONS
		//////////////////////////////////////////////////////////
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
	//////////////////////// STANDARD BINARY OPERATIONS /////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<int DIM, typename T>
	[[nodiscard]] Point<DIM,T> operator+(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result{left};
		return result+=right;
	}

	template<int DIM, typename T>
	[[nodiscard]] Point<DIM,T> operator-(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result{left};
		return result-=right;
	}

	template<int DIM, typename T>
	[[nodiscard]] Point<DIM,T> operator*(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result{left};
		return result*=right;
	}

	template<int DIM, typename T, typename U> requires std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] Point<DIM,T> operator*(const U left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result{right};
		return result*=static_cast<T>(left);
	}

	template<int DIM, typename T, typename U> requires std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] Point<DIM,T> operator*(const Point<DIM,T>& left, const U right) noexcept {
		Point<DIM,T> result{left};
		return result*=static_cast<T>(right);
	}

	template<int DIM, typename T, typename U> requires std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] Point<DIM,T> operator/(const Point<DIM,T>& left, const U right) noexcept {
		Point<DIM,T> result{left};
		return result/=static_cast<T>(right);
	}

	template<int DIM, typename T, typename U> requires std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] Point<DIM,T> operator/(const U left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result = Point<DIM,T>::Filled(static_cast<T>(left));
		return result/=right;
	}

	template<int DIM, typename T>
	[[nodiscard]] Point<DIM,T> operator/(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result{left};
		return result/=right;
	}

	template<int DIM, typename T> requires std::integral<T>
	[[nodiscard]] Point<DIM,T> operator%(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept{
		Point<DIM,T> result{left};
		return result%=right;
	}


	/////////////////////////////////////////////////////////////////////////////
	//////////////////////////// UTILITY OPERATIONS /////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<int DIM, typename T>
	std::ostream& operator<<(std::ostream& os, const Point<DIM,T>& point) {
		for (int i = 0; i < DIM-1; i++) {os << point[i] << " ";}
		os << point[DIM-1];
		return os;
	}

	/////////////////////////////////////////////////////////////////////////////
	////////////////////// CAREFUL FLOATING POINT METHODS ///////////////////////
	/////////////////////////////////////////////////////////////////////////////

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
				[](U a, U b) {return gutil::abs(a) < gutil::abs(b);});

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
	/// Sum the points using Kahan to reduce floating point errors (not pre-sorted)
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

		for (const Point<DIM,T>& p : points) {
			y = static_cast<Point<DIM,U>>(p) - c;
			t = result + y;
			c = (t - result) - y;
			result = t;
		}

		return static_cast<Point<DIM,T>>(result);
	}
	GUTIL_NO_ASSOC_MATH_END



	

}