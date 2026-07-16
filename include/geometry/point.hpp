#pragma once

#include "utility/utility.hpp"
#include "math/math.hpp"

#include <algorithm>
#include <initializer_list>
#include <span>
#include <concepts>
#include <iostream>
#include <cassert>
#include <functional>

namespace gutil
{
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
		////////////////////////////////////////////////////////////////
		// Data and aliases
		////////////////////////////////////////////////////////////////
		using scalar_type = T;

		T data[DIM];

		static constexpr int DIMENSION = DIM;

		////////////////////////////////////////////////////////////////
		// Constructors and move/copy
		////////////////////////////////////////////////////////////////
		constexpr Point() noexcept {} //note data[] is not initialized
		constexpr Point(const Point& other) noexcept = default;
		constexpr Point(Point&& other) noexcept = default;
		constexpr Point& operator=(const Point& other) noexcept = default;
		constexpr Point& operator=(Point&& other) noexcept = default;

		//initialize via Point v(1,2,3)
		template<typename... Ts> requires (sizeof...(Ts)==DIM && (std::is_nothrow_convertible_v<Ts,T> && ...))
		constexpr Point(Ts... args) : data{static_cast<T>(args)...} {}

		//factory methods
		template<IsScalar U> requires std::is_nothrow_convertible_v<T,U>
		[[nodiscard]] static constexpr Point Filled(U val) noexcept {
			Point p;
			std::fill(p.data, p.data+DIM, static_cast<T>(val));
			return p;
		}

		[[nodiscard]] static constexpr Point Zeros() noexcept {return Filled(T{0});}

		
		////////////////////////////////////////////////////////////////
		// Element access
		////////////////////////////////////////////////////////////////
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

		constexpr T*       begin()        noexcept {return data;}
		constexpr const T* begin()  const noexcept {return data;}
		constexpr T*       end()          noexcept {return data + DIM;}
		constexpr const T* end()    const noexcept {return data + DIM;}
		constexpr const T* cbegin() const noexcept {return data;}
		constexpr const T* cend()   const noexcept {return data + DIM;}

		////////////////////////////////////////////////////////////////
		// Type conversion
		////////////////////////////////////////////////////////////////
		template<int OTHER_DIM, IsScalar OTHER_T> requires std::is_nothrow_convertible<T,OTHER_T>::value
		[[nodiscard]] explicit constexpr operator Point<OTHER_DIM, OTHER_T>() const noexcept {
			Point<OTHER_DIM,OTHER_T> result;
			constexpr int min_dim = DIM < OTHER_DIM ? DIM : OTHER_DIM;
			//copy with cast valid data
			for (int i=0; i<min_dim; i++) {result[i] = static_cast<OTHER_T>(data[i]);}
			//append 0 if necessary
			for (int i=min_dim; i<OTHER_DIM; i++) {result[i] = OTHER_T{0};}
			return result;
		}

		[[nodiscard]] constexpr std::span<T,DIM> as_span() noexcept {
			return {data};
		}

		[[nodiscard]] constexpr std::span<const T,DIM> as_span() const noexcept {
			return {data};
		}

		//////////////////////////////////////////////////////////
		/// IN-PLACE BINARY OPERATIONS
		//////////////////////////////////////////////////////////
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

		constexpr Point& operator%=(const T scalar) noexcept requires(std::integral<T>) {
			for (int i=0; i<DIM; i++) {data[i] %= scalar;}
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
			assert(scalar!=U{0} && "Point::operator/=: divide by zero");
			if constexpr (std::numeric_limits<T>::is_integer) {
				for (int i=0; i<DIM; i++) {data[i] /= scalar;}
			}
			else {
				const T s_inv = static_cast<T>(U{1}/scalar);
				for (int i=0; i<DIM; i++) {data[i] *= s_inv;}
			}
			return *this;
		}

		//////////////////////////////////////////////////////////
		/// UNARY OPERATIONS
		//////////////////////////////////////////////////////////
		[[nodiscard]] constexpr Point operator-() const noexcept {
			Point result;
			for (int i=0; i<DIM; i++) {result.data[i] = -data[i];}
			return result;
		}

		Point& normalize() {
			const T nn = gutil::norm2<DIM,T>(data);
			assert(nn>T{0} && "normalized: input was the zero vector");
			return operator/=(nn);
		}

		[[nodiscard]] Point normalized() const {
			Point result{*this};
			result.normalize();
			return result;
		}

		//////////////////////////////////////////////////////////
		/// REDUCTIONS AND NORM OPERATIONS
		//////////////////////////////////////////////////////////
		[[nodiscard]] constexpr T norminfty() const noexcept {
			return gutil::norminfty<DIM,T>(data);
		}

		[[nodiscard]] constexpr T norm1() const noexcept {
			return gutil::norm1<DIM,T>(data);
		}

		[[nodiscard]] constexpr T squared_norm() const noexcept {
			return gutil::squared_norm<DIM,T>(data);
		}

		[[nodiscard]] T norm2() const noexcept {
			return gutil::norm2<DIM,T>(data);
		}

		[[nodiscard]] constexpr T prod() const noexcept {
			return gutil::product_reduce<DIM,T>(data);
		}

		[[nodiscard]] constexpr T sum() const noexcept {
			return gutil::sum_reduce<DIM,T>(data);
		}

		[[nodiscard]] constexpr T max() const noexcept {
			return gutil::max_reduce<DIM,T>(data);
		}

		[[nodiscard]] constexpr T min() const noexcept {
			return gutil::min_reduce<DIM,T>(data);
		}

		//////////////////////////////////////////////////////////
		/// BINARY VECTOR OPERATIONS
		//////////////////////////////////////////////////////////
		[[nodiscard]] constexpr T dot(const Point& other) const noexcept {
			return Point::dot(*this, other);
		}

		[[nodiscard]] constexpr Point<3,T> cross(const Point& other) const noexcept requires (DIM==3) {
			// this x other
			return Point::cross(*this, other);
		}

		//////////////////////////////////////////////////////////
		/// STATIC METHODS
		//////////////////////////////////////////////////////////
		[[nodiscard]] static constexpr T dot(const Point& left, const Point& right) noexcept {
			return gutil::dot_product_reduce<DIM,T>(left.data, right.data);
		}

		[[nodiscard]] static constexpr Point<3,T> cross(const Point& left, const Point& right) noexcept requires(DIM==3) {
			Point<3,T> result;
			result[0] = left.data[1]*right.data[2] - left.data[2]*right.data[1];
			result[1] = left.data[2]*right.data[0] - left.data[0]*right.data[2];
			result[2] = left.data[0]*right.data[1] - left.data[1]*right.data[0];
			return result;
		}

		[[nodiscard]] static constexpr T squared_norm(const Point& vec) noexcept {
			return gutil::squared_norm<DIM,T>(vec.data);
		}

		[[nodiscard]] static constexpr Point midpoint(const Point& left, const Point& right) noexcept {
			if constexpr (IsReal<T>) {return left + T{0.5}*(right - left);}
			else {return (left+right)/T{2};}
		}
	};


	///////////////////////////////////////////////////////////////////
	/// Ensure that the concept 'IsPoint' is valid
	///////////////////////////////////////////////////////////////////
	template<typename T>
	concept IsPoint = GeometryObject<T> && std::same_as<T, Point<T::DIMENSION, typename T::scalar_type>>;
	
	static_assert(IsPoint<Point<3,float>>);
	static_assert(IsPoint<Point<2,float>>);
	static_assert(IsPoint<Point<3,double>>);
	static_assert(IsPoint<Point<2,double>>);
	static_assert(IsPoint<Point<3,int>>);
	static_assert(IsPoint<Point<2,int>>);
	static_assert(IsPoint<Point<3,size_t>>);
	static_assert(IsPoint<Point<2,size_t>>);


	////////////////////////////////////////////////////////////////////
	/////// COMPARISON (CONE/ELEMENT-WISE, NOT A TOTAL ORDERING) ///////
	////////////////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool operator==(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		return gutil::elements_equal<DIM,T>(left.data, right.data);
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool operator<(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		return gutil::cone_compare_less_than<DIM,T>(left.data, right.data);
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool operator<=(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		return gutil::cone_compare_less_than_equal<DIM,T>(left.data, right.data);
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool operator>(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		return gutil::cone_compare_greater_than<DIM,T>(left.data, right.data);
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr bool operator>=(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		return gutil::cone_compare_greater_than_equal<DIM,T>(left.data, right.data);
	}

	/////////////////////////////////////////////////////////////////////////////
	//////////////////////// STANDARD BINARY OPERATIONS /////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] constexpr Point<DIM,T> operator+(Point<DIM,T> left, const Point<DIM,T>& right) noexcept {
		return left+=right;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] constexpr Point<DIM,T> operator-(Point<DIM,T> left, const Point<DIM,T>& right) noexcept {
		return left-=right;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] constexpr Point<DIM,T> operator*(Point<DIM,T> left, const Point<DIM,T>& right) noexcept {
		return left*=right;
	}

	template<int DIM, IsScalar T, IsScalar U>  requires (DIM>0) && std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] constexpr Point<DIM,T> operator*(const U left, Point<DIM,T> right) noexcept {
		return right*=static_cast<T>(left);
	}

	template<int DIM, IsScalar T, IsScalar U>  requires (DIM>0) && std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] constexpr Point<DIM,T> operator*(Point<DIM,T> left, const U right) noexcept {
		return left*=static_cast<T>(right);
	}

	template<int DIM, IsScalar T, IsScalar U>  requires (DIM>0) && std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] constexpr Point<DIM,T> operator/(Point<DIM,T> left, const U right) noexcept {
		return left/=static_cast<T>(right);
	}

	template<int DIM, IsScalar T, IsScalar U>  requires (DIM>0) && std::is_nothrow_convertible<U,T>::value
	[[nodiscard]] constexpr Point<DIM,T> operator/(const U left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result = Point<DIM,T>::Filled(static_cast<T>(left));
		return result/=right;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] constexpr Point<DIM,T> operator/(Point<DIM,T> left, const Point<DIM,T>& right) noexcept {
		return left/=right;
	}

	template<int DIM, IsScalar T> requires (DIM>0) && IsInteger<T>
	[[nodiscard]] constexpr Point<DIM,T> operator%(Point<DIM,T> left, const Point<DIM,T>& right) noexcept{
		return left%=right;
	}


	/////////////////////////////////////////////////////////////////////////////
	//////////////////////////// UTILITY OPERATIONS /////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	std::ostream& operator<<(std::ostream& os, const Point<DIM,T>& point) {
		print_to_stream<DIM,T>(os, point.data, " ");
		return os;
	}

	template<int DIM, IsScalar T> requires(DIM>0)
	[[nodiscard]] inline constexpr bool lexicographic_less(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		return gutil::lexicographic_less_than<DIM,T>(left.data, right.data);
	}


	/////////////////////////////////////////////////////////////////////////////
	//////////////////// POINT ONLY MATH/GEOMETRY OPERATIONS ////////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T dot(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		return Point<DIM,T>::dot(left,right);
	}

	template<IsScalar T>
	[[nodiscard]] inline constexpr Point<3,T> cross(const Point<3,T>& left, const Point<3,T>& right) noexcept {
		return Point<3,T>::cross(left,right);
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> midpoint(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		return Point<DIM,T>::midpoint(left,right);
	}

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline Point<DIM,T> normalized(const Point<DIM,T>& vec) noexcept {
		return vec.normalized();
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> elmin(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result;
		for (int i=0; i<DIM; i++) {
			result.data[i] = gutil::min(left.data[i], right.data[i]);
		}
		return result;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> elmin(std::span<const Point<DIM,T>> points) noexcept {
		Point<DIM,T> result{points[0]};
		for (size_t i=1; i<points.size(); ++i) {
			result = gutil::elmin(result, points[i]);
		}
		return result;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> elmax(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		Point<DIM,T> result;
		for (int i=0; i<DIM; i++) {
			result.data[i] = gutil::max(left.data[i], right.data[i]);
		}
		return result;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> elmax(std::span<const Point<DIM,T>> points) noexcept {
		Point<DIM,T> result{points[0]};
		for (size_t i=1; i<points.size(); ++i) {
			result = gutil::elmax(result, points[i]);
		}
		return result;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> clamp(const Point<DIM,T>& p, const Point<DIM,T>& lo, const Point<DIM,T>& hi) noexcept {
		assert(lo<=hi);
		Point<DIM,T> result;
		for (int i=0; i<DIM; i++) {
			result.data[i] = gutil::clamp(p.data[i], lo.data[i], hi.data[i]);
		}
		return result;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> clamp(const Point<DIM,T>& vec, const T lo, const T hi) noexcept {
		assert(lo<=hi);
		Point<DIM,T> result;
		for (int i=0; i<DIM; i++) {
			result.data[i] = gutil::clamp(vec.data[i], lo, hi);
		}
		return result;
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> lerp(const Point<DIM,T>& left, const Point<DIM,T>& right, const T t) noexcept {
		return left + t*(right-left);
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept {
		return gutil::sse_reduce<DIM,T>(left.data, right.data);
	}

	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T squared_norm(const Point<DIM,T>& vec) noexcept {
		return gutil::squared_norm<DIM,T>(vec.data);
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
	GUTIL_NO_ASSOC_MATH_START
	template<int DIM, IsScalar T, IsScalar U=T> requires (!IsExact<U> && std::is_nothrow_convertible_v<T,U>)
	[[nodiscard]] inline Point<DIM,T> sorted_sum(std::span<const Point<DIM,T>> points) noexcept {
		if (points.empty()) {return Point<DIM,T>::Zeros();}

		Point<DIM,T> result;
		std::vector<U> component;

		const size_t N = points.size();
		component.resize(N);

		for (int i = 0; i < DIM; i++) {
			for (size_t n=0; n<N; ++n) {
				component[n] = static_cast<U>(points[n].data[i]);
			}

			std::sort(component.begin(), component.end(), 
				[](U a, U b) {return gutil::abs(a) < gutil::abs(b);});

			U acc{0};
			for (U val : component) {
				acc += val;
			}

			result[i] = static_cast<T>(acc);
		}
		return result;
	}
	GUTIL_NO_ASSOC_MATH_END

	template<int DIM, IsScalar T, IsScalar U=T> requires (IsExact<U> && std::is_nothrow_convertible_v<T,U>)
	[[nodiscard]] inline Point<DIM,T> sorted_sum(std::span<const Point<DIM,T>> points) noexcept {
		//just add in any order for exact arithemtic types
		Point<DIM,U> result = Point<DIM,U>::Zeros();
		for (const Point<DIM,T>& p : points) {
			result += static_cast<Point<DIM,U>>(p);
		}
		return result;
	}


	/// Convenient ways to call the sorted sum.
	template<int DIM, IsScalar T, IsScalar U=T>
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
	template<int DIM, IsScalar T, IsScalar U=T> requires (!IsExact<U> && std::is_nothrow_convertible_v<T,U>)
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

	template<int DIM, IsScalar T, IsScalar U=T> requires (IsExact<U> && std::is_nothrow_convertible_v<T,U>)
	[[nodiscard]] inline Point<DIM,T> kahan_sum(std::span<const Point<DIM,T>> points) noexcept {
		//just add in any order for exact arithemtic types
		Point<DIM,U> result = Point<DIM,U>::Zeros();
		for (const Point<DIM,T>& p : points) {
			result += p;
		}
		return result;
	}
}


namespace std {
	//inject the hash into std
	template<gutil::IsPoint T>
	struct hash<T> {
		[[nodiscard]] size_t operator()(const T& key) const noexcept{
			size_t seed{0};
			for (int i=0; i<T::DIMENSION; ++i) {
				gutil::hash_combine(seed, key.data[i]);
			}
			return seed;
		}
	};
}
