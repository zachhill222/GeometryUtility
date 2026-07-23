#pragma once

#include "utility/utility.hpp"
#include "math/math.hpp"
#include "geometry/point.hpp"

#include <iostream>
#include <span>
#include <cassert>
#include <algorithm>
#include <functional>

namespace gutil {

	template<int DIM, IsScalar T> requires(DIM>0)
	struct Box {
		////////////////////////////////////////////////////////////////
		// Data and aliases
		////////////////////////////////////////////////////////////////
		using scalar_type = T;
		using point_type  = Point<DIM,T>;

		point_type low;
		point_type high;

		static constexpr int DIMENSION = DIM;
		static constexpr int N_VERTICES = (1 << DIM);

		////////////////////////////////////////////////////////////////
		// Constructors and move/copy
		////////////////////////////////////////////////////////////////
		constexpr Box() noexcept {} //note low and high are not initialized
		constexpr Box(const Box &other) noexcept = default;
		constexpr Box(Box &&other) noexcept = default;
		constexpr Box& operator=(const Box &other) = default;
		constexpr Box& operator=(Box &&other) noexcept = default;
		
		// Constructor from two points (automatically orders them)
		constexpr Box(const point_type& vertex1, const point_type& vertex2) noexcept : 
			low{gutil::elmin(vertex1, vertex2)}, 
			high{gutil::elmax(vertex1, vertex2)} {}

		template<IsScalar U> requires (std::is_nothrow_convertible_v<U,T>)
		explicit constexpr Box(U lo, U hi) noexcept :
			low{point_type::Filled(static_cast<T>(lo))}, 
			high{point_type::Filled(static_cast<T>(hi))} {assert(lo<=hi);}

		// Construct to encompass lots of points
		constexpr Box(std::span<const Point<DIM,T>> points) {
			if (points.empty()) {return;}
			low = points[0];
			high = points[0];
			for (size_t i=1; i<points.size(); ++i) {
				low = gutil::elmin(low, points[i]);
				high = gutil::elmax(high, points[i]);
			}
		}

		////////////////////////////////////////////////////////////////
		// Attributes
		////////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr point_type center() const noexcept {
			return point_type::midpoint(low, high);
		}

		[[nodiscard]] constexpr point_type sidelength() const noexcept {
			return high - low;
		}

		[[nodiscard]] T diameter() const noexcept requires(IsReal<T>) {
			return gutil::distance(high,low);
		}
		
		[[nodiscard]] constexpr T volume() const noexcept {
			return (high-low).prod();
		}


		////////////////////////////////////////////////////////////////
		// Vertex access
		////////////////////////////////////////////////////////////////		
		/// Get i-th vertex in VTK pixel/voxel order
		/// Binary encoding: bit i determines whether to use low[i] or high[i]
		[[nodiscard]] constexpr point_type vertex(const int idx) const noexcept {
			assert(idx >= 0 && idx < (1 << DIM));
			point_type vertex;
			int p = idx;
			for (int i = 0; i < DIM; i++) {
				vertex[i] = (p & 1) ? high[i] : low[i];
				p >>= 1;
			}
			return vertex;
		}

		/// Get i-th vertex in VTK quad/hexahedron order
		/// (swaps vertices 2-3 and 6-7 from voxel ordering)
		[[nodiscard]] constexpr point_type hexvertex(const int idx) const noexcept requires(DIM==2 or DIM==3) {
			switch (idx) {
				case 2: return vertex(3);
				case 3: return vertex(2);
				case 6: return vertex(7);
				case 7: return vertex(6);
				default: return vertex(idx);
			}
		}


		////////////////////////////////////////////////////////////////
		// Containment and intersection
		////////////////////////////////////////////////////////////////
		/// Check if point is in the closed box
		[[nodiscard]] constexpr bool contains(const point_type& point) const noexcept {
			return low <= point && point <= high;
		}
		
		/// Check if point is in the open box
		[[nodiscard]] constexpr bool contains_strict(const point_type& point) const noexcept {
			return low < point && point < high;
		}
		
		/// Check if this box contains the other box
		[[nodiscard]] constexpr bool contains(const Box<DIM,T>& other) const noexcept {
			return low <= other.low && other.high <= high;
		}
		
		/// Check if this box intersects the other box (check projection onto each axis)
		[[nodiscard]] constexpr bool collides_with(const Box<DIM,T>& other) const noexcept {
			for (int i = 0; i < DIM; i++) {
				if (high.data[i] < other.low.data[i] || other.high.data[i] < low.data[i]) {
					return false;
				}
			}
			return true;
		}

		/// Find the support point: vertex that maximizes dot(vertex, direction)
		[[nodiscard]] constexpr point_type support(const point_type& direction) const noexcept {
			point_type result;
			for (int i=0; i<DIM; ++i) {
				result.data[i] = (direction.data[i] < T{0}) ? low.data[i] : high.data[i];
			}
			return result;
		}


		////////////////////////////////////////////////////////////////
		// Geometric transformations
		////////////////////////////////////////////////////////////////
		
		/// Shift box by vector
		constexpr Box& operator+=(const point_type& shift) noexcept {
			low += shift;
			high += shift;
			return *this;
		}

		[[nodiscard]] constexpr Box operator+(const point_type& shift) const noexcept {
			return {low + shift, high + shift};
		}

		constexpr Box& operator-=(const point_type& shift) noexcept {
			low -= shift;
			high -= shift;
			return *this;
		}

		[[nodiscard]] constexpr Box operator-(const point_type& shift) const noexcept {
			return {low - shift, high - shift};
		}


		////////////////////////////////////////////////////////////////
		// Comparison
		////////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr bool operator==(const Box<DIM,T>& other) const noexcept {
			return low == other.low && high == other.high;
		}
	};


	///////////////////////////////////////////////////////////////////
	/// Ensure that the concept 'IsBox' is valid
	///////////////////////////////////////////////////////////////////
	template<typename T>
	concept IsBox = GeometryObject<T> && std::same_as<T, Box<T::DIMENSION, typename T::scalar_type>>;
	
	static_assert(IsBox<Box<3,float>>);
	static_assert(IsBox<Box<2,float>>);
	static_assert(IsBox<Box<3,double>>);
	static_assert(IsBox<Box<2,double>>);
	static_assert(IsBox<Box<3,int>>);
	static_assert(IsBox<Box<2,int>>);
	static_assert(IsBox<Box<3,size_t>>);
	static_assert(IsBox<Box<2,size_t>>);


	////////////////////////////////////////////////////////////////
	// Utility functions
	////////////////////////////////////////////////////////////////
	template<int DIM, IsScalar T> requires (DIM>0)
	std::ostream& operator<<(std::ostream& os, const Box<DIM,T>& box) {
		return os << "Box{low= {" << box.low << "}, high={" << box.high << "}}";
	}

	template<int DIM, IsScalar T>
	[[nodiscard]] inline constexpr bool lexicographic_less(const Box<DIM,T>& left, const Box<DIM,T>& right) noexcept {
		if (gutil::lexicographic_less(left.low, right.low)) {return true;}
		if (gutil::lexicographic_less(right.low, left.low)) {return false;}
		return gutil::lexicographic_less(left.high, right.high);
	}


	/////////////////////////////////////////////////////////////////////////////
	///////////////////// BOX ONLY MATH/GEOMETRY OPERATIONS /////////////////////
	/////////////////////////////////////////////////////////////////////////////
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


	/////////////////////////////////////////////////////////////////////////////
	//////////////////// BOX/POINT MATH/GEOMETRY OPERATIONS /////////////////////
	/////////////////////////////////////////////////////////////////////////////
	/// Return 'union' of two boxes (minimal box that contains both inputs)
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Box<DIM,T> merge(const Box<DIM,T>& A, const Box<DIM,T>& B) noexcept {
		return {gutil::elmin(A.low, B.low), gutil::elmax(A.high, B.high)};
	}

	/// Return the smallest box the contains the given box and point
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Box<DIM,T> expand(const Box<DIM,T>& A, const Point<DIM,T>& B) noexcept {
		return {gutil::elmin(A.low, B), gutil::elmax(A.high, B)};
	}

	/// Return intersection of two boxes (undefined if boxes don't intersect)
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Box<DIM,T> intersection(const Box<DIM,T>& A, const Box<DIM,T>& B) noexcept {
		assert(A.collides_with(B));
		return {gutil::elmax(A.low, B.low), gutil::elmin(A.high, B.high)};
	}

	/// Project/clamp a point to a box
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> clamp(const Point<DIM,T>& A, const Box<DIM,T>& B) noexcept {
		return gutil::clamp(A, B.low, B.high);
	}

	/// Project/clamp a point to a box periodically
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr Point<DIM,T> clamp_periodic(const Point<DIM,T>& A, const Box<DIM,T>& B) noexcept {
		if constexpr (IsInteger<T>) {
			return B.low + (A - B.low)%(B.high - B.low);
		}
		else {
			return B.low + gutil::fmod((A - B.low),(B.high - B.low));
		}
	}

	/// Return the squared distance from a point to the box.
	/// The distance is 0 if the point is contained in the box.
	template<int DIM, IsScalar T> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const Box<DIM,T>& A, const Point<DIM,T>& B) noexcept {
		return gutil::distance_squared(gutil::clamp(B, A), B);
	}
	
	template<int DIM, IsScalar T>
	[[nodiscard]] inline constexpr bool collides( const Box<DIM,T>& A, const Box<DIM,T>& B) noexcept {
		return A.collides_with(B);
	}
}


namespace std
{
	//inject the hash into std
	template<gutil::IsBox T>
	struct hash<T> {
		[[nodiscard]] size_t operator()(const T& key) const noexcept {
			size_t seed{0};
			gutil::hash_combine(seed, key.low);
			gutil::hash_combine(seed, key.high);
			return seed;
		}
	};
}