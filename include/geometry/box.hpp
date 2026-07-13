#pragma once

#include "math/gutilmath.hpp"
#include "geometry/point.hpp"

#include <string>
#include <sstream>
#include <cmath>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <type_traits>

namespace gutil {
	template<int DIM=3, IsScalar T=double> requires(DIM>0)
	struct Box {
		static constexpr int dim = DIM;
		static constexpr int n_vertices = (1 << DIM);
		using scalar_type = T;
		using point_type  = Point<DIM,T>;

		point_type low;
		point_type high;

		////////////////////////////////////////////////////////////////
		// Constructors
		////////////////////////////////////////////////////////////////
		
		// Default constructor: unit box centered at origin
		constexpr Box() noexcept : low(point_type::Filled(-1.0)), high(point_type::Filled(1.0)) {}

		// Constructor from two points (automatically orders them)
		constexpr Box(const point_type &vertex1, const point_type &vertex2) noexcept : 
			low(elmin(vertex1, vertex2)), high(elmax(vertex1, vertex2)) {
			assert(low <= high);
		}

		//use default copy and move constructors and assignment
		constexpr Box(const Box &other) noexcept = default;
		constexpr Box(Box &&other) noexcept = default;
		constexpr Box& operator=(const Box &other) = default;
		constexpr Box& operator=(Box &&other) noexcept = default;
		
		////////////////////////////////////////////////////////////////
		// Attributes
		////////////////////////////////////////////////////////////////
		constexpr point_type center() const noexcept {return T(0.5) * (low + high);}
		constexpr point_type sidelength() const noexcept {return high - low;}
		T diameter() const noexcept {return norm2(high - low);}
		constexpr T volume() const noexcept {
			T vol = 1;
			for (int i = 0; i < DIM; i++) {
				vol *= (high[i] - low[i]);
			}
			return vol;
		}

		////////////////////////////////////////////////////////////////
		// Vertex access
		////////////////////////////////////////////////////////////////
		
		/// Get i-th vertex in VTK pixel/voxel order
		/// Binary encoding: bit i determines whether to use low[i] or high[i]
		constexpr point_type voxelvertex(const int idx) const noexcept {
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
		constexpr point_type hexvertex(const int idx) const noexcept requires(DIM==2 or DIM==3) {
			switch (idx) {
				case 2: return voxelvertex(3);
				case 3: return voxelvertex(2);
				case 6: return voxelvertex(7);
				case 7: return voxelvertex(6);
				default: return voxelvertex(idx);
			}
		}
		////////////////////////////////////////////////////////////////
		// Containment and intersection
		////////////////////////////////////////////////////////////////
		/// Check if point is in the closed box
		constexpr bool contains(const point_type &point) const noexcept {
			return low <= point && point <= high;
		}
		
		/// Check if point is in the open box
		constexpr bool contains_strict(const point_type &point) const noexcept {
			return low < point && point < high;
		}
		
		/// Check if this box contains the other box
		constexpr bool contains(const Box<DIM,T> &other) const noexcept {
			return low <= other.low && other.high <= high;
		}
		
		/// Check if this box intersects the other box (check projection onto each axis)
		constexpr bool intersects(const Box<DIM,T> &other) const noexcept {
			for (int i = 0; i < DIM; i++) {
				if (high[i] < other.low[i] || other.high[i] < low[i]) {
					return false;
				}
			}
			return true;
		}

		/// Find the support point: vertex that maximizes dot(vertex, direction)
		constexpr point_type support(const point_type &direction) const noexcept {
			point_type result{low};
			for (int i=0; i<DIM; ++i) {
				if (direction[i] < T{0}) {result[i] = high[i];}
			}
			return result;
		}

		////////////////////////////////////////////////////////////////
		// Geometric transformations
		////////////////////////////////////////////////////////////////
		
		/// Shift box by vector
		constexpr Box& operator+=(const point_type &shift) noexcept {
			low += shift;
			high += shift;
			return *this;
		}

		constexpr Box operator+(const point_type &shift) const noexcept {
			return Box(low + shift, high + shift);
		}

		constexpr Box& operator-=(const point_type &shift) noexcept {
			low -= shift;
			high -= shift;
			return *this;
		}

		constexpr Box operator-(const point_type &shift) const noexcept {
			return Box(low - shift, high - shift);
		}

		/// Scale box relative to its center
		template<typename U>
		constexpr Box& operator*=(const U& scale) noexcept {
			point_type c = center();
			T s = static_cast<T>(scale);
			low = c + s * (low - c);
			high = c + s * (high - c);
			return *this;
		}

		template<typename U>
		constexpr Box operator*(const U& scale) const noexcept requires(std::is_convertible<U,T>::value) {
			point_type c = center();
			T s = static_cast<T>(scale);
			return Box(c + s * (low - c), c + s * (high - c));
		}

		template<typename U>
		constexpr Box& operator/=(const U& scale) {
			return (*this) *= (T(1) / static_cast<T>(scale));
		}

		template<typename U>
		constexpr Box operator/(const U& scale) const {
			return (*this) * (T(1) / static_cast<T>(scale));
		}

		////////////////////////////////////////////////////////////////
		// Comparison
		////////////////////////////////////////////////////////////////
		
		constexpr bool operator==(const Box<DIM,T> &other) const {
			return low == other.low && high == other.high;
		}

		constexpr bool operator!=(const Box<DIM,T> &other) const {
			return !(*this == other);
		}
	};

	////////////////////////////////////////////////////////////////
	// Free functions
	////////////////////////////////////////////////////////////////
	template <int DIM, typename T>
	Box<DIM,T> combine(const Box<DIM,T>& A, const Box<DIM,T>& B) {
		return {elmin(A.low,B.low), elmax(A.high,B.high)};
	}

	/// Return intersection of two boxes (undefined if boxes don't intersect)
	template <int DIM, typename T>
	Box<DIM,T> intersection(const Box<DIM,T>& A, const Box<DIM,T>& B) {
		assert(A.intersects(B));
		return Box(elmax(A.low, B.low), elmin(A.high, B.high));
	}


	template <int DIM, typename T, typename U>
	constexpr Box<DIM,T> operator*(const U &scale, const Box<DIM,T> &box) {
		return box * scale;
	}

	template <int DIM, typename T>
	constexpr T distance_squared(const Box<DIM,T> &box, const Point<DIM,T> &point) {
		if (box.contains(point)) {
			return T{0};
		}

		// Compute distance to closest point on box surface
		T dist_sq = 0;
		for (int i=0; i<DIM; i++) {
			if (point[i] < box.low[i]) {
				T diff = box.low[i] - point[i];
				dist_sq += diff * diff;
			} else if (point[i] > box.high[i]) {
				T diff = point[i] - box.high[i];
				dist_sq += diff * diff;
			} //no contribution in this axis if the point is in the box interval
		}
		return dist_sq;
	}

	template <int DIM, typename T>
	inline T distance(const Box<DIM,T> &box, const Point<DIM,T> &point) {
		return std::sqrt(distance_squared(box, point));
	}

	template<int DIM, typename T>
	std::ostream& operator<<(std::ostream& os, const Box<DIM,T>& box) {
		return os << "(" << box.low << ") to (" << box.high << ")";
	}
}