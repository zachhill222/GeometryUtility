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
#include <limits>

namespace gutil {
	template<typename T>
	concept IsBox = GutilGeometryObject<T> && std::same_as<T, Box<T::DIMENSION, typename T::scalar_type>>;

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


		////////////////////////////////////////////////////////////////
		// Attributes
		////////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr point_type center() const noexcept {
			if constexpr (IsReal<T>) {return low + T{0.5}*(high-low);}
			else {return (high-low)/T{2};}
		}
		[[nodiscard]] constexpr point_type sidelength() const noexcept {return high - low;}
		[[nodiscard]] T diameter() const noexcept requires(IsReal<T>) {return gutil::distance(high,low);}
		[[nodiscard]] constexpr T volume() const noexcept {return gutil::product_reduce(high-low);}

		////////////////////////////////////////////////////////////////
		// Vertex access
		////////////////////////////////////////////////////////////////
		
		/// Get i-th vertex in VTK pixel/voxel order
		/// Binary encoding: bit i determines whether to use low[i] or high[i]
		[[nodiscard]] constexpr point_type voxelvertex(const int idx) const noexcept {
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
		[[nodiscard]] constexpr bool intersects(const Box<DIM,T>& other) const noexcept {
			return gutil::intersect(*this, other);
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
}


namespace std
{
	//inject the hash into std
	template<gutil::IsBox T>
	struct hash<T> {
		[[nodiscard]] size_t operator()(const T& key) const noexcept{
			size_t seed{0};
			gutil::hash_combine(seed, key.low);
			gutil::hash_combine(seed, key.high);
			return seed;
		}
	};
}