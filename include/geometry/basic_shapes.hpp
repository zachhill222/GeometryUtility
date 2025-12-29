#pragma once

#include "geometry/point.hpp"
#include "geometry/polytope.hpp"

#include <initializer_list>
#include <cmath>

namespace gutil {
	//SIMPLEX DEFINITION
	template<int DIM=3, typename T=double>
	class Simplex : public Polytope<DIM+1,DIM,T> {
	public:
		using BaseType = Polytope<DIM+1,DIM,T>;
		using typename BaseType::Point_t;

		constexpr Simplex() noexcept : BaseType() {
			for (int i=1; i<DIM+1; i++){
				this->_vertices[i] = Point<DIM,T>(T(0));
				this->_vertices[i][i-1] = T(1);
			}
		}

		Simplex(std::initializer_list<Point_t> list) noexcept : BaseType(list) {}
	};

	//REGULAR POLYGON
	template<int N, typename T=double> requires(N>2)
	class RegularPolygon : public Polytope<N,2,T> {
	public:
		using BaseType = Polytope<N,2,T>;
		using Point_t  = Point<2,T>;

		static constexpr T DELTA_THETA = T(6.28318530718) / T(N);

		constexpr RegularPolygon() noexcept :
			BaseType(),
			m_center{T(0)},
			m_inner_radius{std::cos(0.5*DELTA_THETA)},
			m_outer_radius{1}
		{
			for (int i=0; i<N; i++) {
				this->_vertices[i] = std::move(Point_t{std::cos(i*DELTA_THETA), std::sin(i*DELTA_THETA)});
			}
		}

		constexpr RegularPolygon(const Point_t& center) noexcept :
			RegularPolygon()
		{
			*this += center;
		}

		constexpr RegularPolygon& operator+=(const Point_t& shift) noexcept
		{
			m_center += shift;
			for (Point_t& v : *this) {v+=shift;}
			return *this;
		}

		constexpr RegularPolygon& operator*=(const T& scale) noexcept
		{
			assert(scale>T{0});
			m_inner_radius *= scale;
			m_outer_radius *= scale;

			for (Point_t& v : *this) {
				Point_t old_delta = v-m_center;
				v += scale*old_delta;
			}

			return *this;
		}

		inline constexpr const Point_t& center() const noexcept {return m_center;}

		//tangent edge in the positive orientation (not normalized)
		constexpr Point_t edge_tangent(const int i) const noexcept
		{
			const Point_t& p1 = this->vertex(i);
			const Point_t& p2 = this->vertex(i+1);
			return p2 - p1;
		}

		//outward normal of the ith edge (not normalized)
		constexpr Point_t outer_normal(const int i) const noexcept
		{
			const Point_t tangent = edge_tangent(i);
			return Point_t{tangent[1], tangent[0]};
		}

		constexpr bool contains(const Point_t& point) const noexcept
		{
			const Point_t shifted_point = point-m_center;
			T dist2 = squaredNorm(shifted_point);
			if (dist2 < m_inner_radius*m_inner_radius) {return true;}
			else if (dist2 > m_outer_radius*m_outer_radius) {return false;}

			//check dot product with all normal vectors
			for (int i=0; i<N; i++) {
				if (dot(shifted_point, outer_normal(i)) > T{0}) {return false;}
			}

			//point is either interior or on the boundary
			return true;
		}

		constexpr bool intersects(const Box<2,T>& box) const noexcept
		{
			//check if box contains a vertex
			for (const Point_t& v : *this) {
				if (box.contains(v)) {return true;}
			}

			//check if polygon contains a box vertex
			for (int i=0; i<4; i++) {
				if (contains(box.voxelvertex(i))) {return true;}
			}

			//both shapes are convex polygons if they intersect, then one must contain a vertex of the other
			return false;
		}

	private:
		Point_t m_center;
		T m_inner_radius; //center to edge midpoint
		T m_outer_radius; //center to vertex
	};
}