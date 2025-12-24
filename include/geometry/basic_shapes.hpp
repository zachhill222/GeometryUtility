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
	template<int N, typename T=double>
		requires(N>2)
	class RegularPolygon : public Polytope<N,2,T> {
	public:
		using BaseType = Polytope<N,2,T>;
		static constexpr T DELTA_THETA = T(6.28318530718) / T(N);

		constexpr RegularPolygon() noexcept : BaseType() {
			for (int i=0; i<N; i++) {
				this->_vertices[i] = std::move(Point<2,T>{std::cos(i*DELTA_THETA), std::sin(i*DELTA_THETA)});
			}
		}

		constexpr RegularPolygon(const Point<2,T>& center) noexcept : RegularPolygon() {
			for (int i=0; i<N; i++) {
				this->_vertices[i] += center;
			}
		}
	};
}