#pragma once

#include "geometry/point.hpp"

#include <iostream>
#include <array>
#include <initializer_list>

namespace gutil {
	////////////////////////////////
	/// POLYTOPE DEFINITION
	///
	/// @tparam N   the number of vertices in the polytope
	/// @tparam DIM the spacial dimension the polytope is embedded in
	/// @tparam T   the scalar type that models the real line
	////////////////////////////////
	template<int N, int DIM, typename T=double>	requires (N>2 and DIM>1)
	class Polytope {
	public:
		//type aliases and helpful constants
		using Point_t = Point<DIM,T>;
		static constexpr int n_vertices = N;
		static constexpr int dim = DIM;

		//constructors
		constexpr Polytope() noexcept {}
		Polytope(std::initializer_list<Point_t> list) noexcept requires(list.size()==size_t(N)) {
			for (int i=0; i<N; i++) {_vertices[i] = std::move(list[i]);}
		}

		//accessors
		inline constexpr const Point_t& operator[](const int idx) const noexcept;
		inline constexpr Point_t& operator[](const int idx) noexcept;
		inline constexpr const Point_t& vertex(const int idx) const noexcept {return (*this)[idx%N];}
		inline constexpr Point_t& vertex(const int idx) noexcept {return (*this)[idx%N];}

		//iterators
		constexpr auto begin()        noexcept { return _vertices.begin(); }
		constexpr auto begin()  const noexcept { return _vertices.cbegin();}
		constexpr auto cbegin() const noexcept { return _vertices.cbegin();}
		constexpr auto end()          noexcept { return _vertices.end();   }
		constexpr auto end()    const noexcept { return _vertices.cend();  }
		constexpr auto cend()   const noexcept { return _vertices.cend();  }

		//geometry operations
		constexpr Point_t support(const Point_t& direction) const noexcept;
		constexpr Box<DIM,T> bbox() const noexcept; //axis aligned bounding box

	protected:
		std::array<Point_t, N> _vertices;
	};


	//POLYTOPE IMPLEMENTATION
	template<int N, int DIM, typename T> requires (N>2 and DIM>1)
	inline constexpr const Point<DIM,T>& Polytope<N,DIM,T>::operator[](const int idx) const noexcept {
		assert(0<=idx and idx<DIM);
		return _vertices[idx];
	}
	
	template<int N, int DIM, typename T> requires (N>2 and DIM>1)
	inline constexpr Point<DIM,T>& Polytope<N,DIM,T>::operator[](const int idx) noexcept {
		assert(0<=idx and idx<DIM);
		return _vertices[idx];
	}
	
	template<int N, int DIM, typename T> requires (N>2 and DIM>1)
	constexpr Point<DIM,T> Polytope<N,DIM,T>::support(const Point<DIM,T>& direction) const noexcept {
		T maxdot = dot(direction, _vertices[0]);
		int maxind = 0;

		for (int i=1; i<N; i++){
			const T tempdot = dot(direction, _vertices[i]);
			if (tempdot > maxdot){
				maxdot = tempdot;
				maxind = i;
			}
		}

		return _vertices[maxind];
	}
	
	template<int N, int DIM, typename T> requires (N>2 and DIM>1)
	constexpr Box<DIM,T> Polytope<N,DIM,T>::bbox() const noexcept {
		Point<DIM,T> low  = _vertices[0];
		Point<DIM,T> high = _vertices[0]; 

		for (int i=1; i<DIM; i++) {
			low  = elmin(low, _vertices[i]);
			high = elmax(high, _vertices[i]);
		}

		return Box<DIM,T>{low,high};
	}


	template<int N, int DIM, typename T> requires (N>2 and DIM>1)
	std::ostream& operator<<(std::ostream& stream, const Polytope<N,DIM,T>& polytope) {
		for (int i=0; i<polytope.n_vertices(); i++){
			stream << i << ": ";
			stream << polytope[i] << std::endl;
		}
		return stream;
	}
}
