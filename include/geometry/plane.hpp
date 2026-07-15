#pragma once

#include "geometry/point.hpp"
#include "math/quaternion.hpp"

#include <iostream>

namespace gutil {

	template<typename T=double>
	class Plane {
	public:
		using point_type = Point<3,T>;

		Plane(): _origin(point_type{0,0,0}), _normal(point_type{0,0,1}) {calcbasis();}
		Plane( const point_type& origin, const point_type& normal): _origin(origin), _normal(normalize(normal)) {calcbasis();}
		Plane( const point_type& p1, const point_type& p2, const point_type& p3): _origin(p1), _normal(normalize(cross(p3-p1,p2-p1))) {calcbasis();}

		T dist(const point_type& point) const; //signed distence to the plane

		point_type project(const point_type& point) const; //project point from global coordinates to the plane along the _normal direction. return a 2D point in local coordinates.

		point_type tolocal(const point_type& point) const; //write a point in global coordinates in terms of local (_basis[0], _basis[1], _normal) coordinates.

		point_type toglobal(const point_type& point) const; //write a local point in the global coordinate system (works for 2D points on the plane or 3D point in local coordinates).

	private:
		point_type _origin;
		point_type _normal;
		point_type _basis[2];

		void calcbasis();
	};

	template<typename T>
	void Plane<T>::calcbasis()
	{
		point_type vec1, vec2;

		vec1 = cross(_normal,point_type{1,0,0});
		vec2 = cross(_normal,point_type{0,1,0});
		if (squaredNorm(vec2) > squaredNorm(vec1)) {vec1 = vec2;
		}

		vec2 = cross(_normal,point_type{0,0,1});
		if (squaredNorm(vec2) > squaredNorm(vec1)) {vec1 = vec2;}

		_basis[0] = vec1;
		_basis[1] = cross(_normal,_basis[0]);
	}


	template<typename T>
	T Plane<T>::dist(const point_type& point) const {return dot(point-_origin,_normal);}

	template<typename T>
	typename Plane<T>::point_type Plane<T>::project(const point_type& point) const
	{
		point_type local = tolocal(point); //write point in local coordinates
		return local[0]*_basis[0] + local[1]*_basis[1];
	}

	template<typename T>
	typename Plane<T>::point_type Plane<T>::tolocal(const point_type& point) const
	{
		point_type shift = point - _origin;
		T a = dot(shift,_basis[0]);
		T b = dot(shift,_basis[1]);
		T c = dot(shift,_normal);
		return point_type(a,b,c);
	}

	template<typename T>
	typename Plane<T>::point_type Plane<T>::toglobal(const point_type& point) const
	{
		point_type result = _origin + point[0]*_basis[0] + point[1]*_basis[1] + point[2]*_normal;
		return result;
	}
}