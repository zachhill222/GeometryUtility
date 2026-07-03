#pragma once

#include "geometry/point.hpp"
#include <iostream>
#include <cmath>

namespace gutil {

	template <typename T=double>
	class Quaternion {
	public:
		//// INITIALIZERS
		Quaternion(): _q0(1), _qv(Point<3,T> {0,0,0}) {}
		Quaternion(const T q0, const T q1, const T q2, const T q3): _q0(q0), _qv(Point<3,T> {q1,q2,q3} ) {}
		Quaternion(const T q0, const Point<3,T> qv): _q0(q0), _qv(Point<3,T> {qv[0], qv[1], qv[2]}) {}

		//// ATTRIBUTES
		constexpr T operator[](int idx) const;
		constexpr T& operator[](int idx);

		constexpr T q0() const;
		constexpr Point<3,T> qv() const;
		constexpr T& q0();
		constexpr Point<3,T>& qv();

		//// FACTORIES
		static constexpr Quaternion identity() {return Quaternion{T{1}, T{0}, T{0}, T{0}};}
		static constexpr Quaternion angle_axis(const T theta, const Point<3,T>& axis) {
			Quaternion q;
			return q.setrotation(theta, axis);
		}

		//// ROTATIONS
		constexpr Quaternion conj() const;
		constexpr Quaternion inv() const;
		constexpr T squaredNorm() const;
		constexpr T norm() const;
		constexpr Quaternion& normalize(); //normalize this quaternion to a rotation quaternion
		constexpr Quaternion& setrotation(const T& theta, const Point<3,T>& axis);
		constexpr Point<3,T> rotate(const Point<3,T>& point) const; //rotate point
		constexpr bool is_rotation() const;

		///// ARITHMETIC
		constexpr Quaternion& operator+=(const Quaternion& other);
		constexpr Quaternion  operator+(const Quaternion& other) const;
		constexpr Quaternion& operator-=(const Quaternion& other);
		constexpr Quaternion  operator-(const Quaternion& other) const;
		constexpr Quaternion& operator*=(const Quaternion& other);
		constexpr Quaternion  operator*(const Quaternion& other) const;
		constexpr Quaternion& operator/=(const Quaternion& other);
		constexpr Quaternion  operator/(const Quaternion& other) const;
		bool operator==(const Quaternion& other) const;
		inline bool operator!=(const Quaternion& other) const { return !(operator==(other));}
		
	private:
		T _q0;
		Point<3,T> _qv;
	};


	//// IMPLEMENTATION
	template <typename T>
	constexpr T Quaternion<T>::operator[](int idx) const{
		assert(0<=idx && idx<4);
		return idx==0 ? _q0 : _qv[idx-1];
	}

	template <typename T>
	constexpr T& Quaternion<T>::operator[](int idx){
		assert(0<=idx && idx<4);
		return idx==0 ? _q0 : _qv[idx-1];
	}

	template <typename T>
	constexpr T Quaternion<T>::q0() const {return _q0;}

	template <typename T>
	constexpr T& Quaternion<T>::q0() {return _q0;}

	template <typename T>
	constexpr Point<3,T> Quaternion<T>::qv() const {return _qv;}

	template <typename T>
	constexpr Point<3,T>& Quaternion<T>::qv() {return _qv;}

	template <typename T>
	constexpr Quaternion<T> Quaternion<T>::conj() const{
		return Quaternion(_q0, -_qv);
	}

	template <typename T>
	constexpr Quaternion<T> Quaternion<T>::inv() const{
		T C = 1.0/squaredNorm();
		return Quaternion(C*_q0, (-C)*_qv);
	}

	template <typename T>
	constexpr T Quaternion<T>::squaredNorm() const{
		return _q0*_q0 + gv::squaredNorm(_qv);
	}

	template <typename T>
	constexpr T Quaternion<T>::norm() const{
		return std::sqrt(squaredNorm());
	}

	template <typename T>
	constexpr Quaternion<T>& Quaternion<T>::normalize(){
		T C = 1.0/norm();
		_q0*=C;
		_qv*=C;
		return *this;
	}

	template <typename T>
	constexpr Quaternion<T>& Quaternion<T>::setrotation(const T& theta, const Point<3,T>& axis){
		_q0 = std::cos(0.5*theta);
		_qv = std::sin(0.5*theta) * gv::normalize(axis);
		return *this;
	}

	template <typename T>
	constexpr Point<3,T> Quaternion<T>::rotate(const Point<3,T>& point) const {
		Quaternion V = Quaternion(0.0, point);
		V = (*this) * (V*conj());
		return V.qv();
	}

	template <typename T>
	constexpr bool Quaternion<T>::is_rotation() const {
		return gutil::abs(squaredNorm() - T{1}) <= T{1e-12};
	}

	///// ARITHMETIC
	template <typename T>
	constexpr Quaternion<T>& Quaternion<T>::operator+=(const Quaternion& other){
		_q0+=other.q0();
		_qv+=other.qv();
		return *this;
	}

	template <typename T>
	constexpr Quaternion<T> Quaternion<T>::operator+(const Quaternion& other) const{
		return Quaternion(_q0+other.q0(), _qv+other.qv());
	}

	template <typename T>
	constexpr Quaternion<T>& Quaternion<T>::operator-=(const Quaternion& other){
		_q0-=other.q0();
		_qv-=other.qv();
		return *this;
	}

	template <typename T>
	constexpr Quaternion<T> Quaternion<T>::operator-(const Quaternion& other) const{
		return Quaternion(_q0-other.q0(), _qv-other.qv());
	}

	template <typename T>
	constexpr Quaternion<T>& Quaternion<T>::operator*=(const Quaternion& other){
		T Q0 = _q0*other.q0() - dot(_qv,other.qv());
		_qv = _q0*other.qv() + other.q0()*_qv + cross(_qv,other.qv());
		_q0 = Q0;
		return *this;
	}

	template <typename T>
	constexpr Quaternion<T> Quaternion<T>::operator*(const Quaternion& other) const{
		T Q0 = _q0*other.q0() - dot(_qv,other.qv());
		Point<3,T>  QV = _q0*other.qv() + other.q0()*_qv + cross(_qv,other.qv());
		return Quaternion(Q0, QV);
	}

	template <typename T>
	constexpr Quaternion<T>& Quaternion<T>::operator/=(const Quaternion& other){
		operator*=(other.inv());
		return *this;
	}

	template <typename T>
	constexpr Quaternion<T> Quaternion<T>::operator/(const Quaternion& other) const{
		return operator*(other.inv());
	}
	
	template <typename T>
	constexpr bool Quaternion<T>::operator==(const Quaternion& other) const{
		if (_q0 != other.q0()){
			return false;
		}

		if (_qv != other.qv()){
			return false;
		}

		return true;
	}


	///Print to ostream.
	template <typename T>
	std::ostream& operator<<(std::ostream& os, const Quaternion<T> &quaternion){
		for (int i = 0; i < 4; i++){
			os << quaternion[i] << " ";
		}
		return os;
	}
}

