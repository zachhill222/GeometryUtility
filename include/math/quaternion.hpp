#pragma once

#include "utility/utility.hpp"
#include "math/math.hpp"

#include <iostream>
#include <cassert>

namespace gutil {
	template<int DIM, IsScalar T> requires (DIM>0)
	struct Point;


	template<IsReal T>
	struct Quaternion
	{
		////////////////////////////////////////////////////////////////
		// Data and aliases
		////////////////////////////////////////////////////////////////
		using scalar_type = T;
		using point_type = Point<3,T>;
		
		T data[4];

		static constexpr int DIMENSION = 4;

		////////////////////////////////////////////////////////////////
		// Constructors and move/copy
		////////////////////////////////////////////////////////////////
		constexpr Quaternion() noexcept {} //note data[] is not initialized
		constexpr Quaternion(const Quaternion& other) noexcept = default;
		constexpr Quaternion(Quaternion&& other) noexcept = default;
		constexpr Quaternion& operator=(const Quaternion& other) noexcept = default;
		constexpr Quaternion& operator=(Quaternion&& other) noexcept = default;
		
		constexpr Quaternion(const T q0, const T q1, const T q2, const T q3) noexcept : data{q0, q1, q2, q3} {}
		Quaternion(T theta, point_type axis) noexcept {
			//a rotaion quaternion has the form
			// 		cos(theta/2) + sin(theta/2)( x*i + y*j + z*k)
			//where (x,y,z) is a unit vector.
			theta *= T{0.5};
			axis.normalize() *= gutil::sin(theta);
			
			data[0] = gutil::cos(theta);
			data[1] = axis[0];
			data[2] = axis[1];
			data[3] = axis[2];
		}

		static constexpr Quaternion Identity() noexcept {return {T{1}, T{0}, T{0}, T{0}};}
		static Quaternion Rotation(T theta, point_type norm_axis) noexcept {
			theta *= T{0.5};
			norm_axis *= gutil::sin(theta);
			return {gutil::cos(theta), norm_axis[0], norm_axis[1], norm_axis[2]};
		}

		////////////////////////////////////////////////////////////////
		// Element access
		////////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr T operator[](const int idx) const noexcept {
			assert(0<=idx and idx<4);
			return data[idx];
		}
		
		[[nodiscard]] constexpr T& operator[](const int idx) noexcept {
			assert(0<=idx and idx<4);
			return data[idx];
		}

		[[nodiscard]] constexpr T q0() const noexcept {
			return data[0];
		}

		[[nodiscard]] constexpr point_type qv() const noexcept {
			return {data[1], data[2], data[3]};
		}

		constexpr T*       begin()        noexcept {return data;}
		constexpr const T* begin()  const noexcept {return data;}
		constexpr T*       end()          noexcept {return data + 4;}
		constexpr const T* end()    const noexcept {return data + 4;}
		constexpr const T* cbegin() const noexcept {return data;}
		constexpr const T* cend()   const noexcept {return data + 4;}

		////////////////////////////////////////////////////////////////
		// Rotations
		////////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr Quaternion conj() const  noexcept {
			return {data[0], -data[1], -data[2], -data[3]};
		}

		[[nodiscard]] constexpr Quaternion inv() const noexcept {
			assert(this->squared_norm() > T{0});
			const T n_inv = T{-1}/squared_norm();
			return {-n_inv*data[0], n_inv*data[1], n_inv*data[2], n_inv*data[3]};
		}

		[[nodiscard]] constexpr T squared_norm() const noexcept {
			return gutil::squared_norm<4,T>(data);
		}

		[[nodiscard]] T norm() const noexcept {
			return gutil::sqrt(this->squared_norm());
		}

		[[nodiscard]] constexpr point_type rotate(const point_type& point) const noexcept {
			Quaternion V{T{0}, point.data[0], point.data[1], point.data[2]};
			V = V * this->conj();
			V = (*this) * V;
			return V.qv();
		}

		[[nodiscard]] constexpr bool is_rotation() const noexcept {
			return this->squared_norm() == T{1}; //TODO: replace by IsNear
		}

		Quaternion& normalize() noexcept {
			gutil::in_place_divide<4,T>(data, this->norm());
			return *this;
		}

		////////////////////////////////////////////////////////////////
		// In-place arithmetic
		////////////////////////////////////////////////////////////////
		constexpr Quaternion& operator+=(const Quaternion& other) noexcept {
			gutil::in_place_sum<4,T>(data, other.data);
			return *this;
		}

		constexpr Quaternion& operator-=(const Quaternion& other) noexcept {
			gutil::in_place_subtract<4,T>(data, other.data);
			return *this;
		}

		constexpr Quaternion& operator*=(const Quaternion& other) noexcept {
			//q0 = q0*other.q0 - dot(qv,other.qv)
			const T Q0 = data[0]*other.data[0] - (data[1]*other.data[1] + data[2]*other.data[2] + data[3]*other.data[3]);

			//      qv =      q0*other.qv      +      qv*other.q0      +                 qv x other.qv                
			const T Q1 = data[0]*other.data[1] + data[1]*other.data[0] + data[2]*other.data[3] - data[3]*other.data[2];
			const T Q2 = data[0]*other.data[2] + data[2]*other.data[0] + data[3]*other.data[1] - data[1]*other.data[3];
			const T Q3 = data[0]*other.data[3] + data[3]*other.data[0] + data[1]*other.data[2] - data[2]*other.data[1];

			//update data
			data[0] = Q0; data[1] = Q1; data[2] = Q2; data[3] = Q3;
			return *this;
		}

		constexpr Quaternion& operator/=(const Quaternion& other) noexcept {
			return operator*=(other.inv());
		}


		////////////////////////////////////////////////////////////////
		// Unary operators
		////////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr Quaternion operator-() const noexcept {
			return {-data[0], -data[1], -data[2], -data[3]};
		}


		////////////////////////////////////////////////////////////////
		// Static methods
		////////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr point_type rotate(const Quaternion& q, const point_type& p) noexcept {
			assert(q.is_rotation());
			return q.rotate(p);
		}
	};


	/////////////////////////////////////////////////////////////////////////////
	//////////////////////// STANDARD BINARY OPERATIONS /////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<IsReal T>
	[[nodiscard]] inline constexpr Quaternion<T> operator+(Quaternion<T> left, const Quaternion<T>& right) noexcept {
		return left+=right;
	}

	template<IsReal T>
	[[nodiscard]] inline constexpr Quaternion<T> operator-(Quaternion<T> left, const Quaternion<T>& right) noexcept {
		return left-=right;
	}

	template<IsReal T>
	[[nodiscard]] inline constexpr Quaternion<T> operator*(Quaternion<T> left, const Quaternion<T>& right) noexcept {
		return left*=right;
	}

	template<IsReal T>
	[[nodiscard]] inline constexpr Quaternion<T> operator/(Quaternion<T> left, const Quaternion<T>& right) noexcept {
		return left/=right;
	}
	
	template<IsReal T>
	[[nodiscard]] inline constexpr bool operator==(Quaternion<T> left, const Quaternion<T>& right) noexcept {
		return gutil::elements_equal<4,T>(left.data, right.data);
	}


	/////////////////////////////////////////////////////////////////////////////
	//////////////////////////// UTILITY OPERATIONS /////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<IsReal T>
	std::ostream& operator<<(std::ostream& os, const Quaternion<T>& quaternion){
		print_to_stream<4,T>(os, quaternion.data, " ");
		return os;
	}
}

