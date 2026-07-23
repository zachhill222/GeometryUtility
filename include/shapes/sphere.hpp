#pragma once

#include "shapes/shape_base.hpp"
#include "algorithms/convex_collision.hpp"

namespace gutil
{
	template<int DIM, IsReal T> requires (DIM>0)
	struct Sphere : public BaseNonRotatableShape<DIM, T, Sphere<DIM, T>> {
		using BASE = BaseNonRotatableShape<DIM, T,Sphere<DIM, T>>;
		using point_type = typename BASE::point_type;
		using box_type = typename BASE::box_type;

		using BASE::center;
		T radius{1};

		constexpr Sphere() noexcept : BASE() {}
		constexpr Sphere(const point_type& cntr, const T r) noexcept : BASE(cntr), radius(r) {}

		[[nodiscard]] constexpr point_type support_impl(const point_type& direction) const noexcept {
			if (direction == point_type::Zeros()) { return center; }
			return center + radius*gutil::normalized(direction);
		}

		[[nodiscard]] constexpr box_type bbox_impl() const noexcept {
			return box_type{ point_type{center[0]-radius, center[1]-radius, center[2]-radius},
							 point_type{center[0]+radius, center[1]+radius, center[2]+radius} };
		}

		[[nodiscard]] constexpr T signed_distance_impl(const point_type& point) const noexcept {
			//signed distance: positive is outside, negative is inside
			return this->dist2center(point) - radius;
		}

		[[nodiscard]] constexpr T distance_sq_impl(const point_type& point) const noexcept {
			const T dd = signed_distance_impl(point);
			return gutil::max( T{0}, dd*dd);
		}

		[[nodiscard]] constexpr T distance_impl(const point_type& point) const noexcept {
			return gutil::max( T{0}, signed_distance_impl(point));
		}
	};

	//define collides so CPO finds this rather than calling GJK
	template<int DIM, IsReal T>
	inline constexpr bool collides(const Sphere<DIM,T>& A, const Sphere<DIM,T>& B) noexcept {
		const T rr = A.radius + B.radius;
		return gutil::distance_squared(A.center,B.center) <= rr*rr;
	}

	template<int DIM, IsReal T>
	inline constexpr bool collides(const Box<DIM,T>& A, const Sphere<DIM,T>& B) noexcept {
		return gutil::distance_squared(A, B.center) <= B.radius*B.radius;
	}

	template<int DIM, IsReal T>
	inline std::ostream& operator<<(std::ostream& os, const Sphere<DIM,T>& sphere) {
		os << "Sphere{center={ " << sphere.center << "}, radius={ " << sphere.radius << " }}";
		return os; 
	}

}