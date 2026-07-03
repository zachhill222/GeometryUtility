#pragma once

#include "shapes/shape_base.hpp"


namespace gutil
{
	template<typename T=float>
	struct Sphere : public BaseNonRotatableShape<T,Sphere<T>>
	{
		using BASE = BaseNonRotatableShape<T,Sphere<T>>;
		using point_type = typename BASE::point_type;
		using box_type = typename BASE::box_type;

		using BASE::center;
		T radius{1};

		constexpr Sphere() noexcept : BASE() {}
		constexpr Sphere(const T r) noexcept : BASE(), radius(r) {}
		constexpr Sphere(const T r, const point_type& cntr) noexcept : BASE(cntr), radius(r) {}

		constexpr point_type support_impl(const point_type& direction) const noexcept {
			return center + radius*gutil::normalize(direction);
		}

		constexpr box_type bbox_impl() const noexcept {
			return box_type{ point_type{center[0]-radius, center[1]-radius, center[2]-radius},
							 point_type{center[0]+radius, center[1]+radius, center[2]+radius} };
		}

		constexpr T sgndist2surface(const point_type& point) const noexcept {
			//signed distance: positive is outside, negative is inside
			return this->dist2center(point) - radius;
		}
	};
}