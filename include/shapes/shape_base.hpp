#pragma once

#include "math/math.hpp"

#include "geometry/point.hpp"
#include "geometry/box.hpp"

namespace gutil {
	template<int DIM, IsScalar T, typename Derived> requires(DIM>0)
	struct BaseShape {
		using point_type = Point<DIM,T>;
		using box_type = Box<DIM,T>;
		static constexpr int DIMENSION = DIM;

		constexpr BaseShape() noexcept {}
		constexpr BaseShape(const point_type& cntr) noexcept : center{cntr} {}

		//every shape has a center
		point_type center = point_type::Zeros();

		[[nodiscard]] constexpr T dist2center_squared(const point_type& point) const noexcept {
			return gutil::squared_norm(point-center);
		}

		[[nodiscard]] constexpr T dist2center(const point_type& point) const noexcept {
			return gutil::norm2(point-center);
		}

		[[nodiscard]] constexpr T distance_sq(const point_type& point) const noexcept {
			return static_cast<const Derived*>(this) -> distance_sq_impl(point);
		}

		[[nodiscard]] constexpr T distance(const point_type& point) const noexcept {
			return static_cast<const Derived*>(this) -> distance_impl(point);
		}

		[[nodiscard]] constexpr T signed_distance(const point_type& point) const noexcept {
			return static_cast<const Derived*>(this) -> signed_distance_impl(point);
		}

		constexpr void translate_to(const point_type& point) noexcept {
			center = point;
		}

		constexpr void translate_by(const point_type& point) noexcept {
			center+=point;
		}

		//every shape must implement this interface
		[[nodiscard]] constexpr point_type local2global(const point_type& point) const noexcept {
			return static_cast<const Derived*>(this) -> local2global_impl(point);
		}

		[[nodiscard]] constexpr point_type global2local(const point_type& point) const noexcept {
			return static_cast<const Derived*>(this) -> global2local_impl(point);
		}

		[[nodiscard]] constexpr point_type support(const point_type& direction) const noexcept {
			return static_cast<const Derived*>(this) -> support_impl(direction);
		}

		[[nodiscard]] constexpr box_type bbox() const noexcept {
			return static_cast<const Derived*>(this) -> bbox_impl();
		}

		template<typename Shape>
		[[nodiscard]] constexpr bool intersects(const Shape& shape) const noexcept {
			return gutil::collides<Shape,Derived>(shape, *static_cast<const Derived*>(this));
		}
	};


	template<int DIM, IsScalar T, typename Derived> requires (DIM>0)
	[[nodiscard]] inline constexpr T distance_squared(const BaseShape<DIM,T,Derived>& A, const Point<DIM,T>& B) {
		return A.distance_sq(B);
	}



	template<IsScalar T, typename Derived> //quaternions only work for DIM=3
	struct BaseRotatableShape : public BaseShape<3, T, BaseRotatableShape<T,Derived>>
	{
		using BASE = BaseShape<3, T,BaseRotatableShape<T,Derived>>;
		
		using point_type = typename BASE::point_type;
		using box_type = typename BASE::box_type;
		using quaternion_type = Quaternion<T>;

		constexpr BaseRotatableShape() noexcept : BASE() {}
		constexpr BaseRotatableShape(const point_type& cntr, const quaternion_type& quat) noexcept : 
				BASE(cntr), quaternion(quat) {assert(quaternion.is_rotation());}

		//rotations are tracked through the quaternion
		using BASE::center;
		quaternion_type quaternion = quaternion_type::identity();

		constexpr void set_local_to_global_rotation(const quaternion_type& quat) noexcept {
			assert(quat.is_rotation());
			quaternion = quat;
		}

		constexpr void set_local_to_global_rotation(const T theta, const point_type& axis) noexcept {
			quaternion.set_rotation(theta, axis);
			assert(quaternion.is_rotation());
		}

		constexpr void set_global_to_local_rotation(const quaternion_type& quat) noexcept {
			assert(quat.is_rotation());
			quaternion = quat.conj();
		}
		
		constexpr void set_global_to_local_rotation(const T theta, const point_type& axis) noexcept {
			quaternion.set_rotation(theta, -axis); //same as quaternion.set_rotation(theta,axis); quaternion = quaternion.conj();
			assert(quaternion.is_rotation());
		}

		constexpr void rotate_by_local(const quaternion_type& rot) noexcept {
			assert(rot.is_rotation());
			quaternion *= rot;
		}

		constexpr void rotate_by_local(const T theta, const point_type& axis) noexcept {
			quaternion_type rot = quaternion_type::angle_axis(theta, axis);
			assert(rot.is_rotation());
			quaternion *= rot;
		}

		//all rotatable shapes have the same global/local conversions
		[[nodiscard]] constexpr point_type local2global_impl(const point_type& point) const {
			//shift to rotate and shift
			return quaternion.rotate(point) + center;
		}

		[[nodiscard]] constexpr point_type global2local_impl(const point_type& point) const {
			//shift and rotate backwards
			return quaternion.conj().rotate(point - center);
		}

		//pass other implementations to the actual shape
		[[nodiscard]] constexpr box_type bbox_impl() const noexcept {
			return static_cast<const Derived*>(this) -> bbox_impl();
		}

		[[nodiscard]] constexpr point_type support_impl(const point_type& direction) const noexcept {
			return static_cast<const Derived*>(this) -> support_impl(direction);
		}

		[[nodiscard]] constexpr T distance_sq_impl(const point_type& point) const noexcept {
			return static_cast<const Derived*>(this) -> distance_sq_impl(point);
		}

		[[nodiscard]] constexpr T distance_impl(const point_type& point) const noexcept {
			return static_cast<const Derived*>(this) -> distance_impl(point);
		}
		
		[[nodiscard]] constexpr T signed_distance_impl(const point_type& point) const noexcept {
			return static_cast<const Derived*>(this) -> signed_distance_impl(point);
		}
	};

	template<int DIM, IsScalar T, typename Derived> requires (DIM>0)
	struct BaseNonRotatableShape : public BaseShape<DIM, T, BaseNonRotatableShape<DIM, T,Derived>>
	{
		using BASE = BaseShape<DIM, T,BaseNonRotatableShape<DIM, T,Derived>>;
		
		using point_type = typename BASE::point_type;
		using box_type = typename BASE::box_type;

		using BASE::center;
		using BASE::BASE;
		
		//all non-rotatable shapes have the same global/local conversions
		[[nodiscard]] constexpr point_type local2global_impl(const point_type& point) const {
			return point + center;
		}

		[[nodiscard]] constexpr point_type global2local_impl(const point_type& point) const {
			return point - center;
		}

		//pass other implementations to the actual shape
		[[nodiscard]] constexpr box_type bbox_impl() const noexcept {
			return static_cast<const Derived*>(this) -> bbox_impl();
		}

		[[nodiscard]] constexpr point_type support_impl(const point_type& direction) const noexcept {
			return static_cast<const Derived*>(this) -> support_impl(direction);
		}

		[[nodiscard]] constexpr T distance_sq_impl(const point_type& point) const noexcept {
			return static_cast<const Derived*>(this) -> distance_sq_impl(point);
		}

		[[nodiscard]] constexpr T distance_impl(const point_type& point) const noexcept {
			return static_cast<const Derived*>(this) -> distance_impl(point);
		}

		[[nodiscard]] constexpr T signed_distance_impl(const point_type& point) const noexcept {
			return static_cast<const Derived*>(this) -> signed_distance_impl(point);
		}
	};

}