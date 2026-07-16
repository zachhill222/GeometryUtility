#pragma once

#include "geometry/point.hpp"
#include "octree/node.hpp"
#include "octree/octree_base.hpp"

#include <span>

namespace gutil {
	template<IsPoint PointType, size_t MaxData>
	using PointOctreeOpts = NodeOpts<PointType, false, MaxData, PointType::DIMENSION, typename PointType::scalar_type>;

	template<IsPoint PointType, size_t MaxData=64>
	struct PointOctree : public OctreeBase<PointOctreeOpts<PointType,MaxData>, PointOctree<PointType,MaxData>> {
		static constexpr bool HAS_DISTANCE = true;

		using BASE = OctreeBase<PointOctreeOpts<PointType,MaxData>, PointOctree<PointType,MaxData>>;

		using scalar_type = typename BASE::scalar_type;
		using point_type  = typename BASE::point_type;
		using value_type  = typename BASE::value_type;
		using box_type    = typename BASE::box_type;
		static_assert(std::same_as<value_type, point_type>);

		using BASE::BASE;
		using BASE::operator=;

		PointOctree(std::span<value_type> points) : BASE(box_type{points}) {
			BASE::batch_insert(points);
		}

		PointOctree(std::span<const value_type> points) : BASE(box_type{points}) {
			BASE::batch_insert(points);
		}

		bool intersects_impl(const box_type& box, const value_type& value) const {
			return box.contains(value);
		}

		scalar_type distance_squared_impl(const point_type& point, const value_type& value) const {
			return gutil::squared_norm(point-value);
		}

		point_type get_point_imp(const value_type& value) const {
			return value;;
		}
	};
}