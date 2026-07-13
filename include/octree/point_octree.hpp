#pragma once

#include "geometry/point.hpp"
#include "octree/node.hpp"
#include "octree/octree_base.hpp"

namespace gutil {
	template<size_t Dimension, typename ScalarT, size_t MaxData>
	using PointOctreeOpts = NodeOpts<Point<static_cast<int>(Dimension),ScalarT>, false, MaxData, Dimension, ScalarT>;

	template<size_t Dimension, typename ScalarT, size_t MaxData=64>
	struct PointOctree : public OctreeBase<PointOctreeOpts<Dimension,ScalarT,MaxData>, PointOctree<Dimension,ScalarT,MaxData>> {
		static constexpr bool HAS_DISTANCE = true;

		using BASE = OctreeBase<PointOctreeOpts<Dimension,ScalarT,MaxData>, PointOctree<Dimension,ScalarT,MaxData>>;

		using scalar_type = typename BASE::scalar_type;
		using point_type  = typename BASE::point_type;
		using value_type  = typename BASE::value_type;
		using box_type    = typename BASE::box_type;
		static_assert(std::same_as<value_type, point_type>);

		using BASE::BASE;
		using BASE::operator=;

		bool intersects_impl(const box_type& box, const value_type& value) const {
			return box.contains(value);
		}

		scalar_type distance_squared_impl(const point_type& point, const value_type& value) const {
			return gutil::squared_norm(point-value);
		}
	};
}