#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"

#include "octree/base_node.hpp"
#include "octree/base_octree.hpp"

namespace gutil {
	
	template<IsPoint PointType>
	struct PointOctreeOpts {
		static constexpr bool HAS_DISTANCE_SQ = true;
		static constexpr bool VOLUME_DATA = false;
		static constexpr int DIMENSION = PointType::DIMENSION;

		using scalar_type = typename PointType::scalar_type;
		using value_type  = PointType;
		using point_type  = PointType;
		using box_type    = Box<DIMENSION,scalar_type>;
		using node_type   = Node<size_t, DIMENSION, scalar_type, 64, void, void>;
	};


	template<IsPoint PointType>
	struct PointOctree : public BaseOctree<PointOctree<PointType>, PointOctreeOpts<PointType>> {


		////////////////////////////////////////////////////////////////
		// Constants and aliases
		////////////////////////////////////////////////////////////////
		using OPTS = PointOctreeOpts<PointType>;
		using BASE = BaseOctree<PointOctree<PointType>, PointOctreeOpts<PointType>>;
		
		using value_type  = typename OPTS::value_type;
		using box_type    = typename OPTS::box_type;
		using point_type  = typename OPTS::point_type;
		using scalar_type = typename OPTS::scalar_type;

		////////////////////////////////////////////////////////////////
		// Convenient constructors
		////////////////////////////////////////////////////////////////
		using BASE::BASE;

		PointOctree(std::span<value_type> points) : BASE(points) {
			BASE::push_back_range(points);
		}


		////////////////////////////////////////////////////////////////
		// Implementation of CRTP interface
		////////////////////////////////////////////////////////////////
		bool intersects_impl(const box_type& box, const value_type& value) const {
			return box.contains(value);
		}

		scalar_type distance_sq_impl(const point_type& point, const value_type& value) const {
			return gutil::squared_norm(point-value);
		}

		point_type get_point_impl(const value_type& value) const {
			return value;;
		}
	};
}