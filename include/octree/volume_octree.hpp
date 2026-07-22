#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"

#include "algorithms/convex_collision.hpp"

#include "octree/base_node.hpp"
#include "octree/base_octree.hpp"

#include <memory_resource>
#include <concept>

namespace gutil {
	


	template<typename VolumeType>
	struct VolumeOctreeOpts {
		using node_allocator_type = std::pmr::polymorphic_allocator<std::byte>;
		using node_resource_type = std::pmr::synchronized_pool_resource;
		// using node_allocator_type = void;
		// using node_resource_type = void;

		using index_allocator_type = std::pmr::polymorphic_allocator<std::byte>;
		using index_resource_type = std::pmr::synchronized_pool_resource;
		// using index_allocator_type = void;
		// using index_resource_type = void;

		using point_type  = typename VolumeType::point_type;
		using scalar_type = typename point_type::scalar_type;
		using value_type  = VolumeType;
		using node_value_type = size_t;
		using box_type    = Box<DIMENSION,scalar_type>;
		using node_type   = Node<node_value_type, DIMENSION, scalar_type, 64, node_allocator_type, index_allocator_type>;

		static constexpr bool HAS_DISTANCE_SQ = true;
		static constexpr bool VOLUME_DATA = true;
		static constexpr int DIMENSION = point_type::DIMENSION;
	};


	template<typename VolumeType>
	struct VolumeOctree : public BaseOctree<VolumeOctree<VolumeType>, VolumeOctreeOpts<VolumeType>> {


		////////////////////////////////////////////////////////////////
		// Constants and aliases
		////////////////////////////////////////////////////////////////
		using OPTS = VolumeOctreeOpts<VolumeType>;
		using BASE = BaseOctree<VolumeOctree<VolumeType>, PointOctreeOpts<VolumeType>>;
		
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
		bool intersects_impl(const box_type& box, const value_type& value) const requires(std::same_as<value_type,box_type>) {
			return box.intersects(value);
		}

		bool intersects_impl(const box_type& box, const value_type& value) const requires(!std::same_as<value_type,box_type>) {
			if constexpr ( requires { value.intersects(box); }) {
				return value.intersects(box);
			}
			else {
				return gutil::collides_GJK<box_type,value_type,OPTS::DIMENSION, typename OPTS::scalar_type>(box,value);
			}
		}

		scalar_type distance_sq_impl(const point_type& point, const value_type& value) const {
			return value.distance_sq(point);
		}

		point_type get_point_impl(const value_type& value) const {
			return value.center;
		}
	};
}