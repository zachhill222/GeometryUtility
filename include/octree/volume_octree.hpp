#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"

#include "algorithms/convex_collision.hpp"

#include "octree/base_node.hpp"
#include "octree/base_octree.hpp"

#include "shapes/sphere.hpp"

#include <memory_resource>
#include <concepts>

namespace gutil {
	


	template<typename VolumeType>
	struct VolumeOctreeOpts {
		using point_type  = typename VolumeType::point_type;
		using scalar_type = typename point_type::scalar_type;
		using value_type  = VolumeType;
		using node_value_type = size_t;

		static constexpr bool HAS_DISTANCE_SQ = true;
		static constexpr bool VOLUME_DATA = true;
		static constexpr int DIMENSION = point_type::DIMENSION;

		// using node_allocator_type = std::pmr::polymorphic_allocator<std::byte>;
		// using node_resource_type = std::pmr::synchronized_pool_resource;
		using node_allocator_type = void;
		using node_resource_type = void;

		// using index_allocator_type = std::pmr::polymorphic_allocator<std::byte>;
		// using index_resource_type = std::pmr::synchronized_pool_resource;
		using index_allocator_type = void;
		using index_resource_type = void;

		using box_type    = Box<DIMENSION,scalar_type>;
		using node_type   = Node<node_value_type, DIMENSION, scalar_type, 64, node_allocator_type, index_allocator_type>;
	};


	template<typename VolumeType>
	struct VolumeOctree : public BaseOctree<VolumeOctree<VolumeType>, VolumeOctreeOpts<VolumeType>> {


		////////////////////////////////////////////////////////////////
		// Constants and aliases
		////////////////////////////////////////////////////////////////
		using OPTS = VolumeOctreeOpts<VolumeType>;
		using BASE = BaseOctree<VolumeOctree<VolumeType>, VolumeOctreeOpts<VolumeType>>;
		
		using value_type  = typename OPTS::value_type;
		using box_type    = typename OPTS::box_type;
		using point_type  = typename OPTS::point_type;
		using scalar_type = typename OPTS::scalar_type;

		////////////////////////////////////////////////////////////////
		// Convenient constructors
		////////////////////////////////////////////////////////////////
		using BASE::BASE;

		VolumeOctree(std::vector<value_type>&& list) : VolumeOctree(std::span<value_type>(list.begin(), list.end())) {}
		VolumeOctree(std::span<value_type> list) : BASE() {
			if (list.empty()) {return;}

			box_type box = list[0].bbox();
			for (size_t i=1; i<list.size(); ++i) {
				box.expand(list[i].bbox());
			}

			this->construct_root(box);
			this->push_back_range(list);
		}


		////////////////////////////////////////////////////////////////
		// Implementation of CRTP interface
		////////////////////////////////////////////////////////////////
		bool intersects_impl(const box_type& box, const value_type& value) const noexcept requires(std::same_as<value_type,box_type>) {
			return box.intersects(value);
		}

		bool intersects_impl(const box_type& box, const value_type& value) const noexcept requires(!std::same_as<value_type,box_type>) {
			return gutil::collides(box,value);
		}

		bool intersects_impl(const value_type& A, const value_type& B) const noexcept requires(!std::same_as<value_type,box_type>) {
			return gutil::collides(A,B);
		}

		scalar_type distance_sq_impl(const value_type& value, const point_type& point) const {
			return value.distance_sq(point);
		}

		point_type get_point_impl(const value_type& value) const {
			return value.center;
		}

		scalar_type signed_distance(const point_type& point) const {
			size_t idx = this->find_nearest(point);
			return this->data_[idx].signed_distance(point);
		}
	};
}