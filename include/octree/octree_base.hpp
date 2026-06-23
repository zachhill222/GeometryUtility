#pragma once

#include "memory/hetero_slab_allocator.hpp"
#include "octree/node.hpp"

namespace gutil
{
	//concept to determine if it can be determined if data intersects a box
	template<typename Function, typename Box, typename Value>
	concept ValidCheck = requires(Function f, Box b, Value v) {
		{ f(b,v) } -> std::same_as<bool>;
	};

	//concept to determine if the distance from a point to data can be found
	template<typename Function, typename Point, typename Value>
	concept DistanceCheck = requires(Function f, Point p, Value v) {
		{ f(p,v) } -> std::convertible_to<double>;
	};

	//base type for octrees
	template<NodeOpts Opts, typename IsValid, typename DistFun = std::nullptr_t>
	struct OctreeBase
	{
		using interior_node = InteriorNode<Opts>;
		using leaf_node 	= LeafNode<Opts>;

		using value_type    = typename Opts::value_type;	//type that this octree is storing
		using store_type  	= typename Opts::store_type;	//type that is stored in the leaf nodes
		using point_type  	= typename Opts::point_type;	//type of spatial points
		using box_type	  	= typename Opts::box_type;		//type of spatial axis-aligned-bounding-boxes
		using scalar_type 	= typename Opts::scalar_type;	//type that emulates real numbers for the spatial points and aabb

		static constexpr bool STORE_IN_LEAF = Opts::STORE_IN_LEAF;

		static_assert(ValidCheck<IsValid, box_type, value_type>, "OctreeBase: can't check data intersections with boxes");
		static_assert( (!STORE_IN_LEAF) || std::same_as<value_type,store_type>);
	protected:
		//store spatial queries
		static constexpr bool HAS_DISTANCE = DistanceCheck<DistFun, box_type, value_type>;
		[[no_unique_address]] IsValid intersects;	//intersects(box,value) -> bool
		[[no_unique_address]] DistFun distance;		//distance(point,value) -> scalar

		box_type bbox;	//spatial extents that this octree tracks
		interior_node* split(leaf_node* leaf);
		
	private:
		HeterSlabAllocator<interior_node,leaf_node> _nodes_;
	};

}