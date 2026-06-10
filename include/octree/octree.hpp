#pragma once

#include "octree/digit_key.hpp"
#include "octree/node.hpp"

#include "geometry/point.hpp"
#include "geometry/box.hpp"

#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gutil
{
	template<typename NODE_T, uint16_t MAX_DATA_=64, typename DERIVED>
	class OctreeBase
	{
	public:
		//define common aliases for tree logic
		using node_type  	= NODE_T;
		using key_type   	= typename node_type::key_type;
		
		//record constants
		static constexpr uint16_t MAX_DATA    = node_type::MAX_DATA;
		static constexpr uint64_t DIM 		  = key_type::DIM;
		static constexpr uint64_t MAX_DEPTH   = key_type::MAX_DEPTH;
		
		//define common aliases for spatial logic
		using scalar_type	= double;
		using box_type   	= typename key_type::box_type<scalar_type>;
		using point_type 	= typename key_type::point_type<scalar_type>;

		//constructors with just the root node
		OctreeBase() : root_box{point_type(0.0), point_type(1.0)} {leafs[0].push_back(node_type{});}
		OctreeBase(const box_type& bbox) : root_box{bbox} {leafs[0].push_back(node_type{});}
		OctreeBase(const point_type& low, const point_type& high) : root_box{low, high} {leafs[0].push_back(node_type{});}

		//return codes
		enum class InsertResult : int
		{
			OK 			=  1,
			DUPLICATE	=  0,
			OVERFLOW 	= -1
		};
	protected:
		/////////////////////////////////////////////////////////////////////////////
		/// Storage

		//extents of spatial data (axis aligned bounding box)
		box_type root_box;
		
		//store the leaf nodes with data
		std::array<std::vector<node_type>,MAX_DEPTH> leafs;
		
		/////////////////////////////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////////////
		/// Node Querries
		
		//allow lookup into node vectors via their keys (i.e., without creating a temporary node)
		struct KeyCompare {
			using is_transparent = void;
			inline constexpr bool operator()(const node_type& a, const node_type& b) const {return a.key < b.key;}
			inline constexpr bool operator()(const node_type& a, const key_type&  b) const {return a.key < b;}
			inline constexpr bool operator()(const key_type&  a, const node_type& b) const {return a     < b.key;}
		};


		//given a point, find the (first) leaf node that contains it
		node_type& find_leaf(const point_type& point) {
			//initialize the key and box of the current node
			key_type key{}; //root key
			point_type low  = root_box.low();
			point_type high = root_box.high();

			assert(low <= point && point <= high);

			//walk down the tree until we get to a valid leaf that exists
			for (uint16_t dd=0; dd<=MAX_DEPTH; ++dd) {
				//check if the leaf exists
				auto& list = leafs[dd];
				auto it = std::lower_bound(list.begin(), list.end(), key, KeyCompare{});
				if (it!=list.end() && it->key==key) {return *it;}

				//descend to child (updates the key, low, and high)
				key.descend(low, high, point);
			}
		}

		/////////////////////////////////////////////////////////////////////////////
		/// Spatial Logic

		//expand the bounding box to contain the target without changing the spatial extent of any leaf node
		void expand(const box_type& target_box) {
			const auto target_center = target_box.center();

			//get the key from a box that contains the target to the (current) root node
			std::vector<uint16_t> octants;
			while (!root_box.contains(target_box)) {
				assert(octants.size() < key_type::MAX_DEPTH);

				//for each axis, decide which octant we are in
				const auto& old_low  = root_box.low();
				const auto& old_high = root_box.high();
				const auto  side     = old_high - old_low;
				const auto  old_cntr = old_low + 0.5*side;

				uint16_t octant = 0; //also the child number
				
				point_type new_low, new_high;
				for (uint16_t ax=0; ax<DIM; ++ax) {
					if (target_center[ax] >= old_cntr[ax]) {
						new_low[ax]  = old_low[ax];
						new_high[ax] = old_high[ax] + side[ax];
						//octant axis bit is low/zero
					}
					else {
						new_low[ax]  = old_low[ax] - side[ax];
						new_high[ax] = old_high[ax];
						//octant axis bit is high/one
						octant |= (uint16_t{1} << ax);
					}
				}

				octants.push_back(octant);
			}

			//octants were found in reverse order (last was the new root)
			//create the key of the old root relative to the new root
			key_type prefix{}; //root
			for (auto it=octants.rbegin(); it!=octants.rend(); ++it) {prefix = prefix.child(*it);}

			//ensure that there is enough room to 

			//append the prefix to all node keys
			for (auto& node : leafs) {node.prepend(prefix);}
		}
		/////////////////////////////////////////////////////////////////////////////




		/////////////////////////////////////////////////////////////////////////////
		/// Tree Logic

		//find the ancestor leaf corresponding to a given key
		//the leafs must be sorted already. note ancestors have smaller keys.
		auto find_ancestor(const key_type key) {
			assert(key.is_valid());
			auto it = std::lower_bound(leafs.begin(), leafs.end(), node_type{key});
			if (it != leafs.end() && it->key == key) {return it;} //a leaf with the given key exists
		}

		//single insert
		template<typename VALID_T>
		InsertResult insert(DATA_T&& val, VALID_T&& is_valid);

		template<typename VALID_T>
		InsertResult insert(const DATA_T& val, VALID_T&& is_valid) {return insert(std::move(DATA_T{val}), std::forward<VALID_T>(is_valid));}

		//define common utility operations
		void sort_and_merge();


		
	};




}