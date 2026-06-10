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
	template<typename NODE_T>
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
		static constexpr key_type INVALID_KEY = key_type{0};

		//define common aliases for spatial logic
		using scalar_type	= double;
		using box_type   	= typename key_type::box_type<scalar_type>;
		using point_type 	= typename key_type::point_type<scalar_type>;

		//constructors with just the root node
		OctreeBase() : root_box{point_type(0.0), point_type(1.0)} {}
		OctreeBase(const box_type& bbox) : root_box{bbox} {}
		OctreeBase(const point_type& low, const point_type& high) : root_box{low, high} {}

		//return codes
		enum class InsertResult : int
		{
			INSERTED	=  1,
			DUPLICATE	=  0,
			OVERFLOW 	= -1
		};

		//allow public access to nodes
		using node_iterator = std::vector<node_type>::iterator;
		using node_const_iterator = std::vector<node_type>::const_iterator;
		node_iterator node_begin() {return leafs.begin();}
		node_const_iterator node_begin() const {return leafs.cbegin();}
		node_const_iterator node_cbegin() const {return leafs.cbegin();}

		node_iterator node_end() {return leafs.end();}
		node_const_iterator node_end() const {return leafs.cend();}
		node_const_iterator node_cend() const {return leafs.cend();}

		inline node_iterator find_node(const point_type& point) {return find_by_point(point);}
		inline node_iterator find_node(const key_type key) {return find_by_key(key);}



	protected:
		/////////////////////////////////////////////////////////////////////////////
		/// Storage

		//extents of spatial data (axis aligned bounding box)
		box_type root_box;
		
		//store the leaf nodes with data
		std::vector<node_type> leafs{node_type{}};	//this must be sorted for efficent operations
		bool is_sorted = true;						//track if the vector needs to be sorted
		bool needs_erase = false;					//track if there are nodes that need to be erased (i.e., after splitting)
		inline bool is_valid() const {return is_sorted && !needs_erase;}

		//allow lookup into node vectors via their keys (i.e., without creating a temporary node)
		struct KeyCompare {
			using is_transparent = void;
			inline constexpr bool operator()(const node_type& a, const node_type& b) const {return a.key < b.key;}
			inline constexpr bool operator()(const node_type& a, const key_type&  b) const {return a.key < b;}
			inline constexpr bool operator()(const key_type&  a, const node_type& b) const {return a     < b.key;}
		};

		//delete any marked nodes (key set to 0, which is invalid)
		void delete_marked() {
			if (needs_erase) {
				leafs.erase_if([](node_type& node) {return node.key == INVALID_KEY;})
			}
		}
		/////////////////////////////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////////////
		/// Node Querries

		//given a point, find the (first) leaf node that contains it
		node_iterator find_by_point(const point_type& point) {
			assert(is_valid());

			//initialize the key and box of the current node
			key_type key{}; //root key
			point_type low  = root_box.low();
			point_type high = root_box.high();

			assert(low <= point && point <= high);

			//walk down the tree until we get to a valid leaf that exists
			//note deeper nodes have large keys
			node_iterator it = leafs.begin();
			for (uint16_t dd=0; dd<=MAX_DEPTH; ++dd) {
				//check if the leaf exists
				it = std::lower_bound(it, leafs.end(), key, KeyCompare{});
				if (it!=list.end() && it->key==key) {return it;}

				//descend to child (updates the key, low, and high)
				key.descend(low, high, point);
			}
		}

		//given a key, find the leaf (if it exists)
		node_iterator find_by_key(const key_type key) {
			assert(is_valid());
			node_iterator it = std::lower_bound(leafs.begin(), leafs.end(), key, KeyCompare{});
			if (it != leafs.end() && it->key == key) {return it;}
			return leafs.end();
		}

		//find the ancestor (or itself) of a node (if it exists)
		node_iterator find_ancestor_by_key(key_type key) {
			assert(is_valid());
			node_iterator it = std::lower_bound(leafs.begin(), leafs.end(), key, KeyCompare{});

			if (it != leafs.end() && it->key == key) {
				//there exists a leaf with this key
				return it;
			}
			else if (it != leafs.end()) {
				//there are finer leafs in the tree, so no ancestor exists
				return leafs.end();
			}
			else {
				//there is a coarser leaf in the tree, so we can find an ancestor
				while (key.is_valid()) { //getting the parent of the root returns an invalid key
					key = key.parent();
					it = std::lower_bound(leafs.begin(), it, key, KeyCompare{});
					if (it != leafs.end() && it->key == key) {return it;}
				}

				//we should have found an ancestor
				assert(false && "partition invariant violated");
				return leafs.end();
			}
		}

		//split an existing node. the existing node will be replaced with the first child.
		//the remaining children will be inserted to the end of the vector.
		//the tree will be in an invalid state after this
		template<typename IS_VALID>
		void split(node_iterator& it, IS_VALID&& is_valid) {
			auto children = it->split(std::forward<IS_VALID>(is_valid));
			*it = std::move(children[0]);
			leafs.insert(leafs.end(), 
				std::make_move_iterator(children.begin()+1),
				std::make_move_iterator(children.end()));
		}

		//split an existing node. the existing node will be marked for deletion.
		//the children will be inserted to the end of the provided vector.
		//the tree will be in an invalid state after this
		template<typename IS_VALID>
		void split(node_iterator& it, IS_VALID&& is_valid, std::vector<node_type>& list) {
			auto children = it->split(std::forward<IS_VALID>(is_valid));
			it->key = INVALID_KEY;
			list.insert(leafs.end(), 
				std::make_move_iterator(children.begin()),
				std::make_move_iterator(children.end()));
		}

		//merge the nodes from start to end. if any nodes need to be split, the children nodes are added
		//to the end of the node storage vector. all data will be compressed to the nodes nearest start.
		//all empty nodes are marked for deletion.
		template<typename IS_VALID>
		bool merge(node_iterator& start, node_iterator& end, IS_VALID&& is_valid) {
			#ifndef NDEBUG
				for (auto it=start; it!=end; ++it) {assert(it->key == start->key);}
			#endif

			//make a single pass
		}

		/////////////////////////////////////////////////////////////////////////////

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

			//append the prefix to all node keys
			for (auto& node : leafs) {node.prepend(prefix);}
		}
		/////////////////////////////////////////////////////////////////////////////
	};




}