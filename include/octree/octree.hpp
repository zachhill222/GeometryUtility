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

namespace gutil
{
	template<typename DATA_T, uint64_t DIMENSION=3, bool SINGLE_DATA_=true, uint16_t MAX_DATA_=64>
	class Octree
	{
	public:
		//define common aliases
		using value_type 	= DATA_T;
		using key_type   	= OctreeDigitKey<DIMENSION>;
		using node_type  	= OctreeNode<DATA_T,SINGLE_DATA_, MAX_DATA_, true, key_type>; //assume homogeneous data for now
		using scalar_type	= double;
		using box_type   	= typename key_type::box_type<scalar_type>;
		using point_type 	= Point<key_type::DIMENSION,scalar_type>;

		//record constants
		static constexpr uint16_t MAX_DATA    = MAX_DATA_;
		static constexpr bool     SINGLE_DATA = SINGLE_DATA_;
		static constexpr uint64_t DIM 		  = DIMENSION;

		//constructors
		Octree() : root_box{point_type(0.0), point_type(1.0)} {}
		Octree(const box_type& bbox) : root_box{bbox} {}
		Octree(const point_type& low, const point_type& high) : root_box{low, high} {}

		//expand the root bounding box
		void expand(const box_type& target_box) {
			const auto target_center = target_box.center();

			//get the key from a box that contains the target to the (current) root node
			std::vector<uint16_t> octants;
			while (!root_box.contains(target_box)) {
				assert(prefix.depth() < key_type::MAX_DEPTH);

				//for each axis, decide which octant we are in
				const auto& old_low  = root_box.low();
				const auto& old_high = root_box.high();
				const auto  side     = old_high - old_low;

				uint16_t octant   = 0; //also the child number
				
				point_type new_low, new_high;
				for (uint16_t ax=0; ax<DIM; ++ax) {
					if (target_center[ax] >= center[ax]) {
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

		//return codes
		enum class InsertResult : int
		{
			OK 			=  1,
			DUPLICATE	=  0,
			OVERFLOW 	= -1
		};

		//single insert
		template<typename VALID_T>
		InsertResult insert(DATA_T&& val, VALID_T&& is_valid);

		template<typename VALID_T>
		InsertResult insert(const DATA_T& val, VALID_T&& is_valid) {return insert(std::move(DATA_T{val}), std::forward<VALID_T>(is_valid));}

		//define common utility operations
		void sort_and_merge();



	private:
		//main octree storage
		box_type root_box;
		std::vector<node_type> leafs{node_type{}}; //just the root with no data

		//find the ancestor leaf corresponding to a given key
		//the leafs must be sorted already. note ancestors have smaller keys.
		auto find_ancestor(const key_type key) {
			assert(key.is_valid());
			auto it = std::lower_bound(leafs.begin(), leafs.end(), node_type{key});
			if (it != leafs.end() && it->key == key) {return it;} //a leaf with the given key exists
			
		}
	};




}