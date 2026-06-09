#pragma once

#include "octree/digit_key.hpp"

#include <array>
#include <cassert>
#include <concepts>

namespace gutil
{
	//define core components
	template<typename KEY_T=OctreeDigitKey<3>>
	struct OctreeNodeBase
	{
		//constants
		static constexpr uint64_t MAX_DEPTH  = KEY_T::MAX_DEPTH;
		static constexpr uint16_t DIM        = KEY_T::DIM;
		static constexpr uint16_t N_CHILDREN = KEY_T::N_CHILDREN;

		//define common aliases
		using key_type 	 = KEY_T;

		template<typename T=double>
		using box_type = typename key_type::box_type<T>;

		//store the key
		KEY_T key{};
		
		//pass comparisons to the key
		bool constexpr operator==(const OctreeNodeBase other) const {return key == other.key;}
		bool constexpr operator!=(const OctreeNodeBase other) const {return key != other.key;}
		bool constexpr operator<=(const OctreeNodeBase other) const {return key <= other.key;}
		bool constexpr operator>=(const OctreeNodeBase other) const {return key >= other.key;}
		bool constexpr operator<( const OctreeNodeBase other) const {return key <  other.key;}
		bool constexpr operator>( const OctreeNodeBase other) const {return key >  other.key;}

		//simple constructor
		OctreeNodeBase() : key{} {} //root
		OctreeNodeBase(KEY_T key) : key{key} {}
		
		//get the node extents given the root extents
		template<typename T>
		inline constexpr box_type<T> bbox(const box_type<T>& root_bbox) const noexcept {return key.bbox(root_bbox);}

		//change the key to point at a new root
		void prepend(key_type root) {key.prepend(root);}
	};

	//use template specialization for homogeneous vs inhomogeneous data
	template<typename DATA_T, bool SINGLE_DATA=true, uint16_t MAX_DATA=64, bool IS_HOMOGENEOUS=true, typename KEY_T=OctreeDigitKey<3>>
	struct OctreeNode;


	template<typename DATA_T, bool SINGLE_DATA_, uint16_t MAX_DATA_, typename KEY_T>
	struct OctreeNode<DATA_T, SINGLE_DATA_, MAX_DATA_, true, KEY_T> : public OctreeNodeBase<KEY_T>
	{
		//sanity checks
		static_assert(std::equality_comparable<DATA_T>, "OctreeNode - DATA_T must be equality comparable");

		//convenient aliases
		using base_type  = OctreeNodeBase<KEY_T>;
		using value_type = DATA_T;
		using key_type   = KEY_T;

		//constructor
		

		//convenient constants
		static constexpr bool SINGLE_DATA    = SINGLE_DATA_;
		static constexpr uint16_t MAX_DATA   = MAX_DATA_;
		static constexpr bool IS_HOMOGENEOUS = true;
		
		//bring items from the base to this class
		using base_type::base_type;		//constructors
		using base_type::key; 			//key type
		using base_type::N_CHILDREN;	//number of child nodes (constexpr uint64_t)


		//add homogeneous storage
		std::array<DATA_T,MAX_DATA> values{};
		uint16_t cursor{0}; //points to next empty storage, also the number of stored values
		
		//check if data is already included
		bool contains(const DATA_T& val) const {
			for (uint16_t i=0; i<cursor; ++i) {if (values[i]==val) {return true;}}
			return false;
		}

		int insert(DATA_T&& val) {
			if (contains(val)) {return 0;}		//data was already there
			if (cursor>=MAX_DATA) {return -1;}	//data could not be added
			values[cursor] = std::move(val);
			++cursor;
			return 1;							//data was successfully added
		}

		int insert(const DATA_T& val) {return insert(std::move(DATA_T{val}));}

		//merge two nodes when the data is known to be unique
		int merge_unique(OctreeNode&& other) {
			if (cursor + other.cursor > MAX_DATA) {return -1;} //cannot merge
			std::move(other.values.begin(), other.values.begin()+other.cursor, values.begin()+cursor);
			cursor += other.cursor;
			//other.cursor=0; //Uncomment if the other node needs to be in a valid state
			return 1;
		}

		//merge two nodes when the data could be duplicated
		int merge(OctreeNode&& other) {
			while (other.cursor>0 && cursor<MAX_DATA) {
				--other.cursor; //point to newest data
				const int flag = insert(std::move(other.values[other.cursor]));
				assert(flag!=-1); //should never happen with the while loop limits
			}

			if (other.cursor == 0) {return 1;}		//all data was successfully moved
			return -1; 								//merge was unsuccessful, overflow
		}

		//take the data from this node and push it to its children
		template<typename VALID_T>
		std::array<OctreeNode,N_CHILDREN> split(VALID_T&& is_valid) {
			//initialize children
			std::array<OctreeNode,N_CHILDREN> children;
			for (uint64_t c=0; c<N_CHILDREN; ++c) {children[c] = OctreeNode{key.child(c)};}

			//move or copy data to children
			for (uint16_t i=0; i<cursor; ++i) {
				#ifndef NDEBUG
				int n_children = 0;
				#endif
				for (int c=0; c<N_CHILDREN; ++c) {
					if (is_valid(children[c].key, values[i])) {
						#ifndef NDEBUG
						++n_children;
						#endif

						if constexpr (SINGLE_DATA) {
							children[c].insert(std::move(values[i]));
							break;
						}
						else {children[c].insert(values[i]);}
					}
				}

				#ifndef NDEBUG
				assert(n_children>0);
				#endif
			}

			//mark that the data has been deleted from this node
			cursor = 0;

			//return the children with their data
			return children;
		}
	};
}