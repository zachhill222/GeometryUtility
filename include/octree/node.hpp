#pragma once

#include "octree/digit_key.hpp"

#include <array>
#include <cassert>
#include <concepts>

namespace gutil
{
	//use template specialization to either store the data directly (single data)
	//or store an index into a vector of data.



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
	
	//TODO: inheritance probably isn't needed, but I'm not sure if I want other node types
	template<typename STORE_T=uint32_t, bool SINGLE_DATA_=true, uint16_t MAX_DATA_=64, typename KEY_T=OctreeDigitKey<3>, bool ALLOW_SORTED_=true>
	struct OctreeNode: public OctreeNodeBase<KEY_T>
	{
		//sanity checks
		static_assert(std::equality_comparable<STORE_T>, "OctreeNode - STORE_T must be equality comparable");
		static_assert(SINGLE_DATA || std::integral<STORE_T>, "OctreeNode - STORE_T must be an index/integer type for multiple data");

		//convenient aliases
		using base_type  = OctreeNodeBase<KEY_T>;
		using store_type = STORE_T;
		using key_type   = KEY_T;

		//convenient constants
		static constexpr bool SINGLE_DATA    = SINGLE_DATA_;
		static constexpr uint16_t MAX_DATA   = MAX_DATA_;
		static constexpr bool IS_HOMOGENEOUS = true;
		static constexpr bool IS_ORDERED     = std::totally_ordered<STORE_T>;
		static constexpr bool KEEP_SORTED    = (IS_ORDERED&&ALLOW_SORTED_) ? (MAX_DATA>64) : false;

		//bring items from the base to this class
		using base_type::base_type;		//constructors
		using base_type::key; 			//key type
		using base_type::N_CHILDREN;	//number of child nodes (constexpr uint64_t)

		//add homogeneous storage
		std::array<STORE_T,MAX_DATA> values{};
		uint16_t cursor{0}; //points to next empty storage, also the number of stored values
		
		//convenient querries
		inline constexpr bool empty() const {return cursor==0;}
		inline constexpr size_t size() const {return static_cast<size_t>(cursor);}
		inline constexpr bool full() const {return cursor>=MAX_DATA;}
		inline constexpr uint16_t capacity_remaining() const {return full() ? 0 : MAX_DATA-cursor;}

		//iterate through data
		auto begin() {return values.begin();}
		auto begin() const {return values.begin();}
		auto end() {return values.begin()+cursor;}
		auto end() const {return values.begin()+cursor;}

		//check if data is already included
		bool contains(const STORE_T& val) const {
			if constexpr (KEEP_SORTED) {
				return std::binary_search(begin(), end(), val);
			}
			else {
				return std::find(begin(), end(), val) != end();
			}
		}

		//insert data and don't maintain sorting or check containment
		int insert_back(STORE_T&& val) {
			if (cursor>=MAX_DATA) {return -1;}					//data could not be added
			values[cursor++] = std::move(val);
			return 1;											//data was added successfully
		}

		//insert data and maintain sorting. checks for containment.
		int insert_sort(STORE_T&& val) {
			if (cursor>=MAX_DATA) {return -1;}					//data could not be added

			auto it = std::lower_bound(begin(), end(), val);
			if (it != end() && *it == val) {return 0;}			//data was already there
			
			std::move_backward(it, end(), end()+1);				//shift the data back to make room for the new data
			*it = std::move(val);
			++cursor;

			return 1;											//data was added successfully
		}

		//dispatch to insert_back or insert_sort as needed
		inline int insert(const STORE_T& val) {return insert(std::move(STORE_T{val}));}
		inline int insert(STORE_T&& val) {
			if constexpr (KEEP_SORTED) {
				return insert_sort(std::move(val));
			}
			else {
				return contains(val) ? 0 : insert_back(std::move(val));
				
			}
		}

		
		//merge two nodes when the data is known to be unique
		int merge_unique(OctreeNode&& other) {
			if (cursor + other.cursor > MAX_DATA) {return -1;} //cannot merge
			std::move(other.begin(), other.end(), end());
			cursor += other.cursor;
			
			if constexpr (KEEP_SORTED) {
				std::sort(begin(), end());
			}

			return 1;
		}

		//merge two nodes when the data could be duplicated
		int merge(OctreeNode&& other) {
			while (other.cursor>0 && cursor<MAX_DATA) {
				--other.cursor; //point to newest data
				const int flag = insert(std::move(other.values[other.cursor]));
				assert(flag!=-1); //should never happen with the while loop limits
			}

			if (!other.empty()) {return -1;}		//merge was unsuccessful

			if constexpr (KEEP_SORTED) {			//sort if needed
				std::sort(begin(), end());
			}

			return 1; 								//merge was successful
		}

		//take the data from this node and push it to its children
		template<typename VALID_T>
		std::array<OctreeNode,N_CHILDREN> split(VALID_T&& is_valid) {
			//initialize children
			std::array<OctreeNode,N_CHILDREN> children;
			for (uint64_t c=0; c<N_CHILDREN; ++c) {children[c] = OctreeNode{key.child(c)};}

			//move or copy data to children (note if this node data is sorted, then the child data is also sorted)
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
							children[c].insert_back(std::move(values[i]));
							break;
						}
						else {children[c].insert_back(values[i]);}
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