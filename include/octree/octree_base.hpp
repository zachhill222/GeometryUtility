#pragma once

#include "geometry/box.hpp"
#include "memory/hetero_slab_allocator.hpp"
#include "octree/node.hpp"

#include <vector>
#include <array>
#include <algorithm>
#include <cassert>
#include <type_traits>

namespace gutil
{
	//base type for octrees
	template<IsNodeOpts Opts, typename Derived>
	struct OctreeBase
	{
		using internal_node = InternalNode<Opts>;
		using leaf_node 	= LeafNode<Opts>;
		using tag_ptr_type  = typename internal_node::tag_ptr_type;

		using value_type    = typename Opts::value_type;	//type that this octree is storing
		using index_type  	= typename Opts::index_type;	//type that is stored in the leaf nodes
		using point_type  	= typename Opts::point_type;	//type of spatial points
		using box_type	  	= typename Opts::box_type;		//type of spatial axis-aligned-bounding-boxes
		using scalar_type 	= typename Opts::scalar_type;	//type that emulates real numbers for the spatial points and aabb

		OctreeBase(const box_type& box) {
			//always immediately split the root node so that it doen't need to be tracked
			//if it is a leaf or internal node type. this keeps the root node from moving in memory as well.
			root = construct_internal(box);
			root -> construct_child_leafs(_alloc_leaf_());
		}

		static_assert(std::is_trivially_destructible_v<internal_node>);
		static_assert(std::is_trivially_destructible_v<leaf_node>);
		~OctreeBase() {} //_alloc_ free everything automatically if the nodes are trivially destructable

		//some public simple queries
		bool contains(const value_type& value) const {
			return leaf_contains(find_leaf(value), value);
		}

		index_type find(const value_type& value) const {
			const leaf_node* leaf = find_leaf(value);
			if (leaf) {
				for (index_type idx : *leaf) {
					assert(idx<data.size());
					if (data[idx] == value) {return idx;}
				}
			}
			return index_type(-1);
		}

		const box_type& bbox() const {
			assert(root!=nullptr);
			return root->bbox;
		}

		size_t size() const {return data.size();}
		void reserve(const size_t n) {data.reserve(n);}

		//some public simple operations
		void clear() {
			data.clear();
			const box_type box = root->bbox;
			_alloc_.release();
			root = construct_internal(box);
			root -> construct_child_leafs(_alloc_leaf_());
		}

		//iterators for better looping through the data
		//note that changing the dat could invalidate the tree
		auto begin() {return data.begin();}
		auto begin() const {return data.cbegin();}
		auto cbegin() const {return data.cbegin();}

		auto end() {return data.end();}
		auto end() const {return data.cend();}
		auto cend() const {return data.cend();}

		auto rbegin() {return data.rbegin();}
		auto rbegin() const {return data.crbegin();}
		auto crbegin() const {return data.crbegin();}

		auto rend() {return data.rend();}
		auto rend() const {return data.crend();}
		auto crend() const {return data.crend();}

		//accessors to the data, note that changing the data
		//could invalidate the tree
		const value_type& operator[](const size_t idx) const {
			assert(idx<data.size());
			return data[idx];
		}

		value_type& operator[](const size_t idx) {
			assert(idx<data.size());
			return data[idx];
		}

		const value_type& at(const size_t idx) const {
			return data.at(idx);
		}

		value_type& at(const size_t idx) {
			return data.at(idx);
		}

		//insert data into the tree and return its index
		index_type insert(value_type&& value);
		
		index_type insert(const value_type& value) {
			return insert(std::move(value_type{value}));
		}

		//find the nearest data to a point
		index_type nearest(const point_type& point) const 
			requires (Derived::HAS_DISTANCE) {
			if (data.empty()) {
				return index_type(-1);
			}
			else {
				index_type index = 0;
				scalar_type dist2 = distance_squared(point, data[index]);
				nearest_recursive(root->t_ptr(), point, index, dist2);
				return index;
			}
		}

	protected:
		//track where the root is
		internal_node* root{nullptr};

		//store the data in a contiguous vector, leaves store indices into this
		std::vector<value_type> data;

		//get the box from an unkonwn node type
		const box_type& get_box(const tag_ptr_type t_ptr) const {
			assert(!t_ptr.is_null());
			return t_ptr.tag() == NodeTag::LEAF ?
				static_cast<const leaf_node*>(t_ptr)->bbox :
				static_cast<const internal_node*>(t_ptr)->bbox;
		}

		//pass intersection query to the derived class
		bool intersects(const box_type& box, const value_type& value) const {
			return static_cast<const Derived*>(this)->intersects_impl(box, value);
		}

		bool intersects(const box_type& box, const index_type index) const {
			assert(index < data.size());
			return intersects(box, data[index]);
		}

		//pass distance query to the derived class
		scalar_type distance_squared(const point_type& point, const value_type& value) const 
			requires (Derived::HAS_DISTANCE) {
			return static_cast<const Derived*>(this)->distance_squared_impl(point, value);
		}

		scalar_type distance_squared(const point_type& point, const index_type index) const 
			requires (Derived::HAS_DISTANCE) {
			assert(index < data.size());
			return distance_squared(point, data[index]);
		}

		//pass checking if a leaf contains a value to the derived class (necessary if the leaf stores indices)
		bool leaf_contains(const leaf_node* leaf, const value_type& value) const {
			if (leaf) {
				for (index_type idx : *leaf) {
					assert(idx < data.size());
					if (data[idx] == value) {return true;}
				}
			}
			return false;
		}

		bool leaf_contains(const leaf_node* leaf, const index_type index) const {
			assert(index < data.size());
			return (leaf!=nullptr) && (leaf->contains(index));
		}

		//split a leaf and return the internal node that it was converted to
		internal_node* split(leaf_node* leaf);

		//find the first leaf that can contain some value
		leaf_node* find_leaf(const value_type& value) const;

		//find the first leaf node that contains the specified point
		leaf_node* find_leaf(const point_type& point) const requires (!std::same_as<value_type,point_type>);

		//find the first internal node that has multiple children that intersect the data
		//or return the first leaf that intersects the data if no such internal node exists
		tag_ptr_type find_branch(const value_type& value) const;

		//get all leafs that are descendants of a node
		void get_all_leaves(std::vector<leaf_node*>& leaves, const tag_ptr_type node) const;

		//recursive portion of insert to split nodes as needed
		//handles point data and volume data
		void insert_recursive(const tag_ptr_type node, const value_type& value, const index_type index);

		//recursive portion of finding the nearest data to a point
		//index and dist are the current best index/data and its distance
		void nearest_recursive(const tag_ptr_type node, const point_type& point, 
				index_type& index, scalar_type& dist2) const requires (Derived::HAS_DISTANCE);
	private:
		//make allocator for less fragmented node storage and some convenience contructors
		HeteroSlabAllocator<internal_node,leaf_node> _alloc_{};
		auto& _alloc_internal_() {return _alloc_.template pool<internal_node>();}
		auto& _alloc_leaf_() {return _alloc_.template pool<leaf_node>();}

		leaf_node* construct_leaf(const box_type& box) 			{return _alloc_.template construct<leaf_node>(box);}
		internal_node* construct_internal(const box_type& box) 	{return _alloc_.template construct<internal_node>(box);}
		void destroy_leaf(leaf_node* p) 						{_alloc_.template destroy<leaf_node>(p);}
		void destroy_internal(internal_node* p) 				{_alloc_.template destroy<internal_node>(p);}
	};


	template<IsNodeOpts O, typename D>
	typename OctreeBase<O,D>::internal_node* OctreeBase<O,D>::split(leaf_node* leaf) {
		//allocate a new internal node that will take the place of the leaf
		internal_node* internal = construct_internal(leaf->bbox);
		internal->parent = leaf->parent;

		const tag_ptr_type tagged_internal = internal->t_ptr();
		const tag_ptr_type tagged_leaf     = leaf->t_ptr();

		//update the leaf's parent to point to the new internal node
		assert(leaf->parent != nullptr);
		for (tag_ptr_type& c_ptr : *(leaf->parent)) {
			if (c_ptr == tagged_leaf) {
				c_ptr = tagged_internal;
			}
		}

		//construct the children (new leaf) nodes
		internal->construct_child_leafs(_alloc_leaf_());

		//push data into the leafs
		for (index_type& idx : *leaf) {
			for (tag_ptr_type t_ptr : *internal) {
				assert(t_ptr.tag() == NodeTag::LEAF);
				leaf_node* lf = static_cast<leaf_node*>(t_ptr);
				if (intersects(lf->bbox, idx)) {
					lf->insert(idx);
					if constexpr (!O::VOLUME_DATA) {break;}
				}
			}
		}

		//free old leaf and return its internal replacement
		destroy_leaf(leaf);
		return internal;
	}

	template<IsNodeOpts O, typename D>
	typename OctreeBase<O,D>::leaf_node* OctreeBase<O,D>::find_leaf(const value_type& value) const {
		if (!intersects(root->bbox, value)) {return nullptr;}

		tag_ptr_type node = root->t_ptr();
		while (node.tag() == NodeTag::INTERNAL) {
			const internal_node* ptr = static_cast<const internal_node*>(node);
			bool descended = false;
			for (tag_ptr_type c_ptr : *ptr) {
				assert(!c_ptr.is_null());
				if (intersects(get_box(c_ptr), value)) {
					node = c_ptr;
					descended = true;
					break;
				}
			}

			assert(descended && "OctreeBase: could not find child node for data");
			if (!descended) {return nullptr;}
		}

		return static_cast<leaf_node*>(node);
	}

	template<IsNodeOpts O, typename D>
	typename OctreeBase<O,D>::leaf_node* OctreeBase<O,D>::find_leaf(const point_type& point) const 
		requires (!std::same_as<value_type,point_type>) {
		if (!root->bbox.contains(point)) {return nullptr;}

		tag_ptr_type node = root->t_ptr();
		while (node.tag() == NodeTag::INTERNAL) {
			const internal_node* ptr = static_cast<const internal_node*>(node);
			bool descended = false;
			for (tag_ptr_type c_ptr : *ptr) {
				assert(!c_ptr.is_null());
				if (get_box(c_ptr).contains(point)) {
					node = c_ptr;
					descended = true;
					break;
				}
			}

			assert(descended && "OctreeBase: could not find child node for data");
			if (!descended) {return nullptr;}
		}

		return static_cast<leaf_node*>(node);
	}

	template<IsNodeOpts O, typename D>
	typename OctreeBase<O,D>::tag_ptr_type OctreeBase<O,D>::find_branch(const value_type& value) const {
		if (!intersects(root->bbox, value)) {return tag_ptr_type::null();}

		tag_ptr_type node = root->t_ptr();
		while (node.tag() == NodeTag::INTERNAL) {
			const internal_node* ptr = static_cast<const internal_node*>(node);
			int child_count = 0;
			tag_ptr_type temp;
			for (tag_ptr_type c_ptr : *ptr) {
				assert(!c_ptr.is_null());
				if (intersects(get_box(c_ptr), value)) {
					++child_count;
					temp = c_ptr;
				}
			}

			if (child_count>1) {return node;}
			else if (child_count==1) {node = temp;}
			else {
				assert(false && "OctreeBase: could not find child node for data");
				return tag_ptr_type::null();
			}
		}

		return node;
	}

	template<IsNodeOpts O, typename D>
	void OctreeBase<O,D>::get_all_leaves(std::vector<leaf_node*>& leaves, const tag_ptr_type node) const {
		if (node.tag() == NodeTag::INTERNAL) {
			for (const tag_ptr_type c_ptr : *static_cast<const internal_node*>(node)) {
				assert(!c_ptr.is_null());
				get_all_leaves(leaves, c_ptr);
			}
		}
		else if (node.tag() == NodeTag::LEAF) {
			leaves.emplace_back(static_cast<leaf_node*>(node));
		}
	}

	template<IsNodeOpts O, typename D>
	void OctreeBase<O,D>::insert_recursive(const tag_ptr_type node, const value_type& value, const index_type index) {
		if (node.tag() == NodeTag::LEAF) {
			leaf_node* leaf = static_cast<leaf_node*>(node);
			if (leaf->contains(index)) {return;}
			else if (leaf->full()) {
				internal_node* internal = split(leaf);
				insert_recursive(internal->t_ptr(), value, index);
				return;
			}
			else {
				leaf->insert(index);
				return;
			}
		}
		else if (node.tag() == NodeTag::INTERNAL) {
			internal_node* internal = static_cast<internal_node*>(node);
			for (tag_ptr_type c_ptr : *internal) {
				assert(!c_ptr.is_null());
				if (intersects(get_box(c_ptr), value)) {
					insert_recursive(c_ptr, value, index);
					if constexpr (!O::VOLUME_DATA) {return;}
				}
			}
		}
	}

	template<IsNodeOpts O, typename D>
	typename OctreeBase<O,D>::index_type OctreeBase<O,D>::insert(value_type&& value) {
		//get the best leaf for the data
		leaf_node* leaf = find_leaf(value);
		if (leaf==nullptr) {return index_type(-1);}

		//check if the leaf contains the data
		for (index_type idx : *leaf) {
			assert(idx < data.size());
			if (data[idx] == value) {
				return idx;
			}
		}

		//the data is new
		index_type idx = data.size();
		data.push_back(std::move(value));

		if constexpr (!O::VOLUME_DATA) {
			insert_recursive(leaf->t_ptr(), data[idx], idx);
		}
		else {
			const tag_ptr_type branch = find_branch(data[idx]);
			insert_recursive(branch, data[idx], idx);
		}

		return idx;
	}

	//find the nearest data to a point
	template<IsNodeOpts O, typename D>
	void OctreeBase<O,D>:: nearest_recursive(const tag_ptr_type node, 
		const point_type& point, index_type& index, scalar_type& dist2) const requires (D::HAS_DISTANCE) {
		if (node.tag() == NodeTag::INTERNAL) {
			const internal_node* internal = static_cast<const internal_node*>(node);

			//sort children to recurse into closest first
			std::array<std::pair<scalar_type, tag_ptr_type>, O::N_CHILDREN> children_dist_pair;
			size_t i=0;
			for (tag_ptr_type c_ptr : *internal) {
				//note distance_squared(box, point) is a free standing function in box.hpp
				const scalar_type d2 = gutil::distance_squared(get_box(c_ptr), point);
				children_dist_pair[i++] = {d2, c_ptr};
			}
			std::sort(children_dist_pair.begin(), children_dist_pair.end()); //sorts lexigraphically, distance is first

			//descend into viable children
			for (auto [d2, c_ptr] : children_dist_pair) {
				if (d2 < dist2) {
					nearest_recursive(c_ptr, point, index, dist2);
				}
			}
		}
		else if (node.tag() == NodeTag::LEAF) {
			const leaf_node* leaf = static_cast<const leaf_node*>(node);
			for (index_type idx : *leaf) {
				assert(idx < data.size());
				const scalar_type d2 = distance_squared(point, data[idx]);
				if (d2 < dist2) {
					dist2 = d2;
					index = idx;
				}
			}
		}
	}
}