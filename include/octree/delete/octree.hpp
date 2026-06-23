#pragma once

#include "octree/node.hpp"

#include "geometry/point.hpp"
#include "geometry/box.hpp"

#include <vector>
#include <span>
#include <algorithm>
#include <type_traits>
#include <concepts>

#ifdef _OPENMP
#include <omp.h>
#endif

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

	//a class to organize internal nodes, leaf nodes, and stored data
	template<NodeOpts Opts, typename IS_VALID_FUN, typename DIST_FUN = nullptr_t>
	struct Octree
	{
		//bring Opts aliases into this scope (close to the std::Container concept)
		using value_type 		= typename Opts::value_type;
		using point_type 		= typename Opts::point_type;
		using box_type 			= typename Opts::box_type;
		using scalar_type 		= typename Opts::scalar_type;
		using reference 		= value_type&;
		using const_reference   = const value_type&;
		using size_type			= size_t;

		//define callables for spatial queries on data
		using is_valid_function_type = std::function<bool(box_type, value_type)>;
		using distance_function_type = std::function<scalar_type(point_type, value_type)>;

		//define node types
		using internal_node = InternalNode<Opts>;
		using leaf_node 	= LeafNode<Opts>;

		//make the options easy to read
		static constexpr NodeOpts OPTS = Opts;

		//index to return if no data is found
		static constexpr size_type NULL_DATA = size_type(-1);

		//ensure that IS_VALID_FUN is of the form IS_VALID_FUN(box_type, value_type) -> bool
		static_assert(ValidCheck<IS_VALID_FUN, box_type, value_type>);

		//determine if we can compute the distance from a point to data
		static constexpr bool DISTANCE_COMPUTABLE = DistanceCheck<DIST_FUN, point_type, value_type>;

		//bring partial interface of data container to the user
		auto begin() const {
			return _data_.cbegin();
		}
		
		auto end() const {
			return _data_.cend();
		}
		
		auto rbegin() const {
			return _data_.crbegin();
		}
		
		auto rend() const {
			return _data_.crend();
		}
		
		size_type size() const {
			return _data_.size();
		}
		
		size_type capacity() const {
			return _data_.capacity();
		}
		
		bool empty() const {
			return _data_.empty();
		}
		
		void clear() {
			_I_nodes_.clear(); 
			_L_nodes_.clear(); 
			_data_.clear(); 
			_L_nodes_.emplace_back(_root_bbox_);
			_root_index_.index = 0;
			_root_index_.tag = NodeTag::LEAF;
		}
		
		void reserve(const size_type sz) {
			if (sz>size()) {
				_data_.reserve(sz);
				_L_nodes_.reserve(sz/Opts::MAX_DATA);
			}
		}

		void shrink_to_fit() {
			_I_nodes_.shrink_to_fit(); 
			_L_nodes_.shrink_to_fit(); 
			_data_.shrink_to_fit();
		}

		const_reference operator[](const size_type idx) const {
			assert(idx<size()); 
			return _data_[idx];
		}
		
		const_reference at(const size_type idx) const {
			return _data_.at(idx);
		}

		//allow the user to get a reference to the underlying data.
		//note that altering any spatial data could make the octree invalid
		reference get_ref(const size_type idx) {
			assert(idx<size()); 
			return _data_[idx];
		}

		//data insertion methods, return the index of the inserted data
		//rather than being inserted, any duplicate data will return the index of the existing data
		//if data indices are to be stored in multiple leafs (i.e., Opts::SINGLE_DATA is false because the data has volume),
		//then all leafs will be checked to ensure that they all contain the correct index.
		size_type insert(value_type const& value);
		size_type insert(value_type&& value);
		std::vector<size_type> insert(std::span<const value_type> values);
		std::vector<size_type> insert(std::span<value_type> values);

		bool contains(const_reference value) const;
		size_type find(const_reference value) const;
		size_type find_closest(const point_type& point) const requires(DISTANCE_COMPUTABLE);
		std::vector<size_type> find_in_box(const box_type& box) const;

		//constructors
		Octree(const box_type box, IS_VALID_FUN&& validator) : 
			_root_bbox_(box), _validator_{std::move(validator)}, _dist_fun_{}
			{_L_nodes_.emplace_back(box);}

		Octree(const box_type box, IS_VALID_FUN&& validator, DIST_FUN&& dist=nullptr) :
			_root_bbox_(box), _validator_(std::move(validator)), _dist_fun_(std::move(dist))
			{_L_nodes_.emplace_back(box);}

	protected:
		//store the extents of the octree
		box_type _root_bbox_;
		NodeIndex _root_index_ {0, NodeTag::LEAF};

		//make primary storage for nodes and data
		std::vector<internal_node>	_I_nodes_;
		std::vector<leaf_node> 		_L_nodes_;
		std::vector<value_type> 	_data_;

		//store methods to call on data
		IS_VALID_FUN _validator_;
		DIST_FUN _dist_fun_;

		//get the bounding box of a node
		const box_type& get_box(const NodeIndex& index) const {
			assert(valid_index(index));
			return index.tag == NodeTag::INTERNAL ? _I_nodes_[index.index].bbox : _L_nodes_[index.index].bbox;
		}

		bool valid_index(const NodeIndex& index) const {
			return index.tag==NodeTag::INTERNAL ? (index.index < _I_nodes_.size()) : (index.index < _L_nodes_.size());
		}

		//find important nodes
		NodeIndex find_branch(const value_type& value) const;
		NodeIndex find_leaf(const value_type& value) const;
		std::vector<NodeIndex> find_leafs(const value_type& value) const;

		//split a leaf node and return the index to the start of the block of children
		//the children are guaranteed to be contiguous immediately after calling this, but not after
		//any other operations are done
		size_type split(const size_type L_idx);

		//swap two internal nodes and update their connectivities
		void swap_internal(const size_type idx_A, const size_type idx_B);

		//swap two leaf nodes and update their connectivities
		void swap_leaf(const size_type idx_A, const size_type idx_B);

	private:
		//get all leaf nodes that are descendents of a given internal node
		void get_all_leafs(std::vector<NodeIndex>& leafs, const NodeIndex& internal) const;

		//check if a leaf contains a link to given data
		size_type find_value_in_leaf(const NodeIndex& leaf, const value_type& value) const;

		//prepare a leaf to recieve a new index by splitting (multiple times if necessary)
		//return the index of the leaf that is prepared.
		NodeIndex prepare_leaf(const NodeIndex& leaf, const value_type& value);

		//similar to prepare_leaf(NodeIndex), but used for inserting data into multiple leafs
		void prepare_leaf(std::vector<NodeIndex>& leafs )
	};



	///////////////////////////////////////////////////////////////////////
	///	Octree Implementation
	///////////////////////////////////////////////////////////////////////
	template<NodeOpts O, typename V, typename D>
	void Octree<O,V,D>::swap_internal(const size_type idx_A, const size_type idx_B) {
		auto& list = _I_nodes_;
		assert(idx_A < list.size());
		assert(idx_B < list.size());
		if (idx_A == idx_B) {return;}

		//update parent indices
		const size_type parent_A_idx = list[idx_A].parent;
		const size_type parent_B_idx = list[idx_B].parent;

		if (parent_A_idx != parent_B_idx) {
			//the nodes have diferent parents (at least one exists)

			//update parent A
			if (parent_A_idx != NodeIndex::NULL_NODE) {
				internal_node& parent = list[parent_A_idx];
				for (NodeIndex& child_idx : parent.children) {
					if (child_idx.index == idx_A) {
						assert(child_idx.tag == NodeTag::INTERNAL);
						child_idx.index = idx_B;
					}
				}
			}

			//update parent B
			if (parent_B_idx != NodeIndex::NULL_NODE) {
				internal_node& parent = list[parent_B_idx];
				for (NodeIndex& child_idx : parent.children) {
					if (child_idx.index == idx_B) {
						assert(child_idx.tag == NodeTag::INTERNAL);
						child_idx.index = idx_A;
					}
				}
			}
		}
		else if (parent_A_idx != NodeIndex::NULL_NODE) {
			//the nodes are siblings and their parent exists
			internal_node& parent = list[parent_A_idx];
			for (NodeIndex& child_idx : parent.children) {
				if (child_idx.index == idx_A) {
					assert(child_idx.tag == NodeTag::INTERNAL);
					child_idx.index = idx_B;
				}
				else if (child_idx.index == idx_B) {
					assert(child_idx.tag == NodeTag::INTERNAL);
					child_idx.index = idx_A;
				}
			}
		}


		//update the children of node A to point to the parent at idx_B (where A will be moved to)
		for (const NodeIndex& child_idx : list[idx_A].children) {
			if (child_idx.index != NodeIndex::NULL_NODE) {
				if (child_idx.tag == NodeTag::INTERNAL) {
					_I_nodes_[child_idx.index].parent = idx_B;
				}
				else {
					_L_nodes_[child_idx.index].parent = idx_B;
				}
			}
		}

		//update the children of node B to point to the parent at idx_A (where B will be moved to)
		for (const NodeIndex& child_idx : list[idx_B].children) {
			if (child_idx.index != NodeIndex::NULL_NODE) {
				if (child_idx.tag == NodeTag::INTERNAL) {
					_I_nodes_[child_idx.index].parent = idx_A;
				}
				else {
					_L_nodes_[child_idx.index].parent = idx_A;
				}
			}
		}

		//update the root index if necessary
		if (_root_index_ == NodeIndex::internal(idx_A)) {_root_index_.index = idx_B;}
		else if (_root_index == NodeIndex::internal(idx_B)) {_root_index_.index = idx_A;}

		//swap the nodes
		std::swap(list[idx_A], list[idx_B]);
	}


	template<NodeOpts O, typename V, typename D>
	void Octree<O,V,D>::swap_leaf(const size_type idx_A, const size_type idx_B) {
		auto& list = _L_nodes_;
		assert(idx_A < list.size());
		assert(idx_B < list.size());
		if (idx_A == idx_B) {return;}

		//update parent indices
		const size_type parent_A_idx = list[idx_A].parent;
		const size_type parent_B_idx = list[idx_B].parent;

		if (parent_A_idx != parent_B_idx) {
			//the nodes have diferent parents (at least one exists)

			//update parent A
			if (parent_A_idx != NodeIndex::NULL_NODE) {
				internal_node& parent = _I_nodes_[parent_A_idx];
				for (auto& child_idx : parent.children) {
					if (child_idx.index == idx_A) {
						assert(child_idx.tag == NodeTag::LEAF);
						child_idx.index = idx_B;
					}
				}
			}

			//update parent B
			if (parent_B_idx != NodeIndex::NULL_NODE) {
				internal_node& parent = _I_nodes_[parent_B_idx];
				for (auto& child_idx : parent.children) {
					if (child_idx.index == idx_B) {
						assert(child_idx.tag == NodeTag::LEAF);
						child_idx.index = idx_A;
					}
				}
			}
		}
		else if (parent_A_idx != NodeIndex::NULL_NODE) {
			//the nodes are siblings and their parent exists
			internal_node& parent = _I_nodes_[parent_A_idx];
			for (auto& child_idx : parent.children) {
				if (child_idx.index == idx_A) {
					assert(child_idx.tag == NodeTag::LEAF);
					child_idx.index = idx_B;
				}
				else if (child_idx.index == idx_B) {
					assert(child_idx.tag == NodeTag::LEAF);
					child_idx.index = idx_A;
				}
			}
		}


		//note that it is impossible to have two (distinct) leafs with one of them the root

		//swap the nodes
		std::swap(list[idx_A], list[idx_B]);
	}

	template<NodeOpts O, typename V, typename D>
	size_type Octree<O,V,D>::split(const size_type L_idx) {
		assert(L_idx < _L_nodes_.size());
		const bool is_root = (_root_index_.tag == NodeTag::LEAF && _root_index_.index == L_idx);

		//move the leaf to be split to the end so that it can be deleted
		//this will update the connectivity of the parent nodes as well as track the root node if necessary
		const size_type last_idx = _L_nodes_.size()-1;
		swap_leaf(L_idx, last_idx);

		//create the new internal node (last_idx now points to the leaf to be deleted)
		const size_type new_internal_idx = _I_nodes_.size();
		_I_nodes_.emplace_back(_L_nodes_[last_idx].bbox);
		_I_nodes_[new_internal_idx].parent = _L_nodes_[last_idx].parent;

		//update the index to the node to be split (both the type and index change)
		//note that the swap_leaf updated the index from L_idx to last_idx
		if (_I_nodes_[new_internal_idx].parent != NodeIndex::NULL_NODE) {
			const size_type parent_idx = _I_nodes_[new_internal_idx].parent;
			for (auto& child_idx : _I_nodes_[parent_idx].children) {
				if (child_idx.index == last_idx) {
					child_idx.index = new_internal_idx;
					assert(child_idx.tag == NodeTag::LEAF);
					child_idx.tag = NodeTag::INTERNAL;
					break;
				}
			}
		}

		//stash required data and delete the leaf (last_idx now points to the leaf to be deleted)
		std::array<size_type, O::MAX_DATA> data_to_move = std::move(_L_nodes_[last_idx].data);
		size_type cursor = _L_nodes_[last_idx].cursor;

		const point_type low = _L_nodes_[last_idx].bbox.low();
		const point_type high = _L_nodes_[last_idx].bbox.high();
		_L_nodes_.pop_back();

		//initialize the new leaf nodes
		const point_type center = low + scalar_type{0.5}*(high-low);
		const size_type child_index_start = _L_nodes_.size();
		for (size_type child=0; child<O::N_CHILDREN; ++child) {
			//get the region of the child
			//each bit of the child designates the axis low/high region
			point_type vertex = low;
			for (size_type ax=0; ax<O::DIM; ++ax) {
				if ( child & (size_type{1}<<ax)) {
					vertex[ax] = high[ax];
				}
			}

			//create child and update connectivity
			size_type child_index = _L_nodes_.size();
			_L_nodes_.emplace_back(box_type{center, vertex});
			_I_nodes_[new_internal_idx].children[child] = NodeIndex{child_index, NodeTag::LEAF};
			_L_nodes_[child_index].parent = new_internal_idx;
		}

		//push data down to the new leaf nodes
		for (size_type ii=0; ii<cursor; ++ii) {
			const size_type data_idx = data_to_move[ii];
			for (size_type c_idx=child_index_start; c_idx<child_index_start+O::N_CHILDREN; ++c_idx) {
				leaf_node& leaf = _L_nodes_[c_idx];
				if (_validator_(leaf.bbox, _data_[data_idx])) {
					leaf.insert(data_idx);
					if constexpr (O::SINGLE_DATA) {break;}
				}
			}
		}

		//update the root node if it was the root node that was split
		if (is_root) {
			_root_index_.index 	= new_internal_idx;
			_root_index_.tag 	= NodeTag::INTERNAL;
		}

		//return the start of the new children block
		return child_index_start;
	}

	template<NodeOpts O, typename V, typename D>
	void Octree<O,V,D>::get_all_leafs(std::vector<NodeIndex>& leafs, const NodeIndex& internal) const {
		assert(valid_index(internal));
		if (internal.tag == NodeTag::LEAF) {
			leafs.emplace_back(internal);
			return;
		}
		else {
			const internal_node& node = _I_nodes_[internal.index];
			for (const NodeIndex& idx : node.children) {
				get_all_leafs(leafs, idx);
			}
		}
	}

	template<NodeOpts O, typename V, typename D>
	std::vector<NodeIndex> Octree<O,V,D>::find_leafs(const value_type& value) const {
		//get the branch or leaf
		const NodeIndex branch = find_branch(value);
		assert(valid_index(branch));
		if (branch.tag == NodeTag::LEAF) {return {branch};}

		//get all leafs that need the value
		std::vector<NodeIndex> leafs;
		const internal_node& node = _I_nodes_[branch.index];
		for (const NodeIndex& idx : node) {
			if (_validator_(get_box(idx), value)) {
				get_all_leafs(leafs, idx);
			}
		}

		return leafs;
	}

	template<NodeOpts O, typename V, typename D>
	NodeIndex Octree<O,V,D>::find_branch(const value_type& value) const {
		//descend the tree until we have more than one valid child or we hit a leaf
		NodeIndex idx = _root_index_;
		while (idx.tag != NodeIndex::LEAF) {
			assert(valid_index(idx));
			assert(_validator_(get_box(idx), value));
			int n_valid_children = 0;
			NodeIndex last_valid = tag;
			const internal_node& node = _I_nodes_[idx.index];
			

			#ifndef NDEBUG
			//ensure that at least one child is valid
			bool has_valid_child = false;
			#endif
			for (const NodeIndex& child_idx : node.children) {
				if (_validator_(get_box(child_idx), value)) {
					idx = child_idx;
					++n_valid_children;
					#ifndef NDEBUG
					has_valid_child = true;
					#endif
				}
			}

			#ifndef NDEBUG
			assert(has_valid_child);
			#endif

			//check if we hit a branch node
			if (n_valid_children > 1) {return idx;}
			assert(n_valid_children==1);
		}

		//we hit a leaf
		assert(valid_index(idx));
		assert(idx.tag==NodeTag::LEAF);
		return idx;
	}

	template<NodeOpts O, typename V, typename D>
	NodeIndex Octree<O,V,D>::find_leaf(const value_type& value) const {
		//descend the tree until we get to a valid leaf
		NodeIndex idx = _root_index_;
		while (idx.tag != NodeIndex::LEAF) {
			assert(valid_index(idx));
			assert(_validator_(get_box(idx), value));
			const internal_node& node = _I_nodes_[idx.index];
			
			#ifndef NDEBUG
			//ensure that at least one child is valid
			bool has_valid_child = false;
			#endif

			for (const NodeIndex& child_idx : node.children) {
				if (_validator_(get_box(child_idx), value)) {
					idx = child_idx;
					#ifndef NDEBUG
					has_valid_child = true;
					#endif
					break;
				}
			}

			#ifndef NDEBUG
			assert(has_valid_child);
			#endif
		}

		assert(valid_index(idx));
		assert(idx.tag==NodeTag::LEAF);
		return idx;
	}

	template<NodeOpts O, typename V, typename D>
	Octree<O,V,D>::size_type Octree<O,V,D>::find(const value_type& value) const {
		//descend to first valid leaf (always has the index of the value if it is present in the tree)
		const NodeIndex idx = find_leaf(value);
		
		assert(idx.tag == NodeTag::LEAF);
		assert(valid_index(idx));
		const leaf_node& leaf = _L_nodes_[idx.index];

		//check if the leaf has an index to the value
		for (size_type ii=0; ii<leaf.cursor; ++ii) {
			if (_data_[leaf.data[ii]] == value) {
				return leaf.data[ii];
			}
		}

		return NULL_DATA;
	}

	template<NodeOpts O, typename V, typename D>
	Octree<O,V,D>::size_type Octree<O,V,D>::find_value_in_leaf(const NodeIndex& leaf, const value_type& value) const {
		assert(valid_index(leaf));
		assert(leaf.tag == NodeTag::LEAF);
		const leaf_node& node = _L_nodes_[leaf.index];
		for (const auto it : node) {
			if (_data_[*it] == value) {return *it;}
		}
		return NULL_DATA;
	}
	
	template<NodeOpts O, typename V, typename D>
	


	template<NodeOpts O, typename V, typename D>
	NodeIndex Octree<O,V,D>::prepare_leaf(const NodeIndex& leaf, const value_type& value) {
		assert(leaf.tag == NodeTag::LEAF);
		assert(valid_index(leaf));
		assert(_validator_(get_box(leaf), value));

		if (!leaf.full()) {return leaf;}
		
		const size_type child_start = split(leaf.index);
		for (size_type idx = child_start; idx<child_start+O::N_CHILDREN; ++idx) {
			const NodeIndex child = NodeIndex::leaf(idx);
			if (_validator_(get_box(child), value)) {return prepare_leaf(child, value);}
		}

		//we should never reach this point
		assert(false);
		return NodeIndex::null();
	}

	template<NodeOpts O, typename V, typename D>
	Octree<O,V,D>::size_type Octree<O,V,D>::insert(value_type&& value) {
		if constexpr (O::SINGLE_DATA) {
			NodeIndex leaf = find_leaf(value);
			size_type value_index = find_value_in_leaf(leaf, value);
			if (value_index != NULL_DATA) {return value_index;}

			//prepare the leaf (it may need to be split)
			leaf = prepare_leaf(leaf, value);
			assert(valid_index(leaf));
			assert(leaf.tag == NodeTag::LEAF);
			assert(_validator_(get_box(leaf), value));

			//insert the data
			value_index = _data_.size();
			_data_.push_back(std::move(value));
			[[maybe_unused]] InsertReturn flag = _L_nodes_[leaf.index].insert(value_index);
			assert(flag == InsertReturn::SUCCESS);

			return value_index;
		}
		else {
			//find all leafs that need the data
		}
	}

}