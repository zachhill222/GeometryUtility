#pragma once

#include "utility/utility.hpp"
#include "math/math.hpp"
#include "memory/thread_pool.hpp"

#include "geometry/point.hpp"
#include "geometry/box.hpp"

#include "memory/hetero_slab_allocator.hpp"
#include "algorithms/sorting.hpp"

#include "octree/node.hpp"

#include <vector>
#include <array>
#include <span>
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <mutex>

namespace gutil
{
	//concepts for octrees
	template<typename T>
	concept IsOctree = requires(T tree, const T ctree,
			typename T::value_type val,
			typename T::box_type   box,
			std::span<typename T::value_type> sp) {
		// nested types
		typename T::value_type;
		typename T::point_type;
		typename T::box_type;
		typename T::scalar_type;
		typename T::index_type;
		typename T::BASE;

		// construction
		T(box);

		// size and capacity
		{ tree.size()     } -> std::convertible_to<size_t>;
		{ ctree.size()    } -> std::convertible_to<size_t>;
		{ tree.empty()    } -> std::convertible_to<bool>;
		{ tree.reserve(size_t{}) };

		// bbox
		{ ctree.bbox()    } -> std::same_as<const typename T::box_type&>;

		// data access
		{ tree.data()     } -> std::same_as<std::vector<typename T::value_type>&>;
		{ ctree.data()    } -> std::same_as<const std::vector<typename T::value_type>&>;

		// insertion
		{ tree.push_back(val)           } -> std::convertible_to<typename T::index_type>;
		{ tree.push_back(std::move(val))} -> std::convertible_to<typename T::index_type>;
		{ tree.batch_insert(sp)         };

		// queries
		{ ctree.contains(val) } -> std::convertible_to<bool>;
		{ ctree.size()        } -> std::convertible_to<size_t>;

		// maintenance
		{ tree.clear() };
	} && IsReal<typename T::point_type::scalar_type>;

	template<typename T>
	concept IsOctreeWithDistance =
		IsOctree<T> &&
		T::HAS_DISTANCE &&
		requires(const T ctree, typename T::point_type pt) {
	    { ctree.nearest(pt) } -> std::convertible_to<typename T::index_type>;
	};

	template<typename T>
	concept IsVolumeOctree =
		IsOctree<T> &&
		requires { { T::BASE::OPTS::VOLUME_DATA } -> std::convertible_to<bool>; } &&
		T::BASE::OPTS::VOLUME_DATA;

	template<typename T>
	concept IsPointOctree = IsOctree<T> && !IsVolumeOctree<T>;



	//base type for octrees
	template<IsNodeOpts Opts, typename Derived>
	struct OctreeBase
	{
		////////////////////////////////////////////////////////////////
		// Data and aliases
		////////////////////////////////////////////////////////////////
		using internal_node_type = InternalNode<Opts>;
		using leaf_node_type 	 = LeafNode<Opts>;
		using tag_ptr_type       = typename internal_node_type::tag_ptr_type;

		using value_type    = typename Opts::value_type;	//type that this octree is storing
		using index_type  	= typename Opts::index_type;	//type that is stored in the leaf nodes
		using point_type  	= typename Opts::point_type;	//type of spatial points
		using box_type	  	= typename Opts::box_type;		//type of spatial axis-aligned-bounding-boxes
		using scalar_type 	= typename Opts::scalar_type;	//type that emulates real numbers for the spatial points and aabb
		static_assert(IsReal<scalar_type>, "OctreeBase - the scalar type must emulate the real numbers");

		static constexpr Opts OPTS{};	//capture the compile time options (part of the type)

		//define a struct for useful debug information
		struct OctreeStats {
			size_t n_data{0};
			size_t n_internal{0};
			size_t n_leaves{0};
			size_t depth_max{0};
			size_t bytes_data{0};
			size_t bytes_leaves{0};
			size_t bytes_internal{0};
			size_t bytes_nodes{0};
			size_t bytes_reserved{0};
		};

	protected:
		//track where the root is
		internal_node_type* root{nullptr};

		//store the data in a contiguous vector, leaves store indices into this
		std::vector<value_type> _data_;

	private:
		ThreadPool thread_pool{4};
		static constexpr size_t SPAWN_THREAD_THRESHOLD = 512;
		mutable std::mutex mtx;

		//make allocator for less fragmented node storage and some convenience contructors
		HeteroSlabAllocator<internal_node_type,leaf_node_type> _alloc_{};

	public:
		//default constructor has no root or bounding box
		//use set_bbox before inserting data
		OctreeBase() {}

		OctreeBase(const box_type& box) {
			//always immediately split the root node so that it doen't need to be tracked
			//if it is a leaf or internal node type. this keeps the root node from moving in memory as well.
			root = construct_internal(box);
			root -> construct_child_leafs(_alloc_leaf_());
		}

		//moving is trivial
		OctreeBase(OctreeBase&& other) : 
			root(other.root), 
			_data_(std::move(other._data_)), 
			_alloc_(std::move(other._alloc_)) {
			other.root = nullptr;
		}

		OctreeBase& operator=(OctreeBase&& other) {
			if (&other != this) {
				_alloc_.release();
				root = other.root;
				_alloc_ = std::move(other._alloc_);
				_data_  = std::move(other._data_);
				other.root = nullptr;
			}

			return *this;
		}

		//do not copy
		OctreeBase(const OctreeBase& other) = delete;
		OctreeBase operator=(const OctreeBase& other) = delete;

		//ensure the nodes will be correctly freed when the allocator is destroyed
		static_assert(std::is_trivially_destructible_v<internal_node_type>);
		static_assert(std::is_trivially_destructible_v<leaf_node_type>);
		~OctreeBase() {}



		////////////////////////////////////////////////////////////////
		// Public Interface
		////////////////////////////////////////////////////////////////
		bool contains(const value_type& value) const {
			return leaf_contains(find_leaf(value), value);
		}

		index_type find(const value_type& value) const {
			const leaf_node_type* leaf = find_leaf(value);
			if (leaf) {
				for (index_type idx : *leaf) {
					assert(idx<_data_.size());
					if (_data_[idx] == value) {return idx;}
				}
			}
			return index_type(-1);
		}

		const box_type& bbox() const {
			assert(root!=nullptr);
			return root->bbox;
		}

		bool empty() const {return _data_.empty();}
		size_t size() const {return _data_.size();}
		void reserve(const size_t n) {_data_.reserve(n);}

		//some public simple operations
		void clear() {
			reset(root->bbox);
		}

		//iterators for better looping through the data
		//note that changing the dat could invalidate the tree
		auto begin() {return _data_.begin();}
		auto begin() const {return _data_.cbegin();}
		auto cbegin() const {return _data_.cbegin();}

		auto end() {return _data_.end();}
		auto end() const {return _data_.cend();}
		auto cend() const {return _data_.cend();}

		auto rbegin() {return _data_.rbegin();}
		auto rbegin() const {return _data_.crbegin();}
		auto crbegin() const {return _data_.crbegin();}

		auto rend() {return _data_.rend();}
		auto rend() const {return _data_.crend();}
		auto crend() const {return _data_.crend();}

		//access the raw data
		std::vector<value_type>& data() {return _data_;}
		const std::vector<value_type>& data() const {return _data_;}

		//accessors to the data, note that changing the data
		//could invalidate the tree
		const value_type& operator[](const size_t idx) const {
			assert(idx<_data_.size());
			return _data_[idx];
		}

		value_type& operator[](const size_t idx) {
			assert(idx<_data_.size());
			return _data_[idx];
		}

		const value_type& at(const size_t idx) const {
			return _data_.at(idx);
		}

		value_type& at(const size_t idx) {
			return _data_.at(idx);
		}

		//insert data into the tree and return its index
		index_type push_back(value_type&& value);
		
		index_type push_back(const value_type& value) {
			return push_back(std::move(value_type{value}));
		}

		//batch insert _data_. all data that is valid in the current
		//bounding box will be inserted, but the order may not be preserved.
		void batch_insert(std::span<value_type> values) {
			filter_to_bbox(values);
			thread_pool.submit([this, values]() { batch_insert_recursive(root->t_ptr(), values); });
			thread_pool.wait_idle();
			// batch_insert_recursive(root->t_ptr(), values);
		}

		void batch_insert(std::span<const value_type> values) {
			std::vector<value_type> cpy(values.begin(), values.end());
			batch_insert(std::span<value_type>{cpy});
		}

		void batch_insert(const std::vector<value_type>& values) {
			batch_insert(std::span<const value_type>(values.begin(), values.end()));
		}

		void batch_insert(std::vector<value_type>&& values) {
			batch_insert(std::span<value_type>(values.begin(), values.end()));
		}

		//find the nearest data to a point
		index_type nearest(const point_type& point) const requires (Derived::HAS_DISTANCE) {
			if (_data_.empty()) {
				return index_type(-1);
			}
			else {
				index_type index = 0;
				scalar_type dist2 = this->distance_squared(point, _data_[index]);
				nearest_recursive(root->t_ptr(), point, index, dist2);
				return index;
			}
		}

		//set a new bounding box by reconstructing the octree
		void set_bbox(const box_type& box);

		//get octree stats
		OctreeStats get_stats() const {
			//count nodes
			OctreeStats stats{};
			get_stats_recursive(root->t_ptr(), stats, 0);
			
			//set data size
			stats.n_data = _data_.size();
			stats.bytes_data = _data_.size() * sizeof(value_type);
			
			//set node size
			stats.bytes_leaves = stats.n_leaves * sizeof(leaf_node_type);
			stats.bytes_internal = stats.n_internal * sizeof(internal_node_type);
			stats.bytes_nodes = stats.bytes_leaves + stats.bytes_internal;

			//set capacity
			stats.bytes_reserved = _data_.capacity() * sizeof(value_type) + _alloc_.bytes_reserved();

			return stats;
		}

	protected:
		////////////////////////////////////////////////////////////////
		// Private Utility Functions
		////////////////////////////////////////////////////////////////

		//reset the tree to a given box and clear the data
		void reset(const box_type box) {
			_data_.clear();
			_alloc_.release();
			root = construct_internal(box);
			root -> construct_child_leafs(_alloc_leaf_());
		}

		//get the box from an unkonwn node type
		const box_type& get_box(const tag_ptr_type t_ptr) const {
			assert(!t_ptr.is_null());
			return t_ptr.tag() == NodeTag::LEAF ?
				static_cast<const leaf_node_type*>(t_ptr)->bbox :
				static_cast<const internal_node_type*>(t_ptr)->bbox;
		}

		//pass intersection query to the derived class
		bool intersects(const box_type& box, const value_type& value) const {
			return static_cast<const Derived*>(this)->intersects_impl(box, value);
		}

		bool intersects(const box_type& box, const index_type index) const {
			assert(index < _data_.size());
			return intersects(box, _data_[index]);
		}

		//for point data, get the point that a value is located
		point_type get_point(const value_type& value) const {
			return static_cast<const Derived*>(this)->get_point_impl(value);
		}

		//pass distance query to the derived class
		scalar_type distance_squared(const point_type& point, const value_type& value) const 
			requires (Derived::HAS_DISTANCE) {
			return static_cast<const Derived*>(this)->distance_squared_impl(point, value);
		}

		scalar_type distance_squared(const point_type& point, const index_type index) const 
			requires (Derived::HAS_DISTANCE) {
			assert(index < _data_.size());
			return gutil::distance_squared(point, _data_[index]);
		}

		//pass checking if a leaf contains a value to the derived class (necessary if the leaf stores indices)
		bool leaf_contains(const leaf_node_type* leaf, const value_type& value) const {
			if (leaf) {
				for (index_type idx : *leaf) {
					assert(idx < _data_.size());
					if (_data_[idx] == value) {return true;}
				}
			}
			return false;
		}

		bool leaf_contains(const leaf_node_type* leaf, const index_type index) const {
			assert(index < _data_.size());
			return (leaf!=nullptr) && (leaf->contains(index));
		}

		//filter data to the current bounding box
		void filter_to_bbox(std::vector<value_type>& values) const {
			auto pred = [&](const value_type& v) {return !intersects(bbox(), v);};
			std::erase_if(values, pred);
		}

		void filter_to_bbox(std::span<value_type>& values) const {
			auto pred = [&](const value_type& v) {return intersects(bbox(), v);};
			auto mid  = std::partition(values.begin(), values.end(), pred);
			values = std::span<value_type>{values.begin(), mid};
		}

		//split a leaf and return the internal node that it was converted to
		internal_node_type* split(leaf_node_type* leaf);

		//find the first leaf that can contain some value
		leaf_node_type* find_leaf(const value_type& value) const;

		//find the first leaf node that contains the specified point
		leaf_node_type* find_leaf(const point_type& point) const requires (!std::same_as<value_type,point_type>);

		//find the first internal node that has multiple children that intersect the data
		//or return the first leaf that intersects the data if no such internal node exists
		tag_ptr_type find_branch(const value_type& value) const;

		//get all leafs that are descendants of a node
		void get_all_leaves(std::vector<leaf_node_type*>& leaves, const tag_ptr_type node) const;

		//recursive portion of insert to split nodes as needed
		//handles point data and volume data
		void insert_recursive(const tag_ptr_type node, const value_type& value, const index_type index);

		//recusive portion of batch insert
		void batch_insert_recursive(const tag_ptr_type node, std::span<value_type> values) requires (!Opts::VOLUME_DATA);

		//recursive portion of finding the nearest data to a point
		//index and dist are the current best index/data and its distance
		void nearest_recursive(const tag_ptr_type node, const point_type& point, 
				index_type& index, scalar_type& dist2) const requires (Derived::HAS_DISTANCE);
		
		//recursive portion of computing tree stats
		void get_stats_recursive(const tag_ptr_type node, OctreeStats& stats, size_t cur_depth) const;

	private:
		auto& _alloc_internal_() {return _alloc_.template pool<internal_node_type>();}
		auto& _alloc_leaf_() {return _alloc_.template pool<leaf_node_type>();}

		leaf_node_type* construct_leaf(const box_type& box) 		{return _alloc_.template construct<leaf_node_type>(box);}
		internal_node_type* construct_internal(const box_type& box) {return _alloc_.template construct<internal_node_type>(box);}
		void destroy_leaf(leaf_node_type* p) 						{_alloc_.template destroy<leaf_node_type>(p);}
		void destroy_internal(internal_node_type* p) 				{_alloc_.template destroy<internal_node_type>(p);}
	};


	template<IsNodeOpts O, typename D>
	typename OctreeBase<O,D>::internal_node_type* OctreeBase<O,D>::split(leaf_node_type* leaf) {
		//allocate a new internal node that will take the place of the leaf
		internal_node_type* internal = construct_internal(leaf->bbox);
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
				leaf_node_type* lf = static_cast<leaf_node_type*>(t_ptr);
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
	typename OctreeBase<O,D>::leaf_node_type* OctreeBase<O,D>::find_leaf(const value_type& value) const {
		if (!intersects(root->bbox, value)) {return nullptr;}

		tag_ptr_type node = root->t_ptr();
		while (node.tag() == NodeTag::INTERNAL) {
			const internal_node_type* ptr = static_cast<const internal_node_type*>(node);
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

		return static_cast<leaf_node_type*>(node);
	}

	template<IsNodeOpts O, typename D>
	typename OctreeBase<O,D>::leaf_node_type* OctreeBase<O,D>::find_leaf(const point_type& point) const 
		requires (!std::same_as<value_type,point_type>) {
		if (!root->bbox.contains(point)) {return nullptr;}

		tag_ptr_type node = root->t_ptr();
		while (node.tag() == NodeTag::INTERNAL) {
			const internal_node_type* ptr = static_cast<const internal_node_type*>(node);
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

		return static_cast<leaf_node_type*>(node);
	}

	template<IsNodeOpts O, typename D>
	typename OctreeBase<O,D>::tag_ptr_type OctreeBase<O,D>::find_branch(const value_type& value) const {
		if (!intersects(root->bbox, value)) {return tag_ptr_type::null();}

		tag_ptr_type node = root->t_ptr();
		while (node.tag() == NodeTag::INTERNAL) {
			const internal_node_type* ptr = static_cast<const internal_node_type*>(node);
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
	void OctreeBase<O,D>::get_all_leaves(std::vector<leaf_node_type*>& leaves, const tag_ptr_type node) const {
		if (node.tag() == NodeTag::INTERNAL) {
			for (const tag_ptr_type c_ptr : *static_cast<const internal_node_type*>(node)) {
				assert(!c_ptr.is_null());
				get_all_leaves(leaves, c_ptr);
			}
		}
		else if (node.tag() == NodeTag::LEAF) {
			leaves.emplace_back(static_cast<leaf_node_type*>(node));
		}
	}

	template<IsNodeOpts O, typename D>
	void OctreeBase<O,D>::insert_recursive(const tag_ptr_type node, const value_type& value, const index_type index) {
		if (node.tag() == NodeTag::LEAF) {
			leaf_node_type* leaf = static_cast<leaf_node_type*>(node);
			if (leaf->contains(index)) {return;}
			else if (leaf->full()) {
				internal_node_type* internal = split(leaf);
				insert_recursive(internal->t_ptr(), value, index);
				return;
			}
			else {
				leaf->insert(index);
				return;
			}
		}
		else if (node.tag() == NodeTag::INTERNAL) {
			internal_node_type* internal = static_cast<internal_node_type*>(node);
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
	typename OctreeBase<O,D>::index_type OctreeBase<O,D>::push_back(value_type&& value) {
		//get the best leaf for the data
		leaf_node_type* leaf = find_leaf(value);
		if (leaf==nullptr) {return index_type(-1);}

		//check if the leaf contains the data
		for (index_type idx : *leaf) {
			assert(idx < _data_.size());
			if (_data_[idx] == value) {
				return idx;
			}
		}

		//the data is new
		index_type idx;
		{
			std::lock_guard<std::mutex> lock(mtx);
			idx = _data_.size();
			_data_.push_back(std::move(value));
		}

		if constexpr (!O::VOLUME_DATA) {
			insert_recursive(leaf->t_ptr(), _data_[idx], idx);
		}
		else {
			const tag_ptr_type branch = find_branch(_data_[idx]);
			insert_recursive(branch, _data_[idx], idx);
		}

		return idx;
	}

	//find the nearest data to a point
	template<IsNodeOpts O, typename D>
	void OctreeBase<O,D>::nearest_recursive(const tag_ptr_type node, 
		const point_type& point, index_type& index, scalar_type& dist2) const requires (D::HAS_DISTANCE) {
		if (node.tag() == NodeTag::INTERNAL) {
			const internal_node_type* internal = static_cast<const internal_node_type*>(node);

			//sort children to recurse into closest first
			std::array<std::pair<scalar_type, tag_ptr_type>, O::N_CHILDREN> children_dist_pair;
			size_t i=0;
			for (tag_ptr_type c_ptr : *internal) {
				//note distance_squared(box, point) is a free standing function in gutilmath.hpp
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
			const leaf_node_type* leaf = static_cast<const leaf_node_type*>(node);
			for (index_type idx : *leaf) {
				assert(idx < _data_.size());
				const scalar_type d2 = this->distance_squared(point, _data_[idx]);
				if (d2 < dist2) {
					dist2 = d2;
					index = idx;
				}
			}
		}
	}

	template<IsNodeOpts O, typename D>
	void OctreeBase<O,D>::set_bbox(const box_type& box) {
		std::vector<value_type> old_vals = std::move(_data_);
		reset(box);
		batch_insert(std::move(old_vals));
	}

	template<IsNodeOpts O, typename D>
	void OctreeBase<O,D>::batch_insert_recursive(const tag_ptr_type node, std::span<value_type> values) requires (!O::VOLUME_DATA) {
		assert(!node.is_null());

		if (node.tag() == NodeTag::LEAF) {
			leaf_node_type* leaf = static_cast<leaf_node_type*>(node);
			if (leaf->remaining_capacity() >= values.size()) {
				//insert data if there is room
				for (value_type& value : values) {
					if (!leaf_contains(leaf, value)) {
						index_type idx;
						{
							std::lock_guard<std::mutex> lock(mtx);
							idx = _data_.size();
							_data_.push_back(std::move(value));
						}
						leaf->insert(idx);
					}
				}
			}
			else {
				//split leaf and re-try if there is no room
				internal_node_type* internal = split(leaf);
				batch_insert_recursive(internal->t_ptr(), values);
			}
		}
		else {
			internal_node_type* internal = static_cast<internal_node_type*>(node);
			const box_type& box = internal->bbox;
			const point_type cntr = box.center();
			auto pred = [&cntr, this](const value_type& val) {
				return internal_node_type::octant(cntr, this->get_point(val));
			};

			gutil::BinSort sorter{values, O::N_CHILDREN};
			sorter.sort(pred);
			for (int c_idx=0; c_idx<O::N_CHILDREN; ++c_idx) {
				auto c_ptr = internal->children[c_idx];

				const bool spawn_thread = (c_idx!=O::N_CHILDREN-1) && 
						(sorter.bin_size(c_idx)>SPAWN_THREAD_THRESHOLD);

				if (spawn_thread) {
					std::span<value_type> child_data = sorter.get_bin(c_idx);
					auto fun = [this, child_data, c_ptr]() {this->batch_insert_recursive(c_ptr, child_data); };
					thread_pool.submit(fun);
				}
				else {
					batch_insert_recursive(c_ptr, sorter.get_bin(c_idx));
				}
			}
		}
	}

	template<IsNodeOpts O, typename D>
	void OctreeBase<O,D>::get_stats_recursive(const tag_ptr_type node, OctreeStats& stats, size_t cur_depth) const {
		if (node.tag() == NodeTag::LEAF) {
			stats.n_leaves++;
			stats.depth_max = std::max(stats.depth_max, cur_depth);
		}
		else if (node.tag() == NodeTag::INTERNAL) {
			stats.n_internal++;
			stats.depth_max = std::max(stats.depth_max, cur_depth);
			for (tag_ptr_type c_ptr : *static_cast<const internal_node_type*>(node)) {
				get_stats_recursive(c_ptr, stats, cur_depth+1);
			}
		}
	}


	///////////////////////////////////////////////////////////
	/// Free standing functions to join octrees
	///////////////////////////////////////////////////////////

	//join via a single batch insertion (requires lots of memory)
	template<IsOctree Tree, IsOctree... Rest>
    requires (std::same_as<std::remove_cvref_t<Tree>, std::remove_cvref_t<Rest>> && ...)
	std::remove_cvref_t<Tree> join_trees(Tree&& first, Rest&&... rest) {
		using tree_type = std::remove_cvref_t<Tree>;
		using value_type = typename tree_type::value_type;
		using box_type = typename tree_type::box_type;

		//get upper bound on data
		const size_t N_DATA = first.size() + (0 + ... + rest.size());

		//union the bounding boxes
		box_type box = first.bbox();
		( (box = gutil::merge(box, rest.bbox())), ...);

		//move data into a single vector
		std::vector<value_type> buffer;
		buffer.reserve(N_DATA);

		auto move_data = [&](auto&& src_tree) {
			auto& data = src_tree.data();
			buffer.insert(buffer.end(),
				std::make_move_iterator(data.begin()),
				std::make_move_iterator(data.end()));
			data.clear();
		};

		move_data(std::move(first));
		(move_data(std::move(rest)), ...);

		//build the new octree and insert the data
		tree_type tree(box);
		tree.reserve(N_DATA);
		tree.batch_insert(std::move(buffer));

		return tree;
	}

	//join via binary joining (requires less memory)
	template<IsOctree Tree>
	Tree join_trees(std::span<Tree> trees) {
		assert(!trees.empty() && "join_trees: empty span");

		//base cases
		if (trees.size() == 1) {return std::move(trees[0]);}
		else if (trees.size() == 2) {return join_trees(std::move(trees[0]), std::move(trees[1]));}

		//recursive split
		const size_t l_size = trees.size() / 2;
		const size_t r_size = trees.size() - l_size;
		std::span<Tree> left_span = trees.subspan(0, l_size);
		std::span<Tree> right_span = trees.subspan(l_size, r_size);

		//initialize left and right trees
		Tree left_result, right_result;

		//dispatch new thread for left (current thread for right)
		std::thread left_thread([&] {
			left_result = std::move(join_trees(left_span));
		});

		right_result = std::move(join_trees(right_span));

		left_thread.join();

		//combine the left and right trees
		return join_trees(std::move(left_result), std::move(right_result));
	}

	template<IsOctree Tree>
	Tree join_trees(std::vector<Tree>&& trees) {
		return join_trees(std::span{trees});
	}

	template<IsOctree Tree>
	std::ostream& operator<<(std::ostream& os, const Tree& tree) {
		auto s = tree.get_stats();

		// helper to format bytes
		auto fmt_bytes = [](size_t b) -> std::string {
			if (b < 1024)
				return std::to_string(b) + " B";
			else if (b < 1024*1024)
				return std::to_string(b/1024) + " KB";
			else
				return std::to_string(b/(1024*1024)) + " MB";
		};

		//print stats
		os << "OctreeBase {\n"
			<< "  data:             " << s.n_data     << " items\n"
			<< "  internal nodes:   " << s.n_internal << "\n"
			<< "  leaf nodes:       " << s.n_leaves   << "\n"
			<< "  max depth:        " << s.depth_max  << "\n"
			<< "  memory (data):    " << fmt_bytes(s.bytes_data)  << "\n"
			<< "  memory (leaves):  " << fmt_bytes(s.bytes_leaves) << "\n"
			<< "  memory (internal):" << fmt_bytes(s.bytes_internal) << "\n"
			<< "  memory (nodes):   " << fmt_bytes(s.bytes_nodes) << "\n"
			<< "  memory (reserved):" << fmt_bytes(s.bytes_reserved) << "\n"
			<< "  bbox:             " << tree.bbox() << "\n";

		return os;
	}
}