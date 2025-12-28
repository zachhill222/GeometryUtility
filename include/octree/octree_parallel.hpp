#pragma once

#include <cassert>
#include <vector>

#include <cstring>
#include <cmath>

#include <mutex>
#include <shared_mutex>

#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "octree/octree_util.hpp"
#include "octree/thread_queue.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////
/// BasicParallelOctree - Thread-safe spatial data structure
///
/// This octree supports concurrent insertions from multiple OpenMP threads via push_back_async().
/// Space must be pre-allocated via resize() before async insertions. Use shrink_to_fit() to reclaim
/// unused space afterwards.
///
/// Key features:
/// - Immediate data storage: push_back_async() stores data immediately and returns its index
/// - Deferred tree updates: A dedicated worker thread updates the octree structure asynchronously
/// - flush() waits until all pending tree updates complete
///
/// Thread safety:
/// - Multiple threads can call push_back_async() concurrently
/// - If the same new data is inserted by multiple threads, each gets a different index but only
///   one is correct. If data already exists in the tree, the correct index is safely returned.
////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace gutil {
	/////////////////////////////////////////////////
	/// BasicParallelOctree - Main container class
	///
	/// @tparam Data_t           Type of data to store
	/// @tparam SINGLE_DATA      If true, data goes in first valid leaf only.
	///                            If false, data goes in all overlapping leaves.
	/// @tparam DIM              Spatial dimension (typically 3)
	/// @tparam N_DATA           Max data indices per leaf node
	/// @tparam T                Floating-point type for bounding boxes
	/// @tparam BUFFER_CAPACITY  Size of the thread buffers for data to be immediately put into
	/// @tparam BUFFER_ON_STACK  When set to true, put the thread buffers on the stack.
	///                             Otherwise, put them on the heap. If BUFFER_CAPACITY is large enough
	///                             so that stack overflow is a concern, then this should be false.
	/// @tparam N_INSERTER_THREADS The number of threads responsible for inserting the data. Must be between
	///                            1 and N_CHILDREN (4 in 2D, 8 in 3D)
	/////////////////////////////////////////////////
	template<
		typename DATA_T,
		bool SINGLE_DATA,
		int DIM=3,
		int N_DATA=16,
		typename T=float>
	class BasicParallelOctree {
		static_assert(DIM==3 or DIM==2, "The octree must be in 2 or 3 dimensions");
		static_assert(N_DATA > 0, "N_DATA must be positive");

	public:
		// Type aliases
		using Node_t  = OctreeParallelNode<DIM,N_DATA,T>;
		using Point_t = Point<DIM,T>;
		using Box_t   = Box<DIM,T>;
		using Data_t  = DATA_T;

		struct Stats {
			size_t n_used_indices=0;
			size_t n_indices_capacity=0;
			size_t n_leafs=0;
			size_t n_nodes=0;
			size_t memory_reserved_bytes=0;
			size_t memory_used_bytes=0;
			int max_depth=0;
		};

		static constexpr int N_CHILDREN = Node_t::N_CHILDREN;
	private:
		// Tree structure
		Node_t* _root = nullptr;
		mutable std::shared_mutex _tree_mutex; //lock when doing large tree changes

		// Data storage
		std::vector<Data_t> _data;
		std::atomic<size_t> _next_data_idx{0};
		MultipleInMultipleOut<size_t> _queue;
	public:
		////////////////////////////////////////////////////////////
		// Construction and destruction
		////////////////////////////////////////////////////////////
		explicit constexpr BasicParallelOctree(const Box_t &bbox) noexcept;
		constexpr BasicParallelOctree() noexcept : BasicParallelOctree(Box_t{}) {}
		virtual ~BasicParallelOctree() noexcept {delete _root;};

		// Non-copyable, non-movable
		BasicParallelOctree(const BasicParallelOctree&) = delete;
		BasicParallelOctree& operator=(const BasicParallelOctree&) = delete;
		BasicParallelOctree(BasicParallelOctree&&) = delete;
		BasicParallelOctree& operator=(BasicParallelOctree&&) = delete;

		////////////////////////////////////////////////////////////
		// Container interface
		////////////////////////////////////////////////////////////
		void reserve(const size_t length) noexcept {
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
			_data.reserve(length);
		}

		void shrink_to_fit() noexcept {
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
			_data.resize(_next_data_idx.load());
			_data.shrink_to_fit();
		}

		bool empty() const noexcept {
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
			return _data.empty();
		}

		size_t size() const noexcept {
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
			return _next_data_idx.load(std::memory_order_acquire);
		}

		size_t capacity() const noexcept {
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
			return _data.capacity();
		}

		void resize(const size_t length) noexcept {
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
			assert(length > _next_data_idx.load(std::memory_order_acquire));

			// Resize the storage container
			_data.resize(length);
		}

		void clear() noexcept {
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);

			_data.clear();
			_next_data_idx.store(0);
			
			// clear the tree structure
			Node_t* new_root = new Node_t(_root->bbox);
			resetDataIdx(new_root);
			delete _root;
			_root = new_root;
		}

		////////////////////////////////////////////////////////////
		// Element access
		////////////////////////////////////////////////////////////
		inline constexpr const Data_t& operator[](const size_t idx) const noexcept {
			assert(idx < size());
			return _data[idx];
		}

		inline constexpr Data_t& operator[](const size_t idx) noexcept {
			assert(idx < size());
			return _data[idx];
		}

		constexpr const Data_t& at(const size_t idx) const {
			if(idx >= size()) {throw std::runtime_error("BasicParallelOctree: index out of range");}
			return _data[idx];
		}

		constexpr Data_t& at(const size_t idx) {
			if(idx >= size()) {throw std::runtime_error("BasicParallelOctree: index out of range");}
			return _data[idx];
		}

		////////////////////////////////////////////////////////////
		// Querries
		////////////////////////////////////////////////////////////
		/// Find all data indices that MAY be associated with the specified box
		/// This gets all data contained in nodes where the node bounding box intersects the
		/// specified bounding box
		std::vector<size_t> get_data_in_box(const Box_t& bbox) const;

		/// Find existing data in tree
		/// Returns index if found, (size_t)-1 if not found
		size_t find(const Data_t& val) const {
			std::shared_lock<std::shared_mutex> tree_lock(_tree_mutex);
			std::shared_lock<std::shared_mutex> lock(_root->mutex);
			const size_t result = _recursive_find_index(_root, val);
			return result;
		}

		////////////////////////////////////////////////////////////
		// Iterators
		////////////////////////////////////////////////////////////
		auto begin()        { return _data.begin(); }
		auto begin()  const { return _data.cbegin();}
		auto cbegin() const { return _data.cbegin();}
		auto end()          { return _data.end();   }
		auto end()    const { return _data.cend();  }
		auto cend()   const { return _data.cend();  }

		////////////////////////////////////////////////////////////
		// Interact with the bounding box
		////////////////////////////////////////////////////////////
		inline const Box_t& bbox() const {return _root->bbox;}

		void resize_to_fit_data(const Data_t& val) {
			Box_t new_bbox = _root->bbox;
			while (!isValid(new_bbox, val)) {new_bbox = T(2)*new_bbox;}
			set_bbox(new_bbox);
		}

		void set_bbox(const Box_t& new_bbox) {
			assert(new_bbox.contains(_root->bbox));
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
			_recursive_expand_bbox(new_bbox);

			// Reinsert data into new nodes if needed
			if constexpr (!SINGLE_DATA) {
				for (size_t j = 0; j < _data.size(); j++) {
					Node_t* start_node = nullptr;
					{
						Node_t* start_node = _recursive_find_best_node(start_node, _data[j]);
					}

					{
						_recursive_insert_data<true>(start_node, _data[j], j);
					}
				}
			}
		}

		
		////////////////////////////////////////////////////////////
		// Insertion operations
		////////////////////////////////////////////////////////////
		
		/// Single thread copy and move
		size_t push_back(const Data_t &val) {
			std::shared_lock<std::shared_mutex> tree_lock(_tree_mutex);
			Data_t copy(val);
			return push_back(std::move(copy));
		}

		/// Add data to end of _data
		size_t push_back(Data_t &&val);

		/// Wait for all pending async insertions to complete
		void flush();

	private:
		/// Determine if data belongs in the given bounding box (must be overridden)
		virtual constexpr bool isValid(const Box_t &bbox, const Data_t &val) const = 0;

		////////////////////////////////////////////////////////////
		// Recursive helper functions
		////////////////////////////////////////////////////////////
		/// Find best node to start insertion/search
		Node_t* _recursive_find_best_node(const Node_t* node, const Data_t &val) const;

		/// Insert data into tree
		template<bool WAIT>
		int _recursive_insert_data(Node_t* node, const Data_t &val, const size_t idx);

		/// Divide a leaf node into children
		void _divide(Node_t* node);

		/// Find index of data in tree
		size_t _recursive_find_index(const Node_t* node, const Data_t &val) const;

		/// Find all data indices in nodes that intersect the specified box
		void _recursive_data_in_box(const Node_t* node, const Box_t& bbox,
			std::vector<size_t>& data_indices) const;

		/// Expand root bounding box to contain new region
		void _recursive_expand_bbox(const Box_t& new_bbox);

		/// Check if there is duplicated data
		void _recursive_duplicate_data(const Node_t* node) const;

		

		/////////////////////////////////////////////////
		/// Convenience and debug methods
		/////////////////////////////////////////////////
		void _recursive_node_properties(const Node_t* node, Stats& stats) const;
	};


	//////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////    METHOD IMPLEMNTATION    /////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////
	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	constexpr BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::BasicParallelOctree(
		const Box<DIM,T>& bbox) noexcept
	{
		//increase bounding box to closest power of 2 to better avoid floating point arithmetic errors
		Point_t low(bbox.low());
		Point_t high(bbox.high());

		for (int i = 0; i < DIM; i++) {
			int n  = 1+static_cast<int>(std::log2(std::fabs(low[i])));
			if (low[i]<0.0) {low[i] = -static_cast<T>(std::pow(2.0, n));}
			else {low[i] = static_cast<T>(std::exp2(n));}
			
			n  = 1+static_cast<int>(std::log2(std::fabs(high[i])));
			if (high[i]<0.0) {high[i] = -static_cast<T>(std::pow(2.0, n));}
			else {high[i] = static_cast<T>(std::exp2(n));}
		}


		//set _root with rounded bounding box
		_root = new Node_t(Box_t{low, high});
		{
			std::lock_guard<std::shared_mutex> lock(_root->mutex);
			resetDataIdx(_root);
		}
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	std::vector<size_t> BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::get_data_in_box(
		const Box_t& bbox) const
	{
		std::shared_lock<std::shared_mutex> tree_lock(_tree_mutex);

		std::vector<size_t> data_indices;
		_recursive_data_in_box(_root, bbox, data_indices);

		//ensure that data_indices is sorted and has no duplicates
		std::sort(data_indices.begin(), data_indices.end());
		auto last = std::unique(data_indices.begin(), data_indices.end());
		data_indices.erase(last, data_indices.end());

		//return number of data indices found
		return data_indices;
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	size_t BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::push_back(DATA_T&& val)
	{
		std::shared_lock<std::shared_mutex> tree_lock(_tree_mutex);
		Node_t* start_node = _recursive_find_best_node(_root, val);

		//try to find the existing data
		size_t idx = (size_t) -1;
		idx = _recursive_find_index(start_node, val);
		if (idx!=(size_t) -1) {return idx;}

		//data must be inserted
		idx = _next_data_idx.fetch_add(1, std::memory_order_acq_rel);
		assert(idx < _data.size());
		assert(_data[idx] == Data_t{});
		_data[idx] = std::move(val);

		//insert the data into the tree
		int flag = 1;
		{
			flag = _recursive_insert_data<false>(start_node, val, idx);
			if (flag == -1)
			{
				size_t idx_copy = idx;
				while(!_queue.try_push(std::move(idx_copy))) {}
				flush(); //change to an inserter thread until the queue is empty
			}

			//TODO change to try_lock and append to queue if it fails
			//then switch this thread to flush()
		}
		return idx;
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	void BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::flush()
	{
		size_t idx = (size_t) -1;
		while (_queue.try_pop(idx)) {
			assert(idx < (size_t) -1);

			//insert data until successful
			int flag = _recursive_insert_data<true>(_root, _data[idx], idx);

			if (flag == -1)
			{
				throw std::runtime_error("BasicParallelOctree: couldn't insert data at index " + std::to_string(idx));
			}
		}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////    RECURSIVE METHOD IMPLEMNTATION    ////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////
	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	OctreeParallelNode<DIM,N_DATA,T>* 
		BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_find_best_node(
			const OctreeParallelNode<DIM,N_DATA,T>* node,
			const Data_t &val) const
	{
		if (!isValid(node->bbox, val))
		{
			return nullptr;
		}
		
		if (isLeaf(node))
		{
			return const_cast<Node_t*>(node);
		}

		// Traverse to child containing data
		for (int c = 0; c < N_CHILDREN; c++) {
			if (isValid(node->children[c]->bbox, val)) {
				return _recursive_find_best_node(node->children[c], val);
			}
		}

		// Couldn't find a valid leaf
		return const_cast<Node_t*>(node);
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	template<bool WAIT>
	int BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_insert_data(
			OctreeParallelNode<DIM,N_DATA,T>* node,
			const Data_t &val,
			const size_t idx)
	{
		//return 1 on success and -1 on fail, return 0 on data already exists
		//debug sanity check
		assert(isValid(node->bbox, val));
		
		if (!isLeaf(node))
		{
			//recurse into child nodes

			// assert(node->data_idx == nullptr);
			int flag=-1; //failure flag

			for (int c=0; c<N_CHILDREN; c++)
			{
				if (isValid(node->children[c]->bbox, val))
				{
					flag = std::max(flag, _recursive_insert_data<WAIT>(node->children[c], val, idx));
					if constexpr (SINGLE_DATA) {break;}
				}
			}

			return flag;
		}
		else
		{
			{
				if constexpr (WAIT)
				{
					std::lock_guard<std::shared_mutex> lock(node->mutex);
					int flag = appendDataIdx(node, idx);
					if (flag>=0) {return flag;}

					_divide(node);
					return _recursive_insert_data<WAIT>(node, val, idx);
				}
				else
				{
					std::unique_lock<std::shared_mutex> lock(node->mutex, std::try_to_lock);
					if (lock.owns_lock())
					{
						int flag = appendDataIdx(node, idx);
						if (flag>=0) {return flag;} //either the data was found or it was successfully added

						_divide(node);

						//retry now that the node is divided
						lock.unlock();
						return _recursive_insert_data<WAIT>(node, val, idx);
					} else
					{
						//failed to insert data
						return -1;
					}
				}
			}
		}
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	void BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_divide(
			OctreeParallelNode<DIM,N_DATA,T>* node)
	{
		//the write lock must already be aquired for this node
		assert(isLeaf(node));

		//create child nodes
		for (int c=0; c<N_CHILDREN; c++) {
			assert(node->children[c] == nullptr);
			node->children[c] = new Node_t(node, c);
		}

		//copy data down
		for (int d=0; d<node->cursor; d++)
		{
			const size_t d_idx = node->data_idx[d];
			const DATA_T& val  = _data[d_idx];
			for (int c=0; c<N_CHILDREN; c++)
			{
				if (isValid(node->children[c]->bbox, val))
				{
					[[maybe_unsused]] int flag = appendDataIdx(node->children[c], d_idx);
					assert(flag==1);
					if constexpr (SINGLE_DATA) {break;}
				}
			}
		}

		// Clear parent node
		node->is_leaf.store(false);
		clearDataIdx(node);
	}


	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	size_t BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::
		_recursive_find_index(
			const OctreeParallelNode<DIM,N_DATA,T>* node,
			const DATA_T &val) const
	{
		if (!isLeaf(node))
		{
			assert(node->data_idx==nullptr);
			for (int c=0; c<N_CHILDREN; c++)
			{
				if (isValid(node->children[c]->bbox, val))
				{
					const size_t result = _recursive_find_index(node->children[c], val);
					if (result<size()) {return result;}
				}
			}
		}
		else 
		{
			std::shared_lock<std::shared_mutex> lock(node->mutex);
			if (node->data_idx==nullptr) {return (size_t) -1;}

			for (int i=0; i<node->cursor; i++)
			{
				const size_t d_idx = node->data_idx[i];
				if (_data[d_idx]==val) {return d_idx;}
			}
		}

		return (size_t) -1;
	}

	/// Expand root bounding box to contain new region
	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	void BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_expand_bbox(const Box<DIM,T>& new_bbox) {
		if (_root->bbox.contains(new_bbox)) {
			return;
		}

		// Double the bounding box and find best placement
		Box_t expanded_root_bbox = T{2} * _root->bbox;
		int max_vertices = -1;
		int best_sibling_number = -1;

		for (int c = 0; c < N_CHILDREN; c++) {
			Point_t offset = _root->bbox.voxelvertex(c) - expanded_root_bbox.voxelvertex(c);
			Box_t test_box = expanded_root_bbox + offset;
			
			int n_verts = 0;
			for (int i = 0; i < N_CHILDREN; i++) {
				if (test_box.contains(new_bbox.voxelvertex(i))) {
					n_verts++;
				}
			}

			if (n_verts > max_vertices) {
				best_sibling_number = c;
				max_vertices = n_verts;
			}
		}

		// Create new root
		Point_t offset = _root->bbox.voxelvertex(best_sibling_number) 
		               - expanded_root_bbox.voxelvertex(best_sibling_number);
		Box_t new_root_bbox = expanded_root_bbox + offset;

		Node_t* old_root = _root;
		_root = new Node_t(new_root_bbox, old_root->depth - 1);
		_divide(_root);
		
		delete _root->children[best_sibling_number];
		_root->children[best_sibling_number] = old_root;
		old_root->parent = _root;

		assert(_root->bbox.voxelvertex(best_sibling_number) == 
		       old_root->bbox.voxelvertex(best_sibling_number));
		
		// Recursively expand if needed
		if (max_vertices < N_CHILDREN) {
			_recursive_expand_bbox(new_bbox);
		}
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	void BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_data_in_box(
		const OctreeParallelNode<DIM,N_DATA,T>* node,
		const Box<DIM,T>& bbox,
		std::vector<size_t>& data_indices) const
	{
		if (node==nullptr) {return;}
		if(!bbox.intersects(node->bbox)) {return;}

		if (node->data_idx)
		{
			std::shared_lock<std::shared_mutex> lock(node->mutex);
			for (int d=0; d<node->cursor; d++) {
				const size_t d_idx = node->data_idx[d]; 
				if (isValid(bbox, _data[d_idx])) {data_indices.push_back(d_idx);}
			}
		}
		

		//recurse into children
		for (int c = 0; c < N_CHILDREN; c++) {
			_recursive_data_in_box(node->children[c], bbox, data_indices);
		}

		//note that data_indices may contain duplicate entries at this point,
		//but all will be valid data
	}
}