#pragma once

#include "utility/utility.hpp"
#include "geometry/geometry.hpp"

#include "memory/thread_pool.hpp"

#include "octree/base_node.hpp"
#include "octree/node_policies.hpp"


#ifndef GUTIL_N_OCTREE_THREADS
	#define GUTIL_N_OCTREE_THREADS 4
#endif

#ifndef GUTIL_OCTREE_SPAWN_THREAD_THRESHOLD
	#define GUTIL_OCTREE_SPAWN_THREAD_THRESHOLD 512
#endif


namespace gutil {


	/////////////////////////////////////////////////////////////////
	/// A base octree class. Nodes only store an index to the data.
	/// Data is stored contiguously in a vector. A node 'owns' data if:
	/// 	a) the data intersects the node bounding box AND EITHER
	///		b1) the node is a leaf OR
	///		b2) the data intersects more than one bounding box of a child node.
	/////////////////////////////////////////////////////////////////

	template<typename Derived, typename Opts>
	struct BaseOctree {
		

		////////////////////////////////////////////////////////////////
		// Constants and aliases
		////////////////////////////////////////////////////////////////
		using value_type    = typename Opts::value_type;	//type that this octree is storing
		using point_type  	= typename Opts::point_type;	//type of spatial points
		using box_type	  	= typename Opts::box_type;		//type of spatial axis-aligned-bounding-boxes
		using scalar_type 	= typename Opts::scalar_type;	//type that emulates real numbers for the spatial points and aabb
		using node_type     = typename Opts::node_type;
		using key_type      = typename node_type::key_type;

		static_assert( std::same_as< typename node_type::value_type, size_t>,
			"BaseOctree - the nodes must hold index values, not data values");
		
		static constexpr int N_CHILDREN = node_type::N_CHILDREN;
		static constexpr bool HAS_DISTANCE_SQ = Opts::HAS_DISTANCE_SQ;
		static constexpr bool VOLUME_DATA = Opts::VOLUME_DATA;

	protected:
		////////////////////////////////////////////////////////////////
		/// Data, tree, thread resources
		////////////////////////////////////////////////////////////////
		std::vector<value_type> data_{};
		
		node_type* root_{nullptr};
		typename node_type::allocator_type root_alloc_{};
		typename node_type::data_allocator_type index_alloc_{};

		ThreadPool threads_{GUTIL_N_OCTREE_THREADS};
		static constexpr size_t SPAWN_THREAD_THRESHOLD = GUTIL_OCTREE_SPAWN_THREAD_THRESHOLD;


		////////////////////////////////////////////////////////////////
		/// Helper functions to work with the node allocator
		////////////////////////////////////////////////////////////////
		void destroy_root() noexcept {
			if (root_) {
				using traits = typename node_type::alloc_traits;
				traits::destroy(root_alloc_, root_);
				traits::deallocate(root_alloc_, root_, 1);
				root_ = nullptr;
			}
		}

		node_type* construct_root(const box_type& box) {
			assert(root_==nullptr);
			return node_type::ConstructRoot(box, root_alloc_, index_alloc_);
		}


		////////////////////////////////////////////////////////////////
		/// Helper functions from the Derived class
		////////////////////////////////////////////////////////////////
		[[nodiscard]] scalar_type distance_sq(const point_type& pt, const value_type& value) const noexcept requires(Opts::HAS_DISTANCE_SQ) {
			return static_cast<const Derived*>(this) -> distance_sq_impl(value, pt);
		}

		[[nodiscard]] bool intersects(const box_type& box, const value_type& value) const noexcept {
			return static_cast<const Derived*>(this) -> intersects_impl(box, value);
		}

		[[nodiscard]] point_type get_point(const value_type& value)  const noexcept {
			return static_cast<const Derived*>(this) -> get_point_impl(value);
		}


	public:
		////////////////////////////////////////////////////////////////
		/// Constructor, movement
		////////////////////////////////////////////////////////////////
		BaseOctree() = default;
		BaseOctree(const BaseOctree&) = delete;
		BaseOctree& operator=(const BaseOctree&) = delete;
		BaseOctree(BaseOctree&& other) : 
			data_{std::move(other.data_)},
			root_{std::move(other.root_)} { other.root_ = nullptr; }
		BaseOctree& operator=(BaseOctree&& other) {
			if (this != &other) {
				data_.clear();
				data_ = std::move(other.data_);
				destroy_root();
				root_ = other.root_;
				other.root_ = nullptr;
			}
			return *this;
		}

		/// Set the bounding box
		template<typename... Args>
		BaseOctree(Args&&...  args) : 
			root_{nullptr} {
				root_ = construct_root(box_type{std::forward<Args>(args)...});
			}


		////////////////////////////////////////////////////////////////
		/// Public interface to access the data
		////////////////////////////////////////////////////////////////
		[[nodiscard]] std::span<value_type> data() noexcept { return {data_.begin(), data_.end()}; }
		[[nodiscard]] std::span<const value_type> data() const noexcept { return {data_.cbegin(), data_.cend()}; }

		[[nodiscard]] auto begin() {return data_.begin();}
		[[nodiscard]] auto begin() const {return data_.cbegin();}
		[[nodiscard]] auto cbegin() const {return data_.cbegin();}

		[[nodiscard]] auto end() {return data_.end();}
		[[nodiscard]] auto end() const {return data_.cend();}
		[[nodiscard]] auto cend() const {return data_.cend();}

		[[nodiscard]] auto rbegin() {return data_.rbegin();}
		[[nodiscard]] auto rbegin() const {return data_.crbegin();}
		[[nodiscard]] auto crbegin() const {return data_.crbegin();}

		[[nodiscard]] auto rend() {return data_.rend();}
		[[nodiscard]] auto rend() const {return data_.crend();}
		[[nodiscard]] auto crend() const {return data_.crend();}

		[[nodiscard]] const value_type& operator[](const size_t idx) const noexcept {
			assert(idx<data_.size());
			return data_[idx];
		}

		[[nodiscard]] value_type& operator[](const size_t idx) noexcept {
			assert(idx<data_.size());
			return data_[idx];
		}

		[[nodiscard]] const value_type& at(const size_t idx) const {
			return data_.at(idx);
		}

		[[nodiscard]] value_type& at(const size_t idx) {
			return data_.at(idx);
		}

		[[nodiscard]] size_t size() const noexcept { return data_.size(); }
		[[nodiscard]] size_t capacity() const noexcept { return data_.capacity(); }
		void reserve(size_t sz) noexcept { data_.reserve(sz); }


		////////////////////////////////////////////////////////////////
		/// Public interface to add data to the tree or query the tree
		////////////////////////////////////////////////////////////////
		template<typename... Args>
		size_t emplace_back(Args&&... args) noexcept;

		size_t push_back(value_type value) noexcept {return 0;};
		
		void push_back_range(std::span<value_type> values) noexcept {
			size_t idx_start = data_.size();
			data_.insert( 	data_.end(),
							std::make_move_iterator(values.begin()),
							std::make_move_iterator(values.end()));

			std::span<value_type> moved_vals{data_.begin()+idx_start, data_.end()};
			recursive_sort_and_insert(root_, idx_start, moved_vals);
		}
		
		[[nodiscard]] size_t find(const value_type& value) const noexcept;
		
		[[nodiscard]] size_t find_nearest(const point_type& point) const noexcept requires(Opts::HAS_DISTANCE_SQ) {
			if (data_.empty()) {return size_t(-1); }

			size_t idx=0;
			scalar_type d2 = distance_sq(point, data_[idx]);
			recursive_find_nearest(root_, point, d2, idx);
			return idx;
		}

		void sort_and_deduplicate(std::span<value_type> values) noexcept;

	private:
		////////////////////////////////////////////////////////////////
		/// Helper functions
		////////////////////////////////////////////////////////////////
		void recursive_sort_and_insert(node_type* node, size_t idx_start, std::span<value_type> values) noexcept;

		void recursive_find_nearest(const node_type* node, const point_type& point, scalar_type& dist_sq, size_t& idx) const noexcept requires (Opts::HAS_DISTANCE_SQ);

		[[nodiscard]] int get_child_node(const node_type* node, const value_type& value) const noexcept {
			if constexpr (VOLUME_DATA) {
				return node_type::GetChildNumberVolume( value, node, 
						[this](const box_type& b, const value_type& val) { return this->intersects(b,val); });
			}
			else {
				return node_type::GetOrthant( node->bbox.center(), get_point(value) );
			}
		}

		node_type* find_node(const value_type& value) const noexcept {
			if (root_ ==  nullptr) {return nullptr;}
			
			node_type* node = root_;
			while(!node->is_leaf()) {
				assert( intersects(node->bbox, value) );

				const int child = get_child_node(node, value);

				if constexpr (VOLUME_DATA) {
					if (child == N_CHILDREN) { return node; }
				}

				node = node->children + child;	//child nodes are stored contiguously
			}

			return node;
		}

		node_type* find_node(const key_type key) const noexcept {
			return node_type::TraverseToNode(root_, key.morton_index());
		}




	};


	////////////////////////////////////////////////////////////////
	/// Implementation of BaseOctree methods
	////////////////////////////////////////////////////////////////
	template<typename D, typename O>
	void BaseOctree<D,O>::recursive_sort_and_insert(node_type* node, size_t idx_start, std::span<value_type> values ) noexcept {
		//sort data to nodes that it would belong to
		//if the node is does not have enough room to accept data, split and push down its existing data
		//insert indices into nodes that are idx_start + i to match values[i]
		//we assume that the values are already de-duplicated with the existing data or that deduplication
		//is unnecessary.
		//the values span should be a span to the end of the data_ vector

		if (node->is_leaf()) {
			if (node->data.remaining() < values.size()) {
				node->split();
				if constexpr (VOLUME_DATA) {
					node->push_down_volume( [this](const box_type& b, size_t idx) { return this->intersects(b, data_[idx]); });
				}
				else {
					node->push_down_point( [this](size_t idx) { return this->get_point(data_[idx]); });
				}

				//call again to recurse into standard branch
				recursive_sort_and_insert(node, idx_start, values);
			}
			else {
				for (size_t i=0; i<values.size(); ++i) {
					node->data.push_back(idx_start + i);
				}
			}
		}
		else {
			BinSort<value_type> sorted;
			if constexpr (VOLUME_DATA) {
				sorted = node_type::PartitionToOrthantAndNode(values, node->bbox, 
					[this](const value_type& v, const box_type& b) { return this->intersects(v,b); });

				//insert into current node
				assert( node->data.remaining() >= sorted.bin_size(N_CHILDREN) );
				const size_t idx_offset = idx_start + sorted.bin_start(N_CHILDREN);	//values that stay are sorted to the end
				auto list = sorted.get_bin(N_CHILDREN);
				for (size_t i=0; i<list.size(); ++i) {
					node->data.push_back(idx_offset + i);
				}
			}
			else {
				sorted = node_type::PartitionToOrthant(values, node->bbox.center(), 
					[this](const value_type& v) { return this->get_point(v); });
			}

			//recurse
			for (int c=0; c<N_CHILDREN; ++c) {
				recursive_sort_and_insert(node->children + c, idx_start + sorted.bin_start(c), sorted.get_bin(c));
			}
		}
	}

	template<typename D, typename O>
	void BaseOctree<D,O>::recursive_find_nearest(const node_type* node, const point_type& point, scalar_type& dist_sq, size_t& idx) const noexcept requires (O::HAS_DISTANCE_SQ) {
		assert(node);
		if constexpr (VOLUME_DATA) {
			//if using signed distance or setting distance to 0 when inside a volume,
			//there is no need to continue.
			if (dist_sq <= scalar_type{0}) { return; }
		}

		if ( node->is_leaf() ) {
			for (const size_t i : node->data) {
				const scalar_type d2 = distance_sq(point, data_[i]);
				if ( d2 < dist_sq ) {
					dist_sq = d2;
					idx = i;
				}
			}
		}
		else {
			if constexpr (VOLUME_DATA) {
				if ( !node->data.empty() ) {
					for (const size_t i : node->data) {
						const scalar_type d2 = distance_sq(point, data_[i]);
						if (d2 < dist_sq) {
							dist_sq = d2;
							idx = i;
						}
					}
				}
			}

			//sort children to recurse into closest first
			std::array<std::pair<scalar_type, int>, N_CHILDREN> children_dist_pair;
			size_t i=0;
			for (int c=0; c<N_CHILDREN; ++c) {
				//note distance_squared(box, point) is a free standing function in gutilmath.hpp
				const scalar_type d2 = gutil::distance_squared(node->children[c].bbox, point);
				children_dist_pair[i++] = {d2, c};
			}
			std::sort(children_dist_pair.begin(), children_dist_pair.end()); //sorts lexigraphically, distance is first

			//descend into viable children
			for (auto [d2, c] : children_dist_pair) {
				if (d2 < dist_sq) {
					recursive_find_nearest(node->children + c, point, dist_sq, idx);
				}
			}
		}
	}



	template<typename D, typename O>
	[[nodiscard]] size_t BaseOctree<D,O>::find(const value_type& value) const noexcept {
		node_type* node = find_node(value);
		if (node) {
			return node->data.find( [this, &value](size_t idx) { return data_[idx]==value; } );
		}
		return size_t(-1);
	}




}
