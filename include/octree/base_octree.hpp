#pragma once

#include "utility/utility.hpp"
#include "geometry/geometry.hpp"

#include "memory/thread_pool.hpp"

#include "octree/base_node.hpp"
#include "octree/node_policies.hpp"

#include <concepts>

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

		struct DummyResource {};
		
		using node_alloc_type = typename node_type::node_alloc_type;
		using node_resource_type = std::conditional_t< std::same_as<typename Opts::node_resource_type,void>, DummyResource, typename Opts::node_resource_type >;
		[[no_unique_address]] node_resource_type node_resource_{};
		node_alloc_type node_alloc_{};


		using index_alloc_type = typename node_type::data_alloc_type;
		using index_resource_type = std::conditional_t< std::same_as<typename Opts::index_resource_type,void>, DummyResource, typename Opts::index_resource_type >;
		[[no_unique_address]] index_resource_type index_resource_{};
		index_alloc_type index_alloc_{};


		ThreadPool threads_{GUTIL_N_OCTREE_THREADS};
		static constexpr size_t SPAWN_THREAD_THRESHOLD = GUTIL_OCTREE_SPAWN_THREAD_THRESHOLD;


		////////////////////////////////////////////////////////////////
		/// Helper functions to work with the node allocator
		////////////////////////////////////////////////////////////////
		void destroy_root() noexcept {
			if (root_) {
				using traits = typename node_type::alloc_traits;
				traits::destroy(node_alloc_, root_);
				traits::deallocate(node_alloc_, root_, 1);
				root_ = nullptr;
			}
		}

		void construct_root(const box_type& box) noexcept {
			if(root_!=nullptr) { destroy_root(); }
			root_ =  node_type::ConstructRoot(box, node_alloc_, index_alloc_);
		}

		[[nodiscard]] node_alloc_type make_node_allocator() noexcept {
			if constexpr (std::same_as<DummyResource,node_resource_type>) {
				return node_alloc_type{};
			}
			else {
				return node_alloc_type(&node_resource_);
			}
		}

		[[nodiscard]] index_alloc_type make_index_allocator() noexcept {
			if constexpr (std::same_as<DummyResource,index_resource_type>) {
				return index_alloc_type{};
			}
			else {
				return index_alloc_type(&index_resource_);
			}
		}


		////////////////////////////////////////////////////////////////
		/// Helper functions from the Derived class
		////////////////////////////////////////////////////////////////
		[[nodiscard]] scalar_type distance_sq(const value_type& value, const point_type& pt) const noexcept requires(Opts::HAS_DISTANCE_SQ) {
			return static_cast<const Derived*>(this) -> distance_sq_impl(value, pt);
		}

		[[nodiscard]] bool intersects(const box_type& box, const value_type& value) const noexcept {
			return static_cast<const Derived*>(this) -> intersects_impl(box, value);
		}

		[[nodiscard]] bool intersects(const value_type& A, const value_type& B) const noexcept requires(Opts::VOLUME_DATA) {
			return static_cast<const Derived*>(this) -> intersects_impl(A,B);
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
			root_{nullptr},
			node_resource_{},
			node_alloc_(make_node_allocator()),
			index_resource_{},
			index_alloc_(make_index_allocator()) {
				construct_root(box_type{std::forward<Args>(args)...});
			}


		////////////////////////////////////////////////////////////////
		/// Public interface to access the data
		////////////////////////////////////////////////////////////////
		[[nodiscard]] std::span<value_type> as_span() noexcept { return {data_.begin(), data_.end()}; }
		[[nodiscard]] std::span<const value_type> as_span() const noexcept { return {data_.cbegin(), data_.cend()}; }
		[[nodiscard]] std::span<const value_type> as_cspan() const noexcept { return {data_.cbegin(), data_.cend()}; }

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

		size_t push_back(value_type value) noexcept {
			size_t idx = data_.size();
			push_back_range(std::span<value_type>{&value, 1});
			return idx;
		};
		
		void push_back_range(std::vector<value_type>&& values) noexcept {
			push_back_range(std::span<value_type>{values.begin(), values.end()});
		}

		void push_back_range(std::span<value_type> values) noexcept {
			size_t idx_start = data_.size();
			data_.insert( 	data_.end(),
							std::make_move_iterator(values.begin()),
							std::make_move_iterator(values.end()));

			std::span<value_type> moved_vals{data_.begin()+idx_start, data_.end()};

			if constexpr (GUTIL_N_OCTREE_THREADS>0) {
				threads_.submit([&]() { recursive_sort_and_insert(root_, idx_start, moved_vals); });
				threads_.wait_idle();	
			}
			else {
				recursive_sort_and_insert(root_, idx_start, moved_vals);
			}
		}
		

		[[nodiscard]] const box_type& bbox() const noexcept { return root_->bbox; }

		[[nodiscard]] size_t find(const value_type& value) const noexcept;
		
		[[nodiscard]] size_t find_nearest(const point_type& point) const noexcept requires(Opts::HAS_DISTANCE_SQ) {
			if (data_.empty()) { return size_t(-1); }

			size_t idx=0;
			scalar_type d2 = distance_sq(data_[idx], point);
			recursive_find_nearest(root_, point, d2, idx);
			return idx;
		}

		[[nodiscard]] size_t collides_with(const value_type& value) const noexcept requires(Opts::VOLUME_DATA) {
			if (data_.empty()) { return size_t(-1); }

			const node_type* node = find_node(value);
			const size_t* idx = node->data.find( [&](size_t i) { return this->intersects(value, data_[i]); });
			return idx ? *idx : size_t(-1);
		}



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
				GUTIL_ASSERT( intersects(node->bbox, value) );

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
					GUTIL_ASSERT(intersects(node->bbox, data_[idx_start+i]));
					node->data.push_back(idx_start + i);
				}
			}
		}
		else {
			BinSort<value_type> sorted;
			if constexpr (VOLUME_DATA) {
				sorted = node_type::PartitionToOrthantAndNode(values, node->bbox, 
					[this](const box_type& b, const value_type& v) { return this->intersects(b,v); });

				//insert into current node
				GUTIL_ASSERT( node->data.remaining() >= sorted.bin_size(N_CHILDREN) );
				const size_t idx_offset = idx_start + sorted.bin_start(N_CHILDREN);	//values that stay are sorted to the end
				auto list = sorted.get_bin(N_CHILDREN);
				for (size_t i=0; i<list.size(); ++i) {
					node->data.push_back(idx_offset + i);
				}
			}
			else {
				GUTIL_ASSERT(node->data.empty());
				sorted = node_type::PartitionToOrthant(values, node->bbox.center(), 
					[this](const value_type& v) { return this->get_point(v); });
			}

			//recurse
			for (int c=0; c<N_CHILDREN; ++c) {
				if constexpr (GUTIL_N_OCTREE_THREADS>0) {
					const bool spawn_thread = (c==N_CHILDREN-1) ? false : sorted.bin_size(c) >= SPAWN_THREAD_THRESHOLD;

					if (spawn_thread) {
						node_type* child = node->children + c;
						auto list = sorted.get_bin(c);
						size_t start = idx_start + sorted.bin_start(c);

						auto fun = [this,child,list,start]() {this->recursive_sort_and_insert(child, start, list);};
						threads_.submit(fun);
					}
					else {
						recursive_sort_and_insert(node->children + c, idx_start + sorted.bin_start(c), sorted.get_bin(c));
					}
				}
				else {
					recursive_sort_and_insert(node->children + c, idx_start + sorted.bin_start(c), sorted.get_bin(c));
				}
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
				const scalar_type d2 = distance_sq(data_[i], point);
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
						const scalar_type d2 = distance_sq(data_[i], point);
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
			size_t* idx = node->data.find( [this, &value](size_t idx) { return data_[idx]==value; } );
			return idx==nullptr ? size_t(-1) : *idx;
		}
		return size_t(-1);
	}




}
