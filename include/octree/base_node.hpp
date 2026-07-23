#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"

#include "memory/containers.hpp"
#include "algorithms/sorting.hpp"

#include "octree/index_key.hpp"
#include "octree/node_policies.hpp"

#include <memory>
#include <array>
#include <concepts>
#include <cstdint>
#include <bit>
#include <memory_resource>

namespace gutil {
	/////////////////////////////////////////////////////////////////
	/// Define the base node type. Generally ValueType is an index.
	/////////////////////////////////////////////////////////////////
	template<typename ValueType, int DIM, IsScalar T, int MaxData=64, typename NodeAllocator=void, typename DataAllocator=void>
	struct alignas(64) Node {


		/////////////////////////////////////////////////////////////////
		/// Define types and constants.
		///
		/// To specify the allocator type, use a dummy type std::byte.
		/// This dummy type will be rebound to Node. Note that the root node will
		/// need an instantiated allocator passed to it to use when creating the children.
		///
		/// A similar process is used for the data allocator. To use the standard allocator,
		/// you may leave the default 'void' option.
		/////////////////////////////////////////////////////////////////
		using seed_node_alloc_type = std::conditional_t< std::same_as<NodeAllocator,void>, std::allocator<std::byte>, NodeAllocator>;
		using node_alloc_type = typename std::allocator_traits<seed_node_alloc_type>::template rebind_alloc<Node>;
		using alloc_traits   = std::allocator_traits<node_alloc_type>;

		// using allocator_type = std::conditional_t< std::same_as<NodeAllocator,void>, std::allocator<Node>, NodeAllocator>;
		// using alloc_traits   = std::allocator_traits<allocator_type>;

		using seed_data_alloc_type = std::conditional_t< std::same_as<DataAllocator,void>, std::allocator<std::byte>, DataAllocator>;
		using data_alloc_type = typename std::allocator_traits<seed_data_alloc_type>::template rebind_alloc<ValueType>;

		using box_type       = Box<DIM,T>;
		using point_type     = Point<DIM,T>;
		using scalar_type    = T;
		using key_type       = IndexKey<DIM>;
		using container_type = FixedArray<ValueType,MaxData,data_alloc_type>;
		using value_type     = ValueType;

		static constexpr int MAX_DEPTH 	  = static_cast<int>(key_type::MAX_DEPTH);
		static constexpr int MAX_DATA     = MaxData;
		static constexpr int DIMENSION    = DIM;
		static constexpr int N_ORTHANTS   = int{1} << DIMENSION;
		static constexpr int ORTHANT_MASK = N_ORTHANTS - 1;
		static constexpr int N_CHILDREN   = N_ORTHANTS;


		/////////////////////////////////////////////////////////////////
		/// Define quantities that must be stored for all node types.
		/// The tag of (and when using a good allcator the pointer to) this
		/// node are stored in the extra bits of the key. 
		/////////////////////////////////////////////////////////////////
		key_type key = key_type::Root();
		box_type bbox{};

		Node* children{nullptr};
		container_type data{};

		node_alloc_type alloc_{};

		
		/////////////////////////////////////////////////////////////////
		/// Memory management. Use new/delete for now.
		/////////////////////////////////////////////////////////////////
		Node() {}
		Node(const Node&) = delete;
		Node& operator=(const Node&) = delete;
		
		Node(Node&& other) noexcept :
			key{std::move(other.key)},
			bbox{std::move(other.bbox)},
			children{other.children},
			data{std::move(other.data)},
			alloc_{std::move(other.alloc_)} {
				other.children = nullptr;
			}

		Node& operator=(Node&& other) noexcept {
			if (this!=&other) {
				key = other.key;
				bbox = other.bbox;
				if (children) { destroy_children(); }
				children = other.children;
				other.children = nullptr;

				data = std::move(other.data); //releases data
			}
			return *this;
		}

		~Node() noexcept { destroy_children(); }

		/// Initialize the root node with constructors
		Node(const box_type& box, node_alloc_type& n_alloc, data_alloc_type& d_alloc) :
			key{key_type::Root()},
			bbox{box},
			data{d_alloc},
			alloc_{n_alloc} {}


		/// Initialize a node by splitting its parent
		Node(Node* parent, int c) : 
			key{parent->key.child(c)}, 
			bbox{parent->bbox.center(), parent->bbox.vertex(c)},
			data{parent->data.alloc_},
			alloc_{parent->alloc_}
			{}

		/// Destroy and deallocate child nodes
		void destroy_children() noexcept {
			if (children) {
				for (int i=0; i<N_CHILDREN; ++i) {
					alloc_traits::destroy(alloc_, children+i);
				}
				alloc_traits::deallocate(alloc_, children, N_CHILDREN);
				children = nullptr;
			}
		}

		/// Allocate and construct child nodes
		void construct_children() noexcept {
			if (!children) {
				children = alloc_traits::allocate(alloc_, N_CHILDREN);

				for (int i=0; i<N_CHILDREN; ++i) {
					alloc_traits::construct(alloc_, children+i, this, i);
				}
			}
		}

		/// Factory method to contruct the root node
		[[nodiscard]] static Node* ConstructRoot(const box_type& box, node_alloc_type& n_alloc, data_alloc_type& d_alloc) noexcept {
			Node* node = alloc_traits::allocate(n_alloc, 1);
			alloc_traits::construct(n_alloc, node, box, n_alloc, d_alloc);
			return node;
		}

		/////////////////////////////////////////////////////////////////
		/// Node splitting.
		/////////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr bool is_leaf() const noexcept {
			return children == nullptr;
		}

		//prepare child nodes without moving data
		void split() noexcept {
			GUTIL_ASSERT(is_leaf());
			construct_children();
		}

		//push down data that belongs to child nodes
		template<typename IntersectsQuery>
		void push_down_volume(IntersectsQuery&& intersect) noexcept {
			GUTIL_ASSERT(!is_leaf());

			auto sorted = Node::PartitionToOrthantAndNode(data.as_span(), bbox, std::forward<IntersectsQuery>(intersect));
			for (int c=0; c<N_CHILDREN; ++c) {
				children[c].data.push_back_range(sorted.get_bin(c));
			}

			auto keep = sorted.get_bin(N_CHILDREN);
			if (keep.size() == 0) { data.release(); }
			else {
				container_type keep_data;
				keep_data.alloc_ = data.alloc_;
				keep_data.push_back_range(keep);
				data = std::move(keep_data);
			}
		}

		template<typename PointQuery>
		void push_down_point(PointQuery&& point) noexcept {
			GUTIL_ASSERT(!is_leaf());

			auto sorted = Node::PartitionToOrthant(data.as_span(), bbox.center(), std::forward<PointQuery>(point));
			for (int c=0; c<N_CHILDREN; ++c) {
				children[c].data.push_back_range(sorted.get_bin(c));
			}
			data.release();
		}


		/////////////////////////////////////////////////////////////////
		/// Static queries for consistancy
		/////////////////////////////////////////////////////////////////
		/// Get a specific bit of the orthant/child belongs to. one corresponds to the positive side.
		/// axis 0 is the first bit, axis 1 is the second and so on. ties go to the lower side.
		[[nodiscard]] static constexpr bool GetOrthantBit(scalar_type center, scalar_type query) noexcept {
			return query > center;
		}

		/// Get the orthant number that a point belongs to relative to a center
		[[nodiscard]] static constexpr int GetOrthant(const point_type& center, const point_type& query) noexcept {
			int n=0;
			for (int i=0; i<DIMENSION; ++i) {
				if (GetOrthantBit(center.data[i], query.data[i])) {n |= (int{1} << i);} 
			}
			return n;
		}

		/// Given a node and data, determine which child it belongs to using either point or volume criteria
		template<typename VT, typename IntersectsQuery>
		[[nodiscard]] static constexpr int GetChildNumberVolume(const VT& value, const Node* node, IntersectsQuery&& query) noexcept {
			GUTIL_ASSERT(query(node->bbox, value));

			int n_children = 0;
			int child=-1;
			for (int c=0; c<N_CHILDREN; ++c) {
				if ( query(node->children[c].bbox, value)) {
					child = c;
					++n_children;
				}
				if (n_children > 1) { return N_CHILDREN; }	//intersects to multiple children, volume data belongs to original node
			}
			GUTIL_ASSERT( n_children==1 );
			GUTIL_ASSERT( child!=-1 );
			return child;
		}

		/// Sort data in-place by the orthant that they belong to in increasing orthant number.
		/// Return the sorting class with state to extract spans to each data set.
		/// If the data type is not the same as point_type, then a method to get the point type
		/// of the data (e.g., its center or a member that stores its spatial location) must
		/// be provided.
		template<typename PointQuery, typename VT>
		[[nodiscard]] static BinSort<VT> PartitionToOrthant(std::span<VT> data, const point_type& center, PointQuery&& query) noexcept {
			BinSort<VT> sorter(data, N_ORTHANTS);
			sorter.sort( [&query, &center](const VT& value) { return GetOrthant(center, query(value)); } );
			return sorter;
		}

		/// Sort volume data into N_ORTHANTS+1 bins:
		/// The first N_ORTHANTS bins correspond to child nodes while the last bin corresponds to the current node
		/// The original node gets a value if more than one child node intersects the data. otherwise the unique child gets that data
		template<typename IntersectsQuery, typename VT>
		[[nodiscard]] static BinSort<VT> PartitionToOrthantAndNode(std::span<VT> data, const box_type& box, IntersectsQuery&& query) noexcept {
			std::array<box_type,N_ORTHANTS> boxes;
			for (int c=0; c<N_CHILDREN; ++c) { boxes[c] = box_type{box.center(), box.vertex(c)}; }

			auto bin_query = [&](const VT& value) {
				int count=0;
				int child=-1;
				for (int c=0; c<N_CHILDREN; ++c) {
					if ( query(boxes[c], value) ) {
						child=c;
						++count;
					}
					if (count > 1) {return N_CHILDREN;}
				}

				GUTIL_ASSERT(count==1);
				GUTIL_ASSERT(child!=-1);
				return child;
			};

			BinSort<VT> sorter(data, N_ORTHANTS+1);
			sorter.sort( std::move(bin_query) );
			return sorter;
		}


		[[nodiscard]] static BinSort<point_type> PartitionToOrthant(std::span<point_type> data, const point_type& center) noexcept {
			BinSort<point_type> sorter(data, N_ORTHANTS);
			sorter.sort( [&center](const point_type& value) { return GetOrthant(center, value); } );
			return sorter;
		}

		/// Get the bounding box of a child node
		[[nodiscard]] static constexpr box_type ChildBox(const box_type& parent_box, int orthant) noexcept {
			GUTIL_ASSERT(0<=orthant && orthant<N_ORTHANTS);
			return {parent_box.center(), parent_box.vertex(orthant)};
		}

		/// Get the bounding box of the parent node
		/// Note the child number is the child number of the node relative to the parent
		[[nodiscard]] static constexpr box_type ParentBox(const box_type& child_box, int child_number) noexcept {
			GUTIL_ASSERT(0<=child_number && child_number<N_ORTHANTS);
			point_type vertex = child_box.vertex(child_number);			//vertex of the parent box, vertex opposite of this is the

			const int center_index = (~child_number) & ORTHANT_MASK;	//flip each axis from high->low and vice versa
			point_type center = child_box.vertex(center_index);

			//vertex to center is center - vertex
			//vertex to opposite is vertex + 2*(vertex to center)
			//simplifies to v + 2*(c-v) = 2*c-v
			return {vertex, scalar_type{2}*center - vertex};
		}

		/// Get the bounding box of this node given the root box
		[[nodiscard]] static constexpr box_type NodeBox(const box_type& root_box, key_type node_key) noexcept {
			return node_key.compute_box(root_box);
		}

		[[nodiscard]] static constexpr point_type NodeCenter(const box_type& root_box, key_type node_key) noexcept {
			return node_key.compute_center(root_box);
		}

		/// Given a key and the root node, traverse to the requested node
		/// If the requested node is unreachable, return the last node on its path
		[[nodiscard]] static constexpr Node* TraverseToNode(Node* root, uint64_t morton_path) noexcept {
			GUTIL_ASSERT(root);
			GUTIL_DEBUG( key_type key = root->key; )

			//the morton path has a leading 1 can be interpreted as a base-2^DIM number
			const uint64_t depth = (std::bit_width(morton_path)-1) / static_cast<uint64_t>(DIMENSION);
			uint64_t mask = (uint64_t{1} << DIMENSION) - 1;

			for (uint64_t d=0; d<depth; ++d) {
				if (!root->children) { return root; }
				const uint64_t shift = (depth - 1 - d) * static_cast<uint64_t>(DIMENSION);
				int c = static_cast<int>( (morton_path>>shift) & mask);
				root = root->children + c;

				GUTIL_DEBUG( key = key.child(c); )
				GUTIL_DEBUG( GUTIL_ASSERT(key== root->key); )
			}

			return root;
		}
	};
}