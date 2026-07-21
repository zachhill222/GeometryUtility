#pragma once

#include "utility/utility.hpp"
#include "geometry/geometry.hpp"



#include "algorithms/sorting.hpp"

#include "octree/node_policies.hpp"

#include <span>
#include <type_traits>
#include <cassert>
#include <concepts>

#ifndef GUTIL_ORTHTREE_MAX_DEPTH
	#define GUTIL_ORTHTREE_MAX_DEPTH 16
#endif


namespace gutil {


	template<typename ContainerType, bool IS_VOLUME=false>
	struct DataPolicyOpts {
		static constexpr bool ONLY_LEAF_HAS_DATA = !IS_VOLUME; //volume data still belongs to interior nodes
		static constexpr bool VOLUME_DATA = IS_VOLUME;
		using container_type = ContainerType;
	};


	/////////////////////////////////////////////////////////////////
	/// The Orthtree node type. Note that the policies only hold an
	/// allocator view, which is just a pointer to the allocator.
	/////////////////////////////////////////////////////////////////
	template<typename ValueType, IsPoint PointType, typename DataOpts, typename NodeAllocPolicy>
	struct Node {


		/////////////////////////////////////////////////////////////
		/// Get types and do a sanity check.
		/////////////////////////////////////////////////////////////
		static_assert(IsAllocatable<NodeAllocPolicy,Node>, "Node - incompatible node allocator");

		static constexpr int DIMENSION    = PointType::DIMENSION; 
		static constexpr int N_ORTHANTS   = int{1} << DIMENSION;
		static constexpr int ORTHANT_MASK = N_ORTHANTS - 1;
		static constexpr int N_CHILDREN   = N_ORTHANTS;

		static constexpr int MAX_DEPTH = GUTIL_ORTHTREE_MAX_DEPTH;

		using value_type = ValueType;
		using point_type = PointType;
		using scalar_type = typename PointType::scalar_type;
		using box_type = Box<DIMENSION,scalar_type>;
		using container_type = typename DataOpts::container_type;

		using node_allocator_policy_type = NodeAllocPolicy;
		static constexpr bool IS_CONTIGUOUS = node_allocator_policy_type::IS_CONTIGUOUS;

		using child_ptr_type = std::conditional_t<IS_CONTIGUOUS, Node*, Node**>;

		/////////////////////////////////////////////////////////////
		/// Storage required per-node
		/////////////////////////////////////////////////////////////
		Node* parent{nullptr};
		child_ptr_type children{nullptr};

		box_type bbox;
		container_type data;

		static inline node_allocator_policy_type* node_alloc{nullptr};
		static inline void set_node_allocator_policy(NodeAllocPolicy& alloc) { node_alloc = &alloc; }

		/// Helper method to get a child
		[[nodiscard]] Node* child(int c) const noexcept {
			assert(0<=c && c<N_CHILDREN);
			if (!children) { return nullptr; }
			if constexpr (IS_CONTIGUOUS) { return children + c; }
			else { return children[c]; }
		}

		/// Helper method to contruct children
		void construct_children() noexcept {
			assert(!children);
			assert(node_alloc);
			children = node_alloc->construct_children(this);
		}

		/// Helper method to destroy children
		void destroy_children() noexcept {
			assert(children);
			assert(node_alloc);
			node_alloc->destroy_children(children);
			children = nullptr;
		}

		/////////////////////////////////////////////////////////////
		/// Constructor and memory management
		/////////////////////////////////////////////////////////////
		Node() {}

		Node(const Node* parent_, int c) noexcept : 
			parent{parent_}, 
			children{nullptr},
			bbox{ChildBox(parent_->bbox, c)} {}

		Node(const Node&) = delete;

		Node& operator=(const Node&) = delete;

		Node(Node&& node) noexcept :
			parent{node.parent},
			children{node.children},
			bbox{std::move(node.bbox)} {
				node.children = nullptr;
				node.parent = nullptr;
				//TODO take data if it exists
			}

		Node& operator=(Node&& node) noexcept {
			if (this != &node) {
				if (node_alloc && children) {
					destroy_children();
				}
				parent = node.parent;
				children = node.children;
				node.parent = nullptr;
				node.children = nullptr;
			}
			return *this;
		}

		~Node() noexcept {
			if (children) {
				assert(node_alloc);
				destroy_children();
				//TODO: destroy data if needed
			}
		}


		/////////////////////////////////////////////////////////////
		/// Essential tree operations
		/////////////////////////////////////////////////////////////
		template<typename PointQuery>
		BinSort<value_type> partition_data(PointQuery&& query) noexcept requires (!std::same_as<value_type,point_type>) {
			return PartitionToOrthant(data.as_span(), bbox.center(), std::forward<PointQuery>(query));

		}

		BinSort<value_type> partition_data() noexcept requires (std::same_as<value_type,point_type>) {
			return PartitionToOrthant(data.as_span(), bbox.center());
		}
		

		/////////////////////////////////////////////////////////////
		/// Static queries for consistancy
		/////////////////////////////////////////////////////////////
		/// Get a specific bit of the orthant/child belongs to. one corresponds to the positive side.
		/// axis 0 is the first bit, axis 1 is the second and so on. ties go to the lower side.
		GUTIL_DECLARE_SIMD()
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

		/// Sort data in-place by the orthant that they belong to in increasing orthant number.
		/// Return the sorting class with state to extract spans to each data set.
		/// If the data type is not the same as point_type, then a method to get the point type
		/// of the data (e.g., its center or a member that stores its spatial location) must
		/// be provided.
		template<typename PointQuery> requires (!std::same_as<value_type,point_type>)
		[[nodiscard]] static BinSort<value_type> PartitionToOrthant(std::span<value_type> data, const point_type& center, PointQuery&& query) noexcept {
			BinSort<value_type> sorter(data, N_ORTHANTS);
			sorter.sort( [&query, &center](const value_type& value) { return GetOrthant(center, query(value)); } );
			return sorter;
		}

		[[nodiscard]] static BinSort<point_type> PartitionToOrthant(std::span<point_type> data, const point_type& center) noexcept {
			BinSort<point_type> sorter(data, N_ORTHANTS);
			sorter.sort( [&center](const point_type& value) { return GetOrthant(center, value); } );
			return sorter;
		}

		/// Get the bounding box of a child node
		[[nodiscard]] static constexpr box_type ChildBox(const box_type& parent_box, int orthant) noexcept {
			assert(0<=orthant && orthant<N_ORTHANTS);
			return {parent_box.center(), parent_box.vertex(orthant)};
		}

		/// Get the bounding box of the parent node
		/// Note the child number is the child number of the node relative to the parent
		[[nodiscard]] static constexpr box_type ParentBox(const box_type& child_box, int child_number) noexcept {
			assert(0<=child_number && child_number<N_ORTHANTS);
			point_type vertex = child_box.vertex(child_number);			//vertex of the parent box, vertex opposite of this is the

			const int center_index = (~child_number) & ORTHANT_MASK;	//flip each axis from high->low and vice versa
			point_type center = child_box.vertex(center_index);

			//vertex to center is center - vertex
			//vertex to opposite is vertex + 2*(vertex to center)
			//simplifies to v + 2*(c-v) = 2*c-v
			return {vertex, scalar_type{2}*center - vertex};
		}

		/// Traverse the tree up to get the depth of the current node (debug tool only)
		[[nodiscard]] static int NodeDepth(const Node* node) noexcept {
			int depth = 0;
			while (node) {
				++depth;
				node = node->parent;
			}
			return depth;
		}

		/// Find the first leaf that contains the point, starting from the supplied (usually root) node
		[[nodiscard]] static Node* FindLeaf(Node* node, const point_type& point) noexcept {
			if (!node) {return nullptr;}
			GUTIL_DEBUG(int depth = 0;)

			box_type box = node->bbox;

			while(true) {
				assert(box == node->bbox);
				assert(box.contains(point));
				assert(depth<MAX_DEPTH);
				
				int c = GetOrthant(node->bbox.center(), point);
				Node* child = node->child(c);
				if (!child) {return node;}
				
				//update bounding box without looking up the child's box
				box = ChildBox(box, c);
				node = child;

				GUTIL_DEBUG(++depth);
			}
		}

		/// Find the first node where pred(node->bbox) is true and for any of its siblings,
		/// pred(sibling->bbox) is false. This can be used to find the deepest node that
		/// completely contains some volume data (e.g., by having the predicate check if the data intersects the box).
		template<typename Predicate>
		[[nodiscard]] static Node* FindNode(Node* node, Predicate&& pred) noexcept {
			if (!node) {return nullptr;}
			GUTIL_DEBUG(int depth = 0;)

			box_type box = node->bbox;

			while(true) {
				assert(box == node->bbox);
				assert(pred(box));
				assert(depth<MAX_DEPTH);
				
				int count = 0;
				int child_number = -1;
				for (int c=0; c<N_ORTHANTS; ++c) {
					if (pred(ChildBox(box, c))) {
						++count;
						child_number = c;
					}
				}
				
				// note that a node intersecting the data but no children intersecting the data does not make sense.
				assert(count!=0 && "Node::FindNode - the predicate evaluated to true in a node but to false in all child nodes");
				if (count != 1) { return node; }
				
				//update bounding box without looking up the child's box
				Node* child = node->child(child_number);
				if (!child) { return node; }	//hit a leaf node

				box = ChildBox(box, child_number);
				node = child;
				GUTIL_DEBUG(++depth);
			}
		}
	};










	


	
}