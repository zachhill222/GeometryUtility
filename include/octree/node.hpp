#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "memory/base_allocator.hpp"


#include <array>
#include <concepts>
#include <cstdint>
#include <bit>

namespace gutil
{
	//define tags for node types
	namespace NodeTag
	{
		inline constexpr uintptr_t NONE     = 0;
		inline constexpr uintptr_t LEAF     = 1;
		inline constexpr uintptr_t INTERNAL = 2;
		inline constexpr size_t    NODE_ALIGN_BYTES = 8;
	}

	//class for octree compile time options
	template<typename ValueType, bool VolumeData, int MaxData, int DIM, typename ScalarT>
	struct NodeOpts
	{
		static constexpr int DIMENSION 		  = DIM;
		static constexpr int MAX_DATA 	  = MaxData;
		static constexpr int N_CHILDREN 	  = (1 << DIM);
		static constexpr bool   VOLUME_DATA   = VolumeData;

		using value_type 	= ValueType;
		using point_type 	= Point<DIMENSION,ScalarT>;
		using box_type 		= Box<DIMENSION,ScalarT>;
		using scalar_type 	= ScalarT;
		using index_type    = size_t;

		//ensure that the scalar_type is reasonable
		static_assert(std::convertible_to<scalar_type,double>);
		static_assert(MAX_DATA > 0);
		static_assert(DIMENSION > 0);
	};

	//concept to help pass compile time node options
	template<typename T>
	concept IsNodeOpts = requires {
		{T::DIMENSION} -> std::convertible_to<int>;
		{T::MAX_DATA} -> std::convertible_to<int>;
		{T::N_CHILDREN} -> std::convertible_to<int>;
		{T::VOLUME_DATA} -> std::convertible_to<bool>;
		typename T::value_type;
		typename T::point_type;
		typename T::box_type;
		typename T::scalar_type;
		typename T::index_type;
	};

	//we use tagged pointers to differentiate between internal and leaf nodes
	//it is imperative that the nodes are aligned correctly so that the last several bits
	//of the pointer may be used for the tag (they will always be 0 due to the alignment)
	//high bits could be used without the alignment, but that will be platform dependent.
	//note if a type T is aligned to 4 bytes, the last 2 bits will be free. Similarly,
	//if the alignment is 16, the last 4 bits will be free.
	template<size_t AlignBytes>
	struct TaggedPointer
	{
		static_assert( (AlignBytes & (AlignBytes-1)) == 0, "TaggedPointer: alignment must be a power of 2");
		static constexpr uintptr_t TAG_BITS = std::bit_width(AlignBytes) - 1;
		static constexpr uintptr_t TAG_MASK = AlignBytes-1; //same as (uintptr_t{1} << TAG_BITS) - 1
		static constexpr uintptr_t PTR_MASK = ~TAG_MASK;
		uintptr_t _data_{0};	//nullptr, tag is 0

		//allow static cast when the caller knows the type,
		//same as t_ptr.template pointer<T>(), but more readable 
		template<typename T> requires (alignof(T) >= AlignBytes)
		explicit operator T*() const {
			if constexpr (requires {{T::TAG} -> std::convertible_to<uintptr_t>;}) {
				assert(T::TAG == tag() && "TaggedPointer: tag mismatch");
			}
			return reinterpret_cast<T*>(_data_&PTR_MASK);
		}

		template<typename T> requires (alignof(T) >= AlignBytes)
		explicit operator const T*() const {
			if constexpr (requires {{T::TAG} -> std::convertible_to<uintptr_t>;}) {
				assert(T::TAG == tag() && "TaggedPointer: tag mismatch");
			}
			return reinterpret_cast<const T*>(_data_&PTR_MASK);
		}

		//get the tag
		uintptr_t tag() const {return _data_&TAG_MASK;}

		//set the tag
		void set_tag(const uintptr_t t) {
			_data_ &= PTR_MASK;
			_data_ |= (t&TAG_MASK);
		}

		//check if the pointer is null
		bool is_null() const {return (_data_&PTR_MASK) == 0;}

		//factory to make a null pointer
		static constexpr TaggedPointer null() {return TaggedPointer{};}

		//construct a tagged pointer from a pointer
		template<typename T> requires (alignof(T) >= AlignBytes)
		TaggedPointer(T* ptr, const uintptr_t t=0) : _data_(reinterpret_cast<uintptr_t>(ptr)) {set_tag(t);}

		template<typename T> requires (alignof(T) >= AlignBytes)
		TaggedPointer(const T* ptr, const uintptr_t t=0) : _data_(reinterpret_cast<uintptr_t>(ptr)) {set_tag(t);}

		//default is null
		constexpr TaggedPointer() noexcept : _data_{0} {}

		//comparisons
		bool operator==(const TaggedPointer other) const {return _data_ == other._data_;}
		bool operator<(const TaggedPointer other) const {return _data_ < other._data_;}
	};

	//forward declare leaf/internal types
	template<IsNodeOpts Opts>
	struct LeafNode;

	template<IsNodeOpts Opts>
	struct InternalNode;

	//base class for nodes
	template<IsNodeOpts Opts>
	struct alignas(NodeTag::NODE_ALIGN_BYTES) NodeBase
	{
		using box_type     = typename Opts::box_type;
		using point_type   = typename Opts::point_type;
		using scalar_type  = typename Opts::scalar_type;
		using tag_ptr_type = TaggedPointer<NodeTag::NODE_ALIGN_BYTES>;

		//constructor must always set the box
		NodeBase(const box_type& box) : bbox(box) {}

		//store the box for this node
		box_type bbox;

		//always have a parent (except for the root)
		InternalNode<Opts>* parent{nullptr};

		//get the box of a child node
		[[nodiscard]] box_type child_box(const int n) const noexcept {
			assert(n<Opts::N_CHILDREN);

			//the bits of the child number encode the high/low side of the corresponding axis
			const point_type center = bbox.center();
			point_type vertex{bbox.low};
			for (int ax=0; ax<Opts::DIMENSION; ++ax) {
				if ( n & (int{1} << ax) ) {vertex[ax] = bbox.high[ax];}
			}

			return {center, vertex};
		}

		//get a specific bit of the octant/child belongs to. one corresponds to the positive side.
		//axis 0 is the first bit, axis 1 is the second and so on. ties go to the lower side.
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr bool octant_bit(scalar_type cntr, scalar_type query) noexcept {
			return query > cntr;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr bool octant_bit(int octant, int axis) noexcept {
			assert(0<= axis && axis<Opts::DIMENSION);
			assert(0<=octant && octant<Opts::N_CHILDREN);
			return octant & (int(1)<<axis);
		}

		//get the octant/child that a point belongs to
		[[nodiscard]] static constexpr int octant(const point_type& cntr, const point_type& query) noexcept {
			int n=0;
			for (int i=0; i<Opts::DIMENSION; ++i) {
				if (octant_bit(cntr.data[i], query.data[i])) {n |= (int{1} << i);} 
			}
			return n;
		}
	};

	//class for leaf nodes
	template<IsNodeOpts Opts>
	struct LeafNode : public NodeBase<Opts>
	{
		//save options
		using OPTS = Opts;
		static constexpr int MAX_DATA = Opts::MAX_DATA;
		static constexpr uintptr_t TAG 	 = NodeTag::LEAF;

		//save aliases
		using base_type   = NodeBase<Opts>;
		using index_type  = typename Opts::index_type;	//type that is stored in the leaf nodes
		using point_type  = typename Opts::point_type;	//type of spatial points
		using box_type	  = typename Opts::box_type;	//type of spatial axis-aligned-bounding-boxes
		using scalar_type = typename Opts::scalar_type;	//type that emulates real numbers for the spatial points and aabb
		using tag_ptr_type= typename base_type::tag_ptr_type;

		//construct from a box
		using base_type::base_type;

		//store data and cursor
		std::array<index_type, MAX_DATA> data;
		int cursor{0};

		//iterators for easier looping over the data
		auto begin() const {return data.cbegin();}
		auto begin() {return data.begin();}
		auto end() const {return data.cbegin()+cursor;}
		auto end() {return data.begin()+cursor;}

		//simple queries
		bool full() const {return cursor>=MAX_DATA;}
		bool empty() const {return cursor==0;}
		size_t size() const {return static_cast<size_t>(cursor);}
		bool contains(const index_type& index) const {
			for (int i=0; i<cursor; ++i) {
				if (data[i]==index) {return true;}
			}
			return false;
		}
		size_t remaining_capacity() const {return static_cast<size_t>(MAX_DATA-cursor);}

		//move data into the leaf
		void insert(const index_type index) {
			assert(!full());
			data[cursor++] = index;
		}

		//remove data from the leaf
		void remove(const index_type index) {
			for (size_t i=0; i<cursor; ++i) {
				if (data[i] == index) {
					std::swap(data[i], data[cursor-1]);
					--cursor;
					return;
				}
			}
		}

		//convert to a tagged pointer
		tag_ptr_type t_ptr() const {return tag_ptr_type{this, TAG};}
	};

	//class for internal nodes
	template<IsNodeOpts Opts>
	struct InternalNode : public NodeBase<Opts>
	{
		//save options
		using OPTS = Opts;
		static constexpr int DIMENSION  = Opts::DIMENSION;
		static constexpr int N_CHILDREN = Opts::N_CHILDREN;
		static constexpr uintptr_t TAG  = NodeTag::INTERNAL;

		//save aliases
		using base_type   = NodeBase<Opts>;
		using index_type  = typename Opts::index_type;	//type that is stored in the leaf nodes
		using point_type  = typename Opts::point_type;	//type of spatial points
		using box_type	  = typename Opts::box_type;	//type of spatial axis-aligned-bounding-boxes
		using scalar_type = typename Opts::scalar_type;	//type that emulates real numbers for the spatial points and aabb
		using tag_ptr_type= typename base_type::tag_ptr_type;

		//bring bounding box into this scope
		using base_type::bbox;

		//construct from a box
		InternalNode(const box_type& box) : base_type(box) {children.fill(tag_ptr_type{});}

		//store pointers to children
		std::array<tag_ptr_type, N_CHILDREN> children;

		//iterators for easier looping over the children
		auto begin() const {return children.cbegin();}
		auto begin() {return children.begin();}
		auto end() const {return children.cend();}
		auto end() {return children.end();}

		//construct child nodes as leafs given an allocator
		template<typename Allocator>// requires IsAllocatable<Allocator,InternalNode>
		void construct_child_leafs(Allocator _alloc_) {
			const point_type center = bbox.center();

			using leaf_type = LeafNode<Opts>;
			//the bits of the child number encode the octant that it owns
			//ex. if n = (011)_2, then it is the 'high' side of axis 0 and 1, but the 'low' side of axis 2
			for (size_t n=0; n<N_CHILDREN; ++n) {
				point_type vertex = bbox.voxelvertex(n); //same bit operations
				leaf_type* leaf = _alloc_.construct(box_type{vertex, center});

				assert((reinterpret_cast<uintptr_t>(leaf) & 0x7) == 0
				    && "leaf pointer is not 8-byte aligned");

				leaf->parent	= this;
				children[n] 	= tag_ptr_type{leaf, leaf->TAG};
			}
		}

		//convert to a tagged pointer
		tag_ptr_type t_ptr() const {return tag_ptr_type{this, TAG};}
	};
}