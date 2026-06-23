#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"

#include <array>
#include <concepts>
#include <cstdint>

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
	template<typename ValueType, bool StoreInLeaf, size_t MaxData, size_t Dimension, typename ScalarT>
	struct NodeOpts
	{
		static constexpr size_t DIM 		  = Dimension;
		static constexpr size_t MAX_DATA 	  = MaxData;
		static constexpr size_t N_CHILDREN 	  = (1 << Dimension);
		static constexpr bool   STORE_IN_LEAF = StoreInLeaf;

		using value_type 	= ValueType;
		using point_type 	= Point<DIM,ScalarT>;
		using box_type 		= Box<DIM,ScalarT>;
		using scalar_type 	= ScalarT;
		using store_type    = std::conditional_t<StoreInLeaf, value_type, size_t>;

		//ensure that the scalar_type is reasonable
		static_assert(std::convertible_to<scalar_type,double>);
		static_assert(MAX_DATA > 0);
		static_assert(DIM > 0);
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
		uintptr_t _data_{0};	//nullptr, no tag of 0

		//cast the pointer to a specified type
		template<typename T> requires (alignof(T) >= AlignBytes)
		T* pointer() const {return reinterpret_cast<T*>(_data_&PTR_MASK);}
		
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

		//construct a tagged pointer from a reference
		template<typename T> requires (alignof(T) >= AlignBytes)
		TaggedPointer(T& obj, const uintptr_t t=0) : _data_(std::reinterpret_cast<uintptr_t>(&obj)) {set_tag(t);}

		//construct a tagged pointer from a pointer
		template<typename T> requires (alignof(T) >= AlignBytes)
		TaggedPointer(T* ptr, const uintptr_t t=0) : _data_(std::reinterpret_cast<uintptr_t>(ptr)) {set_tag(t);}

		//default is null
		constexpr TaggedPointer() noexcept : _data_{0} {}
	};

	//base class for nodes
	template<NodeOpts Opts>
	struct alignas(NodeTag::NODE_ALIGN_BYTES) NodeBase
	{
		using box_type    = typename Opts::box_type;
		using point_type  = typename Opts::point_type;
		using scalar_type = typename Opts::scalar_type;

		//constructor must always set the box
		NodeBase(const box_type& box) : bbox(box) {}

		//store the box for this node
		box_type bbox;

		//get the box of a child node
		box_type child_box(const size_t n) const {
			assert(n<Opts::N_CHILDREN);

			//the bits of the child number encode the high/low side of the corresponding axis
			const point_type low = bbox.low();
			const point_type high = bbox.high();
			const point_type center = low + scalar_type{0.5}*(high-low);
			point_type vertex = low;
			for (size_t ax=0; ax<Opts::DIM; ++ax) {
				if ( n & (size_t{1} << ax) ) {vertex[ax] = high[ax];}
			}

			return {center, vertex};
		}
	};


	//class for leaf nodes
	template<NodeOpts Opts>
	struct LeafNode : public NodeBase<Opts>
	{
		//save options
		static constexpr NodeOpts OPTS 	 = Opts;
		static constexpr size_t MAX_DATA = Opts::MAX_DATA;
		static constexpr uintptr_t TAG 	 = NodeTag::LEAF;

		//save aliases
		using base_type   = NodeBase<Opts>;
		using store_type  = typename Opts::store_type;	//type that is stored in the leaf nodes
		using point_type  = typename Opts::point_type;	//type of spatial points
		using box_type	  = typename Opts::box_type;	//type of spatial axis-aligned-bounding-boxes
		using scalar_type = typename Opts::scalar_type;	//type that emulates real numbers for the spatial points and aabb

		//construct from a box
		using base_type::base_type;

		//store data and cursor
		std::array<store_type, MAX_DATA> data;
		size_t cursor{0};

		//iterators for easier looping over the data
		auto begin() const {return data.cbegin();}
		auto begin() {return data.begin();}
		auto end() const {return data.cbegin()+cursor;}
		auto end() {return data.begin()+cursor;}

		//simple queries
		bool full() const {return cursor>=MAX_DATA;}
		bool empty() const {return cursor==0;}
		size_t size() const {return cursor;}
	};

	//class for internal nodes
	template<NodeOpts Opts>
	struct InternalNode : public NodeBase<Opts>
	{
		//save options
		static constexpr NodeOpts OPTS = Opts;
		static constexpr size_t N_CHILDREN = Opts::N_CHILDREN;
		static constexpr uintptr_t TAG = NodeTag::INTERNAL;

		//save aliases
		using base_type   = NodeBase<Opts>;
		using store_type  = typename Opts::store_type;	//type that is stored in the leaf nodes
		using point_type  = typename Opts::point_type;	//type of spatial points
		using box_type	  = typename Opts::box_type;	//type of spatial axis-aligned-bounding-boxes
		using scalar_type = typename Opts::scalar_type;	//type that emulates real numbers for the spatial points and aabb

		//construct from a box
		InternalNode(const box_type& box) : base_type(box) {children.fill(TaggedPointer<alignof(base_type)>{});}

		//store pointers to children
		std::array<TaggedPointer<alignof(base_type)>, N_CHILDREN> children;

		//iterators for easier looping over the children
		auto begin() const {return children.cbegin();}
		auto begin() {return children.begin();}
		auto end() const {return children.cend();}
		auto end() {return children.end();}
	};
}