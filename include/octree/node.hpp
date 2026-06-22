#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"

#include <array>
#include <concepts>

namespace gutil
{
	//class for passing compile time options
	template<typename DataT, bool SingleData, size_t MaxData, size_t Dimension, typename ScalarT>
	struct NodeOpts
	{
		static constexpr bool SINGLE_DATA	= SingleData;
		static constexpr size_t DIM 		= Dimension;
		static constexpr size_t MAX_DATA 	= MaxData;
		static constexpr size_t N_CHILDREN 	= (1 << Dimension);

		using value_type 	= DataT;
		using point_type 	= Point<DIM,ScalarT>;
		using box_type 		= Box<DIM,ScalarT>;
		using scalar_type 	= ScalarT;

		//ensure that the scalar_type is reasonable
		static_assert(std::convertible_to<scalar_type,double>);
	};

	//types of nodes
	enum class NodeTag : uint8_t
	{
		INTERNAL,
		LEAF
	};

	//return types for inserting data
	enum class InsertReturn : uint8_t
	{
		SUCCESS,	//unique data was successfully added
		EXISTS,		//data already existed and was not added
		OVERFLOW,	//there was no room to add the data
		FAIL		//the data could not be added (for example, if the data was invalid)
	};

	//a class to determine if a child index is internal or leaf type
	struct NodeIndex
	{
		size_t index;
		NodeTag tag;

		//guard to set if a child does not exist
		static constexpr size_t NULL_NODE = size_t(-1);

		//default constructor
		constexpr NodeIndex() : index{NULL_NODE}, tag{NodeTag::INTERNAL} {}

		//comparisons
		bool operator==(const NodeTag& other) {return (index==other.index) && (tag==other.tag);}
		bool operator!=(const NodeTag& other) {return (index!=other.index) || (tag!=other.tag);}
		
		static constexpr NodeIndex leaf(const size_t idx) {return {idx, NodeTag::LEAF};}
		static constexpr NodeIndex internal(const size_t idx) {return {idx, NodeTag::INTERNAL};}
	};

	//a class for leaf nodes
	template<NodeOpts Opts>
	struct LeafNode
	{
		//determine if we are storing data or indices to data
		//for now, always store indices so the underlying data is unique and contiguous
		using store_type = size_t;

		//store spatial extents
		typename Opts::box_type bbox;

		//data
		std::array<store_type, Opts::MAX_DATA> data;
		size_t cursor{0};
		
		//tag for validity checks
		static constexpr NodeTag TAG = NodeTag::LEAF;

		//tree connectivity
		size_t parent = size_t(-1);

		//queries
		size_t n_idx() const {return cursor;}
		bool full() const {return cursor>=Opts::MAX_DATA;}
		bool contains(const store_type value) const {
			for (size_t ii=0; ii<cursor; ++ii)
				if (data[ii] == value)
					return true;
			return false;
		}

		store_type operator[](size_t ii) const {assert(ii<cursor); return data[ii];}
		store_type& operator[](size_t ii) {assert(ii<cursor); return data[ii];}
		
		InsertReturn insert(const store_type value) {
			assert(cursor<Opts::MAX_DATA);
			if (full()) return InsertReturn::OVERFLOW;
			if (contains(value)) return InsertReturn::EXISTS;
			
			data[cursor++] = value;
			return InsertReturn::SUCCESS;
		}

		//constructor
		LeafNode(const typename Opts::box_type& box) : bbox(box) {}
	};

	//a class for internal nodes
	template<NodeOpts Opts>
	struct InternalNode
	{
		std::array<NodeIndex, Opts::N_CHILDREN> children;
		size_t parent = size_t(-1);

		static constexpr NodeTag TAG = NodeTag::INTERNAL;
		typename Opts::box_type bbox;
		
		//constructor
		InternalNode(const typename Opts::box_type box) : bbox(box) {for (auto& idx : children) idx = NodeIndex{};}
	};
}