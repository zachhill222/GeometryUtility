#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"

#include <cstdint>
#include <bit>
#include <iostream>

namespace gutil
{
	//this key stores octree nodes as a MAX digit base-DIMENSION number stored in a uint64_t value. The first non-zero bit is a sentinel/root bit 
	//and is always 1 on a valid node. Thus any valid key is greater than 0.
	//in 3D, each octant requires 3-bits (base-8), in 2D, each quadrant requires 2-bits (base-4), in 1D, each half requires 1-bit (base-2)
	//The bit layout is as follows:
	//	(0-padding) (1-bit root/depth 0) (D-bit key_depth) (D-bit key_depth-1) ... (D-bit depth 1)
	//in 3D:
	//	root: 0...01
	//  root.child(0): 0...01000
	//	root.child(1): 0...01001
	//	root.child(1).child(0): 0...01001000
	//note that if the bit-width is W (minimum of 1), then we are at depth (W-1)/DIM
	template<uint64_t DIMENSION=3>
	struct OctreeDigitKey
	{	
		//sanity check
		static_assert(0<DIMENSION && DIMENSION<4, "OctreeDigitKey - DIMENSION must be 1, 2, or 3");

		//storage
		uint64_t _data_{0};

		//alias to the bounding box type
		template<typename T=double>
		using box_type = Box<DIMENSION,T>;

		template<typename T=double>
		using point_type = Point<DIMENSION,T>;

		//constants
		static constexpr uint64_t ROOT = 1;
		static constexpr uint64_t DIM  = DIMENSION; //also the number of bits required for each digit
		static constexpr uint64_t MAX_DEPTH = 63/DIM;
		static constexpr uint64_t DIGIT_MASK = (uint64_t{1} << DIM) - 1;
		static constexpr uint64_t N_CHILDREN = uint64_t{1} << DIM;

		//bit masks and shifts
		constexpr uint64_t R_S() const {return std::bit_width(_data_)-1;}
		static constexpr uint64_t D_S(uint64_t dd) {assert(dd<=MAX_DEPTH && dd>0); return 1 + DIM*(dd-1);}			//depth start
		static constexpr uint64_t D_M(uint64_t dd) {assert(dd<=MAX_DEPTH && dd>0); return DIGIT_MASK << D_S(dd);}  //depth mask

		//simple queries
		inline constexpr uint64_t depth() const {return R_S()/DIM;}
		inline constexpr bool is_valid()  const {return _data_ & ROOT;}
		inline constexpr bool is_root()	  const {return _data_ == ROOT;} 

		//get the digit (child number) for depth dd
		inline constexpr uint64_t digit(const uint64_t dd) const {return (_data_ >> D_S(dd)) & DIGIT_MASK;}

		//comparisons
		bool constexpr operator==(const OctreeDigitKey other) const {return _data_ == other._data_;}
		bool constexpr operator!=(const OctreeDigitKey other) const {return _data_ != other._data_;}
		bool constexpr operator<=(const OctreeDigitKey other) const {return _data_ <= other._data_;}
		bool constexpr operator>=(const OctreeDigitKey other) const {return _data_ >= other._data_;}
		bool constexpr operator<( const OctreeDigitKey other) const {return _data_ <  other._data_;}
		bool constexpr operator>( const OctreeDigitKey other) const {return _data_ >  other._data_;}

		//trivial constructors
		OctreeDigitKey() : _data_{ROOT} {}
		explicit OctreeDigitKey(const uint64_t data) : _data_{data} {}

		//copy and move operations
		OctreeDigitKey(const OctreeDigitKey& other) : _data_{other._data_} {}
		OctreeDigitKey(OctreeDigitKey&& other) : _data_{other._data_} {}
		OctreeDigitKey& operator=(const OctreeDigitKey& other) {_data_=other._data_; return *this;}
		OctreeDigitKey& operator=(OctreeDigitKey&& other) {_data_=other._data_; return *this;}

		//hierarchy relations
		inline constexpr OctreeDigitKey parent() const {assert(depth()>0); return OctreeDigitKey{_data_ & ~D_M(depth())};}
		inline constexpr OctreeDigitKey child(const uint64_t c) const {assert(depth()<MAX_DEPTH); assert(c<N_CHILDREN); return OctreeDigitKey{_data_ | (c << D_S(depth()+1))};}

		//given the extents of this node and a query point, descend to the (first) child that contains the querry point.
		//update the low and high to the child
		template<typename T>
		inline void descend(point_type<T>& low, point_type<T>& high, const point_type<T>& query) {
			assert(low<high);
			assert(low <= query && query <= high);
			assert(depth()<MAX_DEPTH);

			const point_type<T> center = low + T{0.5}*(high-low);
			uint64_t c=0;
			for (uint64_t ax=0; ax<DIM; ++ax) {
				if (query[ax] <= center[ax]) {
					high[ax] = center[ax];
					//axis bit is 0 (low coord is constant for this axis)
				}
				else {
					low[ax] = center[ax];
					c |= (uint64_t{1} << ax);
					//axis bit is 1 (high coord is constant for this axis)
				}
			}

			//ensure this stays consistent with child(c)
			_data_ |= (c << D_S(depth()+1));
		}


		//given an octree bbox, determine the bbox of this node
		template<typename T>
		constexpr box_type<T> bbox(const box_type<T>& root_box) const noexcept {
			assert(is_valid());

			auto low  = root_box.low();
			auto high = root_box.high();

			const uint64_t d = depth();
			for (uint64_t dd=1; dd<=d; ++dd) {
				const uint64_t dg = digit(dd);
				auto center = low + T{0.5}*(high-low);
				
				//set each axis extents
				for (uint64_t ax=0; ax<DIM; ++ax) {
					const uint64_t bit_mask = uint64_t{1} << ax;
					if (dg & bit_mask) {low[ax]=center[ax];}
					else {high[ax]=center[ax];}
				}
			}

			return box_type<T>{low, high};
		}

		//determine if this key is an ancestor of another key
		//a key is not an ancestor of itself
		constexpr bool is_ancestor_of(const OctreeDigitKey other) const {
			const uint64_t dd = depth();
			if (dd < other.depth()) {
				const uint64_t shift = DIM * (other.depth()-dd);
				return (other._data_>>shift) == _data_;
			}
			else {
				return false;
			}
		}

		//pre-pend a key (useful for expanding the tree upwards to enlarge the bbox)
		constexpr OctreeDigitKey& prepend(const OctreeDigitKey prefix) noexcept {
			assert(is_valid());
			assert(prefix.is_valid());
			assert(depth() + prefix.depth() <= MAX_DEPTH);
			_data_ &= ~ROOT;
			_data_ <<= DIM * prefix.depth();
			_data_ |= prefix._data_;
			return *this;
		}
		inline constexpr OctreeDigitKey prepended(const OctreeDigitKey prefix) const noexcept {OctreeDigitKey copy{_data_}; return copy.prepend(prefix);}
	};

	template<uint64_t DIM>
	inline std::ostream& operator<<(std::ostream& os, const OctreeDigitKey<DIM> key)
	{
		//sanity check
		if (!key.is_valid()) {os << "<invalid>"; return os;}

		//represent the key as R.a.b.c... where R is root and a,b,c are the digits
		os << 'R';
		const uint64_t d = key.depth();
		for (uint64_t dd=1; dd<=d; ++dd) {
			os << '.' << key.digit(dd);
		}
		return os;
	}
}

//inject the key into the std hash
template<uint64_t DIM>
struct std::hash<gutil::OctreeDigitKey<DIM>>
{
	size_t operator()(const gutil::OctreeDigitKey<DIM>& k) const noexcept {return std::hash<uint64_t>{}(k._data_);}
};