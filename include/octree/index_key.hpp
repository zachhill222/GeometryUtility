#pragma once

#include "utility/utility.hpp"
#include "math/gutilmath.hpp"
#include "geometry/geometry.hpp"

#include <cstdint>
#include <bit>
#include <array>

#ifndef GUTIL_ORTHTREE_MAX_DEPTH
	#define GUTIL_ORTHTREE_MAX_DEPTH 15
#endif


namespace gutil {
	/////////////////////////////////////////////////////////////////
	/// A few convenience methods
	/////////////////////////////////////////////////////////////////
	template<int DIM>
	[[nodiscard]] inline constexpr uint64_t interleave_bits(std::array<uint64_t,DIM> axis_index, int bits_per_axis) noexcept {
		uint64_t result = 0;
		for (int bit=0; bit<bits_per_axis; ++bit) {
			for (int axis=0; axis<DIM; ++axis) {
				const uint64_t b = axis_index[axis] & uint64_t{1};
				result |= b << (bit*DIM + axis);
				axis_index[axis] >>= 1;
			}
		}
		return result;
	}

	template<int DIM>
	[[nodiscard]] inline constexpr std::array<uint64_t,DIM> deinterleave_bits(uint64_t bits, int bits_per_axis) noexcept {
		std::array<uint64_t,DIM> result{};
		for (int axis=0; axis<DIM; ++axis) {
			uint64_t v = 0;
			for (int bit=0; bit<bits_per_axis; ++bit) {
				v |= ((bits >> (bit*DIM + axis)) & uint64_t{1}) << bit;
			}
			result[axis] = v;
		}
		return result;
	}

	[[nodiscard]] inline constexpr uint64_t remove_prefix_one(uint64_t key) noexcept {
		assert(key>0);
		uint64_t bit = std::bit_width(key) - 1;
		uint64_t mask = (uint64_t{1} << bit) - 1;
		return key & mask;
	}

	/////////////////////////////////////////////////////////////////
	/// A key to make integer coordinates for orthtrees. This encodes
	/// a depth, and axis index (axis i=0...DIM-1) as a uint64_t.
	/// TODO: make an allocator where this key can provide a pointer
	/// to the node that it corresponds to (or nullptr).
	///
	/// There are several extra bits that are available that can be
	/// used to tag the node type (e.g., internal/leaf).
	/////////////////////////////////////////////////////////////////
	template<int DIM> requires (DIM>0)
	struct IndexKey {

		//////////////////////////////////////////////////////////////
		/// Ensure this is a literal data type
		//////////////////////////////////////////////////////////////
		uint64_t data;
		constexpr IndexKey(const uint64_t key) noexcept : data{key} {}
		constexpr IndexKey() noexcept : data{0} {}
		constexpr IndexKey(const IndexKey& other) noexcept : data{other.data} {}
		constexpr IndexKey(IndexKey&& other) noexcept : data{other.data} {}
		constexpr IndexKey& operator=(const IndexKey& other) noexcept {data=other.data; return *this;}
		constexpr IndexKey& operator=(IndexKey&& other) noexcept {data=other.data; return *this;}


		//////////////////////////////////////////////////////////////
		/// Supply a few helpful factories
		//////////////////////////////////////////////////////////////
		[[nodiscard]] static constexpr IndexKey Root() noexcept { return IndexKey{CHECK_BIT}; }
		[[nodiscard]] static constexpr IndexKey Null() noexcept { return IndexKey{0}; 		   }


		//////////////////////////////////////////////////////////////
		/// Compute required numbers of bits and ensure there is enough room
		//////////////////////////////////////////////////////////////
		static constexpr uint64_t ONE = 1;	//specifying uint64_t{1} is tedious
		static constexpr uint64_t MAX_DEPTH = GUTIL_ORTHTREE_MAX_DEPTH;
		static constexpr uint64_t MAX_INDEX = (ONE << MAX_DEPTH) - ONE;
		static constexpr uint64_t DEPTH_BIT_WIDTH = std::bit_width(MAX_DEPTH);
		static constexpr uint64_t INDEX_BIT_WIDTH = std::bit_width(MAX_INDEX);
		static constexpr uint64_t TOTAL_BITS = DEPTH_BIT_WIDTH + DIM*INDEX_BIT_WIDTH;
		static constexpr uint64_t EXTRA_BITS = 63 - TOTAL_BITS;	//1 check bit

		//the total key space has B^0 + B^1 + ... + B^M = (B^(M+1) - 1) / (B-1)
		//unique values, where B=2^DIM
		static constexpr uint64_t KEY_SPACE_SIZE = ((ONE << (DIM*(MAX_DEPTH+1))) - 1) / //B^(M+1) - 1
													((ONE << DIM) - 1);
		static_assert(TOTAL_BITS <= 64,
			"IndexKey - requires more than 64 bits");


		//////////////////////////////////////////////////////////////
		/// Compute field masks and shifts
		///		check_bit | extra | depth | axis DIM-1 | axis DIM-2 | ... | axis 0
		///
		/// In 3D, with max depth of 15, we have:
		/// 	1 check bit, 4 depth bits, 15*3=45 index bits, and 64 - (1+4+45) = 14 extra bits.
		//////////////////////////////////////////////////////////////
		static constexpr uint64_t DATA_MASK = (ONE << TOTAL_BITS) - ONE;	//mask of used bits
		static constexpr uint64_t CHECK_BIT = ONE << 63;					//check if a key is valid

		static constexpr uint64_t DM = (ONE << DEPTH_BIT_WIDTH) - ONE;		//unshifted depth mask
		static constexpr uint64_t AM = (ONE << INDEX_BIT_WIDTH) - ONE;		//unshifted axis mask
		static constexpr uint64_t EM = (ONE << EXTRA_BITS)      - ONE;		//unshifted extra mask

		static constexpr uint64_t DS = DIM*INDEX_BIT_WIDTH;					//start of depth field
		[[nodiscard]] static constexpr uint64_t AS(int i) noexcept {		//start of axis field
			assert(0<= i && i<DIM);
			return i*INDEX_BIT_WIDTH;
		}
		static constexpr uint64_t ES = DS + DEPTH_BIT_WIDTH;				//start of extra field


		static constexpr uint64_t DEPTH_MASK = DM << DS;					//shifted depth mask
		[[nodiscard]] static constexpr uint64_t AXIS_MASK(int i) noexcept {	//shifted axis mask
			assert(0<= i && i<DIM);
			return AM << AS(i);
		}
		static constexpr uint64_t EXTRA_MASK = EM << ES;					//shifted extra mask

		static_assert( EXTRA_MASK == ~(CHECK_BIT | DATA_MASK),
			"IndexKey - mask computation is wrong");


		//////////////////////////////////////////////////////////////
		/// Read and set data
		//////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t depth() noexcept	 	{ return (data&DEPTH_MASK)   >> DS;    }
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t index(int i) noexcept	{ return (data&AXIS_MASK(i)) >> AS(i); }
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t extra() noexcept	  	{ return (data&EXTRA_MASK)   >> ES;    }
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr bool check_bit() noexcept 		{ return data & CHECK_BIT; }

		GUTIL_DECLARE_SIMD()
		constexpr void clear_depth() noexcept				  	{ data &= ~DEPTH_MASK;   }
		GUTIL_DECLARE_SIMD()
		constexpr void clear_index(int i) noexcept			  	{ data &= ~AXIS_MASK(i); }
		GUTIL_DECLARE_SIMD()
		constexpr void clear_extra() noexcept				  	{ data &= ~EXTRA_MASK;   }

		GUTIL_DECLARE_SIMD()
		constexpr void set_depth(uint64_t val) noexcept {
			clear_depth(); 
			const uint64_t DEPTH = val & DM;
			data |= (DEPTH << DS);
		}
		GUTIL_DECLARE_SIMD()
		constexpr void set_index(int i, uint64_t val) noexcept {
			clear_index(i); 
			const uint64_t INDEX = val & AM;
			data |= (INDEX << AS(i));
		}
		GUTIL_DECLARE_SIMD()
		constexpr void set_extra(uint64_t val) noexcept {
			clear_extra();
			const uint64_t EXTRA = val & EXTRA_MASK;
			data |= (EXTRA << ES);
		}


		//////////////////////////////////////////////////////////////
		/// Essential utility functions
		//////////////////////////////////////////////////////////////
		/// Allow bitwise operations with this data type to act on the extra bits.
		/// This allows easy comparison with tags (e.g., tagged pointers).
		
		GUTIL_DECLARE_SIMD()
		constexpr IndexKey& operator<<=(uint64_t val) noexcept { set_extra(extra()<<val); return this; }

		GUTIL_DECLARE_SIMD()
		constexpr IndexKey& operator>>=(uint64_t val) noexcept { set_extra(extra()>>val); return this; }

		GUTIL_DECLARE_SIMD()
		constexpr IndexKey& operator|=(uint64_t val) noexcept { set_extra(extra()|val); return this; }

		GUTIL_DECLARE_SIMD()
		constexpr IndexKey& operator&=(uint64_t val) noexcept { set_extra(extra()&val); return this; }

		GUTIL_DECLARE_SIMD()
		constexpr IndexKey& operator^=(uint64_t val) noexcept { set_extra(extra()^val); return this; }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t operator<<(uint64_t val) const noexcept { return extra()<<=val; }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t operator>>(uint64_t val) const noexcept { return extra()>>=val; }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t operator|(uint64_t val) const noexcept { return extra()|=val; }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t operator&(uint64_t val) const noexcept { return extra()&=val; }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t operator^(uint64_t val) const noexcept { return extra()^=val; }


		/// Convert the key into a linear index (from 0 to KEY_SPACE_SIZE-1 )
		/// This key increases whenever the raw data increases
		/// it could be computed by: LOWER + axis[0] + axis[1]*N + ... + axis[DIM-1]*N^(DIM-1)
		/// where LOWER is the number of nodes at lower depths (obtainable by a geometric sum)
		/// and N=2^depth() is the number of nodes per axis at the current depth.
		/// However, this can be computed by compressing the current data bits
		/// using the minimum field width of 2^depth() for each axis.
		///
		/// The linear key is the same as the raw data the would be used to represent
		/// the current key if the max depth were the current depth.
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t linear_key() const noexcept {
			uint64_t idx{0};
			const uint64_t d = depth();
			uint64_t shift = 0;

			for (int i=0; i<DIM; ++i) {
				idx |= (index(i) << shift);
				shift += d;	//d is the width of each (compressed) index
			}
			idx |= (d << shift);


			//verify bit shifting correctness
			#ifndef NDEBUG
			const uint64_t LOWER = ((ONE << (DIM*d)) - 1) / ((ONE<<DIM)-1);
			uint64_t offset = index(DIM-1);
			const uint64_t N = ONE << d;
			//note the offset is ax(0) + N*(ax(1) + N*(ax(2) +N*(... + N*ax(DIM-1)))
			for (int i=DIM-2; i>0; --i) {
				offset = index(i) + N*offset;
			}
			assert(offset == idx);
			#endif

			return idx;
		}


		/// Convert the key into a continuous space filling curve (at the current depth)
		/// The linear index 'curve' (map from Z to Z^DIM) has many jumps
		/// This curve only has jumps at each depth. A leading 1 is prepended to the mixed digits
		/// so that the depth can still be read. This key can be used as a path from the root
		/// to this node/key. TODO: implement MortonKey class.
		[[nodiscard]] uint64_t morton_index() const noexcept {
			const uint64_t dd = depth();
			std::array<uint64_t,DIM> axis_index{};
			for (int i=0; i<DIM; ++i) { axis_index[i] = index(i); }
			uint64_t path = interleave_bits(axis_index, static_cast<int>(dd));
			return path | (ONE << DIM*dd); //prepend a one to record the width (depth) of the number
		}


		//////////////////////////////////////////////////////////////
		/// Get the key of a child or parent
		//////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr IndexKey parent() noexcept {
			assert(depth()>0);
			assert(check_bit());
			
			IndexKey key = Root();
			key.set_depth(depth()-1);
			for (int i=0; i<DIM; ++i) {
				key.set_index(i, index(i)>>1);	//note >>1 is the same as /2
			}
			return key;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr IndexKey child(int c) noexcept {
			assert(depth()<MAX_DEPTH);
			assert(check_bit());

			IndexKey key = Root();
			key.set_depth(depth()+1);
			for (int i=0; i<DIM; ++i) {
				//the i-th bit of c determines the high/low side of the i-th axis
				//this is a standard orthant/vertex ordering scheme
				const uint64_t idx = (c & (1<<i)) ? 1 + (index(i)<<1) : (index(i)<<1);	//note <<1 is the same as *2
				key.set_index(i, idx);
			}
			return key;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr IndexKey neighbor(const std::array<int,DIM>& offsets) const noexcept {
			assert(depth()<MAX_DEPTH);
			assert(check_bit());
			
			IndexKey key{data};
			for (int i=0; i<DIM; ++i) {
				const uint64_t idx = index(i);
				if (offsets[i] > 0) { 
					if (idx == MAX_INDEX) { return Null(); }
					key.set_index(i,idx+1);
				}
				else if (offsets[i] < 0) {
					if (idx == 0) { return Null(); }
					key.set_index(i,idx-1);
				}
			}
			return key;
		}


		//////////////////////////////////////////////////////////////
		/// Comparisons and queries
		/// TODO: think about if the extra bits contribute or not.
		/// For now, the check and
		//////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t raw_key() noexcept { return data & DATA_MASK; }

		template<IsScalar T>
		[[nodiscard]] constexpr Box<DIM,T> compute_box(const Box<DIM,T>& root_box) const {
			Point<DIM,T> depth_diag = root_box.sidelengh();
			Point<DIM,T> lo = root_box.low;

			const int dd = -static_cast<int>(depth());
			for (int i=0; i<DIM; ++i) {
				depth_diag[i] = gutil::ldexp(depth_diag[i], dd);	// root_size * 2^-depth
				lo[i] += static_cast<T>(index(i))*depth_diag[i];
			}

			return {lo, lo+depth_diag};
		}

		template<IsScalar T>
		[[nodiscard]] constexpr Point<DIM,T> compute_center(const Box<DIM,T>& root_box) const {
			Point<DIM,T> depth_diag = root_box.sidelengh();
			Point<DIM,T> lo = root_box.low;

			const int dd = -static_cast<int>(depth());
			for (int i=0; i<DIM; ++i) {
				depth_diag[i] = gutil::ldexp(depth_diag[i], dd);	// root_size * 2^-depth
				lo[i] += static_cast<T>(index(i))*depth_diag[i];
			}

			if constexpr (IsReal<T>) { return lo + T{0.5}*depth_diag; }
			else { return lo + depth_diag/T{2}; }
		}
		
	};


	//////////////////////////////////////////////////////////////////
	/// Out of class methods for IndexKey
	//////////////////////////////////////////////////////////////////
	GUTIL_DECLARE_SIMD()
	template<int DIM> requires (DIM>0)
	[[nodiscard]] inline constexpr bool operator==(IndexKey<DIM> A, IndexKey<DIM> B) noexcept { return A.raw_key() == B.raw_key(); }

	GUTIL_DECLARE_SIMD()
	template<int DIM> requires (DIM>0)
	[[nodiscard]] inline constexpr std::strong_ordering operator<=>(IndexKey<DIM> A, IndexKey<DIM> B) noexcept { return A.raw_key() <=> B.raw_key(); }


	
}


//////////////////////////////////////////////////////////////////
/// Inject a hash into std
//////////////////////////////////////////////////////////////////
namespace std {
	template<int DIM>
	struct hash<gutil::IndexKey<DIM>> {
		[[nodiscard]] size_t operator()(gutil::IndexKey<DIM> key) const noexcept {
			return std::hash<uint64_t>{}( key.raw_key() );
		}
	};
}