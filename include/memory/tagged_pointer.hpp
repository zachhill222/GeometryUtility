#pragma once

#include <cstdint>
#include <bit>

namespace gutil {

	
	//////////////////////////////////////////////////////////////////////////////////////////////
	///	A class to store type information in a raw pointer. Useful for e.g., graphs with a fixed collection
	/// of node/vertex types. Note that the number of distinct tags is limited by the allignment.
	///
	/// we use tagged pointers to differentiate between internal and leaf nodes
	/// it is imperative that the nodes are aligned correctly so that the last several bits
	/// of the pointer may be used for the tag (they will always be 0 due to the alignment)
	/// high bits could be used without the alignment, but that will be platform dependent.
	/// note if a type T is aligned to 4 bytes, the last 2 bits will be free. Similarly,
	/// if the alignment is 16, the last 4 bits will be free.
	/////////////////////////////////////////////////////////////////////////////////////////////
	template<size_t AlignBytes>
	struct TaggedPointer {
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
		bool is_null() const noexcept {return (_data_&PTR_MASK) == 0;}
		operator bool() const noexcept {return !is_null();}

		//factory to make a null pointer
		static constexpr TaggedPointer Null() {return TaggedPointer{0};}

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


	/////////////////////////////////////////////////////////////////
	/// In an orthtree, we allow different types of nodes for the 
	/// internal and leaf nodes. Use a tagged pointer to differentiate
	/// between the two. For portability, the tag is put into the low
	/// bits of the pointer using alignment to guarentee that the low
	/// bits are free. We use free bits to generate unique tags by bitwise or.
	/// For example, a node general pupose node that can be leaf or internal and has data
	/// would have a tag DATA | LEAF | INTERNAL = 0b00111 = 7.
	///
	/// The tags ending in _T should only be used to encode type information.
	/// The tags ending in _R can be used to encode runtime information.
	/// For example if HAS_DATA_R is used to mark if a code has data, we could check with
	/// t_ptr.tag() & HAS_DATA_R before casting to the actual pointer type.
	/////////////////////////////////////////////////////////////////
	namespace NodeTag {
		inline constexpr size_t    NODE_ALIGN_BYTES = 64; //gives us bit_width(63) = 5 bits to work with

		inline constexpr uintptr_t DATA_T     = 1;	//0b00001
		inline constexpr uintptr_t LEAF_T     = 2;	//0b00010
		inline constexpr uintptr_t INTERNAL_T = 4;	//0b00100
		inline constexpr uintptr_t HAS_DATA_R = 8;	//0b01000
		inline constexpr uintptr_t IS_LEAF_R  =16;	//0b10000
		
		inline constexpr uintptr_t T_MASK = DATA_T | LEAF_T | INTERNAL_T;
		inline constexpr uintptr_t R_MASK = HAS_DATA_R | IS_LEAF_R;

		inline constexpr uintptr_t XXX = 0;
		inline constexpr uintptr_t XXI = INTERNAL_T;
		inline constexpr uintptr_t XDX = DATA_T;
		inline constexpr uintptr_t LXX = LEAF_T;
		inline constexpr uintptr_t XDI = DATA_T | INTERNAL_T;
		inline constexpr uintptr_t LXI = LEAF_T | INTERNAL_T;
		inline constexpr uintptr_t LDX = LEAF_T | DATA_T;
		inline constexpr uintptr_t LDI = LEAF_T | DATA_T | INTERNAL_T;
	}
}