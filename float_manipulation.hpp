#pragma once

#include <concepts>
#include <cstdint>
#include <iostream>
#include <cstring>
#include <string>
#include <bitset>
#include <bit>
#include <cmath>

namespace gv::util
{
	///////////////////////////////////////////////////////////////
	/// Union type for integer/float identification and manipulation
	///////////////////////////////////////////////////////////////
	template<int nBits>
		requires (nBits == 32 || nBits == 64) //add 128 later?
	struct FloatingPointBits
	{
		///////////////////////////////////////////////////////////////
		/// Type defs
		///////////////////////////////////////////////////////////////
		static constexpr int BITS = nBits;

		using float_type = decltype([]() {
			if constexpr (BITS == 32) {return float{};}
			else if constexpr (BITS == 64) {return double{};}
			else static_assert(BITS == 32, "FloatingPointBits: unsuported number of bits");
		}());

		using int_type = decltype([]() {
			if constexpr (BITS == 32) {return int32_t{};}
			else if constexpr (BITS == 64) {return int64_t{};}
			else static_assert(BITS == 32, "FloatingPointBits: unsuported number of bits");
		}());


		using uint_type = decltype([]() {
			if constexpr (BITS == 32) {return uint32_t{};}
			else if constexpr (BITS == 64) {return uint64_t{};}
			else static_assert(BITS == 32, "FloatingPointBits: unsuported number of bits");
		}());

		///////////////////////////////////////////////////////////////
		/// Size of bit fields and exponent bias (IEEE 754)
		///////////////////////////////////////////////////////////////
		static constexpr int_type SIGN_BITS{1};

		static constexpr int_type MANTISSA_BITS = []() {
			if constexpr (BITS == 32) {return 23;}
			else if constexpr (BITS == 64) {return 52;}
			else static_assert(BITS == 32, "FloatingPointBits: unsuported number of bits");
		}();
		
		static constexpr int_type EXPONENT_BITS = []() {
			if constexpr (BITS == 32) {return 8;}
			else if constexpr (BITS == 64) {return 11;}
			else static_assert(BITS == 32, "FloatingPointBits: unsuported number of bits");
		}();
		static_assert(SIGN_BITS + MANTISSA_BITS + EXPONENT_BITS == BITS, "FloatingPointBits: inconsistent number of bits");

		static constexpr int_type EXPONENT_BIAS = []() {
			if constexpr (BITS == 32) {return 127;}
			else if constexpr (BITS == 64) {return 1023;}
			else static_assert(BITS == 32, "FloatingPointBits: unsuported number of bits");
		}();


		///////////////////////////////////////////////////////////////
		/// Get bit masks for each field
		///////////////////////////////////////////////////////////////
		static constexpr uint_type SIGN_MASK     =   uint_type(1) << (BITS - 1);                           //only need to shift 1 into place
		static constexpr uint_type EXPONENT_MASK = ((uint_type(1) << EXPONENT_BITS) - 1) << MANTISSA_BITS; //set correct number of ones, then shift into place
		static constexpr uint_type MANTISSA_MASK =  (uint_type(1) << MANTISSA_BITS) - 1;                    //only need to set the correct number of ones
		static_assert(SIGN_MASK + EXPONENT_MASK + MANTISSA_MASK == static_cast<uint_type>(-1), "FloatingPointBits: inconsistent masks");
		static_assert((SIGN_MASK & EXPONENT_MASK) == 0, "FloatingPointBits: inconsistent masks");
		static_assert((SIGN_MASK & MANTISSA_MASK) == 0, "FloatingPointBits: inconsistent masks");
		static_assert((MANTISSA_MASK & EXPONENT_MASK) == 0, "FloatingPointBits: inconsistent masks");

		///////////////////////////////////////////////////////////////
		/// Store as unsigned integer
		///////////////////////////////////////////////////////////////
		uint_type  i;

		///////////////////////////////////////////////////////////////
		/// Constructors
		///////////////////////////////////////////////////////////////
		constexpr FloatingPointBits() noexcept : i(0) {};
		// constexpr explicit FloatingPointBits(float_type val) noexcept : i(std::bit_cast<uint_type>(val)) {};
		// constexpr explicit FloatingPointBits(uint_type  val) noexcept : i(val) {};



		//fallback constructor
		template<typename T>
			requires (sizeof(T) == sizeof(i) and sizeof(T)*8==BITS)
		constexpr explicit FloatingPointBits(T val) noexcept : i(std::bit_cast<uint_type>(val)) {}

		// template<std::signed_integral Mantissa_t, int OFFSET>
		// 	requires(sizeof(Mantissa_t)*8 == BITS)
		// explicit FloatingPointBits(FixedPoint<Mantissa_t,OFFSET> val) noexcept : f(std::bit_cast<float_type>(val.mantissa)) {};

		// template<std::integral T>
		// 	requires (sizeof(T)*8==BITS)
		// explicit FloatingPointBits(T val) noexcept : i(std::bit_cast<uint_type>(val)) {};

		///////////////////////////////////////////////////////////////
		/// Type casts
		///////////////////////////////////////////////////////////////
		constexpr explicit operator uint_type() const noexcept {return i;}
		constexpr explicit operator float_type() const noexcept {return std::bit_cast<float_type>(i);}
		constexpr explicit operator int_type() const noexcept {return std::bit_cast<int_type>(i);}
		constexpr float_type to_float() const noexcept {return std::bit_cast<float_type>(i);}

		///////////////////////////////////////////////////////////////
		/// Bit setting
		///////////////////////////////////////////////////////////////
		constexpr void set_sign(const bool negative) noexcept
		{
			if (negative) {i |= SIGN_MASK;}
			else {i &= ~SIGN_MASK;}
		}
		
		constexpr void set_exponent(const uint_type exp) noexcept
		{
			i &= ~EXPONENT_MASK; //clear bits
			i |= (exp << MANTISSA_BITS) & EXPONENT_MASK; //shift bits into place and set
		}

		constexpr void set_exponent_from_power_of_2(const int_type power) noexcept
		{
			set_exponent(power + EXPONENT_BIAS);
		}

		constexpr void set_mantissa(const uint_type mantissa) noexcept
		{
			i &= ~MANTISSA_MASK; //clear bits
			i |= mantissa & MANTISSA_MASK; //set bits
		}

		///////////////////////////////////////////////////////////////
		/// Bit views
		///////////////////////////////////////////////////////////////
		constexpr uint_type sign() const noexcept
		{
			return (i & SIGN_MASK) >> BITS - 1;
		}

		// extract or set the exponent as it appears in the bits
		constexpr uint_type exponent() const noexcept
		{
			return (i & EXPONENT_MASK) >> MANTISSA_BITS;
		}
		

		// extract exponent as the power for scientific notation
		constexpr int_type exponent_actual() const noexcept
		{
			return static_cast<int_type>(exponent()) - EXPONENT_BIAS;
		}

		// extract or set the mantissa bits
		constexpr uint_type mantissa() const noexcept
		{
			return i & MANTISSA_MASK;
		}
		

		// get bits as a string for debugging
		std::string to_string() const noexcept
		{
			std::string raw = std::bitset<BITS>(i).to_string();
			std::string result;
			for (int i=0; i<BITS/8; i++) {
				result += raw.substr(8*i, 8) + " ";
			}
			return result;
		}

		// get bytes in reverse order for big endian
		uint_type big_endian() const noexcept
		{
			const int N = sizeof(uint_type); //number of bytes
			uint_type result(0);
			uint_type mask = (uint_type(1) << 8) - 1;

			int shift = BITS - 8;
			for (int n=0; n<N; n++) {
				uint_type byte = i & mask;
				mask <<= 8;

				if (shift>0) {byte <<= shift;}
				else {byte >>= -shift;}
				shift-=16;
				
				result |= byte;

			}

			return result;
		}
	};


	//print bytes with colors for each part of the float
	template<int nBits>
	std::ostream& operator<<(std::ostream& os, FloatingPointBits<nBits> bits)
	{
		//set colors
		const char* DEFAULT = "\033[0m";
		const char* SIGN_COLOR = "\033[1;33m";
		const char* EXP_COLOR = "\033[1;35m";
		const char* MANTISSA_COLOR = "\033[1;32m";

		//get raw bits as string
		std::string raw = std::bitset<bits.BITS>(bits.i).to_string();
		
		//get locations of field changes
		int start_exponent = bits.SIGN_BITS;
		int start_mantissa = bits.SIGN_BITS + bits.EXPONENT_BITS;

		for (int i=0; i<bits.BITS; i++) {
			if (i>0 and i%8==0) {os << " ";} //space between bytes

			if (i<start_exponent) {
				//sign bit
				os << SIGN_COLOR << raw[i] << DEFAULT;
			}
			else if (i<start_mantissa)
			{
				//exponent bits
				os << EXP_COLOR << raw[i] << DEFAULT;
			}
			else
			{
				//mantissa bits
				os << MANTISSA_COLOR << raw[i] << DEFAULT;
			}
		}
		return os;
	}


}