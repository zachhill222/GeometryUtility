#pragma once

#include <span>
#include <iostream>
#include <string>

namespace gutil
{
	template<typename T>
	void print_to_stream(std::ostream& os, std::span<const T> data, const std::string delimiter) {
		for (const T val : data) {
			os << val << delimiter;
		}
	}

	template<int DIM, typename T> requires (DIM>0)
	void print_to_stream(std::ostream& os, std::span<const T, DIM> data, const std::string delimiter) {
		for (int i=0; i<DIM-1; ++i) {
			os << data[i] << delimiter;
		}
		os << data[DIM-1];
	}
}