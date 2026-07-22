#pragma once

#include "utility/extra.hpp"

#include <iostream>
#include <string_view>
#include <source_location>


#ifndef NDEBUG
	#define GUTIL_ASSERT(cond) gutil::gutil_assert(cond, #cond);
#else
	#define GUTIL_ASSERT(cond)
#endif

#define GUTIL_ABORT(why) gutil::gutil_abort(#why);

namespace gutil {
	inline void gutil_assert(bool cond, std::string_view condition_str,
			std::source_location loc = std::source_location::current() ) noexcept {
		if (cond) { return; }

		Logger::error("GUTIL_ASSERT : ",condition_str,"\n",
						 "\tat ",loc.file_name()," : ",loc.line()," : ",loc.column(),"\n",
						 "\tin ",loc.function_name());
		std::abort();
	}

	inline void gutil_abort(std::string_view condition_str,
			std::source_location loc = std::source_location::current() ) noexcept {
		Logger::error("GUTIL_ABORT : ",condition_str,"\n",
						 "\tat ",loc.file_name()," : ",loc.line()," : ",loc.column(),"\n",
						 "\tin ",loc.function_name());
		std::abort();
	}
}