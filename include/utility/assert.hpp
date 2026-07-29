#pragma once

#include "utility/extra.hpp"

#include <iostream>
#include <string_view>
#include <source_location>

#ifdef __cpp_lib_stacktrace
	#include <stacktrace>
	#define GUTIL_HAS_STACKTRACE(...) __VA_ARGS__
#else
	#define GUTIL_HAS_STACKTRACE(...)
#endif

#ifndef NDEBUG
	#define GUTIL_ASSERT(cond) gutil::gutil_assert(cond, #cond);
#else
	#define GUTIL_ASSERT(cond)
#endif

#define GUTIL_ABORT(why) gutil::gutil_abort(#why);

namespace gutil {
	inline void gutil_assert(bool cond, std::string_view condition_str,
			std::source_location loc = std::source_location::current()
			GUTIL_HAS_STACKTRACE(,std::stacktrace trace = std::stacktrace::current()) ) noexcept {
		if (cond) { return; }

		Logger::error("GUTIL_ASSERT : ",condition_str,"\n",
						 "\tat ",loc.file_name()," line: ",loc.line()," col: ",loc.column(),"\n",
						 "\tin ",loc.function_name()
						 GUTIL_HAS_STACKTRACE(,"\tstacktrace:", trace));
		std::abort();
	}

	inline void gutil_abort(std::string_view condition_str,
			std::source_location loc = std::source_location::current()
			GUTIL_HAS_STACKTRACE(,std::stacktrace trace = std::stacktrace::current()) ) noexcept {
		Logger::error("GUTIL_ABORT : ",condition_str,"\n",
						 "\tat ",loc.file_name()," : ",loc.line()," : ",loc.column(),"\n",
						 "\tin ",loc.function_name()
						 GUTIL_HAS_STACKTRACE(,"\tstacktrace:", trace));
		std::abort();
	}
}