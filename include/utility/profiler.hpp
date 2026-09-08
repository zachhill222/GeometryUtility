#pragma once

#include "utility/macros.hpp"

#ifdef GUTIL_PROFILE
	#define GUTIL_PROFILE_FUNCTION() \
		static ::gutil::FunctionProfiler GUTIL_CONCAT(_gutil_profiler_,__LINE__){__func__, __FILE__, __LINE__}; \
		auto GUTIL_CONCAT(_gutil_profiler_guard_,__LINE__) = GUTIL_CONCAT(_gutil_profiler_,__LINE__).time()
#else
	#define GUTIL_PROFILE_FUNCTION()
#endif

namespace gutil {

	struct FunctionProfiler {

		struct ThreadStats {
			std::atomic<int64_t> total_ns{0};
			std::atomic<size_t>  calls{0};
		};

		std::string_view name;
		std::string_view file;
		int line;

		mutable std::mutex registry_mutex;
		mutable std::vector<ThreadStats*> registry;

		FunctionProfiler(std::string_view n, std::string_view f, int l) noexcept;

		// keyed by 'this' -- each thread gets one map, but each distinct FunctionProfiler
		// instance gets its own entry within it, so different profilers on the same
		// thread never share an accumulator.
		ThreadStats& this_thread_stats() const noexcept {
			static thread_local std::unordered_map<const FunctionProfiler*, ThreadStats> per_instance;
			auto it = per_instance.find(this);
			if (it != per_instance.end()) { return it->second; }

			auto [inserted, ok] = per_instance.try_emplace(this);
			ThreadStats& stats = inserted->second;
			{
				std::lock_guard<std::mutex> lock(registry_mutex);
				registry.push_back(&stats);
			}
			return stats;
		}

		struct ScopedTimer {
			ThreadStats& stats;
			std::chrono::steady_clock::time_point start;
			explicit ScopedTimer(ThreadStats& s) noexcept : stats(s), start(std::chrono::steady_clock::now()) {}
			~ScopedTimer() noexcept {
				auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
					std::chrono::steady_clock::now() - start).count();
				stats.total_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
				stats.calls.fetch_add(1, std::memory_order_relaxed);
			}
		};

		[[nodiscard]] ScopedTimer time() const noexcept { return ScopedTimer{this_thread_stats()}; }

		[[nodiscard]] double total_seconds() const noexcept {
			std::lock_guard<std::mutex> lock(registry_mutex);
			int64_t sum = 0;
			for (auto* s : registry) {sum += s->total_ns.load(std::memory_order_relaxed);}
			return sum * 1e-9;
		}
		[[nodiscard]] size_t total_calls() const noexcept {
			std::lock_guard<std::mutex> lock(registry_mutex);
			size_t sum = 0;
			for (auto* s : registry) {sum += s->calls.load(std::memory_order_relaxed);}
			return sum;
		}
	};

	inline std::mutex profiler_list_mutex;
	inline std::vector<FunctionProfiler*> function_profiler_list;

	inline FunctionProfiler::FunctionProfiler(std::string_view n, std::string_view f, int l) noexcept : name(n), file(f), line(l) {
		std::lock_guard<std::mutex> lock(profiler_list_mutex);
		function_profiler_list.push_back(this);
	}

	inline void print_all_profiles() {
		std::lock_guard<std::mutex> lock(profiler_list_mutex);
		for (auto* p : function_profiler_list) {
			printf("%-30s (%s:%d): calls=%-8zu total=%.6fs\n",
				std::string(p->name).c_str(), std::string(p->file).c_str(), p->line,
				p->total_calls(), p->total_seconds());
		}
	}
}