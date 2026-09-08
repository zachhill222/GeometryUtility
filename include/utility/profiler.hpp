#pragma once
#include "utility/macros.hpp"
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>
#include <cstdio>

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
		size_t index;   // this profiler's own, fixed position in function_profiler_list, set once at construction

		FunctionProfiler(std::string_view n, std::string_view f, int l) noexcept;

		// heap-allocated, deliberately never freed -- a thread's own stats must
		// survive even after that thread terminates, since aggregation may run
		// long after a worker thread has already joined. Matches the same
		// "never explicitly destroyed, program-lifetime" pattern already used
		// for FunctionProfiler instances themselves.
		//
		// deque (not vector): ThreadStats contains std::atomic members, which
		// are neither copyable nor movable -- vector::resize would require
		// relocating existing elements on growth; deque never does.
		static std::deque<ThreadStats>& this_thread_all_stats() noexcept {
			static thread_local std::deque<ThreadStats>* per_thread_stats = nullptr;
			if (!per_thread_stats) {
				per_thread_stats = new std::deque<ThreadStats>();
				std::lock_guard<std::mutex> lock(thread_stats_registry_mutex());
				thread_stats_registry().push_back(per_thread_stats);
			}
			return *per_thread_stats;
		}

		ThreadStats& this_thread_stats() const noexcept {
			auto& dq = this_thread_all_stats();
			if (index >= dq.size()) { dq.resize(index+1); }
			return dq[index];
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
			std::lock_guard<std::mutex> lock(thread_stats_registry_mutex());
			int64_t sum = 0;
			for (auto* dq : thread_stats_registry()) {
				if (index < dq->size()) { sum += (*dq)[index].total_ns.load(std::memory_order_relaxed); }
			}
			return sum * 1e-9;
		}
		[[nodiscard]] size_t total_calls() const noexcept {
			std::lock_guard<std::mutex> lock(thread_stats_registry_mutex());
			size_t sum = 0;
			for (auto* dq : thread_stats_registry()) {
				if (index < dq->size()) { sum += (*dq)[index].calls.load(std::memory_order_relaxed); }
			}
			return sum;
		}

		static std::mutex& thread_stats_registry_mutex() noexcept { static std::mutex m; return m; }
		static std::vector<std::deque<ThreadStats>*>& thread_stats_registry() noexcept {
			static std::vector<std::deque<ThreadStats>*> reg;
			return reg;
		}
	};

	//define global resources for profiling (but only if they are needed)
	inline std::mutex profiler_list_mutex;
	inline std::vector<FunctionProfiler*> function_profiler_list;

	inline FunctionProfiler::FunctionProfiler(std::string_view n, std::string_view f, int l) noexcept : name(n), file(f), line(l) {
		std::lock_guard<std::mutex> lock(profiler_list_mutex);
		index = function_profiler_list.size();
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