#pragma once
#include "utility/macros.hpp"
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

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

		// CPU time: summed across every thread's own contribution, not wall-clock time.
		// see wall_seconds() for the (approximate) wall-clock counterpart.
		[[nodiscard]] double total_seconds() const noexcept {
			std::lock_guard<std::mutex> lock(thread_stats_registry_mutex());
			int64_t sum = 0;
			for (auto* dq : thread_stats_registry()) {
				if (index < dq->size()) { sum += (*dq)[index].total_ns.load(std::memory_order_relaxed); }
			}
			return static_cast<double>(sum) * 1e-9;
		}
		[[nodiscard]] size_t total_calls() const noexcept {
			std::lock_guard<std::mutex> lock(thread_stats_registry_mutex());
			size_t sum = 0;
			for (auto* dq : thread_stats_registry()) {
				if (index < dq->size()) { sum += (*dq)[index].calls.load(std::memory_order_relaxed); }
			}
			return sum;
		}

		// only counts threads that genuinely called THIS specific profiler at least once,
		// not every thread registered globally across every profiler
		[[nodiscard]] size_t thread_count() const noexcept {
			std::lock_guard<std::mutex> lock(thread_stats_registry_mutex());
			size_t count = 0;
			for (auto* dq : thread_stats_registry()) {
				if (index < dq->size() && (*dq)[index].calls.load(std::memory_order_relaxed) > 0) { ++count; }
			}
			return count;
		}

		// approximation: the max per-thread total time, assuming threads ran concurrently.
		// this is NOT a true wall-clock measurement of the whole parallel region -- it doesn't
		// account for threads starting/finishing at different moments, or gaps between multiple,
		// non-contiguous calls on the same thread. A true wall-clock figure would need to track
		// the actual start/end timestamps of the parallel region itself, not per-call durations.
		[[nodiscard]] double wall_seconds() const noexcept {
			std::lock_guard<std::mutex> lock(thread_stats_registry_mutex());
			int64_t max_ns = 0;
			for (auto* dq : thread_stats_registry()) {
				if (index < dq->size()) {
					max_ns = std::max(max_ns, (*dq)[index].total_ns.load(std::memory_order_relaxed));
				}
			}
			return static_cast<double>(max_ns) * 1e-9;
		}

		// matches usr/bin/time's %CPU convention: (cpu_time / wall_time) * 100.
		// ~100% means no parallelism (purely sequential); ~N*100% means N threads
		// were, on average, concurrently busy for the duration.
		[[nodiscard]] double percent_parallel() const noexcept {
			const double wall = wall_seconds();
			if (wall <= 0.0) { return 0.0; }
			return (total_seconds() / wall) * 100.0;
		}

		static std::mutex& thread_stats_registry_mutex() noexcept { static std::mutex m; return m; }
		static std::vector<std::deque<ThreadStats>*>& thread_stats_registry() noexcept {
			static std::vector<std::deque<ThreadStats>*> reg;
			return reg;
		}
	};

	//define global resources for profiling. these stay unconditional (not guarded by
	//GUTIL_PROFILE) so that downstream libraries can build their own profiling macro
	//(e.g. GV_PROFILE_FUNCTION()) against a fully-formed FunctionProfiler type without
	//needing to redeclare it themselves. GUTIL_PROFILE_FUNCTION() is the only thing
	//actually gated -- with it undefined, function_profiler_list simply stays empty.
	inline std::mutex profiler_list_mutex;
	inline std::vector<FunctionProfiler*> function_profiler_list;

	inline FunctionProfiler::FunctionProfiler(std::string_view n, std::string_view f, int l) noexcept : name(n), file(f), line(l) {
		std::lock_guard<std::mutex> lock(profiler_list_mutex);
		index = function_profiler_list.size();
		function_profiler_list.push_back(this);
	}

	inline void print_all_profiles() {
		std::lock_guard<std::mutex> lock(profiler_list_mutex);
		if (function_profiler_list.empty()) {return;}

		size_t max_name_len = 0, max_location_len = 0;
		for (auto* p : function_profiler_list) {
			max_name_len = std::max(max_name_len, p->name.size());
			max_location_len = std::max(max_location_len, p->file.size() + 1 + std::to_string(p->line).size());
		}

		constexpr size_t BUFFER = 10;
		const size_t name_width = max_name_len + BUFFER;
		const size_t location_width = max_location_len + BUFFER;
		constexpr size_t calls_width = 10, threads_width = 10, cpu_width = 12, wall_width = 12, pct_width = 12;

		std::cout << "\n" << std::left
			<< std::setw(name_width)     << "Function"
			<< std::setw(location_width) << "Location"
			<< std::setw(calls_width)    << "Calls"
			<< std::setw(threads_width)  << "Threads"
			<< std::setw(cpu_width)      << "CPU (s)"
			<< std::setw(wall_width)     << "Wall (s)"
			<< std::setw(pct_width)      << "% Parallel" << "\n";

		for (auto* p : function_profiler_list) {
			std::string location = std::string(p->file) + ":" + std::to_string(p->line);
			std::cout << std::setw(name_width) << std::string(p->name)
				<< std::setw(location_width) << location
				<< std::setw(calls_width) << p->total_calls()
				<< std::setw(threads_width) << p->thread_count()
				<< std::fixed << std::setprecision(6)
				<< std::setw(cpu_width) << p->total_seconds()
				<< std::setw(wall_width) << p->wall_seconds()
				<< std::setprecision(1)
				<< p->percent_parallel() << "%\n";
		}
		std::cout << "\n";
	}
}