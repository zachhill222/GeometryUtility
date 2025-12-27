#pragma once

#include <cassert>
#include <mutex>
#include <shared_mutex>
#include <atomic>

#ifndef GUTIL_OCTREE_BUFFER_USE_STACK
	#define GUTIL_OCTREE_BUFFER_USE_STACK true
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Single producer single consumer queue
////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace gutil {
	/////////////////////////////////////////////////
	/// A queue data structure for use in multithreaded applications.
	/// Making CAPACITY too large can cause stack overflow, especially when
	/// each thread gets its own queue. Sometimes the queue will get full,
	/// so data should be inserted with while(!try_push(data)). The number of
	/// unssuccessful pushes are tracked in buffer_bumps. The buffer/queue can
	/// be moved to the heap if it needs to be larger, but I have found that
	/// having a large buffer doesn't necessarily lead to performance gains.
	///
	/// @tparam Data_t   The type of data to be stored 
	/// @tparam CAPACITY The maximum number of elements that can be stored in the
	///                  queue/buffer at a time. Should be a power of 2 for faster
	///                  mod computation. If this is changed to be re-sizable,
	///                  the mod computation may need to be implemented by hand
	///                  because the compiler won't know it is a power of 2.
	///
	/// @tparam USE_STACK  The buffer can be allocated on either the heap or the stack.
	/////////////////////////////////////////////////
	template<typename Data_t, size_t CAPACITY=2048, bool USE_STACK=true>
	struct ThreadLocalQueue {
		//thread A may only need head while thread B may only need tail.
		//setting alignas(64) keeps one thread from loading both but only using one.
		alignas(64) std::atomic<size_t> head{0};
		alignas(64) std::atomic<size_t> tail{0};
		std::atomic<size_t> count{0};
		size_t queue_id;

		#if USE_STACK
			Data_t queue[CAPACITY];
			virtual ~ThreadLocalQueue() {}
		#else
			Data_t* queue = new Data_t[CAPACITY];
			virtual ~ThreadLocalQueue() {delete[] queue;}
		#endif

		std::atomic<size_t> buffer_bumps{0};

		bool try_push(Data_t&& item) {
			size_t current_tail = tail.load(std::memory_order_acquire);
			size_t next_tail = (current_tail + 1) % CAPACITY;

			if (next_tail == head.load(std::memory_order_acquire)) {
				buffer_bumps.fetch_add(1, std::memory_order_relaxed); //buffer_bumps is mostly of profiling
				return false;  // Queue full
			}

			queue[current_tail] = std::move(item);
			tail.store(next_tail);
			count++;
			return true;
		}

		bool try_pop(Data_t& item) {
			size_t current_head = head.load(std::memory_order_acquire);

			if (current_head == tail.load(std::memory_order_acquire)) {
				return false;  // Queue empty
			}

			item = queue[current_head];
			head.store((current_head + 1) % CAPACITY, std::memory_order_release);
			count.fetch_add(-1, std::memory_order_release);
			return true;
		}

		template<typename DataCondition>
		bool try_pop(Data_t& item) {
			size_t current_head = head.load(std::memory_order_acquire);

			if (current_head == tail.load(std::memory_order_acquire)) {
				return false;  // Queue empty
			} else if (!DataCondition(queue[current_head])) {
				return false; // next data is not valid
			}

			head.store((current_head + 1) % CAPACITY, std::memory_order_release);
			count.fetch_add(-1, std::memory_order_release);
			return true;
		}

		bool empty(std::memory_order order = std::memory_order_seq_cst) const {
			return head.load(order) == tail.load(order);
		}
	};
}