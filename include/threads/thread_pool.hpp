#pragma once

#include "utility/utility.hpp"

#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <memory>
#include <atomic>
#include <condition_variable>
#include <type_traits>


// define GUTIL_DISABLE_THREAD_POOL at compile time to run tasks
// submitted to the pool immediately by the submitting thread.
// this can be helpful for debugging


namespace gutil {

	//////////////////////////////////////////////////////////////////////////////
	/// A simple, fixed-size thread pool. Tasks are started in the order they
	/// were submitted, with up to n_threads() running concurrently.
	///
	/// Two ways to wait for submitted work:
	///   - wait_idle(): waits for ALL currently queued/running work to finish.
	///     Safe to call from the ORIGINAL, external caller only -- never from
	///     a task running on a pool worker (see wait_for below for that case).
	///   - wait_for(handle): waits for one SPECIFIC submitted task, cooperatively
	///     draining the queue in the meantime. Safe to call from anywhere,
	///     including from within another task -- this is what makes recursive,
	///     nested submission (fork-join style algorithms) deadlock-free: a
	///     thread that's "waiting" is never truly unavailable, since it can
	///     always pick up and run other queued work itself while it waits.
	//////////////////////////////////////////////////////////////////////////////
	struct ThreadPool {

		/////////////////////////////////////////////////////////////////////////
		/// Aliases
		/////////////////////////////////////////////////////////////////////////
		using Handle = std::shared_ptr<std::atomic<bool>>;	//completion flag for one submitted task


		/////////////////////////////////////////////////////////////////////////
		/// Constructors
		/////////////////////////////////////////////////////////////////////////
		explicit ThreadPool(size_t n_threads = std::thread::hardware_concurrency()) {
			#ifndef GUTIL_DISABLE_THREAD_POOL
				workers_.reserve(n_threads);
				for (size_t i=0; i<n_threads; ++i) {
					workers_.emplace_back([this, i]() { worker_loop(static_cast<int>(i)); });
				}
			#else
				GUTIL_LOG("Using gutil::ThreadPool at ", this, " as a single thread");
			#endif
		}

		~ThreadPool() {
			{
				std::lock_guard<std::mutex> lock(queue_mtx_);
				stop_ = true;
			}
			cv_.notify_all();
			for (auto& t : workers_) {if (t.joinable()) {t.join();}}
		}

		ThreadPool(const ThreadPool&) = delete;
		ThreadPool& operator=(const ThreadPool&) = delete;
		ThreadPool(ThreadPool&&) = delete;
		ThreadPool& operator=(ThreadPool&&) = delete;


		/////////////////////////////////////////////////////////////////////////
		/// Dispatch work
		///
		/// Valid callable signatures: void(Args...) or void(int, Args...), where
		/// the leading int (if present) is the thread number in [0,n_threads()).
		/// Not marked [[nodiscard]] -- most call sites don't need the returned
		/// Handle at all and should be free to ignore it with no warning; only
		/// callers that need wait_for() should capture it.
		/////////////////////////////////////////////////////////////////////////
		template<typename FunctionType, typename... Args>
		Handle submit(FunctionType&& f, Args&&... args) {
			constexpr bool PASS_THREAD_NUM = std::is_invocable_v<std::decay_t<FunctionType>, int, std::decay_t<Args>...>;
			static_assert(PASS_THREAD_NUM || std::is_invocable_v<std::decay_t<FunctionType>, std::decay_t<Args>...>,
				"submit: callable must be invocable with decayed (by-value) argument types -- "
				"wrap in std::ref() or capture as a reference in the lambda if the task needs to mutate a caller-scope variable.");

			auto done = std::make_shared<std::atomic<bool>>(false);

			#ifdef GUTIL_DISABLE_THREAD_POOL
				if constexpr (PASS_THREAD_NUM) { f(0, std::forward<Args>(args)...); }
				else {f(std::forward<Args>(args)...); }
				done->store(true);
				return done;
			#else

			if (n_threads()==0) {
				if constexpr (PASS_THREAD_NUM) { f(0, std::forward<Args>(args)...); }
				else {f(std::forward<Args>(args)...); }
				done->store(true);
				return done;
			}

			++outstanding_;
			auto packed_function = [this, f = std::forward<FunctionType>(f), ...args = std::forward<Args>(args), done](int thread_number) mutable {
				if constexpr (PASS_THREAD_NUM) { f(thread_number, std::forward<Args>(args)...); }
				else { f(std::forward<Args>(args)...); }

				done->store(true);
				--outstanding_;
				//notify unconditionally: wakes both wait_idle() (checking outstanding_==0)
				//and any wait_for(handle) callers (checking their own handle) without
				//either one needing to know about the other
				{ std::lock_guard<std::mutex> lock(queue_mtx_); }
				done_cv_.notify_all();
			};

			{
				std::lock_guard<std::mutex> lock(queue_mtx_);
				tasks_.emplace(std::move(packed_function));
			}
			cv_.notify_one();
			return done;
			#endif
		}


		/////////////////////////////////////////////////////////////////////////
		/// Waiting
		/////////////////////////////////////////////////////////////////////////

		/// Wait for ALL currently outstanding work to finish. Do not call this
		/// from within a task running on a pool worker -- use wait_for() there
		/// instead, or this can deadlock against sibling work sharing the same
		/// global outstanding_ count.
		void wait_idle() {
			std::unique_lock<std::mutex> lock(queue_mtx_);
			done_cv_.wait(lock, [this] { return outstanding_.load() == 0; });
		}

		/// Wait for one specific submitted task, cooperatively draining the
		/// queue in the meantime. Safe to call from anywhere, including from
		/// within another task -- this is the deadlock-free primitive for
		/// recursive, nested submission.
		void wait_for(const Handle& h) {
			std::unique_lock<std::mutex> lock(queue_mtx_);
			while (!h->load()) {
				if (!tasks_.empty()) {
					auto task = std::move(tasks_.front());
					tasks_.pop();
					lock.unlock();
					task(-1);	//-1: executed cooperatively, not by a dedicated worker
					lock.lock();
				} else {
					done_cv_.wait(lock, [this, &h]{ return h->load() || !tasks_.empty(); });
				}
			}
		}


		/////////////////////////////////////////////////////////////////////////
		/// Queries
		/////////////////////////////////////////////////////////////////////////
		[[nodiscard]] size_t n_threads() const noexcept { return workers_.size(); }
		[[nodiscard]] size_t n_active_tasks() const noexcept { return outstanding_.load(); }


	private:
		/////////////////////////////////////////////////////////////////////////
		/// Worker loop
		/////////////////////////////////////////////////////////////////////////
		void worker_loop(const int thread_number) {
			while (true) {
				std::function<void(int)> task;
				{
					std::unique_lock<std::mutex> lock(queue_mtx_);
					cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
					if (stop_ && tasks_.empty()) {return;}
					task = std::move(tasks_.front());
					tasks_.pop();
				}
				task(thread_number);
			}
		}


		/////////////////////////////////////////////////////////////////////////
		/// Storage
		/////////////////////////////////////////////////////////////////////////
		std::vector<std::thread> 				workers_;
		std::queue<std::function<void(int)>> 	tasks_;
		std::mutex 								queue_mtx_;
		std::condition_variable 				cv_;			//wakes workers when new work is queued
		std::condition_variable 				done_cv_;		//wakes wait_idle()/wait_for() callers on task completion
		std::atomic<size_t> 					outstanding_{0};
		bool 									stop_ = false;
	};
}



	
	// struct ThreadPool {
	// 	/// On construction, prepare the worker threads
	// 	explicit ThreadPool(size_t n_threads = std::thread::hardware_concurrency()) {
	// 		#ifndef GUTIL_DISABLE_THREAD_POOL
	// 		workers_.reserve(n_threads);
	// 		for (size_t i=0; i<n_threads; ++i) {
	// 			workers_.emplace_back([this, i]() { worker_loop(static_cast<int>(i)); });
	// 		}
	// 		#else
	// 		GUTIL_LOG("Using gutil::ThreadPool at ", this, " as a single thread");
	// 		#endif
	// 	}

	// 	/// On destruction, wait for all tasks to end and join the worker threads
	// 	~ThreadPool() {
	// 		{
	// 			std::lock_guard<std::mutex> lock(queue_mtx_);
	// 			stop_ = true;
	// 		}
	// 		cv_.notify_all();
	// 		for (auto& t : workers_) {if (t.joinable()) {t.join();}}
	// 	}

	// 	/// Threads are not copyable or movable
	// 	ThreadPool(const ThreadPool&) = delete;
	// 	ThreadPool& operator=(const ThreadPool&) = delete;
	// 	ThreadPool(ThreadPool&&) = delete;
	// 	ThreadPool& operator=(ThreadPool&&) = delete;

	// 	/// Allow an external thread to dispatch work to the pool
	// 	template<typename FunctionType, typename... Args>
	// 	void submit(FunctionType&& f, Args&&... args) {
	// 		//determine if the function need the thread number
	// 		constexpr bool PASS_THREAD_NUM = std::is_invocable_v<std::decay_t<FunctionType>, int, std::decay_t<Args>...>;


	// 		//put task onto the queue, note that args are generally copied into different threads.
	// 		static_assert(PASS_THREAD_NUM || std::is_invocable_v<std::decay_t<FunctionType>, std::decay_t<Args>...>,
	// 			"submit: callable must be invocable with decayed (by-value) argument types -- "
	// 			"wrap in std::ref() or capture as a reference in the lambda if the task needs to mutate a caller-scope variable.");
			
	// 		#ifdef GUTIL_DISABLE_THREAD_POOL
	// 			if constexpr (PASS_THREAD_NUM) { f(0, std::forward<Args>(args)...);}
	// 			else {f(std::forward<Args>(args)...);}
	// 			return;
	// 		#else

	// 		if (n_threads()==0) {
	// 			if constexpr (PASS_THREAD_NUM) { f(0, std::forward<Args>(args)...); return;}
	// 			else {f(std::forward<Args>(args)...); return;}
	// 		}

	// 		++outstanding_;
	// 		{

	// 			if constexpr (PASS_THREAD_NUM) {
	// 				auto packed_function = [this, f = std::forward<FunctionType>(f), ...args = std::forward<Args>(args)](int thread_number) mutable {
	// 					f(thread_number, std::forward<Args>(args)...);
	// 					if (--outstanding_ == 0) {
	// 						std::lock_guard<std::mutex> lock(queue_mtx_);
	// 						done_cv_.notify_all();
	// 					}
	// 				};
	// 				std::lock_guard<std::mutex> lock(queue_mtx_);
	// 				tasks_.emplace(std::move(packed_function));
	// 			}
	// 			else {
	// 				//append a dummy int argument so the worker thread doesn't have to decide if it needs to pass the thread number
	// 				auto packed_function = [this, f = std::forward<FunctionType>(f), ...args = std::forward<Args>(args)](int) mutable {
	// 					f(std::forward<Args>(args)...);
	// 					if (--outstanding_ == 0) {
	// 						std::lock_guard<std::mutex> lock(queue_mtx_);
	// 						done_cv_.notify_all();
	// 					}
	// 				};

	// 				std::lock_guard<std::mutex> lock(queue_mtx_);
	// 				tasks_.emplace(std::move(packed_function));
	// 			}
	// 		}
	// 		cv_.notify_one();
	// 		#endif
	// 	}

	// 	/// Allow the calling thread to wait until all tasks are done
	// 	/// don't call from a worker thread.
	// 	void wait_idle() {
	// 		std::unique_lock<std::mutex> lock(queue_mtx_);
	// 		done_cv_.wait(lock, [this] { return outstanding_.load() == 0; });
	// 	}

	// 	[[nodiscard]] size_t n_threads() const noexcept { return workers_.size(); }

	// private:
	// 	void worker_loop(const int thread_number) {
	// 		while (true) {
	// 			std::function<void(int)> task;
	// 			{
	// 				std::unique_lock<std::mutex> lock(queue_mtx_);
	// 				cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
	// 				if (stop_ && tasks_.empty()) {return;}
	// 				task = std::move(tasks_.front());
	// 				tasks_.pop();
	// 			}
	// 			task(thread_number);
	// 		}
	// 	}

	// 	std::vector<std::thread> workers_;
	// 	std::queue<std::function<void(int)>> tasks_;
	// 	std::mutex queue_mtx_;
	// 	std::condition_variable cv_;		//get thread to start task
	// 	std::condition_variable done_cv_;	//allow a calling thread to wait until all tasks are done
	// 	std::atomic<size_t> outstanding_{0};
	// 	bool stop_ = false;
	// };

// }