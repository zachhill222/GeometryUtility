#pragma once

#include "utility/utility.hpp"

#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>


// define GUTIL_ENABLE_THREAD_POOL at compile time to enable the thread pool
// actually doing anthing


namespace gutil {
	
	struct ThreadPool {
		/// On construction, prepare the worker threads
		explicit ThreadPool(size_t n_threads = std::thread::hardware_concurrency()) {
			// if (n_threads==0) {n_threads=1;}	//always have at least one thread
			workers_.reserve(n_threads);
			for (size_t i=0; i<n_threads; ++i) {
				workers_.emplace_back([this, i]() { worker_loop(static_cast<int>(i)); });
			}
		}

		/// On destruction, wait for all tasks to end and join the worker threads
		~ThreadPool() {
			{
				std::lock_guard<std::mutex> lock(queue_mtx_);
				stop_ = true;
			}
			cv_.notify_all();
			for (auto& t : workers_) {if (t.joinable()) {t.join();}}
		}

		/// Threads are not copyable or movable
		ThreadPool(const ThreadPool&) = delete;
		ThreadPool& operator=(const ThreadPool&) = delete;
		ThreadPool(ThreadPool&&) = delete;
		ThreadPool& operator=(ThreadPool&&) = delete;

		/// Allow an external thread to dispatch work to the pool
		template<typename FunctionType, typename... Args>
		void submit(FunctionType&& f, Args&&... args) {
			//determine if the function need the thread number
			constexpr bool PASS_THREAD_NUM = std::is_invocable_v<std::decay_t<FunctionType>, int, std::decay_t<Args>...>;


			//put task onto the queue, note that args are generally copied into different threads.
			static_assert(PASS_THREAD_NUM || std::is_invocable_v<std::decay_t<FunctionType>, std::decay_t<Args>...>,
				"submit: callable must be invocable with decayed (by-value) argument types -- "
				"wrap in std::ref() or capture as a reference in the lambda if the task needs to mutate a caller-scope variable.");
			
			#ifndef GUTIL_ENABLE_THREAD_POOL
				if constexpr (PASS_THREAD_NUM) { f(0, std::forward<Args>(args)...);}
				else {f(std::forward<Args>(args)...);}
				return;
			#else

			if (n_threads==0) {
				if constexpr (PASS_THREAD_NUM) { f(0, std::forward<Args>(args)...); return;}
				else {f(std::forward<Args>(args)...); return;}
			}

			++outstanding_;
			{

				if constexpr (PASS_THREAD_NUM) {
					auto packed_function = [this, f = std::forward<FunctionType>(f), ...args = std::forward<Args>(args)](int thread_number) mutable {
						f(thread_number, std::forward<Args>(args)...);
						if (--outstanding_ == 0) {
							std::lock_guard<std::mutex> lock(queue_mtx_);
							done_cv_.notify_all();
						}
					};
					std::lock_guard<std::mutex> lock(queue_mtx_);
					tasks_.emplace(std::move(packed_function));
				}
				else {
					//append a dummy int argument so the worker thread doesn't have to decide if it needs to pass the thread number
					auto packed_function = [this, f = std::forward<FunctionType>(f), ...args = std::forward<Args>(args)](int) mutable {
						f(std::forward<Args>(args)...);
						if (--outstanding_ == 0) {
							std::lock_guard<std::mutex> lock(queue_mtx_);
							done_cv_.notify_all();
						}
					};

					std::lock_guard<std::mutex> lock(queue_mtx_);
					tasks_.emplace(std::move(packed_function));
				}
			}
			cv_.notify_one();
			#endif
		}

		/// Allow the calling thread to wait until all tasks are done
		/// don't call from a worker thread.
		void wait_idle() {
			std::unique_lock<std::mutex> lock(queue_mtx_);
			done_cv_.wait(lock, [this] { return outstanding_.load() == 0; });
		}

		[[nodiscard]] size_t n_threads() const noexcept { return workers_.size(); }

	private:
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

		std::vector<std::thread> workers_;
		std::queue<std::function<void(int)>> tasks_;
		std::mutex queue_mtx_;
		std::condition_variable cv_;		//get thread to start task
		std::condition_variable done_cv_;	//allow a calling thread to wait until all tasks are done
		std::atomic<size_t> outstanding_{0};
		bool stop_ = false;
	};

}