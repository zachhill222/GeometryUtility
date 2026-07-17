#pragma once

#include "utility/utility.hpp"

#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>

namespace gutil {
	
	struct ThreadPool {
		/// On construction, prepare the worker threads
		explicit ThreadPool(size_t n_threads = std::thread::hardware_concurrency()) {
			if (n_threads==0) {n_threads=1;}	//always have at least one thread
			workers_.reserve(n_threads);
			for (size_t i=0; i<n_threads; ++i) {
				workers_.emplace_back([this]() { worker_loop(); });
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
		template<typename FunctionType>
		void submit(FunctionType&& f) {
			++outstanding_;
			{
				//put task onto the queue
				auto packed_function = [this, f = std::forward<FunctionType>(f)]() mutable {
					f();
					if (--outstanding_ == 0) {
						std::lock_guard<std::mutex> lock(queue_mtx_);
						done_cv_.notify_all();
					}
				};

				std::lock_guard<std::mutex> lock(queue_mtx_);
				tasks_.emplace(std::move(packed_function));
			}
			cv_.notify_one();
		}

		/// Allow the calling thread to wait until all tasks are done
		/// don't call from a worker thread.
		void wait_idle() {
			std::unique_lock<std::mutex> lock(queue_mtx_);
			done_cv_.wait(lock, [this] { return outstanding_.load() == 0; });
		}

		[[nodiscard]] size_t n_threads() const noexcept { return workers_.size(); }

	private:
		void worker_loop() {
			while (true) {
				std::function<void()> task;
				{
					std::unique_lock<std::mutex> lock(queue_mtx_);
					cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
					if (stop_ && tasks_.empty()) {return;}
					task = std::move(tasks_.front());
					tasks_.pop();
				}
				task();
			}
		}

		std::vector<std::thread> workers_;
		std::queue<std::function<void()>> tasks_;
		std::mutex queue_mtx_;
		std::condition_variable cv_;		//get thread to start task
		std::condition_variable done_cv_;	//allow a calling thread to wait until all tasks are done
		std::atomic<size_t> outstanding_{0};
		bool stop_ = false;
	};

}