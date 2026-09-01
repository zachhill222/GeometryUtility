#pragma once

#include "utility/utility.hpp"

#include "threads/thread_pool.hpp"

#include <iterator>
#include <concepts>

namespace gutil {


	/////////////////////////////////////////////////////////////////////////
	/// A struct to help partition an index range across OpenMP threads.
	/// This is designed to reduce the repeated logic of something like this
	/// when processing some list (say std::vector<T>).
	///
	///	#pragma omp parallel
	/// {
	/// 	size_t n_idx_per_thread  = list.size()/omp_get_num_threads();
	/// 	size_t this_thread_start = n_idx_per_thread * omp_get_thread_num()
	/// 	size_t this_thread_end   = (omp_get_thread_num() == omp_get_num_threads()-1) ? 
	/// 									end : this_thread_start+n_idx_per_thread;
	/// 	#pragma omp for
	/// 	for (size_t idx=this_thread_start; idx<this_thread_end; ++idx) {...}
	/// }
	///
	/// Instead, we may use the struct. With pragmas so the program runs with or without OpenMP.
	///
	/// GUTIL_OMP(parallel)
	/// {
	///		gutil::OmpIndexRange range(list.size());
	///		GUTIL_OMP(for)
	///		for (size_t idx=range.begin; idx<range.end; ++idx) {...}
	/// }
	///
	/// We may use the iterator version as well.
	///
	/// GUTIL_OMP(parallel)
	/// {
	///		gutil::OmpIteratorRange range(list.begin(), list.end());
	///		GUTIL_OMP(for)
	///		for (auto it=range.begin; it!=range.end; ++it) {...}
	/// }
	///
	/// Note that both OmpIndexRange and OmpIteratorRange track the current
	/// executing thread, the total number of OpenMP threads, and the number of
	/// entries to be processed by the current thread.
	/////////////////////////////////////////////////////////////////////////
	template<std::integral T>
	struct OmpIndexRange {
		T begin;			//this thread begin of range
		T end;				//this thread end of range
		T count;			//the size of this thread's range
		const T tid;		//this thread's openmp number
		const T n_threads;	//total number of openmp threads
		
		//partition the index range into n_threads contiguous blocks
		OmpIndexRange(T global_count) : 
			tid{GUTIL_OMP_TERNARY(static_cast<T>(omp_get_thread_num()), 0)},
			n_threads{GUTIL_OMP_TERNARY(static_cast<T>(omp_get_num_threads()), 1)} {
			
			const T n_idx_per_thread = global_count/n_threads;
			begin 					 = tid*n_idx_per_thread;
			end   					 = (tid==n_threads-1) ? global_count : begin + n_idx_per_thread;
			count 					 = end - begin;
		}
	};


	template<std::random_access_iterator I>
	struct OmpIteratorRange {
		I begin;				//this thread start of range
		I end;					//this thread end of range
		size_t count;			//the size of this thread's range 
		const size_t tid;		//this thread's openmp number
		const size_t n_threads;	//total number of openmp threads
		
		//partition the index range into n_threads contiguous blocks
		OmpIteratorRange(I global_start, I global_end) : 
			tid{GUTIL_OMP_TERNARY(static_cast<size_t>(omp_get_thread_num()), 0)},
			n_threads{GUTIL_OMP_TERNARY(static_cast<size_t>(omp_get_num_threads()), 1)} {
			
			const size_t total 			  = static_cast<size_t>(std::distance(global_start,global_end));
			const size_t n_idx_per_thread = total/n_threads;
			begin 						  = global_start + tid*n_idx_per_thread;
			end   						  = (tid==n_threads-1) ? global_end : begin + n_idx_per_thread;
			count 						  = static_cast<size_t>(std::distance(begin,end));
		}
	};
}
