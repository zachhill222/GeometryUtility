#include "gutil.hpp"

#include <iostream>

#ifdef _OPENMP
	#include <omp.h>
#endif

#ifndef DIMENSION
	#define DIMENSION 2
#endif
static_assert(DIMENSION==2 or DIMENSION==3, "test is enabled for DIMENSION 2 or 3");


// the BasicParallelOctree uses a buffer for each thread
// if these buffers are large, they should be allocated on the heap
// if these buffers are too small, they might fill up
// this defaults to true in thread_queue.hpp but is copied here for visibility.
#ifndef GUTIL_OCTREE_BUFFER_USE_STACK
	#define GUTIL_OCTREE_BUFFER_USE_STACK true
#endif



// using Scalar_t = gutil::FixedPoint<int32_t>;
using Scalar_t = double;
using Octree_t = gutil::PointOctree<DIMENSION,Scalar_t,128>;
using Index_t  = gutil::Point<DIMENSION,size_t>;
using Point_t  = gutil::Point<DIMENSION,Scalar_t>;
using Box_t    = gutil::Box<DIMENSION,Scalar_t>;

void generate_points2(Octree_t& octree, const Index_t& N) {
	assert(N>Index_t{0});

	#ifdef _OPENMP
		//if openmp is enabled and data is being inserted in parallel, then
		//the space must be allocated ahead of time.
		octree.resize(prod(N));
		#pragma omp parallel for collapse(2)
	#else
		//if openmp is not enabled, reserving space is better
		octree.resize(prod(N));
	#endif

	for (size_t i=0; i<N[0]; i++) {
		for (size_t j=0; j<N[1]; j++) {
			//push_back_async can be used even if openmp is not enabled
			//when openmp is enabled, it is the user's responsibility to ensure
			//that no duplicate data is inserted. If openmp is not enambled, then
			//push_back_async defaults to push_back. This can still be called in parallel, but
			//it locks the entire tree with a lock_guard mutex. Data can be searched in parallel
			//if no data is currently being inserted.
			[[maybe_unused]] size_t stored_idx = octree.push_back(Point_t{i,j});
		}
	}

	//if data is inserted in parallel, the buffer must be flushed to ensure
	//that it is up to date. If openmp is not enabled or octree.push_back() is called, then
	//this does not need to be called.
	octree.flush();

	//this isn't needed with precise reserve/resize usage
	//when pushing data back in parallel, it may be necessary (or more efficient) in practice to overestimate
	//the number of data entries to be added when calling resize() and then free unused space after.
	octree.shrink_to_fit();
}

void generate_points3(Octree_t& octree, const Index_t& N) {
	assert(N>Index_t{0});

	#ifdef _OPENMP
		octree.resize(prod(N));
		#pragma omp parallel for collapse(3)
	#else
		octree.resize(prod(N));
	#endif
	
	for (size_t i=0; i<N[0]; i++) {
		for (size_t j=0; j<N[1]; j++) {
			for (size_t k=0; k<N[2]; k++) {
				octree.push_back(Point_t{i,j,k});
			}
		}
	}

	octree.flush();
	octree.shrink_to_fit();
}


int main(int argc, char* argv[]) {
	#if DIMENSION==2
		Index_t N{100,100};
	#elif DIMENSION==3
		Index_t N{100,100,100};
	#endif

	if (argc>1) {N[0]=std::stoull(argv[1]);}
	if (argc>2) {N[1]=std::stoull(argv[2]);}
	if (argc>3 and DIMENSION==3) {N[2]=std::stoull(argv[3]);}

	//the bounding box of the octree will automatically be expanded when needed.
	//this is fairly fast (adds nodes to the top of the tree structure rather than rebuilding)
	//but if the bounding box is known ahead of time, it should be used.
	//to help avoid floating point errors, the bounding box will be expanded to have
	//power of 2 coordinates.
	Octree_t octree(Box_t{Point_t{0}, Point_t{N}});

	if constexpr (DIMENSION==2) {
		generate_points2(octree, N);
	} else if constexpr (DIMENSION==3) {
		generate_points3(octree, N);
	}
	
	std::cout << "The octree has " << octree.size() << " elements ";
	if (octree.size() == prod(N)) {std::cout << "SUCCESS\n";}
	else {std::cout << "FAIL\n";}

	// size_t n_nodes, n_idx, n_idx_cap, n_leafs;
	// int max_depth;
	// octree.treeSummary(n_nodes, n_idx, n_idx_cap, n_leafs, max_depth);

	// std::cout << "The octree has " << n_nodes << " octree nodes with " << n_leafs << " leafs\n";
	// std::cout << "The octree is storing " << n_idx << "/" << n_idx_cap << " data indices\n";
	// std::cout << "The octree has a maximum depth of " << max_depth << std::endl;

	// double memory_overhead_in_bytes = sizeof(octree) + n_nodes*sizeof(typename Octree_t::Node_t);
	// double storage_memory_in_bytes  = octree.size() * sizeof(typename Octree_t::Data_t);
	// double byte2MiB = std::pow(0.5, 20);

	// std::cout << "The octree structure itself is using " << memory_overhead_in_bytes*byte2MiB << " MiB "
	// 		  << "memory overhead\n";
	// std::cout << "The data itself is using " << storage_memory_in_bytes*byte2MiB << " MiB\n";

	Box_t search_box(Point_t{0}, 0.5*Point_t{N});
	std::vector<size_t> found_idx = octree.get_data_in_box(search_box);
	std::cout << "The octree found " << found_idx.size() << " points in the box: " << search_box << std::endl;

	return 0;
}


