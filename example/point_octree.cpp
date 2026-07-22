#include "gutil.hpp"
#include "utility/rng.hpp"

#include <algorithm>
#include <vector>
#include <span>
#include <iostream>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace gutil;

#ifndef TEST_DIM
	#define TEST_DIM 3
#endif

#ifndef TEST_SCALAR
	#define TEST_SCALAR float
#endif

using point_type = Point<TEST_DIM,TEST_SCALAR>;
using Scalar = TEST_SCALAR;
using TreeType = PointOctree<point_type>;


std::vector<point_type> generate_points(size_t N) {
	auto random_point = UniformRandomPoint<point_type,false>();
	random_point.set_parameters(Scalar{-10}, Scalar{10});

	std::vector<point_type> result;
	result.reserve(N);

	for (size_t i=0; i<N; ++i) {
		result.push_back(random_point());
	}

	return result;
}

TreeType move_to_octree(std::span<point_type> list) {
	std::cout << "\n";
	Logger::log("START: move_to_octree");
	LogTime time{"END: move_to_octree"};

	TreeType tree{list};
	std::cout << "\ttree contains " << tree.size() << " points\n";
	return tree;
}

#ifdef _OPENMP
TreeType build_tree_parallel_omp(size_t n_threads, size_t n_points) {
	Logger::log("START: build_tree_parallel_omp");
	LogTime time{"END: build_tree_parallel_omp"};

	std::vector<TreeType> trees(n_threads);
	#pragma omp parallel num_threads(n_threads)
	{
		const size_t t = omp_get_thread_num();
		size_t thread_pts = n_points/n_threads;
		if (t == n_threads) {
			thread_pts = n_points - (n_threads-1)*thread_pts;
		}

		auto list = generate_points(thread_pts);
		trees[t] = std::move(move_to_octree(list));
	}
	return gutil::join_trees(std::move(trees));
}
#endif

TreeType build_tree_parallel_threads(size_t n_threads, size_t n_points) {
	Logger::log("START: build_tree_parallel_threads");
	LogTime time{"END: build_tree_parallel_threads"};

	std::vector<TreeType> trees(n_threads);
	std::vector<std::thread> threads;

	auto fun = [&trees](size_t t, size_t pts) {
		auto list = generate_points(pts);
		trees[t] = std::move(move_to_octree(list));
	};

	//spawn child threads
	for (size_t t=0; t<n_threads-1; ++t) {
		const size_t thread_num = omp_get_thread_num();
		size_t thread_pts = n_points/n_threads;
		threads.emplace_back(fun, t, thread_pts);
	}

	//handle remainder of points on main thread
	fun(n_threads-1, n_points - (n_threads-1)*(n_points/n_threads));

	//join the threads
	for (auto& t : threads) {t.join();}

	//combine trees
	return gutil::join_trees(std::move(trees));
}



int main(int argc, char** argv) {
	const size_t N = argc > 1 ? atoi(argv[1]) : 100000;
	const size_t M = argc > 2 ? atoi(argv[2]) : 4;

	#ifdef _OPENMP
	{
		auto tree = build_tree_parallel_omp(M,N);
		if (tree.size() == N) {
			Logger::log("SUCCESS: OMP tree has ", tree.size(),"/", N, " points");
		}
		else {
			Logger::log("ERROR: OMP tree has ", tree.size(),"/", N, " points");
		}
		
		gutil::print_to_stream(std::cout, std::span<const point_type>(tree.begin(), tree.begin()+5), "\n");
	}
	#endif

	{
		auto tree = build_tree_parallel_threads(M,N);
		if (tree.size() == N) {
			Logger::log("SUCCESS: STD::THREADS tree has ", tree.size(),"/", N, " points");
		}
		else {
			Logger::log("ERROR: STD::THREADS tree has ", tree.size(),"/", N, " points");
		}
		
		gutil::print_to_stream(std::cout, std::span<const point_type>(tree.begin(), tree.begin()+5), "\n");
	}
	return 0;
}