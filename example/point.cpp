#include "gutil.hpp"
#include "utility/rng.hpp"

#include <algorithm>
#include <vector>
#include <span>
#include <unordered_set>
#include <iostream>

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

using tree_type = PointOctree<point_type>;

std::vector<point_type> generate_points(size_t N) {
	std::cout << "\n";
	Logger::log("START: generate_points");
	LogTime time{"END: generate_points"};

	auto random_point = UniformRandomPoint<point_type,false>();
	random_point.set_parameters(Scalar{-10}, Scalar{10});

	std::vector<point_type> result;
	result.reserve(N);

	for (size_t i=0; i<N; ++i) {
		result.push_back(random_point());
	}

	std::cout << "\tgenerated " << std::to_string(result.size()) << " points" << std::endl;

	return result;
}

tree_type generate_random_tree(size_t N) {
	std::cout << "\n";
	Logger::log("START: generate_random_tree");
	LogTime time{"END: generate_random_tree"};

	auto random_point = UniformRandomPoint<point_type,false>();
	random_point.set_parameters(Scalar{-10}, Scalar{10});

	tree_type tree{Box{point_type::Filled(Scalar{-10}), point_type::Filled(Scalar{10})}};
	tree.reserve(N);

	for (size_t i=0; i<N; ++i) {
		tree.push_back(random_point());
	}

	std::cout << "\tgenerated " << std::to_string(tree.size()) << " points" << std::endl;

	return tree;
}

void test_standard_sum(std::span<const point_type> list) {
	std::cout << "\n";
	Logger::log("START: test_standard_sum");
	LogTime time{"END: test_standard_sum"};

	point_type val = point_type::Zeros();
	for (auto& p : list) {val += p;}

	std::cout << "\tsum= " << val << std::endl;
}

void test_sorted_sum(std::span<const point_type> list) {
	std::cout << "\n";
	Logger::log("START: test_sorted_sum");
	LogTime time{"END: test_sorted_sum"};

	std::cout << "\tsum= " << sorted_sum(list) << std::endl;
}

void test_kahan_sum(std::span<const point_type> list) {
	std::cout << "\n";
	Logger::log("START: test_kahan_sum)");
	LogTime time{"END: test_kahan_sum)"};

	std::cout << "\tsum= " << kahan_sum(list) << std::endl;
}

void construct_unordered_set(std::span<const point_type> list) {
	std::cout << "\n";
	Logger::log("START: construct_unordered_set");
	LogTime time{"END: construct_unordered_set"};

	std::unordered_set<point_type> set;
	set.reserve(list.size());
	for (const auto& p : list) {set.insert(p);}

	std::cout << "\tcreated an unordered set with " << set.size() << " points" << std::endl;
}

auto move_to_octree(std::span<point_type> list) {
	std::cout << "\n";
	Logger::log("START: move_to_octree");
	LogTime time{"END: move_to_octree"};

	PointOctree<point_type> tree{point_type::Filled(-10), point_type::Filled(10)};
	tree.push_back_range(list);

	std::cout << "\tcreated the octree with " << tree.size() << " points" << std::endl;
	return tree;
}

void test_octree_find(const PointOctree<point_type>& tree) {
	std::cout << "\n";
	Logger::log("START: test_octree_find");
	LogTime time{"END: test_octree_find"};

	size_t n_miss = 0;
	size_t idx = 0;
	for (const point_type& p : tree) {
		if (tree.find(p) != idx) {++n_miss;}
		++idx;
	}

	if (n_miss>0) {
		std::cerr << "\tERROR : missed " << n_miss << "/" << tree.size() <<  " points" << std::endl; 
	}
	else {
		std::cout << "\tSUCCESS : missed " << n_miss << "/" << tree.size() <<  " points" << std::endl; 
	}
}

std::vector<size_t> find_nearest_octree(const PointOctree<point_type>& tree, std::span<const point_type> query) {
	std::cout << "\n";
	Logger::log("START: find_nearest_octree");
	LogTime time{"END: find_nearest_octree"};

	const size_t N = query.size();
	std::vector<size_t> result(N);

	GUTIL_OMP(parallel for)
	for (size_t i=0; i<N; ++i) {
		result[i] = tree.find_nearest(query[i]);
	}

	return result;
}

std::vector<size_t> find_nearest_brute_force(std::span<const point_type> points, std::span<const point_type> query) {
	std::cout << "\n";
	Logger::log("START: find_nearest_brute_force");
	LogTime time{"END: find_nearest_brute_force"};

	if (points.empty()) {return {};}

	const size_t N = query.size();
	std::vector<size_t> result(N);

	GUTIL_OMP(parallel for)
	for (size_t i=0; i<N; ++i) {
		const point_type& q = query[i];
		result[i] = 0;
		Scalar d2 = gutil::distance_squared(points[0],q);
		for (size_t j=1; j<points.size(); ++j) {
			const Scalar tmp = gutil::distance_squared(points[j],q);
			if (tmp < d2) {d2 = tmp; result[i]=j;}
		}
	}

	return result;
}

void compare_nearest(std::span<const point_type> points, std::span<const point_type> query,
		std::span<const size_t> tree_idx, std::span<const size_t> brute_idx) {
	std::cout << "\n";
	Logger::log("START: compare_nearest");
	LogTime time{"END: compare_nearest"};

	if (query.size() != tree_idx.size() || query.size() != brute_idx.size()) {
		std::cerr << "\tERROR: dimension mismatch" << std::endl;
		throw;
	} 

	size_t n_miss = 0;
	for (size_t i=0; i<query.size(); ++i) {
		if (tree_idx[i] != brute_idx[i]) {
			const auto& q = query[i];
			const auto ti = tree_idx[i];
			const auto bi = brute_idx[i];

			const Scalar dt = gutil::distance_squared(points[ti], q);
			const Scalar db = gutil::distance_squared(points[bi], q);

			++n_miss;
			std::cerr << "\tERROR(" << n_miss << ") at query " << q << "\n"
					  << "\t       tree found " << points[ti] << " (dist= " << dt << ")\n"
					  << "\t       brute found " << points[bi] << " (dist= " << db << ")" << std::endl;
		}
	}

	if (n_miss>0) {
		std::cerr << "\tERROR : " << n_miss << "/" << query.size() <<  " different indices" << std::endl; 
	}
	else {
		std::cout << "\tSUCCESS : " << n_miss << "/" << query.size() <<  " different indices" << std::endl; 
	}
}


int main(int argc, char** argv) {
	size_t N = argc > 1 ? atoi(argv[1]) : 100000;
	std::vector<point_type> points = generate_points(N/2);
	std::vector<point_type> query = generate_points(10);

	test_standard_sum(points);
	// test_sorted_sum(points);
	test_kahan_sum(points);
	// construct_unordered_set(points);
	auto tree = move_to_octree(points);
	{
		auto p2 = generate_points(N/2);
		tree.push_back_range(std::move(p2));
	}

	// {auto tree2 = generate_random_tree(N);}
	test_octree_find(tree);
	auto tree_near = find_nearest_octree(tree, query);
	auto brute_near = find_nearest_brute_force(tree.data(), query);
	compare_nearest(tree.data(), query, tree_near, brute_near);

	std::cout << "\nOctree points:\n";
	gutil::print_to_stream(std::cout, std::span<const point_type>(tree.begin(), tree.begin()+5), "\n");

	// std::cout << tree;

	return 0;
}







