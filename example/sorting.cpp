#include "gutil.hpp"

#include <random>
#include <vector>


int main(int argc, char* argv[]) {
	size_t N = argc>1 ? atoi(argv[1]) : 100000;
	size_t M = argc>2 ? atoi(argv[2]) : 5000;

	gutil::ThreadPool tp{};
	std::mt19937 gen(N);
	std::uniform_int_distribution<size_t> dist(0,M);

	std::vector<size_t> gutil_sort(N);
	std::vector<size_t> std_sort(N);

	{
		GUTIL_TIMER("Fill vectors");
		for (size_t i=0; i<N; ++i) {
			size_t r = dist(gen);
			gutil_sort.push_back(r);
			std_sort.push_back(r);
		}
	}

	{
		GUTIL_TIMER("Gutil sort");
		auto it = gutil::sort_and_unique(gutil_sort, tp);
		GUTIL_LOG("partition at ", std::distance(gutil_sort.begin(), it));
	}

	{
		GUTIL_TIMER("Std sort");
		std::sort(std_sort.begin(), std_sort.end());
		auto it = std::unique(std_sort.begin(), std_sort.end());
		GUTIL_LOG("partition at ", std::distance(std_sort.begin(), it));
	}

	{
		GUTIL_TIMER("Verify results");
		size_t count = 0;
		for (size_t i=0; i<N; ++i) {
			if (gutil_sort[i]!=std_sort[i]) {++count;}
		}
		GUTIL_LOG("vectors differ in ", count, "/", N, " locations");
	}

}