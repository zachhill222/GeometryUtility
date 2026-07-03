#include "gutil.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <string>
#include <omp.h>

//define the tree and point types
using tree_type = gutil::PointOctree<3,float>;
using point_type = typename tree_type::value_type;
using box_type = typename tree_type::box_type;

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
struct Timer
{
    using clock     = std::chrono::high_resolution_clock;
    using duration  = std::chrono::duration<double, std::milli>;

    clock::time_point _start;

    void start() { _start = clock::now(); }

    double elapsed_ms() const {
        return duration(clock::now() - _start).count();
    }
};

int main(int argc, char** argv)
{
    // get number of points and number of octrees
    const size_t N = (argc > 1) ? static_cast<size_t>(std::stoul(argv[1])) : 100'000;
    const size_t T = (argc > 2) ? static_cast<size_t>(std::stoul(argv[2])) : 8;
    std::cout << "=== PointOctree parallel test  N=" << N << ", T=" << T << " ===\n\n";

    //construct trees
    std::vector<tree_type> trees;
    for (size_t t=0; t<T; ++t) {
        trees.emplace_back(box_type{point_type{-1,-1,-1}, point_type{1,1,1}});
    }

    //populate trees
    Timer timer;
    timer.start();

    #pragma omp parallel
    {
        // initialize random number generator
        std::mt19937                          rng(omp_get_thread_num());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        auto rand_point = [&rng, &dist]() {
            return point_type{dist(rng), dist(rng), dist(rng)};
        };

        #pragma omp for
        for (size_t t=0; t<T; ++t) {
            for (size_t n=0; n<N; ++n) {
                trees[t].insert(std::move(rand_point()));
            }
        }    
    }

    double duration = timer.elapsed_ms();
    std::cout << "populated trees: " << duration << " ms" << std::endl;

    //join octrees
    timer.start();
    auto tree = gutil::join_trees(std::move(trees));
    duration = timer.elapsed_ms();
    std::cout << "joined trees: " << duration << " ms" << std::endl;
    std::cout << tree << std::endl; 
}