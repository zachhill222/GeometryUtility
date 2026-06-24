#include "gutil.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <string>

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

// ---------------------------------------------------------------------------
// Pretty printer
// ---------------------------------------------------------------------------
static void print_result(const std::string& label,
                         double elapsed_ms,
                         size_t count)
{
    std::cout << std::left  << std::setw(24) << label
              << std::right << std::setw(10) << count   << " items   "
              << std::setw(10) << std::fixed << std::setprecision(3)
              << elapsed_ms << " ms   ("
              << std::setw(8) << std::setprecision(1)
              << (count / elapsed_ms * 1000.0) << " ops/s)\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Allow N to be passed as a command-line argument, default 100'000
    const size_t N = (argc > 1) ? static_cast<size_t>(std::stoul(argv[1]))
                                 : 100'000;

    std::cout << "=== PointOctree test  N=" << N << " ===\n\n";

    // -----------------------------------------------------------------------
    // 1. Generate random points
    // -----------------------------------------------------------------------
    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    using Tree  = gutil::PointOctree<3, float>;   // 3D, float, default MaxData=64
    using Point = typename Tree::point_type;
    using Box   = typename Tree::box_type;

    std::vector<Point> points;
    points.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        points.push_back({dist(rng), dist(rng), dist(rng)});
    }

    // -----------------------------------------------------------------------
    // 2. Construct the tree
    // -----------------------------------------------------------------------
    const Box root_bbox({-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f});
    Tree tree(root_bbox);
    tree.reserve(N);

    // -----------------------------------------------------------------------
    // 3. Insert all points — timed
    // -----------------------------------------------------------------------
    Timer t;
    t.start();

    std::vector<size_t> indices;
    indices.reserve(N);
    for (const Point& p : points) {
        indices.push_back(tree.insert(p));
    }

    const double insert_ms = t.elapsed_ms();
    print_result("insert", insert_ms, N);

    // Basic sanity: every returned index should be < N and tree size == N
    assert(tree.size() == N && "tree size mismatch after insert");
    for (size_t i = 0; i < N; ++i) {
        assert(indices[i] < N && "bad index returned by insert");
    }

    // -----------------------------------------------------------------------
    // 4. contains() — check every point is found
    // -----------------------------------------------------------------------
    t.start();
    size_t found = 0;
    for (const Point& p : points) {
        if (tree.contains(p)) { ++found; }
    }
    const double contains_ms = t.elapsed_ms();
    print_result("contains", contains_ms, N);

    if (found != N) {
        std::cerr << "ERROR: contains() missed " << (N - found) << " points\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // 5. nearest() — find the nearest stored point to each query point
    //    Use freshly generated query points so we don't just look up existing ones
    // -----------------------------------------------------------------------
    std::vector<Point> queries;
    queries.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        queries.push_back({dist(rng), dist(rng), dist(rng)});
    }

    t.start();
    size_t nearest_found = 0;
    for (const Point& q : queries) {
        const size_t idx = tree.nearest(q);
        if (idx != size_t(-1)) { ++nearest_found; }
    }
    const double nearest_ms = t.elapsed_ms();
    print_result("nearest", nearest_ms, N);

    if (nearest_found != N) {
        std::cerr << "ERROR: nearest() failed for "
                  << (N - nearest_found) << " queries\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // 6. Verify nearest() correctness on a small random sample
    //    Brute-force the answer and compare
    // -----------------------------------------------------------------------
    const size_t VERIFY_N = std::min(N, size_t(1000));
    size_t mismatches = 0;

    for (size_t qi = 0; qi < VERIFY_N; ++qi) {
        const Point& q = queries[qi];

        // brute force
        size_t bf_idx  = 0;
        float  bf_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < N; ++i) {
            const Point  diff = q - points[i];
            const float  d2   = gutil::dot(diff, diff);
            if (d2 < bf_dist) { bf_dist = d2; bf_idx = i; }
        }

        // octree
        const size_t ot_idx  = tree.nearest(q);
        const Point  ot_diff = q - points[ot_idx];
        const float  ot_dist = gutil::dot(ot_diff, ot_diff);

        // distances should match (indices may differ for ties)
        if (std::abs(ot_dist - bf_dist) > 1e-6f) {
            ++mismatches;
        }
    }

    if (mismatches == 0) {
        std::cout << "\nnearest() verified correct on "
                  << VERIFY_N << " samples.\n";
    } else {
        std::cerr << "\nERROR: nearest() gave wrong result on "
                  << mismatches << " / " << VERIFY_N << " samples.\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // 7. clear() and re-insert — sanity check
    // -----------------------------------------------------------------------
    t.start();
    tree.clear();
    for (const Point& p : points) { tree.insert(p); }
    const double reinsert_ms = t.elapsed_ms();
    print_result("clear+reinsert", reinsert_ms, N);

    assert(tree.size() == N);

    std::cout << "\nAll tests passed.\n";
    return 0;
}



