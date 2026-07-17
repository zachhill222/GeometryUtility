#pragma once

#include "math/math.hpp"
#include "utility/utility.hpp"
#include "geometry/geometry.hpp"

#include "octree/node.hpp"
#include "octree/octree_base.hpp"
#include "octree/point_octree.hpp"

#include "memory/slab_allocator.hpp"
#include "memory/hetero_slab_allocator.hpp"
#include "memory/thread_pool.hpp"

#include "algorithms/convex_collision.hpp"

// namespace gutil
// {
// 	template<int offset=0>
// 	using FixedPoint64 = FixedPoint<int64_t,offset>;
// }