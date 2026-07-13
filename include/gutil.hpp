#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "geometry/polytope.hpp"
#include "geometry/basic_shapes.hpp"

#include "octree/node.hpp"
#include "octree/octree_base.hpp"
#include "octree/point_octree.hpp"

#include "math/gutilmath.hpp"
#include "math/fixed_point.hpp"
#include "math/float_manipulation.hpp"
#include "math/matrix.hpp"
#include "math/quaternion.hpp"
#include "math/linear_set.hpp"

#include "memory/slab_allocator.hpp"
#include "memory/hetero_slab_allocator.hpp"

#include "algorithms/convex_collision.hpp"

namespace gutil
{
	template<int offset=0>
	using FixedPoint64 = FixedPoint<int64_t,offset>;
}