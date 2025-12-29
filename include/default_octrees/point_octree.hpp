#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "octree/octree_parallel.hpp"

#ifndef GUTIL_POINT_OCTREE_BUFFER_CAPACITY
	#define GUTIL_POINT_OCTREE_BUFFER_CAPACITY 2048
#endif

#ifndef GUTIL_POINT_OCTREE_BUFFER_USE_STACK
	#define GUTIL_POINT_OCTREE_BUFFER_USE_STACK true
#endif


namespace gutil {
	///Octree for points in space.
	template<int dim=3, typename T=double, int n_data=32>
	class PointOctree : public BasicParallelOctree<
									Point<dim,T>,
									true,
									dim,
									n_data,
									T>
	{
	public:
		using BaseClass = BasicParallelOctree<
									Point<dim,T>,
									true,
									dim,
									n_data,
									T>;
		using typename BaseClass::Data_t;
		using typename BaseClass::Box_t;

		//constructors
		constexpr PointOctree() noexcept : BaseClass() {}
		constexpr PointOctree(const Box_t &bbox) noexcept : BaseClass(bbox) {}

	private:
		constexpr bool isValid(const Box_t& box, const Data_t& data) const override {return box.contains(data);}
	};
}