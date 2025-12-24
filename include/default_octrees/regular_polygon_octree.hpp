#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "geometry/basic_shapes.hpp"
#include "octree/octree_parallel.hpp"

namespace gutil {
	///Octree for points in space.
	///Note that the data should be allowed to be put into multiple leaf nodes
	template<int N, typename T=double, int n_data=32>
	class RegularPolygonOctree : public BasicParallelOctree<RegularPolygon<N,T>, false, 2, n_data, T>
	{
	public:
		using BaseClass = BasicParallelOctree<RegularPolygon<N,T>, false, 2, n_data, T>;
		using typename BaseClass::Data_t;
		using typename BaseClass::Box_t;

		//constructors
		constexpr RegularPolygonOctree() noexcept : BaseClass() {}
		constexpr RegularPolygonOctree(const Box_t &bbox) noexcept : BaseClass(bbox) {}

	private:
		constexpr bool isValid(const Box_t& box, const Data_t& data) const override {
			return box.intersects(data.bbox());}
	};
}