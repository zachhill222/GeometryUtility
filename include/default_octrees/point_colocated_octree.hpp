#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "default_octrees/point_octree.hpp"

#include <vector>
#include <cassert>

namespace gutil {
	//A wrapper for data types colocated at a single point in space
	//stores coordinates in a point octree and extra data in a vector

	template<typename Data_t, int dim=3, typename T=double, int n_data=32, typename U=T>
	struct PointCollocatedOctree
	{
		PointOctree<dim,T,n_data,U> coord_octree;
		std::vector<Data_t> data;

		inline const Data_t& operator[](const size_t& idx) const {assert(idx < data.size()); return data[idx];}
		inline Data_t& operator[](const size_t& idx) {assert(idx < data.size()); return data[idx];}

		
	};
}