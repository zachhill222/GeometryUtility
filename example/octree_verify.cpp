#include "gutil.hpp"

using Sphere     = gutil::Sphere<3,float>;
using point_type = gutil::Point<3,float>;
using box_type   = gutil::Box<3,float>;


int main(int argc, char* argv[]) {
	//set up rng
	auto random_point = gutil::UniformRandomPoint<point_type,false>();
	random_point.set_parameters(float{0}, float{1});

	auto random_scalar = gutil::UniformRandomPoint<point_type,false>();
	random_scalar.set_parameters(0.05f, 0.5f);

	//set up octree and bounding box
	box_type box( point_type::Filled(0), point_type::Filled(1) );
	gutil::VolumeOctree<Sphere> tree(box);

	const int min_size = (argc>1) ? atoi(argv[1]) : 100;
	while (tree.size()<min_size) {
		Sphere s(random_point(), random_scalar.scalar());
		size_t idx = tree.collides_with(s);
		if (idx < tree.size()) {
			gutil::Logger::log("COLLISION: ", s, " collides with ", tree[idx],
					" (idx= ", idx, " center2center= ", gutil::distance(s.center,tree[idx].center),")");
		}
		else {
			tree.push_back(s);
			gutil::Logger::log("INSERTED[", tree.size(), "] ", s);
		}
	}

	//write spheres to a file
	gutil::Logger::log("Tree has ", tree.size(), " spheres");
	gutil::write_spheres_to_file("spheres.txt", tree.as_cspan());

	return 0;
}