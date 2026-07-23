#include "gutil.hpp"

using Sphere     = gutil::Sphere<3,float>;
using point_type = gutil::Point<3,float>;
using box_type   = gutil::Box<3,float>;


int main(int argc, char* argv[]) {
	//set up rng
	auto random_point = gutil::UniformRandomPoint<point_type,false>();
	random_point.set_parameters(float{-10}, float{10});

	auto random_scalar = gutil::UniformRandomPoint<point_type,false>();
	random_scalar.set_parameters(0.1f, 1.0f);

	//set up octree and bounding box
	box_type box( point_type::Filled(-11), point_type::Filled(11) );
	gutil::VolumeOctree<Sphere> tree(box);

	//add spheres to the octree so long as they don't intersect
	int N = (argc>1) ? atoi(argv[1]) : 100;

	for (int i=0; i<N; ++i) {
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

	for (int i=0; i<N; ++i) {
		point_type pt = random_point();
		size_t idx = tree.find_nearest(pt);

		for (const Sphere& s : tree) {
			if (s.distance(pt) < tree[idx].distance(pt)) {
				gutil::Logger::error("ERROR: did not find closest sphere");
			}
		}
	}


	//write spheres to a file
	gutil::write_spheres_to_file("spheres.txt", tree.as_cspan());

	return 0;
}