#include "octree/base_node.hpp"
#include "octree/base_octree.hpp"
#include "octree/point_octree.hpp"

#include "geometry/point.hpp"
#include "geometry/box.hpp"

using point_type = gutil::Point<3,float>;
using box_type = gutil::Box<3,float>;

int main(int argc, char* argv[]) {
	box_type box( point_type{0,0,0}, point_type{1,1,1} );
	gutil::PointOctree<gutil::Point<3,float>> tree(box);
	return 0;
}