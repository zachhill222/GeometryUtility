INCLUDE = -I../include

#default build (debug)
CXX        = g++ -std=c++23
OPT        = -O0
PAR        =
DEBUG      = -pg -g -Wall

#redefine for testing optimizations
OPT        = -O3
PAR        = -fopenmp
DEBUG      = -DNDEBUG

#final compiler command
CC = $(CXX) $(OPT) $(PAR) $(DEBUG)

#final defines
DEFINES = 
ifdef TEST_DIM
	DEFINES += -DTEST_DIM=$(TEST_DIM)
endif
ifdef TEST_SCALAR
	DEFINES += -DTEST_SCALAR=$(TEST_SCALAR)
endif

point: FORCE
	${CC} ${INCLUDE} ${DEFINES} point.cpp -o point

point_parallel: FORCE
	${CC} ${INCLUDE} ${DEFINES} point_parallel.cpp -o point_parallel

point_octree: FORCE
	${CC} ${INCLUDE} ${DEFINES} point_octree.cpp -o point_octree

FORCE:

clean:
	rm -f point point_parallel point_octree *.out