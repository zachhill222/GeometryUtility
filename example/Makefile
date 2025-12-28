INCLUDE = -I../include

#default build (debug)
CXX        = g++ -std=c++20
OPT        = -O0
PAR        =
DEBUG      = -pg -g3 -Wall

#redefine for testing optimizations
OPT        = -O3
PAR        = -fopenmp
DEBUG      = -DNDEBUG

#final compiler command
CC = $(CXX) $(OPT) $(PAR) $(DEBUG)

#testing defines
DIM         = -DDIMENSION=3

#final defines
DEFINES = ${DIM} ${OCT_BUFFER}

point:
	${CC} ${INCLUDE} ${DEFINES} point_octree_example.cpp -o point_octree_example