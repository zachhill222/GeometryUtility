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

#define dimension
# DIM        = -DDIMENSION=2
DIM        = -DDIMENSION=3

#final compiler command
CC = $(CXX) $(OPT) $(PAR) $(DEBUG)

point:
	${CC} ${INCLUDE} ${DIM} point_octree_example.cpp -o point_octree_example