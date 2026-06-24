INCLUDE = -I../include

#default build (debug)
CXX        = g++ -std=c++20
OPT        = -O0
PAR        =
DEBUG      = -pg -g -Wall

#redefine for testing optimizations
OPT        = -O3
# PAR        = -fopenmp
DEBUG      = -DNDEBUG

#final compiler command
CC = $(CXX) $(OPT) $(PAR) $(DEBUG)

#testing defines
DIM        = -DDIMENSION=2

#final defines
DEFINES = ${DIM}

point:
	${CC} ${INCLUDE} ${DEFINES} point.cpp -o point

point_old:
	${CC} ${INCLUDE} ${DEFINES} point_octree_example.cpp -o point_old

polygon:
	${CC} ${INCLUDE} ${DEFINES} polygon_octree_example.cpp -o polygon


clean:
	rm -f point point_old polygon *.out