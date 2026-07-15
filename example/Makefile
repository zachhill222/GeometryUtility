INCLUDE = -I../include

#default build (debug)
CXX        = g++ -std=c++23
OPT        = -O0
PAR        =
DEBUG      = -pg -g -Wall

#redefine for testing optimizations
OPT        = -O3
PAR        = -fopenmp
# DEBUG      = -DNDEBUG

#final compiler command
CC = $(CXX) $(OPT) $(PAR) $(DEBUG)

#final defines
DEFINES = ${DIM}

point: FORCE
	${CC} ${INCLUDE} ${DEFINES} point.cpp -o point

point_parallel: FORCE
	${CC} ${INCLUDE} ${DEFINES} point_parallel.cpp -o point_parallel

FORCE:

clean:
	rm -f point point_parallel *.out