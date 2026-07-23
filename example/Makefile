#header files
INCLUDE = -I../include

#default build (debug)
CXX     = g++ -std=c++23

#explicit defines
DEFINES = 
ifdef TEST_DIM
	DEFINES += -DTEST_DIM=$(TEST_DIM)
endif
ifdef TEST_SCALAR
	DEFINES += -DTEST_SCALAR=$(TEST_SCALAR)
endif
ifdef GUTIL_ORTHTREE_MAX_DEPTH
	DEFINES += -DGUTIL_ORTHTREE_MAX_DEPTH=$(GUTIL_ORTHTREE_MAX_DEPTH)
endif
ifdef GUTIL_N_OCTREE_THREADS
	DEFINES += -DGUTIL_N_OCTREE_THREADS=$(GUTIL_N_OCTREE_THREADS)
endif


#debug vs release mode
ifdef NDEBUG
	CC = $(CXX) -O3
	DEFINES += -DNDEBUG
else
	CC = $(CXX) -O0 -Wall -pg
endif

#link to libraries
LIBS = 
ifdef OMP
	LIBS += -fopenmp
endif

#final compile command
COMPILE = ${CC} ${INCLUDE} ${LIBS} ${DEFINES}


point: FORCE
	${COMPILE} point.cpp -o point

point_parallel: FORCE
	${COMPILE} point_parallel.cpp -o point_parallel

point_octree: FORCE
	${COMPILE} point_octree.cpp -o point_octree

octree_verify: FORCE
	$(COMPILE) octree_verify.cpp -o octree_verify

FORCE:

clean:
	rm -f point point_parallel point_octree octree_verify *.out