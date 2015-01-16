PROJECT = bfs_seq bfs_par bfs_par_2 bfs_par_3
SOURCE = bfs_seq.cpp bfs_par.cu bfs_par_2.cu bfs_novi.cu

all: bfs_seq bfs_par bfs_par_2 bfs_par_3

bfs_seq: bfs_seq.cpp
	g++ bfs_seq.cpp -o bfs_seq

bfs_par: bfs_par.cu
	nvcc bfs_par.cu -o bfs_par

bfs_par_2: bfs_par_2.cu
	nvcc bfs_par_2.cu -o bfs_par_2 -arch=sm_20

bfs_par_3: bfs_novi.cu
	nvcc bfs_novi.cu -o bfs_par_3 -arch=sm_20

clean:
	-rm -f $(PROJECT)
