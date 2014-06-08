PROJECT = bfs_seq bfs_par bfs_par_2
SOURCE = bfs_seq.cpp bfs_par.cu bfs_par_2.cu

install: bfs_seq bfs_par bfs_par_2

bfs_seq: bfs_seq.cpp
	g++ bfs_seq.cpp -o bfs_seq
  
bfs_par: bfs_par.cu
	nvcc bfs_par.cu -o bfs_par

bfs_par_2: bfs_par_2.cu
	nvcc bfs_par_2.cu -o bfs_par_2 -arch=sm_20

clean:
	-rm -f $(PROJECT) $(OBJECTS)
