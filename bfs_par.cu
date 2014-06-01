#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX_THREADS_PER_BLOCK 256

__global__ void kernel(int numberOfNodes, int *C, int *R, int *dist, bool *done, int iteration) {
	
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(threadID >= numberOfNodes || dist[threadID] != iteration) return;
	
	for(int i = R[threadID]; i < R[threadID+1]; ++i) {
		int next = C[i];
		if(dist[next] == -1) {
			dist[next] = dist[threadID] + 1;
			*done = false;
		}
	}
	
}

__global__ void initialize(int *dist, int startingNode, int numberOfNodes) {

	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadID >= numberOfNodes) return;
	dist[threadID] = (threadID == startingNode) ? 0 : -1;
}

int main( void ) {

	int numberOfNodes;
	int numberOfEdges;
	int startingNode;
	
	clock_t inputStartTime = clock();

	scanf("%d", &numberOfNodes);
	scanf("%d", &numberOfEdges);
	scanf("%d", &startingNode);
	
	int *h_C = (int*) malloc(numberOfEdges * sizeof(int)) ;
	int *h_R = (int*) malloc((numberOfNodes + 1) * sizeof(int));
	
	for(int i = 0; i < numberOfEdges; ++i) {
		scanf("%d", &h_C[i] );
	}

	for(int i = 0; i < numberOfNodes + 1; ++i) {
		scanf("%d", &h_R[i]);
	}
	
	int numberOfBlocks;
	int threadsPerBlock;
	
	if(numberOfNodes > MAX_THREADS_PER_BLOCK) {
		numberOfBlocks = (int)ceil(numberOfNodes/(double)MAX_THREADS_PER_BLOCK); //numberOfNodes / MAX_THREADS_PER_BLOCK + 1;
		threadsPerBlock = MAX_THREADS_PER_BLOCK;
	} else {
		numberOfBlocks = 1;
		threadsPerBlock = numberOfNodes;
	}
	
	int *d_C;
	cudaMalloc((void**) &d_C, numberOfEdges * sizeof(int));
	cudaMemcpy(d_C, h_C, numberOfEdges * sizeof(int), cudaMemcpyHostToDevice);

	int *d_R;
	cudaMalloc((void**) &d_R, (numberOfNodes + 1) * sizeof(int));
	cudaMemcpy(d_R, h_R, (numberOfNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

	int *d_dist;
	cudaMalloc((void**) &d_dist, numberOfNodes * sizeof(int));
	
	bool h_done = false;
	bool *d_done;
	cudaMalloc((void**) &d_done, sizeof(bool));
	
	clock_t initStartTime = clock();

	initialize<<<numberOfBlocks, threadsPerBlock>>>(d_dist, startingNode, numberOfNodes);
	cudaThreadSynchronize();

	clock_t startTime = clock();

	for(int iteration = 0; !h_done; ++iteration) {
		h_done = true;
		cudaMemcpy(d_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
		
		kernel<<<numberOfBlocks, threadsPerBlock>>>(numberOfNodes, d_C, d_R, d_dist, d_done, iteration);
		cudaThreadSynchronize();
		
		cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
	}
	
	clock_t endTime = clock();
	printf("Graph input time: %lf\n",  (double)(initStartTime - inputStartTime) / CLOCKS_PER_SEC);
	printf("Initialization time: %lf\n",  (double)(startTime - initStartTime) / CLOCKS_PER_SEC);
	printf("Execution time: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
	
	int *h_dist = (int*) malloc((numberOfNodes) * sizeof(int));
	cudaMemcpy(h_dist, d_dist, numberOfNodes * sizeof(int), cudaMemcpyDeviceToHost);
	
	FILE *out = fopen("results_par.txt", "w");
	
	for( int i = 0; i < numberOfNodes; ++i) {
		fprintf(out, "%d\n", h_dist[i]);
	}
	
	fclose(out);
	
	free(h_C);
	free(h_R);
	free(h_dist);
	cudaFree(d_C);
	cudaFree(d_R);
	cudaFree(d_dist);
	cudaFree(d_done);
	
	return 0;
}