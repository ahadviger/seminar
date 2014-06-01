#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX_THREADS_PER_BLOCK 256

__global__ void kernel(int *C, int *R, int *dist, int *inQueue, int *outQueue, int inQueueSize, int *outQueueSize) {
	
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadID >= inQueueSize) return;
	
	int curr = inQueue[threadID];
		
	for(int i = R[curr]; i < R[curr+1]; ++i) {
		int next = C[i];
		if(dist[next] == -1) {
			dist[next] = dist[curr] + 1;
			int position = atomicAdd(outQueueSize, 1);
			outQueue[position] = next;
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
	
	int h_inQueueSize = 1;
	int h_outQueueSize = 0;

	int *d_inQueue;
	cudaMalloc((void**) &d_inQueue, numberOfNodes * sizeof(int));
	cudaMemcpy(d_inQueue, &startingNode, sizeof(int), cudaMemcpyHostToDevice);

	int *d_outQueue;
	cudaMalloc((void**) &d_outQueue, numberOfNodes * sizeof(int));

	int *d_outQueueSize;
	cudaMalloc((void**) &d_outQueueSize, sizeof(int));

	clock_t initStartTime = clock();

	initialize<<<numberOfBlocks, threadsPerBlock>>>(d_dist, startingNode, numberOfNodes);
	cudaThreadSynchronize();
		
	clock_t startTime = clock();
		
	while(h_inQueueSize != 0) {
	
		cudaMemcpy(d_outQueueSize, &h_outQueueSize, sizeof(int), cudaMemcpyHostToDevice);
		
		if(h_inQueueSize > MAX_THREADS_PER_BLOCK) {
			numberOfBlocks = (int)ceil(h_inQueueSize/(double)MAX_THREADS_PER_BLOCK); //numberOfNodes / MAX_THREADS_PER_BLOCK + 1;
			threadsPerBlock = MAX_THREADS_PER_BLOCK;
		} else {
			numberOfBlocks = 1;
			threadsPerBlock = h_inQueueSize;
		}
		
		kernel<<<numberOfBlocks, threadsPerBlock>>>(d_C, d_R, d_dist, d_inQueue, d_outQueue, h_inQueueSize, d_outQueueSize);
		cudaThreadSynchronize();
		
		cudaMemcpy(&h_inQueueSize, d_outQueueSize, sizeof(int), cudaMemcpyDeviceToHost);
		d_inQueue = d_outQueue;

		//cudaMemcpy(d_inQueue, d_outQueue, numberOfNodes * sizeof(int), cudaMemcpyDeviceToDevice);
	}
	
	clock_t endTime = clock();
	printf("Graph input time: %lf\n",  (double)(initStartTime - inputStartTime) / CLOCKS_PER_SEC);
	printf("Initialization time: %lf\n",  (double)(startTime - initStartTime) / CLOCKS_PER_SEC);
	printf("Execution time: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
	
	int *h_dist = (int*) malloc((numberOfNodes) * sizeof(int));
	cudaMemcpy(h_dist, d_dist, numberOfNodes * sizeof(int), cudaMemcpyDeviceToHost);
	
	FILE *out = fopen("results_par_2.txt", "w");
	
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
	cudaFree(d_outQueue);
	cudaFree(d_inQueue);
	cudaFree(d_outQueueSize);
	
	return 0;
}