#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <deque>
#include <iostream>
#include <time.h>

#define BLOCK_SIZE 1024
#define W_SIZE 32
#define INF 100000
#define WARPS 32
#define TILE_SIZE 64

 #define NUM_BANKS 16  
 #define LOG_NUM_BANKS 4  
 #define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))  
/*
__device__ int warpCull(int neighbor) {
	volatile __shared__ int scratch[WARPS][128];
	int hash = neighbor & 127;
	scratch[warp_id][hash] = neighbor;
	int retrieved = scratch[warp_id][hash];
	if(retrieved == neighbor) {
		scratch[warp_id][hash] = thread_id;
		if(scratch[warp_id][hash] != thread_id) {
			return 1;
		}
	}
	return 0;
}

__device__ int historyCull(int neighbor) {
	
}

/* Threads in a warp together copy their part of array - for INT
 *	- W_OFF - position of a thread in a warp (warp offset)
 *	- size	- size of array to be copied
 */
 /*
__device_ void memcpy_SIMD_int(int W_OFF, int size, int *dest, int *src) {
	for(int i = W_OFF, i < size; i += W_SIZE) {
		dest[i] = src[i];
	}
	
	__threadfence_block();
}
*/
__device__ void gatherWarp(int *queue, int queue_size, int *d_R, int *d_C) {
	volatile __shared__ int comm[WARPS][3];
	
	int lane_id = threadIdx.x % W_SIZE;
	int warp_id = threadIdx.x / W_SIZE;
	
	int node = 0, r = 0, r_end = 0;
	
	if(threadIdx.x < queue_size) {
		node = queue[threadIdx.x];
		r = d_R[node];
		r_end = d_R[node + 1];
	}
		
	while(__any(r_end - r > 0)) {
		
		if(r_end - r > 0) {
			comm[warp_id][0] = lane_id;
		}
		
		if(comm[warp_id][0] == lane_id) {
			comm[warp_id][1] = r;
			comm[warp_id][2] = r_end;
			r = r_end;
		}
		
		int r_gather = comm[warp_id][1] + lane_id;
		int r_gather_end = comm[warp_id][2];
		while(r_gather < r_gather_end) {
			volatile int neighbor = d_C[r_gather];
			r_gather += W_SIZE;
		}
	}
}


__device__ void prefix_sum(int *g_odata, int *g_idata, int n) {

	__shared__ int temp[BLOCK_SIZE]; 
	int thid = threadIdx.x;  
	int offset = 1;  
	
	int ai = thid;  
    int bi = thid + (n/2); 
	
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);  
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);  
    temp[ai + bankOffsetA] = g_idata[ai];  
    temp[bi + bankOffsetB] = g_idata[bi];
	for (int d = n>>1; d > 0; d >>= 1) {                   // build sum in place up the tree  
		__syncthreads();  
		if (thid < d) {         
			int ai = offset*(2*thid+1)-1;  
			int bi = offset*(2*thid+2)-1;  
			ai += CONFLICT_FREE_OFFSET(ai);  
			bi += CONFLICT_FREE_OFFSET(bi);
		   temp[bi] += temp[ai];  
		}  
		offset *= 2;  	
	
	}
	if(thid == 0) {
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}  
	
	for (int d = 1; d < n; d *= 2) {// traverse down tree & build scan  
		offset >>= 1;  
		__syncthreads();  
		if (thid < d) {
			int ai = offset*(2*thid+1)-1;  
			int bi = offset*(2*thid+2)-1;  
			ai += CONFLICT_FREE_OFFSET(ai);  
			bi += CONFLICT_FREE_OFFSET(bi);
			int t = temp[ai];  
			temp[ai] = temp[bi];  
			temp[bi] += t;   
		}  
	}  
	__syncthreads();

	g_odata[ai] = temp[ai + bankOffsetA];  
	g_odata[bi] = temp[bi + bankOffsetB];  

}

__device__ void prescan(int *g_odata, int *g_idata, int n)  {  
    __shared__ int temp[BLOCK_SIZE * 2];
    int thid = threadIdx.x;  
    int offset = 1;  

    temp[2*thid] = g_idata[2*thid];   
    temp[2*thid+1] = g_idata[2*thid+1];  
  	
    for (int d = n>>1; d > 0; d >>= 1) {   
		__syncthreads();  
		if (thid < d) {  
			int ai = offset*(2*thid+1)-1;  
			int bi = offset*(2*thid+2)-1;  
			temp[bi] += temp[ai];  
		}  
		offset *= 2;  
	}
         
    if (thid == 0) {
		temp[n - 1] = 0;
	}
            
    for (int d = 1; d < n; d *= 2) {  
         offset >>= 1;  
         __syncthreads();  
         if (thid < d) {  
			int ai = offset*(2*thid+1)-1;  
			int bi = offset*(2*thid+2)-1;  
			int t = temp[ai];  
			temp[ai] = temp[bi];  
			temp[bi] += t;   
         }  
    }  
     __syncthreads();  
	
    g_odata[2*thid] = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1];  

}  
	


__device__ void gatherCTA(int *queue, int *out_queue, int queue_size, int level, int *dist, int *offset, int *d_R, int *d_C, int *broj, int *obrada) {
	volatile __shared__ int comm[3];
	__shared__ int scan_input[BLOCK_SIZE * 2];
	__shared__ int scan_output[BLOCK_SIZE * 2];
	__shared__ int susjedi;
	int id = threadIdx.x;
	int node = 0, r = 0, r_end = 0;
	
	if(threadIdx.x < queue_size) {
		node = queue[id];
		r = d_R[node];
		r_end = d_R[node + 1];
	}
	
	while(true) {
		if(id == 0) {
			comm[0] = -1;
		}
		
		__syncthreads();
		
		if(r_end - r > 0) {
			comm[0] = id;
		}
		
		__syncthreads();
		
		if(comm[0] == -1) break;
		
		if(comm[0] == id) {
			//printf("inspekcija %d %d %d\n", level, node, r_end - r);
			comm[1] = r;
			comm[2] = r_end;
			r = r_end;
			susjedi = 0;
		}
		
		__syncthreads();
		
		int r_gather = comm[1] + id;
		int r_gather_end = comm[2];
		__shared__ int cta_offset;
		__shared__ int done;
		
		for(int i = 0; true; ++i) {
	
			if(id == 0) done = 1;
			__syncthreads();
			
			volatile int neighbor;
			scan_input[id] = 0;
			
			if(r_gather < r_gather_end) {
				done = 0;
				atomicAdd(&susjedi, 1);
				neighbor = d_C[r_gather];
				if(dist[neighbor] == -1) {
					dist[neighbor] = level;
					scan_input[id] = 1;
					atomicAdd(broj, 1);
					//printf("| %d %d |", queue[comm[0]], neighbor);
				}
			}
			__syncthreads();

			if(done == 1) break;
			
			if(threadIdx.x < BLOCK_SIZE) {
				prescan(scan_output, scan_input, BLOCK_SIZE);
			}
			
			__syncthreads();
			
			if(threadIdx.x == 0) {
				cta_offset = atomicAdd(offset, scan_output[BLOCK_SIZE-1] + scan_input[BLOCK_SIZE-1]);
			}

			__syncthreads();

			if(r_gather < r_gather_end && scan_input[id] == 1) {
				out_queue[scan_output[id] + cta_offset] = neighbor;
			}

			r_gather += BLOCK_SIZE;
		}
	}
}


__global__ void BFS(int *d_R, int *d_C, int numberOfNodes, int *queue, int *out_queue, int *d_dist, int *offset, int level, int *broj, int *obrada) {
	//if(threadIdx.x==0) printf("%d %d\n", blockIdx.x, level);
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	__shared__ int neighbors[TILE_SIZE];
	int chunk_size = min(TILE_SIZE, numberOfNodes - TILE_SIZE * blockIdx.x);
	
	//if(threadIdx.x==0) printf("%d %d %d\n", blockIdx.x, level, chunk_size);

	if(threadIdx.x < chunk_size) {
		neighbors[threadIdx.x] = queue[blockIdx.x * TILE_SIZE + threadIdx.x];
	//	printf("tadadam %d %d %d --> %d\n", blockIdx.x, threadIdx.x, level, neighbors[threadIdx.x]);
	}
	//if(threadIdx.x==0) printf("chunk %d %d %d\n", level, blockIdx.x, chunk_size);
	__syncthreads();
	gatherCTA(neighbors, out_queue, chunk_size, level, d_dist, offset, d_R, d_C, broj, obrada);
	
//	if(threadIdx.x==0) printf("gotovo %d %d\n", blockIdx.x, level);

}

///////////////////////////////////////////////////
//	MAIN PROGRAM
///////////////////////////////////////////////////

int main(void) {

	const int ZERO = 0;

	int numberOfNodes;
	int numberOfEdges;
	int startingNode;
		
	scanf("%d", &numberOfNodes);
	scanf("%d", &numberOfEdges);
	scanf("%d", &startingNode);
	
	int *h_C = (int*) malloc(numberOfEdges * sizeof(int)) ;
	int *h_R = (int*) malloc((numberOfNodes + 1) * sizeof(int));
	int *h_dist = (int*) malloc(numberOfNodes * sizeof(int));
	int *h_q2 = (int*) malloc(numberOfNodes * sizeof(int));

	for(int i = 0; i < numberOfEdges; ++i) {
		scanf("%d", &h_C[i] );
	}

	for(int i = 0; i < numberOfNodes + 1; ++i) {
		scanf("%d", &h_R[i]);
		if(i < numberOfNodes) {
			h_dist[i] = (i == startingNode) ? 0 : -1;
		}
	}
	
	
	int *d_C;
	cudaMalloc((void**) &d_C, numberOfEdges * sizeof(int));
	cudaMemcpy(d_C, h_C, numberOfEdges * sizeof(int), cudaMemcpyHostToDevice);

	int *d_R;
	cudaMalloc((void**) &d_R, (numberOfNodes + 1) * sizeof(int));
	cudaMemcpy(d_R, h_R, (numberOfNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

	int *d_dist;
	cudaMalloc((void**) &d_dist, numberOfNodes * sizeof(int));
	cudaMemcpy(d_dist, h_dist, numberOfNodes * sizeof(int), cudaMemcpyHostToDevice);

	int *d_q1;
	cudaMalloc((void**) &d_q1, numberOfNodes * sizeof(int));
	cudaMemcpy( &d_q1[0], &startingNode, sizeof(int), cudaMemcpyHostToDevice);
 
	int *d_q2;
	cudaMalloc((void**) &d_q2, numberOfNodes * sizeof(int));

	int *offset;
	cudaMalloc((void**) &offset, sizeof(int));
	cudaMemcpy(offset, &ZERO, sizeof(int), cudaMemcpyHostToDevice);
	
	int *broj;
	cudaMalloc((void**) &broj, sizeof(int));
	cudaMemcpy(broj, &ZERO, sizeof(int), cudaMemcpyHostToDevice);
	
	int *obrada;
	cudaMalloc((void**) &obrada, sizeof(int));
	cudaMemcpy(obrada, &ZERO, sizeof(int), cudaMemcpyHostToDevice);
	
	clock_t startTime = clock();
	
	for(int level = 1;; ++level) {

		int queueSize;
		
		if(level == 1) {
			queueSize = 1;
		} else {
			cudaMemcpy(&queueSize, offset, sizeof(int), cudaMemcpyDeviceToHost);
		}
		cudaMemcpy(offset, &ZERO, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(broj, &ZERO, sizeof(int), cudaMemcpyHostToDevice);
		
		if(queueSize == 0 || level > 20) {
			break;
		}
		
		int numberOfBlocks = (int)ceil((double)queueSize / (double)TILE_SIZE);
		
		//printf("size %d blocks %d\n", queueSize, numberOfBlocks);


		BFS<<<numberOfBlocks, BLOCK_SIZE>>>(d_R, d_C, queueSize, d_q1, d_q2, d_dist, offset, level, broj, obrada);

		cudaThreadSynchronize();
		cudaMemcpy(&queueSize, offset, sizeof(int), cudaMemcpyDeviceToHost);
		
		int h_broj;
		cudaMemcpy(&h_broj, broj, sizeof(int), cudaMemcpyDeviceToHost);
		printf("\nvelicine %d %d %d\n", level, h_broj, queueSize);
		
		cudaMemcpy(h_q2, d_q2, queueSize * sizeof(int), cudaMemcpyDeviceToHost);
		printf("\n---------------\n");
		for(int j = 0; j < queueSize; ++j) printf("%d x ", h_q2[j]);
		printf("\n---------------\n");
		fflush(stdout);
		int *tmp = d_q1;
		d_q1 = d_q2;
		d_q2 = tmp;
	}
	
	clock_t endTime = clock();
	printf("Execution time: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

	cudaMemcpy(h_dist, d_dist, numberOfNodes * sizeof(int), cudaMemcpyDeviceToHost);

	FILE *out = fopen("results_par.txt", "w");
	
	for(int i = 0; i < numberOfNodes; ++i) {
		fprintf(out, "%d\n", h_dist[i]);
	}

	fclose(out);

	cudaFree(d_C);
	cudaFree(d_R);
	cudaFree(d_dist);
	cudaFree(offset);
	cudaFree(d_q1);
	cudaFree(d_q2);
	free(h_C);
	free(h_R);
	free(h_dist);
	
	return 0;
}