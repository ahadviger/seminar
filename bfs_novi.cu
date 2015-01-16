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
#define TILE_SIZE 256
#define CTA_THRESHOLD 1023
#define WARP_THRESHOLD 0
#define LABEL_SIZE 32

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

*/

__device__ void prescan2(int *g_odata, int *g_idata, int n) {

	__shared__ int temp[BLOCK_SIZE]; 
	int thid = threadIdx.x;  
	int offset = 1;  
	
	int ai = thid;  
    int bi = thid + (n/2); 
	
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);  
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	
	if(thid < BLOCK_SIZE / 2) {
		temp[ai + bankOffsetA] = g_idata[ai];  
		temp[bi + bankOffsetB] = g_idata[bi];
	}
	
	for (int d = n >> 1; d > 0; d >>= 1) {  
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
	
	for (int d = 1; d < n; d *= 2) {  
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
	
	if(thid < BLOCK_SIZE / 2) {
		g_odata[ai] = temp[ai + bankOffsetA];  
		g_odata[bi] = temp[bi + bankOffsetB];  
	}
}

__device__ void prescan(int *g_odata, int *g_idata, int n)  {  
    __shared__ int temp[BLOCK_SIZE];
	int thid = threadIdx.x;  
    int offset = 1;  

    if(thid < BLOCK_SIZE/2) {
		temp[2*thid] = g_idata[2*thid];   
		temp[2*thid+1] = g_idata[2*thid+1];  
	}
	
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

	if(thid < BLOCK_SIZE/2) {
		g_odata[2*thid] = temp[2*thid]; 
		g_odata[2*thid+1] = temp[2*thid+1];  
	}
}  

/*
__device__ void checkIfVisited(int node) {
	int index = node / LABEL_SIZE;
	unsigned int label = tex1Dfetch(tex_label, index);
	boolean visited = label & (1 << node % LABEL_SIZE) != 0;
	return visited;
}*/
	
__device__ void gatherWarp(int *queue, int *out_queue, int queue_size, int level, int *dist, int *offset, int *d_R, int *d_C, int *scan_input, int *scan_output) {
	volatile __shared__ int comm[WARPS][3];

	int lane_id = threadIdx.x % W_SIZE;
	int warp_id = threadIdx.x / W_SIZE;
	int id = threadIdx.x;
	
	int node = 0, r = 0, r_end = 0;
	__shared__ int done;
	__shared__ int cta_offset;
	__shared__ int curr_node;
	
	if(threadIdx.x < queue_size) {
		node = queue[id];
		r = d_R[node];
		r_end = d_R[node + 1];
	}
		
	while(true) {
		
		if(id == 0) done = 1;
		__syncthreads();
		
		if(r_end - r > WARP_THRESHOLD) {
			comm[warp_id][0] = lane_id;
			done = 0;
		}
		
		__syncthreads();
		if(done == 1) break;
		
		if(comm[warp_id][0] == lane_id) {
			comm[warp_id][1] = r;
			comm[warp_id][2] = r_end;
			r = r_end;
			curr_node = node;
		}
		
		int r_gather = comm[warp_id][1] + lane_id;
		int r_gather_end = comm[warp_id][2];
		
		volatile int neighbor;

		__shared__ int cycle_done;
		
		while(true) {
			if(id == 0) cycle_done = 1;
			__syncthreads();
			scan_input[id] = 0;
			if(r_gather < r_gather_end) {
				cycle_done = 0;
				neighbor = d_C[r_gather];
				//atomicAdd(&d_R[curr_node], 1);
				if(dist[neighbor] == -1) {
					dist[neighbor] = level;
					scan_input[id] = 1;
				}
			}
			
			__syncthreads();
			if(cycle_done == 1) break;
			
			prescan(scan_output, scan_input, BLOCK_SIZE);			
			__syncthreads();
			
			if(threadIdx.x == 0) {
				cta_offset = atomicAdd(offset, scan_output[BLOCK_SIZE-1] + scan_input[BLOCK_SIZE-1]);
			}

			__syncthreads();

			if(r_gather < r_gather_end && scan_input[id] == 1) {
				out_queue[scan_output[id] + cta_offset] = neighbor;
			}
			
			r_gather += W_SIZE;

			//if(r_gather_end - (r_gather - id) < WARP_THRESHOLD) break;
		}
		
		//if(lane_id == 0) d_R[curr_node] = r_gather_end - WARP_THRESHOLD - 1;
	}
}

__device__ void gatherCTA(int *queue, int *out_queue, int queue_size, int level, int *dist, int *offset, int *d_R, int *d_C, int *scan_input, int *scan_output) {
	volatile __shared__ int comm[3];
	__shared__ int curr_node;
	
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
		
		if(r_end - r > CTA_THRESHOLD) {
			comm[0] = id;
		}
		
		__syncthreads();
		
		if(comm[0] == -1) break;
		
		if(comm[0] == id) {
			comm[1] = r;
			comm[2] = r_end;
			r = r_end;
			curr_node = node;
		}
		
		__syncthreads();
		
		int r_gather = comm[1] + id;
		int r_gather_end = comm[2];
		__shared__ int cta_offset;
		__shared__ int done;
		volatile int neighbor;

		while(true) {
			
			scan_input[id] = 0;
			
			if(r_gather_end - r_gather > 0) {
			//	done = 0;
				neighbor = d_C[r_gather];
				if(dist[neighbor] == -1) {
					dist[neighbor] = level;
					scan_input[id] = 1;
				}
			}
			__syncthreads();
			
			prescan(scan_output, scan_input, BLOCK_SIZE);			__syncthreads();
			
			if(threadIdx.x == 0) {
				cta_offset = atomicAdd(offset, scan_output[BLOCK_SIZE-1] + scan_input[BLOCK_SIZE-1]);
			}

			__syncthreads();

			if(r_gather < r_gather_end && scan_input[id] == 1) {
				out_queue[scan_output[id] + cta_offset] = neighbor;
			}

			r_gather += BLOCK_SIZE;
			
			if(r_gather_end - (r_gather - id) < CTA_THRESHOLD) break;
		}
		
		if(threadIdx.x == 0) d_R[curr_node] = r_gather_end - CTA_THRESHOLD - 1;
	}
}

__device__ void gatherScan(int *queue, int *out_queue, int queue_size, int level, int *dist, int *offset, int *d_R, int *d_C, int *scan_input, int *scan_output) {

	volatile __shared__ int comm[BLOCK_SIZE];
	__shared__ int cta_offset;

	int id = threadIdx.x;
	int node = 0, r = 0, r_end = 0;
	
	if(threadIdx.x < queue_size) {
		node = queue[id];
		r = d_R[node];
		r_end = d_R[node + 1];
	}
	
	scan_input[id] = r_end - r;

	__syncthreads();
	prescan(scan_output, scan_input, BLOCK_SIZE);
	__syncthreads();

	int rsv_rank = scan_output[id];
	
	
	int total = scan_output[BLOCK_SIZE-1] + scan_input[BLOCK_SIZE-1];
	int cta_progress = 0;
	volatile int neighbor;
	
	while(total - cta_progress > 0) {
	
		while(rsv_rank < cta_progress + BLOCK_SIZE && r < r_end) {
			comm[rsv_rank - cta_progress] = r;
			rsv_rank++;
			r++;
		}
	
		__syncthreads();
		
		int min = BLOCK_SIZE;
		if(total - cta_progress < BLOCK_SIZE)
			min = total - cta_progress;
			
		scan_input[id] = 0;
		if(id < min) {
			neighbor = d_C[comm[id]];
			if(dist[neighbor] == -1) {
				dist[neighbor] = level;
				scan_input[id] = 1;
			}
		}
		
		__syncthreads();

		prescan(scan_output, scan_input, BLOCK_SIZE);		
		__syncthreads();
			
		if(threadIdx.x == 0) {
			cta_offset = atomicAdd(offset, scan_output[BLOCK_SIZE-1] + scan_input[BLOCK_SIZE-1]);
		}

		__syncthreads();

		if(scan_input[id] == 1) {
			out_queue[scan_output[id] + cta_offset] = neighbor;
		}
		
		cta_progress += BLOCK_SIZE;
		__syncthreads();
	}
}

__global__ void BFS(int *d_R, int *d_C, int numberOfNodes, int *queue, int *out_queue, int *d_dist, int *offset, int level) {
	
	__shared__ int neighbors[TILE_SIZE];
	__shared__ int scan_input[BLOCK_SIZE];
	__shared__ int scan_output[BLOCK_SIZE];

	int chunk_size = min(TILE_SIZE, numberOfNodes - TILE_SIZE * blockIdx.x);

	if(threadIdx.x < chunk_size) {
		neighbors[threadIdx.x] = queue[blockIdx.x * TILE_SIZE + threadIdx.x];
	}

	__syncthreads();
	gatherCTA(neighbors, out_queue, chunk_size, level, d_dist, offset, d_R, d_C, scan_input, scan_output);
	__syncthreads();
//	gatherWarp(neighbors, out_queue, chunk_size, level, d_dist, offset, d_R, d_C, scan_input, scan_output);
//	__syncthreads();
	gatherScan(neighbors, out_queue, chunk_size, level, d_dist, offset, d_R, d_C, scan_input, scan_output);

}

///////////////////////////////////////////////////
//					MAIN PROGRAM
///////////////////////////////////////////////////

int main(int argc, char** argv) {

	const int ZERO = 0;

	int numberOfNodes;
	int numberOfEdges;
	int startingNode;
	
	FILE *input = fopen(argv[1], "r");
		
	fscanf(input, "%d", &numberOfNodes);
	fscanf(input, "%d", &numberOfEdges);
	fscanf(input, "%d", &startingNode);
	
	int *h_C = (int*) malloc(numberOfEdges * sizeof(int)) ;
	int *h_R = (int*) malloc((numberOfNodes + 1) * sizeof(int));
	int *h_dist = (int*) malloc(numberOfNodes * sizeof(int));
	int *h_q2 = (int*) malloc(numberOfNodes * 3 * sizeof(int));

	for(int i = 0; i < numberOfEdges; ++i) {
		fscanf(input, "%d", &h_C[i] );
	}

	for(int i = 0; i < numberOfNodes + 1; ++i) {
		fscanf(input, "%d", &h_R[i]);
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
	cudaMalloc((void**) &d_q1, numberOfNodes * 3 * sizeof(int));
	cudaMemcpy( &d_q1[0], &startingNode, sizeof(int), cudaMemcpyHostToDevice);
 
	int *d_q2;
	cudaMalloc((void**) &d_q2, numberOfNodes * 3 * sizeof(int));

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
	
	for(int level = 1;;	 ++level) {

		int queueSize;
		
		if(level == 1) {
			queueSize = 1;
		} else {
			cudaMemcpy(&queueSize, offset, sizeof(int), cudaMemcpyDeviceToHost);
		}
		cudaMemcpy(offset, &ZERO, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(broj, &ZERO, sizeof(int), cudaMemcpyHostToDevice);
		
		if(queueSize == 0) {
			break;
		}
		
		int numberOfBlocks = (int)ceil((double)queueSize / (double)TILE_SIZE);
		
		BFS<<<numberOfBlocks, BLOCK_SIZE>>>(d_R, d_C, queueSize, d_q1, d_q2, d_dist, offset, level);

		cudaThreadSynchronize();
		cudaMemcpy(&queueSize, offset, sizeof(int), cudaMemcpyDeviceToHost);

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
	fclose(input);
	cudaDeviceReset();
	
	return 0;
}