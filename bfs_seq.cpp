#include <cstdio>
#include <queue>
#include <malloc.h>
#include <time.h>

using namespace std;

int V, E, start;
int *C, *R, *dist;
queue< int > Q;

int main( void ) {

	clock_t inputStartTime = clock();

    scanf("%d%d%d", &V, &E, &start);

    int *C = (int*)malloc(E * sizeof(int));
    int *R = (int*)malloc((V + 1) * sizeof(int));
    int *dist = (int*)malloc(V * sizeof(int));

    for(int i = 0; i < E; ++i) {
        scanf("%d", &C[i]);
    }

    for(int i = 0; i < V + 1; ++i) {
        scanf("%d", &R[i]);
    }

	clock_t initStartTime = clock();

    for(int i = 0; i < V; ++i) {
        dist[i] = -1;
    }

	clock_t startTime = clock();
	
    dist[start] = 0;
    Q.push(start);

    while(!Q.empty()) {
        int curr = Q.front();
        Q.pop();
        for(int i = R[curr]; i < R[curr+1]; ++i) {
            if(dist[C[i]] == -1) {
                dist[C[i]] = dist[curr] + 1;
                Q.push(C[i]);
            }
        }
    }
	
    clock_t endTime = clock();
	printf("Graph input time: %lf\n",  (double)(initStartTime - inputStartTime) / CLOCKS_PER_SEC);
	printf("Initialization time: %lf\n",  (double)(startTime - initStartTime) / CLOCKS_PER_SEC);
    printf("Execution time: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

    FILE *out = fopen("results_seq.txt", "w");
	
    for(int i = 0; i < V; ++i) {
        fprintf(out, "%d\n", dist[i]);
    }
	
    free(C);
    free(R);
    free(dist);
    fclose(out);

    return 0;
}
