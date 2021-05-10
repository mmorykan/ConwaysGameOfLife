/**
 * Conway's Game of Life using Cuda
 * 
 * This version runs in parallel on a GPU using Cuda. Compile with:
 * 	   gcc -Wall -O3 -march=native -c util.c helpers.c
 *     nvcc -arch=sm_20 -O3 game_of_life_cuda.cu util.o -o game_of_life_cuda -lm
 * And run with:
 * 	   ./game_of_life_cuda num-of-iterations input-file output-file
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <inttypes.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/mman.h>

#include "util.h"

#define CHECK(call)                                                       \
{                                                                         \
   const cudaError_t error = call;                                        \
   if (error != cudaSuccess)                                              \
   {                                                                      \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
      exit(1);                                                            \
   }                                                                      \
}

/**
 * Print the grid of organisms.
 */
void print_world(uint8_t* grid, size_t world_size) {
	for (size_t i = 0; i < world_size; i++) {
		for (size_t j = 0; j < world_size; j++) {
			printf("%s", grid[i*world_size+j] ? "█" : "-");
		}
		printf("\n");
	}
    printf("\n");
}

/**
 * Calculates the number of neighbors around an organism on the grid.
 */
__device__
int get_num_neighbors(uint8_t* grid, const size_t i, const size_t sz) {
    const size_t x = i % sz, y = i / sz;
    int neighbor_count = 0;

	// Check horizontally, vertically, and diagonally adjacent neighbors (as necessary)
    neighbor_count += x >= 1     && y >= 1     && grid[i-sz-1];
    neighbor_count +=               y >= 1     && grid[i-sz];
    neighbor_count += x < sz     && y >= 1     && grid[i-sz+1];
    neighbor_count += x >= 1                   && grid[i-1];
    neighbor_count += x < sz-1                 && grid[i+1]; 
    neighbor_count += x >= 1     && (y < sz-1) && grid[i+sz-1]; 
    neighbor_count +=                y < sz-1  && grid[i+sz]; 
    neighbor_count += (x < sz-1) && (y < sz-1) && grid[i+sz+1];

    return neighbor_count;
}

/**
 * Make the current position array the next position array.
 */
__device__
void swap(uint8_t** grid, uint8_t** grid_next) {
    uint8_t* temp = *grid;
    *grid = *grid_next;
    *grid_next = temp;
}

/**
 * The game of life simulation. Checks the number of neighbors each 
 * organism has each iteration and sets them to dead or alive.
 * Doesn't move onto next iteration until all threads are synchronized.
 */
__global__
void simulate(uint8_t* grid, uint8_t* grid_next, const size_t world_size, const size_t iterations) {
	const int i = threadIdx.x;
	const int grid_size = world_size * world_size;
	for (size_t step = 0; step < iterations; step++) {
		for (size_t k = i; k < grid_size; k += blockDim.x) {
			int n = get_num_neighbors(grid, k, world_size);
			grid_next[k] = grid[i] && (n > 1 && n <= 3) || !grid[i] && (n == 3);
		}
		swap(&grid, &grid_next);
		__syncthreads();
	}
}

/**
 * Benchmark function for determining how long memory copies take. 
 * This is used to determine how much time just memory operations take.
 */
 void cuda_memonly(const char* input_file) {
	size_t m, n;
	uint8_t* grid = grid_from_npy_path(input_file, &m, &n);
	if (!grid) { perror("grid_from_npy_path(grid)"); return; }

	// Allocate memory on the host
	size_t grid_size = m * n;
	const size_t grid_bytes = grid_size*sizeof(uint8_t);
	uint8_t* h_grid_copy = (uint8_t*) malloc(grid_bytes);
	uint8_t* h_grid_next = (uint8_t*) malloc(grid_bytes);
	memcpy(h_grid_copy, grid, grid_size);

	// Allocate memory on the device
	uint8_t *d_grid, *d_grid_next;
    CHECK(cudaMalloc(&d_grid, grid_bytes));
    CHECK(cudaMalloc(&d_grid_next, grid_bytes));

	// Copy memory from the host to the device and run the simulation
    CHECK(cudaMemcpy(d_grid, h_grid_copy, grid_bytes, cudaMemcpyHostToDevice));

	// Cleanup
	size_t addr = ((size_t)grid) & ~(sysconf(_SC_PAGE_SIZE)-1);
	munmap((void*)addr, grid_size*sizeof(uint8_t));
    free(h_grid_copy); free(h_grid_next);
    CHECK(cudaFree(d_grid)); CHECK(cudaFree(d_grid_next));
}

int main(int argc, char* const argv[]) {
	size_t iterations = 1000;
	const char * input_file = "examples/input.npy";
	const char * output_file = "output/out.npy";

	// Arguments can either be the number of iterations, the input/output file, or all 3
	if (argc > 4) { printf("Wrong number of arguments!\n"); return 1; }
	if (argc == 2) {
		iterations = atoi(argv[1]);
		if (iterations <= 0) { fprintf(stderr, "Must specify a positive number of iterations\n"); return 1; }
	} else if (argc == 3) {
		input_file = argv[1];
		output_file = argv[2];
	} else {
		iterations = atoi(argv[1]);
		input_file = argv[2];
		output_file = argv[3];
	}

	// Begin timing
	struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

	// Get the initial grid from the input file
	size_t m, n;
	uint8_t* grid = grid_from_npy_path(input_file, &m, &n);
	if (!grid) { perror("grid_from_npy_path(grid)"); return 1; }

	// Allocate memory on the host
	size_t grid_size = m * n;
	const size_t grid_bytes = grid_size*sizeof(uint8_t);
	uint8_t* h_grid_copy = (uint8_t*) malloc(grid_bytes);
	uint8_t* h_grid_next = (uint8_t*) malloc(grid_bytes);
	memcpy(h_grid_copy, grid, grid_size);

	// Allocate memory on the device
	uint8_t *d_grid, *d_grid_next;
    CHECK(cudaMalloc(&d_grid, grid_bytes));
    CHECK(cudaMalloc(&d_grid_next, grid_bytes));

	// Copy memory from the host to the device and run the simulation
    CHECK(cudaMemcpy(d_grid, h_grid_copy, grid_bytes, cudaMemcpyHostToDevice));
	int dimx = 1024, dimy = 1; 
    dim3 block(dimx, dimy);
    dim3 grid_cuda((m + dimx - 1) / dimx, (n + dimy - 1)/ dimy);
    simulate<<<grid_cuda, block>>>(d_grid, d_grid_next, n, iterations);
    CHECK(cudaDeviceSynchronize());

	// Copy memory back from the device to the host and save to output file
    CHECK(cudaMemcpy(h_grid_next, d_grid_next, grid_bytes, cudaMemcpyDeviceToHost));
	
	// End timing
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;
    printf("Time for complete simulation: %g secs\n", time);

	// Time memory allocations/copies only
	clock_gettime(CLOCK_MONOTONIC, &start);
	cuda_memonly(input_file);
	clock_gettime(CLOCK_MONOTONIC, &end);
 	double mem_time = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;
    printf("Time for mem allocs/copies: %g secs\n", time);
	printf("Time running just on device: %g secs\n", time - mem_time);

	// Cleanup
	grid_to_npy_path(output_file, h_grid_next, 1, m, n);
	size_t addr = ((size_t)grid) & ~(sysconf(_SC_PAGE_SIZE)-1);
	munmap((void*)addr, grid_size*sizeof(uint8_t));
    free(h_grid_copy); free(h_grid_next);
    CHECK(cudaFree(d_grid)); CHECK(cudaFree(d_grid_next));
    return 0;
}
