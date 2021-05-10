/**
 * Conway's Game of Life using omp
 * 
 * This version runs in serial. Compile with:
 *     gcc -Wall -O3 -fopenmp -march=native game_of_life_shared.c util.c helpers.c -o game_of_life_shared
 * And run with:
 * 	   ./game_of_life_shared num-of-iterations input-file output-file num-threads
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <string.h>
#include <sys/mman.h>
#include <omp.h>

#include "helpers.h"
#include "util.h"

int main(int argc, char* const argv[]) {
	size_t iterations = 3;
	const char * input_file = "examples/input.npy";
	const char * output_file = "output.npy";
    int num_threads = get_num_cores_affinity();

	// Parse command line arguments
	// Can have num iterations, input and output file, or num iterations and input and output files
	if (argc > 5) { printf("Wrong number of arguments!\n"); return 1; }
	if (argc == 2) {
		iterations = atoi(argv[1]);
	} else if (argc == 3) {
		input_file = argv[1];
		output_file = argv[2];
	} else if (argc == 4) {
		iterations = atoi(argv[1]);
		input_file = argv[2];
		output_file = argv[3];
	} else {
		iterations = atoi(argv[1]);
		input_file = argv[2];
		output_file = argv[3];
		num_threads = atoi(argv[4]);
		if (num_threads <= 0) { fprintf(stderr, "Must specify a positive number of threads\n"); return 1; }
	}
	if (iterations <= 0) { fprintf(stderr, "Must specify a positive number of iterations\n"); return 1; }

	size_t m, n;
	uint8_t* grid = grid_from_npy_path(input_file, &m, &n);
	if (!grid) { perror("grid_from_npy_path(grid)"); return 1; }

	// Begin timing
	struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

	// Allocate a copy of the input-file to not modify it
	size_t grid_size = m * n;
	uint8_t* grid_copy = (uint8_t*)malloc(grid_size*sizeof(uint8_t));
	uint8_t* grid_next = (uint8_t*)malloc(grid_size*sizeof(uint8_t));
	memcpy(grid_copy, grid, grid_size);

	// Begin simulation. Update the grid every iteration and save it
	for (size_t step = 0; step < iterations; step++) {
		#pragma omp parallel for num_threads(num_threads)
		for (size_t i = 0; i < grid_size; i++) {
			update(grid_copy, grid_next, i, n);
		}
		swap(&grid_copy, &grid_next);
  	}

	// End timing
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;
    printf("Time: %g secs\n", time);

	// Save the last updated grid to the output file
    grid_to_npy_path(output_file, grid_copy, 1, m, n);

	// Cleanup
	size_t addr = ((size_t)grid) & ~(sysconf(_SC_PAGE_SIZE)-1);
	munmap((void*)addr, grid_size*sizeof(uint8_t));
	free(grid_next);
	free(grid_copy);
  	return 0;
}
