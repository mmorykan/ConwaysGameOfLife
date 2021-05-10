#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/mman.h>
#include <ctype.h>

#include "helpers.h"

/**
 * Make the current grid the new grid
 */
void swap(uint8_t** grid, uint8_t** grid_next) {
    uint8_t* temp = *grid;
    *grid = *grid_next;
    *grid_next = temp;
}

/**
 * Gets the number of live organisms around a given position and update the next grid based on that neighbor count 
 */
void update(const uint8_t* grid, uint8_t* grid_next, const size_t i, const size_t sz) {
	const size_t x = i % sz, y = i / sz;
	int neighbor_count = 0;

	// Check all 8 possible neighbors
	neighbor_count += x >= 1 && y >= 1 && grid[i-sz-1];
	neighbor_count +=           y >= 1 && grid[i-sz];
	neighbor_count += x < sz && y >= 1 && grid[i-sz+1];
	neighbor_count += x >= 1           && grid[i-1];
	neighbor_count += x < sz           && grid[i+1];
	neighbor_count += x >= 1 && y < sz && grid[i+sz-1];
	neighbor_count +=           y < sz && grid[i+sz];
	neighbor_count += x < sz && y < sz && grid[i+sz+1];

	// Update the grid.
	// If there is a live organism and it has 2 or 3 live neighbors, or, if there is a dead cell and it has 3 live neighbors, 
	// then the grid cell at this position is alive
	grid_next[i] = (grid[i] && (neighbor_count > 1 && neighbor_count <= 3)) || (!grid[i] && (neighbor_count == 3));
}

void print_world(uint8_t* grid, size_t world_size) {
	for (size_t i = 0; i < world_size; i++) {
		for (size_t j = 0; j < world_size; j++) {
			printf("%s", grid[i*world_size+j] ? "█" : "-");
		}
		printf("\n");
	}
    printf("\n");
}

