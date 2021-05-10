#include <stdlib.h>
#include <stdbool.h>

/**
 * Make the current grid the new grid
 */
void swap(uint8_t** grid, uint8_t** grid_next);

/**
 * Gets the number of live organisms around a given position and update the next grid based on that neighbor count 
 */
void update(const uint8_t* grid, uint8_t* grid_next, size_t i, size_t world_size);

void print_world(uint8_t* grid, size_t world_size);

