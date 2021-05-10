/**
 * Several utility functions for displaying results.
 */

#if defined(linux)
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/mman.h>
#include <ctype.h>

#include "matrix_io_helpers.h"
#include "util.h"

/**
 * Prints a positive number with the given number of sigfigs and a unit. The
 * value is scaled to the correct unit (which are mult apart - 1000 for SI and
 * 1024 for digit prefixes).
 */
void print_with_unit(double val, int sigfigs, int mult,
                     const char** units, size_t n_units) {
    size_t i_unit = 0;
    while (i_unit < n_units && val >= mult) { val /= mult; i_unit++; }
    if (i_unit == 0) { sigfigs = 0; }
    else if (val < 10) { sigfigs -= 1; }
    else if (val < 100) { sigfigs -= 2; }
    else { sigfigs -= 3; }
    printf("%.*f %s", sigfigs, val, units[i_unit]);
}

/**
 * Prints a number of bytes after converting to a nicer unit.
 */
void print_bytes(size_t n) {
    static const char* units[4] = {"bytes", "KiB", "MiB", "GiB"};
    print_with_unit(n, 3, 1024, units, 4);
}

/**
 * Print the time (in seconds) with the right units and 3 significant digits.
 */
void print_time(double seconds) {
    static const char* units[4] = {"ns", "us", "ms", "s"};
    print_with_unit(seconds * 1000000000.0, 3, 1000, units, 4);
}

/**
 * Get the difference between two times.
 */
double get_time_diff(struct timespec* start, struct timespec* end) {
    double diff = end->tv_sec - start->tv_sec;
    diff += (end->tv_nsec - start->tv_nsec) / 1000000000.0;
    return diff;
}

// get_num_physical_cores() and get_num_logical_cores() have to be specialized
// for each OS.
#if defined(__APPLE__)
#include <sys/sysctl.h>
size_t __get_sysctl_size_t(const char* name) {
    size_t var = 0, sizeof_var = sizeof(size_t);
    sysctlbyname(name, &var, &sizeof_var, 0, 0);
    return var;
}
size_t get_num_physical_cores() { return __get_sysctl_size_t("hw.physicalcpu"); }
size_t get_num_logical_cores() { return __get_sysctl_size_t("hw.logicalcpu"); }
size_t get_num_cores_affinity() { return get_num_logical_cores(); } // macOS doesn't really support affinity
#elif defined(linux)
#include <unistd.h>
#include <sched.h>
size_t get_num_physical_cores() { return (sysconf(_SC_NPROCESSORS_ONLN) + 1) / 2; } // TODO: this assumes processor has 2 threads per core
size_t get_num_logical_cores() { return sysconf(_SC_NPROCESSORS_ONLN); }
size_t get_num_cores_affinity() { cpu_set_t cs; CPU_ZERO(&cs); sched_getaffinity(0, sizeof(cs), &cs); return CPU_COUNT(&cs); }
#else
#error Unrecognized OS
#endif

/**
 * Creates a new matrix by loading the data from the given NPY file. This is
 * a file format used by the numpy library. This function only supports arrays
 * that are little-endian doubles, c-contiguous, and 1 or 2 dimensional. The
 * file is loaded as memory-mapped so it is backed by the file and loaded
 * on-demand. The file should be opened for reading or reading and writing.
 * 
 * This will return NULL if the data cannot be read, the file format is not
 * recognized, there are memory allocation issues, or the array is not a
 * supported shape or data type.
 */
uint8_t* grid_from_npy(FILE* file, size_t *m, size_t *n) {
    // Read the header, check it, and get the shape of the matrix
    size_t sh[2], offset;
    if (!__npy_read_header(file, sh, &offset)) { return NULL; }
    
    // Get the memory mapped data
    void* x = (void*)mmap(NULL, sh[0]*sh[1] + offset,
                          PROT_READ|PROT_WRITE, MAP_SHARED, fileno(file), 0);
    if (x == MAP_FAILED) { return NULL; }

    // Make the matrix itself
    uint8_t* data = (uint8_t*)(((char*)x) + offset);
    *m = sh[0];
    *n = sh[1];
    return data;
}

/**
 * Same as matrix_from_npy() but takes a file path instead.
 */
uint8_t* grid_from_npy_path(const char* path, size_t *m, size_t *n) {
    FILE* f = fopen(path, "r+b");
    if (!f) { return NULL; }
    uint8_t* grid = grid_from_npy(f, m, n);
    fclose(f);
    return grid;
}

// /**
//  * Saves a matrix to a CSV file.
//  * 
//  * If the file argument is given as stdout, this will print it to the terminal.
//  */
// void matrix_to_csv(FILE* file, const uint8_t* grid, size_t m, size_t n) {
//     printf("rows %lu, cols %lu", m, n);
//     if (m < 1 || n < 1) { return; }
//     for (size_t i = 0; i < m; i++) {
//         for (size_t j = 0; j < n; j++) {
//             fprintf(file, "%hhu,", grid[i*n+j]);
//         }
//         fprintf(file, "\n");
//     }
// }

// /**
//  * Same as matrix_to_csv() but takes a file path instead.
//  */
// bool matrix_to_csv_path(const char* path, const uint8_t* grid, size_t m, size_t n) {
//     FILE* f = fopen(path, "w");
//     if (!f) { return false; }
//     matrix_to_csv(f, grid, m, n);
//     fclose(f);
//     return true;
// }

/**
 * Saves a matrix to a NPY file. This is a file format used by the numpy
 * library. This will return false if the data cannot be written.
 */
bool grid_to_npy(FILE* file, const uint8_t* grid, size_t m, size_t n, size_t p) {
    // create the header
    char header[128];
    size_t len = snprintf(header, sizeof(header), "\x93NUMPY\x01   "
        "{'descr': '<u1', 'fortran_order': False, 'shape': (%zu, %zu, %zu), }",
        m, n, p);
    if (len < 0) { return false; }
    header[7] = 0; // have to after the string is written
    *(unsigned short*)&header[8] = sizeof(header) - 10;
    memset(header + len, ' ', sizeof(header)-len-1);
    header[sizeof(header)-1] = '\n';

    // write the header and the data
    bool head = fwrite(header, 1, sizeof(header), file) == sizeof(header);
    if (!head) return false;
    
    return fwrite(grid, sizeof(uint8_t), n*m*p, file) == n*m;
}

/**
 * Same as matrix_to_npy() but takes a file path instead.
 */
bool grid_to_npy_path(const char* path, const uint8_t* grid, size_t m, size_t n, size_t p) {
    FILE* f = fopen(path, "wb");
    if (!f) { return false; }
    grid_to_npy(f, grid, m, n, p);
    fclose(f);
    return true;
}