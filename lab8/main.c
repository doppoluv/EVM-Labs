#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define NMIN 256
#define NMAX (32 * 1024 * 1024 / sizeof(int))
#define REPEATS 100

#if defined(_MSC_VER)
#include <intrin.h>
uint64_t rdtsc() {
    return __rdtsc();
}
#elif defined(__GNUC__) || defined(__clang__)
uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
#else
#error "RDTSC not supported on this platform"
#endif

void multMatrix() 
{ 
    const size_t size = 2048; 
    float *A = malloc(size * size * sizeof(float)); 
    float *B = malloc(size * size * sizeof(float)); 
    float *C = calloc(size * size, sizeof(float)); 
    for (size_t i = 0; i < size; i++) 
        for (size_t k = 0; k < size; k++) 
        { 
        float a = A[i * size + k]; 
        for (size_t j = 0; j < size; j++) 
            C[i * size + j] += a * B[k * size + j]; 
        } 
    printf("%f %f %f\n", C[0], C[size * size - 1], C[size + 1]); 
    free(A); 
    free(B); 
    free(C); 
}

void fill_sequential(int *array, int size) {
    for (int i = 0; i < size - 1; i++) {
        array[i] = i + 1;
    }
    array[size - 1] = 0;
}

void fill_reverse(int *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (i - 1 + size) % size;
    }
}

void fill_random(int *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = i;
    }
    for (int i = 0; i < size; i++) {
        int j = rand() % size;
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

uint64_t measure_cycles(int *array, int size) {
    volatile int k = 0;
    uint64_t start = rdtsc();
    for (int r = 0; r < REPEATS; r++) {
        for (int i = 0; i < size; i++) {
            k = array[k];
        }
    }
    uint64_t end = rdtsc();
    return (end - start);
}

int main() {
    multMatrix();

    FILE *file_direct = fopen("direct_cycles.csv", "w");
    FILE *file_reverse = fopen("reverse_cycles.csv", "w");
    FILE *file_random = fopen("random_cycles.csv", "w");

    if (!file_direct || !file_reverse || !file_random) {
        fprintf(stderr, "Error opening output files\n");
        exit(EXIT_FAILURE);
    }

    fprintf(file_direct, "Size (elements), Cycles per element\n");
    fprintf(file_reverse, "Size (elements), Cycles per element\n");
    fprintf(file_random, "Size (elements), Cycles per element\n");

    for (int size = NMIN; size <= NMAX; size *= 1.2) {
        int *array = (int *)malloc(size * sizeof(int));
        if (!array) {
            fprintf(stderr, "Memory allocation failed for size %d\n", size);
            exit(EXIT_FAILURE);
        }

        // Прямой обход
        fill_sequential(array, size);
        uint64_t cycles_direct = measure_cycles(array, size);
        double cycles_per_element_direct = (double)cycles_direct / (size * REPEATS);
        fprintf(file_direct, "%d, %.3f\n", size, cycles_per_element_direct);

        // Обратный обход
        fill_reverse(array, size);
        uint64_t cycles_reverse = measure_cycles(array, size);
        double cycles_per_element_reverse = (double)cycles_reverse / (size * REPEATS);
        fprintf(file_reverse, "%d, %.3f\n", size, cycles_per_element_reverse);

        // Случайный обход
        fill_random(array, size);
        uint64_t cycles_random = measure_cycles(array, size);
        double cycles_per_element_random = (double)cycles_random / (size * REPEATS);
        fprintf(file_random, "%d, %.3f\n", size, cycles_per_element_random);

        free(array);
    }

    fclose(file_direct);
    fclose(file_reverse);
    fclose(file_random);

    return 0;
}
