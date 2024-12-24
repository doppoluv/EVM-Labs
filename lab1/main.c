#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

void addDigitsToArray(size_t n, int *arr) {
  for (size_t i = 0; i < n; ++i) {
    arr[i] = rand() % 100;
  }
}

void swap(int* xp, int* yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

// An optimized version of Bubble Sort
void bubbleSort(int *arr, int n)
{
    int i, j;
    bool swapped;

    for (i = 0; i < n - 1; i++) {
        swapped = false;
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
                swapped = true;
            }
        }

        // If no two elements were swapped by inner loop,
        // then break
        if (swapped == false)
            break;
    }
}

int main(int argc, char *argv[]) {
  struct timespec start, end;
  srand(time(NULL));

  size_t n = (size_t) (atoi(argv[1]));
  int *arr = malloc(sizeof(int) * n);
  addDigitsToArray(n, arr);

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  bubbleSort(arr, n);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);

  printf("Time taken: %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));

  free(arr);
  return 0;
}