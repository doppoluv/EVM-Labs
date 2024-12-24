#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/times.h>
#include <cblas.h>
#include <stddef.h>

float *create_identity_matrix(size_t N);
void matrix_invert(const float *A, float *result, size_t N, size_t M);
void print_matrix(const float *matrix, size_t N);

float *create_random_matrix(size_t N)
{
    float *M = calloc(N * N, sizeof(float));
    if (!M) return NULL;

    for (size_t i = 0; i < N * N; ++i)
        M[i] = rand() / (float)RAND_MAX;

    return M;
}

float *create_identity_matrix(size_t N) {
    float *Imatrix = (float *)calloc(N * N, sizeof(float));
    if (!Imatrix) return NULL;

    for (size_t i = 0; i < N; ++i)
        Imatrix[i * N + i] = 1.0f;

    return Imatrix;
}

float *generate_B(const float *A, size_t N)
{
    float max_row_sum = __FLT_MIN__;
    float max_col_sum = __FLT_MIN__;

    float *row_sums = calloc(N, sizeof(float));
    float *col_sums = calloc(N, sizeof(float));
    if (!row_sums || !col_sums) {
        free(row_sums);
        free(col_sums);
        return NULL;
    }

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            row_sums[i] += A[i * N + j];
            col_sums[j] += A[i * N + j];
        }
    }

    for (size_t i = 0; i < N; ++i) {
        if (row_sums[i] > max_row_sum) max_row_sum = row_sums[i];
        if (col_sums[i] > max_col_sum) max_col_sum = col_sums[i];
    }

    free(row_sums);
    free(col_sums);

    float *B = calloc(N * N, sizeof(float));
    if (!B) return NULL;

    float scaling_factor = max_row_sum * max_col_sum;
    if (scaling_factor == 0) return NULL;

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            B[i * N + j] = A[j * N + i] / scaling_factor;

    return B;
}

void matrix_multiply(const float *A, const float *B, float *C, size_t N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
}

void matrix_subtract(const float *A, const float *B, float *C, size_t N) {
    for (size_t i = 0; i < N * N; ++i)
        C[i] = A[i] - B[i];
}

void matrix_add(const float *A, const float *B, float *C, size_t N) {
    for (size_t i = 0; i < N * N; ++i)
        C[i] = A[i] + B[i];
}

void matrix_invert(const float *A, float *result, size_t N, size_t M)
{
    float *Im = create_identity_matrix(N);
    float *B = generate_B(A, N);

    if (!Im || !B) return;

    float *R = calloc(N * N, sizeof(float));
    float *current_power = calloc(N * N, sizeof(float));
    float *temp_result = calloc(N * N, sizeof(float));

    if (!R || !current_power || !temp_result) {
        free(Im); free(B); free(R); free(current_power); free(temp_result);
        return;
    }

    float *BA = calloc(N * N, sizeof(float));
    matrix_multiply(B, A, BA, N);
    matrix_subtract(Im, BA, R, N);

    memcpy(current_power, R, N * N * sizeof(float));
    matrix_add(Im, R, result, N);

    for (size_t i = 2; i <= M; ++i) {
        matrix_multiply(current_power, R, temp_result, N);
        matrix_add(result, temp_result, result, N);
        memcpy(current_power, temp_result, N * N * sizeof(float));
        memset(temp_result, 0, N * N * sizeof(float));
    }

    matrix_multiply(result, B, temp_result, N);
    memcpy(result, temp_result, N * N * sizeof(float));

    free(Im); free(B); free(R); free(BA); free(current_power); free(temp_result);
}


int main(void) {
    size_t N = 0, M = 0;
    printf("Enter matrix size (N): ");
    if (scanf("%zu", &N) == 0) return 0;
    printf("Enter number of iterations (M): ");
    if (scanf("%zu", &M) == 0) return 0;

    srand(time(NULL));
    float *A = create_random_matrix(N);
    float *inverseA = (float *)calloc(N * N, sizeof(float));

    if (!A || !inverseA) return 1;

    struct tms start, end;
    clock_t clock_start = times(&start);

    matrix_invert(A, inverseA, N, M);

    clock_t clock_end = times(&end);

    double elapsed_time = (double)(end.tms_utime - start.tms_utime) / sysconf(_SC_CLK_TCK);

    printf("Elapsed Time: %lf seconds\n", elapsed_time);
    
    printf("A: %f, %f, %f\n", A[0], A[1], A[N]);

    printf("Inverse A: %f, %f, %f\n", inverseA[0], inverseA[1], inverseA[N]);

    free(A);
    free(inverseA);
    return 0;
}