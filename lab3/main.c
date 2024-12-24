#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

double piCalculation(size_t n) {
    double pi = 4;

    for (size_t i = 1; i <= n; ++i) {
        double x = 4. / (2*i + 1);
        if (i % 2) {
            pi -= x;
        } else {
            pi += x;
        }
    }

    return pi;
}

int main() {
    size_t n = 200;

    double pi = piCalculation(n);
    printf("Pi number: %.12lf\n", pi);

    return 0;
}