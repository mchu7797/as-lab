#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

bool is_prime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

int count_primes(int start, int end) {
    int count = 0;
#pragma omp parallel for reduction(+ : count)
    for (int i = start; i <= end; i++) {
        if (is_prime(i)) {
            count++;
        }
    }
    return count;
}

int main() {
    int start, end;
    printf("Enter the start of the range: ");
    scanf("%d", &start);
    printf("Enter the end of the range: ");
    scanf("%d", &end);

    double start_time = omp_get_wtime();

    int prime_count = count_primes(start, end);

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    printf("Number of primes between %d and %d: %d\n", start, end, prime_count);
    printf("Time taken: %f seconds\n", elapsed_time);

    return 0;
}