#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <sys/time.h>
#include <math.h>

#define MAX_NUM 1000000000
#define CHUNK_SIZE 10000
#define NUM_THREADS_ARRAY 4
int num_threads_array[NUM_THREADS_ARRAY] = {4, 8, 16, 32};

int prime_count = 0;
int next_chunk = 0;

bool is_prime(long n) {
    if (n <= 1) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    
    for (long i = 3; i <= sqrt(n); i += 2) {
        if (n % i == 0) return false; 
    }

    return true;
}

void* count_primes(void* arg) {
    while (1) {
        int start = next_chunk;
        int end = start + CHUNK_SIZE - 1;
        next_chunk += CHUNK_SIZE;

        if (start > MAX_NUM) break;
        if (end > MAX_NUM) end = MAX_NUM;

        int local_count = 0;
        for (int i = start; i <= end; i++) {
            if (is_prime(i)) {
                local_count++;
            }
        }

        prime_count += local_count;
    }
    return NULL;
}

double run_threads(int num_threads) {
    prime_count = 0;
    next_chunk = 2;  // Start from 2 as 1 is not a prime
    pthread_t threads[num_threads];

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, count_primes, NULL);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    gettimeofday(&end, NULL);
    double time_spent = (end.tv_sec - start.tv_sec) + 
                        (end.tv_usec - start.tv_usec) / 1000000.0;

    return time_spent;
}

int main() {
    printf("┌────────────┬───────────┬────────────┐\n");
    printf("│ 스레드 수  │ 소수 개수 │ 소요 시간  │\n");
    printf("├────────────┼───────────┼────────────┤\n");

    for (int i = 0; i < NUM_THREADS_ARRAY; i++) {
        int num_threads = num_threads_array[i];
        double time_spent = run_threads(num_threads);
        printf("│ %8d개 │ %9d │ %8.4f초 │\n", num_threads, prime_count, time_spent);
    }

    printf("└────────────┴───────────┴────────────┘\n");
    return 0;
}