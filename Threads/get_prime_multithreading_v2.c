#include <stdbool.h>
#include <stdio.h>
#include <windows.h>

#define MAX_THREADS 32
#define CHUNK_SIZE 5000

typedef struct {
    long long start;
    long long end;
    long long *next_start;
    long long count;
    CRITICAL_SECTION *cs;
} ThreadArgs;

bool is_prime(long long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (long long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

DWORD WINAPI count_primes(LPVOID arg) {
    ThreadArgs *args = (ThreadArgs *) arg;
    long long local_start, local_end;

    while (1) {
        EnterCriticalSection(args->cs);
        if (*(args->next_start) > args->end) {
            LeaveCriticalSection(args->cs);
            break;
        }
        local_start = *(args->next_start);
        local_end = local_start + CHUNK_SIZE - 1;
        if (local_end > args->end) local_end = args->end;
        *(args->next_start) = local_end + 1;
        LeaveCriticalSection(args->cs);

        for (long long i = local_start; i <= local_end; i++) {
            if (is_prime(i)) {
                args->count++;
            }
        }
    }
    return 0;
}

long long parallel_count_primes(long long start, long long end, int num_threads) {
    HANDLE threads[MAX_THREADS];
    ThreadArgs args[MAX_THREADS];
    CRITICAL_SECTION cs;
    long long next_start = start;
    long long total_count = 0;

    InitializeCriticalSection(&cs);

    for (int i = 0; i < num_threads; i++) {
        args[i].start = start;
        args[i].end = end;
        args[i].next_start = &next_start;
        args[i].count = 0;
        args[i].cs = &cs;
        threads[i] = CreateThread(NULL, 0, count_primes, &args[i], 0, NULL);
    }

    WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE);

    for (int i = 0; i < num_threads; i++) {
        CloseHandle(threads[i]);
        total_count += args[i].count;
    }

    DeleteCriticalSection(&cs);
    return total_count;
}

double get_execution_time(long long start, long long end, int num_threads, long long *count) {
    LARGE_INTEGER frequency, begin, finish;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&begin);
    *count = parallel_count_primes(start, end, num_threads);
    QueryPerformanceCounter(&finish);
    return (double) (finish.QuadPart - begin.QuadPart) / frequency.QuadPart;
}

int main() {
    long long start, end;
    printf("Enter the start of the range: ");
    scanf("%lld", &start);
    printf("Enter the end of the range: ");
    scanf("%lld", &end);

    int thread_counts[] = {4, 8, 16, 32};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("\nResults:\n");
    printf("--------------------------------------------\n");
    printf("Threads | Prime Count | Execution Time (s)\n");
    printf("--------------------------------------------\n");

    for (int i = 0; i < num_tests; i++) {
        int num_threads = thread_counts[i];
        long long count;
        double time = get_execution_time(start, end, num_threads, &count);
        printf("%7d | %11lld | %18.6f\n", num_threads, count, time);
    }

    printf("--------------------------------------------\n");

    return 0;
}