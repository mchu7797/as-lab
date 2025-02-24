#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_THREADS 32

typedef struct {
    int id;
    ULONGLONG n;
    int thread_count;
    ULONGLONG count;
} ThreadData;

BOOL isPrime(ULONGLONG n) {
    if (n <= 1) return FALSE;
    if (n <= 3) return TRUE;
    if (n % 2 == 0 || n % 3 == 0) return FALSE;

    for (ULONGLONG i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return FALSE;
    }
    return TRUE;
}

DWORD WINAPI countPrimes(LPVOID lpParam) {
    ThreadData* data = (ThreadData*)lpParam;
    data->count = 0;

    for (ULONGLONG i = data->id + 2; i <= data->n; i += data->thread_count) {
        if (isPrime(i)) {
            data->count++;
        }
    }

    printf("%02d ", data->id + 1);

    return 0;
}

double get_time_ms(void) {
    static LARGE_INTEGER frequency;
    static int init = 0;
    LARGE_INTEGER now;

    if (!init) {
        QueryPerformanceFrequency(&frequency);
        init = 1;
    }

    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / frequency.QuadPart * 1000.0;
}

int main() {
    ULONGLONG n = 5000000;
    HANDLE threads[MAX_THREADS];
    ThreadData threadData[MAX_THREADS];
    ULONGLONG totalCount = 0;
    double startTime, endTime;

    for (int i = 4; i <= MAX_THREADS; i *= 2)
    {
        memset(threads, 0, sizeof(threads));
        memset(threadData, 0, sizeof(threadData));
        totalCount = 0;

        startTime = get_time_ms();

        printf("----------------------------------------\n");
        printf("%02d개의 스레드로 계산 시작!\n", i);
        printf("----------------------------------------\n");

        for (DWORD j = 0; j < i; j++) {
            threadData[j].id = j;
            threadData[j].n = n;
            threadData[j].thread_count = i;

            threads[j] = CreateThread(NULL, 0, countPrimes, &threadData[j], 0, NULL);

            if (threads[j] == NULL) {
                fprintf(stderr, "스레드 생성 실패\n");
                return 1;
            }
        }

        WaitForMultipleObjects(i, threads, TRUE, INFINITE);

        for (DWORD j = 0; j < i; j++) {
            CloseHandle(threads[j]);
            totalCount += threadData[j].count;
        }

        endTime = get_time_ms();

        printf("\n----------------------------------------\n");
        printf("%02d개의 스레드 사용시 소요 시간: %.3f ms\n", i, endTime - startTime);
        printf("1부터 %llu까지의 소수 개수: %llu\n", n, totalCount + 1);  // 2를 소수로 포함
        printf("----------------------------------------\n");
    }

    return 0;
}