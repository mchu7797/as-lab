cmake_minimum_required(VERSION 3.28)
project(Threads C)

set(CMAKE_C_STANDARD 11)

add_executable(get_prime_multithreading get_prime_multithreading.c)
add_executable(get_prime_multithreading_v2 get_prime_multithreading_v2.c)
add_executable(get_prime_openmp get_prime_openmp.c)
