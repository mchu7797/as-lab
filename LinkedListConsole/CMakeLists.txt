cmake_minimum_required(VERSION 3.27)
project(LinkedListConsole)

set(CMAKE_CXX_STANDARD 17)

add_library(LinkedList
        node.h
        linked_list.h
        linked_list.cc)

add_library(LinkedListConsole
    linked_list_console.h
    linked_list_console.cc)

add_executable(LinkedListMain
        linked_list_main.cc)

target_link_libraries(LinkedListMain LinkedList LinkedListConsole)