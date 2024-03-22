cmake_minimum_required(VERSION 3.27)
project(BookKeeper)

set(CMAKE_CXX_STANDARD 20)

add_library(LinkedList
        linked_list.cc
        linked_list.h
        node.h)

add_executable(BookKeeper
        main.cc
        book_keeper.cc
        book_keeper.h
        book_keeper_ui.cc
        book_keeper_ui.h)

target_link_libraries(BookKeeper LinkedList)