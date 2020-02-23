cmake_minimum_required(VERSION 3.15)
project(demo2)

set(CMAKE_CXX_STANDARD 17)

add_executable(demo2 main.cpp src/includes/calc.hpp)

include_directories(src/includes)
include_directories(third-party/boost_1_68_0)

add_subdirectory(test)