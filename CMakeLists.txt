cmake_minimum_required(VERSION 3.10)
project(MetaTensor)

set(CMAKE_CXX_STANDARD 17)

add_executable(MetaTensor main.cpp)

include_directories(src/includes)
include_directories(third-party/boost_1_72_0)
include_directories(third-party/range-v3-0.10.0/include)

add_subdirectory(test)
