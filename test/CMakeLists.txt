project(test)
add_subdirectory(googletest-release-1.10.0)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# adding the Google_Tests_run target
add_executable(Google_Tests_run test.cpp)

target_link_libraries(Google_Tests_run gtest gtest_main)
