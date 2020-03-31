project(test)
add_subdirectory(googletest-release-1.10.0)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# adding the run_tests target
add_executable(run_tests test.cpp TestDims.cpp TestLayout.cpp TestTensorFormat.cpp TestLiterals.cpp TestUtils.cpp)

target_link_libraries(run_tests gtest gtest_main)
