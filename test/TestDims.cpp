#include "gtest/gtest.h"
#include "Dims.hpp"
#include "boost/hana.hpp"

using namespace boost::hana::literals;

TEST(TestDims, Test1) {
    auto dims = Dims(1, 2, 3);
    static_assert(dims.nDims == 3);
    ASSERT_EQ(3, dims.nDims);
}

TEST(TestDims, Test2) {
    auto dims = Dims(1, 2, 3);

    ASSERT_EQ(dims.dim[0_c], 1);
    ASSERT_EQ(dims.dim[1_c], 2);
}

TEST(TestDims, Test3) {
    // Add constexpr
    auto constexpr dims = Dims(1, 2, 3);

    static_assert(dims.dim[0_c] == 1);
    static_assert(dims.dim[1_c] == 2);

    ASSERT_EQ(dims.dim[0_c], 1);
    ASSERT_EQ(dims.dim[1_c], 2);
}

TEST(TestDims, Test4) {
    // No constexpr
    auto dims = Dims(1_c, 2_c, 3_c);

    static_assert(dims.dim[0_c] == 1_c);
    static_assert(dims.dim[1_c] == 2_c);

    ASSERT_EQ(dims.dim[0_c], 1);
    ASSERT_EQ(dims.dim[1_c], 2);
}
