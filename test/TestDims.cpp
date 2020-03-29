#include "gtest/gtest.h"
#include "Dims.hpp"
#include "boost/hana.hpp"
#include "Literals.hpp"
#include "Utils.hpp"

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
    print_type_name(1_c);
    print_type_name(dims);

    static_assert(dims.dim[0_c] == 1_c);
    static_assert(dims.dim[1_c] == 2_c);

    ASSERT_EQ(dims.dim[0_c], 1);
    ASSERT_EQ(dims.dim[1_c], 2);
}

TEST(TestDims, Test5) {
    auto constexpr dims1 = Dims(1, 2, 3);
    auto constexpr dims2 = Dims(dims1);

    static_assert(dims1.dim[0_c] == dims2.dim[0_c]);
    static_assert(dims1.dim[1_c] == dims2.dim[1_c]);
}

auto copy1 = [](auto &&dims) {
    return dims;
};

auto copy2 = [](auto &&dims) {
    return Dims(dims);
};

auto add1 = [](auto &&d1, auto &&d2) {
    auto e0 = d1.dim[0_c] + d2.dim[0_c];
    auto e1 = d1.dim[1_c] + d2.dim[1_c];
    return Dims(e0, e1);
};

TEST(TestDims, Test6) {
    auto d1 = Dims(10_c, 20_c);
    auto d2 = copy1(d1);
    static_assert(d1.dim == d2.dim);

    auto d3 = copy2(d1);
    static_assert(d1.dim == d3.dim);

    auto d4 = add1(d1, d2);
    static_assert(d4.dim[0_c] == 20_c);
    static_assert(d4.dim[1_c] == 40_c);

    print_type_name(d4);
}
