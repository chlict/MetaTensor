#include "gtest/gtest.h"
#include "Literals.hpp"
#include "TensorLayout.hpp"

TEST(TestLayout, Test1) {
    auto layout = TensorLayout(Dims(2_c), Dims(1_c));

    static_assert(layout.shape_.dim[0_c] == 2_c);
    // static_assert(layout.shape().dim[0_c] == 2_c); // compile error

    auto shape = layout.shape();
    static_assert(shape.dim[0_c] == 2_c);
}

TEST(TestLayout, Test2) {
    auto shape = Dims(2_c, 4_c);
    auto stride = Dims(4_c, 1_c);
    auto layout = TensorLayout(shape, stride);

    auto shape0 = layout.shape().dim[0_c];
    auto shape1 = layout.shape().dim[1_c];
    auto stride0 = layout.strides().dim[0_c];
    auto stride1 = layout.strides().dim[1_c];

    static_assert(shape0 == 2_c && shape1 == 4_c);
    static_assert(stride0 == 4_c && stride1 == 1_c);
}