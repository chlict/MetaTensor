#include "gtest/gtest.h"
#include "Literals.hpp"
#include "Layout.hpp"

TEST(TestLayout, Test1) {
    auto layout = Layout(Dims(2_c), Dims(1_c));

    static_assert(layout.pShape.dim[0_c] == 2_c);
    // static_assert(layout.getPShape().dim[0_c] == 2_c); // compile error
    auto p_shape = layout.getPShape();
    static_assert(p_shape.dim[0_c] == 2_c);
}