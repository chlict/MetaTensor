#include "gtest/gtest.h"
#include "Layout.hpp"

using namespace boost::hana::literals;

TEST(TestLayout, Test1) {
    auto layout = Layout(Dims(2_c), Dims(1_c));

    static_assert(layout.pShape.dim[0_c] == 2_c);
//    static_assert(layout.getPShape().dim[0_c] == 2_c); // compile error
}