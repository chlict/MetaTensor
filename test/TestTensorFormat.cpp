#include "gtest/gtest.h"
#include "TensorFormat.hpp"

using namespace boost::hana::literals;

TEST(TestTensorFormat, Test1) {
    auto l_shape = Dims(4_c, 2_c);
    auto layout = Layout(Dims(4_c, 2_c), Dims(2_c, 1_c));
    auto format = make_format<MatrixRowMajor>(l_shape, layout);
    static_assert(format.lShape.dim == l_shape.dim);
    static_assert(format.layout.pShape.dim[0_c] == 4_c);
}

TEST(TestTensorFormat, Test2) {
    auto l_shape = Dims(4, 2);
    auto layout = Layout(Dims(4, 2), Dims(2_c, 1_c));
    auto format = make_format<MatrixRowMajor>(l_shape, layout);
    assert(format.lShape.dim == l_shape.dim);
    assert(format.layout.pShape.dim[0_c] == 4);
}