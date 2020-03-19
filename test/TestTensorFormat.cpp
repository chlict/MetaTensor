#include "gtest/gtest.h"
#include "TensorFormat.hpp"

TEST(TestTensorFormat, Test1) {
    auto view = Dims(4_c, 2_c);
    auto layout = TensorLayout(Dims(4_c, 2_c), Dims(2_c, 1_c));
    auto format = MakeFormat<Customized>(view, layout);
    static_assert(format.view_.dim == view.dim);
    static_assert(format.layout_.shape_.dim[0_c] == 4_c);

    auto view2 = format.get_view();
    auto layout_shape = format.get_layout().shape();
    static_assert(view2.dim == view.dim);
    static_assert(layout_shape.dim[0_c] == 4_c);
}

TEST(TestTensorFormat, Test2) {
    auto view = Dims(4, 2);
    auto layout = TensorLayout(Dims(4, 2), Dims(2_c, 1_c));
    auto format = MakeFormat<Customized>(view, layout);
    assert(format.view_.dim == view.dim);
    assert(format.layout_.shape_.dim[0_c] == 4);
}

TEST(TestTensorFormat, TestMatRowMajor) {
    auto format = MakeFormat<Format::MatRowMajor>(Dims(4_c, 2_c));
    auto shape = format.get_layout().shape();
    auto stride = format.get_layout().stride();

    static_assert(stride.dim[0_c] == shape.dim[0_c] * shape.dim[1_c] && stride.dim[1_c] == 1_c);
}