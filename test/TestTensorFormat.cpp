#include "gtest/gtest.h"
#include "TensorFormat.hpp"
#include "MatrixFormats.hpp"
#include <boost/hana/mult.hpp>

TEST(TestTensorFormat, Test1) {
    auto view = Dims(4_c, 2_c);
    auto layout_function = [](auto &&view) {
        return TensorLayout(Dims(4_c, 2_c), Dims(2_c, 1_c));
    };
    auto format = make_format(view, layout_function);
    static_assert(format.view.dim == view.dim);
    static_assert(format.layout.shape.dim[0_c] == 4_c);

    auto view2 = format.getView();
    auto layout_shape = format.getLayout().getShape();
    static_assert(view2.dim == view.dim);
    static_assert(layout_shape.dim[0_c] == 4_c);
}

TEST(TestTensorFormat, Test2) {
    auto view = Dims(4, 2);
    auto layout_function = [](auto &&view) {
        return TensorLayout(Dims(4, 2), Dims(2_c, 1_c));
    };
    auto format = make_format(view, layout_function);
    assert(format.view.dim == view.dim);
    assert(format.layout.shape.dim[0_c] == 4);
}

TEST(TestTensorFormat, TestRowMajor) {
    auto format = make_format(Dims(2_c, 4_c), RowMajorLayout());
    auto shape = format.getLayout().getShape();
    auto stride = format.getLayout().getStrides();

    static_assert(stride.dim[0_c] == 1_c && stride.dim[1_c] == 4_c);
}

TEST(TestTensorFormat, TestColMajor) {
    auto format = make_format(Dims(2_c, 4_c), ColMajorLayout());
    auto shape = format.getLayout().getShape();
    auto stride = format.getLayout().getStrides();

    static_assert(shape.dim[0_c] == 2_c && shape.dim[1_c] == 4_c);
    static_assert(stride.dim[0_c] == 1_c && stride.dim[1_c] == 2_c);
}


TEST(TestTensorFormat, test4) {
    auto x = boost::hana::mult(4, 2);
    auto y = boost::hana::lift<boost::hana::tuple_tag>('x');
}