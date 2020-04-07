#include "gtest/gtest.h"
#include "TensorFormat.hpp"
#include "MatrixFormat.hpp"
#include <boost/hana/mult.hpp>

struct CustomLayout : AbstractLayoutProvider<CustomLayout> {
    template <typename View>
    constexpr auto operator()(View&& view) const {
        return TensorLayout(Dims(4_c, 2_c), Dims(2_c, 1_c));
    }
};

TEST(TestTensorFormat, Test1) {
    auto view = Dims(4_c, 2_c);
    auto format = make_format(view, CustomLayout());
    static_assert(format.view_.dim == view.dim);
    static_assert(format.layout_.shape_.dim[0_c] == 4_c);

    auto view2 = format.view();
    auto layout_shape = format.layout().shape();
    static_assert(view2.dim == view.dim);
    static_assert(layout_shape.dim[0_c] == 4_c);
}

struct CustomLayout2 : AbstractLayoutProvider<CustomLayout> {
    template <typename View>
    constexpr auto operator()(View&& view) const {
        return TensorLayout(Dims(4, 2), Dims(2_c, 1_c));
    }
};

TEST(TestTensorFormat, Test2) {
    auto view = Dims(4, 2);
    auto format = make_format(view, CustomLayout2());
    assert(format.view_.dim == view.dim);
    assert(format.layout_.shape_.dim[0_c] == 4);
}

TEST(TestTensorFormat, TestRowMajor) {
    auto format = make_format(Dims(2_c, 4_c), RowMajorLayout());
    auto shape = format.layout().shape();
    auto stride = format.layout().strides();

    static_assert(stride.dim[0_c] == 1_c && stride.dim[1_c] == 4_c);
}

TEST(TestTensorFormat, TestColMajor) {
    auto format = make_format(Dims(2_c, 4_c), ColMajorLayout());
    auto shape = format.layout().shape();
    auto stride = format.layout().strides();

    static_assert(shape.dim[0_c] == 2_c && shape.dim[1_c] == 4_c);
    static_assert(stride.dim[0_c] == 1_c && stride.dim[1_c] == 2_c);
}

TEST(TestTensorFormat, TestColMajor2) {
    auto format = make_format(2_c, 4_c, ColMajorLayout());
    auto shape = format.layout().shape();
    auto stride = format.layout().strides();

    static_assert(shape.dim[0_c] == 2_c && shape.dim[1_c] == 4_c);
    static_assert(stride.dim[0_c] == 1_c && stride.dim[1_c] == 2_c);

    auto format2 = make_format(2, 4, ColMajorLayout());
    assert(format2.layout().shape().dim[0_c] == 2);
}

TEST(TestTensorFormat, test4) {
    auto x = boost::hana::mult(4, 2);
    auto y = boost::hana::lift<boost::hana::tuple_tag>('x');
}