#include <boost/hana/mult.hpp>

#include "MatrixFormat.hpp"
#include "TensorFormat.hpp"
#include "gtest/gtest.h"

struct CustomLayout : AbstractLayoutProvider<CustomLayout> {
  template <typename Shape>
  constexpr auto operator()(Shape&& shape) const {
    return TensorLayout(Dims(4_c, 2_c), Dims(2_c, 1_c));
  }
};

TEST(TestTensorFormat, Test1) {
  auto shape = Dims(4_c, 2_c);
  auto format = make_format(shape, CustomLayout());
  static_assert(format.shape_.dim == shape.dim);
  static_assert(format.layout_.dimensions_.dim[0_c] == 4_c);

  auto shape2 = format.shape();
  auto dimensions = format.layout().dimensions();
  static_assert(shape2.dim == shape.dim);
  static_assert(dimensions.dim[0_c] == 4_c);
}

struct CustomLayout2 : AbstractLayoutProvider<CustomLayout> {
  template <typename Shape>
  constexpr auto operator()(Shape&& shape) const {
    return TensorLayout(Dims(4, 2), Dims(2_c, 1_c));
  }
};

TEST(TestTensorFormat, Test2) {
  auto shape = Dims(4, 2);
  auto format = make_format(shape, CustomLayout2());
  assert(format.shape_.dim == shape.dim);
  assert(format.layout_.dimensions_.dim[0_c] == 4);
}

TEST(TestTensorFormat, TestRowMajor) {
  auto format = make_format(Dims(2_c, 4_c), RowMajorLayout());
  auto shape = format.layout().dimensions();
  auto stride = format.layout().strides();

  static_assert(stride.dim[0_c] == 1_c && stride.dim[1_c] == 4_c);

  auto layout = format.layout();
  auto off1 =
      AbstractLayoutProvider<RowMajorLayout>::offset(Dim2(1_c, 2_c), layout);
  auto off2 = RowMajorLayout::offset(Dim2(1_c, 2_c), layout);
  static_assert(off1 == off2);
  static_assert(off1 == 6_c);
}

TEST(TestTensorFormat, TestColMajor) {
  auto format = make_format(Dims(2_c, 4_c), ColMajorLayout());
  auto shape = format.layout().dimensions();
  auto stride = format.layout().strides();

  static_assert(shape.dim[0_c] == 2_c && shape.dim[1_c] == 4_c);
  static_assert(stride.dim[0_c] == 1_c && stride.dim[1_c] == 2_c);

  auto layout = format.layout();
  auto off1 =
      AbstractLayoutProvider<ColMajorLayout>::offset(Dim2(1_c, 2_c), layout);
  auto off2 = ColMajorLayout::offset(Dim2(1_c, 2_c), layout);
  static_assert(off1 == off2);
  static_assert(off1 == 5_c);
}

TEST(TestTensorFormat, TestColMajor2) {
  auto format = make_format(2_c, 4_c, ColMajorLayout());
  auto shape = format.layout().dimensions();
  auto stride = format.layout().strides();

  static_assert(shape.dim[0_c] == 2_c && shape.dim[1_c] == 4_c);
  static_assert(stride.dim[0_c] == 1_c && stride.dim[1_c] == 2_c);

  auto format2 = make_format(2, 4, ColMajorLayout());
  assert(format2.layout().dimensions().dim[0_c] == 2);
}

TEST(TestTensorFormat, TestOffset1) {
  auto rows = 2_c;
  auto cols = 4_c;

  auto format1 = make_format(rows, cols, RowMajorLayout());
  auto layout1 = format1.layout();
  auto range1 = hana::make_range(0_c, rows);
  auto range2 = hana::make_range(0_c, cols);
  auto cp = hana::cartesian_product(hana::make_tuple(range1, range2));

  hana::for_each(cp, [layout1](auto pair) {
    // std::cout << pair[0_c] << ", " << pair[1_c] << std::endl;
    auto row = pair[0_c];
    auto col = pair[1_c];
    auto pos = Dim2(row, col);
    using LayoutProvider =
        typename format_traits<decltype(format1)>::layout_provider_type;
    auto offset = LayoutProvider::offset(pos, layout1);
    static_assert(offset == row * 4_c + col);
    // std::cout << "offset: " << offset << std::endl;
  });
}

TEST(TestTensorFormat, TestOffset2) {
  auto rows = 2_c;
  auto cols = 4_c;

  auto format1 = make_format(rows, cols, ColMajorLayout());
  auto layout1 = format1.layout();
  auto range1 = hana::make_range(0_c, rows);
  auto range2 = hana::make_range(0_c, cols);
  auto cp = hana::cartesian_product(hana::make_tuple(range1, range2));

  hana::for_each(cp, [layout1](auto pair) {
    // std::cout << pair[0_c] << ", " << pair[1_c] << std::endl;
    auto row = pair[0_c];
    auto col = pair[1_c];
    auto pos = Dim2(row, col);
    using LayoutProvider =
        typename format_traits<decltype(format1)>::layout_provider_type;
    auto offset = LayoutProvider::offset(pos, layout1);
    static_assert(offset == col * 2_c + row);
    // std::cout << "offset: " << offset << std::endl;
  });
}