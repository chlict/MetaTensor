#include <array>

#include "Tensor.hpp"
#include "gtest/gtest.h"

TEST(TestTensor, Test1) {
  auto format1 = make_format(Dims(2, 4), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);

  using T = decltype(tensor1);
  static_assert(std::is_same_v<tensor_traits<T>::elem_type, float>);
  static_assert(std::is_same_v<tensor_traits<T>::space, MemSpace::GM>);
}

auto fn_copy = [](auto &&tensor) { return tensor; };

TEST(TestTensor, Test2) {
  auto format1 = make_format(Dims(2_c, 4_c), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
  auto tensor2 = fn_copy(tensor1);

  using T = decltype(tensor2);
  static_assert(std::is_same_v<tensor_traits<T>::elem_type, float>);
  static_assert(std::is_same_v<tensor_traits<T>::space, MemSpace::GM>);

  auto format2 = tensor2.format();
  auto shape = format2.shape();
  auto layout = format2.layout();
  static_assert(shape.dim[0_c] == 2_c);
  static_assert(layout.dimensions_.dim[0_c] == 4_c);

  auto shape_0 = tensor2.shape().dim[0_c];
  auto shape_1 = tensor2.shape().dim[1_c];
  static_assert(shape_0 == 2_c && shape_1 == 4_c);

  auto dimensions_0 = tensor2.dimensions().dim[0_c];
  static_assert(dimensions_0 == 4_c);
  auto stride_0 = tensor2.strides().dim[0_c];
  static_assert(stride_0 == 1_c);
}

TEST(TestTensor, Test3) {
  auto format1 = make_format(Dims(2_c, 4_c), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
  TensorHandle &th = tensor1;
  using T1 = decltype(tensor1);
  auto tensor2 = static_cast<T1 &>(th);
  print_type_name(tensor1);
  print_type_name(tensor2);
  auto shape_0 = tensor2.shape().dim[0_c];
  static_assert(shape_0 == 2_c);

  std::cout << th << std::endl;
  std::cout << tensor2 << std::endl;
}

TEST(TestTensor, Test4) {
  auto format = make_format(Dims(2_c, 4_c), RowMajorLayout());
  auto tensor = Tensor(float(), format, MemSpace::GM(), 0x1000);
  auto layout = tensor.layout();
  auto dimensions = layout.dimensions();
  auto strides = layout.strides();

  auto tile1 = tensor.get_tile(Dim2(0_c, 0_c), Dim2(2_c, 4_c));
  auto tile1_layout = tile1.layout();
  auto tile1_dimensions = tile1_layout.dimensions();
  auto tile1_strides = tile1_layout.strides();
  static_assert(dimensions.dim == tile1_dimensions.dim);
  static_assert(strides.dim == tile1_strides.dim);
  assert(tile1.addr() == tensor.addr());
}

TEST(TestTensor, Test5) {
  std::array<std::array<float, 4>, 2> data = {
      {0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3}};

  auto format = make_format(Dims(2_c, 4_c), RowMajorLayout());
  float *addr = &data[0][0];
  auto tensor = Tensor(float(), format, MemSpace::GM(),
                       reinterpret_cast<uintptr_t>(addr));
  std::cout << "aaa = " << tensor.elem(Dims(0_c, 0_c)) << std::endl;
  tensor.dump();
  // for (std::size_t i = 0; i < 2; i++) {
  //   for (std::size_t j = 0; j < 4; j++) {
  //     std::cout << "&data[" << i << "][" << j << "] =" << &data[i][j]
  //     << "value = " << data[i][j] << std::endl;
  //   }
  // }
  // std::cout << "sizeof(data[0]) = " << sizeof(data[0]) << std::endl;
  // std::cout << "sizeof(data) = " << sizeof(data) << std::endl;
  // std::cout << "data.data = " << data.data() << std::endl;
}
