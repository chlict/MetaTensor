#include "ExprBlock.hpp"
#include "gtest/gtest.h"
using namespace boost::yap::literals;

using namespace mt;

TEST(TestExprList, Test1) {
  auto list1 = ExprBlock{1_p + 2_p, 2_p * 3_p};
  std::cout << list1 << std::endl;

  // Actually only assign expressions can be code generated for.
  list1.gen_code(1, 2, 3);
}

TEST(TestExprBlock, Test2) {
  auto format1 = make_format(Dim2(2, 4), RowMajorLayout());
  auto tensor1 = TensorExpr((float *)0x10, format1);
  auto tensor2 = TensorExpr((float *)0x20, format1);
  auto tensor3 = TensorExpr((float *)0x30, format1);
  auto temp_1  = TensorExpr((float *)0x40, format1);

  auto add_mul =
      ExprBlock{temp_1 = tensor1 + tensor2, tensor3 = temp_1 * tensor3};

  auto kernel = add_mul.gen_code();
  launch(kernel);
}

TEST(TestExprBlock, Test3) {
  auto format1 = make_format(Dim2(2, 4), RowMajorLayout());
  auto tensor1 = Tensor((float *)0x10, format1);
  auto tensor2 = Tensor((float *)0x20, format1);
  auto tensor3 = Tensor((float *)0x30, format1);

  auto add_mul = ExprBlock{_t1 = 1_p + 2_p, _t2 = _t1 * 3_p};

  auto kernel = add_mul.gen_code(tensor1, tensor2, tensor3);
  launch(kernel);
}

TEST(TestExprBlock, Test4) {
  auto format1 = make_format(Dim2(2, 4), RowMajorLayout());
  auto tensor1 = Tensor((float *)0x10, format1);
  auto tensor2 = Tensor((float *)0x20, format1);
  auto tensor3 = Tensor((float *)0x30, format1);

  auto add_mul = ExprBlock{_t1 = 1_p + 2_p, _t2 = _t1 * 3_p};

  add_mul(tensor1, tensor2, tensor3);
}
