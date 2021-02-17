#include "ECompiler.hpp"
#include "gtest/gtest.h"

TEST(TestExprCompiler, Test1) {
  auto format1 = make_format(Dim2(2_c, 4_c), RowMajorLayout());
  auto tensor1 = Tensor((float *)0x10, format1);
  auto tensor2 = Tensor((float *)0x20, format1);
  auto tensor3 = Tensor((float *)0x30, format1);
  auto tensor4 = Tensor((float *)0x40, format1);

  auto term1 = yap::make_terminal(tensor1);
  auto term2 = yap::make_terminal(tensor2);
  auto term3 = yap::make_terminal(tensor3);
  auto term4 = yap::make_terminal(tensor4);
  auto expr1 = term1 + term2 * term3 + term4;

  auto codes = ECompiler(expr1, DumpFlag::ON{}).compile();
  launch(codes);
}

// Use TensorExpr instead of yap::make_terminal
TEST(TestExprCompiler, Test2) {
  auto format1 = make_format(Dim2(2_c, 4_c), RowMajorLayout());
  auto tensor1 = TensorExpr((float *)0x10, format1);
  auto tensor2 = TensorExpr((float *)0x20, format1);
  auto tensor3 = TensorExpr((float *)0x30, format1);
  auto tensor4 = TensorExpr((float *)0x40, format1);

  auto expr = tensor1 + tensor2 * tensor3 + tensor4;

  auto kernel = ECompiler(expr).compile();
  launch(kernel);
}

// Use placeholder and tensor as arguments
TEST(TestExprCompiler, Test3) {
  auto format1 = make_format(Dim2(2, 4), RowMajorLayout());
  auto tensor1 = Tensor((float *)0x10, format1);
  auto tensor2 = Tensor((float *)0x20, format1);
  auto tensor3 = Tensor((float *)0x30, format1);
  auto tensor4 = Tensor((float *)0x40, format1);

  using namespace boost::yap::literals;

  auto expr = 1_p + 2_p * 3_p;
  auto kernel = ECompiler(expr).compile(tensor1, tensor2, tensor3);
  launch(kernel);
}

// Use lambda
TEST(TestExprCompiler, Test4) {
  auto format1 = make_format(Dim2(2, 4), RowMajorLayout());
  auto tensor1 = Tensor((float *)0x10, format1);
  auto tensor2 = Tensor((float *)0x20, format1);
  auto tensor3 = Tensor((float *)0x30, format1);
  auto tensor4 = Tensor((float *)0x40, format1);

  auto add_mul = [](auto &&...args) {
    using namespace boost::yap::literals;
    auto expr = 1_p + 2_p * 3_p;
    auto kernel = ECompiler(expr).compile(args...);
    launch(kernel);
  };

  add_mul(tensor1, tensor2, tensor3);
}
