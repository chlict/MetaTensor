#include <boost/hana.hpp>
#include <boost/yap/expression.hpp>
#include <boost/yap/yap.hpp>

#include "TOperator.hpp"
#include "gtest/gtest.h"

TEST(TestOperator, Test1) {
  auto format1 = make_format(Dims(2, 4), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x10);

  auto range_row = TRange(0_c, 4_c, 2_c);
  auto range_col = TRange(0_c, 8_c, 2_c);
  auto tiling = RowMajorTilingService(range_row, range_col);

  auto src1 = TOperand(tensor1, tiling);
  std::cout << src1 << std::endl;
}

TEST(TestOperator, Test2) {
  auto format1 = make_format(Dims(2, 8), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x10);
  auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x20);

  auto range_row = TRange(0_c, 4_c, 2_c);
  auto range_col = TRange(0_c, 8_c, 2_c);
  auto tiling = RowMajorTilingService(range_row, range_col);

  auto src1 = TOperand(tensor1, tiling);
  auto dest = TOperand(tensor2, tiling);

  using namespace boost::yap::literals;
  auto expr = 1_p + 1_p;
  auto calc = TOperator(expr, dest, src1);
  auto code = calc.gen_code();
  code();
}

TEST(TestOperator, Test3) {
  auto format1 = make_format(Dims(4, 6), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x10);
  auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x20);

  auto range_row = TRange(0_c, 4_c, 2_c);
  auto range_col = TRange(0_c, 6_c, 2_c);
  auto tiling = RowMajorTilingService(range_row, range_col);

  auto src1 = TOperand(tensor1, tiling);
  auto src2 = TOperand(tensor2, tiling);
  auto dest = TOperand(tensor2, tiling);

  auto mul_add = [](auto &&...args) {
    using namespace boost::yap::literals;
    auto op = TOperator(1_p = 2_p + 3_p, args...);
    op.gen_code()();  // with execution
  };

  mul_add(dest, src1, src2);
}