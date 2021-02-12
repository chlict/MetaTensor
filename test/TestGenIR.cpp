#include "Tensor.hpp"
#include "gtest/gtest.h"
#include "xforms/GenIR.hpp"

auto format1 = make_format(Dim2(2_c, 4_c), RowMajorLayout());
auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
auto tensor3 = Tensor(float(), format1, MemSpace::GM(), 0x1000);

TEST(TestGenIR, Test1) {
  auto ten1 = yap::make_terminal(tensor1);
  auto ten2 = yap::make_terminal(tensor2);
  auto add1 = ten1 + ten2;
  yap::print(std::cout, add1);
}

TEST(TestGenIR, Test2) {
  auto ten1 = yap::make_terminal(tensor1);
  auto ten2 = yap::make_terminal(tensor2);
  auto add1 = ten1 + ten2 + tensor3;
  auto gen =
      yap::transform(add1, GenIRXform(hana::make_tuple(), hana::make_tuple()));
  auto ir_list = gen.mIRList;
  print_ir_list(ir_list);
}
