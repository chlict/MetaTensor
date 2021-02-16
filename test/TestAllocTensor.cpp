#include "Tensor.hpp"
#include "gtest/gtest.h"
#include "xforms/AllocTensor.hpp"
#include "xforms/GenIR.hpp"

TEST(TestAllocTensor, Test1) {
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
  auto gen =
      yap::transform(expr1, GenIRXform{hana::make_tuple(), hana::make_tuple()});
  printf("After GenIR:\n");
  print_ir_list_simple(gen.mIRList);

  auto at = AllocTensor();
  auto ir2 = at.transform(gen.mIRList);
  printf("AllocTensor\n");
  print_ir_list_simple(ir2);
}
