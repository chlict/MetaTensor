#include "gtest/gtest.h"
#include "Tensor.hpp"
#include "xforms/GenIR.hpp"
#include "xforms/AllocTensor.hpp"

TEST(TestAllocTensor, Test1) {
    auto format1 = make_format(Dim2(2_c, 4_c), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x10);
    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x20);
    auto tensor3 = Tensor(float(), format1, MemSpace::GM(), 0x30);
    auto tensor4 = Tensor(float(), format1, MemSpace::GM(), 0x40);

    auto term1 = yap::make_terminal(tensor1);
    auto term2 = yap::make_terminal(tensor2);
    auto term3 = yap::make_terminal(tensor3);
    auto term4 = yap::make_terminal(tensor4);
    auto expr1 = term1 + term2 * term3 + term4;
    auto gen = yap::transform(expr1, GenIRXform{hana::make_tuple(), hana::make_tuple()});
    printf("After GenIR:\n");
    print_ir_list_simple(gen.mIRList);

    auto at = AllocTensor();
    auto ir2 = at.transform(gen.mIRList);
    printf("AllocTensor\n");
    print_ir_list_simple(ir2);
}
