#include "gtest/gtest.h"
#include "Tensor.hpp"
#include "xforms/GenIR.hpp"
#include "xforms/AllocTensor.hpp"
#include "xforms/CodeGen.hpp"

template <typename F>
constexpr auto launch2(F const &codes) {
    hana::for_each(codes, [](auto &&f) {
        f();
    });
}

int a = 2;
int b = 4;
TEST(TestCodeGen, Test1) {
    auto format1 = make_format(Dim2(a, b), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x2000);
    auto tensor3 = Tensor(float(), format1, MemSpace::GM(), 0x3000);

    auto term1 = yap::make_terminal(tensor1);
    auto term2 = yap::make_terminal(tensor2);
    auto term3 = yap::make_terminal(tensor3);
    auto expr1 = (term1 + term2) * term3;
    auto gen = yap::transform(expr1, GenIR{hana::make_tuple(), hana::make_tuple()});
//    printf("After GenIR:\n");
//    print_ir_list_simple(gen.mIRList);

    auto at = AllocTensor();
    auto ir2 = at.transform(gen.mIRList);
//    printf("AllocTensor\n");
//    print_ir_list_simple(ir2);

    auto codes = hana::transform(ir2, [](auto &&ir) {
        return yap::transform(ir, CodeGenXform());
    });
    launch2(codes);
}
