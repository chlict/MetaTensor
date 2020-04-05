#include "gtest/gtest.h"
#include "Tensor.hpp"
#include "xforms/GenIR.hpp"
#include "xforms/AllocTensor.hpp"

TEST(TestAllocTensor, Test1) {
    auto format1 = make_format(Dim2(2_c, 4_c), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
    auto tensor3 = Tensor(float(), format1, MemSpace::GM(), 0x1000);

//    auto tensor1 = MakeTensor(1);
//    auto tensor2 = MakeTensor(2);
    auto term1 = yap::make_terminal(tensor1);
    auto term2 = yap::make_terminal(tensor2);
    auto term3 = yap::make_terminal(tensor3);
    auto expr1 = term1 + term2 * term3;
    auto gen = yap::transform(expr1, GenIR{hana::make_tuple(), hana::make_tuple()});
    printf("After transform:\n");
    print_ir_list(gen.mIRList);

    auto ir_list2 = SubstituteTemps(gen.mIRList, alloc_tensor(gen.mIRList));

    print_ir_list(ir_list2);
    //auto &&map = AllocBuffer(gen.mIRList);
    //auto irList2 = SubstituteTemps(gen.mIRList, map);
    //printf("After AllocBuffer and SubstituteTemps:\n");
    //PrintIRList(irList2);


//    auto tensor1 = FakeTensor(1);
//    auto tensor2 = FakeTensor(2);
//    auto ten1 = yap::make_terminal(tensor1);
//    auto ten2 = yap::make_terminal(tensor2);
//    auto add1 = ten1 + ten2;
//    auto gen = yap::transform(add1, GenIR(hana::make_tuple(), hana::make_tuple()));
//    auto ir_list= gen.mIRList;
//    printf("After GenIR\n");
//    print_ir_list(ir_list);
//
//    alloc_tensor(ir_list);

//    print_type_name(ir_list);
//
//    auto ir_list2 = alloc_tensor(ir_list);
//    print_type_name(ir_list2);
//
//    auto ir0 = ir_list2[0_c];
//    yap::print(std::cout, ir0);
//
//    yap::transform(ir0, AllocTensororXform(hana::make_map()));
}

TEST(TestAllocTensor, Test2) {
    auto format1 = make_format(Dim2(2_c, 4_c), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
    auto tensor3 = Tensor(float(), format1, MemSpace::GM(), 0x1000);

    auto ten1 = yap::make_terminal(tensor1);
    auto ten2 = yap::make_terminal(tensor2);
    auto add1 = ten1 + ten2 + tensor3;
    auto gen = yap::transform(add1, GenIR(hana::make_tuple(), hana::make_tuple()));
    auto ir_list= gen.mIRList;
    print_ir_list(ir_list);
}
