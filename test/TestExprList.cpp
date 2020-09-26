#include "gtest/gtest.h"
#include "ExprBlock.hpp"
using namespace boost::yap::literals;


TEST(TestExprList, Test1) {
    auto list1 = ExprBlock {
        1_p + 2_p,
        2_p * 3_p
    };
    std::cout << list1 << std::endl;

    list1.gen_code(1, 2, 3);
}

TEST(TestExprBlock, Test2) {
    auto format1 = make_format(Dim2(2, 4), RowMajorLayout());
    auto tensor1 = TensorE(float(), format1, MemSpace::GM(), 0x10);
    auto tensor2 = TensorE(float(), format1, MemSpace::GM(), 0x20);
    auto tensor3 = TensorE(float(), format1, MemSpace::GM(), 0x30);
    auto temp_1 = TensorE(float(), format1, MemSpace::GM(), 0x40);

    auto add_mul = ExprBlock {
        temp_1 = tensor1 + tensor2,
        tensor3 = temp_1 * tensor3
    };

    add_mul.gen_code()();
}

//TEST(TestExprBlock, Test3) {
//    auto format1 = make_format(Dim2(2, 4), RowMajorLayout());
//    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x10);
//    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x20);
//    auto tensor3 = Tensor(float(), format1, MemSpace::GM(), 0x30);
//    auto temp_1 = Tensor(float(), format1, MemSpace::GM(), 0x40);
//
//    auto add_mul = ExprBlock {
//            _1 = 1_p + 2_p,
//            // sync(),
//            _1 * 3_p
//    };
//
//    add_mul.gen_code(tensor1, tensor2, tensor3)();
//}