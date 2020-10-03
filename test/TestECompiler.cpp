#include "gtest/gtest.h"
#include "ECompiler.hpp"

TEST(TestExprCompiler, Test1) {
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

    auto codes = ECompiler(expr1, DumpFlag::ON{}).compile();
    launch(codes);
}

// Use TensorE instead of yap::make_terminal
TEST(TestExprCompiler, Test2) {
    auto format1 = make_format(Dim2(2_c, 4_c), RowMajorLayout());
    auto tensor1 = TensorE(float(), format1, MemSpace::GM(), 0x10);
    auto tensor2 = TensorE(float(), format1, MemSpace::GM(), 0x20);
    auto tensor3 = TensorE(float(), format1, MemSpace::GM(), 0x30);
    auto tensor4 = TensorE(float(), format1, MemSpace::GM(), 0x40);

    auto expr = tensor1 + tensor2 * tensor3 + tensor4;

    auto kernel = ECompiler(expr).compile();
    launch(kernel);
}

// Use placeholder and tensor as arguments
TEST(TestExprCompiler, Test3) {
    auto format1 = make_format(Dim2(2, 4), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x10);
    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x20);
    auto tensor3 = Tensor(float(), format1, MemSpace::GM(), 0x30);
    auto tensor4 = Tensor(float(), format1, MemSpace::GM(), 0x40);

    using namespace boost::yap::literals;

    auto expr = 1_p + 2_p * 3_p;
    auto kernel = ECompiler(expr).compile(tensor1, tensor2, tensor3);
    launch(kernel);
}

// Use lambda
TEST(TestExprCompiler, Test4) {
    auto format1 = make_format(Dim2(2, 4), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x10);
    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x20);
    auto tensor3 = Tensor(float(), format1, MemSpace::GM(), 0x30);
    auto tensor4 = Tensor(float(), format1, MemSpace::GM(), 0x40);

    auto add_mul = [](auto &&... args) {
        using namespace boost::yap::literals;
        auto expr = 1_p + 2_p * 3_p;
        auto kernel = ECompiler(expr).compile(args...);
        launch(kernel);
    };

    add_mul(tensor1, tensor2, tensor3);
}