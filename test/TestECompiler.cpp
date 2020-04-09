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

    auto codes = ECompiler::compile(expr1);

    hana::for_each(codes, [](auto &&f) {
        f();
    });
}

TEST(TestExprCompiler, TestTensorExpr) {
    auto format1 = make_format(Dim2(2_c, 4_c), RowMajorLayout());
    auto tensor1 = TensorE(float(), format1, MemSpace::GM(), 0x10);
    auto tensor2 = TensorE(float(), format1, MemSpace::GM(), 0x20);
    auto tensor3 = TensorE(float(), format1, MemSpace::GM(), 0x30);
    auto tensor4 = TensorE(float(), format1, MemSpace::GM(), 0x40);

    auto expr = tensor1 + tensor2 * tensor3 + tensor4;

    auto codes = ECompiler::compile(expr);

    hana::for_each(codes, [](auto &&f) {
        f();
    });
}

TEST(TestExprCompiler, Test2) {
    auto format1 = make_format(Dim2(2_c, 4_c), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x10);
    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x20);

    using namespace boost::yap::literals;
    auto expr1 = 1_p + 2_p;
    auto expr2 = boost::yap::replace_placeholders(expr1, std::move(tensor1), std::move(tensor2));
    // boost::yap::print(std::cout, expr2);

    auto codes = ECompiler::compile(expr2);
    hana::for_each(codes, [](auto &&f) {
        f();
    });

}