#include "gtest/gtest.h"
#include <boost/hana.hpp>
#include <boost/yap/yap.hpp>
#include <boost/yap/expression.hpp>
#include "Calculator.hpp"


TEST(TestOperator, Test1) {
    auto format1 = make_format(Dims(2, 4), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x10);

    auto range_row = TRange(0_c, 4_c, 2_c);
    auto range_col = TRange(0_c, 8_c, 2_c);
    auto tiling = Tiling2DRowMajor(range_row, range_col);

    auto src1 = TOperand(tensor1, tiling);
    std::cout << src1 << std::endl;
}

TEST(TestOperator, Test2) {
    auto format1 = make_format(Dims(2, 4), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x10);
    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x20);

    auto range_row = TRange(0_c, 4_c, 2_c);
    auto range_col = TRange(0_c, 8_c, 2_c);
    auto tiling = Tiling2DRowMajor(range_row, range_col);

    auto src1 = TOperand(tensor1, tiling);
    auto dest = TOperand(tensor2, tiling);

    using namespace boost::yap::literals;
    auto expr = 1_p + 1_p;
    auto calc = TCalculator(boost::hana::make_tuple(src1), dest, expr);
    auto code = calc.gen_code_1_I_1_O();
    code();
//    tensor1.tileWith(tiling), tensor2.tileWith(tiling2) | 1_p + 2_p
}

TEST(TestOperator, Test3) {
    auto format1 = make_format(Dims(4, 6), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x10);
    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x20);

    auto range_row = TRange(0_c, 4_c, 2_c);
    auto range_col = TRange(0_c, 6_c, 2_c);
    auto tiling = Tiling2DRowMajor(range_row, range_col);

    auto src1 = TOperand(tensor1, tiling);
    auto src2 = TOperand(tensor2, tiling);
    auto dest = TOperand(tensor2, tiling);

    using namespace boost::yap::literals;
    auto expr = 1_p + 2_p;

    auto inputs = make_inputs(src1, src2);
    auto const calc = TCalculator(inputs, dest, expr);
    auto codes = calc.gen_code_2_I_1_O();
    codes();
}