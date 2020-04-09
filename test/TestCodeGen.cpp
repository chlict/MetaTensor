#include "gtest/gtest.h"
#include "Tensor.hpp"
#include "xforms/GenIR.hpp"
#include "xforms/AllocTensor.hpp"
#include "xforms/CodeGen.hpp"
#include "ECompiler.hpp"

struct TestCodeGen : public ::testing::Test {
private:
    // Some runtime values
    int height_;
    int width_;

public:
    void SetUp() override {
        height_ = 2;
        width_ = 4;
    }

    void TearDown() override {}

    int GetHeight() {
        return height_;
    }

    int GetWidth() {
        return width_;
    }

    // Simulate launching a kernel
    template<typename Codes,
            typename = std::enable_if_t<hana::is_a<hana::tuple_tag, Codes>, void>
    >
    constexpr auto launch(Codes &&codes) const {
        hana::for_each(codes, [](auto &&f) {
            f();
        });
    }
};

TEST_F(TestCodeGen, DynamicShape1) {
    auto height = GetHeight();
    auto width = GetWidth();
    auto format1 = make_format(Dim2(height, width), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x2000);
    auto tensor3 = Tensor(float(), format1, MemSpace::GM(), 0x3000);

    auto term1 = yap::make_terminal(tensor1);
    auto term2 = yap::make_terminal(tensor2);
    auto term3 = yap::make_terminal(tensor3);
    auto expr1 = (term1 + term2) * term3;
    auto gen = yap::transform(expr1, GenIRXform{hana::make_tuple(), hana::make_tuple()});
//    printf("After GenIR:\n");
//    print_ir_list_simple(gen.mIRList);

    auto at = AllocTensor();
    auto ir2 = at.transform(gen.mIRList);
//    printf("AllocTensor\n");
//    print_ir_list_simple(ir2);

    auto codes = hana::transform(ir2, [](auto &&ir) {
        return yap::transform(ir, CodeGenXform());
    });
    launch(codes);
}

TEST_F(TestCodeGen, StaticShape1) {
    auto height = 2_c;
    auto width = 4_c;
    auto format1 = make_format(Dim2(height, width), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x2000);
    auto tensor3 = Tensor(float(), format1, MemSpace::GM(), 0x3000);

    auto term1 = yap::make_terminal(tensor1);
    auto term2 = yap::make_terminal(tensor2);
    auto term3 = yap::make_terminal(tensor3);
    auto expr1 = (term1 + term2) * term3;

    auto codes = ECompiler::compile(expr1);
    launch(codes);
}
