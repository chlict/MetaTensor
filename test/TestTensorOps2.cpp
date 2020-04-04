#include "gtest/gtest.h"
#include "TensorOps.hpp"

auto runtime_gen1() {
    auto format1 = make_format(Dims(2, 4), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
    auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x2000);
    auto tmov = TMov(tensor1, tensor2).gen_code();
    return tmov;
}

template <typename F>
constexpr auto launch(F const &f) {
    f();
}

//int main() {
//    auto gen1 = runtime_gen1();
//    launch(gen1);
//}

TEST(TestTensorOps, Runtime1) {
    auto gen1 = runtime_gen1();
    launch(gen1);
}