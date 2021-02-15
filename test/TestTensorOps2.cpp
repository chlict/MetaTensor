#include <array>

#include "TensorOps.hpp"
// #include "backend/x86/x86_unary.hpp"
#include "gtest/gtest.h"

auto runtime_gen1(float *data1, float *data2) {
  auto format1 = make_format(Dims(2, 4), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), data1);
  auto tensor2 = Tensor(float(), format1, MemSpace::GM(), data2);
  auto tmov = TMov(tensor1, tensor2).gen_code();
  return tmov;
}

template <typename F>
constexpr auto launch(F const &f) {
  f();
}

// int main() {
//    auto gen1 = runtime_gen1();
//    launch(gen1);
//}

TEST(TestTensorOps, Runtime1) {
  std::array<std::array<float, 4>, 2> data1 = {
      0.0,
  };
  std::array<std::array<float, 4>, 2> data2 = {0.0, 0.1, 0.2, 0.3,
                                               1.0, 1.1, 1.2, 1.3};
  auto gen1 = runtime_gen1(&data1[0][0], &data2[0][0]);
  launch(gen1);
  assert(data1[1][3] == data2[1][3]);
}