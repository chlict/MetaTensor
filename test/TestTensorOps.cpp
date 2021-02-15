#include <array>

#include "TensorOps.hpp"
// #include "backend/x86/x86_unary.hpp"
#include "gtest/gtest.h"

void Tester1() {
  auto format1 = make_format(Dims(2_c, 4_c), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
  auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x2000);
  tadd1(tensor1, tensor2);
}

TEST(TestTensorOps, Test1) { Tester1(); }

void Tester2() {
  std::array<std::array<float, 4>, 2> data1 = {
      0.0,
  };
  std::array<std::array<float, 4>, 2> data2 = {0.0, 0.1, 0.2, 0.3,
                                               1.0, 1.1, 1.2, 1.3};
  auto format1 = make_format(Dims(2_c, 4_c), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), &data1[0][0]);
  auto tensor2 = Tensor(float(), format1, MemSpace::GM(), &data2[0][0]);
  auto tmov = TMov(tensor1, tensor2);
  tmov();
  assert(tensor2.elem(Dims(1_c, 3_c)) == tensor1.elem(Dims(1_c, 3_c)));
}

TEST(TestTensorOps, Test2) { Tester2(); }

void Tester3() {
  float *data1 = new float[8];
  float *data2 = new float[8];
  auto format1 = make_format(Dims(2_c, 4_c), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), data1);
  auto tensor2 = Tensor(float(), format1, MemSpace::GM(), data2);
  auto tensor3 = Tensor(tensor1);
  auto tmov = TMov(tensor2, tensor1).gen_code();
  auto tadd = TAdd(tensor3, tensor1, tensor2).gen_code();
  tmov();
  tadd();

  assert(tensor2.elem(Dims(1_c, 3_c)) == tensor1.elem(Dims(1_c, 3_c)));
  delete[](data1);
  delete[](data2);
}

TEST(TestTensorOps, Test3) { Tester3(); }

void Tester4() {
  std::array<std::array<float, 4>, 2> data1 = {
      0.0,
  };
  std::array<std::array<float, 4>, 2> data2 = {0.0, 0.1, 0.2, 0.3,
                                               1.0, 1.1, 1.2, 1.3};
  auto format1 = make_format(Dims(2_c, 4_c), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), &data1[0][0]);
  auto tensor2 = Tensor(float(), format1, MemSpace::GM(), &data2[0][0]);
  auto tensor3 = Tensor(tensor1);
  auto tmov = TMov(tensor2, tensor1).gen_code();
  auto tadd = TAdd(tensor3, tensor1, tensor2).gen_code();
  auto kernel = boost::hana::make_tuple(tmov);
  boost::hana::for_each(kernel, [](auto &op) { op(); });
  auto kernel2 = boost::hana::append(kernel, tadd);
  boost::hana::for_each(kernel2, [](auto &op) { op(); });
}

TEST(TestTensorOps, Test4) { Tester4(); }
