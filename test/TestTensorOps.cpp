#include "TensorOps.hpp"
#include "backend/x86/x86_unary.hpp"
#include "gtest/gtest.h"

void Tester1() {
  auto format1 = make_format(Dims(2_c, 4_c), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
  auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x2000);
  tadd1(tensor1, tensor2);
}

TEST(TestTensorOps, Test1) { Tester1(); }

/**
 * The expected assembly code:

.globl  __Z7Tester2v            ## -- Begin function _Z7Tester2v
.p2align  4, 0x90
__Z7Tester2v:                           ## @_Z7Tester2v
.cfi_startproc
## %bb.0:
pushq %rbp
.cfi_def_cfa_offset 16
.cfi_offset %rbp, -16
movq  %rsp, %rbp
.cfi_def_cfa_register %rbp
        leaq  L_.str.6(%rip), %rdi
        movl  $2, %esi
        movl  $2, %edx
        xorl  %eax, %eax
        popq  %rbp
        jmp _printf                 ## TAILCALL
.cfi_endproc
 */
void Tester2() {
  auto format1 = make_format(Dims(2_c, 4_c), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
  auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x2000);
  auto tmov = TMov(tensor1, tensor2);
  tmov();
}

TEST(TestTensorOps, Test2) { Tester2(); }

/**
 * Expected assembly code:
   .globl  __Z7Tester3v            ## -- Begin function _Z7Tester3v
  .p2align  4, 0x90
__Z7Tester3v:                           ## @_Z7Tester3v
  .cfi_startproc
## %bb.0:
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
  leaq  L_.str.8(%rip), %rdi
  movl  $2, %esi
  movl  $2, %edx
  xorl  %eax, %eax
  callq _printf
  leaq  L_.str.9(%rip), %rdi
  movl  $4096, %esi             ## imm = 0x1000
  movl  $4096, %edx             ## imm = 0x1000
  movl  $8192, %ecx             ## imm = 0x2000
  xorl  %eax, %eax
  popq  %rbp
  jmp _printf                 ## TAILCALL
  .cfi_endproc

 */
void Tester3() {
  auto format1 = make_format(Dims(2_c, 4_c), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
  auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x2000);
  auto tensor3 = Tensor(tensor1);
  auto tmov = TMov(tensor2, tensor1).gen_code();
  auto tadd = TAdd(tensor3, tensor1, tensor2).gen_code();
  tmov();
  tadd();
}

TEST(TestTensorOps, Test3) { Tester3(); }

void Tester4() {
  auto format1 = make_format(Dims(2_c, 4_c), RowMajorLayout());
  auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
  auto tensor2 = Tensor(float(), format1, MemSpace::GM(), 0x2000);
  auto tensor3 = Tensor(tensor1);
  auto tmov = TMov(tensor2, tensor1).gen_code();
  auto tadd = TAdd(tensor3, tensor1, tensor2).gen_code();
  auto kernel = boost::hana::make_tuple(tmov);
  boost::hana::for_each(kernel, [](auto &op) { op(); });
  auto kernel2 = boost::hana::append(kernel, tadd);
  boost::hana::for_each(kernel2, [](auto &op) { op(); });
}

TEST(TestTensorOps, Test4) { Tester4(); }
