#pragma once

#include <cstdio>

#include "Tensor.hpp"

template <typename Tensor1, typename Tensor2,
          typename = std::enable_if_t<
              is_tensor_type<Tensor1> && is_tensor_type<Tensor2>, void>>
struct TMov {
  const Tensor1 &output_;
  const Tensor2 &input_;

  constexpr TMov(Tensor1 const &output, Tensor2 const &input)
      : output_(output), input_(input) {}

  constexpr TMov(TMov const &other)
      : output_(other.output_), input_(other.input_) {}

  constexpr TMov(TMov &&other)
      : output_(std::move(other.output_)), input_(std::move(other.input_)) {}

  constexpr auto output() const { return output_; }

  constexpr auto input() const { return input_; }

  constexpr auto gen_code() const {
    // Calc params
    auto dimensions1 = output().dimensions();
    auto dimensions2 = input().dimensions();

    auto fn = [dimensions1, dimensions2]() {
      printf("tmov(tensor %lu, tensor %lu)\n", dimensions1.nDims,
             dimensions2.nDims);
    };
    return fn;
  }

  constexpr auto operator()() const {
    auto code = gen_code();
    return code();
  }
};

template <typename Dest, typename Src1, typename Src2,
          typename =
              std::enable_if_t<is_tensor_type<Dest> && is_tensor_type<Src1> &&
                                   is_tensor_type<Src2>,
                               void>>
struct TAdd {
  const Dest &dest_;
  const Src1 &src1_;
  const Src2 &src2_;

  constexpr TAdd(Dest const &dest, Src1 const &src1, Src2 const &src2)
      : dest_(dest), src1_(src1), src2_(src2) {}

  constexpr TAdd(TAdd const &other)
      : dest_(other.dest_), src1_(other.src1_), src2_(other.src2_) {}

  constexpr TAdd(TAdd &&other)
      : dest_(std::move(other.dest_)),
        src1_(std::move(other.src1_)),
        src2_(std::move(other.src2_)) {}

  constexpr auto output() const { return dest_; }

  constexpr auto gen_code() const {
    // Calc params
    auto addr_dest = output().addr();
    auto addr_src1 = src1_.addr();
    auto addr_src2 = src2_.addr();

    auto shape1_sum = src1_.format().shape().dim[0_c];
    auto shape2_sum = src1_.format().shape().dim[0_c];
    auto shape_sum = (int)(shape1_sum + shape2_sum);

    auto fn = [addr_dest, addr_src1, addr_src2, shape_sum]() {
      printf("tadd(tensor 0x%x, tensor 0x%x, tensor 0x%x) shape_sum = %d\n",
             (unsigned)addr_dest, (unsigned)addr_src1, (unsigned)addr_src2,
             shape_sum);
    };

    return fn;
  }

  constexpr auto operator()() const {
    auto code = gen_code();
    return code();
  }
};

template <typename Tensor1, typename Tensor2>
constexpr auto tmov1(Tensor1 &&tensor1, Tensor2 &&tensor2) {
  static_assert(is_tensor_type<std::remove_reference_t<Tensor1>>);
  auto dimensions1 = tensor1.dimensions();
  auto dimensions2 = tensor2.dimensions();
  printf("tmov1(tensor %lu, tensor %lu)\n", dimensions1.nDims,
         dimensions2.nDims);
}

template <typename Tensor1, typename Tensor2>
constexpr auto tadd1(Tensor1 &&tensor1, Tensor2 &&tensor2) {
  static_assert(is_tensor_type<std::remove_reference_t<Tensor1>>);
  auto dimensions1 = tensor1.dimensions();
  auto dimensions2 = tensor2.dimensions();
  printf("tadd1(tensor %lu, tensor %lu)\n", dimensions1.nDims,
         dimensions2.nDims);
}