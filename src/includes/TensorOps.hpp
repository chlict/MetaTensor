#pragma once

#include <cstdio>

#include "Tensor.hpp"

// Implement tensor operations on host
namespace arch {
struct Host;
}

template <typename Dst, typename Src, typename ArchTag = arch::Host>
struct UnaryOp {
  static_assert(is_tensor_type<std::remove_reference_t<Dst>>);
  static_assert(is_tensor_type<std::remove_reference_t<Src>>);

  const Dst &dst_;
  const Src &src_;

  constexpr UnaryOp(Dst const &dst, Src const &src) : dst_(dst), src_(src) {}

  constexpr UnaryOp(UnaryOp const &other)
      : dst_(other.dst_), src_(other.src_) {}

  constexpr UnaryOp(UnaryOp &&other)
      : dst_(std::move(other.dst_)), src_(std::move(other.src_)) {}

  constexpr auto dst() const { return dst_; }

  constexpr auto src() const { return src_; }
};

// Subclasses of UnaryOp, which only need to implement:
// (1) constructors
// (2) the gen_code() function
template <typename Dst, typename Src, typename ArchTag = arch::Host>
struct TMov : public UnaryOp<Dst, Src, ArchTag> {
  using Base = UnaryOp<Dst, Src, ArchTag>;

  constexpr TMov(Dst const &dst, Src const &src) : Base(dst, src) {}

  constexpr TMov(TMov const &other) : Base(other) {}

  constexpr TMov(TMov &&other) : Base(std::move(other)) {}

  constexpr auto gen_code() const {
    // Calc params
    auto dimensions1 = Base::dst().dimensions();
    auto dimensions2 = Base::src().dimensions();

    auto fn = [dimensions1, dimensions2]() {
      printf("Host tmov(tensor %lu, tensor %lu)\n", dimensions1.nDims,
             dimensions2.nDims);
      // dimensions1 should be same with dimentions2
      // constexpr int nDims = decltype(dimensions1)::nDims;
      // for (int di = 0; di < nDims; di++) {

      // }
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
