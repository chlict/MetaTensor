#pragma once

#include "Tensor.hpp"
#include <cstdio>

template<typename Tensor1, typename Tensor2,
        typename = std::enable_if_t<
                is_tensor<Tensor1> && is_tensor<Tensor2>,
                void>
>
struct TMov {
    const Tensor1 &output_;
    const Tensor2 &input_;

    constexpr TMov(Tensor1 const &output, Tensor2 const &input) : output_(output), input_(input) {}

    constexpr TMov(TMov const &other) : output_(other.output_), input_(other.input_) {}

    constexpr TMov(TMov &&other) : output_(other.output_), input_(other.input_) {}

    constexpr auto output() const { return output_; }

    constexpr auto input() const { return input_; }

    constexpr auto gen_code() const {
        // Calc params
        auto shape1 = output().shape();
        auto shape2 = input().shape();

        auto fn = [shape1, shape2]() {
            printf("tmov(tensor %lu, tensor %lu)\n", shape1.nDims, shape2.nDims);
        };
        return fn;
    }

    constexpr auto operator()() const {
        auto code = gen_code();
        return code();
    }
};

template<typename Dest, typename Src1, typename Src2,
        typename = std::enable_if_t<
                is_tensor<Dest> && is_tensor<Src1> && is_tensor<Src2>,
                void>
>
struct TAdd {
    const Dest &dest_;
    const Src1 &src1_;
    const Src2 &src2_;

    constexpr TAdd(Dest const &dest, Src1 const &src1, Src2 const &src2) : dest_(dest), src1_(src1), src2_(src2) {}

    constexpr TAdd(TAdd const &other) : dest_(other.dest_), src1_(other.src1_), src2_(other.src2_) {}

    constexpr TAdd(TAdd &&other) : dest_(other.dest_), src1_(other.src1_), src2_(other.src2_) {}

    constexpr auto output() const { return dest_; }

    constexpr auto gen_code() const {
        // Calc params
        auto addr_dest = output().addr();
        auto addr_src1 = src1_.addr();
        auto addr_src2 = src2_.addr();

        auto fn = [addr_dest, addr_src1, addr_src2]() {
            printf("tadd(tensor 0x%x, tensor 0x%x, tensor 0x%x)\n",
                    (unsigned)addr_dest, (unsigned)addr_src1, (unsigned)addr_src2);
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
    static_assert(is_tensor<std::remove_reference_t<Tensor1>>);
    auto shape1 = tensor1.shape();
    auto shape2 = tensor2.shape();
    printf("tmov(tensor %lu, tensor %lu)\n", shape1.nDims, shape2.nDims);
}

template <typename Tensor1, typename Tensor2>
constexpr auto tadd1(Tensor1 &&tensor1, Tensor2 &&tensor2) {
    static_assert(is_tensor<std::remove_reference_t<Tensor1>>);
    auto shape1 = tensor1.shape();
    auto shape2 = tensor2.shape();
    printf("tadd(tensor %lu, tensor %lu)\n", shape1.nDims, shape2.nDims);
}