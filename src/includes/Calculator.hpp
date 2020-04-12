#pragma once

#include <boost/hana.hpp>
#include "Tensor.hpp"
#include "TTiling.hpp"
#include "ECompiler.hpp"

struct toperand_tag;

template <typename T>
constexpr bool is_toperand_type = is_a<toperand_tag, T>;

// Operator input: input tensor with tiling description
template <typename Tensor, typename Tiling>
struct TOperand {
    static_assert(is_tensor_type<Tensor> && is_a<ttiling_tag, Tiling>);

    using tag = toperand_tag;

    const Tensor tensor_;
    const Tiling tiling_;
    // const Padding padding_;

    constexpr TOperand(Tensor const &tensor, Tiling const &tiling) : tensor_(tensor), tiling_(tiling) {}

    constexpr TOperand(Tensor const &other) : tensor_(other.tensor_), tiling_(other.tiling_) {}

    constexpr TOperand(Tensor &&other) : tensor_(other.tensor_), tiling_(other.tiling_) {}

    constexpr auto tensor() const { return tensor_; }

    constexpr auto tiling() const { return tiling_; }

    constexpr auto gen_tiling_indicies() const {

    }

    constexpr auto get_tile() const {

    }

    friend std::ostream& operator << (std::ostream &os, TOperand const &opnd) {
        os << "TOpnd [" << opnd.tensor() << "\n" << opnd.tiling() << "]";
        return os;
    }
};

struct tcalculator_tag;

template <typename T>
constexpr bool is_tcalculator_type = is_a<tcalculator_tag, T>;

template<typename Inputs, typename Output, typename Expr>
struct TCalculator {
    static_assert(is_hana_tuple_type<Inputs> && is_toperand_type<Output> && is_yap_expr_type<Expr>);

    using tag = tcalculator_tag;

    const Inputs inputs_;
    const Output output_;
    const Expr   expr_;

    constexpr TCalculator(Inputs const &inputs, Output const &output, Expr const &expr) :
        inputs_(inputs), output_(output), expr_(expr) {}

    constexpr TCalculator(TCalculator const &other) :
            inputs_(other.inputs_), output_(other.output_), expr_(other.expr_) {}

    constexpr TCalculator(TCalculator &&other) :
            inputs_(other.inputs_), output_(other.output_), expr_(other.expr_) {}

    constexpr auto gen_copy_in() const {}

    constexpr auto gen_copy_out() const {}

    constexpr auto gen_code_2_I_1_O() const {
        auto inputs = inputs_;
        static_assert(hana::length(inputs) == hana::size_c<2>);
        auto expr = expr_;
//        static_assert(yap::get_arity(expr) == 2);
        auto input1 = inputs[0_c];
        auto input2 = inputs[1_c];
        static_assert(is_a<toperand_tag>(input1) && is_a<toperand_tag>(input2));
        auto output = output_;

        auto codes = [input1, input2, output, expr]() {
            auto compiler = ECompiler(expr);
//            auto input1_indicies = input1.gen_tile_indices();
//            auto input2_indicies = input2.gen_tile_indices();
//            for (auto index1 = indicies1.begin(), auto index2 = indicies2.begin();
//                index1 != indicies1.end() || index2 != indices2.end();
//                index1++, index2++) {
//                auto tile1 = input1.get_tile(index1);
//                auto tile2 = input2.get_tile(index2);
//                auto body = compiler.compile(tile1, tile2);
//                body();
//            }
            int i, j;
            for (i = 0, j = 0; i < 4 || j < 6; i += 2, j += 2) {
                printf("i = %d, j = %d\n", i, j);
            }
        };
        return codes;
    }

    constexpr auto gen_code_1_I_1_O() const {
        auto inputs = inputs_;
        static_assert(hana::length(inputs) == hana::size_c<1>);
        auto expr = expr_;
//        static_assert(yap::get_arity(expr) == 2);
        auto input1 = inputs[0_c];
        static_assert(is_a<toperand_tag>(input1));
        auto output = output_;

        auto codes = [input1, output, expr]() {
            // TODO: add constexpr to make sure all these occurred at compile time in case this is executed on device
            auto tensor1 = input1.tensor();
            auto tiling1 = input1.tiling();
            auto range0 = tiling1.ranges()[0_c];
            auto compiler = ECompiler(expr);
            for (unsigned i = (unsigned)range0.begin(); i < (unsigned)range0.end(); i += (unsigned)range0.step()) {
//                auto tile = tensor1.get_tile(i);
                auto core = compiler.compile(tensor1, tensor1);
                core();
            }
        };
        return codes;
    }

};

template <typename ...T>
constexpr auto make_inputs(T &&...opnds) {
    static_assert(all_are<toperand_tag, std::remove_reference_t<T>...>);
    return boost::hana::make_tuple(std::forward<T>(opnds)...);
}
//auto expr = tensor1 + tensor2;
//auto inputs = hana::make_tuple(tensor1);
//auto output = tensor2;
//auto tiling = hana::make_tuple()
//auto add = TCalculator(expr, inputs, output, tiling, padding);
//add.execute();