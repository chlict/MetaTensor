#pragma once

#include <boost/hana.hpp>
#include "Tensor.hpp"
#include "TTiling.hpp"
#include "TilingService.hpp"
#include "TOperand.hpp"
#include "ECompiler.hpp"


struct toperator_tag;

template <typename T>
constexpr bool is_toperator_type = is_a<toperator_tag, T>;

template<typename Inputs, typename Output, typename Expr>
struct TOperator {
    static_assert(is_hana_tuple_type<Inputs> && is_toperand_type<Output> && is_yap_expr_type<Expr>);

    using tag = toperator_tag;

    const Inputs inputs_;
    const Output output_;
    const Expr   expr_;

    constexpr TOperator(Inputs const &inputs, Output const &output, Expr const &expr) :
        inputs_(inputs), output_(output), expr_(expr) {}

    constexpr TOperator(TOperator const &other) :
            inputs_(other.inputs_), output_(other.output_), expr_(other.expr_) {}

    constexpr TOperator(TOperator &&other) :
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
//            auto tensor1 = input1.tensor();
//            auto tiling1 = input1.tiling_provider();
//            auto indices1 = tiling1.gen_tiling_indices(tensor1);
//
//            auto tensor2 = output.tensor();
//            auto tiling2 = output.tiling_provider();
//            auto indices2 = tiling2.gen_tiling_indices(tensor2);
//            auto compiler = ECompiler(expr);
//            auto index = views::zip(indices1, indices2);
//            for (auto i : index) {
//                auto pos1 = tiling1.index_to_pos(std::get<0>(i));
//                auto pos2 = tiling2.index_to_pos(std::get<1>(i));
//                auto tile1 = tensor1.get_tile(pos1, shape1);
//                auto tile2 = tensor2.get_tile(pos2, shape2);
//                auto core = compiler.compile(tile1, tile2);
//                core();
//            }
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