#pragma once

#include <boost/hana.hpp>

#include "ECompiler.hpp"
#include "TOperand.hpp"
#include "TTiling.hpp"
#include "Tensor.hpp"
#include "TilingService.hpp"

namespace mt {

struct toperator_tag;

template <typename T>
constexpr bool is_toperator_type = is_a<toperator_tag, T>;

template <typename Expr, typename Output, typename... Inputs>
struct TOperator {
  static_assert(is_yap_expr_type<Expr> && is_toperand_type<Output> &&
                all_are<toperand_tag, Inputs...>);

  using tag = toperator_tag;

  const boost::hana::tuple<Inputs...> inputs_;
  const Output output_;
  const Expr expr_;

  constexpr TOperator(Expr const &expr, Output const &output,
                      Inputs const &...inputs)
      : inputs_(inputs...), output_(output), expr_(expr) {}

  constexpr TOperator(TOperator const &other)
      : inputs_(other.inputs_), output_(other.output_), expr_(other.expr_) {}

  constexpr TOperator(TOperator &&other) noexcept
      : inputs_(std::move(other.inputs_)),
        output_(std::move(other.output_)),
        expr_(std::move(other.expr_)) {}

  constexpr auto gen_copy_in() const {}

  constexpr auto gen_copy_out() const {}

  constexpr auto gen_code() const {
    if constexpr (sizeof...(Inputs) == 1) {
      return gen_code_1_I_1_O();
    } else if constexpr (sizeof...(Inputs) == 2) {
      return gen_code_2_I_1_O();
    }
  }

  constexpr auto gen_code_1_I_1_O() const {
    auto inputs = inputs_;
    static_assert(hana::length(inputs) == hana::size_c<1>);
    auto expr = expr_;
    // static_assert(yap::get_arity(expr) == 2);
    auto input1 = inputs[0_c];
    static_assert(is_a<toperand_tag>(input1));
    auto output = output_;

    auto codes = [input1, output, expr]() {
      // make sure all these occurred at compile time in case executed on device
      auto compiler = ECompiler(expr);
      auto indices1 = input1.gen_tiling_indices();
      auto indices2 = output.gen_tiling_indices();
      for (auto i1 = indices1.begin(), i2 = indices2.begin();
           i1 != indices1.end() && i2 != indices2.end(); i1++, i2++) {
        auto tile1 = input1.get_tile(*i1);
        auto tile2 = output.get_tile(*i2);
        auto core = compiler.compile(tile1, tile2);
        core();
      }
      // This is simpler, but not inlined by compiler
      //            auto index = ranges::views::zip(indices1, indices2);
      //            for (auto i : index) {
      //                auto tile1 = input1.get_tile(std::get<0>(i));
      //                auto tile2 = output.get_tile(std::get<1>(i));
      //                auto core = compiler.compile(tile1, tile2);
      //                core();
      //            }
    };
    return codes;
  }

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
      // make sure all these occurred at compile time in case executed on device
      auto compiler = ECompiler(expr);
      auto indices1 = input1.gen_tiling_indices();
      auto indices2 = input2.gen_tiling_indices();
      auto indices3 = output.gen_tiling_indices();
      for (auto i1 = indices1.begin(), i2 = indices2.begin(),
                i3 = indices3.begin();
           i1 != indices1.end() && i2 != indices2.end() && i3 != indices3.end();
           i1++, i2++, i3++) {
        auto tile1 = input1.get_tile(*i1);
        auto tile2 = input2.get_tile(*i2);
        auto tile3 = output.get_tile(*i3);
        auto core = compiler.compile(tile3, tile1, tile2);
        core();
      }
    };
    return codes;
  }
};

}  // namespace mt
