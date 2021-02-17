#pragma once

#include <boost/hana.hpp>
#include <boost/yap/expression.hpp>
#include <boost/yap/print.hpp>
#include <cstdio>
#include <iostream>

#include "Tensor.hpp"
#include "TensorOps.hpp"
#include "XformUtils.hpp"
#include "xforms/Placeholder.hpp"
#include "xforms/PrintIR.hpp"

namespace mt {

struct CodeGenXform {
  // Gen code for dest = src1 op src2
  template <yap::expr_kind BinaryOP, typename Dest, typename Src1,
            typename Src2>
  auto operator()(
      yap::expr_tag<yap::expr_kind::assign>, Dest const &dest_expr,
      yap::expression<BinaryOP, hana::tuple<Src1, Src2>> const &bin_expr) {
    auto dest = yap::value(dest_expr);
    auto src1 = yap::value(yap::left(bin_expr));
    auto src2 = yap::value(yap::right(bin_expr));

    static_assert(is_tensor_type<std::remove_reference_t<decltype(dest)>>);
    static_assert(is_tensor_type<std::remove_reference_t<decltype(src1)>>);
    static_assert(is_tensor_type<std::remove_reference_t<decltype(src2)>>);

    // printf("CodeGenXform: tensor = expr1 op expr2 matched\n");

    if constexpr (BinaryOP == yap::expr_kind::plus) {
      auto tadd = TAdd(dest, src1, src2);
      return tadd.gen_code();
    } else if constexpr (BinaryOP == yap::expr_kind::multiplies) {
      return []() {
        printf("Code generation for multiplication is not implemented\n");
      };
    } else {
      return []() {
        printf("Code generation for this operation is not implemented\n");
      };
    }
  }

  // Not work?
  template <yap::expr_kind AnyOP, typename... Children>
  auto operator()(yap::expression<AnyOP, hana::tuple<Children...>> expr) {
    return []() {
      printf("Code generation not implemented for this expression\n");
    };
  }
};

struct CodeGen : StaticTransform {
  using tag = xform_pass_tag;

  // Given an IRList, returns the codes generated
  template <typename IRList, typename Dumping = DumpFlag::OFF>
  constexpr auto transform(IRList &&irlist,
                           Dumping dumping = DumpFlag::OFF{}) const {
    static_assert(is_hana_tuple_type<std::remove_reference_t<IRList>>);

    // For each ir in irList, transform it using CodeGenXform().
    auto code_list = hana::transform(
        irlist, [](auto &&ir) { return yap::transform(ir, CodeGenXform()); });

    if constexpr (need_dump(dumping)) {
      // How to dump codes?
    }

    // Returns a big lambda wrapping each stmt's small lambda
    return
        [code_list]() { hana::for_each(code_list, [](auto &&fn) { fn(); }); };
  }
};

// Simulate launching a kernel to execute on device
auto launch = [](auto &&codes) { codes(); };

// Simulate exeuting in place
auto execute = [](auto &&codes) { codes(); };

// For debug
auto stub_codelet = []() { printf("Stub executed\n"); };

}  // namespace mt
