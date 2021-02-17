#pragma once

#include <iostream>

#include "Tensor.hpp"
#include "Utils.hpp"
#include "xforms/AllocTensor.hpp"
#include "xforms/CodeGen.hpp"
#include "xforms/GenIR.hpp"
#include "xforms/XformUtils.hpp"

namespace mt {

struct ecompiler_tag;

template <typename T>
constexpr bool is_ecompiler_type = is_a<ecompiler_tag, T>;

template <typename Expr, typename Dumping = DumpFlag::OFF>
struct ECompiler {
  static_assert(is_yap_expr_type<Expr>);
  static_assert(std::is_same_v<Dumping, DumpFlag::ON> ||
                std::is_same_v<Dumping, DumpFlag::OFF>);

  const Expr expr_;

  constexpr ECompiler(Expr const &expr, Dumping = DumpFlag::OFF{})
      : expr_(expr) {}

  constexpr ECompiler(ECompiler const &other) : expr_(other.expr_) {}

  constexpr ECompiler(ECompiler &&other) : expr_(other.expr_) {}

  template <typename... Args>
  constexpr auto compile(Args &&...args) const {
    // First replace all the placeholders in expr with args
    auto ast = yap::replace_placeholders(expr_, std::forward<Args>(args)...);
    static_assert(is_yap_expr_type<decltype(ast)>);
    if constexpr (need_dump(Dumping{})) {
      yap::print(std::cout, ast);
    }

    // Go through a set of transforms
    auto xforms = hana::make_tuple(GenIR(), AllocTensor(), CodeGen());

    // Call each xform's transform() method. Each xform's output serves as input
    // of next xform.
    auto apply_xform = [](auto &&ir, auto &&xform) {
      return xform.transform(ir, Dumping{});
    };
    auto codes = hana::fold_left(xforms, ast, apply_xform);

    return codes;
  }
};

}  // namespace mt
