#pragma once

#include "Utils.hpp"
#include "Tensor.hpp"
#include "xforms/GenIR.hpp"
#include "xforms/AllocTensor.hpp"
#include "xforms/CodeGen.hpp"
#include <iostream>

struct ECompiler {
    template <typename Expr, typename Dumping, typename ...Args>
    static constexpr auto compile(Expr &&expr, Dumping dumping, Args && ...args) {
        static_assert(is_yap_expr_type<std::remove_reference_t<Expr>>);
        static_assert(std::is_same_v<Dumping, with_dump> || std::is_same_v<Dumping, nodump>);

        // First replace all the placeholders in expr with args
        auto ast = yap::replace_placeholders(std::forward<Expr>(expr), std::forward<Args>(args)...);
        static_assert(is_yap_expr_type<decltype(ast)>);
        if constexpr (need_dump(dumping)) {
            yap::print(std::cout, ast);
        }

        // Go through a set of transforms
        auto xforms = hana::make_tuple(
                GenIR(),
                AllocTensor(),
                CodeGen()
        );

        // Call each xform's transform() method. Each xform's output serves as input of next xform.
        auto apply_xform = [dumping](auto &&ir, auto &&xform) {
            return xform.transform(ir, dumping);
        };
        auto codes = hana::fold_left(xforms, ast, apply_xform);

        return codes;
    }
};