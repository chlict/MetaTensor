#pragma once

#include <iostream>
#include <boost/yap/yap.hpp>
#include <boost/yap/print.hpp>
#include <boost/yap/expression.hpp>
#include <boost/hana.hpp>
#include "xforms/AllocTensor.hpp"
#include "xforms/CodeGen.hpp"

// Expression block representing a sequence of expressions, like a compound statement:
// ExprBlock {
//     expr1,
//     expr2,
//     expr3
// }
template <typename ...Exprs>
struct ExprBlock {
    // static_assert(all_are yap_expr_type)
    boost::hana::tuple<Exprs...> expr_block_;

    constexpr ExprBlock(Exprs const &... exprs) : expr_block_(exprs...) {}

    constexpr ExprBlock(ExprBlock const& other) : expr_block_(other.expr_block_) {}

    constexpr ExprBlock(ExprBlock&& other) noexcept : expr_block_(other.expr_block_) {}

    template <typename ...Args>
    constexpr auto gen_code(Args &&... args) const {
        // The variadic arguments make it nowhere to place a DumpFlag arg with a default
        // value.
        if constexpr (need_dump(DumpFlag::ON{})) {
           printf("----Initial expressions----\n");
           print_ir_list_simple(expr_block_);
        }

        // First replace all the placeholders in expr with args
        auto ir_list = hana::transform(expr_block_, [&args...](auto const& expr) {
            return boost::yap::replace_placeholders(expr, static_cast<Args &&>(args)...);
        });

       if constexpr (need_dump(DumpFlag::ON{})) {
           printf("----After replace placeholders----\n");
           print_ir_list_simple(ir_list);
       }

        // Go through a set of transforms
        auto xforms = hana::make_tuple(
            AllocTensor(),
            CodeGen()
        );

        // Call each xform's transform() method. Each xform's output serves as input of next xform.
        auto codes = hana::fold_left(xforms, ir_list /* init state */,
            /* accept an irlist and transform to a new irlist */
            [](auto &&irlist, auto &&xform) {
                return xform.transform(irlist, DumpFlag::ON{});
            }
        );

        return codes;
        // return stub_codelet;
    }

    friend std::ostream& operator<< (std::ostream &os, ExprBlock const& L) {
        namespace hana = boost::hana;
        hana::for_each(L.expr_block_, [&os](auto const &expr) {
            boost::yap::print(os, expr);
        });
        return os;
    }
};