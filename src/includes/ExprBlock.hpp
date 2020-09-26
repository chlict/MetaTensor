#pragma once

#include <iostream>
#include <boost/yap/yap.hpp>
#include <boost/yap/print.hpp>
#include <boost/yap/expression.hpp>
#include <boost/hana.hpp>
#include "xforms/AllocTensor.hpp"
#include "xforms/CodeGen.hpp"

template <typename ...Exprs>
struct ExprBlock {
    // static_assert(all_are yap_expr_type)
    boost::hana::tuple<Exprs...> expr_block_;

    constexpr ExprBlock(Exprs const &... exprs) : expr_block_(exprs...) {}

    constexpr ExprBlock(ExprBlock const& other) : expr_block_(other.expr_block_) {}

    constexpr ExprBlock(ExprBlock&& other) noexcept : expr_block_(other.expr_block_) {}

//    template <typename... BodyExprs>
//    constexpr auto operator[] (BodyExprs &&... body_exprs) const {
//        auto body_list = hana::tuple(body_exprs...);
//    }

    template <typename ...Args>
    constexpr auto gen_code(Args &&... args) const {
        namespace yap = boost::yap;

        // First replace all the placeholders in expr with args
        auto ir_list = hana::transform(expr_block_, [&args...](auto const& expr) {
            return yap::replace_placeholders(expr, static_cast<Args &&>(args)...);
        });
//        auto constexpr dumping = Dumping();
//        if constexpr (need_dump(dumping)) {
//            yap::print(std::cout, ast);
//        }

        // Go through a set of transforms
        auto xforms = hana::make_tuple(
                CodeGen()
        );

        // Call each xform's transform() method. Each xform's output serves as input of next xform.
        auto codes = hana::fold_left(xforms, ir_list, [](auto &&ir, auto &&xform) {
            return xform.transform(ir, with_dump{});
        });

        return codes;
    }

    friend std::ostream& operator<< (std::ostream &os, ExprBlock const& L) {
        namespace hana = boost::hana;
        hana::for_each(L.expr_block_, [&os](auto const &expr) {
            boost::yap::print(os, expr);
        });
        return os;
    }
};