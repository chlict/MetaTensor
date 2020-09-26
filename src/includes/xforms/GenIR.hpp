#pragma once

#include <boost/yap/expression.hpp>
#include <boost/hana/maximum.hpp>
#include <boost/hana.hpp>
#include <boost/yap/print.hpp>
#include <iostream>
#include "Placeholder.hpp"
#include "xforms/XformUtils.hpp"

template <typename Sequence, typename Stack, long long I = 1>
struct GenIRXform {
    Sequence mIRList;
    Stack mStack;
    static const long long placeholder_index = I;

    constexpr GenIRXform(const Sequence &seq, const Stack &stk) : mIRList(seq), mStack(stk) {
        // printf("GenIR constructor:\n");
        // printf("mIRList: "); hana::for_each(mIRList, [](const auto &ir) {yap::print(std::cout, ir);});
        // printf("mStack: "); hana::for_each(mStack, [](const auto &expr) {yap::print(std::cout, expr);});
    }

    template <typename T>
    auto operator() (yap::expr_tag<yap::expr_kind::terminal>, T &&t) {
        // printf("GenIR: terminal matched\n");
        // Push result onto stack
        auto stack = hana::append(mStack, std::move(yap::make_terminal(t)));
        return GenIRXform<decltype(mIRList), decltype(stack), I>{mIRList, stack};
    }

    template <yap::expr_kind Kind, typename Expr1, typename Expr2>
    auto operator() (yap::expr_tag<Kind>, Expr1 &&lhs, Expr2 &&rhs) {
        // printf("GenIR: binary op matched\n");
        auto genLhs = yap::transform(yap::as_expr(lhs), GenIRXform<decltype(mIRList), decltype(mStack), I>(mIRList, mStack));
        // printf("genLhs:\n"); PrintIRList(genLhs.mIRList);
        auto genRhs = yap::transform(yap::as_expr(rhs), genLhs);
        // printf("genRhs:\n"); PrintIRList(genRhs.mIRList);

        auto constexpr index = decltype(genRhs)::placeholder_index;
        auto temp = yap::make_terminal(temp_placeholder<index>{});
        auto assign = yap::make_expression<yap::expr_kind::assign>(
                std::move(temp),
                yap::make_expression<Kind>(
                        std::move(hana::back(genLhs.mStack)),  // lhs's result
                        std::move(hana::back(genRhs.mStack))   // rhs's result
                )
        );
        // printf("append:\n"); yap::print(std::cout, assign);
        auto newIRList = hana::append(genRhs.mIRList, std::move(assign));
        // Skip poping operands from mStack
        // Push result onto stack
        auto newStack = hana::append(mStack, std::move(temp));

        return GenIRXform<decltype(newIRList), decltype(newStack), index + 1>(newIRList, newStack);
    }

    template <typename Callable, typename ...Args>
    auto operator() (yap::expr_tag<yap::expr_kind::call>, Callable &&callable, Args &&...args) {
        // printf("GenIR: call matched\n");
        auto assign = yap::make_expression<yap::expr_kind::assign>(
                yap::make_terminal(temp_placeholder<I>{}),
                yap::make_expression<yap::expr_kind::call>(yap::as_expr(callable), yap::as_expr(args)...)
        );
        // printf("assign:\n"); yap::print(std::cout, assign);
        auto newIRList = hana::append(mIRList, std::move(assign));
        auto newStack = hana::append(mStack, std::move(yap::make_terminal(temp_placeholder<I>{})));
        return GenIRXform<decltype(newIRList), decltype(newStack), I + 1>{newIRList, newStack};
    }
};


struct GenIR : StaticTransform {
    using tag = xform_pass_tag;

    // Given a yap expression as an AST, returns a list of flattened IR
    template <typename AST, typename Dumping = nodump>
    constexpr auto transform(AST &&ast, Dumping dumping = nodump{}) const {
        static_assert(is_yap_expr_type<std::remove_reference_t<AST>>);
        auto gen = yap::transform(std::forward<AST>(ast), GenIRXform{hana::make_tuple(), hana::make_tuple()});

        if constexpr (need_dump(dumping)) {
            printf("--------IR After GenIR:--------\n");
            print_ir_list_simple(gen.mIRList);
        }
        return gen.mIRList;
    }
};