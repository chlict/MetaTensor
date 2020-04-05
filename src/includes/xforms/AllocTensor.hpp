#pragma once

#include <boost/yap/expression.hpp>
#include <boost/hana.hpp>
#include <boost/yap/print.hpp>
#include <iostream>
#include <cstdio>
#include "xforms/Placeholder.hpp"
#include "Tensor.hpp"

struct FakeTensor {
    int id;
    constexpr FakeTensor(int id) : id(id) {}
    friend std::ostream &operator << (std::ostream &os, const FakeTensor &t) {
        os << "tensor" << t.id;
        return os;
    }
};

constexpr auto MakeTensor(int id) {
    return FakeTensor(id);
}

template <typename Map>
struct AllocTensorXform {
    const Map &map_; // A map from TempVar to expression

    constexpr AllocTensorXform(const Map &map) : map_(map) {}

//    template <typename Expr1, typename Expr2>
//    auto operator() (yap::expr_tag<yap::expr_kind::assign>, Expr1 &&lhs, Expr2 &&rhs) {
//        // printf("AllocTensorXform: common assign matched\n");
//        assert(false && "Should not reach here");
//    }

//    template <typename Expr2, long long I>
//    auto operator() (yap::expr_tag<yap::expr_kind::assign>, temp_placeholder<I> const &temp, Expr2 &&rhs) const {
//        // printf("AllocTensorXform: assign to temp matched\n");
//        assert(false && "Should not reach here");
//    }

    template <long long I, yap::expr_kind Binary, typename Expr1, typename Expr2>
    auto operator()(yap::expr_tag<yap::expr_kind::assign>,
                    temp_placeholder<I> const &temp,
                    yap::expression<Binary, hana::tuple<Expr1, Expr2>>  &&binaryExpr) {
         printf("AllocTensorXform: assign _temp = lhs op rhs matched\n");
//        auto lhs = yap::left(binaryExpr);
//        auto rhs = yap::right(binaryExpr);
//        static_assert(decltype(lhs)::kind == yap::expr_kind::terminal);
//        static_assert(decltype(rhs)::kind == yap::expr_kind::terminal);
//        // TODO: use rhs's info to deduce information for tensor
////        auto tensor = MakeTensor(I);
//        auto lhs_tensor = yap::value(lhs);  // may be a reference
//        using TensorType = typename std::remove_reference<decltype(lhs_tensor)>::type;
//        static_assert(is_tensor<TensorType>);
//        using ElemType = typename tensor_traits<TensorType>::elem_type;
//        using Space = typename tensor_traits<TensorType>::space;
//        // TODO: alloc address for result_tensor
//        auto result_tensor = Tensor(ElemType(), lhs_tensor.format(), Space(), lhs_tensor.addr());
//        return yap::make_expression<yap::expr_kind::assign>(
//                yap::make_terminal(std::move(result_tensor)),
//                std::move(binaryExpr)
//        );
//        auto expr = yap::make_terminal(std::move(tensor));
//        return hana::insert(mMap, hana::make_pair(hana::llong_c<I>, expr));
    }

//    template <long long I, typename Fn, typename ...Args>
//    auto operator() (yap::expr_tag<yap::expr_kind::assign>, temp_placeholder<I> const &temp,
//                     yap::expression<yap::expr_kind::call, hana::tuple<Fn, Args...>> const &callExpr) {
//        printf("AllocBufferXform: assign _temp = call matched\n");
//        auto tensor = MakeTensor(I);
//        auto expr = yap::make_terminal(std::move(tensor));
//        return hana::insert(mMap, hana::make_pair(hana::llong_c<I>, expr));
//    }
};

template <typename Map>
struct AllocBufferXform {
    const Map &mMap; // A map from TempVar to expression

    constexpr AllocBufferXform(const Map &map) : mMap(map) {}

    template <typename Expr1, typename Expr2>
    auto operator() (yap::expr_tag<yap::expr_kind::assign>, Expr1 &&lhs, Expr2 &&rhs) {
        printf("AllocBufferXform: common assign matched\n");
        assert(false && "Should not reach here");
    }

    template <typename Expr2, long long I>
    auto operator() (yap::expr_tag<yap::expr_kind::assign>, temp_placeholder<I> const &temp, Expr2 &&rhs) const {
        printf("AllocBufferXform: assign to temp matched\n");
        assert(false && "Should not reach here");
    }

    template <long long I, yap::expr_kind Binary, typename Expr1, typename Expr2>
    auto operator() (yap::expr_tag<yap::expr_kind::assign>, temp_placeholder<I> const &temp,
                     yap::expression<Binary, hana::tuple<Expr1, Expr2>> const &binaryExpr) {
        printf("AllocBufferXform: assign _temp = lhs op rhs matched\n");
        auto lhs = yap::left(binaryExpr);
        auto rhs = yap::right(binaryExpr);
        static_assert(decltype(lhs)::kind == yap::expr_kind::terminal);
        static_assert(decltype(rhs)::kind == yap::expr_kind::terminal);
        // TODO: use rhs's info to infer information for tensor
        auto tensor = MakeTensor(I);
        auto expr = yap::make_terminal(std::move(tensor));
        return hana::insert(mMap, hana::make_pair(hana::llong_c<I>, expr));
    }

    template <long long I, typename Fn, typename ...Args>
    auto operator() (yap::expr_tag<yap::expr_kind::assign>, temp_placeholder<I> const &temp,
                     yap::expression<yap::expr_kind::call, hana::tuple<Fn, Args...>> const &callExpr) {
        printf("AllocBufferXform: assign _temp = call matched\n");
        auto tensor = MakeTensor(I);
        auto expr = yap::make_terminal(std::move(tensor));
        return hana::insert(mMap, hana::make_pair(hana::llong_c<I>, expr));
    }
};


template <typename IRList,
        typename = std::enable_if_t<
                hana::is_a<hana::tuple_tag, IRList>,
                void>
>
constexpr auto alloc_tensor(IRList const &ir_list) {
    auto fn = [](auto &&ir) -> decltype(auto) {
        print_type_name(ir);
//        return yap::transform(ir, AllocTensorXform(hana::make_map()));
        return yap::transform(ir, AllocBufferXform{hana::make_map()});
    };
    return hana::for_each(ir_list, fn);
//    return hana::transform(ir_list, [](auto const &ir) {
//        return ir;
//        //return yap::transform(ir, AllocTensorXform(hana::make_map()));
//    });
}
