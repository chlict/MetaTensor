#pragma once

#include <boost/hana.hpp>
#include <boost/yap/expression.hpp>
#include <boost/yap/print.hpp>
#include <cstdio>
#include <iostream>

#include "Tensor.hpp"
#include "XformUtils.hpp"
#include "xforms/Placeholder.hpp"
#include "xforms/PrintIR.hpp"

/**
 * Given a list of IR, this transform allocates new tensors and replace the
 * placeholders with tensors. A list of IR given as: _1 = a * b _2 = _1 + c
 * After the AllocTensor transformation, it will be changed as:
 *      tensor1 = a * b
 *      tensor2 = tensor1 + c
 * where tensor1 and tensor2 are newly allocated tensors with format deductions.
 * The transform is splitted into two steps. The first step is establishing a
 * map mapping from placeholders to newly allocated tensors. The second step is
 * to replace the placeholders with the tensors.
 */

template <typename Lhs, typename ExprMap>
constexpr auto deduce_tensor_helper(Lhs &&lhs, ExprMap &&map) {
  using T = std::remove_reference_t<Lhs>;
  static_assert(is_tensor_type<T> || is_temp<T>);
  if constexpr (is_tensor_type<T>) {
    return lhs;
  } else {
    auto tensor_expr =
        map[hana::llong_c<T::id>];  // should be a terminal of tensor
    return yap::value(tensor_expr);
  }
}

template <typename Lhs, typename ExprMap,
          typename = std::enable_if_t<
              (is_tensor_type<std::remove_reference_t<Lhs>> ||
               is_temp<std::remove_reference_t<Lhs>>)&&hana::
                  is_a<hana::map_tag, std::remove_reference_t<ExprMap>>,
              void>>
constexpr auto deduce_tensor(Lhs &&lhs, ExprMap &&map) {
  if constexpr (is_temp<std::remove_reference_t<Lhs>>) {
    using T = std::remove_reference_t<Lhs>;
    static_assert(
        hana::contains(decltype(hana::keys(map))(), hana::llong_c<T::id>));
  }

  auto src_tensor = deduce_tensor_helper(static_cast<Lhs &&>(lhs),
                                         static_cast<ExprMap &&>(map));
  using TensorType = typename std::remove_reference_t<decltype(src_tensor)>;
  static_assert(is_tensor_type<TensorType>);

  using ElemType = typename tensor_traits<TensorType>::elem_type;
  using Space = typename tensor_traits<TensorType>::space;
  // TODO: alloc address for result_tensor
  auto result_tensor =
      Tensor(ElemType(), src_tensor.format(), Space(), src_tensor.addr());
  return result_tensor;
}

/*
 * This scanner does not change the ir list. It scans for placeholders and
 * creates tensors. During the scan, a map is established to record placeholder
 * => tensor_expression.
 */
template <typename ExprMap>
struct TempScanner {
  // The state holder - a map from temp_placeholder to expression
  const ExprMap map_;

  constexpr TempScanner(const ExprMap &map) : map_(map) {}

  // State update - insert a pair and get a new map
  template <long long I, typename Expr>
  constexpr auto new_state(Expr &&expr) const {
    return hana::insert(map_,
                        hana::make_pair(hana::llong_c<I>, std::move(expr)));
  }

  template <long long I, yap::expr_kind Binary, typename Expr1, typename Expr2>
  auto operator()(
      yap::expr_tag<yap::expr_kind::assign>, temp_placeholder<I> const &temp,
      yap::expression<Binary, hana::tuple<Expr1, Expr2>> const &binaryExpr) {
    // printf("assign _temp = lhs op rhs matched\n");
    auto lhs_expr = yap::left(binaryExpr);
    auto rhs_expr = yap::right(binaryExpr);
    static_assert(decltype(lhs_expr)::kind == yap::expr_kind::terminal);
    static_assert(decltype(rhs_expr)::kind == yap::expr_kind::terminal);

    auto lhs = yap::value(lhs_expr);
    auto rhs = yap::value(rhs_expr);
    using LhsType = typename std::remove_reference<decltype(lhs)>::type;
    using RhsType = typename std::remove_reference<decltype(rhs)>::type;
    static_assert(is_tensor_type<LhsType> || is_temp<LhsType>);
    static_assert(is_tensor_type<RhsType> || is_temp<RhsType>);

    // By default deduces result tensor from lhs
    auto result_tensor = deduce_tensor(lhs, map_);

    auto expr = yap::make_terminal(std::move(result_tensor));
    return new_state<I>(expr);
  }

  template <long long I, typename Fn, typename... Args>
  auto operator()(yap::expr_tag<yap::expr_kind::assign>,
                  temp_placeholder<I> const &temp,
                  yap::expression<yap::expr_kind::call,
                                  hana::tuple<Fn, Args...>> const &callExpr) {
    printf("TempScaner: assign _temp = call matched\n");
    assert(false);
    //        auto tensor = MakeTensor(I);
    //        auto expr = yap::make_terminal(std::move(tensor));
    //        return hana::insert(map_, hana::make_pair(hana::llong_c<I>,
    //        expr)); return expr;
    auto expr = yap::make_terminal(1);
    return new_state<I>(expr);
  }
};

// Substitute the placeholders for tensor expressions according to the ExprMap
template <typename ExprMap>
struct SubstituteXform {
  const ExprMap &map_;
  constexpr SubstituteXform(const ExprMap &map) : map_(map) {}

  template <long long I>
  auto operator()(yap::expr_tag<boost::yap::expr_kind::terminal>,
                  temp_placeholder<I> const &temp) {
    static_assert(
        hana::contains(decltype(hana::keys(map_))(), hana::llong_c<I>));
    // auto tensor = map_[boost::hana::llong_c<I>];
    // std::cout << "I == " << I << std::endl;
    // print_type_name(tensor);
    // boost::yap::print(std::cout, tensor);
    return map_[boost::hana::llong_c<I>];
  }
};

struct AllocTensor : StaticTransform {
  template <
      typename IRList, typename Dumping = DumpFlag::OFF,
      typename = std::enable_if_t<hana::is_a<hana::tuple_tag, IRList>, void>>
  constexpr auto transform(IRList &&ir_list,
                           Dumping dumping = DumpFlag::OFF{}) const {
    // step1 - scan the ir list and create tensors for temps
    auto map = hana::fold_left(ir_list,
                               hana::make_map() /* init state, an empty map */,
                               // For each ir in ir_list, scan for the temp
                               // tensor and make a record in the map
                               [](auto &&map, auto &&ir) {
                                 return yap::transform(ir, TempScanner(map));
                               });

    // step2 - substitute the temps
    auto new_ir_list = hana::transform(ir_list, [&map](auto const &ir) {
      return yap::transform(ir, SubstituteXform(map));
    });

    if constexpr (need_dump(dumping)) {
      printf("--------IR After AllocTensor:--------\n");
      print_ir_list_simple(new_ir_list);
    }

    return new_ir_list;
  }
};

// template <typename IRList,
//        typename = std::enable_if_t<
//                hana::is_a<hana::tuple_tag, IRList>,
//                void>
//>
// constexpr auto alloc_tensor(IRList const &ir_list) {
//    auto fn = [](auto && map, auto &&ir) -> decltype(auto) {
//        return yap::transform(ir, TempScanner(map));
//    };
//
//    return hana::fold_left(ir_list, hana::make_map(), fn);
//}
//
//// Given a sequence of IR and a map from temp_placeholders to expressions,
///substitute each / occurence of temp_placeholder for the corresponding
///expression
// template <typename Sequence, typename Map>
// auto SubstituteTemps(Sequence &&irList, Map &&map) {
//    return hana::transform(irList, [&map](auto const &ir) {
//        return yap::transform(ir, SubstituteXform{map});
//    });
//}
