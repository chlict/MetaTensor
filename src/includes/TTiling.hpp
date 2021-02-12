#pragma once

#include <boost/hana.hpp>
#include <iostream>

#include "Dims.hpp"
#include "ECompiler.hpp"
#include "Literals.hpp"
#include "TRange.hpp"
#include "Utils.hpp"

struct ttiling_tag;

// Describes how to tile a (multi-dimensional) tensor. Orders give the dimension
// order for tiling, the first will be the innermost. Ranges tell how to tile
// each dimension. E.g, for a 2D tensor, if orders = [0, 1], then tiling will be
// first along tensor.view[0] (forms inner loop) and second along tensor.view[1]
// (forms outer loop). The tiling is performed according to ranges[orders[0]]
// and ranges[orders[1]] respectively.
template <typename Orders, typename... Ranges>
struct TTiling {
  static_assert(is_a<dims_tag, Orders> && Orders::nDims == sizeof...(Ranges) &&
                all_are<trange_tag, Ranges...>);

  using tag = ttiling_tag;

  const Orders orders_;
  const boost::hana::tuple<Ranges...> ranges_;

  constexpr TTiling(Orders const &orders, Ranges const &...ranges)
      : orders_(orders), ranges_(ranges...) {}

  constexpr TTiling(Orders &&orders, Ranges &&...ranges)
      : orders_(std::move(orders)), ranges_(std::move(ranges)...) {}

  constexpr TTiling(TTiling const &other)
      : orders_(other.orders_), ranges_(other.ranges_) {}

  constexpr TTiling(TTiling &&other)
      : orders_(std::move(other.orders_)), ranges_(std::move(other.ranges_)) {}

  constexpr auto orders() const { return orders_; }

  constexpr auto ranges() const { return ranges_; }

  template <int I>
  constexpr auto range() const {
    return ranges_[hana::size_c<I>];
  }

  template <typename Tensor, typename Expr>
  constexpr auto apply(Tensor &&tensor, Expr &&expr) const {
    static_assert(is_tensor_type<std::remove_reference_t<Tensor>>);
    static_assert(Orders::nDims == 1);  // only support 1D tensor
    auto range0 = ranges_[0_c];
    // be careful not use size_t or unsigned for i or it will compile with error
    // "not in the same ring" with hana data
    for (long i = range0.begin(); i < range0.end(); i += range0.step()) {
      auto tile = tensor.get_tile(Dims(i), Dims(range0.size()));
      // std::cout << tile << std::endl;
      auto compiler = ECompiler(std::forward<Expr>(expr));
      auto core = compiler.compile(tile, tile);
      core();
    }
  }

  friend std::ostream &operator<<(std::ostream &os, TTiling const &tiling) {
    auto indices = hana::make_range(0_c, hana::size_c<Orders::nDims>);
    hana::for_each(indices, [&os, &tiling](auto i) {
      auto dim_i = tiling.orders_.dim[i];
      os << "order[" << i << "]: " << dim_i << ", " << tiling.ranges_[dim_i]
         << "\n";
    });
    return os;
  }
};

template <typename Range>
constexpr auto Tiling1D(Range &&range) {
  return TTiling(Dims(0_c), std::forward<Range>(range));
}

template <typename RangeRow, typename RangeCol>
constexpr auto Tiling2DRowMajor(RangeRow &&range_row, RangeCol &&range_col) {
  return TTiling(Dims(0_c, 1_c), std::forward<RangeRow>(range_row),
                 std::forward<RangeCol>(range_col));
}

template <typename RangeRow, typename RangeCol>
constexpr auto Tiling2DColMajor(RangeRow &&range_row, RangeCol &&range_col) {
  return TTiling(Dims(1_c, 0_c), std::forward<RangeRow>(range_row),
                 std::forward<RangeCol>(range_col));
}
