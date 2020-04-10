#pragma once

#include <iostream>
#include <boost/hana.hpp>
#include "Literals.hpp"
#include "Dims.hpp"
#include "Utils.hpp"
#include "TRange.hpp"

struct tiling_desc_tag;

// Describes how to tile a (multi-dimensional) tensor. Orders give the dimension order for tiling, the first will be
// the innermost. Ranges tell how to tile each dimension. E.g, for a 2D tensor, if orders = [0, 1], then tiling will
// be first along tensor.view[0] (forms inner loop) and second along tensor.view[1] (forms outer loop). The tiling
// is performed according to ranges[orders[0]] and ranges[orders[1]] respectively.
template <typename Orders, typename ...Ranges>
struct TilingDesc {
    static_assert(is_a<dims_tag, Orders> && Orders::nDims == sizeof...(Ranges) && all_are<trange_tag, Ranges...>);

    using tag = tiling_desc_tag;

    const Orders orders_;
    const boost::hana::tuple<Ranges...> ranges_;

    constexpr TilingDesc(Orders const &orders, Ranges ...ranges) : orders_(orders), ranges_(ranges...) {}

    constexpr TilingDesc(TilingDesc const &other) : orders_(other.orders_), ranges_(other.ranges_) {}

    constexpr TilingDesc(TilingDesc &&other) : orders_(other.orders_), ranges_(other.ranges_) {}

    constexpr auto orders() const { return orders_; }

    constexpr auto ranges() const { return ranges_; }

    template <int I>
    constexpr auto range() const { return ranges_[hana::size_c<I>]; }

    template <typename Inputs, typename Expr>
    constexpr auto apply(Inputs &&inputs, Expr &&expr) const {
        static_assert(hana::is_a<hana::tuple_tag, Inputs>);
    }

    friend std::ostream& operator << (std::ostream &os, TilingDesc const &tiling) {
        auto indices = hana::make_range(0_c, hana::size_c<Orders::nDims>);
        hana::for_each(indices, [&os, &tiling](auto i) {
            auto dim_i = tiling.orders_.dim[i];
            os << "order[" << i << "]: " << dim_i << ", " << tiling.ranges_[dim_i] << "\n";
        });
        return os;
    }
};

template <typename Range>
constexpr auto Tiling1D(Range &&range) {
    return TilingDesc(Dims(0_c), std::forward<Range>(range));
}

template <typename RangeRow, typename RangeCol>
constexpr auto Tiling2DRowMajor(RangeRow &&range_row, RangeCol &&range_col) {
    return TilingDesc(Dims(0_c, 1_c), std::forward<RangeRow>(range_row), std::forward<RangeCol>(range_col));
}

template <typename RangeRow, typename RangeCol>
constexpr auto Tiling2DColMajor(RangeRow &&range_row, RangeCol &&range_col) {
    return TilingDesc(Dims(1_c, 0_c), std::forward<RangeRow>(range_row), std::forward<RangeCol>(range_col));
}

