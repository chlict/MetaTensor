#pragma once

#include <iostream>
#include "boost/hana.hpp"
#include "Literals.hpp"
#include "Dims.hpp"
#include "Utils.hpp"

struct trange_tag;

// A range is made up with [start, end, step, size]. It describes how to tile a dimension of a tensor, e.g,
// for a 1D tensor of which length is 6, say (a0, a1, a2, a3, a4, a5), a tile range of [0, 6, 3, 2] tells
// tiling it as tile0 = (a0, a1) and tile1 = (a3, a4).
// If the 'size' is not specified, it will be equal to the 'step'. So a range of [0, 6, 3] results in
// tile0 = (a0, a1, a2) and tile1 = (a3, a4, a5)
template <typename Start, typename End, typename Step, typename Size = Step,
        typename = std::enable_if_t<
                is_integral_or_constant<Start> &&
                is_integral_or_constant<End> &&
                is_integral_or_constant<Step> &&
                is_integral_or_constant<Size>,
        void> >
struct TRange {
    using tag = trange_tag;

    const Start start_;
    const End end_;
    const Step step_;
    const Size size_;

    constexpr TRange(Start const &start, End const &end, Step const &step, Size const &size = Size()) :
        start_(start), end_(end), step_(step), size_(size) {}

    constexpr TRange(TRange const &other) :
        start_(other.start_), end_(other.end_), step_(other.step_), size_(other.size_) {}

    constexpr TRange(TRange &&other) :
        start_(other.start_), end_(other.end_), step_(other.step_), size_(other.size_) {}

    constexpr auto start() const { return start_; }

    constexpr auto end() const { return end_; }

    constexpr auto step() const { return step_; }

    constexpr auto size() const { return size_; }

    friend std::ostream& operator<<(std::ostream &os, TRange const &r) {
        os << "Range: [" << r.start() << ", " << r.end() << ", " << r.step() << ", " << r.size() << "]";
        return os;
    }
};

struct tiling_desc_tag;

// Describes how to tile a (multi-dimensional) tensor. Orders gives the dimension order for tiling, the first will be
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

