#pragma once

#include <boost/hana.hpp>
#include "Utils.hpp"

namespace hana = boost::hana;

struct dims_tag {};

template <typename ...T>
struct Dims {
    using tag = dims_tag;

    static_assert(all_integral_or_constant<T...>, "only holding integral or constant numbers");

    hana::tuple<T...> dim;

    constexpr explicit Dims(T... t) : dim(t...) {}

    constexpr Dims(Dims &&other) noexcept : dim(std::forward<Dims>(other).dim) {}

    constexpr Dims(const Dims &other) : dim(other.dim) {}

    template <typename ...U>
    constexpr Dims(hana::tuple<U...> &&other) noexcept : dim(other) {}

    template <typename ...U>
    constexpr Dims(const hana::tuple<U...> &other) noexcept : dim(other) {}

    static constexpr auto nDims = sizeof...(T);
};

template <typename D0, typename D1>
constexpr auto Dim2(D0 d0, D1 d1) {
    static_assert(is_integral_or_constant<D0> && is_integral_or_constant<D1>);
    return Dims(d0, d1);
}

template <typename D0, typename D1, typename D2, typename D3>
constexpr auto Dim4(D0 d0, D1 d1, D2 d2, D3 d3) {
    static_assert(is_integral_or_constant<D0> && is_integral_or_constant<D1>);
    static_assert(is_integral_or_constant<D2> && is_integral_or_constant<D3>);
    return Dims(d0, d1, d2, d3);
}

template <typename D0, typename D1, typename D2, typename D3, typename D4>
constexpr auto Dim5(D0 d0, D1 d1, D2 d2, D3 d3, D4 d4) {
    static_assert(is_integral_or_constant<D0> && is_integral_or_constant<D1>);
    static_assert(is_integral_or_constant<D2> && is_integral_or_constant<D3> && is_integral_or_constant<D4>);
    return Dims(d0, d1, d2, d3, d4);
}