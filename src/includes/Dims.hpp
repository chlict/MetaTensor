#pragma once

#include <boost/hana.hpp>

namespace hana = boost::hana;

template <typename ...T>
struct Dims {
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
