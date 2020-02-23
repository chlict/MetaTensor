#pragma once

#include <boost/hana.hpp>

template <typename ...T>
struct Dims {
    boost::hana::tuple<T...> dim;

    constexpr explicit Dims(T... t) : dim(t...) {}

    constexpr Dims(Dims &&other) noexcept : dim(other.dim) {}

    constexpr Dims(const Dims &other) : dim(other.dim) {}

    static constexpr auto nDims = sizeof...(T);
};
