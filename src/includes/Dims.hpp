#pragma once

#include <boost/hana.hpp>

template <typename ...T>
struct Dims {
    boost::hana::tuple<T...> dim;

    constexpr Dims(T... t) : dim(t...) {}

    static constexpr auto nDims = sizeof...(T);
};
