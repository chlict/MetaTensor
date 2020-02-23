#pragma once

#include <boost/hana.hpp>

template <typename ...T>
struct Dims {
    boost::hana::tuple<T...> data;

    constexpr Dims(T... t) : data(t...) {}

    static constexpr auto nDims = sizeof...(T);
};
