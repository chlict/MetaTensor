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

// Sometimes use Dim2, Dim4, Dim5 is more readable than Dims.
#define define_dims(N)                                                          \
template <typename ...T>                                                        \
struct Dim##N {                                                                 \
    static_assert(sizeof...(T) == N);                                           \
                                                                                \
    static_assert(all_integral_or_constant<T...>,                               \
            "only holding integral or constant numbers");                       \
                                                                                \
    using tag = dims_tag;                                                       \
                                                                                \
    hana::tuple<T...> dim;                                                      \
                                                                                \
    constexpr explicit Dim##N(T... t) : dim(t...) {}                            \
                                                                                \
    constexpr Dim##N(Dim##N &&other) noexcept : dim(other.dim) {}               \
                                                                                \
    constexpr Dim##N(const Dim##N &other) : dim(other.dim) {}                   \
                                                                                \
    template <typename ...U>                                                    \
    constexpr Dim##N(hana::tuple<U...> &&other) noexcept : dim(other) {}        \
                                                                                \
    template <typename ...U>                                                    \
    constexpr Dim##N(const hana::tuple<U...> &other) noexcept : dim(other) {}   \
                                                                                \
    static constexpr auto nDims = sizeof...(T);                                 \
};

define_dims(2)
define_dims(4)
define_dims(5)
