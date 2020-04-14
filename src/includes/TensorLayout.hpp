#pragma once

#include "Dims.hpp"

struct tensor_layout_tag;

template <typename T>
constexpr bool is_layout_type = is_a<tensor_layout_tag, T>;

// Dimensions represents physical shape, usually of type Dims.
// Strides holds each dim's stride.
// Innermost dim positioned in leftmost.
template <typename Dimensions, typename Strides>
struct TensorLayout {
    static_assert(is_dims_type<Dimensions> && is_dims_type<Strides>);

    using tag = tensor_layout_tag;

    Dimensions dimensions_;
    Strides strides_;

    constexpr TensorLayout(const Dimensions &dimensions, const Strides &stride) :
            dimensions_(dimensions),
            strides_(stride) {}

    // If 'Dimensions' is a 'Dims' type which consists of a hana::tuple, layout.dimensions() cannot
    // be used in constexpr but layout.dimensions_ can (clang++ fixes this in clang11).
    // However, we can use 'dimensions = layout.dimensions()' outside of a constant evaluation context
    // and use the 'dimensions' in constant evaluation context.
    constexpr auto dimensions() const {
        return dimensions_;
    }

    constexpr auto strides() const {
        return strides_;
    }
};
