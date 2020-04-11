#pragma once

#include "Dims.hpp"

struct tensor_layout_tag;

template <typename T>
constexpr bool is_layout_type = is_a<tensor_layout_tag, T>;

// Shape represents physical shape, usually has a type of Dims.
// Strides represents a type holding each dim's stride.
// Innermost dim positioned in leftmost.
template <typename Shape, typename Strides>
struct TensorLayout {
    static_assert(is_dims_type<Shape> && is_dims_type<Strides>);

    using tag = tensor_layout_tag;

    Shape shape_;
    Strides strides_;

    constexpr TensorLayout(const Shape &shape, const Strides &stride) :
            shape_(shape),
            strides_(stride) {}

    // If 'Shape' is a 'Dims' type which consists of a hana::tuple, layout.shape() cannot be used in constexpr
    // but layout.shape can (clang++ fixes this in clang11).
    // However, we can use 'shape = layout.shape()' outside of a constant evaluation context and use the 'shape'
    // in constant evaluation context.
    constexpr auto shape() const {
        return shape_;
    }

    constexpr auto strides() const {
        return strides_;
    }
};
