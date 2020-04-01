#pragma once

#include "Dims.hpp"

// Shape represents physical shape, usually has a type of Dims.
// Strides represents a type holding each dim's stride.
// Innermost dim positioned in leftmost.
template <typename Shape, typename Strides>
struct TensorLayout {
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
