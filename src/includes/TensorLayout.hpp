#pragma once

#include "Dims.hpp"

// Shape represents physical shape, usually has a type of Dims.
// Strides represents a type holding each dim's stride.
// Innermost dim positioned in leftmost.
template <typename Shape, typename Strides>
struct TensorLayout {
    Shape shape;

    Strides strides;

    constexpr TensorLayout(const Shape &shape, const Strides &stride) :
            shape(shape),
            strides(stride) {}

    constexpr TensorLayout(Shape &&shape, Strides &&stride) noexcept :
            shape(std::forward<Shape>(shape)),
            strides(std::forward<Strides>(stride)) {}

    // If 'Shape' is a 'Dims' type which consists of a hana::tuple, layout.get_shape() cannot be used in constexpr
    // but layout.shape can (clang++ fixes this in clang11).
    // However, we can use 'shape = layout.get_shape()' outside of a constant-evaluation context and use the 'shape'
    // in constant-evaluation context.
    constexpr auto get_shape() const {
        return shape;
    }

    constexpr auto get_strides() const {
        return strides;
    }
};
