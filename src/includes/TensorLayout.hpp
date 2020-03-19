#pragma once

#include "Dims.hpp"

// Shape represents physical shape, usually has a type of Dims.
// Stride represents a type holding each dim's stride.
// Innermost dim positioned in the rightmost element.
template <typename Shape, typename Stride>
struct TensorLayout {
    Shape shape_;

    Stride stride_;

    constexpr TensorLayout(const Shape &shape, const Stride &stride) :
            shape_(shape),
            stride_(stride) {}

    constexpr TensorLayout(Shape &&shape, Stride &&stride) noexcept :
            shape_(std::forward<Shape>(shape)),
            stride_(std::forward<Stride>(stride)) {}

    // If 'Shape' is a 'Dims' type which consists of a hana::tuple, layout.shape() cannot be used in constexpr
    // but layout.shape_ can (clang++ fixes this in clang11).
    // However, we can use 'shape = layout.shape()' outside of a constant-evaluation context and use the 'shape'
    // in constant-evaluation context.
    constexpr auto shape() const {
        return shape_;
    }

    constexpr auto stride() const {
        return stride_;
    }
};
