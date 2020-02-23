#pragma once

#include "Dims.hpp"

// PShape represents physical shape, usually a Dims type
// Stride represents a type holding each dim's stride
// innermost dim positioned in the rightmost element
template <typename PShape, typename Stride>
struct Layout {
    PShape pShape;

    Stride stride;

    constexpr Layout(const PShape &pShape, const Stride &stride) :
            pShape(pShape),
            stride(stride) {}

    constexpr Layout(PShape &&pShape, Stride &&stride) noexcept :
            pShape(std::forward<PShape>(pShape)),
            stride(std::forward<Stride>(stride)) {}

    // layout.pShape() cannot be used in constexpr but layout.mPShape can (but why?). So we delete the getters
//    constexpr auto pShape() const {
//        return pShape;
//    }
//
//    constexpr auto stride() const {
//        return stride;
//    }
};
