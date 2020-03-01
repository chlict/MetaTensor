#pragma once

#include "Layout.hpp"

enum Format {
    MatrixRowMajor,
    MatrixColMajor
};

// LShape represents a logical shape. E.g. a fractal tensor may have a logical shape of Dim2(width, height)
// like a matrix.
// Layout represents a layout of a format.
template<typename LShape, typename Layout>
struct TensorFormat {
    LShape lShape;
    Layout layout;

    constexpr TensorFormat(LShape l_shape, Layout layout) : lShape(l_shape), layout(layout) {}
};

struct AutoLayout {};

template<Format tag, typename LShape, typename Layout>
TensorFormat<LShape, Layout> make_format(LShape l_shape, Layout layout = AutoLayout()) {
    return TensorFormat(l_shape, layout);
}