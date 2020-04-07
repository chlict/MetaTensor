#pragma once
#include "Literals.hpp"
#include "LayoutProvider.hpp"

// Do layout for row-major stored matrix
struct RowMajorLayout : AbstractLayoutProvider<RowMajorLayout> {
    // Given a 2 x 4 matrix:
    // view:           [rows:  2, cols:  4]
    // layout shape:   [inner: 4, outer: 2]
    // layout strides: [inner: 1, outer: 4]
    template <typename View>
    constexpr auto operator()(View&& view) const {
        static_assert(std::remove_reference_t<View>::nDims == 2, "Matrix expected");
        auto shape = view_to_shape(view);
        auto strides = shape_to_strides(shape);
        return TensorLayout(shape, strides);
    }

    template <typename View>
    constexpr auto view_to_shape(View &&view) const {
        static_assert(std::remove_reference_t<View>::nDims == 2, "Matrix expected");
        return Dims(view.dim[1_c], view.dim[0_c]);
    }

    template <typename Shape>
    constexpr auto shape_to_strides(Shape &&shape) const {
        static_assert(std::remove_reference_t<Shape>::nDims == 2, "Matrix expected");
        return Dims(1_c, shape.dim[0_c]);
    }

};

// Do layout for column-major stored matrix
struct ColMajorLayout : AbstractLayoutProvider<ColMajorLayout> {
    // Given a 2 x 4 matrix:
    // view:           [rows:  2, cols:  4]
    // layout shape:   [inner: 2, outer: 4]
    // layout strides: [inner: 1, outer: 2]
    template <typename View>
    constexpr auto operator()(View&& view) const {
        static_assert(std::remove_reference_t<View>::nDims == 2, "Matrix expected");
        auto shape = view_to_shape(view);
        auto strides = shape_to_strides(shape);
        return TensorLayout(shape, strides);
    }

    template <typename View>
    constexpr auto view_to_shape(View &&view) const {
        static_assert(std::remove_reference_t<View>::nDims == 2, "Matrix expected");
        return Dims(view.dim[0_c], view.dim[1_c]);
    }

    template <typename Shape>
    constexpr auto shape_to_strides(Shape &&shape) const {
        static_assert(std::remove_reference_t<Shape>::nDims == 2, "Matrix expected");
        return Dims(1_c, shape.dim[0_c]);
    }
};
