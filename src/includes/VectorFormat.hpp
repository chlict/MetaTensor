#pragma once
#include "Literals.hpp"
#include "LayoutProvider.hpp"

struct VectorLayout : AbstractLayoutProvider<VectorLayout> {
    // Given a length N vector:
    // view:           [N]
    // layout shape:   [N]
    // layout strides: [1]
    template <typename View>
    constexpr auto operator()(View&& view) const {
        static_assert(std::remove_reference_t<View>::nDims == 1, "Vector expected");
        auto shape = view_to_shape(view);
        auto strides = shape_to_strides(shape);
        return TensorLayout(shape, strides);
    }

    template <typename View>
    static constexpr auto view_to_shape(View &&view) {
        static_assert(std::remove_reference_t<View>::nDims == 1, "Vector expected");
        return Dim1(view.dim[0_c]);
    }

    template <typename Shape>
    static constexpr auto shape_to_strides(Shape &&shape) {
        static_assert(std::remove_reference_t<Shape>::nDims == 1, "Vector expected");
        return Dims(1_c);
    }
};

