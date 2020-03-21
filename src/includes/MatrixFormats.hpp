#pragma once

// Do layout for row-major stored matrix
template <>
struct AutoLayout<MATRIX_ROW_MAJOR> {
    // Given a 2 x 4 matrix:
    // view:           [rows:  2, cols:  4]
    // layout shape:   [outer: 2, inner: 4]
    // layout strides: [outer: 4, inner: 1]
    template <typename View>
    constexpr auto operator()(View&& view) const {
        static_assert(std::remove_reference<View>::type::nDims == 2, "Matrix expected");
        auto shape = Dims(view.dim[0_c], view.dim[1_c]);
        auto strides = Dims(view.dim[1_c], 1_c);
        return TensorLayout(shape, strides);
    }
};


// Do layout for column-major stored matrix
template <>
struct AutoLayout<MATRIX_COL_MAJOR> {
    // Given a 2 x 4 matrix:
    // view:           [rows:  2, cols:  4]
    // layout shape:   [outer: 4, inner: 2]
    // layout strides: [outer: 2, inner: 1]
    template <typename View>
    constexpr auto operator()(View&& view) const {
        static_assert(std::remove_reference<View>::type::nDims == 2, "Matrix expected");
        auto shape = Dims(view.dim[1_c], view.dim[0_c]);
        auto strides = Dims(view.dim[1_c], 1_c);
        return TensorLayout(shape, strides);
    }
};
