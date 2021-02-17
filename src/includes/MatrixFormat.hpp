#pragma once
#include "LayoutProvider.hpp"
#include "Literals.hpp"

namespace mt {

// Do layout for row-major stored matrix
struct RowMajorLayout : AbstractLayoutProvider<RowMajorLayout> {
  // Given a 2 x 4 matrix:
  // view (tensor shape): [rows:  2, cols:  4]
  // layout dimensios:    [inner: 4, outer: 2]
  // layout strides:      [inner: 1, outer: 4]
  template <typename View>
  constexpr auto operator()(View const& view) const {
    static_assert(View::nDims == 2, "Matrix expected");
    auto dimensions = view_to_dimensions(view);
    auto strides = dimensions_to_strides(dimensions);
    return TensorLayout(dimensions, strides);
  }

  template <typename View>
  static constexpr auto view_to_dimensions(View const& view) {
    static_assert(View::nDims == 2, "Matrix expected");
    return Dims(view.dim[1_c], view.dim[0_c]);
  }

  template <typename Dimensions>
  static constexpr auto dimensions_to_strides(Dimensions const& dims) {
    static_assert(Dimensions::nDims == 2, "Matrix expected");
    return Dims(1_c, dims.dim[0_c]);
  }
};

// Do layout for column-major stored matrix
struct ColMajorLayout : AbstractLayoutProvider<ColMajorLayout> {
  // Given a 2 x 4 matrix:
  // view (tensor shape): [rows:  2, cols:  4]
  // layout dimensions:   [inner: 2, outer: 4]
  // layout strides:      [inner: 1, outer: 2]
  template <typename View>
  constexpr auto operator()(View const& view) const {
    static_assert(View::nDims == 2, "Matrix expected");
    auto dimensions = view_to_dimensions(view);
    auto strides = dimensions_to_strides(dimensions);
    return TensorLayout(dimensions, strides);
  }

  template <typename View>
  static constexpr auto view_to_dimensions(View const& view) {
    static_assert(View::nDims == 2, "Matrix expected");
    return Dims(view.dim[0_c], view.dim[1_c]);
  }

  template <typename Dimensions>
  static constexpr auto dimensions_to_strides(Dimensions const& dims) {
    static_assert(Dimensions::nDims == 2, "Matrix expected");
    return Dims(1_c, dims.dim[0_c]);
  }
};

}  // namespace mt
