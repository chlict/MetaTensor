#pragma once
#include "LayoutProvider.hpp"
#include "Literals.hpp"

namespace mt {

struct VectorLayout : AbstractLayoutProvider<VectorLayout> {
  // Given a length N vector:
  // view (tensor shape): [N]
  // layout dimensios:    [N]
  // layout strides:      [1]
  template <typename View>
  constexpr auto operator()(View &&view) const {
    static_assert(std::remove_reference_t<View>::nDims == 1, "Vector expected");
    auto dims = view_to_dimensions(view);
    auto strides = dimensions_to_strides(dims);
    return TensorLayout(dims, strides);
  }

  template <typename View>
  static constexpr auto view_to_dimensions(View &&view) {
    static_assert(std::remove_reference_t<View>::nDims == 1, "Vector expected");
    return Dim1(view.dim[0_c]);
  }

  template <typename Dimensions>
  static constexpr auto dimensions_to_strides(Dimensions &&dims) {
    static_assert(std::remove_reference_t<Dimensions>::nDims == 1,
                  "Vector expected");
    return Dim1(1_c);
  }
};

}  // namespace mt
