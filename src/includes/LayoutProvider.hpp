#pragma once

#include "Dims.hpp"
#include "TensorLayout.hpp"

namespace mt {

// Use the CRTP trick to do static polymorphism. Each layout type should inherit
// this struct and provide methods to do the actual layout.
struct layout_provider_tag {};

template <typename Provider>
struct AbstractLayoutProvider {
  using tag = layout_provider_tag;

  template <typename View>
  constexpr auto operator()(View &&view) const {
    const Provider *provider = static_cast<const Provider *>(this);
    return (*provider)(view);
  }

  template <typename View>
  static constexpr auto view_to_shape(View &&view) {
    return Provider::view_to_shape(view);
  }

  template <typename Shape>
  static constexpr auto shape_to_strides(Shape &&shape) {
    return Provider::shape_to_strides(shape);
  }

  template <typename Pos, typename Layout>
  static constexpr auto offset(Pos &&pos, Layout &&layout) {
    static_assert(is_dims_type<std::remove_reference_t<Pos> >);
    static_assert(is_layout_type<std::remove_reference_t<Layout> >);
    auto dimensions = view_to_shape(std::forward<Pos>(pos));
    auto strides = layout.strides();
    // Takes RowMajorLayout for example:
    // view:         [rows: 2, cols: 4]
    // layout.shape  [dim0: 4, dim0: 2]
    // layout.strides[dim0: 1, dim1: 4]
    // pos:          [row:  1, col:  2]
    // dimensions:   [dim0: 2, dim1: 1]
    // offset should be (2 * 1 + 1 * 4)
    return hana::fold(hana::zip_with(hana::mult, dimensions.dim, strides.dim),
                      hana::plus);
  }
};

}  // namespace mt
