#pragma once

#include <range/v3/all.hpp>

#include "TRange.hpp"
#include "TilingService.hpp"

namespace mt {

template <typename TRange>
struct VectorTilingService
    : AbstractTilingService<VectorTilingService<TRange> > {
  static_assert(is_a<trange_tag, TRange>);

  TRange const trange_;

  constexpr explicit VectorTilingService(TRange const &trange)
      : trange_(trange) {}

  constexpr explicit VectorTilingService(TRange &&trange)
      : trange_(std::move(trange)) {}

  constexpr VectorTilingService(VectorTilingService const &other)
      : trange_(other.trange_) {}

  constexpr VectorTilingService(VectorTilingService &&other) noexcept
      : trange_(std::move(other.trange_)) {}

  template <typename Tensor>
  constexpr auto gen_tiling_indices_for(Tensor const &tensor) {
    static_assert(is_tensor_type<Tensor>);
    auto tensor_shape = tensor.shape();
    static_assert(decltype(tensor_shape)::nDims == 1);

    namespace views = ranges::views;
    // Given a tiling range of [4, 10, 2], generates a sequence of [4, 6, 8, 10)
    auto trange = trange_;
    auto b = trange.begin();
    auto e = trange.end();
    auto step = trange.step();
    auto count = int_ceil(e - b, step);

    int bi = (int)b;
    int stepi = (int)step;
    auto indices =
        views::ints(0, (int)count) |
        views::transform([bi, stepi](int i) { return bi + i * stepi; });
    return indices;
  }

  template <typename Index>
  constexpr auto index_to_pos(Index const &i) const {
    static_assert(std::is_integral_v<Index>);
    return Dim1(i);
  }

  constexpr auto gen_tile_shape() const { return Dim1(trange_.size()); }

  friend std::ostream &operator<<(std::ostream &os,
                                  VectorTilingService const service) {
    os << "Vector tiling: " << service.trange_;
    return os;
  }
};

}  // namespace mt
