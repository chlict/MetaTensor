#pragma once

#include "Tensor.hpp"
#include "TilingService.hpp"

struct toperand_tag;

template <typename T>
constexpr bool is_toperand_type = is_a<toperand_tag, T>;

// Operator operand: tensor with tiling description
template <typename Tensor, typename TilingService>
struct TOperand {
  static_assert(is_tensor_type<Tensor> &&
                is_tiling_service_type<TilingService>);

  using tag = toperand_tag;

  const Tensor tensor_;
  const TilingService tiling_service_;
  // const Padding padding_;

  constexpr TOperand(Tensor const &tensor, TilingService const &tiling_service)
      : tensor_(tensor), tiling_service_(tiling_service) {}

  constexpr TOperand(Tensor &&tensor, TilingService &&tiling_service)
      : tensor_(std::move(tensor)),
        tiling_service_(std::move(tiling_service)) {}

  constexpr TOperand(Tensor const &other)
      : tensor_(other.tensor_), tiling_service_(other.tiling_service_) {}

  constexpr TOperand(Tensor &&other) noexcept
      : tensor_(other.tensor_), tiling_service_(other.tiling_service_) {}

  constexpr auto tensor() const { return tensor_; }

  constexpr auto tiling_service() const { return tiling_service_; }

  constexpr auto gen_tiling_indices() const {
    return tiling_service_.gen_tiling_indices_for(tensor_);
  }

  template <typename Index>
  constexpr auto get_tile(Index const &i) const {
    auto tile_pos = tiling_service_.index_to_pos(i);
    auto tile_shape = tiling_service_.gen_tile_shape();
    return tensor_.get_tile(tile_pos, tile_shape);
  }

  friend std::ostream &operator<<(std::ostream &os, TOperand const &opnd) {
    os << "TOpnd [\n\t" << opnd.tensor() << "\n\t" << opnd.tiling_service()
       << "\n]";
    return os;
  }
};
