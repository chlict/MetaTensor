#pragma once

#include <cstdio>

#include "TensorOps.hpp"

template <typename Dst, typename Src>
struct TMov : public BaseTMov<Dst, Src, TMov> {
  static_assert(is_tensor_type<Dst> && is_tensor_type<Src>);

  constexpr TMov(Dst const &output, Src const &input)
      : BaseTMov<Dst, Src, TMov>(output, input) {}

  constexpr TMov(TMov const &other) : BaseTMov<Dst, Src, TMov>(other) {}

  constexpr TMov(TMov &&other) : BaseTMov<Dst, Src, TMov>(std::move(other)) {}

  constexpr auto gen_code() const {
    auto fn = []() { printf("x86::TMov executed\n"); };
    return fn;
  }
};
