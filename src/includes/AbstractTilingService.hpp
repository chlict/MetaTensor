#pragma once

#include "Utils.hpp"
#include "Tensor.hpp"

struct tiling_service_tag {};

// Each tensor format should provide a tiling service which implements all
// the interfaces declared in AbstractTilingService
template <typename TilingService>
struct AbstractTilingService {
    using tag = tiling_service_tag;

    template <typename Tensor>
    constexpr auto gen_tiling_indices_for(Tensor const &tensor) const {
        static_assert(is_tensor_type<Tensor>);
        auto service = static_cast<const TilingService *>(this);
        return service->gen_tiling_indices_for(tensor);
    }

    template <typename Index>
    constexpr auto index_to_pos(Index const &i) const {
        auto service = static_cast<const TilingService *>(this);
        return service->index_to_pos(i);
    }
};
