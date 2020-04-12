#pragma once

#include <range/v3/all.hpp>
#include "Dims.hpp"

struct tiling_provider_tag {};

template <typename Provider>
struct AbstractTilingProvider {
    using tag = tiling_provider_tag;

    static constexpr auto gen_tiling_indices() {
        return Provider::gen_tiling_indices();
    }

    template <typename Index>
    static constexpr auto index_to_pos(Index const &i) {
        return Provider::index_to_pos(i);
    }
};

struct VectorTilingProvider : AbstractTilingProvider<VectorTilingProvider> {
    template <typename Tensor, typename Tiling>
    static constexpr auto gen_tiling_indices(Tensor const &tensor, Tiling const &tiling) {
        static_assert(is_tensor_type<Tensor> && is_a<ttiling_tag, Tiling>);
        auto tensor_view = tensor.view();
        auto dim_orders = tiling.orders();
        static_assert(is_dims_type<decltype(tensor_view)> && is_dims_type<decltype(dim_orders)>);
        static_assert(decltype(tensor_view)::nDims == 1 && decltype(dim_orders)::nDims == 1);

        auto tiling_range = tiling.ranges()[0_c];
        int b = tiling_range.begin();
        int e = tiling_range.end();
        int step = tiling_range.step();
        auto indicies = ranges::view::ints(b, e) | ranges::view::filter([step](int x) { return x / step == 0; });
        return indicies;
    }

    template <typename Index>
    static constexpr auto index_to_pos(Index const &i) {
        static_assert(std::is_same_v<Index, int>);
        return Dim1(i);
    }
};