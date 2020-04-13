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

        namespace views = ranges::views;
        // Given a tiling range of [4, 10, 2], generates a sequence of [4, 6, 8, 10)
        auto trange = tiling.ranges()[0_c];
        auto b = trange.begin();
        auto e = trange.end();
        auto step = trange.step();
        // ceil((e - b) / step)
        int count = ((e - b) % step) == 0 ? (e - b) / step : (e - b) / step + 1;

        int bi = (int)b;
        auto indices = views::ints(0, count) | views::transform([bi, step](int x) { return bi + x * step; });
        return indices;
    }

    template <typename Index>
    static constexpr auto index_to_pos(Index const &i) {
        static_assert(std::is_same_v<Index, int>);
        return Dim1(i);
    }
};