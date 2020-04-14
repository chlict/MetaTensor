#pragma once

#include <range/v3/all.hpp>
#include "Dims.hpp"
#include "TTiling.hpp"

struct RowMajorTiling : AbstractTilingProvider<RowMajorTiling> {

    template <typename Tensor, typename Tiling>
    static constexpr auto gen_tiling_indices(Tensor const &tensor, Tiling const &tiling) {
        static_assert(is_tensor_type<Tensor> && is_a<ttiling_tag, Tiling>);
        auto tensor_shape = tensor.shape();
        auto dim_orders = tiling.orders();
        static_assert(is_dims_type<decltype(tensor_shape)> && is_dims_type<decltype(dim_orders)>);
        static_assert(decltype(tensor_shape)::nDims == 2 && decltype(dim_orders)::nDims == 2);

        namespace views = ranges::views;
        // Given a row major tiling:
        // range[0]: [begin: 0, end: 6, step: 2, size: 2]
        // range[1]: [begin: 0, end: 16, step: 4, size: 4]
        // generates a sequence of [
        // [0, 0], [0, 4], [0, 8], [0, 12],
        // [2, 0], [2, 4], [2, 8], [2, 12],
        // [4, 0], [4, 4], [4, 8], [4, 12],
        auto trange0 = tiling.ranges()[0_c];
        auto b0 = trange0.begin();
        auto e0 = trange0.end();
        auto step0 = trange0.step();
        auto count0 = int_ceil(e0 - b0, step0);

        auto trange1 = tiling.ranges()[1_c];
        auto b1 = trange1.begin();
        auto e1 = trange1.end();
        auto step1 = trange1.step();
        auto count1 = int_ceil(e1 - b1, step1);

        int bi0 = (int)b0;
        int bi1 = (int)b1;
        auto indices0 = views::ints(0, (int)count0) | views::transform([bi0, step0](int x) { return bi0 + x * step0; });
        auto indices1 = views::ints(0, (int)count1) | views::transform([bi1, step1](int x) { return bi1 + x * step1; });
        return views::cartesian_product(indices0, indices1);
    }

    template <typename Index>
    static constexpr auto index_to_pos(Index const &i) {
//        static_assert(std::is_integral_v<Index>);
        return Dim2(std::get<0>(i), std::get<1>(i));
    }
};