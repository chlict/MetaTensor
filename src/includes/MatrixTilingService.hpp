#pragma once

#include <range/v3/all.hpp>
#include "AbstractTilingService.hpp"
#include "TRange.hpp"

template <typename TRangeRow, typename TRangeCol>
struct RowMajorTilingService : AbstractTilingService<
        RowMajorTilingService<TRangeRow, TRangeCol> > {
    static_assert(is_a<trange_tag, TRangeRow> && is_a<trange_tag, TRangeCol>);

    TRangeRow const trange_row_;
    TRangeCol const trange_col_;

    constexpr RowMajorTilingService(TRangeRow const &trange_row, TRangeCol const &trange_col) :
        trange_row_(trange_row), trange_col_(trange_col) {}

    constexpr RowMajorTilingService(RowMajorTilingService const &other) :
            trange_row_(other.trange_row_), trange_col_(other.trange_col_) {}

    constexpr RowMajorTilingService(RowMajorTilingService &&other) noexcept :
            trange_row_(other.trange_row_), trange_col_(other.trange_col_) {}

    template <typename Tensor>
    constexpr auto gen_tiling_indices_for(Tensor const &tensor) const {
        static_assert(is_tensor_type<Tensor>);
        auto tensor_shape = tensor.shape();
        static_assert(decltype(tensor_shape)::nDims == 2);

        namespace views = ranges::views;
        // Given a row major tiling:
        // range[0]: [begin: 0, end: 6, step: 2, size: 2]
        // range[1]: [begin: 0, end: 16, step: 4, size: 4]
        // generates a sequence of [
        // [0, 0], [0, 4], [0, 8], [0, 12],
        // [2, 0], [2, 4], [2, 8], [2, 12],
        // [4, 0], [4, 4], [4, 8], [4, 12],
        auto trange0 = trange_row_;
        auto b0 = trange0.begin();
        auto e0 = trange0.end();
        auto step0 = trange0.step();
        auto count0 = int_ceil(e0 - b0, step0);

        auto trange1 = trange_col_;
        auto b1 = trange1.begin();
        auto e1 = trange1.end();
        auto step1 = trange1.step();
        auto count1 = int_ceil(e1 - b1, step1);

        int bi0 = (int)b0, bi1 = (int)b1;
        int step_i0 = (int)step0, step_i1 = (int)step1;
        auto indices0 = views::ints(0, (int)count0) |
                views::transform([bi0, step_i0](int i) { return bi0 + i * step_i0; });
        auto indices1 = views::ints(0, (int)count1) |
                views::transform([bi1, step_i1](int i) { return bi1 + i * step_i1; });
        return views::cartesian_product(indices0, indices1);
    }

    template <typename Index>
    constexpr auto index_to_pos(Index const &i) const {
        return Dim2(std::get<0>(i), std::get<1>(i));
    }

    constexpr auto gen_tile_shape() const {
        return Dim2(trange_row_.size(), trange_col_.size());
    }

    friend std::ostream& operator << (std::ostream &os, RowMajorTilingService const service) {
        os << "Row major tiling: row" << service.trange_row_ << ", col" << service.trange_col_;
        return os;
    }
};