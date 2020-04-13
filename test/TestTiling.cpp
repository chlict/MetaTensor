#include <Tensor.hpp>
#include <TilingProvider.hpp>
#include "gtest/gtest.h"
#include "TTiling.hpp"
#include "VectorFormat.hpp"

TEST(TestTiling, Test1) {
    auto range1 = TRange(0_c, 4_c, 2_c, 1_c);
    // std::cout << range1 << std::endl;
    auto size1 = range1.size();
    static_assert(size1 == 1_c);

    auto range2 = TRange(0_c, 4_c, 2_c);
    // std::cout << range2 << std::endl;
    auto size2 = range2.size();
    static_assert(size2 == 2_c);

    static_assert(is_a<trange_tag>(range1) && is_a<trange_tag>(range2));
}

TEST(TestTiling, Test2) {
    auto order = Dims(0_c);
    auto range = TRange(0_c, 4_c, 2_c, 1_c);
    auto tiling = TTiling(order, range);
    std::cout << tiling << std::endl;

    auto tiling1d = Tiling1D(range);
    std::cout << tiling1d << std::endl;

    auto range_row = TRange(0_c, 4_c, 2_c);
    auto range_col = TRange(0_c, 8_c, 2_c);
    auto tiling2d_row_major = Tiling2DRowMajor(range_row, range_col);
    std::cout << tiling2d_row_major;

    auto tiling2d_col_major = Tiling2DColMajor(range_row, range_col);
    std::cout << tiling2d_col_major;
}

TEST(TestTiling, Test3) {
    auto format = make_format(Dim1(4_c), VectorLayout());
    auto tensor = Tensor(float(), format, MemSpace::GM(), 0x1000);

    auto tiling1d = Tiling1D(TRange(0_c, 4_c, 2_c, 1_c));
    std::cout << tiling1d << std::endl;

    using namespace boost::yap::literals;
    auto expr = 1_p + 2_p;
    tiling1d.apply(tensor, expr);
}

// Zero-cost expected
TEST(TestTiling, Test4) {
    auto format = make_format(Dim1(100_c), VectorLayout());
    auto tensor = Tensor(float(), format, MemSpace::GM(), 0x1000);

    auto tiling = Tiling1D(TRange(0_c, 100_c, 2_c, 1_c));
    auto tp = VectorTilingProvider();

    auto indices = tp.gen_tiling_indices(tensor, tiling);
    for (auto i : indices) {
        auto pos = tp.index_to_pos(i);
        auto shape = Dim1(1_c);
        auto tile = tensor.get_tile(pos, shape);
//        std::cout << tile << std::endl;
        assert(tile.addr() == 0x1000 + sizeof(float) * pos.dim[0_c]);
    }
}