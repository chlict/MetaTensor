#include <Tensor.hpp>
#include "gtest/gtest.h"
#include "TTiling.hpp"
#include "VectorFormat.hpp"
#include "TilingProvider.hpp"
#include "MatrixTilingProvider.hpp"

// Zero-cost expected
TEST(TestTilingProvider, Test1) {
    auto format = make_format(Dim1(10_c), VectorLayout());
    auto tensor = Tensor(float(), format, MemSpace::GM(), 0x1000);
    auto provider = VectorTilingProvider();

    {
        auto tiling1d = Tiling1D(TRange(0_c, 4_c, 2_c, 1_c));
        auto indices = provider.gen_tiling_indices(tensor, tiling1d);
        assert(indices[0] == 0 && indices[1] == 2);
    }
    {
        auto tiling1d = Tiling1D(TRange(2_c, 10_c, 2_c, 1_c));
        auto indices = provider.gen_tiling_indices(tensor, tiling1d);
        assert(indices.size() == 4 && indices[0] == 2 && indices[1] == 4 && indices[2] == 6 && indices[3] == 8);
    }
    {
        auto tiling1d = Tiling1D(TRange(2_c, 10_c, 3_c, 1_c));
        auto indices = provider.gen_tiling_indices(tensor, tiling1d);
        assert(indices.size() == 3 && indices[0] == 2 && indices[1] == 5 && indices[2] == 8);
    }
//    for (auto i : indices) {
//        printf("i = %d\n", i);
//    }
//
//    for (auto iter = indices.begin(); iter != indices.end(); iter++) {
//        printf("i = %d\n", *iter);
//    }
}

TEST(TestTilingProvider, Test2) {
    auto format = make_format(Dim2(4_c, 6_c), RowMajorLayout());
    auto tensor = Tensor(float(), format, MemSpace::GM(), 0x1000);
    auto provider = RowMajorTiling();

    {
        auto tiling = Tiling2DRowMajor(TRange(0_c, 4_c, 2_c), TRange(0_c, 6_c, 2_c));
        auto indices = provider.gen_tiling_indices(tensor, tiling);
        for (auto i : indices) {
            printf("[%d, %d]\n", std::get<0>(i), std::get<1>(i));
        }
    }
//    {
//        auto tiling1d = Tiling1D(TRange(2_c, 10_c, 2_c, 1_c));
//        auto indices = provider.gen_tiling_indices(tensor, tiling1d);
//        assert(indices.size() == 4 && indices[0] == 2 && indices[1] == 4 && indices[2] == 6 && indices[3] == 8);
//    }
//    {
//        auto tiling1d = Tiling1D(TRange(2_c, 10_c, 3_c, 1_c));
//        auto indices = provider.gen_tiling_indices(tensor, tiling1d);
//        assert(indices.size() == 3 && indices[0] == 2 && indices[1] == 5 && indices[2] == 8);
//    }

//
//    for (auto iter = indices.begin(); iter != indices.end(); iter++) {
//        printf("i = %d\n", *iter);
//    }
}
