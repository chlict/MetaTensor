#include <Tensor.hpp>
#include "gtest/gtest.h"
#include "TTiling.hpp"
#include "VectorFormat.hpp"
#include "TilingProvider.hpp"

TEST(TestTilingProvider, Test1) {
    auto format = make_format(Dim1(4_c), VectorLayout());
    auto tensor = Tensor(float(), format, MemSpace::GM(), 0x1000);

    auto tiling1d = Tiling1D(TRange(0_c, 4_c, 2_c, 1_c));
    // std::cout << tiling1d << std::endl;

    auto provider = VectorTilingProvider();
    auto indicies = provider.gen_tiling_indices(tensor, tiling1d);
    for (auto i : indicies) {
        printf("i = %d\n", i);
    }

    for (auto iter = indicies.begin(); iter != indicies.end(); iter++) {
        printf("i = %d\n", *iter);
    }
}
