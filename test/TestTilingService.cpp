#include <Tensor.hpp>

#include "TTiling.hpp"
#include "TilingService.hpp"
#include "VectorFormat.hpp"
#include "gtest/gtest.h"

// Zero-cost expected
TEST(TestTilingService, Test1) {
  auto format = make_format(Dim1(10_c), VectorLayout());
  auto tensor = Tensor(float(), format, MemSpace::GM(), 0x1000);

  {
    auto ts = VectorTilingService(TRange(0_c, 4_c, 2_c, 1_c));
    auto indices = ts.gen_tiling_indices_for(tensor);
    assert(indices[0] == 0 && indices[1] == 2);
  }
  {
    auto ts = VectorTilingService(TRange(2_c, 10_c, 2_c, 1_c));
    auto indices = ts.gen_tiling_indices_for(tensor);
    assert(indices.size() == 4 && indices[0] == 2 && indices[1] == 4 &&
           indices[2] == 6 && indices[3] == 8);
  }
  {
    auto ts = VectorTilingService(TRange(2_c, 10_c, 3_c, 1_c));
    auto indices = ts.gen_tiling_indices_for(tensor);
    assert(indices.size() == 3 && indices[0] == 2 && indices[1] == 5 &&
           indices[2] == 8);
  }
  //    for (auto i : indices) {
  //        printf("i = %d\n", i);
  //    }
  //
  //    for (auto iter = indices.begin(); iter != indices.end(); iter++) {
  //        printf("i = %d\n", *iter);
  //    }
}

// Zero-cost expected
TEST(TestTilingService, Test2) {
  auto format = make_format(Dim2(4_c, 6_c), RowMajorLayout());
  auto tensor = Tensor(float(), format, MemSpace::GM(), 0x1000);

  {
    auto t_service =
        RowMajorTilingService(TRange(0_c, 4_c, 2_c), TRange(0_c, 6_c, 2_c));
    auto indices = t_service.gen_tiling_indices_for(tensor);
    //        for (auto i : indices) {
    //            printf("[%d, %d]\n", std::get<0>(i), std::get<1>(i));
    //        }
    assert(std::get<0>(indices[0]) == 0 && std::get<1>(indices[0]) == 0);
    assert(std::get<0>(indices[1]) == 0 && std::get<1>(indices[1]) == 2);
    assert(std::get<0>(indices[2]) == 0 && std::get<1>(indices[2]) == 4);
    assert(std::get<0>(indices[3]) == 2 && std::get<1>(indices[3]) == 0);
    assert(std::get<0>(indices[4]) == 2 && std::get<1>(indices[4]) == 2);
    assert(std::get<0>(indices[5]) == 2 && std::get<1>(indices[5]) == 4);
  }
}
