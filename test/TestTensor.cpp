#include "gtest/gtest.h"
#include "Tensor.hpp"

TEST(TestTensor, Test1) {
    auto format1 = make_format(Dims(2, 4), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);

    using T = decltype(tensor1);
    static_assert(std::is_same_v<tensor_traits<T>::elem_type, float>);
    static_assert(std::is_same_v<tensor_traits<T>::space, MemSpace::GM>);
}

auto fn_copy = [](auto &&tensor) { return tensor; };

TEST(TestTensor, Test2) {
    auto format1 = make_format(Dims(2_c, 4_c), RowMajorLayout());
    auto tensor1 = Tensor(float(), format1, MemSpace::GM(), 0x1000);
    auto tensor2 = fn_copy(tensor1);

    using T = decltype(tensor2);
    static_assert(std::is_same_v<tensor_traits<T>::elem_type, float>);
    static_assert(std::is_same_v<tensor_traits<T>::space, MemSpace::GM>);

    auto format2 = tensor2.format();
    auto view = format2.view();
    auto layout = format2.layout();
    static_assert(view.dim[0_c] == 2_c);
    static_assert(layout.shape_.dim[0_c] == 4_c);
}
