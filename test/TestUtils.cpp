#include "gtest/gtest.h"
#include "TensorFormat.hpp"
#include "MatrixFormat.hpp"

TEST(TestUtils, Test1) {
    auto dim2 = Dims(2, 4);
    auto format = make_format(dim2, RowMajorLayout());
    using T = decltype(format);
    static_assert(std::is_same_v<T::tag, tensor_format_tag>);
    static_assert(is_a_t<tensor_format_tag, T>::value);
    static_assert(is_a<tensor_format_tag>(format));
}

TEST(TestUtils, Test2) {
    static_assert(all_integral_or_constant_t<int, int>::value);
    static_assert(all_integral_or_constant<int, int>);
    static_assert(!all_integral_or_constant<float, int>);
    static_assert(all_integral_or_constant<i64<1>, i64<2>>);
}