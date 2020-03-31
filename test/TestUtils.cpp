#include "gtest/gtest.h"
#include "TensorFormat.hpp"
#include "MatrixFormats.hpp"

TEST(TestUtils, Test1) {
    auto dim2 = Dims(2, 4);
    auto format = make_format(dim2, RowMajorLayout());
    using T = decltype(format);
    static_assert(std::is_same_v<T::tag, tensor_format_tag>);
    static_assert(is_a_t<tensor_format_tag, T>::value);
    static_assert(is_a<tensor_format_tag>(format));
}