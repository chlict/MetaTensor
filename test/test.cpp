#include "gtest/gtest.h"
#include "calc.hpp"

TEST(Suite1, Test1) {
    EXPECT_EQ(1, 1);
}

TEST(Suite1, Test2) {
    auto x = calc();
    x.print();
    static_assert(boost::hana::length(boost::hana::make_tuple(1, 2)) == boost::hana::size_c<2>);
    EXPECT_EQ(0, 0);
}



