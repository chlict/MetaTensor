#include "gtest/gtest.h"
#include "Literals.hpp"
#include <iostream>

TEST(TestLiterals, Test1) {
    auto x = 1_c;
    std::cout << x << std::endl;
}