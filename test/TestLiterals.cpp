#include <iostream>

#include "Literals.hpp"
#include "Utils.hpp"
#include "gtest/gtest.h"

TEST(TestLiterals, Test1) {
  auto x = 1_c;
  std::cout << x << std::endl;
  print_type_name(x);
}