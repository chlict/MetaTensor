#include "Test.hpp"

#include <boost/hana.hpp>

#include "gtest/gtest.h"

using namespace boost::hana::literals;

TEST(Suite1, Test1) { EXPECT_EQ(1, 1); }

TEST(Suite1, Test2) {
  auto a = A(1_c);

  static_assert(a.data == 1_c);
  static_assert(a.getData() == 1_c);

  EXPECT_EQ(a.getData(), 1_c);
}

template <typename T>
struct B {
  T x;

  constexpr B(T x) : x(x) {}

  constexpr B(const B &other) : x(other.x) {}

  constexpr T getX() { return x; }
};

TEST(Suite1, Test3) {
  auto a = A(B(1_c));
  static_assert(a.data.x == 1_c);
  static_assert(a.getData().x == 1_c);

  auto b = B(2_c);
  static_assert(b.x == 2_c);
  static_assert(b.getX() == 2_c);
  static_assert(A(b).data.x == 2_c);
  static_assert(A(b).getData().x == 2_c);
}

TEST(Suite1, Test4) {
  auto b = B(boost::hana::make_tuple(2_c));
  static_assert(b.x[0_c] == 2_c);

  auto x = b.getX();
  std::cout << x[0_c] << std::endl;
  static_assert(x[0_c] == 2_c);
  //    static_assert(b.getX()[0_c] == 2_c);
  //    static_assert(b.getX()[0_c] == 2_c);
  //    static_assert(A(b).data.x == 2_c);
  //    static_assert(A(b).getData().x == 2_c);
}
