#include "calc.hpp"
#include <boost/hana.hpp>
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    auto a = calc();
    a.print();
    auto b = boost::hana::make_tuple(1, 2);
    return 0;
}
