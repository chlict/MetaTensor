#pragma once

#include <boost/hana.hpp>
#include <iostream>

class calc {
public:
    void print() {
        auto x = boost::hana::make_tuple(1, 2);
        std::cout << boost::hana::length(x);
        std::cout << "calc\n";
    }
};

