//
// Created by chenlong on 2020/2/23.
//

#ifndef DEMO2_CALC_HPP
#define DEMO2_CALC_HPP

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


#endif //DEMO2_CALC_HPP
