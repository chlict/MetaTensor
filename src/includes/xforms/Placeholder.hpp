#pragma once

#include <boost/yap/expression.hpp>
#include <iostream>

namespace yap = boost::yap;
namespace hana = boost::hana;

// A placeholder for temporary variables
template <long long I>
struct temp_placeholder : boost::hana::llong<I> {
    friend std::ostream& operator<< (std::ostream& os, const temp_placeholder<I> &p) {
        os << "temp" << I;
        return os;
    }
};

auto const _0 = yap::make_terminal(temp_placeholder<0>{});
auto const _1 = yap::make_terminal(temp_placeholder<1>{});
auto const _2 = yap::make_terminal(temp_placeholder<2>{});
auto const _3 = yap::make_terminal(temp_placeholder<3>{});
auto const _4 = yap::make_terminal(temp_placeholder<4>{});
