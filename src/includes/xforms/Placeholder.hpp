#pragma once

#include <boost/yap/expression.hpp>
#include <iostream>
#include "Utils.hpp"

struct temp_placeholder_tag;

template <typename T>
constexpr bool is_temp = is_a<temp_placeholder_tag, T>;

// A placeholder for temporary variables
template <long long I>
struct temp_placeholder : boost::hana::llong<I> {
    using tag = temp_placeholder_tag;

    static constexpr long long id = I;

    friend std::ostream& operator<< (std::ostream& os, const temp_placeholder<I> &p) {
        os << "temp" << I;
        return os;
    }
};

auto const _0 = boost::yap::make_terminal(temp_placeholder<0>{});
auto const _1 = boost::yap::make_terminal(temp_placeholder<1>{});
auto const _2 = boost::yap::make_terminal(temp_placeholder<1>{});

