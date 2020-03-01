#pragma once

#include <boost/hana.hpp>

// An alias type with hana::integral_constant<long long>. This will make the names appeared in gdb shorter.
template <long long v>
struct ic : boost::hana::llong<v> {};

template <char ...c>
constexpr auto operator"" _c() {
    return ic<boost::hana::ic_detail::parse<sizeof...(c)>({c...})>{};
}
