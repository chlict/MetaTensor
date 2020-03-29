#pragma once

#include <boost/hana.hpp>

// An alias type with hana::integral_constant<long long>. This will make the names appeared in gdb shorter.
template <int64_t v>
struct i64 : boost::hana::llong<v> {};

template <char ...c>
constexpr auto operator"" _c() {
    return i64<boost::hana::ic_detail::parse<sizeof...(c)>({c...})>{};
}
