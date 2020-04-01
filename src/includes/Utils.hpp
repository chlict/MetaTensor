#pragma once

#include <cxxabi.h>
#include <cstdio>
#include <cstdlib>
#include <boost/hana.hpp>

// Usage:
// 1) static_assert(is_a_t<tag, T>::value)
// 2) static_assert(is_a<tag>(variable))
template <typename Tag, typename ...Types>
struct is_a_t;

template <typename Tag, typename ...Types>
constexpr is_a_t<Tag, Types...> is_a{};


template <typename Tag, typename Type>
struct is_a_t<Tag, Type> : std::integral_constant<bool,
        std::is_same<typename Type::tag, Tag>::value
        >
{};

template <typename Tag>
struct is_a_t<Tag> {
    template <typename Type>
    constexpr auto operator()(Type const&) const
    { return is_a_t<Tag, Type>::value; }
};

// for debug
template <typename T>
void print_type_name() {
    char *name = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
    printf("\"%s\"\n", name);
    ::free(name);
}

template <typename T>
void print_type_name(T var) {
    print_type_name<T>();
}

// check a given type is either runtime integral type or constant integral type
template <typename T>
struct is_integral_or_constant_t : std::integral_constant<bool,
        std::is_integral_v<T> || boost::hana::is_a<boost::hana::integral_constant_tag<long long>, T> >
{};

template <typename T>
constexpr bool is_integral_or_constant = is_integral_or_constant_t<T>::value;