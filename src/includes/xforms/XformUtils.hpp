#pragma once

#include <iostream>
#include "Utils.hpp"
#include "xforms/PrintIR.hpp"

namespace yap = boost::yap;
namespace hana = boost::hana;

struct xform_pass_tag;

// Transform at compile time
struct StaticTransform {};

// Trsnsform at runtime
struct DynamicTransform {};

struct DumpFlag {
    struct ON  {};
    struct OFF {};
};

template <typename Dumping>
constexpr bool need_dump(Dumping dumping) {
#ifdef NDEBUG  // Release build
    return false;
#else
    return std::is_same_v<Dumping, DumpFlag::ON>;
#endif
}

// Not sure whether it introduces extra cost
template <typename Flag, typename Fn>
constexpr void on_dump(Flag flag, Fn const& fn) {
    if constexpr (need_dump(flag)) {
        fn();
    }
}

template <typename T>
struct is_xform_pass_t : std::integral_constant<bool, is_a<xform_pass_tag, T> > {};

template <typename T>
constexpr bool is_xform_pass = is_xform_pass_t<T>::value;

template <typename ...T>
struct all_xform_passes_t {
    static constexpr auto types = boost::hana::tuple_t<T...>;
    static constexpr auto value = boost::hana::all_of(types, [](auto &&elem) {
        using t = typename std::remove_reference_t<decltype(elem)>::type;
        return is_xform_pass<t>;
    });
};

template <typename ...T>
constexpr bool all_xform_passes = all_xform_passes_t<T...>::value;

template <typename IRList>
constexpr auto print_ir_list(IRList &&irlist) {
    static_assert(hana::is_a<hana::tuple_tag, IRList>);
    hana::for_each(irlist, [](auto &&ir) {
        boost::yap::print(std::cout, ir);
    });
}

// Skip printing the long type name
template <typename IRList>
constexpr auto print_ir_list_simple(IRList &&irlist) {
    static_assert(hana::is_a<hana::tuple_tag, IRList>);
    hana::for_each(irlist, [](auto &&ir) {
        boost::yap2::print_ir(std::cout, ir);
    });
}
