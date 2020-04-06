#pragma once

#include "Utils.hpp"
#include <iostream>

struct xform_pass_tag;

// Transform at compile time
struct StaticTransform {};

// Trsnsform at runtime
struct DynamicTransform {};

template <typename T>
struct is_xform_pass_t : std::integral_constant<bool, is_a<xform_pass_tag, T> > {};

template <typename T>
constexpr bool is_xform_pass = is_xform_pass_t<T>::value;
