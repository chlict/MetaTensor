#pragma once

#include <cxxabi.h>
#include <cstdio>
#include <cstdlib>

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
