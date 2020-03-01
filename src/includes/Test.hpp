#pragma once

template<typename T>
struct A {
    T data;

    constexpr A(const T &data) : data(data) {}

    constexpr A(T &&data) : data(data) {}

    constexpr T getData() {
        return data;
    }
};
