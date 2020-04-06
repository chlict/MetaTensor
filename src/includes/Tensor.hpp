#pragma once

#include "Utils.hpp"
#include "TensorFormat.hpp"
#include "MatrixFormats.hpp"
#include <boost/yap/expression.hpp>
#include <iostream>


struct mem_space_tag;

struct MemSpace {
    struct GM { using tag = mem_space_tag; };
    struct L1 { using tag = mem_space_tag; };
};

struct TensorHandle {
    friend std::ostream& operator << (std::ostream &os, TensorHandle const &handle) {
        os << "TensorHandle";
        return os;
    }
};

struct tensor_tag;

template<typename ElemType, typename Format, typename Space, typename Addr,
        typename = std::enable_if_t<
                is_a<mem_space_tag, Space> &&
                is_a<tensor_format_tag, Format> &&
                is_integral_or_constant<Addr>,
                void> >
struct Tensor : TensorHandle {
    using tag = tensor_tag;

    const Format format_;
    const Addr addr_;

    constexpr Tensor(ElemType const &type, Format const &format, Space const &space, Addr const &addr) :
            format_(format), addr_(addr) {}

    constexpr Tensor(Tensor const &other) noexcept : format_(other.format_), addr_(other.addr_) {}

    constexpr Tensor(Tensor &&other) noexcept : format_(other.format_), addr_(other.addr_) {}

    constexpr auto format() const { return format_; }

    constexpr auto addr() const { return addr_; }

    constexpr auto view() const { return format_.view(); }

    constexpr auto layout() const { return format_.layout(); }

    constexpr auto shape() const { return format_.shape(); }

    constexpr auto strides() const { return format_.strides(); }

    friend std::ostream& operator << (std::ostream &os, Tensor tensor) {
        os << "Tensor@" << tensor.addr();
        return os;
    }
};

// Tensor expression - a boost::yap's terminal of Tensor
template<typename ElemType, typename Format, typename Space, typename Addr,
        typename = std::enable_if_t<
                is_a<mem_space_tag, Space> &&
                is_a<tensor_format_tag, Format> &&
                is_integral_or_constant<Addr>,
                void> >
constexpr auto TensorE(ElemType const &type, Format const &format, Space const &space, Addr const &addr) {
    auto tensor = Tensor(type, format, space, addr);
    return boost::yap::make_terminal(std::move(tensor));
}

template<typename T>
struct tensor_traits;

template<typename ElemType, typename Format, typename Space, typename Addr>
struct tensor_traits<Tensor<ElemType, Format, Space, Addr>> {
    using space = Space;
    using elem_type = ElemType;
};

template <typename T>
struct is_tensor_t : std::integral_constant<bool, is_a<tensor_tag, T> >
{};

template <typename T>
constexpr bool is_tensor = is_tensor_t<T>::value;
