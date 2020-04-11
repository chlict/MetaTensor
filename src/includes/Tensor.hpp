#pragma once

#include "Utils.hpp"
#include "TensorFormat.hpp"
#include "MatrixFormat.hpp"
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

template <typename T>
constexpr bool is_tensor_type = is_a<tensor_tag, T>;

template<typename ElemType, typename Format, typename Space, typename Addr>
struct Tensor : TensorHandle {
    using tag = tensor_tag;

    static_assert(is_a<mem_space_tag, Space> &&
            is_a<tensor_format_tag, Format> &&
            is_integral_or_constant<Addr>);

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

    template <typename Pos, typename SliceView>
    constexpr auto slice(Pos &&pos, SliceView &&slice_view) const {
        using view_type = typename format_traits<Format>::view_type;
        static_assert(is_dims_type<Pos> && is_dims_type<SliceView>);
        static_assert(Pos::nDims == view_type::nDims && SliceView::nDims == view_type::nDims);
        // assert(pos within view && slice_view within view);

        // layout is same as original layout
        using layout_provider_type = typename format_traits<Format>::layout_provider_type;
        auto slice_format = make_format(std::forward<SliceView>(slice_view), layout_provider_type());

//        auto slice_addr = addr() + format_.offset(pos);
    }

    friend std::ostream& operator << (std::ostream &os, Tensor tensor) {
        os << "Tensor@0x" << std::hex << tensor.addr() << std::dec;
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

