#pragma once

#include "Literals.hpp"
#include "TensorLayout.hpp"

enum Format {
    MatRowMajor,    // row major stored matrix
    MatColMajor,    // column major stored matrix
    Customized
};

struct AutoLayout {};

struct TensorFormatHandle {
};

// View represents a logical shape_. E.g. a fractal tensor may have a view of Dim2(width, height)
// like a matrix, but the underlying layout has a Dim4 shape.
template<Format Tag, typename View, typename Layout>
struct TensorFormat : public TensorFormatHandle {
    View view_;
    Layout layout_;

    constexpr TensorFormat(View view, Layout layout) : view_(view), layout_(layout) {}

    constexpr TensorFormat(TensorFormat const &other) noexcept :
        view_(other.view_),
        layout_(other.layout_) {}

    constexpr TensorFormat(TensorFormat &&other) noexcept :
        view_(std::forward<TensorFormat>(other.view_),
        layout_(std::forward<TensorFormat>(other.layout_))) {}

    constexpr auto get_view() const {
        return view_;
    }

    constexpr auto get_layout() const {
        return layout_;
    }
};

template<Format tag, typename View, typename Layout = AutoLayout>
constexpr auto MakeFormat(View view, Layout layout = Layout()) {
    if constexpr (tag == Format::MatRowMajor) {
        static_assert(std::is_same_v<Layout, AutoLayout>);
        auto shape = Dims(view.dim[0_c], view.dim[1_c]);
        auto stride = Dims(view.dim[0_c] * view.dim[1_c], 1_c);
        auto real_layout = TensorLayout(shape, stride);
        return TensorFormat<Format::MatRowMajor, View, decltype(real_layout)>(view, real_layout);
    } else if constexpr (tag == Format::Customized) {
        return TensorFormat<tag, View, Layout>(view, layout);
    }
}

