#pragma once

#include "Literals.hpp"
#include "TensorLayout.hpp"
#include "LayoutProvider.hpp"
#include "Utils.hpp"
#include <type_traits>

struct tensor_format_tag {};

// View represents a logical shape. E.g. a fractal tensor may have a view of Dim2(width, height) like a matrix
// but the underlying layout has a Dim4 shape.
// LayoutProvider can be retrieved from format_traits
template<typename View, typename Layout, typename LayoutProvider>
struct TensorFormat {
    static_assert(is_dims_type<View> && is_layout_type<Layout>);

    using tag = tensor_format_tag;

    View view_;
    Layout layout_;

    constexpr TensorFormat(View const &view, Layout const &layout) :
            view_(view), layout_(layout) {}

    constexpr TensorFormat(TensorFormat const &other) noexcept :
            view_(other.view_),
            layout_(other.layout_) {}

    constexpr TensorFormat(TensorFormat &&other) noexcept :
            view_(other.view_),
            layout_(other.layout_) {}

    constexpr auto view() const {
        return view_;
    }

    constexpr auto layout() const {
        return layout_;
    }

    constexpr auto shape() const {
        return layout_.shape();
    }

    constexpr auto strides() const {
        return layout_.strides();
    }
};

// TODO: View &&view
template <typename View, typename LayoutProvider,
        typename = std::enable_if_t<
                is_a<layout_provider_tag, LayoutProvider>,
                void> >
constexpr auto make_format(View view, AbstractLayoutProvider<LayoutProvider> const &layout_provider) {
    auto layout = layout_provider(view);
    return TensorFormat<View, decltype(layout), LayoutProvider>(view, layout);
}

template <typename Dim0, typename Dim1, typename LayoutProvider,
        typename = std::enable_if_t<
                is_integral_or_constant<Dim0> &&
                is_integral_or_constant<Dim1> &&
                is_a<layout_provider_tag, LayoutProvider>,
                void> >
constexpr auto make_format(Dim0 dim0, Dim1 dim1, AbstractLayoutProvider<LayoutProvider> const &layout_provider) {
    return make_format(Dim2(dim0, dim1), layout_provider);
}

template <typename T>
struct format_traits;

template <typename View, typename Layout, typename LayoutProvider>
struct format_traits<TensorFormat<View, Layout, LayoutProvider>> {
    using view_type = View;
    using layout_type = Layout;
    using layout_provider_type = LayoutProvider;
};
