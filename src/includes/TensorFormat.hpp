#pragma once

#include "Literals.hpp"
#include "TensorLayout.hpp"
#include "LayoutProvider.hpp"
#include "Utils.hpp"
#include <type_traits>

struct tensor_format_tag {};

// View represents a logical shape. E.g. a fractal tensor may have a view of Dim2(width, height) like a matrix
// but the underlying layout has a Dim4 shape.
template<typename View, typename Layout>
struct TensorFormat {
    View view;
    Layout layout;

    using tag = tensor_format_tag;

    constexpr TensorFormat(View view, Layout layout) :
        view(view), layout(layout) {}

    constexpr TensorFormat(TensorFormat const &other) noexcept :
            view(other.view),
            layout(other.layout) {}

    constexpr TensorFormat(TensorFormat &&other) noexcept :
            view(std::forward<TensorFormat>(other.view),
            layout(std::forward<TensorFormat>(other.layout))) {}

    constexpr auto get_view() const {
        return view;
    }

    constexpr auto get_layout() const {
        return layout;
    }
};

// TODO: View &&view
template <typename View, typename LayoutProvider,
        typename = std::enable_if_t<
                is_a<layout_provider_tag, LayoutProvider>,
                void> >
constexpr auto make_format(View view, AbstractLayoutProvider<LayoutProvider> const &layout_provider) {
    auto layout = layout_provider(view);
    return TensorFormat<View, decltype(layout)>(view, layout);
}

template <typename Dim0, typename Dim1, typename LayoutProvider,
        typename = std::enable_if_t<
                is_integral_or_constant<Dim0> &&
                is_integral_or_constant<Dim1> &&
                is_a<layout_provider_tag, LayoutProvider>,
                void> >
constexpr auto make_format(Dim0 dim0, Dim1 dim1, AbstractLayoutProvider<LayoutProvider> const &layout_provider) {
    return make_format(Dims(dim0, dim1), layout_provider);
}


