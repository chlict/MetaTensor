#pragma once

#include "Literals.hpp"
#include "TensorLayout.hpp"
#include "LayoutProvider.hpp"
#include "Utils.hpp"
#include <type_traits>

// View represents a logical shape. E.g. a fractal tensor may have a view of Dim2(width, height) like a matrix
// but the underlying layout has a Dim4 shape.
template<typename View, typename Layout>
struct TensorFormat {
    View view;
    Layout layout;

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

//template<Formats tag, typename Dim0, typename Dim1, typename LayoutProvider = AutoLayout<tag>
//        typename t1 = std::enable_if_t<std::is_integral<Dim0>::value>>
//constexpr auto make_format(Dim0 dim0, Dim1 dim1, LayoutProvider layout_provider = auto_layout<tag>) {
//    return make_format<tag>(Dims(dim0, dim1), layout_provider);
//}

// TODO: View &&view
// TODO: static_assert(LayoutProvider is sub-class of AbstractLayoutProvider<LayoutProvider>)
template <typename View, typename LayoutProvider>
constexpr auto make_format(View view, AbstractLayoutProvider<LayoutProvider> const &layout_provider) {
    auto layout = layout_provider(view);
    return TensorFormat<View, decltype(layout)>(view, layout);
}


