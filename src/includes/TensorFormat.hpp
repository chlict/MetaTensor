#pragma once

#include "Literals.hpp"
#include "TensorLayout.hpp"
#include <type_traits>

enum Formats {
    MATRIX_ROW_MAJOR,    // row major stored matrix
    MATRIX_COL_MAJOR,    // column major stored matrix
    CUSTOMIZED
};

struct TensorFormatHandle {
};

// View represents a logical shape. E.g. a fractal tensor may have a view of Dim2(width, height) like a matrix
// but the underlying layout has a Dim4 shape.
template<typename View, typename Layout>
struct TensorFormat : public TensorFormatHandle {
    View view;
    Layout layout;
    Formats tag;

    constexpr TensorFormat(Formats tag, View view, Layout layout) :
        view(view), layout(layout), tag(tag) {}

    constexpr TensorFormat(TensorFormat const &other) noexcept :
            view(other.view),
            layout(other.layout) {}

    constexpr TensorFormat(TensorFormat &&other) noexcept :
            view(std::forward<TensorFormat>(other.view),
                 layout(std::forward<TensorFormat>(other.layout))) {}

    constexpr auto getView() const {
        return view;
    }

    constexpr auto getLayout() const {
        return layout;
    }

    constexpr auto getTag() const {
        return tag;
    }

};

// Must be specialized before use
template <Formats tag>
struct AutoLayout {
    template <typename View>
    constexpr auto operator()(View &&view) const = delete;
};

template <Formats tag>
AutoLayout<tag> auto_layout;

#include "MatrixFormats.hpp"

template<Formats tag, typename View, typename LayoutProvider = AutoLayout<tag>>
        // TODO: View &&view
constexpr auto make_format(View view, LayoutProvider layout_provider = auto_layout<tag>) {
    auto layout = layout_provider(static_cast<View &&>(view));
    return TensorFormat<View, decltype(layout)>(tag, view, layout);
}

//template<Formats tag, typename Dim0, typename Dim1, typename LayoutProvider = AutoLayout<tag>
//        typename t1 = std::enable_if_t<std::is_integral<Dim0>::value>>
//constexpr auto make_format(Dim0 dim0, Dim1 dim1, LayoutProvider layout_provider = auto_layout<tag>) {
//    return make_format<tag>(Dims(dim0, dim1), layout_provider);
//}


