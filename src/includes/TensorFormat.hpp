#pragma once

#include "Literals.hpp"
#include "TensorLayout.hpp"
#include "LayoutProvider.hpp"
#include "Utils.hpp"
#include <type_traits>

struct tensor_format_tag;

// Shape represents a logical shape. E.g. a fractal tensor may have a Dim2(width, height)
// shape like a matrix, but the underlying layout has a Dim4 shape (dimensions).
// LayoutProvider can be retrieved from format_traits
template<typename Shape, typename Layout, typename LayoutProvider>
struct TensorFormat {
    static_assert(is_dims_type<Shape> && is_layout_type<Layout>);

    using tag = tensor_format_tag;

    Shape shape_;
    Layout layout_;

    constexpr TensorFormat(Shape const& shape, Layout const& layout) :
            shape_(shape), layout_(layout) {}

    constexpr TensorFormat(Shape&& shape, Layout&& layout) :
            shape_(std::move(shape)), layout_(std::move(layout)) {}

    constexpr TensorFormat(TensorFormat const &other) noexcept :
            shape_(other.shape_),
            layout_(other.layout_) {}

    constexpr TensorFormat(TensorFormat &&other) noexcept :
            shape_(std::move(other.shape_)),
            layout_(std::move(other.layout_)) {}

    constexpr auto shape() const {
        return shape_;
    }

    constexpr auto layout() const {
        return layout_;
    }

    constexpr auto dimensions() const {
        return layout_.dimensions();
    }

    constexpr auto strides() const {
        return layout_.strides();
    }
};

// TODO: Shape &&shape
template <typename Shape, typename LayoutProvider,
        typename = std::enable_if_t<
                is_a<layout_provider_tag, LayoutProvider>,
                void> >
constexpr auto make_format(Shape shape, AbstractLayoutProvider<LayoutProvider> const &layout_provider) {
    auto layout = layout_provider(shape);
    return TensorFormat<Shape, decltype(layout), LayoutProvider>(shape, layout);
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

template <typename Shape, typename Layout, typename LayoutProvider>
struct format_traits<TensorFormat<Shape, Layout, LayoutProvider>> {
    using shape_type = Shape;
    using layout_type = Layout;
    using layout_provider_type = LayoutProvider;
};
