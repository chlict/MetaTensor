#pragma once

// Use the CRTP trick to do static polymorphism. Each layout type should inherit this struct
// and provide methods to do the actual layout.
struct layout_provider_tag {};

template <typename Provider>
struct AbstractLayoutProvider {
    using tag = layout_provider_tag;

    template <typename View>
    constexpr auto operator()(View&& view) const {
        const Provider *provider = static_cast<const Provider *>(this);
        return (*provider)(view);
    }

    template <typename View>
    constexpr auto view_to_shape(View &&view) const {
        const Provider *provider = static_cast<const Provider *>(this);
        return provider->view_to_shape(view);
    }

    template <typename Shape>
    constexpr auto shape_to_strides(Shape &&shape) const {
        const Provider *provider = static_cast<const Provider *>(this);
        return provider->shape_to_strides(shape);
    }
};