#pragma once

#include <boost/yap/expression.hpp>
#include <iostream>

#include "MatrixFormat.hpp"
#include "TensorFormat.hpp"
#include "Utils.hpp"

namespace mt {

struct mem_space_tag;

struct MemSpace {
  struct Host {
    using tag = mem_space_tag;
  };
  struct GM {
    using tag = mem_space_tag;
  };
  struct L1 {
    using tag = mem_space_tag;
  };
};

namespace detail {

const long ELEM_PER_LINE = 8;

template <int DIM, typename Tensor>
struct TensorHelper;

// Specilization for DIM 1
template <typename Tensor>
struct TensorHelper<1, Tensor> {
  static void dump(Tensor const &tensor) {
    // NOTE: do NOT use size_t or unsigned for i or it will compile with error
    // "not in the same ring"
    const long size = tensor.shape().dim[0_c];
    for (long i = 0; i < size; i++) {
      if (i % ELEM_PER_LINE == 0) std::cout << "\n";
      std::cout << " " << tensor.elem(Dims(i));
    }
    std::cout << "\n";
  }
};

// Specilization for DIM 2
template <typename Tensor>
struct TensorHelper<2, Tensor> {
  static void dump(Tensor const &tensor) {
    // Here we use standard loop rather hana:for_each() because
    // hana::for_each() might unroll the loop heavily. Compiler can also
    // unroll and vectorize this oridinary loop.
    const long size0 = tensor.shape().dim[0_c];
    const long size1 = tensor.shape().dim[1_c];
    for (long i0 = 0; i0 < size0; i0++) {
      for (long i1 = 0; i1 < size1; i1++) {
        if (i1 % ELEM_PER_LINE == 0) std::cout << "\n";
        std::cout << " " << tensor.elem(Dims(i0, i1));
      }
    }
    std::cout << "\n";
  }
};

}  // namespace detail

struct TensorHandle {
  friend std::ostream &operator<<(std::ostream &os,
                                  TensorHandle const &handle) {
    os << "TensorHandle";
    return os;
  }
};

struct tensor_tag;

template <typename T>
constexpr bool is_tensor_type = is_a<tensor_tag, T>;

template <typename ElemType, typename Format, typename Space = MemSpace::GM>
struct Tensor : TensorHandle {
  using tag = tensor_tag;
  static_assert(is_a<tensor_format_tag, Format>);
  static_assert(is_a<mem_space_tag, Space>);

 private:
  const Format format_;
  const ElemType *addr_;

 public:
  constexpr Tensor(ElemType *addr, Format const &format,
                   Space const &space = MemSpace::GM{})
      : format_(format), addr_(addr) {}

  constexpr Tensor(Tensor const &other) noexcept
      : format_(other.format_), addr_(other.addr_) {}

  constexpr Tensor(Tensor &&other) noexcept
      : format_(std::move(other.format_)), addr_(std::move(other.addr_)) {}

 public:
  constexpr auto format() const { return format_; }

  constexpr auto addr() const { return addr_; }

  constexpr auto shape() const { return format_.shape(); }

  constexpr auto layout() const { return format_.layout(); }

  // Get the address of an element at specified postion (coorination).
  template <typename Pos>
  constexpr ElemType *elem_addr(Pos const &pos) const {
    using Layout = typename format_traits<Format>::layout_provider_type;
    auto offset = Layout::offset(layout(), pos);
    return (ElemType *)addr() + (long)offset;
  }

  // Get an element at specified postion
  template <typename Pos>
  constexpr ElemType elem(Pos const &pos) const {
    return *(ElemType *)elem_addr(pos);
  }

  // Set an element at specified postion
  template <typename Pos>
  constexpr void set(Pos const &pos, ElemType const &elem) const {
    ElemType *addr = elem_addr(pos);
    *addr = elem;
  }

  template <typename Pos, typename TileShape>
  constexpr auto get_tile(Pos &&pos, TileShape &&tile_shape) const {
    get_tile_check<Format, Pos, TileShape>();
    // layout is same as original layout
    using LayoutType = typename format_traits<Format>::layout_provider_type;
    auto tile_format =
        make_format(std::forward<TileShape>(tile_shape), LayoutType());
    using TileFormat = decltype(tile_format);
    ElemType *tile_addr = elem_addr(pos);
    return Tensor<ElemType, TileFormat, Space>(tile_addr, tile_format);
  }

  friend std::ostream &operator<<(std::ostream &os, Tensor tensor) {
    os << "Tensor@0x" << std::hex << tensor.addr() << std::dec;
    os << "  shape: [";
    // print each dim of tensor shape
    hana::fold_left(tensor.shape().dim, [&os](auto const &v1, auto const &v2) {
      os << v1 << ", " << v2;
      return "";  // accumulated state
    });
    os << "]";
    return os;
  }

  // Dump each element of the tensor
  void dump() const {
    std::cout << *this;
    using Shape = typename format_traits<Format>::shape_type;
    detail::TensorHelper<Shape::nDims, Tensor>::dump(*this);
  }

 private:
  static_assert(is_a<mem_space_tag, Space> && is_a<tensor_format_tag, Format>);

  template <typename TensorFormat, typename Pos, typename TileShape>
  constexpr void get_tile_check() const {
    using shape_type = typename format_traits<TensorFormat>::shape_type;
    static_assert(is_dims_type<std::remove_reference_t<Pos>> &&
                  is_dims_type<std::remove_reference_t<TileShape>>);
    static_assert(std::remove_reference_t<Pos>::nDims == shape_type::nDims &&
                  std::remove_reference_t<TileShape>::nDims ==
                      shape_type::nDims);
    // assert(pos within shape && tile_shape within shape);
  }
};

template <typename ElemType, typename Space, typename Format>
constexpr auto make_tensor(ElemType *data, Format const &format) {
  return Tensor<ElemType, Format, Space>(data, format, Space());
}

template <typename T>
struct tensor_traits;

template <typename ElemType, typename Format, typename Space>
struct tensor_traits<Tensor<ElemType, Format, Space>> {
  using space_type = Space;
  using elem_type = ElemType;
};

// Tensor expression - a boost::yap's terminal of Tensor
template <typename ElemType, typename Format, typename Space = MemSpace::GM>
constexpr auto TensorExpr(ElemType *addr, Format const &format,
                          Space const &space = MemSpace::GM()) {
  static_assert(is_a<mem_space_tag, Space>);
  static_assert(is_a<tensor_format_tag, Format>);

  auto tensor = Tensor<ElemType, Format, Space>(addr, format);
  return boost::yap::make_terminal(std::move(tensor));
}

}  // namespace mt
