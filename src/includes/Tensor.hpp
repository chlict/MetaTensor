#pragma once

#include <boost/yap/expression.hpp>
#include <iostream>

#include "MatrixFormat.hpp"
#include "TensorFormat.hpp"
#include "Utils.hpp"

struct mem_space_tag;

struct MemSpace {
  struct GM {
    using tag = mem_space_tag;
  };
  struct L1 {
    using tag = mem_space_tag;
  };
};

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

template <typename ElemType, typename Format, typename Space, typename Addr>
struct Tensor : TensorHandle {
  using tag = tensor_tag;

 private:
  const Format format_;
  const Addr addr_;

 public:
  constexpr Tensor(ElemType const &type, Format const &format,
                   Space const &space, Addr const &addr)
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

  constexpr auto dimensions() const { return format_.dimensions(); }

  constexpr auto strides() const { return format_.strides(); }

  // Get the address of an element at specified postion (coorination).
  // The return value might be an integral constant or an uint64_t, so remember
  // to cast it to ElemType * before use.
  template <typename Pos>
  constexpr auto elem_addr(Pos const &pos) const {
    using Layout = typename format_traits<Format>::layout_provider_type;
    auto offset = Layout::offset(pos, layout());
    constexpr bool is_constant =
        hana::is_a<hana::integral_constant_tag<long long>>(offset) &&
        hana::is_a<hana::integral_constant_tag<long long>, Addr>;

    if constexpr (is_constant) {
      return addr() + offset * hana::llong_c<sizeof(ElemType)>;
    } else {
      return (uint64_t)addr() + sizeof(ElemType) * (uint64_t)offset;
    }
  }

  // Get an element at specified postion
  template <typename Pos>
  constexpr ElemType elem(Pos const &pos) const {
    return *(ElemType *)elem_addr(pos);
  }

  // Set an element at specified postion
  template <typename Pos>
  constexpr void set(Pos const &pos, ElemType const &elem) const {
    ElemType *addr = static_cast<ElemType *>(elem_addr(pos));
    *addr = elem;
  }

  template <typename Pos, typename TileShape>
  constexpr auto get_tile(Pos &&pos, TileShape &&tile_shape) const {
    get_tile_check<Format, Pos, TileShape>();
    // layout is same as original layout
    using LayoutType = typename format_traits<Format>::layout_provider_type;
    auto tile_format =
        make_format(std::forward<TileShape>(tile_shape), LayoutType());
    auto tile_addr = elem_addr(pos);
    using TileFormat = decltype(tile_format);
    using TileAddr = decltype(tile_addr);
    return Tensor<ElemType, TileFormat, Space, TileAddr>(
        ElemType(), tile_format, Space(), tile_addr);
  }

  friend std::ostream &operator<<(std::ostream &os, Tensor tensor) {
    os << "Tensor@0x" << std::hex << tensor.addr() << std::dec;
    os << "  shape: [";
    hana::for_each(tensor.shape().dim,
                   [&os](auto const &v) { os << v << ", "; });
    os << "]";
    return os;
  }

  void dump() const {
    using Shape = typename format_traits<Format>::shape_type;
    const long ELEM_PER_LINE = 8;
    std::cout << *this;
    if constexpr (Shape::nDims == 1) {
      const long size = static_cast<long>(shape().dim[0_c]);
      for (long i = 0; i < size; i++) {
        if (i % ELEM_PER_LINE == 0) std::cout << "\n";
        std::cout << " " << elem(Dims(i));
      }
      std::cout << "\n";
    }
    if constexpr (Shape::nDims == 2) {
      // Here we use standard loop rather hana:for_each() because
      // hana::for_each() might unroll the loop heavily. Compiler can also
      // unroll and vectorize this oridinary loop. NOTE: do NOT use size_t or
      // unsigned for i or it will compile with error "not in the same ring"
      const long size0 = static_cast<long>(shape().dim[0_c]);
      const long size1 = static_cast<long>(shape().dim[1_c]);
      for (long i0 = 0; i0 < size0; i0++) {
        for (long i1 = 0; i1 < size1; i1++) {
          if (i1 % ELEM_PER_LINE == 0) std::cout << "\n";
          std::cout << " " << elem(Dims(i0, i1));
        }
      }
      std::cout << "\n";
    }
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

// Tensor expression - a boost::yap's terminal of Tensor
template <typename ElemType, typename Format, typename Space, typename Addr,
          typename = std::enable_if_t<is_a<mem_space_tag, Space> &&
                                          is_a<tensor_format_tag, Format> &&
                                          is_integral_or_constant<Addr>,
                                      void>>
constexpr auto TensorE(ElemType const &type, Format const &format,
                       Space const &space, Addr const &addr) {
  auto tensor = Tensor(type, format, space, addr);
  return boost::yap::make_terminal(std::move(tensor));
}

template <typename T>
struct tensor_traits;

template <typename ElemType, typename Format, typename Space, typename Addr>
struct tensor_traits<Tensor<ElemType, Format, Space, Addr>> {
  using space = Space;
  using elem_type = ElemType;
};
