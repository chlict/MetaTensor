#pragma once

#include <iostream>

struct trange_tag;

// A range is made up with [begin, end, step, size]. It describes how to tile a
// dimension of a tensor, e.g, for a 1D tensor of which length is 6, say (a0,
// a1, a2, a3, a4, a5), a tile range of [0, 6, 3, 2] tells tiling it as tile0 =
// (a0, a1) and tile1 = (a3, a4). If the 'size' is not specified, it will be
// equal to the 'step'. So a range of [0, 6, 3] results in tile0 = (a0, a1, a2)
// and tile1 = (a3, a4, a5)
template <typename Begin, typename End, typename Step, typename Size = Step>
struct TRange {
  static_assert(is_integral_or_constant<Begin> &&
                is_integral_or_constant<End> && is_integral_or_constant<Step> &&
                is_integral_or_constant<Size>);
  using tag = trange_tag;

  const Begin begin_;
  const End end_;
  const Step step_;
  const Size size_;

  constexpr TRange(Begin const &begin, End const &end, Step const &step,
                   Size const &size = Size())
      : begin_(begin), end_(end), step_(step), size_(size) {}

  constexpr TRange(Begin &&begin, End &&end, Step &&step, Size &&size = Size())
      : begin_(std::move(begin)),
        end_(std::move(end)),
        step_(std::move(step)),
        size_(std::move(size)) {}

  constexpr TRange(TRange const &other)
      : begin_(other.begin_),
        end_(other.end_),
        step_(other.step_),
        size_(other.size_) {}

  constexpr TRange(TRange &&other)
      : begin_(std::move(other.begin_)),
        end_(std::move(other.end_)),
        step_(std::move(other.step_)),
        size_(std::move(other.size_)) {}

  constexpr auto begin() const { return begin_; }

  constexpr auto end() const { return end_; }

  constexpr auto step() const { return step_; }

  constexpr auto size() const { return size_; }

  friend std::ostream &operator<<(std::ostream &os, TRange const &r) {
    os << "Range: [" << r.begin() << ", " << r.end() << ", " << r.step() << ", "
       << r.size() << "]";
    return os;
  }
};
