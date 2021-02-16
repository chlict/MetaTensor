# MetaTensor

## Introduction

This is a tiny project using C++ meta programming to do Tensor arithmetics. This is a header-only library. One of the major features provided by this library is uniform programming of static shaped tensor and dynamic shaped tensor. For static shaped tensor, which means the tensor shape is known at compile time, this library provides zero-cost abstraction for the operations, i.e., no runtime cost introduced by shape calculation or element position calculation. On the other hand, if the tensor shape is known only at runtime, the code is almost same with the code programmed for static shaped tensor. There will be some necessary runtime cost for dynamic shaped tensor, which may be considered heavy compared to static shaped tensor for some architecture, we want to make them as small as possible.

To construct a tensor with MetaTensor, the first step is to create a `TensorFormat` object, which describes the shape and layout of a tensor. If the shape is known at compile time, use the `boost::hana::integral_constant` literals to describe each dim of the shape, otherwise, use regular integer values or variables as arguments.

```cpp
#include "Tensor.hpp"
using boost::hana::literals;

std::array<float, 8> buffer = { /* ... */ };

// Create a 2D row-major tensor, its shape is 2 x 4 known at compile time
auto format1 = make_format(Dim2(2_c, 4_c), RowMajorLayout());
auto tensor1 = Tensor(buffer.data(), format1);

// Tensor shape is m x n, but m and n are known at runtime
auto format2 = make_format(Dim2(m, n), RowMajorLayout());
auto tensor2 = Tensor(buffer.data(), format2);
```

See, the only difference is `Dim2(2_c, 4_c)` and `Dim2(m, n)` passed to `make_format`. Note that `Dim2(2, 4)` is treated as dynamic shape as `2` and `4` will be stored into variables and no guaranteed to be constant all the time (constant propagation can help but not guaranteed). On the contrary, `2_c` and `4_c` carry their values in their types so the value information will never lost during compilation.

Another feature of MetaTensor is it provides lazy evaluation via expression templates. With lazy evaluation, tensor operators can be preprocessed, calculating some parameters, and invoke the actual calculation. For a heterogenous platform, the preprocessing can be performed on host CPU, and the rest can be performed as kernel on device. MetaTensor provides convenient way to write lazy evaluated tensor expressions, e.g., to write a add-mul operator:

```cpp
  auto format1 = make_format(Dim2(2, 4), RowMajorLayout());
  auto tensor1 = Tensor((float *)data1, format1);
  auto tensor2 = Tensor((float *)data2, format1);
  auto tensor3 = Tensor((float *)data3, format1);

  auto add_mul = ExprBlock{_t1 = 1_p + 2_p, _t2 = _t1 * 3_p};
  add_mul(tensor1, tensor2, tensor3);
```

## Build

```shell
mkdir build
cd build
cmake ..
make
```

Requires a c++17 compliant compiler.
