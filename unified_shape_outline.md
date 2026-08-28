<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# `Shape` as a template parameter -- what it looks like, and what it costs

> **Status.** COMPLETE. The `Extent` rename and all four stages have landed. Neither
> geometry is declared by hand anywhere any more -- both are derived from the operand
> shapes, and every stage kept the selftest traces byte-identical. Stage 1 came in at **+188/-158 lines** across
> `api.h`, `impl.hpp`, `shape.hpp`, seven kernels and the selftest, with all three
> selftest traces byte-identical and all eight device tests passing.

Outline for phase 5 of [unified_llama_prefill.md](unified_llama_prefill.md). Every
code block below either compiles today (the `Shape` type and its metafunctions were
built and verified at `-std=c++17 -Wall -Wextra -Werror`) or is a mechanical rewrite
of a signature that exists now.

## 1. The type

Constexpr-only, variadic, no storage. **Verified compiling**, all `static_assert`s
holding, including at rank 1 and rank 3.

```cpp
// Innermost dimension LAST: Shape<4,4> is a 4x4 tile block, Shape<2,4,4> is two of
// them. Only the last two dims take part in the math ops; anything to the left is a
// leading extent the kernel loop walks.
template <uint32_t... Dims>
struct Shape {
    static constexpr uint32_t rank = sizeof...(Dims);
    static_assert(rank > 0, "a Shape needs at least one dimension");

    static constexpr uint32_t dims[rank] = {Dims...};
    static constexpr uint32_t dim(uint32_t i) { return dims[i]; }

    // One tile per page in v1, so the page count is just the product.
    static constexpr uint32_t num_pages = (uint32_t{1} * ... * Dims);

    static constexpr uint32_t cols = dims[rank - 1];
    static constexpr uint32_t rows = rank >= 2 ? dims[rank >= 2 ? rank - 2 : 0] : 1;
    static constexpr uint32_t leading = num_pages / (rows * cols);
};
```

**Shape equality is type identity.** That is the whole point: `Shape<1,4>` and
`Shape<4>` hold the same number of pages and are different types, so the mistake the
current `num_pages` cannot see becomes a compile error.

### Derived shapes: one helper covers every op, at any rank

```cpp
// Rebuild a shape with the last two dims replaced.
template <typename S, uint32_t H, uint32_t W, typename Idx> struct with_hw_impl;
template <uint32_t... D, uint32_t H, uint32_t W, std::size_t... I>
struct with_hw_impl<Shape<D...>, H, W, std::index_sequence<I...>> {
    using S = Shape<D...>;
    using type = Shape<(I == S::rank - 2 ? H : (I == S::rank - 1 ? W : S::dim(I)))...>;
};
template <typename S, uint32_t H, uint32_t W>
using with_hw = typename with_hw_impl<S, H, W, std::make_index_sequence<S::rank>>::type;

template <typename S, ReduceAxis A>
using reduce_shape = with_hw<S, (A == ReduceAxis::Cols ? S::rows : 1),
                                (A == ReduceAxis::Rows ? S::cols : 1)>;

template <typename A, typename B>
struct matmul_shape {
    static_assert(A::cols == B::rows, "matmul inner dimension disagrees: A's columns must equal B's rows");
    static_assert(A::leading == B::leading, "matmul operands disagree on their leading (batch) extent");
    using type = with_hw<A, A::rows, B::cols>;
};
```

Verified:

```cpp
reduce_shape<Shape<4,4>, Rows>     == Shape<1,4>
reduce_shape<Shape<4,4>, Cols>     == Shape<4,1>
reduce_shape<Shape<4,4>, Both>     == Shape<1,1>
reduce_shape<Shape<2,4,4>, Cols>   == Shape<2,4,1>     // leading extent preserved
matmul_shape_t<Shape<2,3>, Shape<3,5>>       == Shape<2,5>
matmul_shape_t<Shape<7,2,3>, Shape<7,3,5>>   == Shape<7,2,5>
```

**Variadic rank costs nothing extra.** `with_hw` is rank-generic, so supporting a
leading extent is free rather than a later migration.

### Error quality, measured

`matmul_shape_t<Shape<2,3>, Shape<4,5>>` gives:

```
error: static assertion failed due to requirement 'Shape<2, 3>::cols == Shape<4, 5>::rows':
       matmul inner dimension disagrees: A's columns must equal B's rows
```

clang prints the real shapes. This is the property to protect: every check must be a
`static_assert` that fires before overload resolution can produce a wall of
substitution failures.

## 2. Name collision

`Shape` is currently the multicast rectangle extent (`LogicalMcast{coord, Shape{h,w}}`),
used in 6 places: `api.h:90`, `api.h:133`, `impl.hpp:788`, `mcast_bcast.cpp:63`,
`matmul_mcast.cpp:102-103`. Rename it to **`Extent`** -- it is an h x w core-rectangle
size, and the name is accurate. Six-site mechanical rename, done first and separately.

## 3. The three types

The single most important burden finding: **`num_pages` becomes a `static constexpr`
member, and reading it through an instance still compiles.** Verified. So the ~40
places in `impl.hpp` that say `storage.num_pages`, `block.num_pages`,
`src.num_pages` need **no edit at all**. Only signatures change, not bodies.

```cpp
// before                                    // after
struct Storage {                             template <typename S>
    Storage(uint32_t cb_id,                  struct Storage {
            uint32_t num_pages);                 using shape = S;
    template <typename Node>                     explicit Storage(uint32_t cb_id);
    Block store(const Node&) const;              template <typename Node>
    uint32_t cb_id;                              Block<S> store(const Node&) const;   // static_asserts node_shape<Node> == S
    uint32_t num_pages;                          uint32_t cb_id;
};                                               static constexpr uint32_t num_pages = S::num_pages;
                                             };
```

`Block<S>` and `ComputeBlock<S>` follow the same pattern: drop the runtime
`num_pages` field, gain `static constexpr num_pages = S::num_pages`.

## 4. The signature burden

**20 declarations in `api.h`, 15 definitions in `impl.hpp`.** The mechanical
majority gain one template parameter and one substitution:

```cpp
// before
template <int thread, typename Accessor>
NocAsyncReadTx<thread> noc_load(const Storage& storage, const Accessor& acc, uint32_t block_idx);
// after
template <int thread, typename S, typename Accessor>
NocAsyncReadTx<thread, S> noc_load(const Storage<S>& storage, const Accessor& acc, uint32_t block_idx);

// before
template <int thread, typename Accessor>
NocAsyncWriteTx<thread> noc_store(Block block, const Accessor& acc, uint32_t block_idx);
// after
template <int thread, typename S, typename Accessor>
NocAsyncWriteTx<thread, S> noc_store(Block<S> block, const Accessor& acc, uint32_t block_idx);
```

### The eight sites that are NOT mechanical

| site | change | what it buys |
|---|---|---|
| `is_operand<ComputeBlock>` | full -> partial specialisation over `S` | -- |
| `as_node(const ComputeBlock&)` | becomes a template; `TileSource` gains `S` | lets `node_shape` see leaf shapes |
| the 5 unary overloads + `matmul` on `ComputeBlock` | become templates on `S` | -- |
| `Accumulator<Mode>` -> `Accumulator<Mode, S>` | one shape for both Storages | its "two Storages, same shape" rule is **documented only** today |
| `noc_core_read/write(const Storage<D>&, Block<S>, ...)` | two shape params + `static_assert(D == S)` | src/dst size agreement is **unchecked** today |
| `fill_reduce_scaler(const Storage<S>&)` | `static_assert(S == Shape<1,1>)` | a scaler is one tile by definition |
| `.bias(operand)` | `static_assert(shape == Shape<1, Ct>)` | replaces a **runtime** `ASSERT` with a compile-time one |
| `node_shape<>` | new metafunction in `math.hpp` | `expr.hpp` stays shape-agnostic |

## 5. What kernels look like

### `matmul.cpp` -- net neutral, three derived products deleted

```cpp
// before                                              // after
using Geom = u::MatmulGeometry<MM_RT_DIM, MM_CT_DIM,    using In0  = u::Shape<MM_RT_DIM, MM_KT_DIM>;
                               MM_KT_DIM, MM_K_BLOCKS>; using In1  = u::Shape<MM_KT_DIM, MM_CT_DIM>;
constexpr uint32_t kIn0Tiles = MM_RT_DIM * MM_KT_DIM;   using Out  = u::Shape<MM_RT_DIM, MM_CT_DIM>;
constexpr uint32_t kIn1Tiles = MM_KT_DIM * MM_CT_DIM;   using Bias = u::Shape<1, MM_CT_DIM>;
constexpr uint32_t kOutTiles = MM_RT_DIM * MM_CT_DIM;
u::Storage in0_storage(kCbIn0, kIn0Tiles);              u::Storage<In0>  in0_storage(kCbIn0);
u::Storage in1_storage(kCbIn1, kIn1Tiles);              u::Storage<In1>  in1_storage(kCbIn1);
u::Storage acc_storage(kCbAcc, kOutTiles);              u::Storage<Out>  acc_storage(kCbAcc);
u::Storage out_storage(kCbOut, kOutTiles);              u::Storage<Out>  out_storage(kCbOut);
u::Storage bias_storage(kCbBias, MM_CT_DIM);            u::Storage<Bias> bias_storage(kCbBias);
...                                                     ...
acc.accumulate(u::matmul<Geom>(a, b), finish)           acc.accumulate(u::matmul(a, b), finish)
```

Same line count. The geometry is gone, the three hand-multiplied products are gone,
`Bias` states its rank so the runtime assert becomes compile-time, and a wrong
`MM_KT_DIM` now fails to compile instead of producing silent garbage.

### `reduction_tree.cpp` -- the real win

```cpp
// before                                                     // after
using PerCore   = u::ReduceGeometry<in_ht, in_wt>;             using In       = u::Shape<in_ht, in_wt>;
using PerColumn = u::ReduceGeometry<num_cores_y, in_wt>;       using PerCore  = u::reduce_shape<In, kAxis>;
constexpr uint32_t reduced_tiles_per_block                     using Gathered = u::Shape<num_cores_y * PerCore::rows,
    = PerCore::out_tiles(kAxis);                                                          PerCore::cols>;
                                                               using Out      = u::reduce_shape<Gathered, kAxis>;
u::Storage in0_storage(kCbIn0, PerCore::num_tiles);            u::Storage<In>             in0_storage(kCbIn0);
u::Storage scaler_storage(kCbScaler, 1);                       u::Storage<u::Shape<1, 1>> scaler_storage(kCbScaler);
u::Storage tmp0_storage(kCbTmp0, reduced_tiles_per_block);     u::Storage<PerCore>        tmp0_storage(kCbTmp0);
u::Storage tmp1_storage(kCbTmp1,                               u::Storage<Gathered>       tmp1_storage(kCbTmp1);
    reduced_tiles_per_block * num_cores_y);                    u::Storage<Out>            out_storage(kCbOut);
u::Storage out_storage(kCbOut, reduced_tiles_per_block);
```

`reduced_tiles_per_block * num_cores_y` -- the hand arithmetic the out-block-index
collision hid in -- becomes a shape the compiler derives and `store()` verifies.

## 6. Burden verdict

**Cheaper than it looks**, for one reason: bodies don't change. The `static constexpr
num_pages` trick means the ~40 internal `.num_pages` reads compile untouched.

| | |
|---|---|
| Mechanical signature edits | ~27 of 35, one template param each |
| Non-mechanical sites | 8, listed above -- and 4 of them *add a check that does not exist today* |
| Kernel line count | net neutral |
| Bodies in `impl.hpp` | **unchanged** |
| Verification | selftest trace must stay **byte-identical** -- no emitted instruction changes |

Real costs, not hidden:

- **No CTAD.** `u::Storage in0(cb, 4)` becomes `u::Storage<In0> in0(cb)`; the shape must
  be named. That is the feature, but it is more to type at each declaration. A
  `using In0 = ...` alias per buffer is the idiom, as above.
- **The ADL hooks get touched again.** `is_operand` and `as_node` are exactly what
  produced the `Fluent` ordering trap. Expect that class of error.
- **`Block`'s moved-from poison loses one line.** It stamps `cb_id` and `num_pages` with
  `kMovedFrom`; only `cb_id` can be poisoned now. The contract survives.
- **Template error walls** if a check is missed. Mitigated by making every constraint a
  `static_assert` first -- proven achievable in section 1.

## 7. Decisions

1. **Rank: variadic.** Arbitrary rank is a useful abstraction and `with_hw` is already
   rank-generic, so it costs nothing in the implementation. No `static_assert(rank == 2)`.
2. **Eltwise: strict.** `a + b` demands identical shapes, not merely equal `num_pages`.
   That is what catches `Shape<1,4>` against `Shape<4>`. An explicit `reshape<S2>(block)`
   escape can be added if a legitimate reinterpretation ever needs one.
3. **Dynamic: not yet.** A `DynamicShape` class, used *instead of* `Shape`, is the
   eventual escape for the runtime-extent cases. Leave room for it; do not build or
   design it out now.

## 8. What Stage 2 found

Six checks landed, each proven by a violation that must fail to compile, plus a
control proving the legal case still passes. What building them turned up:

- **The core-copy invariant is not equality.** The first version asserted
  `same_shape_v<D, S>` and `test_unified_reduction.py` rejected it immediately: a
  GATHER has n writers each depositing its own source at its own `byte_offset`, so the
  destination is legitimately n times the source. The correct invariant is
  `S::num_pages <= D::num_pages` together with `D::num_pages % S::num_pages == 0` --
  the source fits, and it tiles the destination evenly, one slot per writer.
- **The trace harness was nondeterministic for one commit.** Giving the selftest a real
  L1 stand-in -- needed because `fill_reduce_scaler` dereferences the write pointer --
  let an ASLR-randomised mmap address reach the trace through the NOC address
  `noc_core_write` builds. Byte-identical trace comparison is the entire safety net for
  this refactor, so it is now mapped at a fixed address with `MAP_FIXED_NOREPLACE`,
  verified deterministic over five runs.
- **The syntax probe had been silently inconsistent.** It stored a
  `MatmulGeometry<2,2,2>` result -- `Shape<2,2>` -- into a `Storage<Shape<1,2>>`. The
  new `store` conformance check caught it on its first compile.
- **One runtime `ASSERT` became redundant and is gone.** `Strategy<ReduceFusion>`
  checked `num_tiles == kOut` at runtime and only in asserts-enabled builds. `store`
  conformance subsumes it: compile-time, unconditional, and on the full shape rather
  than just the page count.
- **The bias check now catches a rank error.** `Shape<ct_dim, 1>` and
  `Shape<1, ct_dim>` have identical page counts, so the old runtime page-count `ASSERT`
  could not tell a column from a row. Both are now rejected; verified on device.

## 9. What Stage 3 found

`MatmulGeometry`'s five `uint32_t` parameters became two shapes, and it is now derived
rather than declared -- no kernel writes one. `rt_dim`, `ct_dim` and `kt_dim` are read
off the operands with their agreement checked; `NumBlocks` and `In1RowStride` are
deleted outright. `MatmulNode` carries `lhs_shape`/`rhs_shape` and exposes
`MatmulGeometry<SA, SB>` as `geometry`, so `Strategy<FPUFusion>`'s body did not change
at all -- which is why the trace stayed byte-identical through the whole stage.

- **`Accumulator` bypassed `store` conformance, and that was a real hole.** It drives
  the strategy directly, so nothing checked that its two Storages matched the matmul's
  output block. Proven the worst way: a `Storage<Shape<2,1>>` accumulator on a
  `Shape<1,2>` matmul compiled AND ran correctly on device, because the two shapes hold
  the same number of pages. Now a `static_assert` on the node's shape, the same one
  `store` applies.
- **A sabotage that looks fine can be a no-op.** The first attempt at the wrong-output
  test swapped `Shape<rt, ct>` for `Shape<ct, rt>` in the default configuration, where
  `rt == ct == 1` -- identical types, nothing to catch. Asymmetric dimensions are
  required to make that test mean anything.
- **`matmul_init` needs the shapes before the buffers.** It programs the block
  dimensions, so in `matmul_mcast.cpp` the `using` aliases had to move above the
  `matmul_init` call they feed.

## 10. What Stage 4 found

`ReduceGeometry<Ht, Wt>` became `ReduceGeometry<S>`, derived from the operand's shape.
It needed no other change: every member -- `out_tiles`, `elements`, `group`,
`contributor` -- was already a pure function of `(rows, cols, axis)`, so only where the
two extents come from moved. `reduce_sum<RG, Axis>(a, sc)` is `reduce_sum<Axis>(a, sc)`.

`reduction_tree.cpp` is where it shows. Gone: both `using PerCore/PerColumn` lines, the
hand-derived `reduced_tiles_per_block`, and the geometry argument to its `RT_REDUCE`
macro. What remains is four shape aliases and the ops that read them.

- **A batched reduce is now an explicit refusal rather than a silent wrong answer.**
  `reduce_shape<Shape<2,4,4>, Cols>` correctly gives `Shape<2,4,1>`, so the shape
  algebra handles a leading extent -- but `Strategy<ReduceFusion>` walks a single 2-D
  grid and would have quietly reduced only the first slice. `ReduceGeometry` now
  `static_assert`s `S::leading == 1` and says to loop per batch from the kernel.
- **One legitimate geometry USE survives**, and it should:
  `ReduceGeometry<In>::elements(kAxis)` supplies the mean scaler's 1/N. Querying a
  derived property is the point; declaring the geometry was the problem.
