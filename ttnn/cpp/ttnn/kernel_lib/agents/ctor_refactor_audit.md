# Eltwise chain element ctor refactor — audit

Branch: `astancov/eltwise_run7_refined` HEAD `07f7a94b66f`

## Goal

Replace `Elt elt{}; elt.<field> = v;` with `Elt{v}` (positional ctor). C++17.

## Aggregate-init feasibility

C++17 aggregate-init with empty (non-`std::is_empty`-base + brace-required) bases requires `Elt{{}, val1, val2}` — verified locally with gcc -std=c++17. The cleanest path that keeps `Elt{}` (default-init) and `Elt{val1, val2}` (positional) both compiling is to add **explicit ctors** to the chain-element structs.

The SFPU op-structs in `eltwise_math.hpp` / `eltwise_scalar.hpp` / `eltwise_activations.hpp` / `eltwise_fill.hpp` / `eltwise_rand.hpp` **already** have explicit ctors — no changes required for those.

## Inventory — element types with runtime member fields

### Chain elements in `eltwise_chain.inl` (need ctor additions)

| # | Struct | File:line | Member fields (in declaration order) | Default | Current ctor? |
|---|---|---|---|---|---|
| 1 | `CopyTile<Cb, DstSlot, Policy, IndexMode, Reconfig>` | eltwise_chain.inl:247 | `uint32_t cb_tile_idx` | `= 0` | implicit aggregate |
| 2 | `PackTile<Cb, DstSlot, Policy, IndexMode, Reconfig, EnableFp32DestAccV>` | eltwise_chain.inl:340 | `uint32_t output_tile_idx` | `= 0` | implicit aggregate |
| 3 | `BinaryFpu<CbA, CbB, CbOut, Op, Bcast, DfReconfig, APolicy, BPolicy, Index, DstSlot, EnableFp32DestAccV>` | eltwise_chain.inl:503-504 | `uint32_t a_tile_idx`, `uint32_t b_tile_idx` | both `= 0` | implicit aggregate |
| 4 | `DestReuseBinary<Cb, Op, ReuseType, DstIn, DstOut, Reconfig, Policy, IndexMode, EnableFp32DestAccV>` | eltwise_chain.inl:630 | `uint32_t cb_tile_idx` | `= 0` | implicit aggregate |
| 5 | `UnaryBcast<Dim, Cb, CbOut, DstSlot, Policy, Reconfig, EnableFp32DestAccV>` | eltwise_chain.inl:687 | (none — exec uses fixed in_tile_index=0) | — | n/a |
| 6 | `PackTileBlock<Cb, FirstSlot, NTiles, Policy, Reconfig, EnableFp32DestAccV>` | eltwise_chain.inl:395 | (none) | — | n/a |

### Block elements in `eltwise_block.hpp` (none — block-indexed via loop var)

| Struct | Member fields |
|---|---|
| `BlockCopyTile` | (none) |
| `BlockBinaryFpu` | (none) |
| `BlockPackTile` | (none) |

### SFPU / Fill / Rand structs (already have ctors)

These structs already have explicit ctors — they work today with `Elt{val}` positional construction.

| Struct | File:line | Fields |
|---|---|---|
| `Power` | eltwise_math.hpp:80 | `uint32_t exponent` |
| `Rpow` | eltwise_math.hpp:91 | `uint32_t base` |
| `FillScalar` | eltwise_fill.hpp:20 | `float value` |
| `FillInt` | eltwise_fill.hpp:31 | `uint32_t value` |
| `FillBitcast` | eltwise_fill.hpp:43 | `uint32_t bits` |
| `RandTile` | eltwise_rand.hpp:23 | `uint32_t from_`, `uint32_t scale_` |
| `Threshold` | eltwise_scalar.hpp:24 | `uint32_t threshold`, `uint32_t value` |
| `Clamp` | eltwise_scalar.hpp:36 | `uint32_t min_param`, `uint32_t max_param` |
| `AddUnary`/`SubUnary`/`MulUnary`/`DivUnary`/`RsubUnary` | eltwise_scalar.hpp:58-62 | `uint32_t param0` |
| `RdivUnary` | eltwise_scalar.hpp:66 | `uint32_t param0` |
| `Hardtanh` | eltwise_activations.hpp:74 | `uint32_t min_param`, `uint32_t max_param` |
| `Elu` | eltwise_activations.hpp:88 | `uint32_t alpha` |
| `Selu` | eltwise_activations.hpp:99 | `uint32_t scale`, `uint32_t alpha` |
| `Softplus` | eltwise_activations.hpp:110 | `uint32_t beta`, `uint32_t beta_recip`, `uint32_t threshold` |
| `Prelu` | eltwise_activations.hpp:122 | `uint32_t param0` |
| `LeakyRelu` | eltwise_activations.hpp:133 | `uint32_t slope` |

## Helper API design — chain elements

For each of `CopyTile`, `PackTile`, `BinaryFpu`, `DestReuseBinary`:
- Add `constexpr <T>() noexcept = default;` (keeps `Elt{}` working).
- Add `constexpr <T>(uint32_t a [, uint32_t b]) noexcept : member-init {}`.
- Keep public member fields as-is (no API breakage for any code that reads `.member` directly).
- Adding explicit ctors makes the struct non-aggregate, so the existing pattern `Elt{}; e.field = v;` still works because the default ctor zero-inits + assignable members remain.

### `OptionalChainElement<true, Inner>`

Pattern is `using Inner::Inner;` — inherits Inner's ctors. Verified locally that this propagates the new positional ctors correctly.

### `OptionalChainElement<false, Inner>`

Has a variadic `template <class... Ignored> constexpr explicit OptionalChainElement(Ignored&&...)` that swallows args. Already compatible with positional ctor args.

## Callsite inventory

Pattern `\.\(a\|b\|cb\|output\)_tile_idx\s*=` across `ttnn/cpp/ttnn/operations/`:

| File | Occurrences |
|---|---|
| `moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_w.cpp` | 7 |
| `moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_h.cpp` | 7 |
| `moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp` | 7 |
| `moreh/moreh_adamw/device/kernels/moreh_adamw.cpp` | 5 |
| `moreh/moreh_adam/device/kernels/moreh_adam.cpp` | 5 |
| `moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_w_large.cpp` | 3 |
| `moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_h_large.cpp` | 3 |
| `moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_c_large.cpp` | 3 |
| `moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp` | 3 |
| `moreh/moreh_softmax/device/kernels/moreh_softmax_h_large.cpp` | 3 |
| `moreh/moreh_softmax/device/kernels/moreh_softmax_c_large.cpp` | 3 |
| `moreh/moreh_sgd/device/kernels/moreh_sgd.cpp` | 3 |
| `moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp` | 1 |

**Totals:** 13 files, 53 assignment occurrences across operations. Zero in `ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/*.cpp`.

## Cluster grouping for commits

1. `moreh_adam` + `moreh_adamw` — adam family (2 files, 10 sites)
2. `moreh_softmax` (3 files) + `moreh_softmax_backward` (5 files) — softmax family (8 files, 32 sites)
3. `moreh_layer_norm_backward` (2 files) — 8 sites
4. `moreh_sgd` — 3 sites

## Commit plan

- **Commit 1:** Helper API additions — add `Elt() = default; Elt(uint32_t ...) : ...` ctors to `CopyTile`, `PackTile`, `BinaryFpu`, `DestReuseBinary` in `eltwise_chain.inl`. No callsite touched. Gate: `test_eltwise.py` 453P/7S.
- **Commit 2:** Adam family callsite sweep (adam + adamw). Gate: `moreh_adam` 132P/0F + `moreh_adamw` 19P/0F.
- **Commit 3:** Softmax family callsite sweep. Gate: `moreh_softmax` 77P/16F.
- **Commit 4:** Layer-norm-bwd callsite sweep. Gate: `moreh_layer_norm` 48P/48S.
- **Commit 5:** SGD callsite sweep. Gate: `moreh_sgd` 164P.
- **End-of-pipeline gate:** all listed suites green at pre-stated counts.
