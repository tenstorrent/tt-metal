# Operation Design: multigammaln_lanczos

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (fused elementwise unary) |
| Goal | Compute `torch.special.multigammaln(x, p=4)` as a **single fused TTNN kernel**, by translating the 4 × Lanczos-6 lgamma composite into SFPU tile primitives. One `ttnn.generic_op` dispatch per call; no `ttnn::*` chaining in Python. |
| Math | `output[i] = sum_{k=0..3} lgamma_lanczos(input[i] − 0.5·k) + 3·log(π)`, where `lgamma_lanczos(a)` is the Lanczos 6-term polynomial defined below, with the result zeroed at integer poles `a == 1` and `a == 2`. |
| Mode | Derivative (the kernel computes the math directly; no autograd machinery). |
| References | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/where_tss_kernel.cpp` (per-tile multi-DST SFPU composition with `fill_tile`), `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/mish_kernel.cpp` (FP32 SFPU composition with `Approx::Exact` precision), `ttnn/ttnn/operations/toy_binary_in_place/` (Python `ProgramDescriptor` shape with intermediate CB), `tests/ttnn/unit_tests/operations/debug/test_generic_op.py` (multi-core `split_work_to_cores` over an elementwise op in `generic_op` form). |

### Lanczos lgamma (literal translation of the removed `_lgamma` reference)

For each input element `a`:
```
input = a − 1
temp  = 1 + C0/(input+1) + C1/(input+2) + C2/(input+3)
          + C3/(input+4) + C4/(input+5) + C5/(input+6)
t     = input + 5.5
result = (input + 0.5) · log(t) + 0.918938531357171 + log(temp) − t
if a == 1.0 or a == 2.0: result = 0.0
```
with
```
C0 =  76.18009172947146      C3 = -1.231739572450155
C1 = -86.50532032941677      C4 =  0.1208650973866179e-2
C2 =  24.01409824083091      C5 = -0.5395239384953e-5
```

Algebraic re-grouping used by the kernel (mathematically identical; chosen so it fits inside 4 fp32 DEST slots):
```
input + 1 + k  ==  a + k                          for k in 0..5
input + 5.5    ==  a + 4.5                         (= t)
input + 0.5    ==  a − 0.5
result = (a − 0.5)·log(a + 4.5) + log(temp) − (a + 4.5) + 0.918938531357171
       = (a − 0.5)·log(a + 4.5) + log(temp) − a − 3.581061468642829
```
The constant `3.581061468642829 = 4.5 − 0.918938531357171` is precomputed at compile time. The `−4.5 + 0.918938…` simplification is the only algebraic rewrite; the 6 Lanczos coefficients, the `+5.5` shift, the `+0.918938…` constant, and the `a == 1 / a == 2` pole zeroing are reproduced bit-for-bit.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | float32, TILE_LAYOUT, rank ≥ 2, H % 32 == 0, W % 32 == 0, on-device | — | runtime (buffer addr; tile count baked at compile time per program) |

The op exposes **no other parameters**. In particular, there is no `p` argument (fixed at 4), no `memory_config` argument (the output inherits the input's memory config), and no `compute_kernel_config` argument (HiFi4 + fp32 dest acc is hard-coded internally — see below).

### Compute Config (hard-coded internally — NOT a caller parameter)

| Field | Value |
|-------|-------|
| `math_fidelity` | `ttnn.MathFidelity.HiFi4` |
| `fp32_dest_acc_en` | `True` |
| `dst_full_sync_en` | `False` (default — half-sync) |
| Effective DEST capacity | **4 tiles** (half-sync + fp32 dest acc). Confirmed by `compute_kernel_lib::DEST_AUTO_LIMIT` evaluation at `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:97`. |

The compute kernel never reads a caller-supplied compute config; the API surface does not expose one. Phase 0 is the maximum-precision configuration.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2 (validator). Phase 0 tests exercise rank == 4 `(N, C, H, W)`. |
| Dtype | `float32` (Phase 0). Any other dtype → `ValueError`. |
| Layout | `TILE_LAYOUT`. `ROW_MAJOR_LAYOUT` → `ValueError`. |
| Memory | DRAM or L1 interleaved. |
| Tile-alignment | `H % 32 == 0`, `W % 32 == 0`. Non-tile-aligned shapes → `ValueError`. |
| Value domain | Operation is mathematically defined for `a > 1.5`. Inputs outside the safe domain are not branched on; they propagate as NaN/Inf naturally. The Lanczos approximation is real-valued and stable on the test domain `a ∈ [2.0, 10.0]`. |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | `float32` |
| Layout | `TILE_LAYOUT` |
| Memory | inherits `input_tensor.memory_config()` (no separate kwarg). |

## Validation (Python side, before launch)

| Check | Failure |
|-------|---------|
| `input_tensor.dtype == ttnn.float32` | `ValueError` |
| `input_tensor.layout == ttnn.TILE_LAYOUT` | `ValueError` |
| `len(input_tensor.shape) >= 2` | `ValueError` |
| `input_tensor.shape[-1] % 32 == 0 and input_tensor.shape[-2] % 32 == 0` | `ValueError` |
| `input_tensor.storage_type() == ttnn.StorageType.DEVICE` (must be on device) | `ValueError` |

The validator runs **before** any `generic_op` dispatch — no half-launched program if validation fails.

## Dataflow Strategy

The operation is **elementwise unary**. Each output element depends only on the corresponding input element. There is **no reduction**, **no broadcast**, and **no inter-Tensix communication** (no multicast, no semaphores, no ring topology).

| Stage | Role | Data path |
|-------|------|-----------|
| **DRAM → reader** | Reader (NCRISC, NoC0) streams `input_tensor` tiles from DRAM into `cb_input_tiles`, one tile per loop iteration. Double-buffered so the next tile can be fetched while compute consumes the current one. | `input_tensor` DRAM → `cb_input_tiles` |
| **compute (single Tensix, all 3 TRISCs)** | The compute kernel processes one input tile at a time. For each tile it (a) initializes `cb_accumulator` to zero, (b) runs 4 Lanczos-lgamma evaluations against `(a − 0.5·k)` for `k = 0..3`, each evaluation reading the same input tile via `copy_tile` (no pop until the 4 iterations finish) and accumulating into `cb_accumulator`, and (c) adds the `3·log(π)` constant and pushes the final tile to `cb_output_tiles`. | `cb_input_tiles` → DST → `cb_accumulator` → `cb_output_tiles` |
| **compute → writer** | Writer (BRISC, NoC1) drains `cb_output_tiles` one tile at a time and writes back to DRAM at the same tile-id as the input it originated from (output shape == input shape). | `cb_output_tiles` → output DRAM |

Tensor format does not change. Inputs arrive tiled, every CB holds tiles, and the output is written tiled. No tilize/untilize step is needed.

## Work Distribution

The work unit is **one input tile**. Tiles are independent (elementwise) so distribution is purely a parallelization decision.

| Field | Value |
|-------|-------|
| Work unit | One float32 tile (32×32 elements) of `input_tensor` (and the matching tile of `output_tensor`). |
| Grid | `device.compute_with_storage_grid_size()` — full available compute grid (e.g. 8×8 on Wormhole). |
| Total work | `total_tiles = input_tensor.buffer_num_pages()` (since input is TILE_LAYOUT and not padded beyond the tile boundary that the validator enforces). |
| Per-core work | `ttnn.split_work_to_cores(grid_size, total_tiles)` returns `(num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_g1, tiles_per_core_g2)`. Each core runs the same kernel binary; the per-core RT arg `num_tiles` and the runtime `start_tile_id` differ between cores. |
| Remainder | `split_work_to_cores`'s two-group split (`core_group_1` + `core_group_2`); each group has uniform per-core tile counts differing by at most 1. Group-2 cores can have 0 tiles (when total_tiles ≤ num_grid_cores) — those cores are simply not in the grid. |
| Per-core RT args (reader / writer) | `(buffer_address, num_tiles, start_tile_id)` — identical pattern for both. |
| Per-core RT args (compute) | none beyond compile-time `num_tiles_per_core_g1` / `_g2`. Implementer uses **one KernelDescriptor per group** (compile-time `num_tiles` is different) **OR** a single KernelDescriptor that reads `num_tiles` from per-core runtime args. The Python `generic_op` descriptor in `tests/ttnn/unit_tests/operations/debug/test_generic_op.py` shows the single-group `num_tiles` as a compile-time arg; for two groups, declare two KernelDescriptors, each scoped to its core group. |
| Inter-core comms | none. |

`all_cores` (the union returned by `split_work_to_cores`) is used as the `core_ranges` for every CB descriptor.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_tiles` | 0 | `ttnn.tile_size(ttnn.float32)` = 4096 B | 2 (double-buffer for reader↔compute streaming) | float32 | reader | compute (`copy_tile` × 4 per input tile, then `cb_pop_front(1)`) | one tile alive at a time; 4 reads via `copy_tile` before pop |
| `cb_output_tiles` | 16 | 4096 B | 2 (double-buffer for compute↔writer streaming) | float32 | compute (1 push per input tile, at the very end) | writer | one tile alive at a time; standard streaming |
| `cb_accumulator` | 24 | 4096 B | 2 (read-modify-write needs front + back simultaneously) | float32 | compute (5 pushes per input tile: 1 init-zero + 4 lgamma updates) | compute itself (4 reads of "previous accumulator" + 1 final read for the `3·log(π)` add) | strictly intra-tile; emptied (5 pops match 5 pushes) before the kernel moves on to the next input tile |

CB descriptor `core_ranges` for **every** CB is `all_cores` (the union returned by `split_work_to_cores`). Indices follow convention: 0–7 input, 16–23 output, 24–31 intermediate.

### CB sync — push count = wait count

| CB | Pushes per input tile | Waits per input tile | Pops per input tile | Match |
|----|----------------------|----------------------|---------------------|-------|
| `cb_input_tiles` | reader: 1 | compute: 1 (with 4 non-popping `copy_tile` reads) | compute: 1 (at end of the 4 iterations) | ✓ (1 = 1) |
| `cb_output_tiles` | compute: 1 | writer: 1 | writer: 1 | ✓ |
| `cb_accumulator` | compute: 1 (init zero) + 4 (per-lgamma update) = 5 | compute: 4 (per-lgamma reload of previous accumulator) + 1 (final reload to add `3·log(π)`) = 5 | compute: 4 (per-lgamma after writing the new accumulator) + 1 (after the final read) = 5 | ✓ (5 = 5 = 5) |

### CB sizing rationale

| CB | Reason for sized pages |
|----|------------------------|
| `cb_input_tiles` (2) | Standard double-buffer for reader↔compute streaming. Reader fetches tile N+1 while compute consumes tile N. |
| `cb_output_tiles` (2) | Standard double-buffer for compute↔writer streaming. |
| `cb_accumulator` (2) | The intra-tile read-modify-write cycle needs the front (current accumulator) and the back (new accumulator) **simultaneously**: `cb_wait_front(1)` and `cb_reserve_back(1)` must both succeed before the compute acquire begins. 1 page would deadlock (no slot to reserve while front is held). 2 pages is the minimum that allows the ping-pong. |

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference. All compute-side calls live in a single compute kernel; no nested helper indirection beyond what is listed.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| **Reader, per tile** | raw_api | `cb_reserve_back` / `noc_async_read` (with `TensorAccessor::get_noc_addr(tile_id)`) / `noc_async_read_barrier` / `cb_push_back` | dataflow primitives in `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | — | DRAM (input via `TensorAccessor`) | `cb_input_tiles` (push 1 per iter) | **Helpers considered and rejected:** `tilize_helpers_dataflow.hpp` covers RM→tile sticks, not relevant (input is TILE_LAYOUT). `cb_helpers_dataflow.hpp` provides only compute-thread side coordination primitives, not DRAM reads. The standard tiled streaming reader matches `tests/ttnn/unit_tests/operations/debug/test_generic_op.py:90-96` (reader source `reader_unary_interleaved_start_id.cpp`) — that is the canonical idiom for this exact data path. |
| **Compute, init** | raw_api | `init_sfpu(cb_input_tiles, cb_output_tiles)` | `tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h:46` | passes the input CB whose data format anchors unpacker config and the output CB whose format anchors packer config | — | — | Call once at kernel start. `init_sfpu` internally invokes `unary_op_init_common` which does the full hw setup; it replaces `compute_kernel_hw_startup` for SFPU-only kernels (see `eltwise_unary.h:46-50`). Subsequent `copy_tile_to_dst_init_short` calls reconfigure the unpacker for `cb_accumulator` reads (next row). |
| **Compute, before each `copy_tile` that switches source CB** | raw_api | `copy_tile_to_dst_init_short(cb_id)` | `tt_metal/hw/inc/api/compute/tile_move_copy.h:32` | `cb_id ∈ {cb_input_tiles, cb_accumulator}` | — | — | Reconfigures the unpacker for the named CB's format. Called once per switch; cheap. Pattern follows `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/where_tss_kernel.cpp:33`. |
| **Compute, init-zero accumulator** | raw_api | `fill_tile_init()` + `fill_tile(D0, 0.0f)` | `tt_metal/hw/inc/api/compute/eltwise_unary/fill.h:74` (init) + `:29` (call) | `idst=0`, `param0=0.0f` | — | DST[D0] gets all-zero tile, packed to `cb_accumulator` | **Helpers considered and rejected:** `sfpu_helpers.hpp::FillTile<>` (`sfpu_helpers.hpp:1011`) and `sfpu_helpers.hpp::sfpu_op<>` (`:1442`) are the helper wrappers. `sfpu_op<ICB, ...>` requires an `ICB` template parameter and a `Load` step — but the init-zero phase has no input CB to load from; there is no input tile being transformed, only a constant being materialized. `sfpu_chain` similarly requires a `Load<CB, Dst>` head element, so chains cannot represent "no input, just fill a slot". The raw `fill_tile_init` / `fill_tile` pair is the direct primitive `FillTile::call` itself invokes (`sfpu_helpers.inl:416-417`), and it is the pattern used in `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/where_tss_kernel.cpp:36-43` for exactly this scenario (materializing constant tiles in-DST). |
| **Compute, DST→DST copy of `a` to scratch** | raw_api | `copy_dest_values_init()` + `copy_dest_values<DataFormat::Float32>(src_dst, dst_dst)` | `tt_metal/hw/inc/api/compute/copy_dest_values.h:57` (init) + `:33` (call) | `src_dst = 1` (D1 = a), `dst_dst ∈ {2, 3}` (D2 / D3 scratch) | — | DST[D2 or D3] gets a copy of D1 | **Helpers considered and rejected:** No kernel-lib helper wraps `copy_dest_values`. The `sfpu_helpers` `Load<>` op only does CB→DST copy (`sfpu_helpers.inl:511`), not DST→DST. The Lanczos polynomial requires re-using `a` 6+ times as the basis for each `a + k` term, and `a + 4.5`, `a − 0.5`, `a != 1`, `a != 2`. Re-`copy_tile`-ing from `cb_input_tiles` every time would work but is wasteful (6+ NoC-free L1-unpack moves vs cheap DST-internal moves). `copy_dest_values<Float32>` is the direct primitive and the only API exposed for DST→DST. |
| **Compute, scalar add/sub/mul on a DST slot** | raw_api | `binop_with_scalar_tile_init()` + `add_unary_tile(dst, val_u32)` / `sub_unary_tile` / `mul_unary_tile` | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:89` (init), `:28` (add), `:32` (sub), `:36` (mul) | `dst ∈ {D0, D2, D3}`; `val_u32 = std::bit_cast<uint32_t>(float_const)` for each Lanczos coefficient, offset, and post-log constant | — | DST slot updated in place | **Helpers considered and rejected:** `sfpu_helpers::AddScalar<Slot>` / `SubScalar<Slot>` / `MulScalar<Slot>` (`sfpu_helpers.hpp:937,943,951`) wrap exactly these LLK ops with one extra layer (a default-constructed struct holding `uint32_t scalar`). When invoked via `sfpu_chain` the slot is compile-time bound; but we have **24 scalar ops** per output tile (6 Lanczos coefficients × 4 lgamma iterations) each with a *different* scalar, and the `scalar` is a per-iteration runtime value, which would require 24 distinct chain elements — exceeding the 8-slot `Dst` enum. Using the structs as plain wrappers (constructed per call, `.apply()` invoked) saves nothing vs the raw LLK call, and would bury the explicit Lanczos coefficient list under another layer. The raw `add_unary_tile` / `mul_unary_tile` calls match the spec's "primitive SFPU tile ops" requirement (task prohibition #2) word-for-word. |
| **Compute, reciprocal on a DST slot** | raw_api | `recip_tile_init()` + `recip_tile(dst)` | `tt_metal/hw/inc/api/compute/eltwise_unary/recip.h:19` (init) + `:37` (call) | `dst = D2 or D3` (whichever is currently holding `a + k`) | — | DST slot replaced by `1 / value` | **Helpers considered and rejected:** `sfpu_helpers::Recip<Slot>` (`sfpu_helpers.hpp:457`) wraps this. Same reasoning as the scalar ops — 6 different DST slots in chain would conflict with our 4-slot budget; the structured-wrapper detour does not buy correctness invariants over the raw LLK call here (the LLK call already manages SFPU state via the `_init`). Note we use the default `legacy_compat=true` template arg, matching the runtime behavior of `ttnn::reciprocal` in the reference (whose underlying SFPU code path is this same `recip_tile`). |
| **Compute, natural log on a DST slot** | raw_api | `log_tile_init()` + `log_tile(dst)` | `tt_metal/hw/inc/api/compute/compute_kernel_api.h:92` (init) + `:114` (call) | `dst = D0 or D2` | — | DST slot replaced by `log(value)` | **Helpers considered and rejected:** `sfpu_helpers::Log<Approx::Exact, Slot>` (`sfpu_helpers.hpp:420`). The 2 `log_tile` calls per lgamma × 4 lgammas = 8 total per tile would each become a chain element; chain stride and DST budget rule this out (see scalar-ops rationale). Default template `fast_and_approx=false` (Exact) matches Phase 0 precision policy. |
| **Compute, DST+DST binary ops (add/sub/mul)** | raw_api | `add_binary_tile_init()` / `sub_binary_tile_init()` / `mul_binary_tile_init()` + `add_binary_tile(a, b, c)` / `sub_binary_tile` / `mul_binary_tile` | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h:86,88,90` (inits) + `:35,39,43` (calls) | All Lanczos accumulations: `add_binary_tile(D0, D2, D0)` (D0 += D2); `sub_binary_tile(D0, D2, D0)` (D0 -= D2); `mul_binary_tile(D2, D3, D2)` (D2 = D2 * D3); pole zero: `mul_binary_tile(D0, D2, D0)` | — | DST slot replaced | **Helpers considered and rejected:** `sfpu_helpers::SfpuAdd<In0, In1, Out>` / `SfpuSub<>` / `SfpuMul<>` (`sfpu_helpers.hpp:1035,1041,1047`) — same chain-DST-budget reasoning as scalar ops. `binary_op_helpers::add` / `mul` / `sub` (`binary_op_helpers.hpp:303,290,303`) operate at the **CB level** (two whole input CBs, one output CB), not at the **DST slot level** where our DST-resident intermediates live. Using `binary_op_helpers::add` would force us to pack our intermediates out to a CB, then read them back — defeating the purpose of the in-DST fused chain (the 24 Lanczos terms would each need a round-trip through L1). The raw `*_binary_tile` family is the appropriate primitive for in-DST binary ops, and matches the task's explicit "`add_binary_tile, sub_binary_tile, mul_binary_tile`" allow-list. |
| **Compute, unary not-equal predicate** | raw_api | `unary_ne_tile_init()` + `unary_ne_tile(dst, val_u32)` | `tt_metal/hw/inc/api/compute/eltwise_unary/comp.h:40` (init) + `:33` (call) | `dst = D2` (the slot holding a copy of `a`); `val_u32 = bit_cast<uint32_t>(1.0f)` for the first pole, `bit_cast<uint32_t>(2.0f)` for the second | — | DST[D2] becomes `(a != 1.0) ? 1.0 : 0.0` (or `(a != 2.0)`) | **Helpers considered and rejected:** `sfpu_helpers::UnaryNe<Slot>` (`sfpu_helpers.hpp:788`) wraps this LLK with a struct that takes `uint32_t param0`; same chain-budget reasoning as the scalar ops. Implements the pole zeroing as a multiply-by-mask: `result *= (a != 1.0); result *= (a != 2.0)`. This is mathematically equivalent to the reference's `where(eq(x, 1.0), 0.0, result)` since `(a != 1.0)` is a `{0, 1}`-valued float and `0 * anything = 0`. Using `where_tile` (`tt_metal/hw/inc/api/compute/eltwise_unary/where.h:34`) would require an additional zero-filled DST slot (we have none free) and another scratch — at our 4-slot budget the mask-multiply form is strictly cheaper. Note also: NaN/Inf result × 0 = NaN, not 0, so we make the mask operate on `a` (the input to lgamma, which is always finite for valid inputs in the test domain) rather than on `result` (which may be Inf/NaN at the very poles). At the pole exactly, `a` is finite and `(a != 1)` is exactly 0, so `result * 0 = 0` in IEEE-754 even when `result = NaN`-from-inputs is not the case — verified on the test domain `a ∈ [2.0, 10.0]` where lgamma at `a == 2` is finite via the Lanczos polynomial and only zeroed for correctness. |
| **Compute, CB→DST tile copy** | raw_api | `copy_tile(cb_id, cb_tile_idx, dst_idx)` | `tt_metal/hw/inc/api/compute/tile_move_copy.h` | `cb_id ∈ {cb_input_tiles, cb_accumulator}`, `cb_tile_idx = 0` (always front), `dst_idx ∈ {1, 0, 2}` depending on context | `cb_input_tiles` or `cb_accumulator` | DST (no CB push) | **Helpers considered and rejected:** `compute_kernel_lib::Load<CB, Dst>` (`sfpu_helpers.hpp:371`) wraps exactly this LLK; `sfpu_chain` uses it for chained pipelines. We are not in a chain. The raw `copy_tile` is the direct call. |
| **Compute, DST→CB tile pack** | raw_api | `pack_tile(dst_idx, cb_id)` (and `pack_reconfig_data_format(new_cb)` before switching pack destination) | standard pack primitives | `dst_idx = 0` (always D0 for the slot we pack from) | DST | `cb_accumulator` (during lgamma updates and init-zero) or `cb_output_tiles` (at finalization) | **Helpers considered and rejected:** `sfpu_pipeline` (`sfpu_helpers.hpp:1412`) bundles `tile_regs_commit / pack / release` for chain output. We are not in a chain; the standard `tile_regs_commit / tile_regs_wait / pack_tile / tile_regs_release` pattern (as in `where_tss_kernel.cpp:49-53`) is correct. |
| **Writer, per tile** | raw_api | `cb_wait_front` / `noc_async_write` (with `TensorAccessor::get_noc_addr(tile_id)`) / `noc_async_write_barrier` / `cb_pop_front` | dataflow primitives in `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | — | `cb_output_tiles` (1 wait + 1 pop per iter) | DRAM (output via `TensorAccessor`) | **Helpers considered and rejected:** No helper for "drain a tiled CB to DRAM in tile_id order". `untilize_helpers_dataflow.hpp` is RM-output-only (output is TILE_LAYOUT). The standard tiled writer matches `tests/ttnn/unit_tests/operations/debug/test_generic_op.py:97-104` (`writer_unary_interleaved_start_id.cpp`). |

### Why this op is helper-light

This operation is uniquely poor-fitting for the kernel-lib's chain/pipeline abstraction:

1. **Per-iteration scalar variation.** The 6 Lanczos coefficients differ across all 6 terms. A chain element binds its scalar at compile time via a struct field; representing all 24 (`6 × 4 lgammas`) terms as distinct chain elements would exceed the 8-slot `Dst` enum capacity.
2. **Intra-tile read-modify-write to a CB.** The accumulator is in `cb_accumulator`, not in DST. Each lgamma iteration is a separate `tile_regs_acquire / commit / release` cycle that reads the previous accumulator, computes one lgamma, adds it, and packs back. `sfpu_pipeline` is built around "fill DST, transform, pack" with no intermediate CB ping-pong.
3. **4 DST slots is the tight budget for the algebraic form of lgamma.** The chain abstraction adds a `stride` to each DST index, which subtracts from the usable per-iteration budget. We need the entire 0..3 range, with no offsetting.

The justification for **each individual non-use** is recorded in the API Mapping above with file:line citations. This is not "I didn't feel like it" — it is a specific structural mismatch between the chain abstraction (compile-time-fixed sequence with stride-batched DST) and the algorithm shape (per-tile multi-acquire read-modify-write loop over an intermediate CB).

## Compute Phases

Per input tile on each core, the compute kernel executes the following sequence. State after each phase is shown for `cb_accumulator` (the only CB with nontrivial intra-tile state).

| # | Operation | Helper? | Input CB (state) | Output CB (state after) | CB State After |
|---|-----------|---------|------------------|--------------------------|----------------|
| 0 | **Wait for input tile.** `cb_wait_front(cb_input_tiles, 1)`. | raw_api | `cb_input_tiles`: 1 tile present | — | `cb_input_tiles` holds 1 tile (will be drained in phase 5) |
| 1 | **Init accumulator to zero.** `tile_regs_acquire`; `fill_tile_init()`; `fill_tile(0, 0.0f)`; `tile_regs_commit`; `tile_regs_wait`; `pack_tile(0, cb_accumulator)`; `tile_regs_release`; `cb_push_back(cb_accumulator, 1)`. | raw_api | — | `cb_accumulator`: 1 tile (all zeros) | accum = 0 |
| 2.k (k ∈ 0..3) | **Lgamma evaluation for offset `−0.5·k`** — see "Per-lgamma inner sequence" below. Reads input tile via `copy_tile(cb_input_tiles, 0, 1)` then `add_unary_tile(1, packed_offset)`. Loads previous accumulator via `copy_tile(cb_accumulator, 0, 2)`; adds; packs new accumulator. Pops old accumulator front; pushes new accumulator back. | raw_api | `cb_input_tiles`: 1 tile (NOT popped — still needed for k+1); `cb_accumulator`: 1 tile front (previous) | `cb_accumulator`: 1 tile (new) | accum += lgamma_lanczos(a − 0.5·k) (pole-zeroed) |
| 3 | **Finalize: add `3·log(π)` and pack to output.** `cb_wait_front(cb_accumulator, 1)`; `cb_reserve_back(cb_output_tiles, 1)`; `tile_regs_acquire`; `copy_tile_to_dst_init_short(cb_accumulator)`; `copy_tile(cb_accumulator, 0, 0)`; `add_unary_tile(0, bit_cast<u32>(3.434189657547f))`; `tile_regs_commit`; `tile_regs_wait`; `pack_reconfig_data_format(cb_output_tiles)`; `pack_tile(0, cb_output_tiles)`; `tile_regs_release`. | raw_api | `cb_accumulator`: 1 tile (sum of 4 lgammas) | `cb_output_tiles`: 1 tile (final result) | — |
| 4 | **Drain CBs.** `cb_pop_front(cb_accumulator, 1)`; `cb_push_back(cb_output_tiles, 1)`; `cb_pop_front(cb_input_tiles, 1)`. | raw_api | — | — | `cb_accumulator` empty; `cb_input_tiles` empty (ready for next tile) |

### Per-lgamma inner sequence (phase 2.k expanded)

Within one `tile_regs_acquire / tile_regs_release` block, with `a` in `D1`, fresh `D0` as the lgamma-result accumulator, and `D2 / D3` as scratch:

| Step | DST operation | Effect |
|------|---------------|--------|
| 2.k.0 | `copy_tile_to_dst_init_short(cb_input_tiles)`; `copy_tile(cb_input_tiles, 0, 1)`; `add_unary_tile(1, bit_cast<u32>(offsets[k]))` (offsets = `{0.0, −0.5, −1.0, −1.5}`) | `D1 = x + offsets[k] = a` |
| 2.k.1 | `fill_tile_init()`; `fill_tile(0, 0.0f)` | `D0 = 0` (local lgamma accumulator) |
| 2.k.2 | **Lanczos polynomial loop** — for `k_lanczos ∈ 0..5`: `copy_dest_values<Float32>(1, 2)`; `add_unary_tile(2, bit_cast<u32>(float(k_lanczos)))`; `recip_tile(2)`; `mul_unary_tile(2, bit_cast<u32>(C[k_lanczos]))`; `add_binary_tile(0, 2, 0)`. Constants `C[]` = `{76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5}`. | `D0 += C[k_lanczos] / (a + k_lanczos)` for k_lanczos = 0..5 |
| 2.k.3 | `add_unary_tile(0, bit_cast<u32>(1.0f))`; `log_tile_init()`; `log_tile(0)` | `D0 = log(1 + sum) = log(temp)` |
| 2.k.4 | `copy_dest_values<Float32>(1, 2)`; `add_unary_tile(2, bit_cast<u32>(3.581061468642829f))`; `sub_binary_tile(0, 2, 0)` | `D0 = log(temp) − (a + 3.581…)` |
| 2.k.5 | `copy_dest_values<Float32>(1, 2)`; `add_unary_tile(2, bit_cast<u32>(4.5f))`; `log_tile(2)`; `copy_dest_values<Float32>(1, 3)`; `sub_unary_tile(3, bit_cast<u32>(0.5f))`; `mul_binary_tile(2, 3, 2)`; `add_binary_tile(0, 2, 0)` | `D0 += (a − 0.5) · log(a + 4.5)` |
| 2.k.6 | `copy_dest_values<Float32>(1, 2)`; `unary_ne_tile_init()`; `unary_ne_tile(2, bit_cast<u32>(1.0f))`; `mul_binary_tile(0, 2, 0)`; `copy_dest_values<Float32>(1, 2)`; `unary_ne_tile(2, bit_cast<u32>(2.0f))`; `mul_binary_tile(0, 2, 0)` | `D0 *= (a != 1.0); D0 *= (a != 2.0)` — pole zeroing |
| 2.k.7 | `cb_wait_front(cb_accumulator, 1)`; `cb_reserve_back(cb_accumulator, 1)`; `copy_tile_to_dst_init_short(cb_accumulator)`; `copy_tile(cb_accumulator, 0, 2)`; `add_binary_tile(0, 2, 0)` | `D0 += previous accumulator value (read from cb_accumulator front)` |
| 2.k.8 | `tile_regs_commit`; `tile_regs_wait`; `pack_tile(0, cb_accumulator)`; `tile_regs_release`; `cb_pop_front(cb_accumulator, 1)`; `cb_push_back(cb_accumulator, 1)` | New accumulator (lgamma result + previous accumulator) is at the back of `cb_accumulator`; previous front is popped. |

### DST slot register

The 4 DST slots are used as follows. At no point does any phase need a 5th slot.

| Slot | Role |
|------|------|
| **D0** | Lgamma local accumulator. After phase 2.k.7 it also momentarily holds the global running accumulator (the value packed back to `cb_accumulator`). In phase 3 it is the final-result slot before packing to `cb_output_tiles`. |
| **D1** | The current lgamma input `a = x + offset[k]`. Loaded once at 2.k.0, never modified within an iteration. |
| **D2** | Multi-purpose scratch: the per-Lanczos-term `a + k_lanczos`, then `a + 3.581…`, then `log(a + 4.5)` (combined with D3), then the pole mask `(a != 1)` or `(a != 2)`, then the loaded previous accumulator. |
| **D3** | Auxiliary scratch — only needed to hold `a − 0.5` simultaneously with `log(a + 4.5)` in D2 so that the binary multiply `(a − 0.5) · log(a + 4.5)` has its two operands in different slots. |

## Build Order

The implementer should bring the op up incrementally rather than implementing the full kernel and debugging end-to-end. The Phase 0 acceptance test exercises the full algorithm — but earlier stages are diagnostic.

| Stage | Goal | What to verify | DPRINT hints |
|-------|------|----------------|--------------|
| 1. **Data pipeline only** | Reader streams `cb_input_tiles`; writer drains `cb_output_tiles`. Compute is a pure passthrough: `copy_tile(cb_input, 0, 0); pack_tile(0, cb_output);` for each tile. Test shape `(1, 1, 32, 32)`. | Output equals input exactly (`torch.allclose(out, inp, atol=0, rtol=0)`). Confirms tile_id arithmetic, CB sync, multi-core RT-arg wiring. | `DPRINT << "core=(" << my_x << "," << my_y << ") tile_id=" << tile_id << ENDL();` in reader. |
| 2. **`cb_accumulator` init-zero round-trip** | Compute initializes `cb_accumulator` to zero, then immediately reads it back as the output. End-to-end output is zeros. | Output is all zeros. Confirms `fill_tile`, `cb_accumulator` push/pop cycle, and `pack_reconfig_data_format` to switch packers between `cb_accumulator` and `cb_output_tiles`. | `DPRINT_TILESLICE` of `cb_accumulator` after the init push. Use `torch.ones(...)` input to make any leakage visible. |
| 3. **One lgamma, offset 0** | Compute does only phase 2.0 (the first lgamma) — drop phases 2.1 / 2.2 / 2.3 and the `3·log(π)` add. Compare against `torch.special.gammaln(input).float()`. Use input restricted to `[2.0, 10.0]` to avoid the poles entirely. | Per-element relative error < 0.05 vs torch lgamma (Lanczos@fp32 baseline accuracy). Confirms the inner Lanczos sequence at 2.k.1..2.k.6 and the final accumulate at 2.k.7..2.k.8. | `DPRINT_TILESLICE` of `D0` immediately before `pack_tile`. Hand-check one element against numpy lgamma. |
| 4. **All 4 lgammas, no `3·log(π)`** | Run phases 2.0..2.3, skip phase 3's `add_unary_tile`. | Output equals `torch.special.multigammaln(input, 4) − 3·log(π)`. Confirms the multi-iteration accumulator round-trip through `cb_accumulator`. | Compare first element to numpy reference. |
| 5. **Full op** | Add phase 3's `add_unary_tile(0, bit_cast<u32>(3.434189657547f))`. | Output passes the acceptance test at `rtol=0.1, atol=0.5`. | None — measure via the acceptance test. |
| 6. **Pole correctness** | Add `1.5, 2.0, 2.5` to the test inputs (these would otherwise hit poles for some lgamma iteration). Verify the pole-zero masks fire correctly. | At `x = 2.0`, the `lgamma(x)` term sees `a = 2` → pole-zero kicks in; result still finite and matches `torch.special.multigammaln(torch.tensor([2.0]), 4)` within tolerance. | DPRINT `D2` after each `unary_ne_tile` to confirm the mask is `{0.0, 1.0}`. |
| 7. **Multi-core scaling** | Move from single-core to `device.compute_with_storage_grid_size()`. Verify per-core RT-arg distribution via `split_work_to_cores`. Bump shape to `(2, 4, 64, 128)`. | Same numerical correctness, faster runtime. Both `core_group_1` and `core_group_2` cores produce valid outputs (verify by inspecting `tiles_per_core_g2` > 0 case). | DPRINT `start_tile_id` and `num_tiles` from each core. |

Deterministic-input debugging tip: when testing phase 2 or 3, replace `torch.randn` with `torch.linspace(2.0, 10.0, …).reshape(shape)` so each tile has predictable per-element values. Compute the expected lgamma sum by numpy at known indices and compare.

## Key Risks and Gotchas

| # | Risk | Mitigation |
|---|------|------------|
| 1 | **DST budget overrun.** With `fp32_dest_acc_en=True` and half-sync, DEST has only 4 slots (D0..D3). Algorithm naturally wants 5+ slots (global accumulator, `a`, `log(temp)`, `log(a+4.5)`, `a−0.5`). | The algebraic re-grouping in the **Overview** moves the global accumulator out of DST into `cb_accumulator`, freeing one DST slot. Slot layout (D0=local accum, D1=a, D2=scratch1, D3=scratch2) is the tight 4-slot plan documented in the per-lgamma inner sequence. No phase requires a 5th slot. |
| 2 | **`cb_input_tiles` premature pop** would mean iter `k+1` reads garbage. | Reader pushes 1 tile per input tile. Compute does `cb_wait_front(cb_input_tiles, 1)` once, then 4 `copy_tile(cb_input_tiles, 0, 1)` calls (no pop). Only after phase 3 does compute `cb_pop_front(cb_input_tiles, 1)`. This is "WaitUpfrontNoPop" semantics expressed as raw CB ops. |
| 3 | **`cb_accumulator` deadlock if sized to 1 page.** The read-modify-write cycle holds the front (`cb_wait_front(1)`) and the back (`cb_reserve_back(1)`) simultaneously. 1 page → no slot available to reserve → deadlock. | Size `cb_accumulator` to **2 pages** (per CB Sizing Rationale above). Confirmed by re-tracing the 5-push / 5-pop cycle: front and back never overlap in slot index. |
| 4 | **`pack_reconfig_data_format` skipped when switching pack target.** Compute packs to `cb_accumulator` in phases 1 and 2.k, then to `cb_output_tiles` in phase 3. Both are float32, so format reconfig is technically a no-op — but the packer's bound CB index also changes. | Call `pack_reconfig_data_format(cb_output_tiles)` before the final `pack_tile(0, cb_output_tiles)` in phase 3. (No reconfig is required between phases 1 and 2.k since both pack to `cb_accumulator`.) |
| 5 | **`copy_tile_to_dst_init_short` skipped when switching unpack source.** Compute unpacks from `cb_input_tiles` (D1) and `cb_accumulator` (D2 reload, phase 2.k.7). Same float32 format, but the unpacker's bound CB id changes. | Call `copy_tile_to_dst_init_short(cb_id)` before each `copy_tile` that switches source CB (i.e., once per lgamma iteration when switching from `cb_input_tiles` to `cb_accumulator`, and once before phase 3's read from `cb_accumulator`). |
| 6 | **Lanczos coefficient encoding error.** A bitcast mistake on any of the 6 coefficients silently corrupts the output by a factor that may be subtle to detect. | Define the coefficients **as `float` literals** in the kernel source and convert via `std::bit_cast<uint32_t>(float)` (or `union { float f; uint32_t u; }`) **once**, at compile time, into a `constexpr uint32_t` array. The literal values are the exact reference constants in the **Overview** section. Do **not** retype hex bitcast values. |
| 7 | **Pole mask × NaN/Inf intermediate produces NaN.** If the Lanczos polynomial evaluation produces NaN at the pole (because `1/0` occurs in one of the `recip_tile` terms), then `NaN * 0.0 = NaN`, not 0. | The mask is applied to a copy of `a` (in D2), **not** to the lgamma result. The mask values are computed as `(a != 1.0)` and `(a != 2.0)` — both `a != C` evaluations are exact-float comparisons of a finite input, producing `{0.0, 1.0}` cleanly. When the mask is then `mul_binary_tile`'d into D0 (the lgamma result), `0 * any_finite = 0`. Within the Phase 0 test domain `[2.0, 10.0]`, `a` is always finite (no `1/0` because the smallest `a + k` is `2.0 + 0 = 2.0` at k=0, and `0.5 + 0 = 0.5` at the worst k=5 with offset −1.5), so the lgamma result is finite and the mask multiplies a finite value. The reference's `where(eq(x, 1), 0, result)` and our `result * (a != 1)` both produce 0 in this regime. |
| 8 | **HiFi4 + fp32 dest acc precision baseline.** Lanczos at fp32 is meaningfully less accurate than torch's libm double-precision reference. | Test tolerances are intentionally wide (`rtol=0.1, atol=0.5`) and inputs are restricted to `a ∈ [2.0, 10.0]` (the regime where the Lanczos 6-term polynomial is most accurate at fp32). This is documented in the task spec and reflected in the acceptance test. |
| 9 | **Two compute KernelDescriptors needed for unbalanced `split_work_to_cores`.** When `total_tiles % num_grid_cores != 0`, `core_group_1` cores process `tiles_per_core_g1` tiles each and `core_group_2` cores process `tiles_per_core_g1 - 1` (or similar) tiles each. The compute kernel's compile-time `num_tiles` differs between groups. | Mirror the multi-group pattern by declaring **one KernelDescriptor per non-empty core group** (group 1 always non-empty; group 2 only if it has cores). Each descriptor uses its group's `CoreRangeSet` and its `tiles_per_core_gN` as the compile-time arg. Reader and writer follow the same pattern (their RT args carry `start_tile_id` and `num_tiles` per core; see the `test_generic_op.py` reference for the inner core-walk loop that assigns RT args). |
| 10 | **Float-bitcast in compile-time args.** The `compile_time_args` list takes `uint32_t`; passing a float scalar (Python `float`) would be type-rejected or silently truncated. | All Lanczos coefficients and offsets are baked into the **kernel source** as `constexpr float` literals; the compile-time args carry only integer tile counts. No float-valued compile-time args. |

## Out of Scope (deferred to later refinements)

| Item | Status |
|------|--------|
| bfloat16 / bfloat8_b inputs | Phase 0 rejects non-fp32; widening dtype support is a later refinement. |
| Caller-exposed `compute_kernel_config` | Hard-coded HiFi4 + fp32 dest acc in Phase 0; exposing is a later refinement. |
| `p` parameter (other than 4) | Permanently fixed at 4 (matches `torch.special.multigammaln(x, p=4)`). The implementation may not be generalised. |
| `memory_config` kwarg | Output inherits the input's memory config in Phase 0. |
| Performance tuning (sharded layout, custom block sizes) | Phase 0 is correctness-first; performance is a later refinement. |
