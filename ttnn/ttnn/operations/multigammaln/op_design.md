# Operation Design: multigammaln

## Overview

| Field | Value |
|-------|-------|
| Classification | compute |
| Goal | Elementwise multivariate log-gamma function at order p = 4. For each scalar `a` in the input tensor, produce `lgamma(a) + lgamma(a - 0.5) + lgamma(a - 1.0) + lgamma(a - 1.5) + 3·log(π)`. Order is hard-coded; no `p` argument is exposed. |
| Math | `output[idx] = lgamma(a) + lgamma(a − 0.5) + lgamma(a − 1.0) + lgamma(a − 1.5) + 3.434189657547`, where `a = input[idx]`. Domain `a > 1.5`; outside the domain, the math falls through to NaN naturally — no input branching. |
| Mode | Derivative (matches `torch.special.multigammaln(x, p=4)`) |
| References | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/lgamma_kernel.cpp` (the fp32 Stirling + reflection-adjusted recipe replicated four times here), `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp` (helper chain used for the final 4-way sum + constant), `ttnn/ttnn/operations/backward_softmax/` (Phase-0 fp32 + HiFi4 + work-distribution pattern), `ttnn/ttnn/operations/toy_tilize_untilize/` (generic_op program-descriptor scaffolding) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | `float32`, `TILE_LAYOUT`, rank 4 `(N, C, H, W)`, `H % 32 == 0`, `W % 32 == 0`, on-device (DRAM or L1 interleaved) | — | runtime (buffer addr + tile range) |

`p` is **not** a parameter; the order is wired to 4 at the kernel level. The compute config (HiFi4, `fp32_dest_acc_en=True`) is hard-coded inside `create_program_descriptor` and not exposed.

### Compute Config (hard-coded internally — NOT a caller parameter)

| Field | Value |
|-------|-------|
| `math_fidelity` | `ttnn.MathFidelity.HiFi4` |
| `fp32_dest_acc_en` | `True` |
| Effective DEST capacity | 4 tiles (half-sync + fp32 acc) — kernel-lib helpers honor this via `DEST_AUTO_LIMIT` (`ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:88–102`) |

Wired in `create_program_descriptor` as `ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)` on the compute kernel descriptor (pattern from `ttnn/ttnn/operations/backward_softmax/backward_softmax_program_descriptor.py:261–264`).

## Tensors

### Input — `input_tensor`

| Property | Requirement |
|----------|-------------|
| Shape | `(N, C, H, W)` — rank == 4 (validated). |
| Dtype | `float32` (validated). Other dtypes → `ValueError`. |
| Layout | `TILE_LAYOUT` (validated). `ROW_MAJOR_LAYOUT` → `ValueError`. |
| Memory | DRAM or L1 interleaved |
| Tile-alignment | `H % 32 == 0`, `W % 32 == 0`. Non-aligned → `ValueError`. |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to `input_tensor.shape` |
| Dtype | `float32` |
| Layout | `TILE_LAYOUT` |
| Memory | inherited from `memory_config` arg, defaults to DRAM interleaved |

## Validation (Python side, before launch)

| Check | Failure |
|-------|---------|
| `input_tensor.dtype == ttnn.float32` | `ValueError` |
| `input_tensor.layout == ttnn.TILE_LAYOUT` | `ValueError` |
| `len(input_tensor.shape) == 4` | `ValueError` (Phase 0 requires rank 4) |
| `input_tensor.shape[-1] % 32 == 0 and input_tensor.shape[-2] % 32 == 0` | `ValueError` |
| `input_tensor.storage_type() == ttnn.StorageType.DEVICE` | `RuntimeError` (must be on device — caller already on-device for `generic_op`) |

Domain validation (`a > 1.5`) is **not** performed in Python. Out-of-domain inputs naturally produce NaN via the lgamma reflection (matches `torch.special.multigammaln`).

## Dataflow Strategy

`multigammaln` is purely elementwise — every output tile depends on exactly one input tile at the same logical tile id. There is no reduction, no broadcast, and no cross-tile data dependency.

| Stage | Role | Data path |
|-------|------|-----------|
| **DRAM → reader** | NCRISC reader streams the per-core slice of input tiles from DRAM into `cb_input_tiles`, one tile at a time. | `input_tensor` DRAM → `cb_input_tiles` |
| **Compute, sub-phase A** | TRISCs hold the input tile (no pop), run the full fp32-precision `lgamma` recipe four times in succession on `(a - 0.0)`, `(a - 0.5)`, `(a - 1.0)`, `(a - 1.5)`. Each invocation packs its scalar lgamma result to a dedicated per-term intermediate CB (`cb_lgamma_a`, `cb_lgamma_a_half`, `cb_lgamma_a_one`, `cb_lgamma_a_three_halves`). After the four sub-phases, the input tile is popped. | `cb_input_tiles` → `cb_lgamma_a`, `cb_lgamma_a_half`, `cb_lgamma_a_one`, `cb_lgamma_a_three_halves` (one tile each per input tile) |
| **Compute, sub-phase B** | TRISCs run an SFPU chain that loads all four lgamma terms into D0..D3, sums them into D0 with three `SfpuAdd` ops, and adds the compile-time constant `3·log(π)` (≈ `3.434189657547f`) with `AddScalar`. The result is packed into `cb_output_tiles`. | `cb_lgamma_a`, `cb_lgamma_a_half`, `cb_lgamma_a_one`, `cb_lgamma_a_three_halves` → `cb_output_tiles` |
| **compute → writer** | BRISC writer drains `cb_output_tiles` and writes each tile back to DRAM at the matching logical tile id. | `cb_output_tiles` → output-tensor DRAM |

**No inter-Tensix communication.** The work is embarrassingly parallel across output tiles; each core processes its assigned slice end-to-end.

**Tensor format does not change.** Inputs arrive tiled (TILE_LAYOUT), every CB holds tiles, the output is written tiled. No tilize/untilize needed.

**Why four intermediate CBs (not one):** the final sum is expressed through the SFPU helper chain `sfpu_chain(Load<cb_a, D0>, Load<cb_a_half, D1>, Load<cb_a_one, D2>, Load<cb_a_three_halves, D3>, SfpuAdd<...>×3, AddScalar)`. `CompactLoad` (`sfpu_helpers.inl:507–515`) issues one `cb_wait_front(CB, 1) + copy_tile(CB, 0, slot) + cb_pop_front(CB, 1)` per CB, so each per-tile DEST-slot load must come from a distinct CB. A single 4-page accumulator CB would copy the SAME head tile into four slots — wrong.

## Work Distribution

The work unit is **one output tile** (equivalently, one input tile). Every tile produces one output tile independently.

| Field | Value |
|-------|-------|
| Work unit | One 32×32 fp32 tile. |
| Total tiles | `total_tiles = (N · C · H · W) / (32 · 32)` — also retrievable as `input_tensor.buffer_num_pages()`. |
| Grid | `device.compute_with_storage_grid_size()` (full Tensix compute grid available). |
| Per-core work | `pages_per_core_g{1,2}` tiles, computed by `ttnn.split_work_to_cores(grid_size, total_tiles)` (`ttnn/ttnn/__init__.py` / Python utility binding documented in `.claude/references/ttnn-python-utility-bindings.md`). Returns `(num_cores, all_cores, core_group_1, core_group_2, pages_per_core_g1, pages_per_core_g2)`. |
| Remainder | Handled by `ttnn.split_work_to_cores`'s two-group split. Each group has a uniform per-core tile count differing by at most 1. |
| Per-core RT args | Reader: `(input_addr, start_tile_id, num_tiles_for_this_core)`. Writer: `(output_addr, start_tile_id, num_tiles_for_this_core)`. Compute: `(num_tiles_for_this_core)`. `start_tile_id` is the cumulative tile offset for that core (assigned by walking `core_group_1` then `core_group_2`). |

`all_cores` (the union) is used for CB descriptors. Reader/writer/compute KernelDescriptors are placed on `all_cores`; the per-core RT arg arrays are populated by iterating `core_group_1` (each core gets `pages_per_core_g1`) and then `core_group_2` (each core gets `pages_per_core_g2`), tracking a running `start_tile_id`.

## Circular Buffers

CB index convention (`.claude/references/op-design-template.md` and `.claude/references/generic_op_template/template_op_program_descriptor.py:74–76`): `0–7` for input-tensor CBs, `16–23` for output-tensor CBs, `24–31` for intermediates.

All CBs are float32, page size `ttnn.tile_size(ttnn.float32) = 4096 B`.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_tiles` | 0 | `tile_size(float32)` = 4096 B | 2 (double-buffer) | float32 | reader | compute (lgamma sub-phases — held with explicit `cb_wait_front(..., 1)` then `cb_pop_front(..., 1)` after all four lgammas of a tile are done) | per-tile streaming |
| `cb_lgamma_a` | 24 | `tile_size(float32)` = 4096 B | 2 (double-buffer) | float32 | compute (lgamma sub-phase, offset 0.0) | compute (sum sub-phase, Load slot D0) | per-tile streaming; drained one tile per input tile |
| `cb_lgamma_a_half` | 25 | `tile_size(float32)` = 4096 B | 2 (double-buffer) | float32 | compute (lgamma sub-phase, offset 0.5) | compute (sum sub-phase, Load slot D1) | same |
| `cb_lgamma_a_one` | 26 | `tile_size(float32)` = 4096 B | 2 (double-buffer) | float32 | compute (lgamma sub-phase, offset 1.0) | compute (sum sub-phase, Load slot D2) | same |
| `cb_lgamma_a_three_halves` | 27 | `tile_size(float32)` = 4096 B | 2 (double-buffer) | float32 | compute (lgamma sub-phase, offset 1.5) | compute (sum sub-phase, Load slot D3) | same |
| `cb_output_tiles` | 16 | `tile_size(float32)` = 4096 B | 2 (double-buffer) | float32 | compute (sum sub-phase) | writer | per-tile streaming |

### CB sizing rationale

- **`cb_input_tiles` = 2 pages.** Streaming reader → compute consumer with full pipelining (reader is on NCRISC, compute on TRISCs — different processors that pipeline naturally). Standard double buffer (`.claude/references/ttnn-cb-memory-fundamentals.md` "double-buffered streaming CB" pattern).
- **`cb_lgamma_*` = 2 pages each.** Producer and consumer are both the **same compute kernel** running sequentially (sub-phase A produces, sub-phase B consumes within the same tile iteration), but the per-tile push/pop is exactly 1, and double-buffering allows the writer-facing `cb_output_tiles` push of one iteration to overlap with the next iteration's lgamma sub-phases without forcing a back-pressure stall. Since the sequence is `push 1 → push 1 → push 1 → push 1 → pop 1 + pop 1 + pop 1 + pop 1` per tile, 2 pages is the minimum-safe size (1 page would still work but loses the headroom; 2 pages is the standard double-buffer convention).
- **`cb_output_tiles` = 2 pages.** Streaming compute → writer with full pipelining (writer on BRISC; different processor). Standard double buffer.

### CB sync verification (per output tile, single core)

| CB | Producer pushes | Consumer waits | Consumer pops | Match? |
|----|-----------------|----------------|---------------|--------|
| `cb_input_tiles` | reader: 1 per tile | compute: `cb_wait_front(cb_input_tiles, 1)` once at the start of the tile; reused across all 4 lgamma sub-phases without intermediate pops | compute: `cb_pop_front(cb_input_tiles, 1)` once at the end of the tile | ✓ (1 push ↔ 1 pop per tile; 4 reads of the same in-front tile permitted without pop) |
| `cb_lgamma_a` | compute: 1 per tile | sum-phase Load: 1 per tile | sum-phase CompactLoad pops: 1 per tile (`sfpu_helpers.inl:512–514`, `DoPop=true` after compact-fold) | ✓ |
| `cb_lgamma_a_half` | compute: 1 per tile | sum-phase Load: 1 per tile | sum-phase CompactLoad pops: 1 per tile | ✓ |
| `cb_lgamma_a_one` | compute: 1 per tile | sum-phase Load: 1 per tile | sum-phase CompactLoad pops: 1 per tile | ✓ |
| `cb_lgamma_a_three_halves` | compute: 1 per tile | sum-phase Load: 1 per tile | sum-phase CompactLoad pops: 1 per tile | ✓ |
| `cb_output_tiles` | compute: 1 per tile | writer: 1 per tile | writer: 1 per tile | ✓ |

## API Mapping

Every operation in the compute kernel is enumerated here with file:line citations. Sub-phase A (the per-offset lgamma recipe) uses raw APIs because the kernel-lib `Lgamma<>` helper does **not** cover the fp32 high-precision path with reflection adjustment (see `Helpers considered and rejected` block below). Sub-phase B (the 4-way sum + constant) uses the SFPU helper chain exclusively.

### Helpers used

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|------------------------|--------------------------|---------------------------|--------------|
| Kernel hardware init | raw_api | `init_sfpu(icb, ocb)` | `tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h` (declaration) — invoked per pattern documented in `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:72–73` ("PREREQUISITE: Call `init_sfpu(icb, ocb)` …") | `(cb_input_tiles, cb_output_tiles)` | — | — | Must be called exactly once at the top of `kernel_main`, before any `tile_regs_acquire` / SFPU op. No helper wraps this; it is the standard kernel prologue. |
| Sub-phase B (per-tile sum + constant) | helper | `compute_kernel_lib::sfpu_pipeline` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:1406–1412` | `sfpu_pipeline<SfpuBatching::Auto, SfpuInputPolicy::WaitAndPopPerTile, SfpuOutputPolicy::PerTile, SfpuDataFormatReconfig::INPUT_AND_OUTPUT>(chain, cb_output_tiles, /*num_tiles=*/1)` (called once per input tile) | `cb_lgamma_a`, `cb_lgamma_a_half`, `cb_lgamma_a_one`, `cb_lgamma_a_three_halves` (via chain Loads) | `cb_output_tiles` | Manages `tile_regs_acquire/commit/wait/release`, the per-CB `cb_wait_front`/`cb_pop_front`, the output `cb_reserve_back`/`cb_push_back`, and data-format reconfig. The chain has stride 4 (D0..D3); with `fp32_dest_acc_en=True`, `DEST_AUTO_LIMIT = 4`, so `batch_size = 4/4 = 1` — one tile per pipeline call is the correct (and only) choice. |
| Sub-phase B chain composition | helper | `compute_kernel_lib::sfpu_chain` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:1364–1371` | `sfpu_chain(Load<cb_lgamma_a, Dst::D0>{}, Load<cb_lgamma_a_half, Dst::D1>{}, Load<cb_lgamma_a_one, Dst::D2>{}, Load<cb_lgamma_a_three_halves, Dst::D3>{}, SfpuAdd<Dst::D0, Dst::D1, Dst::D0>{}, SfpuAdd<Dst::D0, Dst::D2, Dst::D0>{}, SfpuAdd<Dst::D0, Dst::D3, Dst::D0>{}, AddScalar<Dst::D0>{0x405BA32Eu /*≈ 3.434189657547f, 3·log(π)*/})` | — | — | Each `Load` from a distinct CB stays a separate `CompactLoad` (compaction only merges adjacent loads from the **same** CB — `sfpu_helpers.hpp:1234–1287`). The `NoMultiGroupCB` static_assert (`sfpu_helpers.hpp:1366–1369`) passes because each CB appears in exactly one Load. |
| Sub-phase B Load | helper | `compute_kernel_lib::Load<CB, Slot>` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:370–376` | One `Load<cb_lgamma_*, Dst::Dn>{}` per term (4 total). | `cb_lgamma_*` | DEST | After chain compaction each load becomes `CompactLoad<CB, /*DoWait=*/true, /*DoPop=*/true, Slot>` (`sfpu_helpers.hpp:390–405`). |
| Sub-phase B addition | helper | `compute_kernel_lib::SfpuAdd<In0, In1, Out>` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:1035–1039` | Three instances: `SfpuAdd<Dst::D0, Dst::D1, Dst::D0>{}`, `SfpuAdd<Dst::D0, Dst::D2, Dst::D0>{}`, `SfpuAdd<Dst::D0, Dst::D3, Dst::D0>{}` | DEST | DEST | Each one re-runs `add_binary_tile_init()` (`sfpu_helpers.inl` — same pattern as other binary SFPU helpers via `BinaryOp` CRTP at `sfpu_helpers.hpp:297–313`). |
| Sub-phase B constant addition | helper | `compute_kernel_lib::AddScalar<Slot>` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:937–941` | `AddScalar<Dst::D0>{0x405BA32Eu}` — the `uint32_t scalar` is the IEEE-754 bit pattern of `3·log(π) = 3.434189657547f`. Confirmed encoding requirement at `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:24` ("fp32 value scalar encoded as uint32"). | DEST | DEST | Implementation calls `binop_with_scalar_tile_init()` + `add_unary_tile(d0, scalar)` (`sfpu_helpers.inl:378–379`). |
| `DEST_AUTO_LIMIT` constant | helper (read-only) | `compute_kernel_lib::DEST_AUTO_LIMIT` | `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:88–102` | — | — | — | Resolves to 4 under `fp32_dest_acc_en=True` + half-sync. The chain stride is 4 (max DEST slot = D3, stride = max_dst + 1 = 4 — `sfpu_helpers.hpp:1178–1179`), so the pipeline correctly batches one tile per acquire/release. Used implicitly by `sfpu_pipeline`. |

### Raw APIs used for the per-offset lgamma sub-procedure

The lgamma sub-procedure is run four times per input tile (`offset ∈ {0.0, 0.5, 1.0, 1.5}`), inline in the kernel. Each invocation mirrors `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/lgamma_kernel.cpp:34–137` with one addition: a `sub_unary_tile(slot, bit_cast<uint32_t>(offset))` applied to every DEST slot that holds `x` before any of the lgamma math runs. With offset == 0.0 this is a `subtract zero` no-op preserving uniformity of the code path (no input-value branching).

| Step | Type | Function | File:Line | Args (DEST slot indices use the lgamma_kernel.cpp convention) | Notes |
|------|------|----------|-----------|---------------------------------------------------------------|-------|
| 1 | raw_api | `copy_tile_to_dst_init_short(cb)` | `tt_metal/hw/inc/api/compute/tile_move_copy.h` (declaration) | `(cb_input_tiles)` — re-issued at each step that pulls a fresh copy of `x` from `cb_input_tiles` | Needed because `copy_tile` requires the matching unpacker config; the lgamma_kernel.cpp reference issues this before every `copy_tile` block. |
| 2 | raw_api | `copy_tile(cb, src_idx, dst_idx)` | `tt_metal/hw/inc/api/compute/tile_move_copy.h` | `(cb_input_tiles, 0, 0)` then `(cb_input_tiles, 0, 1)` — populate D0, D1 with `x` | Source index 0 = head of CB; tile is held via `cb_wait_front(cb_input_tiles, 1)` and is NOT popped here. |
| 3 | raw_api | `binop_with_scalar_tile_init()` | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:89` | — | Re-init required before any `sub_unary_tile` / `add_unary_tile` block (other SFPU ops switch state). |
| 4 | raw_api | `sub_unary_tile(idst, param1)` | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:32–34` | `(0, bit_cast<uint32_t>(offset))`, `(1, bit_cast<uint32_t>(offset))` — apply per-offset subtraction. `param1` is the fp32 bit pattern of the scalar (lines 22–24). For `offset == 0.0f` (i.e., `param1 == 0u`), this is a no-op pass-through; we keep the call for code-path uniformity. | After this step: D0 = `x - offset`, D1 = `x - offset`. |
| 5 | raw_api | `sub_unary_tile(idst, param1)` | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:32–34` | `(1, 0x3F000000u /*0.5*/)` — D1 becomes `(x - offset) - 0.5`. | Used to test the reflection condition `(x - offset) < 0.5` via `ltz_tile` next. |
| 6 | raw_api | `ltz_tile_init()` + `ltz_tile(idst)` | `tt_metal/hw/inc/api/compute/eltwise_unary/comp.h` (declarations) | `ltz_tile(1)` — D1 becomes the `(x_off < 0.5)` mask (1.0 / 0.0). | Mirrors lgamma_kernel.cpp:52–53. |
| 7 | raw_api | `fill_tile_init()` + `fill_tile(idst, val)` | `tt_metal/hw/inc/api/compute/eltwise_unary/fill.h` (declarations) | `fill_tile(2, 1.0f)` — D2 = 1.0 | Prep for `1 - x_off`. Mirrors lgamma_kernel.cpp:55–56. |
| 8 | raw_api | `sub_binary_tile_init()` + `sub_binary_tile(a, b, out)` | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h:39, 88` | `sub_binary_tile(2, 0, 2)` — D2 = D2 - D0 = `1 - x_off` | Mirrors lgamma_kernel.cpp:59–60. |
| 9 | raw_api | `where_tile_init()` + `where_tile<DataFormat::Float32>(c, a, b, out)` | `tt_metal/hw/inc/api/compute/eltwise_unary/where.h` (declarations) | `where_tile<DataFormat::Float32>(1, 2, 0, 1)` — D1 = mask(D1) ? D2 : D0 = `z = (x_off < 0.5) ? 1-x_off : x_off` | Mirrors lgamma_kernel.cpp:63–64. |
| 10 | raw_api | `log_tile_init<false>()` + `log_tile<false>(idst)` | `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` → `log.h` (declarations) | `log_tile<false>(1)` — D1 = log(z) | High-precision (non-approx) log; matches lgamma_kernel.cpp:67–68. |
| 11 | raw_api | `lgamma_stirling_float_tile_init()` + `lgamma_stirling_float_tile(idst0, idst1, idst2)` | `tt_metal/hw/inc/api/compute/eltwise_unary/lgamma.h:68, 61` (also `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_sfpu_lgamma.h:48–62`) | `lgamma_stirling_float_tile(0, 1, 0)` — D0 = `lgamma_stirling(x_off)` with precomputed log(z) in D1 | The **fp32-precision** Stirling form (with Taylor polynomial bridges around z=1 and z=2 — `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lgamma.h:55–124`). The reflection correction for x < 0.5 is NOT yet applied here (lgamma.h:18–19, 44–46). |
| 12 | raw_api | `fill_tile(2, M_PI)` | `tt_metal/hw/inc/api/compute/eltwise_unary/fill.h` | D2 = π | Mirrors lgamma_kernel.cpp:74–76. |
| 13 | raw_api | `copy_tile_to_dst_init_short(cb)` + `copy_tile(cb_input_tiles, 0, 1)` | `tt_metal/hw/inc/api/compute/tile_move_copy.h` | reload D1 = x | Need to recompute `frac(x_off)` from a fresh copy. |
| 14 | raw_api | `sub_unary_tile(1, bit_cast<uint32_t>(offset))` | `binop_with_scalar.h:32–34` (re-init not needed; binop_with_scalar was the previous binop state from step 4) | D1 = `x - offset = x_off` | For `offset == 0.0` this is again a no-op pass-through. |
| 15 | raw_api | `rounding_op_tile_init()` + `frac_tile(idst)` | `tt_metal/hw/inc/api/compute/eltwise_unary/rounding.h` (declarations) | `frac_tile(1)` — D1 = `frac(x_off)` | Mirrors lgamma_kernel.cpp:82–83. |
| 16 | raw_api | `mul_binary_tile_init()` + `mul_binary_tile(a, b, out)` | `eltwise_binary_sfpu.h` (declarations, alongside line 39, 88) | `mul_binary_tile(1, 2, 1)` — D1 = `frac(x_off) · π` | Mirrors lgamma_kernel.cpp:86–87. |
| 17 | raw_api | `sin_tile_init()` + `sin_tile(idst)` | `tt_metal/hw/inc/api/compute/eltwise_unary/trigonometry.h` (declarations) | `sin_tile(1)` — D1 = `sin(π · frac(x_off))` | Mirrors lgamma_kernel.cpp:90–91. |
| 18 | raw_api | `copy_tile_to_dst_init_short(cb)` + `copy_tile(cb_input_tiles, 0, 2)` + `copy_tile(cb_input_tiles, 0, 3)` | `tile_move_copy.h` | D2 = x, D3 = x | Re-population for integer-handling. |
| 19 | raw_api | `sub_unary_tile(2, bit_cast<uint32_t>(offset))` + `sub_unary_tile(3, bit_cast<uint32_t>(offset))` | `binop_with_scalar.h:32–34` (re-init `binop_with_scalar_tile_init()` first because step 15–17 changed state) | D2 = x_off, D3 = x_off | Apply the per-offset subtraction. |
| 20 | raw_api | `rounding_op_tile_init()` + `floor_tile(idst)` | `rounding.h` | `floor_tile(3)` — D3 = `floor(x_off)` | Mirrors lgamma_kernel.cpp:97–99. |
| 21 | raw_api | `eq_binary_tile_init()` + `eq_binary_tile(a, b, out)` | `eltwise_binary_sfpu.h` (declarations) | `eq_binary_tile(2, 3, 2)` — D2 = `(x_off == floor(x_off))` mask | Detect integer x_off for sin-zero handling. |
| 22 | raw_api | `fill_tile_init()` + `fill_tile(3, 0.0f)` | `fill.h` | D3 = 0 | Zero value for `where` masking. |
| 23 | raw_api | `where_tile_init()` + `where_tile<DataFormat::Float32>(2, 3, 1, 1)` | `where.h` | D1 = (D2 mask) ? D3 : D1 — zero out `sin(π · frac)` when x_off is integer | Avoids `log(0)` blowup at integers. |
| 24 | raw_api | `abs_tile_init()` + `abs_tile(idst)` | `tt_metal/hw/inc/api/compute/eltwise_unary/negative.h` (declarations) | `abs_tile(1)` — D1 = `|sin(π · frac(x_off))|` | Mirrors lgamma_kernel.cpp:114–115. |
| 25 | raw_api | `log_tile_init()` + `log_tile(idst)` | `sfpu_split_includes.h` → `log.h` | `log_tile(1)` — D1 = `log|sin(π · frac(x_off))|` (= -∞ where x_off is integer, naturally producing NaN downstream when the reflection branch is taken — torch-compatible). | Mirrors lgamma_kernel.cpp:118–119. Default (non-approx) log; lgamma_kernel.cpp uses the no-template-arg variant here. |
| 26 | raw_api | `copy_tile_to_dst_init_short(cb)` + `copy_tile(cb_input_tiles, 0, 2)` + `sub_unary_tile(2, bit_cast<uint32_t>(offset))` (with `binop_with_scalar_tile_init()` first) | `tile_move_copy.h`, `binop_with_scalar.h:32–34, 89` | D2 = x_off (reload because step 21 clobbered it with the mask) | Required for `lgamma_adjusted_tile`'s third argument. |
| 27 | raw_api | `lgamma_adjusted_tile_init()` + `lgamma_adjusted_tile(idst_stirling, idst_log_sin, idst_input, idst_out)` | `tt_metal/hw/inc/api/compute/eltwise_unary/lgamma.h:101, 94` (also `llk_math_eltwise_sfpu_lgamma.h:32–46` and `ckernel_sfpu_lgamma.h:126–162`) | `lgamma_adjusted_tile(0, 1, 2, 0)` — D0 = `lgamma(x_off)`, with reflection applied when `x_off < 0.5` (using the `log\|sin\|` term computed above) | The full reflection correction (`ln(π) − log\|sin(π·frac(x))\| − lgamma_stirling(1 − x_off)` when `x_off < 0.5`, else `lgamma_stirling(x_off)`) — see `ckernel_sfpu_lgamma.h:144–148`. With `is_fp32_dest_acc_en=True` (template threaded through), produces fp32 results without bfloat16 truncation. |
| 28 | raw_api | `cb_reserve_back(cb_lgamma_term, 1)` | `tt_metal/hw/inc/api/compute/cb_api.h` (declaration) | `(cb_lgamma_<offset>, 1)` — reserve one tile in the matching per-offset CB | Called BEFORE `tile_regs_acquire` per the standard pattern in lgamma_kernel.cpp:37. |
| 29 | raw_api | `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()` | `tt_metal/hw/inc/api/compute/reg_api.h` (declarations) | — | Bracketing the DEST manipulation in steps 1–27. Standard MATH↔PACK sync protocol (lgamma_kernel.cpp:35, 127, 129, 136). |
| 30 | raw_api | `pack_tile(idst, cb)` | `tt_metal/hw/inc/api/compute/pack.h` (declaration) | `pack_tile(0, cb_lgamma_<offset>)` — pack D0 result into the matching per-offset CB | Default `pack_tile<false>` — sequential pack into the reserved slot at `fifo_wr_ptr`. |
| 31 | raw_api | `cb_push_back(cb_lgamma_term, 1)` | `cb_api.h` | `(cb_lgamma_<offset>, 1)` | Standard end-of-tile push, advances `fifo_wr_ptr` by one tile (see memory note in `~/.claude/projects/.../MEMORY.md` "cb_push_back Always Advances fifo_wr_ptr"). |

After all four lgamma sub-phases of an input tile have completed, the compute kernel issues a single `cb_pop_front(cb_input_tiles, 1)`. The four lgamma CBs each contain one fresh tile, ready to feed the sum sub-phase.

### Helpers considered and rejected (per-offset lgamma sub-procedure)

| Helper considered | File:Line | Reason rejected (concrete) |
|-------------------|-----------|----------------------------|
| `compute_kernel_lib::Lgamma<Slot>` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:700–704`; impl at `sfpu_helpers.inl:276–277` | The helper's `init()` and `call()` resolve to `lgamma_stirling_tile_init()` / `lgamma_stirling_tile(idst)` — the **single-arg unary** variant. The header docstring at `tt_metal/hw/inc/api/compute/eltwise_unary/lgamma.h:15–17` states this form is "suitable for bfloat16 inputs" (the **fp32-suitable** form is the 3-arg `lgamma_stirling_float_tile(idst0, idst1, idst2)` at `lgamma.h:61`). Additionally, lgamma.h:18–19 and the ckernel comment at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lgamma.h:45–46, 120` explicitly state "the final reflection formula correction for (inputs < 0.5) is not part of this kernel" — only `lgamma_adjusted_tile` (`lgamma.h:94`) applies it. Multigammaln has the term `lgamma(a - 1.5)` whose argument lies in `(0, 0.5)` for `a ∈ (1.5, 2.0)` — a real-valued, in-domain region per `torch.special.multigammaln`. Without `lgamma_adjusted_tile`, that term would be wrong by the reflection delta `ln(π) − log\|sin(π·x)\|`. No helper exposes `lgamma_stirling_float_tile` or `lgamma_adjusted_tile`. Therefore the per-offset lgamma sub-procedure necessarily uses raw APIs (mirroring `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/lgamma_kernel.cpp` line-for-line, plus the per-offset subtraction). |
| `compute_kernel_lib::SubScalar<Slot>` for the per-offset `x - offset` subtraction | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:943–948`; impl at `sfpu_helpers.inl:380–381` | The helper IS structurally usable for the subtraction itself, but it can only appear inside a `sfpu_chain` and `sfpu_pipeline` invocation. Sub-phase A is not expressible as a chain: it consumes 23+ distinct SFPU op calls per offset (Stirling + reflection setup + adjusted), of which only a few have helper equivalents and none of those helpers compose with `lgamma_stirling_float_tile` or `lgamma_adjusted_tile`. We therefore inline `sub_unary_tile` directly with the same `binop_with_scalar_tile_init()` (`binop_with_scalar.h:89`) the helper would call, keeping the raw sequence consistent with lgamma_kernel.cpp. The semantics — `D[slot] = D[slot] - bit_cast<float>(scalar)` — match `SubScalar` exactly. |
| `compute_kernel_lib::Where<...>`, `SfpuSub`, `SfpuMul`, `SfpuEq` for the in-sequence `where_tile` / `sub_binary_tile` / `mul_binary_tile` / `eq_binary_tile` calls | `sfpu_helpers.hpp:1041–1046` (`SfpuSub`), `1047–1051` (`SfpuMul`), `1071–1075` (`SfpuEq`), `1097–1100` (`Where`) | These ops appear in the **middle** of the lgamma sequence between `lgamma_stirling_float_tile` and `lgamma_adjusted_tile` calls. They cannot be lifted into a `sfpu_chain` because the chain framework requires the entire pipeline to be expressible as `Load*` → `compute*` → pack, with DEST contents owned by the chain. Our DEST state is hand-managed across multi-tile operations (the LLK ternary `lgamma_adjusted_tile` reads D0/D1/D2 simultaneously, then writes D0), which is outside what `sfpu_chain` represents. Using the raw `*_binary_tile` / `where_tile` calls directly preserves DEST register layout across the recipe — exactly what `lgamma_kernel.cpp` already does. |
| `compute_kernel_lib::FillTile<Slot>`, `Log<Slot>`, `Abs<Slot>`, `Sin<Slot>`, `Floor<Slot>`, `Frac<Slot>` | `sfpu_helpers.hpp:1011–1015`, `420–423`, `463–466`, `603–606`, `886–889`, `905–908` | Same reason as above: these ops are interleaved through hand-managed DEST slots inside a multi-step recipe that cannot be expressed as a `sfpu_chain` because of the embedded ternary `lgamma_adjusted_tile` and the simultaneous reads of D0/D1/D2 with write-back to D0. Each helper would inject its own implicit `tile_regs_*` / `cb_*` framing if used via `sfpu_pipeline`, which conflicts with the explicit, monolithic `tile_regs_acquire/commit/wait/release` block that lgamma_kernel.cpp uses (and that this design replicates). Using the underlying raw `*_tile_init` + `*_tile` is the documented and verified pattern in lgamma_kernel.cpp. |
| `compute_kernel_lib::binary_op` and `add`, `sub`, `mul` (FPU binary helpers) for combining lgamma terms via CB→CB chains | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:250–314` | The FPU-binary helpers consume two CBs and produce a third CB, requiring chained intermediate CBs to combine four terms (`add(a, half, t1) → add(t1, one, t2) → add(t2, three_halves, t3) → add_const(t3, K)`). That uses two additional intermediate CBs (`t1`, `t2`, plus an `AddScalar` step that has no FPU equivalent — `add_unary_tile` is SFPU). The SFPU chain expresses the entire 4-way sum + constant within one DEST acquire/release using 4 slots — strictly cheaper, fewer CBs, fewer `tile_regs_*` cycles. Helpers chosen on this basis: `sfpu_chain` + `sfpu_pipeline`. |

## Compute Phases

Sequential phase execution per output tile. Phase boundaries are explicit because each phase is its own helper invocation or `tile_regs_acquire/release` block.

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `cb_wait_front(cb_input_tiles, 1)` | raw | `cb_input_tiles` (1 tile, ready) | — | `cb_input_tiles` front held (NOT popped) |
| 1 | Per-offset lgamma sub-procedure with `offset = 0.0f` (steps 1–31 in the API Mapping above) — 31 raw API calls inside one `tile_regs_acquire/commit/wait/release` block, packing the final DEST D0 to `cb_lgamma_a` | raw | `cb_input_tiles` (1 tile, front, not popped) | `cb_lgamma_a` (1 tile pushed) | `cb_input_tiles` still front-held; `cb_lgamma_a` +1 tile |
| 2 | Per-offset lgamma sub-procedure with `offset = 0.5f` (= 0x3F000000u in IEEE-754) — same 31 raw steps, but step 4 sub_unary_tile subtracts 0.5 first | raw | `cb_input_tiles` (1 tile, front, not popped) | `cb_lgamma_a_half` (1 tile pushed) | `cb_input_tiles` still front-held; `cb_lgamma_a_half` +1 tile |
| 3 | Per-offset lgamma sub-procedure with `offset = 1.0f` (= 0x3F800000u) | raw | `cb_input_tiles` (1 tile, front, not popped) | `cb_lgamma_a_one` (1 tile pushed) | `cb_input_tiles` still front-held; `cb_lgamma_a_one` +1 tile |
| 4 | Per-offset lgamma sub-procedure with `offset = 1.5f` (= 0x3FC00000u) | raw | `cb_input_tiles` (1 tile, front, not popped) | `cb_lgamma_a_three_halves` (1 tile pushed) | `cb_input_tiles` still front-held; `cb_lgamma_a_three_halves` +1 tile |
| 5 | `cb_pop_front(cb_input_tiles, 1)` — release the input tile now that all four lgammas have read it | raw | — | — | `cb_input_tiles` -1 tile |
| 6 | Sum sub-phase: `sfpu_pipeline(sfpu_chain(Load<cb_lgamma_a, D0>, Load<cb_lgamma_a_half, D1>, Load<cb_lgamma_a_one, D2>, Load<cb_lgamma_a_three_halves, D3>, SfpuAdd<D0,D1,D0>, SfpuAdd<D0,D2,D0>, SfpuAdd<D0,D3,D0>, AddScalar<D0>{0x405BA32Eu /*≈3·log(π)*/}), cb_output_tiles, 1)`. The helper internally does: acquire, four wait+copy+pop pairs (one per CB), three `add_binary_tile` (D0+=Dn), one `add_unary_tile` (D0+=K), commit, wait, reserve output, `pack_tile(0, cb_output_tiles)`, push output, release. | helper (`sfpu_pipeline` + `sfpu_chain`) | `cb_lgamma_a` (1 tile), `cb_lgamma_a_half` (1), `cb_lgamma_a_one` (1), `cb_lgamma_a_three_halves` (1) | `cb_output_tiles` (1 tile pushed) | All four `cb_lgamma_*` -1 tile; `cb_output_tiles` +1 tile |

The kernel-main outer loop wraps phases 0–6 inside `for (uint32_t i = 0; i < num_tiles; ++i)`. With 2-page intermediate CBs and 1 push/pop per CB per iteration, the loop is self-balanced (`push 1 → pop 1` per CB per iteration, never accumulates beyond 1 in-flight tile).

## Build Order

Suggested incremental bring-up sequence for the implementer. Each step yields a runnable artifact that can be inspected via `tt-probe.sh` with deterministic inputs (e.g., a constant fill of `2.5f` → expected `lgamma(2.5) + lgamma(2.0) + lgamma(1.5) + lgamma(1.0) + 3·log(π)`).

| Step | Build target | Verification |
|------|--------------|--------------|
| 1 | Python entrypoint (`multigammaln.py`) with validation and `__init__.py`. Stub program descriptor that allocates the output tensor and returns a no-op. | `from ttnn.operations.multigammaln import multigammaln` succeeds; validation `pytest -k rejects` cases pass. |
| 2 | Pass-through pipeline: reader → cb_input_tiles → compute copies tile to cb_output_tiles → writer. No lgamma yet. Verifies CB sync, work distribution, TensorAccessor wiring. | A constant-fill input round-trips through the kernel unchanged. |
| 3 | Sum sub-phase only (replace lgamma sub-phases with simple `pack_tile` of `x` into all four `cb_lgamma_*`). Output should equal `4·x + 3·log(π)`. | Constant input `x = 1.0` → output `4.0 + 3.434189... ≈ 7.434189`. Validates `sfpu_chain`, `SfpuAdd`, `AddScalar` and the 4-input CompactLoad pattern. |
| 4 | Add lgamma sub-phase for `offset = 0.0` only (other three `cb_lgamma_*` still produce zero or `x`). Output = `lgamma(x) + 3·x + 3·log(π)`. | Constant `x = 1.0` → `lgamma(1.0) = 0`, so output `0 + 3·1 + 3·log(π) ≈ 6.434189`. Validates the full Stirling-+-adjusted recipe. |
| 5 | Enable all four offsets. Output = full multigammaln. | Constant `x = 2.5` → `lgamma(2.5) + lgamma(2.0) + lgamma(1.5) + lgamma(1.0) + 3·log(π)` ≈ `0.2846828 + 0 + (-0.1207822) + 0 + 3.434189... ≈ 3.598090`. Cross-check against `torch.special.multigammaln(torch.tensor(2.5), 4)`. |
| 6 | Multi-tile, multi-core. Use `ttnn.split_work_to_cores` and per-core RT args. | Run the acceptance test (`tests/ttnn/unit_tests/operations/multigammaln/test_multigammaln.py`). |

`tt-probe.sh --dev` is recommended through step 4 (asserts and CB sanitization catch DEST-overflow and CB-overrun bugs at the failing instruction). Switch to default mode for step 5–6 once the recipe is stable.

## Kernel Arguments

### Compile-Time

| Kernel | Index | Name | Type | Source/Formula |
|--------|-------|------|------|----------------|
| Reader | 0 | `num_tiles` | `uint32_t` | unused at CT — passed at RT instead (see below). _Optional alternative_: if the implementer prefers CT, set to `total_tiles` per kernel (single-grid CT). The implementer chooses based on whether tile counts are uniform across cores. |
| Reader | 0… | `TensorAccessorArgs(input_tensor)` | TensorAccessor CT args | `ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()` (`.claude/references/ttnn-python-utility-bindings.md`) |
| Writer | 0… | `TensorAccessorArgs(output_tensor)` | TensorAccessor CT args | same |
| Compute | — | — | — | none required at CT; the kernel reads `num_tiles` as RT. (Optional: encode `cb_input_tiles=0`, `cb_lgamma_a=24`, …, `cb_output_tiles=16` as CT constants for cleaner kernel source. Indices are also hard-coded in `op_design.md` and may be inlined as `constexpr` in the kernel.) |

Scalar CT args always go FIRST; `TensorAccessorArgs(...)` always go LAST (`.claude/references/generic_op_template/template_op_program_descriptor.py:106–110`).

### Runtime

| Kernel | Index | Name | Type | Source/Formula |
|--------|-------|------|------|----------------|
| Reader | 0 | `input_addr` | `uint32_t` | `input_tensor.buffer_address()` |
| Reader | 1 | `start_tile_id` | `uint32_t` | Running cumulative tile offset for this core (assigned by walking `core_group_1` then `core_group_2`). |
| Reader | 2 | `num_tiles_for_this_core` | `uint32_t` | `pages_per_core_g1` if the core is in `core_group_1` else `pages_per_core_g2`. |
| Writer | 0 | `output_addr` | `uint32_t` | `output_tensor.buffer_address()` |
| Writer | 1 | `start_tile_id` | `uint32_t` | same as reader |
| Writer | 2 | `num_tiles_for_this_core` | `uint32_t` | same as reader |
| Compute | 0 | `num_tiles_for_this_core` | `uint32_t` | same as reader |

## Key Risks and Gotchas

| Risk | Mitigation |
|------|------------|
| **DEST budget** — `fp32_dest_acc_en=True` halves DEST to 4 tiles per acquire/release (`dest_helpers.hpp:88–102`). The per-offset lgamma sub-procedure uses exactly D0..D3 (matches `lgamma_kernel.cpp` reference). The sum chain also uses D0..D3 (`stride = 4`, batch_size = 1). No room for headroom — adding any additional concurrent tile in DEST will overflow. | Document explicitly in the kernel source; do NOT bump batch size; do NOT add more concurrent slots. |
| **`Lgamma<>` helper is wrong for fp32** — calls the bf16 single-arg Stirling variant without reflection (see "Helpers considered and rejected"). Using it would silently produce incorrect values for `a ∈ (1.5, 2.0)`. | Use raw `lgamma_stirling_float_tile` + `lgamma_adjusted_tile` as documented. |
| **Domain `a ≤ 1.5` produces NaN naturally** — `torch.special.multigammaln(x, 4)` returns NaN there. The kernel must not branch on input; the reflection inside `lgamma_adjusted_tile` plus the log-of-negative inside `lgamma_stirling_float_tile` produce NaN for non-positive args. | The implementer must NOT add a `where` predicate on `a > 1.5`. The acceptance test verifies the output is NaN exactly where torch outputs NaN. |
| **`pack_tile<false>` advances `fifo_wr_ptr`** — combined with `cb_push_back`, the reserved-then-pushed CB region must be re-reserved per tile. Don't try to "stream-write multiple lgamma terms into one accumulator CB" — that would require `pack_tile<true>` with held wr_ptr, a much trickier protocol. | Use four distinct CBs (`cb_lgamma_a`, `_half`, `_one`, `_three_halves`), each with its own `cb_reserve_back` / `pack_tile<false>` / `cb_push_back` per tile. |
| **`init_sfpu(cb_input_tiles, cb_output_tiles)` is called once** — but the kernel writes to four intermediate CBs (`cb_lgamma_*`) AND `cb_output_tiles`. All five output CBs have the same dtype (float32), so the packer config from `init_sfpu(cb_input_tiles, cb_output_tiles)` is correct for all of them; no reconfig needed mid-kernel. The SFPU pipeline at phase 6 internally reconfigures (per the `SfpuDataFormatReconfig::INPUT_AND_OUTPUT` default — `sfpu_helpers.hpp:1410`), which is a no-op when input/output formats are identical. | Don't call `init_sfpu` more than once. The helper's per-call reconfig is sufficient for phase 6. |
| **`cb_input_tiles` is read four times without pop** — phases 1–4 each `copy_tile(cb_input_tiles, 0, …)` against the same front tile. `cb_pop_front` is deferred until after phase 4. If the implementer accidentally pops the front between phases, the next phase reads stale (or unmapped) memory. | Single `cb_wait_front(cb_input_tiles, 1)` at phase 0, single `cb_pop_front(cb_input_tiles, 1)` at phase 5. Verify in code review. |
| **The constant `3·log(π)` is bit-cast** — `AddScalar<D0>{0x405BA32Eu}` encodes `3.434189657547f` as an `uint32_t`. The IEEE-754 round-trip is exact for this value to 7 sig figs (matches `numpy.float32(3 * numpy.log(numpy.pi))` = `3.4341897`). | Compute the bit pattern from `bit_cast<uint32_t>(3.434189657547f)` (`__builtin_bit_cast` in C++20, or `reinterpret_cast` via `union`). For clarity, define `constexpr uint32_t THREE_LOG_PI_BITS = 0x405BA32Eu;` in the kernel with a comment showing the source float literal. |
| **`offset == 0.0f` case** — step 4's `sub_unary_tile(slot, 0u)` IS a real SFPU op (not eliminated). It subtracts +0.0 (which is a no-op semantically). Subtracting 0 from a NaN or inf preserves the value. We pay a few extra cycles for code-path uniformity. | Acceptable cost (<1% kernel time per offset); avoids `if constexpr` branching that would complicate the source. The implementer may convert this into a `constexpr` template `if` to elide the call if perf becomes an issue in a later refinement. |
| **Math fidelity = HiFi4 is required** for the precision target. `fp32_dest_acc_en=True` alone is not sufficient — without HiFi4, FPU operations would still run at LoFi precision. | Wire both in `ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)`. Do NOT expose to caller (Phase 0). |
