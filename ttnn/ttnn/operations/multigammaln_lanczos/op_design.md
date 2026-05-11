# Operation Design: multigammaln_lanczos

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (elementwise, fused) |
| Goal | Elementwise multivariate log-gamma at order `p = 4`, implemented as a faithful translation of the Lanczos 6-term polynomial recipe into a single fused TTNN kernel. The Python entry point dispatches exactly one program via `ttnn.generic_op`; no `ttnn::*` op chaining is allowed and no SFPU `lgamma_tile` / `lgamma_stirling_float_tile` / `lgamma_adjusted_tile` helper may appear in the compute kernel. |
| Math | `output[idx] = L(a) + L(a − 0.5) + L(a − 1.0) + L(a − 1.5) + 3.434189657547`, where `a = input[idx]` and `L(·)` is the Lanczos 6-term polynomial approximation of `lgamma` (see Algorithm below). |
| Mode | Derivative (matches `torch.special.multigammaln(x, p=4)` within Lanczos-at-fp32 precision). |
| References | `ttnn/ttnn/operations/multigammaln/op_design.md` (same per-tile architecture — 4 per-offset CBs → SFPU sum chain; we adopt that scaffolding verbatim and replace only the per-offset compute with the Lanczos polynomial), `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp` (`sfpu_chain` + `sfpu_pipeline` for the final sum), `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h` (`add_unary_tile`, `sub_unary_tile`, `mul_unary_tile` — the per-offset scalar arithmetic), `tt_metal/hw/inc/api/compute/eltwise_unary/recip.h` (`recip_tile`), `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` → `log.h` (`log_tile`), `tt_metal/hw/inc/api/compute/eltwise_unary/fill.h` (`fill_tile`), `tt_metal/hw/inc/api/compute/eltwise_unary/comp.h` (`unary_eq_tile`), `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` (`add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`), `tt_metal/hw/inc/api/compute/tile_move_copy.h` (`copy_tile`, `copy_tile_to_dst_init_short`). |

### Algorithm — Lanczos 6-term polynomial (verbatim from the spec)

For a single scalar argument `y` (the argument to one of the four `_lgamma` calls), the Lanczos sub-recipe is:

```
input  = y - 1
series = 1
       + 76.18009172947146f          / (input + 1)
       + (-86.50532032941677f)       / (input + 2)
       + 24.01409824083091f          / (input + 3)
       + (-1.231739572450155f)       / (input + 4)
       + 0.1208650973866179e-2f      / (input + 5)
       + (-0.5395239384953e-5f)      / (input + 6)
t      = input + 5.5
L(y)   = (input + 0.5) * log(t)
       + 0.918938531357171f          // == log(sqrt(2π))
       + log(series)
       - t
// Reflection-free; zero at the integer poles 1 and 2
if y == 1 or y == 2: L(y) = 0
```

Algebraic identity used in the kernel (avoids needing both `t` and `log(t)` in DEST at the same time):

```
L(y) = (input + 0.5) * log(t) + log(series) - input - 4.581061468643f
     // since (0.918938531357171 - 5.5) == -4.581061468643
     // and t = input + 5.5, so log(sqrt(2π)) - t = -input - 4.581061468643
```

For `multigammaln(a, 4)`:

```
output = L(a) + L(a - 0.5) + L(a - 1.0) + L(a - 1.5) + 3.434189657547f  // == 3 * log(π)
```

Each `L(y)` zero-clamps at its own integer poles `{1, 2}` — i.e. `L(a − off)` is zeroed where `a == off + 1` or `a == off + 2`. This is a `where`-style elementwise select on a scalar bitmask, not an input-value branch.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | `float32`, `TILE_LAYOUT`, rank `4` `(N, C, H, W)`, `H % 32 == 0`, `W % 32 == 0`, on-device (DRAM or L1 interleaved) | — | runtime (buffer addr + per-core tile range) |
| `memory_config` (keyword) | `ttnn.MemoryConfig` | no | DRAM or L1 interleaved | `ttnn.DRAM_MEMORY_CONFIG` | host (does not appear in kernel) |

`p` is **not** a parameter — order is wired to 4 in the kernel.

### Compute Config (hard-coded inside `create_program_descriptor` — NOT a caller parameter)

| Field | Value | Source |
|-------|-------|--------|
| `math_fidelity` | `ttnn.MathFidelity.HiFi4` | Required for fp32 FPU precision on the polynomial / log / recip steps. |
| `fp32_dest_acc_en` | `True` | Required to keep mantissa precision through the multi-step recipe. |
| Effective DEST capacity | **4 tiles** (half-sync + fp32 DEST acc) — auto-detected by `compute_kernel_lib::DEST_AUTO_LIMIT` (`ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:88–102`). | Limits the per-offset compute to D0..D3 and forces the final-sum SFPU pipeline `batch_size = 4 / stride 4 = 1`. |
| `unpack_to_dest_mode` | `UnpackToDestFp32` on every fp32 CB (`cb_input_tiles`, `cb_lgamma_a`, `cb_lgamma_a_half`, `cb_lgamma_a_one`, `cb_lgamma_a_three_halves`, `cb_output_tiles`) | Mirrors the precision lever flagged in `ttnn/ttnn/operations/multigammaln/multigammaln_program_descriptor.py:170–183`. Without this, `copy_tile`-ing an intermediate fp32 tile into DEST can lose mantissa bits via the SrcA/SrcB TF32-truncation path even though the CBs are declared Float32 and DEST is fp32-accumulated. |

Compute config is wired as:

```python
compute_config = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
)
compute_config.unpack_to_dest_mode = unpack_modes  # see above
```

## Tensors

### Input — `input_tensor`

| Property | Requirement |
|----------|-------------|
| Shape | `(N, C, H, W)` — rank == 4 (validated). |
| Dtype | `float32` (validated). Other dtypes → `ValueError`. |
| Layout | `TILE_LAYOUT` (validated). `ROW_MAJOR_LAYOUT` → `ValueError`. |
| Memory | DRAM or L1 interleaved. |
| Tile-alignment | `H % 32 == 0`, `W % 32 == 0` on the logical shape (validated). |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to `input_tensor.shape`. |
| Dtype | `float32`. |
| Layout | `TILE_LAYOUT`. |
| Memory | inherited from `memory_config` arg; defaults to DRAM interleaved. |

## Validation (Python side, before `ttnn.generic_op`)

| Check | Failure |
|-------|---------|
| `input_tensor.dtype == ttnn.float32` | `ValueError("multigammaln_lanczos: only float32 is supported in Phase 0, got …")` |
| `input_tensor.layout == ttnn.TILE_LAYOUT` | `ValueError("multigammaln_lanczos: only TILE_LAYOUT is supported in Phase 0, got …")` |
| `len(input_tensor.shape) == 4` | `ValueError("multigammaln_lanczos: input must be rank-4 (N, C, H, W), got shape …")` |
| `input_tensor.shape[-1] % 32 == 0 and input_tensor.shape[-2] % 32 == 0` | `ValueError("multigammaln_lanczos: input H and W must be divisible by 32, got shape …")` |

Domain validation (`a > 1.5`) is **not** performed in Python. Out-of-domain inputs naturally produce NaN / −Inf via the math falling through (the polynomial has poles, `log` of negative / zero, `recip` near zero). This matches `torch.special.multigammaln(x, 4)`. The kernel MUST NOT branch on input value.

## Dataflow Strategy

`multigammaln_lanczos` is purely elementwise — every output tile depends on exactly one input tile at the same logical tile id. No reduction, no broadcast, no inter-tile or inter-Tensix data dependency.

| Stage | Role | Data path |
|-------|------|-----------|
| **DRAM → reader** | NCRISC reader streams the per-core slice of input tiles from DRAM into `cb_input_tiles`, one tile at a time. | `input_tensor` DRAM → `cb_input_tiles` |
| **Compute, sub-phase A (×4 per tile)** | TRISCs hold the input tile (`cb_wait_front(cb_input_tiles, 1)` once, NOT popped between sub-phases), run the full Lanczos polynomial recipe four times in succession on `(a − 0.0)`, `(a − 0.5)`, `(a − 1.0)`, `(a − 1.5)`. Each invocation packs its single-tile result to a dedicated per-term intermediate CB. After the four sub-phases complete, the input tile is popped. | `cb_input_tiles` → `cb_lgamma_a`, `cb_lgamma_a_half`, `cb_lgamma_a_one`, `cb_lgamma_a_three_halves` (one tile each per input tile) |
| **Compute, sub-phase B** | TRISCs run a single `sfpu_pipeline` over a 4-load SFPU chain: `Load<cb_lgamma_a, D0>`, `Load<cb_lgamma_a_half, D1>`, `Load<cb_lgamma_a_one, D2>`, `Load<cb_lgamma_a_three_halves, D3>`, `SfpuAdd<D0,D1,D0>`, `SfpuAdd<D0,D2,D0>`, `SfpuAdd<D0,D3,D0>`, `AddScalar<D0>{0x405BA32Eu /* ≈3·log(π) */}`. The result is packed into `cb_output_tiles`. | `cb_lgamma_*` → `cb_output_tiles` |
| **compute → writer** | BRISC writer drains `cb_output_tiles` and writes each tile back to DRAM at the matching logical tile id. | `cb_output_tiles` → output-tensor DRAM |

**No inter-Tensix communication.** The work is embarrassingly parallel across output tiles; each core processes its assigned slice end-to-end. No semaphores, no multicast.

**Tensor format does not change.** Inputs arrive tiled (TILE_LAYOUT), every CB holds tiles, the output is written tiled. No tilize / untilize.

**Why four intermediate CBs (not one):** the final 4-way sum is expressed through `sfpu_chain(Load<cb_a, D0>, Load<cb_a_half, D1>, Load<cb_a_one, D2>, Load<cb_a_three_halves, D3>, …)`. `CompactLoad` (`ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:390–405`) issues one `cb_wait_front(CB, 1) + copy_tile(CB, 0, slot) + cb_pop_front(CB, 1)` per CB, so each per-tile DEST-slot load must come from a distinct CB. A single 4-page accumulator CB would copy the SAME head tile into four slots — wrong. The compaction rule and the `NoMultiGroupCB` static_assert (`sfpu_helpers.hpp:1366–1369`) also require each CB to appear in exactly one Load group.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One 32 × 32 fp32 tile (≡ one output tile, ≡ one input tile). |
| Total tiles | `total_tiles = (N · C · H · W) / (32 · 32)` — equivalently `input_tensor.buffer_num_pages()` (`.claude/references/ttnn-python-utility-bindings.md:40–50`). |
| Grid | `device.compute_with_storage_grid_size()` — the full Tensix compute grid. |
| Per-core work | `ttnn.split_work_to_cores(grid_size, total_tiles)` (`.claude/references/ttnn-python-utility-bindings.md:159–176`) returns `(num_cores, all_cores, core_group_1, core_group_2, pages_per_core_g1, pages_per_core_g2)`. Group sizes differ by at most one tile. |
| Remainder | Handled by the two-group split — no explicit per-core remainder math in the kernels. |
| Per-core RT args | Reader: `(input_addr, start_tile_id, num_tiles_for_this_core)`. Writer: `(output_addr, start_tile_id, num_tiles_for_this_core)`. Compute: `(num_tiles_for_this_core,)`. `start_tile_id` is the running cumulative offset assigned by walking `core_group_1` first, then `core_group_2`. |

Reader, writer and compute KernelDescriptors are placed on `all_cores`; per-core RT-arg arrays are populated by iterating `core_group_1` then `core_group_2` while accumulating `start_tile_id` (pattern from `ttnn/ttnn/operations/multigammaln/multigammaln_program_descriptor.py:131–146`).

## Circular Buffers

CB index convention (`.claude/references/op-design-template.md` §"Circular Buffers", and `.claude/references/generic_op_template/template_op_program_descriptor.py:74–80`): `0–7` for input-tensor CBs, `16–23` for output-tensor CBs, `24–31` for intermediates.

All CBs are float32 (`tile_size(float32) == 4096 B`).

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_tiles` | 0 | `tile_size(float32)` = 4096 B | 2 (double-buffer) | float32 | reader (NCRISC) | compute (held with `cb_wait_front(..., 1)`, **not popped** between the four Lanczos sub-phases of one tile; `cb_pop_front(..., 1)` once after all four sub-phases) | per-tile streaming |
| `cb_lgamma_a` | 24 | `tile_size(float32)` = 4096 B | 2 (double-buffer) | float32 | compute (Lanczos sub-phase, offset = 0.0f) | compute (sum sub-phase, Load → D0) | per-tile streaming |
| `cb_lgamma_a_half` | 25 | `tile_size(float32)` = 4096 B | 2 (double-buffer) | float32 | compute (Lanczos sub-phase, offset = 0.5f) | compute (sum sub-phase, Load → D1) | per-tile streaming |
| `cb_lgamma_a_one` | 26 | `tile_size(float32)` = 4096 B | 2 (double-buffer) | float32 | compute (Lanczos sub-phase, offset = 1.0f) | compute (sum sub-phase, Load → D2) | per-tile streaming |
| `cb_lgamma_a_three_halves` | 27 | `tile_size(float32)` = 4096 B | 2 (double-buffer) | float32 | compute (Lanczos sub-phase, offset = 1.5f) | compute (sum sub-phase, Load → D3) | per-tile streaming |
| `cb_output_tiles` | 16 | `tile_size(float32)` = 4096 B | 2 (double-buffer) | float32 | compute (sum sub-phase) | writer (BRISC) | per-tile streaming |

### CB sizing rationale

- **`cb_input_tiles` = 2 pages.** Streaming reader (NCRISC) → compute (TRISCs) — different processors, pipelining works → standard double-buffer is correct (`.claude/references/ttnn-cb-memory-fundamentals.md` "double-buffered streaming CB").
- **`cb_lgamma_*` = 2 pages each.** Producer and consumer are both the **same compute kernel** running sequentially within one tile (sub-phase A produces 1 page, sub-phase B consumes 1 page). Per-tile balance is `+1, −1` per CB, so 1 page would functionally suffice — but a 2-page allocation lets the next iteration's sub-phase A begin while sub-phase B is still draining the previous iteration's tiles. This is the standard double-buffer convention; the cost is 4 KB extra L1 per CB.
- **`cb_output_tiles` = 2 pages.** Streaming compute (TRISCs) → writer (BRISC) — different processors, pipelining works → standard double-buffer.

### CB sync verification (per output tile, single core)

| CB | Producer pushes | Consumer waits | Consumer pops | Match? |
|----|-----------------|----------------|---------------|--------|
| `cb_input_tiles` | reader: 1 per tile | compute: `cb_wait_front(cb_input_tiles, 1)` once at the start of the tile; reused across all 4 Lanczos sub-phases with `copy_tile(cb_input_tiles, 0, slot)` (no intermediate pop) | compute: `cb_pop_front(cb_input_tiles, 1)` once at the end of the four sub-phases | ✓ |
| `cb_lgamma_a` | compute (Lanczos sub-phase, offset=0.0): 1 per tile | sum-phase Load (D0): 1 per tile | sum-phase CompactLoad pop (DoPop=true after compact-fold — `sfpu_helpers.hpp:1280–1281`): 1 per tile | ✓ |
| `cb_lgamma_a_half` | compute (offset=0.5): 1 per tile | sum-phase Load (D1): 1 per tile | sum-phase CompactLoad pop: 1 per tile | ✓ |
| `cb_lgamma_a_one` | compute (offset=1.0): 1 per tile | sum-phase Load (D2): 1 per tile | sum-phase CompactLoad pop: 1 per tile | ✓ |
| `cb_lgamma_a_three_halves` | compute (offset=1.5): 1 per tile | sum-phase Load (D3): 1 per tile | sum-phase CompactLoad pop: 1 per tile | ✓ |
| `cb_output_tiles` | compute (sum sub-phase): 1 per tile | writer: 1 per tile | writer: 1 per tile | ✓ |

## API Mapping

Every operation in the compute kernel is enumerated here with file:line citations. Sub-phase B (the 4-way sum + constant) is fully helper-based — `sfpu_chain` + `sfpu_pipeline`. Sub-phase A (the Lanczos polynomial per offset) is raw-API based because the chain abstraction cannot express the multi-init, multi-binary-op recipe with cross-pass DEST state; concrete justifications appear in the "Helpers considered and rejected" block below.

### Helpers used

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|------------------------|--------------------------|---------------------------|--------------|
| Kernel hardware init (once, top of `kernel_main`) | raw_api (no helper wraps this) | `init_sfpu(icb, ocb)` — documented entry point at `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:72–73` ("PREREQUISITE: Call `init_sfpu(icb, ocb)` at the start of your kernel"); implementation pulls `eltwise_unary_init`/`copy_init` from `tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h`. | declared via `sfpu_helpers.hpp:72`; impl resolved from `sfpu_helpers.inl` | `(cb_input_tiles, cb_output_tiles)` | — | — | Must be called exactly once at the top of `kernel_main`, before any `tile_regs_acquire` or SFPU op. |
| Sub-phase B (final per-tile sum + constant) | helper | `compute_kernel_lib::sfpu_pipeline` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:1406–1412` | `sfpu_pipeline<SfpuBatching::Auto, SfpuInputPolicy::WaitAndPopPerTile, SfpuOutputPolicy::PerTile, SfpuDataFormatReconfig::INPUT_AND_OUTPUT>(chain, cb_output_tiles, /*num_tiles=*/1)` invoked once per input tile. | `cb_lgamma_a`, `cb_lgamma_a_half`, `cb_lgamma_a_one`, `cb_lgamma_a_three_halves` (via chain Loads) | `cb_output_tiles` | Manages `tile_regs_acquire/commit/wait/release`, the per-CB `cb_wait_front`/`cb_pop_front`, the output `cb_reserve_back`/`cb_push_back`, and unpack/pack data-format reconfig. Chain stride is `max_dst + 1 = 4` (D0..D3), `DEST_AUTO_LIMIT = 4` under fp32 half-sync → `batch_size = 4 / 4 = 1` tile per pipeline call — exactly correct. |
| Sub-phase B chain composition | helper | `compute_kernel_lib::sfpu_chain` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:1364–1371` | `sfpu_chain(Load<cb_lgamma_a, Dst::D0>{}, Load<cb_lgamma_a_half, Dst::D1>{}, Load<cb_lgamma_a_one, Dst::D2>{}, Load<cb_lgamma_a_three_halves, Dst::D3>{}, SfpuAdd<Dst::D0, Dst::D1, Dst::D0>{}, SfpuAdd<Dst::D0, Dst::D2, Dst::D0>{}, SfpuAdd<Dst::D0, Dst::D3, Dst::D0>{}, AddScalar<Dst::D0>{0x405BA32Eu /* ≈ 3.434189657547f, 3·log(π) */})` | — | — | Each `Load` from a distinct CB stays a separate `CompactLoad` (compaction only merges adjacent loads from the **same** CB — `sfpu_helpers.hpp:1234–1287`). The `NoMultiGroupCB` static_assert (`sfpu_helpers.hpp:1366–1369`) passes because each CB appears in exactly one Load. |
| Sub-phase B Load | helper | `compute_kernel_lib::Load<CB, Slot>` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:370–376` | One `Load<cb_lgamma_*, Dst::Dn>{}` per term (4 total). | `cb_lgamma_*` | DEST | After chain compaction each load becomes `CompactLoad<CB, /*DoWait=*/true, /*DoPop=*/true, Slot>` (`sfpu_helpers.hpp:390–405`). |
| Sub-phase B addition | helper | `compute_kernel_lib::SfpuAdd<In0, In1, Out>` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:1035–1039` | Three instances: `SfpuAdd<Dst::D0, Dst::D1, Dst::D0>{}`, `SfpuAdd<Dst::D0, Dst::D2, Dst::D0>{}`, `SfpuAdd<Dst::D0, Dst::D3, Dst::D0>{}`. | DEST | DEST | Wraps `add_binary_tile_init()` + `add_binary_tile(in0, in1, out)` via the `BinaryOp` CRTP base (`sfpu_helpers.hpp:297–313`). The pipeline calls `init()` once per op and `exec()` once per tile (batch_size = 1). |
| Sub-phase B constant addition | helper | `compute_kernel_lib::AddScalar<Slot>` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:937–941` | `AddScalar<Dst::D0>{0x405BA32Eu}` — `0x405BA32Eu` is the IEEE-754 bit pattern of `3.434189657547f`. The `uint32_t scalar` field is the fp32 bit pattern per `binop_with_scalar.h:24` ("fp32 value scalar encoded as uint32"). | DEST | DEST | Wraps `binop_with_scalar_tile_init()` + `add_unary_tile(d0, scalar)`. |
| `DEST_AUTO_LIMIT` constant | helper (read-only) | `compute_kernel_lib::DEST_AUTO_LIMIT` | `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:88–102` | — | — | — | Resolves to `4` under `fp32_dest_acc_en=True` + half-sync. Used implicitly by `sfpu_pipeline` to pick `batch_size`. |

### Raw APIs used for the per-offset Lanczos sub-procedure (one invocation per offset; runs 4× per input tile)

The Lanczos sub-procedure runs inside one `tile_regs_acquire / commit / wait / release` block per offset. DEST slot allocation across one invocation:

| Slot | Holds | Lifetime |
|------|-------|----------|
| **D0** | `input = a − (offset + 1)` then later overwritten with the **lgamma result** then pole-zero masked | start of recipe → packed at end |
| **D1** | `t = input + 5.5` → overwritten with `log(t)` then with the **pole-zero mask** | mid → end |
| **D2** | polynomial accumulator `series` (initialised to `1.0`) → overwritten with `log(series)` | full recipe |
| **D3** | scratch — used per-`j` for `(input + j) → coef[j] / (input + j)`, and for `(input + 0.5)` at the end | rolling |

Algebraic note: the kernel uses the simplification `L(y) = (input + 0.5) · log(t) + log(series) − input − 4.581061468643f` (derived under the Algorithm section). This avoids needing un-logged `t` in DEST after `log_tile` overwrites D1.

| # | Type | Function | File:Line | Args | Notes |
|---|------|----------|-----------|------|-------|
| 1 | raw_api | `cb_reserve_back(cb_lgamma_<offset>, 1)` | `tt_metal/hw/inc/api/compute/cb_api.h` (declaration) | — | Reserve the per-offset output CB BEFORE `tile_regs_acquire`. Pattern from `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/lgamma_kernel.cpp:37`. |
| 2 | raw_api | `tile_regs_acquire()` | `tt_metal/hw/inc/api/compute/reg_api.h` | — | Begin the DEST-acquire block; all subsequent SFPU ops up through the pack run inside this block. |
| 3 | raw_api | `copy_tile_to_dst_init_short(cb_input_tiles)` | `tt_metal/hw/inc/api/compute/tile_move_copy.h` | `(cb_input_tiles)` | Required before every `copy_tile` block (unpacker config). Re-issued whenever another SFPU init has run since the last `copy_tile`. |
| 4 | raw_api | `copy_tile(cb_input_tiles, 0, 1)` | `tile_move_copy.h` | `(cb_input_tiles, 0, 1)` — head of CB into D1 | D1 = `a`. The head tile is held via the surrounding `cb_wait_front(cb_input_tiles, 1)` and is NOT popped between sub-phases. |
| 5 | raw_api | `binop_with_scalar_tile_init()` | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:89` | — | Switch SFPU state to binop-with-scalar. |
| 6 | raw_api | `sub_unary_tile(idst, param1)` | `binop_with_scalar.h:32–34` | `(1, bit_cast<uint32_t>(offset - 4.5f))` | D1 = `a − (offset − 4.5)` = `(a − offset) + 4.5` = `input + 5.5` = **t**. `param1` is the fp32 bit pattern of the signed scalar (lines 22–24). |
| 7 | raw_api | `log_tile_init()` + `log_tile(idst)` | `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` → `log.h` (declarations); APPROX-mode log impl threads through `LOG_APPROX_MODE`. | `log_tile(1)` | D1 = `log(t)`. Default (non-approx) log — high precision. |
| 8 | raw_api | `fill_tile_init()` + `fill_tile(idst, val)` | `tt_metal/hw/inc/api/compute/eltwise_unary/fill.h:29, 74` | `fill_tile(2, 1.0f)` | D2 = 1.0 — polynomial accumulator seed. |
| 9–14 | raw_api (looped per `j ∈ {1,2,3,4,5,6}`) | per-term: `copy_tile_to_dst_init_short(cb_input_tiles)` + `copy_tile(cb_input_tiles, 0, 3)` + `binop_with_scalar_tile_init()` + `sub_unary_tile(3, bit_cast<uint32_t>(offset + 1.0f - j))` + `recip_tile_init()` + `recip_tile(3)` + `binop_with_scalar_tile_init()` + `mul_unary_tile(3, bit_cast<uint32_t>(coef[j]))` + `add_binary_tile_init()` + `add_binary_tile(2, 3, 2)` | `tile_move_copy.h`; `binop_with_scalar.h:32–34, 36–38, 89`; `eltwise_unary/recip.h:18–40`; `eltwise_binary_sfpu.h:35–37, 86` | per `j`: `coef[1] = 76.18009172947146f`; `coef[2] = -86.50532032941677f`; `coef[3] = 24.01409824083091f`; `coef[4] = -1.231739572450155f`; `coef[5] = 0.1208650973866179e-2f`; `coef[6] = -0.5395239384953e-5f` | After loop: D2 = `series = 1 + Σ coef[j] / (input + j)`. The init switches each iteration are unavoidable — `recip`, `binop_with_scalar`, and `add_binary` use disjoint SFPU state. The 6 iterations are written out as 6 inlined blocks (or one `for` loop in the kernel source — implementer's choice; both compile identically because `j` is a compile-time `constexpr`). |
| 15 | raw_api | `log_tile_init()` + `log_tile(idst)` | `log.h` (declarations) | `log_tile(2)` | D2 = `log(series)`. |
| 16 | raw_api | `copy_tile_to_dst_init_short(cb_input_tiles)` + `copy_tile(cb_input_tiles, 0, 0)` | `tile_move_copy.h` | `(cb_input_tiles, 0, 0)` | D0 = `a` (fresh from CB). |
| 17 | raw_api | `binop_with_scalar_tile_init()` + `sub_unary_tile(0, bit_cast<uint32_t>(offset + 1.0f))` | `binop_with_scalar.h:32–34, 89` | — | D0 = `a − (offset + 1)` = **input**. |
| 18 | raw_api | `copy_tile_to_dst_init_short(cb_input_tiles)` + `copy_tile(cb_input_tiles, 0, 3)` + `binop_with_scalar_tile_init()` + `sub_unary_tile(3, bit_cast<uint32_t>(offset + 0.5f))` | `tile_move_copy.h`, `binop_with_scalar.h:32–34, 89` | — | D3 = `a − (offset + 0.5)` = `input + 0.5`. |
| 19 | raw_api | `mul_binary_tile_init()` + `mul_binary_tile(a, b, out)` | `eltwise_binary_sfpu.h:43–45, 90` | `mul_binary_tile(3, 1, 3)` | D3 = `(input + 0.5) · log(t)`. |
| 20 | raw_api | `add_binary_tile_init()` + `add_binary_tile(3, 2, 3)` | `eltwise_binary_sfpu.h:35–37, 86` | — | D3 += `log(series)`. Now D3 = `(input + 0.5) · log(t) + log(series)`. |
| 21 | raw_api | `sub_binary_tile_init()` + `sub_binary_tile(3, 0, 3)` | `eltwise_binary_sfpu.h:39–41, 88` | `sub_binary_tile(3, 0, 3)` | D3 −= D0 = D3 − `input`. Now D3 = `(input + 0.5) · log(t) + log(series) − input`. |
| 22 | raw_api | `binop_with_scalar_tile_init()` + `sub_unary_tile(3, bit_cast<uint32_t>(4.581061468643f))` | `binop_with_scalar.h:32–34, 89` | — | D3 −= `4.581061468643f` = `5.5 − 0.918938531357171`. Now D3 = `L(y)` (Lanczos result for this offset, BEFORE pole zeroing). The constant `4.581061468643f` is `5.5f − 0.918938531357171f` per the algebraic simplification in the Algorithm section. |
| 23 | raw_api | `copy_tile_to_dst_init_short(cb_input_tiles)` + `copy_tile(cb_input_tiles, 0, 0)` | `tile_move_copy.h` | — | D0 = `a` (reload — was overwritten as `input` in step 17, then used in step 21). |
| 24 | raw_api | `unary_eq_tile_init()` + `unary_eq_tile(idst, param0)` | `tt_metal/hw/inc/api/compute/eltwise_unary/comp.h:78–85` | `unary_eq_tile(0, bit_cast<uint32_t>(offset + 1.0f))` | D0 = `(a == offset + 1)` mask. `param0` is the fp32 bit pattern of the comparison value (lines 30, 78). |
| 25 | raw_api | `binop_with_scalar_tile_init()` + `rsub_unary_tile(0, bit_cast<uint32_t>(1.0f))` | `binop_with_scalar.h:44–46, 89` | — | D0 = `1 − D0` = `(a != offset + 1)` mask. |
| 26 | raw_api | `mul_binary_tile_init()` + `mul_binary_tile(3, 0, 3)` | `eltwise_binary_sfpu.h:43–45, 90` | — | D3 *= D0 — zero the Lanczos result where `a == offset + 1`. |
| 27 | raw_api | `copy_tile_to_dst_init_short(cb_input_tiles)` + `copy_tile(cb_input_tiles, 0, 0)` | `tile_move_copy.h` | — | D0 = `a` (reload). |
| 28 | raw_api | `unary_eq_tile_init()` + `unary_eq_tile(0, bit_cast<uint32_t>(offset + 2.0f))` | `comp.h:78–85` | — | D0 = `(a == offset + 2)` mask. |
| 29 | raw_api | `binop_with_scalar_tile_init()` + `rsub_unary_tile(0, bit_cast<uint32_t>(1.0f))` | `binop_with_scalar.h:44–46, 89` | — | D0 = `(a != offset + 2)` mask. |
| 30 | raw_api | `mul_binary_tile_init()` + `mul_binary_tile(3, 0, 3)` | `eltwise_binary_sfpu.h:43–45, 90` | — | D3 *= D0 — zero the Lanczos result where `a == offset + 2`. D3 now holds the final `L(y)` for this offset. |
| 31 | raw_api | `tile_regs_commit()` / `tile_regs_wait()` | `reg_api.h` | — | MATH↔PACK sync. |
| 32 | raw_api | `pack_tile(idst, cb)` | `tt_metal/hw/inc/api/compute/pack.h` (declaration) | `pack_tile(3, cb_lgamma_<offset>)` — default `pack_tile<false>`, sequential pack at `fifo_wr_ptr`. | — |
| 33 | raw_api | `tile_regs_release()` | `reg_api.h` | — | Release DEST for the next sub-phase. |
| 34 | raw_api | `cb_push_back(cb_lgamma_<offset>, 1)` | `cb_api.h` (declaration) | — | Advances `fifo_wr_ptr` by one tile. |

After all four Lanczos sub-phases for a tile have completed, the compute kernel issues a single `cb_pop_front(cb_input_tiles, 1)`. The four `cb_lgamma_*` CBs each contain one fresh tile, ready for the sum sub-phase (helper-based, see "Helpers used" above).

### Helpers considered and rejected (per-offset Lanczos sub-procedure)

| Helper considered | File:Line | Reason rejected (concrete) |
|-------------------|-----------|----------------------------|
| `compute_kernel_lib::Lgamma<Slot>` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:700–704`; impl resolves to `lgamma_stirling_tile` (`tt_metal/hw/inc/api/compute/eltwise_unary/lgamma.h:15–17, 18–19` — Stirling, single-arg, bf16-suitable, **no reflection**). | **Hard prohibition.** The task spec ("No SFPU lgamma helpers in the compute kernel") forbids `lgamma_tile`, `lgamma_stirling_float_tile`, `lgamma_adjusted_tile`. `Lgamma<>` is the helper wrapper around `lgamma_stirling_tile`, so it inherits the prohibition. Also: it implements **Stirling**, not Lanczos — the helper's algorithm does not match the spec's required Lanczos recipe, so even without the prohibition the helper would compute the wrong polynomial. |
| `compute_kernel_lib::Recip<Slot>`, `Log<Slot>`, `SfpuAdd`, `SfpuSub`, `SfpuMul`, `AddScalar`, `SubScalar`, `MulScalar`, `FillTile`, `UnaryEq` for the in-recipe primitives | `sfpu_helpers.hpp:457–460` (`Recip`), `420–423` (`Log`), `1035–1039` (`SfpuAdd`), `1041–1046` (`SfpuSub`), `1047–1051` (`SfpuMul`), `937–941` (`AddScalar`), `943–948` (`SubScalar`), `949–955` (`MulScalar`), `1011–1015` (`FillTile`), `781–786` (`UnaryEq`) | These op structs are **only** usable inside an `sfpu_chain` + `sfpu_pipeline` invocation (`sfpu_helpers.hpp:1406–1412`). The pipeline takes a fixed chain, manages `tile_regs_acquire/commit/wait/release` for a single per-iteration DEST-fill pattern, and packs to one output CB. The Lanczos sub-procedure (steps 3–34 above) does NOT fit that shape: it is one monolithic `tile_regs_acquire` block containing 30+ ops with **interleaved** unpack-state init switches (every `recip_tile` follows a `binop_with_scalar` follows an `add_binary_tile` …). The chain framework expects ops to compose by DEST slot only — it has no representation for the per-step `*_tile_init()` re-issues that `lgamma_kernel.cpp` and our Lanczos recipe both need. Concretely, `sfpu_chain` compaction at `sfpu_helpers.hpp:1234–1287` only merges adjacent `Load` ops; it does NOT re-emit `*_tile_init()` between compute ops. The `apply` / `exec` dispatch in `SfpuChain::apply` (`sfpu_helpers.hpp:1186–1189`) calls `init()` once per element and `exec()` once per element, while our recipe requires re-init mid-chain (e.g. `binop_with_scalar` → `recip` → `binop_with_scalar` → `add_binary`). Wrapping each primitive as a single-op `sfpu_pipeline` would also wrap each with its own `tile_regs_acquire/release`, force a `pack_tile` per step, and require us to ferry intermediate values through CBs — explicitly forbidden by the kernel's DEST-slot allocation plan. Therefore the recipe is expressed with raw `*_tile_init()` + `*_tile()` calls inside a single `tile_regs_acquire/commit/wait/release` block, matching the pattern of `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/lgamma_kernel.cpp`. |
| `compute_kernel_lib::Where<...>` | `sfpu_helpers.hpp:1097–1100`; wraps `where_tile` (`tt_metal/hw/inc/api/compute/eltwise_unary/where.h:33–35`) | The pole-zeroing step is implementable either as `where_tile(mask, 0, result)` or as `result *= (1 − mask)` (rsub + mul). The `where_tile` form requires **three** DEST slots simultaneously (cond, true_val, false_val) plus the output, i.e. 4 slots in the worst case. Under fp32 + half-sync we only have D0..D3 — exactly 4 slots — but the Lanczos accumulator (D0=input, D1=log(t), D2=log(series), D3=result) already occupies all four. The `rsub + mul` form needs only 2 free slots at the moment of zeroing (one for the mask, one for the result, with D1/D2 reusable as scratch since their values are no longer needed), so it fits. The `Where` helper would also require its inputs to be in adjacent DEST slots picked at compile time, conflicting with our hand-allocated layout. The raw `unary_eq_tile` + `rsub_unary_tile` + `mul_binary_tile` sequence in steps 23–30 fits the slot budget exactly. |
| `compute_kernel_lib::binary_op` / `add`, `sub`, `mul` (FPU binary helpers in `binary_op_helpers.hpp`) for combining lgamma terms via CB → CB chains | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:250–314` | These helpers consume two CBs and produce a third CB, requiring chained intermediate CBs to combine four terms (`add(a, half, t1) → add(t1, one, t2) → add(t2, three_halves, t3) → add_const(t3, K)`). That uses two extra intermediate CBs (`t1`, `t2`) and forces a `tile_regs_acquire/release` cycle per add — three more cycles than the SFPU chain solution. The SFPU chain expresses the 4-way sum + constant within ONE DEST acquire/release using 4 slots — strictly cheaper, fewer CBs, fewer cycles. Choice for sub-phase B: `sfpu_chain` + `sfpu_pipeline`. |

### Per-offset constant table (precomputed bit patterns)

To prevent any drift between the design and the implementation, the implementer should encode these as `constexpr uint32_t` in the kernel with a comment showing the source `float` literal.

| Symbol | Float value | uint32 bit-cast | Used in (step) |
|--------|-------------|-----------------|----------------|
| `0.5` | `0.5f` | `0x3F000000u` | per-offset step 18; `bit_cast<uint32_t>(offset + 0.5f)` at offset=0 |
| `1.0` | `1.0f` | `0x3F800000u` | rsub mask completion (steps 25, 29); `bit_cast<uint32_t>(offset + 1.0f)` at offset=0 (step 17, 24) |
| `2.0` | `2.0f` | `0x40000000u` | `bit_cast<uint32_t>(offset + 2.0f)` at offset=0 (step 28) |
| `4.5` | `4.5f` | `0x40900000u` | `bit_cast<uint32_t>(offset − 4.5f)` at offset=0 → `bit_cast<uint32_t>(-4.5f) = 0xC0900000u` (step 6) |
| `-4.5` | `-4.5f` | `0xC0900000u` | step 6 at offset=0 (= subtract `−4.5` → add `4.5`) |
| `4.581061468643` | `4.581061468643f` | `0x40928D27u` (≈) | step 22 |
| `76.18009172947146` | `0x42985263u` (≈) | `coef[1]` (mul_unary_tile in step 9–14, j=1) |
| `-86.50532032941677` | `0xC2AD0440u` (≈) | `coef[2]` (j=2) |
| `24.01409824083091` | `0x41C03A35u` (≈) | `coef[3]` (j=3) |
| `-1.231739572450155` | `0xBF9DA6B7u` (≈) | `coef[4]` (j=4) |
| `0.1208650973866179e-2` | `0x3A9E66D6u` (≈) | `coef[5]` (j=5) |
| `-0.5395239384953e-5` | `0xB6B5236Au` (≈) | `coef[6]` (j=6) |
| `3.434189657547` | `3.434189657547f` (≈ `3 · log(π)`) | `0x405BA32Eu` | `AddScalar<D0>` in sub-phase B |

The implementer MUST recompute each bit pattern with `std::bit_cast<uint32_t>(literal_f)` (C++20) or an equivalent `union` punning at kernel compile time — do not hand-copy hex values. The hex values above are reference / sanity targets; small last-bit differences in encoding are acceptable as long as the underlying `float` literal matches.

For `offset ≠ 0`, the per-offset constants (`offset + 1.0f`, `offset + 0.5f`, etc.) become `0.5+1.0 = 1.5`, `1.0+1.0 = 2.0`, etc. — implementer recomputes via `constexpr float` arithmetic, then bit-casts. There are 4 offsets, so the implementer should template the Lanczos sub-procedure on `float OFFSET` (`template <float OFFSET>` is allowed in C++20 NTTP; alternatively use a function-like macro with `OFFSET` as a token, expanded four times).

## Compute Phases

Sequential per-tile execution. Phase boundaries are explicit because each phase is its own helper invocation or `tile_regs_acquire/release` block.

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `cb_wait_front(cb_input_tiles, 1)` | raw | `cb_input_tiles` (1 tile, ready) | — | `cb_input_tiles` front held (NOT popped) |
| 1 | Lanczos sub-procedure with `OFFSET = 0.0f` — steps 1–34 in the API Mapping inside one `tile_regs_acquire/commit/wait/release` block, packing D3 to `cb_lgamma_a` | raw | `cb_input_tiles` (1 tile, front, not popped) | `cb_lgamma_a` (1 tile pushed) | `cb_input_tiles` still front-held; `cb_lgamma_a` +1 tile |
| 2 | Lanczos sub-procedure with `OFFSET = 0.5f` — same 34 raw steps with `OFFSET = 0.5f` substituted | raw | `cb_input_tiles` (1 tile, front, not popped) | `cb_lgamma_a_half` (1 tile pushed) | `cb_input_tiles` still front-held; `cb_lgamma_a_half` +1 tile |
| 3 | Lanczos sub-procedure with `OFFSET = 1.0f` | raw | `cb_input_tiles` (1 tile, front, not popped) | `cb_lgamma_a_one` (1 tile pushed) | `cb_input_tiles` still front-held; `cb_lgamma_a_one` +1 tile |
| 4 | Lanczos sub-procedure with `OFFSET = 1.5f` | raw | `cb_input_tiles` (1 tile, front, not popped) | `cb_lgamma_a_three_halves` (1 tile pushed) | `cb_input_tiles` still front-held; `cb_lgamma_a_three_halves` +1 tile |
| 5 | `cb_pop_front(cb_input_tiles, 1)` — release the input tile now that all four Lanczos sub-phases have read it | raw | `cb_input_tiles` (1 tile, front) | — | `cb_input_tiles` −1 tile |
| 6 | Sum sub-phase: `sfpu_pipeline(sfpu_chain(Load<cb_lgamma_a, D0>, Load<cb_lgamma_a_half, D1>, Load<cb_lgamma_a_one, D2>, Load<cb_lgamma_a_three_halves, D3>, SfpuAdd<D0, D1, D0>, SfpuAdd<D0, D2, D0>, SfpuAdd<D0, D3, D0>, AddScalar<D0>{0x405BA32Eu}), cb_output_tiles, 1)`. Internally: acquire → 4× `cb_wait_front + copy_tile + cb_pop_front` → 3× `add_binary_tile` (D0+=Dn) → 1× `add_unary_tile` (D0+=K) → commit/wait → `cb_reserve_back(cb_output_tiles, 1) + pack_tile(0, cb_output_tiles) + cb_push_back(cb_output_tiles, 1)` → release. | helper (`sfpu_pipeline` + `sfpu_chain`) | `cb_lgamma_a` (1), `cb_lgamma_a_half` (1), `cb_lgamma_a_one` (1), `cb_lgamma_a_three_halves` (1) | `cb_output_tiles` (1 tile pushed) | All four `cb_lgamma_*` −1 tile; `cb_output_tiles` +1 tile |

The compute kernel wraps phases 0–6 in `for (uint32_t i = 0; i < num_tiles; ++i)`. With 2-page intermediate CBs and 1 push/pop per CB per iteration, the loop is self-balanced (`push 1 → pop 1` per CB per iteration; at most one in-flight tile per intermediate CB at any time).

## Build Order

Suggested incremental bring-up sequence for the implementer. Each step yields a runnable artifact that can be inspected via `scripts/tt-probe.sh --dev` with deterministic inputs.

| Step | Build target | Verification |
|------|--------------|--------------|
| 1 | Python entrypoint (`multigammaln_lanczos.py`) with validation and `__init__.py`. Stub program descriptor that allocates the output tensor and dispatches a no-op compute kernel. | `from ttnn.operations.multigammaln_lanczos import multigammaln_lanczos` succeeds; the `_validate_input` negative tests pass (bf16, RM, rank, alignment). |
| 2 | Pass-through pipeline: reader → `cb_input_tiles` → compute does `cb_wait_front + copy_tile(0,0) + pack_tile(0, cb_output_tiles) + cb_push_back + cb_pop_front` → writer. No Lanczos yet. | A constant-fill input round-trips through the kernel unchanged for shape `(1,1,32,32)`. Validates CB sync, work distribution, TensorAccessor wiring. Probe with constant `a = 5.0` → expect output `5.0`. |
| 3 | Sum sub-phase only — replace Lanczos sub-phases with `pack_tile` of `a` (the input itself) into all four `cb_lgamma_*`. Output should equal `4·a + 3·log(π) ≈ 4·a + 3.434190`. | Probe with constant `a = 1.0` → expect output `≈ 7.434190`. Validates `sfpu_chain` + `SfpuAdd` + `AddScalar` and the 4-input CompactLoad pattern. |
| 4 | Lanczos sub-phase for `OFFSET = 0.0f` only; the other three `cb_lgamma_*` are still filled with `a` (pass-through). Output should equal `L(a) + 3·a + 3·log(π)`. | Probe with constant `a = 5.0` → expect `lgamma(5) + 3·5 + 3·log(π) = 3.178054 + 15 + 3.434190 = 21.612244`. Probe also with `a = 1.0` to verify pole zeroing: `L(1) = 0`, so output `= 0 + 3·1 + 3·log(π) = 6.434190`. |
| 5 | Enable all four `OFFSET` values. Output should equal full `multigammaln(a, 4)`. | Probe with constant `a = 3.0` → expect `torch.special.multigammaln(torch.tensor(3.0), 4)` ≈ `lgamma(3) + lgamma(2.5) + lgamma(2) + lgamma(1.5) + 3·log(π) = 0.693147 + 0.284683 + 0 + (-0.120782) + 3.434190 = 4.291237`. Probe with constant `a = 5.0` → ≈ `3.178054 + 2.453736 + 1.791759 + 1.200974 + 3.434190 = 12.058713`. |
| 6 | Multi-tile, multi-core. Use `ttnn.split_work_to_cores` and per-core RT args. | Run the acceptance test (`tests/ttnn/unit_tests/operations/multigammaln_lanczos/test_multigammaln_lanczos.py`). |

`scripts/tt-probe.sh --dev` is recommended through step 5 (CB sanitizer + `LLK_ASSERT` halts catch DEST-overflow and CB-overrun bugs at the failing instruction). Switch to default mode for step 6 once the recipe is stable.

## Key Risks and Gotchas

| Risk | Mitigation |
|------|------------|
| **Hard prohibition on SFPU lgamma helpers** — `lgamma_tile`, `lgamma_stirling_float_tile`, `lgamma_adjusted_tile`, and `compute_kernel_lib::Lgamma<>` are explicitly forbidden by the task spec. Using any of them invalidates the operation. | The implementer must use only the raw primitives listed in the "Raw APIs used" table. Code review must grep the compute kernel for these names and fail the review if found. |
| **DEST budget = 4 tiles** — `fp32_dest_acc_en=True` halves DEST to D0..D3 per `tile_regs_acquire/release` (`dest_helpers.hpp:88–102`). The Lanczos sub-procedure uses exactly D0..D3. The sum chain also uses D0..D3 (`stride = max_dst + 1 = 4`, `batch_size = DEST_AUTO_LIMIT / stride = 4/4 = 1`). | No room for additional concurrent slots. Do NOT bump batch size in the sum chain. Do NOT add extra DEST holdings in the Lanczos recipe. The slot allocation (D0=input/result, D1=log(t)/mask, D2=log(series), D3=scratch/final-accumulator) is fixed. |
| **Init-state switching** — each `recip_tile`, `binop_with_scalar` (add/sub/mul/rsub_unary), `add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`, `log_tile`, `fill_tile`, `unary_eq_tile`, `where_tile` requires its own `*_tile_init()` before the next call. Re-issuing an init when state already matches is cheap; omitting an init when state has changed produces silent wrong results. | When in doubt, re-issue the init. The Raw APIs table above re-issues `binop_with_scalar_tile_init()` between every binop-with-scalar block and after any other SFPU op that changed the unpacker state. The implementer should keep this convention. |
| **`recip_tile` legacy mode default** — `recip_tile<bool legacy_compat = true>` defaults to `true` (`tt_metal/hw/inc/api/compute/eltwise_unary/recip.h:18, 36`). Under `fp32_dest_acc_en=True`, legacy mode and modern mode may differ in precision. | Use the default (`recip_tile(idst)` without explicit template argument) for consistency with the rest of the TTNN codebase. If precision is later flagged as inadequate at the integer-pole boundary, retry with `recip_tile<false>` and benchmark. |
| **Pole zeroing must not branch on input** — the task spec explicitly forbids "branching on the input value". `where`-style elementwise select via a mask tile is NOT a branch; it is a select expressed as `result *= (1 − mask)`. | Use exactly the `unary_eq_tile → rsub_unary_tile → mul_binary_tile` sequence in the Raw APIs table. Do NOT add an `if (a > 1.5)` guard in the kernel. NaN / −Inf for out-of-domain inputs must fall through naturally. |
| **Domain (1.5, 2.0) is the hardest sub-region** — `L(a − 1.5)` for `a ∈ (1.5, 2.0)` has argument `(0, 0.5)` where the Lanczos polynomial without reflection is meaningfully less accurate than `torch`'s libm `lgamma` (double precision). The "+5.5" shift in `t` plus the `log(series)` term keep the result finite, but absolute error can exceed 0.1 in this region. | The acceptance test uses `rtol = 0.1`, `atol = 0.5` and restricts random inputs to `a ∈ [2.0, 10.0]` to avoid spurious failures from the worst sub-region. A separate `pytest.mark.parametrize` covers `a ∈ (1.55, 1.95)` with the same loose tolerances and explicit acceptance that this region is the limit of Lanczos-at-fp32. |
| **`AddScalar`, `sub_unary_tile`, `mul_unary_tile` all expect a `uint32_t` bit-cast of an fp32 value** (`binop_with_scalar.h:24`). Passing a literal `int` or implicit `int → uint32_t` conversion produces a denormal / wrong-magnitude scalar. | Use `std::bit_cast<uint32_t>(literal_f)` (C++20) or an explicit `union { float f; uint32_t u; }` punning. Pre-compute as `constexpr uint32_t` with a comment showing the source float literal (see "Per-offset constant table"). |
| **`pack_tile<false>` advances `fifo_wr_ptr`** — combined with `cb_push_back`, the reserved-then-pushed CB region must be re-reserved per tile (see `~/.claude/projects/.../MEMORY.md` "cb_push_back Always Advances fifo_wr_ptr"). Don't try to "stream-write multiple Lanczos terms into one accumulator CB" — that requires `pack_tile<true>` with held wr_ptr, a much trickier protocol. | Use four distinct CBs (`cb_lgamma_a`, `_half`, `_one`, `_three_halves`), each with its own `cb_reserve_back / pack_tile<false> / cb_push_back` per tile. |
| **`cb_input_tiles` is read four times without pop** — phases 1–4 each `copy_tile(cb_input_tiles, 0, …)` against the same front tile. `cb_pop_front` is deferred until after phase 4. If the implementer accidentally pops the front between sub-phases, the next sub-phase reads stale (or unmapped) memory. | One `cb_wait_front(cb_input_tiles, 1)` at phase 0, one `cb_pop_front(cb_input_tiles, 1)` at phase 5. Add a code-review checklist item. |
| **`init_sfpu(cb_input_tiles, cb_output_tiles)` is called once** — but the kernel writes to five distinct CBs (`cb_lgamma_*` ×4, `cb_output_tiles`). All five have the same dtype (float32), so the packer config from `init_sfpu(cb_input_tiles, cb_output_tiles)` is correct for all of them; no mid-kernel reconfig needed. The SFPU pipeline at phase 6 internally reconfigures (per `SfpuDataFormatReconfig::INPUT_AND_OUTPUT` default — `sfpu_helpers.hpp:1410`), which is a no-op when input/output formats are identical. | Do NOT call `init_sfpu` more than once. |
| **`UnpackToDestMode.UnpackToDestFp32`** must be set on every fp32 CB in the compute config's `unpack_to_dest_mode` list, mirroring `ttnn/ttnn/operations/multigammaln/multigammaln_program_descriptor.py:170–183`. Without it, the unpacker truncates fp32 mantissa to TF32 (~10 bits) when copying CB tiles into DEST — silently degrading precision on the `cb_lgamma_*` reload during sub-phase B. | Build the `unpack_modes` list in the program descriptor; set `UnpackToDestFp32` for indices `(0, 16, 24, 25, 26, 27)`. Test step 5 will surface this regression if missed. |
| **`a == off + 1` mask uses `unary_eq_tile` (fp32 exact equality)** — only fires when `a` is *exactly* the integer pole. Random fp32 inputs almost never hit exact `1.0`, `1.5`, `2.0`, `2.5`, `3.0`, `3.5` — so for the acceptance test, the pole-zero codepath is mostly dormant. Probe testing at exact integer / half-integer inputs is required to validate it. | Build-order step 4 probes `a = 1.0` and `a = 2.0` explicitly. The acceptance test also includes integer probes (see "test_pole_zeroing_at_exact_integers"). |
| **Math fidelity = HiFi4 is required** for the precision target. `fp32_dest_acc_en=True` alone is not sufficient — without HiFi4, FPU multiplies inside `mul_binary_tile` / `mul_unary_tile` would run at LoFi precision and the polynomial would lose precision. | Wire both in `ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)`. Do NOT expose to caller (Phase 0). |
