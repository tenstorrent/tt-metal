# Operation Design: groupnorm_sc_N_1_HW_C

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (normalization) |
| Goal | Single-core GroupNorm on `(N, 1, H*W, C)` tensors, including `(C/num_groups) % 32 != 0` from Phase 0. |
| Math | For each group `g`: `mean_g = (1/N_per_g) · Σ x[h, c∈g]`; `var_g = (1/N_per_g) · Σ x²[h, c∈g] − mean_g²`; `rcp_std_g = 1 / sqrt(var_g + eps)`; `y[h, c] = (x[h, c] − mean_g(c)) · rcp_std_g(c) · γ[c] + β[c]` where `g(c)` is the group containing channel `c`. `N_per_g = HW · (C/num_groups)`. |
| Mode | Derivative (TTNN generic op + custom kernels via `ttnn.generic_op`) |
| References | `.claude/skills/groupnorm-partial-channels/SKILL.md`, `tech_reports/tensor_layouts/tensor_layouts.md`, `.claude/references/ttnn-cb-memory-fundamentals.md`, `.claude/references/generic_op_template/` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank=4, `shape[1]==1`, dtype∈{bf16}, layout∈{TILE, ROW_MAJOR}, interleaved DRAM | — | tensor metadata → host; addr → RT |
| `num_groups` | `int` | yes | `C % num_groups == 0`, `1 ≤ num_groups ≤ C` | — | CT (`G`, `Cg = C/G`) |
| `gamma` | `ttnn.Tensor` (kw, optional) | no | shape `(1, 1, 1, C)`, dtype=bf16, layout=ROW_MAJOR, DRAM | `None` | host: dtype/shape/addr; flag `HAS_GAMMA` → CT |
| `beta` | `ttnn.Tensor` (kw, optional) | no | shape `(1, 1, 1, C)`, dtype=bf16, layout=ROW_MAJOR, DRAM | `None` | host: dtype/shape/addr; flag `HAS_BETA` → CT |
| `eps` | `float` | no | `> 0` | `1e-5` | CT (bit-packed `float`) |

Validation (raises `ValueError`, separate from the registry gates):
- `input_tensor.shape` rank != 4 → `ValueError("input must be rank 4 (N, 1, H*W, C)")`
- `input_tensor.shape[1] != 1` → `ValueError("dim[1] must be 1")`
- `C % num_groups != 0` → `ValueError("C must be divisible by num_groups")`
- `gamma is not None` and `gamma.shape != (1, 1, 1, C)` → `ValueError("gamma shape must be (1, 1, 1, C)")`
- `beta is not None` and `beta.shape != (1, 1, 1, C)` → `ValueError("beta shape must be (1, 1, 1, C)")`

Registry-model declarations (in the op file, separate from the above):

```python
def tag_alignment(inputs, axes):
    shape = inputs[0]
    HW, C = shape[-2], shape[-1]
    if HW % 32 == 0 and C % 32 == 0: return "tile_aligned"
    if C % 32 != 0:                   return "c_non_aligned"
    return "hw_non_aligned"

INPUT_TAGGERS = {"alignment": tag_alignment}

SUPPORTED = {
    "dtype":         [ttnn.bfloat16],
    "layout":        [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment":     ["tile_aligned", "c_non_aligned"],
    "affine":        ["gamma_beta", "gamma_only", "no_affine"],
    "affine_dtype":  [ttnn.bfloat16],
    "affine_layout": [ttnn.ROW_MAJOR_LAYOUT],
}

EXCLUSIONS = []
```

`validate()` raises `NotImplementedError` for any axis value outside `SUPPORTED`. `INVALID` lives in `eval/golden_tests/groupnorm_sc_N_1_HW_C/feature_spec.py`; the op file is INVALID-agnostic.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(N, 1, HW, C)`. `HW % 32 == 0` enforced by SUPPORTED.alignment (Phase 0). `C` may be non-tile-aligned (`c_non_aligned` bucket). |
| Dtype | `ttnn.bfloat16` (Phase 0) |
| Layout | `ttnn.TILE_LAYOUT` or `ttnn.ROW_MAJOR_LAYOUT` |
| Memory | interleaved DRAM |

### Optional gamma / beta

| Property | Requirement |
|----------|-------------|
| Shape | `(1, 1, 1, C)` (one stick of `C` bf16 elements) |
| Dtype | `ttnn.bfloat16` |
| Layout | `ttnn.ROW_MAJOR_LAYOUT` |
| Memory | interleaved DRAM |

### Output

| Property | Value |
|----------|-------|
| Shape | `(N, 1, HW, C)` (same as input) |
| Dtype | same as input (`ttnn.bfloat16` in Phase 0; contract: `output.dtype == input.dtype`) |
| Layout | `ttnn.TILE_LAYOUT` (always — independent of input layout) |
| Memory | interleaved DRAM |

## Dataflow Strategy

Single-core: no inter-Tensix communication. Within the one Tensix, the contract is reader (NCRISC) → compute (TRISCs) → writer (BRISC), all via L1 CBs.

The op runs in **two phases** per batch `n`. Both phases read `input` from DRAM (input is too large to cache in L1 — we re-stream).

**Phase R — reduce (per batch `n`, per group `g`, per `c-tile` in g's span, per `HW`-row):**
- Reader reads input tile `input[n, r, T]` (`T` = current c-tile, `r` = HW tile row) and pushes to `cb_input_tiles_R`.
- Reader pre-populates `cb_group_masks` ONCE at startup with `G × tile_span_per_group` row-replicated full mask tiles (mask `g` at tile `T` = 32×32 tile where every row equals the 1×32 pattern with 1.0 at lanes belonging to group `g` within tile `T`, 0.0 elsewhere). Reader never pops from `cb_group_masks`.
- Compute multiplies input by the appropriate mask tile (`mul<NONE>`), accumulates the masked product's scalar reduction into `cb_group_sum[g]`, and similarly accumulates the squared product into `cb_group_sumsq[g]`.

**Phase R/post — finalize stats (per batch `n`, per group `g`):**
- Compute reads `cb_group_sum[g]` and `cb_group_sumsq[g]`, divides by `N_per_g = HW · Cg`, computes `var_g = sumsq/N − mean²`, then `rcp_std_g = rsqrt(var_g + eps)`. Stats live in `cb_group_mean` and `cb_group_rcp_std` (G tiles each, `(0,0)` = scalar).

**Phase A — apply (per batch `n`, per output `c-tile` `T`, per HW row `r`):**
- Reader re-reads `input[n, r, T]` and pushes to `cb_input_tiles_A`. Reader also reads `gamma[T]` and `beta[T]` ONCE per `T` (small one-stick reads) and pushes to `cb_gamma_tile_T` and `cb_beta_tile_T`; both consumed once per `T` then re-loaded for next `T`.
- Compute first builds two row-replicated full tiles for output tile `T`: `cb_means_tile_T` and `cb_rcp_std_tile_T`, by iterating over groups that touch `T` and doing `mul<SCALAR>(mask_g[T], mean_g) → add into means_tile_T`; same for `rcp_std_tile_T`. These persist across the inner HW loop (`WaitUpfrontNoPop` policy) and are popped at the end of each `T`.
- For each `r`: compute `out = (input − means_tile_T) · rcp_std_tile_T · gamma_tile_T + beta_tile_T` (skipping the multiply by `gamma_tile_T` if `HAS_GAMMA=0` and the add of `beta_tile_T` if `HAS_BETA=0`). Push to `cb_output_tiles`. Writer writes to DRAM.

Both phases stream input from DRAM. Stats and masks live in L1 throughout (per the skill's L1 invariant: footprint scales with `G`, not with `HW` or `C`).

For `ROW_MAJOR_LAYOUT` input: in **both** phases, the reader uses `dataflow_kernel_lib::read_sticks_for_tilize` to write row-major sticks into `cb_input_rm_R` / `cb_input_rm_A`, and the compute uses `compute_kernel_lib::tilize` to convert into `cb_input_tiles_R` / `cb_input_tiles_A`. For `TILE_LAYOUT`, the reader writes tiles directly into `cb_input_tiles_R` / `cb_input_tiles_A` (no tilize step on compute).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | Whole tensor — one core handles the entire `(N, 1, HW, C)` input. |
| Grid | `1 × 1` (single core at `CoreCoord(0, 0)`) per the SINGLE-CORE constraint. |
| Per-core work | All `N` batches, all `G` groups, all `Ct` output channel-tiles, all `Ht` HW-tile-rows. |
| Remainder | None — single core owns everything. The kernel iterates `for n in 0..N: phase_R(n); phase_A(n)`. |

`Ht = HW / 32` (always integer in Phase 0 since `HW % 32 == 0`).
`Ct = ceil(C / 32)` (Ct tiles along the channel axis; the last tile may be partially-padded when `C % 32 != 0`).
`Cg = C / num_groups`. `tile_span_per_group_max = ceil((Cg + 31) / 32) + (1 if Cg % 32 != 0 else 0)` — bounded above by 2 when `Cg < 32`, otherwise `ceil(Cg/32)+1`. Host computes a per-group start-tile / end-tile list and passes via runtime args.

## Circular Buffers

CB indices follow the convention 0–7 input, 8–15 special, 16–23 output, 24–31 intermediate. Page size in bytes uses `tile_size(bf16) = 2048`. Stick size `stick_bytes = C * 2`.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_rm_R` | 0 | `stick_bytes` (rounded up to DRAM align) | 32 | bf16 | reader (RM input only) | compute tilize (R) | only when input layout = ROW_MAJOR; phase R streaming. Double-/multi-buffered to 32 sticks = 1 tile-row. Per `dataflow_kernel_lib::read_sticks_for_tilize` (TILE granularity). |
| `cb_input_rm_A` | 1 | `stick_bytes` | 32 | bf16 | reader (RM input only) | compute tilize (A) | only when input layout = ROW_MAJOR; phase A streaming. |
| `cb_input_tiles_R` | 2 | 2048 | 2 | bf16 | reader (tile input) or compute tilize (R) | compute (R) | phase R streaming; double-buffered. |
| `cb_input_tiles_A` | 3 | 2048 | 2 | bf16 | reader (tile input) or compute tilize (A) | compute (A) | phase A streaming; double-buffered. |
| `cb_gamma_rm` | 4 | `stick_bytes` | 32 | bf16 | reader (RM gamma stick replicated 32×, only when `HAS_GAMMA=1`) | compute tilize gamma | one-shot per output tile T in phase A (re-loaded per T). |
| `cb_beta_rm` | 5 | `stick_bytes` | 32 | bf16 | reader (RM beta stick replicated 32×, only when `HAS_BETA=1`) | compute tilize beta | one-shot per output tile T in phase A. |
| `cb_gamma_tile` | 6 | 2048 | 1 | bf16 | compute tilize | compute apply | only when `HAS_GAMMA=1`; one tile per output tile T, popped after the HW loop for that T. Row-replicated (each row = γ row 0 of T). |
| `cb_beta_tile` | 7 | 2048 | 1 | bf16 | compute tilize | compute apply | only when `HAS_BETA=1`; same as gamma. |
| `cb_scaler_one` | 8 | 2048 | 1 | bf16 | reader (one-shot at startup) | compute reduce (R) | populated once at startup; held forever; used by every `reduce<SUM, REDUCE_SCALAR>` call. Scaler value = 1.0 (we do `/N` manually after reduce). |
| `cb_group_masks` | 9 | 2048 | `mask_count = G × tile_span_per_group` (host-supplied CT bound) | bf16 | reader (built once at startup) | compute (R and A) | populated once; held forever; never popped. Each tile = full 32×32 row-replicated mask for `(g, T)` slot `g × tile_span_per_group + (T − first_tile_of_g)`. |
| `cb_inv_N_scalar` | 10 | 2048 | 1 | bf16 | reader (one-shot at startup) | compute (R/post) | scalar tile with `1/N_per_g` at `(0,0)` for the divide-by-N step. Persisted with `WaitUpfrontNoPop`. |
| `cb_group_sum` | 24 | 2048 | `G` | bf16 | compute (R) via reduce-accumulate | compute (R/post) | one slot per group; output of `reduce<SUM, REDUCE_SCALAR>` with `Accumulate` template across all `(T, r)` iterations for group `g`. Lifetime: phase R fill → phase R/post consume. |
| `cb_group_sumsq` | 25 | 2048 | `G` | bf16 | compute (R) via reduce-accumulate on squared input | compute (R/post) | same shape and lifetime as `cb_group_sum`. |
| `cb_group_mean` | 26 | 2048 | `G` | bf16 | compute (R/post) | compute (A) | one tile per group; scalar `mean_g` at `(0,0)`. Persisted across all of phase A using `WaitUpfrontNoPop`. Popped once at end of phase A for batch n. |
| `cb_group_rcp_std` | 27 | 2048 | `G` | bf16 | compute (R/post) | compute (A) | one tile per group; scalar `rcp_std_g` at `(0,0)`. Persisted with `WaitUpfrontNoPop`. |
| `cb_scratch_a` | 28 | 2048 | 2 | bf16 | compute | compute | scratch for intermediate computations in R/post (e.g., `mean²`, `sumsq/N`) and in A (per-T expansion intermediate `mul<SCALAR>(mask, scalar)` before accumulating into `cb_means_tile_T` / `cb_rcp_std_tile_T`). Double-buffered. |
| `cb_means_tile_T` | 29 | 2048 | 1 | bf16 | compute (A, expansion sub-phase) | compute (A, apply sub-phase) | one tile, rebuilt per output T. Row-replicated full tile of expanded per-channel means. `WaitUpfrontNoPop` across the HW loop for T; popped at end of T. |
| `cb_rcp_std_tile_T` | 30 | 2048 | 1 | bf16 | compute (A, expansion sub-phase) | compute (A, apply sub-phase) | same as `cb_means_tile_T` but for rcp_std. |
| `cb_output_tiles` | 16 | 2048 | 2 | bf16 | compute (A) | writer | phase A streaming output; double-buffered. |

L1 budget sanity check (Phase 0, `G ≤ 32`, `tile_span_per_group ≤ 2`, `Ht`, `Ct` arbitrary):
- Input/output streaming: ~`8 × 2KB` = 16 KB.
- RM-input staging (if RM): `2 × 32 × stick_bytes`. For `C = 1024`: `2 × 32 × 2048` = 128 KB. For `C = 320`: 40 KB.
- γ/β staging (if affine): `2 × 32 × stick_bytes` + `2 × 2KB` = same order as input RM staging but only one T at a time. Bounded by `Ct=1` worth of γ.
- Scaler / inv_N: ~4 KB.
- Masks: `G × tile_span_per_group × 2KB` ≤ `32 × 2 × 2KB` = 128 KB.
- Stats: `4G × 2KB` ≤ `128 × 2KB` = 256 KB.
- Expansion + scratch: `4 × 2KB` = 8 KB.
- **Total worst case ≤ ~540 KB**, well under Wormhole L1 1.5 MB. The invariant "scales with `G`, not with `HW` or `C`" is preserved — the only `C`-scaling component is the RM staging (which only exists when input layout is RM and can be lowered by reducing the double-buffer rows).

CB sync invariants (producer push count = consumer wait count):
- `cb_input_tiles_R`: reader pushes 1 tile per `(g, T_in_g_span, r)` triplet; compute waits 1 tile per same triplet (via `binary_op` `WaitAndPopPerTile`). For one batch: total pushes = `Σ_g (tile_span_g × Ht) = Ht × Σ_g tile_span_g`. For `Cg < 32`: typically `≤ 2 × G × Ht`; for `Cg ≥ 32`: `Σ_g ceil(Cg/32)+1 ≈ Ct + G × Ht`.
- `cb_input_tiles_A`: reader pushes 1 tile per `(T, r)`; compute waits 1 per same. Total per batch = `Ct × Ht`.
- `cb_group_masks`: reader pushes `mask_count` once at startup; compute waits via `cb_wait_front(cb_group_masks, mask_count)` once at startup and never pops.
- `cb_group_sum` / `cb_group_sumsq`: written `G` times via `reduce` (one push at the final-iteration `Accumulate` of each group); read `G` times in R/post; popped at end of R/post.
- `cb_group_mean` / `cb_group_rcp_std`: written `G` times in R/post; in A waited `G` upfront and never popped; popped once at end of A for batch n.

## API Mapping

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|------------------------|--------------------------|---------------------------|--------------|
| R: scaler setup (one-shot) | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:100-101` | `<cb_id=cb_scaler_one, PoolType=SUM, ReduceDim=REDUCE_SCALAR>` (reduce_factor default 1 — SUM uses 1.0). Called by reader. | — | `cb_scaler_one` | scaler CB must be bf16; pool-type-aware overload used as mandated. Helper manages reserve+push. |
| R: inv_N scalar tile | raw_api | `cb_reserve_back` + L1 write + `cb_push_back` | `tt_metal/include/dataflow_api.h` (CB ops) | Reader writes one bf16 = `1.0f / N_per_g` to position `(0,0)` of one tile in `cb_inv_N_scalar`. Rest zero. | — | `cb_inv_N_scalar` | **Helpers considered and rejected**: `prepare_reduce_scaler` (`reduce_helpers_dataflow.hpp:65-67`) builds scaler tiles for the **reduce LLK** only — its docstring at `reduce_helpers_dataflow.hpp:21-25` says *"They must ONLY be used for that purpose — not for arbitrary constant tiles."* We're using this tile as the operand of `mul<SCALAR>` (a BINARY op), not as a reduce scaler. Bypassing this restriction would mis-use the helper contract. No `fill_constant_tile` helper exists in `cb_helpers_dataflow.hpp` (verified by reading `ttnn/cpp/ttnn/kernel_lib/cb_helpers_dataflow.hpp` — only CB sync helpers). Manual fill is the only correct option. |
| R: mask construction (one-shot) | raw_api | Reader fills `mask_count` tiles in `cb_group_masks` | — | For each `(g, T)` slot, reader writes the row pattern (bf16 1.0 at lanes in group g, 0.0 elsewhere) replicated down 32 rows of a tile, using tile face layout (4 × 16×16 faces). | — | `cb_group_masks` | **Helpers considered and rejected**: No mask-tile-construction helper exists in `kernel_lib/` (checked `tilize_helpers_dataflow.hpp`, `cb_helpers_dataflow.hpp`, `reduce_helpers_dataflow.hpp`). The skill explicitly says mask construction is op-specific control flow that the implementer writes (SKILL.md:124–130: *"There is no `reduce_with_mask_per_group` helper, and there shouldn't be"*). FillTile-style SFPU helpers (`sfpu_helpers.hpp:**` — see `FillTile` in compute helpers) fill a tile with a single uniform value, not a per-lane pattern — they cannot build the row mask. |
| R/A: input read (tile-layout input) | raw_api | `noc_async_read` + `noc_async_read_barrier` via `TensorAccessor` | `tech_reports/tensor_accessor/tensor_accessor.md`; `tt_metal/api/dataflow/dataflow_api.h` | Per (n, r, T) page id read. | — | `cb_input_tiles_R` / `cb_input_tiles_A` | **Helpers considered and rejected**: No "read interleaved tile into CB" helper covers this — the standard pattern is direct `TensorAccessor::get_noc_addr` + `noc_async_read` + `cb_reserve_back`/`cb_push_back`. This is the unanimous template-op pattern (see `.claude/references/generic_op_template/kernels/template_op_reader.cpp`). |
| R/A: input read (RM input) | helper | `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-93` | `<cb_id=cb_input_rm_R, TilizeGranularity=TILE>`. Reads `Ht × 32 = HW` sticks total per c-tile column; uses `start_page` per HW row and `byte_offset_within_page = T * 32 * elem_size` to chunk along C if needed. | — | `cb_input_rm_R` / `cb_input_rm_A` | Helper manages CB reserve+push and the partial-row pad. |
| R/A: tilize input (RM only) | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:178-187` | `<block_width_tiles=1, input_cb=cb_input_rm_R, output_cb=cb_input_tiles_R>` invoked per HW tile-row block. We tilize 1 c-tile column at a time to keep RM staging small. | `cb_input_rm_R` / `cb_input_rm_A` | `cb_input_tiles_R` / `cb_input_tiles_A` | Helper handles reserve/wait/pop and packs into tile-row block. |
| R: input × mask | helper | `compute_kernel_lib::mul` (alias for `binary_op<MUL, NONE>`) | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:296-306` | `<BroadcastDim=NONE, input_a_policy=WaitAndPopPerTile, input_b_policy=NoWaitNoPop, output_policy=PerTile>` with `BinaryInputBlockShape::single()`. Caller did `cb_wait_front(cb_group_masks, mask_count)` once at startup. | A = `cb_input_tiles_R` (1 tile), B = `cb_group_masks[mask_idx(g, T)]` (1 tile, no pop) | `cb_scratch_a` | Helper handles DEST acquire/commit/wait/release. Manual indexing into `cb_group_masks` via the unused-tile policy `NoWaitNoPop` (we waited upfront once). |
| R: reduce sum (with Accumulate) | helper | `compute_kernel_lib::reduce` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:400-415` | `<reduce_type=SUM, reduce_dim=REDUCE_SCALAR, input_policy=WaitAndPopPerTile>` `(cb_scratch_a, cb_scaler_one, cb_group_sum, ReduceInputBlockShape::single(), {} , Accumulate(AccumulationConfig{cb_group_sum, /*dst*/ 0}, iteration), NoOp{})`. `iteration` runs from 0 (first `(T, r)` in group g) to `(tile_span_g × Ht) − 1` (last). | A = `cb_scratch_a` (1 tile), scaler = `cb_scaler_one` | `cb_group_sum[g]` | Last iteration packs into `cb_group_sum[g]`; intermediate iterations reload from `cb_group_sum` via the `Accumulate` template. `cb_scaler_one` is held with `WaitUpfrontNoPop` (default policy of reduce is wait per tile — but we set this up with one tile that is never popped; reduce will call wait on it which is harmless since the tile is already present). The implementer should configure the reduce helper to NOT pop the scaler (using the reduce input policy variant that retains the scaler) — alternatively, the implementer can issue an explicit `cb_wait_front(cb_scaler_one, 1)` once and use a reduce input policy whose scaler handling matches. **Implementer note**: `reduce<>` calls `cb_wait_front(cb_scaler_one, 1)` internally per `reduce_helpers_compute.hpp` semantics; never popping the scaler from the kernel side keeps the CB depth at 1 forever. |
| R: square(masked) | helper | `compute_kernel_lib::square` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:308-316` | `<input_policy=WaitUpfrontNoPop>` so the masked tile remains in `cb_scratch_a` for the sum-reduce-then-square sequence. Squares the tile in `cb_scratch_a`. | `cb_scratch_a` (the masked tile, retained) | `cb_scratch_a` (second slot — double-buffered) | DEST managed internally. **Alternative ordering**: since `cb_scratch_a` holds the masked tile, we can: (a) compute `cb_group_sum` reduce first while retaining the masked tile, then (b) square in place via `square` into a second scratch slot, then (c) reduce-accumulate into `cb_group_sumsq`. **Implementer choice**: use two scratch CBs (`cb_scratch_a`, `cb_scratch_b`), masking writes to one, squaring writes the other. Reduce-accumulate consumes each. |
| R/post: divide sum by N → mean | helper | `compute_kernel_lib::mul` (`<BroadcastDim=SCALAR>`) | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:296-306` | `<BroadcastDim=SCALAR, input_a_policy=WaitAndPopPerTile, input_b_policy=NoWaitNoPop>` with `BinaryInputBlockShape::single()`. | A = `cb_group_sum[g]` (1 tile, popped); B = `cb_inv_N_scalar` (1 tile, retained) | `cb_group_mean[g]` | mean_g lives at `(0,0)`; SCALAR broadcast of `cb_inv_N_scalar(0,0) = 1/N_per_g` multiplies into A's `(0,0)`. |
| R/post: divide sumsq by N | helper | `compute_kernel_lib::mul` (`<BroadcastDim=SCALAR>`) | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:296-306` | Same template params as above. | A = `cb_group_sumsq[g]`; B = `cb_inv_N_scalar` | `cb_scratch_a` (one slot — `E[X²]`) | Scalar tile at `(0,0) = sumsq/N`. |
| R/post: mean² | helper | `compute_kernel_lib::square` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:308-316` | `<input_policy=WaitUpfrontNoPop>` so `cb_group_mean[g]` is retained for the next phase. | `cb_group_mean[g]` | `cb_scratch_b` (slot) | scalar tile at `(0,0) = mean²`. |
| R/post: var = E[X²] − mean² | helper | `compute_kernel_lib::sub` (`<BroadcastDim=NONE>`) | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:282-293` | `<BroadcastDim=NONE>`. Both operands are scalar tiles with only `(0,0)` valid. | A = `cb_scratch_a` (E[X²]); B = `cb_scratch_b` (mean²) | `cb_scratch_a` (overwrite) | sub at `(0,0)`. |
| R/post: var + eps → rcp_std (composed) | helper | `compute_kernel_lib::sfpu_chain` + `compute_kernel_lib::sfpu_pipeline` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:1409-1415` | Chain: `Load<cb_scratch_a, Dst::D0>{} → SfpuAddScalar<>{eps} → Rsqrt<>{}`. Then `sfpu_pipeline(chain, cb_group_rcp_std[g], 1)`. Note: if the kernel_lib lacks `SfpuAddScalar`, the implementer composes via two-step: `binary_op<ADD, SCALAR>(cb_scratch_a, cb_eps_scalar, cb_scratch_b)` then `sfpu_rsqrt<cb_scratch_b>(cb_group_rcp_std[g], 1)`. The chain approach is preferred if supported. | `cb_scratch_a` (var) | `cb_group_rcp_std[g]` | The `eps` constant is bit-packed into CT args; a small constant tile `cb_eps_scalar` is built at reader startup the same way `cb_inv_N_scalar` is. |
| A: build means_tile_T (per group g touching T, accumulate) | helper | `compute_kernel_lib::mul` (`<BroadcastDim=SCALAR>`) with `BinaryAccumulate` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:296-306` | `<BroadcastDim=SCALAR, input_a_policy=NoWaitNoPop, input_b_policy=WaitUpfrontNoPop, accum=BinaryAccumulate{cb_means_tile_T, /*dst*/ 0}>` with `BinaryInputBlockShape::single()`. Per `g` touching T, iteration 0..(num_g_in_T − 1). | A = `cb_group_masks[mask_idx(g, T)]` (no pop, no wait — held since startup); B = `cb_group_mean[g]` (no pop — retained across A) | `cb_means_tile_T` | Accumulate template adds successive products into the destination. First iteration initializes (skip-reload), later iterations reload + add. Result is row-replicated `means_tile_T`. The implementer reads `BinaryAccumulate` semantics in `binary_op_helpers.hpp:190-192`. |
| A: build rcp_std_tile_T | helper | `compute_kernel_lib::mul` (`<BroadcastDim=SCALAR>`) with `BinaryAccumulate` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:296-306` | Same template params as means construction. | A = `cb_group_masks[mask_idx(g, T)]`; B = `cb_group_rcp_std[g]` | `cb_rcp_std_tile_T` | row-replicated `rcp_std_tile_T`. |
| A: tilize gamma (if `HAS_GAMMA`) | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:178-187` | `<block_width_tiles=1, cb_input=cb_gamma_rm, cb_output=cb_gamma_tile>`. Reader writes the gamma stick **32 times** (replicating row 0 across 32 rows) into `cb_gamma_rm` so the tilize produces a row-replicated tile. | `cb_gamma_rm` (32 sticks) | `cb_gamma_tile` (1 tile) | Once per output T. |
| A: tilize beta (if `HAS_BETA`) | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:178-187` | Same as gamma. | `cb_beta_rm` (32 sticks) | `cb_beta_tile` (1 tile) | Once per output T. |
| A: out = (input − mean) | helper | `compute_kernel_lib::sub` (`<BroadcastDim=NONE>`) | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:282-293` | `<BroadcastDim=NONE, input_a_policy=WaitAndPopPerTile, input_b_policy=NoWaitNoPop>` with `BinaryInputBlockShape::single()`. | A = `cb_input_tiles_A` (popped per HW row); B = `cb_means_tile_T` (held with WaitUpfrontNoPop) | `cb_scratch_a` | streaming. |
| A: × rcp_std | helper | `compute_kernel_lib::mul` (`<BroadcastDim=NONE>`) | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:296-306` | `<BroadcastDim=NONE, input_a_policy=WaitAndPopPerTile, input_b_policy=NoWaitNoPop>`. | A = `cb_scratch_a`; B = `cb_rcp_std_tile_T` (no pop) | `cb_scratch_b` | streaming. |
| A: × gamma (if `HAS_GAMMA`) | helper | `compute_kernel_lib::mul` (`<BroadcastDim=NONE>`) | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:296-306` | Same template params as above. | A = `cb_scratch_b`; B = `cb_gamma_tile` (no pop) | `cb_scratch_a` (overwrite) | gamma is row-replicated full tile (NONE broadcast). Skip when `HAS_GAMMA=0`. |
| A: + beta (if `HAS_BETA`) → final | helper | `compute_kernel_lib::add` (`<BroadcastDim=NONE>`) | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:269-280` | Same template params; output is `cb_output_tiles`. | A = current scratch; B = `cb_beta_tile` (no pop) | `cb_output_tiles` | streaming. Skip → identity copy when `HAS_BETA=0` (the prior stage already writes to `cb_output_tiles`). |
| A: copy → output (when `HAS_BETA=0`) | helper | `compute_kernel_lib::copy_tile_helpers` | `ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp` (see header for exact symbol) | Used to forward the scratch tile to `cb_output_tiles`. Alternatively, route the last multiply directly to `cb_output_tiles`. | scratch | `cb_output_tiles` | **Implementer's choice**: rather than a copy, change the destination CB of the last stage (mul × gamma → `cb_output_tiles` when `HAS_BETA=0`). |
| A: write output | raw_api | `noc_async_write` + `noc_async_writes_flushed` via `TensorAccessor` | `tech_reports/tensor_accessor/tensor_accessor.md` | Per (n, r, T) page id write. | `cb_output_tiles` | — | **Helpers considered and rejected**: No "interleaved tile-write" helper. Same rationale as the tile-layout reader entry. |
| compute init | raw_api | `compute_kernel_hw_startup(...)` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:14-19` (helper precondition) | Pass CBs that participate in initial register configuration (e.g., `cb_input_tiles_R, cb_group_masks, cb_scratch_a`). Called exactly once at the start of the compute kernel. | — | — | Required by all helpers. |

## Compute Phases

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `compute_kernel_hw_startup(...)` (once at kernel boot) | raw | — | — | hardware initialized |
| 1 | (RM input only, phase R) tilize HW tile-row block in c-tile column T | helper `tilize` | `cb_input_rm_R` (32 sticks) | `cb_input_tiles_R` (1 tile) | input rm consumed, tile pushed |
| 2 | Mask input: `mul<NONE>(cb_input_tiles_R, cb_group_masks[mask_idx])` | helper `mul` | `cb_input_tiles_R` (1, popped); `cb_group_masks` (no pop, retained) | `cb_scratch_a` (1) | masked tile available |
| 3 | Sum-accumulate into `cb_group_sum[g]`: `reduce<SUM, REDUCE_SCALAR>(cb_scratch_a, cb_scaler_one, cb_group_sum[g], single(), {}, Accumulate{cb_group_sum[g], iter})` — retains the masked tile (NoPop variant) so it can be squared | helper `reduce` | `cb_scratch_a` (1, retained for step 4); `cb_scaler_one` (1, no pop) | `cb_group_sum[g]` (push on last iter only; reload on others) | masked tile still in `cb_scratch_a` |
| 4 | Square: `square(cb_scratch_a → cb_scratch_b)` (input policy: WaitAndPop — now consume the masked tile) | helper `square` | `cb_scratch_a` (1, popped) | `cb_scratch_b` (1) | masked-squared tile available |
| 5 | Sumsq-accumulate: `reduce<SUM, REDUCE_SCALAR>(cb_scratch_b, cb_scaler_one, cb_group_sumsq[g], single(), {}, Accumulate{cb_group_sumsq[g], iter})` | helper `reduce` | `cb_scratch_b` (1, popped); `cb_scaler_one` (no pop) | `cb_group_sumsq[g]` | one (T, r) iteration done |
| 6 | Repeat steps 1-5 for every `(g, T_in_g_span, r)` triplet for batch n | — | — | — | `cb_group_sum[0..G-1]`, `cb_group_sumsq[0..G-1]` ready |
| 7 | R/post: divide sum by N: `mul<SCALAR>(cb_group_sum[g], cb_inv_N_scalar) → cb_group_mean[g]` | helper `mul` | `cb_group_sum[g]` (pop); `cb_inv_N_scalar` (no pop) | `cb_group_mean[g]` | mean_g ready |
| 8 | R/post: divide sumsq by N: `mul<SCALAR>(cb_group_sumsq[g], cb_inv_N_scalar) → cb_scratch_a` | helper `mul` | `cb_group_sumsq[g]` (pop); `cb_inv_N_scalar` (no pop) | `cb_scratch_a` | E[X²] ready |
| 9 | R/post: mean²: `square(cb_group_mean[g]) → cb_scratch_b` (retain cb_group_mean[g]) | helper `square` `<WaitUpfrontNoPop>` | `cb_group_mean[g]` (no pop) | `cb_scratch_b` | mean² ready, mean preserved |
| 10 | R/post: var = E[X²] − mean²: `sub<NONE>(cb_scratch_a, cb_scratch_b) → cb_scratch_a` (overwrite) | helper `sub` | `cb_scratch_a`, `cb_scratch_b` (both popped) | `cb_scratch_a` | var ready |
| 11 | R/post: rcp_std = rsqrt(var + eps): chained SFPU `sfpu_chain(Load<cb_scratch_a> + AddScalar(eps) + Rsqrt) → sfpu_pipeline(chain, cb_group_rcp_std[g], 1)` | helper `sfpu_pipeline` | `cb_scratch_a` (pop) | `cb_group_rcp_std[g]` | rcp_std_g ready |
| 12 | Repeat steps 7-11 for every group g. After: `cb_group_mean[0..G-1]`, `cb_group_rcp_std[0..G-1]` populated. | — | — | — | stats ready for apply |
| 13 | A: for each output T: build means_tile_T accumulator with `mul<SCALAR>(cb_group_masks[mask_idx(g, T)], cb_group_mean[g], cb_means_tile_T, single(), BinaryAccumulate{cb_means_tile_T, 0})` for every g touching T | helper `mul` + `BinaryAccumulate` | `cb_group_masks[...]` (no pop); `cb_group_mean[g]` (no pop — held across all A) | `cb_means_tile_T` (1, accumulating) | means_tile_T ready (row-replicated) |
| 14 | A: similar for rcp_std_tile_T: `mul<SCALAR>(cb_group_masks[...], cb_group_rcp_std[g], cb_rcp_std_tile_T, ..., BinaryAccumulate)` | helper `mul` + `BinaryAccumulate` | masks (no pop); rcp_std (no pop) | `cb_rcp_std_tile_T` | rcp_std_tile_T ready |
| 15 | (HAS_GAMMA) tilize gamma for T: `tilize<1, cb_gamma_rm, cb_gamma_tile>(1, 32 /* total sticks */)` | helper `tilize` | `cb_gamma_rm` (32 sticks) | `cb_gamma_tile` (1) | gamma_tile for T ready (row-replicated) |
| 16 | (HAS_BETA) tilize beta for T | helper `tilize` | `cb_beta_rm` (32 sticks) | `cb_beta_tile` (1) | beta_tile for T ready |
| 17 | A: inner HW loop, per r: `sub<NONE>(cb_input_tiles_A, cb_means_tile_T) → cb_scratch_a` | helper `sub` | input (pop); means_tile_T (no pop) | `cb_scratch_a` | (input − mean) |
| 18 | `mul<NONE>(cb_scratch_a, cb_rcp_std_tile_T) → cb_scratch_b` | helper `mul` | scratch_a (pop); rcp_std_tile_T (no pop) | `cb_scratch_b` | × rcp_std |
| 19 | (HAS_GAMMA) `mul<NONE>(cb_scratch_b, cb_gamma_tile) → cb_scratch_a` | helper `mul` | scratch_b (pop); gamma_tile (no pop) | `cb_scratch_a` | × gamma |
| 20 | (HAS_BETA) `add<NONE>(<current>, cb_beta_tile) → cb_output_tiles` (output policy PerTile pushes to writer); else final stage routes directly into `cb_output_tiles` | helper `add` | current scratch (pop); beta_tile (no pop) | `cb_output_tiles` | output tile ready |
| 21 | After inner HW loop: pop `cb_means_tile_T`, `cb_rcp_std_tile_T`, `cb_gamma_tile`, `cb_beta_tile` (each: 1 tile) | raw `cb_pop_front` | — | — | T iteration cleaned up |
| 22 | Repeat steps 13-21 for every output T = 0..Ct-1 for batch n. After all T's: pop `cb_group_mean[0..G-1]`, `cb_group_rcp_std[0..G-1]`. | — | — | — | batch n done |
| 23 | Repeat phases 1-22 for every batch n = 0..N-1. | — | — | — | full op done |

After kernel completion: `cb_group_masks`, `cb_scaler_one`, `cb_inv_N_scalar`, `cb_eps_scalar` still hold their one-shot data (never popped during the kernel — they go out of scope when the program ends).

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| R: input × mask | `mul` | All (32×32 full tile) | All (32×32 row-replicated full tile) | NONE |
| R/post: sum × inv_N | `mul` | `(0,0)` (scalar tile) | `(0,0)` (scalar tile) | SCALAR |
| R/post: sumsq × inv_N | `mul` | `(0,0)` | `(0,0)` | SCALAR |
| R/post: mean × mean | `square` | `(0,0)` | (same) | NONE (square = self-mul) |
| R/post: E[X²] − mean² | `sub` | `(0,0)` | `(0,0)` | NONE |
| A: mask × mean (expansion) | `mul` | All (mask row-replicated full tile) | `(0,0)` (mean scalar tile) | SCALAR |
| A: mask × rcp_std (expansion) | `mul` | All (mask) | `(0,0)` (rcp_std scalar) | SCALAR |
| A: input − means_tile_T | `sub` | All (input full tile) | All (means_tile_T row-replicated full tile) | NONE |
| A: result × rcp_std_tile_T | `mul` | All | All (rcp_std_tile_T row-replicated) | NONE |
| A: result × gamma_tile | `mul` | All | All (gamma_tile row-replicated, built via tilize-from-32-replicated-sticks) | NONE |
| A: result + beta_tile | `add` | All | All (beta_tile row-replicated) | NONE |

All "row-replicated full tile" CBs (masks, expansion outputs, gamma_tile, beta_tile) hold the same 1×32 pattern in every row of the tile. NONE broadcast is correct because elementwise read of row `r` of B at col `c` equals the pattern value at `c` for any `r`. This is the design's central simplification — by row-replicating up front, every apply-phase op is NONE-broadcast.

## Reduce Direction Verification

| Logical Dim | Tile ReduceDim | Output Valid Region | BroadcastDim | ReduceInputBlockShape | BinaryInputBlockShape |
|-------------|----------------|--------------------|--------------|-----------------------|-----------------------|
| Per-group scalar (sum over HW × Cg masked lanes) | `REDUCE_SCALAR` | `(0,0)` of one tile per group | SCALAR (when consuming the scalar later) | `ReduceInputBlockShape::single()` per tile fed in; `Accumulate` template stitches them across iterations | `BinaryInputBlockShape::single()` for all per-group scalar binary ops |

## Key Risks and Gotchas

| Risk | Mitigation |
|------|-----------|
| **Mask CB sync — produced once, consumed many times.** A typo where the compute pops `cb_group_masks` mid-kernel turns the rest of the masks into reads-past-front and produces silent garbage. | Reader pushes `mask_count` tiles ONCE at startup. Compute calls `cb_wait_front(cb_group_masks, mask_count)` once at startup. Every helper invocation that uses a mask MUST use a policy that does NOT pop (`NoWaitNoPop` or `WaitUpfrontNoPop`). The implementer treats `cb_group_masks` as read-only after startup. |
| **Scaler CB depth = 1, never popped.** The reduce helper waits on the scaler internally; if the kernel ever pops the scaler the next reduce call hangs on `cb_wait_front`. | `cb_scaler_one` (and `cb_inv_N_scalar`, `cb_eps_scalar`) are reader-built one-shot; the compute kernel never pops them. The reduce helper's "wait" on the scaler is satisfied permanently after the first push. |
| **`(C/G) % 32 != 0` is the default path, not a special case.** The kernel must not have a fast path for tile-aligned C/G that masks-only-when-needed. | The reader builds masks unconditionally; the compute multiplies by mask unconditionally. When `Cg = 32k`, the masks are "all ones at the right lanes, zero elsewhere," which works identically to the partial case. (One small optimization: for interior tiles where the entire tile belongs to one group, the mask is all-ones and the multiply is wasted — but the skill explicitly says "do not write two implementations." Cost is one wasted multiply per interior tile, acceptable per Phase 0 goals.) |
| **C non-aligned (`C % 32 != 0`) padding poisoning.** The last C-tile has only `C % 32` valid lanes. If the mask for the last group's last tile extends beyond `C % 32`, the padded junk poisons the sum. | The reader computes each group's mask using `min(g_end, C)` — the last group's mask stops at `C`, so padding lanes are always 0 in the mask. Verified by construction: the (g, T) mask covers `[max(g*Cg, T*32) − T*32, min((g+1)*Cg, C) − T*32)` within the tile. |
| **Stats CB written G times by the same logical sequence — accumulate iteration index correctness.** `Accumulate(cfg, iter=0)` is "no reload, init"; `iter>0` is "reload + accumulate". A typo in the iteration index causes silent corruption (stats initialized to garbage on every iteration). | The iteration index for each group `g` is the running count of `(T, r)` pairs processed so far for that group, starting from 0. Implementer: track this with a per-group counter incremented inside the (T, r) loop. The host can pre-compute `iter_max_per_group[g]` and pass via RT args; the kernel uses iter = 0..iter_max-1 and pushes the result only when iter == iter_max - 1 (the reduce helper handles this automatically via the `Accumulate::is_first()` and final-iteration logic). |
| **Re-reading input for the apply phase is required and costs DRAM bandwidth.** Keeping input in L1 across phases would violate the L1-scales-with-G invariant (input scales with HW × C). | Accept the re-read. The reader has the input address; it walks the same tiles a second time. DRAM bandwidth is the right tradeoff vs L1 budget. |
| **Output dtype must equal input dtype.** Phase 0 is bf16 only, so trivially satisfied. The contract still needs to be enforced by the program factory and by the acceptance test. | Program factory: allocate output as `input_tensor.dtype` (not hardcoded bfloat16). Acceptance test: assert `out.dtype == in.dtype`. |
| **Mask construction (reader side) must respect tile face layout.** A bf16 tile in L1 is laid out as 4 × 16×16 faces. Writing 32 contiguous bf16 values does NOT produce row 0 of a tile — it spans face 0 then face 1, but only the top row of each. | Implementer code: for a row-replicated mask, write 16 bf16 values to face 0 row 0, face 0 row 1, …, face 0 row 15 (same 16 values 16 times), then write 16 bf16 values to face 1 rows 0..15 (the other 16 mask positions), then duplicate to faces 2 and 3 (bottom 16 rows of the tile). Tile face layout reference: `tech_reports/tensor_layouts/tensor_layouts.md`. |
| **`HAS_GAMMA`, `HAS_BETA` are compile-time.** Conditional helper calls must use `if constexpr` (or be elided at host CT-arg level). A runtime branch on `has_gamma` would force the compute kernel to consume non-existent γ tiles or stall. | Program factory sets CT args `HAS_GAMMA` and `HAS_BETA` based on whether `gamma` / `beta` are passed. Compute kernel uses `if constexpr` to gate the γ/β multiply/add. When `HAS_GAMMA=0`, neither `cb_gamma_rm` nor `cb_gamma_tile` is configured (saves L1). Same for β. |
| **`cb_group_sum[g]` is one slot in a G-slot CB — addressing by group index needs explicit `cb_push_back` / `cb_pop_front` semantics or per-group offset within the CB.** | The reduce helper writes one tile per call. The implementer writes ALL G group sums sequentially in the order g=0..G-1 (with the inner `(T, r)` Accumulate loop completing each one before moving to the next). The CB holds G tiles after phase R; R/post pops them in order g=0..G-1. No special addressing — order is enough. |

## Structural impossibilities (note for INVALID; do NOT edit feature_spec.py)

The op-specific structural-impossibility candidates implied by this design are already in `eval/golden_tests/groupnorm_sc_N_1_HW_C/feature_spec.py::INVALID`. None to add.
