# Operation Design: rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (fused reduction + broadcast eltwise) |
| Goal | Normalize a tensor along its last dimension by the root-mean-square of that dimension, then optionally scale by a per-channel weight `gamma`. Native support for ROW_MAJOR and TILE layout and for non-tile-aligned H and W, with no host-side layout/padding workarounds. |
| Math | `out[..., w] = x[..., w] * rsqrt( (1/W) * Σ_{w'=0}^{W-1} x[..., w']² + eps ) * gamma[w]` |
| Mode | Hybrid (Python `ProgramDescriptor` generic op; reader / compute / writer kernels) |
| Entry point | `from ttnn.operations.rms_norm import rms_norm` |
| Feature spec | `eval/golden_tests/rms_norm/feature_spec.py` (authoritative; read-only for this design) |
| References | `.claude/references/op-design-template.md`, `.claude/references/precision_convention.md`, `.claude/references/ttnn-cb-memory-fundamentals.md`, `.claude/references/generic_op_template/`, `ttnn/ttnn/operations/toy_variance/`, `ttnn/ttnn/operations/toy_reduce_partial/`, `ttnn/ttnn/operations/examples/master.md` |

---

## 1. Blocking Model

**This section is the design.** Everything below it is a realization of the decisions made here.

### 1.1 Axes of this op

The input is a rank-2/3/4 tensor. Its dims collapse into exactly two axes for this op's math, because RMSNorm reduces the last dim and is elementwise in everything else:

| Op axis | What it is | Tile extent |
|---------|-----------|-------------|
| `row` | all leading dims folded together (`N·C·H`, `B·S`, or `B`) — one *logical row* is one reduction group | `Rt` tile-rows (see §4 for the layout-specific formula) |
| `width` | the last dim `W` — the reduced/normalized dim | `Wt = ceil(W / 32)` tiles |
| `gammaW` | `gamma`'s only dim — aliases `width`, absent along `row` | `Wt` tiles |

### 1.2 The table

| Axis | Character | Block-factor knob | Phase 0 value | Core-assignment | Later unlock |
|------|-----------|-------------------|---------------|-----------------|--------------|
| `row` | **independent** — each output row's RMS denominator is computed from that row alone; no cross-row dependency of any kind | `BLOCK_ROWS` (tile-rows per compute block) | `min(MAX_ROWS_PER_CORE, BLOCK_ROWS_L1_MAX)` — the **coarsest** chunk of the per-core assignment that fits L1, i.e. the whole assignment whenever it fits (§1.4) | `ttnn.split_work_to_cores(grid, Rt, row_wise=True)` over the **full** `compute_with_storage_grid_size()` — Phase 0 is multi-core on day 1 | **knob-turn** (change `BLOCK_ROWS`; `HEIGHT_SHARDED` is this same split with extent + placement pinned by the caller) |
| `width` | **dependent** — the result at every `w` depends on `Σ x²` over *all* of `W`. Within a core this is a cheap sequential accumulate; across cores it needs a NoC combine | `WT_CHUNK` (width tiles per compute block) | `Wt` (the **whole** row — one chunk) in the RESIDENT regime; `WT_CHUNK_L1_MAX` in the STREAM regime (§1.4) | `GRID_W = 1` — one core owns the entire width of every row it owns | **scheme-change** (cross-core partial-sum combine + mcast → also unlocks `WIDTH_SHARDED` / `BLOCK_SHARDED`) — see Lamp L1 |
| `gammaW` | **reuse-shared** — `gamma` is indexed by `width` only, so it does **not** vary along `row`, the axis Phase 0 splits across cores. Every core in the row-split therefore re-reads the identical `Wt` gamma tiles | `GAMMA_RESIDENT` (gamma held for a core's whole assignment) | `True` in RESIDENT (read once per core, `Wt` tiles, never popped); `False` in STREAM (chunked, re-read per row-block) | replicated: every core reads gamma from DRAM itself | **scheme-change** (one injector reads gamma, multicasts to the grid row) — see Lamp L2 |

**Buffer-depth knobs** (per streaming CB; a *distinct* knob from block factor — it buys data-movement↔compute overlap, not reuse):

| CB | Depth knob | Phase 0 value | Why |
|----|-----------|---------------|-----|
| `cb_input_tiles` | `CB_X_DEPTH` | `2` when the input is TILE (reader is the producer → cross-processor overlap is real); `1` when the input is ROW_MAJOR (producer is the `tilize` compute helper → sequential helpers cannot overlap, depth > 1 buys nothing) | `double_buffer/report.md`: depth 1 saturates ≈13 GB/s and cannot overlap read+write; depth 2 at block 4–8 tiles reaches 17.9 GB/s (2.78×) |
| `cb_output_tiles` | `CB_OUT_DEPTH` | `2` when the output is TILE (writer is the consumer); `1` when ROW_MAJOR (consumer is the `untilize` compute helper) | same |
| `cb_input_sticks`, `cb_output_sticks` (ROW_MAJOR only) | `CB_RM_STAGE_DEPTH` | `2` (unit = one tile-row = `WT_CHUNK` pages) | reader/writer ↔ tilize/untilize overlap |
| `cb_scaler` | — | `1 + (1 if PARTIAL_W else 0)` | constant CB; single page (pair when the partial scaler is needed) |
| `cb_row_stat` | — | `BLOCK_ROWS` | accumulator + in-place finalize; `transform_in_place` pops before reserving, so `BLOCK_ROWS` pages suffice |
| `cb_x_squared`, `cb_normalized` | — | `BLOCK_ROWS · WT_CHUNK` (full block) | sequential compute helpers own all 3 TRISCs and cannot pipeline — the CB must hold everything the producing helper emits (`ttnn-cb-memory-fundamentals.md` → "Intermediate CB Sizing Between Compute Helpers") |

### 1.3 Bandwidth ranking of the candidate splits (qualitative, structural — no ns)

Ranked over *all* axes that could be split, including the dependent one.

| Rank | Candidate split | Bytes moved / fan-out | Combine needed? | Verdict |
|------|-----------------|------------------------|-----------------|---------|
| 1 | **`row` across cores** (chosen) | x read **once**, out written once, contiguous whole-tile/whole-stick transfers. Only redundancy: gamma, `num_cores · Wt` tiles | none — every core owns disjoint output rows | **Phase 0 primary split.** Cheapest total bytes whenever `Rt ≥ num_cores` |
| 2 | **`width` across cores** (Lamp L1) | x read once, out written once, **zero** gamma redundancy (each core reads only its own `Wt/GRID_W` slice). Adds: 1 partial tile per row per core gathered to a root, plus one mcast of `BLOCK_ROWS` finalized tiles back to the group | **yes** — cross-core partial-`Σx²` combine | Strictly fewer *DRAM* bytes than (1); pays for it with NoC combine traffic. **The available parallelism when `Rt < num_cores`** (decode-shaped: `Rt=1` gives split (1) exactly one core) and the only way to keep x resident when `Wt·bytes > L1` |
| 3 | **`width` chunked *within* a core** (Phase 0 STREAM regime) | x read **twice** (pass A accumulates `Σx²`, pass B re-reads x to scale it) and gamma re-read per row-block → ≈2× the DRAM bytes of (1) | none | Not a parallelization at all — an **L1 fallback**. Used only when one tile-row's width cannot be resident. Its existence is what makes the op correct at any `W`; Lamp L1 is what makes it *fast* |
| 4 | single-core / no split | — | — | **rejected**; forbidden by the blocking model |

Both splits are **logical shards** of an interleaved tensor. Cutting `row` (the leading/`H` dims) is a **height**-flavoured cut; cutting `width` (the last dim) is a **width**-flavoured cut; cutting both is **block**-flavoured. The *flavour* is geometry; the *cost* is the character of the axis being cut — and for this op the row-wise reading happens to be the familiar one: `row` is independent (height cut → no combine, knob-turn) and `width` is dependent (width/block cut → combine, scheme-change). §5.3 states this per `memory_layout` value.

### 1.4 Knob derivation — one source of truth each, everything else derived (DRY)

All of the following live **once**, in `rms_norm_program_descriptor.py`. No block/chunk/tile count is restated as a second literal anywhere; kernels receive them as CT/RT args and never re-derive them.

```
# ---- primary knobs (the ONLY hand-set numbers) -------------------------------
TILE_DIM              = 32
# L1 available for this op's CBs, per core — DERIVED from the device, never a literal.
# ttnn.get_max_worker_l1_unreserved_size() is the usable-L1-per-core query
# (ttnn/cpp/ttnn-nanobind/device.cpp:645-647, exported ttnn/ttnn/device.py:20).
# L1_SAFETY_FRACTION is the ONE hand-set number here; lower it if S3 CB-OOM appears.
L1_SAFETY_FRACTION    = 0.85
_L1_CB_BUDGET_BYTES   = int(ttnn.get_max_worker_l1_unreserved_size() * L1_SAFETY_FRACTION)
CB_X_DEPTH            = 2 if input_layout is TILE else 1
CB_OUT_DEPTH          = 2 if input_layout is TILE else 1
CB_RM_STAGE_DEPTH     = 2
REDUCE_INPUT_POLICY   = ReduceInputPolicy::BulkWaitBulkPop
REDUCE_FP32_MODE      = (ReduceFp32Mode::Accurate if input_dtype is float32
                         else ReduceFp32Mode::Fast)     # CT knob, see R14
GRID_W                = 1                              # cores along `width` (Lamp L1)

# ---- geometry (alignment-aware: ceil everywhere, per-image) ------------------
W        = shape[-1]
Wt       = div_up(W, TILE_DIM)
PARTIAL_W = W % TILE_DIM                               # 0 => tile-aligned width
# TILE layout: every (…, H, W) image is tile-padded independently
Rt       = prod(shape[:-2]) * div_up(shape[-2], TILE_DIM)      # if input_layout is TILE
# ROW_MAJOR layout: no implicit H padding; pages are sticks
R_rm     = prod(shape[:-1])
Rt       = div_up(R_rm, TILE_DIM)                              # if input_layout is ROW_MAJOR
ROWS_IN_LAST_TILEROW = R_rm - TILE_DIM * (Rt - 1)              # ROW_MAJOR only

# ---- cross-core split of the independent axis: SIZE and COUNT together ------
grid = device.compute_with_storage_grid_size()
(num_cores, all_cores, core_group_1, core_group_2,
 rows_per_core_g1, rows_per_core_g2) = ttnn.split_work_to_cores(grid, Rt, row_wise=True)
MAX_ROWS_PER_CORE = max(rows_per_core_g1, rows_per_core_g2)

# ---- per-core block size = coarsest chunk of the assignment that fits L1 ----
bt   = ttnn.tile_size(input_dtype)                     # x / x² / normalized / out
gt   = ttnn.tile_size(gamma_dtype) if HAS_GAMMA else 0
st   = ttnn.tile_size(ttnn.bfloat16)                   # scaler CB (see §6, Risk R3)
ft   = ttnn.tile_size(ttnn.float32)                    # cb_row_stat

# tiles-per-block multiplier over the block-scoped CBs
CB_BLOCK_MULT = CB_X_DEPTH + 1 + (1 if HAS_GAMMA else 0) + CB_OUT_DEPTH
#               cb_input_tiles  cb_x_squared  cb_normalized       cb_output_tiles

FIXED_RES = (Wt * gt) \
          + (Wt * gt if GAMMA_IS_RM else 0) \
          + (2 * CB_RM_STAGE_DEPTH * Wt * bt if input_layout is ROW_MAJOR else 0) \
          + st * (2 if PARTIAL_W else 1)

BLOCK_ROWS_L1_MAX = max(0, (_L1_CB_BUDGET_BYTES - FIXED_RES)
                            // (Wt * bt * CB_BLOCK_MULT + ft))

# ---- REGIME SELECTION FUNCTION (pinned; see §5.2) --------------------------
if BLOCK_ROWS_L1_MAX >= 1:                             # regime RESIDENT
    BLOCK_ROWS     = min(MAX_ROWS_PER_CORE, BLOCK_ROWS_L1_MAX)
    WT_CHUNK       = Wt
    NUM_W_CHUNKS   = 1
    X_RESIDENT     = True                              # gamma resident too
else:                                                  # regime STREAM
    BLOCK_ROWS     = 1
    X_RESIDENT     = False
    FIXED_STREAM   = (2 * CB_RM_STAGE_DEPTH * Wt * bt if input_layout is ROW_MAJOR else 0) \
                   + st * (2 if PARTIAL_W else 1) + ft
    WT_CHUNK       = min(Wt, max(1, (_L1_CB_BUDGET_BYTES - FIXED_STREAM)
                                     // (bt * CB_BLOCK_MULT + gt * (2 if GAMMA_IS_RM else 1))))
    NUM_W_CHUNKS   = div_up(Wt, WT_CHUNK)

BLOCKS_PER_CORE(c) = div_up(rows_of(c), BLOCK_ROWS)     # ragged tail handled at runtime
```

`X_RESIDENT ≡ (NUM_W_CHUNKS == 1) ≡ GAMMA_RESIDENT`. The two regimes share **one loop nest** (§7); the only differences are (a) whether pass B re-reads x, and (b) `NUM_W_CHUNKS`. Nothing else branches.

### 1.5 The scheme Phase 0 commits to

> **Row-parallel, multi-core, coarse-blocked, dual-path on an explicit fits-in-L1 predicate.**
> The independent `row` axis is spread across the entire compute grid (`split_work_to_cores(grid, Rt, row_wise=True)`); each core processes its assignment in the coarsest whole-row blocks that fit L1 (`BLOCK_ROWS`, defaulting to the *entire* per-core assignment); the dependent `width` axis stays inside a core, taken in **one chunk** (`WT_CHUNK = Wt`) whenever the working set fits, and chunked only as an L1 fallback. Buffer depths are the minimal overlap-enabling value (2 on cross-processor CBs, 1 where the producer/consumer are both compute helpers).

Every knob above — including `num_cores` (via `grid`) and `GRID_W` — is a **parameter**, not an inlined constant. `GRID_W` exists at its trivial value `1` in Phase 0 precisely so Lamp L1 is a knob change plus a combine, not a rewrite.

### 1.6 Lamps — the scheme-changes Phase 0 leaves room for

| # | Unlock | Class | Why Phase 0's structure keeps it reachable |
|---|--------|-------|--------------------------------------------|
| **L1** | **Cross-core `width` split + partial-sum combine** (`GRID_W > 1`). The available parallelism when `Rt < num_cores` (every decode-shaped case, `Rt=1`), and what makes `Wt` resident again for wide `W` | scheme-change | Per-core work is already described by `(row_start, row_count, w_start, w_count)` runtime args with `w_start=0, w_count=Wt`; the reduce already writes through an accumulator CB (`cb_row_stat`); the eps+rsqrt finalize is already a **separate** step (`transform_in_place`) that in Scheme W runs on the group root only. Adding the combine adds a dedicated handoff CB + semaphores; the loop nest is unchanged. Full Tensix↔Tensix contract in §3.3 |
| **L2** | **gamma broadcast** — one injector per grid row reads gamma from DRAM and multicasts it | scheme-change (reuse) | `cb_gamma_tiles` is already a dedicated CB filled **once per core** before the row-block loop and never popped in RESIDENT. Only its *filler* changes (DRAM read → `ReceiverPipe::receive()`). Contract in §3.4. Evidence caveat: `shared_input_reuse/report.md` needed 2.4 MB × 22 cores of redundancy for 1.71×; gamma redundancy here is `num_cores · Wt` tiles, so this pays only in the wide-`W` / many-core corner — measure before shipping |
| **L3** | **`HEIGHT_SHARDED` input/output** | knob-turn | Identical logical scheme with core-assignment and per-core extent pinned by the caller and the data **already in L1**. `cb_input_tiles` becomes `ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor)` (zero-copy, **no NoC read** — re-reading a local shard through a `TensorAccessor` is not sharding); `cb_output_tiles` likewise. `BLOCK_ROWS` defaults to the shard's full tile-row count (sub-chunk only under L1 pressure) |
| **L4** | **`WIDTH_SHARDED` / `BLOCK_SHARDED` input/output** | scheme-change | Same combine as L1, with `GRID_W`/`GRID_H` and per-core extents read off the shard spec instead of chosen. L1 must land first |
| **L5** | **Row-resident, W-chunked third regime** — hold x + gamma resident for one tile-row while chunking only the derived CBs, removing STREAM's pass-B re-read for `Wt` between the RESIDENT and STREAM bounds | knob-turn | `X_RESIDENT` is already an explicit CT flag decoupled from `NUM_W_CHUNKS`; needs `TileOffset` column offsets on the resident operands (`eltwise_chain.hpp:311`). Lower priority than L1, which fixes the same shapes *and* fills the grid |
| **L6** | Perf micro-knobs, all already parameters: **(a)** flip the A2 `reduce<>` call's `ReduceAlgorithm` template arg from `ReduceTile` to `AccumulateViaAdd` (+ `prepare_reduce_mask` instead of the partial scaler for partial W) — measured **2.87–2.94×** on `REDUCE_ROW` at `Wt ≥ 4` (`row_reduce_accumulate/report.md`, `reduce_accumulate/README.md`). Phase 0 pins `ReduceTile` explicitly (see R14): `AccumulateViaAdd` restricts `Accumulate` to SUM + `BulkWaitBulkPop` only (`reduce_helpers_compute.inl:778-832`) and swaps the partial-W mechanism from a scaler tile to a 0/1 mask tile, so it is a coupled change, not a one-word swap. **(b)** scope the `rsqrt` to `VectorMode::C` — the stat lives in column 0, clean **1.94×** (`sfpu_tile_scope/report.md`). **(c)** drop dtype reconfig when every CB shares one format — up to **1.19×** (`compute_block_size/report_reconfig_ablation.md`, ≈110–150 ns per reconfig). **(d)** fuse square+accumulate via `DestAccumulation::PerRow` to delete `cb_x_squared`'s L1 round-trip (needs the mask moved ahead of the accumulation) | knob-turn |

### 1.7 Catalog evidence → knob mapping

| Knob | Value chosen | Measured evidence |
|------|--------------|-------------------|
| `BLOCK_ROWS` default = coarsest that fits | one compute pass per phase per block | `compute_block_size/report.md`: 1 pass vs per-tile-row = **1.65×**; ≈1.6 µs fixed cost per extra pass; win grows with phase count (this op has 3–5 phases) |
| `CB_X_DEPTH`/`CB_OUT_DEPTH` = 2 | double-buffered | `double_buffer/report.md`: b1/single 6.5 GB/s → b4/double **17.9 GB/s (2.78×)**; depth 1 caps at ≈13 GB/s |
| reader transaction granularity | whole tiles / whole sticks, ≥4 tiles per barrier where the block allows | `double_buffer/report.md` (4–8 tiles per barrier is the plateau; `block=256` OOMs L1), `tile_reorder/README.md` (move whole pages, never sub-tile faces) |
| `row_wise=True` in `split_work_to_cores` | row line, not column line | `noc_placement/README.md`: **2.91×** row vs column; reads on NoC0 / writes on NoC1 **2.5–4.8×** (keep the `ReaderConfigDescriptor` / `WriterConfigDescriptor` defaults) |
| primary split = `row` while `Rt ≥ num_cores` | — | `width_split/report.md`: width-split gains track *cores filled* (2.24× @ Wt=8 → 7.76× @ Wt=256) and are **1.00× at Wt=1** — i.e. splitting only helps by filling the grid, which the row split already does when `Rt` is large |
| Lamp L1 combine budget | root-reduce + mcast on a 1-D group | `tensix_all_reduce/README.md`: at **1 tile/core** (exactly a per-row partial sum) grid-two-stage/flat-root = **1.27–1.55 µs**; two-phase is the *worst* there. On a 1-D group grid-two-stage collapses to the flat root reduce |
| fp32 accumulation of `Σx²` | `fp32_dest_acc_en=True` (Phase 0 mandate) | `row_reduce_accumulate/report.md`: bf16 accumulation error **grows** with reduce width and is worst on all-positive data — a sum of squares is exactly that case (`reduce_fold` 5.83 ULP @ 32t) |
| gamma multiply is a separate L1-mediated pass, never DEST-reuse, never SFPU | two `mul` calls with `cb_normalized` between | `compute_fusion/README.md`: FPU consumer + dest-reuse = **0.82–1.02×** (no win); SFPU multiply = **0.58×** |
| eps+rsqrt as a small separate SFPU step on the reduce output | `transform_in_place` | `compute_fusion/report.md`: fusing an SFPU post-op onto a reduce is only **1.01–1.07×**, and shrinks as reduce width grows — not worth coupling it to the reduce call |
| Lamp L2 skepticism | gamma stays per-core in Phase 0 | `shared_input_reuse/report.md`: 1.71× required 2.4 MB × 22 cores; win ≪ the DRAM-read-count reduction because the injector reads serially |
| global caveat | do not over-tune the compute | `reduce_accumulate/README.md`: "most real reductions are data-movement-bound, where it won't show" — this op is DRAM-bound in most regimes; the DRAM-traffic decisions (x read once; gamma read once per core) dominate |

---

## 2. Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2; `bfloat16` \| `float32` (Phase 0); `TILE_LAYOUT` \| `ROW_MAJOR_LAYOUT`; `INTERLEAVED` (Phase 0) | — | — |
| `gamma` | `Optional[ttnn.Tensor]` | no (keyword-only) | shape `(1,1,1,W)` with `gamma.shape[-1] == input.shape[-1]`; any of `float32` / `bfloat16` / `bfloat8_b`; `TILE_LAYOUT` \| `ROW_MAJOR_LAYOUT` | `None` | — |
| `epsilon` | `float` | no (keyword-only) | `> 0` | `1e-6` | CT (raw fp32 bits) |
| `compute_kernel_config` | `Optional[ttnn.ComputeConfigDescriptor]` | no (keyword-only) | `fp32_dest_acc_en=True` in Phase 0; `math_fidelity` / `math_approx_mode` **ungated** (any value accepted) | resolved through `default_compute_kernel_config()` | passed as `config=compute_kernel_config` to the compute `KernelDescriptor` |
| `memory_config` | `Optional[ttnn.MemoryConfig]` | no (keyword-only) | Phase 0: `INTERLEAVED` (DRAM/L1). Accepted-and-honoured so the golden harness's sharded cells reach `validate()` and are refused there, not at the signature | input's memory config flavour, DRAM interleaved | — |
| `program_config` | `Optional[Any]` | no (keyword-only) | reserved / ignored when `None` | `None` | — |

Derived (host, all single-source per §1.4): `Wt`, `PARTIAL_W`, `Rt`, `R_rm`, `ROWS_IN_LAST_TILEROW`, `BLOCK_ROWS`, `WT_CHUNK`, `NUM_W_CHUNKS`, `X_RESIDENT`, `HAS_GAMMA`, `GAMMA_IS_RM`, `INV_W = 1.0/float(W)` (raw fp32 bits), `EPS` (raw fp32 bits).

### 2.1 Compute config contract

`default_compute_kernel_config()` is a **factory** exported from `ttnn/ttnn/operations/rms_norm/rms_norm.py` and is the *single source of truth* for what `None` means (`precision_convention.md:61-85`; the golden tagger already imports it — `eval/golden_tests/rms_norm/axes.py:22,40-43`):

```python
def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
```

`validate()` reads `cfg.fp32_dest_acc_en` and gates on it; after validation the config is passed through **unmodified** (`config=compute_kernel_config`). `math_fidelity` / `math_approx_mode` are never gated.

### 2.2 Registry surface the op module must export

`rms_norm`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `default_compute_kernel_config`.

`INPUT_TAGGERS` (names and value sets fixed by `feature_spec.py:14-22`):

| Tagger | Returns | Definition |
|--------|---------|-----------|
| `tag_alignment(inputs, axes)` | `"tile_aligned"` \| `"w_non_aligned"` \| `"h_non_aligned"` | `"w_non_aligned"` if `shape[-1] % 32 != 0` (regardless of H); else `"h_non_aligned"` if `shape[-2] % 32 != 0`; else `"tile_aligned"`. The three values map to genuinely different kernel paths — W-mask, H-row-padding, neither (§6 R1/R2) |
| `tag_rank(inputs, axes)` | `int` | `len(inputs[0])` |

`EXCLUSIONS` must contain `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}` (`precision_convention.md:31-39`). `SUPPORTED["gamma_dtype"]` and `SUPPORTED["gamma_layout"]` must both include the string `"none"` — it means "no weight tensor" and is always legal (`axes.py:50-52`). `validate()` never refuses `"none"` for those axes. The op file declares no `INVALID`. There is no index axis (`dim`/`axis`) to canonicalize.

---

## 3. Tensors and Dataflow Strategy

### 3.1 Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2, any `(…, H, W)`; H and W need **not** be multiples of 32 |
| Dtype | Phase 0: `bfloat16`, `float32`. TARGET also lists `bfloat8_b` (refinement) |
| Layout | `TILE_LAYOUT` or `ROW_MAJOR_LAYOUT` — both **native**. The Python entry point performs **no** `to_layout` / `tilize` / `untilize` / `pad` / `slice` |
| Memory | Phase 0: `INTERLEAVED`. `*_SHARDED` per Lamps L3/L4 |

`gamma`: shape `(1,1,1,W)`, dtype and layout independent of the input's (mixed precision is expected — bf16 activations + fp32 weights). In `TILE_LAYOUT` a `(1,1,1,W)` tensor occupies `Wt` tiles with only **row 0** valid, which is exactly the layout `BroadcastDim::Row` consumes.

### 3.2 Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | input dtype |
| Layout | **input layout** (TILE in → TILE out; ROW_MAJOR in → ROW_MAJOR out) |
| Memory | `memory_config` if given, else the input's memory config; Phase 0 DRAM interleaved |

### 3.3 Phase 0 data path

```
                        ┌──────────────── one Tensix core, one row-block ────────────────┐
 DRAM (interleaved)     │                                                                │
   x tiles / sticks ──► │ READER (NCRISC, NoC0)                                          │
   gamma tiles/stick ──►│   • TensorAccessor reads                                       │
                        │   • fills cb_scaler once at boot                               │
                        │   • fills cb_gamma_* once per core (RESIDENT)                   │
                        │                     │                                          │
                        │                     ▼ CBs                                      │
                        │   COMPUTE (UNPACK / MATH / PACK)                                │
                        │     [RM only] tilize  cb_input_sticks  → cb_input_tiles         │
                        │     pass A: square    cb_input_tiles   → cb_x_squared           │
                        │             reduce    cb_x_squared     → cb_row_stat  (Σx²)     │
                        │             finalize  cb_row_stat      → cb_row_stat  (1/rms)   │
                        │     pass B: mul<Col>  cb_input_tiles   → cb_normalized          │
                        │             mul<Row>  cb_normalized    → cb_output_tiles        │
                        │     [RM only] untilize cb_output_tiles → cb_output_sticks       │
                        │                     │                                          │
                        │                     ▼                                          │
                        │ WRITER (BRISC, NoC1) ──────────────────────────────────────────►│ DRAM
                        └────────────────────────────────────────────────────────────────┘
```

Format at each stage:

| Stage | TILE input | ROW_MAJOR input |
|-------|-----------|-----------------|
| DRAM → L1 | tiles (page = tile) | sticks (page = `W · elem_size`), staged into a padded `32 × (WT_CHUNK·32)` row-major block; reader **zero-fills** the `[W, WT_CHUNK·32)` tail of every staged row and every row ≥ `ROWS_IN_LAST_TILEROW` of the final tile-row |
| L1 → compute | tiles | tiles after `tilize<WT_CHUNK>` |
| compute → L1 | tiles | tiles → `untilize<WT_CHUNK>` → padded row-major block |
| L1 → DRAM | tiles | writer emits `ROWS_IN_LAST_TILEROW`-aware stick counts, `W · elem_size` bytes each |

Reader on NoC0, writer on NoC1 (`ReaderConfigDescriptor` / `WriterConfigDescriptor` defaults) — reversing them costs 2.5–4.8× (`noc_placement/README.md`).

### 3.4 Tensix-to-Tensix contract for Lamp L1 (`GRID_W > 1`) — designed, not executed in Phase 0

Phase 0 does not use inter-Tensix communication. The unlocked scheme is specified here so Phase 0's structure is provably non-blocking for it.

**Topology.** The grid is viewed as `GRID_H × GRID_W`. Core `(h, w)` owns tile-rows assigned to grid-row `h` and width tiles `[w_start(w), w_start(w) + w_count(w))`. Each grid **row** is a 1-D W-group of `GRID_W` cores. Selection: `GRID_H = min(Rt, num_cores)`, `GRID_W = clamp(num_cores // GRID_H, 1, Wt)`, forced `> 1` when `Wt · bt` exceeds one core's resident budget.

**Per row-block, per group:**

| Step | Who | What | Sync |
|------|-----|------|------|
| 1 | every core `(h,w)` | pass A over its own width slice → `BLOCK_ROWS` raw partial-`Σx²` tiles, packed into **`cb_sum_handoff`** — a CB dedicated to the cross-kernel handoff, *distinct* from `cb_row_stat` (a CB read by a dataflow kernel and used as a compute accumulator would be two consumers — `ttnn-cb-memory-fundamentals.md:94-118`) | — |
| 2 | non-root cores `w>0` | `noc_async_write` the `BLOCK_ROWS` tiles into a per-sender slot of the root's `cb_partials_gathered`, then `noc_semaphore_inc` the root's arrival semaphore | root spins until arrivals == `GRID_W-1` |
| 3 | root `(h,0)` | sum the `(GRID_W-1) · BLOCK_ROWS` gathered tiles into its own partials (pairwise `add_tiles(acc_to_dest)`, fp32 DEST, live batch from `DEST_AUTO_LIMIT`, odd count seeded by one `copy_tile`), then run the **same** `transform_in_place` finalize (×`INV_W`, +eps, rsqrt) | — |
| 4 | root | `SenderPipe::send()` the `BLOCK_ROWS` finalized stat tiles to the group's `cb_row_stat` | `send()` is atomic: consumer-ready wait → mcast → data-ready signal on the same VC (data-before-signal) → fence |
| 5 | every core | `ReceiverPipe::receive()`, then pass B on its own width slice with its own gamma slice → its own output tiles | one handshake round per row-block |

No output combine is needed — the output is width-partitioned, so every core writes disjoint tiles.

**Host wiring:** `Mcast1D(device, all_cores, Mcast1DShape::PerRow, starting_sender_index=0, McastConfig{noc=NOC_0, handshake=true})` supplies the semaphores (`owned_semaphores()`), CT args (`compile_time_args()`) and per-core RT args (`runtime_args(core)`); the kernel side reconstructs the pipes through `McastArgs<CT_BASE, RT_BASE>`. Plus one arrival semaphore per group for step 2.

**Combine cost budget:** a per-row partial sum is the **1 tile/core** regime, where `tensix_all_reduce/README.md` measures flat-root / grid-two-stage at **1.27–1.55 µs** per group and explicitly rules out tile-index two-phase (worst there). On a 1-D group grid-two-stage collapses to the flat root, which is why step 3 is a flat root reduce.

### 3.5 Tensix-to-Tensix contract for Lamp L2 (gamma broadcast)

`Mcast1D` with `Mcast1DShape::PerRow`; injector = column 0 of each grid row. Payload = the `Wt` gamma tiles, chunked to `cb_gamma_tiles`' capacity, `cb_in`-style double-buffered so the injector prefetches while consumers drain (`shared_input_reuse/README.md`). One `SenderPipe::send()` / `ReceiverPipe::receive()` round per chunk; receivers consume `cb_gamma_tiles` exactly as in Phase 0. Only the CB's *filler* changes; the compute kernel is untouched.

---

## 4. Work Distribution

This is §1's core-assignment knob made concrete.

| Field | Value |
|-------|-------|
| **Work unit** | one **row-block** = `BLOCK_ROWS` tile-rows × `WT_CHUNK` width tiles (`NUM_W_CHUNKS` chunks make a full row) |
| **Grid** | `device.compute_with_storage_grid_size()` — the full compute grid, `GRID_W = 1` (all width on one core) |
| **Split** | `ttnn.split_work_to_cores(grid, Rt, row_wise=True)` → `(num_cores, all_cores, core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2)` (`ttnn/cpp/ttnn-nanobind/operations/core.cpp:466-498`). `row_wise=True` is mandatory (2.91× vs a column line) |
| **Per-core work** | core `c` owns tile-rows `[row_start(c), row_start(c) + rows_of(c))` where `rows_of(c)` is `rows_per_core_g1` or `rows_per_core_g2` and `row_start(c)` is the running prefix sum over `ttnn.corerange_to_cores(all_cores, None, True)`. It runs `div_up(rows_of(c), BLOCK_ROWS)` blocks |
| **Remainder** | two-group split handles the cross-core remainder (`core_group_2` gets one fewer row). The **per-core** remainder is a ragged final block of `rows_of(c) - (nblk-1)·BLOCK_ROWS` tile-rows: CBs are sized for `BLOCK_ROWS`, and the helper shapes (`EltwiseShape::grid(rows, cols)`, `ReduceInputBlockShape::of(rows, cols)`) take the runtime `rows` for that block. Likewise `NUM_W_CHUNKS` may have a short final chunk of `Wt - (NUM_W_CHUNKS-1)·WT_CHUNK` tiles |
| **Cores when `Rt < num_cores`** | `split_work_to_cores` returns `num_cores = Rt`; the remaining cores are idle. **This is the shape Lamp L1 exists for** and the golden suite's `_WIDE` / decode perf cases target it |

### 4.1 Alignment-aware tile geometry (`ceil`, per-image — never `floor`/`//`)

| Quantity | TILE layout | ROW_MAJOR layout |
|----------|-------------|------------------|
| width tiles | `Wt = div_up(W, 32)` | `Wt = div_up(W, 32)` |
| tile-rows | `Rt = prod(shape[:-2]) * div_up(shape[-2], 32)` — each image tile-pads H **independently**; `div_up(prod(shape[:-2])*shape[-2], 32)` would be wrong | `Rt = div_up(prod(shape[:-1]), 32)` — RM has **no** implicit H padding, so the rows of all images are contiguous |
| partial W | `PARTIAL_W = W % 32` (0 ⇒ aligned) | same |
| partial rows | last tile-row of **each image** holds `shape[-2] % 32` real rows (padding rows are self-contained; see R2) | only the **global** last tile-row is short: `ROWS_IN_LAST_TILEROW` |
| global tile id | `tile_id(r, wt) = r * Wt + wt` for global tile-row `r ∈ [0, Rt)` — valid for every rank because images are contiguous and each contributes `Ht_img · Wt` tiles row-major | stick id `= g` for global row `g ∈ [0, R_rm)`; tile-row `r` covers `[r·32, min((r+1)·32, R_rm))` |
| gamma tile id | `wt` | gamma is one stick |

### 4.2 Regime-selection function (pinned) and required regime-pinned tests

The op selects between **two compute regimes**. The selection function is exactly the `if BLOCK_ROWS_L1_MAX >= 1` branch in §1.4 — a pure function of `(input_layout, input_dtype, gamma_dtype, gamma_layout, HAS_GAMMA, Wt, _L1_CB_BUDGET_BYTES)`. It does **not** depend on `Rt`, `num_cores`, or the device grid — so it is independent of *work distribution* and reproducible for a fixed device.

It is **not** device-independent: `_L1_CB_BUDGET_BYTES` now derives from `ttnn.get_max_worker_l1_unreserved_size()` (§1.4), which differs between Wormhole and Blackhole and between L1-small configurations. The regime boundary therefore moves with the device, which is precisely why the regime-pinned tests below are **mandatory** rather than nice-to-have: a shape that lands in RESIDENT on one part can land in STREAM on another, and only a test that pins each regime by shape will catch a STREAM-only bug on a device whose budget hides it.

| Regime | Predicate | Which input class lands here |
|--------|-----------|-----------------------------|
| **RESIDENT** | `BLOCK_ROWS_L1_MAX >= 1`, i.e. one tile-row's whole working set fits the budget | narrow-to-medium `W`. For bf16 / TILE / bf16-TILE-gamma this is roughly `Wt ≤ 55` (`W ≲ 1800`); for fp32 roughly half that; wider for the no-gamma and ROW_MAJOR (smaller `CB_BLOCK_MULT`) builds |
| **STREAM** | `BLOCK_ROWS_L1_MAX == 0` | wide `W` (`W = 4096`, `8192`, and every `_WIDE` loose case). `BLOCK_ROWS = 1`, x re-read in pass B |

**Regime-pinned tests are required** (a regime that only triggers on some grids/dtypes can pass on one device and fail on another). The acceptance test (`tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm.py`) pins:
* RESIDENT × {TILE, ROW_MAJOR} × {bf16, fp32} × {gamma, no gamma} — `W = 64/128/512`;
* STREAM × {TILE, ROW_MAJOR} × {bf16, fp32} — `W = 4096`;
* both regimes crossed with `w_non_aligned` and `h_non_aligned`.

---

## 5. Circular Buffers

`bt = tile_size(input_dtype)`, `gt = tile_size(gamma_dtype)`, `st = tile_size(bfloat16)`, `ft = tile_size(float32)`. **Every `Num Pages` entry is a function of the block/depth knobs only — never of a whole-op dimension.** `Wt` appears only in `cb_gamma_*`, whose extent *is* the gamma tensor and which is bounded by the same L1 budget through §1.4's `FIXED_RES`/`FIXED_STREAM` terms (in STREAM it collapses to `WT_CHUNK`).

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_sticks` | 0 | `bt` (tile-sized page holding 32 padded RM rows' worth of one tile column) | `CB_RM_STAGE_DEPTH * WT_CHUNK` | input dtype | reader | compute | ROW_MAJOR input only; per tile-row |
| `cb_input_tiles` | 1 | `bt` | `CB_X_DEPTH * BLOCK_ROWS * WT_CHUNK` | input dtype | **TILE build:** reader. **ROW_MAJOR build:** compute (`tilize`) | compute | one row-block; **held across pass A and pass B** in RESIDENT |
| `cb_x_squared` | 2 | `bt` | `BLOCK_ROWS * WT_CHUNK` | input dtype | compute | compute | within pass A of one block; full block (sequential helpers) |
| `cb_scaler` | 3 | `st` | `1 + (1 if PARTIAL_W else 0)` | **`ttnn.bfloat16`** | reader | compute | whole kernel; filled once at boot, never popped by `reduce` |
| `cb_row_stat` | 4 | `ft` | `BLOCK_ROWS` | **`ttnn.float32`** | compute | compute | one block: `Σx²` accumulator → in-place `1/rms` → `Col` operand of pass B |
| `cb_gamma_sticks` | 5 | `gt` | `WT_CHUNK` | gamma dtype | reader | compute | ROW_MAJOR gamma only; one padded tile-row (row 0 = gamma, rows 1–31 zero) |
| `cb_gamma_tiles` | 6 | `gt` | `WT_CHUNK` | gamma dtype | **TILE gamma:** reader. **ROW_MAJOR gamma:** compute (`tilize`) | compute | RESIDENT: filled once per core, never popped. STREAM: per chunk |
| `cb_normalized` | 7 | `bt` | `BLOCK_ROWS * WT_CHUNK` | input dtype | compute | compute | **allocated only when `HAS_GAMMA`**; full block (sequential helpers) |
| `cb_output_tiles` | 8 | `bt` | `CB_OUT_DEPTH * BLOCK_ROWS * WT_CHUNK` | output dtype (= input dtype) | compute | **TILE build:** writer. **ROW_MAJOR build:** compute (`untilize`) | one row-block |
| `cb_output_sticks` | 9 | `bt` | `CB_RM_STAGE_DEPTH * WT_CHUNK` | output dtype | compute (`untilize`) | writer | ROW_MAJOR output only; per tile-row |

Lamp-only CBs (not allocated in Phase 0): `cb_sum_handoff` (`BLOCK_ROWS` fp32 pages, producer compute, consumer reader-as-mcast-source), `cb_partials_gathered` (`(GRID_W-1) * BLOCK_ROWS` fp32 pages on the root, producer reader, consumer compute).

### 5.1 CB sync ledger — push count = wait count, per CB, per regime

| CB | Producer pushes (per row-block) | Consumer waits / pops (per row-block) |
|----|-------------------------------|----------------------------------------|
| `cb_input_sticks` | reader: `WT_CHUNK` per tile-row × `BLOCK_ROWS` × `NUM_W_CHUNKS` × (1 if `X_RESIDENT` else 2) | `tilize<WT_CHUNK>` waits + pops `WT_CHUNK` per block, `num_blocks = BLOCK_ROWS` per chunk |
| `cb_input_tiles` | **RESIDENT:** `BLOCK_ROWS·Wt` once. **STREAM:** `rows·wt_c` per chunk, **twice** (pass A, pass B) | **RESIDENT:** `square` waits `BLOCK_ROWS·Wt` (`WaitPolicy::Upfront`, `PopPolicy::None`); pass-B `mul` waits the same count (idempotent) and pops it (`PopPolicy::AtEnd`). Net pops == pushes. **STREAM:** each pass waits + pops `rows·wt_c` per chunk |
| `cb_x_squared` | `square`: `rows·wt_c` per chunk | `reduce` with `BulkWaitBulkPop`: waits `wt_c` and pops `wt_c` **per row** ⇒ `rows·wt_c` per chunk |
| `cb_scaler` | reader: `1 + PARTIAL_W?1:0`, **once at boot** | `reduce` waits `last_tile_scaler_idx + 1`, never pops. Compute issues one explicit `cb_pop_front(cb_scaler, 1 + PARTIAL_W?1:0)` at kernel end (mirrors `toy_variance/kernels/compute.cpp:126`) |
| `cb_row_stat` | `reduce` output: `rows` per chunk (accumulator CB is also the output CB on the `ReduceTile` datapath); then `transform_in_place` pops 1 / pushes 1, `rows` times | pass-B `mul` `Col` operand: `WaitPolicy::Upfront`, `PopPolicy::None`; compute issues one explicit `cb_pop_front(cb_row_stat, rows)` after the pass-B chunk loop |
| `cb_gamma_sticks` | reader: `WT_CHUNK` (RESIDENT: once per core; STREAM: per chunk per block) | `tilize<WT_CHUNK>(1)` waits + pops `WT_CHUNK` |
| `cb_gamma_tiles` | `WT_CHUNK` (RESIDENT: once per core; STREAM: per chunk per block) | pass-B gamma `mul` `Row` operand: `WaitPolicy::Upfront`, `PopPolicy::None`; explicit `cb_pop_front(cb_gamma_tiles, wt_c)` after each chunk **only when `!X_RESIDENT`** (CT flag `GAMMA_POP_PER_CHUNK = !X_RESIDENT`) |
| `cb_normalized` | normalize `mul`: `rows·wt_c` per chunk | gamma `mul`: `rows·wt_c` per chunk |
| `cb_output_tiles` | final `mul` (or the normalize `mul` when `!HAS_GAMMA`): `rows·wt_c` per chunk | **TILE:** writer waits + pops `rows·wt_c`. **ROW_MAJOR:** `untilize<WT_CHUNK>(rows)` waits + pops `rows·wt_c` |
| `cb_output_sticks` | `untilize<WT_CHUNK>`: `WT_CHUNK` per tile-row | writer waits + pops `WT_CHUNK` per tile-row |

### 5.2 CB ownership — exactly one producer kernel and one consumer kernel per CB, per build

Verified for both builds. `cb_input_tiles` and `cb_gamma_tiles` change producer between the TILE and ROW_MAJOR builds (reader vs `tilize`); `cb_output_tiles` changes consumer (writer vs `untilize`). Within any one build each is single-producer / single-consumer. `cb_row_stat` and `cb_gamma_tiles` are read *and* written only by compute — no dataflow kernel touches them. **No in-place eltwise ever targets a reader-fed CB**: normalizing in place into `cb_input_tiles` would make compute a *second producer* of a reader-produced CB (silent UB), which is why `cb_normalized` exists as a separate CB. `cb_sum_handoff` exists (in Lamp L1) precisely so the mcast source is not a second consumer of `cb_row_stat`.

### 5.3 `memory_layout` TARGET values → blocking-model classification

| `memory_layout` | Which axis it cuts | Character of that axis for **this** op | Unlock class | Consumption in the design |
|-----------------|--------------------|---------------------------------------|--------------|---------------------------|
| `INTERLEAVED` | none (logical split imposed by the op) | — | Phase 0 | `TensorAccessor` reads/writes |
| `HEIGHT_SHARDED` | leading/row dims → `row` | **independent** | **knob-turn** (Lamp L3) | `ttnn.cb_descriptor_from_sharded_tensor` for `cb_input_tiles`/`cb_output_tiles` — zero-copy, **no NoC read for x**; `BLOCK_ROWS` = the shard's tile-rows |
| `WIDTH_SHARDED` | last dim → `width` | **dependent** | **scheme-change** (Lamp L4 ⊃ L1) | per-core slice pre-placed in L1 (CB from sharded tensor) + the §3.4 cross-core combine |
| `BLOCK_SHARDED` | both | `row` independent, `width` dependent | **scheme-change** (Lamp L4 ⊃ L1) | as `WIDTH_SHARDED`, with `GRID_H > 1` |

The pairing here (height = trivial, width/block = combine) is not a property of the flavours — it falls out of *this* op's axis characters, read off the math in §1.1.

---

## 6. API Mapping

Every mechanism has a verified file:line reference. Knob columns flag which template params / args are the tunable block factors from §1.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| boot (compute) | raw_api | `compute_kernel_hw_startup(cb_a, cb_scaler, cb_output_tiles)` | required by `reduce_helpers_compute.hpp:408-412`; precedent `toy_variance/kernels/compute.cpp:52` | `cb_a` = `cb_input_sticks` (RM) or `cb_input_tiles` (TILE) | — | — | Exactly **once**, first statement of `MAIN()`; never re-called |
| boot (reader) — aligned W | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f)` | `reduce_helpers_dataflow.hpp:58-60` | pool-type-aware overload (PoolType + ReduceDim template args) | — | `cb_scaler` | Value is exactly **1.0** (bit-exact in bf16). `1/W` is *not* folded here — see R3 |
| boot (reader) — partial W | helper | `dataflow_kernel_lib::prepare_partial_reduce_scalers<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW, PARTIAL_W>(1.0f)` | `reduce_helpers_dataflow.hpp:131-132` (semantics `:108-113`) | `PARTIAL_W` ∈ `[1,31]` is a CT arg | — | `cb_scaler` | Emits tile 0 = full scaler, tile 1 = partial scaler. Precedent: `toy_variance/kernels/reader.cpp:42-52`, `toy_reduce_partial/kernels/reader.cpp:30-39` |
| tilize x (RM only) | helper | `compute_kernel_lib::tilize<WT_CHUNK, cb_input_sticks, cb_input_tiles>(rows)` | `tilize_helpers.hpp:187-197` | **`WT_CHUNK` is the block knob** (`block_width_tiles`); `num_blocks = rows` | `cb_input_sticks` | `cb_input_tiles` | Symmetric (tile-sized) pages both sides; `input_dfb != output_dfb` |
| tilize gamma (RM gamma only) | helper | `compute_kernel_lib::tilize<WT_CHUNK, cb_gamma_sticks, cb_gamma_tiles>(1)` | `tilize_helpers.hpp:187-197` | `num_blocks = 1` | `cb_gamma_sticks` | `cb_gamma_tiles` | Reader zero-fills the staged block and writes gamma into row 0 only ⇒ tiles have gamma in row 0, zeros elsewhere — exactly what `BroadcastDim::Row` reads |
| A1 — x² | helper | `compute_kernel_lib::square<input(cb_input_tiles, wait, pop, OperandKind::Block), output(cb_x_squared)>(EltwiseShape::grid(rows, wt_c))` | `eltwise_convenience.hpp:73-74` (forwards to `BinaryFpu<In,In,Mul>` — `eltwise_convenience.inl:33-38`) | **`EltwiseShape::grid(rows, wt_c)` is the block knob** (`eltwise_chain.hpp:192`). `pop = PopPolicy::None` when `X_RESIDENT`, `PopPolicy::AtEnd` otherwise | `cb_input_tiles` | `cb_x_squared` | One call, one wait/pop of the input, same buffer as both operands |
| A2 — `Σx²` | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_x_squared, cb_scaler, cb_row_stat, ReduceInputPolicy::BulkWaitBulkPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, REDUCE_FP32_MODE, ReduceAlgorithm::ReduceTile>(ReduceInputBlockShape::of(rows, wt_c), ReduceInputMemoryLayout::contiguous(), accumulate_c, NoOp{}, partial_scaler)` | `reduce_helpers_compute.hpp:525-542`; `ReduceInputBlockShape::of` `:216`; `ReduceInputPolicy` `:109`; `ReduceAlgorithm` `:153`; `Accumulate::at` `:328`; `ReduceFp32Mode` `reduce_helpers_common.hpp:18`. Chunked-accumulate idiom documented at `reduce_helpers_compute.hpp:563-578` | **`ReduceInputBlockShape::of(rows, wt_c)` is the block knob**; `accumulate_c` is the width-chunk knob: `NoAccumulation{}` when `NUM_W_CHUNKS == 1`, else `Accumulate::at(cb_row_stat, c)`. `REDUCE_FP32_MODE` and `ReduceAlgorithm` are **explicit**, not defaulted (R14) | `cb_x_squared` | `cb_row_stat` | Scaler CB filled first. `cb_row_stat` sized `rows` pages and is **both** output and accumulator on the `ReduceTile` datapath (the accumulator CB *is* the output CB — `reduce_helpers_compute.inl:657`/`:678` reload-pop/push). `partial_scaler` is `last_tile_at(1)` **only on the chunk containing width tile `Wt-1`**, `none()` otherwise — the caller routes it, since calling `reduce<>` directly means no wrapper does it. Precedent: `toy_variance/kernels/compute.cpp:99-115` |
| A2 (partial W) | helper | `ReducePartialScaler::last_tile_at(1)` / `::none()` | `reduce_helpers_compute.hpp:261` / `:260`; doc `:224-249` | CT-selected on `PARTIAL_W != 0` | — | — | Pairs with `prepare_partial_reduce_scalers`. Precedent: `toy_variance/kernels/compute.cpp:60-61`, `toy_reduce_partial/kernels/compute.cpp:31-32` |
| A3 — finalize | helper | `compute_kernel_lib::transform_in_place(cb_row_stat, λ)`, `rows` times; λ issues scalar-multiply by `INV_W`, scalar-add of `EPS`, then `rsqrt` | `streaming_reduce_helpers.hpp:110-111`; recommended for exactly this case at `:75-78` ("multi-instruction finalizers (e.g. rsqrt-with-eps)"). Chain-struct equivalents: `eltwise_scalar.hpp` (`MulUnary`/`AddUnary`), `eltwise_math.hpp:37-38` (`Rsqrt<>`) | `INV_W`, `EPS` are CT args (raw fp32 bits) | `cb_row_stat` | `cb_row_stat` | Pops before reserving ⇒ a `rows`-page CB is sufficient; SRCA/packer reconfig is bundled by the helper. Lamp L6b: scope the rsqrt to `VectorMode::C` (the stat lives in column 0) |
| B1 — normalize | helper | `compute_kernel_lib::mul<input(cb_input_tiles, Upfront, pop, OperandKind::Block), input(cb_row_stat, Upfront, PopPolicy::None, OperandKind::Col), output(NORM_OUT), BroadcastDim::Col>(EltwiseShape::grid(rows, wt_c))` | `eltwise_convenience.hpp:58-64`; `BroadcastDim` `eltwise_chain.hpp:461-466` + doc `:451-460`; `OperandKind` `:282-287` + doc `:270-280`; `input()` `:370-379`; `output()` `:384-393` | `NORM_OUT = HAS_GAMMA ? cb_normalized : cb_output_tiles` (CT). **Block knob:** `EltwiseShape::grid` | `cb_input_tiles`, `cb_row_stat` | `cb_normalized` \| `cb_output_tiles` | `BroadcastDim::Col` because a `REDUCE_ROW` result is column-shaped and broadcasts back **across columns** (`eltwise_chain.hpp:451-455`). The broadcast source is always operand **B** (`:456-458`) ⇒ the stat must be the second operand. `OperandKind::Col` (indexed by row only) requires the 2-D `grid` shape (`:278-279`); the default `Scalar` would silently pin tile 0 |
| B2 — gamma | helper | `compute_kernel_lib::mul<input(cb_normalized, …, OperandKind::Block), input(cb_gamma_tiles, Upfront, PopPolicy::None, OperandKind::Row), output(cb_output_tiles), BroadcastDim::Row>(EltwiseShape::grid(rows, wt_c))` | `eltwise_convenience.hpp:58-64` | skipped entirely when `!HAS_GAMMA` | `cb_normalized`, `cb_gamma_tiles` | `cb_output_tiles` | `BroadcastDim::Row` because gamma is row-shaped (`1 × W`, valid in row 0) and broadcasts **down rows**; `OperandKind::Row` = indexed by column only. Do **not** fuse B1+B2 via `DestReuseBinary` — it has no broadcast param (`eltwise_chain.hpp:518-520`) and dest-reuse on an FPU consumer measured **0.82–1.02×** (`compute_fusion/README.md:91-97`) |
| untilize (RM only) | helper | `compute_kernel_lib::untilize<WT_CHUNK, cb_output_tiles, cb_output_sticks>(rows)` | `untilize_helpers.hpp:145-154` | **`WT_CHUNK` is the block knob** | `cb_output_tiles` | `cb_output_sticks` | Symmetric tile-sized pages both sides |
| DEST budget | helper | `DEST_AUTO_LIMIT` | `dest_helpers.hpp:103`, `get_dest_limit()` `:89-100` | — | — | — | `fp32_dest_acc_en=True` halves the budget (8→4 half-sync, 16→8 full-sync). Never hardcode `8` (`eltwise_chain.hpp:399-401`). All helpers used here derive their live batch from it |
| reader/writer I/O | raw_api | `TensorAccessor` + `noc_async_read` / `noc_async_write` | `.claude/references/ttnn-cb-memory-fundamentals.md:244-268`; host `ttnn.TensorAccessorArgs` `ttnn/cpp/ttnn-nanobind/tensor_accessor_args.cpp:15-42` | one accessor per tensor slot; `TensorAccessorArgs` CT args go **last** | — | — | Include `"api/dataflow/dataflow_api.h"`. Interleaved I/O only; Lamp L3 replaces the x path with a sharded CB |
| RM stick read (x) | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks, TilizeGranularity::TILE>(accessor, total_num_rows, row_bytes, start_page, byte_offset_within_page)` | `tilize_helpers_dataflow.hpp:87-93` (contract `:46-86`) | `TilizeGranularity::TILE` matches `cb_input_sticks`' tile-sized pages and `tilize`'s **symmetric** mode (`:16-19`, `:54-58`). `start_page = row_start(c)·32` is the per-core offset (`:76-78`); **`byte_offset_within_page = c · row_bytes` is the `WT_CHUNK` width-chunk knob** (`:79-85`, which states CB sizing then scales with the chunk width, not full `W` — exactly §1.4's bound) | — | `cb_input_sticks` | Handles non-tile-aligned **W** by padding the L1 stride and non-tile-aligned **H** by pushing full tile pages for the last partial block. **Untouched rows contain stale data, not zeros** (`:51-52`) — safe here by R2/R15, not by zero-fill |
| RM stick write (out) | helper | `dataflow_kernel_lib::write_sticks_after_untilize<cb_output_sticks>(accessor, total_num_rows, row_bytes, start_page, byte_offset_within_page)` | `tilize_helpers_dataflow.hpp:129-135` (contract `:95-128`) | same `start_page` / `byte_offset_within_page` knobs, symmetric to the read helper (`:123-127`) | `cb_output_sticks` | — | Skips the L1 W-padding and **writes only the valid rows** of the last partial block (`:102-104`), so stale H-padding rows are computed and then dropped — no output padding exists in ROW_MAJOR |
| RM gamma stick read | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_sticks, TilizeGranularity::TILE>(gamma_accessor, 1, row_bytes, 0, c · row_bytes)` | `tilize_helpers_dataflow.hpp:87-93` | `total_num_rows = 1` — gamma is a single stick | — | `cb_gamma_sticks` | With `total_num_rows = 1` the helper pushes full tile pages whose rows 1–31 are **stale**, so `BroadcastDim::Row` (which reads row 0 only) gets the right values regardless — see R15 |
| host — program | raw_api | `ttnn.ProgramDescriptor` / `KernelDescriptor` / `CBDescriptor` / `CBFormatDescriptor` / `ComputeConfigDescriptor` / `RuntimeArgs` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:930-961`, `:694-737`, `:398-419`, `:325-363`, `:627-655`, `:167-243` | `ReaderConfigDescriptor` / `WriterConfigDescriptor` (`:577-592`) keep reader on NoC0 / writer on NoC1 | — | — | Output tensor **last** in `ttnn.generic_op([...], pd)` (`ttnn/cpp/ttnn/operations/generic/generic_op.hpp:25`; `.claude/references/generic_op_template/template_op.py:58-59`) |
| host — geometry | raw_api | `ttnn.div_up`, `ttnn.tile_size`, `tensor.buffer_page_size()`, `tensor.buffer_num_pages()`, `ttnn.split_work_to_cores`, `ttnn.corerange_to_cores` | `.claude/references/ttnn-python-utility-bindings.md:78-140`, `:9-51`; `ttnn/cpp/ttnn-nanobind/operations/core.cpp:466-498`; `ttnn/cpp/ttnn-nanobind/tensor.cpp:611-612` | — | — | — | `div_up` everywhere — never `//` for a tile count |
| Lamp L1/L2 only | helper | `McastArgs<>` / `SenderPipe::send()` / `ReceiverPipe::receive()`; host `Mcast1D` + `McastConfig` | `mcast_pipe.hpp:327-395`, `:197`, `:274`; `host/mcast_host.hpp:156-164`, `:87-103`, `:222`, `:251`, `:266` | — | — | — | Not built in Phase 0; contract fixed in §3.4/§3.5 |

### 6.1 Helpers considered and rejected

Every compute phase and the entire ROW_MAJOR dataflow path go through helpers. Four entries below record a non-use or a deliberate choice between helpers; each carries a concrete, file:line-cited reason. Only **one** genuine raw-API fallback survives — the TILE-layout whole-tile accessor I/O, which no dataflow helper in `kernel_lib/` covers.

| Raw API used | Helper considered | File:line of the mismatch | Concrete reason |
|--------------|-------------------|---------------------------|-----------------|
| reader/writer `TensorAccessor` + `noc_async_read/write` for the **TILE** x and out paths only | `dataflow_kernel_lib::read_sticks_for_tilize` / `write_sticks_after_untilize` | `tilize_helpers_dataflow.hpp:87-93`, `:129-135` | **Decided, not deferred:** those two helpers are **mandated** for the whole ROW_MAJOR path (x sticks, gamma stick, output sticks) — see the three RM rows in §6. They cover exactly this op's needs: `TilizeGranularity::TILE` matches `cb_input_sticks`' tile-sized pages and `tilize`'s symmetric mode (`:16-19`), `start_page` carries the per-core row offset (`:76-78`), and `byte_offset_within_page` carries the `WT_CHUNK` width chunking with CB sizing bounded by the chunk width rather than full `W` (`:79-85`). They do **not** cover the **TILE** layout path, which reads and writes whole interleaved *tiles* rather than row-major sticks and has no tilize stage at all — that is the only remaining accessor-level I/O, and it is a genuine gap in the helper's scope, not a preference. |
| explicit `reduce<>` call at A2 instead of the `accumulate_reduce_block` wrapper | `compute_kernel_lib::accumulate_reduce_block` | `streaming_reduce_helpers.hpp:47-61` — the template list is `<pool, rdim, cb_in, cb_scaler, cb_acc, in_policy, reconfig_mode, PostOp>`: **no `ReduceFp32Mode` and no `ReduceAlgorithm` parameter** | `reduce<>` *is* the helper; `accumulate_reduce_block` is a 16-line convenience wrapper over it whose only added value is auto-routing `partial`/`post_op` to the last chunk. It hard-codes `fp32_mode` and `algorithm` to their defaults, so it cannot express either of the two settings this op pins in §1.4 / R14: `ReduceFp32Mode::Accurate` for the `float32` corner, and an explicit `ReduceAlgorithm::ReduceTile`. Calling `reduce<>` with its own `Accumulate::at(...)` argument is the idiom the header itself documents for chunked accumulation (`reduce_helpers_compute.hpp:563-578`). Cost of the swap: the caller routes `partial_scaler` to the last chunk itself (one ternary). |
| the eps/`INV_W`/rsqrt lambda inside `transform_in_place` | `eltwise_chain` with `MulUnary` + `AddUnary` + `Rsqrt<>` + `PackTile` | `streaming_reduce_helpers.hpp:75-78` | The reduce-helper family explicitly directs multi-instruction finalizers (**"e.g. rsqrt-with-eps"**) to `transform_in_place` rather than a chain, because it pops-before-reserves (so a small CB cannot deadlock) and bundles the SRCA/packer reconfigs around the accumulator read (`:100-108`). A chain would need a second `BLOCK_ROWS`-page fp32 CB for no benefit. `transform_in_place` **is** a helper; only the 3 SFPU ops inside the lambda are raw, which is that helper's documented calling convention. |
| explicit `cb_pop_front(cb_row_stat, rows)` / `cb_pop_front(cb_gamma_tiles, wt_c)` / `cb_pop_front(cb_scaler, …)` | letting the `mul`/`reduce` helpers' `PopPolicy` do it | `eltwise_chain.hpp:262` (`PopPolicy`), `reduce_helpers_compute.hpp:906` (scaler never popped) | The stat and gamma operands must survive **all `NUM_W_CHUNKS` chunks** of pass B, so no per-call `PopPolicy` can release them at the right moment; the scaler must survive the whole kernel because `reduce` waits on it but never pops it. Explicit pops at the exact lifetime boundary is the sanctioned pattern — precedent `toy_variance/kernels/compute.cpp:119` (`cb_pop_front(cb_mean, Ht)`) and `:126` (scaler). |

---

## 7. Compute Phases

One loop nest for both regimes. `rows = min(BLOCK_ROWS, rows_remaining)`; `wt_c = min(WT_CHUNK, Wt - c*WT_CHUNK)`.

```
compute_kernel_hw_startup(CB_A, cb_scaler, cb_output_tiles)          # exactly once
[RESIDENT] fill cb_gamma_tiles once (tilize if GAMMA_IS_RM)          # gamma resident per core
for blk in range(BLOCKS_PER_CORE):                                   # rows across cores (§4)
    # ---- pass A: Σx² over the whole width -------------------------------
    for c in range(NUM_W_CHUNKS):
        [RM] tilize<WT_CHUNK>(rows)                 cb_input_sticks -> cb_input_tiles
        square(grid(rows, wt_c))                    cb_input_tiles  -> cb_x_squared
        reduce<SUM, REDUCE_ROW, .., REDUCE_FP32_MODE, ReduceTile>(of(rows, wt_c),
                contiguous(), (NUM_W_CHUNKS == 1 ? NoAccumulation{}
                                                 : Accumulate::at(cb_row_stat, c)),
                NoOp{}, (c == LAST_W_CHUNK ? partial_scaler : none()))
                                                    cb_x_squared    -> cb_row_stat
    # ---- finalize: mean, eps, rsqrt --------------------------------------
    repeat rows times: transform_in_place(cb_row_stat, λ[×INV_W, +EPS, rsqrt])
    # ---- pass B: scale ---------------------------------------------------
    for c in range(NUM_W_CHUNKS):
        [STREAM, RM] tilize<WT_CHUNK>(rows)         cb_input_sticks -> cb_input_tiles
        mul<Col>(grid(rows, wt_c))                  cb_input_tiles, cb_row_stat -> NORM_OUT
        if HAS_GAMMA:
            mul<Row>(grid(rows, wt_c))              cb_normalized, cb_gamma_tiles -> cb_output_tiles
        [RM] untilize<WT_CHUNK>(rows)               cb_output_tiles -> cb_output_sticks
        if not X_RESIDENT: cb_pop_front(cb_gamma_tiles, wt_c)
    cb_pop_front(cb_row_stat, rows)
cb_pop_front(cb_scaler, 1 + (1 if PARTIAL_W else 0))
[RESIDENT] cb_pop_front(cb_gamma_tiles, Wt)
```

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|--------------------------|-------------------|----------------|
| 0a | fill scaler | yes (`prepare_[partial_]reduce_scaler[s]`) | — | `cb_scaler` (1–2, bf16, value 1.0) | persists for the whole kernel; `reduce` waits, never pops |
| 0b | fill gamma (RESIDENT) | yes (`tilize` when RM gamma) | `cb_gamma_sticks` (`Wt`) | `cb_gamma_tiles` (`Wt`, row 0 valid) | **persists across every row-block**; popped only at kernel end |
| 1 | tilize x (RM only) | yes (`tilize<WT_CHUNK>`) | `cb_input_sticks` (`WT_CHUNK` per tile-row, padded RM) | `cb_input_tiles` (`rows·wt_c`) | `cb_input_sticks` drained |
| 2 | `x²` | yes (`square`) | `cb_input_tiles` (`rows·wt_c`, **not popped when `X_RESIDENT`**) | `cb_x_squared` (`rows·wt_c`) | `cb_input_tiles` still full in RESIDENT — this is what makes pass B free of a re-read |
| 3 | `Σx²` (chunk-accumulating row reduce) | yes (`reduce<>` + `Accumulate::at`) | `cb_x_squared` (`rows·wt_c`), `cb_scaler` (1–2) | `cb_row_stat` (`rows`, fp32) | `cb_x_squared` drained; `cb_row_stat` holds the running raw sum (final after chunk `NUM_W_CHUNKS-1`). Accumulator CB **is** the output CB on the `ReduceTile` datapath |
| 4 | `1/rms = rsqrt(Σx²·(1/W) + eps)` | yes (`transform_in_place`) | `cb_row_stat` (1 per call, `rows` calls) | `cb_row_stat` (in place) | `cb_row_stat` holds `1/rms` per row, value in **column 0** of each tile |
| 5 | `x · (1/rms)` | yes (`mul`, `BroadcastDim::Col`) | `cb_input_tiles` (`rows·wt_c`, popped here), `cb_row_stat` (`rows`, **not popped**) | `cb_normalized` (`rows·wt_c`) or `cb_output_tiles` when `!HAS_GAMMA` | `cb_input_tiles` released back to the reader; `cb_row_stat` retained for the remaining chunks |
| 6 | `· gamma` | yes (`mul`, `BroadcastDim::Row`) | `cb_normalized` (`rows·wt_c`), `cb_gamma_tiles` (`wt_c`, **not popped**) | `cb_output_tiles` (`rows·wt_c`) | skipped entirely when `!HAS_GAMMA` |
| 7 | untilize (RM only) | yes (`untilize<WT_CHUNK>`) | `cb_output_tiles` (`rows·wt_c`) | `cb_output_sticks` (`WT_CHUNK` per tile-row) | writer drains `cb_output_sticks` |

---

## 8. Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| 2 — `x²` | `mul(cb_input_tiles, cb_input_tiles)` | 2D `[H,W]` → **All** | 2D `[H,W]` → **All** | `BroadcastDim::None` |
| 5 — normalize | `mul(cb_input_tiles, cb_row_stat)` | 2D `[H,W]` → **All** | `REDUCE_ROW` out → **Col0** | **`BroadcastDim::Col`** |
| 6 — gamma | `mul(cb_normalized, cb_gamma_tiles)` | 2D `[H,W]` → **All** | 1D `[W]` → **Row0** | **`BroadcastDim::Row`** |

The dim names which axis is *broadcast*, not which was reduced: a `REDUCE_ROW` result is column-shaped `(N,1)` and broadcasts back across columns via `Col`; a row-shaped `1×W` operand broadcasts down rows via `Row` (`eltwise_chain.hpp:451-455`). In both broadcast phases the broadcast operand is **B** — `BinaryFpu` always applies the intra-tile broadcast to operand B and never to A (`eltwise_chain.hpp:456-458`), so the stat and gamma must be the *second* operand. `OperandKind` is orthogonal to `BroadcastDim` and selects the *tile* index: `Col` (stat, advances with the row) and `Row` (gamma, advances with the column); both require the 2-D `EltwiseShape::grid(rows, wt_c)` shape (`eltwise_chain.hpp:278-279`), and the `input()` default of `OperandKind::Scalar` would silently pin tile 0.

*(No "Reduce Direction Verification" section: this op reduces exactly one dimension — the last — and exposes no `dim` parameter.)*

---

## 9. Key Risks and Gotchas

| # | Risk | Mitigation / decision |
|---|------|-----------------------|
| **R1** | **Non-tile-aligned W must not pollute the RMS denominator.** The reduce must see only the `W` real elements, and the divisor must be `W`, not `Wt·32`. The golden suite tests this adversarially: `_PAD_POISON_SHAPES` fills the implicit tile padding with `1000.0` on narrow widths where padding is 11–38 % of the row (`feature_spec.py:448-493`), so a leak is a 6–27 % error, not a sub-noise-floor one | Two independent mechanisms, both explicit: (a) **values** — `prepare_partial_reduce_scalers<…, PARTIAL_W>` + `ReducePartialScaler::last_tile_at(1)` zero the pad lanes' contribution on the last width tile; (b) **divisor** — `INV_W = 1/W` (the *logical* W) is applied by the finalize step, never derived from `Wt·32`. Poisoning also probes the second failure mode (masking values but dividing by the padded width); (b) covers it |
| **R2** | **Non-tile-aligned H.** In TILE layout each image tile-pads H independently, so padding *rows* exist in the buffer and may be poisoned | Harmless **by construction**: the reduction is per-row, so a padding row's `Σx²` and `1/rms` are confined to that row, and its output lands in the output tensor's own padding, which the golden checker never reads. No masking needed. This is why `tag_alignment` separates `h_non_aligned` from `w_non_aligned` — genuinely different paths. In ROW_MAJOR there is no H padding at all; only `ROWS_IN_LAST_TILEROW` bounds the last tile-row |
| **R3** | **A non-finite value in the input's W padding becomes NaN for the whole row**: `inf² · 0 = NaN` inside the reduce's masked accumulation (the same hazard `toy_variance` documents at `toy_variance.py:45-52`) | Contract: pad contributions are zeroed exactly for **finite** padding (which is what `ttnn.from_torch` and `ttnn.fill_implicit_tile_padding` produce, and what the suite's `1000.0` poison is). Callers holding non-finite padding must call `ttnn.fill_implicit_tile_padding(x, 0.0)` first. If a hard guarantee is ever needed, the fix is a reader-side zero of the last width tile's pad columns — **not** a change to the blocking scheme |
| **R4** | Folding `1/W` into the bf16 reduce scaler would inject a **uniform per-row scale error** of up to ≈0.4 % (bf16 has 8 mantissa bits) — and a uniform scale error is exactly the class of bug PCC is blind to | `cb_scaler` carries exactly **1.0** (bit-exact in bf16, satisfying the bf16-scaler rule) and `1/W` is applied as an **fp32 SFPU** scalar-multiply inside `transform_in_place`. `INV_W` has one definition on the host |
| **R5** | **Sequential compute helpers cannot pipeline** — each owns all 3 TRISCs. An intermediate CB smaller than what its producing helper emits deadlocks on `cb_reserve_back` (`ttnn-cb-memory-fundamentals.md:122-155`) | `cb_x_squared` and `cb_normalized` are sized to the **full block** (`BLOCK_ROWS · WT_CHUNK`). In the ROW_MAJOR build `cb_input_tiles` (produced by `tilize`) and `cb_output_tiles` (consumed by `untilize`) are likewise full-block, which is why `CB_X_DEPTH`/`CB_OUT_DEPTH` drop to 1 there |
| **R6** | **Data that must persist across phases.** `cb_input_tiles` is read twice (pass A and pass B) in RESIDENT; `cb_row_stat` is read by every pass-B chunk; `cb_gamma_tiles` is read by every block; `cb_scaler` by every reduce | Encoded as `PopPolicy::None` + explicit `cb_pop_front` at the exact lifetime boundary (§5.1). Getting any of these wrong is a hang (`cb_wait_front`) or silent reuse of the wrong tile |
| **R7** | **Never normalize in place into `cb_input_tiles`.** It would give a reader-produced CB a second producer — silent UB that passes the push==wait check | `cb_normalized` is a separate CB. Same reasoning creates `cb_sum_handoff` in Lamp L1 |
| **R8** | **`fp32_dest_acc_en=True` halves DEST** (8→4 tiles half-sync, 16→8 full-sync) | Every compute helper here derives its live batch from `DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`). No literal DEST bound appears anywhere; `Dst` slot use is static-asserted (`eltwise_chain.hpp:399-401`) |
| **R9** | `Rt < num_cores` (every decode-shaped case, `Rt = 1`) leaves most of the grid idle in Phase 0 | Acknowledged and quantified: this is **exactly** Lamp L1's target, and `feature_spec.py`'s `_WIDE` cases and the decode perf table exist to drive it. Phase 0 carries `GRID_W` and `(w_start, w_count)` at their trivial values so enabling it adds a combine, not a rewrite |
| **R10** | STREAM's pass-B re-read makes wide `W` ≈2× the DRAM bytes of RESIDENT | Deliberate: STREAM's job is *correctness at any `W`*. The perf fix is Lamp L1 (which also fills the grid); Lamp L5 is the cheaper-but-lesser alternative |
| **R11** | The scaler CB must use the **pool-type-aware** overload; different `PoolType`/`ReduceDim` combinations need different tile fill patterns | Both call sites carry `<…, PoolType::SUM, ReduceDim::REDUCE_ROW>` template args. The caller-provided-value form (`prepare_reduce_scaler`, not `calculate_and_prepare_…`) is the sanctioned choice for a non-derived scaler (`reduce_helpers_dataflow.hpp:31-37`) |
| **R12** | Mixed input/gamma dtype and layout (bf16 activations + fp32 or bf8b TILE gamma) | Handled structurally: `cb_gamma_tiles`' `CBFormatDescriptor.data_format` is the **gamma** dtype and its page size `tile_size(gamma_dtype)`; the eltwise chain's `DataFormatReconfig::Enabled` (`eltwise_chain.hpp:319`) reconfigures the unpacker at the phase-6 boundary. Cost ≈110–150 ns per reconfig (`compute_block_size/report_reconfig_ablation.md`) — Lamp L6c elides it in the all-one-format build |
| **R13** | Bit-pattern host↔kernel transfer of `epsilon` and `INV_W` | Both packed as raw fp32 bits with `struct.unpack("I", struct.pack("f", v))[0]` and read with `__builtin_bit_cast(float, bits)` — precedent `toy_variance_program_descriptor.py:76-77`, `toy_variance/kernels/reader.cpp:42` |
| **R14** | **Two `reduce<>` template args must be pinned explicitly, and both are invisible if the `accumulate_reduce_block` wrapper is used.** (a) `ReduceFp32Mode` defaults to `Fast`, which keeps fp32 SUM on the FPU and truncates the source operands to tf32; the Phase 0 mandate is fp32 accumulation with no downcast, so `float32` input takes `Accurate`, which routes fp32 SUM through the SFPU (`reduce_helpers_common.hpp:31-53` — `Accurate` + `Float32` + `SUM` + ROW/COL only). (b) `ReduceAlgorithm` defaults to `Auto`, which today resolves to `ReduceTile` (`reduce_helpers_compute.inl:776-777`) but is not contractually pinned | Both are **explicit** template args at the A2 call site (§6), sourced from the single `REDUCE_FP32_MODE` knob in §1.4 and a literal `ReduceAlgorithm::ReduceTile`. Pinning `ReduceTile` also keeps `AccumulateReloadMode` at its default `CopySeedPairs`, which matters because **`FoldViaAdd` is documented incorrect when the accumulator CB is tagged `UnpackToDestFp32`** (`reduce_helpers_compute.hpp:160-163`) — and `cb_row_stat` is exactly an fp32 accumulator. Fallback if `Accurate` + `Accumulate` proves unsupported on device: flip `REDUCE_FP32_MODE` to `Fast` — a one-line CT change that still accumulates in fp32 DEST (that is `fp32_dest_acc_en`, a separate axis) and sits well inside the 0.999 float32 PCC gate |
| **R15** | **The RM staging helpers leave stale L1, not zeros** — `read_sticks_for_tilize` pads the L1 stride for non-tile-aligned `W` and pushes full tile pages for a partial last block, with "untouched rows contain stale data" (`tilize_helpers_dataflow.hpp:51-52`). So in the ROW_MAJOR build both the W pad lanes *and* the H pad rows of `cb_input_sticks`, and rows 1–31 of `cb_gamma_sticks`, hold whatever the previous kernel left there | Safe by three separate arguments, none of which is zero-filling: **W pad lanes** — the partial reduce scaler zeroes their *contribution* (R1a), so any finite stale value is neutralised; **H pad rows** — reductions are per-row (R2) and `write_sticks_after_untilize` writes only the valid rows (`:102-104`), so stale rows are computed and dropped; **gamma rows 1–31** — `BroadcastDim::Row` reads row 0 only. The residual exposure is identical to R3: non-finite stale L1 in the W pad lanes would give `inf²·0 = NaN` for that row. If that is ever observed, the fix is a reader-side zero of the last width tile's pad lanes via `l1_helpers.hpp` — a local addition to the reader, **not** a change to the blocking scheme |
| **R16** | **`tilize`'s fp32 mode is coupled to the input CB's `UnpackToDestMode` by a `static_assert`.** The default `Fp32Mode::Fast` static-asserts `!has_unpack_to_dest_fp32<input_dfb, /*pack_default=*/false>()` (`tilize_helpers.inl:132`), while `Lossless` requires *both* `DST_ACCUM_MODE` and `has_unpack_to_dest_fp32<...>` (`:117-120`). Since Phase 0 mandates `fp32_dest_acc_en=True`, an implementer who also sets `ComputeConfigDescriptor.unpack_to_dest_mode` for the input CB will trip the `Fast` assert at compile time | Use **`Fp32Mode::Fast`** (the default) and **do not** set `unpack_to_dest_mode` on `cb_input_sticks`. `Lossless` is wrong here on its own terms anyway: it is valid only when the tilized output is consumed exclusively by SFPU-from-DEST ops (`tilize_helpers.hpp:55-62`), and this op's output feeds an **FPU** reduce (A2) and two **FPU** broadcast multiplies (B1, B2). Related: `tilize` also asserts its input is not a block-float format (`tilize_helpers.inl:156`), which is consistent with `feature_spec.py`'s existing `{bfloat8_b, ROW_MAJOR}` `INVALID` entry — the bf8b refinement can never reach the RM tilize path |

### 9.1 Validation (Python side, before any device work)

| Condition | Exception | Message requirement |
|-----------|-----------|---------------------|
| `len(input_tensor.shape) < 2` | `ValueError` | must contain the word **`rank`** (case-insensitive) — the acceptance test matches on it via the `expect_error` fixture so CI log triage can attribute the failure |
| `gamma is not None and gamma.shape[-1] != input_tensor.shape[-1]` | `ValueError` | must contain the word **`gamma`** (case-insensitive) |
| axis value outside `SUPPORTED` | `UnsupportedAxisValue` (`ttnn/ttnn/operations/_op_contract.py:27`) |
| cell in `EXCLUSIONS` (e.g. `float32` + `fp32_dest_acc_en=False`) | `ExcludedCell` (`_op_contract.py:31`) |

`"none"` is never refused for `gamma_dtype` / `gamma_layout`.

### 9.2 Structural impossibilities (candidates for a future `/golden-tests` pass — do **not** edit `feature_spec.py`)

One candidate the current `INVALID` list does not cover, by exact analogy with its own `{dtype: bfloat8_b, alignment: w_non_aligned/h_non_aligned}` entries (`feature_spec.py:97-101`):

```python
{"gamma_dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"},
{"gamma_dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"},
```

Rationale (single-tensor coupling holds — both axes describe the weight tensor's realisation of the same shape): a `bfloat8_b` gamma shares one exponent per 16-element block, so a non-tile-aligned `W` puts gamma's pad lanes in the *same* quantization block as real weights and perturbs them. That is the identical impossibility already accepted for the activation tensor. Everything else in the TARGET cartesian is reachable by this design.

---

## 10. Hardware Constraints Checklist

- [x] Blocking model stated: every axis tagged (independent / dependent / reuse-shared) with block, core and buffer knobs named and Phase 0 values given; lamps recorded (§1)
- [x] **Phase 0 is multi-core**: `split_work_to_cores(grid, Rt, row_wise=True)` over the full compute grid; only the dependent-axis combine (L1) and the reuse-shared mcast (L2) are deferred
- [x] **No knob is a constant, each has one source**: no unconditional op-parameter-sized CB (§5 note; `cb_gamma_*` is bounded by the same L1 predicate and collapses to `WT_CHUNK` in STREAM); every block factor and depth is a §1.4 parameter with all dependents derived from it. The L1 budget itself is **derived from the device** (`ttnn.get_max_worker_l1_unreserved_size()`), not a hardcoded byte count — `L1_SAFETY_FRACTION` is the single hand-set number and every block factor is solved from it
- [x] **Block-size defaults are coarse, not minimal**: `BLOCK_ROWS = min(MAX_ROWS_PER_CORE, BLOCK_ROWS_L1_MAX)` (whole per-core assignment when it fits); `WT_CHUNK = Wt` (whole row). `BLOCK_ROWS = 1` / `WT_CHUNK < Wt` occur **only** in STREAM, whose predicate *is* the L1-pressure justification. A `HEIGHT_SHARDED` path takes the shard as the block (L3)
- [x] CB sync: push count = wait count for every CB, both regimes (§5.1)
- [x] CB ownership: exactly one producer kernel and one consumer kernel per CB per build; no in-place eltwise on a reader-fed CB (§5.2, R7)
- [x] Tile geometry alignment-aware: `div_up` everywhere, `Rt` computed **per image** for TILE and per stick-count for ROW_MAJOR (§4.1)
- [x] `>1` compute regime ⇒ regime-selection function pinned and regime-pinned tests required (§4.2)
- [x] Reduce scaler CB is `ttnn.bfloat16`, value exactly 1.0, filled through the **pool-type-aware** overloads (`<…, PoolType::SUM, ReduceDim::REDUCE_ROW>`) — R4, R11
- [x] DEST: max 8 tiles (bf16) / 4 tiles (fp32) — taken from `DEST_AUTO_LIMIT`, never hardcoded (R8)
- [x] Sequential helper intermediates (`cb_x_squared`, `cb_normalized`, and the RM `cb_input_tiles` / `cb_output_tiles`) sized to the full block (R5)
- [x] Every compute phase uses a helper, and the whole ROW_MAJOR dataflow path uses `read_sticks_for_tilize` / `write_sticks_after_untilize`; the single remaining raw-API fallback (TILE whole-tile accessor I/O) and every deliberate choice-between-helpers carries a concrete file:line justification (§6.1)
- [x] Reduce fp32 mode and algorithm are **explicit**, not defaulted: `REDUCE_FP32_MODE` (`Accurate` for float32) and `ReduceAlgorithm::ReduceTile`, which also pins the fp32-safe `AccumulateReloadMode::CopySeedPairs` (R14)
- [x] `tilize` uses `Fp32Mode::Fast` with no `unpack_to_dest_mode` on its input CB, satisfying the mode's `static_assert` under the mandated `fp32_dest_acc_en=True` (R16)
- [x] `compute_kernel_hw_startup()` called exactly once, before any helper (§6, §7)
- [x] Page sizes are tile-sized (or stick-sized where the CB counts sticks); reader on NoC0, writer on NoC1
