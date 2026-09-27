# Operation Design: rms_norm_ttnn

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (fused normalization) |
| Goal | Root-mean-square normalization over the last dimension, with an optional pre-statistics residual add and optional post-normalize per-channel scale and shift, in **exactly one** device program per call. |
| Math | `t = x + r` (optional) → `m = mean(t², dim=-1)` → `y = t * rsqrt(m + eps)` → `y *= w` (optional) → `y += b` (optional) |
| Mode | **Derivative** — extends the designated seed `ttnn/ttnn/operations/rms_norm/`. Every scheme, regime, knob, CB and kernel of the seed is preserved; this design adds capability at the seams. |
| References | Seed code (ground truth): `ttnn/ttnn/operations/rms_norm/rms_norm.py`, `.../rms_norm_program_descriptor.py` (module docstring notes D1–D28 are the maintained record), `.../kernels/{rms_norm_reader,rms_norm_compute,rms_norm_writer}.cpp`. Golden universe: `eval/golden_tests/rms_norm_ttnn/feature_spec.py`. Helpers: `ttnn/cpp/ttnn/kernel_lib/`. Conventions: `tt_metal/third_party/tt_ops_code_gen/references/precision_convention.md`. |

### What this design changes relative to the seed

The seed is preserved and extended. The table below is the complete delta; **everything not listed
is the seed's, unchanged**, and the operand-free configurations must build a byte-identical program.

| # | Addition | Where it lands |
|---|----------|----------------|
| A1 | `residual_input_tensor` — a second full-shape activation added **before** the statistics | new CBs `cb_residual_sticks` / `cb_residual_tiles` / `cb_x_sum`; new block operation `residual_add_block`; new terms in the L1 solve |
| A2 | `bias` — a per-channel shift applied **after** the scale | new CBs `cb_bias_sticks` / `cb_bias_tiles`; new block operation `bias_block`; the gamma stage becomes in-place when bias follows it |
| A3 | Ranks 0, 1, 5 (no floor, no ceiling) | host-side axis folding only; ranks 0/1 are the ROW_MAJOR path at `R = 1`, `W ∈ {1, shape[-1]}` |
| A4 | Zero-volume inputs | one degenerate device program (all cores inactive), output is an empty tensor at the requested placement |
| A5 | `program_config` consumed: two variants, validated, `subblock_w` honoured, `inplace` honoured | host-side; `subblock_w` binds the existing `PASS_B_BLK` knob; `inplace` binds `native_out` onto the input buffer |
| A6 | Two compute-config object types accepted (`ComputeConfigDescriptor`, `WormholeComputeKernelConfig`/`GrayskullComputeKernelConfig`) | one normalizing adapter at the entry point |
| A7 | Default compute config becomes HiFi4 / `math_approx_mode=True` / `fp32_dest_acc_en=False` | `default_compute_kernel_config()` |
| A8 | `{float32, fp32_dest_acc_en=False}` becomes a supported cell (the seed's only EXCLUSION is removed) | no kernel change — the cell already worked; the seed refused it on policy |
| A9 | `epsilon = 0.0` accepted (no domain check) | validation only |
| A10 | Per-channel operand shape rule becomes a **logical floor**, and the blocked ROW_MAJOR `(Wt, 32)` physical form is accepted | validation + one new reader branch |
| A11 | Host-resident input raises at the entry point | validation |
| A12 | Public surface renamed: `rms_norm_ttnn(input_tensor, *, epsilon, weight, bias, residual_input_tensor, memory_config, program_config, compute_kernel_config)` | entry point |
| A13 | A torch reference consuming **every** operand is exported alongside the op | `torch_rms_norm_ttnn()` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 0, device-resident, TILE or ROW_MAJOR, INTERLEAVED or any `*_SHARDED` | — | RT (`x_addr`) + CT (`TensorAccessorArgs`) |
| `epsilon` | `float` | no | any float, `0.0` included — **no domain check** | `1e-12` | CT (`EPS_BITS`, raw fp32 bits) |
| `weight` | `Optional[ttnn.Tensor]` | no | per-channel over the last dim; flat `(1,1,1,Wg)` at either layout, or blocked `(Wt, 32)` ROW_MAJOR only; dtype ∈ {fp32, bf16, bf8b} at **both** layouts | `None` | CT (`HAS_GAMMA`, `GAMMA_IS_RM`, `GAMMA_BLOCKED`, `GAMMA_ELEM_BYTES`, `GAMMA_TRIM`, its own `TensorAccessorArgs`) + RT (`g_addr`) |
| `bias` | `Optional[ttnn.Tensor]` | no | same rules as `weight`; must share `weight`'s **layout** when both present; dtype independent of `weight`'s | `None` | CT (`HAS_BIAS`, `BIAS_ELEM_BYTES`, `BIAS_TRIM`, its own `TensorAccessorArgs`) + RT (`b_addr`) |
| `residual_input_tensor` | `Optional[ttnn.Tensor]` | no | exact match to the input: dtype, layout, logical shape, padded shape, device, memory config (shard spec included) | `None` | CT (`HAS_RESIDUAL`, `NATIVE_RESIDUAL`, its own `TensorAccessorArgs`) + RT (`r_addr`) |
| `memory_config` | `Optional[ttnn.MemoryConfig]` | no | any placement in SUPPORTED; under `inplace` must agree with the input's | input's | host only |
| `program_config` | `Optional[object]` | no | DEFAULT variant (3 flags) or SHARDED variant (grid + blocking + 3 flags); the variant must match the input's placement | `None` → for a sharded input, synthesized from the input's shard spec with `inplace=False` | host only, except `subblock_w` → `PASS_B_BLK` (CT) |
| `compute_kernel_config` | `Optional[object]` | no | `ttnn.ComputeConfigDescriptor` **or** a device compute-kernel config; `math_fidelity` / `math_approx_mode` ungated; `fp32_dest_acc_en` ∈ {True, False} at every dtype; `packer_l1_acc` accepted and unread | `default_compute_kernel_config()` | passed as-is to `ComputeKernelDescriptor(config=…)` |

### `program_config` field contract

| Field | Variant | Treatment | Refusal |
|-------|---------|-----------|---------|
| `legacy_reduction` | both | read, **not acted upon** — this op picks its own reduction and is numerically correct at either value | never |
| `legacy_rsqrt` | both | read, **not acted upon** | never |
| `use_welford` | both | **rejected** — this op has no Welford path | `ValueError` naming the field |
| `compute_with_storage_grid_size` | sharded | range-checked against `device.compute_with_storage_grid_size()`; **not** a placement contract — placement follows the input's shard spec, so every in-range value builds the same program (X-02) | out of range |
| `subblock_w` | sharded | **honoured** — binds `PASS_B_BLK`, the pass-B DEST-lane block size | `< 1` (X-06); or does not divide `block_w`; or exceeds `DEST_AUTO_LIMIT` for the resolved `fp32_dest_acc_en`. Each refusal names both operands of the constraint |
| `block_h` / `block_w` | sharded | **checked as a restatement** of the input's shard geometry (`shard_shape[0] // 32`, `shard_shape[1] // 32`); the geometry itself is taken from the shard (X-01) | mismatch, naming both the supplied value and the shard's |
| `inplace` | sharded | when true **and** the output placement is sharded, the op returns the **input tensor object**; `cb_output_tiles` aliases the input buffer (X-03) | a `memory_config` disagreeing with the input's placement |
| variant vs. placement | both | DEFAULT + sharded input, or SHARDED + interleaved input, is refused with a message naming both (X-05); the sharded branch is never entered on an absent shard spec | mismatch |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | any rank 0…N; leading dims fold into the `row` axis. `W = shape[-1]` (rank ≥ 1) else `1`. TILE: `tensor_row_tiles = prod(shape[:-2]) * ceil_div(shape[-2], 32)`, `tensor_width_tiles = ceil_div(W, 32)`. ROW_MAJOR: `tensor_sticks = prod(shape[:-1])` (rank ≥ 1) else `1`. Any dimension `0` ⇒ zero-volume path |
| Dtype | `float32`, `bfloat16`, `bfloat8_b` |
| Layout | TILE and ROW_MAJOR, both native. No `to_layout` / `tilize` / `untilize` / `pad` / `slice` on the host |
| Memory | device-resident (host input refused, X-07); INTERLEAVED, HEIGHT/WIDTH/BLOCK_SHARDED |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to the input, at every rank including 0 |
| Dtype | input's |
| Layout | input's |
| Memory | `memory_config` if supplied, else the input's. Under `inplace` the output **is** the input tensor object |

### Optional operands

| Operand | Shape | Dtype | Layout | Placement |
|---------|-------|-------|--------|-----------|
| `weight` / `bias` | flat `(1,1,1,Wg)` with `Wg ≥ W`, **or** blocked `(Wt, 32)` ROW_MAJOR with `Wt*32 ≥ W`. At TILE layout the padded last dim equals the input's padded last dim, the logical last dim is ≥ the input's, and the padded second-to-last dim is one tile height | fp32 / bf16 / bf8b, **independent of the input's and of each other's**. Same accepted set checked at **both** layouts (X-08) | TILE or ROW_MAJOR, independent of the input's; when both are present they must **share** a layout | any (read through a `TensorAccessor`) |
| `residual_input_tensor` | exactly the input's logical **and** padded shape | exactly the input's | exactly the input's | exactly the input's, shard spec included |

## Blocking Model

The first design section. Everything below is its realization.

### Scope glossary

The seed's names are kept verbatim (a rename is divergence without benefit). Their scope:

| Name | Scope | Meaning |
|------|-------|---------|
| `Rt`, `Wt`, `R_rm`, `W` | tensor | tile-rows, width tiles, sticks, logical width of the whole tensor |
| `wt_per_core`, `row_count`, `stick_count` | core assignment | this core's share |
| `BLOCK_ROWS`, `WT_CHUNK` | block | the two block extents |
| `x_hold_wt` | block | width tiles the *held* CB spans = `wt_per_core` when `X_RESIDENT`, else `WT_CHUNK` |
| `PASS_B_BLK` | sub-block | DEST-lane block size inside pass B (a divisor of `WT_CHUNK`) |
| `GRID_W` / `group_size` | core assignment | cores per width-combine group |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `row` — all leading dims folded (batch, channel, H) | **independent** — each row is normalized in isolation; no result spans two rows | `BLOCK_ROWS` (tile-rows per block) | `min(core_row_tiles, coarsest that fits L1)`, clamped to 32 on a combine path | `_solve_blocking()` → reader/compute/writer CT arg 4 | **spread across the full grid** by `split_work_to_cores(..., row_wise=True)`; on a HEIGHT/BLOCK shard the shard already fixes it | knob-turn (raise the block, add cores) |
| `width` — the reduced last dim | **dependent** — `sum(t²)` spans the whole axis, so a cross-block combine is required | `WT_CHUNK` (width tiles per block) | `wt_per_core` (**the whole per-core width, one chunk**) whenever the working set fits; the coarsest **divisor** of `wt_per_core` that fits otherwise | `_solve_blocking()` → CT arg 2 on all three kernels | **already split across cores** in three built regimes: the AUTO interleaved width split (`GRID_W`), a WIDTH/BLOCK shard, and the RM BAND. Within a core the chunks are sequential with an fp32 carry | scheme-change **already built** (gather + finalize + multicast). Widening the group is a knob-turn on `WIDTH_SPLIT_MAX_GROUP_CORES` |
| `width` sub-block (pass B DEST lanes) | derived from `width`; not a tensor axis but a real extent | `PASS_B_BLK` | largest divisor of `WT_CHUNK` ≤ `DEST_AUTO_LIMIT`; **overridden by `program_config.subblock_w`** when supplied | `pass_b_blk(WT_CHUNK, DEST_AUTO_LIMIT)` in the compute kernel, or the host-validated `subblock_w` | not a core-assignment axis — it lives inside one core's block | knob-turn (the caller's, via `subblock_w`) |
| `channel` — the per-channel index of `weight` / `bias` | **reuse-shared** — one vector over `width` feeds *every* row, so splitting `row` across cores makes every core re-read it | spans with `width`: the operand CBs are sized `x_hold_wt` | `x_hold_wt` (held for the whole row when `X_RESIDENT`) | `x_hold_wt` (one source, shared with the x CBs) | **not assigned across cores** — each core holds the slice of the vector its `width` slice needs. Under the row split that is the whole vector on every core; under a width split it is only that core's slice | **deferred**: read once on an injector and multicast (`mcast_pipe`). See the `GAMMA_MCAST` regime row |
| `residual` | **no new axis** — carries the input's exact `row × width` geometry, so it inherits both extents. It is a second operand along both, not a third axis | `BLOCK_ROWS` × `WT_CHUNK` (same knobs) | same values | same source | same assignment as `x`, by construction (identical shard spec) | knob-turn (follows `x`) |
| `pass` (statistics pass A, normalize pass B) | **not a tensor axis** — a phase, listed so it is not silently dropped | `NUM_PASSES` = `1` when `X_RESIDENT`, `2` otherwise | derived from `X_RESIDENT`, never set directly | `_solve_blocking()` → reader CT arg 18 | no core-assignment: both passes run on the core that owns the block | knob-turn (making a shape `X_RESIDENT` removes pass 2's re-read) |
| `width-group slot` | **communication role**, not a work assignment — every member does equal reduce work; only the *fold* is unequal | `group_size` (`GRID_W`), `COMBINE_TREE_F0` | AUTO policy (see D11); `f0 = 4` | `_auto_width_split()` / `_combine_tree_arity()` | the tree already spreads the fold across `f1` cores instead of leaving it all on the root (D28), so the "one core computes while the rest wait" hazard is *already answered* for the fold. The **finalize** is one rsqrt per *round* (D27) and stays root-only: Refinement 2 built the spread, measured it a loss at every combine geometry, and parked it as a live knob — see Lamp L-FIN | knob-turn (`f0`); the finalize spread is a scheme-change, **built and measured** |

**Every axis carries a decision; no cell is blank.** The two "not a tensor axis" rows are decisions,
not omissions: `pass` is a phase count derived from residency, and `width-group slot` is a
communication role whose *work* imbalance is bounded to the finalize and named as a lamp.

#### Re-running the table on the stages this design adds

`residual_add_block` produces `cb_x_sum`, whose axes are the input's `row × width`; it is blocked at
`BLOCK_ROWS × WT_CHUNK` and assigned to the same core that owns those blocks — no new decision.
`bias_block` produces the output block; same. Neither introduces a stage that computes on one core
while the rest wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `CB_X_DEPTH` (searched over `CB_DEPTH_CANDIDATES`) | `2` on the TILE path, `1` on ROW_MAJOR (producer is a sequential tilize) | reader ↔ compute overlap on the x stream |
| `cb_residual_tiles` | `CB_R_DEPTH` = `CB_X_DEPTH` | same as x | reader ↔ compute overlap on the residual stream; deliberately tied to x so the two streams stay in step and one knob moves both |
| `cb_output_tiles` | `CB_OUT_DEPTH` (same search) | `2` / `1` | compute ↔ writer overlap |
| `cb_input_sticks`, `cb_output_sticks`, `cb_residual_sticks` | `CB_RM_STAGE_DEPTH` | `2` | reader ↔ tilize (and untilize ↔ writer) overlap on the ROW_MAJOR path |
| `cb_row_stat` | `CB_ROW_STAT_DEPTH` | `2` | **not** a perf depth — a correctness floor for the partial final row-block (D6): `transform_in_place` rotates the ring, and a ring of exactly `BLOCK_ROWS` makes a partial block's finalized tiles straddle the wrap |
| `cb_sum_handoff` | `CB_ROW_STAT_DEPTH` | `2` | the D25 combine pipeline (block `blk+1`'s pass A issued before block `blk`'s combine) |
| `cb_stat_handoff`, `cb_compact_handoff`, `cb_mcast_in`, `cb_node_out` | `CB_COMBINE_FLAT_DEPTH` | `2` | one round in flight; flat in `BLOCK_ROWS` since D27 makes the unit one compact tile |
| `cb_partials_gathered` | **not a depth** — `GATHER_SLOTS` (or `f0`) pages, one round | — | deepening it was a measured regression twice (D25) |
| `cb_x_sum`, `cb_normalized`, `cb_x_squared`, `cb_gamma_tiles`, `cb_bias_tiles` | none (depth 1) | 1 | compute-private, produced and consumed by sequential helpers on the same TRISC set — depth buys no overlap |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Compact partial transpose (D27): one tile-row's stat becomes one **column** of one tile | `BLOCK_ROWS` on a combine path | `BLOCK_ROWS = min(BLOCK_ROWS, 32)`; host `assert`, kernel `static_assert` | silently wrong results — a 101-row block measured pcc 0.949 |
| `compute_kernel_lib::tilize` / `untilize` take `block_width_tiles` as a **compile-time** template param (`tilize_helpers.hpp:187`) | `WT_CHUNK` must be one value for every core and every chunk | `WT_CHUNK` is a **divisor** of `wt_per_core` (D1) — or, since Refinement 4 (D33), the coarsest *fitting* value with the last chunk **padded** out to it, which keeps the cap satisfied verbatim; on the BAND scheme it is the **widest** tile span any core's band touches, narrower bands staging an all-zero pad tile column — D33 is that same pad moved from the shard axis to the chunk axis | a ragged tail would need a second instantiation, or would read tiles the core does not own |
| `reduce()`'s `BulkWaitBulkPop` asserts `num_pages(cb_in) % cols == 0` (`reduce_helpers_compute.inl:698`) | `WT_CHUNK` | same divisor clamp | assertion / mis-strided reduce |
| Multi-page `cb_reserve_back` + `get_write_ptr` must not straddle the ring | ring size a multiple of the push unit | same divisor clamp | silent read past the ring end |
| DEST lane capacity (`DEST_AUTO_LIMIT`, `dest_helpers.hpp:88` — 8 at `fp32_dest_acc_en=False`, **4** at True, half-sync) | `PASS_B_BLK`, `COMBINE_DEST_BATCH` | `pass_b_blk(WT_CHUNK, DEST_AUTO_LIMIT)`; a caller's `subblock_w` above it is **refused**, not clamped | DEST overflow / corrupted pack |
| DEST-resident serial accumulation depth in pass A's square fold | `WT_CHUNK` | `square_dest_acc_per_row` only when `WT_CHUNK ≤ DEST_ACC_SQUARE_MAX_WT (8)` **and** `PARTIAL_W == 0` (D12) | precision loss at 16-bit DEST; and with `PARTIAL_W != 0` the fold folds pad lanes in before the mask can reach them |
| An unaligned DRAM source offset is **silently truncated** down to the alignment | the byte offset of every per-channel and band read | stage in the tensor's **global tile frame** so every fetch sits on a tile column, a multiple of 64 B for every dtype (D10) | measured whole-tensor pcc 0.32 with a spot-check of band 0 reading perfect |
| `bfloat8_b`'s 1088-byte tile has a 272-byte face — not 64-B aligned | `GAMMA_TRIM` / `BIAS_TRIM` | demote face-row trim (2) to a half-page read (1) for bf8b (D23) | truncated read |
| **In-place eltwise chain** requires an *incrementally popping* input and an *incrementally reserving* output (`inplace_chain.cpp:5-21`); a `Row`/`Col` operand may never be the aliased CB (`chain.inl:82-85`) | the gamma stage's srcA when `HAS_BIAS`, and only there | gamma-in-place uses `input(cb_normalized, PerBlockSize, PerBlockSize, Block)` + `output(cb_normalized, PerBlockSize, PerBlockSize)`; **the held x CB is never aliased** (it is `Upfront`/`None`) | CB self-deadlock (hang), not a wrong answer |
| `cb_scaler` value is exactly `1.0` in bfloat16; `1/W` is applied in fp32 by the finalize | — | never fold `1/W` into the scaler | a bf16-quantized reciprocal of the width |

### Regimes

`Wt_c` = this core's width tiles. `budget` = `L1_SAFETY_FRACTION * (usable L1 − resident shards − arena base reserve)`.
`working_set(BR, WC, depth, present operands)` is the closed-form footprint in `l1_ledger.md`.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **ROWS · RESIDENT** | **built** | interleaved (or a HEIGHT shard) **and** `working_set(BR≥1, WT_CHUNK=Wt_c, depth, operands) ≤ budget` at some depth in `CB_DEPTH_CANDIDATES` | `BLOCK_ROWS × Wt_c` | **minimum**: x once, residual once, output once across DRAM. Per-channel operands cross `num_cores` times (reuse-shared, held per core) | amortizes: pass-A/B init pairs, the tilize/untilize template instantiation's per-block setup, the reduce's `Accumulate` seed/finalize, one CB handshake per block instead of per tile-row. Intended: **one init set + one barrier per block per phase** |
| **ROWS · ROW_RESIDENT** (Lamp L5 / D14) | **built** | RESIDENT does not fit, **and** a divisor `WT_CHUNK < Wt_c` exists with the *whole* tile-row of x (and of `cb_x_sum` when residual is present) plus the whole row of each per-channel operand held; gated on `ROW_RESIDENT_MIN_ROWS_PER_CORE` when it forces a shallower depth | `1 × WT_CHUNK`, with x/x_sum/gamma/bias **held at `Wt_c`** across both passes | above minimum by **nothing** on x, residual and output; per-channel operands stay at `num_cores`. Measured 470 MB → 285 MB against STREAM on `(1,1,8192,7168)` | same fixed costs, plus: it removes STREAM's whole pass-B re-read of x **and of the residual** — with a residual present that saving doubles, which is the single biggest reason this regime matters more here than in the seed |
| **ROWS · STREAM** | **built** | neither above fits | `1 × WT_CHUNK` (coarsest divisor that fits), nothing held | **above minimum**: x crosses DRAM **twice**, residual **twice**, and each per-channel operand `num_cores × num_blocks × NUM_W_CHUNKS` times. Structurally unavoidable at this footprint — there is no L1 to hold anything | as above; a bigger `WT_CHUNK` directly divides the per-channel re-read count |
| **ROWS + WIDTH SPLIT** (Lamp L1 / D11) | **built** | interleaved, TILE, `Wt > 1`, and `_auto_width_split` finds `(gw, gh)` with `gw` a divisor of `Wt`, `Wt/gw ≥ WIDTH_SPLIT_MIN_WT_PER_CORE`, `gw ≤ WIDTH_SPLIT_MAX_GROUP_CORES`, and `gw*gh ≥ WIDTH_SPLIT_MIN_GAIN × row_cores` | `BLOCK_ROWS × (Wt/gw)`, one chunk | x / residual / output at **minimum**; per-channel operands drop from `num_cores` crossings to `gh` (each core reads only its slice) — the split *reduces* reuse-shared traffic. Adds cross-core: one compact fp32 tile per member per round up the tree, one multicast tile down | as RESIDENT, plus it amortizes the **combine round** — one gather + fold + finalize + multicast per block instead of per tile-row |
| **SHARD_H · native** | **built** | TILE input, HEIGHT_SHARDED, `shard_w_t == Wt` | the shard: `shard_h_t × Wt`, **one block** unless L1 forces a cut | x and output **never cross DRAM** (zero-copy CBs over the resident shards); residual likewise; the reduce is entirely core-local | one block = one pass over the resident shard, no per-chunk re-init |
| **SHARD_W / SHARD_B · native + combine** | **built** | TILE input, WIDTH or BLOCK_SHARDED, not (`ragged_w` and `PARTIAL_W`) | the shard: `shard_h_t × shard_w_t`, one chunk (`NUM_W_CHUNKS == 1` is asserted) | x / residual / output resident, zero DRAM. Adds the same tree gather + multicast as the width split | amortizes the combine round over more tile-rows per round |
| **BAND** (D10) | **built** | ROW_MAJOR input, WIDTH or BLOCK_SHARDED | the shard band, staged into the tensor's **global tile frame**; `WT_CHUNK` = the widest band span in the group | x / residual staged from the core's own L1 (no DRAM, no NoC); output written back locally when the out shard matches | one staged read per tile-row instead of per stick when the band fills its tile columns and the shard stride matches |
| **DEGENERATE_RANK** (rank ≤ 1) | **built** | `rank ≤ 1` | `1 × 1` — one stick of `W ∈ {1, shape[-1]}` elements, ROW_MAJOR | one stick in, one stick out — minimum. Rank 0 uses `W = 1`, `INV_W = 1.0`, `PARTIAL_W = 1`, so `mean(t²) = t²` and `t / sqrt(t² + eps)` falls out of the same finalize | nothing — the block is the whole tensor. This is a **parameterization of ROWS·RESIDENT**, not a separate scheme, which is why it is one row and not a fourth kernel path |
| **ZERO_VOLUME** | **built** | any dimension `== 0` | none — every core is `row_count == 0` and returns immediately | zero crossings. Still **one dispatch** (the requirement is one device program per invocation, not one per element) | nothing |
| **GAMMA_MCAST** — read the per-channel vectors once on an injector core and multicast them to the row-split group | **BUILT AND MEASURED IN AN ISOLATED BENCH (Perf 1); NOT GRADUATED — the deferral is now a number, not an argument.** Against a same-build control it is **+2.2% to +7.4% on the row-split interleaved prefill** (the boot per-channel read collapses 34.4 µs → 5.7 µs per core, a 6× cut, which converts to only ~5% of the op because the DRAM time it frees is immediately re-absorbed by `reader_read_x` — those shapes are DRAM-throughput-bound end to end, exactly as this row argued). It is a **measured regression on BLOCK shards** (−1.6% / −2.3%: a 4-tile per-core slice is latency-bound tiny reads, and the mcast adds a handshake plus an 8-of-64-core boot straggler), **structurally inapplicable on a WIDTH shard** (one tile-row ⇒ every core owns a disjoint slice ⇒ no reuse at all), and **unbuilt on STREAM**, which carries the largest remaining prize in the whole op (+20.1% ablation ceiling) but needs the mcast round count padded to the column's max because it is a per-core quantity. Not graduated in Perf 1 for one honest reason: the candidate integration carries an unattributed same-build overhead that regresses non-engaged plans ~4–5% against the shipped baseline, i.e. the off path is not yet byte-identical. Requeued. Original deferral rationale, unchanged:  the reuse-shared traffic it removes is `num_cores × Wt × tile_bytes`, which is second-order at every measured shape (on `(1,1,8192,7168)` gamma is 118 MB against 234 MB of x+out, and ROW_RESIDENT already cuts it to 50 MB); and the built **width split** removes it structurally wherever the grid is under-filled, which is exactly where it would bite hardest. Reachable: the group topology, the semaphore allocation and `mcast_pipe` are already in the program for the combine, and the per-channel CBs are already boot-filled once per core — the change is *who* fills them | would apply when `row_cores > 1` and the operands are held (`X_RESIDENT`) | unchanged | would take per-channel crossings from `num_cores` to `1`, adding one multicast of `Wt` tiles per operand | — |
| **RAGGED WIDTH CHUNK** | **built** (Refinement 4, descriptor D33) — and built *without* the runtime `wt_c` this row proposed, which is why no helper changed. `_width_chunk` takes the coarsest **balanced** chunk `ceil(wt_c / ceil(wt_c / cap))` and **pads** the last one out to it, so the chunk stays UNIFORM and all three `Mechanism caps` rows above hold verbatim: `tilize`/`untilize` keep their compile-time `block_width_tiles`, the reduce's `num_pages % cols == 0` still holds, and every ring is still a multiple of the push unit. The pad tiles are the ragged-**shard** pad this op already had, moved one axis over: the reader zeroes them (device zero API, `publish_native_shard`'s mechanism) so they add exactly 0 to `sum(t²)`, and the writer skips them with the `wt < WT` predicate it already carried | `WT_CHUNK` is now the coarsest **fitting** value, not the coarsest fitting **divisor**. `wt_per_core` is replaced by the PADDED width `NUM_W_CHUNKS × WT_CHUNK` wherever a held CB spans the row. Gated on `PARTIAL_W == 0` — D1 still governs a non-tile-aligned width, where the reduce's partial scaler is aimed at the block's last tile | unchanged | unchanged — the pad is read from nowhere and written nowhere (pad < `NUM_W_CHUNKS`, i.e. ONE tile in 128 at `Wt = 127`) | **1.34–10.46×** on the prime-`Wt` resilience shapes and 1.05× on `(1,1,1024,16384)` STREAM, at *better* pcc (a coarse chunk clears D7/D8's reduce-datapath floors). The compute kernel is byte-for-byte unchanged |
| **WELFORD single-pass statistics** | **rejected** — the contract requires `use_welford` to raise. Superseded by the two-pass reduce, which is already at the DRAM roofline on the prefill profiles | — | — | — | — |
| **3+ level gather tree** | **rejected** — measured a loss at 6 of 7 cells in the isolated bench; superseded by the shipped 2-level tree (D28) | — | — | — | — |
| **Fuse the residual add into pass B's normalize chain** (no `cb_x_sum`) | **rejected** — inexpressible: a second `BinaryFpu` reads CBs, not DEST, and `DestReuseBinary` carries no broadcast parameter (`chain.hpp:526`), so `(x+r) * stat<Col>` cannot share one DEST window. Superseded by the `residual_add_block` + held `cb_x_sum` structure, which additionally makes the sum available to **both** passes for one materialization | — | — | — |
| **Re-read x and r in pass B and re-add, holding nothing** | **rejected as a distinct regime** — it *is* STREAM, and STREAM is built. Listed so it is not re-proposed as an L1 saving: the saving is `cb_x_sum`, the cost is a second full DRAM crossing of **two** activations | — | — | — |

**Selection predicate.** `_plan_placement()` picks the scheme from layout × `memory_layout` × shard
geometry; `_solve_blocking()` then picks RESIDENT → ROW_RESIDENT → STREAM in that order, returning
`None` to force a re-plan on `SCHEME_ROWS` when a shard-derived width cannot be chunked. Both are
pure functions of (shape, dtypes, layouts, placement, operand presence, `fp32_dest_acc_en`,
device L1 size, grid size) — **device-independent given the grid**, so a regime is reproducible.
**Regime-pinned tests are required**: `feature_spec.LOOSE_CASES` already pins RESIDENT (the small
`INPUTS`), ROW_RESIDENT / STREAM (the wide prefill perf cases), the width split (`_WIDE`), all four
placements (`_SHARDED`), BAND (the ROW_MAJOR × `*_SHARDED` resilience cells), and the degenerate
ranks — the acceptance test adds a direct pin on each operand combination.

### Traffic ranking

Candidate splits, ranked by aggregate movement for the shape class each serves. Bytes, not
nanoseconds. `C` = cores at work, `T` = tile bytes.

| Rank | Split | DRAM crossings (x, r, out) | Per-channel DRAM crossings | Cross-core | When it is the right answer |
|------|-------|---------------------------|---------------------------|-----------|-----------------------------|
| 1 | **`row` across the grid, `width` whole per core** (independent axis, no combine) | 1, 1, 1 | `num_cores × Wt` tiles | none | `Rt ≥ num_cores` — the prefill profile. Chosen wherever it fills the grid |
| 2 | **`row` across the grid + `width` across a group** (dependent axis + combine) | 1, 1, 1 | `gh × Wt` tiles — **strictly fewer** than rank 1 | `group_size` compact fp32 tiles up the tree + 1 multicast tile down, **per row-block** | `Rt < num_cores` — the decode profile. Measured 3.24× on `(1,1,32,7168)`. Also the *residency* mechanism: cutting the reduced extent is what lets a 16 384-wide row fit L1 at all |
| 3 | **shard-pinned** (`row` for HEIGHT, `width` for WIDTH/BLOCK) | 0, 0, 0 — resident | `num_cores × Wt` (HEIGHT) or `gh × shard_w_t` (WIDTH/BLOCK) | as rank 2 for WIDTH/BLOCK | the caller already placed the data; the split is not ours to choose, only the block within it |
| 4 | **`row` on one core, `width` whole** (no split) | 1, 1, 1 | `Wt` | none | never chosen — it moves the fewest total bytes but uses one core; the traffic ranking does not price occupancy, and the grid-fill step (Rule 2, step 1) rules it out before the extent is chosen |
| 5 | **`width` chunked within a core, nothing held** (STREAM) | **2, 2, 1** | `num_cores × num_blocks × NUM_W_CHUNKS × WT_CHUNK` | none | only when nothing else fits — it is the fallback, not a choice |

**Chosen:** rank 1 where the row axis fills the grid, rank 2 where it does not (AUTO policy), rank 3
where the caller pinned it. The cheapest-traffic split (rank 2, which beats rank 1 on the
reuse-shared operands) is **implemented**, and the `WIDTH_SPLIT_MIN_GAIN = 4` gate keeps it off the
shapes where the combine's rendezvous is not paid for — a gate that was measured, not assumed
(`MIN_GAIN = 2` produced a 0.92× regression on `(1024,1024)`).

**The residual changes the ranking's magnitude, not its order.** Every activation crossing above
doubles, so rank 5's penalty doubles and the gap between ROW_RESIDENT and STREAM roughly doubles
with it. That is the whole mechanism by which "an operand changes what the budget has to cover, so
the thresholds move": the same shape that was RESIDENT without a residual can land in
ROW_RESIDENT with one, and the L1 solve — which now counts `cb_residual_tiles` and `cb_x_sum` —
re-decides it without a new code path.

**Stall-shadow check.** Three stages in this scheme WAIT on something they do not produce:

| Stage that waits | What runs in the shadow | Status |
|------------------|------------------------|--------|
| the root waits for the group's partials to arrive | block `blk+1`'s **pass A** is issued before block `blk`'s combine (D25). Legal because pass A of a later block is independent of the current block's stat — an *ordering* property of the two-pass algorithm, not something a profile could authorize | **built**, carved out to `native_in` (on a reader-fed input ring a tile offset cannot cross a ring wrap) |
| the members wait for the root's multicast | the root publishes its **own** stat copy before broadcasting (D24), so its pass B does not wait out its own multicast | **built** |
| **compute's pass A waits for the reader's first hand-off of x** | since Perf 1 / D35, the reader's ENTIRE boot — the scaler synthesis, the one-hot bank, and the per-channel operand's DRAM read — runs in that shadow instead of ahead of it. The publish is a CB hand-off of already-resident L1, so it has no dependency on any of them; the per-channel read (~1 µs, and the largest single source of cross-core skew at 541 ns) is not needed until pass B, ~3 µs later | **built (Perf 1)**. The price is that `cb_scaler` and `cb_bank` lost their implicit ordering and now need EXPLICIT one-shot fronts at first use in the compute kernel — see D35 for the non-deterministic corruption that appears without them |
| every core waits for the root's **finalize** (one rsqrt per *round* since D27, not per tile-row) | nothing is independent of it — pass B's first operand *is* the finalized stat | **BUILT AND MEASURED (Refinement 2), and the spread LOSES — parked as a live knob.** `COMBINE_FIN_SPREAD` forwards the RAW group sum and finalizes on every core in parallel; it measured 0.953–1.004x, i.e. a loss at every combine geometry. The reason is structural: the finalize is a **replicated** term, not a divisible one — every core needs the *same* value, so relocating the rsqrt moves it along the identical serial chain (fold → [rsqrt] → send → recv → [rsqrt] → pass B) instead of dividing it. D22 also fused the root's rsqrt into the fold's DEST window (no pack at all), while a spread one needs its own copy+pack+unpack per core; and **D27 already removed the expensive half** — the compact tile makes the finalize ONE tile-op per round whatever `BLOCK_ROWS` is, where the lamp was written against an O(`BLOCK_ROWS`) cost. D15's scoped rsqrt remains the built mitigation |

### Block schedule

Named block operations. This is the **logical** schedule; reader, compute and writer realize their
parts asynchronously and adjacent blocks pipeline.

```cpp
// ---- boot, once per core ----
stage_scaler();                                  // reader: 1.0 scaler pair, or the 0/1 mask tile
stage_permutation_bank();                        // reader: COMPACT combine only, BLOCK_ROWS one-hot pages
stage_gamma_row();                               // reader: HAS_GAMMA && X_RESIDENT
stage_bias_row();                                // reader: HAS_BIAS   && X_RESIDENT
tilize_per_channel_rows();                       // compute: PER_CHANNEL_IS_RM && X_RESIDENT

for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {

    // ---- pass A: statistics over t = x + r ----
    for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
        load_x_block(block_idx, c);              // reader  (elided when NATIVE_IN)
        load_residual_block(block_idx, c);       // reader  (HAS_RESIDUAL; elided when NATIVE_RESIDUAL)
        tilize_x_block(block_idx, c);            // compute (ROW_MAJOR only)
        residual_add_block(block_idx, c);        // compute (HAS_RESIDUAL)  -> cb_x_sum
        square_block(block_idx, c);              // compute
        reduce_accumulate_block(block_idx, c);   // compute
    }

    // ---- the statistic becomes a per-row rsqrt ----
    combine_block(block_idx);                    // writer+compute, cross-core schemes only
    finalize_stat_block(block_idx);              // compute, local schemes only

    // ---- pass B: normalize, scale, shift ----
    for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
        reload_x_block(block_idx, c);            // reader  (!X_RESIDENT)
        reload_residual_block(block_idx, c);     // reader  (!X_RESIDENT && HAS_RESIDUAL)
        residual_add_block(block_idx, c);        // compute (!X_RESIDENT && HAS_RESIDUAL)
        normalize_block(block_idx, c);           // compute
        scale_block(block_idx, c);               // compute (HAS_GAMMA)
        bias_block(block_idx, c);                // compute (HAS_BIAS)
        untilize_out_block(block_idx, c);        // compute (ROW_MAJOR only)
        store_block(block_idx, c);               // writer  (elided when NATIVE_OUT)
    }
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `stage_scaler` | 1–2 tiles | `cb_scaler` for the whole kernel | **once per kernel** |
| `stage_permutation_bank` | `BLOCK_ROWS` tiles | `cb_bank` for the whole kernel, never popped | **once per kernel** |
| `stage_gamma_row` / `stage_bias_row` | `x_hold_wt` tiles each | the operand CBs for the whole kernel when `X_RESIDENT` | **once per kernel** when `X_RESIDENT`; once per pass-B chunk in STREAM |
| `tilize_per_channel_rows` | `1 × WT_CHUNK`, `NUM_W_CHUNKS` times | as above | one tilize init per kernel (`init_uninit_mode` amortizes across the back-to-back calls) |
| `load_x_block` / `load_residual_block` | `BLOCK_ROWS × WT_CHUNK` | — | **one NoC barrier per (block, chunk) per stream** on the TILE path; both streams share the barrier since it fences NoC-0 globally. On the ROW_MAJOR path each stream costs its own barrier (`read_sticks_for_tilize` owns its reserve/push) |
| `tilize_x_block` | `BLOCK_ROWS × WT_CHUNK` | — | one init per kernel; `block_width_tiles` is compile-time |
| `residual_add_block` | `BLOCK_ROWS × WT_CHUNK` | writes into `cb_x_sum`, which is **held at `x_hold_wt`** when `X_RESIDENT` | one binary-FPU init per block, one dst-sync window per `PASS_B_BLK` group |
| `square_block` | `BLOCK_ROWS × WT_CHUNK` | `cb_x_sum` (or `cb_input_tiles`) | one init per block; with the DEST fold, one pack per **tile-row** instead of per tile |
| `reduce_accumulate_block` | `BLOCK_ROWS × X_SQUARED_WT` | the fp32 carry in `cb_row_stat` / `cb_sum_handoff` across chunks | one reduce init per block; the cross-chunk `Accumulate` seed once per block, `at_last` once |
| `combine_block` | one compact fp32 tile per member | the gather ring for one round | **one gather + one fold + one finalize + one multicast per block** — never per tile-row (that is what D27's compact transpose bought) |
| `finalize_stat_block` | `BLOCK_ROWS` tiles | — | one `rsqrt_tile_init` per block |
| `normalize_block` | `BLOCK_ROWS × WT_CHUNK`, DEST-blocked at `PASS_B_BLK` | the held x/x_sum CB and the stat | one init per block, one dst-sync window per `PASS_B_BLK` group |
| `scale_block` | as above | the held `cb_gamma_tiles` | as above. **Not fused with `normalize_block`** — measured 0.84× (`changelog.md:881`), and **re-measured at a different geometry and by a different mechanism in Perf 1**: 0.972× on the 28-core decode WIDTH shard and 0.818× on the 64-core BLOCK shard, built on the raw `DEST (op) bcast(CB)` path rather than a helper chain. The lamp is closed by MECHANISM now, not by one number — pass B's three compute threads are already balanced (unpack 8520 / math 8556 / pack 8110 ns per core), so the unpack and pack the fusion deletes were never the constraint; and `eltwise_binary_run_with_dest_reuse` restarts the MOP **per face** with a `move_d2a_fixed_face` + `TT_ZEROACC`, making the DEST-reuse mul ~71 ns/tile against the helper's one-MOP bcast mul at ~32 ns/tile |
| `bias_block` | as above | the held `cb_bias_tiles` | as above |
| `untilize_out_block` | `BLOCK_ROWS × WT_CHUNK` | — | one init per kernel |
| `store_block` | `BLOCK_ROWS × WT_CHUNK` | — | one NoC barrier per (block, chunk); interleaved with the combine rounds, never batched to the end (batching deadlocks once `num_blocks` exceeds the output CB depth) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| ~~**L-RES-FUSE**~~ **CLOSED (Refinement 3)** — `residual_add_block` materializes `cb_x_sum` in both passes | The lamp's own four-element chain (`Add → PackTile(cb_x_sum) → Square → PackTile(cb_x_squared)`) is **structurally impossible**, and it fails SILENTLY: in `eltwise_chain` **pack is its own cohort**, disjoint from math-MOP/SFPU (`chain.inl elem_pack_init`), so every pack runs after every compute element and `cb_x_sum` receives the SQUARE. Built and measured at **pcc 0.260** on `(1,1,8192,5120)` ROW_RESIDENT `gamma_bias_residual` (and 0.947x). Publishing `t` and squaring it in one DEST window needs a DEST→DEST copy element the chain does not expose — a helper gap, recorded, not worked around with raw LLK | **Done: built, measured, parked.** The STREAM-only framing was right — `t` need not survive only where pass B rebuilds it — and the correct three-element form (`Add → Square → PackTile`) is shipped as the `RES_FUSE` knob at its byte-identical default: correct (pcc 0.999980) and **0.989x** on `(1,1,1024,16384)` STREAM `gamma_bias_residual`, because the SFPU `square_tile` costs more than the saved unpack and pack |
| **L-RES-DEPTH** — `CB_R_DEPTH` is tied to `CB_X_DEPTH` | Two double-buffered streams cost two extra blocks of L1 at RESIDENT, which can be what pushes a shape into ROW_RESIDENT. Depth 1 on the residual alone may buy back a coarser block | `CB_R_DEPTH ∈ {1, 2}` at fixed `CB_X_DEPTH = 2`, on the block-sharded `gamma_bias_residual` perf case (`(7168,1024)`, 112 tiles/core) |
| **L-BIAS-INPLACE** — the gamma stage transforms `cb_normalized` in place when bias follows | In-place forces `PerBlockSize` policies where the seed's non-bias path uses `Upfront`/`AtEnd`, and the front rotates one full revolution per block. If that costs more than the block CB it saves, a dedicated `cb_scaled` is better | in-place gamma vs. a third block CB `cb_scaled`, on `(8192,5120)` `gamma_bias` interleaved |
| ~~**L-FIN**~~ **CLOSED (Refinement 2)** — the finalize is root-only and `group_size` cores wait on it | D15 established that a per-tile SFPU cost invisible in a tile-op count *dominated* the sharded geometries. The tree spread the fold but not the finalize | **Done: built, measured, parked.** The spread measured 0.953–1.004x (loss) at every combine geometry — see the stall-shadow row above for why a *replicated* term cannot be spread, and why D27 had already deleted the O(`BLOCK_ROWS`) half of this lamp. What the phase DID buy is on the same round's **transport**: the stat multicast's pre-handshake is elided on a single-round combine and its identity-path payload is trimmed to faces 0..2, together 1.035x on the op's worst combine cell |
| ~~**L-OVERLAP**~~ **CLOSED (Perf 3, D42)** — the block is now PICKED, not inherited from what fits | The lamp was right that the coarsest fitting block can leave a core with a SINGLE row-block and nothing to pipeline, and D41 (Perf 2) reached for it with the wrong lever — it raised the block count only by DEEPENING the ring, and at each depth still took the LARGEST block that fits, so the search never offered itself the finest split at all | **Done: built, measured, shipped, and the lamp's own framing was half-wrong.** The lamp said *a depth sacrifice is free while a coarse block buys nothing*; the measurement says **depth is irrelevant** once the block is picked directly (82,570–83,153 ns across depths 2/3/4/5/6/8 on `(1,1,8192,1024)`, a 0.7% band), so D41's second ladder is deleted and its L1 comes back. What the lamp missed is that the rule is **REGIME-SPLIT**: a core that READS x over the NoC wants ONE tile-row (the blocks *are* the pipeline — 1.011x on `(1,1,8192,1024)`, 1.015x on the masked `(1,1,8192,1000)`, flat where a core already holds many blocks), while a core whose x is a **zero-copy resident shard** has no read to overlap and wants the coarsest block — `br 16 → 1` there is **0.50x** on the `(1,1,8192,1024)` BLOCK shard and 0.42x on `(1,1,7168,1024)` gbr. What a native shard *can* take is the BALANCED block at the SAME block count (20+12 → 16+16), **1.073x**. The un-picked half — `Rt = 1`, where the core has one block and no finer split exists — stays with `ROW_RESIDENT_MIN_ROWS_PER_CORE`, unchanged |
| ~~**L-SUBBLOCK**~~ **CLOSED (Perf 1, D38)** — the default is now the largest divisor at `BLOCK_ROWS > 1` and the **smallest divisor ≥ 2** at `BLOCK_ROWS == 1`. The lamp was right that one rule cannot serve both: a many-tile-row block is throughput-bound and wants the whole DEST window, a one-tile-row block is the tail after the multicast and wants the packer to overlap the math. Measured 1.02x–1.06x where the small rule applies and 0.958x/0.976x where it does not, which is what earns the carve-out. `subblock_w` still overrides both, unclamped | The perf spec records `subblock_w = block_w` beating `subblock_w = 1` by 1.12× on the 64-core BLOCK shard — but only together with `inplace`, because a separate output shard makes the buffers clash. The interaction is real and the default does not know about it | `PASS_B_BLK` at 1 vs. `block_w`, with and without `inplace`, on `(8192,1024)` BLOCK-sharded |
| ~~**L-OPERAND-TRIM**~~ **CLOSED (Refinement 3)** — `BIAS_TRIM` copies `GAMMA_TRIM`'s policy | The hypothesis was that two trimmed reads per chunk instead of one changes the transaction count, which D13 found actually mattered | **Done: swept, and the hypothesis is REFUTED with a number.** Each operand got its own override (`PER_CHANNEL_TRIM_GAMMA` / `_BIAS`), filtered through the same legality rule so a forced granularity can never truncate a block-float read. Everything coarser than D23's two-face-row form LOSES: half page 0.93–1.00x, whole tile **0.76–0.96x**, `gamma=2 / bias=0` 0.85–1.00x, across the prefill and the sharded combine geometries alike. The trim is a BYTE-count win, not a transaction-count one; the bias copying gamma's policy — derived from its OWN tile size — is the measured optimum, and both ship derived |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x: DRAM/L1 → `cb_input_sticks` | RM sticks, input dtype | `read_sticks_for_tilize<cb_input_sticks>` (`tilize_helpers_dataflow.hpp:88`), or `stage_band` from the core's own L1 | ROW_MAJOR path. Ring boot-zeroed when `STAGE_ZERO` |
| x: DRAM → `cb_input_tiles` | tiles, input dtype | `noc_async_read_tile` per width tile, one barrier per chunk | TILE interleaved path |
| x: resident shard → `cb_input_tiles` | tiles / sticks | **zero-copy** `ttnn.cb_descriptor_from_sharded_tensor` — 0 arena bytes, **no NoC re-read** | every `native_in` scheme |
| residual: → `cb_residual_sticks` / `cb_residual_tiles` | identical to x, by contract | **the same three mechanisms**, on the same schedule. On the TILE path both streams are issued inside one width-tile loop and covered by the **one** shared `noc_async_read_barrier()`; on the RM path each costs its own barrier because the helper owns its reserve/push | Under `native_in` the residual is **also** zero-copy over its own shard — it carries the identical shard spec, so it is already in this core's L1 and must not cross the NoC |
| per-channel: DRAM → `cb_gamma_sticks` / `cb_bias_sticks` | one staged stick per chunk | flat form: one wide read at byte offset `first_wt * 32 * elem`. **Blocked form**: `WT_CHUNK` page reads of `32 * elem` bytes each, page `first_wt + w`, landing at byte offset `w * 32 * elem` inside the *same* staged stick | ROW_MAJOR path. Both forms produce byte-identical staging, which is why only the reader can tell them apart |
| per-channel: DRAM → `cb_gamma_tiles` / `cb_bias_tiles` | tiles, operand dtype | `noc_async_read` at the trim granularity (`GAMMA_TRIM` / `BIAS_TRIM`), tile id clamped to `Wt-1` for pad columns | TILE path |
| RM → tiles | `cb_*_sticks` → `cb_*_tiles` | `compute_kernel_lib::tilize<WT_CHUNK, in, out>` | in-kernel; **never** `ttnn.to_layout` |
| compute → compute | tiles | `cb_x_sum`, `cb_x_squared`, `cb_normalized`, `cb_row_stat` | all compute-private |
| member → gatherer | one compact fp32 tile | `noc_async_write` into the gatherer's `cb_partials_gathered` slot + `Semaphore::up`, one arrival semaphore **per tree level** | writer owns it (NoC-1 is idle through pass A) |
| root → group | one compact fp32 tile | `mcast_pipe` `SenderPipe::send` / `ReceiverPipe::receive` (`mcast_pipe.hpp:154,219`), `Mcast1D::PerRow` or `Mcast2D` over the bounding box | root publishes its own copy *before* sending (D24) |
| tiles → RM | `cb_output_tiles` → `cb_output_sticks` | `compute_kernel_lib::untilize<WT_CHUNK, in, out>` | in-kernel |
| out: → DRAM / L1 | tiles or sticks | `noc_async_write_tile`, `write_sticks_after_untilize`, `write_band`, or **nothing** when `native_out` | under `inplace`, `native_out` aliases the **input** buffer |

**Physical shards are consumed in place.** Every `*_SHARDED` value of the `memory_layout` TARGET
axis is a placement of a logical split this design already makes, so adding one is a CB/placement
change and not a new algorithm:

| Placement | Which axis the shard cuts | Character of that axis | Class |
|-----------|--------------------------|------------------------|-------|
| HEIGHT_SHARDED | `row` | independent | **knob-turn** — the shard *is* the per-core block, the reduce stays local, CBs go zero-copy |
| WIDTH_SHARDED | `width` | dependent | **scheme-change**, already built — per-core partials gathered to a group root, finalized, multicast back |
| BLOCK_SHARDED | both | `row` independent, `width` dependent | **scheme-change**, already built — one group per grid row |
| any, ROW_MAJOR, cutting `width` | `width` with a sub-tile edge | dependent | **scheme-change**, already built (BAND) — the combine sums per-row partials **elementwise**, so a partial may cover any contiguous element range and no core needs a whole width tile |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one **block** = `BLOCK_ROWS` tile-rows × `WT_CHUNK` width tiles (× 1 for each present activation operand) |
| Grid | `device.compute_with_storage_grid_size()` — a **runtime** quantity, never inlined. On a sharded input, the shard's own core grid |
| Per-core work | `SCHEME_ROWS`: `split_work_to_cores(Rt_or_groups, grid, row_wise=True)`, so `row_count ∈ {⌈Rt/C⌉, ⌊Rt/C⌋}`. With a width split, the row axis is first cut into `gh` row-groups and each group's rows are spread over `gw` cores that each own `Wt/gw` width tiles. Sharded: exactly the shard |
| Remainder | explicit. `row_count == 0` marks an **inactive** core, which still joins the program so the multicast box lands in a CB this program owns. A core's last row-block is partial whenever `BLOCK_ROWS ∤ row_count`; `CB_ROW_STAT_DEPTH = 2` is the correctness floor that makes the partial block's ring rotation contiguous (D6) |
| Tile geometry | **alignment-aware from the start**: `tensor_row_tiles = prod(shape[:-2]) * ceil_div(shape[-2], 32)` — per-image `ceil`, never `floor(prod(shape[:-1])/32)`, because each image is tile-padded independently. `tensor_width_tiles = ceil_div(W, 32)`. `PARTIAL_W = W % 32` |
| Regime selection | `_plan_placement()` then `_solve_blocking()`, both pure functions of shape × dtypes × layouts × placement × **operand presence** × `fp32_dest_acc_en` × grid × L1 size. **Regime-pinned tests required** (see the Regimes section) |

### Rank folding

| Rank | `W` | `R` (row extent) | Path |
|------|-----|------------------|------|
| 0 | `1` | `1` | ROW_MAJOR, `PARTIAL_W = 1`, `INV_W = 1.0` → `t / sqrt(t² + eps)`; epsilon stays in the denominator so a zero scalar returns **zero**, not NaN |
| 1 | `shape[-1]` | `1` | ROW_MAJOR, one stick |
| 2…N | `shape[-1]` | TILE: `prod(shape[:-2]) * ceil_div(shape[-2],32)`; RM: `prod(shape[:-1])` | the ordinary paths; rank 5 is rank 4 with one more leading factor |
| any with a `0` dim | — | — | ZERO_VOLUME |

`INPUT_TAGGERS` must tolerate rank < 2: read `W = shape[-1] if rank ≥ 1 else 1` and
`H = shape[-2] if rank ≥ 2 else 32` (a tensor with no second-to-last extent is not *mis*aligned).

## Circular Buffers

Indices 0–18 are the seed's, **unchanged and at the same indices** — that is a load-bearing part of
"the operand-free program is byte-identical". Indices 19–23 are new and are allocated only when
their operand is present.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_sticks` | 0 | `tile_size(in.dtype)` | `CB_RM_STAGE_DEPTH * WT_CHUNK` | spans `width` at chunk scope; streams over `row` (32 sticks fill one tile-row's worth) | input dtype | reader | compute | ROW_MAJOR only, whole kernel |
| `cb_input_tiles` | 1 | `tile_size(in.dtype)` | zero-copy `shard_h_t*shard_w_t`; else `CB_X_DEPTH * BLOCK_ROWS * x_hold_wt` when `!HAS_RESIDUAL`, or `CB_X_DEPTH * BLOCK_ROWS * WT_CHUNK` when `HAS_RESIDUAL` | spans `row` and `width` at block scope. **With a residual it drops from held to streaming** — the held role moves to `cb_x_sum` | input dtype | reader | compute | whole kernel |
| `cb_x_squared` | 2 | `tile_size(in.dtype)` | `BLOCK_ROWS * X_SQUARED_WT` (`X_SQUARED_WT ∈ {1, WT_CHUNK}`) | spans `row`; spans `width` only when the DEST fold is off | input dtype | compute | compute | pass A |
| `cb_scaler` | 3 | `tile_size(bf16)` | `scaler_pages` = 2 when `PARTIAL_W` else 1 | constant — a scaler pair or a 0/1 mask tile | **bfloat16** | reader | compute | whole kernel |
| `cb_row_stat` | 4 | `tile_size(fp32)` | `CB_ROW_STAT_DEPTH * BLOCK_ROWS`; **not allocated on a combine path** | spans `row`; the depth is a correctness floor, not overlap | **float32 always** — the cross-chunk accumulator must stay lossless even at 16-bit DEST | compute | compute | pass A → pass B |
| `cb_gamma_sticks` | 5 | `tile_size(w.dtype)` | `WT_CHUNK` | spans `width` at chunk scope; streams over `row` (a row vector) | weight dtype | reader | compute | `HAS_GAMMA && PER_CHANNEL_IS_RM` |
| `cb_gamma_tiles` | 6 | `tile_size(w.dtype)` | `x_hold_wt` | spans `width` at hold scope; **does not span `row`** — reuse-shared | weight dtype | reader (TILE) / compute (RM tilize) | compute | `HAS_GAMMA` |
| `cb_normalized` | 7 | `tile_size(in.dtype)` | `BLOCK_ROWS * WT_CHUNK` | spans `row` and `width` at block scope. Allocated only when `!HAS_RESIDUAL && (HAS_GAMMA \|\| HAS_BIAS)` — with a residual, `cb_x_sum` plays this role too | input dtype | compute | compute | pass B |
| `cb_output_tiles` | 8 | `tile_size(out.dtype)` | zero-copy `out_shard_pages`; else `CB_OUT_DEPTH * BLOCK_ROWS * WT_CHUNK` | spans `row` and `width` at block scope | output dtype | compute | writer | pass B |
| `cb_output_sticks` | 9 | `tile_size(out.dtype)` | `CB_RM_STAGE_DEPTH * WT_CHUNK` | spans `width` at chunk scope | output dtype | compute | writer | ROW_MAJOR only |
| `cb_sum_handoff` | 10 | `tile_size(fp32)` | `CB_ROW_STAT_DEPTH * BLOCK_ROWS` | spans `row`; depth 2 is the D25 pipeline | float32 | compute | writer | combine only |
| `cb_partials_gathered` | 11 | `tile_size(fp32)` | flat: `GATHER_SLOTS` = `group_size + group_size%2`; tree: `f0 + f0%2` | spans **neither** block axis since D27 — one compact tile per member per round; scales with the *group*, a bounded non-block parameter | float32 | writer | compute | combine only |
| `cb_stat_handoff` | 12 | `tile_size(fp32)` | `CB_COMBINE_FLAT_DEPTH` | constant — one finalized tile per round | float32 | compute | writer | combine only |
| `cb_row_final` | 13 | `tile_size(fp32)` | `CB_ROW_STAT_DEPTH * BLOCK_ROWS` | spans `row` — the un-permuted per-row stats pass B reads | float32 | compute | compute | combine only |
| `cb_bank` | 14 | `tile_size(bf16)` | `BLOCK_ROWS` | spans `row`; a one-hot permutation basis, synthesized once and never popped | **bfloat16** | reader | compute | compact combine only |
| `cb_compact_handoff` | 15 | `tile_size(fp32)` | `CB_COMBINE_FLAT_DEPTH` | constant | float32 | compute | writer | compact combine only |
| `cb_mcast_in` | 16 | `tile_size(fp32)` | `CB_COMBINE_FLAT_DEPTH` | constant; declared on **all** box cores incl. inactive ones so its L1 address is identical | float32 | writer | compute | compact combine only |
| `cb_gather_l1` | 17 | `tile_size(fp32)` | `f1 + f1%2` | scales with the tree's level-1 fan-in | float32 | writer | compute | tree combine only |
| `cb_node_out` | 18 | `tile_size(fp32)` | `CB_COMBINE_FLAT_DEPTH` | constant | float32 | compute | writer | tree combine only |
| **`cb_residual_sticks`** | **19** | `tile_size(in.dtype)` | `CB_RM_STAGE_DEPTH * WT_CHUNK` | mirrors `cb_input_sticks` exactly — the residual carries the input's geometry | input dtype | reader | compute | `HAS_RESIDUAL` && ROW_MAJOR |
| **`cb_residual_tiles`** | **20** | `tile_size(in.dtype)` | zero-copy `shard_h_t*shard_w_t` when `NATIVE_RESIDUAL`; else `CB_R_DEPTH * BLOCK_ROWS * WT_CHUNK` | spans `row` and `width` at block scope. **Never held** — consumed by `residual_add_block` and popped | input dtype | reader | compute | `HAS_RESIDUAL` |
| **`cb_x_sum`** | **21** | `tile_size(in.dtype)` | `BLOCK_ROWS * x_hold_wt` | spans `row` and `width` at **hold** scope — it takes over `cb_input_tiles`'s held role, so its capacity is `x_hold_wt` and not `WT_CHUNK` | input dtype | compute | compute | `HAS_RESIDUAL`, pass A → pass B |
| **`cb_bias_sticks`** | **22** | `tile_size(b.dtype)` | `WT_CHUNK` | mirrors `cb_gamma_sticks` | bias dtype | reader | compute | `HAS_BIAS && PER_CHANNEL_IS_RM` |
| **`cb_bias_tiles`** | **23** | `tile_size(b.dtype)` | `x_hold_wt` | mirrors `cb_gamma_tiles`; **does not span `row`** — reuse-shared | bias dtype | reader (TILE) / compute (RM tilize) | compute | `HAS_BIAS` |

**Every CB has exactly one producer and one consumer.** Two entries name a producer conditionally
(`cb_gamma_tiles` / `cb_bias_tiles`: the reader on the TILE path, the compute tilize on the RM path)
— these are **compile-time-disjoint builds**, not two concurrent producers; only one exists in any
given program. `cb_output_tiles` under `native_out` is pushed by compute and only *waited on* by the
writer as a completion barrier; the writer performs no read of the payload, so there is still one
consumer of the data.

**Sync rule.** Producer push count = consumer wait count on every CB. The three CBs read with
`PopPolicy::None` (`cb_scaler`, the held x/x_sum CB, the per-channel tile CBs) are popped by an
explicit `cb_pop_front` at the block or kernel boundary — the sanctioned held-CB pattern, and the
one place where push and wait counts are reconciled by hand rather than by the chain.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| B0a | `stage_scaler` | 1–2 tiles | yes | — | `cb_scaler`, 1–2 pages | pushed once, popped at kernel end |
| B0b | `stage_permutation_bank` | `BLOCK_ROWS` tiles | no (raw L1 stores) | — | `cb_bank`, `BLOCK_ROWS` | pushed once, **never popped** |
| B0c | `stage_gamma_row` / `stage_bias_row` | `x_hold_wt` tiles | partly | — | `cb_gamma_tiles` / `cb_bias_tiles` (TILE) or `cb_gamma_sticks` / `cb_bias_sticks` (RM), `WT_CHUNK` per chunk | held; popped once at kernel end when `X_RESIDENT` |
| B0d | `tilize_per_channel_rows` | `1 × WT_CHUNK` × `NUM_W_CHUNKS` | yes | `cb_gamma_sticks` / `cb_bias_sticks`, `WT_CHUNK`, popped | `cb_gamma_tiles` / `cb_bias_tiles`, `x_hold_wt` accumulated | held |
| B1 | `load_x_block` | `BLOCK_ROWS × WT_CHUNK` | partly | — | `cb_input_tiles` or `cb_input_sticks` | pushed |
| B2 | `load_residual_block` | `BLOCK_ROWS × WT_CHUNK` | partly | — | `cb_residual_tiles` or `cb_residual_sticks` | pushed |
| B3 | `tilize_x_block` (+ the residual's) | `BLOCK_ROWS × WT_CHUNK` | yes | `cb_input_sticks` / `cb_residual_sticks`, popped | `cb_input_tiles` / `cb_residual_tiles` | pushed |
| B4 | `residual_add_block` | `BLOCK_ROWS × WT_CHUNK` | yes | `cb_input_tiles` (`Block`, base `hold_base`) + `cb_residual_tiles` (`Block`), both popped | `cb_x_sum`, `WT_CHUNK` pages appended per chunk | `cb_x_sum` accumulates to `BLOCK_ROWS * x_hold_wt` when `X_RESIDENT`, else is popped by B5 |
| B5 | `square_block` | `BLOCK_ROWS × WT_CHUNK` | yes | `CB_X` = `cb_x_sum` if `HAS_RESIDUAL` else `cb_input_tiles`; `Upfront`/`None` + `TileOffset::Set` base `c*WT_CHUNK` when held, streaming otherwise | `cb_x_squared`, `BLOCK_ROWS * X_SQUARED_WT` | held CB **not** popped here |
| B6 | `reduce_accumulate_block` | `BLOCK_ROWS × X_SQUARED_WT` | yes | `cb_x_squared` (popped) + `cb_scaler` (`None`) | `cb_row_stat` or `cb_sum_handoff`, `BLOCK_ROWS` | fp32 carry live across chunks |
| B7 | `combine_block` | 1 compact tile per member | no (raw: matmul permute, `add_tiles` fold, `mcast_pipe`) | `cb_sum_handoff` → `cb_compact_handoff` → peers' `cb_partials_gathered` / `cb_gather_l1` → `cb_stat_handoff` → `cb_mcast_in` | `cb_row_final`, `BLOCK_ROWS` | rings return to base each round |
| B8 | `finalize_stat_block` | `BLOCK_ROWS` tiles | no (raw: scoped SFPU rsqrt) | `cb_row_stat` | `cb_row_stat` (in place, pop-then-reserve) | rotated one revolution |
| B9 | `normalize_block` | `BLOCK_ROWS × WT_CHUNK`, DEST-blocked `PASS_B_BLK` | yes | `CB_X` (`Block`, held or streaming) + `CB_STAT_B` (`Col`, `Upfront`/`None`) | `cb_normalized` if a post-stage follows, else `cb_output_tiles` | held CB popped at block end |
| B10 | `scale_block` | as B9 | yes | `cb_normalized` (`Block`) + `cb_gamma_tiles` (`Row`, `Upfront`/`None`) | **`cb_normalized` in place** when `HAS_BIAS`, else `cb_output_tiles` | when in place: `PerBlockSize` reserve/push, front rotated exactly one revolution |
| B11 | `bias_block` | as B9 | yes | `cb_normalized` (`Block`) + `cb_bias_tiles` (`Row`, `Upfront`/`None`) | `cb_output_tiles` | — |
| B12 | `untilize_out_block` | `BLOCK_ROWS × WT_CHUNK` | yes | `cb_output_tiles`, popped | `cb_output_sticks` | pushed |
| B13 | `store_block` | `BLOCK_ROWS × WT_CHUNK` | partly | `cb_output_tiles` / `cb_output_sticks`, popped | — | — |

**Pass-B output routing, stated once.** Let the post-normalize stages present be
`S = [scale] if HAS_GAMMA else [] + [bias] if HAS_BIAS else []`.
`normalize_block` writes `cb_output_tiles` when `S` is empty, else `cb_normalized`.
Each stage in `S` except the last writes **in place** into `cb_normalized`; the last writes
`cb_output_tiles`. With `S = [scale]` this collapses to exactly the seed's routing — the
operand-free and gamma-only programs are unchanged.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `stage_scaler` (aligned) | helper | `prepare_reduce_scaler<cb, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:59` | pool-type-aware overload, **mandatory** | — | `cb_scaler` | none |
| `stage_scaler` (partial, ReduceTile) | helper | `prepare_partial_reduce_scalers<cb, SUM, REDUCE_ROW, PARTIAL_W>(1.0f)` | `reduce_helpers_dataflow.hpp:132` | `PARTIAL_W` | — | `cb_scaler` (2 pages) | `PARTIAL_W` |
| `stage_scaler` (partial, AccViaAdd) | helper | `prepare_reduce_mask<cb, ReduceDim::REDUCE_ROW>(PARTIAL_W)` | `reduce_helpers_dataflow.hpp:74` | `PARTIAL_W` | — | `cb_scaler` (1 page) | `PARTIAL_W` |
| `load_x_block` / `load_residual_block` (RM) | helper | `read_sticks_for_tilize<cb>(acc, sticks, row_bytes, start_page, byte_off)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:88` | CB id; owns its reserve/push at TILE granularity | — | `cb_input_sticks` / `cb_residual_sticks` | `sticks` = `BLOCK_ROWS*32`, `byte_off` = `c*CHUNK_ROW_BYTES` |
| `load_x_block` / `load_residual_block` (TILE) | raw_api | `noc_async_read_tile` + one shared `noc_async_read_barrier()` | `tt_metal/.../dataflow_api.h` | — | — | `cb_input_tiles` / `cb_residual_tiles` | `WT_CHUNK` reads per tile-row |
| | | *Helpers considered and rejected*: `read_sticks_for_tilize` (`tilize_helpers_dataflow.hpp:88`) stages **sticks**, not tiles — it has no tiled-source form, and the tiled path needs per-tile `TensorAccessor` ids. No tiled block-read helper exists in `kernel_lib`. Raw is the only expression, and it is what lets **both** activation streams share one barrier | | | | | |
| `stage_gamma_row` / `stage_bias_row` (RM, flat) | helper | `read_sticks_for_tilize<cb>(acc, 1, row_bytes, 0, first_wt*32*elem)` | `tilize_helpers_dataflow.hpp:88` | one stick | — | `cb_gamma_sticks` / `cb_bias_sticks` | `WT_CHUNK` via `row_bytes` |
| `stage_gamma_row` / `stage_bias_row` (RM, **blocked**) | raw_api | `noc_async_read` per page: `WT_CHUNK` reads of `32*elem` bytes from page `first_wt+w` into offset `w*32*elem` of one staged stick | `dataflow_api.h` | — | — | as above | `WT_CHUNK` = read count |
| | | *Helpers considered and rejected*: `read_sticks_for_tilize` (`tilize_helpers_dataflow.hpp:88`) maps page *p* to tile **row** *p*; the blocked form needs page *p* to land at tile **column** *p* of one row. The helper's stick→row mapping is structural, not a parameter, so it cannot express a column-major staging of a row vector | | | | | |
| `stage_gamma_row` / `stage_bias_row` (TILE) | raw_api | `noc_async_read` at the trim granularity | `dataflow_api.h` | `GAMMA_TRIM` / `BIAS_TRIM` ∈ {0,1,2} | — | `cb_gamma_tiles` / `cb_bias_tiles` | `WT_CHUNK` |
| | | *Helpers considered and rejected*: `noc_async_read_tile` reads a **whole** tile; D23 measured 1.107–1.182× from reading only the face-rows the `BroadcastDim::Row` consumer touches, bit-exactly. No helper exposes a sub-tile read granularity | | | | | |
| `tilize_x_block`, `tilize_per_channel_rows` | helper | `compute_kernel_lib::tilize<block_width_tiles, in_dfb, out_dfb>(num_blocks)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:187` | `block_width_tiles = WT_CHUNK` (**compile-time**) | `cb_*_sticks` | `cb_*_tiles` | **`WT_CHUNK`** |
| `residual_add_block` | helper | `eltwise_chain(IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK), BinaryFpu<Add, input(cb_input_tiles, …, Block, Set), input(cb_residual_tiles, …, Block)>{x_base, 0}, PackTile<output(cb_x_sum, PerBlockSize, PerBlockSize)>{})` | `ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp:456` (`BinaryFpuOp::Add`), `:488` (`BinaryFpu`), `:391` (`output`) | `BinaryFpuOp::Add`, `BroadcastDim::None`, `OperandKind::Block` | `cb_input_tiles`, `cb_residual_tiles` | `cb_x_sum` | **`BLOCK_ROWS`, `WT_CHUNK`, `PASS_B_BLK`** |
| | | *Why `eltwise_chain` and not the `add<>` one-liner*: the convenience wrappers default-construct their elements (`convenience.inl:8-71`) and cannot pass the runtime tile base a `NATIVE_IN`/`X_RESIDENT` ring needs — the same reason recorded at `rms_norm_compute.cpp:49-53`. This is a *helper*, not a raw-API entry | | | | | |
| `square_block` | helper | `eltwise_chain(grid(rows, WT_CHUNK), BinaryFpu<Mul, CB_X, CB_X, D0, SQ_OUT.dest_accumulation>{base, base}, PackTile<SQ_OUT>{})` | `chain.hpp:488` | `DestAccumulation::PerRow` when the D12 fold is on | `cb_x_sum` / `cb_input_tiles` | `cb_x_squared` | **`BLOCK_ROWS`, `WT_CHUNK`, `X_SQUARED_WT`** |
| `reduce_accumulate_block` | helper | `ckl::reduce<SUM, REDUCE_ROW, …>` via the local `accumulate_reduce_block` router | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:616` | `ReduceInputBlockShape::of(rows, X_SQUARED_WT)`, `ReduceInputPolicy::BulkWaitBulkPop`, `ReduceAlgorithm::AccumulateViaAdd \| Auto`, `Accumulate::at/at_last` | `cb_x_squared`, `cb_scaler` | `cb_row_stat` / `cb_sum_handoff` | **`BLOCK_ROWS`, `X_SQUARED_WT`**, the chunk index and count |
| `finalize_stat_block` | raw_api | scoped SFPU rsqrt inside `transform_in_place` | `rms_norm_compute.cpp:202-236` | `VectorMode::C` scope | `cb_row_stat` | `cb_row_stat` | `BLOCK_ROWS` |
| | | *Helpers considered and rejected*: `unary<Rsqrt<>, …>` (`convenience.hpp`) forwards to `rsqrt_tile`, which hard-codes `VectorMode::RC` and exposes no seam (`rms_norm_compute.cpp:149-165`). The consumer reads only tile column 0 = faces 0 and 2 = `VectorMode::C`; scoping measured 1.03–1.14× and the unscoped form was the slowest cell at every geometry. The chain also cannot fold `*(1/W)` and `+eps` into the same DEST pass, which removes a bf16 rounding at 16-bit DEST (measured 2.04×) | | | | | |
| `combine_block` — permute / un-permute | raw_api | `matmul_init` + `matmul_tiles` against the one-hot `cb_bank` | `tt_metal/.../compute/matmul.h`; sites `rms_norm_compute.cpp:987-999`, `:1246-1263` | `transpose ∈ {0,1}` | `cb_sum_handoff`, `cb_bank` | `cb_compact_handoff`, `cb_row_final` | `BLOCK_ROWS` (columns used) |
| | | *Helpers considered and rejected*: a **column permutation** has no `kernel_lib` expression — eltwise/bcast/reduce preserve or collapse the column axis, and `transpose_wh` is a different map. The FPU's only horizontal-mixing primitive is the matmul (`rms_norm_compute.cpp:944-953`). `matmul_block_helpers.hpp:…` `matmul_block()` computes a *block* product and would reserve/push a block-shaped output; the permute needs one accumulated tile from `rows` rank-1 products | | | | | |
| `combine_block` — fold + finalize | raw_api | `add_tiles_init(acc_to_dest=true)` + `add_tiles` pairwise, then the scoped rsqrt, one `pack_tile` | `rms_norm_compute.cpp:376-438` | `GATHER_SLOTS` / `TREE_SL0` / `TREE_SL1` | `cb_partials_gathered` / `cb_gather_l1` | `cb_node_out` / `cb_stat_handoff` | `group_size`, `f0`, `f1` |
| | | *Helpers considered and rejected*: `eltwise_chain` cannot express fold+finalize in one DEST window — every element's `apply` runs on **every** inner iteration, so a `StatFinalize` element would rsqrt a partial sum `GROUP_SIZE/2` times (`rms_norm_compute.cpp:1146-1157`). Measured 2.18× fused vs. 1.93× for the expressible split | | | | | |
| `combine_block` — transport | helper | `McastArgs<CT,RT>`, `SenderPipe::send`, `ReceiverPipe::receive`, `Semaphore::up/wait_min` | `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp:243-289`, `:154`, `:219` | `Mcast1D::PerRow` / `Mcast2D` | `cb_stat_handoff` | `cb_mcast_in` / `cb_row_final` | `group_size` |
| `normalize_block` | helper | `eltwise_chain(grid(rows, WT_CHUNK).block_size(PASS_B_BLK), BinaryFpu<Mul, CB_X(Block,Set), input(CB_STAT_B, BroadcastDim::Col, Upfront, None, Col)>{base}, PackTile<…>{})` | `chain.hpp:488`, `:314` (`BroadcastDim`), `:369` (`input(spec, bcast)`) | `BroadcastDim::Col` — the stat is a `REDUCE_ROW` result, column-shaped, and must be srcB | `cb_x_sum`/`cb_input_tiles`, `cb_row_stat`/`cb_row_final` | `cb_normalized` / `cb_output_tiles` | **`BLOCK_ROWS`, `WT_CHUNK`, `PASS_B_BLK`** |
| `scale_block` | helper | same chain with `BinaryFpu<Mul, input(cb_normalized, …, Block), input(cb_gamma_tiles, BroadcastDim::Row, Upfront, None, Row, Set)>` | `chain.hpp:488`, `:314` | `BroadcastDim::Row` — gamma is `1×W`, valid in row 0. **In-place** variant when `HAS_BIAS`: `input(cb_normalized, PerBlockSize, PerBlockSize, Block)` + `output(cb_normalized, PerBlockSize, PerBlockSize)`, the device-verified case 1 of `ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/lifecycle/inplace_chain.cpp:47-64` | `cb_normalized`, `cb_gamma_tiles` | `cb_normalized` (in place) / `cb_output_tiles` | **`PASS_B_BLK`** |
| `bias_block` | helper | same chain with `BinaryFpu<Add, input(cb_normalized, …, Block), input(cb_bias_tiles, BroadcastDim::Row, Upfront, None, Row, Set)>` | `chain.hpp:456` (`BinaryFpuOp::Add`), device-tested Add+Row at `ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/bcast_binary_add.cpp:31-39` | `BroadcastDim::Row` | `cb_normalized`, `cb_bias_tiles` | `cb_output_tiles` | **`PASS_B_BLK`** |
| `untilize_out_block` | helper | `compute_kernel_lib::untilize<block_width_tiles, in, out>(num_blocks)` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:145` | `block_width_tiles = WT_CHUNK` | `cb_output_tiles` | `cb_output_sticks` | **`WT_CHUNK`** |
| `store_block` (RM) | helper | `write_sticks_after_untilize<cb>(acc, sticks, row_bytes, start_page, byte_off)` | `tilize_helpers_dataflow.hpp:130` | — | `cb_output_sticks` | — | `BLOCK_ROWS`, `WT_CHUNK` |
| `store_block` (TILE) | raw_api | `noc_async_write_tile` + barrier, skipping `wt ≥ Wt` pad tiles | `dataflow_api.h` | — | `cb_output_tiles` | — | `WT_CHUNK` |
| | | *Helpers considered and rejected*: `write_sticks_after_untilize` is the stick-form counterpart only (`tilize_helpers_dataflow.hpp:130`); there is no tiled block-write helper, and the pad-tile skip is per-tile | | | | | |
| DEST capacity | helper | `DEST_AUTO_LIMIT` | `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:88` | honours `fp32_dest_acc_en` and full/half sync automatically | — | — | bounds `PASS_B_BLK`, `COMBINE_DEST_BATCH` |
| kernel boot | required | `compute_kernel_hw_startup(CB_A, cb_scaler, cb_output_tiles)` | first statement of `MAIN()` | — | — | — | — |

`matmul_block_helpers.hpp` / `bias_add_helpers.hpp` / `reblock_untilize_helpers.hpp` /
`sfpu_activation_helpers.hpp` are **not applicable**: this op has no matmul phase, so
`add_bias_bcast_rows()` — which exists to consume `matmul_block`'s `LastBlockTarget::Interm`
partials — has no partials to consume. Its bias-add is expressed here by the `eltwise_chain`
`BinaryFpu<Add, …, BroadcastDim::Row>` above, which is the same broadcast on the same operand shape
without the matmul contract.

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `residual_add_block` | `Add` | `cb_input_tiles` — All `[H,W]` | `cb_residual_tiles` — All `[H,W]` | `None` |
| `square_block` | `Mul` | `cb_x_sum` / `cb_input_tiles` — All | same CB, same tile — All | `None` |
| `reduce_accumulate_block` | `SUM`/`REDUCE_ROW` | `cb_x_squared` — All | `cb_scaler` — All (constant `1.0`) or the 0/1 mask | n/a (reduce) |
| `normalize_block` | `Mul` | `cb_x_sum` / `cb_input_tiles` — All | `cb_row_stat` / `cb_row_final` — **Col0** (a `REDUCE_ROW` result) | **`Col`** |
| `scale_block` | `Mul` | `cb_normalized` — All | `cb_gamma_tiles` — **Row0** (`1×W`) | **`Row`** |
| `bias_block` | `Add` | `cb_normalized` — All | `cb_bias_tiles` — **Row0** (`1×W`) | **`Row`** |
| `combine_block` fold | `add_tiles` | `cb_partials_gathered[p]` — All | `cb_partials_gathered[HALF+p]` — All | `None` (pairwise DEST accumulate) |

`BroadcastDim` names **the axis being broadcast, not the axis reduced** (`chain.hpp:314-319`).
Only srcB may carry a broadcast (`BinaryFpuInputSpec`, `chain.hpp:336-341`) — which is why the
column-shaped stat must be the *second* operand of `normalize_block` and never the first.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| **The operand-free program must be byte-identical to the seed's** | The verifier builds the program with and without each operand and compares; a stray CB, CT arg or blocking term makes an operand-free build differ | Every new CB is at a **new index** (19–23) and is allocated only under its `HAS_*` flag; every new CT arg is **appended**; `_cb_block_mult` and both L1 solves gain terms multiplied by `HAS_RESIDUAL` / `HAS_BIAS`; pass-B routing collapses to the seed's when `S = [scale]` or `S = []`. Optionality is **compile-time specialization** — no zero tensor, no ones vector, no worst-case bucket over the possible operand sets |
| **A held CB can never be aliased in place** | `Row`/`Col` and `Upfront`/`None` operands are illegal as an in-place target (`chain.inl:82-85`, `inplace_chain.cpp:5-21`); an upfront-reserve output on an aliased CB **deadlocks** rather than returning a wrong answer | The only in-place site is `scale_block` when `HAS_BIAS`, on the compute-private `cb_normalized`, with `PerBlockSize` on both sides. `cb_x_sum` — the held CB — is written by `residual_add_block` and read by `square_block`/`normalize_block`, never aliased. Fallback if it hangs: a third block CB `cb_scaled` (Lamp L-BIAS-INPLACE, ledger delta recorded) |
| **`cb_x_sum` doubles the held footprint at RESIDENT** | With a residual, RESIDENT holds `cb_x_sum` at `BLOCK_ROWS*Wt_c` *and* streams `cb_input_tiles` + `cb_residual_tiles` at `CB_X_DEPTH*BLOCK_ROWS*WT_CHUNK` each. At `WT_CHUNK == Wt_c` that is 5 blocks where the seed had 1 | The L1 solve counts all three, so the regime search simply lands on a smaller `BLOCK_ROWS` or on ROW_RESIDENT. That is the required behaviour ("let the existing machinery re-decide"), not a failure. Lamp L-RES-DEPTH is the knob to measure if it decides too conservatively |
| **A residual on a `native_in` scheme must not cross the NoC** | The residual carries the identical shard spec, so it is *already* in this core's L1; reading it through a `TensorAccessor` would be a pure waste and would also blow the L1 budget with a redundant arena copy | `cb_residual_tiles` is zero-copy via `ttnn.cb_descriptor_from_sharded_tensor(residual)` whenever `native_in` holds. `l1_reserved` gains `_shard_l1_bytes(residual)` — the second resident shard the perf spec warns about (block-sharded fits ≤ 120 tiles/core, and `fp32_dest_acc_en=True` does not fit block-sharded at that extent at all) |
| **The reader's `NATIVE_IN` early return skips *all* streams** | The seed's `stage_x_chunk` returns before any read when `NATIVE_IN`; a residual that is not shard-backed would be silently skipped, producing `norm(x)` instead of `norm(x+r)` — a plausible-looking wrong answer | The early return is gated per stream: `NATIVE_IN` elides x's read, `NATIVE_RESIDUAL` elides the residual's. Since the contract pins the residual's memory config to the input's, the two flags always agree — but they are separate flags so a future divergence fails loudly rather than silently |
| **`STAGE_ZERO` must cover the residual ring too** | The boot zeroing establishes "every pad byte is zero or real tensor data" so the reduce's `×0` never meets an `inf`/`nan`. A residual ring left unzeroed reintroduces exactly that NaN, and `x_pad + r_pad` propagates it into `cb_x_sum` where the mask can still catch it — but only if it is finite | `STAGE_ZERO` zeroes `cb_input_sticks` **and** `cb_residual_sticks` at boot |
| **The D12 DEST square fold is gated on `PARTIAL_W == 0`** | The fold folds the row's last width tile *including its pad lanes* before the reduce runs, so the mask can no longer reach them | Unchanged from the seed, and it still holds with a residual: `cb_x_sum`'s pad lanes are `x_pad + r_pad`, which the same gate excludes. The BAND scheme keeps the fold because it passes `kernel_partial_w == 0` and zeroes its staging rings |
| **Rank 0 must not return NaN for a zero scalar** | `mean(x²)` over a one-element row is `x²`; dropping epsilon turns `0/0` loose | `W = 1`, `INV_W = 1.0f`, `PARTIAL_W = 1`, and epsilon stays inside the finalize's `rsqrt(sum*INV_W + eps)`. `0 * rsqrt(1e-12) = 0` |
| **Rank 0 must still consume every operand, in one program** | It is tempting to short-circuit a scalar on the host | Rank 0 is `ROWS·RESIDENT` at `R=1, W=1` — the *same* program, the same CBs, the same kernels. `weight`/`bias` are 1-element per-channel vectors and the residual is a 1-element activation; all three ride the ordinary paths |
| **Zero-volume must still be one dispatch** | Returning early without dispatching, or dispatching a copy op, both break "exactly one native device-program dispatch per public invocation" | One degenerate program on a single core with `row_count = 0`, so all three kernels take the inactive-core early return. `TensorAccessorArgs` are still emitted for the (empty) buffers. **Risk**: a 0-volume buffer may not survive accessor construction — if it does not, the fallback is a 1-page dummy CB set with every kernel returning on a `ZERO_VOLUME` compile-time flag before touching an accessor |
| **`inplace` aliases the output onto the input buffer** | `cb_input_tiles` and `cb_output_tiles` then name the same L1 region, and pass B reads x while writing out | Safe **by tile-index lockstep**: both CBs are pushed once with the whole shard and the last pass-B stage writes tile `(r*WT_CHUNK+w)` after every stage has read that same index; with gamma or bias present the intervening stages run to completion over the whole block before the last one writes anything. The op returns the **input tensor object**, and a `memory_config` disagreeing with the input's placement is refused rather than discarded |
| **`subblock_w` must be honoured, not absorbed** | Accepting the object and silently blocking differently is worse than refusing it, because the caller cannot see it | `subblock_w` **is** `PASS_B_BLK`. It is refused below 1, refused when it does not divide `block_w`, and refused above `DEST_AUTO_LIMIT` for the resolved `fp32_dest_acc_en` — each with a message naming both operands. It is never clamped |
| **`block_h` / `block_w` are not knobs** | Reading them as a blocking instruction would let a caller express a blocking the shard cannot support | Checked as a restatement of `shard_shape[0]//32`, `shard_shape[1]//32`; the geometry is then taken from the shard |
| **A per-channel operand at a dtype other than the input's must be *correct*, not merely accepted** | X-09 rules out "guard-clean but near-uncorrelated" | Every CB declares `data_format` = the dtype of the tensor it carries, and the `eltwise_chain` reconfig fold emits the format switch between `cb_normalized` (input dtype) and `cb_gamma_tiles`/`cb_bias_tiles` (their own dtypes). `weight` and `bias` may differ from each other, so the two stages carry two independent formats and the fold must not elide the second |
| **The blocked `(Wt, 32)` per-channel form is a *shape*, not a layout** | The `gamma_layout` axis cannot express it, and a validation rule that reads the trailing `32` as the channel count wrongly refuses every blocked operand with `W > 32` | Detection: rank 2 **and** `shape[-1] == 32` **and** `shape[0] == ceil_div(W,32)` **and** ROW_MAJOR ⇒ blocked, channel extent `shape[0]*32`. Otherwise flat, channel extent `shape[-1]`. The floor is `channel_extent ≥ W`, an inequality, at both layouts |
| **DRAM interleaving breaks the "blocked form is byte-identical" intuition** | The blocked form's `Wt` pages are spread across banks, so the single wide read the flat form uses would read the wrong bytes | The blocked branch issues `WT_CHUNK` per-page reads through the operand's own `TensorAccessor` |
| **`bfloat8_b` has no `element_size()`** | The ROW_MAJOR stick byte math would fault | Goes through the seed's `_stick_elem_bytes()`; block-float cannot be ROW_MAJOR anyway (`INVALID`), so 0 is the correct answer there |
| **`cb_row_stat` must stay float32 in both DEST modes** | It is the cross-chunk accumulator the reduce reloads; demoting it to the input dtype erases exactly the precision this op is about | Format pinned to float32 unconditionally. This is the one CB where the ledger's "page format follows DEST width" default is deliberately overridden, and the reason is recorded in the ledger row |
| **`unpack_to_dest_mode` must stay Default** | An `UnpackToDestFp32` CB may never be an FPU operand (`reduce_helpers_compute.inl:127-137`), and pass B consumes the stat as srcB of an FPU broadcast multiply | Left entirely at Default; tagging `cb_input_sticks` is separately forbidden by `tilize`'s `Fp32Mode::Fast` static_assert |
| **A partial final row-block straddles `cb_row_stat`'s ring** | `transform_in_place` rotates the CB; a ring of exactly `BLOCK_ROWS` makes the finalized tiles of a partial block wrap, and pass B's linear indexing reads past the end — pcc 0.55–0.93, invisible whenever `BLOCK_ROWS == 1` | `CB_ROW_STAT_DEPTH = 2`, counted through the same constant by both L1 solves so raising it cannot drift from the budget |

## Structural impossibilities (observations for a future `/golden-tests` pass)

Not edits to `feature_spec.py` — noted here only so the user can fold them in if they agree.

| Candidate | Why it is structural |
|-----------|---------------------|
| `{rank: 0, layout: TILE_LAYOUT}` and `{rank: 1, layout: TILE_LAYOUT}` | A tensor with no second-to-last dimension has no tile grid to lay out on. The cartesian never generates it today (rank is shape-derived and `INPUTS` carries no rank ≤ 1 shape), so it costs nothing to leave out — but a future rank-0 `INPUTS` entry would make it reachable |
| `{rank: 0, gamma_layout: ROW_MAJOR_LAYOUT, …blocked form}` | The blocked `(Wt, 32)` form of a 1-channel operand is `(1, 32)` with 31 padding lanes — legal, but indistinguishable from a flat rank-2 operand under the detection rule above. Only reachable if a rank-0 case ever supplies a blocked operand |
