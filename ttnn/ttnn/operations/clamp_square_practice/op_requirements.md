# Operation Requirements: clamp_square_practice

## Definition
- **Formula**: `y = clamp(x, min_val, max_val) ** 2`, elementwise, with `min_val <= max_val` finite fp32 scalars.
- **PyTorch Reference**: `def ref(x, min_val, max_val): return torch.clamp(x.float(), min_val, max_val) ** 2`
- **Import Path**: `from ttnn.operations.clamp_square_practice import clamp_square_practice`
- **Function Signature**: `clamp_square_practice(input_tensor: ttnn.Tensor, min_val: float, max_val: float) -> ttnn.Tensor`

## Queue source (provisional TARGET)

`eval/golden_tests/clamp_square_practice/feature_spec.py` **does not exist** (see `verification_report.md`). Until it
is authored, TARGET is taken from `op_design.md` → "TARGET axes and the regime table":

| Axis | SUPPORTED (Phase 0) | Provisional TARGET | Gap → refinement |
|---|---|---|---|
| `dtype` | bfloat16 | bfloat16, float32, bfloat8_b | float32, bfloat8_b → R4 |
| `layout` | TILE | TILE, ROW_MAJOR | ROW_MAJOR → R2 |
| `alignment` | tile_aligned | tile_aligned, non_tile_aligned | non_tile_aligned → R2 |
| `memory_layout` | INTERLEAVED | INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED | 3 sharded values → R1 |
| `buffer_type` | DRAM | DRAM, L1 | L1 → R1 |
| `rank` | rank_ge_2 | rank_ge_2 | none. `rank_lt_2` is a documented omission: the op contract mandates rejecting rank < 2. |

Proposed structural `INVALID` for the spec author: `{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}`.

**There is no perf-flagged `LOOSE_CASE`** (no spec), so perf regions are free-selected from the measured baseline in
`verification_report.md`. Perf-1's target contract (bf16 / TILE / interleaved DRAM / default compute config) is already
fully in SUPPORTED at Phase 0, so R1 and R2 are not needed to unlock it and are ordered by difficulty.

**Every axis is independent** (pure elementwise; no dependent axis exists). So every generality unlock below is a
knob-turn in Blocking-Model terms, and none requires a cross-core scheme-change. The only new topology anywhere in the
queue is the zero-copy sharded CB placement in R1. That is a local placement change, not a combine.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Acceptance-test boundary note**: `tests/.../test_clamp_square_practice.py` contains Phase-0 rejection tests for values this queue adds (`test_rejects_wrong_dtype` [float32], `test_rejects_row_major_layout`, `test_rejects_non_tile_aligned`, `test_rejects_l1_interleaved`, `test_rejects_sharded`). The refinement that adds a value retargets that test to a still-unsupported value (or retires it once nothing on that axis is rejected). This is the expected update to the SUPPORTED boundary and does not count as a regression.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED memory_layout / buffer_type**: [INTERLEAVED] / [DRAM]
- **SUPPORTED shape-derived axes**: rank=rank_ge_2, alignment=tile_aligned
- **SUPPORTED op-specific axes**: none (`min_val`, `max_val` are runtime scalars, not axes)
- **Cores**: full grid. The flattened tile axis is split with `split_work_to_cores(row_wise=True)`, giving 120/120
  cores on BH at ≥ 120 tiles.
- **Compute config**: fixed `fp32_dest_acc_en=False`, half-sync DEST. SFPU-only chain, so fidelity is irrelevant.
- **Blocking**: `block_tiles = min(8, dest_limit, core_tiles_g1)`, with input and output depth = 2 blocks.
- **Golden baseline**: n/a (no golden suite). Acceptance 48/48, precision 12/12, extended 8/8.

### [ ] Refinement 1 — L1 placement + local sharding (`sharded_resident` regime)

**Goal**: add `ttnn.BufferType.L1` to `SUPPORTED["buffer_type"]`. Add `HEIGHT_SHARDED`, `WIDTH_SHARDED` and
`BLOCK_SHARDED` to `SUPPORTED["memory_layout"]`. All of these must be implemented natively in the kernels.

- **Interleaved L1** uses the existing `interleaved_stream` kernels unchanged: `TensorAccessor` already maps L1 banks.
  Only the validate gate and the output memory config change. The output should follow the input's memory config.
- **Sharded inputs** take the design's deferred `sharded_resident` regime:
  - Back `cb_input_tiles` on the input shard and `cb_output_tiles` on an output tensor with the **same** shard spec,
    using `ttnn.cb_descriptor_from_sharded_tensor`. That makes them zero-copy.
  - The reader and writer reduce to a `cb_push_back` / `cb_wait_front` of the core's shard (no NoC traffic).
  - The core grid is the shard grid, and `core_tiles` is the shard's tile count.
  - The compute chain keeps DEST groups of `block_tiles` *inside* the resident shard.

This regime has no dependent axis: all three shard strategies keep each core's work local, including block sharding,
because the op is elementwise.

**Verifier notes**:
- This is the hardest generality item, so it goes first. It adds a second regime, and later refinements (R2 layout,
  R4 dtypes) must extend both regimes. Building the regime first means they extend it once rather than retrofitting.
- No skill covers `memory_layout` placement, so do **not** use `/memory-layouts` here.
- **The native path is the requirement.** Re-reading a core's own local shard through a `TensorAccessor` as if it
  were interleaved does **not** count as implementing sharding. Send it back even if the tests pass.
- The CB ring/quantum contract changes for the resident shard: the CB capacity is the whole shard, so there is no
  wrap. The compute chain's `FullBlock` + `PerBlockSize` policies must still close on the shard's tile count. Use a
  `BlockTailSync` / bulk-wait choice that is legal for a non-multiple-of-`block_tiles` shard.
- Update `l1_ledger.md`: for the sharded regime the CB "capacity" is the tensor's own shard and is not an op
  allocation. Add a regime row.
- Mixed placements (sharded in, interleaved out or the reverse) are out of scope unless the spec's TARGET asks for
  them. The output follows the input.
- Shard specs for tests come from `eval.sharding.auto_shard_config`. Retarget `test_rejects_l1_interleaved` and
  `test_rejects_sharded`. Once all values are supported there is nothing left on these axes to reject, so retire
  both tests.

**Done when**: interleaved-L1 inputs and height-, width- and block-sharded L1 inputs pass the acceptance reference
at the acceptance PCC. The shard-specific cases must include a ragged shard (tile count not a multiple of
`block_tiles`) and a multi-core block-sharded grid. No sharded path issues a NoC read of its own shard. All prior
tests stay green.

### [ ] Refinement 2 — ROW_MAJOR layout + non-tile-aligned shapes

**Goal**:
- Add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` through in-kernel data access. Two shapes are acceptable:
  - a tilize-wrapped reader/compute (tilize → Clamp → Square → untilize, inside the same dispatch), or
  - since the op is pure elementwise with no tile-shape dependency, a stick-paged path where the reader brings RM
    sticks into a CB the chain treats as tile-sized chunks.

  The skill decides which. The output layout follows the input layout.
- Add `"non_tile_aligned"` to `SUPPORTED["alignment"]` for **both** TILE and ROW_MAJOR:
  - **TILE:** the geometry is already `ceil` per image, so the padding is computed on and never observed. This is
    mostly a validate relaxation plus tests.
  - **ROW_MAJOR:** the last partial tile/stick must be handled in the reader or compute.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- RM must also run on the sharded regime from R1. RM sharded tensors are page = stick, which the skill's retrofit
  workflow covers.
- Do **not** wrap the op in `ttnn.to_layout` / `ttnn.tilize` / `ttnn.untilize`. It also violates the
  one-dispatch-per-call rule.
- If an RM + sharded combination turns out to need more than a focused pass, partial-tick (`[~]`) and file that
  combination as `Refinement 2b`. Do not use a manipulation-op wrapper.
- Keep `block_tiles` as the single block source for any new CBs, e.g. a tilize staging CB sized
  `depth · block_tiles`, never `Wt`. Add the new rows to `l1_ledger.md`.
- Retarget `test_rejects_row_major_layout`. Retarget `test_rejects_non_tile_aligned` to a still-rejected case, or
  retire it.

**Done when**: RM and non-tile-aligned (H or W not a multiple of 32) inputs pass the reference at the acceptance PCC
on interleaved DRAM, interleaved L1 and at least height-sharded placement. Rank-2 and rank-3 RM cases are included.
All prior tests stay green.

### [ ] Refinement 3 — Speed up the bandwidth-bound large interleaved-DRAM cells

**Type**: perf

**Goal**: the large bf16 / TILE / interleaved-DRAM cells reach only about 370–385 GB/s of Blackhole's ~512 GB/s
nominal. Measured: `(8,1,1024,1024)` 87.2 µs = 385 GB/s, and `(1,1,4096,4096)` 182.5 µs = 368 GB/s, both on 120
cores. Each core's NoC batch is capped at 8 tiles because `block_tiles` is pinned by the DEST cap and does double
duty as the NoC batch.

Relevant `ttnn/ttnn/operations/examples/master.md` levers:
- Decouple the NoC batch from the DEST group: issue several blocks of reads/writes before one barrier
  (`trid_double_issue`, "batch your barriers first", with optimum `cb_blocks/2`), co-tuned with
  `input_depth_blocks` / `output_depth_blocks` in the 2–4 range (`double_buffer`).
- `bank_stagger` for contiguous per-core ranges that all start on the same bank.
- `noc_placement` row-vs-diagonal on BH's asymmetric grid.

Keep the compute quantum at the DEST-capped `block_tiles`. Introduce any new NoC-batch factor as its own
single-source knob. No SUPPORTED change.

**Verifier notes**:
- Filed after R1 and R2 on the 2:1 cadence. It touches only the `interleaved_stream` reader and writer, and those R1
  and R2 do not restructure for this path.
- Optimize the exact Phase-0 config (bf16, TILE, interleaved DRAM, `fp32_dest_acc_en=False`). Do not optimize a proxy.

**Done when**: measured device-ns improves on `(8,1,1024,1024)` and `(1,1,4096,4096)` (bf16 / TILE / DRAM) with all
op tests green. There must be no regression across the config-spanning guard set: one representative per kernel path
× layout × placement existing at that point — interleaved DRAM, interleaved L1, sharded, RM, plus a small
`(1,1,256,256)` and a mid `(2,4,256,512)` shape.

### [ ] Refinement 4 — Numerical configurability (float32, bfloat8_b, compute_kernel_config)

**Goal**:
- Add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`.
- Expose `compute_kernel_config: ttnn.ComputeKernelConfig | None = None` on the entry point: `fp32_dest_acc_en`,
  `math_approx_mode`, `dst_full_sync_en`, and `math_fidelity` (accepted, though the chain is SFPU-only).
- Derive the CB page formats from the tensor dtypes.

Cells that fail out of the box go to `EXCLUSIONS`, not into their own refinement. Typical cases are
`bfloat8_b + non_tile_aligned` and anything structurally covered by the proposed INVALID `bfloat8_b + ROW_MAJOR`.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Precision knobs are the cheapest tier, so they come last among the generality refinements. They must extend
  **every** regime and layout built by R1 and R2.
- The `block_tiles` clamp already reads `dest_limit(fp32_dest_acc_en, dst_full_sync_en)`. Wire the *exposed* config
  flags into that one function rather than the module constants, so the DEST cap of 4 tiles for fp32 DEST follows
  automatically (single source of truth).
- float32 input will normally need `fp32_dest_acc_en=True` and `UnpackToDestFp32` on `cb_input_tiles`.
- Retarget `test_rejects_wrong_dtype`, for example to `ttnn.uint32`, or retire it.
- Adding fp32 DEST also makes non-bf16-exact bounds exact on clamped elements. See the report's precision note.

**Done when**: float32 and bfloat8_b inputs pass the reference at the acceptance PCC table's thresholds (0.999 for
fp32, 0.99 for bf8b) across interleaved, sharded and RM placements. The exceptions are cells explicitly listed in
`EXCLUSIONS` or proposed-INVALID. `compute_kernel_config` round-trips, with `fp32_dest_acc_en` True and False both
tested. All prior tests stay green.

### [ ] Refinement 5 — Speed up the small and mid-size (launch/fill-bound) cells

**Type**: perf

**Goal**: tensors of up to about 1k tiles give each core ≤ 9 tiles (one or two blocks), so per-core fill, first-read
latency and launch dominate. Measured on BH:
- `(1,1,256,256)`: 64 tiles on 64 cores, 2.35 µs = 112 GB/s
- `(1,1,32,8192)`: 256 tiles on 120 cores, 4.31 µs = 243 GB/s
- `(2,4,256,512)`: 1 024 tiles on 120 cores, 13.75 µs = 305 GB/s

Speed these up using the relevant `master.md` patterns. Candidates:
- the design's "grid synchronization / occupancy" lamp: use the `core_grid` knob to cap participating cores so each
  gets at least one full block or a tuned minimum, versus full participation;
- `noc_placement` (row vs diagonal for a partial line of cores on BH);
- `bank_stagger`.

No SUPPORTED change.

**Verifier notes**:
- After R4, generality is exhausted, so this slot is perf.
- The occupancy cap must be a measured, shape-dependent rule derived from `tensor_tiles`, defined once on the host.
  Do not hard-code a per-shape table.

**Done when**: measured device-ns improves on `(2,4,256,512)` and `(1,1,32,8192)` (bf16 / TILE / DRAM). The large
shapes from Refinement 3 must not regress, all op tests stay green, and there is no regression across the
config-spanning guard set: one representative per kernel path × layout × placement × dtype.
