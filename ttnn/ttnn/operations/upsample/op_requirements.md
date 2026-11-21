# Operation Requirements: upsample

## Definition
- **Formula**: `output[n, i, j, c] = input[n, i // sh, j // sw, c]`; output `[N, H·sh, W·sw, C]`, the input's dtype, layout and buffer (interleaved). Every output element is an input element, unchanged (spec R1).
- **PyTorch Reference**: `reference(input_tensor, scale_factor)` in `eval/specs/upsample/spec.py` — `input.double().repeat_interleave(sh, dim=1).repeat_interleave(sw, dim=2)` (= `torch.nn.functional.interpolate(mode="nearest")` on the channels-first permutation).
- **Import Path**: `from ttnn.operations.upsample import upsample`
- **Function Signature**: `upsample(input_tensor: ttnn.Tensor, scale_factor, mode: str = "nearest") -> ttnn.Tensor` — `scale_factor` an int or a pair `(sh, sw)` of whole numbers ≥ 1 (form never gated, R2).
- **Specification**: `eval/specs/upsample/spec.py` (the only source of requirements). One `ttnn.generic_op` dispatch per call (run rule); Metal 2.0 `ttnn.ProgramSpec` only (Quasar).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend (this run: Quasar on ttsim)**: there is no clock. A perf refinement is gated on the **structural count** its entry names (read off the op by a test or a probe, before and after recorded in `changelog.md`); its gain in time is unconfirmed until a timed backend (`ttsim-perf`) or hardware measures it. Never report a simulator timing as a speedup.
> **One dispatch**: wrapping the op in `ttnn.to_layout` / `ttnn.tilize` / `ttnn.untilize` / `ttnn.to_memory_config` is not even a partial-tick fallback here — it is a second dispatch, which the run's rules forbid. Every capability below is in-kernel.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [ROW_MAJOR_LAYOUT]
- **SUPPORTED memory_config / sharding**: DRAM, L1 / interleaved
- **SUPPORTED shape-derived axes**: channel_alignment = tile_aligned; scale = uniform (factors 1–4, int and pair)
- **SUPPORTED op-specific axes**: mode = nearest
- **Regime**: `stick_stream` — input sticks (N·H·W flattened) split over the full 8×4 grid (`split_work_to_cores`, row-wise); per cluster blocks of `block_sticks` sticks, one STRIDED DM→DM ring entry each; each stick read once, written `sh·sw` times.
- **Cores / threads**: full grid; thread plan (R, C, W) = (2, 0, 4), no compute kernel (data-movement op).
- **Compute config**: none (spec DECISIONS: pure data movement, CONVENTIONS §2.2 exemption).
- **Golden baseline**: 26 / 26 supported cells pass (bit-exact); 142 xfail_expected, 84 invalid_skipped, 0 loud (per `verifier_report.json`).

### [ ] Refinement 0 — The `memory_config` argument and a height-sharded input (the YOLOv8 sharded upsample)

**Goal**: two things the specification gained after Phase 0, in one refinement because the model's call needs both.
1. The signature gains a keyword-only `memory_config=None` (the stub in `eval/specs/upsample/spec.py`; rule R7; `CALL["canonical"]`, axis `memory_argument`). Add `memory_argument` to SUPPORTED with `None` and `"input"`: `memory_config=` the input tensor's own memory config gives the same result as leaving it out, and any other config reads as `"other"` and is refused by name.
2. Add `"height"` to SUPPORTED["sharding"]: a height-sharded input in L1, at the Phase 0 values of every other axis. The result is height-sharded in L1 over the input's cores with the shard shape `[input shard height * sh * sw, C]` (R5).

**Origin**: the model-level test. YOLOv8 upsamples a sharded neck feature map in place (`models/experimental/ops/quasar/tests/yolo_ops/test_upsample_small_grid.py::test_upsample_sharded_small_grid`): `upsample(x, (2, 2), memory_config=x.memory_config())` with `x` a `[1, 4, 4, 64]` bfloat16 ROW_MAJOR tensor height-sharded in L1 over two cores (shard shape [8, 64]); the result is `[1, 8, 8, 64]`, height-sharded over the same two cores (shard shape [32, 64]). The Phase 0 op has no `memory_config` parameter, so the call is a TypeError there, and tt-metal's op serves it.

**Operator notes** (queued by the operator of the run, not by the verifier):
- The baseline golden run of this refinement is red in every cell: the runner passes `memory_config=None` for the new axis and the Phase 0 entry point does not take the keyword. That is the signature change, not a regression; the Phase 0 cells must pass again once the parameter exists.
- A sibling op gives a hint, not a proof: the generated `add`, which reads and writes pages through `TensorAccessor`, served height-sharded operands once its gate let them through, with no change to its kernels. Here the output is not sharded like the input (four times the sticks per core for a factor of (2, 2)), so the output's memory config has to be built: same cores, same orientation, the new shard shape.
- The runner sizes a cell's shard grid from the shape and the device (`eval/CONVENTIONS.md` §2.3, `eval/memory.py`): a shard need not hold whole image rows, and the op must not depend on how many cores a grid spans. A cell whose input shards do not fit one core's L1 is skipped by the runner.
- Keep the data path: upsampling each shard on the core that holds it is a performance refinement of its own and is not asked for here.

**Done when**: the entry point's signature equals the stub (`eval.validate_contract upsample` reports no `signature` error); every golden cell with `sharding=height` or `memory_argument=input` at the Phase 0 values of the other axes passes exactly; the Phase 0 cells pass again; `verify_supported` loud categories are 0; a test in the op's own test file makes the model's call and asserts the values and the result's placement (height-sharded in L1, the input's cores, shard shape [32, 64]).

### [ ] Refinement 1 — TILE layout (`tile_layout` regime)

**Goal**: add `ttnn.TILE_LAYOUT` to `SUPPORTED["layout"]` with a native tile-layout reader/writer path, on the Phase 0 cells' other axes (bfloat16, tile_aligned C, uniform factors 1–4, DRAM and L1 interleaved). This is the design's deferred `tile_layout` regime (`op_design.md`, Regimes): a **scheme-change** — a second kernel pair, selected per call by `layout`, inside the same single `generic_op` dispatch.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- Hardest generality refinement, so it goes first: land the tile structure on the small surface, and Refinement 2 extends both layouts for the light axes afterwards.
- Tensor geometry: in `[N, H, W, C]` TILE layout the tiles span `(W, C)` per image row `(n, h)`. H replication is whole tile-rows written `sh` times (whole-tile NoC writes from one resident copy — catalog `tile_reorder`: move whole pages, never scatter faces to DRAM). W replication regathers tile *rows* (each row of a tile = two 16-element face rows): output tile row `w'` comes from input row `w' // sw`, so an output tile is assembled in L1 from face rows of one or more input tiles and then written once. W is not a multiple of 32 in most INPUTS entries even when C is (`[1,16,16,32]`, `[1,7,9,32]`, `[1,256,4,32]`), so the row gather across tile boundaries and the padded last W tile (input padding rows are not output data, spec GUIDANCE) are in scope here, not deferred.
- Keep it a byte mover for R1: if the W regather goes through a compute kernel (untilize → replicate → tilize helpers), its formats must carry float32 bit-exact once Refinement 2 adds it; a DM face-row shuffle has no such risk. Either is the implementer's call (scope boundary), but the choice must not need reworking in Refinement 2.
- Split: keep the cheapest-traffic split (input read once, output written once, zero cross-core traffic) over the full grid — contiguous ranges of input tile-rows / tiles per cluster, as `stick_stream` does with sticks; block knobs on the host, single source of truth, ledger rows for any new ring. Re-use `THREAD_PLAN` and its tests (each thread moves its blocks; session order with a tt-metal program before and after) for the new kernel pair.
- Not a wrapper: `ttnn.to_layout` around the RM kernels is a second dispatch (forbidden), and tile→RM→tile conversion would also be forbidden by the one-dispatch rule. No partial-tick escape hatch exists for this refinement; if a facet is out of reach, partial-tick by leaving TILE cells of that facet in `EXCLUSIONS` and file `Refinement 1b`.

**Done when**: `TILE_LAYOUT` is in `SUPPORTED["layout"]`; every golden cell with `layout=TILE` × bfloat16 × tile_aligned × uniform × {DRAM, L1} passes bit-exact; Phase 0 cells still pass; thread-plan and session-order tests cover the TILE kernels; `l1_ledger.md` and `op_design.md` (regime row → built) updated.

### [ ] Refinement 2 — float32, non-tile-aligned channels and per-axis factors on both layouts

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]`, `non_tile_aligned` to `SUPPORTED["channel_alignment"]`, and `per_axis` to `SUPPORTED["scale"]`, on both the ROW_MAJOR (`stick_stream`) and TILE paths. All are **knob-turns** of the built regimes.

**Implementation skill**: /numeric-formats-metal, /memory-layouts

**Verifier notes**:
- ROW_MAJOR path: verified gate-only by the verifier probe `tests/ttnn/unit_tests/operations/upsample/probes/probe_002.py` (validate() bypassed via `_dispatch`): float32 `[1,16,16,32]`, `[1,40,40,256]`, `[1,17,50,17]`, `[2,8,8,50]` (2,4); bf16 C=3 (6 B sticks), C=17 (34 B, L1), C=50 (100 B, L1); per-axis (2,1), (1,2), (3,2), (2,4) — all bit-exact. `stick_stride = align_up(stick_bytes, max(l1_align, dram_align))` already covers sub-alignment sticks. So on RM this is a SUPPORTED/validate() change plus the cells' confirmation.
- TILE path (from Refinement 1): float32 tiles (64 B face rows, 4 KiB tiles), padded C columns of the last tile (carried along, not output data) and `sh ≠ sw` must all work. R1 holds: no float32 value may pass through a 16-bit format on the way.
- `/numeric-formats-metal` applies for the dtype only: this op has **no** `compute_kernel_config` (spec DECISIONS, CONVENTIONS §2.2), so do not add one; ring `data_format_metadata` follows the tensor dtype.
- `bfloat8_b` is not part of this refinement (unbuildable on Quasar; see the Omitted line below).

**Done when**: the three values are in SUPPORTED; every golden cell over {bf16, fp32} × {RM, TILE} × {tile_aligned, non_tile_aligned} × {uniform, per_axis} × {DRAM, L1} passes bit-exact (0 `xfail_expected` left: the bfloat8_b cells skip as unbuildable or INVALID), and prior tests pass.

### [ ] Refinement 3 — Writer/reader balance and pipeline fill on the model's calls (RM)

**Type**: perf

**Goal**: `LOOSE_CASES` flags no perf case, so the target is the free-selected YOLOv8 calls — `[1,40,40,512]` and `[1,40,40,256]` by (2, 2), bfloat16 ROW_MAJOR, DRAM (and `[1,20,20,512]` (2,2) as the second shape). Each cluster gets 50 sticks; `block_sticks = ceil(50 / (W·BLOCKS_PER_WRITER_THREAD)) = 7` makes 8 blocks of 7,7,7,7,7,7,7,1, so the writer threads carry 14/14/14/8 sticks (critical path 14, balanced 13) and the readers 28/22 (balanced 25), and a cluster reads a whole 7-stick block before its first write can start. Candidate levers, from `ttnn/ttnn/operations/examples/master.md` and the design's Perf lamps: a balanced cut (block extents differing by ≤ 1, ring entry sized to the larger), `BLOCKS_PER_WRITER_THREAD` / `RING_ENTRIES_PER_THREAD` co-tune, `trid_double_issue` (retire the oldest of several write trids instead of draining a thread's NoC queue at each block barrier), an implicit-sync reader on one-stick entries. The implementer picks the lever. No SUPPORTED change.

**Gate (structural, ttsim — no clock)**: on `[1,40,40,512]` (2,2): (a) max output pages written by one writer thread per cluster, 56 → ≤ 52 (= `sh·sw·ceil(core_sticks / W)`); (b) sticks read on a cluster before its first output write, 7 → ≤ 2. A test or probe reads both off the host split / block schedule (the existing `split_sticks` + `block_sticks_for` arithmetic, or its replacement) and asserts them. The gain in time is unconfirmed on this backend; say so in the changelog. If a `ttsim-perf` run is available later, the design's thread-plan alternatives `(1,0,1)`, `(1,0,2)`, `(2,0,2)`, `(1,0,4)` are measured there, not here.

**Done when**: both counts meet the gate on the target shapes; the golden suite is green (bit-exact); thread-plan tests still show each thread moving its own blocks; and no count regresses (max pages per writer thread, input reads per stick = 1, output writes per output page = 1) on the guard set: one representative per kernel path × layout × placement — RM DRAM `[1,64,64,32]` (small sticks), RM L1 `[2,8,8,128]`, RM DRAM `[1,8,8,4096]` (wide C), and the TILE path's representative from Refinement 1.

### [ ] Refinement 4 — Whole-tile transfers on the TILE path for the model's maps

**Type**: perf

**Goal**: after Refinement 1, speed up the TILE path on `[1,40,40,512]` (2,2) bfloat16 TILE DRAM (the model's map in tile layout). Relevant catalog entries in `ttnn/ttnn/operations/examples/master.md`: `tile_reorder` (DRAM writes as whole tiles, never faces), `double_buffer` (several transfers per barrier, ring depth 2·block), `trid_double_issue`. The W regather is core-local L1 work; it should not add DRAM transactions. No SUPPORTED change.

**Gate (structural, ttsim — no clock)**: per output tile, DRAM/NoC write transactions = 1 and input DRAM reads per input tile = 1 (sub-tile face writes to DRAM count against it); transfers per barrier ≥ the block's tile count. Record the values Refinement 1 shipped with as "before". The gain in time is unconfirmed on this backend.

**Done when**: the gate holds on the target shape with the golden suite green, and no count regresses on the guard set (one representative per kernel path × layout × placement, as in Refinement 3).

**Verifier notes**: if Refinement 1 already ships at the gate (whole-tile writes, batched barriers), mark this `[x]` with the counts recorded and no code change; do not invent work.

Omitted: SUPPORTED["dtype"] bfloat8_b — unbuildable on Quasar (target page *Family facts*: no bfloat8_b path; its golden cells skip as `unbuildable on quasar`); validate() refuses it and no refinement is queued. Its ROW_MAJOR cells are also INVALID in the specification.
