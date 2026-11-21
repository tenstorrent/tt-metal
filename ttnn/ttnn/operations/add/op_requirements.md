# Operation Requirements: add

## Definition
- **Formula**: `out = a + b`, elementwise, with torch broadcasting (a size-1 dimension repeats; a lower-rank operand gets leading 1s). The output has the broadcast shape, the operands' layout, the dtype `dtype` asks for (else the operands'), and lives in the operands' buffer, interleaved (R9).
- **PyTorch Reference**: `torch.add(input, other)`. Graded against `a.double() + b.double()` on the stored inputs (the specification's `reference`).
- **Import Path**: `from ttnn.operations.add import add`
- **Function Signature**:
  ```python
  def add(
      input_tensor_a: ttnn.Tensor,
      input_tensor_b: ttnn.Tensor | float,
      *,
      dtype: ttnn.DataType = None,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
  ) -> ttnn.Tensor: ...
  ```
- **Specification**: `eval/specs/add/spec.py` (the only source of requirements). Target: Quasar on ttsim (`.claude/references/targets/quasar.md`, `.claude/references/backends/ttsim.md`). Host program: Metal 2.0 `ttnn.ProgramSpec` only.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Perf on this backend**: ttsim has no clock. A perf refinement here is gated on a **structural count** read off the op by a test or probe (backend page, *Performance measurement*); its gain in time is unconfirmed until a timed backend or hardware measures it. Never report a simulator timing as a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED placement**: memory_config ∈ {DRAM, L1}, sharding = interleaved
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4}, broadcast = none
- **SUPPORTED op-specific axes**: output_dtype = None
- **Cores**: full grid (flat page split over `compute_with_storage_grid_size()`, `row_wise=True`); thread plan (1, 1, 1), stated as data-movement bound
- **Compute config**: `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True, approx off; any fidelity / approx accepted (R3)
- **Golden baseline**: 87 / 4694 passed (86 supported_pass + rejection X1), 3046 xfail_expected, 1560 skipped (780 INVALID bf8b×ROW_MAJOR + 780 bf8b×TILE unbuildable on quasar), 0 loud (per `verifier_report.json`)

Omitted: SUPPORTED["dtype"] bfloat8_b — no path for block floats on Quasar (target page, *Family facts*: unbuildable on quasar, its cells skip; refuse in validate(), queue no refinement).

### [x] Refinement 0 — Height-sharded operands in L1 (the YOLOv8 bottleneck add)

**Goal**: add `"height"` to SUPPORTED["sharding"]: both operands height-sharded in L1, at the Phase 0 values of every other axis. The specification's TARGET gained this value after Phase 0 (rules R9 and R10, the INVALID entry DRAM x height, the DECISIONS entry of 2026-11-21). The result is allocated where the first operand is: height-sharded in L1 with that operand's shard shape and cores (R9).

**Origin**: the model-level test. YOLOv8's bottleneck shortcut (`models/experimental/ops/quasar/tests/yolo_ops/test_add.py::test_add_residual_sharded`, small-grid form `[1, 1, 64, 128]`) calls add on two bfloat16 TILE operands height-sharded in L1 over two cores (shard shape [32, 128]) and gets a tensor sharded the same way. The Phase 0 op refuses the call by name, and tt-metal's op serves it.

**Operator notes** (queued by the operator of the run, not by the verifier):
- A probe: with `"height"` added to SUPPORTED by hand and nothing else changed, the Phase 0 op passed the 39 golden cells `sharding=height` x bfloat16 x TILE x broadcast=none x output_dtype=None that the runner could build on ttsim, and the YOLOv8 test above. So the page walk through `TensorAccessor` already reaches the shards. What is owed is the claim and its evidence, and what the probe did not look at: where the result is and how it is sharded (R9), the op in one device session with tt-metal programs before and after it on sharded tensors, and operands whose shard grid is not the set of cores the op works on (a shard grid of one core, of two, of the full grid).
- The runner sizes a cell's shard grid from the shape and the device (`eval/CONVENTIONS.md` §2.3, `eval/memory.py`); the op must not depend on how many cores a grid spans. A cell whose input shards do not fit one core's L1 is skipped by the runner (on this simulator `[1, 2, 33600]`).
- Keep the data path: reading each shard on the core that holds it is a performance refinement of its own and is not asked for here.

**Done when**: every golden cell with `sharding=height` at the Phase 0 values of the other axes passes; the interleaved cells are unchanged; `verify_supported` loud categories are 0; a test in the op's own test file asserts that the result of a height-sharded call is height-sharded in L1 with the first operand's shard shape and cores.

### [ ] Refinement 1 — Broadcasting: tensor broadcast and a Python-number operand

**Goal**: add `"over_w"`, `"over_h"`, `"over_hw"`, `"over_leading"` and `"number"` to SUPPORTED["broadcast"] (all five values in one refinement), for TILE operands in DRAM or L1, at the Phase 0 dtype. The output is allocated with the broadcast shape (no longer `a.spec`), still in the operands' buffer (R9).

**Verifier notes**:
- Scheme-change (design regime `bcast_resident_b` + `number_scalar`), so it stands alone and goes first: the flat page walk unfolds into (leading, tile_rows, tile_cols), and every later refinement (layout, numeric) extends that walk rather than the flat one. The host split, the three-kernel skeleton and the implicit-sync DM protocol stay.
- b is **reuse-shared** along the broadcast axes: read each cluster's b slice once and hold it resident in its ring (`WaitPolicy::Upfront`, no per-block pop), never re-read it per output tile (`bcast_reread_b_per_tile` is the rejected regime). Intra-tile broadcast via the FPU binary's `BroadcastDim` (`over_w` → `Col`, `over_h` → `Row`, `over_hw` → `Scalar`, `over_leading` → none, page remap only). FPU broadcast is srcB-only: when a is the smaller operand (INPUTS has `(1,1,32,1)+(1,1,32,64)` and `(1,512)+(128,512)`), the host swaps the operands (R4; addition commutes).
- A resident b ring whose capacity depends on a shape needs an L1 bound and an `l1_ledger.md` row (symbol, bound, predicate, streaming fallback if it can exceed the budget); update the ledger's data-movement budget with b's crossings per cluster.
- `number` (R7): the number is taken as a **float32** value, not rounded to the operand dtype first. A scalar tile in bf16 would round it before the add; prefer the number as fp32 bits in a compute runtime arg (SFPU unary add on the fp32 DEST, `eltwise/unary/scalar.hpp`, availability on Quasar to confirm), or a 32-bit scalar tile. No b ring and no b traffic on this path. `validate()` already maps a number to `broadcast="number"` before touching tensor attributes; keep that.
- Under a thread plan with C > 1 the resident b ring must be visible to every NEO that needs it (the STRIDED pattern deals entries round-robin; see the target page and `quasar_op_examples/README.md` "Not covered": ALL access pattern). At the Phase 0 plan (1,1,1) this does not arise; if the implementer keeps C = 1 it stays out of scope, but say so in the changelog.
- Build it performantly (full grid, both DM halves batched), regime-pinned tests for each broadcast value, the session test (another program before, `to_layout` after) still green.

**Done when**: every golden cell with broadcast ∈ {over_w, over_h, over_hw, over_leading, number} at dtype bfloat16, TILE, DRAM/L1 passes; `verify_supported` loud categories are 0; b's bytes read per cluster ≤ one b slice (read off the program by a test).

### [ ] Refinement 2 — ROW_MAJOR layout for both operands and the output

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to SUPPORTED["layout"] for **both** operands and the output (output layout = a's), natively in the kernels (design regime `row_major_tilize`): the reader reads sticks into a 32-stick tile-row block, compute tilizes, adds, untilizes, the writer writes sticks. Cover every broadcast value Refinement 1 added (on RM), every alignment and rank, in DRAM and L1, and keep the full-grid split.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- Depends on Refinement 1 (the RM path must serve the broadcast cells it adds, including the number operand on an RM a). Hardest remaining generality axis, so it precedes the numeric bundle.
- The unit dealt to cores and NEOs becomes a tile-row (32 sticks), not a tile: under C > 1 rows dealt to NEOs need padded rounds and whole rounds per cluster (`quasar_op_examples/toy_row_passes`). Non-aligned H: the last tile-row has < 32 sticks; W not a multiple of 32: pad/mask in the reader so tilize sees whole tiles.
- An RM reader assembles a ring entry from several stick reads: that ring is **explicit sync** (listed in `disable_dfb_implicit_sync_for`, every transfer tagged `{.trid = 0}`, barrier on the same id), per the target page's ring-protocol rules; the session test must cover the RM path too.
- Wrapping the op in `ttnn.to_layout` / `ttnn.tilize` / `ttnn.untilize` is **not** the plan and would also break the one-dispatch rule. If the in-kernel tilize path cannot be finished in one pass, partial-tick (`[~]`) with whatever native RM cells landed and file `Refinement 2b` for the rest; a host-side layout conversion is not an acceptable escape hatch here.
- Keep `THREAD_PLAN` single-sourced; the per-regime thread plan for the compute-heavier RM path is Refinement 5's.

**Done when**: every golden cell with layout ROW_MAJOR (bfloat16, all broadcast values, DRAM/L1) passes, TILE cells unchanged, loud categories 0, session test green on an RM call.

### [ ] Refinement 3 — Skip the all-padding faces of the YOLOv8 dist2bbox tiles

**Type**: perf

**Goal**: no LOOSE_CASE is perf-flagged, so the region is free-selected: the YOLOv8 dist2bbox calls `[1, 2, 8400]` and `[1, 2, 33600]` (bfloat16, TILE, DRAM; 1,313 of the 1,513 tiles the model's five traced calls move). Their H is 2, so every tile page holds 2 valid rows: faces 2 and 3 (rows 16–31, the second 1,024 B of each 2,048 B page) are pure padding, yet the op reads both operands' full pages and writes full output pages, 6,144 B per output tile. Read (and, where the padding contract allows, write) only the first half page of every tile in a tile-row whose valid height is ≤ 16 (`H % 32 ∈ 1..16`, last tile-row of each image). This is a transfer-shape change on a DRAM-bound stream (catalog `ttnn/ttnn/operations/examples/master.md`: `double_buffer` — bytes in flight; `tile_reorder` — whole contiguous transfers; faces 0–1 are one contiguous 1,024 B run, so it stays one transfer per page). The same predicate applies to any shape with such a last tile-row; the W analogue (faces 1/3, two runs) is optional. No SUPPORTED change.

**Verifier notes**: an implicit-sync ring posts one credit per tagged transfer; if a sub-page transfer cannot fill a whole entry under implicit sync, that ring goes explicit-sync with tagged transfers (target page). Output padding becomes unspecified (stale ring bytes or untouched DRAM); the specification compares logical elements only (GUIDANCE, *Padding*), but keep the session test (`to_layout` after, and `add(add(a, b), b)`) green. After Refinement 1 the broadcast walk must use the same predicate per tile-row; after Refinement 2 it does not apply to ROW_MAJOR.

**Done when**: the structural count **bytes read from DRAM per output tile** on `[1, 2, 8400]` and `[1, 2, 33600]` drops from 4,096 to 2,048 (and bytes written per output tile from 2,048 to 1,024 if the writer half is taken), read off the op by a test (e.g. the host-computed transfer size handed to the reader, asserted against the shape); unchanged on the config-spanning guard set (one representative per kernel path × layout × placement, incl. an aligned shape, a non-aligned H > 16 shape and an L1 call), where results stay correct; golden suite green. The gain in time is unconfirmed on ttsim; record before/after counts in the changelog.

### [ ] Refinement 4 — Numerical configurability: float32 operands, 16-bit DEST, output dtype

**Goal**: add `ttnn.float32` to SUPPORTED["dtype"], `False` to SUPPORTED["fp32_dest_acc_en"] (bfloat16 operands; float32 + False stays refused by R1), and `ttnn.float32` and `ttnn.bfloat16` to SUPPORTED["output_dtype"] (the sum rounded once into the requested dtype, R8) — across every layout, broadcast value and placement already supported.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- float32 is graded at **0 ulp** (NUMERIC_RESOLVED `('float32', 'default')`), and `float32 → bfloat16` at 0 ulp too. The FPU source registers do not carry a full float32 (the gaps file: the FPU path round-trips through tf32), so the float32 add is the SFPU path the design's regime `stream_flat_fp32_sfpu` names: `add_binary_tile` with 32-bit pages unpacked to DEST, and for broadcast cells the exact `sfpu_add_bcast_col/row` (`tt_metal/hw/inc/api/compute/sfpu_binary_bcast.h`) — all exact on the simulator per `.claude/references/quasar-library-gaps.md` (*An accurate float32 path exists*). Mind that file's `UnpackToDest` rules: 32-bit pages only (a bf16 page into 32-bit DEST aborts ttsim), and `DataFormatReconfig::Disabled` on a chain whose operand is unpacked to DEST. This is the one compute-element change; DM, split and rings stay (ring entry size from `buffer_page_size()`).
- `output_dtype` differs from the operands': allocate the output from a spec with the requested dtype; the output ring's format follows it. `bfloat16 → float32` is graded at 0 ulp: with fp32 DEST the bf16 sum is exact in DEST and packed without rounding.
- `fp32_dest_acc_en=False` raises the DEST section to 8 tiles: `dest_section_tiles()` already clamps `block_tiles`; `BLOCK_TILES` may follow (single source in `add_program_artifacts.py`).
- Hardest first: lands after the broadcast walk and the RM path, extending both, so float32 / output_dtype cells exist on every path once.

**Done when**: every golden cell with dtype float32, fp32_dest_acc_en False, or output_dtype ∈ {float32, bfloat16} passes (float32 + fp32_dest_acc_en=False still refused by R1); the three float32 numeric regression cases (`small_magnitude`, `large_magnitude`, `uniform`) pass; loud categories 0.

### [ ] Refinement 5 — Thread plan per regime for the compute-heavier paths

**Type**: perf

**Goal**: Phase 0's plan (1, 1, 1) is right for the FPU `stream_flat` regime (one FPU op per tile: data-movement bound, matching the twin op's measured floor in `.claude/references/quasar_op_examples/README.md`). The regimes Refinements 2 and 4 add are not: the ROW_MAJOR path does tilize + add + untilize per block, and the float32 path does an SFPU add with unpack-to-DEST per tile. By the target page's *Choosing the plan* rule these are compute-bound, and a compute-bound regime takes C = 4 with readers to feed two inputs, (4, 4, 2), fed by whole rounds per cluster. Select the plan per regime from one host table (no second literal), and keep (1, 1, 1) on `stream_flat`. Catalog levers: `ttnn/ttnn/operations/examples/master.md` (`split_reader` for the reader side). No SUPPORTED change.

**Verifier notes**: depends on Refinements 2 and 4. The RM unit is a tile-row: padded rounds per `toy_row_passes`. A resident b ring under broadcast (Refinement 1) must reach every NEO that reads it. Keep the NaN-prefilled "every thread ran" test for each regime, and the session test.

**Done when**: the structural count **compute threads at work** on the ROW_MAJOR and the float32 regimes goes from 1 to 4 (readers ≥ 2), read off the program spec by a test (`KernelSpec.num_threads` per regime) together with a NaN-prefilled output test proving every NEO wrote its tiles; `stream_flat` stays at (1, 1, 1); the guard set (one representative per kernel path × layout × placement) stays correct; golden suite green. The gain in time is unconfirmed on ttsim; record it as such in the changelog.
