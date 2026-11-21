# Operation Requirements: subtract

## Definition
- **Formula**: `out = a - b`, element by element, over `torch.broadcast_shapes(a.shape, b.shape)`. Either operand may be the expanded one (rule R5).
- **PyTorch Reference**: `def reference(a, b): return a.double() - b.double()` (`eval/specs/subtract/spec.py`, `reference`)
- **Import Path**: `from ttnn.operations.subtract import subtract`
- **Function Signature**:
  ```python
  def subtract(
      input_tensor_a: ttnn.Tensor,
      input_tensor_b: ttnn.Tensor,
      *,
      dtype: ttnn.DataType = None,                                  # the output's dtype (axis `output_dtype`)
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,   # None -> default_compute_kernel_config()
  ) -> ttnn.Tensor
  ```
- **Target / backend**: Quasar (`.claude/references/targets/quasar.md`) on ttsim (`.claude/references/backends/ttsim.md`). Only Metal 2.0 `ttnn.ProgramSpec`; one `ttnn.generic_op` dispatch per call (no host-side sequencing of TTNN ops, so **no `ttnn.to_layout` / `ttnn.typecast` / expand wrapper is allowed, not even as a partial-tick fallback**).
- **Perf gate on this backend**: ttsim has no clock. Every perf refinement here is gated on a **structural count** named in its entry (value before → value that makes it done), read off the op by a test or probe; the gain in time is **unconfirmed** until a timed backend or hardware measures it.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED memory_config / sharding**: DRAM, L1 / interleaved (output in the first operand's buffer, R9)
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4}, broadcast = none
- **SUPPORTED op-specific axes**: output_dtype ∈ {None, bfloat16}
- **EXCLUSIONS**: R1 `{dtype: float32, fp32_dest_acc_en: False}`, R2 `{output_dtype: float32, fp32_dest_acc_en: False}` (MUST-refuse rules, pre-armed for Refinement 4)
- **Regime**: R0 `tile_stream_same_shape` — flat output-tile range split over the full 8×4 grid (`split_work_to_cores`, row-wise, contiguous ranges), one `compute_kernel_lib::sub` call per NEO over its whole share, `sync_tiles = 4` per DEST section / ring handshake, implicit-sync tagged reader/writer with the tag handed back.
- **Thread plan**: (R, C, W) = (1, 1, 1), host constant `THREAD_PLAN`. The op is bound by data movement (one FPU op per tile over DRAM tensors). The kernels work for any legal plan (tested at (2,2,2), (1,4,1), (4,4,2), (2,4,2)).
- **Compute config**: `default_compute_kernel_config()` = HiFi4, approx off, `fp32_dest_acc_en=True`, half-sync DEST. Translated to `ComputeHardwareConfig` with `fpu_math_fidelity` always set.
- **Golden baseline**: 78 supported_pass (72 cells + 6 numeric regression), 1,512 xfail_expected, X1 rejection pass, 0 supported_fail / xpass_drift / xfail_wrong_mode (`verifier_report.json`).
- **Accuracy**: PCC ≥ 0.999998, relative RMS ≤ 0.0019, golden `ulp_p99` = 0; max 1 bf16 ULP (10 of 1,048,576 elements on `[2,512,1024]`). The spec's grade is 1 ULP.

### [ ] Refinement 1 — Broadcasting, both orders (TILE)

**Goal**: add over_w, over_h, over_hw, over_leading, first_operand to `SUPPORTED["broadcast"]`, on the existing TILE / bfloat16 / fp32-DEST path, in DRAM and L1, at every rank and alignment already supported. This builds regime **R1 `bcast_b_index_mapped`** and **R2 `bcast_a`** of `op_design.md`:
- each line here is one value: SUPPORTED["broadcast"] over_w — `b` has W = 1: the reader maps output tile `(n, ht, wt)` to `b` page `(n_b, ht, 0)`; compute `sub` with `b` as `input(..., BroadcastDim::Col)`.
- SUPPORTED["broadcast"] over_h — `b` page `(n_b, 0, wt)`; `BroadcastDim::Row`.
- SUPPORTED["broadcast"] over_hw — `b` page `(n_b, 0, 0)`; `BroadcastDim::Scalar`.
- SUPPORTED["broadcast"] over_leading — index mapping only (size-1 or missing leading dims of `b` pinned to 0); `BroadcastDim::None`. Includes `b` of lower rank (`(4,8,32,64)` against `(32,64)`).
- SUPPORTED["broadcast"] first_operand — `a` is the expanded operand. The FPU broadcasts only srcB, so compute `-(b - bcast(a))` in one chain: `BinaryFpu<Sub>(b, input(a, BroadcastDim::…))` → `Negative` → `PackTile` (negation is exact, so the result is the once-rounded `a - b`, rules R5 / R6).

The split stays the flat **output** tile range over the full grid (row-wise, contiguous), with `tensor_tiles` from the output. The reader walks `(n, ht, wt)` of each output tile in its range and computes each operand's page from its own shape. The output goes in `a`'s buffer (R9) even when `a` is the smaller operand.

**Verifier notes**:
- Hardest generality item and a true dependency for Refinement 2: the ROW_MAJOR broadcast path (R5b) reuses this refinement's index mapping and broadcast compute. Land it first.
- It is a scheme change inside one cluster (new reader index walk + new compute call), not cross-core. No inter-cluster traffic: the multicast of `b` (R1c) is rejected in the design.
- Ship it performant. Keep the full-grid split, the implicit-sync tagged reads with no barrier in the loop, the tag hand-back on every reader/writer exit path, and one helper call per NEO over its whole share with `sync_tiles` per handshake (no per-tile init). Here the broadcast operand is read once per output tile (same bytes as a full-size `b`, never more than R0). Making it resident is Refinement 3's job. Do not build it here unless it falls out for free.
- Check first on ttsim, with a probe at C = 1, that the FPU in-tile broadcast (`BroadcastDim::Col/Row/Scalar`) works on Quasar. It has not been exercised by a pipeline op on this family. If it does not work, the documented-exact fallback is the SFPU broadcast family (`sfpu_sub_bcast_col/row`, `.claude/references/quasar-library-gaps.md`, "An accurate float32 path exists"). Add a library-gaps style note to the report.
- `first_operand` with a chain of two elements: one helper call, one output ring, so the pack-target gap does not bite. If you split it into two calls, `pack_init(<ring>)` is required.
- A call where **both** operands need an in-tile broadcast (`a` size-1 in H, `b` size-1 in W) is tagged `first_operand` but is not in `INPUTS`. Either handle it (e.g. replicate one operand's row/column in the reader) or refuse it from `validate()` with `NotImplementedError`. It must never return wrong values.
- L1: no new ring; the ledger rows gain the broadcast operand's axis accounting (the `b` ring streams over `tile_id`, spanning `sync_tiles`). The block knobs stay single-sourced (`SYNC_TILES`, `RING_DEPTH_QUANTA`, `THREAD_PLAN`).
- Tests: regime-pinned tests per broadcast value with ragged per-cluster counts, a non-default plan (2,2,2) on one broadcast shape, and the one-session test (another program → subtract with broadcast → `ttnn.to_layout(out, ROW_MAJOR)`).

**Done when**: every golden cell with `broadcast ∈ {over_w, over_h, over_hw, over_leading, first_operand}` inside the new SUPPORTED passes. Phase 0 cells still pass, and `verify_supported` shows 0 supported_fail / xpass_drift / xfail_wrong_mode.

### [ ] Refinement 2 — ROW_MAJOR layout (both operands and the output), native in-kernel

**Goal**: add ROW_MAJOR to `SUPPORTED["layout"]` for both operands (the output takes `a`'s layout), for every broadcast value Refinement 1 supports, in DRAM and L1, at every rank and alignment. Two regimes of `op_design.md`:
- **R5 `row_major_flat`** (`broadcast = none`): the operation is elementwise and the shapes are equal, so element order does not matter. The reader packs consecutive elements of the flat row-major buffer into tile-sized ring entries (an explicit-sync ring assembled from stick reads). The same `sub` call runs on them, and the writer scatters back to sticks. No tilize/untilize.
- **R5b `row_major_tilize`** (broadcast ≠ none): the reader gathers 32 sticks per tile-row. Compute runs `tilize` → `sub` with Refinement 1's broadcast → `untilize`, inside the one program.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- Depends on Refinement 1 (R5b reuses its index mapping and broadcast compute).
- **The one-native-dispatch rule forbids the usual partial-tick escape hatch** (`ttnn.to_layout` / `ttnn.tilize` / `ttnn.untilize` around the TILE kernel). If R5b is more than one focused pass, ship R5 (ROW_MAJOR × broadcast = none) natively as `[~]` and file **Refinement 2b** right after this entry for ROW_MAJOR × broadcast ≠ none. Keep the RM-broadcast cells out of SUPPORTED with an `EXCLUSIONS` entry `{layout: ROW_MAJOR, broadcast: <value>}` (structural capability gap, allowed) until 2b deletes it.
- A ring the reader assembles from several transfers is **explicit sync**. List it in `disable_dfb_implicit_sync_for`, give every NoC transfer of that kernel `NocOptions::TXN_ID` with `{.trid = 0}` and barrier on the same id (target page, *Ring protocols*). The writer likewise if it scatters one entry into several sticks.
- Alignment hazard for R5: a 1,024-element entry boundary falls inside a stick whenever W·elem_size does not divide 2,048 B. That needs sub-page reads/writes at offsets that are not 16 B (L1) / 64 B (DRAM) aligned (`ttnn.get_l1_alignment()` / `ttnn.get_dram_alignment()`). The non-aligned golden shapes (`W = 50, 17, 8400`) hit this. Design the entry around whole sticks (k sticks per entry, k·W ≤ 1,024, padding the rest) or stage unaligned pieces through an aligned scratch. Do not let a transfer straddle the alignment rule.
- Keep Phase 0's full-grid contiguous split (in sticks or flat chunks) and batch both halves. A writer that barriers per stick is half-done.
- Thread plan: keep the regime's kernels plan-agnostic (counts from compile-time args). R5 is data-movement bound, so (1,1,1) is right. R5b's three passes per tile make it compute-heavier, and its plan is turned in Refinement 5.
- L1 ledger: add rows for any stick-staging / tilize / untilize ring, with lifetime and sharing justification. `cb_tilized` → `sub` → `cb_untilize_in` can often pack in place. Justify any intermediate ring by its live set.
- Tests: the one-session test with an RM output (subtract → `ttnn.to_layout(out, TILE)`), and RM × L1 output placement (R9).

**Done when**: every golden cell with `layout = ROW_MAJOR` inside the new SUPPORTED passes (all broadcast values, or broadcast = none only on a `[~]` with Refinement 2b filed). Prior cells still pass, and the loud categories are 0.

### [ ] Refinement 3 — Read the broadcast operand once per cluster (resident window)

**Type**: perf

**Goal**: after Refinement 1, the expanded operand is fetched once **per output tile** (regime R1). On shapes where a cluster's output range repeats a broadcast tile, that re-reads the same page over the NoC. Build **R1b `bcast_b_resident`** (`op_design.md`): the reader fetches each distinct broadcast tile the cluster's range needs once, and compute holds it across the run of output tiles that use it. Use the helper's held-operand form (`input(cb, BroadcastDim::…, WaitPolicy::PerTile / Upfront, PopPolicy::None)` with an explicit pop at the end of the run, or a resident window indexed on a `grid(Ht_c, Wt)` shape). Relevant catalog patterns in `ttnn/ttnn/operations/examples/master.md`: `double_buffer` (bytes in flight per useful tile) and `compute_block_size` (hold an operand across a block). Target shapes (bf16, TILE, DRAM, `fp32_dest_acc_en=True`, `dtype=None`, all landed by Refinement 1):
- `(4,8,32,256) − (4,8,32,1)` (over_w): 256 output tiles, 8 per cluster, one `b` tile per tile-row.
- `(4,128,512) − (4,1,1)` (over_hw): 256 output tiles, 8 per cluster, one `b` tile per image.

**Structural count (the gate)**: NoC reads of the broadcast operand per cluster on the target shapes. **Before: 8** (one per output tile; bytes read per output tile = 2 tiles). **Done: 1** (the distinct broadcast tiles the cluster's range touches; bytes read per output tile = 1.125 tiles). A test or probe reads the count off the op, e.g. the reader's per-cluster runtime-arg schedule or a per-cluster counter written by the reader. The changelog records before/after. No SUPPORTED change. The gain in time is **unconfirmed** on ttsim.

**Verifier notes**:
- Bound the resident window by an L1 predicate with the per-tile R1 path as the fallback (`b_tiles_needed_by_core · page_bytes ≤` budget share from `ttnn.get_max_worker_l1_unreserved_size()`). An over_leading `b` can be large (`(1,512,1024)` = 512 tiles). Add the ledger row with its bound and predicate.
- Under a thread plan with C > 1 a STRIDED ring deals each entry to one NEO. A tile held by every NEO needs one entry per NEO (or the ALL access pattern, which no pipeline example covers yet). At C = 1 this does not arise; state what the implementation does at C > 1 and test it at (2,2,2).
- Shapes with no per-cluster reuse (over_leading, the small broadcast shapes) cannot improve here. Their count must not get worse.

**Done when**: the count is 8 → 1 on both target shapes, and the golden suite is green. No regression in the count, or in correctness, across the guard set: one representative per path × layout × placement — R0 TILE DRAM, R0 TILE L1, each broadcast value TILE, ROW_MAJOR flat, ROW_MAJOR broadcast.

### [ ] Refinement 4 — Numerical configurability: float32 operands, 16-bit DEST, float32 output

**Goal**: add float32 to `SUPPORTED["dtype"]`, False to `SUPPORTED["fp32_dest_acc_en"]`, and float32 to `SUPPORTED["output_dtype"]`, across every layout and broadcast value already supported. Rules R1 and R2 stay enforced: the two `EXCLUSIONS` entries already in the op file become live gates and must not be removed. Per the design's regimes:
- **R3 `fp32_operands`**: FPU source registers hold 19 bits, so float32 operands need the SFPU subtract to meet the spec's **0-ULP** float32 grade. Both input rings `UnpackToDest` (32-bit pages only), `binary_sfpu<SubBinary…>` or the raw `sub_binary_tile`, and `DataFormatReconfig::Disabled` on that chain (library gap: the chain's reconfig asserts on an `UnpackToDest` buffer). Load every tile of a DEST section before the first SFPU op (library gap on `UnpackToDest`).
- **R4 `fp32_output` / `acc16`**: `output_dtype = float32` from bf16 operands keeps the FPU path (bf16 is exact in the source registers; the difference is exact in 32-bit DEST for these inputs) with a Float32 output ring. `fp32_dest_acc_en = False` for bf16 operands runs 16-bit DEST: `dest_tiles()` already raises the `sync_tiles` clamp to 8, so ring depth follows from the one source.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Precision is the cheapest axis, so it comes last of the generality refinements. It extends the broadcast and RM structure of Refinements 1–2 instead of being reworked under them.
- "Zero kernel changes" does **not** hold for float32 operands: R3 is a different compute call. With broadcast it also needs a float32 broadcast. The FPU broadcast reads 19-bit sources, so use `sfpu_sub_bcast_col/row` or replicate in the reader. The RM paths need 4-byte element arithmetic in the reader/writer and Float32 tilize/untilize formats.
- `bfloat16` with a 16-bit page into 32-bit DEST must keep `UnpackToSrc` (ttsim aborts on `UnpackToDest` of a bf16 page into 32-bit DEST).
- Risk to watch: `float32 → bfloat16` output is graded at **0 ULP** (provisional in the list). It is met only if the packer rounds the SFPU's float32 difference to nearest-even exactly once. The Phase 0 bf16 path shows 10 in 10⁶ elements at 1 ULP, inside its 1-ULP grade. If that pattern shows up at 0 ULP, it is a `numerical-precision` failure to keep failing and name in a follow-up. It is not a reason for EXCLUSIONS, and it is a spec-defect question only if a correctly-rounded path provably cannot exist.
- L1: Float32 pages double the ring bytes. The ledger's worst case (`6 · 16 · 4 · 4,096 B` = 1.5 MiB) already covers it. Update the worked values and the `Page format` column. A 16-bit output ring under 32-bit DEST stays justified only by the output ABI (R6).
- Keep the thread plan at (1,1,1) here with plan-agnostic kernels. Turning the plan on the SFPU path is Refinement 5.

**Done when**: every golden cell with `dtype = float32`, `fp32_dest_acc_en = False` or `output_dtype = float32` inside the new SUPPORTED passes. The R1/R2 cells are refused with `NotImplementedError`, prior cells still pass, and the loud categories are 0.

### [ ] Refinement 5 — Compute threads on the compute-heavier regimes; drop the redundant format reconfig

**Type**: perf

**Goal**: Phase 0's (1,1,1) plan is right for R0, where the op is bound by data movement (the toy measured no gain from more NEOs). Refinements 1, 2 and 4 add regimes with real work per tile, which the target page classes as **compute-bound**:
- R3: `UnpackToDest` + SFPU subtract per tile.
- R5b: tilize + subtract + untilize per tile.
- R2: FPU subtract + SFPU negate per tile.

Give each of these regimes its own plan constant (one host constant per regime, passed as compile-time args as today): C = 4 with (2,4,2), or (4,4,2) where the regime reads more than it writes. Engage `min(grid, ceil(tiles / C))` clusters on those regimes (`MIN_TILES_PER_CORE = C`), so no NEO idles for lack of tiles. Second lever (⭐ T1, catalog `compute_block_size`): `DataFormatReconfig::Disabled` on chains whose input and output formats are all equal (R0 and R1 at bf16→bf16). Target shapes: `(2,512,1024)` float32 TILE DRAM (1,024 tiles, 32 per cluster) and `(2,512,1024) − (1,512,1024)` bf16 ROW_MAJOR DRAM.

**Structural count (the gate)**: (a) compute threads at work per cluster on the target shapes: **before 1, done 4**. Shown by the existing "every thread ran" test pattern: plan read off the built `ProgramSpec`, and a ragged per-NEO tail completing correctly under STRIDED rings. (b) Format reconfigs emitted per helper call on the all-bf16 R0/R1 chains: **before 1, done 0**. No SUPPORTED change. The gain in time is **unconfirmed** on ttsim. If a timed backend (`ttsim-perf`) becomes available, measure each regime's plan against (1,1,1) once on the target shape and revert any regime where it loses.

**Verifier notes**:
- R0 (bf16, TILE, broadcast = none) keeps (1,1,1): it sits at the data-movement floor. Do not turn its plan.
- The resident broadcast window of Refinement 3 under C > 1 must still hold. Re-run its count test at the new plan.
- Run the one-session test (another program → subtract → `ttnn.to_layout`) at each new plan.

**Done when**: both counts reach their done values on the target shapes. The golden suite is green, and the guard set (one representative per path × layout × placement) is correct with no count regressions.
