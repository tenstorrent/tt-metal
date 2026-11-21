# Operation Requirements: permute

## Definition
- **Formula**: `output[i_0, ..., i_{r-1}] = input[j_0, ..., j_{r-1}]` with `j_{dims[k]} = i_k`; output size along dim i = input size along `dims[i]`. Every output element is an input element, bit for bit (spec R1).
- **PyTorch Reference**: `torch.permute(input_tensor.double(), tuple(dims))` (spec `reference`)
- **Import Path**: `from ttnn.operations.permute import permute`
- **Function Signature**: `permute(input_tensor: ttnn.Tensor, dims) -> ttnn.Tensor`. `dims` is a list or tuple of ints, one per dimension; negative entries count from the last dimension (R2).

Specification: `eval/specs/permute/spec.py`. Design: `op_design.md` (Blocking Model, Regimes). Target: Quasar on ttsim.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Perf gates on this backend**: ttsim has no clock. Every `**Type**: perf` entry below is gated on a **structural count** read off the op by a test or probe (backend page, *Performance measurement*), with its value now and the value that makes it done. Any gain in time is **unconfirmed** until a timed backend (ttsim-perf) or hardware measures it; never report a simulator timing as a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE, ROW_MAJOR]
- **SUPPORTED memory_config / sharding**: DRAM, L1 / interleaved
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {3, 4}
- **SUPPORTED op-specific axes**: permutation ∈ {leading, last_two, general}
- **Cores**: full grid (8×4 = 32 clusters), `split_work_to_cores(row_wise=True)`; three regimes (`page_gather` (1,0,1), `tile_transpose` (1,1,1), `staged_gather` 6 worker harts)
- **Compute config**: none public (spec: the op computes nothing). `tile_transpose` runs LoFi, 16-bit DEST (bit-exact datacopy).
- **Golden baseline**: 133 / 133 supported cells pass + 3 / 3 rejections; 508 xfail_expected (all `dtype ∉ {bf16}` or `rank = 2`); 0 loud (per `verifier_report.json`)

### [ ] Refinement 1 — 32-bit dtypes and rank 2

**Goal**: add `ttnn.float32`, `ttnn.int32`, `ttnn.uint32` to `SUPPORTED["dtype"]` and `2` to `SUPPORTED["rank"]`, natively in all three regimes, so the 508 xfail_expected cells (396 32-bit rank-3/4 cells, 112 rank-2 cells across all four dtypes) pass bit-exact. Line by line:
- `SUPPORTED["dtype"]` float32: element moves of 4 B in `page_gather` / `staged_gather` (the gather kernel is already templated on the CT `element_bytes`); the 32-bit transpose path in `tile_transpose`.
- `SUPPORTED["dtype"]` int32: same as float32. Exact integer moves (R1: an int32 value must not pass through a bfloat16 or float32-rounded buffer; R1's check is the int32 cells).
- `SUPPORTED["dtype"]` uint32: same, with every ring declaring `int32` (uint32 is not a buffer format on Quasar; `_ring_format()` already does this).
- `SUPPORTED["rank"]` 2: canonicalization already lifts rank 2 to `(1, 1, H, W)`; no kernel change expected.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- No `compute_kernel_config` is exposed: the spec gives the op no compute config (DEFINITION: "It therefore has no compute config"). Use the skill for the dtype→ring-format derivation and the precision-matrix tests, not for a public config knob.
- `tile_transpose` 32-bit: Quasar's `transpose_init` has a 32-bit branch that **requires** `UnpackToDestEn` (`tt_metal/hw/inc/api/compute/transpose.h`, `#else` arm: "32-bit transpose on Quasar requires unpack-to-dest to be enabled"), i.e. `enable_32_bit_dest=True` plus `UnpackToDest` on `cb_input_tiles` for Float32 / Int32 rings. bf16 must keep 16-bit DEST and the default unpack. The DEST window halves at 32-bit (`DEST_AUTO_LIMIT` = 4 in half sync): keep `TRANSPOSE_BLOCK_TILES` clamped by it (the kernel's `static_assert` already reads `DEST_AUTO_LIMIT`). Test the 32-bit transpose alone at C = 1 first.
- Fallback if the NEO 32-bit transpose does not run on ttsim: send 32-bit TILE H↔W swaps to `staged_gather` (pure DM, exact by construction). This makes `select_regime` depend on dtype too; record it in the changelog and the design. It is an in-kernel path, not a manipulation-op wrapper.
- `staged_gather` TILE footprint doubles at 4 B. The only shape class whose minimum box then exceeds L1 is TILE with **both** output in-tile axes leading and ≥ 32 (32·32 tiles × 4 KiB = 4 MiB), the design's deferred `staged_gather_subtile` row. No INPUTS entry reaches it (checked: every TILE `staged_gather` entry has `a` or `b` ∈ {h, w}, or leading extents < 32), so it is not a goal of this refinement; such a call is refused by `plan_staged_gather()` with `NotImplementedError`.
- `page_gather` RM ring cap: 4-B rows halve the W limit to 32,767 (largest INPUTS W is 8192).
- Keep the Phase 0 splits and thread plans on the new paths (performance-conformance bar), and extend the device-session tests (`test_permute_plan.py`) to one 32-bit case per regime: the `UnpackToDest` compute config is a new program shape after which tt-metal's `to_layout` must still run.

**Done when**: all 508 currently-xfail golden cells pass exactly (0 xfail_expected, 0 loud categories), all Phase 0 cells still pass, and the session / every-thread-ran tests cover one 32-bit case per regime.

### [ ] Refinement 2 — Speed up the transaction-bound YOLOv8 RM gathers

**Type**: perf

**Goal**: no LOOSE_CASE is perf-flagged, so the region is chosen here: YOLOv8's two large traced RM calls, both `staged_gather`, both bound by NoC transactions at the layout minimum (each output or input page is its own transfer):
- RM `[1,4,16,8400]` dims `(0,1,3,2)`: 176 boxes, 2,816 reads of 384 B, **33,600 writes of 32 B**;
- RM `[1,4,8400,1]` dims `(0,3,1,2)`: 176 boxes, **33,600 reads of 2 B**, 176 writes of 384 B.

Today every one of those transfers goes on NoC0: the gather kernel's single `Noc` uses the reader config for its writes too (design perf lamp "NoC for `staged_gather` writes"). It also waits `async_writes_flushed` after every `OUT_SLOTS` = 8 writes, whatever the slot size. Apply the cheap (⭐ T1) levers from `ttnn/ttnn/operations/examples/master.md` that fit this profile: `noc_placement` (writes on NoC1, or the two NoCs split across the harts) and `double_buffer` / keeping more bytes in flight (`out_slots` chosen per slot size, e.g. ≥ 32 for slots ≤ 64 B; the L1 cost is ≤ 24 × 64 B per thread, recorded in `l1_ledger.md`). The implementer owns the exact lever and values. No SUPPORTED change.

**Done when** (structural counts, read off the op by a test or probe; time gain unconfirmed on ttsim):
- writes issued on NoC1 for `[1,4,16,8400]` `(0,1,3,2)`: 0 of 33,600 now → all of them, or a documented half-and-half split across the two NoCs;
- writes in flight per `async_writes_flushed` wait on that shape: 8 now → ≥ 32;
- golden suite green, the session tests green (a second `Noc` object must still hand its tag back / tag every transfer), and no regression across the config-spanning guard set (one representative per regime × layout × DRAM/L1). The changelog records each count before and after.

### [ ] Refinement 3 — Cut the per-element cost of the staged_gather shuffle

**Type**: perf

**Goal**: the design states `staged_gather` as bound by compute on the DM harts ("several RISC-V instructions per element"). Today `shuffle_block` moves one element per load/store pair, and in TILE it tests `col_off[x] == INVALID_OFF` on every element. Reduce the instructions per output element. Two levers: (a) register blocking. Pair adjacent output pages so that one 32-bit staging load feeds two bf16 outputs (in RM, output rows `r, r+1` read adjacent elements of one staged input row; in TILE, the same holds along the input's in-tile column axis). (b) Hoist the padding test out of the TILE inner loop: a valid-column prefix copy plus a zero tail, computed once per output tile. The relevant catalog material is `ttnn/ttnn/operations/examples/master.md`, on compute-side costs per element. Target shapes: RM `[1,4,16,8400]` `(0,1,3,2)` (YOLOv8; 537,600 output elements) and TILE `[1,16,20,20]` `(0,2,3,1)` (the YOLOv8 stem shape in TILE layout, a staged TILE gather). No SUPPORTED change.

**Done when** (structural counts; time gain unconfirmed on ttsim):
- staging loads per output element in `shuffle_block` for bf16: 1 now → ≤ 0.5 on the RM target shape (count read off the kernel by a probe or a debug counter compiled in under a test-only CT flag);
- per-element `INVALID_OFF` branches in the TILE inner loop: 1 now → 0;
- bit-exactness kept for every dtype then in SUPPORTED (bf16, and the 4-B dtypes once Refinement 1 lands: a pair of 4-B elements is a 64-bit move or stays scalar). Golden suite green; no regression across the guard set (one representative per regime × layout × DRAM/L1).

**Verifier notes**: lands after Refinement 1, so the 2-B and 4-B element paths are designed together (`EB` is a CT arg; the blocking must be correct for both).

### [ ] Refinement 4 — Fill the grid on whole-tile-quantum TILE gathers

**Type**: perf

**Goal**: TILE `staged_gather` boxes cannot be split below one 32-element quantum on `{h, w, a, b}`, so shapes whose output pages each draw on many input tiles engage few workers. TILE `[2,32,64,128]` dims `(0,2,1,3)` (C and H swapped, W stays) makes 16 boxes of 32 staged tiles: **16 of 192 worker harts** (16 of 32 clusters, 1 of 6 harts each) do all the work. Lower the quantum on the input's in-tile row axis `h` when `h ∉ {a, b}`. With quantum 16 (one face pair, a contiguous 1 KiB of a bf16 tile, 64-B aligned) or smaller (a row pair of both faces: 2 reads of 64 B in DRAM), a box stages only the input rows its output tiles use. This is the face-granular quantum of the design's deferred `staged_gather_subtile` row, applied for occupancy. It also brings the 4-B both-leading TILE case under the L1 budget, the row's original purpose. The catalog levers are grid occupancy and keeping bytes in flight (`ttnn/ttnn/operations/examples/master.md`). This is a ⭐⭐ T2 change confined to `choose_gather_block()`'s quanta and the TILE reader's sub-page read sizes and offsets; the box machinery and the shuffle's offset tables stay. No SUPPORTED change.

**Done when** (structural counts; time gain unconfirmed on ttsim):
- worker harts with ≥ 1 box on TILE `[2,32,64,128]` `(0,2,1,3)`: 16 now → ≥ 64 (bytes read stay at one crossing of the input: each input byte staged once, counted by the probe);
- golden suite green, the every-thread-ran test extended to a sub-tile-quantum TILE case, and no regression across the guard set (one representative per regime × layout × DRAM/L1).

**Verifier notes**: a scheme-local change (no cross-core data). It stays a perf entry because no INPUTS cell fails today. Order after Refinement 1, so the 4-B tile layout (faces of 1 KiB) is in place when the quanta are derived from `element_bytes`.
