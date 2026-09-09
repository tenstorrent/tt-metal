# Verification Report: tilize

Phase 0 verification pass. Op file `ttnn/ttnn/operations/tilize/tilize.py`, design
`op_design.md`, footprint ledger `l1_ledger.md`, golden gate
`eval/golden_tests/tilize/`.

Headline: **all three loud verifier categories are zero**, and the pass moved
`supported_pass` from **41 to 249** golden cells — not by weakening the gate but by
correcting a validate() ordering bug and promoting three axes the kernel already
handled bit-exactly but SUPPORTED under-claimed.

---

## Code Review

### Fixed

**1. `validate()` raised the wrong exception class for 83 unsupported cells.**
`_check_alignment_request` and the `output_padded_shape` checks ran *ahead* of the
per-axis SUPPORTED loop, so a cell whose `alignment` or `pad_mode` was outside
SUPPORTED came back as a `ValueError` about the argument, not as a registry support
refusal. The golden harness decorates out-of-SUPPORTED cells
`xfail(strict=True, raises=NotImplementedError)`, so all 83 landed in
`xfail_wrong_mode`.

Fixed by splitting the malformed-request checks on one question — *is this request
malformed whatever the op eventually supports?*

  * **Unconditional** (`_check_request`, still first): not on device, a layout that
    is neither ROW_MAJOR nor TILE, a TILE input with no `tile=`, a tile whose width
    isn't 32 or whose height isn't a power-of-two fraction of 32. No future
    refinement makes any of these legal.
  * **Support-conditional** (`_check_pad_target`, `_check_alignment_request`, now
    after the SUPPORTED/EXCLUSIONS gate): both are statements about the *padding*
    contract, and padding is a registry-gated axis. While `pad_mode`/`alignment`
    sit outside SUPPORTED the honest refusal is "this op does not do padding yet";
    they re-arm as real `ValueError`s the moment the padding refinement lands.

  This keeps the immutable acceptance test green without touching it:
  `UnsupportedAxisValue` is a `NotImplementedError`, which *is* a `RuntimeError`, so
  `test_tilize_rejects_unaligned_input_without_padding`'s
  `expect_error((ValueError, RuntimeError), "tilize")` still matches.

**2. `_check_pad_target` rejected the legal rank-0 / rank-1 pad target.** 30 of the
83 above were `output_padded_shape rank 2 != input rank 0/1`. A rank-0 or rank-1
input has no tile dims of its own — the pad *synthesizes* them (`rank 0 -> [H,W]`,
`rank 1 -> [H,W]`), which is exactly why `feature_spec.TARGET["rank"]` lists 0 and 1
as pad-only ranks and `tag_alignment` reports `hw_non_aligned` there. The rank check
now left-pads the input shape to 2 before comparing, so a rank-2 target against such
an input is well-formed.

**3. `_spec_of` mis-tagged every sharded call as `nd`.** It tested "does the
memory_config carry an `nd_shard_spec`". Measured on device
(`probes/probe_003.py`–`probe_005.py`): a **live tensor's** `memory_config()` populates *both*
`shard_spec` and `nd_shard_spec` whichever API allocated it, so that test tags every
sharded call `nd` and disagrees with the declared `tag_shard_api` (which reads the
scenario's `scheme`) on every `legacy_2d` row. Replaced with the discriminator that
holds for a freshly built config *and* a live one: nd iff there is an nd spec and
**no** legacy 2-D spec (a fresh nd config also reports `ND_SHARDED`, a live one
reports the equivalent HEIGHT/WIDTH/BLOCK layout). Latent at Phase 0 (both values
are refused), load-bearing the moment the sharding refinement lands. Verified
against fresh-legacy, fresh-nd and both live forms.

**4. The op was unreachable under the name the graded external suite calls.**
`eval/golden_tests/tilize/test_golden_main_tests.py` — the externally-authored case
set, provenance-tracked to main's own tilize tests — dispatches through
`ttnn.tilize(...)`, the name this op occupies. `ttnn.tilize` did not exist, so all
**159** of those cases died at `AttributeError` *before entering the op*, reading as
159 op failures while testing nothing. Bound the public alias in the op package's
`__init__.py` (guarded, self-contained, no `ttnn/__init__.py` edit — `ttnn.operations`
is pkgutil-walked partway through ttnn's own import, so `sys.modules["ttnn"]` already
exists). 10 of those cases now genuinely **pass**; the rest fail honestly with
`UnsupportedAxisValue` naming the axis they need.

### Auto-fixed SUPPORTED (measured under-claim — see Registry Conformance)

`rank` `[4] -> [2,3,4,5,6]`, `low_l1` `[False] -> [False,True]`,
`buffer` `[dram_to_dram] -> all four transitions`.

### Checked and found correct — no change

* **Registry shape.** `INPUT_TAGGERS` (all twelve, all `(inputs, axes)`), `SUPPORTED`
  (every gated axis present), `EXCLUSIONS` (empty, correctly), `validate()` present
  and called as the entry point's first line. `INVALID` is **not** declared in the op
  file. The twelve taggers match `feature_spec.py`'s `Expected INPUT_TAGGERS` block
  rule-for-rule, including the `tile_grid` evaluation order
  (`single_tile -> short_wide -> tall_narrow -> small -> square_large`) and
  `tag_alignment`'s H-against-output-tile-height / W-against-literal-32 asymmetry.
* **One native dispatch.** A single `ttnn.generic_op` over one `ProgramDescriptor`
  (3 kernels, 2 CBs, 0 semaphores). No `to_layout` / `tilize` / `to_memory_config`
  wrapper anywhere; no host round-trip; no second DRAM pass. Structural, visible in
  the descriptor.
* **Helper usage.** `compute_kernel_lib::tilize<bw, in, out>(block_row_extent)` is one
  call per *block* (not per tile-row) — one `tilize_init`/`uninit` and one data-format
  reconfig per block, with the helper's own per-tile-row traversal inside. Symmetric
  (tile-page) mode, correctly paired with the reader's `TilizeGranularity::TILE`.
  `compute_kernel_hw_startup` is present as the helper's documented prerequisite.
  Reader is `dataflow_kernel_lib::read_sticks_for_tilize`, one call per block, using
  `byte_offset_within_page` as the documented wide-W chunking parameter.
* **Writer's raw-API justification holds.** I re-checked both candidate helpers
  against this destination. `write_sticks_after_untilize` addresses by **stick**
  index and advances by `padded_row_bytes`; this destination is TILE-layout pages
  addressed by **tile** index (`row * C + col`), so every address it computes is
  wrong — and its barrier granularity is pinned at one tile-row, which is the
  `write_rows_per_barrier` knob this op must expose. `local_copy_helpers_dataflow` is
  an L1→L1 family requiring `AddressType::LOCAL_L1`. The kernel is that missing block
  helper written out; per the scope boundary, mechanism is not the finding.
* **Writer wrap-safety cap is real, not cargo-culted.** `cb_pop_front` asserts
  `fifo_rd_ptr <= fifo_limit` and wraps only on exact equality. The output CB's
  capacity is `2 * wrpb * bw` pages against a pop quantum of `rows * bw` with
  `rows <= wrpb`; a short final batch (when `block_row_extent % wrpb != 0`) makes the
  quantum vary, so the contiguous-span cap is load-bearing. It can never clamp to 0:
  every pop is a multiple of `bw` pages, so the residual span is too, and is `>= bw`.
* **Syntax / API hygiene.** `void kernel_main()` in all three kernels;
  `api/dataflow/dataflow_api.h` (not the bare path); `TensorAccessor` +
  `TensorAccessorArgs` on both legs, no `InterleavedAddrGen`; reads on NoC0
  (`ReaderConfigDescriptor`) and writes on NoC1 (`WriterConfigDescriptor`) — the
  measured-correct assignment, not the reverse.
* **CB push/pop balance.** `cb_input_rows`: reader pushes `bw` per tile-row, the
  helper pops `bw` per iteration over `block_row_extent` iterations. `cb_output_tiles`:
  the helper pushes `bw` per tile-row, the writer pops `rows_this_batch * bw` and the
  batch loop covers `block_row_extent` exactly. Balanced per block on both.
* **No broadcast anywhere** — tilize has no reuse-shared operand (each input byte
  feeds exactly one output byte), so there is no broadcast dim to get wrong and no
  redundant full-tile fill to remove.

### Blocking-model fidelity

Read against `op_design.md`'s Blocking Model. **No collapsed knob, no DRY violation.**

* Every block factor is a named host constant or a quantity derived from one in
  `derive_plan`: `WRITE_BATCH_MIN_TILES`, `FAST_TILIZE_WIDTH_CAP`,
  `LOW_L1_WIDTH_CAP`, `INPUT_DEPTH_ROWS`, `OUTPUT_DEPTH_BATCHES` are the five
  sources; `block_width_tiles`, `num_w_chunks`, `num_row_groups`,
  `write_rows_per_barrier`, `block_row_bytes` are derived. CB page counts, kernel CT
  args, grid sizing and loop trip counts are all computed *from* the plan object —
  no block value is restated as a literal in two places, and `TILE_WIDTH` has one
  definition that `tilize.py` imports.
* **No CB scales with a whole-op dimension.** The per-core total is
  `2*bw*tb_in + 2*wrpb*bw*tb_out`, in which no tensor dimension appears at either
  `low_l1` setting. `block_row_extent` is correctly absent — the helpers wait/push
  one tile-row at a time, so the row extent is *streamed over*, never spanned.
* **Both halves of the knob are turned, not just the count.** `tile_row`: split
  across `num_row_groups` cores *and* each core's compute loop takes its whole
  row-group in one `tilize` call (coarsest, and free in L1). `tile_col`: split across
  `num_w_chunks` cores *and* the per-core extent is the coarsest divisor that fits
  L1. Neither axis is spread-and-then-walked-one-unit-at-a-time.
* **The schedule reads as blocks in all three kernels.** Reader: one
  `read_sticks_for_tilize` per block (the per-tile-row loop is *internal to the block
  operation's implementation* — explicitly not a finding). Compute: one helper call
  per block. Writer: `wrpb` tile-rows acquired, written and retired per barrier, with
  `wrpb*bw` writes in flight. No kernel handshakes per minimum unit.
* **Work distribution fills the machine, as designed.** Re-measured under `--profile`:
  `[1,1,32,16384]` (R=1, C=512 — the geometry a row-only split collapses on) reaches
  **64/64 cores**, and so does its transposed counterpart `[1,1,16384,32]` (R=512,
  C=1). Both indices of the 2-D block grid really are core-assignment indices, so the
  design's `tile_grid`-complete claim is earned rather than asserted.
* Two recorded deviations from the design (divisor-constrained `block_width_tiles`;
  output depth counted in write *batches*) are both forced by the same hard mechanism
  cap — neither CB endpoint may wrap mid-transfer — and both are documented at their
  point of use and in the ledger. Verified against `llk_io_pack.h` and
  `dataflow_api.h:267-272`. `WRITE_BATCH_MIN_TILES = 4` rather than the design's 8 is
  a measured knob value with the sweep recorded, not a drift.

### Not fixed — noted for a refinement (architectural, out of scope for this pass)

* **The divisor constraint overshoots the read-transaction minimum on rough `C`.**
  See the L1 Ledger Audit below. Folded into perf Refinement 6.
* **`fp32_dest_acc_en` is keyed on the *input* dtype alone.** Correct at Phase 0
  (bf16 only, so it is always `False`) and forward-looking, but it does not yet
  encode the fp32-output rule the design's mechanism-cap table names: an fp32
  **output** drops off `can_use_fast_tilize` and must be paired with
  `Fp32Mode::Lossless` + `fp32_dest_acc_en=true` + `UnpackToDestMode::UnpackToDestFp32`
  on `cb_input_rows`, or it comes back truncated through tf32. Confirmed by probe:
  `fp32 -> fp32` today gives `max_abs_err = 1.95e-3` (tf32 truncation), not the
  bit-identity the contract requires. This is exactly the numeric refinement's job
  (Refinement 5) and is not a Phase 0 defect — fp32 is not in SUPPORTED.
* **`_PLAN_CACHE` is unbounded.** Keyed on shape/dtype/memory-config/tile/low_l1/grid,
  so it is bounded in practice by the number of distinct call signatures a process
  sees, but there is no eviction. Harmless today (entries are tiny), worth a bound if
  the op ever serves an unbounded shape stream.

---

## Registry Conformance

Confirmed present and correctly wired in the op file: `INPUT_TAGGERS` (twelve, each
with the `(inputs, axes)` signature), `SUPPORTED` (every axis the kernel gates on,
including all twelve tagger keys plus `dtype`/`output_dtype`), `EXCLUSIONS` (an empty
list — nothing inside the Phase 0 rectangle is refused, which is the honest state),
and `validate()` — SUPPORTED per-axis, then EXCLUSIONS, both raising from
`ttnn.operations._op_contract`. `validate()` is the entry point's first statement.
The op file does **not** declare `INVALID`.

### Auto-fixes applied to SUPPORTED

Three axes were under-claimed. Each was probed on device before promotion, each came
back **bit-exact against the torch input**, and each required **zero** kernel or
descriptor change:

| Axis | Was | Now | Evidence |
|---|---|---|---|
| `rank` | `[4]` | `[2, 3, 4, 5, 6]` | `derive_plan` folds `shape[:-2]` into `R` generically and the reader indexes sticks linearly, so ranks ≥ 2 exercise no new path. `(32,64)`, `(2,32,64)`, `(2,1,3,32,64)`, `(2,1,1,3,32,64)` all bit-exact. Ranks **0 and 1 stay out** — they have no tile dims of their own and are reachable only with a pad, so they belong to Refinement 2. |
| `low_l1` | `[False]` | `[False, True]` | `low_l1` is a value of the `block_width_tiles` knob (`W_CAP = min(W_FIT, LOW_L1_WIDTH_CAP)`), not a code path — kernels and CBs are byte-identical. `[1,1,32,4096]`, `[1,1,32,8192]` (the `low_l1_forcing_width` geometry) and `[1,1,2048,2048]` all bit-exact at `True`, no OOM, and the A/B pair `helpers.run_tilize` grades is **bit-identical** on the forcing width. The ledger's total contains no tensor dimension at either setting, so the O(1)-in-dims contract is discharged structurally. |
| `buffer` | `[dram_to_dram]` | all four transitions | Placement is a `TensorAccessor` concern on both legs; no kernel names a buffer type. `dram_to_l1`, `l1_to_l1`, `l1_to_dram` all bit-exact. |

These three plus the validate() ordering fix are what took `supported_pass` from 41
to 249 with `supported_fail` still 0.

### INVALID audit (`eval/golden_tests/tilize/feature_spec.py`)

38 entries, all in the `dtype × output_dtype` plane. **Well-formed on all three sanity
rules** — do not change:

* **Single-tensor coupling.** Every entry couples `dtype` (the input's storage
  format) with `output_dtype` (the same tensor's output format). Those are the two
  ends of one tensor's flow through one op, not two different tensors' axes. No
  cross-tensor entry.
* **Structural, not "unimplemented".** tilize re-lays bytes; it does not reinterpret
  them. An int↔float cross asks for a numeric conversion the op never defined, and an
  integer *width* change resizes the datum. Both are outside the mathematical
  contract, so `INVALID` is right and `EXCLUSIONS` would be wrong. `int32 <-> uint32`
  is correctly **kept** (same width, a signedness bit_cast).
* **Universe changes** for every entry (each prunes real cells from the cartesian).
* **Canonical bf8b + ROW_MAJOR: correctly handled by omission, not by an entry.**
  `TARGET["dtype"]` (the *input* dtype) simply does not list `bfloat8_b` or
  `bfloat4_b`, with the reason stated inline: block-float has no ROW_MAJOR form, so
  either can only ever be an output. That is stronger than an INVALID row — the cells
  are never generated. Not a gap.
* No weight axes, so the norm-like no-weight canonicalization rule does not apply.

**One candidate gap, flagged not acted on** (this pass does not edit `feature_spec.py`;
raise via `/golden-tests` if you agree). `TARGET["dtype"]`'s own comment says
`fp8_e4m3` is *"ROW_MAJOR-only … an INPUT that tilizes to any float output and never
an output itself"* — but the `tile_geometry_retile` cases build a **TILE** input, so
the cartesian generates `{dtype: fp8_e4m3, in_tile_height: 32|16|8|4|2|1}`: an fp8
tensor in TILE layout, which by that same comment has no TILE form.
`helpers.create_ttnn_input_tensor` would have to materialize one. On this box those
cells are masked by `skip_if_fp8_unsupported` (Wormhole), which hides rather than
resolves it — they will surface on Blackhole. If the layout really is impossible,
`{"dtype": ttnn.fp8_e4m3, "in_tile_height": <each retile height>}` belongs in
`INVALID`. (This is the same item `op_design.md`'s "Structural impossibilities"
section flagged; the verification pass confirms it is still open.)

### Harness observation (not an op defect): 137 declared/captured axis mismatches

`merge_axes` reports `137 row(s) with declared/captured axis mismatch (classify_call
bug)`. Both classes are on the harness side and neither changes any decoration
(decoration comes from the *declared* taggers):

* **84 × `shard_api`: declared `legacy_2d`, captured `nd`.** This is finding 3 above,
  in `eval/golden_tests/tilize/axes.py:_spec_of`, which still tests "has an
  `nd_shard_spec`". I fixed the op's copy; `axes.py` is gate code and out of bounds
  for this pass. The fix is the same three-line discriminator. Worth raising, because
  it will mis-attribute every legacy-2D row once sharding lands.
* **53 × `low_l1`: declared `True`, captured `False`.** `helpers.run_tilize` runs every
  `low_l1` scenario at **both** settings (that is the bit-identity A/B the rules
  require) and the last call recorded wins. An artifact of the A/B, not a bug.

---

## L1 Ledger Audit

`l1_ledger.md` read against the CB declarations in `tilize_program_descriptor.py`.

1. **Ledger currency — clean.** Two CBs declared, two rows. `cb_input_rows` capacity
   `input_depth_rows * block_width_tiles` pages and `cb_output_tiles` capacity
   `output_depth_batches * write_rows_per_barrier * block_width_tiles` pages match
   `TilizePlan.input_cb_pages` / `output_cb_pages` exactly, and `total_size` is
   `pages * page_bytes` in both. Page formats match the descriptors
   (`input_tensor.dtype`, `output_tensor.dtype`), and `tb_out` really is
   `output_tensor.buffer_page_size()` read off the buffer rather than computed.
2. **Capacity vs live set, both directions — clean.** *Over:* both CBs sit at exactly
   2× their live set, and both reasons are recorded and non-hand-wavy — reader/compute
   double buffering for the input (catalog-measured 1.24–1.99×), and for the output
   *both* a compute overlap window *and* the wrap requirement that the capacity be an
   integer number of write-batch quanta. Neither is a whole-block buffer around a
   one-tile live set. *Under:* the axis accounting is correct and I verified it against
   the helpers rather than taking it on trust — `block_row_extent` is tagged `streams`
   in both rows and is correctly absent from both capacities, because
   `compute_kernel_lib::tilize` waits/pushes/pops one tile-row per iteration
   (`tilize_helpers.inl:233-259`) and `read_sticks_for_tilize` reserves/pushes one
   tile-row per iteration (`tilize_helpers_dataflow.inl:110-127`). The only spanned
   axes (`tile_col` on both, a `wrpb`-deep `tile_row` window on the output) do have
   capacity scaling with them. No collapsed extent, no inflated streaming window.
3. **Page format vs DEST width — clean at Phase 0.** `fp32_dest_acc_en` is set iff the
   input dtype is `float32`, so at bf16 there is no `Float32` page under a 16-bit DEST
   and no 16-bit page under an fp32 DEST. tilize performs no arithmetic, so there is no
   accumulation to truncate at a phase boundary. Forward note: once `dtype=` can widen
   (bf16 in → fp32 out) the output page would be `Float32` under a 16-bit DEST. That is
   *not* a precision loss here (the DEST value is already the tensor's own bf16 value,
   losslessly widened at pack) but it does double the bytes through the packer and in
   every read of that CB — the numeric refinement should decide it deliberately rather
   than inherit it.
4. **Disjoint lifetime — none, and the non-sharing is justified.** Both CBs are live for
   the whole program and deliberately pipelined against each other. Each `Shares with /
   why not` cell carries three concrete reasons, one of them a citation
   (`compute_kernel_lib::tilize` `static_assert`s `input_dfb != output_dfb`, because
   tilize is not an in-place permutation). No blank cell.
5. **Bounds and closed form — clean.** Every symbol in the total appears in the symbol
   table with a bound and its predicate; the total is closed-form in
   `{input_depth_rows, output_depth_batches, write_rows_per_barrier, block_width_tiles,
   tb_in, tb_out}`; `W_FIT` is the *inversion* of that closed form rather than a search
   that settled. `budget` is read from the device
   (`ttnn.get_max_worker_l1_unreserved_size()`), never a literal. `R`, `C`,
   `num_w_chunks`, `num_row_groups` are explicitly recorded as absent from every
   capacity expression — that absence is the whole `low_l1` proof, and it is what let
   me promote `low_l1=True` on evidence rather than hope.

**Block-size defaults.** Rule 2 is followed in its two ordered steps: spread the split's
work units across the full grid first (`w_chunks_for_occupancy = min(C, ceil(num_cores/R))`),
then take the coarsest block that fits (`w_chunks_for_l1 = ceil(C/W_CAP)`), with
`num_w_chunks` **minimized** as `max(...)` of the two. The departure from a full-width
block is forced by occupancy, not by a budget solve settling, and the inventory was
minimized first (two CBs, the thread-boundary floor, with the in-place candidate ruled
out by citation). Both lever departures — `WRITE_BATCH_MIN_TILES` 8 → 4, and
`INPUT_DEPTH_ROWS` *kept* at 2 — are backed by recorded on-device sweeps, including
the informative negative result (input depth is flat across {1,2,3,4} on both wide
shapes, which says the reads are the wall).

**One accuracy defect found and fixed in the ledger** (fixed in place, not filed as a
refinement, per the filing rule). The data-movement budget closed with *"the fewest
read transactions of any split that fills the grid"*. That is exact on smooth `C` but
an overshoot on rough `C`, because the divisor constraint means `num_w_chunks` can only
land *above* the target: on the logits row `[1,1,1,50304]`, `C = 1572 = 2²·3·131`, the
coarsest divisor `<= 1572/64 = 24` is **12**, so the split lands on **131** chunks of 12
tiles instead of the target 63 of 25 — DRAM crossings unchanged at 1 in / 1 out, but
**2.08× the read-transaction minimum**, at 768 B per read instead of 1600 B. The ledger
now quantifies this, and records the escape (two disjoint core ranges in the one
`ProgramDescriptor` — full-width cores and tail cores, each with its own CB sizing and
`block_width_tiles` CT arg — restores the design's ragged tail without weakening the
wrap invariant). Folded into perf **Refinement 6**, whose scope already touches these
buffers. The perf-flagged shape is unaffected (`C = 512` is smooth).

**Per-core footprint, as an expression:**

```
L1_per_core = 2 * block_width_tiles * tb_in                      # cb_input_rows
            + 2 * write_rows_per_barrier * block_width_tiles * tb_out   # cb_output_tiles
```

scaling with `block_width_tiles` (the `tile_col` extent), the two depth knobs, `tile_h`
and the element widths — and with **no** tensor dimension, at either `low_l1` setting.
Measured range across the suite: 20 KiB (`[1,1,32,32]`, `[1,1,16384,32]`) to 512 KiB
(`[1,1,2048,2048]`, `bw=64`), against a 1.43 MiB budget. 64 KiB on the perf-focus shape.

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py`, bfloat16,
rank 4, tile-aligned, interleaved DRAM→DRAM (the Phase 0 rectangle). tilize performs no
arithmetic, so the contract is bit-identity and every figure below is an equality the
test asserts, not a tolerance it tolerates.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio (median, p5, p95) |
|-------|-----|-------------|--------------|------------------|----------------------------------|
| (1,1,32,32) `single_tile` | 1.0 | 0.0 | 0.0 | 0.0 | 1.0, 1.0, 1.0 |
| (1,1,64,128) `multi_tile` | 1.0 | 0.0 | 0.0 | 0.0 | 1.0, 1.0, 1.0 |
| (2,3,128,256) `leading_fold` | 1.0 | 0.0 | 0.0 | 0.0 | 1.0, 1.0, 1.0 |
| (1,1,1024,1024) `square_large` | 1.0 | 0.0 | 0.0 | 0.0 | 1.0, 1.0, 1.0 |
| (1,1,32,16384) `short_wide` (perf focus) | 1.0 | 0.0 | 0.0 | 0.0 | 1.0, 1.0, 1.0 |

`comp_allclose` reports `Max ATOL Delta: 0.0, Max RTOL Delta: 0.0` on every row.

**Assessment**: exact. The got/true ratio is identically 1.0 with zero spread on all
five shapes, which is the scale-bug detector coming back clean — no constant-ish
non-1.0 ratio (a leaked scaler / broadcast-reduce mistake) and no spread around 1.0
(rounding). The `leading_fold` and `square_large` rows are the load-bearing ones: they
are where a mis-derived row count (`floor(N*H/tile_h)` instead of
`N*ceil(H/tile_h)`) or a wrong output page id would show up as displaced values while
still correlating well.

**Recommended tolerances**: PCC = 1.0, `rtol = atol = 0` for every no-cast float and
integer cell. These are floors, not budgets — do not loosen them in a refinement. The
tolerances that *will* need a budget are the lossy output casts the numeric refinement
adds; measured here for reference on `(1,1,64,128)`: bf16→bf8b `max_abs_err = 0.031`,
bf16→bf4b `max_abs_err = 0.926`, bf16→fp32 exact.

---

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/tilize/ /tmp/tilize_verify3` then
`python3 -m eval.verify_supported /tmp/tilize_verify3 ttnn.operations.tilize`.
Trimmed copy of the report committed alongside as `verifier_report.json`.

| Category | Count | |
|---|---|---|
| `supported_pass` | **249** | ✓ (was 41 before this pass) |
| `xfail_expected` | 1430 | ✓ refused with a support refusal, as declared |
| `invalid_skipped` | 2090 | ✓ |
| `xfail_other` | 304 | ✓ arch gates only — `fp8_e4m3 requires Blackhole` (196) and `retile requires Blackhole` (108). Not signal. |
| `no_axes_found` | 475 | the non-registry suites (`test_golden_main_tests.py`, `test_regression.py`, and skipped `test_translated.py` rows) — see below |
| **`supported_fail`** | **0** | ✓ ship |
| **`xpass_drift`** | **0** | ✓ ship |
| **`xfail_wrong_mode`** | **0** | ✓ ship (was 83) |
| `supported_marked_xfail` | 0 | ✓ |
| `invalid_unexpected` | 0 | ✓ |

Run totals: **266 / 4548 passed**, 152 failed, 4 errors, 2696 skipped, **0 hangs**.
Progression across this pass: 41 → 51 (validate() ordering + `ttnn.tilize` alias) → 266
(SUPPORTED promotions), with **zero** tests lost at any step.

### The 152 failures and 4 errors, itemised (all outside the registry categories)

None of these is an op defect at Phase 0 — every one names an axis value the refinement
queue already carries:

* **145 in `test_golden_main_tests.py`** — the externally-authored graded case set. All
  now reach the op (they previously died at `AttributeError`) and refuse honestly:
  `shard_api='nd'` 84, `shard_api='legacy_2d'` 13 → Refinement 1; `dtype` fp32/int32/
  uint32/uint16 32 → Refinement 5; `pad_mode` auto/explicit 10 and `rank=0` 3 →
  Refinement 2.
* **10 in `test_regression.py`** — not registry-driven, so undecorated; fp32 /
  uint16 / int32 passthrough and pad-value extremes. Refinements 5 and 2.
* **4 setup errors** (2 in `test_golden_main_tests.py`, 2 in `test_golden_main_trace.py`)
  — `ValueError: Cannot use @pytest.mark.use_module_device with
  @pytest.mark.parametrize('device_params', ...)`. A suite-authoring conflict that
  fails **before the op is entered**; `test_golden_main_trace.py`'s own docstring
  documents the split that was supposed to avoid it. Gate-side, not fixed here.

---

## Recommendations

**Queue ordering** is hardest-structural-first, which the difficulty ranking and the
"land the hard refinement on the small test surface" rule both point to. The
`dtype × output_dtype` cartesian is 18 legal pairs, so the numeric refinement multiplies
*every* scenario by 18 — it is by far the largest cell-count unlock and also the
cheapest to build, which is exactly why it goes **last**: landing it first would force
every later structural refinement to be built and debugged across 18 dtype pairs.

**The perf anchor needs nothing unlocked.** `feature_spec.LOOSE_CASES`'s
`attention:`-flagged entry is `[1,1,32,16384]` at bf16→bf16, rank 4, tile-aligned,
interleaved DRAM→DRAM, 32×32 tile — every facet of that contract is already in
Phase 0 SUPPORTED, and the shape already reaches 64/64 cores. So the first perf slot
(Refinement 3) can measure and optimize the real config from day one, and Refinements 1
and 2 are free to be ordered purely by difficulty.

**Measured perf headroom** (this pass, `--profile`, 8×8 Wormhole, fresh cache, two
dispatches each): perf focus `[1,1,32,16384]` **13106 / 14133 ns** at 64/64 cores
(≈160 GB/s over the 2 MiB moved); transposed pair `[1,1,16384,32]` **21220 / 20268 ns**
at 64/64 (≈101 GB/s). `double_buffer/report.md` puts an untuned 64-core DRAM→DRAM
stream at 190.8 GB/s (≈ this part's peak), so there is roughly **1.19×** left on the
perf-focus geometry and **1.9×** on its transposed counterpart. The gap between the two
is transaction shape, not occupancy — both fill the grid, and the narrow side's 64 B
reads are the tensor's own stick width, which the blocking cannot coarsen (consecutive
ROW_MAJOR sticks live on different DRAM banks). That makes the transposed entry a
`split_reader` candidate (both data-movement RISC-Vs issuing) rather than a blocking
candidate, and it is filed that way.

**Cross-cutting concerns for the implementer:**

* **`eval/golden_tests/tilize/axes.py:_spec_of` carries the same nd-vs-legacy bug I
  fixed in the op** (finding 3). Until it is fixed, `merge_axes` will keep reporting 84
  `shard_api` mismatches and the *captured* axes on every legacy-2D row will read `nd`.
  The xfail decoration is unaffected (it comes from the declared taggers), so Refinement
  1 is not blocked — but do not chase the mismatch report as if it were your bug.
* **The fp32 output path is a correctness trap, not a precision knob.** `fp32 -> fp32`
  measures `max_abs_err = 1.95e-3` today because `can_use_fast_tilize` truncates through
  tf32. The golden oracle for a byte re-lay is bit-identity, so Refinement 5 must pair
  `Fp32Mode::Lossless` with `fp32_dest_acc_en=true` *and*
  `UnpackToDestMode::UnpackToDestFp32` on `cb_input_rows` — all three, or the cell
  fails. This is the one place in this op where the usual "prefer Fast, downstream FPU
  truncates anyway" advice is wrong: there is no downstream FPU op, the output *is* the
  product.
* **`uint8` looks broken, not merely unsupported.** A probe of `uint8 -> uint8` on
  `(1,1,64,128)` came back with `max_abs_err = 99` on data in `[0, 100)` — i.e. the
  output is not the input. Every other integer width (`uint32`, `int32`, `uint16`) is
  bit-exact. Refinement 5 should treat `uint8` as an investigation rather than a
  list-widening, and land it in `EXCLUSIONS` if the 1-byte element size does not survive
  the helper's `elem_size = tile_size / tile_hw` derivation.
* **The CB L1 budget does not account for L1-resident tensors.** `derive_plan` sizes
  against `ttnn.get_max_worker_l1_unreserved_size()`, but with `buffer` now including
  `l1_to_l1` the tensors themselves compete for the same L1. No cell in the suite trips
  it (L1 is not the binding constraint on any of these shapes — the grid is, and `W_FIT`
  is 176 at bf16 against an occupancy-driven `bw` of at most 64), but it is a latent
  OOM on a large L1-interleaved tensor. Refinement 1 already touches placement; fold the
  budget adjustment in there.
* **`PROPERTIES` is an extra declaration** beyond the registry's four (`multi_core`,
  `bounded_cb`, both `source: declared`). Harmless and both claims are now *measured*
  rather than declared — 64/64 cores on the transposed pair, and a footprint with no
  tensor dimension in it. Left as-is; if a refinement changes either, update it.
