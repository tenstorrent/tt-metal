# Verification Report: rms_norm

Verified 2026-07-28 on **blackhole_p150b** (11 × 10 = 110-core compute grid, AICLK 1350 MHz).

- Acceptance suite: `scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/unit_tests/operations/rms_norm/` → **145 passed**
- Golden suite: `eval/eval_test_runner.sh eval/golden_tests/rms_norm/ <results>` → **770 passed / 0 failed / 0 errors / 0 hangs**
- Verifier CLI: `python3 -m eval.verify_supported <results> ttnn.operations.rms_norm` → **all three loud categories 0**

---

## Code Review

### Fixed in this pass

| # | Issue | Where | Fix |
|---|---|---|---|
| 1 | **DRY violation on the shared compile-time-arg block.** The reader's and the writer's CT arg lists were two separate literal `regime + knobs + [...]` expressions building the *identical* 18-entry block. Both kernels read the same indices (`WT_CHUNK` at 7, `NW` at 9, `HT_BLOCK` at 10, `TensorAccessorArgs<18>`), so a knob added to one list and not the other would silently re-index the other kernel's args — the exact drift the single-source rule exists to prevent. | `rms_norm_program_descriptor.py:create_program_descriptor` | Built **once** as `dataflow_ct_args`; reader and writer each `list(...)` it and append only their own accessor args. Added `DATAFLOW_ACCESSOR_ARG_BASE` + an assert that the block is still 18 long, so the `TensorAccessorArgs<18>` both kernels hard-code can't silently desync from the host. |
| 2 | **Buffer-depth knob applied where it cannot pay.** `X_RESIDENT_DEPTH` (the resident-input-strip depth) was raised to 2 on the ROW_MAJOR path too. On that path `cb_input_tiles` is produced by *compute's own* `tilize()` and consumed by compute's `square`/`mul` — one RISC on both ends, so a second strip can never be filled ahead. It bought nothing and spent `HT_BLOCK * Wt * tile_bytes` of L1 that can evict `GAMMA_RESIDENT`. | `rms_norm_program_descriptor.py:_Blocking.__init__` | Gated the depth bump on `not self.is_rm`, with the reason recorded inline next to the existing `grid_full` / `nh_core_max` gates. |
| 3 | **Grid derived twice.** `create_program_descriptor` computed the grid, sized the blocking against `grid.x*grid.y`, and then `_core_assignment(device, ...)` re-derived the grid from the device. Two sources for the one quantity `grid_full` keys off. | `rms_norm_program_descriptor.py:_core_assignment` | `_core_assignment(grid, ht_total)` now takes the caller's grid object; the two unused return values (`num_cores`, `grid_cores`) were dropped. |

All three are single-source/DRY repairs on the blocking model; none changes the math. Acceptance (145) and golden (770/0) are green after them.

### Checked and found correct (no change needed)

- **Block factors are parameters, not constants.** `L1_BLOCK_BUDGET_BYTES` is the one knob; `TILE_BLOCK_BUDGET`, `WT_CHUNK`, `HT_BLOCK`, `NW`, `WT_LAST`, every CB page count and every kernel loop trip count derive from it. `X_DEPTH` / `OUT_DEPTH` / `GAMMA_DEPTH` / `X_RESIDENT_DEPTH` are named depth knobs. The grid comes from `device.compute_with_storage_grid_size()`, never an inlined core count.
- **No CB is unconditionally sized to a whole-op dimension.** The only two `Wt`-sized CBs — `cb_input_tiles` under `X_RESIDENT` and `cb_gamma` under `GAMMA_RESIDENT` — are the sanctioned **predicate-guarded resident fast path** with a bounded streaming fallback, and the host asserts the final per-core total against `L1_CB_BUDGET_BYTES` (halving the block budget and re-deriving if it misses). Measured totals on the perf shapes: 408–920 KB against a 1 100 KB budget.
- **The split is not half-turned.** The independent `ht` axis is split across the grid (count) *and* the per-core compute loop runs a coarse `HT_BLOCK × WT_CHUNK` block (size) — 32 tiles per helper call on bf16 TILE, never one tile-row at a time. `HT_BLOCK` collapses to 1 only when `NW > 1`, which is the load-bearing `TileOffset` invariant (R7), and both the host and all three kernels `static_assert` it.
- **Design conformance / does it fill the machine.** `ttnn.split_work_to_cores(grid, ht_total, row_wise=True)` — `row_wise=True` is the measured-fast line orientation (`examples/noc_placement`, 2.91×). Prefill shapes occupy all 110 cores. Both dataflow halves batch: the reader issues `X_READ_CHUNKS × ht × WT_CHUNK` reads per barrier and the writer `ht × WT_CHUNK` writes per barrier — well above the 4–8-tile plateau in `examples/double_buffer`. Streaming CBs are depth-2. **Decode shapes (`ht_total == 1`) run on exactly 1 of 110 cores** — that is not a conformance break: the design commits phase 1 to the independent-axis split and names the dependent-`W` cross-core split as Lamp L1, a deliberate scheme-change. It is filed as Refinement 2 and is the headline perf gap.
- **Helper usage.** Every compute phase goes through `compute_kernel_lib` (`tilize`/`untilize`, `square`/`mul`/`eltwise_chain`, `reduce`/`reduce_mean`, `AddUnary`+`Rsqrt` fused into one dst-sync window). The two drops from the `square`/`mul` convenience wrappers to `eltwise_chain` in the resident regimes are justified: the wrappers (`eltwise_convenience.hpp:60-120`) have no `TileOffset` template parameter, so they cannot express the resident-strip chunk offset. That is an overload choice, not a raw-LLK substitution. The reader/writer TILE paths use `TensorAccessor` + `noc_async_read_page`/`noc_async_write_page` (the non-deprecated generic form) because no dataflow helper expresses a tile-page read of an already-tiled tensor — `read_sticks_for_tilize` is stick-indexed. `mcast_pipe.hpp` is not applicable: phase 1 issues zero inter-Tensix traffic.
- **CB sync ledger.** Push count = wait count on every CB. `cb_rms_recip` is `HeldBulk` across pass B and explicitly popped at phase 8 (R2); `cb_input_tiles` under `X_RESIDENT` is `CallerManaged`, waited cumulatively and popped once per row-block (R8); `cb_scaler` is pushed once and never popped (R4). Design risk **R3** (does `Accumulate::at_last` drain `cb_partials`?) resolves in the helper's favour — `reduce_helpers_compute.inl:375,493` wait and pop the accumulator itself — so the extra defensive `cb_pop_front` the design allowed for is correctly *absent*; multi-row-block × `NW>1` shapes (e.g. `(1,1,8192,5120)`: 3 row-blocks × 5 chunks per core) run clean.
- **Kernel hygiene.** `void kernel_main()` in all three kernels; includes are `api/dataflow/dataflow_api.h` / `api/compute/compute_kernel_hw_startup.h` (not the bare legacy paths); `TensorAccessor` throughout, no `InterleavedAddrGen`; `compute_kernel_hw_startup` is the first statement of the compute kernel and never inside a loop.
- **Broadcast correctness.** Phase 5 uses `BroadcastDim::Col` for the column-shaped `REDUCE_ROW` result and phase 6 `BroadcastDim::Row` for the row-0-valid `gamma` — no CB is filled with full tiles of repeated data. `OperandKind` switches to `Col`/`Row` only when `HT_BLOCK > 1`, which is when the block is genuinely 2-D.

### Documented deviations from `op_design.md` (implementation is right, the design text is stale)

1. **`cb_scaler` format is the input dtype, not a fixed `Float16_b`** (design R4). `reduce_accumulate_via_add`'s `fold_partial_last` reads the mask through srcB via `llk_unpack_AB<ROW>` *without* reconfiguring srcB, and reduce entry already programmed srcB at the input format — so a `Float16_b` mask under an `fp32` input is reinterpreted as fp32. The implementer found this as 82 golden failures (all `float32 × w_non_aligned`) and fixed it correctly; `test_rms_norm_debug.py` pins it with hand-calculable all-ones inputs. **Design R4 should be amended** — its `Float16_b` is right for the `ReduceTile` datapath, wrong for `AccumulateViaAdd`.
2. **`cb_output_rm` page size is `tile_bytes`, not `C*32*elem_bytes`** (design §6). `untilize_helpers.hpp` states plainly that the untilize helper "always uses symmetric (tile-sized) entries for both input and output DFBs", and `write_sticks_after_untilize` derives `width_in_tiles` from the CB's tile geometry and pops that many pages. The implementation matches the helper contract; the design table's row-page size would deadlock. (`cb_input_rm` genuinely *is* row-paged — `TilizeGranularity::ROW` — and is sized that way.)
3. **`WT_CHUNK` is the coarsest *divisor* of `Wt` under the budget, not `min(Wt, TILE_BLOCK_BUDGET)`.** The implementer documented why at the top of `rms_norm_program_descriptor.py`: a mixed `[C, C, …, L]` push sequence would straddle `fifo_limit`, and the RM row-page size *is* the chunk width so two widths cannot both satisfy the tilize stride. The consequence is bounded and correct — only a prime `Wt` above the budget degrades to more chunks. `WT_LAST` stays an emitted knob and every kernel `static_assert`s `WT_LAST == WT_CHUNK`.

### Not fixed — advisory only

- `PROPERTIES["multi_core"]` is tagged `"declared"`. `test_report_blocking` *prints* the resolved core count (110 on the prefill shapes) but nothing asserts it, so per `eval/op_template.py`'s rule the tag is honest as-is. Promoting it to `"verified"` needs a one-line assertion, which is cleaner to land together with Refinement 2's core-assignment change.
- `EXCLUSIONS = [{dtype: float32, fp32_dest_acc_en: False}]` is currently *unreachable*: `SUPPORTED["fp32_dest_acc_en"] == [True]`, so `validate()` refuses on the SUPPORTED check first. This is deliberate and load-bearing the moment Refinement 1 adds `False` — keep it.
- `cb_gamma_rm` is allocated `TILIZE_ROWS` (32) row-pages although the single-stick gamma tilize only ever needs 1. It is a few KB and charged conservatively in `BLOCK_CB_UNITS`; not worth churn.

---

## Registry Conformance

**Confirmed present and correctly wired in `ttnn/ttnn/operations/rms_norm/rms_norm.py`:**

- `INPUT_TAGGERS = {"alignment": tag_alignment, "rank": tag_rank}` — both taggers take `(inputs, axes)`, both read only `inputs[0]`. W-not-divisible-by-32 wins over H, matching `feature_spec.py`'s documented contract.
- `SUPPORTED` — nine axes, one per finite `TARGET` axis plus both taggers: `dtype`, `fp32_dest_acc_en`, `layout`, `alignment`, `rank`, `gamma_mode`, `gamma_dtype`, `gamma_layout`, `memory_layout`. Nothing the kernel gates on is missing. `"none"` is present on `gamma_dtype` / `gamma_layout` (the absent-weight sentinel, always legal).
- `EXCLUSIONS` — one cell-dict, `{float32, fp32_dest_acc_en=False}`, an op-side permanent refusal per `references/precision_convention.md` (correctly *not* an INVALID entry).
- `validate()` — argument errors (`rank`, `gamma`, `epsilon`) first, then the non-axis refusals (`program_config`, output `memory_config.memory_layout`), then **SUPPORTED per-axis**, then **EXCLUSIONS cell-level**. Both raise `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`. Order is correct.
- `rms_norm()` calls `validate(...)` as its **first statement**, before the output allocation and before any program-descriptor work.
- **The op file does NOT declare `INVALID`.** Confirmed by inspection; INVALID is sourced only from `eval/golden_tests/rms_norm/feature_spec.py`.

**Auto-fixes applied to SUPPORTED from XPASS evidence:** none required — `xpass_drift` is 0.

### INVALID audit (`eval/golden_tests/rms_norm/feature_spec.py`)

Structurally sound entries (no action):

- ✅ Canonical **bf8b + ROW_MAJOR** present for the activation (`{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}`) *and* for the weight (`{gamma_dtype: bfloat8_b, gamma_layout: ROW_MAJOR_LAYOUT}`). Each couples two axes of the *same* tensor. Correct.
- ✅ The **no-weight canonicalization** block is complete and coupled both ways: `gamma present ⇒ never "none"` (2 entries) and `gamma absent ⇒ sentinel only` (5 entries). Exactly one canonical `("none","none")` cell survives, which is what the runtime tagger in `axes.py` produces.
- ✅ No entry that is merely "my kernel doesn't do this yet" *for a shipped axis value* — every such case is correctly in the op's `EXCLUSIONS` or simply outside `SUPPORTED`.

Two clusters to raise with the golden-test author (**I did not edit `feature_spec.py`**):

1. **Cross-tensor coupling — recommend removing.** The three "author-scoped" entries
   `{layout: ROW_MAJOR_LAYOUT, memory_layout: <*_SHARDED>, gamma_layout: TILE_LAYOUT}`
   couple the **activation's** `layout` and `memory_layout` with the **gamma tensor's** `gamma_layout`. That is the canonical INVALID authoring mistake (the conv2d_nhwc `layout × weights_dtype` pattern) — there is no kernel-level coupling that makes an RM activation on a sharded buffer *impossible* when the weight happens to be tiled. The file itself labels them "NOT structural impossibility … deliberately parked in INVALID to keep them out of the refinement backlog", which is precisely the EXCLUSIONS/backlog role, not INVALID's. Effect today: **1 260 cells are skipped that should be xfailing** (counted as cells no *other* INVALID entry also matches), so the sharded refinements' true reach is under-reported. Recommend deleting the three entries; if the op should genuinely refuse them for now, the op-side `EXCLUSIONS` is the honest home.
2. **"Out of scope for now" in INVALID — recommend re-homing.** `{dtype: bfloat8_b, alignment: w_non_aligned}` and `{dtype: bfloat8_b, alignment: h_non_aligned}` encode a capability gap ("bf8b block-quantization + a masked/padded reduce is out of scope"), not an impossibility. Under the registry model that belongs in the op's `EXCLUSIONS` once bf8b enters `SUPPORTED` — which is exactly where the `/numeric-formats-metal` skill routes the standard `bfloat8_b + non_tile_aligned` failure. They uniquely account for **720 skipped cells**. **Refinement 1 is written to work either way**: if they move to EXCLUSIONS it covers those 720 cells too; if they stay in INVALID the refinement is unchanged and simply narrower.

Neither issue blocks shipping — both currently *shrink* the exercised universe, so nothing is over-claimed.

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py` — 16 cells (4 shapes × {bf16, fp32} × {TILE, ROW_MAJOR}), gamma present, `epsilon=1e-6`, default config (HiFi4 / `fp32_dest_acc_en=True`). PCC via `assert_with_pcc`, abs errors via `comp_allclose`, plus the got/true **ratio spread** (median, p5, p95) as the scale-bug detector.

### TILE layout

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ratio med (p5 / p95) |
|-------|-------|-----|-------------|--------------|------------------|----------------------|
| (1,1,32,64) | bf16 | ≥ 0.995 ✓ | 2.40e-02 | 1.21e-03 | 2.36e-03 | 0.999672 (0.99568 / 1.00349) |
| (1,1,64,128) | bf16 | ≥ 0.995 ✓ | 2.79e-02 | 1.12e-03 | 2.34e-03 | 0.999550 (0.99562 / 1.00361) |
| (2,4,128,512) | bf16 | ≥ 0.995 ✓ | 4.64e-02 | 1.22e-03 | 2.39e-03 | 0.999664 (0.99577 / 1.00360) |
| (1,1,32,4096) | bf16 | ≥ 0.995 ✓ | 4.81e-02 | 1.27e-03 | 2.44e-03 | 0.999690 (0.99584 / 1.00388) |
| (1,1,32,64) | fp32 | ≥ 0.999 ✓ | 1.35e-02 | 8.11e-04 | 1.52e-03 | 0.998728 (0.99744 / 0.99984) |
| (1,1,64,128) | fp32 | ≥ 0.999 ✓ | 1.38e-02 | 7.86e-04 | 1.46e-03 | 0.998827 (0.99754 / 0.99994) |
| (2,4,128,512) | fp32 | ≥ 0.999 ✓ | 2.00e-02 | 7.72e-04 | 1.40e-03 | 0.998800 (0.99749 / 0.99992) |
| (1,1,32,4096) | fp32 | ≥ 0.999 ✓ | 1.81e-02 | 4.69e-04 | 9.30e-04 | 0.999405 (0.99819 / 1.00049) |

### ROW_MAJOR layout

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ratio med (p5 / p95) |
|-------|-------|-----|-------------|--------------|------------------|----------------------|
| (1,1,32,64) | bf16 | ≥ 0.995 ✓ | 2.40e-02 | 1.21e-03 | 2.36e-03 | 0.999672 (0.99568 / 1.00349) |
| (1,1,64,128) | bf16 | ≥ 0.995 ✓ | 2.79e-02 | 1.12e-03 | 2.34e-03 | 0.999550 (0.99562 / 1.00361) |
| (2,4,128,512) | bf16 | ≥ 0.995 ✓ | 4.64e-02 | 1.22e-03 | 2.39e-03 | 0.999664 (0.99577 / 1.00360) |
| (1,1,32,4096) | bf16 | ≥ 0.995 ✓ | 3.70e-02 | 1.29e-03 | 2.46e-03 | 1.000297 (0.99631 / 1.00447) |
| (1,1,32,64) | fp32 | ≥ 0.999 ✓ | 1.48e-02 | 1.02e-03 | 1.84e-03 | 0.998396 (0.99705 / 0.99954) |
| (1,1,64,128) | fp32 | ≥ 0.999 ✓ | 1.70e-02 | 1.01e-03 | 1.80e-03 | 0.998477 (0.99712 / 0.99964) |
| (2,4,128,512) | fp32 | ≥ 0.999 ✓ | 2.17e-02 | 9.33e-04 | 1.65e-03 | 0.998533 (0.99717 / 0.99972) |
| (1,1,32,4096) | fp32 | ≥ 0.999 ✓ | 1.68e-02 | 4.10e-04 | 8.28e-04 | 0.999772 (0.99838 / 1.00096) |

**Assessment.** Every cell clears the golden-suite gates with room to spare (bf16 rel-RMS 2.4e-3 vs a 4.0e-2 gate; fp32 9e-4–1.8e-3 vs a 2.0e-2 gate). Layout is essentially free — TILE and ROW_MAJOR agree to the last digit on bf16 and to ~20 % on fp32 (the RM path adds a tilize/untilize round-trip). Error does **not** grow with `W`: the widest shape `(1,1,32,4096)`, which is also the only chunked-reduce (`NW = 5`) cell, is the *most* accurate fp32 row — evidence that the `AccumulateViaAdd` cross-chunk accumulation through fp32 DEST is not losing anything.

**Scale-bug triage.** The fp32 rows show the shape the triage table warns about — a ratio median systematically below 1 (0.9984–0.9994) — so it was run down explicitly rather than assumed:

- `probes/probe_001.py` measures the implied `mean(x²)` **without gamma** across `W ∈ {64,128,1024}`: median ratio 0.99989–1.00030, i.e. the *reduce* is unbiased. The within-row spread of the implied RMS is ~1.9e-3 and the per-row scale is constant to that tolerance (also pinned by `test_rms_norm_debug.py::test_monotonic_ratio_is_position_independent`).
- The bias therefore enters at the **FPU multiplies** (`x·(1/rms)` and `·gamma`), whose operands pass through 19-bit (1-8-10) SrcA/SrcB registers. Truncating a magnitude biases it toward zero, and two truncated operands per multiply × two multiplies gives ≈ −1e-3 — matching the observed −1.2e-3 exactly.
- **Verdict: not a scale/structural bug.** The offset (1.2e-3) is *smaller* than the random spread (2.4e-3), which is the opposite of the tight-cluster-at-a-constant signature; and it is absent once gamma is dropped. It is ordinary Tensix fp32 datapath truncation, ~30× inside the fp32 gate. Not routed to a precision refinement; recorded in Recommendations below because `UnpackToDestFp32` is the lever if fp32 ever needs to be better than ~11 effective mantissa bits.

The test carries this as a live tripwire: it asserts `|median − 1| ≤ max(2 × spread, 5e-3)`, which fires on a systematic offset that dominates the noise (a real scale bug) but not on symmetric rounding.

**Recommended tolerances** (unchanged from the golden suite; the measurements justify them):
`bfloat16` PCC ≥ 0.995, rel-RMS ≤ 0.04 (≈ 17× headroom) · `float32` PCC ≥ 0.999, rel-RMS ≤ 0.02 (≈ 13× headroom) · `rtol = 1e-2`, `atol = 1e-2` for `comp_allclose`.

---

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/rms_norm/ /tmp/rms_norm_verify` then
`python3 -m eval.verify_supported /tmp/rms_norm_verify ttnn.operations.rms_norm --output /tmp/rms_norm_verify/verifier_report.json`

| Category | Count |
|---|---|
| total | 40 438 |
| `supported_pass` | **755** |
| `xfail_expected` | 5 768 |
| `invalid_skipped` | 33 900 |
| `infeasible_skipped` | 0 |
| **`supported_fail`** | **0** ✓ |
| **`xpass_drift`** | **0** ✓ |
| **`xfail_wrong_mode`** | **0** ✓ |
| `supported_marked_xfail` | 0 ✓ |
| `invalid_unexpected` | 0 ✓ |
| `no_axes_found` | 15 (uncharged — `test_regression.py` numerics tests carry no registry axes; all 15 passed) |

Golden run line: `770/40438 passed (0 failed, 0 errors, 33900 skipped, 0 hangs)` = 755 registry cells + 15 numerics regressions.

A trimmed artifact (summary + loud-category node lists + the xfail axis histogram) is committed as `verifier_report.json` next to this file; the full 40 438-row report stays in the results dir.

### The `xfail_expected` bucket is fully accounted for

Every one of the 5 768 xfails is explained by exactly three axis gaps — there is **no queue gap**:

| Missing axis value | xfail cells touched | Refinement |
|---|---:|---|
| `fp32_dest_acc_en = False` | 3 255 | **R1** |
| `memory_layout = WIDTH_SHARDED` | 1 505 | **R2** |
| `memory_layout = BLOCK_SHARDED` | 1 502 | **R2** |
| `memory_layout = HEIGHT_SHARDED` | 1 501 | **R4** |
| `dtype = BFLOAT8_B` | 960 | **R1** |
| `gamma_dtype = BFLOAT8_B` | 860 | **R1** |

(Counts overlap — a cell missing two axes is counted under both.) `TARGET − SUPPORTED` is therefore exactly `{dtype: bf8b, gamma_dtype: bf8b, fp32_dest_acc_en: False, memory_layout: HEIGHT/WIDTH/BLOCK_SHARDED}`, and all six values are queued. No axis value is omitted.

**All 13 perf-target loose cases are currently xfail** — every one pins `fp32_dest_acc_en=False`, and five additionally pin a sharded `memory_layout`. This is what fixes the queue order: R1 and R2 exist to make the perf slots measurable on the *real* config rather than a supported stand-in.

---

## Recommendations

### Cross-cutting, for whoever picks up the queue

1. **R1 before everything.** Not for tidiness — until `fp32_dest_acc_en=False` is in `SUPPORTED`, *no* perf-flagged loose case can even run, so any perf number measured before it would be against a different datapath (fp32 DEST vs bf16 DEST) than the one the `achievable_ns` references were taken at. The op's `default_compute_kernel_config()` must keep returning `fp32_dest_acc_en=True` — `axes.py:40-43` and the acceptance test both read that factory.
2. **Watch the reduce accuracy when DEST goes bf16.** With `fp32_dest_acc_en=False` the `AccumulateViaAdd` running sum lands in bf16 DEST. `examples/row_reduce_accumulate` measures bf16 *accumulation* error growing with reduce width (13.3 ULP at 32 tiles for a single folded reduce, 1.4 ULP for the pairwise-add path this op uses). The perf loose cases carry a **tighter-than-default soft gate, `pcc_threshold = 0.9995`**, at `W` up to 7168 (224 tiles → `NW = 7`). If it misses, the fix is the SFPU finalize (same source), not widening the gate.
3. **The decode column is one core out of 110.** Measured blocking: `(1,1,32,W)` → `ht_total = 1` → 1 core, for every `W`. Everything the decode perf targets ask for lives behind the dependent-`W` cross-core split; there is no knob-turn that reaches it. Conversely the prefill column is **reader-bound** (NCRISC 90–99 % of kernel time, e.g. 442 221 of 482 655 ns on `(1,1,8192,5120)`), i.e. sitting at the interleaved-DRAM read floor — do not expect overlap levers to move it, and do not file one that only has overlap levers.
4. **`gamma` is reuse-shared and currently re-read by every core** (Lamp L2: `Wt` tiles × `num_cores` DRAM reads). Under R2's `W`-split each core owns a *disjoint* gamma slice, so gamma is read exactly once and L2 becomes unnecessary in that regime. Do **not** file a separate gamma-broadcast refinement before R2 lands — R2 may delete the need for it. If R2 keeps a row-parallel path for tall shapes, re-evaluate L2 there (`examples/shared_input_reuse`: 1.71×, `mcast_pipe`).

### L1 / memory-pressure observations (no OOM today)

- Per-core CB totals on the perf shapes: 408 KB … 920 KB against `L1_CB_BUDGET_BYTES = 1 100 000`. The widest supported shape `(1,1,32,32768)` (`Wt = 1024`) falls to the bounded streaming fallback and still fits — no OOM anywhere in the suite, and `supported_fail` has no `OOM` entries.
- The headroom is thinnest on the **fp32 + ROW_MAJOR + gamma** cell, where `BLOCK_CB_UNITS = 11` and `unit_bytes = 4096` drive `TILE_BLOCK_BUDGET` down to 8. Adding `bfloat8_b` (R1) *reduces* pressure; adding a resident **sharded** input (R2/R4) *adds* the shard on top of the CBs. Re-check `blk.cb_total_bytes` against the budget on the sharded path — the existing halve-and-re-derive loop in `_derive_blocking` will absorb it, but silently, by shrinking the block.

### Numerical-precision observations (no concrete failing cell, so no refinement filed)

- **fp32 delivers ≈ 11 effective mantissa bits, not 24.** Documented in the baseline above; the cause is FPU SrcA/SrcB truncation on the two multiplies, not the reduce. If fp32 ever needs to be genuinely fp32-accurate, the lever is `UnpackToDestFp32` on the multiply inputs — which is inside `/numeric-formats-metal`'s scope, so it is worth *trying* while R1 is open (it is already touching intermediate-CB precision). It is explicitly **not** the reason for R1 and must not gate it: every cell passes today with 13–30× tolerance headroom.

### Deferred code items (none blocking)

- Promote `PROPERTIES["multi_core"]` from `"declared"` to `"verified"` with an assertion, alongside R2's core-assignment rework.
- Amend `op_design.md` R4 (`cb_scaler` format) and §6 (`cb_output_rm` page size) to match the shipped, helper-mandated behaviour — see "Documented deviations" above. The design is the planner's artifact, so this is a note rather than an edit.
