# Verification Report: rms_norm

Verified 2026-08-11 on **blackhole p150b** (11×10 = 110-core compute grid, `l1_cb_budget = 1 441 792 B`).

Artifacts: `verifier_report.summary.json` (this directory — counts + the two non-empty diagnostic
categories; the full per-test `verifier_report.json` is 25 MB and stays in the results dir, above the
repo's 500 KB file limit), `op_requirements.md` (refinement queue), `changelog.md`,
`tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py`.

Reproduce with:

```bash
eval/eval_test_runner.sh eval/golden_tests/rms_norm/ <results_dir>
python3 -m eval.verify_supported <results_dir> ttnn.operations.rms_norm \
        --output <results_dir>/verifier_report.json
```

---

## Code Review

Phase 0 arrived in unusually good shape: no functional failures, no SUPPORTED drift, and every
`raw_api` substitution already carried a file:line justification in the kernel header comments.
Four things were **fixed**; nothing was deferred as a known issue.

### Fixed

1. **The CB inventory was stated twice (DRY / single-source-of-truth violation).**
   `rms_norm_program_descriptor.py` declared each CB's page count once in the descriptor's `cbs`
   list and *again*, independently, inside `_cb_bytes()`'s `fixed_bytes` / `per_row_bytes`
   expressions — which the L1 residency solve (and therefore the choice of `G`, `C`, `R`) reads.
   Both were "parameters", so no knob was collapsed, but the two statements drift the moment a
   refinement changes a page count: the solve would keep sizing against a buffer layout that is no
   longer allocated, silently mis-choosing the block factors or OOMing. Fixed by introducing
   `_cb_specs(geo, C, G, R, is_rm_out, has_tail, input_cb_depth)` as the **one** table of
   `(index, num_pages, page_size, format)`; the descriptor's CB list is now a `map` over it, and
   `_cb_bytes()` *derives* `(fixed_bytes, per_row_bytes)` from it by differencing at `R = 1, 2`
   (every page count is affine in `R`, which the file documents). Verified byte-identical to the
   previous closed form over 4608 `(in dtype, gamma dtype, rm_in, rm_out, rm_γ, has_gamma, has_tail,
   C, G)` combinations, so no chosen geometry changed.

2. **`cb_input_tiles` reserved a prefetch slot that provably could not be used.**
   Its capacity was unconditionally `input_cb_depth · R · C`. `input_cb_depth` buys exactly one
   thing — the reader filling block `b+1` while compute runs block `b` — and the selection function
   picks `R == core_row_tiles` (i.e. `num_blocks == 1`) for a real, common configuration:
   `(1,1,8192,1024)` gets `G=1, C=32, R=3, core_row_tiles=3` on this grid. There is no block `b+1`
   there, so half of the op's largest buffer was dead L1 and the ledger's "capacity is 2× the live
   set for pipelining" justification was false for that cell. The descriptor now allocates depth 1
   when the busiest core's whole assignment is one block. The residency **solve** deliberately still
   uses the full knob, so this only ever lowers the footprint and cannot change the selected
   `(G, C, R)`.

3. **Three hardcoded kernel arg offsets were unpinned.** `rms_norm_writer.cpp` reads its mcast args
   at `McastArgs<MCAST_CT_BASE=5, MCAST_RT_BASE=15>` and `rms_norm_reader.cpp` its accessor args at
   `TensorAccessorArgs<6>`; nothing on the host asserted that its own arg lists were exactly that
   long. Appending one runtime arg to the writer would have silently handed `McastArgs` a shifted
   window — a class of bug that manifests as a hang or a wrong multicast destination, not a
   compile error. The three offsets are now named host constants (`READER_ACCESSOR_CT_BASE`,
   `WRITER_MCAST_CT_BASE`, `WRITER_MCAST_RT_BASE`) with build-time asserts naming the kernel constant
   to bump.

4. **`l1_ledger.md` currency.** Its "measured selections" table recorded `(1,1,8192,7168) → G=5,
   C=45, R=3`, which the current score tuple does not select (`G=2, C=112, R=1, 5 blocks/core`,
   re-read off `_select_regime` on the device). Corrected, and the four changes above are recorded
   in the ledger's implementation-delta table together with a previously unrecorded row
   (`cb_stat_sq` carries `R·(1+has_tail_global)` pages on cores that own no tail, live set `R` —
   the accounted price of the group-uniform L1 map).

### Checked and deliberately left alone

- **Every `raw_api` substitution.** The gather leg (`noc_async_write` + semaphore) genuinely cannot
  be `mcast_pipe`'s `SenderPipe` — that is one-to-many into an identical address, the gather is
  many-to-one into disjoint slots (`mcast_pipe.hpp:44-45`); the return multicast *does* use
  `SenderPipe`/`ReceiverPipe`. `read_sticks_for_tilize` / `write_sticks_after_untilize` derive both
  page count and L1 row stride from `row_bytes` (`tilize_helpers_dataflow.inl:91-93,120-124`), so
  they cannot produce the **group-uniform** row stride that `tilize<CB_W_TILES>` consumes on a
  ragged core; the local `read_slice_rows` / `write_slice_rows` are those helpers' bodies with the
  stride taken from the uniform CB width. Per the scope boundary, mechanism choice is the
  implementer's — and here the semantic effects (batching, barrier frequency, residency) are the
  ones the design asked for.
- **The caller-managed `(WaitPolicy::None, PopPolicy::None)` policies** on `sumsq_block` /
  `mask_tail_block`. `TileOffset::Strided` *requires* them (`eltwise_chain.inl:1169-1172`), and
  strided addressing is what lets both phases walk the block's hidden tiles at row stride
  `CB_W_TILES` without a contiguous scratch copy. The kernel wraps them in an explicit
  `cb_wait_front` / `cb_reserve_back` … `cb_push_back` pair, so push count == wait count holds.
- **Uninitialized L1 in the pad columns / stale rows of a block.** Safe *because* every math phase
  over `cb_input_tiles` is row-independent (`REDUCE_ROW` + elementwise), so a stale Inf/NaN cannot
  migrate into a valid row, and the writer never stores those bytes. The reader header states this
  invariant and names what would break it (any `REDUCE_COL` / `REDUCE_SCALAR` /
  `DestAccumulation::WholeShape` phase). Left as-is, but it is a standing constraint on Refinement
  1 and 4: a future phase that crosses rows must zero-fill first.
- **The CB ring-wrap invariant** (`capacity_pages % quantum == 0`, `dataflow_api.h:216-221`, the
  implementer's high-cost friction note). It holds here for a non-obvious reason worth recording:
  every `R`-scaled CB has capacity `depth × (full quantum)`, so full-block pushes wrap **exactly**,
  and the single ragged `last_block_row_tiles` push is always the *last* one, ending strictly below
  `fifo_limit`. That is why `R ∤ capacity` cases (e.g. `R=4`, `last=3`) are nonetheless legal. Any
  refinement that makes a partial-quantum push non-final must re-derive this.

### Advisories (no change made)

- **Perf lamp P4 (`scale_block` + `gamma_block` fusion)** is not applied. `DestReuseBinary` would
  remove a full pack+unpack of the block and the `cb_normed` buffer, but it routes DEST through a
  Src register (bf16 on Wormhole) — a real precision loss on the `float32` cells — and
  `compute_fusion` measures FPU-consumer dest-reuse at 0.94×/0.82× in isolation. Correct default;
  re-open only with a PCC measurement on the fp32 cells (noted in Refinement 4's lever set).
- **`eval/prompts/rms_norm.txt` has no `## Rules` section**, so only the stock policies apply. All
  of its prose constraints are met: no host-side `to_layout` / `tilize` / `untilize` / `pad` /
  `slice` anywhere in the op (verified by reading `rms_norm.py`); output layout and dtype match the
  input (asserted in the descriptor); `default_compute_kernel_config()` is the single exported
  factory and the golden axis-tagger imports it; the caller's descriptor is passed through as
  `config=compute_kernel_config`; the three rejection messages contain `rank` / `gamma` /
  `fp32_dest_acc_en` as the acceptance test's regexes require.
- **The prompt asks for `tag_gamma_dtype` / `tag_gamma_layout` in `INPUT_TAGGERS`**; the op derives
  those two axes inline in `validate()` instead. This is not a defect: `INPUT_TAGGERS` here take
  `(inputs, axes)` where `inputs` is a tuple of **shapes**, from which a gamma dtype is not
  derivable, and `feature_spec.py`'s own docstring names exactly the two taggers the op declares
  (`tag_alignment`, `tag_rank`). The `"none"` sentinel semantics the prompt actually cares about are
  implemented, and `verify_supported` confirms the op-side and golden-side axis dicts agree
  (0 `xfail_wrong_mode`, 0 `xpass_drift`).

---

## Registry Conformance

- **Confirmed present and correctly wired** in `ttnn/ttnn/operations/rms_norm/rms_norm.py`:
  `INPUT_TAGGERS` (both taggers take `(inputs, axes)`), `SUPPORTED` (9 axes — every axis the kernel
  gates on, including both `INPUT_TAGGERS` keys), `EXCLUSIONS` (one cell-dict), and `validate()`,
  which checks **SUPPORTED per-axis first, then EXCLUSIONS**, raising `UnsupportedAxisValue` /
  `ExcludedCell` from `ttnn.operations._op_contract`. Structural `ValueError`s (rank < 2, gamma
  width/leading-dim mismatch, `epsilon <= 0`) are correctly *not* support refusals.
  `rms_norm()` calls `validate(...)` as its first statement, before any allocation or device work.
- **Confirmed the op file does NOT declare `INVALID`** — it is sourced from
  `eval/golden_tests/rms_norm/feature_spec.py`, as the registry model requires.
- **No auto-fixes to SUPPORTED were needed**: `xpass_drift = 0`, so nothing works that the op does
  not already claim, and `supported_fail = 0`, so nothing it claims is broken.
- `PROPERTIES` is present (`multi_core` verified, `bounded_cb` declared, `math_fidelity` declared) —
  consistent with what the descriptor does.

### INVALID audit (`eval/golden_tests/rms_norm/feature_spec.py`)

**Well-formed (structural impossibility, single-tensor coupling):**

- `{dtype: bfloat8_b, layout: ROW_MAJOR}` — the canonical bf8b+ROW_MAJOR activation entry. Present ✓.
- `{gamma_dtype: bfloat8_b, gamma_layout: ROW_MAJOR}` — the same impossibility on the gamma tensor;
  both axes describe gamma ✓.
- The six `no_gamma` canonicalization entries — presence ↔ `"none"` sentinel coupled both ways, so
  exactly one canonical `("none","none")` cell survives ✓. This is the norm-like-op
  "no-weight canonicalization" requirement, and it is complete.

**Flagged — recommend the author move these out of INVALID** (report-only; I did not edit
`feature_spec.py`):

- `{layout: ROW_MAJOR, memory_layout: *_SHARDED, gamma_layout: TILE}` (three entries) **couple axes
  describing two different tensors** — the activation's layout *and* placement with the **gamma**
  tensor's layout — with no kernel-level coupling documented. The file itself labels the block
  "Author-scoped exclusions (**NOT** structural impossibility)". That is the canonical INVALID
  authoring mistake and it is load-bearing here: because the harness *skips* rather than xfails
  them, these cells will stay invisible even after Refinement 2 lands sharding, so the queue can
  never see them come back. Recommended: delete from `INVALID` and let the op's `EXCLUSIONS` refuse
  them if it must.
- `{dtype: bfloat8_b, alignment: w_non_aligned}` and `{… h_non_aligned}` are single-tensor (both
  axes describe the activation), but they encode "my kernel doesn't do this yet", which is
  `EXCLUSIONS` material, not INVALID. Note the consequence for Refinement 1: `bfloat8_b` ×
  non-tile-aligned is exactly the cell the `/numeric-formats-metal` skill expects to route to
  `EXCLUSIONS`, and it cannot, because those cells are skipped upstream. No cell movement should be
  expected from them.
- No `INVALID` entry is missing for a cell that is genuinely impossible, and none is redundant with
  SUPPORTED.

### Other verifier-CLI categories

- `invalid_unexpected = 2`: two `test_translated.py` cases
  (`test_rms_norm_sharded_uneven_multicore_logical_width[w{72,200}_c{2,3}_nonaligned-bfloat8_b]`)
  whose cell matches the INVALID entry `{dtype: bfloat8_b, alignment: w_non_aligned}` but which the
  translated suite marks **xfail** instead of skipping. Harness-side only (the op refuses them
  correctly); worth a `/golden-tests` tidy, no op change.
- `no_axes_found = 15`: `test_regression.py`'s magnitude/uniform/sign cases record no axes tag. All
  15 **passed**. Harness-side; the op is unaffected.

---

## L1 Ledger Audit

- **Ledger currency**: after the fixes above, every declared CB has a row, every row corresponds to
  a live CB, and every size expression matches what `_cb_specs()` allocates. Two currency defects
  were found and repaired: the stale `(1,1,8192,7168)` selection row, and the missing
  `cb_input_tiles` depth / `cb_stat_sq` tail-column rows. The footprint expressions are now
  explicitly labelled as *derived* from the CB table rather than a second source of truth.
- **Capacity vs live set — over**: one real hit, `cb_input_tiles` at depth 2 with `num_blocks == 1`
  → **fixed in place** (item 2 above). Two accounted-and-justified over-capacities remain and are
  correct: `cb_stat_gather` (`R·G` pages allocated on **every** group member, not just roots,
  because `mcast_pipe` requires an identical `dst_l1` on all receivers and a root-only CB would
  shift every later address on the roots) and `cb_stat_sq` (`R·(1+has_tail_global)` on cores owning
  no ragged tile) — same uniform-L1-map price, now both recorded.
- **Capacity vs live set — under (collapsed extent)**: none. Every row whose live set *spans* a
  block axis scales with that axis (`cb_input_tiles`, `cb_normed`: `R·C`; `cb_stat_gather`: `R·G`;
  the RM-output `cb_output_tiles`: `R·C`), and every fixed-window row genuinely only *streams* over
  the axis it does not scale with (`cb_output_tiles` on the tiled path, `cb_input_rm`,
  `cb_output_rm`: one tile-row live, depth 2 for overlap).
- **Disjoint lifetime with no justification**: none. Every row's `Shares with / why not` cell names
  a concrete reason, and the two rejected reuses are recorded with theirs (in-placing `gamma_block`
  onto `cb_output_tiles` would give the writer and compute two concurrent consumers; folding
  `cb_rstd_send` into `cb_rstd` would make one CB a compute product, a writer multicast source and
  a compute input). The one *available, deliberately unclaimed* reuse (`cb_tail_masked` ↔
  `cb_stat_sum`) no longer exists — `cb_tail_masked` was fused away entirely.
- **Bounds and closed form**: every non-block symbol in a capacity expression is in the symbol
  table with a bound and its predicate. The only op dimension reaching a capacity is
  `tensor_w_tiles`, and only through `C = ceil(tensor_w_tiles / G)`, which the residency predicate
  bounds by *raising `G`* — a real, host-checkable predicate, not a hope: the widest shape in the
  declared universe (`W = 32768`, `Wt = 1024`) lands at `C = 47` on this grid, against a grid
  capacity of ~110 × ~110 hidden tiles. Beyond it, `_select_regime` raises a `RuntimeError` naming
  regime R3 (lamped) instead of OOMing — an explicit, bounded failure.
- **Block-size defaults**: interleaved wants "spread the split's work units across the full grid,
  then take the coarsest block that fits". Held on the prefill side (110/110 cores; `R` = the
  coarsest that fits the closed-form solve). **Departed from, with measurement**, on the decode side:
  `MAX_W_GROUP_SIZE = 32` deliberately does *not* fill the grid at `tensor_row_tiles == 1`, and the
  constant carries the device-ns numbers that justify it (110 → 22 cores: 17676→11088 ns at
  `W=5120`, 18624→12165 ns at `W=7168`). `MIN_PIPELINE_BLOCKS = 1` is likewise a measured default
  (a 1/2/3/4-block sweep came out flat within 2%, DRAM-bound). Both are single host constants.
  The unmeasured consequence — the same cap leaves 44/110 cores at `tensor_row_tiles = 2` and 66/110
  at 3 — is folded into Refinement 3, not filed separately.
- **Data-movement budget**: present, consistent with the implemented split, and re-verified on
  device. Input **1×** DRAM (the block stays resident from `sumsq_block` through `scale_block`,
  which is the entire reason the hidden axis is split), output **1×**, gamma **`num_row_groups`×**
  (structurally irreducible: gamma does not vary along `row`). The re-read of `cb_input_tiles` by
  the apply pass is L1, not DRAM, and is counted. Cheapest-traffic split is **lamped, not silent**:
  forcing `num_row_groups = 1` saves `(num_row_groups−1)·W·2` bytes ≈ 0.44 MiB = **0.19 %** of DRAM
  traffic on the widest prefill case, at the cost of an unmeasured number of extra combine rounds
  (perf lamp P3, reachable by one line in the score tuple). Occupancy is nowhere used as the
  justification for the traffic choice.
- **Per-core footprint** (what the host computes, all terms in the block knobs):
  `footprint(R) = fixed_bytes + R · per_row_bytes` with
  `per_row_bytes = T_in·C·(input_cb_depth + has_gamma + rm_out) + 4096·(4 + (1+has_tail) + G)` and
  `fixed_bytes = T_in·C·((1−rm_out)·output_cb_depth + (rm_in+rm_out)·rm_cb_depth)
  + T_γ·has_gamma·C·(1+rm_γ) + 2048·(1+has_tail) + 4096`.
  Scaling: everything in `per_row_bytes` scales with `R`; within it the two block buffers scale with
  `C` and only `cb_stat_gather` with `G`; `fixed_bytes` scales with `C` alone. Nothing scales with
  `tensor_row_tiles`, and `tensor_w_tiles` reaches it only via the bounded `C`.

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py` — 40 cells
(5 shapes × {bf16, fp32} × {TILE, ROW_MAJOR} × {gamma, no_gamma}), all passing. TILE + gamma rows:

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio (median, p5–p95) |
|-------|-------|-----|-------------|--------------|------------------|----------------------------------|
| (1,1,32,32) | bfloat16 | 0.9999968 | 0.019962 | 0.001442 | 0.002592 | 0.999904 (0.99599–1.00388) |
| (1,1,128,512) | bfloat16 | 0.9999973 | 0.031619 | 0.001212 | 0.002333 | 1.000075 (0.99624–1.00400) |
| (1,1,32,4096) | bfloat16 | 0.9999975 | 0.046088 | 0.001154 | 0.002264 | 1.000216 (0.99657–1.00401) |
| (2,1,1024,1024) | bfloat16 | 0.9999972 | 0.066194 | 0.001202 | 0.002366 | 1.000156 (0.99626–1.00408) |
| (1,1,40,200) *(both dims non-aligned)* | bfloat16 | 0.9999970 | 0.029738 | 0.001129 | 0.002434 | 1.000032 (0.99608–1.00405) |
| (1,1,32,32) | float32 | 0.9999997 | 0.012485 | 0.000877 | 0.001487 | 0.998867 (0.99755–0.99997) |
| (1,1,128,512) | float32 | 0.9999997 | 0.017711 | 0.000706 | 0.001311 | 0.998963 (0.99765–1.00008) |
| (1,1,32,4096) | float32 | 0.9999997 | 0.022491 | 0.000636 | 0.001220 | 0.999079 (0.99771–1.00022) |
| (2,1,1024,1024) | float32 | 0.9999997 | 0.026640 | 0.000696 | 0.001312 | 0.998939 (0.99760–1.00008) |
| (1,1,40,200) *(both dims non-aligned)* | float32 | 0.9999997 | 0.021823 | 0.000647 | 0.001347 | 0.998940 (0.99764–1.00005) |

ROW_MAJOR is bit-identical to TILE for bfloat16 and within 25 % on the error metrics for float32
(the extra tilize/untilize round-trip); `no_gamma` is uniformly ~1.4× better (one fewer lossy FPU
multiply): bf16 rel_rms ≈ 0.0017, fp32 rel_rms ≈ 0.00055.

**Assessment**: no scale bug anywhere — the got/true ratio is centred on 1.0 with a spread wider
than its offset in every cell, which is the rounding signature, not the "tight cluster at a non-1.0
constant" signature. PCC is ≥ 0.999996 on every cell, i.e. 3+ orders of magnitude of headroom over
the golden gates, and error does **not** grow with the reduce width (rel_rms is flat from `W=32` to
`W=4096`), which is the direct evidence that the `Σ x²` accumulation really is happening in fp32
DEST and not in a bf16 intermediate. Two systematic effects are worth recording because Refinement 1
will move both: (1) `float32` shows a consistent **~0.1 % shrink** (ratio median 0.9989) and a
rel_rms of ~1.3e-3 rather than the ~1e-7 a true-fp32 datapath would give — that is the FPU's
truncating fp32 mantissa (~19 bits) across three multiplies, expected, and it is why the fp32 gate
is 0.999 not 0.9999999; (2) the non-tile-aligned cell `(1,1,40,200)` is indistinguishable from the
aligned cells, confirming the ragged-tile mask is exact (`(x·mask)² == x²·mask`) rather than
approximately right.

**Recommended tolerances**: unchanged from the golden suite — bfloat16 `PCC ≥ 0.995`, rel_rms ≤ 0.04;
float32 `PCC ≥ 0.999`, rel_rms ≤ 0.02. Measured margins are ≥ 15× on rel_rms, so these are safe
gates for the refinements; the perf loose cases' soft `pcc_threshold = 0.9995` is met with ~3 orders
of magnitude to spare today, which matters because Refinement 1 spends precisely that margin
(bf16 DEST accumulation, then bfloat8_b).

---

## Verifier CLI Summary

Golden suite: `PASSED=752 FAILED=0 ERRORS=0 SKIPPED=33900 HANGS=0 TOTAL=40828`.
`python3 -m eval.verify_supported`:

- supported_pass: **737**
- xfail_expected: **6174**
- invalid_skipped: **33900**
- supported_fail: **0**   (must be 0 to ship) ✓
- xpass_drift: **0**      (must be 0 to ship) ✓
- xfail_wrong_mode: **0** (must be 0 to ship) ✓
- supported_marked_xfail: 0; supported_skipped: 0; xfail_other: 0; infeasible_skipped: 0
- invalid_unexpected: 2 (harness-side, see above); no_axes_found: 15 (harness-side, all passed)

Identical before and after the four code-review fixes, i.e. the fixes are behaviour-preserving.
Unit suites: `test_rms_norm.py` 82/82, `test_rms_norm_shapes.py` 92/92,
`test_rms_norm_precision_baseline.py` 40/40, `test_rms_norm_perf.py` collected and passing.

**Every `(axis, missing_value)` in `TARGET − SUPPORTED` is accounted for.** Grouped by which axis
value puts them outside SUPPORTED, the 6174 `xfail_expected` cells decompose into exactly four
missing values — `fp32_dest_acc_en=False`, `memory_layout ∈ {HEIGHT,WIDTH,BLOCK}_SHARDED`,
`dtype=bfloat8_b`, `gamma_dtype=bfloat8_b` — and their combinations. Refinement 1 covers the first,
third and fourth; Refinement 2 covers the second. No axis value is left without a queue entry, and
no queue entry claims an axis value the op already has.

---

## Recommendations

1. **Refinement 1 is the gate on the whole perf programme, not just a dtype widening.** Every
   `group="perf"`, `resilience` and `pad_poison` loose case in `feature_spec.py` runs at
   `fp32_dest_acc_en=False` (+ `math_fidelity=HiFi2` for the perf ones). Until it lands, *no* perf
   number can be taken at a flagged case's real config, and the temptation to measure at
   `fp32_dest_acc_en=True` instead must be refused: DEST capacity doubles when fp32 accumulation is
   off, which changes `DEST_AUTO_LIMIT`, the helpers' internal walk, and plausibly the selected `R`.
   The op's own `test_rms_norm_perf.py` currently measures at `True` for exactly this reason and
   should be re-pointed at `False` as part of Refinement 1.
2. **Keep the statistics path fp32 through Refinement 1.** `x²` is the all-positive,
   monotonically-swamping accumulation that `row_reduce_accumulate` singles out as the case where
   bf16 accumulation error grows with width. The precision baseline's flat rel_rms across
   `W = 32 → 4096` is today's evidence that it is fp32; if that flatness disappears after
   Refinement 1, a stat CB has been downgraded.
3. **The idle-grid regime is narrow but real.** `MAX_W_GROUP_SIZE = 32` was measured only at
   `tensor_row_tiles == 1`; at 2/3/4 tile-rows the same cap leaves 44/66/88 of 110 cores active
   (verified on device via `_select_regime`). `(1,1,64,12288)` is exactly such a case and is a
   `_WIDE` loose case. It is a knob-turn (one host constant) and is folded into Refinement 3 rather
   than filed separately.
4. **Writer batching is below `double_buffer`'s sweet spot on narrow-hidden shapes.** The TILE-path
   drain is one `noc_async_write_barrier` per tile-row (`core_w` tiles): 11–112 tiles on the perf
   shapes (fine), but 2–3 tiles on `(99991,64)` / `(1,1,3232,96)` (below the measured 4–8 floor).
   Raising it means raising `output_cb_depth` and draining several tile-rows per barrier — L1 traded
   for overlap, so it belongs in Refinement 4's co-tune, measured, not applied blind.
5. **L1 headroom is not currently a risk, and the reason is a predicate rather than luck.** No cell
   in the declared universe reaches the `RuntimeError` (regime R3) path, because the selection
   raises `G` until `C` fits. Two things would change that and should be re-checked when they land:
   a much smaller compute grid (the predicate has fewer cores to spend), and Refinement 2's sharded
   placements, where the shard spec — not the solve — dictates `C`, so a legal-looking shard can
   demand a `C` that does not fit. Refinement 2 should assert that explicitly rather than discover
   it as an allocation failure.
6. **Perf lamp P4 (fuse `scale_block` + `gamma_block`)** is the one structural compute-side win left
   on the table (a pack + unpack of the whole block, plus the `R·C`-page `cb_normed` buffer). It is
   correctly not the default — dest-reuse routes DEST through a bf16 Src register on Wormhole, and
   `compute_fusion` measures FPU-consumer dest-reuse at 0.94×/0.82× — but on the bf16 perf cases
   neither objection applies with force. Worth one measurement inside Refinement 4, gated on the
   fp32 cells' PCC.
