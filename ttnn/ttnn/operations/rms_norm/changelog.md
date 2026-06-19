# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-06-19
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Two-regime performance design:
  Regime A (row-parallel, full row resident, embarrassingly-parallel multi-core)
  and Regime B (wide-W cross-core W-split with an mcast all-gather of partial
  Σx²). Verifier pass: registry conformance hardening, golden run, precision
  baseline, refinement queue.
- **SUPPORTED at Phase 0**:
  - dtype = [bfloat16]
  - layout = [TILE_LAYOUT]
  - alignment = [tile_aligned]
  - rank = [2, 3, 4]
  - fp32_dest_acc_en = [True]
  - gamma_mode = [gamma, no_gamma]; gamma_dtype = [bfloat16, float32(no_gamma canonical only)]; gamma_layout = [TILE_LAYOUT]
  - EXCLUSIONS = [{gamma_mode: gamma, gamma_dtype: float32}]
- **Accuracy achieved (Regime A, bf16, measured on 8 cases via
  test_rms_norm_precision_baseline.py)**:
  PCC ≥ 0.999; max_abs_err ≤ 0.078 (gamma) / ≤ 0.032 (no gamma);
  mean_abs_err ≈ 0.002; relative RMS ≈ 0.003–0.004.
- **Golden suite at Phase 0** (per `verifier_report.json`):
  total 5142 — supported_pass 22, xfail_expected 2144, invalid_skipped 2940,
  **supported_fail 21** (all Regime B), xpass_drift 0, xfail_wrong_mode 0,
  supported_marked_xfail 0, no_axes_found 15 (float32 test_regression.py).
- **Issues encountered / fixed this pass**:
  - `__init__.py` did not re-export `INPUT_TAGGERS`/`SUPPORTED`/`EXCLUSIONS` →
    whole golden suite failed at collection. Fixed (now re-exported).
  - `tag_alignment` was a 2-value split returning an out-of-universe value;
    replaced with the feature_spec-mandated 3-value split. Added missing
    `tag_rank`. Both taggers now take `(inputs, axes)`.
  - `SUPPORTED` was missing `rank`, `fp32_dest_acc_en`, `gamma_mode`,
    `gamma_dtype`, `gamma_layout` → fp32/bf8b/ROW_MAJOR gamma cells would have
    run-and-failed (silent over-claim). Added all; gating now honest.
  - `validate()` now takes `gamma` + `compute_kernel_config` and mirrors
    `helpers.classify_call`; added prompt-required `ValueError` guards (rank < 2,
    gamma last-dim mismatch). Entry point forwards both args.
  - **Known blocker (NOT fixed — filed as Refinement 1):** Regime B
    (cross-core all-gather) is numerically broken — output too large by
    `sqrt(2·num_chunks)`; gathered Σx² underflows by exactly 1/(2·num_chunks).
    Regime A is correct. This is the op's headline feature and gates the queue.
  - Deferred (noted in verification_report.md, to fold into Refinement 1):
    `cb_normalized`/`cb_gamma` sized to `Wt` (not constant `REDUCE_BLOCK`), which
    understates the resident-L1 budget; writer barrier-per-tile (perf only).
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py`
    (PCC + max/mean abs + relative RMS over 4 shapes × gamma/no-gamma, Regime A).
  - (`test_rms_norm.py` acceptance suite already present — 20/20 passing.)

## Refinement 1 — Fix Regime B cross-core all-gather correctness (BLOCKER)
- **Date**: 2026-06-19
- **What was done**:
  - **Root-caused two compounding bugs** at the `cb_partial_sumsq` cross-thread
    handshake that made Regime B output too large by `sqrt(2·num_chunks)`:
    1. *(factor 1/num_chunks)* The mcast reader's `cb_wait_front(cb_partial_sumsq, 1)`
       was satisfied after PASS-1's **first** W-chunk (the reduce-accumulate pushes
       once per chunk), so it grabbed an early front L1 address holding only chunk-0's
       partial. The accumulator pops/repushes across chunks, landing the final sum in a
       **different physical page**, so the reader mcast only chunk-0's contribution.
    2. *(factor 1/2)* The K-partial combine read the **stale, un-popped PASS-1 local
       sum** as its in-place accumulator front (the reader pops `cb_partial_sumsq` only
       after pushing the gathered CB, racing compute's reuse).
  - **Fix**: added a dedicated single-push CB `cb_local_sumsq` (CB 28, Regime B only).
    Compute copies the fully-accumulated local Σx² into it exactly once after PASS-1
    completes (`copy<>` also pops `cb_partial_sumsq`, emptying it). The reader now waits
    on `cb_local_sumsq` (observes only the final value); the combine writes into the
    now-empty `cb_partial_sumsq` with no stale-front aliasing. Regime A (num_partials==1)
    skips this path entirely and is unchanged.
  - **Folded in the deferred `cb_normalized = Wt` sizing fix**: the gamma pass-2 now
    streams the Col→Row multiply per `REDUCE_BLOCK` with `cb_normalized` sized to one
    block (was `Wt`/`Wt_s`), so per-core L1 no longer scales with row/shard width and the
    A/B resident-budget heuristic is sound. (The design's single-`eltwise_chain` fusion
    was not used — the helper lib has no broadcast `DestReuseBinary` and `BinaryFpu`
    cannot take DST as an operand; the streaming two-helper chunked form is the bounded
    equivalent.)
- **Accuracy achieved (Regime B, bf16, measured)**: all-ones → exactly 1.0 across
  num_chunks ∈ {1,2,3} and LOOSE W ∈ {16384, 32768}; random standard-normal vs torch
  PCC ≥ 0.99999, relative RMS 0.0035–0.0091 (±gamma) — well inside the bf16 band
  (relRMS ≤ 0.04). Regime A precision baseline unchanged (8/8).
- **Golden test progress**: 43/43 supported cells passing (was 22/43 — the +21 were all
  Regime B); 0 failed, 0 xpass-drift. (The 15 `test_regression.py` failures are float32,
  out of scope until Refinement 2.)
- **Issues encountered**: None blocking. The clean single-chain pass-2 fusion the design
  references is not expressible with the current helper library (noted above); the
  bounded chunked-streaming form was used instead with identical accuracy.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_regime_b.py` — 39 cases:
    all-ones exact (==1.0), per-shard-distinguishable ramp (verifies the all-gather
    delivers all K distinct partials), random vs torch (±gamma), and the LOOSE wide-W
    cases. All pass in production timing.

## Refinement 2 — Numerical configurability expansion
- **Date**: 2026-06-19
- **What was done**:
  - **SUPPORTED grown**: `dtype += [float32, bfloat8_b]`; `fp32_dest_acc_en += [False]`;
    `gamma_dtype += [bfloat8_b]`. EXCLUSIONS: removed the old gamma-float32 entry
    (gamma-present float32 now works) and added `{dtype: float32, fp32_dest_acc_en: False}`
    (fp32 input mandates fp32 accumulation — the prompt's documented EXCLUSION).
  - **Per-CB format derivation in the program descriptor** (both regimes): input/output
    CBs follow the tensor dtype; the gamma CB follows `gamma.dtype` (this is what unblocks
    mixed-precision gamma — bf16 activations + fp32/bf8b gamma); accumulator intermediates
    (Σx², scaler, recip-rms, normalized block, Regime-B gathered partials) follow
    `_intermediate_dtype`: promoted to `Float32` when `fp32_dest_acc_en`, else bf16 — and
    **never bf8b** (a block-float accumulator is wrong). bf16 input keeps bf16 intermediates,
    byte-identical to Phase 0 / Refinement 1 (no regression). Per-CB tile bytes via
    `ttnn.tile_size(format)` (was a single shared `buffer_page_size`).
  - **No compute-kernel changes** — the eltwise/reduce helpers reconfig unpack (BinaryFpu
    `Input`, CopyTile `Input`) and pack (PackTile `Output`) data formats automatically, so
    mixed input/intermediate/gamma/output formats just work (numeric-formats skill pass
    condition held).
  - **Regime-B mcast reader fix (the one real bug uncovered)**: the all-gather strided its
    slots and sized the cross-core transfer with `get_tile_size(cb_input_resident)`, a latent
    assumption that input format == partials format. True for bf16 (both 2048 B), false for
    bf8b input (1088 B) with fp32 partials (4096 B) → the gather copied/strode the wrong byte
    count → `Inf` in a subset of outputs (72 golden cells). Now uses
    `get_tile_size(cb_partials_gathered)` for the gather slot stride / local copy / `sender.send`
    size, and the input tile bytes only for the input read.
- **Accuracy achieved** (precision matrix, fp32_dest_acc_en=True, HiFi4, 128x512):
  float32 PCC ≥ 0.99999, relRMS ≤ 0.0015; bfloat16 PCC ≥ 0.99999, relRMS ≤ 0.004;
  bfloat8_b PCC ≥ 0.9999, relRMS ≤ 0.015. fp32_dest_acc_en=False (bf16/bf8b) and the full
  LoFi→HiFi4 sweep all stay above the asserted floors (bf16/fp32 ≥ 0.99, bf8b ≥ 0.98);
  no Inf/NaN in any cell.
- **Golden test progress**: 418/418 supported passing (was 346/346 before this refinement;
  the +72 are the bf8b Regime-B cells the mcast fix unblocked), 2940 skipped, 1784 xfailed,
  0 failed, 0 xpass-drift. The 15 float32 `test_regression.py` `no_axes_found` cases are
  cleared (float32 is now in SUPPORTED).
- **Issues encountered**: the bf8b Regime-B `Inf` bug (root-caused + fixed, above). No other
  blockers — fp32, bf8b, the False precision corner, and mixed-precision gamma all landed.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_matrix.py` — the
    authoritative precision matrix: 8 shapes (Regime A + B) × {bf16, fp32, bf8b} ×
    {HiFi4..LoFi} × {fp32_acc, bf16_acc} × {uniform, normal} × {gamma, no_gamma} = 641 cases
    (+128 skipped EXCLUSION cells), plus `test_rms_norm_precision_matrix_fp32_no_acc_refused`
    asserting the EXCLUSION raises a support refusal.
  - `tests/ttnn/unit_tests/operations/rms_norm/precision_matrix_results.md` — results table.

## Refinement 3 — ROW_MAJOR layout + non-tile-aligned shapes (native)
- **Date**: 2026-06-19
- **What was done**:
  - **SUPPORTED grown**: `layout += [ROW_MAJOR_LAYOUT]`; `gamma_layout += [ROW_MAJOR_LAYOUT]`;
    `alignment += [w_non_aligned, h_non_aligned]`. No new EXCLUSIONS — every newly-claimed
    cell passes natively at the bf16/fp32/bf8b tolerance band. INVALID ({bf8b, ROW_MAJOR}
    on either tensor) stays in feature_spec, never reaching validate().
  - **TILE input + non-aligned: already native.** Verified that TILE-layout non-tile-aligned
    shapes work unchanged — `ttnn.from_torch` zero-pads the tensor and the kernel's
    `inv_W = 1/W` already carries the TRUE element count, so padding columns add 0 to the
    per-stick Sum(x^2). Only required adding the alignment values to SUPPORTED.
  - **ROW_MAJOR input: new dedicated tilize-wrapped, row-parallel path**
    (`_regime_rm_descriptor` + `rms_norm_{reader,compute,writer}_rm.cpp`). Each stick (one
    (b,c,h) row of W elements) is RMS-normalized independently; sticks are processed in
    32-stick tile-blocks, W chunked by `reduce_block` tiles so the per-core L1 footprint is
    bounded regardless of W (input held resident per block; gamma streamed). Reader reads
    sticks → compute tilizes → TILE math (square→reduce→rsqrt→normalize) → compute untilizes
    → writer writes sticks. **No host-side to_layout / tilize / untilize / pad / slice.**
    Output layout matches input.
  - **Native non-aligned handling in the RM dataflow kernels**: the reader ZEROES the W-padding
    columns of the last real tile (and any synthetic padding tiles rounding Wt up to a
    reduce_block multiple) so they contribute 0 to Sum(x^2); H non-alignment is a partial last
    tile-block whose extra sticks are zeroed and never written by the writer. Per-stick
    independence means H alignment needs no math change.
  - **Mixed gamma layout (both input layouts)**: gamma may be TILE or ROW_MAJOR independent of
    input. TILE gamma is read as column tiles directly into the gamma-tiled CB; ROW_MAJOR gamma
    is read as a stick and tilized in compute. For the TILE-input path this added a constexpr
    `gamma_is_rm` branch to both readers (Regime A + Regime B mcast) and a one-time gamma tilize
    at compute boot — fully gated so the TILE-gamma path is byte-identical to prior phases.
  - **Bug fixed (the one real blocker)**: adding non-aligned to SUPPORTED surfaced 350 golden
    failures, all `ValueError: datum for bfp8 invalid` — a host-side crash from calling
    `tensor.element_size()` on a bf8b (block) tensor, which has no single datum size. `gamma_elem`
    is now computed only when gamma is ROW_MAJOR (guaranteed non-bf8b); bf8b + non-aligned then
    works numerically (pcc >= 0.9999).
- **Accuracy achieved** (measured on probes, fp32_dest_acc_en=True, HiFi4):
  - RM bf16: PCC >= 0.99999, maxerr <= 0.09 on shapes incl. (1,1,32,50), (1,1,17,64),
    (1,1,17,50), (4,8,32,47), (1,1,32,4096), (1,32,4096).
  - RM fp32: PCC = 1.00000 incl. (1,1,32,8192) (fits L1: bounded streaming).
  - TILE input + RM gamma (Regime A + B): PCC >= 0.99999.
  - bf8b + non-aligned (TILE): PCC >= 0.9999.
- **Golden test progress**: 1683/1683 supported passing, 2940 skipped, 420 xfailed, 0 failed,
  0 xpass-drift (was 1333 passed / 350 failed before the element_size fix). test_regression.py
  15/15.
- **Issues encountered**: (1) TILE-layout gamma initially read as RM sticks in the RM path
  (pcc ~0.2) — fixed with a `gamma_is_tile` branch reading tiles directly. (2) bf8b
  element_size() host crash (above). Both root-caused and fixed; no deferrals.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_layout_matrix.py` — the
    authoritative layout matrix: 10 shapes (aligned + W/H/both non-aligned, 2D/3D/4D) ×
    {bf16, fp32} × {TILE->TILE, RM->RM} × {no_gamma, gamma_tile, gamma_rm} = 120 cases, all
    asserting output-layout-matches-input + PCC/relRMS. All pass.

## Refinement 4 — Unify the kernel set (7 → 3–4 max; single compute non-negotiable)
- **Date**: 2026-06-19
- **What was done**: Collapsed the **7-kernel set to 3** — exactly one of each
  (`rms_norm_reader.cpp`, `rms_norm_compute.cpp`, `rms_norm_writer.cpp`) — beating
  the ≤4 budget and meeting the single-compute non-negotiable. Done in three
  validated phases:
  1. **Single compute** (defect #1): merged `rms_norm_compute_rm.cpp` into
     `rms_norm_compute.cpp` gated by a `layout_is_rm` constexpr — RM is the shared
     TILE math with a tilize prologue + untilize epilogue. RM adopted the
     resident-gamma model (gamma fed/tilized once, indexed by per-chunk
     `TileOffset` in PASS-2) so pass-2 is byte-for-byte shared with TILE. The
     bf16/TILE Phase-0 anchor is numerically byte-identical (no-gamma TILE keeps
     the single whole-shard `mul`).
  2. **Unified reader + writer** (defect #2): one reader gated by
     `(layout_is_rm, num_partials)` — TILE tile-read vs RM stick-read, with the
     Regime-B `SenderPipe`/`ReceiverPipe` all-gather gated by `num_partials>1`
     (mcast RT args read only under that constexpr via a superset RT layout). One
     writer gated by `layout_is_rm`. Deleted `reader_mcast.cpp`, `reader_rm.cpp`,
     `writer_rm.cpp`.
  3. **ROW_MAJOR through mcast** (defect #3): RM dispatch now runs the same A/B
     L1-fit heuristic as TILE. New `_regime_rm_b_descriptor` mirrors TILE Regime B
     — each core owns one 32-stick block-group × one W-column shard (offset
     `shard_col0`), all-gathers the K shard partials to the GLOBAL Σx²
     (`inv_W = 1/W` over the full row), normalizes its shard, untilizes, writes
     its columns. Wide-W RM no longer single-cores / OOMs.
- **Kernel count**: 7 → 3 (reader, compute, writer); **exactly one compute**.
- **Accuracy achieved** (RM Regime B mcast path, measured):
  - bf16 wide-W (incl. (1,1,32,8192), (1,1,64,8192), (1,32,8192), W-non-aligned
    (1,1,32,8190)): PCC ≥ 0.999, relRMS ≤ 0.0083.
  - fp32 wide-W: PCC ≥ 0.9999, relRMS ≤ 0.02.
  - TILE bf16 anchor byte-identical (regression anchor preserved).
- **Golden test progress**: **1683/1683 supported passing** (2940 skipped, 420
  xfailed) — identical to the Refinement 3 baseline, no regression. SUPPORTED
  unchanged (this is a non-registry structural refinement: 0 xpass-drift, 0
  supported_fail). test_regression.py + test_translated.py: 99/99.
- **Issues encountered**: None blocking. One test-only bug (asserted `.layout` on a
  torch tensor instead of the ttnn tensor) — fixed. The clean single-`eltwise_chain`
  pass-2 fusion the design references is still not expressible with the current
  helper library (noted in Refinement 1); the bounded chunked-streaming form is
  retained, now shared by both layouts.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_rm_regime_b.py` — 40
    cases proving wide-W ROW_MAJOR routes through the Regime B mcast all-gather
    (descriptor carries 2 semaphores) AND is numerically correct vs torch
    (bf16/fp32 × {no_gamma, gamma_tile, gamma_rm} × 5 wide shapes incl.
    W-non-aligned). All other suites (acceptance, layout_matrix 120, regime_b,
    precision_matrix 641) re-run green against the unified 3-kernel set.

## Refinement 5 — Remove the redundant DEST-level reduce-accumulate chunking
- **Date**: 2026-06-19
- **What was done**:
  - **Removed the PASS-1 `num_chunks` / `reduce_block` / `Accumulate` loop** from the
    unified compute kernel (`rms_norm_compute.cpp`). PASS-1 sum-of-squares is now a
    **single `square` (eltwise_chain BinaryFpu Mul over the whole shard) + single
    `reduce<SUM, REDUCE_ROW, BulkWaitBulkPop>`** producing the local Σx² directly — no
    cross-chunk accumulation. The `using ckl::Accumulate` import and the header
    dataflow comment were updated to match.
  - **Memory-budget reasoning (per /memory-budget-metal)**: the DEST-level chunking was
    a hand-rolled view of DEST capacity, not a real hardware limit — `ckl::eltwise_chain`
    and `ckl::reduce` are L1→L1 helpers that tile their own work through DEST internally,
    so a square / reduce over the full shard is one call regardless of width. The
    per-core resident shard (`Wt` Regime A, `Wt_s` Regime B, `Wt_padded` RM) is already
    L1-bounded by the host A/B W-split heuristic, so the second in-kernel chunking was
    pure redundancy. Confirmed empirically: across **every** golden-routed shape the
    resident shard is **≤32 tiles** (≤128 KB fp32) — see below — so growing `cb_squared`
    from `reduce_block` to the full shard width adds ≤32 tiles and never approaches the
    1.5 MB L1 limit. **Routing (RESIDENT_BUDGET_TILES / `_select_k`) was left unchanged**,
    so no shape re-routes and golden is unaffected.
  - **`cb_squared` resized** to the shard width in all four descriptors
    (`_regime_a_descriptor` → `Wt`, `_regime_b_descriptor` → `Wt_s`,
    `_regime_rm_descriptor` / `_regime_rm_b_descriptor` → `Wt_padded`).
  - **PASS-2 kept chunked — deliberate, not an oversight.** PASS-2's `num_chunks` loop is
    NOT a mirror of PASS-1: it is load-bearing. (a) The gamma path streams `x·recip` →
    `cb_normalized` → `·gamma` with `cb_normalized` sized to **one** `reduce_block` (the
    Refinement-1 fix that keeps per-core L1 from scaling with shard width); unchunking it
    would regrow `cb_normalized` to the full shard. (b) The RM path interleaves a per-chunk
    `untilize` so the writer can drain sticks block-by-block — the chunk granularity is the
    untilize granularity. Removing PASS-2 chunking would regress R1's memory bound and the
    RM untilize pipeline, so only the genuinely-redundant PASS-1 chunking was removed.
- **Accuracy achieved** (measured): all-ones multi-chunk shards (Regime A/B, TILE/RM,
  ±gamma) → max|out−1| < 0.05; random vs torch PCC ≥ 0.999 on
  [(4,1,512,512), (2,512,1024), (1024,1024), (1,1,32,32768), (128,8192), (1,1,64,12288)]
  (TILE) and [(1,1,32,1024), (1,32,1024), (1,1,32,8192), (1,32,8192)] (RM). Single-chunk
  (num_chunks==1) Phase-0 corner is byte-identical (single reduce ≡ old `Accumulate(.,0)`);
  multi-chunk bf16 is if anything slightly *more* precise (no per-chunk pack/reload
  rounding) and well inside the tolerance band.
- **Golden test progress**: **1683/1683 supported passing** (2940 skipped, 420 xfailed) —
  identical to the Refinement 4 baseline, 0 regression, 0 xpass-drift. test_regression.py +
  test_translated.py green. Non-registry refinement: SUPPORTED unchanged.
- **Issues encountered**: None. The single reduce over `cols=Wt` is correct for REDUCE_ROW
  (the LLK accumulates all input tiles into one DST register; output is always 1 tile, so
  DEST capacity is never the constraint for a row reduction regardless of width).
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_reduce_simplification.py` — 32
    cases targeting shards WIDER than DEST_AUTO_LIMIT (the shapes the old kernel chunked):
    all-ones exactness + random-vs-torch PCC across both regimes, TILE + ROW_MAJOR, ±gamma.
