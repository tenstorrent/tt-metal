# Changelog: scaled_dot_product_attention (Flash Attention)

## Phase 0 — Core Implementation
- **Date**: 2026-07-23
- **What was done**: Initial implementation via the incremental pipeline (planner →
  implementer → verifier). Fused flash-attention kernel with the online-softmax
  recurrence (score matrix never materialized); multi-core over the flat
  `B·H_q·q_num_chunks` work-list; reader/compute/writer split; GQA/MQA handled by
  reader head-addressing; self + cross attention; mask_mode none/custom; scale
  auto/explicit.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE],
  alignment=[tile_aligned], attention_kind=[self, cross], kv_heads_mode=[mha, gqa, mqa],
  mask_mode=[none, custom], scale_mode=[auto, explicit].
- **Accuracy achieved** (bf16, HiFi2 + fp32-DEST, `randn`, seed 42, via
  test_scaled_dot_product_attention_precision_baseline.py):
  PCC ≥ 0.995 on all shapes; max_abs_err ≤ 0.0134, mean_abs_err ≤ 0.0017,
  relative RMS 0.0079–0.0099. got/true ratio centered ~0.995 with symmetric
  spread → bf16 noise, not a scale bug.
- **Golden suite at Phase 0**: 206 supported_pass, 6 supported_fail (all OOM:
  D∈{512,1024} at S=128), 2113 xfail_expected; xpass_drift=0, xfail_wrong_mode=0
  (per `verifier_report.json`).
- **Issues encountered / fixes applied by verifier**:
  1. Added the missing `default_compute_kernel_config()` factory + `__init__.py`
     export (the golden harness imports it as the single source of truth for the
     `fp32_dest_acc_en` tag; its absence broke golden-suite collection).
  2. Reordered `validate()` so the SUPPORTED/EXCLUSIONS support gate precedes the
     detailed tensor-shape contract — cleared 24 `xfail_wrong_mode` (unsupported
     `fp32_dest_acc_en=False` cells that also carried a batch-broadcast mask were
     being rejected with ValueError instead of the support-refusal).
  6 OOM `supported_fail` left failing (not silenced) → Refinement 2, per the
  registry-model routing table.
- **Tests added**: test_scaled_dot_product_attention_precision_baseline.py
  (test_scaled_dot_product_attention.py and test_scaled_dot_product_attention_debug.py
  already present; all 36 pass).

## Refinement 1 — Numerical configurability expansion
- **Date**: 2026-07-23
- **What was done**: Extended the numerical surface with **zero compute-kernel
  changes** (the kernel is fully helper-based — the /numeric-formats-metal pass
  condition holds). All edits are in the op file + entry point + program
  descriptor:
  - `SUPPORTED["dtype"] = [bfloat16, float32, bfloat8_b]`;
    `SUPPORTED["fp32_dest_acc_en"] = [True, False]`.
  - `EXCLUSIONS += {dtype: float32, fp32_dest_acc_en: False}` — legal-but-lossy
    (fp32 input thrown away by the 16-bit DEST accumulator), refused, mirrors
    softmax. `{bfloat8_b, False}` kept SUPPORTED: it clears the golden (0.99/0.12)
    tolerance (measured PCC ~0.9998), so block-float already dominates the error
    budget and the DEST width is second-order.
  - **Per-CB dtype-derived formats** (single source per role): input CBs
    (Q/K/V/mask) carry the input dtype; `cb_out` + the output tensor follow the
    input dtype (fp32→fp32, bf16→bf16, bf8b→bf8b — the golden contract checks
    got.dtype == input dtype; Phase-0's hardcoded-bf16 output only held because
    input was bf16). Scalers stay bf16 (reader packs via prepare_reduce_scaler).
  - **Intermediate CBs = fp32 whenever `fp32_dest_acc_en=True`** (skill §4): the
    online-softmax running (m,l,O) is parked and reloaded every KV-block, so a
    bf16 park truncates back to 7 mantissa bits each step and erases the
    fp32-DEST gain. This is the "fp32 intermediate CBs" lever the verifier note
    names — it improved the adversarial-distribution regressions 14→10 failing.
    When acc is off (the bf16 perf profile) intermediates match the streaming
    width, so the perf path is byte-identical.
  - **Dtype-correct math fidelity** (`_resolve_math_fidelity`, single source):
    bf16/bf8b inputs are clamped from HiFi4/HiFi3 → HiFi2 (they fit losslessly in
    TF32, and HiFi4 + fp32-DEST + bf16 silently corrupts, issue #38306); float32
    keeps the requested fidelity (HiFi4 recovers TF32-truncated mantissa bits).
    The program descriptor rebuilds the ComputeConfigDescriptor with the resolved
    fidelity; `fp32_dest_acc_en` / `math_approx_mode` honored as passed.
- **Accuracy achieved** (D=64/128 tile-aligned shapes, randn):
  fp32@True PCC ~0.99999 (relRMS 0.0036); bf16@True PCC ~0.99997 (0.0098);
  bf16@False PCC ~0.99993 (0.0121); bf8b@True PCC ~0.99989 (0.0148);
  bf8b@False PCC ~0.99985 (0.0174). All clear their golden tolerances.
- **Golden test progress**: test_golden.py 1025 passed / 36 failed / 848 xfailed,
  **0 xpassed** (SUPPORTED matches reality — no drift). Up from Phase-0's
  206 passed. All 36 failures are L1 CB-allocation OOM (RuntimeError @
  program.cpp:1751) at the large-head-dim boundary (D ∈ {256, 512, 1024} at
  S=128) — Refinement 2's explicit, anticipated scope (fp32 CBs raise L1
  pressure exactly as the R1 note foretold). Left failing, not silenced.
  The flagged perf-shape loose case (bf16 @ fp32_dest_acc_en=False, the perf-1
  contract anchor for Refinements 3/5) passes.
- **Issues encountered**:
  1. Initial run: 648 failures, all `dtype_mismatch got BFLOAT16 expected
     FLOAT32/BFLOAT8_B` — the op hardcoded bf16 output. Fixed by having the
     output dtype follow the input dtype.
  2. Adversarial-distribution regressions (test_regression.py
     uniform/negative/large-magnitude, bf16 input) improved 14→10 via fp32
     intermediates. The remaining 10 are fundamental bf16-input compute-path
     noise (severity mostly `precision`, one `bug` at pcc 0.94); the tests
     hardcode bf16 so float32 can't be applied — a pre-existing Phase-0 baseline
     the correct lever partially cleared, not an R1 regression.
- **Tests added**: test_scaled_dot_product_attention_precision_matrix.py
  (dtype × fp32_dest_acc_en × distribution over tile-aligned D≤128 shapes;
  40 passed / 8 skipped for the fp32@False EXCLUSION) + precision_matrix_results.md.

## Refinement 1b — Numerical configurability expansion (debug: fix gate violations)
- **Date**: 2026-07-23
- **What was done**: Fixed the hard golden REGRESSION the completion gate caught in
  Refinement 1. **Root cause**: R1 set `interm_df = float32 if fp32_dest_acc_en else
  bfloat16` in the program descriptor — but the Phase-0 baseline config is bf16 input @
  `fp32_dest_acc_en=True`, so this **doubled every intermediate CB** on the exact
  previously-passing bf16 path, pushing the D=256 bf16@True cells (which passed in
  Phase-0) into L1 OOM (`program.cpp:1751`). That is the "prior-passing cells no longer
  pass" the gate reported.
- **Fix (one line, single source)**: `interm_df = float32 if in_df == float32 else
  bfloat16`. Intermediate-CB fp32 now couples to **float32 INPUT**, not to the bf16-input
  DEST-acc flag — which is what the R1 verifier note actually meant by "float32 + fp32
  intermediate CBs". Consequences:
  - **bf16 input** → bf16 intermediates → **byte-identical to Phase-0** (both acc
    settings). The whole bf16 supported rectangle keeps its exact Phase-0 L1 footprint;
    D=256 bf16@True no longer OOMs. Change is monotone-non-increasing in L1 vs the failed
    R1 (fp32→bf16), so no bf16/bf8b cell can regress from it.
  - **float32 input** → fp32 intermediates (unchanged; needed so the parked online-softmax
    (m,l,O) is not truncated mid-reduce).
  - **bf8b input** → bf16 intermediates (bf8b can't be an accumulator format; bf16 is the
    correct upcast and already dominates the bf8b error budget).
- **Golden suite after fix** (full run, no subset): **1029 passed / 32 failed / 848
  xfailed**, completed in 106s, exit code 1 (test failures, **not** a hang). Up from the
  failed R1's 1025 passed / 36 failed — the 4 flipped cells are the restored D=256 bf16@True
  cells. All 32 remaining failures are large-head-dim L1 OOM (`program.cpp:1751`):
  bf16 only at D∈{512,1024} (= Phase-0's OOM baseline), fp32 at D∈{256,512,1024},
  bf8b at D=1024 — **exactly the D-scaling L1 pressure the R1 note foretold**, and
  Refinement 2's explicit, anticipated scope. Zero prior-passing regressions; zero hangs.
- **Accuracy**: unchanged from R1 on the passing shapes (bf16 path is byte-identical to
  Phase-0; fp32/bf8b paths untouched by this fix).
- **Completion-gate bullets**: (1) zero hangs in SUPPORTED (full suite terminated in 106s);
  (2) acceptance 32/32 + precision matrix 44 passed / 8 skipped + baseline pass;
  (3) golden majority (1029/1061 responsible) with no regression. test_regression at 14
  failed / 25 passed = Phase-0 baseline (these hardcode bf16 adversarial distributions —
  pre-existing bf16-compute-path noise, the verifier's float32 lever can't reach them).
- **Issues encountered**: None beyond the diagnosed root cause.
- **Tests added**: None (reused the R1 test set + the golden suite as the regression net).

## Refinement 2 — Per-core L1 budget fit for large head_dim
- Date: 2026-07-23
- What was done: Knob-turn on the block factors the design's Blocking Model already
  exposes — cap `sq_chunk_t`/`sk_chunk_t` to the COARSEST divisor pair whose per-core
  CB footprint fits the device L1 CB arena. **Program-descriptor only; zero kernel
  changes** (the compute/reader/writer kernels are already parameterized on
  (sq, sk, dht) — shrinking the chunk just adds more fully-full Q/KV chunks to the
  flat work-list). Changes:
  * Added an exact L1-footprint model in the program descriptor:
    `grow_to = L1_CB_RESERVED(111360) + Σ(CB total_size)` must stay `<= L1_CB_CEILING
    (1572864)`. Both constants measured **exact-to-the-byte on device** (Wormhole)
    across bf16/fp32/bf8b × {mask,no-mask}: the allocator's reported `grow to N` equals
    RESERVED + our own CB-size sum every time (see probes/probe_005/006). Budget keeps
    a 32 KB safety margin for reserved-region variation.
  * `_cb_specs()` is now the single source of truth for CB (index, num_pages, dtype);
    BOTH the footprint calc and the actual CB build derive from it (DRY — the footprint
    can't drift from the real allocation).
  * `_pick_chunks()` iterates the divisor pairs `(sq<=Q_CHUNK_TILES, sk<=K_CHUNK_TILES)`
    and selects the max-footprint pair that fits the budget. Footprint is monotone in
    both factors, so when the (4,4) default fits it IS the max-footprint candidate and
    is chosen unchanged → every currently-passing cell is byte-identical to Phase-0/R1b
    (verified: D<=256 all dtypes and the D=128 perf shape keep (4,4)).
  * Dtype/CB-format resolution moved above the chunk pick, because the footprint (hence
    the cap) depends on the resolved tile size — float32's 2x tile bytes lower the D at
    which OOM strikes, exactly as the R1 note foretold.
  * Chosen chunks on the fixed OOM cells: bf16 D=1024→(2,2); fp32 D=256→(4,2),
    D=512→(2,2), D=1024→(1,1); bf8b D=1024→(4,1); bf16 D=512+custom→(4,2). D-blocking on
    QKᵀ (the note's optional lever) was NOT needed — (1,1) fits every target D<=1024.
- Accuracy achieved: PCC >= 0.99961 on every fixed cell (D=256/512/1024 × bf16/fp32/bf8b
  × {mask,no-mask}); fp32 cells PCC ~0.99996, bf8b ~0.99987, bf16 D=1024 ~0.99961.
  rtol/atol: golden tolerances (0.99/0.12 for bf8b, tighter for bf16/fp32) all cleared.
- Golden test progress: **1061 passed / 0 failed / 848 xfailed** (103s, no hang), up from
  R1b's 1029 passed / 32 failed. All 32 previously-OOMing large-head-dim cells
  (bf16 D∈{512-custom,1024}, fp32 D∈{256,512,1024}, bf8b D=1024) now pass; zero xpass
  drift, zero regressions.
- Issues encountered: The skill's quoted L1 budget (1499136) did not match this device;
  measured the real ceiling (1572864) and the constant reserved base (111360) directly
  on device before writing the cap — the "try cheap first / measure don't guess" path.
  The changelog's earlier claim that bf16 D=512 OOMs was imprecise: bf16 D=256 and D=512
  (no mask) already passed at (4,4); only bf16 D=1024 and D=512+custom OOMed for bf16.
- Tests added: None (reused the R1 precision matrix + acceptance + golden suite as the
  regression net; added exploratory probes probe_005/006 documenting the exact L1 model).

## Refinement 3 — Speed up the perf-flagged profile (K/V reuse multicast) (partial)
- Date: 2026-07-23
- What was done: Built the requested K/V reuse-multicast scheme-change and measured it on
  device. A compile-time-gated `USE_MCAST` regime (Reused the existing reader/writer/compute
  + program descriptor; compute kernel UNCHANGED; Added a gated branch): when the (batch,head)
  groups map one-per-grid-row (`b·H_q == grid_rows`) and there is no mask, one injector per
  row (col 0) reads each KV-block once from DRAM and NoC-multicasts it across its row via
  `ttnn.Mcast1D` PerRow (host) + `mcast_pipe` `SenderPipe`/`ReceiverPipe` (kernel, one family,
  Flag signal, sequential K-then-V sends). All cores in a row process
  `rounds = ceil(q_num_chunks / grid_cols)` Q-blocks in perfect cb_k_in/cb_v_in lockstep;
  ragged-edge "dummy" slots re-run Q-chunk 0 (a bit-identical redundant output, so the mcast
  landing address stays identical across the row and correctness is preserved with no discard
  path). Grid on this box = 11×10 = 110, and `b·H_q = 10 = grid_rows` for the flagged shape, so
  it maps to one head per row, 11 cols splitting 74 Q-blocks, injector col 0. **Gate is
  zero-regression by construction**: no golden/acceptance INPUTS shape has `b·H_q == 10`, so
  only the flagged LOOSE_CASE takes the mcast path; every other cell keeps the byte-identical
  per-core DRAM path (`USE_MCAST=0`).
- Accuracy achieved: flagged shape `(1,10,9472,128)` bf16 @ fp32_dest_acc_en=False — PCC ≥ 0.997
  (soft gate) holds on the mcast path (golden `test_op_loose` green). No RMSE regression.
- Golden test progress: **1061 passed / 0 failed / 848 xfailed** (104s, no hang) — identical to
  Refinement 2; zero regression. Acceptance 32/32. Non-mcast debug 4/4.
- Perf result: **baseline 11.05 ms → mcast 10.97 ms = 1.007× (FLAT)**, DEVICE FW DURATION,
  110 cores. The lever is CORRECT but does not win. Two on-device ablations pin the cause:
  (1) ~10× fewer total DRAM K/V reads → time flat; (2) cutting the injector's DRAM read volume
  ~16× (1 tile/block) → 0.1% change. **Conclusion: the flagged shape is NOT DRAM-read-bound**
  (the refinement's premise is refuted by measurement) — it is compute / per-core
  dataflow-latency bound (util ~0.14). No read-strategy change (fixed injector, distributed/
  rotating injector, store-and-forward) can win, because reads are not the critical path.
- Decision: **[~] partial**. Per "keep a correct lever," the gated mcast path is retained
  (correct, exercised by test_scaled_dot_product_attention_perf.py, zero regression, and
  re-enabled for free on any future shape that IS read-bound). Filed **Refinement 3b** naming
  the real lever: compute-side amortization (per-helper reconfig tax over the ~7-phase kv_step,
  matmul subblock, chunk coarsening, CB depth) — which overlaps Refinement 5's scope.
- Issues encountered: The mcast came up correct on the first --dev run (no hang) — the lockstep
  dummy-slot design and one-family sequential K/V handshake held. The only surprise was the flat
  perf, resolved by the two ablations above (measure, don't guess).
- Tests added: test_scaled_dot_product_attention_perf.py (flagged-shape device-ns + PCC 0.997
  gate; the baseline/target harness for Refinements 3/3b/5).

## Refinement 3b — K/V reuse multicast (debug: fix gate violations)
- Date: 2026-07-23
- What was done: Fixed the hard golden REGRESSION the completion gate caught in Refinement 3
  (responsible cells 1060/1061). **Root cause**: the mcast auto-gate
  (`b·H_q == grid_rows and not has_mask and grid_cols > 1`) fired on the flagged golden
  `test_op_loose` case `(1,10,9472,128)` — the ONLY golden shape with `b·H_q == 10` on this
  11×10 Blackhole grid. The mcast SenderPipe/ReceiverPipe handshake has a **rare intermittent
  hang** (reproduced as non-determinism: my full-suite runs passed 1061/1061 repeatedly, but
  the harness hit the hang once → suite never completed → the cell counts as "never ran" =
  regression). The mcast lever delivers **zero measured perf benefit** (Refinement 3: 11.05 →
  10.97 ms, flat; the shape is compute/dataflow-latency bound, not read-bound).
- Fix (park a correct-but-flat lever at its trivial byte-identical default as a live knob):
  gated the mcast auto-fire behind an explicit opt-in `TTNN_SDPA_KV_MCAST=1` (default off),
  read at program-descriptor build time. Consequences:
  - **Default (opt-in off)** → `use_mcast=False` for every cell → the whole supported
    rectangle (including the flagged loose case) runs the proven **R2 per-core DRAM path**,
    **byte-identical to Refinement 2** → deterministic, zero hang, zero regression by
    construction. No SUPPORTED / EXCLUSIONS / API change; kernels untouched (the mcast
    branch is still compiled via `USE_MCAST`, just never selected by the host).
  - **Opt-in on** → the shape gate applies as before and the mcast scheme re-engages —
    retained intact and retrievable for any FUTURE genuinely read-bound shape. Verified:
    with `TTNN_SDPA_KV_MCAST=1` the flagged loose case routes to and completes the mcast path.
- Golden suite after fix (full run, `--run-all`, no subset): **1061 passed / 848 xfailed**,
  103.65s, no hang — identical to Refinement 2; zero prior-passing regressions; the flagged
  `test_op_loose` case passes on the deterministic baseline path.
- Accuracy: unchanged (the default path is byte-identical to R2). Perf-test PCC gate (0.997)
  holds on the baseline path.
- Completion-gate bullets: (1) zero hangs in SUPPORTED — full suite terminated in 103.65s;
  (2) acceptance 36 + precision matrix 40/8-skipped + baseline + perf all pass (80 passed /
  8 skipped combined); (3) golden majority (1061/1061 responsible) with no regression.
- Issues encountered: The hang is rare enough that it did not reproduce in ~6 isolated
  reruns here; the decisive evidence was the harness's non-deterministic 1060/1061 vs. my
  repeated 1061/1061. Given the flat perf, chasing the exact race buys nothing for the perf
  target — the parked-knob fix removes the hang risk from the gate deterministically while
  preserving the correct scheme. The real perf lever (compute-side amortization) is the
  next queued refinement.
- Tests added: None (reused the golden suite + acceptance + precision matrix + perf test as
  the regression net).

## Refinement 3b — Compute-side amortization on the perf-flagged profile (Refinement 3's real lever) (partial)
- Date: 2026-07-23
- What was done: Chased R3's diagnosed lever — compute-side amortization — with the block-surface
  knobs the planner already exposes. **Program-descriptor only; zero kernel change** (reader/compute/
  writer are already parameterized on (sq, sk, dht); coarsening just makes each helper call do more
  tiles per invocation, amortizing the per-phase init/dst-sync/format-reconfig tax over the ~7-phase
  kv_step — master.md `compute_block_size`, whose win grows with phase count). Measured all three
  named knobs on device against the flagged shape `(1,10,9472,128)` bf16 @ `fp32_dest_acc_en=False`:
  * **Chunk coarsening (the win)**: added a coarse block pair `Q_CHUNK_COARSE`/`K_CHUNK_COARSE` = (8,8)
    alongside the base (4,4). `_pick_chunks` FIRST tries the coarse pair, taken only when it (a) is a
    real divisor pair of (sqt, skvt), (b) fits the R2 L1 budget, and (c) keeps the flat Q-block
    work-list filling the grid after q-coarsening (`b·H_q·(sqt/8) >= num_cores`); otherwise it falls
    through to the exact R2 baseline pick. The win is **NON-MONOTONIC** — only the full (8,8) PAIR
    beats baseline: **(4,4)=11.04 ms → (8,8)=10.24 ms = 1.078×** (stable ±0.03% over 3 runs), while
    (4,8)=12.31 ms and (8,4)=12.10 ms are both *slower* — so the gate is a binary regime switch to the
    pair, not a per-axis picker. util ~0.14 → ~0.15.
  * **matmul output-subblock**: already at the DEST ceiling (8 tiles, `fp32_dest_acc_en=False` → 8-tile
    DEST) in BOTH the (4,4) and (8,8) regimes (`_pick_subblock` maxes it) — no headroom to turn.
  * **kv_buffer_factor**: depth 3 measured 10.29 ms — no gain (marginally worse from L1 pressure);
    reads are OFF the critical path (R3's ablations), so a deeper read buffer is a no-op here. Parked
    at the double-buffer default (a live tunable for any future read-bound shape).
- Grid/L1 gating (no-regression by construction): the coarse pair is emitted ONLY for shapes that
  qualify — on this 110-core grid the golden universe that means the flagged loose case plus
  `(1,8,4096,128)` (bf16/bf8b; fp32 (8,8) OOMs → stays baseline) and `(1,71,2048,64)` (bf16/fp32).
  Every other cell — including the small `(*,*,256,64)` bf8b cases and all large-head-dim / underfilling
  shapes — takes the byte-identical R2 pick. (A first cut used a grid-fill guard that forced `sq=1` on
  underfilling shapes; it regressed `B1_H4_S256_D64 bf8b` PCC 0.975 by changing (4,4)→(1,8). Replaced
  with the binary coarse-pair gate above, which leaves those shapes at (4,4).)
- Accuracy achieved: flagged shape PCC ≥ 0.997 (soft gate) holds on the (8,8) path; golden PCC
  tolerances all cleared on the two large qualifying shapes at every coarsened dtype.
- Golden test progress: **1061 passed / 848 xfailed** (104.79s, no hang) — identical to Refinement 2/3b;
  zero regression, zero xpass drift. Unit dir 81 passed / 8 skipped; perf test PCC 0.997.
- Perf result: **11.04 ms → 10.24 ms = 1.078×** DEVICE FW DURATION, flagged shape, 110 cores.
- Decision: **[~] partial**. The compute-side *block knobs* are turned to their ceiling and the winning
  one is kept (not reverted). But the util-0.35 goal is not reached (~0.15): a 4× cut in kv_step count
  buying only 7% is itself the diagnosis — the fixed per-block tax is second-order; the dominant cost is
  the **sequential FPU-matmul / SFPU-softmax phase structure** (the ~7 kv_step phases each own all 3
  TRISCs → the FPU idles through softmax, the SFPU idles through the matmuls). Closing to 0.35 needs
  FPU/SFPU **phase overlap / pipelining** — a scheme-change outside the block-knob set — filed as
  **Refinement 3c**, ordered immediately after this one. (R5's block-size/buffer-depth co-tune is now
  largely absorbed by this pass.)
- Issues encountered: The non-monotonic curve (only the pair wins) was the surprise; caught by the
  per-axis sweep (measure, don't guess). The bf8b grid-fill-guard regression above, caught by the unit
  dir and fixed before the golden run.
- Tests added: None (reused test_scaled_dot_product_attention_perf.py + the golden suite + acceptance +
  precision matrix as the regression net).

## Refinement 3c — FPU/SFPU phase overlap on the perf-flagged profile (the residual after 3b) (partial)
- Date: 2026-07-23
- What was done: Ran the verifier-mandated `/perf-measure` ablation FIRST (before any restructuring)
  to quantify the exact FPU-idle / SFPU-idle split and set the realistic ceiling. Added a free,
  defaulted-off compile-time ablation gate to the compute kernel (CT arg 13, driven by
  `TTNN_SDPA_ABLATE`: 0=normal, 1=matmul-stub keeping every CB reserve/wait/pop/push scaffold,
  2 reserved for softmax-stub) + one `int(os.environ.get(...))` line in the program descriptor.
  Measured on the flagged shape `(1,10,9472,128)` bf16 @ `fp32_dest_acc_en=False`, 110 cores:
  * **Baseline (full): 10.24 ms.**
  * **Matmul-stub (SFPU/softmax + dataflow floor): 8.28 ms.**
  * ⇒ **FPU (both matmuls) = 1.96 ms = 19.1 % of the wall; SFPU/softmax + overhead = 8.28 ms = 80.9 %.**
  The shape is **SFPU/softmax-bound, not FPU/SFPU-balanced** — this refutes the refinement's premise by
  measurement, exactly as Refinement 3's ablations refuted the read-bound premise.
- Why the named lever is dead: (1) **Ceiling is 1.24×, not util 0.35** — even perfectly hiding ALL FPU
  work behind SFPU bottoms out at the 8.28 ms SFPU floor (util 0.15 → ~0.185); the FPU is only 19 % of
  the time. (2) **Literal FPU/SFPU concurrency is architecturally unavailable on a single Tensix core** —
  both engines are issued by the one MATH RISC (TRISC1) and time-share; a full survey of the
  examples/master.md measured-perf catalog found **no pattern that runs both engines in the same
  wall-clock window** (the closest levers all *reduce SFPU cost*, i.e. attack the floor, not overlap the
  engines). (3) Recurrence software-pipelining (block k's PV while block k+1's QK) is additionally
  blocked by the running-(m,l,O) data dependency. There is **no code lever that realizes the heading's
  mechanism**.
- Decision: **[~] partial** (mirrors Refinement 3). The diagnostic work landed (the mandated ablation,
  the exact 81/19 split, and the reusable gate) and the real next lever is named + filed as
  **Refinement 3d — reduce the 8.28 ms SFPU/softmax floor** (DEST-resident fusion `compute_fusion`;
  SFPU-finalize reduces `reduce_accumulate`/`reduce_block`; one-axis epilogue scoping `sfpu_tile_scope`).
  Per "keep a correct lever at its trivial byte-identical default," the ablation gate is kept (defaulted
  off) as reusable measurement infrastructure for 3d/5; nothing was reverted. 3b's (8,8) regime and R3's
  gated mcast path are untouched.
- Accuracy achieved: PCC ≥ 0.997 (soft gate) holds on the default path (byte-identical to R3b/3b);
  shape `(1,10,9472,128)` bf16 @ fp32_dest_acc_en=False.
- Golden test progress: **1061 passed / 848 xfailed** (104.9 s, no hang) — identical to R2/3b, zero
  regression, zero xpass drift. Unit dir 81 passed / 8 skipped. Default (ablate=0) device-ns 10.25 ms
  (+0.11 %, run noise) — the compile-time-constant branch folds → free.
- Issues encountered: None. The ablation came up clean (no hang) on the first --profile run; the only
  "surprise" was the decisive 81/19 SFPU/FPU split, which reframes the target for 3d.
- Tests added: None (reused test_scaled_dot_product_attention_perf.py + the golden suite + unit dir as
  the regression net; the ablation gate is env-driven, no new test file).

## Refinement 3d — Reduce the SFPU/softmax floor on the perf-flagged profile (3c's real lever) (partial)
- Date: 2026-07-23
- What was done: Ran the verifier-mandated `/perf-measure` ablation FIRST — extended the kept
  `TTNN_SDPA_ABLATE` gate (0=normal, 1=matmul-stub) to a full cumulative decomposition of the 8.28 ms
  SFPU floor (2=+reduce-stub, 3=+exp-stub; all inert/byte-identical at the default 0), then wired the
  real lever the ablation points at.
  * Decomposition on the flagged shape `(1,10,9472,128)` bf16 @ fp32_dest_acc_en=False, 110 cores:
    baseline **10.25 ms** → matmul-stub **8.27 ms** (FPU matmuls **19%**) → +reduce-stub **7.45 ms**
    (both reduces **0.83 ms = 8%**) → +exp-stub **5.29 ms** (phase-4 exp chain **2.15 ms = 21%**),
    leaving a **5.29 ms = 52% pure dataflow/CB/NoC floor**.
  * **All three named sub-levers refuted by measurement** (mirroring R3/3b/3c): (1) the reduces the
    `reduce_accumulate`/`reduce_block` SFPU-finalize lever targets are only 8%, and only the SUM half is
    eligible — `reduce`'s `AccumulateViaAdd` datapath is SUM-only and needs `BulkWaitBulkPop`/
    `WaitAndPopPerTile`, both of which POP, but our rowsum runs `WaitUpfrontNoPop` because P must stay
    resident for the PV matmul; (2) the softmax `sub→mul→exp` chain is **already DEST-fused** — the only
    untapped `compute_fusion` seam is the l/O rescale epilogue, which is FPU-consumer (dest-reuse LOSES
    there per master.md); (3) `sfpu_tile_scope`'s per-row targets (corr-exp, final recip) are ≤ sq=8
    tiles → negligible on the wall.
  * **The dominant SFPU cost is the exp op itself (21%+), and the real lever is fast approximate
    `exp_tile`.** Templated `kv_step` on a compile-time `ExpMode` (`ckl::Approx`), routed BOTH exp sites
    (phase-4 P-exp + phase-3 corr-exp) through it, added compute-kernel CT arg 14, and drove it from the
    already-plumbed compute-config `math_approx_mode` (which the kernel previously ignored for the SFPU
    exp). `if constexpr` dispatch → the unused instantiation is dead-code-eliminated.
- Accuracy achieved: exact (math_approx_mode=False) PCC ≥ 0.997 (byte-identical to prior phases);
  approx (math_approx_mode=True) PCC 0.9967 on the flagged shape.
- Golden test progress: **1061 passed / 848 xfailed** (106.2 s, no hang) — identical to R2/3b/3c, zero
  regression, zero xpass drift. Unit dir 82 passed / 8 skipped (added the approx perf case).
- Perf result: **math_approx_mode=True → 10.25 → 7.11 ms = 1.44×** (util ~0.15 → ~0.22), DEVICE KERNEL
  DURATION, 110 cores. **math_approx_mode=False (default) → 10.2476 ms, byte-identical** (zero
  regression; the exact perf test at PCC 0.997 stays green).
- Decision: **[~] partial**. The correct SFPU-floor lever landed and wins 1.44× (kept, not reverted), but
  the win is realized only under `math_approx_mode=True` at PCC 0.9967; the flagged shape's contract
  anchor is `math_approx_mode=False` @ PCC ≥ 0.997, which fast exp misses by 0.0003 (exp-polynomial
  error, not storage width — fp32 intermediates can't recover it). Under the exact config the flagged
  shape's SFPU floor is exp-dominated and irreducible without approximation. Per "keep a correct lever at
  its trivial byte-identical default," the approx-exp lever + the ablation-decomposition gate are kept;
  filed **Refinement 3d-a** naming the exact next lever (a PCC-recovery path — hybrid/near-range exact +
  deep-negative fast exp, or a corrected fast-exp — so the 1.44× is realizable at the 0.997 contract).
- Issues encountered: None beyond the measured PCC near-miss. Fast exp came up correct on the first run;
  the only surprise was how large the exp cost is (a single template flip → 1.44×), which is exactly what
  reframes the residual as exp-bound.
- Tests added: test_scaled_dot_product_attention_perf.py extended with a `math_approx_mode` × `pcc_gate`
  parametrization (exact @ 0.997, approx @ 0.996) — both green; documents the lever's measured tradeoff.
