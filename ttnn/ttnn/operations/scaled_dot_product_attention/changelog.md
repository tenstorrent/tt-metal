# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-06-19
- **What was done**: Initial Flash-Attention implementation via the incremental
  pipeline (planner → implementer → verifier). Tiled online softmax with O(S)
  memory — `cb_scores`/`cb_p` hold one `(1 × kv_chunk_t)` block; running m/l/O are
  the only KV-surviving state. Multi-core via `split_work_to_cores` over `B·H·Sq_t`
  units (embarrassingly parallel, no inter-core communication).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE_LAYOUT],
  alignment=[tile_aligned], attention_kind=[self, cross], kv_heads_mode=[mha],
  mask_mode=[none, causal], scale_mode=[auto, explicit]. EXCLUSIONS=[].
- **Accuracy achieved** (bf16, randn, seed 0, via
  test_scaled_dot_product_attention_precision_baseline.py over 4 shapes):
  PCC=0.99997 / 0.99997 / 0.99997 / 0.99996; max_abs_err ≤ 0.0137;
  mean_abs_err ≤ 0.0016; relative RMS err 0.0101 → 0.0114 as S/H scale.
- **Golden suite at Phase 0**: **140 / 140** supported cells passing
  (per `verifier_report.json`: supported_pass=140, supported_fail=0,
  xpass_drift=0, xfail_wrong_mode=0; 604 cells correctly xfail toward TARGET;
  40 non-registry rows = test_regression.py numerics + empty loose placeholder).
- **Issues encountered**:
  - No drift, no code-review fixes required — the implementation is clean and
    idiomatic (helpers used throughout, CB push/pop balanced, TensorAccessor,
    correct includes, correct broadcast/reduce dims).
  - Three documented design deviations, all forced by real LLK constraints and
    mirrored by the production SDPA op (not bugs): (1) bf16 intermediate CBs with
    fp32 DEST accumulation instead of fp32 CBs (Issue #13364 `pack_reconfig` hang);
    (2) scale folded after the QK matmul instead of into Q (datacopy→matmul-AB
    hang); (3) `mm_init` boot instead of `compute_kernel_hw_startup`+`mm_block_init`.
  - Precision limitation (no in-scope lever, documented in verification_report.md,
    NOT a refinement): bf16-CB storage of the running softmax statistics makes error
    grow across the KV recurrence under adversarial distributions (uniform/negative
    inputs in test_regression.py reach rel RMS up to 0.30–0.67). The fp32-CB lever
    hangs the device; HiFi2 + fp32_dest_acc are already set. Revisit when Issue
    #13364 is resolved.
- **Tests added**: test_scaled_dot_product_attention_precision_baseline.py
  (PCC + abs/RMS baseline over 4 shapes). Acceptance test
  (test_scaled_dot_product_attention.py, 24 cases) and golden + regression suites
  pre-existed; all 24 acceptance cases pass.
- **Refinement queue** (op_requirements.md): R1 numerical configurability
  (float32 + bfloat8_b + compute_kernel_config, /numeric-formats-metal) →
  R2 GQA/MQA (kv_heads_mode, reader already wired) → R3 non-tile-aligned
  (alignment w/h, /memory-layouts). Covers every TARGET−SUPPORTED gap.

## Refinement 1 — Numerical configurability (float32 + bfloat8_b + compute_kernel_config)
- **Date**: 2026-06-19
- **What was done** (descriptor-level only — zero compute-kernel changes; the
  helper-based kernel carried all data-format reconfig automatically):
  - `SUPPORTED["dtype"]` += `ttnn.float32`, `ttnn.bfloat8_b`.
  - Public entry point gained `compute_kernel_config: ttnn.ComputeKernelConfig = None`.
    Resolved via `init_device_compute_kernel_config(arch, cfg, …)` — `None` reproduces
    the Phase-0 defaults byte-for-byte (HiFi2, fp32_dest_acc_en=True, approx=False,
    full_sync=False); a user config overrides per-field.
  - Output dtype now follows `Q.dtype` (was hard-coded bf16) — satisfies the golden
    `output.dtype == input.dtype` contract for fp32/bf8b.
  - Program descriptor: input-side CBs (cb_q/k/v/mask) and the output CB (cb_out)
    follow the tensor dtype (`ttnn.tile_size(dtype)`); **all** matmul/reduce/eltwise
    intermediate CBs stay bf16 with fp32-DEST accumulation (Issue #13364 — fp32 CB
    storage hangs this LLK). fp32 inputs unpack through srcA/srcB (→ TF32) for the
    matmuls — production SDPA behavior. ComputeConfigDescriptor now driven by the
    resolved config. No `UnpackToDestFp32` tagging (intermediates are bf16 and feed
    FPU matmul/reduce — tag would be incompatible).
- **Accuracy achieved** (test_..._numerical.py precision matrix, randn, 5 shapes
  32×32…512×64/128×128 × {bf16,fp32,bf8b} × {HiFi4,HiFi2,LoFi} × fp32_acc{T,F},
  90/90 pass):
  - bf16:  PCC ≥ 0.99992 (HiFi4/2), ≥ 0.99896 (LoFi); rel-RMS 0.5–1.0% (HiFi), 6–8% (LoFi).
  - fp32:  PCC ≥ 0.99989 (HiFi4/2), ≥ 0.99887 (LoFi); rel-RMS ~same as bf16 (inputs
    unpack to TF32, so fp32 storage adds range/contract, not mantissa precision).
  - bf8b:  PCC ≥ 0.99980 (HiFi4/2), ≥ 0.99905 (LoFi); rel-RMS 1.7–2.2%.
  - bf8b + causal additive −inf mask: PCC 0.9999 (the flagged sharp edge — passes,
    no EXCLUSIONS needed). Full table in tests/.../precision_matrix_results.md.
- **Golden test progress**: **414 / 420** supported cells passing (Phase-0 was
  140/140 bf16; 324 cells correctly xfail toward R2/R3; 1 skipped loose placeholder).
  - `bfloat8_b`: **fully landed** — every tile-aligned/MHA bf8b cell passes.
  - `float32`: all pass **except 6 corners**, both left failing (NOT silenced in
    EXCLUSIONS, per the OOM / precision-near-miss protocol):
    1. `float32 + Q1x1x128x1024` (D=1024, 4 cells) — **hard L1 OOM**: static CBs
       grow to 2,767,616 B > 1,572,864 B because fp32 input CBs double the footprint
       (cb_k/cb_v ≈ 1 MB each at Dt=32). **Lever proven** (temp `kv_chunk_t=1` →
       all 4 run+pass). Deferred to **R4** (footprint-aware kv_chunk_t,
       /memory-budget-metal) rather than hardcoding an L1 heuristic into a
       numeric-formats refinement.
    2. `float32 + Q1x1x8192x64 + no-mask` (2 cells) — **precision near-miss**:
       rms=0.0206 > 0.02 target (PCC=0.9999). bf16-CB softmax-accumulator limit over
       2048 KV chunks (Issue #13364; fp32 CBs hang). Lever-less for the default-config
       golden — tracked in verification_report.md, not a refinement. (Causal S=8192
       fp32 passes — masking halves the accumulation depth.)
- **Issues encountered**: the 6 float32 corners above. The bf8b+causal and
  bf8b+non_aligned EXCLUSIONS candidates flagged in the R1 plan did **not**
  materialize (bf8b+causal passes; bf8b+non_aligned isn't reachable until R3 adds
  the alignment axis). EXCLUSIONS stays `[]`.
- **Tests added**:
  - tests/.../test_scaled_dot_product_attention_numerical.py — test_dtype_mask_scale
    (48: dtype×mask×scale×attn-kind, incl. bf8b+causal), test_compute_kernel_config_honored
    (18: config-knob plumbing), test_scaled_dot_product_attention_precision_matrix (90).
  - tests/.../test_sdpa_dtype_probe.py — minimal isolated dtype smoke test.
  - tests/.../precision_matrix_results.md — characterization table + observations.

## Refinement 2 — GQA / MQA (kv_heads_mode)
- **Date**: 2026-06-19
- **What was done** (op-file gate only — zero kernel / descriptor changes):
  - `SUPPORTED["kv_heads_mode"]` += `"gqa"`, `"mqa"`. This is the *only* change.
    The Flash-Attention pipeline already supported GQA/MQA end-to-end:
    - **Reader**: `head_group = H / H_kv`, `h_kv = h / head_group`,
      `kv_head_base = (b*H_kv + h_kv)*Skv_t` — each Q head reads its grouped KV
      head. Work split is per Q head (`B·H·Sq_t`), so GQA/MQA is embarrassingly
      parallel — no new distribution, no inter-core comm.
    - **Writer**: indexes output purely by Q head (`(b*H + h)*Sq_t + q`).
    - **`_check_structural`**: already enforces `H_q % H_kv == 0`.
    - **Mask head index**: `mh = (mask_H == 1) ? 0 : h` already handles
      `mask_H ∈ {1, H_q}` (broadcast and per-head masks).
  - No EXCLUSIONS change (stays `[]`), no compute/program-descriptor edit.
- **Accuracy achieved** (golden + new R2 unit tests, randn, TILE):
  - bf16  GQA/MQA: PCC ≥ 0.995, rel-RMS within 0.05 target across 8:2, 32:8,
    32:1, 12:4, 16:4 ratios, self + cross, none/causal/per-head masks.
  - fp32  GQA/MQA: PCC ≥ 0.999 (target 0.02 rms) on all tile-aligned shapes
    **except** the one long-context corner below.
  - bf8b  GQA/MQA: PCC ≥ 0.99, rel-RMS within 0.12 target (incl. causal).
- **Golden test progress**: **203 / 204** supported gqa/mqa cells pass; **24**
  non-aligned gqa/mqa cells correctly xfail (toward R3); registry total now
  624 expected-pass (was 420 at R1) + 120 xfail. The 140 Phase-0 bf16 cells
  stay green (MHA `validate()` byte-identical — widening SUPPORTED cannot
  regress MHA). `test_gqa_mqa_forward` (4 cases) now passes (were
  `UnsupportedAxisValue` rejections).
- **Issues encountered** (1 deferred corner, NOT silenced in EXCLUSIONS, per the
  precision-near-miss protocol):
  - `float32 + Q1x8x4096x128 (GQA) + no-mask + explicit-scale` — rms=0.0222 >
    0.02 fp32 target, PCC=0.99988. **Bounded precisely**: the same shape passes
    at bf16 (4/4), bf8b (4/4), fp32+causal (2/2), and fp32+none+auto — only
    fp32+none+explicit tips over the tight 0.02 target. This is **not** a
    head-remapping gap (GQA is bit-exact); it is the *identical* bf16-CB
    softmax-accumulator limitation (Issue #13364) that R1 already declared
    lever-less for the MHA equivalent (`Q1x1x8192x64` fp32 no-mask, rms=0.0206).
    Surfaced on one additional GQA long-context shape. **Lever-less** for the
    default-config golden: fp32 CBs hang the LLK (#13364); math_fidelity is
    accumulation-dominated, not matmul-dominated, and the default must stay
    HiFi2; the golden harness uses the default compute_kernel_config so the R1
    config surface can't reach it. Left red; tracked in op_requirements.md's
    bottom "Not a refinement" note (now lists both the R1-MHA and R2-GQA data
    points). **Not** filed as a standalone refinement — no in-scope lever, same
    rationale as R1/R4.
  - The 14 `test_regression.py` failures (large_magnitude / uniform / negative,
    all MHA) are pre-existing adversarial-distribution precision limits from
    Phase 0 — unaffected by R2 (those inputs are MHA, whose `validate()` path is
    unchanged).
- **Tests added**:
  - tests/.../test_scaled_dot_product_attention_gqa_mqa.py (50 cases):
    test_gqa_mqa_forward (27: {bf16,fp32,bf8b} × 9 LLM ratios), causal-mask (9),
    per-head-mask (9), cross-attention (3), non-divisible-heads structural
    rejection (1), MHA regression guard (1).
- **Checkbox**: `[~]` partial — both named axis values (gqa, mqa) fully landed
  and confirmed by 203 passing golden cells; one inherited fp32-no-mask-
  long-context precision corner (Issue #13364, lever-less) left red rather than
  excluded, so the literal "every gqa/mqa cell at supported dtype passes"
  Done-when is not 100% met.

## Refinement 3 — Non-tile-aligned shapes (alignment w / h)
- **Date**: 2026-06-19
- **What was done**: added `w_non_aligned` and `h_non_aligned` to
  `SUPPORTED["alignment"]` via **in-kernel** edge handling (no `ttnn.to_layout`/
  `tilize` wrapper — SDPA stays TILE-layout). Three distinct edges:
  - **Program descriptor**: tile counts (`Sq_t`/`Skv_t`/`Dt`) now round UP
    (were floor + a `tile-aligned` assert). `ttnn.from_torch(TILE_LAYOUT)`
    zero-pads the padded tile region, which makes the two "easy" edges fall out:
    - *w_non_aligned (D not %32)*: padded D-columns of Q/K are zero → contribute
      0 to the Q·Kᵀ contraction; padded D-columns of the PV output are zero (V
      padding is zero) and dropped on read-back (logical-shape slice).
    - *h_non_aligned (S_q not %32)*: padded query rows are zero → per-row softmax
      is independent and the writer's padded rows are dropped on read-back.
    Both were confirmed in an isolation probe (D=47 non-aligned + S_kv=64 aligned
    passes at PCC 0.99 with **zero** masking).
  - **The structural edge — S_kv not %32** (bundled into BOTH alignment values
    because every non-aligned INPUT also carries a non-aligned S_kv): the padded
    KEY columns of the last KV tile scored 0 and over-counted the softmax
    denominator. Diagnosed precisely on the golden subset before the fix: no-mask
    rms ~0.19 pcc ~0.999 (severity=precision, output scaled down by
    Z_correct/Z_inflated); causal pcc ~0.86 rms ~0.6 (severity=bug). 24/120 of the
    subset passed pre-fix — exactly the S_kv-aligned cases (`Q…32x50` S_kv=32,
    `Q…64x47` S_kv=64), proving the D/S_q edges were already correct.
  - **Fix** (mirrors production SDPA `fill_vertical_tile_bf16`): the reader
    generates a bf16 −inf vertical column tile (cols ≥ S_kv%32 = 0xFF80) on-device
    (no NoC read) once per work unit into `cb_pad_mask` (kv_chunk_t tiles, zeros
    except the last). Compute adds it to `cb_scores` on the **last KV block only**,
    before the row-max/row-sum — same `binary_add<…, BroadcastDim::None>` helper
    and broadcast as the user-mask add (streaming WaitAndPop walks all
    kv_chunk_t tiles). Robust for kv_chunk_t ∈ {1,3,4} (verified S_kv 47/33/50/65/100).
    No new sync primitives; reuses the existing helper. EXCLUSIONS stays `[]`.
- **Accuracy achieved** (golden tolerances: bf16 pcc 0.995/rms 0.05, fp32
  0.999/0.02, bf8b 0.99/0.12): all non-aligned cells within target. The
  dedicated `test_skv_keypad_denominator` (no-mask, S_q/D aligned, S_kv varied)
  confirms the denominator is now correct: post-fix rms well under target vs the
  ~0.19 pre-fix over-count.
- **Golden test progress**: **737 / 744** passing (was 624/744 at R2: +120
  non-aligned now pass — all 10 non-aligned INPUT shapes × 3 dtypes × 2 mask × 2
  scale, incl. bf8b+non_aligned which the R3 plan flagged as an EXCLUSIONS
  candidate but which passes cleanly). The 7 remaining failures are ALL
  `alignment=tile_aligned` fp32 pre-existing corners untouched by R3:
  `Q1x1x128x1024` L1 OOM ×4 (R4, `program.cpp:1487`), `Q1x1x8192x64` fp32 no-mask
  rms 0.0206 ×2 (R1, Issue #13364), `Q1x8x4096x128` GQA fp32 no-mask explicit
  rms 0.0222 ×1 (R2, Issue #13364). Left red per the OOM / precision-near-miss
  protocol (not silenced in EXCLUSIONS).
- **Issues encountered**: None new. The bf8b+non_aligned EXCLUSIONS candidate
  the R3 plan anticipated did not materialize — bf8b non-aligned passes. The
  per-unit pad-mask refill (advisory efficiency tradeoff vs a persistent no-pop
  CB) is negligible for the non-aligned shapes (num_units ≤ 32) and was chosen
  for correctness-first simplicity (reuses the exact user-mask `binary_add`).
- **Non-regression**: 74/74 acceptance + gqa/mqa, 156/156 numerical precision
  matrix, 130/130 new alignment tests; no hang under `--dev`, no race in non-dev.
- **Tests added**:
  - tests/.../test_scaled_dot_product_attention_alignment.py (130 cases):
    test_alignment_matrix (120: 10 edge shapes × {bf16,fp32,bf8b} × {none,causal}
    × {auto,explicit}, covering w/h/both non-aligned, cross-attn, GQA/MQA),
    test_skv_keypad_denominator (10: S_kv ∈ {47,33,50,65,100} × {bf16,fp32},
    directly targeting the key-padding denominator edge).
  - tests/.../test_sdpa_nonaligned_probe.py (3 isolation probes, kept as debug
    artifacts): from_torch zero-pad inspection, w_non_aligned+S_kv-aligned
    isolation, h+S_kv-non-aligned.
- **Checkbox**: `[x]` full — every `alignment ∈ {w_non_aligned, h_non_aligned}`
  cell at supported dtype/kv_heads passes; prior phases still green (the 7 red
  cells are inherited R1/R2/R4 tile-aligned fp32 corners, not regressions).
