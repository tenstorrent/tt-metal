# matmul — changelog

## Phase 0 — 2026-06-27 — fused 2D dual-multicast matmul (initial implementation)

**What was done.** Implemented `C = A @ B` as fused on-device kernels with the
2D dual-orthogonal-multicast topology from `op_design.md`:

- **Reader (`matmul_reader.cpp`, NCRISC):** per output block, per K-block, the
  first column (X==0) reads its activation row-block from DRAM and multicasts it
  ACROSS the grid-row; the first row (Y==0) reads its weight column-block and
  multicasts it DOWN the grid-column. Uses `dataflow_kernel_lib::SenderPipe` /
  `ReceiverPipe` (`mcast_pipe.hpp`) for the NoC-multicast + semaphore handshake,
  and raw `TensorAccessor` + `Noc::async_read` for the DRAM tile gather (no
  kernel-lib helper covers interleaved tile addressing). in0 phase fully precedes
  in1 phase each K-block (acyclic — in0 confined to rows, in1 to columns).
- **Compute (`matmul_compute.cpp`, TRISC):** one `compute_kernel_lib::matmul_block`
  per output block (all defaults: `last_block_target=Out`, `SubblockMajor`,
  `init_mode=Short`, `WaitAndPopPerKBlock`, `reconfig=INPUT_AND_OUTPUT`). The
  helper loops the K-blocks internally and spills/reloads via `cb_interm`. Boot:
  `compute_kernel_hw_startup` + `mm_block_init` once.
- **Writer (`matmul_writer.cpp`, BRISC):** drains one SubblockMajor output block
  from `cb_out` and scatters tiles to interleaved DRAM, skipping out-of-range
  (ragged/phantom) tiles.
- **L1 bounded for arbitrary M/N/K:** K streamed in K-blocks of
  `in0_block_w = find_max_divisor(Kt, 4)`; M/N output blocks bounded by an L1
  budget with an outer per-core block loop run lock-step across the grid.

**Accuracy (measured).** fp32 activation + fp32 weight, HiFi4, `fp32_dest_acc_en=True`:
- PCC = 1.000000 on 256x256, 512x512, and batched (1,2,128,256)@(256,128).
- relative RMS ≈ 0.0013–0.0017.
- Acceptance suite tolerance is PCC ≥ 0.999; golden tolerance (float32, acc=True)
  is (PCC 0.999, RMS 0.02). Comfortably inside both.

**Tests.**
- Acceptance (`tests/.../matmul/test_matmul.py`): **16/16 pass** in BOTH `--dev`
  (watcher clean) and non-dev (production timing — no race in the mcast handshake).
- Golden (`eval/golden_tests/matmul/test_golden.py`): **20 passed, 1 skipped,
  646 xfailed**. Every Phase 0 SUPPORTED cell passes; all out-of-contract cells
  xfail-strict; NO XPASS-strict (SUPPORTED/EXCLUSIONS consistent with behavior).

**SUPPORTED (Phase 0).** dtype=float32, weight_dtype=float32, layout=TILE,
fp32_dest_acc_en=True, alignment=tile_aligned, weight_batch=single.
**EXCLUSIONS:** `{dtype=float32, fp32_dest_acc_en=False}` (maxed input demands
maxed accumulator).

**Issues encountered.** None — passed first device run.

**Advisory deviations (from `op_design.md`).**
- The design's RISC-split table suggests the reader as "one source, 4
  KernelDescriptors keyed by per-group CT role flags `(is_in0_sender,
  is_in1_sender)`." Implemented instead as a SINGLE reader KernelDescriptor with
  per-core RUNTIME role flags (derived from `grid_row`/`grid_col`) and per-K-block
  pipe construction. This realizes the IDENTICAL inter-core contract (same 4
  semaphores, same sender/receiver roles, same in0→in1 ordering) with simpler host
  wiring. Per-call pipe construction is the proven-safe pattern from
  `pipe_rotating.cpp` (the ReceiverPipe ctor's INVALID always precedes the sender's
  VALID because the sender's mcast is gated behind the receiver's ack). Verified
  correct under both `--dev` and non-dev timing.
- `cb_out` is double-buffered (×2) for writer pipelining; `cb_interm` is ×1. The
  design listed the out-CB ×2 as optional and interm as ×1 — within latitude.

**Follow-up axes (TARGET, not yet implemented — all currently xfail-strict).**
mixed/low precision (`dtype`/`weight_dtype` ∈ {bfloat16, bfloat8_b}),
`fp32_dest_acc_en=False`, non-tile-aligned M/K/N (`alignment` masking paths),
batched weight (`weight_batch=batched`).

## Phase 0 — 2026-06-27 — verification (incremental-verifier)

- **What was done**: Code review, golden-suite + `eval.verify_supported` run,
  precision baseline, refinement queue. One helper-usage fix applied (below).
- **Code-review fix**: `kernels/matmul_compute.cpp` — replaced the external
  `for (i < total_blocks) matmul_block<>(... batch=1 ...)` loop with a single
  `matmul_block<>(... batch=total_blocks ...)` call. This is the helper's
  documented matmul-only pattern (init once, internal batch loop) — strictly fewer
  `mm_block_init_short` re-issues, byte-identical CB push/pop counts (multicast
  lock-step unchanged). Verified numerically identical and no regression on the
  multi-block large/wide-K golden shapes (where `total_blocks > 1`).
- **SUPPORTED at Phase 0** (unchanged — verification only): dtype=[float32],
  weight_dtype=[float32], layout=[TILE], fp32_dest_acc_en=[True],
  alignment=[tile_aligned], weight_batch=[single].
- **Accuracy achieved** (test_matmul_precision_baseline.py, 4 shapes):
  PCC ≥ 0.99999 (min 0.99999414 at 512×4096×4096); max_abs_err 0.037→63.9
  (tracks deep-K magnitude); mean_abs_err 0.0075→0.37; relative RMS 0.0012→0.0073
  (all ≤ the golden 0.02 band).
- **Golden suite at Phase 0** (`eval/eval_test_runner.sh --no-precompile
  --no-jit-server`, cold, current kernels): registry-driven `test_golden.py`
  = **20 / 20 supported pass, 646 xfail_expected, 0 loud signals**
  (supported_fail / xpass_drift / xfail_wrong_mode all 0). The 3 loud signals in
  the raw whole-directory verifier output are all `test_translated.py` artifacts
  (LoFi-fidelity precision; a K-mismatch rejection-expecting test; a batched-weight
  refusal on a non-xfail test) — none are registry drift or kernel bugs; see
  `verification_report.md`.
- **Issues encountered**: Kernel binary cache staleness — `ttnn.generic_op` keys
  the JIT cache on kernel source PATH (not content), so kernel `.cpp` edits silently
  reuse the stale binary. Proven via a deliberate-compile-error probe; worked around
  by `rm -rf`-ing the `matmul_{compute,reader,writer}` cache dirs + `--no-precompile`.
  Documented at the top of `op_requirements.md` for refinement implementers.
- **Tests added**: `tests/.../matmul/test_matmul_precision_baseline.py` (4 shapes,
  PCC + abs/RMS). Acceptance `test_matmul.py` unchanged (immutable spec), 16/16 pass.
- **Artifacts**: `verification_report.md`, `op_requirements.md`, `verifier_report.json`.

## Refinement 1 — Numerical configurability (dtypes + fp32_dest_acc_en) — 2026-06-27 — **[~] partial**

- **What was done**:
  - **SUPPORTED widened** (`matmul.py`): `dtype` and `weight_dtype` (independent
    axes) → `{float32, bfloat16, bfloat8_b}`; `fp32_dest_acc_en` → `{True, False}`.
    Kept the mandated `{dtype=float32, fp32_dest_acc_en=False}` EXCLUSION. No new
    EXCLUSIONS added (see "deferred" below — the residual misses are precision
    near-misses, deliberately left failing per protocol, not silenced).
  - **Program descriptor dtype-awareness** (`matmul_program_descriptor.py`):
    - `cb_interm` format is now dtype-aware. The matmul_block last-K-block
      "pack-to-out" data-format reconfig is gated on
      `(packer_l1_acc || fp32_dest_acc_en)` (`matmul_block_helpers.inl:394`), so:
      `fp32_acc=True` → interm = **Float32** (fp32 K-accumulator; the gated
      reconfig swaps the packer to the output format for the final pack, so out
      may be any dtype); `fp32_acc=False` (software spill) → interm **must equal**
      the output format (no reconfig fires before the final pack).
    - **L1 footprint estimate is per-CB dtype-aware** (each of in0/in1/out/interm
      carries its own tile-byte size; Phase 0 used a single fp32 size).
    - **Effective compute config** clamps **HiFi4→HiFi2 for bf16/bf8b inputs**
      (issue #38306 on Wormhole B0: HiFi4+fp32_dest_acc_en+bf16 silently corrupts
      the K-accumulator; bf16's ≤7 mantissa bits gain nothing from HiFi4, so the
      clamp costs zero precision — verified A/B in probe_004). fp32 keeps HiFi4.
  - **Lever B (compute-kernel change, narrowly scoped)** (`matmul_compute.cpp` +
    descriptor): for **bf8b output + acc=False** only, opt into hardware packer-L1
    K-accumulation (`matmul_block<false,true>`). This makes the gated reconfig fire
    so `cb_interm` can be **Float16_b** while `cb_out` stays bf8b — the running
    K-sum accumulates in bf16 instead of re-quantizing to bf8b on every K-block
    spill. Drops deep-K bf8b acc=False relRMS **0.385 → 0.022**. Every other path
    (fp32 / bf16 / any-dtype acc=True / bf8b acc=True) keeps the proven software
    spill/reload (`matmul_block<>` unchanged). #28800 (packer-L1 vs fp32_acc) does
    not apply — this branch is acc=False only.
- **Accuracy achieved** (golden tolerance bands, all PASS unless noted):
  - bf16 (eff): acc=True PCC ≥ 0.99998 relRMS ≤ 0.012; acc=False PCC ≥ 0.999
    relRMS ≤ 0.07 up to K=4096.
  - bf8b (eff): acc=True PCC ≥ 0.99990 relRMS ≤ 0.014; acc=False (Lever B) PCC
    ≥ 0.9998 relRMS ≤ 0.022 even at K=8192.
  - fp32: unchanged (PCC ≥ 0.99999, relRMS ≤ 0.0073) — byte-identical path.
- **Golden test progress**: **298 / 300 supported cells pass** (was 20/20 at
  Phase 0); 366 xfail_expected; 1 skipped; **0 XPASS / 0 supported-vs-excluded
  drift**. (Full fresh-cache re-run after the kernel edit.)
- **Issues encountered / deferred** (the 2 residual golden fails):
  - `A256x8192` (K=8192) with **bf16 OUTPUT + acc=False** (bf16/bf16 relRMS
    0.1279, bf16/fp32 relRMS 0.1254 vs golden band 0.10; PCC 0.997). This is the
    fundamental **16-bit-DEST accumulation floor** (~O(√K)) — the expert-debugger
    confirmed it is K-block-independent and Lever-B-immune (bf16 interm is already
    bf16; the error is the in-DEST FMA rounding, not the spill). The **only** fix
    is `fp32_dest_acc_en=True`, which these cells deliberately disable. NOT added
    to EXCLUSIONS: it is shape-dependent (deep-K only; shallow/medium-K bf16
    acc=False all pass) and cannot be expressed as an axis-cell without
    over-refusing ~100 working cells. Left failing per the precision-near-miss
    protocol; filed as **Refinement 1b**.
  - bfp8_pack_precise was probed (probe_005) and had **zero** effect — confirming
    the dominant deep-K acc=False error is DEST accumulation, not bf8b packing.
- **Tests added**:
  - `tests/.../matmul/test_matmul_precision_matrix.py` — dtype × weight_dtype ×
    fp32_acc × {HiFi4,HiFi2} × 4 shapes (PCC-asserted, all metrics printed):
    **120 passed, 24 skipped** (EXCLUSION).
  - `tests/.../matmul/precision_matrix_results.md` — characterization table.
  - probes 003–005 (A/B clamp, #38306, bfp8_pack_precise lever).
  - Acceptance `test_matmul.py` (16/16) + `test_matmul_precision_baseline.py`
    (4/4) unchanged — fp32 non-regression confirmed.
- **Sub-agents**: ttnn-expert-debugger (diagnosed the deep-K floor; found & landed
  Lever B; full audit trail in git `3b8c7f01d5`..`792ac69661`).
