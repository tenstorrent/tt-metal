# Changelog: genify_matmul

## Phase 0 — Core Implementation
- **Date**: 2026-11-17
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer →
  verifier). Regime R1 `interleaved_out_split`:
  - an independent M×N output split over the full device grid;
  - per-core blocks solved coarsest-first against the L1 budget;
  - the K-walk accumulated in L1 by the `matmul_block` helper (packer L1 acc, ragged last K-block);
  - one `ttnn.generic_op` dispatch per call.
- **SUPPORTED at Phase 0**:

  | Axis | Values |
  |---|---|
  | dtype | bfloat16 |
  | weight_dtype | bfloat16 |
  | layout | TILE |
  | fp32_dest_acc_en | True |
  | placement | interleaved (DRAM or L1) |
  | epilogue | none |
  | alignment | tile_aligned, m_non_aligned |
  | weight_batch | single |

- **Accuracy achieved**: PCC ≥ 0.9999985, max_abs_err ≤ 0.018, relative RMS 0.0017, ULP p99 3–4.
  Measured on 5 shapes (single tile through 2048×4096×2048 and K = 16384) with
  `test_genify_matmul_precision_baseline.py`. Bit-identical across repeated calls.
- **Golden suite at Phase 0**:
  - 94 supported_pass, 2088 xfail_expected, 0 xpass_drift, 0 xfail_wrong_mode
    (`verifier_report.json`);
  - 1 supported_fail: a harness L1-capture false-negative on a correctly raised `ValueError`.
- **Issues encountered (fixed by the verifier)**:
  - **Grid factorization.** Ranking W strictly first produced DRAM-traffic-pathological splits:
    9472×5120×3456 took 28 ms with every core reading all of A. It is now `choose_split()`: within
    `GRID_W_SLACK` (0.25) of the best W, take the min of aggregate read traffic *including* the
    block re-reads. Measured 0.21–0.46× device time on the mis-split shapes and no regression beyond
    noise on the guard set; golden geomean 0.95×.
  - **L1 CB budget.** It ignored fragmentation (a freed L1 tensor above a live one), which gave a
    CB/L1-buffer clash TT_THROW. It now ends at the lowest allocated L1 block.
  - **DRY.** The duplicate `TILE` constant was removed.
- **Tests added**:
  - `test_genify_matmul_precision_baseline.py`;
  - `test_genify_matmul_extended.py`: output-dtype override, fragmented L1, L1 weight;
  - `perf/test_genify_matmul_perf_guard.py`: opt-in device-ns guard set (`GENIFY_MATMUL_PERF=1`).
  - `test_genify_matmul.py` (acceptance, 19 tests) already existed and is unchanged.

## Refinement 1 — Numerical configurability (activation, weight and DEST formats)
- Date: 2026-11-17
- What was done:
  - **SUPPORTED widened.** `dtype` += bfloat8_b, float32. `weight_dtype` += bfloat8_b, bfloat4_b.
    `fp32_dest_acc_en` += False. `{"dtype": float32, "fp32_dest_acc_en": False}` stays the permanent
    EXCLUSION. No new EXCLUSION was needed: every cell this refinement unlocks passed out of the box.
  - **Formats needed no change.** Phase 0 already derived every CB page format and tile size from
    its tensor (A, B, output) and the partials from the DEST width (Float32 / Float16_b), plus
    `dest_limit` 4 / 8. The `matmul_block` helper owns the unpack/pack reconfig, so there is no
    kernel change.
  - **Precision fix: new knob `DEST16_BLOCK_K_MAX = 8`** (program descriptor;
    `Geometry.block_k_cap`).
    - *Problem.* With 16-bit DEST the solve put all of K into one DEST chain. The bf16 running sum
      swamps small addends, so the error grows with K: rel-RMS 0.04 / 0.07 / 0.18 at
      K = 2k / 4k / 16k. Stock ttnn.matmul measures 0.011 / 0.014 / 0.027 on the same shapes.
    - *Fix.* With 16-bit DEST, `block_k` is now capped at 8 tiles. The cross-K-block sum moves to
      the packer: L1 accumulation in `cb_partials`, kept Float16_b as specified.
    - *Effect.* K = 16k drops to rel-RMS 0.019. Cap 2: 0.020. Float32 partials only reached 0.0145,
      a marginal gain, so Float16_b is kept.
    - *Scope.* fp32 DEST is uncapped (`block_k_cap = tensor_k_tiles`), so the bf16 / fp32-DEST
      path's blocking is unchanged.
    - *Perf.* The cap also wins on device time from better read/compute overlap:
      64×16384×256 108 µs vs 143 µs uncapped; 384×1536×1024 89 vs 101 µs; 1×2048×1536 35 vs 41 µs;
      2048×2048×1536 LoFi 363 vs 377 µs. Cap 4 is within noise of 8 on time and slightly worse on
      LoFi, so 8 stays.
- Accuracy achieved (`test_genify_matmul_precision_matrix`, HiFi4, randn; full table in
  `tests/.../genify_matmul/precision_matrix_results.md`):
  - fp32 DEST:
    - bf16 out: PCC 0.999999, rel-RMS ≤ 0.0017;
    - bf8b out: PCC 0.99997, rel-RMS ≤ 0.0085;
    - fp32 out: PCC 1.000000, rel-RMS ≤ 0.00073. The tf32 partials reload (median got/true
      0.9994) stays well inside 0.005, so no `UnpackToDestFp32` was needed.
  - 16-bit DEST: bf16 out rel-RMS ≤ 0.019; bf8b out ≤ 0.025 (gate 0.15).
  - Shapes: 32×32×32, 1×2048×1536, 384×1536×1024, 2·64×4096×512, 64×16384×256.
  - The weight dtype barely moves these numbers, because the reference holds the quantized
    operands.
- Golden test progress, on the cells this refinement targets (interleaved, epilogue none, single
  weight):
  - cartesian: 60/60 supported pass, 12 xfail (the fp32 × acc=False EXCLUSION);
  - loose cases: 27/27 newly unlocked pass. These are the 20 Qwen prefill bf4b, 2 FLUX fp32
    timestep, 4 text-encoder bf8b and the PDL upsample W pass (bf8b, 16-bit DEST, L1 A/out).
  - After the cap change, the 37 cells with 16-bit DEST were re-run and still pass.
- Issues encountered:
  - **Uniform-input metric artifact.** An all-positive `torch.rand` input gives outputs whose
    mean dwarfs their std, so std-relative metrics only measure bf16 output rounding at the mean.
    The matrix uses zero-centred uniform instead.
  - **fp32 out at LoFi.** It measures rel-RMS ~0.025, which is the hardware's LoFi floor. The
    LOW_FIDELITY fp32 row was calibrated at HiFi2, which is what FLUX uses. Per the skill, the matrix
    gates below-HiFi4 cells on PCC only.
  - **Finding (Phase-0 split, not a format regression).** 512×5120×3072 picks grid (1,1,96), so
    every core reads all of A. That costs 1.38 ms in bf16 and 1.33 ms with a bf8b weight, against
    139 µs stock. The 2-D factorizations that would cut A traffic fail `_placement_for`
    (gn = 12 > gx = 11, or gm = 12 > gy = 10). This is Refinement 3's mcast topology / factorization
    co-tune.
  - **Format vs bf16 twin (device ns).**
    - 2048×5120×4160 LoFi: bf4b weight 1.02 ms vs bf16 1.55 ms. The bf4b tiles are 576 B, so the
      solve picks coarser K-blocks.
    - 512×5120×3072 HiFi2: bf8b 1.33 ms vs bf16 1.38 ms.
    - 1×6144×6144 HiFi2: fp32 A 401 µs vs bf16 305 µs. A is read whole by each of the 96 cores,
      so 2× A bytes cost 1.3×.
- Tests added:
  - `test_genify_matmul_precision_matrix.py`: 540 cases, 5 shapes × A {bf16, bf8b, fp32} ×
    B {bf16, bf8b, bf4b} × acc × {HiFi4, HiFi2, LoFi} × {randn, uniform}. The fp32 × acc=False
    cells assert `ExcludedCell`.
  - `perf/test_genify_matmul_perf_guard.py::test_perf_formats`: one representative per new format
    path (bf4b LoFi, bf8b HiFi2, fp32 A, bf8b × bf8b 16-bit DEST ×2, capped 16-bit-DEST K-walk).
    Baseline device ns: 1016036, 1343873, 404437, 20310, 44777, 108567.

## Refinement 2 — Fused epilogues: bias, bias_relu, swiglu, addcmul
- Date: 2026-11-17
- What was done: `SUPPORTED["epilogue"]` is now {none, bias, bias_relu, swiglu, addcmul}. Every
  epilogue runs inside the same single `generic_op` program, on the block accumulated in L1, before
  the output is packed. There is no second pass over the output and no host-side or ttnn-op epilogue.
  - **Reused**: the R1 split and blocking, `cb_partials`, the matmul_block call (now templated on
    `LastBlockTarget`), the writer, and the reader's K-walk.
  - **Compute**: with an epilogue, `matmul_block<…, Interm>` lands the block in `cb_partials` (once
    per batch slice), then `pack_reconfig_l1_acc(0)`, then one epilogue block op:
    - bias / bias_relu: `add_bias_bcast_rows<RowBroadcast, TileRowMajor>`, with PACK_RELU set and
      reset around it for bias_relu;
    - swiglu: `eltwise_chain` over (block_m, block_n/2), Strided gate/up `CopyTile` + a
      kernel-local `Silu` chain element + `MulBinary`;
    - addcmul: `eltwise_chain` with an FPU `Mul` against the row-broadcast gate, then a `CopyTile`
      of the residual and an SFPU `AddBinary` (no DEST reuse ahead of an FPU op).
    - The epilogue-none path is unchanged (`LastBlockTarget::Out`).
  - **Reader**:
    - swiglu gate/up column permutation inside each N-block;
    - the bias / gate N-block row, read once per block and pushed after the first batch's K-walk so
      it never stalls the K-block prefetch;
    - residual row groups (sb_h × block_n, depth 2) after each batch's K-walk.
  - **Host**:
    - `q_n ≥ 2` for swiglu, `out_block_n = block_n/2`, and the writer region in output columns;
    - the `E_bias` / `E_addcmul` / `cb_partials` terms are inside `_footprint`, so the solve
      prices them; `_block_k_max` derives its fixed part from `_footprint`;
    - bias / gate / residual join the tensor list, and the `validate()` gate requires
      interleaved TILE epilogue tensors.
  - **Knobs** (single source in the program descriptor): `DEPTH_RESIDUAL = 2`,
    `EPILOGUE_BLOCK_TILES = 4` (measured flat, 1708 → 1696 µs, kept as a live knob),
    `SWIGLU_FAST_SILU = True` (see below).
- Accuracy achieved (PCC / relative RMS, limit in brackets):
  - PERF FOCUS 1, Qwen prefill 2048×5120×8704 bf16×bf4b LoFi swiglu: 0.999988 / 0.0085 [0.9995 / 0.03].
    The other Qwen prefill and FLUX swiglu cases are the same to 3 digits.
  - FLUX addcmul (128 / 256 / 2048 × 6144×3072, HiFi2): 0.999994 / 0.0037–0.0038.
  - FLUX time_mod 1×6144×52224, fp32 A, bias → bf16: 0.999991 / 0.0060.
  - 384×1536×1024, fp32 out: bias 0.00071, bias_relu 0.00086, swiglu 0.00159, addcmul 0.00077
    [0.99999 / 0.005].
  - 384×1536×1024, bf8b with 16-bit DEST: rms 0.010–0.024 [0.15].
  - 384×1536×1024, bf16×bf4b: rms 0.0017–0.0025 [0.01].
- Golden test progress (slices run):
  - cartesian interleaved epilogue cells: 240/240 pass, 48 xfail (the permanent fp32 × acc=False
    EXCLUSION);
  - interleaved loose epilogue cases: 13/13 pass, among them PERF FOCUS 1 (prefill 2048 swiglu,
    bf4b), PERF FOCUS 4 (FLUX 256 swiglu), the 3 FLUX addcmul cases and the fp32 time_mod bias.
  - The epilogue cells on height / block / dram-sharded placements stay xfail until Refinements 4, 5
    and 7.
  - Unit dir: 611 passed, 42 skipped (env-gated perf guard).
- Perf (device ns, fused vs the epilogue-none twin of the same shape and formats, one fresh run each):
  - swiglu 2048×5120×8704 bf4b LoFi: 1655 µs vs 1572 µs;
  - addcmul 2048×6144×3072 HiFi2: 1773 µs vs 1716 µs;
  - bias_relu 2048×256×512 bf8b: 20.4 µs vs 20.8 µs (noise);
  - bias 384×1536×1024: 102.9 µs vs 101.6 µs.
  - The split and the block are unchanged by the epilogue (11×9, 6×32 on PERF FOCUS 1). The output
    CB is half the size, which gives K-block 9 instead of 8.
  - Ablation with the chain compute stubbed (`CKL_ELTWISE_CHAIN_SKIP_COMPUTE`) gives 1564 µs, the
    same as the twin. The Interm / `cb_partials` plumbing is free, and the remaining overhead is the
    epilogue's own compute, which runs serialized after the matmul on each core.
  - That compute was SFPU-bound: the fp32-precision sigmoid inside silu took ~1.7k cycles per output
    tile. `SWIGLU_FAST_SILU` (bf16-precision sigmoid, applied only to non-fp32 outputs) cut it from
    124 to 83 µs, with swiglu accuracy unchanged at 3 digits.
- Issues encountered: none blocking.
  - The precompile farm's best-effort up-front compile fails on this box with an sfpi toolchain
    mismatch (`__xtt_vector8`). The tests compile locally and are unaffected.
  - **Finding for Refinement 3.** The swiglu epilogue still costs ~5% of PERF FOCUS 1. It sits on
    the critical path because each core runs it after its last K-block. Two levers: overlap it with
    the next block's K-walk (only matters when there are several blocks per core), or fold
    silu·up into the last K-block's PostComputeFn with a subblock-local gate/up permutation. The
    second needs a helper change: the packer would have to pack only half the subblock.
- Tests added:
  - `test_genify_matmul_epilogue.py`, 40 cases:
    - 4 epilogues × 5 shapes (single K-block, 2-D grid, folded batch, M = 1, deep ragged K);
    - × 3 formats (bf8b with 16-bit DEST, fp32 out, bf16×bf4b with a bf8b bias);
    - × L1 A / out;
    - × a pinned multi-block-per-core walk.
  - `perf/test_genify_matmul_perf_guard.py::test_perf_epilogues`: one representative per epilogue
    path at the model config, each with its epilogue-none twin.
