# Verification Report: wq_b_linear

Board: Blackhole **p150** (11×10 workers, `dram_grid_size` 8×1, AICLK 1350 MHz). `feature_spec.py`'s SOL (16,018 ns) was computed for a p100a with 7 DRAM channels. With 8 channels (512 GB/s) the same DRAM bytes (7,176,192 B) give **SOL ≈ 14,016 ns on this board**, which is also the op model's DRAM lower bound (14.02 µs).

## Code Review

Fixes applied:
- **DEST subblock limit tied to the config** (`wq_b_linear_program_descriptor.py`). `DEST_TILE_LIMIT = 8` was a constant, and the compute config hard-coded `fp32_dest_acc_en=False`. Both now come from `compute_config.fp32_dest_acc_en` (`_dest_tile_limit()`: 8 for 16-bit DEST, 4 for fp32 DEST), so there is one source of truth. There is no behaviour change today because `validate()` admits only `False`.
- **`block_k_tiles` precision ceiling documented** next to `BLOCK_K_TILES` (see Precision Baseline). This is a hard constraint on the perf knob, not a style note.
- **L1 ledger currency**: the symbol table, data-movement budget and split-search notes still stated `num_k_blocks = 12` / 12 rounds × 40,960 B from the design's kb 4. They are corrected to the built kb 8 (6 rounds × 81,920 B), and a verifier audit section is appended.

Checked and found conformant (no change needed):
- Kernel syntax `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor` everywhere.
- CB push/wait balance:
  - x: 6 × 40 pushed by the worker writer = 6 × 40 popped by compute. The sender core's x-reader pushes 6 × 40 and the multicast kernel pops 6 × 40.
  - W: 6 × 16 = 6 × 16.
  - out: 10 = 10.
- Multicast uses the helper (`ttnn.Mcast` + `McastArgs` `sender().send()` / `receiver().receive()`), not raw semaphores. Compute is one `matmul_block` helper call with packer L1 accumulation. No raw compute APIs.
- One dispatch per call: a single `ProgramDescriptor` passed to `ttnn.generic_op`.

Prompt `## Rules` check:
- Grid and coordinates come from the device: `compute_with_storage_grid_size`, `dram_grid_size`, and the Mcast helper's physical mapping.
- Each output tile's K sum is owned by one core, so the result is deterministic (`test_regression` passes, 20 repeats bit-identical).
- The output is L1 interleaved, bf16, TILE.
- `validate()` raises `ValueError` for shape and placement violations.

No MUST violations.

Design conformance:
- **Algorithm, topology, split** match `op_design.md`: an N-split over 64 workers (an 8×8 rectangle) plus 1 x-sender. x is read once from DRAM and multicast to resident CBs with no handshake. Each worker streams only its own W columns, in the bank-spread column map.
- **Knobs** are host constants with derived quantities: `BLOCK_N_TILES`, `BLOCK_K_TILES`, `W_DEPTH_BLOCKS`, `OUT_DEPTH_BLOCKS`. `num_workers`, `num_k_blocks`, CB sizes, the round size and the subblock are all derived from them.
- **Implementer deviations, both measured**: kb 4 → 8 (Counter-mode rounds have a ~1 µs fixed cost each), and the sender split over NCRISC (DRAM read, NoC0) and BRISC (multicast, NoC1). Both are recorded in `l1_ledger.md`.
- **Block expression**:
  - The W reader is one barrier and one push per K-step (16 tiles).
  - The writer pushes 40 tiles per x round and stores the output block with one barrier.
  - The sender multicasts one 81,920 B round per `send()`.
  - No per-tile handshakes anywhere.
- **Performance conformance**: 65 of 110 cores, as the design chose (the op-model curve flattens at 65). However, the x-delivery path runs well below its modelled rate; see Performance. That is filed as perf Refinement 1, not as a defect, because the topology is as designed.

## Registry Conformance

- `INPUT_TAGGERS = {}`, `SUPPORTED` (`dtype`, `weight_dtype`, `layout`, `fp32_dest_acc_en`), `EXCLUSIONS = []` and `validate()` are all present.
- `validate()` checks SUPPORTED per axis, then EXCLUSIONS, then the shape/placement contract (`ValueError`). It is the entry point's first line. `default_compute_kernel_config()` is exported.
- The op file does not declare `INVALID`.
- No auto-fixes were needed: there was no XPASS evidence, and SUPPORTED == TARGET.
- **INVALID audit** (`feature_spec.py`): `INVALID = []` is correct. TARGET is one cell with no ROW_MAJOR, so the canonical bf8b+ROW_MAJOR entry does not apply. There are no weight-canonicalization cells.

## L1 Ledger Audit

- **Currency**:
  - 4 CBs and 4 rows (`cb_x_resident` is declared on workers + sender).
  - The size expressions match the descriptor.
  - Stale kb-4 figures were fixed (see Code Review).
- **Capacity vs live set**:
  - `cb_x_resident` (240 pages, whole x) and `cb_w_block` (96 pages, whole per-core W slice) are resident by scheme. They are bounded by `validate()`'s single-shape predicate. They are not collapsed knobs: x residency is what makes the handshake-free multicast legal, and W depth is the stall-shadow choice.
  - `cb_w_block` at full depth buys nothing measurable today: the W reader finishes at ~9 µs of a 26.6 µs span. Its depth-3/4 lamp is folded into perf Refinement 2.
  - Nothing is over- or under-spanned.
- **Page format vs DEST**: all pages are Float16_b / Bfp8_b with `fp32_dest_acc_en = False`. This is consistent.
- **Disjoint lifetimes**: `cb_interm_acc` and `cb_out_block` are justified separately: the TileRowMajor output needs its own region.
- **Bounds**: every op dimension in a capacity expression is fixed by `validate()`, and the total is closed-form.
- **Block-size defaults**:
  - Interleaved: the N-split over 64 cores with one output block per core is the coarsest possible.
  - kb was changed from 4 to 8 on measurement (31.8 → 26.4 µs).
- **Footprint**: worker `2048·m_t·k_t + 1088·k_t·block_n_tiles + 2·2048·m_t·block_n_tiles` = **636,928 B**. x is a fixed term. W, acc and out scale with `block_n_tiles`. `block_k_tiles` does not change the total. Sender: **491,520 B**.
- **Data movement**:
  - x and W cross DRAM once each (7.18 MB, the minimum). y stays in L1.
  - The cheapest-traffic split is the one implemented. K-split and 2-D are `deferred` with positive, model-backed reasons (18.15 / 22.3 µs vs 16.6 µs).

## Precision Baseline

The op accepts exactly one shape, so the baseline sweeps input profiles (`test_wq_b_linear_precision_baseline.py`). Default config: HiFi2, 16-bit DEST, kb 8. The reference is an fp64 matmul against the device-quantized W.

| Profile (x, W) | PCC | Max Abs Err | Mean Abs Err | Rel RMS | ULP p99 | Global scale | Ratio p5 / median / p95 |
|---|---|---|---|---|---|---|---|
| randn seed 0 (golden dist.) | 0.999931 | 0.0899 | 0.00827 | 0.0139 | 103 | 1.0069 | 0.957 / 1.0079 / 1.067 |
| randn seed 7 | 0.999931 | 0.0837 | 0.00824 | 0.0139 | 107 | 1.0069 | 0.957 / 1.0079 / 1.066 |
| x 8×N(0,1) (wide range) | 0.999931 | 0.672 | 0.0659 | 0.0139 | 105 | 1.0069 | 0.956 / 1.0078 / 1.067 |
| x U(0,1), W mean 0.5·0.02 (non-zero mean) | 0.995411 | 0.242 | 0.0364 | 0.0971 | 3 | 0.9990 | 0.990 / 0.9990 / 1.009 |

**Assessment**:
- **Golden distribution**: PCC 0.99993 clears the 0.9998 gate by a wide margin. This is better than the stock `ttnn.linear` (0.99985) and the hand-written reference (0.99990) in `feature_spec.py`.
- **Scale triage**: a uniform +0.69% scale explains about half the RMS. It was checked as a possible scale bug and **ruled out as an op bug**: it is 16-bit DEST rounding.
  - It grows with the in-DEST accumulation length (`block_k_tiles`): slope 0.9989 @1, 1.0002 @2, 1.0027 @4, 1.0069 @8, 1.0105 @12, 1.0134 @16, 1.0176 @24/48.
  - It does not depend on fidelity (HiFi4: 1.0084).
  - Stock `ttnn.matmul` with `packer_l1_acc=False` (whole K in DEST) shows the identical 1.0176 and PCC 0.999786. With `packer_l1_acc=True`, stock matmul equals our kb 2 result (0.999963).
- **Consequence for perf work**: `block_k_tiles ≥ 24` drops PCC to 0.999786, below the 0.9998 gate. The PCC-legal range is `block_k_tiles ≤ 16` (16 gives 0.999863). Smaller kb improves precision (kb 2: 0.999963, rel-RMS 0.0088).
- **Non-zero-mean profile**: format-limited. Rounding the exact result to bf16 alone gives PCC 0.99968 and rel-RMS 0.025, because the output mean is about 8× its std. It is gated at the shared default (0.99, 0.13). This profile is outside the golden distribution and is a characterization only.

**Recommended tolerances**: the defaults hold: `helpers.TOLERANCES[(bfloat8_b, False)] = 0.9998` PCC for the golden distribution. In the baseline test, rel-RMS is gated at 0.03 and global scale at |s−1| < 0.02.

## Performance (measured, Phase 0)

`tests/ttnn/unit_tests/operations/wq_b_linear/_op_bench.py` run under `run_safe_pytest.sh --profile`, 6 dispatches. The device-kernel span comes from `profile_log_device.csv`. The ops CSV's `DEVICE KERNEL DURATION` column is unusable on this run: one core's profiler clock was not synced, which put ~22 s into the column. That core was excluded.

- **Whole-op span**: median **26.6 µs** (25.8–27.1). That is **1.90× the 14.0 µs SOL** (p150) and 1.66× the p100a SOL figure.
- **Per-RISC timing, worker (3,9)**:
  - W reader (NCRISC) done at **9.2 µs**.
  - Compute (TRISC) done at 24.7 µs. Ideal compute is 480 tile-matmuls × 32 cycles = 11.4 µs.
  - Writer done at 27.0 µs.
  - Worker spans are balanced (max/p50 ≈ 1.03).
- **Sender core**:
  - The x DRAM read (NCRISC) takes **14.6 µs** for 491,520 B, about 34 GB/s. The model predicts 8.3 µs.
  - The multicast (BRISC) completes at **22.4 µs**. The model predicts 10.8 µs.
- **Classification**: **x-delivery-bound**. Compute is starved waiting for x rounds; W is done long before. The ~2.3 µs tail after compute is the 10-tile output store plus drain.
- **Calculator** (`op_model.py matmul 5 48 128 --designs --c-mem l1 --fidelity HiFi2 --a-dtype bf16 --b-dtype bfp8_b --c-dtype bf16`): rank 1 is 16.53 µs (73 cores, kb 2, linked x multicast). Rank 2 is 16.60 µs (65 cores, kb 2). The as-built config (`--kb 8 --workers 64`) is predicted at 18.75 µs. The model does not include the per-round Counter/write-barrier cost or the slow single-sender read that the measurements show.

## Verifier CLI Summary

Results: `/tmp/wqb_verify_golden2/`. A copy is in `verify/verifier_report.json`.

- supported_pass: 2 (cartesian cell + perf-flagged loose case)
- xfail_expected: 0
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 1 (`test_regression.py::test_repeated_calls_bit_identical`, passed; it is not registry-driven)

The unit tests pass 14/14: acceptance (10: 4 numeric, 1 repeat bit-identical, 1 default config, 4 validation rejections) and precision baseline (4).

## Recommendations

- **TARGET − SUPPORTED is empty**: every axis has its one TARGET value supported. There are no generality refinements, and the queue is all perf (see `op_requirements.md`).
- **Perf order**:
  1. x delivery: the single-sender DRAM read plus per-round multicast cost, ~12 µs above the model.
  2. Then the tail and W-side knobs: the output-store tail, W read coalescing, `w_depth_blocks`, and the column map for the actual bank count.
- **Hard constraint for any K-block re-tune**: `block_k_tiles ≤ 16` (precision ceiling above).
- **Ruled out by the calculator** (not filed):
  - K-split (ksplit 3: 18.15 µs, ≈ +10% vs rank 1, adds 2.6 MB NoC partials).
  - 2-D M×N with W multicast (22.3 µs, +35%).
  - Per-worker DRAM x reads (64× x DRAM traffic).
- **Profiler note for perf phases**: read device time from `profile_log_device.csv` and drop cores with unsynced timestamps. The ops-CSV duration column was corrupted by one core on this board.
