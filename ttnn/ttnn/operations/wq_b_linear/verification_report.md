# Verification Report: wq_b_linear

Board: Blackhole p100a, 11×10 = 110 worker cores, AICLK 1350 MHz. Op state: commit `ec22f5fc67e` plus the fixes below.

## Code Review

Fixed:
- **Fake buffer-depth knobs removed** (`wq_b_linear_program_descriptor.py`). `x_cb_k_blocks` and `w_cb_k_blocks` were host aliases for `num_k_blocks` that no kernel read. The writer and reader reserve all `num_k_blocks` blocks up front, so setting either alias below `num_k_blocks` would shrink the CB under a kernel that still reserves the whole slice and hang. CB capacities are now written in `num_k_blocks` directly. Whole-K residency is a regime property bounded by `validate()`'s fixed shape, not a tunable depth.
- **Design drift recorded** (`op_design.md`, new "As built (Phase 0)" subsection). The implementer changed four planned values, each backed by a measurement: `K_BLOCK_TILES` 8→6, `w_reads_in_flight` 6→1, `num_x_injectors` 6→8, and the x multicast from Counter without handshake to Flag with handshake (linked multicast). The design tables still showed the planned values, and the next refinement plans against this file.
- **Ledger currency** (`l1_ledger.md`). The capacity expressions no longer reference the removed aliases. The R5 deferral note now records that measurement contradicts its premise (see L1 Ledger Audit).

Checked, no change needed:
- Helpers: compute is one `compute_kernel_lib::matmul_block` call with packer L1 accumulation in place. The x broadcast uses the `ttnn.Mcast` + `McastArgs` sender/receiver helper. W reads use the trid helpers. Interleaved page I/O goes through `TensorAccessor` (no kernel-lib helper wraps interleaved-DRAM tile reads).
- CB sync: `cb_x_resident` 8 pushes × 30 = 240 = compute pops. `cb_w_stream` 8 × `6·core_n` matches. `cb_y_tiles` push `5·core_n` = writer wait/pop.
- Kernel syntax: `void kernel_main()`, `api/...` include paths, `TensorAccessor` only.
- Broadcast efficiency: there are no broadcast operands. The x replica is the matmul's in0 and is multicast once.
- Blocking-model fidelity:
  - `K_BLOCK_TILES` is the single source for `k_block_tiles`, `num_k_blocks`, the x round size, the W batch size, the CB push quanta and the trid count.
  - `core_n_tiles` comes from `split_work_to_cores(grid, …, row_wise=True)`.
  - The subblock comes from the DEST-limit rule.
  - No kernel holds a literal tile count, and no value is duplicated as a literal.
  - Whole-K residency of x and W is the sanctioned predicate-guarded case: one shape, footprint proven in the ledger.
- Expression: per-core compute is one `matmul_block` over the whole `(5, core_n, 48)` block. The reader synchronizes per K-block batch, the writer per multicast round and once for the y store. No per-tile handshake.
- Performance conformance: 110/110 cores active, n split row-wise, both dataflow halves batched.

Deferred to refinements (performance only): see Recommendations and `op_requirements.md`.

## Prompt Rules (`eval/prompts/wq_b_linear.txt`)

| Rule | Status |
|------|--------|
| MUST derive grid/coords from the device, no hardcoded harvesting | met: `compute_with_storage_grid_size()`; `ttnn.Mcast` resolves physical coords |
| MUST be bit-identical across calls when several cores contribute to one tile | met trivially: n-only split, no cross-core reduction. 20-call golden regression passes. Refinement 3 (K-remainder split) must preserve it with a fixed-order combine |
| MUST allocate output L1 interleaved, activation dtype, TILE | met (asserted by golden `_check_output_placement`) |
| One native dispatch per call | met: one `ttnn.generic_op` |

## Registry Conformance

- `INPUT_TAGGERS = {}`, `SUPPORTED` (`dtype`, `weight_dtype`, `layout`, `fp32_dest_acc_en`), `EXCLUSIONS = []` and `validate()` are present inline. `validate()` checks SUPPORTED per axis (plus W layout), then EXCLUSIONS, then the shape/placement contract (`ValueError`) and the core-count predicate (`RuntimeError`). The public entry calls it first.
- The op file declares no `INVALID`.
- SUPPORTED == TARGET (one value per axis, one cell). No XPASS drift, no auto-fixes.
- INVALID audit (`feature_spec.py`): `INVALID = []`, correct for a one-cell universe. No ROW_MAJOR in TARGET, so the canonical bf8b+ROW_MAJOR entry is not applicable. No cross-tensor entries. Not a norm op.

## L1 Ledger Audit

- **Currency**: three CBs declared, three rows. Size expressions match the descriptor after the alias fix. Measured `device_l1_peak_bytes` = 628,736 B = ledger 616,448 B + 12,288 B of the L1-interleaved y share.
- **Capacity vs live set**: `cb_w_stream` capacity is 2× its live set on the 92 one-column cores (52,224 B). Justified in the ledger (one descriptor for both groups; L1 is not binding). No collapsed extents: x and W span k with capacity scaling in `num_k_blocks`, and `cb_y_tiles` spans m and n.
- **Page format vs DEST**: all pages are Float16_b / Bfp8_b with 16-bit DEST. No over- or under-width page.
- **Disjoint lifetimes**: none. All three CBs are live for the whole kernel.
- **Bounds**: every capacity symbol is bounded in the symbol table by `validate()`'s shape and core-count predicates.
- **Block-size defaults**: held. Interleaved, so the split spreads n over the full grid and the per-core block is the whole assignment (coarsest).
- **Footprint**: `491,520 + 62,464 · core_n_tiles_max` B. Only the `cb_w_stream` and `cb_y_tiles` terms scale, with the n-share knob; `k_block_tiles` changes push quanta only.
- **Data-movement budget**: present and consistent with R1 (W ×1, x ×1 DRAM, y 0). The cheapest-traffic split (R5) was deferred because "the cross-core tier does not bind". **The ablation below falsifies that**: the x broadcast (cross-core) is the critical path. The ledger now says so, and Refinement 1 names the x-topology scheme-change as its escalation. Folded into Refinement 1; no standalone entry.

## Precision Baseline

The op accepts exactly one shape (`validate()` rejects all others), so the baseline varies input distribution and seed instead of shape. Reference: fp64 `x @ W_device_quantized`. `test_wq_b_linear_precision_baseline.py`, 8/8 pass.

| Case (x, W distribution) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p99 (bf16) | got/true median [p5, p95] |
|---|---|---|---|---|---|---|
| standard N(0,1), 0.02·N(0,1), seed 0 | 0.999968 | 0.0452 | 0.00512 | 0.00831 | 85.5 | 1.0010 [0.957, 1.046] |
| standard, seed 1 | 0.999968 | 0.0415 | 0.00512 | 0.00831 | 84.5 | 1.0010 [0.956, 1.046] |
| x ×8 | 0.999968 | 0.3617 | 0.04094 | 0.00831 | 85.5 | 1.0010 [0.957, 1.046] |
| x + 0.5 | 0.999968 | 0.0464 | 0.00571 | 0.00830 | 85.8 | 1.0010 [0.956, 1.046] |
| uniform [-1,1], 0.02·U[-1,1] | 0.999969 | 0.0142 | 0.00171 | 0.00835 | 84.9 | 1.0016 [0.957, 1.046] |

The golden cells measure the same: PCC 0.999967, rel-RMS 0.0085.

**Assessment**: Relative error is scale-invariant (rel-RMS 0.0083 whether x is scaled ×8 or offset). The got/true ratio is a broad ±4.5% spread centred near 1.0, which is rounding noise, not a scale bug. The +0.1% median bias matches bf16 rounding of 8 packer L1-accumulation passes in 16-bit DEST (`fp32_dest_acc_en=False` is the contract). The ULP p99 of ~85 bf16 ULP is dominated by outputs near zero relative to the 48-tile K-sum magnitude. PCC margin over the 0.9998 gate: 1.7e-4.

**Recommended tolerances**: `(bfloat8_b weight, fp32_dest_acc_en=False)`: PCC ≥ 0.9998 (the `helpers.TOLERANCES` value holds). Relative RMS ≤ 0.02 is a safe additional gate (measured 0.0083); the golden suite does not gate RMS.

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/wq_b_linear/` → 3/3 passed; `verifier_report.json` copied into the op directory.

- supported_pass: 2 (`test_op`, `test_op_loose`)
- xfail_expected: 0
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 1 (`test_regression.py::test_repeated_calls_bit_identical`, untagged by design, passes)

Acceptance suite: 18/18 (plus 8/8 precision baseline), `--dev`.

TARGET − SUPPORTED = ∅ on every axis. No generality refinement exists; the queue is perf-only.

## Performance (flagged LOOSE_CASE, the whole input space)

SOL (feature_spec) 16,018 ns, DRAM-bound. Golden device kernel time 27.0 µs, 1.69× SOL.

Ablation (`tests/ttnn/unit_tests/operations/wq_b_linear/_op_bench.py`, `--profile`, 3 calls each, `WQB_VARIANT=`):

| Variant | Device ns | What is left running |
|---|---|---|
| full | 26,580–26,690 | everything |
| no_compute | 25,460–25,690 | x broadcast, W reads, y store (`SKIP_COMPUTE`) |
| no_w_read | 26,300–26,400 | x broadcast, compute, y store |
| no_compute_no_w_read | 25,330–25,440 | x broadcast + y store |
| w_only | 20,700–21,040 | W reads; x multicast shrunk to 1 page/round, no x prefetch |
| compute_only | 15,680–15,750 | compute; 1-page x rounds |
| sync_only | 14,700–14,760 | 8 one-page handshake multicast rounds + y store |

Reading:
- **The x broadcast binds**: 25.4 µs alone vs 26.6 µs full. W reads and compute are almost entirely hidden under it.
- **Sync-only is 14.7 µs**: 8 serialized handshake rounds cost about 1.6–1.8 µs each with a one-page payload, before any bytes. The x bytes add about 10.7 µs (≈ 46 GB/s ≈ 34 B/cycle linked multicast). Bytes and handshake latency serialize because rounds are chained.
- **W stream alone is 20.8 µs**, 1.30× the 16.0 µs DRAM SOL. Per-core NCRISC end times (w_only): max 20.86 µs, p50 16.68 µs (max/p50 = 1.25). The 20 slowest cores are the 18 two-column cores of core_group_1 (logical rows 0–1). They read 104 KB vs 52 KB, and their tail runs with low DRAM concurrency.
- **Compute** is ≤ 15.7 µs including the 14.7 µs sync floor it overlaps, so not binding.

## Recommendations

- Queue order (`op_requirements.md`):
  1. x-broadcast critical path (knob-turns: round granularity decoupled from the compute K-block, protocol / injector choice).
  2. W-stream knob co-tune (per-group depth, even 64×2 split, placement, bank stagger).
  3. Balanced W split via the design's R4 K-remainder combine (scheme-change), only if imbalance survives Refinement 2.

  Each one exposes the next binding term, which is why they run in this order.
- `mcast` helper friction (implementer breadcrumb, confirmed by the ablation): `data_ready=Counter` silently selects unlinked multicast writes. Worth documenting in `kernel_lib/mcast/README.md`. Rotating senders with `Flag` force a handshake (`mcast_impl.cpp:459`), and that handshake is what costs about 1.7 µs per round here.
- The precision margin is small (1.7e-4 PCC). A refinement that reorders the K-sum, or adds a cross-core combine (Refinement 3), must re-run the precision baseline. A combine that accumulates partials in bf16 through an extra L1 pass will move the median bias.
- Not binding today: the `cb_w_stream` over-allocation on one-column cores (52 KB), and the x prefetch issued on NoC1 (8 injectors × 61 KB, under the W stream). Revisit only if a refinement makes L1 or the prefetch bind.
