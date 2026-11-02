# Verification Report: qkv_proj_heads_rope

Board: Blackhole p100a (110 Tensix, 108 used). Built regime: R1 (2D token_rows x head_cols,
output-stationary, unicast reads). One `generic_op` dispatch per call.

## Code Review

Fixed:

- **DRY: table geometry restated in the reader.** The reader hard-coded `table_tiles_per_row = 4`
  and `st * 2` (tiles per head) while the host (`TABLE_TILES_PER_ROW`, `TILES_PER_HEAD`) and compute
  (`2 * tiles_per_head` from a CT arg) derive them. `TILES_PER_HEAD` is now a reader CT arg
  (index 8, accessors shifted to 9); the row size and page index derive from it, with a
  `static_assert(tiles_per_head == 2)` matching compute's.

Checked, no change needed:

- Kernels use `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor` (no
  `InterleavedAddrGen`).
- CB sync balanced on every CB: activation / weight push `block_m*k_block` / `k_block*N_c` per
  K-block and `matmul_block` pops per K-block; table 4 per valid row on q/k cores, both sides; out
  `N_c` per valid row, both sides; partials reserve/push/wait/pop the nominal block.
- Helpers: `matmul_block` (Interm, TileRowMajor, L1 acc, caller-owned target) for the projection,
  `eltwise_chain` for rotary (BinaryFpu Mul, PerRow DEST acc) and convert. The single raw call
  (`pack_reconfig_l1_acc(0)`) has no helper equivalent. No multicast exists yet, so there is no raw
  semaphore code to migrate to `McastArgs`.
- Broadcast: bias folded as the 25th K tile (ones-column tile x zero-padded bias row), so there is no
  broadcast pass and no redundant fill.

Not fixed (perf, measured gate needed, filed as levers in Refinement 4):

- The rotary chain is called once per (row, head), so its init and reconfig run `h_c * rows`
  times per block. The design asked for `InitReconfigOwner::Caller`, hoisted once per block. The op
  is data-movement bound today (ablation in `l1_ledger.md`: the compute floor of the matmul phase
  is 21 us), so this only shows once the multicasts land.

### Prompt rules

| Rule | Applies now | Status |
|------|-------------|--------|
| One program, no composed ttnn ops | yes | OK: one `generic_op` |
| No 1/sqrt(D) scale | yes | OK |
| cos/sin row s for position s, any S_tab >= S | yes | OK: `st = r % St` on x's tile rows, pages index the table's own grid; golden `test_long_tables` (inf beyond S) passes |
| No angles on device | yes | OK |
| Padded rows exactly 0 when S % 32 != 0 | no (`h_non_aligned` not in SUPPORTED) | Re-arms with Refinement 2. Probe: v padding currently non-zero, so the writer-side zeroing is required. |
| Default HiFi3 + fp32 acc; honor caller config | yes | OK |
| Must beat stock on every perf row (Performance target) | yes | **Violated on every row**: 200-283 us vs stock 75-246 us. R1 is DM-bound by design (x unicast re-read). Refinements 1, 3, 4 and 5 address it. |

## Design Conformance

- Algorithm, pipeline topology and RISC ownership match `op_design.md` (R1 row).
- Work distribution fills the machine: 108/110 cores for every perf row (selection function,
  runtime grid). Both dataflow halves batch per K-block (one barrier per 40-55 tile transfer); the
  writer flushes per output row (`N_c` tiles), as designed.
- Blocking knobs (`block_m`, `heads_per_core`, `k_block`, the four depths, `CB_BUDGET_BYTES`) are
  host constants passed as CT args, and every CB size is derived in `_cb_pages`. No CB scales with a
  whole-op dimension. After the DRY fix no block value is restated as a literal.
- Expression: the reader and writer handshake per K-block and per row, and compute holds the whole
  block live. No per-unit completion boundary sits inside a block-scoped phase.
- Deviation, recorded rather than fixed: the per-call chain init above (performance only).

## Registry Conformance

- `INPUT_TAGGERS` (`tag_alignment(inputs, axes)`), `SUPPORTED` (all 8 TARGET axes), `EXCLUSIONS = []`
  and `validate()` (SUPPORTED, then EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell`)
  are present and wired. `validate()` is the first line of the entry point. The op file does not
  declare `INVALID`.
- No SUPPORTED auto-fix was applied (no XPASS evidence: validate rejects every unsupported cell
  first). Probe evidence (below) shows the three DRAM placements are already numerically correct;
  they are deliberately left to Refinements 1 and 2 (see Recommendations).
- INVALID audit (`feature_spec.py`): one entry, `{dtype: bfloat8_b, layout: ROW_MAJOR}`. It
  couples a single tensor and is structural (bf8b needs tiles). It is inert under the current
  single-valued TARGET. There is no weight-optional axis, so no canonicalization cells are needed.
  OK.

## Golden harness defects (not fixed here: eval suite is out of the verifier's write scope)

The committed golden suite cannot pass against any implementation:

1. `helpers.py::default_test_compute_config` passes `packer_l1_acc=True` to
   `ttnn.ComputeConfigDescriptor`, which has no such field (`program_descriptors.cpp` nanobind
   `__init__`). Every `test_op` cell raises `TypeError` before the op runs. As committed, the CLI
   reports 17 `supported_fail` and 175 `xfail_wrong_mode`; only the 8 loose cells (default config)
   pass.
2. `test_regression.py` sets `_TOL = TOLERANCES[...]`, a `(pcc, rms)` tuple, and passes it to
   `check_output(tolerance=...)`, which reads `.pcc`: `AttributeError` on every regression test.
   `test_lane_pairing_rotate_half` also indexes `_TOL[1]`, so the file uses `_TOL` both as a
   tuple and as a `Tolerance`. No single fix to `_TOL` satisfies both uses.

Fix needed in `/golden-tests`: drop the `packer_l1_acc` kwarg (other ops' translated tests already
drop it), and make `_TOL` a `Tolerance` and use `_TOL.rms` in `test_lane_pairing_rotate_half`.
**Every result below came from a run with this local, uncommitted patch** (kwarg dropped,
`_TOL = tolerance_for(...)`); the patch was reverted afterwards.

## L1 Ledger Audit

- **Currency**: the 5 ledger rows match the 5 CBs in `_cb_pages` / `CB_FORMATS`; size expressions
  match the allocation.
- **Capacity vs live set**: activation / weight / table / out are 2x their live sets (double
  buffering, stated). Partials capacity == live set, which spans `block_m x N_c` and scales with
  both. Nothing is over- or under-sized.
- **Page format vs DEST**: partials use Float32 pages with `fp32_dest_acc_en = True` (the only
  supported value), carrying the L1 accumulation across K-blocks. Every other CB is bf16 because it
  is never a DEST spill. OK.
- **Disjoint lifetimes**: none unjustified. The table/out vs activation aliasing is ruled out by the
  table prefetch during the K-loop tail (a pipelining decision, stated).
- **Bounds**: every non-block symbol (`h_c`, `k_block`, `block_m_cap`, depths, `cb_budget_bytes`) is
  in the symbol table with its predicate. `Mt` enters only through `min(..., block_m_cap)`. The total
  is closed-form.
- **Block-size defaults**: held. Each core's whole assignment is one block (11x8 at 8x512, 780 KiB);
  the multi-block path (`num_blocks > 1`) is reachable only at tiny S with huge B. Probed at
  B=4096, S=1 (multi-block): correct.
- **Footprint**: `2048*(2*block_m*k_block + 2*k_block*N_c + 8 + 2*N_c) + 4096*block_m*N_c`.
  `block_m` scales activation + partials, `N_c` scales weight + out + partials, `k_block` scales
  activation + weight.
- **Data-movement budget**: present and consistent with R1 (W 12x DRAM, x 9x NoC unicast at
  8x512). The cheapest split (R1+R2+R3, two multicasts) is deferred with "stepping stone" as the
  reason. That is not a positive reason under the audit rules, and the measurement contradicts the
  deferral: the x re-read is the binding resource. It is folded into the queue as Refinement 1
  (x multicast) and Refinement 3 (W multicast). The verifier added measured placement sensitivity
  to the ledger.

## Precision Baseline

`test_qkv_proj_heads_rope_precision_baseline.py`, default config (HiFi3, fp32 acc), x / tables /
outputs in L1. Worst of q, k, v per shape:

| Shape (B,S,S_tab) | PCC (min) | Max Abs Err | Mean Abs Err | Rel RMS (max) | ULP p99 (max) | got/true median q,k / v | p5..p95 |
|-------------------|-----------|-------------|--------------|---------------|---------------|-------------------------|---------|
| (1,32,32) | 0.9999984 | 0.0117 | 1.28e-3 | 1.91e-3 | 4.9 | 0.99927 / 1.00005 | 0.996..1.003 |
| (4,256,512) | 0.9999985 | 0.0210 | 1.29e-3 | 1.90e-3 | 4.5 | 0.99926 / 0.99998 | 0.996..1.003 |
| (8,512,512) | 0.9999985 | 0.0212 | 1.29e-3 | 1.91e-3 | 4.4 | 0.99925 / 0.99999 | 0.996..1.003 |
| (32,128,128) | 0.9999985 | 0.0194 | 1.29e-3 | 1.90e-3 | 4.2 | 0.99925 / 0.99998 | 0.996..1.003 |

**Assessment**: no scale/structural bug. The spread is +-0.4% around a median of about 1, and rel
RMS is about 2x below the 4e-3 gate. q/k carry a small systematic negative bias (median ratio
0.99925, -7.5e-4) that v does not: the rotary FPU multiply truncates (fp32 partials enter srcA as
tf32, plus the HiFi3 fidelity phases), so it rounds toward zero. The bias is well inside tolerance
and matches the stock chain's q/k vs v RMS split (3.3e-3 vs 1.7e-3). The lever if ever needed is
HiFi4 (caller config) or an SFPU combine (lamp L7). No refinement is filed for it.

**Recommended tolerances**: PCC >= 0.9999, rel RMS <= 4e-3 (golden). Measured headroom ~2x on RMS.

## Verifier CLI Summary

Golden run with the local harness patch above, `verifier_results/verifier_report.json`:

- supported_pass: 25 (17 `test_op` + 8 loose perf cases)
- xfail_expected: 175 (all `h_non_aligned` and/or a DRAM placement)
- invalid_skipped: 0 (INVALID inert under TARGET)
- no_axes_found: 10 (the `test_regression.py` numerics tests: 4 pass, 4 refused as `h_non_aligned`,
  `test_lane_pairing_rotate_half` fails on the harness `_TOL[1]` bug)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

Loose (perf-flagged) cells, golden-run device time vs the stock four-op sequence:

| (B,S) | 8x512 | 16x256 | 32x128 | 8x384 | 4x512 | 8x256 | 2x512 | 4x256 |
|-------|-------|--------|--------|-------|-------|-------|-------|-------|
| op us | 281.2 | 273.8 | 282.7 | 252.4 | 234.6 | 200.6 | 239.3 | 227.0 |
| stock us | 226.3 | 228.5 | 246.4 | 178.8 | 123.5 | 124.1 | 75.0 | 79.8 |
| target us | 90 | 90 | 90 | 68 | 46 | 46 | 28 | 28 |

(The profiler run measured 2x512 at 189.5 us. The small rows vary by about 20% run to run.)

## Recommendations

- **Perf is the dominant debt.** The op is 1.1-3.2x slower than the four ops it replaces. The ledger
  ablation (x unicast re-read binds the K-loop: 185 -> 45 us with x reads stubbed at 2x512) puts the
  x multicast first (Refinement 1). Refinement 3 (W multicast) follows on cadence.
- **DRAM placements**: probes with SUPPORTED widened at runtime (`probes/probe_002.py`) show all
  three DRAM axes numerically correct with zero kernel change (PCC 0.9999985, rel RMS 1.9e-3).
  Output and table DRAM are traffic-neutral (275 / 281 us vs 279 us). x in DRAM costs +13% at 8x512
  (317 us) and +17% at 2x512 under R1, so it ships with the x multicast, not alone.
- **h_non_aligned**: the probe confirms the valid rows are already correct (2x37, 13x300, 16x8,
  4096x1). Only the writer-side pad zeroing is missing (v padding non-zero; q/k padding is zero only
  because the probe's tables had zero padding, and inf in x's padding would make it NaN).
- **Multi-block regime (R4 row)**: reachable only once `h_non_aligned` lands (for example B=4096,
  S=1). It is correct today but re-crosses W per block. That is outside the perf rows, so no
  refinement is filed.
- **Harness**: fix the two golden-suite defects above before the refinement loop relies on raw
  `eval_test_runner.sh` results.
