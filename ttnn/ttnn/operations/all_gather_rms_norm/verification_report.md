# Verification Report: all_gather_rms_norm

Box: Blackhole QuietBox 2 (4 chips, snake ring opened 1×4 / 4×1, FABRIC_2D_TORUS_XY, 14 KiB + 64 B payload).
Golden runs use the **gate** tier (`GOLDEN_TIER=gate`, 1720 cells), the tier `feature_spec.TIERS` defines for this
gate; the exhaustive tier is ~0.5 M cells. Test mode: production. `--dev` was attempted first. The first test passed,
then every device bring-up stalled on "Timed out while waiting for active ethernet core ... to become active again"
(fixture errors, the same environment issue the implementer logged). The watcher run is not usable on this box.

## Code Review

Fixed:
1. **Persistent L1 stats table leaked and collided with the CB region (correctness, 404 golden failures).**
   `stats_table()` created a mesh L1 tensor per distinct table size and cached it forever. L1 allocates top-down, so a
   probe showed each new shape placing another table lower (1570624, 1529664, …). After enough shapes, programs
   failed with *"Statically allocated circular buffers ... clash with L1 buffers ... allocated at 288000"* (49 cells).
   The golden conftest then deallocates every device tensor a failed call's frames hold, including the cached table
   (and staging buffer). Every later call failed with *"Tensor is not allocated"* (321 cells). The cells pass in
   isolation, which confirms state accumulating across calls.
   **Fix:** the table is now `cb_stats_table` (CB 25), a program-scoped CB on the full program core range.
   - Identical CB lists give it the same L1 address on every core of every chip, which is the property the
     identical-address rule already relies on for `cb_out_block`.
   - Peers write into it only after the ready fence: the sender waits for its peer's ready before its first stats
     packet, so the receiving program, and with it its CB region, is live.
   - Kernels take the address from `get_write_ptr(cb_stats_table)`; the common / per-port runtime arg is gone.
   - The cached DRAM staging buffer (R_U) is re-created whenever it is found deallocated.
2. **`select_split` ignored the stats-table cap** (27 cells). LN at R = 64 chose k = 3, a 288 KiB table over
   `STATS_TABLE_CAP` = 256 KiB (2048×1280, 2048×384, 4×1×512×512). New `stats_k_cap(R, G, nstats)` folds the cap
   into `k_cap`, single-sourced from `STATS_TABLE_CAP`. The RuntimeError now fires only when k = 1 does not fit.
3. **RMS partial sums were truncated to tf32 when summed.** This violated the prompt MUST: *"any per-row partial sum
   MUST stay fp32 on the wire and when summed"*. Both sums went through the FPU / AccumulateViaAdd reduce, which
   reads a Float32 CB through srcA (tf32):
   - the local collapse `cb_sumsq → cb_partial` (the value that goes on the wire);
   - the cross-chip combine `cb_gathered → cb_inv`.

   **Fix:** under fp32 DEST, both CBs are tagged UnpackToDestFp32 and reduced with `ReduceFp32Mode::Accurate`
   (SFPU, exact copies into DEST). This is the pattern LN's S1 already uses. `partial_sum_fp32_mode` and the host
   tag list both key on `fp32_dest_acc_en`, so Refinement 1's 16-bit-DEST path inherits a consistent switch. Perf
   was null: 35.542 → 35.557 µs on the focus case, noise 0.050.
4. **EXCLUSIONS re-measured**, as the from-base brief requires:
   - `{bfloat8_b, h_non_aligned}` is **lifted**: 19/19 gate cells pass (RMS + LN, 6 shapes). H padding is never
     element-masked here because rows are independent.
   - `{bfloat8_b, w_non_aligned}` is **kept** and now carries its measurement in the op file: 19/19 cells
     severity=bug (PCC ≈ 0) with it lifted. The R_U element-shift assembly and the reader's W-tail zero-fill
     address datums by byte offset, which shared-exponent blocks cannot express.
5. The stale reader header comment (it described a word-copy mean broadcast and a column scatter, both gone) is
   rewritten. `op_design.md` (Mechanism caps, Key Risks) and `l1_ledger.md` are updated for `cb_stats_table` and
   the k clamp.

Deferred to refinements (architectural): sharded I/O and 16-bit DEST (Refinement 1, from-base port); bulk transport
efficiency, overlap and the LN chain (Refinements 2–4, perf).

## Prompt rules (`eval/prompts/all_gather_rms_norm.txt` `## Rules` + MUST clauses)

- One fused native implementation: one `ttnn.generic_op(MeshProgramDescriptor)` per call, own kernels, Fabric
  transport, no other norm / CCL op. The example module is imported only for host-side planning helpers
  (`build_groups`, `_schedule`, `_worker_nearest_noc1`), never dispatched. ✓
- Every device returns the full (..., M, W) output: checked per device and per slice by `check_output`. ✓
- LayerNorm mean and variance over the full W, centered before scaling, beta at its global column: shifted-sum
  combine plus `(x − K) − d` centering; gamma and beta at `p·W_local + local column`; the big_offset regression
  passes. ✓
- Partial sums fp32 on the wire and when summed: was ✗ for RMS (item 3), now ✓. LN was already exact.
- Non-aligned shapes: padding excluded from Σ (RMS zero-fill; LN zero-fill for S1, `only_partial` mask for T),
  1/W uses the true W. R_U builds the dense concatenation. ✓
- Determinism (bit-identical repeats): `test_deterministic` and the 4-call acceptance test pass. ✓
- Advisory (soft): the prompt's "no host synchronization inside the op". `global_semaphores()` calls
  `synchronize_device` on the **first** call per port set, outside any trace (op_design.md Key Risks documents it).
  It is not on the per-call path.

## Registry Conformance

- `INPUT_TAGGERS` holds exactly `alignment` and `rank`, both with the `(inputs, axes)` signature, reading the
  per-device input shape. ✓
- `SUPPORTED` lists all 14 registry axes. gamma_dtype and gamma_layout are projected inside `validate()` with the
  `"none"` sentinel, which is listed. ✓
- `EXCLUSIONS` are cell-level dicts inside SUPPORTED. ✓
- `validate()` is the first line of the entry point: SUPPORTED per axis, then EXCLUSIONS
  (`UnsupportedAxisValue` / `ExcludedCell`), then caller errors (ValueError) and the live-fabric checks. ✓
- The op file does **not** declare `INVALID`. ✓
- No drift fixes were needed (xpass_drift = 0). SUPPORTED gained nothing; EXCLUSIONS lost `{bfloat8_b, h_non_aligned}`
  (item 4).
- **INVALID audit** (`feature_spec.py`), 8 entries, all well-formed:
  - the gamma "none" canonicalization, both directions;
  - rms_norm × beta;
  - no_gamma × beta (beta shares gamma's format axes).

  No entry couples axes of different tensors without a reason: beta × gamma is a documented shared-format
  coupling. No entry encodes "not yet supported". There is no bf8b × ROW_MAJOR activation entry, correctly: the
  input is TILE only, and gamma_dtype has no bfloat8_b, as the spec comment notes.

## Design Conformance

- **Algorithm (SF-GN):** matches. Stats-first, one fabric stats round, own-slice normalize, normalized slices sent
  to the peers' final pages (R_A) or to slot-contiguous staging plus assembly (R_U).
- **Topology / RISC ownership:** as designed. Hubs are the link-0 port cores; workers run reader (NoC0) / writer
  (NoC1) / compute; ports run reader + sender. The stats-ready signal is a raw `noc_semaphore_set_multicast` from the
  hub's reader instead of the `McastArgs` helper. The deviation is documented: the hub sits inside the multicast
  rectangle and the signal is a set, not a data channel.
- **Work distribution:** fills the fused grid (110 − 2L = 106 workers, 2 links).
  - Focus 640×384: k = 4 → 80 workers × 3 tiles, the planner's figure.
  - LN 640×32: 20 workers. R = 20 tile-rows × 1 tile is the whole slice; nothing finer exists.
  - Both dataflow halves are batched: the reader does one barrier per tile-row; the writer one barrier per block;
    ports use batches of 4 packets.
- **Blocking-model fidelity:**
  - Knobs are host single-sourced: `BATCH_CHUNKS`, `INC_EVERY`, `SCATTER_RUNS`, `BALANCE_MIN_TILES`, `P_CAP`,
    `COMB_TILE_COST_FUSED`, `STATS_TABLE_CAP`, plus `_knob()` env overrides for sweeps.
  - `crt` / `cw` derive from `select_split` and reach the kernels as CT args.
  - CB page counts are `crt_max·cw_max` (block), not op dimensions, with two bounded exceptions:
    `cb_stats_table` (`G·R·k·nstats·128`, bounded by `STATS_TABLE_CAP` via the k clamp + RuntimeError) and
    `cb_record_landing` (`G·k·nstats`, bounded by `P_CAP`).
  - `block_row_tiles = core_row_tiles` is the design's mechanism cap: x stays resident across the one stats round.
    R_WIN is the `deferred` row.
  - No collapsed or half-turned knobs found. Expression is fine: reader and writer barriers group by tile-row /
    block, compute runs block-shaped chains, and the per-tile loops in `sum_squares_block` / `transpose_gathered`
    are inside one block operation.
- `RECORD_BYTES` (128) and the CB indices are mirrored in Python and `agrn_common.hpp`. That is a necessary
  cross-language mirror; both sides cite each other.

## L1 Ledger Audit

- **Currency:** the ledger's "As built" table matches the code's CB list (CB 0–25), including the new
  `cb_stats_table` row. The main inventory table predates the as-built departures; the As-built table overrides it
  and says so.
- **Capacity vs live set:**
  - Every worker CB is one block (one compute thread: a compute→compute CB must hold the block).
  - `cb_ln_sq` = `crt·cw` is justified (masked AccumulateViaAdd, BulkWaitBulkPop, single compute thread).
  - `cb_port_chunks` (≈ 112 KiB) and `cb_stats_table` are unused on most cores but allocated by the
    identical-address rule (one CB list on the full range); both are bounded and stated.
  - Nothing to fix in place.
- **Page format vs DEST:** with fp32 DEST, every Float32 page is justified (wire contract, exact LN paths, the
  UnpackToDestFp32 tags). `cb_inv` is a Float32 page regardless of `fp32_dest_acc_en`, which will be over-wide under
  16-bit DEST. That is folded into Refinement 1 (`T_acc` per the base op's Refinement 1).
- **Disjoint lifetimes:** `cb_ln_m` aliases m and then d = mean − K (stated). The remaining pairs have reasons in
  the ledger.
- **Bounds:** `crt` ≤ R (footprint), `k` ≤ min(Wlt, P_CAP/G, 32/G for LN, `stats_k_cap`), `R·k` bounded by
  STATS_TABLE_CAP, payload pages by the fabric payload. Closed form below.
- **Per-core footprint:**
  `crt·cw·(T_in + T_out) + (γ+β)·cw·T_g + crt·(T_acc[RMS] + 4096·nstats·(1 + 2·gather_tiles) + 128·nstats + 4096)`
  `+ [LN](crt·4096·6 + crt·cw·4096·(1 + [x≠fp32])) + G·k·nstats·128 + G·R·k·nstats·128 + 2·batch_chunks·payload`.
  - Block knobs scale `crt·cw`; `k` and `G` scale the stats terms.
  - Focus case: ≈ 12 KiB block + ~40 KiB stats table + ~30 KiB stats CBs + 112 KiB port chunks.
- **Block-size defaults:** interleaved spreads `select_split` units over the fused grid and takes the whole per-group
  assignment as the block (mechanism-capped). No measured departure.
- **Data-movement budget:** present and consistent: x 1, own write 1, landed 3, relay read 1 → 6S per chip vs 5S
  minimum. The cheapest split, `fabric_mcast` (5S), is a `deferred` regime row with a positive reason: the route
  setter has no wrap handling on the ring's wrap hop. It is folded into Refinement 2 as lever (b).

## Precision Baseline

`tests/ttnn/unit_tests/operations/all_gather_rms_norm/test_all_gather_rms_norm_precision_baseline.py`.
Setup: 1×4 ring, Ring, 2 links, gamma (and beta for LN) ROW_MAJOR sticks, fp32_dest_acc_en=True, HiFi4. The
reference uses device-quantized inputs. Metrics run over all 4 devices. rel_rms = RMS error / reference RMS (the
golden helpers' definition; equal to the std form here). The ratio is got/true over elements with
|ref| > 1% of max. Numbers are after the item-3 fix.

| Shape | Norm | dtype | PCC | Max Abs Err | Mean Abs Err | Rel RMS | ratio median [p5, p95] |
|-------|------|-------|-----|-------------|--------------|---------|------------------------|
| (1,1,32,384) | rms | bf16 | 0.999997 | 0.0647 | 0.00219 | 0.00401 | 0.9968 [0.9928, 1.0007] |
| (1,1,640,384) | rms | bf16 | 0.999997 | 0.0818 | 0.00214 | 0.00396 | 0.9969 [0.9928, 1.0008] |
| (1,1,640,384) | rms | fp32 | 1.000000 | 0.0327 | 0.00102 | 0.00174 | 0.9985 [0.9972, 0.9995] |
| (1,1,640,384) | rms | bf8b | 0.999953 | 0.1032 | 0.00751 | 0.00970 | 1.0001 [0.9600, 1.0443] |
| (1,1,640,1792) | rms | bf16 | 0.999997 | 0.0883 | 0.00208 | 0.00390 | 0.9970 [0.9929, 1.0008] |
| (1,1,17,50) | rms | bf16 | 0.999997 | 0.0410 | 0.00216 | 0.00401 | 0.9969 [0.9927, 1.0008] |
| (1,1,640,32) | ln | bf16 | 0.999998 | 0.0258 | 0.00192 | 0.00194 | 0.9999 [0.9959, 1.0038] |
| (1,1,640,32) | ln | fp32 | 1.000000 | 0.0225 | 0.00137 | 0.00144 | 0.9993 [0.9955, 1.0018] |
| (1,1,640,384) | ln | bf16 | 0.999998 | 0.0396 | 0.00184 | 0.00193 | 0.9998 [0.9961, 1.0034] |
| (1,1,640,1792) | ln | bf16 | 0.999998 | 0.0452 | 0.00182 | 0.00193 | 0.9998 [0.9961, 1.0034] |

**Assessment:** every case passes its `helpers.TOLERANCES` row with 2.5–5× margin on rel RMS. The scale-bug triage
is negative: rel RMS is ~0.002–0.01, far from the high-RMS signature, and the ratio spread is a band, not a tight
cluster.

The RMS path does carry a **small systematic low bias**: median ratio 0.9969 (bf16) and 0.9985 (fp32), every p95
close to 1. That size is what round-toward-zero truncation predicts in the normalize chain:
- x and inv truncated to tf32 into srcA/srcB of the FPU `x·inv` multiply (~0.05% each on average);
- the DEST → srcA reuse in the raw `x·γ` row-broadcast multiply (~0.05%);
- for bf16, the pack to bf16 (~0.15–0.2%).

LN centers and scales in SFPU fp32 and sits at 0.9998. Before item 3, the tf32 truncation of Σx² partly offset this
bias (fp32 median 0.9988, rel RMS 0.0014). The fp32-sum fix removed that offset, which is the right trade for a
MUST rule. No cell fails, so this is an advisory (Recommendations), not a refinement.

**Recommended tolerances:** defaults hold (`helpers.TOLERANCES`).

## Verifier CLI Summary

Final gate run (after all fixes), `/tmp/agrn_verify/gate4/verifier_report.json` (summary committed as
`agent_logs/verifier_report_summary.json`):
- supported_pass: **1080**
- xfail_expected: **480**
  - 188 fp32_dest_acc_en=False and 169 HEIGHT/WIDTH_SHARDED: Refinement 1.
  - 88 `{float32, False}`: a permanent EXCLUSION (precision convention).
  - 35 `{bfloat8_b, w_non_aligned}`: a measured EXCLUSION (structural for the R_U element-shift path). This is the
    documented omission: no refinement is filed for it, because block-float assembly at non-tile offsets would
    need a re-quantizing assembly path no model call site uses.
- invalid_skipped: 0. INVALID cells are not collected in a tier.
- no_axes_found: 160. This is 32 `test_regression.py` cells (all pass) and 128 `test_ring_mock.py` cells, skipped
  by design outside the exhaustive tier.
- supported_fail: **0**
- xpass_drift: **0**
- xfail_wrong_mode: **0**

History: the first gate run had **404 supported_fail** and **3 xfail_wrong_mode** (items 1–2). The 3 wrong-mode
cells were the harness's own sharded `from_torch` failing (tilize work split) while the leaked tables ate L1. They
xfail normally after the fix.

Acceptance: `test_all_gather_rms_norm.py` 15/15. Precision baseline: 10/10.

**TARGET − SUPPORTED coverage:**
- fp32_dest_acc_en=False → Refinement 1.
- memory_layout HEIGHT_SHARDED, WIDTH_SHARDED → Refinement 1.
- No other gaps.

## Performance (eval harness, 5 experiments, 2500 calls each)

| Case | fastest chip | slowest chip | spread | roofline / target |
|---|---|---|---|---|
| focus 640×384 RMS (bf16, γ RM sticks, Ring, 2 links) | 35.56 µs | 35.94 µs | 0.38 µs | 7.6 / 10.1 µs |
| LN 640×32 + γ + β | 29.60 µs | 29.88 µs | 0.27 µs | 0.6 / 0.8 µs |

The run is not dispatch-bound (max gap 0.61 µs).

Per-chip breakdown on the focus case, from the implementer's zone timeline in this run; my fixes are perf-null:

| Phase | Time | Notes |
|---|---|---|
| Local stats | ~2.3 µs | Hidden behind the ~5 µs fence |
| Stats round | to ~6 µs | |
| Combine + normalize + own write | to ~11 µs | |
| Bulk own entry (TRANSPORT) | ~15 µs | ~16 GB/s per port; final interleaved pages give 1–2-page bank runs |
| Relay | ~4 µs | |
| Landing tail | ~3 µs | |

TRANSPORT therefore dominates (~22 µs). For scale: transport-only `fabric_all_gather` at 256 tiles per chip runs in
≈ 18 µs on 2 links.

On the LN case:
- Every component is fixed cost: fence, stats round, ~7 µs of serial LN compute per tile-row, and ~11 µs bulk for
  a 20-page slot.
- The 0.6 µs roofline excludes the ~8.4 µs exchange fixed cost by construction.

The queue ranks accordingly: Refinement 2 TRANSPORT, Refinement 3 OVERLAP, Refinement 4 LN local chain.

## Recommendations

- **Queue order:** Refinement 1 (from-base port) first, then the cross-device perf refinements by the measured
  breakdown (2 TRANSPORT, 3 OVERLAP), then the LN local chain (4). There are no other generality candidates, so
  every slot after Refinement 1 is perf.
- **RMS normalize bias (advisory, no failing cell):** the ~0.15% (fp32) / ~0.3% (bf16) low bias comes from FPU
  operand truncation in `x·inv·γ`. An SFPU normalize, as LN does, would remove the fp32 part at some compute cost.
  Worth a precision refinement only if a model-level tolerance needs it.
- **Persistent state:** the only remaining persistent device resources are the global semaphores (per port set)
  and the R_U staging DRAM buffer (per geometry). The staging cache grows by one DRAM buffer per distinct
  R_U geometry. This is harmless for the golden sweep, but a long-running process with many unaligned shapes
  would accumulate DRAM. A grow-only single buffer per (mesh, dtype) would bound it, but a captured trace would
  then pin the buffer's address, so it needs care.
- **Env-var probe hooks** in the production path (`AGRN_FORCE_STAGED`, `AGRN_OWN_FROM_DST`, `AGRN_PROFILE_ZONES`,
  `AGRN_<KNOB>`) are perf-sweep aids. They are hashed into the program cache key; keep them, but do not let a perf
  round leave one set by default.
- **Watcher coverage:** `--dev` cannot bring the fabric up twice on this box (Ethernet core timeout). Hang triage for
  the refinements must rely on production-mode dispatch timeouts.
