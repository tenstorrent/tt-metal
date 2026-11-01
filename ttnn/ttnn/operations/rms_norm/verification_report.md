# Verification Report: rms_norm

Board: Blackhole p150b, 13×10 = 130 Tensix cores (`compute_with_storage_grid_size()`), L1 1 572 864 bytes per Tensix core, allocator base 111 744 bytes.

## Code Review

Fixed in place:

1. **L1 budget overshoot: a real out-of-memory (OOM) failure, now fixed.** `select_blocking()` sized circular buffers (CBs) against `ttnn.get_max_worker_l1_unreserved_size()` (1 531 776 bytes on BH). That figure starts at the kernel-config ring buffer, which sits *below* the allocator base where the CBs actually start. The usable CB region is 1 461 120 bytes, so the op over-counted by 70 656 bytes. Any shape whose footprint fell in that gap threw `Statically allocated circular buffers ... grow to 1594496 B which is beyond max L1 size`. Reproduced on (1,1,544,11008) bf16 TILE: the Llama FFN width with ≥ 17 tile-rows, so k = 1 and the footprint is 1 482 752 bytes. The new `cb_l1_budget(device)` reads `ttnn.get_memory_view(device, L1)` and returns the free region between the allocator base and the lowest allocated L1 buffer. This also accounts for L1-resident input and output tensors, which the old budget ignored; L1-interleaved inputs were therefore also exposed to the overflow. Regression test: `test_rms_norm_debug_l1_budget_boundary`, TILE/ROW_MAJOR × DRAM/L1-interleaved.
2. **DRY: CB capacities were stated twice.** `footprint()`, which drives the L1-fit solve, and the CB declarations in `create_program_descriptor()` restated the same nine page-count expressions independently, so a knob change could make the solve and the allocation drift apart. Both now read one `cb_plan(sz, k, slice_w, b, d)` dict, `{cb_index: (pages, page_bytes)}`. The writer's `out_cb_pages` comes from the same dict. The TILE writer unit `min(slice_w, 16)`, previously written separately in `cb_plan`'s ancestor and in `Blocking.out_unit_tiles`, now comes from one `_out_unit()`.
3. **DRY: writer runtime-argument layout.** The writer kernel hard-coded `NOC_XY_ARG_BASE = 9`, which had to match the length of the host's base runtime-argument list. It is now a compile-time argument computed from the host constant `WRITER_NOC_XY_ARG_BASE`, and a host assert checks it against the list.

Checked and found correct (no change):
- Kernel syntax (`void kernel_main()`), `api/dataflow/dataflow_api.h` includes, and `TensorAccessor` everywhere (no `InterleavedAddrGen`).
- CB push/wait parity on every CB:
  - `cb_x`: pushes `x_row_pitch` per tile-row; compute waits and pops `rows·x_row_pitch`.
  - `cb_gathered_sumsq`: `k·rows`.
  - `cb_sumsq_tiles`, `cb_partial_sumsq`, `cb_inv_rms`: `rows`.
  - `cb_out`: per unit (TILE) or per chunk (RM).
  - `cb_gamma`: one push and one pop per program.
- Helper usage:
  - Every compute phase is a kernel-lib helper (`eltwise_chain` with DEST-accumulated square; `reduce` with `AccumulateViaAdd`, Collapse or Skip; `tilize` / `untilize`).
  - Raw API is used only where documented: the finalize post-op inside `reduce()`'s `post_reduce_op` hook, and the partial all-gather.
  - The gamma multiply is fused as a `DestReuseBinary` post-op of the `x·inv_rms` chain.
  - No redundant broadcast fills: `inv_rms` is consumed with `BroadcastDim::Col`, and gamma is expanded once per program.
- Multicast helper: the all-gather is a *symmetric unicast* round (k senders, no multicast). Per the design and catalog `tensix_all_reduce` (k ≤ 8), it is not a multicast-and-handshake pattern the `McastArgs` helper expresses, so it stays raw.

Deferred to refinements: none from code review. The perf lamps below are measured and filed in the queue as perf refinements.

## Prompt Rules

`eval/prompts/rms_norm.txt` has no `## Rules` section, so the stock policy applies and is met:
- One native `ttnn.generic_op` dispatch per call.
- No host-side `to_layout` / `tilize` / `pad` / `slice`.
- Non-aligned H and W are handled in-kernel; the 1/W uses the logical W.
- `default_compute_kernel_config()` is the single default.
- The config is passed as-is.

**Advisory.** The prompt says `tag_gamma_dtype` / `tag_gamma_layout` "MUST" be `INPUT_TAGGERS`. The implementer deliberately did not register them, and `validate()` projects them instead. This is the right call: the harness treats every `INPUT_TAGGERS` key as shape-derived and stops iterating it from TARGET (`eval/feature_matrix.py` `cartesian()`), which would collapse the golden `gamma_dtype` / `gamma_layout` axes. Taggers also receive only shapes. The "none"-sentinel behaviour the rule exists for is preserved. Recommend rewording the prompt (implementer breadcrumb already filed).

## Registry Conformance

- `INPUT_TAGGERS` = {`alignment`, `rank`}, both with the `(inputs, axes)` signature. `SUPPORTED` lists every axis the kernel gates on: dtype, fp32_dest_acc_en, layout, alignment, rank, gamma_mode, gamma_dtype, gamma_layout, memory_layout. `EXCLUSIONS` = `[{dtype: float32, fp32_dest_acc_en: False}]`.
- `validate()` order: structural `ValueError`s first, then SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`). `rms_norm()` calls `validate()` first.
- The op file does **not** declare `INVALID` ✓.
- No auto-fixes to SUPPORTED: xpass_drift = 0.

**INVALID audit** (`eval/golden_tests/rms_norm/feature_spec.py`; not edited — please update via `/golden-tests`):
- ✓ Canonical `{dtype: bf8b, layout: ROW_MAJOR}` is present; the gamma twin `{gamma_dtype: bf8b, gamma_layout: ROW_MAJOR}` is single-tensor and fine.
- ✓ No-gamma canonicalization through the "none" sentinel: 7 entries, both directions.
- ✗ **Cross-tensor, not structural:** `{layout: ROW_MAJOR, memory_layout: *_SHARDED, gamma_layout: TILE}` (3 entries). These couple the *activation* layout and placement with the *weight* layout, and their own comment says "out of scope for now". There is no kernel-level coupling: the op already reads TILE gamma with an RM activation on interleaved memory. Recommend deleting them; if the op ever cannot do a combination, that is an op-side EXCLUSION.
- ✗ **"For now", not structural:** `{dtype: bf8b, alignment: w_non_aligned | h_non_aligned}` (2 entries). Block-float storage of a non-aligned tensor is legal; the comment says these are parked. They belong in op EXCLUSIONS, not INVALID. Side effect today: translated tests do not consult INVALID, so the CLI reports 2 `invalid_unexpected` cells (`test_rms_norm_sharded_uneven_multicore_logical_width[*-bfloat8_b]`). These are harmless: validate refuses them with NotImplementedError. Refinement 1's notes ask the implementer to mirror them as op EXCLUSIONS.

## Design Conformance

- **Algorithm:** matches. One DRAM read of x, kept resident across both passes. Σx² is accumulated in DEST (no x² buffer). The padding is zeroed by the reader and 1/W uses the logical W. The fused `x·inv_rms·gamma` chain is as designed.
- **Data pipeline, RISC-V ownership, inter-core communication:** match the design. The combine is a symmetric unicast all-gather with a monotonic semaphore and a 2-block gathered CB, all on one identical CB list.
- **Parallelization** (performance conformance): the machine is filled as the design specifies.
  - R1 spreads tile-rows over the whole 130-core grid. For example, 7168×1024 uses all 130 Tensix cores.
  - R2 splits W when tile-rows ≤ G/8 or W does not fit. For example, 32×16384 uses 8 Tensix cores; 64×12288 uses 16.
  - Both dataflow halves are batched: the reader does one barrier per tile-row with `core_w_tiles` reads in flight; the writer waits, writes and flushes ≤ 16 tiles per unit.
  - Measured under-fill at mid tile-row counts (G/8 < tile-rows < G, e.g. 20 or 32 Tensix cores) follows the design's occupancy predicate (catalog `width_split` band). It is a design perf lamp, not a contract violation, and is filed as Refinement 3.
- **Blocking-model fidelity:** `block_row_tiles`, `x_block_depth`, `w_split_cores`, `rm_chunk_w_tiles` and `out_unit_tiles` are each computed once on the host and passed as runtime or compile-time arguments.
  - Every CB capacity and trip count derives from them (DRY now enforced through `cb_plan`).
  - No CB scales with a whole-op dimension unconditionally: `cb_x` spans the core's W *share* by design (residency), bounded by the `k_fit` predicate.
  - The per-core loop is block-granular: one block = the whole assignment when it fits, so there is no half-turned split.
  - Expression: compute is block-expressed (one chain or `reduce` call per block). The reader reserves the whole block and pushes per tile-row, and the writer streams through a `cb_out` window; both are design-sanctioned streaming boundaries, not per-unit serialization of a block operation.
- **Implementation deviation (sound, recorded by the implementer):** `x_block_depth = 2` is realized as two one-block CBs (`cb_x` / `cb_x_alt`, alternating by block parity) instead of one 2-block ring, because indexed CB reads cannot wrap the ring. The footprint is unchanged.

## L1 Ledger Audit

- **Currency:** all 10 declared CBs have rows, including the implementer-added `cb_x_alt`, and every capacity expression matches `cb_plan()`. Updated in place: the `budget` symbol (new budget source) and a note that `cb_plan()` is the single source.
- **Capacity vs live set:** no over-sized buffers.
  - `cb_x` exceeds its live set only by depth 2 or the last W member's gap pages; both are justified.
  - `cb_gathered_sumsq` is 2 blocks because of the lockstep mechanism cap.
  - `cb_out` (TILE) is a 2 × unit streaming window. `cb_out` (RM) spans row × one chunk, and that is its live set (untilize input).
  - No collapsed extent: every axis marked as spanned has capacity that scales with it.
- **Page format vs DEST width:** Phase 0 is fp32 DEST only, so the Float32 intermediates (`cb_sumsq_tiles`, `cb_partial_sumsq`, `cb_gathered_sumsq`, `cb_inv_rms`) are correct. Under `fp32_dest_acc_en=False` they would be an "over" finding. This is folded into Refinement 1 as a note: derive their format and page size from the flag.
- **Disjoint lifetimes:** `cb_sumsq_tiles` / `cb_inv_rms` (disjoint in R2, overlapping in R1, saving < 1 %) and `cb_x_rm` / `cb_out_rm` (concurrent) are justified.
- **Bounds:** every symbol has a bound and its predicate. The total is closed-form in `k`, `b` and `d`.
- **Block-size defaults:** held. Interleaved spreads row-groups over the full grid, then takes the coarsest block (the whole assignment at depth 1; depth 2 only when that does not fit). No departure. The resulting "one block per core, no overlap" on full-grid prefill is the design's overlap lamp; Refinement 3 measures it.
- **Data-movement budget:** present and consistent with the code. x and out cross DRAM once; gamma is read once per active Tensix core (its slice); R2 adds k²·rows·4 KB of NoC traffic. The cheapest-traffic alternative, gamma read-once + multicast (−(G/k−1)·W bytes), is a `deferred` row with a positive reason: r ≤ 1/32 is in the catalog's measured-null band for `shared_input_reuse`.
- **Per-core footprint**, in bytes:

  ```
  d·b·S_x·T_in + [γ]·S_x·T_g + 2·b·T32 + [k>1]·(b + 2kb)·T32 + (TILE: 2·min(S,16)·T_out | RM: 2C·T_in + bC·T_out + 2C·T_out)
  ```

  The dominant terms `d·b·S_x·T_in` and `S_x·T_g` scale with `block_row_tiles`, `x_block_depth` and 1/k.
- **Findings filed:** none beyond the fixed budget bug and the Refinement 1 page-format note.

## Precision Baseline

Measured on (1,1,32,32), (1,1,256,1024), (2,1,512,2048) and (1,1,32,16384) with gamma (bf16/fp32 randn, ROW_MAJOR gamma), HiFi4, fp32 DEST, ε = 1e-6, TILE and ROW_MAJOR. Source: `test_rms_norm_precision_baseline.py`. ULP is measured in the output dtype's ULP at the reference value; ratio = got/true over |ref| > 1e-3.

| Shape | dtype / layout | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | ULP p50 / p99 / max | ratio p5 / p50 / p95 |
|---|---|---|---|---|---|---|---|
| (1,1,32,32) | bf16 TILE | 0.9999975 | 0.0258 | 0.00164 | 0.00318 | 0.51 / 1.66 / 2.19 | 0.9930 / 0.9972 / 1.0009 |
| (1,1,256,1024) | bf16 TILE | 0.9999970 | 0.0614 | 0.00194 | 0.00373 | 0.50 / 1.64 / 2.35 | 0.9931 / 0.9973 / 1.0011 |
| (2,1,512,2048) | bf16 TILE | 0.9999971 | 0.0940 | 0.00196 | 0.00379 | 0.52 / 1.65 / 2.44 | 0.9930 / 0.9971 / 1.0010 |
| (1,1,32,16384) | bf16 TILE (R2) | 0.9999969 | 0.0623 | 0.00195 | 0.00373 | 0.50 / 1.62 / 2.37 | 0.9931 / 0.9973 / 1.0013 |
| (1,1,32,32) | fp32 TILE | 0.99999977 | 0.0183 | 0.00082 | 0.00167 | 1.7e4 / 4.7e4 / 5.2e4 | 0.9972 / 0.9985 / 0.9997 |
| (1,1,256,1024) | fp32 TILE | 0.99999974 | 0.0204 | 0.00079 | 0.00144 | 1.4e4 / 4.1e4 / 6.5e4 | 0.9975 / 0.9988 / 0.9999 |
| (2,1,512,2048) | fp32 TILE | 0.99999974 | 0.0298 | 0.00088 | 0.00159 | 1.6e4 / 4.3e4 / 6.8e4 | 0.9973 / 0.9986 / 0.9997 |
| (1,1,32,16384) | fp32 TILE (R2) | 0.99999974 | 0.0286 | 0.00074 | 0.00136 | 1.3e4 / 4.0e4 / 6.5e4 | 0.9976 / 0.9989 / 1.0000 |

ROW_MAJOR results are identical for bf16. For fp32 they are slightly worse (rel RMS 0.0017–0.0020) because `tilize` `Fp32Mode::Fast` truncates fp32 to tf32.

**Assessment.**
- bf16: within ~0.5 ULP median and ≤ 2.5 ULP max, at the bf16 output-rounding floor.
- fp32: about 1.5e4 fp32 ULP median (≈ 2 tf32 ULP). This is the FPU's tf32 operand resolution: the square, `x·inv_rms` and `·gamma` all run on the FPU.
- **Scale-vs-precision triage:** the ratio median is slightly below 1 (−0.28 % bf16, −0.15 % fp32), with a narrow spread. Rel RMS is *low* (≤ 0.4 %) and the bias is independent of W (32 → 16384) and of the regime (R1 / R2). So this is *not* the high-PCC / high-RMS scale-bug signature. It is consistent with truncating (round-toward-zero) operand and pack formats.
- **Lever, not filed** because no cell fails: fp32 accuracy could move toward fp32 resolution by squaring on the SFPU (`CopyTile` + `Square`) with `UnpackToDestFp32` on `cb_x` (design risk table).

**Recommended tolerances:** bf16 PCC ≥ 0.9999, rel RMS ≤ 6e-3, atol 0.1 (gamma-scaled), rtol 2e-2. fp32 PCC ≥ 0.99999, rel RMS ≤ 3e-3, atol 0.04.

## Verifier CLI Summary

Golden suite `eval/golden_tests/rms_norm/` on the final code. Results dir: `/tmp/rms_golden_v2` (full `verifier_report.json` is 25 MB; a compact copy is committed as `verifier_report_summary.json`).

- supported_pass: 737
- xfail_expected: 6180
- invalid_skipped: 33900
- infeasible_skipped: 2 (99991×64 WIDTH_SHARDED: shard geometry vs L1, harness-side, uncharged)
- invalid_unexpected: 2 (translated bf8b × w_non_aligned; see INVALID audit)
- no_axes_found: 15 (`test_regression.py` numerics tests; all passed)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

Unit tests: `tests/ttnn/unit_tests/operations/rms_norm/` 145/145 in `--dev` (acceptance 109, debug 20 incl. the new L1-budget cases, precision baseline 16).

**Queue coverage of `xfail_expected`.** Each blocker value maps to a refinement:

| Only blocker | Cells | Refinement |
|---|---|---|
| fp32_dest_acc_en=False | 822 | 1 |
| gamma_dtype=bf8b | 140 | 1 |
| dtype=bf8b | 100 | 1 |
| WIDTH_SHARDED | 586 | 2 |
| BLOCK_SHARDED | 586 | 2 |
| HEIGHT_SHARDED | 561 | 4 |

The remaining 3385 cells carry two blockers (e.g. fp32_dest_acc_en=False with a sharded `memory_layout`), and both values are covered. TARGET − SUPPORTED = {dtype: bf8b, fp32_dest_acc_en: False, gamma_dtype: bf8b, memory_layout: HEIGHT/WIDTH/BLOCK_SHARDED}; every value is in the queue and none is omitted.

## Perf Scoreboard (Phase 0 config: bf16, fp32 DEST, HiFi4, TILE, TILE gamma, INTERLEAVED)

Measured with `tests/ttnn/unit_tests/operations/rms_norm/perf_scoreboard_rms_norm.py` through the eval runner's `device_kernel_ns`. The DRAM ceiling is (in + out bytes) / ~420 GB/s (BH p150b achieved full-grid read+write, catalog index). `achievable_ns` is the feature_spec reference at the *perf* config (fp32 DEST off, HiFi2), so it is not directly comparable until Refinement 1 lands.

| Shape | Tensix cores | device ns | DRAM ceiling ns | achievable_ns | Class |
|---|---|---|---|---|---|
| (1,1,32,1024) | 8 | 9 168 | ~0.3 k | 9 149 | decode (small) |
| (1,1,32,7168) | 8 | 21 828 | ~2.2 k | 14 894 | decode, k capped at 8 → Refinement 5 |
| (1,1,32,32768) | 8 | 73 525 | ~10 k | — | few-row wide → Refinement 5 |
| (1,1,640,6144) | 20 | 116 401 | ~37 k | 101 968 | mid tile-rows, grid under-filled → Refinement 3a |
| (1,1,640,7168) | 20 | 135 241 | ~44 k | 119 307 | same |
| (1,1,1024,1024) | 32 | 25 549 | ~10 k | — | same |
| (1,1,7168,1024) | 130 | 96 600 | ~70 k | 89 392 | prefill, one block per core, no overlap → Refinement 3b |
| (1,1,8192,2304) | 130 | 246 173 | ~180 k | 211 345 | same |
| (1,1,8192,7168) | 130 | 789 403 | ~559 k | 1 032 281 | same |

## Recommendations

- **Queue order:**
  1. Numerics: bf8b and fp32-DEST-off. This lands the perf profiles' config.
  2. WIDTH / BLOCK sharding with a scalable combine: the hardest structural change.
  3. Perf on large interleaved cells.
  4. HEIGHT sharding: a local extension of 2's plumbing.
  5. Perf on few-row wide-W cells: needs 2's k > 8 combine.
- **L1 budget for sharded refinements:** shards are allocated in L1 before the program is built, and `cb_l1_budget()` already subtracts them. Keep using it — never the HAL "unreserved" figure.
- **Precision without a failing cell:** fp32 outputs sit at tf32 resolution (see Precision Baseline). If a future fp32 cell fails RMS, the lever is SFPU square / multiply with `UnpackToDestFp32` (Refinement 1's skill covers the CB tagging).
- **Combine payload:** the R2 combine ships whole 4 KB fp32 partial tiles and uses a write barrier before each semaphore increment. Catalog `tensix_all_reduce` says to ship faces 0/2 (or a 128 B row at K ≥ 32) and to flush rather than barrier. This belongs to Refinement 5, measured on its shapes.
- **INVALID cleanup in `feature_spec.py`:** remove the 3 cross-tensor RM × sharded × TILE-gamma entries, and move the 2 bf8b × non-aligned entries to op EXCLUSIONS. See the INVALID audit.
