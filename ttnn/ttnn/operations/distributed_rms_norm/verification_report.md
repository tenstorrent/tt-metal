# Verification Report: distributed_rms_norm

Phase 0 is regime R1 resident-flat from `op_design.md`, as one fused `ttnn.generic_op` (MeshProgramDescriptor) per call. Each chip runs:
- a worker grid of row-groups × W-chunks (reader / compute / writer);
- `2 · num_links` port cores (Fabric sender) that exchange the fp32 per-row partials;
- one leader per row-group that combines the partials and fans inv_rms out to its row-group.

Verified on a Blackhole QuietBox 2 (4 chips): the system mesh is 2×2, and the ring mock opens it as 1×4 / 4×1, under FABRIC_2D_TORUS_XY with the 14 KiB router payload.

## Blocker for grading: harness L1 capture fails on every fabric-mesh test

With the runner's default `EVAL_CAPTURE_L1=1`, **765 of 773 failures** in the first golden run are `eval.l1_profiling.L1ProfilingError: L1 profiling returned peak_total=0`. They are instrumentation failures, not op failures:
- graph capture around one call of this op records `buffer_allocate` / `circular_buffer_deallocate_all` nodes but **no `circular_buffer_allocate` node**, so `extract_resource_usage_per_core(...).peak_total == 0`;
- **native `ttnn.add` on the same fabric mesh also captures peak 0** (`tests/.../distributed_rms_norm/probes/probe_l1_capture.py`), so it is not specific to `generic_op` or to this op;
- only a test whose call allocates an L1 *buffer* gets a positive peak: the first call that creates the persistent board scratch (25 632 B), which is why 32 cells "passed" in that run;
- `tt_metal/impl/program/program.cpp:2225-2330` tracks CB allocations per sub-device of a MeshDevice. Why the processor drops them on this mesh was not root-caused, because that is harness / tt-metal territory and out of scope for an op verifier.

All results below come from `EVAL_CAPTURE_L1=0 eval/eval_test_runner.sh ...`, the runner's supported opt-out. **Action for the harness owner:** fix the CB tracking for MeshDevice graph capture, or exempt fabric-mesh ops. Until then the graded run will report this op, and any other mesh op, as failing.

## Code Review

Fixed in place:
- **Dead host constant**: `HALF_ROW_BYTES` in `distributed_rms_norm_program_descriptor.py` was unused; removed. The kernels own their 64 B half-row constant.
- **`L1_BUDGET` provenance**: the ledger claimed it was a "device query", but the code uses a host constant. I checked `ttnn.get_max_worker_l1_unreserved_size()` (1 531 904 B here). It is the static HAL ceiling, not this device's allocator base, so swapping it in would over-promise and turn a clean refusal into a CB clash. I kept the conservative constant, documented why in code, and corrected the ledger's symbol table.
- **`SUPPORTED` drift, three values promoted** (see Registry Conformance): `num_links=None`, `dtype=bfloat8_b`, `alignment=h_non_aligned`.

Reviewed and found conformant (no change):
- **Kernel API**: all four kernels use `void kernel_main()` and `api/dataflow/dataflow_api.h` includes. DRAM I/O goes through `TensorAccessor` (no `InterleavedAddrGen`). CB push and wait counts match per CB (`h_act·w_act` for x; `h_act` for the stats CBs; `n_gather_tiles·h_act` for the gather; `w_act` per out batch).
- **Helpers**:
  - `sumsq` is one `eltwise_chain` over `grid(h_act, w_act)` using `BinaryFpu<Mul, x, x, DestAccumulation::PerRow>`, the DEST-accumulated square that is the `row_reduce_accumulate` fast path.
  - Both reductions are `reduce<SUM, REDUCE_COL, …, Accurate>`. The combine fuses `×1/W, +ε, rsqrt` into the reduce's `post_reduce_op` (`compute_fusion`: SFPU consumer).
  - `scale_by_gamma` and `normalize` are single block-level chains with Row / Col broadcast.
  - The only raw LLK is the in-place transpose (kernel_lib has no transpose element; justified in the kernel header).
  - `WaitUpfrontNoPop` plus a caller pop replaces the design's `BulkWaitBulkPop`, because the latter has a CB-size precondition (implementer friction note). That is correct.
- **Broadcasts**: gamma is read as row-0 slivers only (2 × 64 B per tile), which is exactly what the Row broadcast consumes; no full-tile fill. inv_rms is a row vector transposed in place to Col0 for the Col broadcast. The SUM scaler is a single 1.0 tile.
- **Dataflow inter-core**: the inv_rms fan-out is a unicast of 64 B half-rows to each core of the row-group, followed by one semaphore increment each. `mcast_pipe` was considered and rejected by the design for Phase 0 (it targets a CB-handshaked rectangle payload). The mcast fan-out is listed as a perf lever in refinement 3.

Deferred (architectural; filed):
- **Collapsed `exchange_round` / blocks-per-core knob.** `ROWS_PER_ROUND` exists on the host but raises `NotImplementedError`, and no kernel has the design's `for block_idx < num_blocks_this_core` loop. The solve therefore requires `num_row_groups ≤ num_workers`, which is the root cause of the 8 OOM supported_fails. Filed as **refinement 1** (it needs real kernel rework: per-round exchange accounting in four kernels).

## Design Conformance

| Dimension | Verdict |
|-----------|---------|
| Algorithm | ✓ One fused program: x is resident and read from DRAM once; sumsq is DEST-accumulated; fp32 compact partials go on the wire; there is a per-row-group leader combine, and the stall-shadow `x⊙gamma` runs before the stats wait. |
| Pipeline / RISC ownership | ✓ Reader on NCRISC/NoC0, writer and port sender on BRISC/NoC1, as designed. |
| Work distribution | ✓ The grid is filled by the solve: 100 of 106 workers at 640×1792 (20 row-groups × 5 chunks), 104 at 2048×1280. The W split fills the grid when rows cannot; decode (one tile row) uses 14 workers, limited by `MIN_BLOCK_W = 4` (a design lamp, perf refinement 3). Both dataflow halves are batched: x reads take one barrier per block row, and out writes are batched per block row behind `OUT_DEPTH = 2`. |
| Inter-core / inter-chip | ✓ Ring/Linear schedule, ready fence, fused write + arrival increment on each slot's last packet, re-armed counters; back-to-back calls are bit-identical (acceptance test). **Documented deviation:** port cores use "simple" placement (logical row 0) instead of the Ethernet-probe placement, because the probe is a second device dispatch and violates the one-dispatch rule. The design's own port-placement lamp lists "simple" as the alternative. |
| Blocking-model fidelity | `block_h`, `block_w`, `num_w_chunks`, `num_row_groups`, `n_gather_tiles`, `OUT_DEPTH`, `HEADER_RING` and `num_links` are host values threaded once into CT/RT args, and kernels only derive from them (`wc = col0 / block_w`). There are no duplicate block literals. Ragged blocks pass smaller runtime extents (`h_act`, `w_act`) through the same block operations. **Exception:** the collapsed `rows_per_round` knob above (refinement 1). |
| Expression | ✓ Every phase acts on the whole block: one push of x, one chain per phase, one publish barrier, one gather barrier, out batched per block row. No per-tile handshakes at the scheduling boundary. |

## Prompt rules

- **MUST be ONE fused native implementation** (no ttnn.rms_norm*, all_gather, CCL ops, or example ops): ✓. `run_distributed_rms_norm` builds one MeshProgramDescriptor and calls `ttnn.generic_op` once. The only other device work is a one-time cached creation of the global semaphores and the board tensor, done before any trace.
- **MUST NOT let W tile padding enter the sum / MUST divide by the true W**: ✓ for the supported cells. `w_non_aligned` stays outside SUPPORTED until refinement 4 masks it; `1/W_global` is computed on the host from `G · W_local`.
- **fp32 on the wire and when summing**: ✓. The partials, board, gather and inv vectors are all Float32, and both reduces use `ReduceFp32Mode::Accurate` with `UnpackToDestFp32` inputs.
- **Must not change the fabric config; discover links with `get_forwarding_link_indices`**: ✓.
- **Determinism**: ✓ (bit-identical repeat test plus the golden `test_deterministic`).

## Registry Conformance

- `INPUT_TAGGERS` (`alignment`, `rank`, both `(inputs, axes)`), `SUPPORTED`, `EXCLUSIONS` and `validate()` are present and correctly wired. `validate()` raises caller errors (ValueError) first, then checks SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`). The public entry point calls `validate()` as its first statement. The op file does **not** declare `INVALID`.
- `default_compute_kernel_config()` is the single source of the None default, and the golden `axes.classify_call` reads it.
- **Fix-in-place drift (verifier):**
  - `num_links`: added `None`. The op accepts `None` (= every discovered link) and canonicalizes it before the gate, but the harness's observe-only `classify_call` records the raw kwarg. Every translated test calls with the default, so all of them were classified outside SUPPORTED: 7 `xpass_drift` + 199 `xfail_wrong_mode` in run 0. Listing `None` makes the classification match what the op actually does; no behavior change. (Harness suggestion: canonicalize `num_links` in `classify_call` like the op does.)
  - `alignment`: added `h_non_aligned`. Probe (`probes/probe_unlock.py`, validate bypassed): 10/10 pass on `(1,1,1,1792)`, `(1,1,17,1280)`, `(1,1,100,512)` fp32, `(1,17,128)`, `(17,64)` fp32, each ± gamma. The structural reason it works: sum of squares, exchange, combine and normalize are all per row, so the padded rows of the last tile row never touch a valid row.
  - `dtype`: added `bfloat8_b`. Probe: 8/8 pass, including `(1,1,640,1792)`, `(2,4,128,512)` and bf8b × h_non_aligned, within the harness's `(bf8b, fp32 DEST)` tolerance. The x/out CB formats derive from the tensor dtype, and every intermediate is fp32. Its `fp32_dest_acc_en=False` corner stays with refinement 5.
  - Confirmed by the final golden run (below).
- **INVALID audit** (`eval/golden_tests/distributed_rms_norm/feature_spec.py`): six entries, all the gamma "none" canonicalization coupled to `gamma_mode`, both ways. All are single-tensor (gamma's own axes) and pure canonicalization, so none violates the three sanity rules. The canonical bf8b + ROW_MAJOR activation entry is correctly absent: the input layout is TILE-only in TARGET and gamma_dtype has no bf8b (stated in the spec). No cross-tensor entries. **No change requested.**
- **CLI false positive (not drift):** `test_translated.py::test_pre_all_gather_non_welford_fp32_requires_fp32_dest_acc` is a *negative* test that passes because the op correctly refuses `{float32, fp32_dest_acc_en=False}` (EXCLUSION). The CLI cannot tell "expect-raise passed" from XPASS, so it lands in `xpass_drift`. The op is correct and nothing should change.

## L1 Ledger Audit

- **Ledger currency**: every declared CB has a row, and the size expressions match the descriptor:
  - `cb_x_block`: `block_h·block_w·T_x`
  - `cb_gamma`: `block_w·T_g` (only with gamma)
  - `cb_scaler`: `T_16`
  - alias group A {3, 4, 5}: `block_h·T_32`
  - `cb_partial_row`: `block_h·T_32`
  - `cb_gather`: `n_gather_tiles·block_h·T_32`
  - alias group B {8, 9}: `block_h·T_32`
  - `cb_xg`: `block_h·block_w·T_32` (only with gamma)
  - `cb_out_tiles`: `min(OUT_DEPTH, block_h)·block_w·T_x`

  `worker_bytes()` in the solve is the same expression. One text fix: the `L1_BUDGET` symbol row (above).
- **Capacity vs live set**:
  - `cb_out_tiles` exceeds its live set by one block row. That is the stated double-buffer, so it is justified.
  - `cb_gather` exists on non-leaders for address uniformity, at a cost of 4 KB at perf focus. Justified.
  - `cb_gamma` pages are full tiles holding two 64 B slivers each. The Row-broadcast unpacker addresses whole tile pages (mechanism reason), and the cost is `block_w·T_g` (24 KB at perf focus). Accepted.
  - Under direction: none. Each spanned axis scales with its extent (`cb_x_block` and `cb_xg` with `block_h·block_w`, the stats CBs with `block_h`, `cb_gather` with `n_gather_tiles·block_h`).
- **Page format vs DEST width**: `fp32_dest_acc_en=True` is the only SUPPORTED mode, so the Float32 stats / `cb_xg` pages are correct (not over-wide). Once `False` is added, `cb_xg` must drop to 16-bit (over-wide otherwise). The stats CBs keep Float32 for the prompt's fp32-wire rule. Folded into **refinement 5**'s verifier notes.
- **Disjoint lifetime with no justification**: none. Every non-shared pair has a written reason: `cb_partial_row` is concurrent with group A, `cb_gather` has no cross-kernel ordering edge, and `cb_xg` vs `cb_x_block` differ in format and are concurrent during the scale.
- **Bounds / closed form**: every term is closed-form in `block_h`, `block_w`, `n_gather_tiles`, `OUT_DEPTH`, `T_*`. **Finding:** the solve's `num_row_groups ≤ num_workers` constraint makes the block size grow with the tensor, so `fits_l1` fails past ~16k tiles per device with gamma at bf16 (e.g. `(2,1,2048,4096)` → 8 golden cells refused). The op refuses cleanly (no unbounded allocation), but it is a real capacity gap. Folded into **refinement 1** (the block loop makes the footprint one block per round). Noted in the ledger.
- **Data-movement budget**: present, and consistent with the implemented split:
  - x and out cross DRAM once each;
  - gamma slivers are counted per row-group;
  - partials, Fabric and fan-out bytes are itemized.

  The cheapest-traffic split (rows-only R5) is recorded with its delta and a positive reason: DRAM, the dominant tier, is identical, and R5 is the `num_w_chunks = 1` point of the same structure. The local pre-combine (R3) and gamma broadcast (R4) are `deferred` regime rows, each with a positive reason. R4 is taken up as a measured lever in refinement 6.
- **Block-size defaults**: interleaved. The solve spreads row-groups × chunks over the full worker grid first, then takes the coarsest block (key: tiles/core, then fewer chunks, then fewer rows). No departure.
- **Per-core footprint** (perf focus, bf16 x + gamma): 138 KB = x 24 + gamma 24 + scaler 2 + 4 × 4 stats + xg 48 + out 24. Terms scaling with `block_h·block_w`: x, xg. With `block_w`: gamma, out. With `block_h`: stats. With `n_gather_tiles·block_h`: gather.

## Precision Baseline

`tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_precision_baseline.py`. Setup:
- 2×2 system mesh, `cluster_axis=1` (G = 2), Linear;
- bf16 TILE gamma, ε = 1e-6, default config (HiFi4, fp32 DEST);
- reference: fp32 torch `rms_norm` over the concatenated row;
- ULP is measured in the output dtype;
- ratio = got/true over |ref| > 1e-3.

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP max / mean / p99 | got/true p5 / median / p95 |
|-------|-------|-----|-------------|--------------|------------------|----------------------|----------------------------|
| (1,1,32,32) | bf16 | 0.999998 | 0.0252 | 9.4e-4 | 0.00205 | 1 / 0.187 / 1 | 0.9962 / 0.99900 / 1.0017 |
| (1,1,32,1792) | bf16 | 0.999998 | 0.0308 | 1.0e-3 | 0.00197 | 1 / 0.167 / 1 | 0.9963 / 0.99908 / 1.0018 |
| (1,1,640,1792) | bf16 | 0.999999 | 0.0475 | 1.0e-3 | 0.00199 | 1 / 0.177 / 1 | 0.9962 / 0.99904 / 1.0017 |
| (1,1,2048,1280) | bf16 | 0.999999 | 0.0491 | 1.0e-3 | 0.00200 | 1 / 0.181 / 1 | 0.9962 / 0.99902 / 1.0017 |
| (1,1,32,32) | fp32 | 1.000000 | 0.0144 | 7.4e-4 | 0.00146 | 46650 / 15316 / 37088 | 0.9976 / 0.99876 / 0.9998 |
| (1,1,32,1792) | fp32 | 1.000000 | 0.0263 | 7.5e-4 | 0.00137 | 57403 / 14255 / 34547 | 0.9976 / 0.99883 / 0.9999 |
| (1,1,640,1792) | fp32 | 1.000000 | 0.0317 | 7.9e-4 | 0.00140 | 61330 / 14706 / 35003 | 0.9976 / 0.99879 / 0.9998 |
| (1,1,2048,1280) | fp32 | 1.000000 | 0.0324 | 7.8e-4 | 0.00139 | 61853 / 14593 / 34830 | 0.9976 / 0.99880 / 0.9998 |

**Assessment**:
- **bf16**: at most 1 ULP everywhere (≈ 18% of elements off by one). This is output-rounding-limited and independent of shape and of the W-split / row-group count.
- **fp32**: PCC 1.0 and rel-RMS 0.0014, about 3.5× inside the harness target of 0.005. The ULP counts look large only because 0.1% is ~8k fp32 ULP.

**Scale-vs-precision triage.** The ratio median sits at 0.9988 (fp32) / 0.9990 (bf16) with a tight spread, which is the signature the triage table warns about. A discriminating probe (`probes/probe_ratio.py`, fp32 input, 1×(64×1024)) shows the bias scales with the number of FPU multiply operands that are fp32 values truncated to TF32 on unpack, and with nothing else:

| Input | Median ratio | Truncated operands |
|-------|--------------|--------------------|
| bf16-valued x, no gamma | 0.99963 | 1 (inv) |
| bf16-valued x + gamma | 0.99900 | 2 (xg, inv) |
| fp32 random x, no gamma | 0.99946 | 2, partly cancelled by a low ss |
| fp32 random x + gamma | 0.99878 | 3 |

A structural or scale bug would not depend on whether x is TF32-representable. **Verdict: precision (FPU srcA/srcB TF32 truncation, a round-toward-zero bias of ≈ 2⁻¹¹ per operand), not a bug.** There are no failing cells. The lever, if the fp32 corner ever needs it, is to run the post-stats multiply(ies) on the SFPU in fp32 (both operands in DEST via `UnpackToDestFp32`) for `dtype=float32`; `compute_fusion` measures that as slower. This is noted for refinement 5 and is not a queue entry.

**Recommended tolerances**:
- bf16: PCC ≥ 0.9995, rel-RMS ≤ 0.004, atol 0.06;
- fp32: PCC ≥ 0.99999, rel-RMS ≤ 0.003, rtol 2e-3 (allow the −0.12% bias), atol 0.04.

The harness `TOLERANCES` (bf16 0.999 / 0.01, fp32 0.9999 / 0.005) are met with margin.

## Verifier CLI Summary

Final run (`EVAL_CAPTURE_L1=0`, post-fix SUPPORTED), `verifier_report.json` committed alongside:

| Category | Count | |
|----------|-------|---|
| supported_pass | 1283 | |
| xfail_expected | 11790 | every cell maps to refinement 2 (sharded), 4 (w_non_aligned / RM gamma), 5 (fp32_dest_acc_en=False) or the `{float32, False}` EXCLUSION |
| invalid_skipped | 33696 | gamma "none" canonicalization |
| infeasible_skipped | 160 | harness shard-geometry OOM |
| **supported_fail** | **8** | all 8 are `test_distributed_norm_allclose[*-8192-2048-2]`: the op's own L1 refusal (`working set exceeds L1`). Kept failing on purpose and filed as **refinement 1**; per the rules an OOM is not silenced with EXCLUSIONS or a shape tagger |
| **xpass_drift** | **1** | false positive: the negative test `test_pre_all_gather_non_welford_fp32_requires_fp32_dest_acc` passes *because* the op refuses `{float32, False}` |
| **xfail_wrong_mode** | **0** | |
| supported_marked_xfail | 0 | |
| no_axes_found | 87 | `test_ring_mock` / `test_regression` / `test_sweep` call paths not tagged by the harness; 36 passed, 51 skipped, **0 failed** |

Runner line: PASSED=1320 FAILED=8 ERRORS=0 SKIPPED=33907 HANGS=0 TOTAL=47025 (`/tmp/drms/golden2`).
Progression:
- run 0, L1 capture on: 48 passed / 773 failed (765 L1ProfilingError);
- run 1, capture off: 813 / 8;
- run 2, after the SUPPORTED promotions: 1320 / 8.

`verifier_report.json` in this directory is a slim copy (summary plus the loud categories in full).

## Performance snapshot (no change made; informs the perf refinements)

The three LOOSE cases on the 1×4 ring (G = 4, Ring, 2 links, bf16 x + gamma) all pass. Two measurements:
- **Harness `device_kernel_ns`** (untraced): 189.8 µs (640×1792), 59.9 µs (32×1792), 302.4 µs (2048×1280). This metric is ≈ the **sum** of the four chips' kernel times. Each chip's kernel begins at its own dispatch, and the early chips wait at the Fabric fence for the last one launched (implementer's per-chip times at 640×1792: 72.8 / 54.8 / 42.1 / 22.2 µs), so the metric is dominated by host dispatch skew, not op work.
- **Last-launched chip** (the op's own cost): 22.2 / 10.5 / 48.7 µs, against targets of 22.2 / 10.9 / 37.6 µs and the 92.1 / 87.2 / 93.0 µs composite baselines. The targets are *traced* numbers.

The perf refinements must state their measurement method (traced, or last-launched chip). Otherwise skew noise swamps a win.

## Queue coverage (TARGET − SUPPORTED)

| Axis | Missing value | Disposition |
|------|---------------|-------------|
| memory_layout | HEIGHT_SHARDED, WIDTH_SHARDED | Refinement 2 |
| alignment | w_non_aligned | Refinement 4 |
| gamma_layout | ROW_MAJOR | Refinement 4 |
| fp32_dest_acc_en | False | Refinement 5 (`{float32, False}` stays EXCLUDED, precision-convention MUST) |
| dtype | — (bfloat8_b promoted by the verifier) | — |
| alignment | — (h_non_aligned promoted by the verifier) | — |
| (OOM cells, not an axis) | `test_distributed_norm_allclose[*-8192-2048-2]` | Refinement 1 |

Every `xfail_expected` cell maps to one of these refinements. None is covered by INVALID or omitted.

## Recommendations

- **Harness**:
  - fix the MeshDevice CB tracking in graph capture, or exempt fabric ops from `EVAL_CAPTURE_L1`;
  - canonicalize `num_links=None` in `axes.classify_call`;
  - route the 87 `no_axes_found` rows from `test_ring_mock` / `test_regression` / `test_sweep` through `axes.observed()` so the CLI can classify them. All of them pass in the final run.
- **Testing**: under `--dev`, run one mesh shape per pytest invocation. A fabric mesh close under the watcher leaves ERISCs stuck, so the next mesh open fails (reproduced by the implementer with the fabric_all_gather example).
- **Arch portability**: `L1_BUDGET` is a Blackhole-calibrated constant. On Wormhole (1464 KiB L1) it still under-promises (safe), but a real allocator-base query would let the solve use more L1. Exposing one to Python is a framework item.
- **Decode occupancy**: 32×1792 uses 14 of 106 workers (`MIN_BLOCK_W = 4`). It is latency-bound, so more chunks lengthen the gather and fan-out. Refinement 3 measures `MIN_BLOCK_W` both ways.
- **Ordering rationale**:
  - refinement 1 fixes the only real supported_fail and restores the collapsed block loop that refinement 6's overlap lever needs;
  - sharding (2) is the hardest axis unlock, because it moves the port cores and derives the worker set from the shard spec;
  - alignment + RM gamma (4) and the numeric corner (5) extend those kernels;
  - perf passes go in slots 3 and 6 and both target the flagged 640×1792 ring case, whose full config contract was already in Phase 0 SUPPORTED.
