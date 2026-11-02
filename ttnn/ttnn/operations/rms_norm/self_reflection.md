# Self-Reflection: rms_norm (run 1)

## Summary
- **Blind pass was clean.** Golden: 5,429 passed, 0 failed, 1,365 xfail (all `{FLOAT32, fp32_dest_acc_en=False}`, a permanent precision convention), 21 `INFEASIBLE_L1`. Translated: 104 passed, 2 xfail. Regression: 15 passed. `verify_supported`: 0 `supported_fail`, 0 `xpass_drift`, 2 `invalid_unexpected`.
- Every refinement (R1–R5) is ticked `[x]`. Two perf rounds ran after them (Perf 1: 7 graduations; Perf 2: 4 graduations, 1 null).
- **Most important finding: golden is blind to two correctness bugs that shipped green through R2–R5.** The perf subagents found both, not golden:
  1. HEIGHT_SHARDED with a streamed side operand and ≥ 2 tile-rows per shard gave PCC 0.11.
  2. BLOCK_SHARDED with a W tail and a ragged last shard row overran the CB ring.

  Both bugs are fixed now. Neither facet is a golden axis, so a future run would ship the same bugs.
- **The problems are framework-level, not op-level:**
  - golden has axis-blind gaps;
  - a reference is silent on indexed CB reads wrapping the ring (the rule was violated twice this run);
  - the task prompt contradicts the harness about `INPUT_TAGGERS`;
  - golden snapshot timings for DRAM-bound cells swing about 1.4× between snapshots, which sent the perf coordinator after a phantom regression.

## 1. Golden coverage → `eval/golden_tests/rms_norm/feature_spec.py`

No blind failures, so there are no failure clusters. Both findings below come from **escaped bugs** that later phases measured. Both are in TARGET, and both are axis-blind.

**G1. Streamed side operand × multi-tile-row HEIGHT shard (axis-blind).**
- **What:** The bug needs three things at once: shard-backed output, gamma streamed in column windows (L1 pressure from a resident shard plus a wide fp32 gamma), and a shard of ≥ 2 tile-rows. Pass 2 then packed window-major into a row-major shard.
  - Every golden HEIGHT cell has either a resident gamma or 1 tile-row per Tensix core, so `golden_refinement_4/5` were green.
  - `(1,1,7136,736)` HEIGHT passed in every snapshot (gamma resident at W=736).
- **Evidence:**
  - `perf_experiments/gamma_stream_l1/runs/log_run11.txt:2`: `hgt_tile_8320x4608_gf32 op: pcc_ref=0.113505`. Plan line: `b=2 k=1 slice_w=144 … budget=281472`.
  - `changelog.md:335`: "streamed gamma on a shard-backed output packed multi-row blocks window-major".
  - The fix is in aaa60521c87.
- **Recommendation:** Propose a `LOOSE_CASES` entry:
  `{"inputs": ((1,1,8320,4608),), dtype=bf16, fp32_dest_acc_en=True, layout=TILE, gamma_mode="gamma", gamma_dtype=float32, gamma_layout=TILE, memory_layout=HEIGHT_SHARDED}`
  (260 tile-rows on 130 Tensix cores = 2 tile-rows per shard). Consider a new tagger axis `rows_per_shard ∈ {1, >1}` (tile-rows per Tensix core under the harness's `auto_shard_config`). It separates the local-shard paths that only break when a Tensix core owns more than one tile-row.
- **Confidence:** high (bug and repro are measured). Med on whether this exact shape still streams under every future implementation; the facet (L1-pressured side operand × multi-row shard) is the durable part.

**G2. W tail × ragged last BLOCK shard row with `b/2 < rows_last < b` (axis-blind).**
- **What:** The pre-existing `RMS_MASK_TAIL` path pushed rows twice into `cb_sumsq` (b pages) per front, and the collapse then read by index across the ring end. The result was nondeterministic.
  - It triggers only when the last shard row's tile-rows satisfy `rows_last < b < 2·rows_last` **and** W is non-aligned.
  - The golden BLOCK perf geometry `(7168,1024) [736,128] (8,10)` is ragged but W-aligned.
  - The resilience BLOCK w-tail shapes use the auto geometry, which does not hit the window.
- **Evidence:**
  - `agent_logs/perf-part-optimizer_breadcrumbs.jsonl:34`: "PRE-EXISTING op bug: RMS_MASK_TAIL path pushes rows twice … Base nondeterministic on blk (7000,1000) BLOCK [736,128]".
  - Repro: `perf_experiments/combine_pipeline/bench_combine_pipeline.py:73-77`.
- **Recommendation:** Propose the W-tail twin of the existing BLOCK perf profile:
  `{"inputs": ((1,1,7000,1000),), **_PERF_BASE, memory_layout=BLOCK_SHARDED, "extras": {"shard_shape": [736,128], "core_grid": (8,10), "math_fidelity": HiFi2, "pcc_threshold": 0.9995}}`.
  The last shard row is 376 rows = 12 tile-rows against b = 23, so it sits inside the wrap window. Because the failure is nondeterministic, consider also `(1,1,1000,1000)` with `[192,128] (8,6)` (last shard row = 2 tile-rows). Longer term, an axis `ragged_last_shard ∈ {none, < half, > half}` would capture the facet.
- **Confidence:** high on the gap; med on single-run detection (nondeterministic).

## 2. SUPPORTED honesty → `rms_norm.py` `SUPPORTED` / `EXCLUSIONS`

- **No over-claim or under-claim:** 0 `supported_fail`, 0 `xpass_drift`, 0 `xfail_wrong_mode` in `golden_blind_final/verifier_report.json`. All 1,365 xfails are the `{FLOAT32, fp32_dest_acc_en=False}` exclusion, spread evenly over placement × `Layout` × alignment. Nothing to promote or demote.

**S1. `invalid_unexpected` stays loud for a correct native refusal (harness).**
- **What:** These 2 cells are bf8b × `w_non_aligned` in `test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width[w72_c2_nonaligned-bfloat8_b]` and `[w200_c3_nonaligned-bfloat8_b]`.
  - The op refuses them natively (`EXCLUSIONS` mirrors `INVALID` exactly as the R1 verifier note asked, `rms_norm.py:105-110`), and they xfail.
  - `verify_supported._categorize` still maps any non-`skipped` INVALID cell to `invalid_unexpected` (`eval/verify_supported.py:190-194`), which is in `_LOUD_CATEGORIES` (`:361`).
  - The flag stayed loud from Phase 0 (the R1 note says "the CLI already reports 2 `invalid_unexpected`") through the blind pass, with no action available to the op.
- **Recommendation:** Classify INVALID × `xfail` as a quiet `invalid_refused`, and keep `invalid_unexpected` for INVALID × `passed`/`failed`. Alternatively, have `test_translated` consult `INVALID`.
- **Confidence:** high.

## 3. Helper / reference docs

**H1. Absent rule: an indexed CB read must not straddle the ring end. Violated twice this run.**
- **What:** `ttnn-cb-memory-fundamentals.md:84-125` (CB Ring-Wrap Invariant) and `blocking-model.md:179-190` cover only push/pop quanta closing the cycle. `static-analysis-checklist.md:228-233` says "hard to catch statically". None of them states that an **indexed** access (`*_tile(cb, idx)` with `idx > 0` after `cb_wait_front`) reads `fifo_rd_ptr + idx·page` with no modulo, and is bounded by `cb_access_within_bounds` (`tt_metal/hw/inc/internal/circular_buffer_interface.h:139-142`).
  - Instance 1: the planner designed a two-block `cb_x` ring with Offset-addressed lookahead (`ttnn-implementer_breadcrumbs.jsonl:4`).
  - Instance 2: the shipped `RMS_MASK_TAIL` double-push bug (§1 G2).
- **Recommendation:** Add to the Ring-Wrap section: "Indexed reads never wrap: every window a consumer addresses by index (`wait_front` count) must sit wholly below `fifo_limit` from the current read pointer. Either capacity = an exact multiple of that window, or use two CBs (ping-pong) for simultaneous access to two blocks." Mirror it as a check in `static-analysis-checklist.md` §4.5.
- **Confidence:** high.

**H2. Misleading example: `tilize_helpers.hpp:168-171` (example 6) corrupts output as written.**
- **What:** Back-to-back `InitOnly` / `Neither` / `UninitOnly` calls keep the default `reconfig_mode` (`tilize_helpers.hpp:193`). The `.inl` then runs `reconfig_data_format_srca` / `pack_reconfig_data_format` on every call (`tilize_helpers.inl:150-167`), including after `fast_tilize_init`. Rows 1..31 of later tiles come out as zero.
- **Evidence:**
  - `ttnn-implementer_breadcrumbs.jsonl:3` (cost med).
  - The workaround is at `kernels/rms_norm_compute.cpp:405-421`: "a reconfig issued after fast_tilize_init clobbers the fast-tilize unpack config".
- **Recommendation:** In example 6, pass `ReconfigureRegisterDatatypeMode::NoReconfigure` on the `Neither`/`UninitOnly` calls. Better, have the helper skip the reconfig whenever it does not init.
- **Confidence:** high.

**H3. Absent rule: which host API gives the CB L1 budget.**
- **What:** The implementer used `ttnn.get_max_worker_l1_unreserved_size()`. It overshoots the CB region by about 69 KB on BH (1,531,776 vs 1,461,120 bytes) and ignores L1-resident tensors. Fit-boundary shapes OOMed at runtime.
- **Evidence:**
  - `incremental-verifier_breadcrumbs.jsonl:1`; fix commit 940840b62fd.
  - The correct derivation now lives only in op code (`rms_norm_program_descriptor.py:888-900`, `ttnn.get_memory_view(device, BufferType.L1)`).
  - `grep` finds neither API in `.claude/references/` or `.claude/skills/memory-budget-metal`.
- **Recommendation:** State in `l1-footprint-discipline.md` and `/memory-budget-metal`: "CB budget = lowest allocated L1 buffer address from `get_memory_view(device, BufferType.L1)`, never `get_max_worker_l1_unreserved_size()`".
- **Confidence:** high.

#### Helper gaps (perf)

Rows are clustered per helper. Each verdict was checked against the `.hpp` + `.inl` (or the compute API header that the helper forwards to).

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `DestReuseBinary` / `*_reuse_dest_init` (Perf 1 gamma row-bcast; Perf 2 narrow-flat pass 2) | capability | **missing** (at both layers) | `api/compute/eltwise_binary.h:63-64` hard-codes `BroadcastType::NONE` in `binary_reuse_dest_init`; `chain.inl:1339-1349` forwards only the reuse type. 119,794 → 70,276 ns (hgt_tile_grm). The raw subclass `MulGammaRowBcast` (`rms_norm_compute.cpp:322-350`) inherits every chain trait and only swaps init/exec. | Add a `BroadcastDim` template parameter to `DestReuseBinary` and to `{add,sub,mul}_reuse_dest_{init,tiles}`. The narrow-flat "chain cannot start from a matmul result in DEST" part is separate and stays an open request (confidence low). |
| SFPU unary API (`rsqrt_tile`, `binop_with_scalar`): sub-tile scope (Perf 1 col-0 finalize; Perf 2 flat one-vector finalize; narrow-flat FoldFinalize) | capability | **missing** | `api/compute/eltwise_unary/rsqrt.h:52` passes `VectorMode::RC` with no parameter, so not even the face scope the rows mention is exposed. All three sites call raw `_llk_math_eltwise_unary_sfpu_params_` with a custom SFPI body (`rms_norm_compute.cpp:130`, `rms_norm_flat_compute.cpp:72`, `rms_norm_narrow_flat_compute.cpp:105`). Note: the recorded Perf 2 pair is **3,270 vs 3,279 ns (equal)**. Its win (≈ 210 ns per stick) only shows at many sticks, and the row should cite such a cell. | Expose `VectorMode` on `rsqrt_tile` and friends. Add a fused `scale_add_rsqrt_tile(dst, scale, add, VectorMode)`, with a column-0 (even-parity) scope. |
| `compute_kernel_lib::reduce` (owner combine + member collapse) | capability | **split:** member row-form collapse = **missing**; owner `REDUCE_COL → transpose_dest → finalize` = **undocumented** | The helper re-inits on every call (`reduce_helpers_compute.inl:973`). For REDUCE_COL, `post_reduce_op` runs after the last `reduce_tile` of the chunk (`.inl:1249-1252`). So one call per owned row (`Ht=row_gather, Wt=1`, BulkWaitBulkPop) with a post-op `{transpose_dest_init; transpose_dest; finalize}` appears expressible. The raw loop already re-inits per row (`rms_norm_compute.cpp:266`). What is missing is a doc statement: the `.hpp:517-521` post-op contract never says whether a post-op may reprogram the FPU MOP. The member collapse needs a transposed-input reduce (row sums into row 0), which no option offers. | Document in `reduce_helpers_compute.hpp:517`: "a post_reduce_op may reprogram FPU/MOP state only when it is the last op before pack in a single-batch, single-chunk call; the next call re-inits". For the member collapse, add `ReduceOutputForm::Row`. (confidence med; not device-verified) |
| `eltwise_chain` `BinaryFpu<…, DestAccumulation::PerRow>` (pass 1) | capability | **missing** (heterogeneous tail element), plus one DEST acquire per row (`chain.hpp:280`) | **≈ equal ns on no-tail cores: 44,507 vs 43,953 (1.2 %)**. On those cores the bypass bought one uniform code path, not speed. The helper was abandoned on every core because of one tail tile. | Add a "last-column override" element (e.g. `TailFold<BinaryFpu<Mul, x, bcast_row(mask)>, DestReuseBinary<Mul, x>>`) and multi-row batching for PerRow. |
| `eltwise_chain` `TileAddressing::Strided` (HEIGHT RM flat pass 1/2) | capability | **missing** | `StridedTileRange{base, row_stride}` only (`chain.hpp:263-266`). There is no column stride for page ≠ tile (`page = gcd(stick pitch, tile bytes)`) and no second-CB tail slot. | Add `col_stride` to `StridedTileRange`. |
| `dataflow_kernel_lib` local copy `set_read_state` / `read_with_state` (Perf 1 RM staging; Perf 2 flat tail) | capability | **missing** | `local_copy_helpers_dataflow.hpp:92`: "src_addr … must be the address set_read_state() programmed". There are no write variants. The raw call it replaces, `noc_async_read_one_packet_with_state(src + s·pitch, dst + s·row)` (`rms_norm_reader.cpp:162-165`), already varies both addresses, so the helper is narrower than the API under it. 163,563 → 157,073 ns (4 %). | Let `read_with_state` take any `src` on the programmed NoC/size, and add `set_write_state` / `write_with_state`. |
| kernel_lib `Mcast` (narrow-flat inv return) | capability | **missing** (confidence low) | `mcast_args.hpp:147`: `num_senders = rotating ? rotating_span : 1`. The flat-unit case is every member being a sender of its own unit at the same time. | Request a "per-unit leader" pipe. Do not graduate it before a second op needs it. |
| `matmul_block` (narrow-flat pass 1/2) | capability | **missing** (confidence low) | No DEST-resident output for a following DEST-reuse op; no per-tile constant indexing (`matmul_block_helpers.hpp:25-60` describes only CB-lifecycle pack modes). | Add a `PackMode::LeaveInDest` / caller-owned DEST option. |
| no helper: partial-row scatter, zero-row fill, 0/1 selection-matrix construction | capability | **missing** (no API exists) | `rms_norm_writer.cpp:410,499`; `rms_norm_narrow_flat_reader.cpp:22`. | Low priority; one-op needs. |

- **Unrecorded bypass (gap in the record):** `rms_norm_narrow_flat_compute.cpp:120` uses raw `mul_tiles(x_cb(f), x_cb(f), …)` for the x² of narrow-flat pass 1. No row covers it: the Perf 2 `matmul_block` row cites only `:145` / `:240`. It most likely has the same cause as the Strided row (flat-tile index). Propose that the Perf 2 table add it. Confidence med.

API sketches (proposals, derived from the raw sequences written):
```cpp
// DestReuseBinary with a broadcast CB operand — replaces MulGammaRowBcast (rms_norm_compute.cpp:322-350)
template <BinaryFpuOp Op, InputSpec In, DestReuseType R, BroadcastDim B = BroadcastDim::None, Dst D = Dst::D0>
using DestReuseBinary = /* … */;
// call site:
using MulGamma = DestReuseBinary<BinaryFpuOp::Mul, gamma_in, DestReuseType::DEST_TO_SRCA, BroadcastDim::Row>;

// Stateful local copy with both addresses stepping — replaces rms_norm_reader.cpp:162-165
dataflow_kernel_lib::set_read_state(noc, src_noc_xy, bytes);           // fixes NoC coords + size only
for (s...) dataflow_kernel_lib::read_with_state(noc, l1 + s * row, src + s * pitch);
```

## 4. Agent prompts

**P1. The task prompt contradicts the harness about `INPUT_TAGGERS`.**
- **What:** `eval/prompts/rms_norm.txt:92-93` says `tag_gamma_dtype` / `tag_gamma_layout` "MUST" be `INPUT_TAGGERS`. But `eval/feature_matrix.py cartesian()` treats every `INPUT_TAGGERS` key as shape-derived and stops iterating it from TARGET, which collapses both gamma axes. Taggers also only see shapes.
  - The implementer correctly deviated (`rms_norm.py:57-65` comment).
  - A less careful run would have shrunk golden silently.
- **Evidence:** `ttnn-implementer_breadcrumbs.jsonl:5`.
- **Recommendation:** Reword the prompt: "project gamma_dtype/gamma_layout to "none" inside `validate()`; do not register them in `INPUT_TAGGERS`". Consider adding the general rule to the planner prompt: only shape-derived facets go in `INPUT_TAGGERS`.
- **Confidence:** high.

**P2. The earlier-snapshot priority rule followed a timing artifact.**
- **What:** `perf-coordinator.md:120-123` makes any axis value > 5 % faster in an earlier `golden_*` snapshot a priority target. Snapshot timings of DRAM-bound INTERLEAVED cells swing about 1.4× while L1-resident cells stay at 1.00×:
  - R5/R4 median for INTERLEAVED > 50 µs is 0.854×, against 0.996–1.005× for sharded.
  - blind/`golden_perf_2` for `(1,1,8192,7168)` perf config is 431 → 605 µs. The coordinator's same-session A/B measured about 600 µs for both.

  Both coordinators chased the artifact. Perf 1 called it non-reproducible (`changelog.md:311-316`). Perf 2 spent a whole subagent (`prefill_dram`, NULL) and diagnosed it as throttled-AICLK ticks (`changelog.md:474-481`).
- **Recommendation:**
  - Harness: record the measured AICLK per golden snapshot, and either normalize `device_kernel_ns` or mark the snapshot `clock_suspect`.
  - Prompt: before treating a cross-snapshot regression as a target, re-measure one affected cell at round start; if it does not reproduce within 5 %, discard the snapshot.
- **Confidence:** high on the artifact; med on the AICLK cause.

**P3. Golden-suite guard runs are the safety net; the isolated benches are not.**
- **What:** Perf 1's `gamma_rowbcast` regressed per-Tensix-core RM-gamma loads (409 vs 113 µs). The subagent's bench never took that path, and only the coordinator's full-golden guard run caught it (`changelog.md:362-367`, fix 9b355328ef7).
- **Recommendation:** In `perf-part-optimizer.md`, require one bench cell per loader path (multicast sender, per-core, streamed) whenever an idea changes a load that several paths share. Keep the coordinator's full-golden-per-graduation requirement as it is.
- **Confidence:** med.

**P4. No prompt caught either escaped correctness bug before the perf phase.**
- **What:** Verifier gates for R2 and R4 were "every golden cell whose only gap was this value passes". That inherits golden's blind spots (§1 G1/G2).
- **Recommendation:** In `incremental-verifier.md`, for each new placement value, require acceptance cases at the facet boundaries the design itself names: ragged last shard (`< half`, `> half`), ≥ 2 tile-rows per shard, and the L1-fallback (streamed) path forced on. Not only golden cells.
- **Confidence:** med.
