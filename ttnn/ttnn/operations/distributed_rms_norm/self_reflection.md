# Self-Reflection: distributed_rms_norm

## Summary
- **Blind final:** golden 10763/10763 SUPPORTED cells pass, with 0 hangs. Translated: 264 pass, 3 fail, 12 xfail. `verify_supported` reports `supported_fail=2`, `supported_marked_xfail=12`, `xfail_wrong_mode=1` and `xpass_drift=0`. All six refinements and two perf rounds completed (`op_requirements.md` has no unchecked `[ ]`). Every TARGET value is in SUPPORTED.
- **Most important finding:** the two real failures are tall-prefill cells (per-device M ≥ 32768). The cause is not the L1 working set that the error message names. The exchange **board** is sized one region per round, so its bytes grow with the total row count. Once `BOARD_CAP_BYTES` binds, the solve has to use fewer W-chunks, which makes each block too wide for L1. Golden never goes above M=4096, and no tagger captures row magnitude, so the gap is axis-blind.
- **Op-level vs framework-level:** the problems are mostly framework-level:
  - Translated cells that become SUPPORTED partway through the queue are never re-gated.
  - A verifier note claimed a bound ("holds for any `tensor_h_tiles`") that its Done-when never tested.
  - The L1-profiling harness rejects tests that expect the op to refuse.
- The perf phase ran 2 rounds (640×1792 went from 18.43 to 16.79 µs) and recorded 3 helper bypasses. All 3 are real capability gaps. One has an undocumented partial overlap with the helper.

## 1. Golden coverage → `eval/golden_tests/distributed_rms_norm/feature_spec.py`

**G1. Tall row count is an untested, axis-blind region (in TARGET)** — confidence **high**
- **What:** In `test_ds_distributed_norm`, the cells with M ∈ {128, 1024, 8192} pass. The cells with M ∈ {32768, 131072} fail. All of them have identical observed axes (bf16, fp32_dest_acc_en=False, RM gamma, INTERLEAVED, Linear, cluster_axis 1, tile_aligned, rank 4). Row magnitude is the only thing that separates pass from fail.
  - The axes-tuple is "covered" by golden, but golden's largest M is 4096 (`feature_spec.py` INPUTS `(1,1,4096,1024)`).
  - The `alignment` and `rank` taggers cannot see this facet.
- **Evidence:**
  - `golden_blind_final/test_results.json`: `test_ds_distributed_norm[random_weights-prefill-32768-0.98]` and `[...-131072-...]` fail with `RuntimeError: ... working set exceeds L1 for every block (streaming regime deferred)`. Per-device shape is `(1,1,M,3584)`, G=2.
  - I replayed `plan_blocks` on the host (110 workers assumed, bf16): `(1,1,32768,1792)` fails at G=4 with 1 or 2 links, and at G=2 with 1 link. `(1,1,16384,1792)` fails at G=4 with 1 link. `(1,1,131072,*)` fails everywhere.
- **Recommendation:** Add a `LOOSE_CASES` entry `{"inputs": ((1, 1, 32768, 1792),), **_PHASE0}`. This is DeepSeek/Kimi TP4 at a 32K prefill, and it fails on the 1×4 ring placement. Optionally add `(1,1,16384,1792)` with `num_links=1` as the boundary case. Consider also promoting a `row_regime` tagger (e.g. `decode` ≤32 / `prefill` ≤4096 / `long_prefill` >4096) to `INPUT_TAGGERS` and TARGET, so the facet appears in the cartesian product rather than only in a pinned case.

**G2. Output placement ≠ input placement has no axis** — confidence **med**
- **What:** 12 translated cells pass a sharded `memory_config` that differs from the input's shard spec (e.g. offset grid, or `output_core_grid` ≠ `core_grid`). `validate()` refuses them with `UnsupportedAxisValue`, but no axis describes output placement, so they show up as `supported_marked_xfail`. The `core_grid0` cells, where output matches input, pass.
- **Evidence:** `test_simulated_distributed_layernorm[core_grid{1,2,3}-...]` → "a sharded output must have the input's memory config"; refusal site `distributed_rms_norm.py:207-209`.
- **Recommendation:** Consider adding an `output_placement` axis {`interleaved`, `same_as_input`, `resharded`} to TARGET (via a tagger on `memory_config`). Then either refine `resharded`, or leave it out of SUPPORTED so these cells become honest `xfail_expected`. This is an ambition decision for a human. Do not add a LOOSE_CASE until TARGET includes the value.

## 2. SUPPORTED honesty → op file `SUPPORTED` / `EXCLUSIONS`

**S1. `supported_fail=2` — fix, do not demote (board sized per round)** — confidence **high**
- **What:** The real binding constraint is the board. `_candidate` charges `G · rounds · board_slot_bytes(...)` against `BOARD_CAP_BYTES` (256 KiB). Because the board keeps one disjoint region per round, its bytes are proportional to the total tile rows × `num_w_chunks`.
  - At tall M the board check forces the chunk count down, and the wider block then fails `L1_BUDGET`. My replay at G=2, W=3584, M=32768 found candidates that pass the board check and candidates that pass L1, but none that pass both. At 131072 even the board check alone has no candidate.
  - The error text ("working set exceeds L1 … streaming regime deferred") blames the wrong constraint. The deferred R2 streaming regime would not fix this, because it does not shrink the board.
- **Evidence:**
  - `distributed_rms_norm_program_descriptor.py:288`: `if G * rounds * board_slot_bytes(...) > BOARD_CAP_BYTES`
  - `:368` (the raise)
  - `l1_ledger.md:30`: "one disjoint region per round: no intra-call fence"
  - `op_design.md:153` already establishes "a worker is at most one round ahead".
- **Recommendation:**
  - **Fix:** bound the board to a small ring of round regions (e.g. 2–3, reused once the leaders have drained them; the parity semaphores already exist). As an alternative, let the interleaved solve fall back to R3 pre-combine when the board cap binds, the same way the sharded solve already does. Either change makes the board independent of `tensor_h_tiles`.
  - Separately, make the error name the constraint that actually bound (board vs L1).
  - A shape-size limit is not an axis, so demoting via EXCLUSIONS is the wrong tool here.

**S2. `xfail_wrong_mode=1` is a harness artifact, not op dishonesty** — confidence **high**
- **What:** `test_pre_all_gather_non_welford_fp32_requires_fp32_dest_acc` asserts that the op refuses `{float32, False}`. The op does refuse (EXCLUSION). The test then fails because the L1 profiler requires a positive allocation peak.
- **Evidence:** blind message `L1ProfilingError: L1 profiling returned peak_total=0`; `eval/l1_profiling.py:85-88`. Already flagged in commit `7252c9d9f3f` ("negative test refusing before any allocation; harness artifact") and in a verifier breadcrumb (`incremental-verifier_breadcrumbs.jsonl` #4). It was never fixed.
- **Recommendation:** Make `l1_profiling` skip the positive-peak check when the test body raised or passed through `pytest.raises` (or add an opt-out marker for expect-refusal tests). No op change is needed.

**S3. `num_links: None` is in SUPPORTED to work around the classifier** — confidence **med**
- **What:** The op lists `None` in `SUPPORTED["num_links"]` only so that `classify_call` (which records the raw kwarg) classifies default-link translated calls correctly. The registry is being bent to fit the harness.
- **Evidence:** `distributed_rms_norm.py:96-99`; `verification_report.md:65`; verifier breadcrumb #2 ("7 xpass_drift + 199 xfail_wrong_mode of pure classification noise").
- **Recommendation:** Canonicalize `num_links=None` in `eval/golden_tests/.../axes.py` `classify_call` the same way the op does, then drop `None` from SUPPORTED.

No `xpass_drift`, so there is nothing to promote.

## 3. Helper / reference docs

**H1. `ReduceInputPolicy::BulkWaitBulkPop` doc is wrong for the SFPU path** — confidence **high**
- **What:** The header says REDUCE_COL Bulk = `Ht×chunk` with `chunk = DEST_AUTO_LIMIT`. The `.inl` uses `DEST_AUTO_LIMIT - 1` on the SFPU (Accurate) path, asserts the input CB holds at least `Ht*chunk` pages, and indexes tiles chunk-major. Because the design followed the header, the first device run hit an ASSERT/hang.
- **Evidence:** `reduce_helpers_compute.hpp:101` vs `reduce_helpers_compute.inl:1163-1167`; implementer breadcrumb ("requires the input CB to hold >= Ht*(DEST_AUTO_LIMIT-1) pages … indexes tiles chunk-major").
- **Recommendation:** Change hpp:101 to: "chunk = DEST_AUTO_LIMIT (FPU) or DEST_AUTO_LIMIT−1 (SFPU/Accurate); the input CB must hold ≥ Ht·chunk pages (asserted); tiles are consumed chunk-major (ht·chunk + i)".

**H2. `ReceiverPipe` (Flag) constructor clears the data-ready cell, and the header does not say so** — confidence **high**
- **What:** In Flag mode the constructor runs `data_ready_.set(INVALID)`. If a sender signals before the receiver object is constructed, the signal is lost and the receiver hangs. Refinement 6's gamma multicast hit exactly this hang and switched to Counter mode.
- **Evidence:** `mcast_pipe.inl:381-384`; `mcast_pipe.hpp:16` only says "semaphores are initialized to INVALID". Implementer breadcrumb: "Flag ReceiverPipe ctor sets INVALID after the sender signal already landed … fix = Counter".
- **Recommendation:** Add to the `ReceiverPipe` constructor doc in hpp: "Flag mode: the constructor resets the data-ready cell to INVALID. Construct every receiver before any sender can signal (e.g. before the first PRE_HANDSHAKE ack or the first data read), or use Counter."

**H3. Design's raw-fabric rejection does not actually evaluate `ccl_helpers_dataflow`** — confidence **low**
- **What:** The port sender uses raw `WorkerToFabricEdmSender` with fused write + atomic-inc. `ccl_helpers_dataflow.hpp` advertises exactly that shape (`arm_fused_write_inc`). The design's rejection row gives a reason that only applies to `mcast_pipe`.
- **Evidence:** `op_design.md:321`; `distributed_rms_norm_port_sender.cpp:115-181`; `ccl/ccl_helpers_dataflow.hpp:21-38`.
- **Recommendation:** Planner checklist: a "helpers considered and rejected" row must give a reason for each named helper. The relay/arrival-counter pattern may still be out of reach for the CCL helper; that is unverified.

### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `ckl::eltwise_chain` + `ckl::reduce` (sumsq x²→transpose→col-fold) | capability | **missing** | No transpose element anywhere in `kernel_lib/` (only `matmul_block_helpers.hpp:277` and the reduce-internal reload at `reduce_helpers_compute.inl:706-713`). Every `ReduceInputPolicy` reads from a CB (`reduce_helpers_compute.hpp:112`), so nothing can fold a tile already in DEST. Measured 1442→1040 ns. | Add a `TransposeDest<Dst>` chain element plus a DEST-sourced reduce entry (`reduce_in_dest<SUM, REDUCE_COL>(dst)`). The 2b/8a `transpose_tile` raw use, recorded in the design and kernel header, has the same root cause. |
| `mul_unary_tile` / `add_unary_tile` / `rsqrt_tile` inside `post_reduce_op` | ergonomics | **missing** (API, not ergonomics) | `binop_with_scalar.h:42,70` and `rsqrt.h:52` hard-code `VectorMode::RC`. `kernel_lib/eltwise` exposes no vector-mode/iterations knob (grep: none). The bypass needed `SFPU_UNARY_CALL` plus `TRISC_MATH`-guarded `ckernel_sfpu_*` includes (`compute.cpp:53-56`). 1670→785 ns. | Add a `VectorMode` parameter (default RC) to the unary compute API, and a `.rows(0)` scope on kernel_lib unary elements. |
| `mcast_pipe` `SenderPipe`/`ReceiverPipe` (inv_rms fan-out) | capability | **missing**, with an **undocumented** partial overlap | Confirmed: on `send()` the Flag source is always the pipe's own cell, with value VALID (`mcast_pipe.inl:277`). The receiver resets the cell to INVALID on every receive (`inl:415`), so there is no monotone "set r+1" mode. Counter mode barriers before the atomic (`inl:245-246`). However, Flag mode already sends data then SET with no barrier on the same path (`inl:268-283`). Multi-rectangle sends (`MAX_RECTS` ≤ 3, `hpp:56`) and the singleton local-copy fallback (`inl:134`) also exist. The bypass row describes these as missing. Measured 1318→420 ns. | Add `DataReadySignal::Monotone`: SET from a sender-owned scratch cell holding a caller value; the receiver uses `wait_min(round+1)` without a reset. Separately, document `MAX_RECTS` and the singleton fallback in the class comment. |

No row has roughly equal helper and raw ns, so no zero-gain ergonomics cost appears. I found no unrecorded perf-phase bypass: the raw calls in the files that `3129bee48cc` and `a10b63c0f92` touched are covered by the three rows.

There is one record nit. Perf 2 says the worker transpose bypass is in the "Perf 1 table". It is not there; it is recorded only in `op_design.md:319` and the kernel header.

```cpp
// sumsq, as the caller wanted it (derived from compute.cpp fused path) — confidence: low
eltwise_chain(IterationShape::tiles(w), Mul<CbX, CbX, AccumulatePerRow>{}, TransposeDest<D0>{},
              SfpuReduce<SUM, REDUCE_COL, D0>{}, PackTo<cb_partial_row>{});
// fan-out, derived from writer.cpp:186-215 — confidence: low
SenderPipe<NOC1, SEM_INV_READY, false, UNUSED_SEM_ID, DataReadySignal::Monotone, false, Unknown, 4> pipe(noc, args);
pipe.send(src, dst, inv_row_bytes, /*signal_value=*/r + 1);   // receivers: rx.receive(r) == wait_min(r+1)
```

## 4. Agent prompts → `.claude/agents/*.md`

**P1. Verifier: a refinement that adds an axis value must re-run the translated cells it newly makes SUPPORTED** (`incremental-verifier.md`) — confidence **high**
- **What:** `test_ds_distributed_norm` prefill cells became SUPPORTED once refinement 5 added `fp32_dest_acc_en=False` (refinement 4 had added RM gamma). Refinement 5's Done-when was "golden … `supported_fail = 0`", checked on an 80-cell golden sample. No translated cells were run. The failures first appeared in Perf 1, where they were waved through.
- **Evidence:** `op_requirements.md:214-218`; changelog Refinement 5 ("random sample of 80 buildable `test_op` cells"); per-phase `golden_refinement_*` dirs contain no `test_translated` cells.
- **Recommendation:** Add to the Done-when template: "re-classify `test_translated.py` under the new registry and run every cell whose classification flips into SUPPORTED".

**P2. Verifier: if a note claims a scaling bound, the Done-when must test that bound** (`incremental-verifier.md`) — confidence **high**
- **What:** Refinement 1's note offered "board slot region per round" as an option and asserted that the footprint is bounded "for any `tensor_h_tiles`". The Done-when only tested up to M=8192. The option the verifier offered made board bytes grow with `tensor_h_tiles`, which falsified the claim.
- **Evidence:** `op_requirements.md:55-59` vs `:63-68`; `l1_ledger.md:30`.
- **Recommendation:** Add a rule: "When a note claims a resource is bounded independent of a dimension, the Done-when includes a probe at ≥8× the largest INPUTS value of that dimension (or the translated suite's max), and every per-round or per-row persistent buffer is listed with its scaling."

**P3. Perf coordinator: a newly observed `supported_fail` must be filed as a refinement, not called "pre-existing"** (`perf-coordinator.md`) — confidence **med**
- **What:** Both perf rounds recorded the 2 failures as a "pre-existing streaming-regime generality gap … reported upstream" without checking the cause (it is the board, not streaming). Golden-green was satisfied only because translated cells are outside the golden directory gate.
- **Evidence:** changelog Perf 1 Correctness, `changelog.md:329`; Perf 2, `changelog.md:429`.
- **Recommendation:** Add to perf-coordinator §"Golden green throughout": "a supported_fail that also fails on baseline is not a perf regression, but record its binding constraint (from the raise site) and file it to `op_requirements.md` as an unchecked refinement".

**P4. The pipeline lets op refinements patch core framework code to work around the eval harness** (`ttnn-implementer.md` / harness) — confidence **med**
- **What:** Refinements 1 and 2 changed `tt_metal/distributed/mesh_workload.cpp` and `ttnn/core/graph/graph_trace_utils.cpp` so that L1 capture would work on mesh workloads. Before that, every golden cell failed with `peak_total=0` (phase 0: 1078/1078 failures). These are framework fixes buried in op commits.
- **Evidence:** commits `7252c9d9f3f` and `5bc32471e61`; `golden_phase0` failure messages; verifier breadcrumb #1 (cost: high).
- **Recommendation:** Upstream the two fixes separately. In the implementer prompt, require harness and framework fixes to be split into their own commit, flagged `[framework]`.

**P5. Verifier prompt assumes the golden run fits in a foreground Bash call** — confidence **med**
- **What:** For a CCL op the golden runner takes 25–30 minutes, but the Bash tool caps at 10 minutes.
- **Evidence:** verifier breadcrumb #3.
- **Recommendation:** In the incremental-verifier Testing Protocol step 2, launch the runner detached, poll its log, and retry when the device lock times out.
