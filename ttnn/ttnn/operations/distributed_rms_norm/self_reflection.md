# Self-Reflection: distributed_rms_norm

## Summary
- **Blind pass:** 11,076 passed / 2 failed / 2,160 strict xfail / 33,787 skipped (of 47,025). Golden `test_op` + `test_op_loose`: 0 failures. Translated: 277 passed / 2 failed. Regression and ring-mock: all green. `verify_supported`: supported_fail 1, xfail_wrong_mode 1, xpass_drift 0.
- **Queue:** all refinements (1, 1b, 2–5) are ticked `[x]`. Two perf rounds ran: Perf 1 graduated 2 ideas, Perf 2 graduated 1 and had 1 null. Focus case went 22.08 → ≈18.8 µs against a 14.1 µs target.
- **Most important finding:** the only real op failure is an **axis-blind tall-M limit**. The persistent landing scratch grows with the per-device row count (`land = nw * G * seg`). So per-device M = 131072 raises `footprint(R=1) > budget` even though every declared axis of that cell is SUPPORTED. Golden never goes past M = 4096. Refinement 4 promoted this cell into SUPPORTED without anyone noticing. Both perf rounds then labelled it "pre-existing" and moved on.
- **Op-level vs framework-level:** the op is in good shape. Most findings are framework-level:
  - the L1-capture harness rejects expected-raise tests;
  - neither the reference nor the skill says that `UnpackToDestFp32` requires 32-bit DEST;
  - gates never re-run translated tests after a SUPPORTED widening;
  - the perf bypass record mis-describes one helper's contract.

## 1. Golden coverage → `eval/golden_tests/distributed_rms_norm/feature_spec.py`

**1.1 Tall per-device M is axis-blind and untested (in TARGET).**
- **What:** `test_ds_distributed_norm[random_weights-prefill-131072-0.98]` fails. Its axes tuple (bf16, TILE, 16-bit DEST, γ bf16 ROW_MAJOR, INTERLEAVED, axis 1, Linear, tile_aligned, rank 4) is fully SUPPORTED and well covered by golden. The only thing that separates it from its passing siblings is per-device M:
  - passes: 128, 1024, 8192 and 32768;
  - fails: 131072;
  - golden INPUTS go up to M = 4096 (`feature_spec.py`, `((1, 1, 4096, 1024),)`).

  The mechanism is that the landing scratch is sized for every wave at once. So it scales with M_t·G·128 B / L, independent of W. Once it no longer fits, the plan raises.
- **Evidence:**
  - `distributed_rms_norm_program_descriptor.py:853`: `land = nw * G * _align_up(seg * STATS_ROW_BYTES, SEG_ALIGN)`
  - `:956`: `landing_bytes=num_waves * G * seg_stride`
  - `:903`: "one resident tile-row (block_rows = 1) does not fit L1 (footprint(R=1) > budget)"
  - blind `test_results.json`: the one failing translated cell.
- **Recommendation:** two proposals.
  - **Pin the boundary with a `LOOSE_CASES` entry** that holds every axis at the failing cell's values except the dtype config. The exact config: `{"inputs": ((1, 1, 131072, 1792),), **{**_PHASE0, "topology": ttnn.Topology.Linear, "num_links": 1}}`. One link doubles the landing per port, so this is the hardest variant.
  - **Add a cheap sibling**, `((1, 1, 131072, 64),)`. The landing does not depend on W_local, so this should hit the same wall with much less host memory (unverified).
  - **Consider an `INPUT_TAGGERS` axis** for per-device tile-row magnitude, for example `rows ∈ {decode ≤ 1 tile-row, prefill ≤ 128 tile-rows, long_prefill > 1024 tile-rows}`. Today no tagger can tell a 4K-row cell from a 128K-row cell.
- **Confidence:** high that the failure is a gap. Med on the exact boundary between 32768 and 131072, and on whether the W = 64 sibling reproduces it.

No other blind failure cluster maps to an untested in-TARGET region. The 40 `INFEASIBLE_L1` skips (fp32 32×8192 and 128×8192 sharded) and the 45 "2-device ring is the line" skips are honest structural skips.

## 2. SUPPORTED honesty → `distributed_rms_norm.py` `SUPPORTED` / `EXCLUSIONS`

**2.1 `supported_fail` = 1: tall M (cluster 1.1). Recommend fix, not demote.**
- **What:** this is an over-claim, but no axis can express it, so there is nothing to move to `EXCLUSIONS`. A demotion would have to remove `fp32_dest_acc_en=False` or `ROW_MAJOR`, which is wrong.
- **History:**
  - The cell only became "supported" in Refinement 4, which added `fp32_dest_acc_en=False`. Before that, the translated test's `HIFI4_NOFP32_ACC` config was outside SUPPORTED.
  - The Refinement 4 gate ran no translated tests: `golden_refinement_4/test_results.json` contains only `test_golden`, `test_regression` and `test_ring_mock`.
- **Evidence:** `verifier_report.json` → `supported_fail[0]` = `test_ds_distributed_norm[random_weights-prefill-131072-0.98]`, `failure_category: validation`. `changelog.md:530` and `:684` say "It is **pre-existing**".
- **Recommendation (fix):**
  - Size the landing scratch for the in-flight waves only, not `num_waves`. The port already has a ready fence per wave, and `slot_depth` exists.
  - Equivalently, recycle wave slots once the roots have consumed them. Then `landing_bytes` becomes `slot_depth * G * seg_stride`.
  - Until then, have `validate()` state the M ceiling in its error message.
- **Confidence:** med. This is inferred from the solve code; the scratch size for this cell was not instrumented.

**2.2 `xfail_wrong_mode` = 1: harness artifact, not op dishonesty.**
- **What:** `test_pre_all_gather_non_welford_fp32_requires_fp32_dest_acc` passes its `pytest.raises`: the op correctly refuses fp32 + 16-bit DEST before dispatch. The L1 capture plugin then fails the test because nothing was allocated.
- **Evidence:** `eval/l1_profiling.py:86`: `"L1 profiling returned peak_total=…; expected a positive allocation peak"`. The same failure was already diagnosed in the Refinement 1b outcome (`op_requirements.md`, "an expected-raise test where the op correctly refuses before dispatch").
- **Recommendation:** in `capture_for_item`, skip the positive-peak assertion when no program was dispatched (empty trace), or when the test is an expected-raise test. Alternatively, let translated expected-raise tests opt out with a marker. This is framework-level and affects every op that has refusal tests.
- **Confidence:** high.

**2.3 `xpass_drift` = 0.** No under-claims.

## 3. Helper / reference docs

**3.1 Absent rule: `UnpackToDestFp32` requires `fp32_dest_acc_en=True`.**
- **What:** Refinement 4 followed the plan and tagged `cb_stats_slots` `UnpackToDestFp32` regardless of DEST width. With 16-bit DEST, every output was inf: the first slice passed 31/108 cells, with pcc = 0. Both the skill and the reference describe the tag's *exclusivity* (no FPU consumers) in detail, but neither states the DEST-width precondition. The skill's §5 example even tags the CB unconditionally right next to a user-supplied `fp32_dest_acc_en`.
- **Evidence:**
  - `changelog.md` Refinement 4: "With 16-bit DEST the tag made the unpacker write 32-bit words into a DEST that math / pack read as 16-bit, and every output was inf".
  - implementer breadcrumb (2026-10-23T00:04): "UnpackToDestFp32 gated on fp32 DEST (16-bit DEST gave inf)".
  - `.claude/skills/numeric-formats-metal/SKILL.md:224-229`: "Apply this only to CBs that meet both conditions" (only two conditions, neither about DEST width).
  - `.claude/references/numerical_stability_analysis_reference.md:135-151` (§2.7).
- **Recommendation:**
  - Add a third condition to SKILL.md §5 and to reference §2.7: "`fp32_dest_acc_en=True`. With 16-bit DEST the tag writes 32-bit words into a 16-bit DEST (inf / garbage), so gate the tag on the flag."
  - In both code samples, make the tag conditional (`if compute_kernel_config.fp32_dest_acc_en:`).
- **Confidence:** high.

**3.2 The perf prompt points at a header that does not exist.**
- **What:** the perf tournament prompt mandates `MaybeDeviceZoneScope` from `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`. That file is not in this tree, and both perf rounds had to note it and fall back to `DeviceZoneScopedN`.
- **Evidence:** `.claude/eval/prompts/perf_tournament_prompt.txt:9`; `changelog.md` Perf 1 Instrumentation: "`perf_instrumentation.hpp` (`MaybeDeviceZoneScope`) does not exist in this tree".
- **Recommendation:** either ship the header (`.claude/references/device-zone-scope-attribution.md:11` already defines it as an alias of `DeviceZoneScopedN`), or change the prompt to "use `DeviceZoneScopedN` (alias `MaybeDeviceZoneScope` where available)".
- **Confidence:** high.

**3.3 The mcast helper documents value-carrying signals only on the control channel.**
- **What:** see the `McastFamily` row in the table below. The perf author concluded that "Flag mode writes only a fixed VALID word". In fact `send_signal(value)` sends a caller value, but the `McastArgs` usage block (`mcast_args.hpp:14-50`) only shows `send()` and `receive()`.
- **Recommendation:** add one line to the `mcast_args.hpp` example: `sender.send_signal(v)` sends an arbitrary value ≥ VALID in Flag mode, read with `receiver.receive_signal()`.
- **Confidence:** med.

#### Helper gaps (perf)
| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| kernel_lib `McastArgs` → `SenderPipe` (mcast) | capability: Flag writes a fixed VALID, Counter barriers before its atomic | **missing**, narrower than claimed, and partly **undocumented** | The Counter barrier claim holds: `mcast_pipe.inl:245-246` (`async_write_barrier` when Counter). The Flag claim is wrong: `send_signal(uint32_t value)` multicasts a caller value (`mcast_pipe.hpp:78-82`; `.inl:273-277` "a typed control signal instead writes its caller-supplied value"). The real gap is that `send()` always closes each payload with its own VALID signal (`.inl:143`, `signal_ready_(rectangle, loopback, destinations)` with the default value). So "N linked payloads, then one valued SET per rectangle" cannot be expressed: `send_signal` also re-runs the PRE_HANDSHAKE wait. Site: `distributed_rms_norm_reader.cpp:402-404`. | Add a data-only `send_payload` (linked, no signal) to pair with `send_signal(value)`. Also document the valued Flag signal in `mcast_args.hpp`. |
| kernel_lib `eltwise_chain` SFPU ops (`AddBinary`, `Rsqrt`, `MulUnary`, `AddUnary`) | capability: no way to scope an op to a subset of DEST vectors | **missing**. The bypass itself stays inside the chain. | `grep VectorMode ttnn/cpp/ttnn/kernel_lib/eltwise` finds nothing; every stock op runs over the full tile. The author used the chain's CRTP extension point (`AddRow01 : ckl::BinaryOp`, `FinalizeRow01 : ckl::UnaryOp`, `distributed_rms_norm_compute.cpp:118-145`). Only the LLK params call is raw. That extension point is documented only in `eltwise/core/chain.inl:565-577`, not in `api/chain.hpp`. The ns pair (19,210 vs 18,830) is protocol-level: it includes the deleted RISC-V expansion, so it is not helper-vs-raw. | Add a `VectorMode` template parameter (default `RC`) to the stock SFPU chain ops. Move the "custom op" recipe into `api/chain.hpp`. |
| `transpose_dest` / inter-face SFPU move | capability: no chain element | **missing** | No `transpose_dest` wrapper anywhere in `ttnn/cpp/ttnn/kernel_lib` (grep). Raw at `distributed_rms_norm_compute.cpp:150-157` (`Row01ToCol0`). | Add a `TransposeDest<Slot>` chain element (both DEST widths). |

**Record completeness:** checked. No equal-ns ergonomics rows. Two raw paths in the kernels have no bypass row, and both predate the perf phase, so they are outside its record:
- the reader's in-chip unicast gather and R3-RS inv all-gather (`reader.cpp:248-262` and `:515-571`, from commits `58ca92ecbad` and `766c002c61c`);
- the port kernel's raw Fabric sends.

No unrecorded *perf-phase* bypass.

```cpp
// Sketch (confidence: low — one call site). What forward_and_signal wanted, per rectangle:
sender.send_payload</*linked=*/true>(rect, src_l1, dst_l1, ROW_BYTES);   // ×n, no data-ready signal
sender.send_signal</*SKIP_HANDSHAKE=*/true>(rect, (block + 1) << 24 | mask);
// call site:
for (auto& r : rects) { for (i < n) sender.send_payload(r, wire(i), wire(i), ROW_BYTES); sender.send_signal(r, v); }
```
```cpp
// Sketch: VectorMode on stock SFPU chain ops (removes AddRow01 / FinalizeRow01)
ckl::AddBinary<D0, D1, D0, ckl::VectorMode::Rows01>{}, ckl::MulUnary<D0, ckl::VectorMode::Rows01>{inv_w},
ckl::AddUnary<D0, ckl::VectorMode::Rows01>{eps}, ckl::Rsqrt<D0, ckl::VectorMode::Rows01>{}, ckl::TransposeDest<D0>{}
```

## 4. Agent prompts → `.claude/agents/*.md`

**4.1 Gates never re-run translated tests after a SUPPORTED widening, and the perf phase gave a known `supported_fail` no disposition.**
- **What:** Refinement 4 widened `SUPPORTED["fp32_dest_acc_en"]`, which silently promoted a translated cell into SUPPORTED. That cell fails. Nothing in that gate would have caught it:
  - the gate ran golden slices and the precision matrix only;
  - `golden_refinement_{1..5}` and `golden_perf_{1,2}` contain no `test_translated` results.

  Both perf rounds then saw the failure and filed it as "pre-existing". There was no EXCLUSION, follow-up refinement or `validate()` message, so it reached the blind pass.
- **Evidence:**
  - `changelog.md:530`: "It is **pre-existing**: it fails identically on HEAD with Perf 1 stashed".
  - `changelog.md:684`: the same failure recorded again in Perf 2.
  - `incremental-verifier.md` and `ttnn-implementer.md` never mention `test_translated`.
- **Recommendation:**
  - **`incremental-verifier.md`:** "When a refinement widens SUPPORTED, the gate re-runs `test_translated.py` and any newly-supported translated failure blocks the tick."
  - **`perf-coordinator.md`:** "A failing SUPPORTED cell is never 'pre-existing': file a `Refinement Nb` or an EXCLUSION, and record it in the result."
- **Confidence:** high.

**4.2 The planner prescribed a hardware-invalid CB tag across a DEST-width axis.**
- **What:** the Refinement 4 goal ordered "`cb_stats_slots` stays Float32 with `UnpackToDestFp32` … whatever the flag (prompt requirement)". The op prompt only asks that the partials stay fp32 *on the wire and when summing* (`.claude/eval/prompts/distributed_rms_norm.txt:203`). The planner turned that into a CB unpack tag under 16-bit DEST, which produces inf (see 3.1).
- **Evidence:** `op_requirements.md:135`; commit `8b2510ff643`: "UnpackToDestFp32 on cb_stats_slots only with 32-bit DEST (16-bit DEST + tag -> inf, measured)".
- **Recommendation:** add to `incremental-planner.md`: "For a refinement that adds a precision-axis value, do not prescribe per-CB unpack / DEST tags. State the numeric intent (e.g. 'fp32 sums') and defer the mechanism to `/numeric-formats-metal`." This depends on 3.1 also landing.
- **Confidence:** med.

**4.3 The Phase-0 baseline was certified with the harness's L1 capture disabled.**
- **What:**
  - The official Phase-0 golden pass had 569 `supported_fail`, all `L1ProfilingError peak_total=0`: graph capture could not see heterogeneous MeshWorkload CBs.
  - The verifier reran with `EVAL_CAPTURE_L1=0` and recorded `supported_fail: 0` as the baseline.
  - The real infrastructure gap then cost a debug refinement (1b). That refinement patched tt_metal C++ (`mesh_workload.cpp`, the program's CB tracking, `graph_trace_utils`) inside an op commit.
- **Evidence:** `golden_phase0/verifier_report.json` (supported_fail 569); `verification_report.md:143-172` (`EVAL_CAPTURE_L1=0 eval/eval_test_runner.sh …`); commit `b9603640a5c`.
- **Recommendation:**
  - **`incremental-verifier.md`:** "Never certify a baseline with harness instrumentation disabled. If the harness itself fails, file an infrastructure blocker or debug refinement immediately and report the official counts."
  - **For a human:** review and upstream the graph-capture fix from `b9603640a5c` separately, since it changes framework C++ that all fused-CCL ops share.
- **Confidence:** med.
