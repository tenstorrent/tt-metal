# Self-Reflection: conv2d_nhwc

## Summary
- Blind pass: **4578 passed, 0 failed, 0 hangs** (5712 total). Golden: 2414/2414 supported pass, 1104 INVALID skips. Translated: 2148 pass, **15 op refusals** (lenient xfail), and 15 skips carried over from the reference test (`test_conv_core_nondivis`: "New API needs to be tested"). Regression: 16/16.
- `verifier_report.json`: `supported_fail` 0, `xpass_drift` 0, **`supported_marked_xfail` 15**. All 9 refinements are `[x]`, and 2 perf rounds ran (SDXL L1 focus case 237.9k → 188.6k cycles).
- **Most important finding:** all 15 refusals are cells whose axes are all in SUPPORTED but that the op refuses because they do not fit in L1 (`input window exceeds L1 — … R10 not built`, `sharded K-exchange blocks do not fit L1`). No tagger captures that facet and no golden cell reaches it. The verifier prompt reads this category as "decoration drift", and the queue only accepts golden-driven refinements, so the gap was found in Refinement 2 but never queued.
- The problems are **framework-level** (axis model, queue rules, prompt wording, one reference doc). The kernels look correct everywhere they claim support.

## 1. Golden coverage → `eval/golden_tests/conv2d_nhwc/feature_spec.py`

**1a. Axis-blind gap: L1-fit / per-block input footprint (13 of 15 refusals)**
- **What:** 13 translated cells with `dtype=BFLOAT16, memory_layout=INTERLEAVED` (and every other axis in SUPPORTED) are refused because one output tile-row's input window exceeds the L1 hard cap. The distinguishing feature is `kernel_rows·W·C_in·elem_bytes` above about 1.2 MiB. The `test_conv2d_with_fold` cells fail at C_in=320 and pass at C_in=3 with the same k, s and W. A tile-row of 32 output pixels at s=16 spans 512 input columns, which is more than W, so the column clip cannot help. Golden's largest HEIGHT/INTERLEAVED footprint is about 0.58 MB, so golden never gets near the boundary.
- **Evidence:** `golden_blind_final/test_results.json`: `test_conv2d_with_fold[*-320-32-{224,512,208}-*-{16,32}-*]` ×10, `test_conv_dram__sanity[1-2904-2904-48-48-…]`, `test_ttnn_pytorch_sweep[input_spec166|294]`, each with "op refused … input window exceeds L1 — DRAM-slicing regime R10 not built" (`conv2d_nhwc_program_descriptor.py:554`). `changelog.md:91`: "Finding: R10 has no refinement in the queue yet."
- **Recommendation:** Propose a boundary `LOOSE_CASES` entry: `{"inputs": ((1, 208, 208, 320), 32, 16, 8, 16, 1, 1), "dtype": ttnn.bfloat16, "weight_dtype": ttnn.bfloat16, "layout": ttnn.ROW_MAJOR_LAYOUT, "bias_mode": "no_bias"}` (the smallest refused fold shape). Golden does not turn refusals into xfail (`conftest.py:126-135` only applies that to `translated_from`), so this would surface as `supported_fail`. Also consider a new `INPUT_TAGGERS` axis such as `l1_regime ∈ {fits, needs_dram_slicing}`, derived from `(k-1)·d+1`, W, C_in and dtype, and listed in TARGET. The R10 gap would then be an ordinary refinement candidate instead of an invisible one.
- **Confidence:** high

**1b. HEIGHT_SHARDED with a large per-core footprint (2 of 15 refusals)**
- **What:** 2 HEIGHT cells are refused with "sharded K-exchange blocks do not fit L1": vanilla-UNet 30×40, 256→512, 3×3 (bf16) and SD 2×64×64, 960→320, 1×1 (bf8b weights). Similar HEIGHT cells pass, e.g. 60×80, 288→288 at a 1.34 MB L1 peak. No single raw facet clearly separates the two groups. Golden HEIGHT tops out at C_in ≤ 320 and C_out ≤ 320, with peak ≤ 0.58 MB.
- **Evidence:** `test_conv_for_vanilla_unet[…-1-512-256-30-40-3-3-…-HEIGHT_SHARDED]`, `test_sd_conv_wh[…-2-320-960-64-64-1-1-…-HEIGHT_SHARDED]`, both raised at `conv2d_nhwc_program_descriptor.py:915`.
- **Recommendation:** Propose a `LOOSE_CASES` entry `((1, 30, 40, 256), 512, 3, 1, 1, 1, 1, ttnn.TensorMemoryLayout.HEIGHT_SHARDED)` with bf16/bf16 and bias. It is model-derived and is the smallest refused HEIGHT shape. The same `l1_regime` tagger idea applies.
- **Confidence:** medium (the exact discriminating facet is unclear)

**1c. The declared EXCLUSION `{layout: TILE, groups_mode: depthwise}` is never exercised**
- **What:** TILE×depthwise is in TARGET and in EXCLUSIONS, but no golden cell hits it (`xfail_expected` = 0). Nobody can check whether the exclusion is honest, and it produced no refinement signal.
- **Evidence:** `conv2d_nhwc.py:134-136`: "No golden TILE cell is depthwise." `verifier_report.json`: `xfail_expected: 0`.
- **Recommendation:** Propose `LOOSE_CASES` entry `((1, 28, 28, 32), 32, 3, 1, 1, 32, 1, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TILE_LAYOUT)` with bf16/bf16.
- **Confidence:** high

## 2. SUPPORTED honesty → `conv2d_nhwc.py` `SUPPORTED` / `EXCLUSIONS`
- `supported_fail` 0 and `xpass_drift` 0: every axis value the op declares holds on the golden universe.
- **2a. The 15 `supported_marked_xfail` cells are a size-facet over-claim, not decoration drift.** 13 are INTERLEAVED windows over the L1 cap (cluster 1a) and 2 are HEIGHT-sharded fit failures (cluster 1b). The axis registry cannot express "supported except above an L1 bound", so the descriptor raises `UnsupportedAxisValue` for a non-axis reason (`conv2d_nhwc_program_descriptor.py:29`, `:554`, `:915`). **Recommend fix, not demote**: demoting a whole axis value would be wrong because thousands of cells pass. Concretely, queue R10 (DRAM output-row slicing / direct DRAM im2col gather, already a `deferred` row at `op_design.md:188`) as a refinement, gated by the new tagger from 1a. Confidence: high.
- **2b. Minor wording issue.** The `:915` refusal says "K-exchange" but also fires on HEIGHT (S=1, no exchange). Consider rewording it to "sharded blocks do not fit L1". Confidence: high.

## 3. Helper / reference docs

**3a. `numerical_stability_analysis_reference.md` says packer L1 accumulation "does not work" with fp32 DEST. This run contradicts that.**
- **What:** The reference gives a blanket "do not use it" with no conditions. This op uses `matmul_block<packer_l1_acc=true, Interm>` with `fp32_dest_acc_en=True` and Float32 `cb_partials` on every path. It passes 2414 golden cells, with fp32 PCC ≥ 0.999997 (Phase 0) and bf16 rel-RMS ≤ 0.009.
- **Evidence:** `.claude/references/numerical_stability_analysis_reference.md:126`: "Packer L1 accumulation does not work with `fp32_dest_acc_en=true`". `op_design.md:257,378`; `changelog.md:24`.
- **Recommendation:** Scope that line: say it applies when the L1 accumulation target is not Float32 (or name the arch: this run was Blackhole), and say that a Float32 accumulation CB with fp32 DEST was validated here. Today the reference pushes planners away from a proven, 5× faster pattern (`op_design.md:246`).
- **Confidence:** medium (the failure mode in issue #28800 may be format- or arch-specific; validated on BH only)

**3b. `matmul_block` leaves packer L1-acc on when it returns, and the header does not say so**
- **What:** With `packer_l1_acc=true` and an `Interm` target, the helper leaves `llk_pack_reconfig_l1_acc(1)` set after K-block ≥ 1. It is also still set when a `PreKBlockFn` runs. Callers must reset it with raw LLK. The `.hpp` restore contract only covers unpack/math/pack *format*. The planner had to find this in the `.inl`, and the kernel carries 3 raw calls.
- **Evidence:** `matmul_block_helpers.inl:410-414` (`block == 0 ? 0 : 1`), with no reset on exit. `matmul_block_helpers.hpp:182-193` (restore contract, silent on L1-acc). `conv2d_nhwc_compute.cpp:24,132,148,400`. `op_design.md:343`: "leaves `llk_pack_reconfig_l1_acc(1)` set from the previous K-block".
- **Recommendation:** Either have the helper reset L1-acc to 0 before `PreKBlockFn` and on return (preferred; this removes the raw calls), or add one line to the hpp contract: "On return, and inside PreKBlockFn, packer L1-acc is left ON for K-block ≥ 1; caller must `llk_pack_reconfig_l1_acc(0)` before any other pack."
- **Confidence:** high

#### Helper gaps (perf)
| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `noc_async_read_with_state` (dataflow_api, two cmd buffers) | capability | **missing**, but **~equal ns**: 171.2 vs 169.9 µs (−0.8%, inside the stated ±4% noise) | `dataflow_api.h:677-699` hard-codes `read_cmd_buf`. Raw NOC register writes at `conv2d_nhwc_gather.hpp:167-189`. It cost a hang carve-out under `WR_WEIGHTS` (`changelog.md:412`, trid barrier blind to the second buffer). On the focus case "only the padding part … remains active" (`changelog.md:413`). | The raw dual-buffer part bought no measurable speed and added a hang hazard. Consider reverting that half to the stateful API (keep the padding fast path). Separately, a `cmd_buf` parameter on `set_state`/`with_state` plus trid coverage would make the pattern safe if a later measurement justifies it. |
| Mcast `SenderPipe` / `ReceiverPipe` (non-blocking readiness) | capability | **missing** | `mcast_pipe.hpp:83-101` (sender: only blocking `send`/`send_signal`). `:157-185` (receiver: blocking `acknowledge`/`wait_delivery`/`receive`). The kernel reaches into `dataflow_kernel_lib::detail::make_mcast_semaphore` (`conv2d_nhwc_reader.cpp:90-91`) and peeks at `:136-140`, `:176`. 170.1 → 159.5 µs (−6%). | Add public non-blocking queries (sketch below). This also removes the `detail::` dependency. |
| CB co-producer (two RISCs fill one CB) | capability | **missing** | No kernel_lib dataflow CB helper supports split reserve/fill/push across RISCs. Raw `noc_semaphore_set/wait_min` plus a 64 B mailbox CB at `conv2d_nhwc_gather.hpp:1011,1085`. 146.9 → 139.7 µs (−5%). | Single site. Consider a "reserve here, fill elsewhere, push here" primitive only if a second op needs it (confidence: low). |
| `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` | — (**unrecorded**) | **bypassed although the helper exists** | Commit `624cc0c529c` uses raw `noc_async_read_set_trid` / `noc_async_read_barrier_with_trid` at `conv2d_nhwc_reader.cpp:162,167` and `conv2d_nhwc_gather.hpp:706,1062`. The writer uses the helpers for the same pattern (`conv2d_nhwc_writer.cpp:293,315,419,441`; helper at `local_copy_helpers_dataflow.hpp:147,158`). No row in either `### Helper bypasses` table. | Switch the 4 raw calls to the helpers. Record any remaining bypass. |
| trid "flushed" query (non-blocking) | — (**unrecorded**) | **missing** | `conv2d_nhwc_reader.cpp:174` calls the internal `ncrisc_noc_read_with_transaction_id_flushed`. kernel_lib offers only the blocking `async_read_barrier_with_trid` (`local_copy_helpers_dataflow.hpp:158`). | Add `bool read_trid_flushed(Noc, trid)` next to the barrier (sketch below). |

Also from a non-graduated perf idea (no bypass shipped): `matmul_block` reprograms the pack data format and L1-acc per subblock, about 49k of 150k cycles on the compute-only bench (`changelog.md:547`; patch in `perf_experiments/matmul_kblock_cost/kernel_lib_hoist.patch`). This is a library-side win worth reviewing on its own.

```cpp
// Mcast pipes — non-blocking readiness (derived from conv2d_nhwc_reader.cpp:136-180)
bool ReceiverPipe::delivered(uint32_t round) const;   // true ⇒ wait_delivery(round) returns immediately
bool SenderPipe::consumers_ready() const;             // true ⇒ send() will not block on CONSUMER_READY
// call site, polled from inside the gather loop:
if (w_rx.delivered(g)) { w_rx.wait_delivery(g); cb_push_back(cb_weight_tiles, n); }
if (owns(g_push) && read_trid_flushed(noc, trid(g_push)) && w_tx.consumers_ready()) w_tx.send(src, dst, bytes);

// trid — non-blocking (derived from conv2d_nhwc_reader.cpp:174)
bool dataflow_kernel_lib::read_trid_flushed(Noc noc, uint32_t trid);
```

## 4. Agent prompts

**4a. The verifier misreads `supported_marked_xfail`**
- **What:** The prompt says this category means "decoration drift — re-collect tests". Here it meant real op refusals on SUPPORTED cells, caught by `conftest.py`'s lenient-xfail hook. Nobody acted on the 15 cells, and the category's code comment (`eval/verify_supported.py:211-218`) has the same assumption.
- **Evidence:** `.claude/agents/incremental-verifier.md:218`: "Decoration drift — out-of-sync state. Re-collect tests after fixing." Blind messages all start with "op refused (SUPPORTED/EXCLUSIONS)".
- **Recommendation:** Add a row: "If the message is `op refused …` from the descriptor (not `validate()`), treat it as an over-claim on a non-axis facet: file a refinement or propose a tagger axis."
- **Confidence:** high

**4b. The queue can only be fed from golden, so gaps found in translated tests are dropped**
- **What:** The implementer found the R10 gap in Refinement 2 ("R10 has no refinement in the queue yet") but could only record it. The verifier's queue accepts only changes "that unlock more golden-test cells". The planner's "no golden shape reaches R10" claim (`op_design.md:127,188`) was wrong twice: 9 translated fp32 failures at Phase 0, then 15 refusals in the blind pass.
- **Evidence:** `incremental-verifier.md:278`, `changelog.md:27,91`, `verification_report.md:52`.
- **Recommendation:** Planner prompt: check reachability claims for `deferred` regimes against `test_translated.py` shapes too, not just golden. Verifier prompt: allow a refinement whose target is translated `supported_marked_xfail` cells, or require that such a gap come with a proposed tagger axis.
- **Confidence:** medium

**4c. The verifier prompt points to the wrong import path**
- **What:** The precision-baseline step tells the agent to import `comp_allclose` from `models.common.utility_functions`. The function lives in `models.common.metrics`.
- **Evidence:** `incremental-verifier.md:236`. Verifier friction breadcrumb (2026-11-02T09:13). `models/common/metrics.py:165`; not defined in `utility_functions.py`.
- **Recommendation:** Change the path to `models.common.metrics`.
- **Confidence:** high

**4d. The perf coordinator requires a header that is not in the base tree**
- **What:** The prompt requires `kernel_lib/perf_instrumentation.hpp`. This run had to recreate it ("did not exist in this tree"), as several earlier runs in git history did.
- **Evidence:** `perf-coordinator.md:49-50`, `changelog.md:360`, commits `ed24abbe3ec`, `9e3a5647fe5`, `77a477a74ff` ("restored kernel_lib/perf_instrumentation.hpp").
- **Recommendation:** Commit the header to the base branch, or have the prompt give its full contents for creation when it is missing.
- **Confidence:** high
