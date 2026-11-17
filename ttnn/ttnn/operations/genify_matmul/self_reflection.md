# Self-Reflection: genify_matmul

## Summary
- **Blind pass:** golden is **1512 passed / 375 xfail / 0 failed**. Translated is 197 passed / 80 xfail / 20 skipped / **19 failed**. Regression is 10/10.
  - SUPPORTED equals TARGET.
  - Both EXCLUSIONs are permanent and justified: fp32 × 16-bit DEST, and dram_sharded × batched.
- **None of the 19 failures is an op failure.** Every one is a harness or test-translation artifact. The 16+2 program-cache-count cells still record PCC ≥ 0.99985.
- **Most important finding (framework):** the graded run's L1 capture (`eval/l1_profiling.py:46`) disables and clears the program cache around every test call.
  - Any translated `num_program_cache_entries()` delta assertion therefore reads `0 - 0`. That accounts for 18 of the 19 failures.
  - It also makes `test_program_cache_reuses_the_program` pass vacuously.
- **The run itself was clean.** All 11 refinements were checked `[x]`. Golden showed zero regressions across all 15 phase snapshots and 2 perf rounds.
- **Remaining feedback** is on helpers and references:
  - `matmul_block` lacks a per-K-block A row stride, and its LLK writes DEST per tile.
  - kernel_lib has no fused GLU element.
  - The runtime L1 CB budget recipe is missing from the references.

## 1. Golden coverage (feature_spec.py / test universe)

**No in-TARGET golden gap found.** Every golden cell in SUPPORTED passes. The 80 translated xfails sit on the out-of-TARGET axis values `k_non_aligned`, `n_non_aligned` and `weight_dtype=float32` (`verification_report.md:210-212`), so they are correct to fail. No `LOOSE_CASES` proposal.

**F1.1 — Translated program-cache tests are wrong under the eval harness (test universe, not feature_spec).**
- **What:** 18 failures form one cluster. The commonality is the assertion `num_program_cache_entries() - n0 == 1`. Two independent mechanisms break it:
  - **(a) Graded run.** `start_for_item` calls `device.disable_and_clear_program_cache()` before the test body, so `n0 = 0` and the post-call count is also 0. This breaks all 16 `test_matmul_1d_multiple_output_blocks_per_core` cells and both `test_matmul_2d_multiple_output_blocks_per_core` cells.
  - **(b) Agent runs, without capture.** The 16 parametrizations of the 1d test differ only in reference-op knobs that genify_matmul ignores (`num_out_block_h/w`, `uneven_width`). They collapse to 2 programs, so 14 cells hit the cache that earlier cells filled.
- **Evidence:**
  - `golden_blind_final/test_results.json`: `assert (0 - 0) == 1` on all 18. The stdout logs `Disabling and clearing program cache` at each test start.
  - `changelog.md:707-709`: "parametrizations differ only in reference-op knobs that genify_matmul ignores, so later cases hit the program cache".
  - `test_translated.py:2046-2048`: `test_program_cache_reuses_the_program` asserts `== entries` and passes as `0 == 0`.
- **Recommendation:**
  - (a) Propose an opt-out marker, e.g. `@pytest.mark.no_l1_capture`, honoured by `eval/l1_profiling._device_from_item`. Apply it to every test that reads `num_program_cache_entries`.
  - (b) Rewrite the "same config reuses" form as: warm call, then `n0`, then a repeat call, then `assert delta == 0`. It is then robust to programs already cached by sibling parametrizations.
- **Confidence:** high.

**F1.2 — Expected-error test fails because of L1 capture (singleton, harness).**
- **What:** `test_matmul_same_shape_but_invalid` raises the expected `ValueError` before anything is allocated. Capture then raises `L1ProfilingError: peak_total=0`.
- **Evidence:**
  - `eval/l1_profiling.py:84-88`.
  - Verifier breadcrumb `incremental-verifier_breadcrumbs.jsonl:2`, filed at Phase 0 (02:52) and still unfixed at blind.
- **Recommendation:** in `capture_for_item`, skip the peak (do not raise) when `peak_total == 0` and the test passed with no dispatch.
- **Confidence:** high.

**F1.3 (observation, no action) — kernel path is axis-blind.**
- Regime choice (R1/R1a/R1b, raw vs helper K-walk, head ramp, tail split, R11 none-raw-walk) is made by a cost model from shape. No tagger captures it.
- The golden `INPUTS` are all small (largest 384×1536×1024). Path coverage therefore rests on the `LOOSE_CASES` model shapes. Golden is green, so this is not a gap today.
- **Consider:** let the op report its chosen regime/walk into test extras, so path coverage becomes checkable.
- **Confidence:** low.

## 2. SUPPORTED honesty

- **Counts** (`golden_blind_final/verifier_report.json`): supported_pass 1709, **supported_fail 19**, xfail_expected 455, xpass_drift 0, xfail_wrong_mode 0.
- **The 19 `supported_fail` → neither fix nor demote.**
  - 18 are the program-cache cluster (F1.1). They tag as interleaved / none / bf8b weight / tile_aligned.
  - 1 is the expected-error capture artifact (F1.2).
  - In every case the op produced correct numerics or raised correctly. Demoting `interleaved × BFLOAT8_B` would be dishonest: hundreds of golden cells with that tuple pass.
  - **Recommend:** fix the harness and translation (F1.1, F1.2). After that, the verify_supported over-claim count should drop to 0.
  - **Confidence:** high.
- **No `xpass_drift`.** Both EXCLUSIONs are documented as permanent (`genify_matmul.py:117-122`), and their 375 golden cells xfail as expected. No promote recommendation.
- **Process note.** The Phase-0 verifier correctly refused to silence F1.2 with an EXCLUSION (`verification_report.md:214-221`). That is the right behaviour, and it is why the category stays loud until the harness is fixed.

## 3. Helper / reference docs

**F3.1 — The reference has no recipe for the runtime L1 CB budget (absence).**
- **What:** the design prescribed `ttnn.get_max_worker_l1_unreserved_size() − tensors`, which overstates the CB region by about 70 KB on Blackhole.
  - The implementer's replacement, `total_bytes_per_bank − allocated`, ignores fragmentation.
  - Both versions TT_THREW (CB/L1 clash) before the verifier fixed it with the lowest-allocated-block rule.
- **Evidence:**
  - `op_design.md:155`.
  - `ttnn-implementer_breadcrumbs.jsonl:6`.
  - `incremental-verifier_breadcrumbs.jsonl:1`: "No reference says how to compute the per-core CB budget at runtime".
  - `grep` finds no `memory_view`, `block_table` or `lowest` in `.claude/references/l1-footprint-discipline.md`.
- **Recommendation:** add one line to `l1-footprint-discipline.md`:
  - CB budget = `min(lowest allocated L1 block offset from get_memory_view(device, L1).block_table, bank − own L1 tensors) − reserve`.
  - Never use `get_max_worker_l1_unreserved_size()`, which counts from below the CB base.
  - Alternatively, bind `lowest_occupied_compute_l1_address`.
- **Confidence:** high. Two agents independently hit it.

**F3.2 — The `matmul_block` precision note is silent on the 16-bit-DEST mitigation (absence).**
- **What:** `matmul_block_helpers.hpp:260-261` only says to use fp32 DEST when Kt > 1. TARGET requires `fp32_dest_acc_en=False`.
  - The run found that error grows with the in-DEST chain length: rel-RMS 0.185 at K=16k.
  - Capping `block_k ≤ 8` and accumulating across K-blocks with packer L1 accumulation brings it back to stock level.
- **Evidence:**
  - `ttnn-implementer_breadcrumbs.jsonl:9-11`.
  - `changelog.md:56` (`DEST16_BLOCK_K_MAX = 8`).
- **Recommendation:** add to the Precision block: "with 16-bit DEST and Kt > 1, cap in0_block_k (≈8) and use packer_l1_acc; error tracks the in-DEST chain, not K".
- **Confidence:** med.

#### Helper gaps (perf)

Two perf rounds ran. The matmul_block rows are clustered per helper.

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `compute_kernel_lib::matmul_block`: per-tile DEST offset write (Perf 1, part 1) | capability | **missing** (below the helper: LLK) | `tt_llk_blackhole/llk_lib/llk_math_matmul.h:814` calls `set_dst_write_addr` inside the per-tile loop. The helper (`matmul_block_helpers.inl:361`) can only call `ckernel::matmul_block`. Measured 353.9 → 330.6 µs isolated, bit-identical. | Add an LLK multi-tile pass that writes DEST once and advances it by address modifier; expose it through the helper. |
| `compute_kernel_lib::matmul_block`: per-K-block partials drain on UNPACK (Perf 1, part 2) | capability | **present and documented**, so not a capability gap. Production does not adopt it, which suggests `too-hard`. | Documented at `matmul_block_helpers.hpp:27-28,293`. The row itself says `caller_owns_pack_target` "fixes (2) alone (−3.9 µs)". The surviving helper call for the carve-outs (`genify_matmul_compute.cpp:194-207`, Interm, sb_h>1 / residency) leaves it at its default `false`, because it is the 15th positional template argument (`hpp:354`). | Op side: enable it on the Interm carve-out call. Helper side: named options instead of 18 positional template parameters (sketch B). Confidence: low on the "too-hard" reading, since no breadcrumb says it was hard. |
| `compute_kernel_lib::matmul_block`: per-K-block A row stride (Perf 2; recorded as prose, **not a row**) | capability | **missing** | `matmul_block_helpers.inl:193` computes `in0_subblock_num_tiles = out_subblock_h * in0_block_k`, and `:498` advances by it. `hpp:208` says "kt_dim stays block_k". `KBlockInnerDimFn` shrinks only the loop bound. Cost: the head ramp is carved out of the R1a helper walk, measured flat to +2.3% there (`perf-part-optimizer_breadcrumbs.jsonl:24`). | Extend the per-K-block functor to return the A row stride as well (sketch A). Also record it as a table row. |
| `compute_kernel_lib::matmul_block`, `LastBlockTarget::Out` (Refinement 11 none-raw-walk) | **no row** | **unrecorded bypass extension** | `genify_matmul_compute.cpp:177-182` and `changelog.md:989`: `kw::matmul_accumulate_interm` plus an identity copy replaces `matmul_block<Out>`. The Perf 1 row still says the epilogue-none path "stays on the helper". | Add or update a bypass row whenever a refinement widens a raw-LLK path. The root cause is the same SETC16 gap as row 1. |
| eltwise_chain: fused `silu(a)·b` (Perf 1) | capability | **missing** | `kernel_lib/eltwise` has only `Sigmoid` (`unary/activations.hpp:24`) and `MulBinary` (`binary/sfpu/basic.hpp:28`); no GLU/SwiGLU element. 85,317 → 61,810 ns. | Add a fused GLU binary element (`SiluMul<gate_slot, up_slot>`) to eltwise. |
| eltwise_chain / SFPI: SFPLOADMACRO scheduling (Perf 2) | capability | **missing** (toolchain/LLK, not the helper) | `genify_matmul_swiglu_sfpu.hpp:19-22`. Record-quality issue: the row's "helper ns" of 61,827 is the round-1 *raw* element, not a helper path. The true helper baseline is 85,317 (`changelog.md:717`). | SFPI/LLK roadmap item. Ask perf rows to always give the stock-helper ns. |

- **No `ergonomics` row with ~equal ns.** Both recorded kinds are `capability` with large wins.
- **Raw-LLK grep:** `genify_matmul_compute.cpp` itself has no raw `TTI_`/`llk_` calls. Raw calls appear only in `genify_matmul_kwalk.hpp` and `genify_matmul_swiglu_sfpu.hpp`, and both are recorded (the R11 extension is the exception above).

Sketch A — per-K-block A stride. Confidence: low; derived from `genify_matmul_kwalk.hpp`, where `a_stride = compact_a ? inner : block_k`.
```cpp
struct KBlockGeom { uint32_t inner_steps; uint32_t in0_row_stride; };
template <..., typename KBlockGeomFn = DefaultKBlockGeom /* {block_k, block_k} */, ...>
void matmul_block(..., KBlockGeomFn k_geom = {});
// call site (head ramp):
matmul_block<...>(act, w, out, partials, shape, ..., HeadRamp{});  // returns {kv(kb), compact ? kv(kb) : block_k}
```
Sketch B — named options. Confidence: low.
```cpp
constexpr MatmulOpts kOpts{.packer_l1_acc = true, .target = LastBlockTarget::Interm,
                           .order = OutputCBLayout::TileRowMajor, .caller_owns_pack_target = true};
matmul_block<kOpts>(act, w, out, partials, shape, Hooks{.post_k_block = RetireOwnRoundSlot{kb}});
```

## 4. Agent prompts

**F4.1 — `translate-reference-tests` rule C prescribes a delta form that is fragile under the eval harness.**
- **What:** `.claude/skills/translate-reference-tests/SKILL.md:467-481` prescribes `n0 = ...; <run once or twice>; assert delta == N`. This is the exact pattern in the 18 failures. It assumes:
  - no sibling parametrization has already cached the program (false whenever the agentic op ignores reference knobs);
  - the program cache is enabled (false under graded L1 capture).
- **Evidence:** F1.1; `test_translated.py:841-848`, `:1756-1775`.
- **Recommendation:**
  - Amend rule C. For "same config reuses", do a warm call, take `n0`, repeat the call, and assert `delta == 0`.
  - For "distinct configs → distinct programs", assert per-config deltas ≥ 1 only on configs that are distinct after projecting through the agentic op's signature.
  - Mark cache-asserting tests with the L1-capture opt-out.
- **Confidence:** high.

**F4.2 — Known non-op failures went 4 phases without a friction breadcrumb.**
- **What:** Perf 1, Perf 2, Refinement 10 and Refinement 11 each re-reported "14 pre-existing program-cache failures". None filed a `friction` event against the test or harness. The only trace is changelog prose and a coordinator `round_complete` string. Without a breadcrumb, the defect never reached a framework owner.
- **Evidence:**
  - `changelog.md:707`, `:879`, `:996`.
  - `perf-coordinator_breadcrumbs.jsonl:7,13`: status strings only, no `friction` event.
- **Recommendation:** in `perf-coordinator.md` and `ttnn-implementer.md`, add: "a failure you classify as pre-existing / not-op-caused must also be logged once as a `friction` breadcrumb with `surface: test|harness` and the nodeid".
- **Confidence:** med.

**F4.3 — Helper-bypass record scope.**
- **What:** the `### Helper bypasses` schema lives only in `perf-coordinator.md:231-272`. Refinements that widen a bypass (R11, F3 table row 4) have no obligation to update it. Perf 2 also recorded a capability gap as a prose bullet instead of a row (`changelog.md:887`).
- **Recommendation:**
  - In `ttnn-implementer.md`, require a bypass row whenever a refinement adds or extends a raw-LLK path.
  - In `perf-coordinator.md`, state that every capability gap blocking a graduation (including carve-outs) gets its own row.
- **Confidence:** med.
