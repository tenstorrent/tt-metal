# Self-Reflection: gen_qkv_causal_conv1d_silu

_Advisory only. Nothing here is applied automatically; every item is a proposal for a human to accept or reject._

## Summary

- **Final blind pass: 21/21 golden; no translated cells.** `verify_supported` (re-run) gives 21 `supported_pass` and 0 for every signal category.
  - No `test_translated.py` exists for this op, so the "blind" pass re-ran the 21 golden shapes on fresh inputs.
  - The trajectory was Phase 0 17 + 4 xfail → R1 21/21, which held through R2, R3, Perf 1, Perf 2 and the blind pass. All 3 refinements are `[x]`.
- **Perf.** Two perf rounds, one graduation: a hand-scheduled raw-TTI SiLU body.
  - conv1d-004 is at 124 µs against a stretch of 117 µs; its 164 µs target is met.
  - The lever that met the stretch, cross-core work stealing, was blocked by an intermittent device hang.
- **Most important finding.** The graduated SiLU body uses `SFPLOADMACRO` with no `DISABLE_SFPLOADMACRO` fallback.
  - Default plans run it on 17 of the 21 golden cells.
  - Simulator runs set that switch, so those cells are likely to break in sim (§3 D2).
  - The portable sfpi body that would have been the fallback was deleted under the coordinator's "spread and delete" rule.
- **Op level or framework level?** The issues look framework-level: a reference with no request-count rule, a verifier prompt without the "balanced is not bound" rule, and perf tooling that the prompts prescribe but that is missing or broken in the tree. No op-correctness failure was observed.

## 1. Golden coverage → `eval/golden_tests/gen_qkv_causal_conv1d_silu/feature_spec.py`

There are no blind failures to cluster. The findings below are in-TARGET regions that no golden cell exercises (absence).

**G1. Axis-blind gap: the graduated raw-TTI SiLU body at full DEST-group size has no golden cell.**
- **What.** Which body runs, and how large its DEST group is, depends on the plan: tile-rows per core and `out_group_tiles`. No tagger captures either.
  - Realistic prefill shapes run `silu_bf16::rows` with 5- or 8-tile groups, 3 rows per core (offline `select_plan` on BH 11×10):
    - `(1,1024,2560)`: og=8;
    - `(1,512,2560)`: og=5, in both RM and TILE.
  - Golden's only multi-row cell on this body is conv1d-008 (og=2, RM). No golden cell covers TILE + 16-bit DEST + multi-row.
  - `(1,1024,2560)` has exactly the axes tuple of conv1d-004, so it looks "covered" while running a different kernel path (004 keeps the library body).
  - This body keeps state per DEST group (`group_begin`/`group_end`, `compute.cpp:155-161`, `:298-306`) and per block (borrows L11, `restore_neg1` at `:148`, `:414`). That state is exactly what grows with group size and block count.
- **Evidence.** `default_silu_body` uses the library body only at ≥ 5 rows per core (`program_descriptor.py:106,340`).
  - The only full-group coverage is the op's own *pinned* unit test (`test_silu_body.py:98` `conv1d-020_multi_row`), which agents can edit.
- **Recommendation.** Add two LOOSE_CASES through `make_cases.py` (the file says to regenerate rather than edit). Both are all-DRAM with default compute (HiFi4, `fp32_dest_acc_en=False`):
  - `((1, 1024, 2560), (512, 512, 1536), 256, None, 0)`, layout RM;
  - `((1, 512, 2560), (512, 512, 1536), 256, None, 0)`, layout TILE.
  - Do **not** promote this to an axis. The facet depends on the plan and the device grid, not on the inputs, so a tagger cannot compute it portably.
- **Confidence.** Med. The path is validated under a pin, but no immutable test runs it on a default plan.

**G2. There is no held-out suite: the blind pass contributes no generalization signal.**
- **What.** `run_refinements.py:338-340` only adds the translated cells when `test_translated.py` exists. It does not exist for this op, so the blind pass is the golden set again.
- **Evidence.** `golden_blind_final/test_results.json` holds 21 nodeids, all from `test_golden.py::test_op_loose`.
  - `native.md` cites the native nightly test, which has three production cases at T=64: 512³ chunk 1536, 1024³ chunk 768, and (512,256,128) chunk 896.
- **Recommendation.** Run `/translate-reference-tests` on `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_qkv_causal_conv1d_silu.py`.
  - 1024³ chunk 768 (equal widths, straddling blocks, C=3072) is not a golden cell today.
- **Confidence.** Med.

**G3. A W that is not a multiple of 32 is supported natively but only tested by the implementer.**
- **What.** R1 made the TILE page stride `ceil(W/32)`, which "supports" a W that is not a multiple of 32 (`changelog.md:37`). That case is in TARGET: `native.md` requires only `W ≥ offset + C`.
  - The `input_window` tagger folds it into `column_window`. Every golden W is a multiple of 32.
- **Evidence.** The implementer's `test_reader_addressing.py` uses W=2200 (`changelog.md:49`). No golden case does.
- **Recommendation.**
  - Add a LOOSE_CASE: `((1, 64, 2200), (1024, 512, 512), 64, None, 32)`, layout TILE, with in/history in DRAM.
  - Optionally add an `input_window` value for an unaligned width (a tagger can compute that from the inputs).
- **Confidence.** Low–med.

## 2. SUPPORTED honesty → `SUPPORTED` / `EXCLUSIONS`

No evidence-backed finding.
- Blind pass: 21 `supported_pass`, 0 `supported_fail`, 0 `xpass_drift`, 0 `xfail_wrong_mode`; the re-run of `verify_supported` agrees.
- `SUPPORTED == TARGET`, and `EXCLUSIONS = []`.
- The claim covers about 11.5k legal cells with 21 tests. G1 is where it is least evidenced.

## 3. Helper / reference docs

**D1. `blocking-model.md` ranks splits by bytes only; nothing says to count DRAM requests.**
- **What.**
  - **The rule:** §4 gives the split ranking's unit as "bytes crossing DRAM / NoC" (`.claude/references/blocking-model.md:233`). "request" appears 0 times in `blocking-model.md`, `op-design-template.md` and `incremental-planner.md`.
  - **What the run measured:** it hit request-rate-bound DRAM twice.
- **Evidence.**
  - **Planner:** it deferred `tap_multicast_per_group` as "≤ 1.1% of the DRAM bytes" (`op_design.md:143`).
  - **Implementer, Phase 0:** about 14k tiny tap reads stalled readers by 3–80 µs. Fixing that was the largest single win on 004, 185.9 → 152.9 µs (implementer friction breadcrumb, 19:03).
  - **R2:** decode is "request-rate-bound … ~12.5 ns per request per bank port". `select_plan`'s makespan ignores per-request cost (`op_requirements.md` R2). The split reader then cut 001–003 by 24–30%.
- **Recommendation.** Add a third unit to §4: requests per DRAM bank port. Count sub-64 B reads per core for reuse-shared operands and decode-size sticks, and treat "cores × tiny-request fan-in" as critical-path unless measured otherwise.
- **Confidence.** High.

**D2. The graduated `SFPLOADMACRO` body has no `DISABLE_SFPLOADMACRO` fallback, and no doc states the convention.**
- **What.** `silu_bf16::rows` issues `TTI_SFPLOADMACRO` unconditionally (`compute.cpp:176-177`, `:205-206`).
  - **The convention:** LLK bodies that use load macros keep an sfpi path under `#ifdef DISABLE_SFPLOADMACRO` (e.g. `ckernel_sfpu_exp.h:726`; 10 files in `llk_sfpu/`).
  - **Who sets the switch:** `TT_METAL_DISABLE_SFPLOADMACRO=1` becomes `-DDISABLE_SFPLOADMACRO` (`jit_build/build.cpp:521`), and sim runs export it (`scripts/run_safe_pytest.sh:184`, `.claude/sim_env.py:95`).
  - **What was lost:** Perf 2 deleted R3's sfpi quartic (`changelog.md:422`), which was the natural fallback.
  - **Docs:** no `.claude` reference or perf prompt mentions the convention.
  - The block also has no arch guard, although the kernel head calls its mechanisms Blackhole-specific (`compute.cpp:42`).
- **Evidence.** The default plan selects this body on 17 of 21 golden cells (offline `select_plan`).
- **Recommendation.**
  - **Op:** restore the sfpi quartic under `#ifdef DISABLE_SFPLOADMACRO`.
  - **Docs:** add one line to `perf-part-optimizer.md` rule 4 and `perf-coordinator.md:203`: "a raw SFPU body using SFPLOADMACRO keeps an sfpi path under `#ifdef DISABLE_SFPLOADMACRO` (LLK convention; sim mode sets it). The portable path it replaces is that fallback, so do not delete it."
- **Confidence.** Med. This was not run under sim; the hazard is read from the code.

**D3. The perf-measure ablation rules do not say *how* to gate a variant; a source-edit ablation poisoned the JIT cache.**
- **What.** `.claude/skills/perf-measure/SKILL.md:110-125` says "Gate the payload behind an off flag", and `:277` says "measure a fresh kernel build". Neither names `KernelDescriptor.defines` or forbids editing and then restoring kernel sources.
- **Evidence.** The verifier's friction breadcrumb (cost: high) and `verification_report.md:166` record it:
  - after an edited source was restored, stale ablated binaries kept running;
  - the outputs were NaN, with compute-only timings (2.9 / 6.1 µs);
  - the dependency check did not invalidate them (`tt_metal/jit_build/depend.cpp:239`).
  - The lesson survived only through the R2 verifier note; Perf 1 then used define-gated `GQKV_ABLATE_*`.
- **Recommendation.** Add one rule: "inject ablation flags through `KernelDescriptor.defines` (part of the kernel hash). Never edit and restore kernel sources; if you must, delete `built/tt-metal-cache*/kernels/<op>*` and re-run a correctness-checked case."
- **Confidence.** High.

**D4. There is no supported atomic fetch-and-add-with-return for worker kernels; this blocked the lever that met 004's stretch.**
- **What.** Work stealing reached 112.4–117.2 µs on 004. It hung in 7 of 16 suite runs: the INCR_GET return value never landed (`changelog.md` Perf 2 W).
  - The public `noc_semaphore_inc` returns void and points the return at a scratch address (`dataflow_api.h:2445-2454`).
  - The internal `noc_fast_atomic_increment<…, program_ret_addr=true>` writes only `NOC_RET_ADDR_LO` of an XY address (`noc_nonblocking_api.h:1283`). On BH the X/Y coordinates sit above bit 36 (`noc_parameters.h:221`), so they are not in what it writes.
  - The coordinator's variant that programmed the full RET_ADDR still hung 2 of 4 times, so the root cause is open.
- **Recommendation.** Add a reference note:
  - "no documented fetch-and-add-with-return on BH; claim work through semaphore-acked mailbox writes";
  - track either a public `noc_atomic_fetch_add` API or the changelog's minimal claim-only stress bench.
- **Confidence.** Med that the API is absent; low on the root cause.

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `compute_kernel_lib::apply_activation_from_pack` / `ActivationApplyHelper<SILU>` | capability | **missing** | `sfpu_activation_helpers.hpp:158-172`: the loop is hard-wired to `ActivationApplyHelper<ACT,…>::apply(i)`. `ACT` is a closed enum (`:112-117`). There is no functor slot and no per-group hook, and no `.inl` exists, so the `.hpp` is the whole contract. The mirror `silu_from_pack` (`compute.cpp:281`) is the helper's sequence plus `group_begin`/`group_end`. 78.3 → 42.1 µs per pass. | An overload taking a caller SFPU functor plus optional per-DEST-group begin/end hooks, with a matching custom-init entry. |
| LLK SFPU SiLU (`ckernel::sfpu::calculate_silu`) | capability | **missing** | `ckernel_sfpu_silu.h` is sfpi-only (46 lines, 0 `SFPLOADMACRO`). Sibling bodies carry TTI load-macro paths (`ckernel_sfpu_exp.h` 39 uses, `ckernel_sfpu_recip.h` 15). 61.5 → 42.1 µs. | Upstream `silu_bf16::rows` (`compute.cpp:82-209`) as a bf16 load-macro SiLU in `ckernel_sfpu_silu.h`, **with** the `DISABLE_SFPLOADMACRO` sfpi fallback (D2). |
| `eltwise_chain` `BinaryFpu` (pre-perf, Phase 0; recorded at the kernel head only) | capability | **missing** | `chain.inl:1469` calls `mul_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst)` with no `bcast_row_idx`. Site: `compute.cpp:398`. | `bcast_row_idx` on `BinaryFpu`, plus post-accumulation DEST-only elements on the sticky slot (the gap closer the kernel head names). |
| `dataflow_kernel_lib::read_sticks_for_tilize` (pre-perf, Phase 0; recorded at the kernel head only) | capability | **missing** (the rationale is imprecise) | ROW granularity barriers after every stick (`tilize_helpers_dataflow.inl:146-157`). TILE granularity already batches one barrier per 32 rows (`:110-127`) but needs tile-sized pages. Neither takes a halo prefix from another source. `reader.cpp:31` says "barriers after every stick" without saying which mode. | A batched-barrier option for row pages, a caller prefix, and a comment reworded to "in ROW granularity". |

- **Ergonomics rows.** Neither has roughly equal ns: both perf rows bought real speed (−46% and −32%).
- **Unrecorded bypasses.** None. Every raw call in the shipped kernels (`TTI_*`, `_llk_math_eltwise_sfpu_apply_vector_mode_`, `set_dst_write_addr`, `compute.cpp:257-310`) falls under rows 1–2.
- **Pre-perf rows.** Rows 3–4 predate the tournament, so no `### Helper bypasses` table carries them. Consider recording Phase 0 / refinement deviations in the same schema so they can be read across runs.

API sketch for row 1 (confidence: low, from one call site; derived from `silu_from_pack`, `compute.cpp:281-307`):

```cpp
template <SfpuType T, typename InitFn>
FORCE_INLINE void sfpu_init_pack(InitFn init);
template <typename SfpuBody, typename GroupBegin = NoHook, typename GroupEnd = NoHook>
FORCE_INLINE void apply_sfpu_from_pack(uint32_t num_tiles, SfpuBody body, GroupBegin begin = {}, GroupEnd end = {});

// call site
compute_kernel_lib::sfpu_init_pack<SfpuType::silu>(silu_bf16::init);
compute_kernel_lib::apply_sfpu_from_pack(group, silu_bf16::rows<8>, silu_bf16::group_begin, silu_bf16::group_end);
```

## 4. Agent prompts → `.claude/agents/*.md`

**P1. The verifier called 004 "compute-bound" from two solo ablations; they were balanced, so R3's focus lever could not move 004.**
- **What.**
  - **The call:** `verification_report.md:147` says "004 is compute-bound. Compute alone is 109 µs; dataflow alone is 99 µs." R3 Goal (a) was built on that.
  - **The outcome:** R3 left 004 at 129.1 → 129.6 µs and found that a faster SiLU makes 004 slower (~150 µs).
  - **Perf 1's measurement:** "Compute (108.6) and DM (111.0) are balanced … Every variant that makes compute faster makes the op slower" (`changelog.md:252-254`).
  - **Where the rule lives:** `perf-coordinator.md:79-84` has it ("two such runs … say the stages are balanced"). `incremental-verifier.md` has no hit for balanced, peel or cumulative.
  - The R3 lever still paid off on single-row plans (−5..−8%). The cost was a 1h39m phase aimed at the focus shape that did not move it.
- **Recommendation.** Add to `incremental-verifier.md` § Perf refinements: "solo ablations within ~15% of each other mean *balanced*, not X-bound. Require the refinement to A/B a faster-stage variant on the whole op before committing to a lever family."
- **Confidence.** Med–high.

**P2. The perf prompts prescribe tooling that is missing or broken in the tree; four agents worked around it independently.**
- **Evidence.**
  - `perf-coordinator.md:50` names `kernel_lib/perf_instrumentation.hpp`. It was absent, and the coordinator wrote it into the shared kernel_lib (`0aa6e9dbf08`, `changelog.md:220`).
  - `perf-coordinator.md:66` names `run_safe_pytest.sh --profile`, where tracy-capture failed:
    - exit 3 for the verifier and for the R1 implementer (both breadcrumbs);
    - "still crashes" in Perf 2 (`changelog.md:477`);
    - "nested sessions" for a part-optimizer, which switched to `ttnn.ReadDeviceProfiler` in-process (part-optimizer breadcrumb #11).
  - The `noc_estimate` CLI was not built, so the roofline gate fell back to the DRAM floor in both rounds (`changelog.md:270`, `:366`).
  - Infra: `/home` hit ENOSPC mid-Perf 2 and broke the JIT ccache (`perf_2_stderr.log`, coordinator breadcrumb 02:10).
- **Recommendation.**
  - Add a pipeline pre-flight that builds `noc_estimate`, lands `perf_instrumentation.hpp` in the base tree, and smoke-tests `--profile`.
  - Name the working fallback in `perf-coordinator.md` Step 1.3: `generated/profiler/.logs/cpp_device_perf_report.csv`, or in-process `ReadDeviceProfiler` with `eval.metrics_plugin.op_window`.
- **Confidence.** High.

**P3. The planner handed off an "immutable" acceptance test that could not run.**
- **What.** `test_repeat_is_bit_identical` called `case.values[0]` on an already-unpacked dict, a TypeError before the op ever ran.
  - The implementer was not allowed to edit it (implementer friction breadcrumb, 19:03); the verifier fixed it (`6032bc62d81`).
  - `incremental-planner.md:325-335` has no "execute it once" step.
- **Recommendation.** Add to the planner's Acceptance Test section: "before committing, run the test once against the PyTorch reference wrapped as a stub op, so harness errors surface. The implementer cannot edit this file."
- **Confidence.** Med.

**P4. The coordinator's spread-and-delete rule removed the portable fallback (see D2).**
- **Recommendation.** In `perf-coordinator.md` (description, and §graduation at `:203`), exempt a portable sfpi or helper path that serves as the `DISABLE_SFPLOADMACRO` / sim fallback from "deleting what it replaces".
- **Confidence.** Med.
