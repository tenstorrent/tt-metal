# Self-reflection: groupnorm_sc_N_1_HW_C (run_1, post-blind, advisory)

Everything below is a proposal for a human to ratify. Nothing here is auto-applied.

## Summary

- **Blind pass**: 3720 passed / 0 failed / 0 xfail (golden 3685 + regression 35; no translated cells in this universe). 10788 of 14508 generated cells (74 %) were `INVALID` skips. `verify_supported`: `supported_fail 0`, `xpass_drift 0`. SUPPORTED == TARGET, `EXCLUSIONS = []` since Refinement 2 (Phase 0 carried 88 `xfail_expected` on `ROW_MAJOR × hw_non_aligned`, lifted with no drift).
- **Most important finding**: the only near-miss cluster is a *reference* artefact, not a kernel one — every `dtype=FLOAT32 ∧ affine_dtype=BFLOAT8_B` cell sits at **0.90–0.93 of its rms tolerance** (all other dtype pairs ≤ 0.49) because `helpers.py` builds the bf8b-weight reference from bf16-rounded torch values. One bad seed flips a passing suite.
- **Perf phase ran (2 rounds)**; three recorded helper bypasses all verify as genuine **`missing`** capabilities (none is an undocumented feature), plus one pre-existing raw-LLK site that never received a table row. The compute bypass gained only ~3 % over the best helper-only variant.
- Problems are **framework-level** (golden template reference path, registry model's blindness to a compute-config axis, helper API holes, three reference-doc absences that each cost a real bug or hang) rather than op-level. The op itself looks honest and correct.

## 1. Golden coverage → `eval/golden_tests/groupnorm_sc_N_1_HW_C/feature_spec.py`

No blind failures, so no failure-derived `LOOSE_CASES`. Findings are near-miss and absence findings.

**1.1 `FLOAT32 × BFLOAT8_B-affine` cells at 0.93 of tolerance — reference quantisation, not kernel error.**
- What: 268 cells share the conjunction `dtype=FLOAT32 ∧ affine_dtype=BFLOAT8_B`; their worst `rms/tol` is 0.929 (`1x1x2048x128 … affine=gamma_only-affine_dtype=BFLOAT8_B`, both layouts), vs ≤ 0.49 for every other `(dtype, affine_dtype)` pair. Discriminative facet is the *pair*, enriched at `affine=gamma_only`. The cause is in the golden template: `helpers.py:59` `ttnn.bfloat8_b: torch.bfloat16, # no native torch bf8b; reference in bf16` — the reference sees bf16-rounded weights while the device consumes bf8b-quantised ones, so a fixed quantisation error is charged against the tight fp32 tolerance (`helpers.py:40` `(0.999, 0.01)`).
- Evidence: `golden_blind_final/test_results.json` (rms 0.00929 on the cells above); `incremental-verifier_breadcrumbs.jsonl:2` "fp32-activation x bf8b-affine cells therefore sit at rms 0.0093 against the fp32 target 0.01 purely from reference quantisation, a seed-flip risk".
- Recommendation: in the golden template (`helpers.py`), dequantise bf8b *weight* tensors for the reference via `ttnn.to_torch(device_tensor)` (as already done for activations), or key `TOLERANCES` on `(dtype, affine_dtype)`. This is a template-level fix (op-agnostic: any op with a bf8b side-input has it).
- Confidence: high.

**1.2 Axis-blind gap: `alignment` collapses two independent facets, and the conjunction is never exercised.**
- What: `tag_alignment` (op file `:50-58`, mirrored by `axes.py` `tile_alignment(..., w_first)`) returns a single value — "C wins when both are off" — so a shape with **both** `HW % 32 != 0` and `C % 32 != 0` tags `c_non_aligned` and looks covered, yet no `INPUTS` entry has both (the four `hw_non_aligned` entries all have `C ∈ {64,128,256}`; the eleven `c_non_aligned` entries all have `HW ∈ {64,128}`). The two paths are distinct kernel mechanisms (Refinement 2: partial `REDUCE_COL` scaler on the last tile-row for HW; membership zero-columns / ragged stick lanes for C) and were never tested together, in either layout. Same blindness: `hw_non_aligned ∧ group_straddling` is never exercised (all `hw_non_aligned` INPUTS use `num_groups=1`).
- Evidence: `feature_spec.py:190-213` (INPUTS blocks); `changelog.md:30-37` (Refinement 2 describes the two mechanisms separately).
- Recommendation: add `LOOSE_CASES` entries pinning the conjunctions, minimal boundary shapes: `{"inputs": {"input_tensor": (2, 1, 50, 48), "num_groups": 2}, dtype: bfloat16, layout: ROW_MAJOR, affine: gamma_beta, affine_dtype: bfloat16, affine_layout: ROW_MAJOR}` (HW ragged ∧ C ragged ∧ straddling ∧ N>1 ∧ RM stick path) and the TILE twin; plus `{"input_tensor": (1, 1, 17, 320), "num_groups": 32}` (`hw_non_aligned ∧ group_straddling`). Consider a registry-level note that `tile_alignment(w_first=True)` is lossy and that `INPUTS` must realise the both-off region explicitly.
- Confidence: high that the region is untested; medium that it hides a bug (the op passes its own `layout_matrix` on `(2,1,100,80)` per implementer breadcrumb 23, but that is `c_non_aligned` only).

**1.3 `LOOSE_CASES` is empty → the perf phase free-selected its focus shape twice, and the harness emits a phantom skip.**
- Evidence: `perf-coordinator_breadcrumbs.jsonl:0` and `:8` "LOOSE_CASES empty -> free-selected"; `test_results.json` one `skipped` cell "got empty parameter set for (cell, bundle, extras)" from `test_op_loose`.
- Recommendation: golden authoring should pin at least one perf-flagged loose case for the op's dominant regime (the verifier prompt `:377` already assumes one exists); harness: don't emit a parametrised `test_op_loose` node when `LOOSE_CASES == []`.
- Confidence: high.

**1.4 74 % of the cartesian is `INVALID`, and `verify_supported` files those skips under `no_axes_found`, not `invalid_skipped`.**
- What: `affine_dtype`/`affine_layout` are dependent on `affine`; the cartesian generates 10788 structurally impossible cells and skips each with `INVALID: {...}`. `verifier_report.json` shows `invalid_skipped: 0` and `no_axes_found: 10823` — the INVALID skips carry `observed_axes: {}` so the verifier cannot classify them, and the `invalid_skipped` counter is dead for this op.
- Evidence: `golden_blind_final/verifier_report.json` summary; `test_results.json` skip messages.
- Recommendation: (a) harness — prune `INVALID` at enumeration time rather than at run time (or let a spec mark dependent axes so the cartesian is conditional); (b) `verify_supported` — attach the *generated* cell axes to INVALID skips so they land in `invalid_skipped`. Framework-level, op-agnostic.
- Confidence: high.

## 2. SUPPORTED honesty → op file `SUPPORTED` / `EXCLUSIONS`

**2.1 No over- or under-claim.** `supported_fail 0`, `xpass_drift 0`, `supported_pass 3685` (`verifier_report.json`). SUPPORTED covers every TARGET value on all 7 axes, `EXCLUSIONS = []` (`groupnorm_sc_N_1_HW_C.py:113`). The Phase-0 exclusion `{ROW_MAJOR, hw_non_aligned}` (88 `xfail_expected` in `golden_phase0`) was lifted in Refinement 2 (`77c7c69fbb`) and the cells went straight to `supported_pass` — honest trajectory, nothing to promote or demote.

**2.2 A gated knob lives *outside* the registry model and is therefore invisible to golden and to `verify_supported`.**
- What: the op refuses `fp32_dest_acc_en=False ∧ dtype=float32` (`EXCLUSIONS_COMPUTE_CONFIG`, op file `:133`) but had to keep it out of `SUPPORTED`/`EXCLUSIONS` because `eval/registry.py:259-260` returns `"axis missing from cell"` for any SUPPORTED key the golden cell lacks — a literal `SUPPORTED["fp32_dest_acc_en"]` would xfail the whole suite (op file `:120-126`). Golden therefore never exercises the 16-bit-DEST path that Refinement 3 added (322 executed cells in the op's own precision matrix, `136bcc7012`), and the exclusion's honesty is unverified by the pipeline.
- Recommendation: registry model — either let `feature_spec.TARGET` carry non-tensor knobs (`fp32_dest_acc_en`) with `TOLERANCES` keyed on them, or make `unsupported_reason` treat a SUPPORTED key absent from the cell as "unconstrained" instead of xfail. Until then, document `SUPPORTED_COMPUTE_CONFIG` as a recognised sidecar in `eval/op_template.py`.
- Confidence: high (rule verified in `registry.py`), medium on the preferred fix.

## 3. Helper / reference docs → helper docstrings + `.claude/references/`

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `ckl::DestReuseBinary` (eltwise chain) | capability: no `BroadcastDim` on the CB operand | **missing** | `chain.hpp:496` takes `InputSpec Input` (not `BroadcastInputSpec` as `BinaryFpu` does at `:484`); `chain.inl:1347-1351` → `{add,sub,mul}_reuse_dest_tiles`; `eltwise_binary.h:412` `llk_unpack_A<BroadcastType::NONE, acc_to_dest, reuse_dest>` — no broadcast variant exists at any public layer. Raw site `compute.cpp:617-640`. **Note the ns pair**: helper-only best 2529 vs raw 2456 (~3 %); the graduated raw block buys almost nothing over the helper-only `lane_l1` variant. | Add `DestReuseBinary<Op, input(cb, BroadcastDim::Row), DEST_TO_SRCA>` (static_assert `DEST_TO_SRCA` for broadcast; reconfig srcA to the CB format before MOVD2A — see 3.1 below). Or, given the 3 % delta, revert to the helper-only path and drop the raw block. |
| `calculate_and_prepare_[partial_]reduce_scaler[s]` (dataflow) | capability: entry points own `reserve→zero→full barrier→fill→push`; fill primitives `.inl`-only | **missing** (+ doc absence) | `reduce_helpers_dataflow.inl:199-200` `noc.async_write_zeros(dfb, …); noc.write_zeros_l1_barrier();` inside `prepare_reduce_scaler`; `fill_each_face_row0[_partial]` declared only in the `.inl` (`:48`, `:93`), absent from the `.hpp` public surface (`:58-153`). The `.hpp:26-37` docstring never says the helper issues a **full NoC read barrier** (WH/BH zero-fill is a loopback read). Site `reader.cpp:171-177`. | Promote `fill_each_face_row0[_partial]` to the `.hpp` as a documented fill-only entry (`prepare_reduce_scaler_in_zeroed_page(ptr, valid)`), and add one doc line: "issues `write_zeros_l1_barrier` = full `noc_async_read_barrier`; do not call with reads in flight you don't want to wait on". |
| `Noc` wrapper zero-fill (`async_write_zeros` / `write_zeros_l1_barrier`) — "no helper covers this" | capability: no trid plumbing on the zero path | **missing** | `noc.h:758,802` `async_write_zeros(const Dst&, uint32_t, args)` has no `NocOptions` template, while `async_read` honours `TXN_ID` (`:196-198`); `write_zeros_l1_barrier()` (`:813`) takes no trid. Raw sites `reader.cpp:272,285-286,364-378` (`noc_async_read_set_trid`, `noc_async_read_barrier_with_trid`, manual `NOC_ZERO_MODE_EXIT()`). | `template <NocOptions opts> async_write_zeros(..., NocOptVals{.trid})` + `write_zeros_l1_barrier(trid)`; document that on WH/BH zeros are reads and so a tagged barrier is the *only* way to shadow them. |
| chain SFPU elements (`MulUnary/AddUnary/Relu/Rsqrt/MulBinary/SubBinary`) — **no table row** (Refinement-5 bypass, mentioned only in prose) | (prose) capability: no vector-mode / iteration knob | **missing**, and **unrecorded in the table** | `grep VectorMode ttnn/cpp/ttnn/kernel_lib/eltwise` → 0 hits; raw site `groupnorm_sc_N_1_HW_C_lane_sfpu.hpp:44-46` `MATH(SFPU_UNARY_CALL(..., ckernel::VectorMode::R, ...))`; `changelog.md:83,111` "The pre-existing Refinement-5 lane-form SFPU bypass … is unchanged" — no `helper ns / raw ns / site` row in either round. Measured 2.95 → 1.03 µs finalize (`changelog.md:58`). | Add a `VectorMode`/iterations template knob to the chain unary/binary SFPU elements (row-0-only passes for lane-form tiles). Record the row retroactively; see §4.3 for the prompt rule that let it go unrecorded. |

API sketches derived from the raw sequences actually written (`confidence: low` — single call site each):

```cpp
// compute.cpp:617-640 wanted:   DEST[c] = DEST[c] + bcast_row(cb_b_row[c])   after   mul_tiles_bcast_rows(x, a_row)
eltwise_chain(tiles(cols),
    BinaryFpu<Mul, input(cb_x), input(cb_a_row, BroadcastDim::Row)>,
    DestReuseBinary<Add, input(cb_b_row, BroadcastDim::Row), DestReuseType::DEST_TO_SRCA>,   // reconfigs srcA to cb_b_row's format itself
    PackTile<output(cb_y)>);

// reader.cpp:171-286 wanted: zeros that can be shadowed under in-flight reads
noc.async_write_zeros<NocOptions::TXN_ID>(cb_scaler, tile_bytes, {}, NocOptVals{.trid = TRID_ZERO});
noc.write_zeros_l1_barrier(NocOptVals{.trid = TRID_ZERO});                     // awaits only the zeros; exits zero mode
dataflow_kernel_lib::fill_reduce_scaler_row0<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_COL>(ptr, valid_elems);  // fill-only, no lifecycle

// lane_sfpu.hpp:44-58 wanted:
ckl::MulUnary<..., ckl::VectorMode::R, /*iterations=*/2>
```

Unrecorded-bypass audit: every other raw-LLK site in the kernels (`reader.cpp:598-610` trid resets) belongs to row 3. `ckernel::PoolType/ReduceDim` uses in `compute.cpp:298-385` are helper template args, not bypasses. Gaps *observed but not bypassed* and correctly recorded in prose (`changelog.md:83,111`): `chain_has_duplicate_upfront_cbs_v` forbidding two `Upfront` stages of one CB (ergonomics), `BinaryFpu` `DestAccumulation` tied to a single-slot lifecycle, `read_sticks_for_tilize` having no issue/barrier seam (which is why the RM path kept the legacy order — a real carve-out cost).

**3.1 `DestReuseBinary<…, DEST_TO_SRCA>` reconfigures the wrong unpacker for the CB operand.**
- What: the chain folds the CB operand's dtype reconfig onto srcB for `DEST_TO_SRCA`, but the LLK unpacks the CB through unpacker A for either reuse side; a CB whose format differs from the previous srcA CB (fp32 `b` after bf16 `x`) fires `is_unpacker_A_configured_correctly` under `--dev` and would unpack with the wrong format in production. The perf bypass in row 1 had to hand-write `reconfig_data_format_srca(cb_x_pass2, cb_b_row)` (`compute.cpp:619`) for the same reason.
- Evidence: `ttnn-implementer_breadcrumbs.jsonl:4` citing `chain.inl:1300-1316` vs `eltwise_binary.h:48-63`; `changelog.md:107` "reconfigures srcA to a 32-bit format before the MOVD2A".
- Recommendation: `DestReuseBinaryImpl` reconfigures srcA for both reuse sides (or `static_assert`s format equality); `chain.hpp` doc line: "the CB operand always goes through unpacker A, whichever side DEST is routed to".
- Confidence: high (two independent observers, one of them measured).

**3.2 `WaitPolicy::None` docs are silent on the pack→unpack race across two chains in one kernel.**
- What: `chain.hpp:206` defines `None`; `:251,256` describe it only as "caller-managed". A `None` input addressing a tile the *previous* `eltwise_chain` packed is unordered — only `cb_wait_front` orders the packer write against the next unpack. Manifested as non-deterministic per-element errors on one layout×affine cell; fixed in `72115e31c7` "fix CB race".
- Evidence: `incremental-verifier_breadcrumbs.jsonl:1`; commit `72115e31c7`.
- Recommendation: add to the `WaitPolicy::None` doc: "if a preceding chain in this kernel produced the tile, `cb_wait_front(cb, index+1)` is mandatory before the consuming chain"; optional debug-build assert that a `None` input CB has ≥ offset+1 tiles fronted.
- Confidence: high.

**3.3 `device-zone-scope-attribution.md §3` prescribes hoisting helper waits without warning that a duplicate `cb_wait_front` costs ~90 cycles.**
- What: `.claude/references/device-zone-scope-attribution.md:112` "hoist the wait out of the helper call (wait explicitly, then call the helper on already-present tiles)". Perf 1 measured four such hoists at **+0.10…+0.45 µs** with zones compiled out, exceeding several of the stages they attributed; they were removed.
- Evidence: `changelog.md:79` "Finding — hoisted waits are not free … a duplicate `cb_wait_front` on the unpack thread costs ~90 cycles even when the tiles are present"; `perf-coordinator_breadcrumbs.jsonl:6`.
- Recommendation: one line in §3: "a hoisted wait is a real duplicate `cb_wait_front` (~90 cycles on unpack); compile it out with the zones or measure the whole op with and without it before ranking".
- Confidence: high.

**3.4 Absent rule: NoC read source/destination address-residue alignment.**
- What: no reference states that on Blackhole `noc_async_read(src, dst)` needs `src % 64 == dst % 64` (WH: 32). The design's RM gamma-row read (stick `+32 B` into face 1 `+512`) violated it and silently returned wrong data (PCC 0.5), caught only by the `--dev` watcher.
- Evidence: `ttnn-implementer_breadcrumbs.jsonl:6` (ref `.claude/references/ttnn-python-utility-bindings.md`, `op_design.md` gamma row read).
- Recommendation: add the rule to `data_transfer_analysis_reference.md` / the memory-layouts reference, with the idiom "aligned read + local fix-up" for sub-tile chunks of an RM stick.
- Confidence: high.

**3.5 Absent rule: `DEST_AUTO_LIMIT = 8` under fp32 DEST assumes `dst_full_sync_en=True`.**
- What: the default `ComputeConfigDescriptor` (half sync) gives 4; no reference names `dst_full_sync_en` (`grep full_sync .claude/references/{precision_convention,op-design-template,ttnn-op-constraints}.md` → 0 hits), so a literal reading of the default config silently halves every DEST-derived extent in the design.
- Evidence: `ttnn-implementer_breadcrumbs.jsonl:8`.
- Recommendation: `precision_convention.md` + op-design template knob table: state the `(fp32_dest_acc_en, dst_full_sync_en)` pair the DEST cap assumes and mirror `dest_helpers.hpp`'s table on the host.
- Confidence: high.

**3.6 `mcast_pipe` docs do not warn that `ReceiverPipe`'s ctor reset races an early sender when `handshake=False`.**
- Evidence: `ttnn-implementer_breadcrumbs.jsonl:5` "ReceiverPipe ctor (data_ready_.set(INVALID)) … lost signal -> hang"; the fix shipped in `94c7af7b59` "handshake-free combine". Related: `reduce_helpers_compute.inl:1181` asserts CB capacity is a multiple of `Ht*DEST_AUTO_LIMIT`, not of the block shape passed; undocumented in the `BulkWaitBulkPop` header doc (`ttnn-implementer_breadcrumbs.jsonl:3`).
- Recommendation: `mcast_pipe.hpp`: "without PRE_HANDSHAKE the sender must have a happens-before with every receiver's ctor". `reduce_helpers_compute.hpp` (`BulkWaitBulkPop`): "REDUCE_COL bulk = Ht·min(Wt, DEST_AUTO_LIMIT); input CB pages must be a multiple of Ht·DEST_AUTO_LIMIT".
- Confidence: medium (single observer each, but each produced a hang).

**3.7 bf8b padding semantics undocumented in the layout references.** Pad lanes of a bf8b *output* tile share an exponent with valid lanes per 16-element segment, so garbage there corrupts valid data; fixed in `72115e31c7` ("bf8b/pad-lane bugs"). Evidence: `incremental-verifier_breadcrumbs.jsonl:2`. Recommendation: one line in the tensor-layouts reference: "bf8b pad lanes must be finite and small, not merely masked". Confidence: medium.

## 4. Agent prompts → `.claude/agents/*.md`

**4.1 `incremental-verifier.md` Testing Protocol presents a ~50-min golden run as a plain foreground command.**
- Evidence: `incremental-verifier.md:204` `eval/eval_test_runner.sh eval/golden_tests/{op_name}/ <results_dir>`; `incremental-verifier_breadcrumbs.jsonl:0` "far above the 10-min Bash tool cap … First attempt was killed by the cap and the run lost … kernel .cpp files must NOT be edited while a golden run is in flight (JIT compiles source per program …)".
- Recommendation: state in the prompt: launch the runner detached (`setsid nohup … &`) and poll `golden_results.txt`; never edit kernel sources or the descriptor while a run is live; budget the runs (as-handed / post-fix / final).
- Confidence: high.

**4.2 `incremental-planner.md` let two hazardous synchronisation choices ship without a stated ordering invariant.**
- What: `op_design.md` prescribed (a) `handshake=False` multicast with a gather-ready signal (lost-signal hang, 3.6) and (b) aliasing `cb_x_pass1/cb_x_pass2` in streaming while asking the reader to prefetch pass-2 chunks during the combine (overwrite of chunks still being reduced). Both were discovered by the implementer on device.
- Evidence: `ttnn-implementer_breadcrumbs.jsonl:5,7`; commits `94c7af7b59`, `060884cdb2` ("record implemented CB deviations in l1_ledger.md").
- Recommendation: planner/op-design template: every semaphore protocol and every CB alias must carry a one-line happens-before statement ("X is complete before Y because Z") and name the credit that protects an aliased region; `blocking-model.md` reviewers check for it.
- Confidence: medium-high.

**4.3 `perf-coordinator.md` scopes the `### Helper bypasses` table to "every graduated raw-LLK path" of *that round*, so a pre-existing bypass never gets a row.**
- Evidence: `perf-coordinator.md:230` "Every graduated raw-LLK path states … why the helper wasn't used"; `changelog.md:83,111` record the Refinement-5 `lane_sfpu.hpp` bypass only as "unchanged", with no `helper ns / raw ns / site` row in either round (see §3 table row 4).
- Recommendation: make the table **cumulative at round end** — "one row per raw-LLK site present in the kernels when the round closes, including sites inherited from refinements", and require the refinement phase (`ttnn-implementer.md`) to emit the same row schema when a refinement introduces raw LLK.
- Confidence: high.

**4.4 Perf-coordinator focus selection depends on a perf-flagged `LOOSE_CASES` entry that golden authoring does not require.**
- Evidence: `incremental-verifier.md:377` "Target the perf-flagged case first"; `perf-coordinator_breadcrumbs.jsonl:0,8` "LOOSE_CASES empty -> free-selected". Two rounds worked from an agent's own shape choice with no human-ratified target.
- Recommendation: the golden-tests authoring step (`/golden-tests`) should require ≥ 1 perf-flagged loose case whenever TARGET is non-trivial, or the coordinator prompt should ask the human before free-selecting.
- Confidence: medium.

**4.5 (infra, low)** `--dev` was unusable throughout the perf phase — `WATCHER_ENABLED + PROFILE_KERNEL` overflows the BRISC firmware region — so the watcher-guarded `NOC_ZERO_MODE_EXIT()` path graduated untested under the watcher (`perf-part-optimizer_breadcrumbs.jsonl:7`; `changelog.md:101`). Recommendation: perf prompts should name this build conflict and require a non-profiler `--dev` confirmation run before graduation. Confidence: low (firmware-size limit, not a prompt defect per se).
