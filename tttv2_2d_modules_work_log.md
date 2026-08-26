# TTTv2 2D Modules Milestone A Work Log

## Checkpoint 1 - Scope and repository audit

- Date: 2026-08-19
- Active goal: Complete Milestone A from `tttv2_2d_modules_plan.md`.
- Confirmed scope: reusable Galaxy CCL collaboration, `Prefetcher2D`, six new 2D modules, completion of `RMSNorm2D` and `MLP2D`, generic immutable batched-prefill policy, focused tests, documentation, and modularity audit.
- Confirmed exit constraints: Wormhole `(8, 4)` only; no 1D module implementation changes; no `from_model_args` dependency in 2D modules; default runtime behavior unchanged; TT hardware tests serialized.
- Existing reusable 2D implementation is limited to `mlp_2d.py`, `rmsnorm_2d.py`, and basic tests. The requested plan document and other planning artifacts are untracked user files and will be preserved.
- Execution strategy: parallelize disjoint code and host-test ownership; integrate centrally; run all WH Galaxy hardware tests sequentially, using `tt-smi -r` only if recovery is required.

### Modularity scorecard (initial)

- New 2D/model files added: none yet.
- Existing shared files changed: none yet.
- 1D module implementation files changed: 0.
- Default runtime behaviors changed: 0.
- 1D regression suites run: none yet.
- Common-code topology assumptions discovered: hard-coded supported physical batches in prefill planning, as documented by the plan.
- Boundary status: planning remains within modules/config/model boundaries.

## Checkpoint 2 - Parallel work dispatched

- Six agents received dedicated `/goal` scopes with disjoint write sets.
- Agent lanes: Galaxy CCL plus `Prefetcher2D`; `Embedding2D` plus `RotarySetup2D` plus `LMHead2D`; `RMSNorm2D` plus `MLP2D`; `Attention2D`; `Sampling2D`; generic batched-prefill policy.
- All agent work is limited to host-safe implementation and tests. No agent may reserve or reset TT hardware.
- Main-thread ownership: cross-slice contract review, work-log checkpoints, documentation and scorecard, complete host integration, and serialized WH Galaxy testing.

## Checkpoint 3 - Batched-prefill policy agent baseline and design

- Scoped agent goal: Milestone A step 7 only, confined to `llm_runtime` prefill config/planning and focused host tests.
- Existing planner audit found an isolated legacy implicit path plus an explicit path controlled by batch cap, chunk limit, strict 128K token ceiling, no-prefix eligibility, and batched-output sampling safety.
- Baseline focused result: 24 policy/config tests passed before edits.
- Chosen compatibility design: add a frozen topology-neutral policy, synthesize the exact existing explicit policy by default, preserve legacy implicit planning unchanged, and delegate only physical selection and generic eligibility facts from runtime to planner.
- Hardware status: no TT hardware used or reserved by this agent.

## Checkpoint 4 - Batched-prefill policy implementation and focused verification

- Added frozen `BatchedPrefillPolicy` ownership to `PrefillRuntimeConfig` with physical batch sizes, active-row floor, physical/sequence/token ceilings, cached-prefix eligibility, and sampling/output requirements.
- Default resolution materializes the exact former explicit planner limits: physical sizes `(1, 2, 4, 8)`, minimum 2 active rows, maximum physical batch 8, sequence ceiling 2048, strict 128K token ceiling, no cached prefixes, and batched extraction required for device sampling.
- Planner/runtime plumbing is topology-neutral and delegates generic request facts to the resolved policy. The separate implicit legacy path remains unchanged.
- Added focused host coverage for active rows 15/16/31/32, padding sentinels and row metadata, cached and over-2048 fallbacks, greedy/seeded sampling eligibility, and physical-32 program/trace signatures.
- Verification: 34 focused policy/config tests passed; the complete `test_prefill_runtime.py` suite passed (141 tests). Black formatting check passed. `ruff` and `flake8` are unavailable in this environment.
- Hardware status: no TT hardware used or reserved by this agent.

## Checkpoint 5 - Batched-prefill policy final regression and boundary audit

- Adjacent host regression result: 458 executor-integration, program-compiler, trace-compiler, and warmup tests passed.
- The full `models/common/tests/llm_runtime` run is currently blocked by two unrelated stale tests in the checkout: one expects an older four-field `Llama3ExecutorConfig`, and one execution fixture omits the existing `capture_plan.prime` attribute. Neither failure touches this agent's files or policy path.
- Final scoped files: `prefill/config.py`, `prefill/plan.py`, `prefill/runtime.py`, and `test_prefill_runtime.py`; this work log is the only project-level artifact updated.
- Boundary audit: no 1D module implementation, 2D/model code, warmup, signature, trace, decode, KV, or output-reader file changed by this agent. Runtime execution change is limited to mechanical delegation of generic sampling/output facts to resolved config.
- Default-behavior proof: the complete prefill runtime suite passed all 141 tests without changing pre-existing expectations, default policy values are asserted explicitly, and adjacent 1D executor bindings passed their host integration contracts.
- Hardware status: no TT hardware used or reset.

## Checkpoint 3 - Reference and hardware audit

- Legacy Galaxy CCL is not reusable as-is: it owns model configuration dictionaries, mode/model flags, topology decisions, semaphore cycling, and model-shaped persistent buffers. The new shared implementation must receive resolved resources explicitly.
- Legacy prefetch setup imports model-local core-range policy and mutates device mode during construction. `Prefetcher2D` must instead expose immutable contexts and explicit activation/cleanup boundaries.
- Existing 2D tests accept `(4, 8)` and exercise `from_model_args`; both conflict with Milestone A and must be replaced or narrowed to canonical `(8, 4)` direct configs.
- Local TTNN reports `wormhole_b0`; `tt-smi -s` sees the WH Galaxy hardware. No reset has been needed.
- Hardware execution remains deferred until all parallel host slices have been integrated and reviewed.

## Checkpoint 4 - Pre-change runtime regression baseline

- Command: `pytest -q models/common/tests/llm_runtime`
- Result: 1017 passed, 2 failed, 1 skipped in 201.83 seconds.
- Existing failure 1: `test_executor_config_has_exact_static_policy_owners_and_is_frozen` expects four fields, while the current implementation already has `allow_batched_prefill_with_device_sampling_for_diagnostics`.
- Existing failure 2: `test_traced_prefill_compile_does_not_interpret_request_eligibility` supplies an older capture-plan stub without the current `prime` attribute.
- These failures occurred before any tracked Milestone A implementation landed. Final comparison criterion: no additional common-runtime failures; Milestone A policy tests must pass.

## Checkpoint 5 - First module slices integrated

- `Embedding2D`, `RotarySetup2D`, and `LMHead2D` landed with 25 focused host tests passing independently.
- `Sampling2D` landed with canonical Galaxy placement, `LazyBuffer` state, padded-vocabulary exclusion, and slot-stable seeded sampling; its agent reports 44 host tests passing across the selected sampling coverage.
- Cross-slice compilation succeeds. No forbidden legacy model imports or `from_model_args` methods were introduced.
- The remaining shared, attention, MLP/RMSNorm, and runtime agents are still completing verification; their visible patches are under central contract review but are not yet treated as final.
## Integration checkpoint: parallel implementation complete

- All six delegated implementation goals completed: Galaxy CCL/Prefetcher2D; Embedding2D/RotarySetup2D/LMHead2D; RMSNorm2D/MLP2D; Attention2D; Sampling2D; and generic batched-prefill policy.
- Delegated host verification reported 18, 25, 24, 28, 44, and 141 focused tests passing respectively, plus the runtime agent's 458 adjacent regression tests.
- No delegated goal used TT hardware. Attention2D still requires concrete TTNN operation recipes; CCL resource allocation, fused RMSNorm decode, real decode/prefill PCC, KV-cache PCC, cleanup, and repeat invocation remain Milestone A exit-gate work.
- Integration audit found two cross-lane contract issues to correct before qualification: MLP2D/RMSNorm2D retain fallback imports from common `TT_CCL`, and Sampling2DConfig is not frozen.

## Integration checkpoint: shared contracts reconciled

- Made `Sampling2DConfig` frozen.
- Removed MLP2D/RMSNorm2D imports and fallback discovery from common `TT_CCL`; both now require an injected model-owned Galaxy collaborator and validate mesh identity during config resolution.
- Bound Galaxy's linear topology and CCL tuning values in immutable MLP2D config before execution.
- Focused integrated host suite: `259 passed in 39.63s`.
- Hardware exit-gate status remains open; host mocks do not count as decode/prefill PCC or KV-cache qualification.

## Regression checkpoint: full common runtime

- Structural audit: zero 1D module implementation changes; zero `from_model_args`, legacy model-stack imports, or common `TT_CCL` imports in the 2D/Galaxy implementation.
- `compileall` and `git diff --check` passed; Black formatting was applied to the integrated MLP2D file.
- Full `models/common/tests/llm_runtime`: `1027 passed, 2 failed, 1 skipped in 200.88s`.
- The two failures exactly match the pre-change baseline: the stale exact-field assertion for the already-present diagnostics config field, and the stale trace test double without `capture_plan.prime`. No additional runtime failures were introduced.

## Qualification checkpoint: hardware-readiness audit failed

- Three independent read-only audits were dispatched with dedicated goals while the serialized 1D hardware suite owned the device.
- MLP2D/RMSNorm2D audit found P0 gaps: unusable simple constructors after mandatory CCL injection, no production fused residual RMSNorm operation, incorrect distributed geometry for Qwen's head-local 128-wide Q/K norm, and invalid chunked-prefill stats shape. It also found ignored CCL resources and transient tensor leaks.
- Embedding/RoPE/LMHead/Sampling audit found critical LM-head 2D padding and duplicate weight-materialization defects, no qualified real collective, incomplete decode-index preparation, and repeat-invocation leaks in embedding/sampling.
- The current host suite is therefore necessary but insufficient evidence. Milestone A remains open; no real 2D module PCC claim will be made until these defects are corrected and serialized WH `(8,4)` tests pass.

## Regression checkpoint: serialized 1D hardware sample

- Started the existing 1D module matrix as one serialized pytest process; collection selected 2,032 cases spanning multiple mesh shapes and long contexts.
- Stopped after `497.05s` when the suite moved into redundant `1x2` long-context coverage: `96 passed, 2 failed, 24 skipped` before interruption.
- Passing real-WH attention coverage included decode and prefill, paged and chunked KV paths, prefill/decode transitions, mesh shapes `1x1` and `1x2`, and sequence lengths through 32768. Reported PCC values met each existing test threshold.
- The two failures were host-only fake-TTNN tests for chunk-start overload selection and occurred before device execution; no 1D implementation file was changed by this work. The broad suite was intentionally interrupted to return the only Galaxy to new 2D qualification.
- Device teardown completed cleanly and `tt-smi` reports all 32 devices; no reset was required.

## Hardware-reference checkpoint: Qwen Q/K norm

- Ran the existing legacy Galaxy Q/K norm test only as a behavioral baseline while correction agents used no hardware.
- Result: `1 passed in 15.26s` on the full `(8,4)` mesh; Q PCC `0.9999867986`, K PCC `0.9999862581`.
- This is not Milestone A evidence because it imports the legacy stack. It establishes the head-local normalization geometry and numerical target the new `RMSNorm2D`/`Attention2D` path must independently match.
- Device and fabric teardown completed normally; no reset was required.

## Correction checkpoint: six parallel lanes complete

- Four correction agents completed disjoint implementation lanes: Attention2D; MLP2D/RMSNorm2D; Galaxy CCL/Prefetcher2D; and Embedding2D/RotarySetup2D/LMHead2D/Sampling2D.
- Reported focused host results are 51, 33, 27, and 64 passing tests respectively. Fixes cover direct attention TTNN stages, strict KV/placement policy, head-local Q/K norm, fused residual norm, keyed CCL ownership, prefetch start/stop lifecycle, LM-head padding/materialization, RoPE index preparation, and transient cleanup.
- Two additional agents authored seven basic-module and ten MLP/RMSNorm real-WH cases in new test files. They compiled and collected without using hardware.
- Central integration is now required because independently changed CCL/resource consumers may disagree on exact keyed-resource selection and production factory naming. No hardware tests will run until the combined host suite is clean.

## Integration checkpoint: exact CCL resource identity

- Reconciled the corrected CCL owner with MLP2D and RMSNorm2D: axis-0 all-reduce is canonical, conventional resource lookup selects axis 0 for all-reduce/all-gather-concat, and both consumers retain and cycle the exact operation/axis/geometry/sequence key resolved during construction.
- Corrected distributed RMSNorm prefill statistics from flattened token geometry to rank-preserving `(N, C, S, 32)` geometry.
- Updated host doubles to exercise the keyed resource API. Focused integrated result: `47 passed in 6.99s` across Galaxy CCL, MLP2D, and RMSNorm2D.
- Production resource creation is still absent from the common Galaxy package. Hardware qualification will begin with modules that do not depend on that factory while the allocation boundary is implemented.

## Hardware checkpoint: Embedding2D qualified

- The first Llama hardware run exposed BF16 token-index upload: `LazyWeight`'s default dtype prevented the requested `uint32` from resolving, producing PCC `0.0287529` for high vocabulary IDs.
- Fixed Embedding2D to enforce replicated `uint32` token IDs, reject foreign-mesh inputs, normalize the internal table to TTNN's rank-4 embedding layout, and return rank-4 residual output. Host result: `11 passed`.
- Real WH `(8,4)` Llama result: `1 passed in 31.72s`; decode batch 32 and prefill 128/2048 each passed twice at PCC >= 0.99.
- Real WH `(8,4)` Qwen result: `1 passed in 25.42s`; the same decode/prefill/repeat matrix passed with Qwen's 151936 vocabulary, 5120 hidden size, and embedding scale.
- Both processes completed full 32-device teardown cleanly. No hardware reset was required.

## Hardware checkpoint: RotarySetup2D qualified

- Corrected the hardware case to provide explicit fabric-safe core resources and use the module-owned grouped index preparation. Fixed validation to inspect the local 8-user column shard rather than the global batch.
- Restored the proven decode embedding reshape to logical `[users_per_column, 1, head_dim]` with tile-padded storage before the rank-4 view.
- Isolated decode and prefill table storage at one-time load and made prefill outputs owned copies; this prevents caller output cleanup from invalidating persistent tables during repeated calls or mode transitions.
- Real WH `(8,4)` results: Llama `1 passed in 2.59s`; Qwen `1 passed in 2.25s`. Both cover decode and prefill 128/2048 twice each at PCC >= 0.99. Host RoPE contracts remain `13 passed`.
- Full device teardown was clean after both runs. No hardware reset was required.

## Hardware checkpoint: Sampling2D qualified

- The initial real-device run rejected the test's worker ranges because the mesh fixture used the default dispatch axis while the production Llama/Qwen Galaxy ranges reserve cores for column dispatch.
- Aligned the hardware fixture with the production Galaxy dispatch contract by selecting `DispatchCoreAxis.COL`; the explicit model-owned sampling and top-k core ranges remain unchanged.
- Real WH `(8,4)` result: `1 passed in 21.66s`. The Qwen padded-vocabulary case ran forced argmax twice, matched all 32 expected tokens exactly, and proved that logits in `[vocab_size, padded_vocab_size)` cannot be selected.
- Full 32-device teardown was clean. No hardware reset was required.

## Hardware checkpoint: LMHead2D test contract rejected

- The first Llama real-WH attempt failed before weight materialization or device compute: the hardware test supplied a bare `_column_all_reduce` function, while `LMHead2D` correctly requires each injected collective to declare its mesh and canonical cluster axis.
- Failure: `ValueError: LMHead2D decode collective must declare the resolved mesh_device`.
- This is a test/resource-adapter defect, not numerical evidence for or against LMHead2D. The strict implementation contract will remain intact; the hardware test must supply an explicit production-contract adapter before rerun.
- Full 32-device teardown was clean. No hardware reset was required.

## Regression checkpoint: integrated 2D host suite

- Ran the complete focused host suite for Galaxy CCL, Prefetcher2D, Attention2D, Embedding2D, LMHead2D, MLP2D, RMSNorm2D, RotarySetup2D, and Sampling2D.
- Result: `176 passed in 26.42s`.
- This establishes a clean integration baseline before adding the production Galaxy resource allocator and the remaining real-device qualification cases.

## Audit checkpoint: module and runtime boundaries

- `compileall` and `git diff --check` pass for the integrated implementation.
- No 1D module implementation file is changed. The new Galaxy/2D implementation contains no `from_model_args`, common `TT_CCL`, or legacy Llama/Qwen model-stack dependency.
- The runtime diff contains no Galaxy, Llama, Qwen, Wormhole, 2D, or `(8,4)` execution branch; its change remains topology-neutral delegation to immutable prefill policy.
- Production-resource review found that async MLP/RMSNorm all-gather buffers are geometry-specific. The provisional hardware helper's `expected_weight_count` input is insufficient to allocate them honestly, so the production factory boundary must accept explicit keyed mode/geometry specifications before those hardware gates can run.

## Hardware checkpoint: LMHead2D decode qualified

- Added an explicit column-all-reduce adapter declaring the resolved mesh, canonical cluster axis, borrowed-input ownership, and owned-output behavior; strict LMHead2D collective validation remains unchanged.
- Strengthened host contracts for input geometry/cross-mesh rejection, caller-owned materialized inputs, repeat projection failure cleanup, and deduplicated release. Focused host result: `19 passed in 2.86s`.
- Real WH `(8,4)` Llama decode result: `1 passed in 38.82s`; batch 32 ran twice at PCC >= 0.99.
- Real WH `(8,4)` Qwen decode result: `1 passed in 29.39s`; batch 32 ran twice at PCC >= 0.99 and all padded-vocabulary logits remained negative infinity.
- Both processes completed clean full-device teardown without reset. Prefill 128/2048 qualification remains open.

## Hardware checkpoint: LMHead2D mode coverage complete

- Extended the real-WH test to run both `decode_forward` and `prefill_forward` twice. Prefill uses the applicable physical 32-row final-token batch produced by runtime extraction rather than projecting every prompt token through the vocabulary head.
- Real WH `(8,4)` Llama decode/prefill repeat result: `1 passed in 36.72s` at PCC >= 0.99.
- Real WH `(8,4)` Qwen decode/prefill repeat result: `1 passed in 27.15s` at PCC >= 0.99 with exact padded-vocabulary masking.
- Both processes completed clean full-device teardown without reset.

## Infrastructure checkpoint: production Galaxy resource owner

- Added an explicit production owner for WH `(8,4)` collective buffers, semaphores, and prefill/decode subdevice managers in `models/common/models/galaxy/resources.py`.
- The API requires resolved tensor shape, dtype, layout, memory placement, operation, axis, geometry, and sequence plans; it intentionally does not infer allocations from model weight counts.
- Lifecycle coverage includes allocation rollback, mode-switch serialization and rollback, reverse-order cleanup, idempotency, and cleanup continuation after release errors.
- Combined Galaxy resource/CCL host result: `20 passed in 3.04s`.
- The provisional shared hardware fixture still uses the obsolete `expected_weight_count` convention. MLP2D, RMSNorm2D, and Attention2D hardware gates remain blocked until their tests supply exact plans.

## Test-authoring checkpoint: Attention2D Galaxy coverage

- Added a standalone real-WH test with Llama 70B and Qwen 7B geometries, repeated batch-32 decode, repeated prefill 128/2048, Qwen Q/K normalization, attention-output PCC, and KV-cache PCC.
- The test uses only common modules and the production Galaxy resource boundary, with exception-safe output, weight, cache, and owner cleanup.
- Verification without taking the shared hardware: the Attention2D host suite reports `51 passed`, the new hardware file compiles and collects two model cases, and its pure CPU reference smoke passed.
- Real-device execution remains pending after explicit attention resource plans are integrated.

## Integration checkpoint: exact sequence-keyed resources

- Replaced the hardware helper's inferred weight-count boundary with mandatory `GalaxyResourcesConfig` input and exact operation-input tensor selection.
- Added an injected topology-neutral resource selector to MLP2D and RMSNorm2D. Existing single-resource callers retain construction-time binding; production tests can now select distinct 128/2048 buffers by operation, axis, tensor shape, and flattened sequence.
- Hardened Galaxy resource lifecycle after independent audit: mode plans now require the worker in the synchronized stall group, and cleanup retries manager reset/clear after a failed activation rollback.
- Combined Galaxy/MLP2D/RMSNorm2D host result: `54 passed in 8.18s`; all 12 MLP, RMSNorm, and Attention hardware cases compile and collect.

## Hardware checkpoint: head-local RMSNorm2D qualified

- The initial Q/K run exposed a test composer error: head-local outputs are replicated across mesh columns, but the helper concatenated four replicas as if they were width-sharded. The values aligned; result composition now selects one logical 128-wide replica.
- Real WH `(8,4)` result after correction: `2 passed in 5.68s`.
- Both Q and K normalization cases cover decode batch 32 and prefill 128/2048 twice each at PCC >= 0.99.
- Full device and fabric teardown was clean; no reset was required.

## Hardware checkpoint: distributed RMSNorm2D correction required

- The first distributed decode attempt rejected the generic 32-core Qwen input layout because its physical width shard was 40 elements, not tile-aligned.
- Replaced the generic grid derivation with the proven column-dispatch layouts: 16 cores with width 128 for Llama and 10 cores with width 128 for Qwen, both offset from reserved dispatch column x=0. Host RMSNorm result returned to `15 passed`.
- The corrected Llama kernel reached execution but the process segfaulted during output readback, preventing pytest teardown. `tt-smi -r` was run and successfully reset/reinitialized all 32 devices.
- Review identified that the fused residual policy still composed `add` with two-phase RMSNorm rather than owning the proven WH fused operation. Distributed qualification remains open while that path is corrected to `fused_rms_minimal`.

## Hardware checkpoint: fused RMSNorm2D decode hung

- Replaced the residual decode composition with direct `ttnn.fused_rms_minimal`, corrected the call to use one global semaphore and an explicit subdevice ID, and retained exact sequence-keyed persistent-buffer selection.
- The focused Llama batch-32 decode reached the fused call but did not return after more than 18 minutes. All 32 device-worker threads remained busy with no compiler subprocesses, indicating command-queue polling rather than a productive cold build.
- Terminated the pytest process with `SIGTERM`; no numerical result was produced. Ran `tt-smi -glx_reset`, which successfully issued reset/post-reset and reinitialized all 32 boards.
- Distributed decode remains unqualified. The next attempt requires comparison against the proven fused call site's resource ownership, subdevice-manager, and tensor-layout contracts before taking the hardware again.

## Hardware checkpoint: fused RMSNorm2D bounded retry failed

- Aligned the decode layout with the proven Galaxy Llama contract by reserving stats core `x=1`, moving norm shards to `x=2..3`, and creating the fused semaphore on the exact norm shard grid. The focused host/resource suite remained clean at `22 passed`.
- The Llama fused decode still failed to return and was terminated by a hard five-minute process timeout (`exit 124`), producing no numerical result.
- Ran `tt-smi -glx_reset` after the forced termination; all 32 boards completed post-reset reinitialization.
- The direct fused kernel is rejected for the focused no-prefetch Milestone A qualification path. Residual decode will be qualified through explicit residual addition followed by the reusable two-phase distributed norm.

## Hardware checkpoint: two-phase RMSNorm2D readback fault

- The module-owned residual-add plus two-phase Llama decode returned from device execution promptly, but host composition segfaulted in `ttnn.to_torch`; no PCC result was produced.
- Verified the resolved layouts against the proven Galaxy recipe: input/output shards are `[32,128]` on `x=2..3`, and gathered stats are `[32,128]` on `x=1`.
- Ran `tt-smi -glx_reset`; all 32 boards reinitialized successfully.
- The proven distributed tests synchronize the mesh before reading sharded L1 output. Added the missing explicit synchronization before hardware-test composition for the next bounded retry.

## Hardware checkpoint: RMSNorm2D CCL contract corrected

- Explicit synchronization blocked, proving the earlier readback segfault was downstream of an incomplete async collective rather than a host-only composition race. Stopped the diagnostic run and successfully reset all 32 boards.
- Compared the call against the existing production Galaxy CCL: persistent-buffer async all-gather requires a two-handle semaphore set and no barrier semaphore. The new path had supplied one semaphore plus a barrier, which cannot complete the persistent-buffer protocol.
- Updated decode to allocate two semaphores per slot and both decode/prefill RMS all-gathers to omit the barrier when using their mandatory persistent output buffers.
- The first retry still blocked. A second call-site comparison found that the 2D collective supplied `cluster_axis=1` without the resolved `mesh_device`; stopped the run and reset all 32 boards. Both RMS all-gathers now pass the mesh explicitly so TTNN resolves four-device column lines rather than an ambiguous flattened device set.
- The first explicit-mesh attempt failed immediately and cleanly at Python overload resolution. Updated both call sites to TTNN's dedicated 2D overload: positional dimension/axis/mesh/topology/semaphores, `persistent_output_tensor`, and no unsupported `chunks_per_sync` keyword.
- The dedicated overload accepted the call but still blocked at synchronization. Restored semaphore allocation to the full worker subdevice, matching production Galaxy CCL; narrowing semaphores to norm shards is valid for the fused primitive's containment check but not for the standalone async all-gather worker protocol.
- Further comparison showed that the focused helper's single full-core decode subdevice was itself not the production WH envelope. Added the canonical static partition: sender columns `x=0,4` as subdevice 0 and worker columns `x=1..3,5..6` as subdevice 1, with CCL semaphores and the stall group bound to worker subdevice 1.
- The canonical partition alone still blocked. Added explicit adjacent-slot semaphore windows to the Galaxy CCL context so persistent async all-gather advances by one double-buffer slot and receives `[current, next]`, matching production rather than grouping unrelated handles inside each slot.

## Verification checkpoint: fused RMSNorm2D retry prepared

- Restored the distributed decode hardware gate to the direct `FUSED_DECODE` policy and added PCC validation for both the normalized output and returned residual sum.
- Retained the canonical Galaxy sender/worker subdevice partition and one semaphore per rotating slot; the fused primitive therefore receives one exact worker-subdevice semaphore while the two-phase path retains adjacent windows.
- Focused RMSNorm/Galaxy host result: `36 passed in 5.50s`. Python compilation and `git diff --check` also pass.
- The next hardware action is a hard-bounded Llama-only fused decode run. Hardware remains serialized under the main agent.

## Hardware checkpoint: fused semaphore adapter corrected

- The bounded Llama retry failed immediately at overload resolution and completed clean device teardown. No reset was needed and no kernel executed.
- Root cause: `fused_rms_minimal` requires one scalar global semaphore, while the shared CCL context correctly returns a one-element handle sequence for each slot.
- Added a strict single-semaphore adapter that unwraps exactly one handle and rejects ambiguous cardinality. Updated the host fake to use production-shaped tuple handles and assert the scalar fused argument.
- Focused RMSNorm/Galaxy host result after correction: `36 passed in 5.44s`; compilation and diff checks pass.

## Hardware checkpoint: fused RMSNorm2D requires active prefetch ownership

- The scalar-semaphore Llama fused decode passed Python overload resolution and entered device execution, but produced no output before the hard five-minute timeout (`exit 124`).
- This reproduces the command-queue block under the canonical sender/worker manager when no prefetch session is running. Static subdevice topology and semaphore placement are therefore insufficient for this fused primitive.
- Terminated the process through the external timeout and ran `tt-smi -glx_reset`; all 32 boards completed post-reset reinitialization.
- The parallel ownership lane completed a host implementation in which sealed `Prefetcher2D` exclusively owns managers, stall groups, global CBs, activation, and cleanup, while `GalaxyResources` owns only CCL allocations and delegates mode activation. That integration is the next qualification target.

## Integration checkpoint: parallel readiness lanes synthesized

- Integrated the Prefetcher2D/Galaxy ownership lane, Attention2D readiness lane, MLP2D readiness lane, and Milestone A documentation/scorecard lane; each agent maintained a dedicated goal and checkpoint log.
- Tightened the production prefetch lifecycle against the proven TTNN contract: the asynchronous input list ends with packed address metadata, each decode activation starts a fresh session, and the stop callback deallocates the returned sentinel before mode transition or cleanup.
- Attention2D now isolates concat-32 contiguous/paged KV writes by physical source row, validates mode-specific prefetch contexts, propagates projection contexts, and requires explicit head-local Q/K norm geometry.
- MLP2D now resolves callable selectors/program factories, activation enum, and positive cutoff policy before execution, with decode/prefill collective and SiLU/GELU coverage.
- Combined Prefetcher/Galaxy/MLP/Attention/RMS host result: `143 passed in 21.18s`. Compilation and `git diff --check` pass.
- Added `models/common/modules/MILESTONE_A_STATUS.md` and updated the module README with an evidence-based scorecard that explicitly leaves all unqualified hardware gates open.

## Integration checkpoint: production prefetch object graph wired into hardware gates

- Replaced the provisional hardware adapter with explicit production construction: canonical 12-sender/receiver mapping, exact prefill/decode mode policies, packed address placement, sealed registered device weights, one lifecycle-aware activation, and CCL-before-prefetcher cleanup.
- Decode plans for RMSNorm2D, MLP2D, and Attention2D now use the canonical sender/worker subdevice partition. MLP and attention register their exact projection weights in consumption order and borrow the resulting mode contexts.
- Corrected configured global-CB sizing semantics: an explicit resolved size is authoritative because the proven block-prefetch allocation is intentionally smaller than a complete sharded tensor buffer; derivation remains available when no size is supplied.
- Registered weights remain alive until the prefetch session and CCL resources are stopped and released.
- Combined host result remains `143 passed in 21.86s`; all 12 RMSNorm/MLP/Attention WH cases compile and collect, and diff checks pass.

## Hardware checkpoint: invalid standalone RMS prefetch payload rejected

- The first integrated Llama RMSNorm attempt segfaulted inside `ttnn.dram_prefetcher` before launching the norm kernel when the queue contained the row-major RMS gamma tensor.
- The reference Galaxy stack registers only attention and MLP matmul weights with the prefetcher; norm weights are not valid queue payloads. The failure is therefore in the standalone test composition, not a numerical RMSNorm result.
- Ran `tt-smi -glx_reset`; all 32 boards reinitialized successfully.
- The next production-owner hardware gate uses MLP's proven `w1`, `w3`, `w2` registration order. RMSNorm must later run alongside a valid projection prefetch queue rather than registering gamma.

## Hardware checkpoint: exact active-sender partition required

- The first Llama MLP prefetch launch also segfaulted inside `ttnn.dram_prefetcher`, before MLP execution, despite using the proven `w1`, `w3`, `w2` weight order.
- Comparison with the reference topology found that the decode sender subdevice must contain exactly 12 active sender points. The provisional plan incorrectly included all 20 cores in columns `x=0,4`, including eight dummy sender locations that are intentionally outside the manager.
- Reset and reinitialized all 32 boards. Updated the decode plan to the exact 12-point sender set with the unchanged worker ranges `x=1..3,5..6`, and matched the prefetch input list form used by TTNN.
- Focused Prefetcher/Galaxy/MLP/RMS host result after correction: `83 passed in 12.59s`; compilation and diff checks pass.

## Hardware checkpoint: prefetch requires DRAM-sharded weights

- With the exact sender partition, Llama MLP prefetch no longer crashed. TTNN rejected the launch cleanly with `num_readers > 0`, showing that the registered common-module weights were DRAM-interleaved and therefore exposed no programmable DRAM reader cores.
- Updated the hardware MLP config to use the reference 12-bank DRAM width-sharded placements for local W1/W3 and W2 geometry and threaded those memory configs into `MLP2DConfig`.
- Corrected Prefetcher2D cleanup: global circular buffers are RAII objects and are released by dropping ownership; only tensor metadata and explicit stop results go through `ttnn.deallocate`.
- Focused Prefetcher/Galaxy/MLP host result: `68 passed in 10.04s`; compilation and diff checks pass. Device teardown from the clean TTNN validation failure completed normally, so no reset was required.

## Hardware checkpoint: ring matmul policy required

- DRAM readers launched with width-sharded weights, then the default decode matmul rejected sharded B because it resolved an interleaved-only program.
- The process blocked during failure cleanup, was terminated with `SIGTERM`, and all 32 boards were reset and reinitialized.
- Added a resolved `decode_w2_input_memcfg` field and wired the exact 24-core prefetch ring inputs, receiver outputs, hop core, global-CB receiver count, and 1D gather matmul program configs into the hardware MLP case.
- Focused MLP/Prefetcher/Galaxy host result: `68 passed in 10.16s`; compilation and diff checks pass.

## Hardware checkpoint: global-CB containment mismatch isolated

- The Llama decode retry launched the production prefetch session and reached the ring matmul, then failed with `Specified cores are not contained in associated GlobalCircularBuffer`.
- Terminated the blocked pytest/timeout process pair explicitly and ran `tt-smi -glx_reset`; all 32 boards completed post-reset reinitialization.
- The active hypothesis is a topology-construction mismatch: the sender subdevice and DRAM address metadata correctly use only 12 active sender points, while the proven global circular buffer mapping also includes eight dummy sender/receiver mappings so its receiver coverage contains the complete worker program core set.
- Dispatched independent C++ containment, reference-topology, and remaining-exit-gate audits, each with a dedicated goal and markdown checkpoint log. Hardware remains serialized under the main agent.

## Verification checkpoint: full reference global-CB topology restored

- Confirmed from `get_core_ranges` and `TtLlamaPrefetcherSetup` that the proven configuration separates 12 active DRAM-reader senders from 20 total global-CB mappings.
- Relaxed `Prefetcher2DConfig` so `address_repeat_count` may be smaller than the mapping count but cannot exceed it, and added focused validation for the active-plus-dummy contract.
- Extended the WH hardware helper with the exact eight dummy sender/receiver mappings. Their receiver ranges complete coverage of worker columns `x=1..3,5..6`; the sender subdevice and address metadata remain restricted to the 12 active points.
- Matched the reference MLP ring program's largest-divisor output subblock policy.
- Focused Prefetcher/Galaxy/MLP host result: `69 passed in 10.44s`; compilation and `git diff --check` pass.

## Hardware checkpoint: containment fixed, output placement mismatch exposed

- The bounded Llama decode run passed global-CB containment and advanced into the ring matmul.
- TTNN then rejected the first projection because its preallocated output tensor was DRAM-interleaved while `decode_w1_w3_output_memcfg` requires the 24-receiver L1 width-sharded placement.
- Terminated the blocked cleanup process and reset/reinitialized all 32 boards.
- Synthesized the parallel topology and C++ audits: both independently confirm that the 20-entry mapping is required and that the previous missing hop core `(3,6)` caused the containment failure. Their dedicated logs are `tttv2_wh_galaxy_prefetch_topology_audit_20260819.md` and `tttv2_mlp2d_gcb_containment_diagnosis_work_log_20260819.md`.
- The next correction is confined to MLP output-tensor allocation/forwarding; hardware remains serialized.

## Verification checkpoint: decode all-gather placement aligned

- Traced the memory-config fatal to `all_gather_async`: the model-owned persistent output was DRAM-interleaved while `MLP2D` requested the earlier projection's L1 layout.
- Aligned the decode path with the reference dataflow by requesting `decode_w2_input_memcfg` directly from all-gather and allocating the decode persistent output buffer in that exact 24-core ring layout.
- Removed the now-redundant post-gather conversion and added host assertions that W2 consumes the gathered persistent tensor directly.
- Focused MLP/Prefetcher/Galaxy host result: `69 passed in 10.34s`; compilation and `git diff --check` pass.

## Hardware checkpoint: decode reaches final all-reduce

- The next Llama decode run passed both ring projections, both reduce-scatters, activation, the ring-layout all-gather, and W2.
- Final `all_reduce_async` then rejected its DRAM-interleaved persistent buffer; the primitive requires width-sharded input, output, and buffer tensors, with buffer shard volume at least eight times the output shard volume for mesh axis 0.
- Terminated the blocked cleanup process and reset/reinitialized all 32 boards.
- The remaining correction is in the hardware resource plan: allocate the reference decode residual output layout and an eight-times-larger width-sharded persistent all-reduce buffer.

## Verification checkpoint: axis-0 all-reduce geometry restored

- Added the exact reference residual output layouts: 16 cores for Llama and 10 cores for Qwen, both with `[32,128]` L1 width shards.
- Allocated the axis-0 persistent buffer across all 50 worker cores with `[32,1024]` L1 width shards, satisfying the eight-device intermediate-volume requirement while containing each residual output grid.
- Kept logical all-reduce output placement separate from the larger workspace buffer specification.
- Focused host result remains `69 passed in 10.33s`; all four MLP hardware cases collect, compilation and `git diff --check` pass.

## Hardware checkpoint: all-reduce launches but stalls

- With valid width-sharded input/output/workspace geometry, the Llama decode reached and launched final all-reduce without a host validation failure, then made no forward progress for more than three minutes.
- Terminated the bounded process and reset/reinitialized all 32 boards.
- Compared the launch against the proven 6U Galaxy path: the remaining visible policy mismatch is `num_links=1` in the focused plan versus four links on this WH Galaxy. The next bounded retry uses the proven four-link axis-0 configuration.

## Hardware checkpoint: four links alone do not resolve all-reduce stall

- Retried axis-0 all-reduce with the proven four-link count; it still launched without validation errors and stalled.
- Terminated the run and reset/reinitialized all 32 boards.
- Found the remaining 6U-specific topology mismatch: the reference selects `Topology.Ring`, while the focused plan selected `Topology.Linear`. The next retry changes only final all-reduce to ring topology.

## Hardware checkpoint: ring topology still stalls at eventual synchronization

- The four-link ring-topology retry also produced no completion before the diagnostic stop; terminated it and reset/reinitialized all 32 boards.
- Because all MLP collectives are asynchronous, the eventual readback wait cannot identify which queued primitive failed to complete. The next diagnostic adds worker-subdevice synchronization immediately after each collective adapter in the hardware test path to isolate the first stalled stage.

## Hardware checkpoint: first stalled primitive isolated to all-gather

- Diagnostic synchronization completed after both decode reduce-scatters and then blocked immediately after `all_gather_async`, proving the first incomplete primitive is all-gather rather than all-reduce.
- Terminated the run and reset/reinitialized all 32 boards.
- Compared the launch with the production prefetcher path: persistent all-gather requires one semaphore from each adjacent double-buffer slot and no barrier. The common path incorrectly supplied two semaphores from one slot plus a barrier.

## Verification checkpoint: persistent all-gather protocol corrected

- Added `next_semaphore_window` to MLP2D's required Galaxy CCL context contract and now supplies an adjacent two-slot window to persistent all-gather.
- Removed the barrier from persistent all-gather and changed the hardware plan to allocate one gather semaphore per double-buffer slot.
- Host assertions lock the exact two-handle window and absent barrier. Focused MLP/Prefetcher/Galaxy result: `69 passed in 10.46s`; compilation and `git diff --check` pass.

## Hardware checkpoint: all-gather still stalls with corrected semaphore window

- The adjacent-window/no-barrier retry still completed both reduce-scatters and blocked at the all-gather synchronization boundary.
- Terminated the run and reset/reinitialized all 32 boards.
- The remaining launch differences from the proven 6U prefetcher path are ring/four-link routing and TTNN-default channel tuning. The next retry aligns both.

## Hardware checkpoint: all-gather requires explicit 2D mesh overload

- Ring/four-link routing with default channel tuning still blocked at the all-gather synchronization boundary; terminated and reset all 32 boards.
- The common call supplied `cluster_axis=1` without `mesh_device`, unlike the proven 2D overload. Added the explicit mesh argument so TTNN resolves four-device column lines rather than an ambiguous flattened device set.

## Hardware checkpoint: legacy all-gather keyword selects wrong overload

- With explicit mesh, the all-gather call blocked before returning rather than at the following synchronization boundary; terminated and reset all 32 boards.
- The remaining call-shape mismatch is overload selection: the reference passes positional `dim` and `persistent_output_tensor`, while the common path used the legacy `persistent_output_buffer` keyword. The next retry uses the exact dedicated 2D persistent signature.

## Hardware checkpoint: exact decode all-gather routing policy identified

- The dedicated persistent overload still blocked at the all-gather synchronization boundary; terminated and reset all 32 boards.
- Re-read `line_all_gather`: decode intentionally remains `Topology.Linear` even on 6U, uses four links, and enables `use_optimal_ccl_for_llama` for the MLP prefetcher path. The next retry applies that exact routing combination.

## Hardware checkpoint: all-gather input placement mismatch identified

- Exact linear/four-link/optimized routing still blocked at all-gather synchronization; terminated and reset all 32 boards.
- The remaining data-layout mismatch is upstream: the focused plan emits reduce-scatter results in DRAM-interleaved memory, while the reference feeds all-gather from a 30-core L1 width-sharded layout with `[32,32]` shards. The next retry aligns both the persistent reduce-scatter output and MLP request.

## Hardware checkpoint: all-gather qualified, all-reduce workspace mapping remains

- With the 30-core L1 reduce-scatter outputs, both reduce-scatters and all-gather completed explicit worker-subdevice synchronization.
- Final all-reduce remained the first stalled stage; terminated the run and reset/reinitialized all 32 boards.
- Its workspace is the remaining reference mismatch: the generic allocator replicated a local `(1,1,M,N)` tensor, while production constructs global `(8,4,M,N)` storage sharded over mesh axes 0 and 1. The next retry adds explicit mapper support to the Galaxy tensor allocation spec and uses it for this workspace only.

## Hardware checkpoint: global workspace mapping does not resolve all-reduce stall

- Retried Llama decode with an axis-0 workspace created from global `(8,4,M,N)` storage and an explicit two-dimensional mesh mapper, matching the legacy Galaxy allocation shape.
- Both reduce-scatter stages and the persistent all-gather completed worker-subdevice synchronization; the final all-reduce remained the first stalled stage.
- Terminated the bounded diagnostic process, reset the Galaxy, and verified that all 32 devices re-enumerated.
- The next diagnostic removes the redundant same-layout conversion after `all_reduce_async` and synchronizes directly around the primitive, distinguishing primitive completion from queued post-processing.

## Verification checkpoint: all-reduce diagnostic boundary tightened

- Removed `_all_reduce_tg`'s redundant final `to_memory_config`; `all_reduce_async` already receives the exact requested output memory config.
- Added a host assertion that a same-dtype, sharded all-reduce path performs no post-collective memory conversion.
- Added hardware-only synchronization immediately after `all_reduce_async` returns, before reshape or later decode output conversion.
- Focused MLP/Galaxy/Prefetcher result: `55 passed in 8.14s`; compilation and `git diff --check` pass.

## Hardware checkpoint: deadlock confirmed inside all-reduce primitive

- The Llama decode retry completed both reduce-scatter stages and persistent all-gather, then blocked at synchronization placed immediately after `all_reduce_async` returned.
- This rules out `_all_reduce_tg` reshape, deallocation, and memory conversion as the source of the stall.
- Terminated the diagnostic process and successfully reset all 32 Galaxy devices with `tt-smi -r`.
- The proven production MLP call passes `use_optimal_ccl_for_llama=True` to this final axis-0 collective; the common MLP launch omitted that routing policy. The next retry aligns this exact flag.

## Verification checkpoint: production all-reduce routing policy aligned

- Added `use_optimal_ccl_for_llama=True` to MLP2D's final all-reduce launch and locked it with a host assertion.
- Focused MLP result: `32 passed in 4.91s`; compilation and `git diff --check` pass.

## Hardware checkpoint: warm reset produced incompatible mesh descriptor

- The first optimized-routing retry failed during fixture setup, before model or collective execution.
- After `tt-smi -r`, topology discovery exposed a `(16,2)` system mesh, which cannot be rotated into the required `(8,4)` logical mesh.
- This run provides no result for the all-reduce change. A full `tt-smi -glx_reset` is required before the serialized retry.

## Hardware checkpoint: optimized workers do not fix incompatible active fabric

- After the full tray reset restored `(8,4)`, the optimized-worker retry again completed both reduce-scatter stages and all-gather but stalled inside axis-0 all-reduce.
- The parallel C++ audit identified a device-setup mismatch: the common fixture converts `fabric_config=True` to `FABRIC_1D_NEIGHBOR_EXCHANGE`, while the collective explicitly requests `Topology.Ring`.
- The proven 6U model fixture instead normalizes this setting to `FABRIC_1D_RING`; current all-reduce validation does not reject the incompatible neighbor-exchange/ring pairing before device execution.
- Terminated the stalled process and performed a full Galaxy tray reset. The next retry uses explicit ring fabric in both MLP hardware cases.

## Hardware checkpoint: explicit ring fabric resolves collective deadlock

- With `FABRIC_1D_RING`, both reduce-scatter stages, persistent all-gather, and final axis-0 all-reduce completed explicit worker-subdevice synchronization.
- The first Llama decode invocation reached host readback without a reset, proving the prior stall was the neighbor-exchange/ring mismatch.
- Numerical correctness remains open: the end-to-end output produced PCC `0.00221497` against the Torch MLP reference.
- The next step is stage-level correctness isolation, starting with tensor mesh composition and projection/collective shard ordering; no further topology changes are indicated.

## Hardware checkpoint: decode corruption isolated to GCB-backed ring matmul

- Stage readback showed all-gather gated PCC `0.0291092`; per-row local comparisons were similarly low, ruling out host composition.
- Both reduce-scatter outputs were already incorrect, and direct reads of both preceding ring matmul outputs were uncorrelated with their exact per-device Torch partial products.
- Representative matmul input shards were exact (`PCC 1.0`) and DRAM weight shards were correct (`PCC ~0.999974`), isolating corruption to the GCB-backed matmul/prefetch launch path.
- The legacy launch stalls both sender and worker subdevices before `dram_prefetcher`, then switches to worker-only. Prefetcher2D currently omits the pre-launch all-subdevice stall; the next correction restores this two-step transition.

## Verification checkpoint: decode prefetch launch transition restored

- Prefetcher2D now stalls every decode-manager subdevice before launching `dram_prefetcher`, then applies the configured steady-state stall group after launch.
- Added a host assertion for the exact pre-launch all-subdevice and post-launch configured-stall sequence.
- Focused Prefetcher/Galaxy/MLP result: `55 passed in 8.20s`; compilation and `git diff --check` pass.

## Hardware checkpoint: standalone GCB matmul path remains invalid

- Retried after restoring the two-step stall transition; representative input and weight shards remained correct, while the first standalone ring matmul remained uncorrelated with exact partial products.
- The proven decode implementation consumes prefetched W1/W3 through fused `llama_rs_matmul`, not standalone `ttnn.linear` calls followed by generic reduce-scatter.
- MLP2D decode now uses that fused operation and the hardware plan allocates its exact global `(8,4,32,4096)` L1 packet workspace on the eight reference packet cores.

## Verification checkpoint: fused decode front half integrated

- Updated decode ownership tests for the fused W1/W3 reduction path while retaining straight-line W2/all-reduce behavior.
- Focused MLP/Prefetcher/Galaxy result: `55 passed in 8.27s`; compilation and `git diff --check` pass.

## Hardware checkpoint: fused decode reduction requires ring topology

- The first fused retry blocked inside `llama_rs_matmul` before returning to the diagnostic wrapper.
- Its shared reduce-scatter resource still requested Linear topology; the proven WH 6U fused path requests Ring on the active ring fabric.
- Terminated the bounded run, reset all 32 boards, and aligned the decode fused reduction resource to Ring for the next retry.

## Hardware checkpoint: fused decode reduction also requires four links

- The ring-topology retry remained blocked inside the fused primitive for more than 90 seconds.
- The remaining call-policy mismatch was the generic one-link reduce-scatter plan versus the proven fused MLP's four-link launch.
- Terminated the run, reset all 32 boards, and aligned decode fused reduce-scatter to four links.

## Hardware checkpoint: four-link fused call still stalls with BF16 input

- The four-link retry still remained inside `llama_rs_matmul` for more than one minute.
- The focused input helper preserved LazyWeight's BF16 default, while the proven specialized fused path receives BF8 input and MLP2D's decode activation policy is BF8.
- Terminated the run, reset all 32 boards, and made the hardware input conversion explicitly BF8.

## Hardware checkpoint: BF8 input does not resolve fused stall

- The explicit BF8 retry also remained inside `llama_rs_matmul` for more than one minute.
- Terminated the run and reset all 32 boards; the next diagnostic records the concrete semaphore, packet-buffer, link, topology, and subdevice arguments at the primitive boundary.

## Hardware checkpoint: lazy input materialization stalls after decode activation

- Coarse markers showed resource and module construction completed, decode resources activated, and module entry began, but execution never reached the fused primitive wrapper.
- The hardware harness was materializing its lazy host input only after the decode worker subdevice was stalled; production callers supply device-resident activations.
- The harness now materializes the caller-owned input before decode activation and passes the TTNN tensor directly to MLP2D.

## Hardware checkpoint: static resource key unblocks fused primitive

- The fused adapter was blocking while reading `input_tensor.shape` after the worker subdevice was stalled; replaced that dynamic metadata read with the resolved decode geometry key.
- `llama_rs_matmul` then launched and completed with BF8 input, the exact `(32,512)` eight-core packet workspace, four links, Ring topology, and worker subdevice 1.
- Execution next failed in activation/multiply with `Invalid subtile broadcast type`; the next diagnostic records all fused output shapes and layouts to verify return ordering.

## Hardware checkpoint: fused output contract identifies missing W3 reduction

- The fused primitive completed and returned output widths `3584`, `3584`, and `960`: two W1/W3 matmul projections followed by W1's axis-1 reduce-scatter result.
- The established Galaxy MLP contract confirms `llama_rs_matmul` fuses only W1 reduction; its W3 projection must pass through a second axis-1 reduce-scatter before gated multiplication.
- MLP2D decode now preserves the static resource key, returns W1-reduced plus W3-projection from the adapter, explicitly reduces W3, and uses the configured reduce-scatter memory layout for `mul`.

## Verification checkpoint: decode fused-contract coverage

- Added host coverage for the static fused resource key, primitive output ordering, the explicit W3 reduce-scatter, and projection ownership.
- Focused result: `4 passed, 29 deselected`; Python compilation and `git diff --check` pass.

## Design checkpoint: W3 reduction aligned with padded fused geometry

- Parallel C++ and reference audits independently confirmed that generic minimal reduce-scatter would yield Llama width `896`, while the specialized fused W1 reduction pads to width `960`.
- Decode W3 now uses `ttnn.experimental.llama_reduce_scatter` with the same static resource key, packet workspace, Ring topology, four links, and output memory configuration as fused W1.
- Added a host contract for the specialized W3 primitive call; the next hardware run will verify matching operand geometry before multiplication.

## Verification checkpoint: specialized W3 reduction contract

- Focused decode contracts pass: `4 passed, 30 deselected`.
- Python compilation and `git diff --check` pass after adding hardware-stage synchronization around the specialized W3 reduction.

## Hardware checkpoint: overlapping diagnostic processes invalidated retry

- Process inspection found two lingering bounded MLP pytest trees; the earlier shell pipeline had returned control while its nested pytest remained active, so the subsequent run violated hardware serialization.
- Terminated both exact timeout/pytest process trees. No result from either run is considered valid.
- The tray will be reset before a single unpiped pytest invocation whose session is tracked directly to completion.

## Hardware checkpoint: both specialized reductions complete

- A clean serialized run completed fused W1 reduce-scatter and the separate specialized W3 reduce-scatter without deadlock.
- The test failed before entering all-gather, then stalled during fixture cleanup; terminated the exact process tree.
- The next diagnostic records W3's logical/padded shape and memory layout immediately before multiplication to distinguish a remaining geometry mismatch from an elementwise layout constraint.

## Hardware checkpoint: reduced operand geometry now matches

- W1 and W3 both reach multiplication with logical and padded shape `(1,1,32,960)` and identical L1 width-sharded `[32,32]` memory configuration.
- The test still fails before all-gather, so the original width mismatch is fixed but another call-site contract remains.
- Added a test-only `ttnn.mul` wrapper that reports both dtype/tile contracts and prints any exception immediately before fixture cleanup.

## Hardware checkpoint: multiply completes; padded all-gather key was stale

- The call-site diagnostic confirmed both BF8 tile operands match and `ttnn.mul` returns successfully.
- Decode then failed before entering all-gather because its resource plan used unpadded width `hidden_dim/32` (`896` for Llama), while specialized reduction emits padded width `960`.
- The hardware resource plan now derives decode reduced width with the same 24-core shard padding rule as `LlamaReduceScatterDeviceOperation`; Qwen remains `800` because its local width divides the shard width exactly.

## Verification checkpoint: padded resource-plan update

- Hardware test compilation and `git diff --check` pass after the decode all-gather key correction.

## Hardware checkpoint: optimal all-gather requires unpadded persistent shape

- With key width `960`, optimal all-gather completed, but the auto-sized persistent output retained width `3840`; gated PCC was `0.0208`, and W2 correctly rejected K `3840` against weight K `3584`.
- The proven Galaxy CCL preallocates BINARY_MUL output at unpadded logical width `3584` (Qwen `3200`) while accepting the padded reduced shard as input.
- Decode all-gather now selects resources by padded input width but preallocates its persistent output at unpadded `hidden_dim/8`, allowing the optimal kernel to discard per-column padding and restore W2's contract.

## Verification checkpoint: unpadded persistent all-gather plan

- Hardware test compilation and `git diff --check` pass after separating the padded selector key from the unpadded persistent output shape.

## Hardware checkpoint: decode executes end to end; projection data remains incorrect

- Optimal all-gather now returns unpadded W2 width `3584`; W2 and final ring all-reduce both complete, and fixture teardown closes all 32 devices cleanly.
- Numerical checks still fail: gated all-gather PCC is about `0.141` and final PCC is about `-0.00058`.
- Added per-device diagnostics comparing each reduced W1/W3 shard's first `896` meaningful channels against its exact Torch row/column segment to isolate the first corrupt projection.

## Diagnostic checkpoint: raw fused matmul comparison added

- Closed and synthesized the completed fused-output contract, subtile-failure, and host-contract agents.
- Dispatched three independent hardware-free audits, each with a dedicated goal, for raw device mapping, GCB ordering, and padded reduce-scatter ownership.
- Added a test-only `llama_rs_matmul` checkpoint that compares raw W1/W3 outputs on all 32 devices with exact row/column Torch partial products before reduction.
- Next: run the Llama decode case alone and use those raw PCCs to localize the remaining numerical failure.

## Hardware checkpoint: corruption precedes reduce-scatter

- The isolated Llama decode hardware case completed end to end and closed all 32 device drivers cleanly.
- All 32 raw W1 partial-product PCCs were approximately zero (range about `-0.0033` to `0.0079`); all 32 raw W3 PCCs were likewise approximately zero (range about `-0.0108` to `0.0062`).
- This rules out padded reduce-scatter ownership as the first numerical defect. The fused matmul is consuming incorrect input or weight data before reduction.
- Next: compare the common Prefetcher2D registration/address sequence and fused launch arguments with the proven legacy Galaxy path, then run one narrowly targeted isolation experiment.

## Audit checkpoint: padded reduce-scatter ownership resolved

- The padded-layout agent established that Llama's 256 channels of padding form one global tail rather than four 64-channel local tails.
- Output columns 0-2 each own 960 valid channels; column 3 owns 704 valid channels followed by 256 padded channels. Both specialized reduce-scatter operations use this ownership.
- The finding is documented in `tttv2_llama_padded_rs_channel_audit_20260819T055625Z.md`; it does not explain raw projection corruption.
- Added a test-only isolation override that launches `llama_rs_matmul` with `global_cb=None`, bypassing prefetched weight consumption while retaining the same input, weights, program, and collective resources.

## Audit checkpoint: raw mapping and prefetch order validated

- The shard-mapping audit independently confirmed the diagnostic formula, no-transpose convention, and `row * 4 + column` device ordering for both fused raw outputs. Findings are in `tttv2_llama_rs_matmul_shard_mapping_audit_20260819_0916.md`.
- The prefetch audit found no registration, packed-address, or consumption-order mismatch: common and legacy paths consistently use W1, W3, W2. Findings are in `tttv2_mlp2d_prefetch_gcb_order_audit_20260819_055725.md`.
- The no-GCB isolation is unsupported for the three-input fused program (`Must have exactly 2 input tensors, got: 3`) and did not execute. The resulting stalled pytest process tree was terminated, and the test-only override was removed.
- Next: reset hardware, then isolate runtime GCB pointer/geometry behavior against the proven legacy MLP hardware test.

## Hardware checkpoint: reset complete; legacy test blocked by model environment

- `tt-smi -glx_reset` completed successfully and restored all 32 Galaxy chips after the unsupported no-GCB launch.
- The legacy MLP unit test opened and closed the Galaxy cleanly but stopped before model construction because neither `LLAMA_DIR` nor `HF_MODEL` is configured; it provided no fused-kernel correctness signal.
- Next: compare packet-buffer allocation, semaphore sequencing, and GCB launch geometry directly with the legacy implementation using the random-weight common hardware case.

## Implementation checkpoint: decode manager now owns prefetch allocations

- Found a lifecycle mismatch with the proven legacy setup: common `Prefetcher2D.seal()` allocated the global CB and address metadata before loading the decode subdevice manager, and the MLP test subsequently allocated its L1 input before `activate()`.
- Updated `seal()` to load the decode manager and stall both decode subdevices before allocating decode prefetch resources, matching legacy allocation order.
- Added separate loaded-mode tracking so cleanup clears a manager prepared during sealing and failed activation rollback resets that state.
- Added host coverage for the required decode-manager/stall ordering. Next: run focused Prefetcher2D host tests, then retry the isolated Llama hardware case.

## Verification checkpoint: decode-allocation lifecycle host coverage

- Focused Prefetcher2D and MLP2D host tests reached `50 passed, 1 failed`.
- The sole failure was the pre-existing activation test expecting its first worker-only stall at index 1; sealing now intentionally contributes an earlier all-subdevice preparation stall.
- Updated the assertion to require the exact sequence: seal stalls all decode subdevices, activation stalls all while launching prefetch, then activation releases to the configured worker stall group.

## Verification checkpoint: decode-allocation lifecycle host tests pass

- Focused Prefetcher2D and MLP2D suites pass: `51 passed`.
- `git diff --check` passes for the lifecycle implementation, host test, MLP hardware diagnostic, and main work log.
- Next: run the isolated Llama decode hardware case and inspect raw W1/W3 partial-product PCCs before reduction.

## Hardware checkpoint: manager-owned allocation does not fix raw projections

- The isolated Llama decode run completed and closed all 32 devices cleanly after the lifecycle change.
- Raw W1/W3 partial-product PCCs and final output were bit-for-bit identical to the prior failure, ruling out decode-manager allocation timing as the numerical cause.
- The fused launch repeatedly warned that `program_config.allowed_worker_cores` is empty and is being auto-populated from `compute_with_storage_grid_size`; fused CCL callers bypass normal matmul program-config normalization.
- Next: determine the required allowed-worker core set for the 24-core prefetch ring program and provide it explicitly if the binding supports it.

## Diagnostic checkpoint: allowed-worker fallback assessed

- The matmul binding exposes `allowed_worker_cores`, but normal normalization would populate the same dense rectangle implied by `(8,3)`; setting it explicitly would only suppress the warning and does not explain the unchanged corruption.
- Expanded the raw diagnostic to compare fused output 0 and output 1 independently against both W1 and W3 partial-product references on all devices.
- Next: one isolated Llama run to detect or rule out a GCB stream-slot swap.

## Hardware checkpoint: GCB W1/W3 slot swap ruled out

- The isolated Llama run completed and closed all 32 devices cleanly.
- Fused output 0 and output 1 were each uncorrelated with both W1 and W3 partial references; no one-slot W1/W3 swap or reversal is present.
- Next: compare ordered ring/receiver core construction with legacy, because deterministic block permutation at the 24-core output boundary would make raw logical readback and downstream reduction incorrect while preserving stable values.

## Root-cause checkpoint: ring core order was discarded

- Verified directly that `CoreRangeSet` preserves constructor sequence: building the same point cores from a list versus a set produces unequal objects with different iteration orders and hashes.
- Legacy constructs the 24-core input ring and 24-core output receiver ring from ordered lists. The common MLP hardware config used Python sets, sorting both physical shard streams and breaking the kernel's ring block order.
- Updated input/output ring and packet-buffer core sets to preserve declared sequence. Updated the prefetch sender address grid to preserve active sender order as well.
- Next: host syntax/diff verification, then rerun the isolated Llama decode case and inspect raw PCC recovery.

## Verification checkpoint: ordered ring construction passes host tests

- MLP hardware test and shared Galaxy hardware helper compile after preserving ring order.
- Focused Prefetcher2D and MLP2D host suites pass: `51 passed`; `git diff --check` is clean.
- Next: isolated Llama decode hardware validation.

## Hardware checkpoint: ordered ring fix qualifies Llama decode

- The isolated Llama decode test passed both repeated invocations end to end and closed all 32 devices cleanly.
- Every raw W1 and W3 partial-product PCC is about `0.9996`; the gated all-gather PCC is about `0.99888`.
- Root cause was loss of the explicit 24-core ring/receiver ordering when constructing `CoreRangeSet` from Python sets.
- Next: remove temporary diagnostic wrappers and prints, then run clean Llama and Qwen repeated decode cases sequentially.

## Cleanup checkpoint: MLP hardware test restored

- Removed all temporary collective monkeypatches, forced stage synchronizations, raw tensor PCC dumps, and diagnostic prints from the Galaxy MLP test.
- Restored the decode reference to the shared `_torch_mlp` path while retaining two repeated production-like invocations and final-output PCC validation.
- Next: run syntax, diff, and focused host verification before clean sequential hardware qualification.

## Verification checkpoint: cleaned MLP path passes host tests

- Python compilation and `git diff --check` pass after removing the diagnostics.
- Focused Prefetcher2D and MLP2D host suites pass: `51 passed`.
- Next: run clean repeated decode qualification for both Llama and Qwen sequentially on the Galaxy.

## Hardware checkpoint: clean decode exposes masked async defect

- The clean decode selector timed out after 15 minutes without completing the first Llama case; pytest output stopped after the fused matmul launch warnings and produced no test report.
- Sustained host CPU activity did not represent Qwen progress: the first test never emitted PASS. The prior diagnostic test's forced per-stage synchronizations therefore masked an asynchronous ordering or tensor-lifetime defect.
- The external timeout interrupted an open mesh, so the Galaxy must be reset before further hardware work.
- Next: reset hardware and isolate the earliest synchronization boundary required by the clean path without restoring broad diagnostic monkeypatching.

## Recovery checkpoint: Galaxy reset and readback boundary isolated

- `tt-smi -glx_reset` completed successfully and re-initialized all 32 boards after the timed-out mesh process.
- The hardware test now synchronizes only the CCL worker subdevice after module execution and before host tensor composition; the persistent decode prefetch kernel remains active on its separate sender subdevice.
- Dispatched parallel dedicated-goal audits for prefetch/readback lifecycle and async CCL/deallocation behavior.
- Next: run the isolated clean Llama case with this single qualification-harness boundary.

## Hardware checkpoint: worker-scoped readback synchronization fixes timeout

- The isolated clean Llama repeated-decode case passes in `22.93s` with one worker-subdevice synchronization before host composition; all 32 devices close cleanly.
- This demonstrates that the module pipeline completes correctly and the timeout came from host readback attempting a whole-device synchronization while the persistent prefetch kernel was intentionally running on the sender subdevice.
- The synchronization belongs in the hardware qualification harness, not the production module hot path.
- Next: qualify the Qwen decode geometry with the same worker-scoped readback boundary.

## Audit checkpoint: async pipeline and persistent readback contract

- Both dedicated-goal audits agree that the production MLP collective order, semaphore cycling, and enqueue-before-deallocate lifetimes match legacy; the final worker synchronization was the diagnostic boundary that masked whole-device host readback.
- Findings are documented in `tttv2_mlp_async_ccl_audit_20260819T063505Z.md` and `tttv2_mlp_prefetch_readback_audit_20260819_063533.md`.
- The prefetch audit recommends exposing worker-scoped synchronization through the Galaxy resource owner rather than reaching through the CCL collaborator in tests; this API cleanup remains pending integration.
- The async audit also identified unreleased reduced W1/W3 intermediates after `mul`, which is a repeat-run L1 retention risk and requires a scoped ownership fix.
- Next: finish the active Qwen qualification run, then address any geometry-specific failure and the intermediate ownership gap.

## Hardware checkpoint: Qwen decode has a geometry-specific stall

- The isolated Qwen repeated-decode case timed out after five minutes at the worker-subdevice completion boundary; it never reached host composition or emitted a test result.
- Unlike the resolved Llama readback timeout, this means at least one Qwen worker operation did not complete.
- The timed-out process interrupted an open mesh, so another Galaxy reset is required.
- Next: reset all boards and use temporary worker-scoped stage waits to identify the first incomplete Qwen collective.

## Hardware checkpoint: Qwen stall isolated to BF8 final all-reduce

- Temporary worker-scoped stage waits show that Qwen completes fused W1/W3 plus W1 reduce-scatter, standalone W3 reduce-scatter, and gated all-gather; it stalls in the final axis-0 `all_reduce_async`.
- The repository's 6U CCL qualification matrix explicitly uses BF16 for the Qwen `[1, 1, 32, 1280]` FF2 all-reduce, while the common hardware case configured Qwen weights, activations, outputs, and persistent collective resources as BF8.
- The stage-local timeout again requires a Galaxy reset.
- Next: reset hardware, bind the Qwen precision recipe as BF16 through module config and resources, remove diagnostics, and rerun the isolated case.

## Implementation checkpoint: Qwen BF16 recipe and ownership preserved

- Reset and re-initialized all 32 boards successfully.
- Bound Qwen weights, activation/multiply/CCL dtypes, inputs, and collective persistent resources to BF16 through the explicit hardware-case module/resource config; Llama remains BF8.
- Removed the temporary stage instrumentation.
- Rejected the proposed intermediate deallocations after host ownership tests proved those tensors are borrowed CCL persistent outputs; existing explicit ownership semantics remain intact.
- Python compilation, `git diff --check`, and focused Prefetcher2D/MLP2D host suites pass: `51 passed`.
- Next: rerun isolated Qwen repeated decode with the qualified BF16 recipe.

## Hardware checkpoint: BF16 is insufficient for Qwen all-reduce

- The clean BF16 Qwen case still timed out at five minutes, so matching the qualified primitive dtype does not by itself resolve the final all-reduce stall.
- Comparison with `test_new_all_reduce.py` exposed remaining order loss: known-good subdevice and persistent-buffer core grids use ordered lists, while the common hardware helper still constructs equivalent multi-range `CoreRangeSet` values from sets.
- The timeout requires another Galaxy reset.
- Next: reset, make the remaining CCL/subdevice core sequences deterministic, and independently diff the Qwen all-reduce setup against the known-good 6U unit case.

## Audit checkpoint: Qwen all-reduce setup diff

- Reset and re-initialized all 32 boards, then changed remaining multi-range sender/worker/persistent-buffer core sets to preserve the known-good list order.
- The dedicated-goal field audit is recorded in `tttv2_qwen_final_all_reduce_audit_20260819T065535Z.md`.
- Input/output logical and padded shapes, BF16 dtype, ring coordinates, topology, links, output sharding, and mesh mapper match the known-good 6U Qwen primitive case.
- The major remaining difference is intentional optimized-path geometry: common MLP uses `use_optimal_ccl_for_llama=True` with a 50-core buffer, while the generic primitive test uses the default path with a 10-core buffer. This exact hybrid Qwen contract is not independently qualified.
- Python compilation, diff validation, and focused host suites pass after ordered core construction: `51 passed`.
- Next: determine whether ordered resource grids fix the integrated Qwen case before changing the injected all-reduce strategy.

## Hardware checkpoint: resource ordering alone does not fix Qwen

- The ordered-grid BF16 Qwen run still timed out after three minutes, so the final stall is not caused solely by worker/persistent-buffer `CoreRangeSet` ordering.
- The parallel precision/padding audit is recorded in `tttv2_qwen3_32b_wh_galaxy_mlp_decode_audit_20260819T065815Z.md`.
- It found a concrete upstream geometry mismatch: common Qwen W1/W3 DRAM storage pads local N=3200 to 3456, but the fused ring contract and legacy implementation require physical N=3840. Llama already lands on 3840 under either alignment.
- It also established the legacy recipe: Qwen sharded weights are BF16 while decode activations, multiply, CCL outputs, and optimized 50-core persistent buffers remain BF8.
- The timeout requires another reset.
- Next: reset, align DRAM N to the 24-core ring and restore the legacy Qwen mixed-precision recipe before retesting.

## Hardware checkpoint: Qwen repeated decode qualifies

- Reset and re-initialized all 32 boards, then aligned DRAM output-channel storage to the 24-core ring (`3200 -> 3840` physical channels for Qwen).
- Restored the proven Qwen mixed-precision contract: BF16 sharded weights with BF8 activations, multiply, collective outputs, and optimized persistent buffers.
- The isolated Qwen decode test passes both repeated invocations in `13.67s` and closes all devices cleanly.
- The final worker wait had surfaced an upstream physical-padding/W2 dependency rather than an intrinsic Qwen all-reduce defect.
- Next: rerun combined Llama and Qwen decode qualification, then run prefill 128/2048 sequentially.

## Hardware checkpoint: combined MLP decode exit case passes

- Clean combined Llama and Qwen decode qualification passes: `2 passed, 2 deselected in 33.50s`.
- Each geometry executes two repeated batch-32 invocations with final-output PCC validation and worker-scoped readback synchronization.
- Both mesh fixtures close all 32 devices cleanly.
- Next: run Llama and Qwen prefill sequence lengths 128 and 2048, each repeated twice, in one serialized hardware process.

## Hardware checkpoint: prefill requires dual weight layouts

- Both Llama and Qwen prefill cases fail immediately at the first W1 linear with `Input B memory layout must be INTERLEAVED, got WIDTH_SHARDED`; both mesh fixtures still close cleanly.
- Decode's fused ring path requires DRAM width-sharded weights, while prefill's standard matmuls require DRAM interleaved weights. The legacy module keeps both representations explicitly.
- Next: add optional prefill `LazyWeight` fields and mode-specific lazy materialization to MLP2D, then supply interleaved copies through hardware config without changing the decode path.

## Implementation checkpoint: dual decode/prefill weight layouts

- Added explicit optional prefill W1/W2/W3 LazyWeights and mode-specific materialization to `MLP2D`; decode retains ring-sharded DRAM weights while prefill resolves distinct interleaved DRAM weights.
- Corrected the hardware case wiring so only the prefill qualification allocates the interleaved copies, with deduplicated cleanup covering both representations.
- Python compilation and `git diff --check` pass.
- Focused MLP2D, Prefetcher2D, Galaxy CCL, and Galaxy resource host suites pass: `72 passed in 10.83s`.
- Next: run the isolated Llama prefill 128/2048 repeated hardware case, then qualify Qwen after the Galaxy is released cleanly.

## Hardware checkpoint: prefill requires composite all-reduce

- The isolated Llama prefill case now completes W1/W3, both axis-1 reductions, gating, gather, and W2, then fails at final `all_reduce_async` because its input is DRAM interleaved rather than width-sharded.
- The fixture closes all 32 devices cleanly after the assertion.
- The legacy Galaxy MLP confirms this is a strategy mismatch: decode uses optimized `all_reduce_async`, while prefill implements all-reduce as axis-0 reduce-scatter followed by axis-0 all-gather and supports interleaved DRAM tensors.
- Next: generalize the shared reduce-scatter/all-gather helpers to explicit axes, register exact prefill axis-0 resources, and route prefill through the composite strategy.

## Implementation checkpoint: prefill composite all-reduce

- Generalized MLP2D reduce-scatter/all-gather internals to select an explicit cluster axis while preserving the existing axis-1 wrappers.
- Prefill final reduction now performs axis-0 reduce-scatter followed by axis-0 all-gather; decode remains on the qualified optimized `all_reduce_async` path.
- Added exact sequence-keyed axis-0 persistent resources for prefill lengths 128 and 2048.
- Added a host regression proving prefill collective order and exclusion of `all_reduce_async`.
- Python compilation, `git diff --check`, and focused host suites pass: `73 passed in 11.08s`.
- Next: rerun isolated Llama repeated prefill qualification on hardware.

## Contract checkpoint: axis-0 reduce-scatter resource keys

- The first composite-path rerun failed before module execution because `GalaxyResourceKey` rejected reduce-scatter on cluster axis 0.
- Widened the shared CCL schema to allow reduce-scatter on Galaxy axes 0 and 1 and retained rejection for non-mesh axis 2.
- Updated the CCL validation test to cover the new MLP output key.
- The fixture closed cleanly; Python compilation, diff validation, and focused host suites pass: `73 passed in 11.06s`.
- Next: rerun isolated Llama prefill with axis-0 resources now constructible.

## Hardware checkpoint: prefill executes with incorrect collective semantics

- Llama prefill 128 completes end to end but returns PCC `0.173187`, failing before the remaining sequence/repeat cases.
- The fixture releases all devices cleanly.
- Comparison with the proven ring path found that prefill reduce-scatter intermediates had an erroneous extra leading dimension and all prefill collectives were configured as Linear despite Ring fabric execution.
- Corrected prefill intermediate/output shapes, Ring topology, four-link policy, and replicated mesh allocation on both axes.
- Python compilation, diff validation, and focused host suites pass after the resource correction: `73 passed in 10.88s`.
- Next: rerun isolated Llama prefill to validate corrected ring resource semantics.

## Hardware checkpoint: W1/W3 persistent output alias identified

- Corrected ring shapes/topology leave Llama prefill PCC effectively unchanged at `0.173059`, ruling those settings out as the primary semantic defect.
- W1 and W3 axis-1 reductions currently select the same exact geometry key and therefore the same persistent output buffer; both async results remain live until gating, so W3 can overwrite W1 before `mul` consumes it.
- Legacy CCL allocates distinct `FF1` and `FF3` buffers for this reason.
- The fixture closes cleanly after the PCC assertion.
- Next: add explicit stage sequence keys to resource selection and allocate distinct W1/W3 persistent resources before rerunning hardware.

## Implementation checkpoint: stage-keyed prefill collectives

- Extended the MLP collective selector contract with an optional stage key backed by `GalaxyResourceKey.sequence_key`.
- Prefill now selects distinct axis-1 reduction resources for W1 and W3, plus independently keyed gated gather and final axis-0 composite resources.
- Decode continues to use its existing geometry-derived sequence keys.
- Updated hardware resource plans and host selector/forward contracts for the explicit stage identity.
- Python compilation, diff validation, and focused host suites pass: `73 passed in 10.84s`.
- Next: rerun isolated Llama prefill to verify removal of the live persistent-buffer alias.

## Hardware checkpoint: final prefill reduce-scatter handle is unallocated

- With stage-keyed resources, Llama prefill advances through W1/W3 gating and W2 but fails when the final axis-0 all-gather receives an unallocated tensor handle from `reduce_scatter_minimal_async`.
- This is a new failure after removing W1/W3 aliasing and localizes the remaining problem to the final composite boundary.
- The common helper still passes decode-oriented `intermediate_memory_config`, chunk, worker, and channel tuning on prefill calls; the qualified Galaxy prefill primitive omits these except for a sequence-dependent worker count.
- The fixture closes all devices cleanly.
- Next: make reduce-scatter invocation policy mode-specific and match the qualified ring prefill signature.

## Implementation checkpoint: qualified prefill reduction arguments

- Split `reduce_scatter_minimal_async` arguments by execution mode.
- Prefill now matches the qualified ring path and selects one worker per link at sequence 128 and four above 128; decode retains its existing intermediate-memory, chunk, worker, and channel tuning.
- Python compilation, diff validation, and focused host suites pass: `73 passed in 10.83s`.
- Next: rerun isolated Llama prefill against the corrected final composite call.

## Hardware checkpoint: axis-0 minimal output remains unsupported

- The qualified prefill argument set still produces an unallocated output handle from the final axis-0 `reduce_scatter_minimal_async`; the following all-gather rejects it immediately.
- Axis-1 minimal reductions remain operational, so the failure is specific to this axis-0 minimal-kernel combination rather than general resource allocation.
- The fixture closes all devices cleanly.
- Next: retain injected axis-0 topology/link strategy but use the public reduce-scatter/all-gather primitives for the final prefill composite, matching the stable fallback already present in the Galaxy CCL implementation.

## Implementation checkpoint: stable final prefill composite

- Added an explicit persistent/stable strategy switch to MLP2D collective helpers.
- Prefill W1/W3 and gated collectives remain on stage-keyed persistent async resources; only the final axis-0 reduction/gather uses public primitives.
- The stable final operations still consume the injected resource topology, link count, memory policy, mesh axis, and worker subdevice.
- Python compilation, diff validation, and focused host suites pass: `73 passed in 11.04s`.
- Next: qualify isolated Llama prefill numerical correctness and repeated sequence behavior.

## Hardware checkpoint: Llama repeated prefill qualifies

- Isolated Llama prefill passes sequence lengths 128 and 2048, each invoked twice, in `40.32s` with PCC validation on every output.
- The stage-keyed W1/W3 resources restore numerical correctness, and the stable axis-0 final composite avoids the minimal-kernel unallocated output.
- The mesh fixture closes all 32 devices cleanly.
- Public all-gather emits a September 2026 deprecation warning for explicit topology/link arguments; this does not affect correctness but should be tracked for later API migration.
- Next: run the same repeated prefill qualification for Qwen geometry on the released Galaxy.

## Hardware checkpoint: Qwen repeated prefill qualifies

- Isolated Qwen prefill passes sequence lengths 128 and 2048, each invoked twice, in `31.54s` with PCC validation on every output.
- BF16 interleaved Qwen weights and BF8 execution/collectives satisfy the representative geometry contract.
- The mesh fixture closes all 32 devices cleanly.
- Next: run the complete MLP2D Galaxy file to verify decode and prefill resources coexist across both representative models.

## Hardware checkpoint: complete MLP2D matrix qualifies

- The complete MLP2D Galaxy hardware file passes all four representative cases: Llama decode, Llama prefill 128/2048, Qwen decode, and Qwen prefill 128/2048.
- Every case invokes the module twice in-process and validates numerical output; the combined result is `4 passed in 90.93s`.
- Decode and prefill resource plans coexist across both model geometries, and all 32 devices close cleanly at fixture teardown.
- Next: refresh the Milestone A status ledger and audit the remaining hardware and regression exit gates.

## Audit checkpoint: remaining hardware gates collect cleanly

- Refreshed the Milestone A status ledger with the qualified MLP2D matrix and removed MLP2D from the open hardware list.
- The remaining standalone WH files collect nine cases cleanly: two Attention2D model cases, six RMSNorm2D cases, and one Sampling2D case.
- No pytest or `tt-smi` process is using the Galaxy. Four independent host-only audit agents are reviewing Attention, distributed RMSNorm, Prefetcher/Galaxy ownership, and physical-32 trace plus regression coverage under dedicated goals.
- Attention2D is the next serialized gate because each model case already covers repeated decode, prefill 128/2048, output PCC, and KV-cache PCC through production resources.
- Next: run the Llama Attention2D case alone with a hard process timeout.

## Hardware checkpoint: Attention prefetch placement rejected

- The isolated Llama Attention2D case opened the full mesh but segfaulted in `ttnn.dram_prefetcher` during decode activation, before any Attention2D operation executed.
- Both registered projection weights are DRAM-interleaved. This reproduces the previously diagnosed MLP prefetch failure class: the WH block prefetcher requires DRAM width-sharded decode weights with programmable reader banks.
- Because the process could not run fixture teardown, `tt-smi -r` reset and reinitialized all 32 boards successfully.
- The qualified Galaxy recipe uses separate DRAM-sharded decode ring weights and interleaved prefill weights, with ring-specific projection program and memory configs.
- Next: add mode-specific Attention2D weight materialization and align the decode projection recipe before rerunning hardware.

## Integration checkpoint: Attention ownership and mode-specific weights

- Synthesized four dedicated-agent audits covering Attention2D, distributed RMSNorm2D, Prefetcher/Galaxy ownership, and physical-32 trace plus 1D regressions.
- Added public worker-scoped `GalaxyResources.synchronize(mode)` and hardware-adapter forwarding; MLP hardware readback no longer reaches through private CCL context state.
- Corrected Attention all-reduce persistent buffers to preserve input shape and added an explicit borrowed-output contract so resource-owned CCL buffers are not deallocated as module transients.
- Added independent DRAM-sharded decode and interleaved prefill projection weights. Only decode weights enter the persistent prefetch queue.
- Targeted host verification passes `105 passed in 15.60s`; Python compilation and `git diff --check` pass.
- Distributed RMSNorm audit recommends prefill-128 as the next norm experiment and defers fused decode pending ownership/topology corrections. Physical-32 host planning passes, but a real WH trace test is still missing and the broad 1D regression remains outstanding.
- Next: rerun the isolated Llama Attention2D case and localize the first actual module-stage result.

## Hardware checkpoint: post-reset Galaxy topology degraded

- The Attention retry failed during mesh fixture setup, before opening the requested `(8,4)` mesh or executing module code.
- Auto-discovery now reports a `16x2` system mesh and one physical edge with only three active channels; the 32-node Galaxy graph cannot be mapped under strict validation.
- The failed setup closed device drivers cleanly and produced no module qualification result.
- Next: run the Galaxy-specific reset and require healthy topology discovery before any further serialized TT test.

## Hardware checkpoint: Attention reaches decode projection

- `tt-smi -glx_reset` restored healthy `4:32` logical and physical Galaxy adjacency and reinitialized every board.
- After separating WQKV and WO memory placement policy, the Llama retry starts the production prefetch session and reaches the first decode `ttnn.linear` call.
- TTNN rejects the generic program config because its `(8,1)` compute grid does not fit the column-dispatch `(7,10)` worker grid. This is a clean recipe error, not a numerical result.
- Cleanup did not return after the projection exception; the externally bounded pytest process was terminated and requires a device reset before reuse.
- Next: reset, replace the generic decode projection config with the proven Galaxy ring grid/memory recipe, and rerun the isolated case.

## Hardware checkpoint: Attention decode ring recipe executes to assertion

- Ported the qualified Galaxy decode projection recipe into the Attention2D hardware test: 24 ordered ring/receiver cores, padded 768-element local widths, `(8,3)` gather-in0 programs, global-CB receivers, ring-sharded activations, and projection-specific resource geometries.
- The isolated Llama case now advances beyond the former column-dispatch grid rejection and reaches a pytest failure during the first decode invocation.
- TT runtime cleanup deadlocks after the assertion and prevents pytest from flushing the traceback; the externally bounded process was terminated and `tt-smi -glx_reset` successfully restored all 32 boards.
- This is a test-diagnostic blocker rather than a qualification result. The next run will expose stage/failure details before resource teardown and then correct the first failing decode contract.
- Next: run one instrumented decode invocation, capture the exact failing operation or PCC/cache assertion, and continue serialized hardware qualification.

## Hardware checkpoint: Attention resource key uses physical padding

- Failure-local diagnostics exposed the first decode exception before the teardown deadlock: axis-1 all-reduce could not find geometry `(1,1,32,1280)`.
- The registry incorrectly used the projection's padded 1536-element shard width as tensor identity. TTNN preserves the logical 1280-element QKV width while using padded physical storage underneath.
- Corrected decode collective plans to key and allocate persistent outputs by logical QKV/output shapes; the padded widths remain confined to projection program and memory configurations.
- The failed process was terminated and `tt-smi -glx_reset` reinitialized all 32 boards successfully.
- Next: rerun Llama decode and localize the next module stage with logical collective geometry.

## Hardware checkpoint: Attention all-reduce requires width-sharded L1

- With logical resource keys, the first decode projection and resource lookup succeed.
- Optimized `all_reduce_async` then rejects the interleaved persistent buffer before launch: its buffer tensor must be width-sharded.
- The qualified Galaxy Attention path converts QKV to 10 one-head L1 shards, emits residual output to model-width L1 shards, and allocates each persistent buffer across the complete 50-core worker grid with per-shard width multiplied by the collective axis size.
- The failed process was terminated and another Galaxy reset completed successfully.
- Next: encode those axis-specific all-reduce layouts in the injected resource plan and rerun decode.

## Implementation checkpoint: Attention decode CCL layouts

- Added axis-specific decode all-reduce policy to the hardware adapter while keeping ownership in `GalaxyResources`.
- Axis 1 converts QKV to ten 128-wide head shards; axis 0 converts projection output to 128-wide model shards. Both use Ring topology, four links, the worker subdevice, and optimal Llama CCL selection.
- Persistent BF8 interim buffers are width-sharded across the complete 50-core worker grid and sized by the corresponding four- or eight-device ring cardinality.
- Attention's declared decode output placement now matches the axis-0 L1 result, and temporary conversion tensors are released after enqueue.
- Python compilation, diff validation, and the focused Attention host suite pass: `62 passed in 9.21s`.
- Next: rerun isolated Llama decode with the qualified CCL memory contracts.

## Hardware checkpoint: QKV CCL interim buffer underallocated

- Axis-1 input conversion and width-sharded persistent-buffer validation now pass.
- Program creation fails because BF16 all-reduce requests a 32768-byte global circular buffer while the four-device BF8 minimum shard provides only 17408 bytes.
- The qualified shared MLP path overprovisions optimized all-reduce interim shards to width 1024; applied the same minimum to both Attention axes so physical storage satisfies output-format CB sizing.
- The failed process was terminated and all 32 boards were reset successfully.
- Next: rerun decode with the 1024-wide persistent interim shards.

## Hardware checkpoint: decode head creation leaves worker subdevice

- The overprovisioned QKV all-reduce compiles and launches, advancing decode into `nlp_create_qkv_heads_decode`.
- Head creation fails subdevice validation because an interleaved DRAM output lets TTNN select a kernel grid that is not wholly contained in the active 50-core worker subdevice.
- Changed decode head outputs to the qualified height-sharded L1 layout over the exact worker core ranges; Q/K/V shards remain one tile high by one head wide.
- The failed process was terminated and Galaxy reset completed across all 32 boards.
- Next: rerun decode and validate head creation, cache update, and SDPA in sequence.

## Hardware checkpoint: contiguous cache update API mismatch

- Height-sharded head creation succeeds, advancing decode through Q/K/V creation and placement.
- The contiguous cache branch then calls scalar-only `ttnn.update_cache` with the replicated positions tensor; this build rejects the signature before a cache kernel launches.
- Current TTNN provides tensor-indexed contiguous cache writes through `ttnn.experimental.paged_update_cache` without a page table, and the transformer stack uses that path for per-user positions.
- Updated the contiguous branch to use tensor-indexed writes while retaining nonpaged SDPA selection; the paged-cache branch remains unchanged.
- The failed process was terminated and all Galaxy boards reset successfully.
- Next: verify the revised host contract and rerun Llama decode through cache update and SDPA.

## Verification checkpoint: tensor-indexed contiguous cache writes

- Added a host assertion that contiguous decode issues two tensor-indexed cache writes with the exact positions tensor and no page-table argument, followed by nonpaged SDPA.
- Python compilation and `git diff --check` pass; the focused Attention suite passes `62 passed in 9.24s`.
- Next: rerun the isolated Llama hardware case through cache update and decode SDPA.

## Hardware checkpoint: tensor-indexed cache input must remain sharded

- Tensor-indexed cache update selection is accepted, but its kernel rejects K/V after the module converts them to interleaved DRAM.
- Changed decode K/V placement to the same height-sharded worker-grid layout produced by head creation; `to_memory_config` now changes cache dtype without discarding the sharding required by cache update.
- The failed process was terminated and the full Galaxy reset completed successfully.
- Next: rerun decode through cache writes and SDPA with sharded K/V inputs.

## Hardware checkpoint: redundant cache dtype copy leaves worker subdevice

- Height-sharded K/V placement reaches cache preparation, but `ttnn.to_memory_config` still launches a same-layout BF16-to-BF8 copy whose default kernel grid extends outside the active worker subdevice.
- TTNN's paged-update-cache coverage explicitly qualifies BF16 input tensors writing BF8 caches, and the production transformer decode path passes those dtypes directly to the cache kernel.
- The required fix is to preserve K/V tensors that already have the declared worker-grid placement and let `paged_update_cache` perform the supported cache-format conversion.
- The failed process was terminated and a full Galaxy reset was started before further hardware use.
- Next: remove only the redundant same-placement copy, extend the host contract, and rerun the isolated Llama case.

## Verification checkpoint: cache kernel owns format conversion

- Decode now retains K/V tensors when their placement already equals `decode_kv_memory_config`; mismatched placement is still converted explicitly, without forcing cache dtype.
- This preserves the worker-grid sharding contract and delegates the separately qualified BF16-input/BF8-cache conversion to `paged_update_cache`.
- Python compilation and `git diff --check` pass; the focused Attention host suite passes `62 passed in 9.25s`.
- The Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun the isolated Llama hardware case through cache update and decode SDPA.

## Hardware checkpoint: decode reaches cache kernel validation

- The revised path advances through projection, QKV all-reduce, head creation, and redundant-copy elimination into `ttnn.experimental.paged_update_cache`.
- The cache primitive raises during device-operation validation; its exact fatal message was clipped by the first bounded output capture, after which fixture cleanup hung.
- The failed process was terminated and another full Galaxy reset was started.
- Next: rerun with a persistent bounded diagnostic log, retain the cache geometry fatal, and correct that contract before proceeding to SDPA.

## Hardware checkpoint: cache update requires one core per user

- Persistent diagnostic capture retained the exact validation: the K/V shard grid has 50 cores, but `paged_update_cache` requires `input_num_shards == num_users`, which is 32 for the representative batch.
- Head creation only requires its grid to remain inside the active worker subdevice; cache update additionally requires one height shard per user.
- The failed process was terminated and another full Galaxy reset was started.
- Next: use a 32-core subset of the worker subdevice for decode heads/K/V while retaining all 50 cores for collective persistent buffers, then rerun.

## Implementation checkpoint: 32-core decode head layout

- Decode head and K/V output memory now use 32 height-sharded cores selected from the active 50-core worker subdevice, matching the fixed batch and cache-kernel dispatch contract.
- Axis-specific collective inputs and persistent outputs retain their independently qualified 10-core, model-width, and 50-core layouts.
- Python compilation and `git diff --check` pass, and the Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun isolated Llama decode through cache update and localize the next SDPA or output-projection boundary.

## Hardware checkpoint: decode reaches SDPA placement

- The 32-core head layout passes both cache updates and advances into nonpaged decode SDPA.
- SDPA's default rectangular `(8,4)` physical placement includes a dispatch core and fails before kernel launch.
- SDPA supports explicit sub-core grids whose core count must equal the configured grid area; the 32-core head subset satisfies both that rule and the operation's minimum of one available core per batch row.
- The failed process was terminated and a full Galaxy reset was started.
- Next: bind decode SDPA's 32 logical slots to the existing 32-core worker subset and rerun through concat/output projection.

## Implementation checkpoint: worker-scoped decode SDPA

- Decode SDPA now maps its `(8,4)` logical compute area through `sub_core_grids` to the exact 32-core head/KV subset, avoiding dispatch columns while preserving one core per batch row.
- Python compilation and `git diff --check` pass, and the Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun isolated Llama decode through SDPA, concat, WO, axis-0 reduction, and numerical checks.

## Hardware checkpoint: decode reaches head concat

- Worker-scoped nonpaged SDPA compiles and completes, advancing decode into `nlp_concat_heads_decode`.
- Concat rejects the interleaved SDPA result; the hardware recipe still declares DRAM for `decode_sdpa_output_memory_config` even though the module correctly forwards that policy.
- The failed process was terminated and a full Galaxy reset was started.
- Next: use the 32-core head layout for SDPA output and rerun through concat and WO.

## Implementation checkpoint: sharded SDPA output

- The hardware recipe now emits decode SDPA output into the same 32-core height-sharded layout consumed by cache update and head concat.
- Python compilation and `git diff --check` pass, and the Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun isolated Llama decode through concat, WO, axis-0 reduction, and numerical checks.

## Hardware checkpoint: concat requires explicit sub-core policy

- Sharded SDPA output passes validation, but concat raises `bad optional access`: TTNN detects the nonzero/multi-range input grid as a sub-core-grid invocation while the module has no way to provide that grid.
- TTNN's concat implementation also ignores its public `memory_config` argument and derives a width-sharded output, so the module must enforce its declared WO-input placement after concat.
- The failed process was terminated and a full Galaxy reset was started.
- Next: add optional decode concat sub-core policy, pass the worker grid, normalize concat output to `decode_concat_memory_config`, and rerun through WO.

## Verification checkpoint: subdevice-aware head concat

- Added optional `decode_concat_sub_core_grids`; default callers retain their prior signature, while Galaxy decode passes the full worker grid required by TTNN's subcore concat factory.
- Attention2D now checks concat's actual placement and explicitly converts it to `decode_concat_memory_config` when the operation ignores that argument.
- Python compilation and `git diff --check` pass; the focused Attention host suite passes `62 passed in 9.05s`.
- The Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun isolated Llama decode through concat conversion, WO, axis-0 reduction, and numerical checks.

## Hardware checkpoint: decode reaches final axis-0 reduction

- Subdevice concat, explicit ring-layout conversion, and WO projection all complete; decode reaches the final axis-0 optimized all-reduce.
- The eight-device BF16 ring requests a 65,536-byte global circular buffer, but its 1,024-element BF8 persistent shard provides only 34,816 allocated bytes.
- The four-device QKV ring remains valid at the 1,024-element minimum; the axis-0 output ring requires a 2,048-element BF8 shard.
- The failed process was terminated and a full Galaxy reset was started.
- Next: size persistent shards by collective-axis cardinality and rerun through numerical/cache checks.

## Implementation checkpoint: axis-sized Attention CCL buffers

- Persistent decode CCL shards now reserve `max(1024, 256 * axis_cardinality)` BF8 elements: 1,024 for the four-device QKV ring and 2,048 for the eight-device output ring.
- Python compilation and `git diff --check` pass, and the Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun isolated Llama decode through final reduction and numerical/cache checks.

## Hardware checkpoint: widened BF8 output ring stalls

- The 2,048-element BF8 axis-0 buffer clears circular-buffer allocation validation, but the optimized final ring remains stationary with no host or device progress for over one minute.
- The qualified optimized collective geometry uses 1,024 elements per worker shard; changing logical shard width is not a viable capacity fix.
- A 1,024-element BF16 persistent shard supplies the required 65,536 physical bytes while matching the collective's requested BF16 output dtype and retaining qualified geometry.
- The stalled process was terminated and a full Galaxy reset was started.
- Next: restore 1,024-element shard geometry with BF16 persistent storage and rerun final reduction.

## Implementation checkpoint: dtype-sized persistent output

- Both Attention decode persistent outputs now retain the qualified minimum 1,024-element worker shard and use BF16 storage, matching the adapter's requested reduction dtype and providing a 65,536-byte bank.
- Python compilation and `git diff --check` pass, and the Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun isolated Llama decode through final reduction and numerical/cache checks.

## Hardware checkpoint: CCL input dtype must match persistent output

- A BF16 persistent tensor with unchanged geometry also stalls because the adapter still feeds BF16 to both optimized rings.
- The qualified Galaxy MLP contract casts optimized axis-0 reduction input/output to BF8. Attention axis 1 must remain BF16 because `nlp_create_qkv_heads_decode` only accepts BF16/FP32, while final axis 0 can use BF8.
- Each persistent buffer must match its ring's requested output dtype: BF16 for QKV axis 1 and BF8 for final output axis 0, both with the qualified 1,024-element minimum shard.
- The stalled process was terminated and a full Galaxy reset was started.
- Next: enforce per-axis CCL dtype before all-reduce and rerun final reduction.

## Implementation checkpoint: per-axis Attention CCL dtype

- Decode QKV axis 1 now converts to and reduces in BF16 with a BF16 persistent output; final axis 0 converts to and reduces in BF8 with a BF8 persistent output.
- Both rings retain 1,024-element persistent worker shards and their existing worker-scoped topology/semaphores.
- Python compilation and `git diff --check` pass, and the Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun isolated Llama decode through final reduction and numerical/cache checks.

## Hardware checkpoint: same-dtype copy selects invalid subdevice grid

- Passing `dtype=BF16` explicitly during the axis-1 placement conversion selects a different TTNN copy path whose kernel group is not contained in the worker subdevice.
- Axis 1 already has BF16 input and previously qualified with placement-only conversion; only axis 0 requires an actual BF16-to-BF8 conversion.
- The failed process was terminated and a full Galaxy reset was started.
- Next: request dtype conversion only when source and target CCL dtypes differ, then rerun.

## Implementation checkpoint: conditional CCL cast

- Axis-1 QKV uses placement-only conversion for its existing BF16 dtype; axis-0 output uses the explicit BF16-to-BF8 conversion required by the optimized ring.
- Python compilation and `git diff --check` pass, and the Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun isolated Llama decode through both reductions and numerical/cache checks.

## Hardware checkpoint: axis-0 cast and reshard must be split

- Axis-1 QKV reduction is restored and decode again reaches final axis 0.
- A single BF16-ring-layout to BF8-CCL-layout copy fails worker-subdevice validation. The qualified MLP path separates dtype conversion and resharding into two TTNN operations.
- The failed process was terminated and a full Galaxy reset was started.
- Next: cast axis-0 input to interleaved L1 BF8, reshard to the axis-0 CCL layout, release the cast intermediate, and rerun.

## Implementation checkpoint: two-stage axis-0 CCL conversion

- Axis-0 decode now casts projected output to interleaved L1 BF8, reshards that tensor into the qualified width-sharded CCL layout, and releases the cast intermediate before reduction.
- Axis-1 retains its placement-only BF16 conversion.
- Python compilation and `git diff --check` pass, and the Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun isolated Llama decode through final reduction and numerical/cache checks.

## Hardware checkpoint: both decode reductions complete

- The two-stage axis-0 conversion and optimized BF8 ring complete, advancing the full decode invocation to its final output contract.
- Attention2D currently requires output dtype to equal `decode_activation_dtype` (BF16), but the qualified optimized ring intentionally returns BF8, matching the established Galaxy MLP CCL policy.
- The failed process was terminated and a full Galaxy reset was started.
- Next: add an optional decode output dtype that defaults to activation dtype, select BF8 for this recipe, and proceed to numerical/cache checks.

## Verification checkpoint: independent decode output dtype

- Added optional `decode_output_dtype`; omitted values preserve the prior activation-dtype contract, while the qualified Galaxy recipe declares BF8 final output.
- Python compilation and `git diff --check` pass; the focused Attention host suite passes `62 passed in 9.24s`.
- The Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun isolated Llama decode through numerical and KV-cache checks.

## Hardware checkpoint: optimized axis-0 ring hangs on synchronization

- With the BF8 output contract accepted, decode returns to the test, but the first host composition remains stationary while synchronizing the optimized axis-0 ring.
- The optimized call's enqueue return is asynchronous and did not represent device completion; both widened-BF8 and dtype-matched variants hang at the same synchronization point.
- LMHead2D already qualifies synchronous `ttnn.all_reduce` on this Galaxy. Attention can retain optimized persistent axis-1 QKV CCL and use the stable synchronous primitive for final axis 0.
- The stalled process was terminated and a full Galaxy reset was started.
- Next: use synchronous BF16 axis-0 all-reduce, restore the original BF16 output contract, and rerun numerical/cache checks.

## Verification checkpoint: stable final Attention reduction

- Decode axis 0 now uses synchronous BF16 `ttnn.all_reduce` with the declared output memory config; optimized persistent BF16 CCL remains in place for axis-1 QKV.
- Removed the temporary independent output-dtype policy and restored Attention2D's original BF16 decode output contract.
- Python compilation and `git diff --check` pass; the focused Attention host suite passes `62 passed in 9.08s`.
- The Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun isolated Llama decode through host synchronization, numerical/cache checks, repetition, and prefill.

## Hardware checkpoint: optimized QKV ring also blocks device progress

- Replacing only final axis 0 does not unblock the first host synchronization, so the optimized axis-1 QKV command earlier in the queue is also a possible blocking command.
- Subsequent TTNN Python calls only enqueue behind optimized CCL and therefore do not prove that the QKV ring completed on device.
- The failed process was terminated and a full Galaxy reset was started.
- Next: use public `ttnn.all_reduce` for both Attention axes, retaining exact shared-resource geometry lookup and owned-output semantics, then rerun with real synchronization.

## Implementation checkpoint: stable owned Attention collectives

- Both Attention axes now resolve their exact `GalaxyResources` plan and execute public `ttnn.all_reduce` with the plan's topology/link count and mode-specific output memory config.
- The adapter reports outputs as owned and no longer consumes persistent semaphore handles; persistent resources remain allocated and validated by the shared registry for composition coverage.
- Python compilation and `git diff --check` pass, and the Galaxy reset completed with all 32 devices reinitialized.
- Next: rerun isolated Llama decode through real synchronization, numerical/cache checks, repetition, and prefill.

## Hardware checkpoint: public collectives still leave a blocked queue

- Replacing both optimized collectives with public `ttnn.all_reduce` still leaves the first host composition blocked, so CCL has not been isolated as the failing stage.
- The failed process was terminated and a full Galaxy reset was started.
- Next: fence the worker subdevice after QKV reduction, before final reduction, and after final reduction; use emitted stage markers to distinguish QKV CCL, cache/SDPA/WO, and output CCL.

## Hardware checkpoint: queue blocks by first post-QKV fence

- No marker appears after the first worker fence following axis-1 reduction, localizing the blocked queue to either QKV projection or the immediately following public all-reduce.
- The failed process was terminated and a full Galaxy reset was started.
- Next: add a pre-axis-1 fence to distinguish projection completion from axis-1 collective completion.

## Hardware checkpoint: Ring/4-link public axis-1 reduction blocks

- The pre-axis-1 marker appears and the post-axis-1 marker does not, proving QKV projection completes and the public column all-reduce blocks.
- The resource plan still specifies the Ring/4-link topology inherited from optimized CCL. The independently qualified LMHead2D public column all-reduce uses Linear topology with one link.
- The failed process was terminated and a full Galaxy reset was started.
- Next: change Attention public collective plans to Linear/1-link and rerun with stage fences retained.

## Hardware checkpoint: public CCL is incompatible with active decode subdevice

- Linear/1-link public axis-1 all-reduce also blocks after the pre-collective fence, so topology is not the issue; public CCL is incompatible with this active prefetch subdevice setup.
- The qualified MLP decode path uses optimized Ring/4-link CCL in BF8 on both axes. Attention previously requested BF16 on axis 1, which differs from that proven contract.
- The failed process was terminated and a full Galaxy reset was started.
- Next: restore optimized BF8 CCL for both axes, cast ring inputs through interleaved L1, and convert each persistent BF8 result back to owned BF16 for Attention consumers.

## Hardware checkpoint: optimized all-reduce blocks in BF8 as well

- Optimized BF8 axis-1 all-reduce also blocks between the pre/post fences, ruling out BF16 output format as the cause.
- MLP's prefill reduction composes standard `reduce_scatter` and `all_gather` while explicitly passing the active worker `subdevice_id`; this is mathematically equivalent to all-reduce and avoids both blocked all-reduce implementations.
- The failed process was terminated and a full Galaxy reset was started.
- Next: compose both Attention reductions from Linear/1-link reduce-scatter plus all-gather on the worker subdevice and rerun with fences retained.

## Hardware checkpoint: standard reduce-scatter spans loaded subdevices

- Standard reduce-scatter fails fast with `Programs must be executed on a single sub-device` even when passed the worker ID, so it cannot replace decode all-reduce under the active prefetch manager.
- Axis-1 optimized BF8 was only tested with Ring/4-link. A Linear/1-link optimized column collective remains untested, while axis-0 Ring/4-link is independently qualified by MLP.
- The failed process was terminated and a full Galaxy reset was started.
- Next: use optimized BF8 CCL with per-axis topology (axis 1 Linear/1-link, axis 0 Ring/4-link) and rerun with fences retained.

## Hardware checkpoint: Llama-optimal column all-reduce blocks

- Optimized BF8 Linear/1-link axis 1 still blocks between the stage fences.
- `use_optimal_ccl_for_llama` is qualified by MLP for the final axis-0 ring, not for a column collective; the generic async all-reduce kernel remains untested for axis 1.
- The failed process was terminated and a full Galaxy reset was started.
- Next: disable the Llama-optimal kernel only for axis 1 and rerun with all other resource contracts unchanged.

## Hardware checkpoint: generic async column all-reduce also blocks

- Disabling the Llama-optimal kernel does not unblock axis 1, exhausting the all-reduce implementations tried under decode prefetch.
- The earlier reduce-scatter failure used a DRAM intermediate, allowing TTNN to place kernels outside the worker subdevice. Worker-only width-sharded intermediates can constrain both halves of the equivalent reduction.
- The failed process was terminated and a full Galaxy reset was started.
- Next: use 10 worker cores for the 320-wide axis-1 reduced shard and 8 worker cores for the 256-wide axis-0 reduced shard, then all-gather into the declared output layouts.

## Hardware checkpoint: reduce-scatter input still spans subdevices

- Worker-sharded reduce-scatter intermediates still fail single-subdevice validation because the projection tensor itself remains on ring/receiver cores spanning loaded subdevices.
- Both all-reduce adapters previously converted projection output into the worker-only CCL layout before launch; the composed path omitted that prerequisite.
- The failed process was terminated and a full Galaxy reset was started.
- Next: convert projection output to the declared worker CCL layout before reduce-scatter, then rerun the composed reduction.

## Hardware checkpoint: generic axis-1 CCL remains multi-subdevice

- Converting projection input and reduce-scatter output to worker-only width-sharded layouts still produces `Programs must be executed on a single sub-device` in the standard CCL implementation.
- Decode projection itself completes at the pre-axis-1 fence. The unresolved boundary is specifically Attention's axis-1 reduction while the persistent decode sender subdevice is active.
- Public all-reduce, optimized all-reduce (BF16/BF8, Ring/Linear, optimal/generic), and standard reduce-scatter/all-gather have now been ruled out for this boundary.
- The failed process was terminated and a full Galaxy reset was started.
- Next: preserve this diagnostic boundary, advance independent Milestone A exit gates, then return with a specialized persistent axis-1 reduce-scatter/all-gather resource plan.

## Verification checkpoint: stale regression fixtures repaired

- Updated the exact frozen `Llama3ExecutorConfig` field expectation to include the existing diagnostic batching flag.
- Updated the traced-prefill operation-plan fake with the current nullable `prime` and `release_prime_output` protocol.
- Moved the three prefill kernel-config values onto the Attention1D config fake where unchanged production code reads them; no `*_1d.py` implementation changed.
- Focused verification passes: `4 passed in 2.11s` across both runtime nodes and both Attention1D parametrizations.
- Next: run the complete default-runtime suite, then the serialized 1D regression files.

## Verification checkpoint: default runtime exit gate green

- The complete default-runtime suite passes after the test-only fixture repairs: `1029 passed, 1 skipped, 9 warnings in 201.26s`.
- This clears the prior two stale failures and preserves every existing runtime expectation alongside the new fixed-physical-32 planning policy.
- Next: run the complete 2,006-case 1D module regression suite in one serialized process and confirm no 1D implementation files changed.

## Verification checkpoint: serialized 1D matrix clean through penalties

- The one-hour guard expired with exit code 124 during the final sampling module; this is an incomplete gate, not a test failure.
- Every completed case passed. The run completed Attention1D, Embedding1D, LMHead1D, MLP1D, RMSNorm1D, RoPE1D, and Sampling Penalties1D across the selected 1-, 2-, and 8-device meshes.
- Representative results included 32K paged/chunked Attention prefill, 8-device prefill/decode transitions above 0.9995 PCC, exact Embedding and RoPE comparisons, and passing low-precision PCC contracts for LM head, MLP, and RMSNorm.
- `git diff --name-only -- 'models/common/modules/**/*_1d.py'` is empty: no 1D implementation file changed. The only 1D test edit is the previously documented stale Attention fixture repair.
- Next: reset the Galaxy after timeout termination and run `test_sampling_1d.py` alone to close the remaining serialized 1D gate.

## Hardware checkpoint: Galaxy topology restored after warm-reset degradation

- A first post-timeout `tt-smi -r` returned success but left two physical links at three channels; `SystemMeshDescriptor` then failed before module setup, producing 136 sampling fixture errors after four host tests passed.
- The focused traceback was a strict 32-chip topology mapping failure, not a Sampling1D assertion or implementation failure.
- `tt-smi -glx_reset` reinitialized all 32 boards. A focused `1x1` mesh fixture then passed and topology discovery reported physical degree `{4:32}`.
- Next: rerun the complete Sampling1D file on the restored Galaxy.

## Verification checkpoint: complete 1D module gate green

- The remaining Sampling1D suite passes on restored hardware: `140 passed, 50 deselected in 88.95s`.
- Combined with the clean one-hour matrix through Sampling Penalties1D, all selected 1D module files completed without a test failure.
- No `models/common/modules/**/*_1d.py` implementation file changed.
- Next: qualify the remaining Milestone A 2D hardware boundaries sequentially, beginning with distributed RMSNorm2D prefill.

## Verification checkpoint: distributed RMSNorm2D prefill harness bounded

- Replaced whole-mesh readback synchronization in the RMSNorm Galaxy harness with the public worker-scoped `resources.synchronize(mode)` API.
- Split distributed prefill sequence lengths into independent `128` and `2048` pytest cases so the smallest Llama residual smoke can run alone before expansion.
- Hardware collection now exposes eight cases cleanly. Focused RMSNorm2D and Galaxy resource host contracts pass: `23 passed in 3.47s`.
- Next: run only Llama final-norm prefill at sequence 128 under a five-minute external timeout.

## Hardware checkpoint: RMSNorm prefill semaphore nesting rejected at launch

- The bounded Llama sequence-128 node failed fast and tore down cleanly before collective execution.
- `all_gather_async` received `list[list[global_semaphore]]`: the test plan allocated two semaphores per slot and RMSNorm then requested the required adjacent two-slot window.
- The qualified MLP async all-gather contract is one semaphore per slot with a two-slot window. RMSNorm's prefill test plan now matches that allocation shape while preserving two cycling slots.
- Next: rerun the same bounded Llama sequence-128 residual prefill node.

## Hardware checkpoint: RMSNorm async prefill all-gather dumps core

- With one semaphore per slot and a valid adjacent two-slot window, the Llama sequence-128 collective launched but the process dumped core before pytest teardown.
- This reproduces the audit's unresolved async-prefill boundary: residual add and local RMS statistics reach the collective, while the persistent 2D `all_gather_async` path does not return safely.
- The failed process contaminated hardware state; a full Galaxy tray reset is required before further device work.
- Next: compare against qualified distributed RMS and MLP prefill collectives, then replace or isolate the failing async path with the smallest statically resolved strategy.

## Verification checkpoint: RMSNorm prefill strategy isolated from async CCL

- Distributed prefill now uses standard axis-aware `ttnn.all_gather` on the single full worker subdevice; topology, links, axis, subdevice, and DRAM output placement remain resolved in immutable module resources before the hot path.
- Decode's distinct persistent/fused strategies are unchanged.
- Added a host contract proving the exact standard-gather arguments and that prefill does not call `all_gather_async`. Focused RMSNorm/Galaxy tests pass: `24 passed in 3.59s`.
- The Galaxy was fully tray-reset after the async core dump and all 32 boards reinitialized.
- Next: rerun the bounded Llama sequence-128 residual prefill node with the isolated prefill strategy.

## Hardware checkpoint: RMSNorm standard prefill gather also dumps core

- Replacing persistent async gather with standard axis-aware gather did not remove the device core dump, ruling out the async operation as the sole cause.
- The shared lifetime difference is now the leading boundary: RMSNorm2D force-deallocates the local stats tensor immediately after enqueueing the collective, while qualified MLP and existing distributed RMSNorm1D paths do not.
- The failed process contaminated hardware state; another full Galaxy tray reset is required.
- Next: remove the unsafe eager stats deallocation, preserve TTNN queue ownership, rerun host contracts, and retry the same smallest hardware node.

## Hardware checkpoint: RMSNorm stats lifetime hypothesis ruled out

- Keeping local stats alive through the queued gather and post-gather RMS did not change the core dump.
- Host contracts remained green (`24 passed in 3.62s`), but the hardware failure still occurs before module return/readback.
- The process again contaminated hardware state and requires a full tray reset.
- Next: add temporary worker-only stage fences after local stats, gather, and post-gather RMS to identify the first non-returning program precisely.

## Hardware checkpoint: first recovery reset incomplete

- The first post-crash `tt-smi -glx_reset` found all 32 PCI devices but failed board reinitialization at 23 detected chips with `Timed out while waiting 1s for ARC to respond`.
- Hardware remains unavailable for testing in this state.
- Next: repeat the full tray reset and require a 32-chip mesh smoke before continuing RMSNorm stage fencing.

## Hardware checkpoint: RMSNorm compute and gather stages complete

- The second tray reset restored all 32 boards and a focused mesh fixture passed.
- Temporary worker-only fences proved local RMS statistics, standard axis-1 gather, and post-gather RMS all complete. The segmentation fault occurs only afterward.
- The remaining post-compute path redundantly converted an already produced output to the configured placement and force-deallocated both the residual sum and unplaced output while queued work could still reference them.
- Removed all temporary diagnostic fences. Post-gather RMS now receives the pre-resolved output memory config directly, and the prefill path no longer performs redundant placement or forced intermediate deallocation.
- Next: reset after the diagnostic crash, rerun host contracts, then retry the bounded Llama sequence-128 node.

## Hardware checkpoint: RMSNorm2D Llama distributed prefill-128 qualified

- The corrected path passes two repeated Llama residual-prefill invocations at sequence 128 with PCC `>= 0.99` and clean 32-device teardown: `1 passed in 2.85s`.
- Root cause was redundant post-norm output conversion plus forced intermediate deallocation after queued compute. Passing the resolved output memory config directly to `rms_norm_post_all_gather` removes that unsafe lifetime boundary.
- Standard axis-1 gather remains statically configured from resolved Galaxy resources; no diagnostic synchronization or print remains in production.
- Focused host contracts and diff hygiene remain green: `24 passed in 3.62s`, `git diff --check` clean for the touched RMS files.
- Next: qualify Qwen sequence 128, then Llama and Qwen sequence 2048 in separate serialized processes.

## Hardware checkpoint: RMSNorm2D distributed prefill matrix qualified

- Qwen sequence 128 passes repeated residual-prefill invocation and teardown: `1 passed in 7.30s`.
- Llama sequence 2048 passes repeated residual-prefill invocation and teardown: `1 passed in 11.88s`.
- Qwen sequence 2048 passes repeated residual-prefill invocation and teardown: `1 passed in 6.44s`.
- Together with Llama sequence 128, distributed prefill is qualified across both representative geometries and the required 128-to-2048 range at PCC `>= 0.99`.
- Next: isolate fused distributed decode from the unsafe gamma-prefetch harness, use semaphore cores matching the decode input shard, and qualify Llama then Qwen batch 32.

## Verification checkpoint: fused RMSNorm2D decode harness uses qualified ownership

- Decode now uses explicit `FABRIC_1D_RING`, Ring topology, and a known-good DRAM-sharded tiled projection payload for the persistent prefetch queue; row-major norm gamma is no longer registered with `dram_prefetcher`.
- The fused semaphore is allocated on the exact x=2..3 norm input shard grid, independently of the x=1 persistent stats buffer.
- The projection payload is test-only and borrowed by Prefetcher2D; root cleanup remains Galaxy resources first, then explicit projection and module-weight release.
- Hardware collection remains eight cases, RMSNorm host contracts pass `16 passed in 2.35s`, and the touched diff is whitespace-clean.
- Next: run the isolated Llama fused batch-32 residual decode node under a five-minute guard.

## Hardware checkpoint: decode projection setup times out before RMS launch

- The bounded Llama decode process opened the 32-device mesh but emitted no module or prefetch progress before the five-minute guard expired.
- The fused RMS kernel did not launch. The new boundary is test-only projection materialization versus Prefetcher2D initialization, not RMS arithmetic.
- Hardware state after forced timeout requires a full tray reset.
- Next: add setup markers around projection materialization and resource construction, reduce the payload if needed, and rerun only after reset.

## Hardware checkpoint: decode projection and Prefetcher setup complete

- A reduced but valid DRAM-sharded projection payload materializes successfully, and concrete Prefetcher2D plus Galaxy resources report ready.
- The guarded process still times out afterward, narrowing the boundary to decode activation, fused RMS launch, or worker synchronization.
- The timed-out state requires another full tray reset.
- Next: bracket those three operation stages with test-only markers and rerun the same Llama node.

## Hardware checkpoint: fused RMS decode first invocation passes, repeat stalls

- Test-only markers show projection setup, decode activation, fused module launch/return, first worker synchronization, and first numerical readback all complete.
- The second invocation again activates resources and returns from `fused_rms_minimal`, but its worker synchronization does not return before the external timeout.
- This isolates the defect to repeated fused-collective cycle state, not first-use arithmetic, prefetch startup, semaphore placement, or output readback.
- The timed-out process requires a full tray reset.
- Next: align repeated eager resource provisioning with fused RMS reference tests, especially per-invocation semaphore and persistent stats ownership.

## Verification checkpoint: fused RMS repeat lifecycle aligned to reference

- TTNN fused RMS reference tests allocate one semaphore per iteration, reuse one persistent stats tensor, enqueue all eager iterations, then synchronize and read back once.
- Galaxy resources already provide distinct cycling semaphore slots and one persistent stats buffer; the harness differed by synchronizing and reading back between launches.
- The decode repeat test is being changed to activate once, enqueue two calls, synchronize the worker once, and then validate both normalized and in-place residual outputs.
- Next: reset the timed-out state and rerun Llama fused decode with the reference-aligned lifecycle.

## Hardware checkpoint: fused RMS repeat passes; synthetic prefetch teardown stalls

- The reference-aligned Llama run enqueued two fused calls, synchronized once, completed both normalized/residual PCC checks, and reached pytest `PASSED`.
- Teardown then timed out because the synthetic projection payload was prefetched into the global circular buffer but no RMS operation consumes it, preventing the persistent sender from draining its stop protocol.
- This is an isolated harness ownership error, not a fused RMS compute or repeat failure.
- The timeout requires a full tray reset.
- Next: qualify fused RMS with Galaxy CCL resources but no unrelated active prefetch sender; retain Prefetcher2D integration proof in the already-green repeated MLP decode matrix.

## Implementation checkpoint: fused RMS CCL-only ownership fixture

- A full Galaxy tray reset restored all 32 boards after the synthetic-prefetch teardown timeout.
- Added a test-only structural subdevice owner that loads the exact production Galaxy mode plans while deliberately launching no Prefetcher2D GCB program.
- The fused decode harness now allocates production Galaxy CCL resources through `create_galaxy_resources` without registering an unrelated projection payload; repeated Prefetcher2D consumption remains covered by the qualified MLP matrix.
- Python compilation and touched-file diff checks pass.
- Next: run the focused Llama fused decode hardware node and verify both repeated PCC checks and clean resource teardown.

## Hardware checkpoint: Llama fused RMS repeat and teardown pass

- The Llama 8192-wide fused decode test passed two queued invocations with normalized and in-place residual PCC >= 0.99.
- Worker-scoped synchronization returned, production Galaxy CCL resources cleaned up, subdevice managers were removed, and all 32 devices closed normally.
- Result: `1 passed in 2.71s`; teardown completed in 0.29s.
- Next: run the Qwen 5120-wide fused decode node through the same serialized CCL-only ownership path.

## Hardware checkpoint: fused distributed RMS decode matrix passes

- Qwen 5120-wide fused decode passed two repeated normalized/residual invocations and clean teardown: `1 passed in 4.29s`.
- Together with Llama 8192, distributed RMSNorm2D now has real-hardware coverage for batch-32 fused decode and repeated 128/2048 prefill across both target model widths.
- The CCL-only fixture cleanly separates RMS collective ownership from the already-qualified MLP Prefetcher2D consumer lifecycle.
- Next: return to the Attention2D axis-1 projection reduction and replace the generic multi-subdevice collective with an explicitly provisioned asynchronous recipe.

## Integration checkpoint: concrete Prefetcher + Galaxy ownership contracts

- Integrated and reviewed a new host-only suite that composes concrete `Prefetcher2D` and `GalaxyResources` owners with injected hardware bindings.
- Contracts cover shared contexts, repeated decode restart and mode-switch serialization, worker-scoped synchronization, split cleanup ownership, policy mismatch rejection before CCL allocation, and activation rollback.
- Focused recheck passes: `4 passed in 0.63s`; the agent's combined Prefetcher/Galaxy regression matrix passed `29 passed in 4.31s`.
- No hardware commands ran in the delegated lane, preserving serialized device ownership.
- Next: adapt Attention2D decode collectives to the already-qualified MLP persistent asynchronous resource pattern.

## Integration checkpoint: physical batch-32 trace lifecycle contract

- Integrated and reviewed a host trace lifecycle test covering padded sequence lengths 128, 1024, and 2048.
- Each case captures once at physical batch 32, then replays the same program/trace identity for refreshed 31/32 active rows and reversed slots while checking padding, persistent-input refresh, replay counters, and trace release.
- Delegated verification passed: focused `3 passed in 0.84s`, physical-32 set `23 passed`, and full prefill runtime `144 passed in 21.89s`; diff check is clean.
- The delegated lane ran no TT hardware commands.

## Verification checkpoint: Attention asynchronous adapter ready

- Replaced the decode-only generic reduce-scatter/all-gather adapter with the same persistent `all_reduce_async` call contract used by the qualified MLP decode path.
- The adapter now consumes exact Galaxy resource keys, cycling semaphores, persistent buffers, resolved topology/link count, worker subdevice ID, and BF8 L1 placement.
- Attention host contracts remain green: `62 passed in 9.21s`; Python compilation and diff checks pass.
- Next: run the focused Llama Attention2D hardware node under a bounded timeout.

## Hardware checkpoint: Attention decode reaches placement conversion failure

- The persistent async adapter was selected, but the first decode failed before collective launch while converting the projected tensor into the requested BF8 L1 sharding; TT Metal rejected the resulting circular-buffer allocation.
- Because the concrete decode prefetch producer had already started, exception cleanup could not drain and the failed process was terminated.
- This is a narrower placement/configuration defect than the previous generic multi-subdevice CCL failure; no evidence indicates the persistent all-reduce itself launched.
- Hardware requires a full tray reset before the next run.
- Next: capture the full allocation diagnostic after reset and align the collective input memory config with the known-good MLP all-reduce geometry.

## Hardware checkpoint: Attention axis-1 persistent all-reduce completes

- Matching MLP's two-step BF16 receiver-sharded -> BF8 interleaved -> BF8 worker-sharded conversion removed the circular-buffer allocation failure.
- The exact persistent axis-1 all-reduce completed and execution advanced to `nlp_create_qkv_heads_decode`.
- That kernel rejected the BF8 collective result because it requires BF16 or FP32; the failed process again left the unrelated prefetch sender unable to drain and was terminated.
- Added an owned two-step BF16 restoration after the borrowed persistent collective result, preserving the persistent buffer while satisfying the heads kernel contract.
- Next: reset all trays and rerun the focused Llama decode path.

## Implementation checkpoint: Attention axis-1 minimal RS/AG recipe

- BF16 restoration forced the queued axis-1 `all_reduce_async` to execute and confirmed that operation stalls under the decode partition; the process was terminated.
- Replaced only decode axis 1 with the MLP-qualified persistent sequence: Ring-4 `reduce_scatter_minimal_async` using three semaphores, a barrier, packet intermediates, and a reduced output, followed by Ring-4 `all_gather_async` using a two-slot semaphore window and persistent output.
- Decode axis 0 retains persistent all-reduce; prefill retains standard eager collectives.
- Added exact operation/geometry plans for both axis-1 stages. Python compilation and diff checks pass.
- Next: complete the required tray reset, then rerun focused Llama Attention2D.

## Hardware checkpoint: axis-1 minimal RS resource validation reached

- The new minimal RS/AG adapter selected the exact axis-1 resource and entered `reduce_scatter_minimal_async`.
- Validation rejected the copied MLP packet intermediate because its padded width was 4096 while Attention QKV input width is 1280.
- Replaced it with the operation's explicitly supported input-shaped BF8 tiled DRAM persistent intermediate, aligned with `intermediate_memory_config=DRAM`.
- The failed process was terminated and hardware requires reset before rerun.
- Next: reset all trays and rerun the focused Llama node.

## Hardware checkpoint: tiled RS fallback stalls; contiguous staging selected

- The input-shaped tiled persistent intermediate passes validation but the Ring-4 reduce-scatter does not complete under the decode partition.
- The process was terminated; no later Attention stage ran.
- Updated the hardware adapter to lazily allocate the operation's exact correlated contiguous and penultimate staging buffers with `reduce_scatter_minimal_async_create_intermediate_buffer`, pass them in the required `[intermediate, output, penult]` order, and release them explicitly during fixture cleanup.
- Next: reset all trays and rerun Llama with the contiguous BF8 Ring fast path.

## Hardware checkpoint: contiguous RS still stalls; input placement mismatch found

- Exact contiguous staging validates but Ring-4 RS still does not make forward progress; the process was terminated.
- Comparison with qualified MLP revealed the Attention adapter was placing the QKV input on the reduced-output cores before RS. MLP keeps the RS input on the matmul receiver cores and uses a distinct memory config only for the reduced result.
- Added explicit per-axis collective input placements: QKV/WO matmul receiver sharding for inputs, reduced/final sharding for outputs.
- Next: reset all trays and rerun focused Llama with corrected RS input ownership.

## Hardware checkpoint: Ring-4 axis-1 remains stalled; Linear minimal selected

- Keeping the projection on its matmul receiver cores did not unblock the Ring-4 minimal RS; the process was terminated.
- The remaining variable is the Ring protocol for the narrow 1280-wide QKV geometry.
- Switched decode axis 1 to Linear-1 minimal RS/AG. Linear uses the already validated input-shaped tiled intermediate and has no contiguous or penultimate Ring staging protocol; the adapter now selects persistent buffer order by topology.
- Next: reset all trays and run focused Llama through the Linear minimal path.

## Hardware checkpoint: persistent axis-1 conflicts with active decode sender

- Linear-1 minimal RS also stalls under the two-subdevice decode sender partition; the process was terminated.
- Ring/Linear, tiled/contiguous staging, and receiver/reduced input placement have now been independently ruled out. The common condition is the active unrelated persistent sender.
- Split Attention arithmetic qualification onto a CCL-only single-worker Galaxy owner, matching the accepted fused-RMS isolation strategy. Decode uses regular matmuls, no GCB kwargs, DRAM projection/concat placements, and standard axis-aware collectives.
- Prefetcher plus Galaxy repeated ownership remains proven by MLP hardware and the concrete host composition suite.
- Next: reset all trays and run the isolated Llama Attention2D decode/prefill matrix.

## Hardware checkpoint: CCL-only Attention reaches regular matmul validation

- The isolated Llama test opened, failed, cleaned all resources, and closed all 32 devices normally in 7.08s; no reset is required.
- Failure is regular program tuning, not ownership or CCL: column dispatch exposes a `(7, 10)` worker grid, while `_matmul_program` requested `(8, 1)`.
- Resolved the shared regular decode/prefill matmul helper to seven columns and recomputed per-core N partitioning accordingly.
- Next: rerun focused Llama without resetting clean hardware.

## Hardware checkpoint: regular QKV matmul passes; selector mismatch fails cleanly

- Seven-column regular matmul tuning passed and execution reached the axis-1 collective selector.
- The CCL-only branch incorrectly retained the persistent branch's `reduce_scatter` resource lookup, while its plan intentionally exposes an `all_reduce` identity for the standard RS/AG composition.
- Corrected operation selection to request `reduce_scatter` only for persistent decode axis 1.
- Failure tore down and closed all devices normally: `1 failed in 10.65s`; no reset required.
- Next: rerun focused Llama.

## Hardware checkpoint: isolated standard axis-1 CCL also stalls

- The CCL-only Llama run completed regular QKV projection and submitted standard axis-1 reduce-scatter/all-gather on a single full-grid worker subdevice.
- The worker still did not return after the final bounded interval; the process was terminated.
- This establishes the remaining blocker below Prefetcher ownership and below Attention2D arithmetic: every available axis-1 reduction API stalls for the concrete `(8, 4)`, `(1, 1, 32, 1280)` QKV geometry on this runtime, including generic RS/AG, async all-reduce, Ring/Linear minimal RS/AG, and isolated single-worker standard RS/AG.
- Hardware requires one final full tray reset. Further speculative retries are stopped; remaining host, scope, and status gates will be completed with this hardware limitation recorded explicitly.

## Verification checkpoint: final integrated host gate passes

- Ran runtime, Galaxy CCL/resources, concrete Prefetcher composition, Prefetcher2D, and every 2D module host suite together.
- Result: `1257 passed, 1 skipped, 9 warnings in 250.72s`.
- This includes the physical-batch-32 capture/replay lifecycle at 128/1024/2048 and the concrete Prefetcher/Galaxy ownership contracts added by parallel agents.
- Final `git diff --check` is clean.
- Updated `MILESTONE_A_STATUS.md` with current qualified RMS/MLP/resource/runtime/1D evidence and the exact unresolved Attention axis-1 hardware boundary.

## Investigation checkpoint: Attention CCL blocker resumed with tt-buddy lead

- Resumed the Milestone A goal to investigate the sole remaining Attention2D hardware blocker rather than treating the prior axis-1 stall boundary as final.
- Attempted to access `tenstorrent/tt-buddy` through the public GitHub URL, the authenticated GitHub connector, direct clone, and the local filesystem. The repository currently returns 404 through GitHub surfaces, direct clone has no usable credentials, and no local checkout was found.
- Closed four completed stale agents and dispatched three fresh independent audits, each with a dedicated goal and checkpointed work log: known-passing WH Galaxy axis-1 examples, local CCL implementation/validation, and alternate `tt-buddy` access or equivalent guidance.
- Kept TT hardware ownership on the main lane only. The next local step is a standalone sequential collective probe based on the closest known-passing QKV and 2D CCL tests, followed by a one-variable experiment matrix.

## Hardware checkpoint: exact 6U fused QKV collective passes

- Found the repository's purpose-built Galaxy test in `tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py`, which uses the exact per-device Attention QKV shape `[1, 1, 32, 1280]`.
- The linear-fabric variant was correctly skipped because localhost is detected as a 6U topology.
- Ran the unchanged 6U variant under serialized hardware ownership. It uses `FABRIC_1D_RING`, Ring topology, `all_reduce_create_qkv_heads`, 3 links, 24 input cores, 10 output cores, BF8 input, and BF16 head outputs.
- Result: `1 passed, 1 warning in 3.71s`, including 10 warmups, 30 traced iterations, per-output numerical validation, and clean device teardown.
- This disproves the prior hypothesis that the concrete `(8,4) / (1,1,32,1280)` geometry is unsupported. The remaining work is to integrate the qualified fused QKV/head primitive and its 6U fabric/subdevice contract into the Attention2D hardware adapter.

## Implementation checkpoint: optional fused decode boundary passes host contracts

- Added an optional `reduce_create_qkv_heads` callable to `Attention2DLowLevelCallables`; existing callers retain the reduce-then-split route unchanged.
- Updated `decode_forward` to use the fused hook when supplied, release the QKV projection exactly once, and continue with the same Q/K normalization, rotary, cache, SDPA, concat, and output projection stages.
- Added a host ownership/dispatch test proving the fused hook bypasses both `reduce_qkv` and `nlp_create_qkv_heads_decode` without leaking the QKV projection.
- Result: `63 passed in 9.49s` for the complete host Attention2D contract suite.
- The passing-examples agent independently confirmed that exact-shape working cases use width-sharded L1 and a single specialized asynchronous collective, while the stalled adapter used a DRAM-backed synchronous RS/AG composition. Its checkpointed report is `tttv2_ccl_passing_examples_audit.md`.

## Investigation checkpoint: tt-buddy and C++ audits identify concrete deadlock causes

- A parallel agent established that the private `tenstorrent/tt-buddy` repository is accessible through this host's configured GitHub SSH identity and inspected commit `ba9021417442d59756aa8cdf154a25648c9a0de5`.
- `tt-buddy` documents incorrect Galaxy `num_links` as a deadlock cause, requires live triage before reset, and recommends deriving links from qualified sibling models. Its full evidence and triage commands are recorded in `tttv2_tt_buddy_access_audit.md`.
- Independent source audits found two additional harness defects: `fabric_config=True` resolves to neighbor exchange rather than the 6U Ring fabric, and the old adapter force-deallocated queued RS/AG dependencies before synchronization.
- Replaced the decode resource geometry with the exact qualified fused-QKV contract: explicit `FABRIC_1D_RING`, Ring topology, three links, the canonical 50-core worker subdevice, 24-core BF8 input placement, 10-core BF8 scratch/BF16 reduced placement, and 32-core head output placement.
- Added an explicit mode synchronization before releasing the fused collective input, preserving asynchronous tensor lifetimes. Compilation, hardware-test collection, and `git diff --check` pass.

## Implementation checkpoint: Galaxy resource vocabulary admits fused QKV

- The first integrated Llama attempt failed before kernel launch because the shared Galaxy resource-key vocabulary did not yet include the specialized fused operation; teardown closed all 32 devices normally.
- Added the canonical `all_reduce_create_qkv_heads` resource name with axis-1 validation to shared Galaxy CCL infrastructure and covered it in host key validation.
- Combined shared Galaxy CCL/resource and Attention2D host verification passes: `85 passed in 12.49s`.
- Hardware remained clean, so the same focused Llama case can be rerun without reset.

## Hardware checkpoint: dedicated CCL subdevice conflicts with regular QKV matmul

- The explicit Ring-fabric Llama run reached decode execution but failed before the fused collective: the regular rectangular QKV matmul touches cores outside the canonical 50-core CCL-only subdevice.
- Runtime rejected this correctly with `Kernel group cores do not match sub device cores`; all devices closed normally.
- Retained the exact fused CCL core placements, buffers, links, topology, and semaphore grid, but widened the single decode scheduling subdevice to the full compute grid so producer and consumer share one valid runtime domain.

## Hardware checkpoint: real matmul-derived fused QKV passes

- With the full-grid scheduling domain, the real Attention QKV matmul completed, converted to the exact 24-core BF8 input layout, and the Ring-3 `all_reduce_create_qkv_heads` completed under explicit worker synchronization.
- Execution advanced through fused Q/K/V creation to KV-cache update, proving the sole CCL blocker is resolved for the real producer tensor rather than only synthetic input.
- The next fail-fast validation exposed an independent cache mapping defect: fused outputs carry eight column-local users, while the test cache was replicated with local batch 32.
- Changed the cache mapper to shard batch over mesh columns and replicate over rows, matching both the fused operation's `batch_offset=(0,8,16,24)` contract and the existing 2D cache composer.

## Implementation checkpoint: contiguous KV cache validates column-local batch

- The sharded cache then failed the module's host-side validation because `Attention2D` incorrectly required local batch 32 despite its explicit `users_per_column=8` contract.
- Corrected contiguous-cache validation to require the column-local batch and updated affected host fixtures; global composition still reconstructs all 32 users across columns.

## Hardware checkpoint: decode positions must follow column-local users

- The corrected cache reached `paged_update_cache`, which rejected the still-replicated 32-element positions tensor against the local batch of eight.
- Sharded positions over mesh columns with the same batch mapping as the cache and allowed the public decode validator to recognize both column-local distributed shapes and legacy/global host shapes.

## Hardware checkpoint: KV cache update requires one shard per local user

- Cache update now validates local batch and index count, then rejected fused K/V's inherited 32-core head placement because its kernel requires one shard per user.
- Added a distinct eight-core height-sharded KV placement. The module already owns the required K/V placement transition, while Q retains the 32-core fused head layout used by decode SDPA.

## Hardware checkpoint: fused decode reaches concat-heads

- The next focused Llama run completed the real QKV projection, Ring-3 fused QKV collective, both KV-cache updates, and decode SDPA.
- Execution then failed cleanly in `nlp_concat_heads_decode` with `bad optional access`; all 32 devices closed normally and no reset was required.
- Source validation showed this is a downstream API-contract defect, not CCL: the non-origin fused head grid selects the concat operation's sub-core program, which requires an explicit `sub_core_grids` argument.

## Implementation checkpoint: concat-heads receives its qualified sub-core grid

- Wired the fused 32-core head grid into `decode_concat_sub_core_grids`; that grid contains the eight local-head compute cores required by `nlp_concat_heads_decode` and matches the SDPA output's scheduling domain.
- Kept concat's intrinsic width-sharded L1 result and the module's existing explicit transition to the configured DRAM placement before WO projection.
- Two read-only parallel agents are independently auditing the C++ operation and production Galaxy usage while the main lane proceeds with host checks and the next serialized hardware run.

## Hardware checkpoint: concat requires one SDPA-output core per local user

- The explicit sub-core argument removed `bad optional access`; concat validation then reported that its 32-core input grid did not match the eight local users.
- The fused Q tensor legitimately uses 32 cores for decode SDPA, but concat's input contract is different: decode SDPA must emit one height shard on each of eight local-user cores.
- Reused the already-qualified eight-core KV/user placement for the SDPA result and exposed that exact grid to concat; the focused failure closed all devices normally, so no hardware reset was needed.

## Hardware checkpoint: complete decode compute reaches output contract

- With the eight-core SDPA/concat boundary, the focused Llama run completed concat-heads, WO projection, and the axis-0 output collective without a CCL stall.
- The only failure was a declared-placement mismatch: the adapter's qualified output collective returns width-sharded L1, while `decode_output_placement` still claimed interleaved DRAM.
- Aligned the module's public decode output contract with the collective's actual configured placement. Teardown was clean on all devices, so the next serialized run proceeds without reset.

## Hardware checkpoint: column-local users require a pre-concat gather

- Full decode reached numerical validation, but PCC was `-0.00218`; the first local users were plausible while padded rows contained large values.
- The cause is the adapter's incorrect claim that every mesh column already owns the same users. Fused QKV deliberately selects distinct eight-user slices per column, so production Galaxy gathers SDPA output over axis 1 before concat-heads.
- Added an optional `gather_users` low-level boundary before decode concat, preserving the existing path when omitted. The hardware adapter now uses an explicit Ring, one-link axis-1 all-gather resource and synchronizes before the module releases its input.

## Hardware checkpoint: gathered-user resource keys use logical geometry

- The first pre-concat gather attempt failed before kernel submission because its resource plan used the padded head count in the key.
- TTNN exposes the SDPA tensor's logical geometry as `(1, 8, 8, 128)` with sequence key `64`; corrected the all-gather key and its logical global output specification to `(1, 32, 8, 128)` while retaining the required physical 32-head tile shard.
- All devices closed normally and no reset was required.

## Hardware checkpoint: pre-concat gather executes; numerical localization continues

- The corrected Ring axis-1 gather executed and synchronized, and the full decode again reached output comparison without a stall.
- Whole-output PCC improved only to `0.00621`, indicating that user gathering removed padded-row garbage but a user/head/order mismatch remains.
- Reordered the mandatory cache assertion ahead of output PCC so the next single run will separate fused-QKV/cache correctness from post-SDPA composition. The failed run tore down cleanly.

## Hardware checkpoint: KV caches pass; output reduction ignored its resource topology

- Both K and V cache comparisons passed PCC for all 32 users before output comparison, qualifying fused QKV selection, batch offsets, cache mapping, and cache updates end to end.
- The remaining output path exposed a direct tt-buddy rule violation: the axis-0 resource declares the sibling-qualified Ring topology with four links, but the standard RS/AG adapter hardcoded Linear topology and one link.
- Updated both stages to use the selected resource's topology and link count and synchronized before releasing the reduced dependency. The diagnostic run closed all devices normally.

## Hardware checkpoint: output reduction is sound; SDPA positions are column-local metadata

- The Ring/four-link output reduction executed cleanly but produced the same `0.00621` PCC, ruling that stage out while retaining its corrected resource contract.
- The output magnitude localizes the defect to SDPA metadata: users 0-7 have the expected 128-token averaging scale, while later column groups look like position-zero/single-token attention.
- Replaced the globally sourced, mesh-sharded 32-position tensor with a replicated local eight-position tensor. This matches both the cache-update local batch and every device's column-local SDPA input contract.

## Hardware checkpoint: local position replication is numerically neutral

- Replicated local-eight positions produced bit-identical output and the same PCC, ruling out position tensor distribution as the source of the remaining mismatch.
- Added failure-only per-user correlation, best-user matching, and norm-ratio diagnostics to distinguish a user permutation from head ordering or SDPA scaling without retaining device intermediates.

## Hardware checkpoint: per-user diagnostics localize the mismatch to SDPA layout

- The focused Llama decode run again completed all collectives and closed all 32 devices normally; compile and diff checks passed before hardware execution.
- The diagnostic ruled out a global user permutation: only 12 output rows reached approximately `0.9994` PCC, matching expected users in the repeating groups `(0, 3, 6)`, `(8, 11, 14)`, `(16, 19, 22)`, and `(24, 27, 30)`.
- Remaining rows had output norms approximately `45x` or `67x` the reference. Since all 32 K/V cache rows already pass, the active investigation is narrowed to Q head/user interpretation, decode SDPA program/output sharding, and the concat input contract.
- Dispatched two new read-only agents with dedicated goals to audit tt-buddy mesh-ordering guidance and production Galaxy Q/SDPA/head-layout usage; neither may access hardware.

## Hardware checkpoint: production SDPA core ordering fixes repeated decode

- Production Galaxy selects both its 32-core decode SDPA scheduling grid and eight-core SDPA output grid with `row_wise=True`; the test adapter had selected both column-wise.
- Separated cache-update K/V placement from SDPA placement and matched the production row-wise scheduling/output grids. Both Llama decode invocations then passed cache and output PCC checks.
- The run advanced to prefill and failed cleanly in the first `fill_cache`: a same-placement `to_memory_config` returned a storage alias that `_transition` invalidated by deallocating the old Python tensor wrapper. The next fix is a production ownership guard that skips no-op K/V transitions.

## Implementation checkpoint: prefill cache-fill allocation diagnostics

- The initial no-op guard and then an atomic K/V dtype conversion both preserved the same first-prefill `Input Tensor is not allocated` failure, while all host attention tests remained green (`64 passed`).
- Added a fail-fast allocation-state check for the four cache-fill operands so the next serialized hardware run identifies whether invalidation affects the bound cache or one of the converted head tensors.

## Hardware checkpoint: explicit cache typecast reaches prefill SDPA

- Allocation diagnostics identified invalid BF16 key heads: `to_memory_config(..., dtype=bf8)` returned a same-DRAM alias without casting, then old-wrapper release invalidated it.
- Matched production Galaxy by using explicit `ttnn.typecast` for K/V cache dtype conversion and preserved sibling lifetimes until both converted outputs exist. Prefill cache fill then completed.
- The next fail-fast error is prefill SDPA kernel placement on dispatch cores. The adapter is being aligned to production's `(7,10)` prefill geometry and the already-qualified worker sub-core grid.

## Hardware checkpoint: complete repeated Llama attention gate passes

- Qualified prefill SDPA with production Galaxy's `(7,10)` geometry and the full worker sub-core grid, removing dispatch-core placement from the active subdevice.
- The focused Llama test passed both decode invocations and both repetitions of 128- and 2048-token prefill, including output PCC and K/V cache checks: `1 passed, 1 deselected in 44.15s`.
- All 32 devices closed normally; no `tt-smi -r` was needed.
- Both dedicated tt-buddy/production audits completed and independently identified the column-wise SDPA grids as the numerical root cause; reports are in `tttv2_attention_mapping_audit.md` and `tttv2_attention_head_layout_audit.md`.

## Hardware checkpoint: Qwen fixture geometry corrected before compute

- The first Qwen variant stopped before device execution because its fixture combined hidden size 5120 with 64 heads of dimension 128, yielding an impossible 8192-channel attention result.
- Corrected the fixture to the Milestone A target, Qwen3-32B: hidden size 5120, 40 query heads, eight KV heads, head dimension 128, and Q/K norm enabled.
- Device setup and teardown were clean; no hardware reset was needed.

## Hardware checkpoint: Qwen head-local norm placement is explicit

- Correct Qwen geometry reached Q/K RMSNorm, which rejects the fused height-sharded create-head output.
- Added an atomic Q/K norm boundary in `Attention2D` so sibling head tensors are both relocated and normalized before old shared inputs are released; host attention tests remain green (`64 passed`).
- The default one-core head-local width shard only describes height 32 and cannot fit decode Q's physical height 256. The hardware Q/K norm recipe now explicitly uses interleaved DRAM input/output for regular RMSNorm.

## Hardware checkpoint: model-derived fused-QKV geometry clears Qwen

- Qwen3-32B reached numerical cache validation after its Q/K norm and gather resources were parameterized, but V-cache output contained zero head shards. Preserving the shared create-head backing allocation improved coverage without resolving the missing shards.
- The fused-QKV scratch/output grid was still fixed at Llama's ten local head cores (`8Q + 1K + 1V`). Derived the core count from each model's local fused QKV width, giving Qwen the required seven cores (`5Q + 1K + 1V`) while retaining ten for Llama.
- The focused Qwen test now passes two decode invocations and repeated 128- and 2048-token prefill, including output and K/V cache PCC checks: `1 passed, 1 deselected in 41.43s`.
- All 32 devices closed normally; no `tt-smi -r` was needed.

## Hardware checkpoint: combined Llama and Qwen gate passes

- Ran both WH Galaxy `(8,4)` model variants sequentially in one pytest process to exercise fixture teardown, fabric reinitialization, and JIT cache reuse across model geometry changes.
- Llama-70B and Qwen3-32B each passed two decode invocations plus repeated 128- and 2048-token prefill with output and K/V cache PCC validation: `2 passed in 53.93s`.
- Device teardown completed normally after both variants. Hardware reset was not required.

## Verification checkpoint: refreshed integrated host gate passes

- Re-ran the runtime, Galaxy CCL/resources, Prefetcher2D, concrete Prefetcher composition, and every 2D module host suite after the Attention hardware fixes.
- Result: `1259 passed, 1 skipped, 9 warnings in 251.51s`. The two additional passing tests cover Attention's explicit cache typecast and shared Q/K/V ownership behavior.
- The run includes the physical-batch-32 capture/replay lifecycle at sequence lengths 128/1024/2048 and all static strategy/config validation contracts.

## Verification checkpoint: final scope and supplemental 1D audit

- Compileall and `git diff --check` pass. No `models/common/modules/**/*_1d.py` implementation file is changed, no 2D module exposes `from_model_args`, and the common prefill runtime contains no Galaxy/model/architecture/mesh branch.
- An attempted affected-file 1D Attention check expanded to 516 hardware cases. Stopped the redundant expansion after 41 selected cases passed, including standard and paged prefill plus repeated decode; pytest reported `41 passed, 960 deselected` before the intentional interrupt.
- The interrupted run is supplemental only. Formal 1D exit evidence remains the previously completed selected matrix (`140 passed, 50 deselected`). All devices closed normally after the interrupt.

## Hardware checkpoint: unsupervised Milestone A device evidence re-run 2026-08-24

- Re-ran the 21 collected WH Galaxy `(8,4)` device cases at `de4c8f4e659` unsupervised, one pytest process at a time, on a complete 32-device 6U host. Full evidence package: `tttv2_milestone_a_device_evidence/REPORT.md`.
- Result is not a clean sweep: `16 passed, 3 failed, 2 BLOCKED (infra)` over 2 h 54 m. No source or test file was modified.
- Collection check: the brief's `-k wh_galaxy` sweep reports 27 cases, but six are the host-only `test_resolution_fails_closed_on_non_wh_galaxy` parametrizations in the sibling `_2d.py` files. The seven `*_wh_galaxy.py` files collect exactly the expected 21.
- Embedding2D, RotarySetup2D, MLP2D, LMHead2D and Sampling2D reproduced their recorded evidence exactly: 11 of 11 passed, all 32 devices closed normally, no reset required.
- RMSNorm2D confirmed only in part: distributed prefill 128/2048 passed for both dims, and fused-residual decode passed for Llama-8192.
- `final_norm_decode_batch_32_fused_residual_repeat[qwen-final-5120]` failed numerically at invocation 0: PCC `0.09771` in the file run and `0.13944` on an isolated re-run. Inputs are seeded identically, so the failure reproduces but its magnitude does not, pointing at a race or uninitialized read rather than a fixed mapping error.
- Both `head_local_128_qk_decode_and_prefill_repeat` parametrizations aborted before any kernel, in `LayerNormDeviceOperation::validate_on_program_cache_miss`: `Shard-padded width (2x128 = 256) does not align with tensor width 128`. Head-local Q/K normalization therefore has no numerical evidence at all in this run.
- All three RMSNorm failures were confirmed deterministic by individual node-ID re-runs, and every RMSNorm process closed all 32 devices normally, so no reset was needed.
- Attention2D produced no numerical evidence. The whole-file run and both individually-run node IDs each consumed the full 2700 s bound with zero pytest result lines, on a freshly `tt-smi -glx_reset` Galaxy each time; three resets, all reporting `Re-initialized 32 boards`.
- The hang is not JIT compilation: `/proc` sampling during the Llama run found one thread spinning at ~100 % CPU, the main thread parked in `futex_wait_queue`, ~290 sibling threads in `hrtimer_nanosleep`, and zero `sfpi`/`riscv`/`clang` subprocesses. UMD topology discovery, firmware bundle 18.12.1, and `Fabric initialized on Device 0..31` all completed first, so the stall sits inside Attention2D device execution after mesh bring-up and before the first `comp_pcc`.
- Recovery cap of two attempts per group is spent, so `llama-70b` and `qwen3-32b` are terminal at `BLOCKED (infra)`. This contradicts the recorded `2 passed in 53.93s` combined gate but does not establish an Attention2D numerical defect — no PCC was ever computed.
- Passing cases print no PCC value: `comp_pcc` returns it but logs nothing on success and no `*_wh_galaxy.py` test logs it, so passes are assertion-backed at threshold 0.99 rather than numerically quoted.
- Not covered by this device set, matching the status page's own caveats: stochastic `Sampling2D` hardware, real-device physical-32 trace runs, the Galaxy CCL/resources and Prefetcher2D host suites (evidenced only indirectly through MLP2D/RMSNorm2D), and the batched-prefill runtime.
- Device left clean: final `tt-smi -glx_reset` re-initialized all 32 boards and `tt-smi -ls` confirms 32 present.

## Implementation checkpoint: 2D module tests share the 1D reference 2026-08-25

- The 2D WH Galaxy tests qualified their modules against hand-written torch re-implementations while the 1D suites compared against the real reference (HuggingFace / `torch.nn`). A re-implementation can only prove the test agrees with itself, so the 2D tests now use the same reference the 1D tests do.
- New `models/common/tests/modules/_hf_reference.py` owns the shared plumbing: `HfAttentionWrapper`, `IdentityRotaryEmbedding`, `reverse_permute` / `reverse_permute_1d`, `get_attention_weights_from_ref_model`, `get_mlp_weights_from_ref_model`. `test_attention_1d.py` and `test_mlp_1d.py` now import these instead of defining their own copies; no 1D behaviour changed.
- `test_attention_2d_wh_galaxy.py`: dropped `_project_qkv`, `_decode_reference`, `_prefill_reference` and `_fused_qkv_weight`. It builds a locally configured `LlamaConfig` / `Qwen3Config` attention block, drives it through `HfAttentionWrapper`, and takes the fused-QKV weight straight from `get_attention_weights_from_ref_model(..., num_devices=8)` — the row-fused layout Attention2D expects.
- Rotation stays identity on both sides (`IdentityRotaryEmbedding`), matching how the module is wired: RoPE has its own Milestone-A qualification, so this test remains about projection, SDPA, cache and CCL.
- Two layout facts make the comparison exact. Q/K reach the device in Meta (interleaved) layout, so the reference K cache is compared after the same per-head permutation (`reverse_permute_1d`); Q·Kᵀ is invariant under a permutation shared by Q and K, so the output needs no adjustment. Decode starts mid-cache with no prefill, so `reset_cache_to_zeros` seeds the reference with the same zero-filled history the freshly allocated device cache holds.
- `test_mlp_2d_wh_galaxy.py`: `_torch_mlp` replaced by the HF `LlamaMLP`, weights read back with `get_mlp_weights_from_ref_model`. `test_rmsnorm_2d_wh_galaxy.py`: `_torch_rms_norm` replaced by `torch.nn.RMSNorm`.
- Equivalence checked on host before any device time: at matched geometry the HF attention reference reproduces the old hand-rolled decode/prefill outputs and K/V caches at PCC 1.0 for both Llama and Qwen3, and the MLP and RMSNorm references are bit-exact (max abs diff 0.0). Weight draws now consume the RNG in HF parameter order, so the input tensors differ from earlier runs even at the same seed.
- Deliberately unchanged: `test_embedding_2d_wh_galaxy.py` already compares against `torch.nn.functional.embedding`, `test_lm_head_2d_wh_galaxy.py` against a plain `torch.matmul`, and `test_rope_2d_wh_galaxy.py` builds cos/sin tables the same way `test_rope_1d.py` does — and there they are module *inputs*, not a re-implementation of what the module computes. `test_sampling_2d_wh_galaxy.py` is stochastic and has no math reference. The host-only `*_2d.py` suites are mock-based and hold no numerical reference.
- Host gates: pre-commit clean on all six files; `models/common/tests/modules/{attention,mlp,rmsnorm}/test_*_2d.py` `115 passed`; the device-free 1D attention selection `14 passed, 6 skipped`; every touched file collects. No WH Galaxy device evidence for the rewritten 2D tests yet — the Attention2D device case was already `BLOCKED (infra)` in the 2026-08-24 run.

## Hardware checkpoint: RMSNorm2D fused decode and head-local Q/K fixed 2026-08-25

- Started from the three RMSNorm2D failures in `tttv2_milestone_a_device_evidence/REPORT.md`. First confirmed the shared-reference change is numerically neutral: on host, `torch.nn.RMSNorm` and the removed `_torch_rms_norm` agree at PCC ~1.0 (max ratio 1.0) for 8192/5120/128, so it cannot explain any device result.
- Re-running the file at the current tree gave **4** failures, not 3: `final_norm_decode_batch_32_fused_residual_repeat` now failed for **Llama-8192 too** (PCC 0.1555) alongside Qwen-5120 (0.0977). The recorded Llama pass was luck, not a passing configuration.
- Isolating the fused decode showed the failure is not the queued repeat: `count=1` and `count=2` produce identical PCC (~0.14) on both invocations, and the in-place residual sum is correct at 0.9999979. Only the normalization scale was wrong.
- Dumping the tensors explained the magnitude: the device output equals the expected output times ~1.45e37, i.e. `rsqrt` of ~zero. The stats the kernel reduced were not the gathered stats.
- Root cause: `fused_rms_minimal` creates its stats circular buffer (`cb_stats`) on the **first core of the norm input shard grid** and binds it to the stats tensor's L1 address (`rms_allgather_program_factory.cpp`: `CreateCircularBuffer(program, sender_cores, ...set_globally_allocated_address(*stats->buffer()))`); it never reads that tensor over the NoC. The test's persistent stats buffer sat on `x=1` while the norm grid starts at `x=2`, so the kernel reduced whatever the allocator had left at that address on the sender core.
- Confirmed by readback and sweep. The stats tensor itself is always correct — device 0 holds `[2.047, 2.109, 1.992, 1.953]` against torch per-device `mean(x^2)` `[1.984, 1.959, 1.991, 1.969]` — so the fabric all-gather was never at fault. Sweeping placement: stats on the norm sender core passed every run (5 runs, PCC stable at 0.9999860); stats on `x=1` produced 0.0977 / 0.1394 / 0.1555 / 0.1701 in some processes and ~0.9999 in others. The same parametrization flipped between fail and pass across processes, which is the signature of reading aliased L1, and explains both the earlier "passed" checkpoints and the report's non-reproducible magnitudes.
- Module fix 1 (fused stats placement): `decode_stats_memcfg` now defaults to the norm grid's first core instead of a hardcoded `x=1`, and `_decode_fused_residual_norm` calls the new `_require_fused_stats_placement`, which rejects an injected persistent stats buffer that is not L1-sharded on that core. Silent numerical corruption becomes a loud `ValueError`.
- Module fix 2 (head-local Q/K): head-local resolution built a `128`-wide width shard over the hardcoded two-core `x=2..3` range, so a 128-wide tensor declared `2 x 128 = 256` of padded width and aborted in `shard_spec_validation.cpp:104` before any kernel. Head-local decode now defaults to interleaved DRAM like prefill - it is a plain `rms_norm` with a placement-derived program config, and decode feeds it `batch * local heads` rows rather than the fixed 32 a width-sharded recipe would pin. `attention_2d`'s only head-local consumer already overrode all four memcfgs to DRAM, so this makes the default match the one qualified usage. The distributed norm grid is now derived from `grid_width` rather than hardcoding `x=3` as its end.
- Test fix: `_resources_config` allocates the decode persistent stats buffer on `(2, 0)`, the norm sender core.
- Device evidence on the full 6U WH Galaxy `(8,4)`: `8 passed in 33.06s`, repeated three times, plus both fused decode node IDs run alone in fresh processes (`1 passed, 1 deselected` each) - the exact worst case that previously failed. All 32 devices closed normally every run; no `tt-smi -glx_reset` was needed at any point.
- Host gates: `models/common/tests/modules/rmsnorm/test_rmsnorm_2d.py` `19 passed` (three new contracts: the fused stats-placement rejection, head-local staying interleaved, and stats sharing the norm sender core); `test_attention_2d.py` plus `models/common/tests/models/galaxy` `90 passed`.
- Note for the reference model: `models/demos/llama3_70b_galaxy` has the same shape of exposure. Its `LAYERNORM` persistent buffer is on `(1,0)` while `DistributedNorm` puts the non-`qk_norm` (Llama) norm grid at `x=2..3`, so its sender core is `(2,0)`; only the `qk_norm` (Qwen) layout at `grid_offset=(1,0)` co-locates the two. Not investigated further here, and no change was made outside `models/common`.

## Hardware checkpoint: Attention2D decode CCL hang fixed 2026-08-25

- Picked up the last open item from `tttv2_milestone_a_device_evidence/REPORT.md`: Attention2D was `BLOCKED (infra)` with no numerical evidence, after four attempts and three galaxy resets produced zero pytest result lines. First confirmed the shared-reference rewrite is not implicated - the hang predates it and reproduces identically.
- Reproduced the hang and located it exactly. A repeating `faulthandler.dump_traceback_later` plugin (diagnostic only, never committed) caught two identical all-thread dumps 90 s apart, both parked at `resources.synchronize("decode")` in `test_attention_2d_wh_galaxy.py:487`, immediately after `ttnn.experimental.all_reduce_create_qkv_heads` in decode invocation 0. So it is a hard stall in the fused QKV collective, not slow progress, not JIT, and not a numerical failure.
- Ruled out the op, the topology, and the host. The 6U-qualified `tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py::test_all_reduce_qkv_heads_fuse_perf_6U` - same `Topology.Ring`, `FABRIC_1D_RING`, `num_links=3`, `cluster_axis=1`, same `[1,1,32,1280]` geometry, same 24-core ring input and 10-core output grids - passes here in `6.48s`. The Linear/`FABRIC_1D` sibling is `skipif(is_6u())`, so Ring is the correct qualified choice and matches what the test already used.
- Root cause is the decode resource plan, not the module. `choose_worker_cores_fuse` (`all_reduce_create_qkv_heads_program_factory.cpp:49-93`) takes `device->worker_cores(TENSIX, sub_device_id)`, subtracts the reserved output cores, and claims the first `num_links` cores in `(y, x)` order; each sender then receives `semaphore.address()` as an absolute L1 address. `galaxy_mode_plan` builds its worker subdevice as the whole compute grid `(0,0)-(6,9)`, but the test narrowed `semaphore_cores` to the production `worker_cores` (`x in {1,2,3,5,6}`). With the 10-core output grid occupying all of column `x=1`, the senders land on `(0,0)`, `(2,0)`, `(3,0)` - and `(0,0)` is outside the semaphore's core range set, where `create_global_semaphore` never reserved or zeroed that address. The collective polls uninitialized L1 forever.
- This also explains the report's open puzzle. Whether the run hangs depends on whatever the allocator left at that address on `(0,0)`, so the same parametrization can pass in one process and hang in another - exactly the earlier `2 passed in 53.93s` versus four consecutive full-bound hangs, and the same signature as the RMSNorm fused-stats aliasing fixed earlier today.
- Test fix: the decode plan no longer narrows `semaphore_cores`, so the mode's global semaphores cover the whole worker subdevice. Production narrows the subdevice itself (`galaxy_prefetch_decode_mode_plan` makes the worker subdevice equal `worker_cores`, keeping the two consistent); this test cannot, because its decode QKV/WO matmuls use a `(7,1)` grid that spans `x=0..6`.
- Recorded the invariant on `galaxy_mode_plan` so it is not reintroduced: narrow `semaphore_cores` only for a collective that binds its semaphore to a grid it owns, the way the fused RMS all-gather does. The generic async CCLs (`all_gather_async`, `reduce_scatter_minimal_async`, `all_reduce_async`, `all_reduce_create_qkv_heads`) choose senders from the subdevice and must keep the default. No hard guard was added, because `test_rmsnorm_2d_wh_galaxy.py` legitimately narrows to `norm_cores` and passes.
- Device evidence on the full 6U WH Galaxy `(8,4)`: the isolated `llama-70b` node ID `1 passed in 164.64s` from a cold JIT cache, then the complete file `2 passed in 175.40s`, then three further whole-file repeats in fresh processes at `75.77s`, `76.15s` and `74.90s`. Both Llama-70B and Qwen3-32B cover two decode invocations plus repeated 128- and 2048-token prefill with output and K/V cache PCC at 0.99. All 32 devices closed normally in every run and no `tt-smi -glx_reset` was needed after the fix.
- Host gates: `test_attention_2d.py` plus `models/common/tests/models/galaxy` `90 passed`; pre-commit clean on both changed files. Only two files changed, both under `models/common/tests`: the decode plan kwarg and the `galaxy_mode_plan` docstring. No module implementation and no threshold was touched.
- `REPORT.md` is now out of date on this row: Attention2D is no longer `BLOCKED (infra)`, and its exit-gate claim is reproducible again.

## Hardware checkpoint: Sampling2D stochastic hardware coverage 2026-08-25

- Closed the first Milestone A gap: `Sampling2D` had no stochastic device evidence at all. `test_sampling_2d_wh_galaxy.py` passes both `forced_argmax=True` and `temperature=0.0`, which collapses every slot to `k=1, p=0.0, temp=1.0`, so `ttnn.sampling` never took its stochastic branch on hardware. New `models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py` adds 9 device cases on the same qualified `(8,4)` geometry: top-k/top-p containment over 5 parametrizations, padded-vocabulary exclusion under stochastic sampling, seeded repeatability with per-slot seed stability, unseeded freshness, and per-slot heterogeneous k/p/temperature.
- **The new coverage found a real module defect.** `ttnn.sampling`'s `temp` argument is the *reciprocal* temperature - its kernel multiplies candidate logits by `temp` before the softmax - but `_update_call_buffers` was writing the raw `T`. Every request at `T != 1.0` sampled from a distribution warped in the inverse direction: `T = 0.8`, intended to sharpen, instead flattened and admitted tokens the nucleus should have excluded.
- Root cause of the *miss*, not just the bug: `1.0` is its own reciprocal, so the defect is exactly invisible at `T = 1.0`, and the greedy path forces `temp = 1.0` unconditionally. The only pre-existing hardware test is greedy. The defect was structurally unreachable by the existing coverage - any test that could catch it had to be stochastic *and* off `T = 1.0`. Both new cases satisfying that caught it immediately.
- Fix is one line, `sampling_2d.py:213`: `temperature_values[slot] = 1.0 if force_greedy else 1.0 / call.temperature[index]`, with the convention recorded in a comment at `:202-205`. `temperature == 0.0` folds into `force_greedy` on the line above, so the reciprocal is never evaluated at zero. `sample_host` was already correct (it divides, `:260`), so host and device now agree semantically and no host-path change was needed.
- Evidence chain, all in `tttv2_milestone_a_gap1_evidence/logs/`: `run01_calibration` `9 passed` with the two nucleus bounds deliberately unconstrained, to observe rather than assert; `run02_prefix_defect_demo` `2 failed, 7 passed` against the final bounds with the pre-fix code - the two failures are *exactly* the two `T != 1.0` cases, 4/32 and 2/8; then `run03`/`run04`/`run05` `9 passed` each in three fresh processes, 0 violations on all 23 report lines in all three.
- The correlation is the argument. Every case at `T = 1.0` showed 0 violations pre-fix and post-fix alike - including the `p=0.9` nucleus case, which exercises the same bf16 cumsum boundary as the failing `p=0.5` case, so the nucleus arithmetic is not the discriminator. Only the two `T != 1.0` cases ever violated, and they vanished to 0 after a change touching nothing but the temperature buffer. `T = 1.0` is the fixed point of `T -> 1/T`; that is exactly what a multiply-vs-divide mismatch predicts.
- Tolerances were **tightened**, not relaxed: the two `p in (0,1)` nucleus bounds went from the unconstrained calibration values (32 and 8) to 1. The observed post-fix maximum across runs 03-05 is **0** everywhere, including both nucleus cases, so the 1 is deliberate headroom for the bf16 softmax/cumsum boundary - which did not manifest on this geometry at all once the temperature was correct - not an observed requirement. Any violation reported here is a regression. `p in {0.0, 1.0}` cases stay at zero tolerance.
- Host gates: `test_sampling_2d.py` `27 passed`, including new `test_device_temperature_buffer_holds_reciprocal_temperature`, which pins the buffer to `1/T` so the fix cannot silently regress without device time; one pre-existing assertion at `:168` corrected, it had pinned the buggy raw-`T` value. `test_sampling_1d.py` `140 passed, 50 deselected`; combined `167 passed, 50 deselected`. `_hf_valid_token_set` moved out of the 1D test into shared `models/common/tests/modules/_hf_reference.py` as `hf_valid_token_set` so both suites share one HuggingFace-derived reference; no 1D behaviour changed.
- The handoff's expected "166 passed" for the 1D suite was a combined two-file count recorded against the 1D file alone, taken before the new host test existed: `140 + 26 = 166`, and `140 + 27 = 167` now. The 50 deselections are the `mesh_shape` cross-product filter in `models/common/tests/conftest.py:95`, deterministic and device-free. Not a regression.
- `pre-commit` is clean on the four files this job created or rewrote. `test_sampling_2d.py` fails `prefer-expect-error` on seven `pytest.raises` blocks that exist verbatim at HEAD and that this job's diff neither adds nor removes; the hook landed in `31e21e4d190` and 47 repo test files currently trip it. Left standing rather than silenced with `# allow-pytest.raises` overrides - suppressing a lint signal on code this job did not write is the same category as relaxing a threshold. Reported in `REPORT.md` §6.
- All 32 devices closed normally in every run, `tt-smi -ls` confirms 32 boards present afterwards, and no `tt-smi -glx_reset` was needed at any point.
- Not established, deliberately: RNG *distributional* correctness (containment proves support membership, not probabilities); device-vs-`sample_host` token equality, which is not a design property - `_device_seed` masks to 31 bits and feeds `ttnn.manual_seed`, `_host_seed` to 63 and feeds `torch.multinomial`; `top_k > 32`, which is out of contract and where the containment argument stops being sound; trace/capture coverage; any geometry other than `(8,4)`.
- **Follow-up filed - 1D reference temperature, a testable prediction.** The 1D tolerances are asymmetric in exactly the direction a multiply-vs-divide mismatch predicts: `t=0.5` allows 6 violations, `t=2.0` allows 2 (`test_sampling_1d.py:832-834`). The module is not wrong there - `Sampling1D` passes `temp` straight through, so *its* `temp` already is the op's `1/T`. The mismatch is in the test's HF reference, which warps with the raw value while `TemperatureLogitsWarper` divides, so the disagreement is worst where `T` is furthest from 1.0 - larger at 0.5 than at 2.0, which is the observed asymmetry. Prediction: passing `1/temp` to `hf_valid_token_set` should collapse `t=0.5` and `t=2.0` toward the `t=1.0` baseline and remove the asymmetry. Note the baseline is not zero - the same block has `k32-p0.5-t1` allowing 3, and at `T = 1.0` the reciprocal is a no-op, so a genuine bf16 nucleus effect exists in 1D independent of temperature. The prediction is about the temperature-dependent excess (6 and 2 against a baseline of 3), not that all three go to 0. The 2D bounds could reach 0 because those tests use a tie-free candidate ladder; the 1D tests draw over full vocabularies where bf16 top-k ties are common. Not touched here - it is 1D test surface, needs its own hardware run, and is out of this brief's scope. If it holds, the 1D tolerances are masking a reference bug rather than a hardware effect.

## Hardware checkpoint: Prefetcher2D and Galaxy resource hardware qualification 2026-08-26

- Closed the second Milestone A gap. `Prefetcher2D` and the Galaxy resource owner had 446 lines of host coverage built entirely on `FakeMesh`/`FakeTensor`/`MagicMock`, and ran on silicon only *incidentally* through MLP2D and one RMSNorm2D test - and every one of those device tests pins a single mode for its whole lifetime, so on real hardware **no prefill<->decode transition had ever executed at all.** New `models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py` (863 lines, 7 tests, 8 cases) closes that. The job spanned three sessions: session 1 (08-25) wrote the suite and was killed by an org monthly spend limit; session 2 (08-26) ran the attention case once and killed itself by `kill -TERM`-ing its own `claude -p` wrapper after `pgrep -af pytest` matched the prompt on its command line; session 3 (08-26) finished it. Neither early death was technical, and both sessions' findings were on disk before they died.
- Payload is the qualified MLP2D geometry, imported wholesale. New `models/common/tests/modules/_mlp_2d_galaxy.py` (468 lines) holds the resource plan, ring core map, DRAM-sharded weight layout and HF reference; `test_mlp_2d_wh_galaxy.py` is gutted to `+22 -421` and imports what it used to define. Same shared-plumbing pattern as `_hf_reference.py`. So the PCC numbers here are directly comparable with the recorded MLP2D evidence and a wrong core grid cannot creep in by transcription.
- `_wh_galaxy_hardware.py` `+71 -23`: `_create_hardware_prefetcher` split so `GALAXY_PREFETCH_SENDER_COORDS`, `galaxy_prefetcher_sender_cores()` and `galaxy_prefetcher_config(..., global_cb_size=...)` are public. Behaviour-preserving - the old function now just calls the new config factory with the same values.
- Transition matrix actually executed: `("decode","prefill","decode","prefill","decode") * 2 + ("prefill","prefill")`. Twelve steps, one real MLP2D invocation and one PCC assertion each, so the plan's full list is covered including the `decode->decode` seam between the two cycles and the `prefill->prefill` tail. **Every decode step returns exactly 0.9982190 and every prefill step exactly 0.9993101** - the transition is numerically inert, which is the result you want: activating a mode restores that mode's context bit-for-bit rather than leaving it perturbed by the mode it came from. Threshold 0.99, untouched.
- At each step `_SubdeviceRecorder` asserts the device really loaded that mode's `sub_device_manager_id` and set that mode's `stall_group`, that `active_mode` matches, and that the DRAM prefetch producer exists iff the mode is decode. ttnn exposes no getter for either, so the four `ttnn.MeshDevice` lifecycle methods are shadowed with forwarding wrappers (`ttnn.MeshDevice` is bound with nanobind `dynamic_attr`); the device sees an unchanged call sequence.
- Rollback is proved by numbers, not bookkeeping: inject a `RuntimeError` at `_dram_prefetch_start` during `activate("decode")`, confirm the rollback re-loaded the prefill manager and stall group, then show the **next prefill invocation still hits 0.9993101** - the same value it hit pre-failure - and that the transition that failed succeeds once the injection is gone (0.9982190).
- Case 1 reads the packed address tensor back off the senders with `ConcatMeshToTensor(dim=0)` and asserts all `32 x 12 = 384` rows equal `[2968640, 3664960, 4361280]`, each entry the registered tensor's actual `buffer_address()`. Nothing proved that on silicon before.
- **Finding F1 - `cleanup()` cannot free the global circular buffer, and this is the ownership defect the gap existed to find.** ttnn exposes no free for a `global_circular_buffer`, so its L1 is reclaimed by RAII when the last handle dies - and `seal()` publishes the live handle inside the immutable decode context, which every consumer keeps (`MLP2D` as `decode_prefetch_context`). `cleanup()` does drop its own `_global_cb` and clear `_contexts`; it cannot drop anyone else's. A consumer that outlives the owner keeps 55444480 B of L1 pinned and the next owner's `seal()` dies in `create_global_circular_buffer` with `Out of Memory: Not enough space to allocate 55444480 B L1 buffer across 70 banks`. After cleanup the owner truthfully reports `owned_resources == ()` while the L1 is still resident: accurate about ownership, silent about residency.
- F1's evidence chain. It **failed the real suite first** (`dev03`: both cleanup cases, frames landing in `_wh_galaxy_hardware.py:297 -> prefetcher.seal()`) - the gap2 handoff attributes both `dev03` and `dev04` failures to the agent's own `UnboundLocalError`, which is true of `dev04` only, and `dev04`'s bug was introduced *by the fix for F1*. A ten-step scratch probe then isolated it: holding the whole context across cleanup fails; holding **only** `context.global_cb` fails identically; holding only `weight_address_metadata` is fine and the metadata is already deallocated; and holding nothing - never taking a context handle at all - is fine, so the owner's own bookkeeping is sufficient whenever no consumer handle exists.
- F1 recommendation, argued in the report and **not applied**: adopt and document the ordering contract now (docstring plus the README prefetcher-ownership paragraph), because `modules/README.md:212` currently calls `Prefetcher2D` the resource root "for ... the global circular buffer" and says the executor "owns deterministic cleanup", which overstates what `cleanup()` can deliver. Fix the design in Milestone B/C by having `Prefetcher2DContext` expose `global_cb` as a property that fetches from the owner instead of storing the handle, which makes the contract enforceable rather than documented. Rejected: refcount introspection in `cleanup()` - fragile, makes the one method that must always succeed throw, and inverts the teardown order. `prefetcher_2d.py` was not modified; nothing under `models/common/modules/` was.
- **Test 5 (Attention2D decode with an active prefetch producer) is terminal FAILED and is an incompatibility by construction, not a defect. Recommended for Milestone B.** It aborts deterministically in ~40 s with `TT_FATAL @ fd_mesh_command_queue.cpp:388: sub_device_ids.size() == 1 / Programs must be executed on a single sub-device`, reproduced on two separate days. The raising call is pinned to `attention_2d.py:851`, the decode QKV `ttnn.linear` on `[1,1,32,2048] x [2048,1280]`.
- Test 5, what conflicts with what. `galaxy_prefetch_decode_mode_plan` splits the `7x10` grid into senders on `x in {0,4}` (12 cores) and workers `x in {1,2,3} u {5,6}` (50 cores), with 8 dummy senders in neither. `_matmul_program(_BATCH_SIZE=32, ...)` hardcodes `grid_x=7` and computes `grid_y = min(4, ceil(32/32)) = 1`, so `compute_with_storage_grid_size=(7,1)`, which ttnn normalizes to `allowed_worker_cores = CoreRange((0,0),(6,0))`: **2 sender cores `(0,0)`/`(4,0)` and 5 worker cores**. `program.cpp:2166 determine_sub_device_ids` intersects each kernel group's core ranges with every subdevice and collects all that intersect, so the set has two elements and the workload is refused before it runs.
- Test 5, why it cannot be narrowed. `ttnn.linear` does take a `sub_device_id` and `Attention2D` already forwards one when given a prefetch context, but the subdevice set is derived from kernel placement, never from that argument. `allowed_worker_cores` exists on the multicast config and would be the mechanism, but `matmul_program_config.cpp:1075` `TT_FATAL`s unless it is a dense rectangle, and the worker subdevice is 50 cores in a `6x10` bounding box. Every dense rectangle anchored at the origin includes column `x=0`, a sender column. Offset rectangles inside the worker subdevice exist (`((1,0),(3,0))`, `((1,0),(3,2))`, `((5,0),(6,3))`) but none has 7 cores, so each changes `per_core_N` and moves the output shard grid - the factory derives `start_core` from `allowed_worker_cores.bounding_box().start_coord` (`matmul_device_operation.cpp:2541`) - which re-tiles the qualified decode projections and re-derives the grids the fused QKV collective is qualified against. No such value can be argued to be "the correct one", so none was chosen.
- Test 5, and it is the wrong target anyway. Production decode attention with a prefetcher uses the ring/`gather_in0` matmul that reads weights out of the global CB, the way MLP2D does - and that form already exists in the attention suite: `_decode_ring_config` builds `qkv_program`/`wo_program` as `MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8,3), gather_in0=True, hop_cores=((3,6),), num_global_cb_receivers=2)` over 24 `ring_cores`, **all 24 plus the hop core inside the worker subdevice.** Those two configs are built and never passed to `Attention2D.from_config`; `_make_module` wires the `(7,1)` DRAM-sharded form and `decode_prefetch_context=None`. So even with the grid conflict gone, the case as written would prove coexistence, not consumption - the producer streams `attn.wqkv`/`attn.wo` into the CB and nothing reads them. Wiring attention decode onto the ring/global-CB matmul is choosing a production grid, which is Milestone B's job; doing it here would mean qualifying a new attention decode geometry under cover of a prefetcher test.
- **A `TT_FATAL` abort inside a multi-subdevice program leaves the mesh un-drainable, and this costs a reset for anyone sequencing Milestone B device work.** After the abort the process sits forever in `mesh_device` fixture teardown; a gdb backtrace puts the main thread in `FDMeshCommandQueue::~FDMeshCommandQueue -> clear_expected_num_workers_completed -> wait_for_outstanding_reads -> pthread_cond_wait` under `MeshDevice::close()`. SIGTERM cannot be serviced - the main thread is blocked in a C++ destructor with the GIL released - so it takes SIGKILL. Not specific to attention or the prefetcher: any `TT_FATAL` out of `enqueue_mesh_workload` does it.
- Operational note for the next teardown stall: **the in-process `faulthandler.dump_traceback_later` trick does not work past a test failure.** pytest's built-in faulthandler plugin cancels every pending dump in `pytest_exception_interact` (`_pytest/faulthandler.py:114`, `tryfirst`), so the failure you want to trace past disarms the dumper. Use gdb from outside, as here. What *does* survive is a plugin that flushes `report.longrepr` at the end of the call phase - that is how the raising call was pinned, since the terminal summary never prints when teardown hangs.
- **Finding F2 - an undersized `global_cb_size` is silently accepted at `seal()`.** The gap brief asked for a device test that it is rejected; there is no such rejection on host or device. `seal()` validates only `resolved_cb_size > 0`, and on device `global_cb_size=1024` and `4096` were both accepted against weights needing far more. The host test named in the brief, `test_seal_derives_cb_size_and_rejects_undersized_configuration`, asserts derivation only - its name overstates it. Left as a decision to make, not a test to write, which is why `galaxy_prefetcher_config(..., global_cb_size=...)` currently has no caller outside the scratch probe.
- Device evidence on the full 6U WH Galaxy `(8,4)`, `FABRIC_1D_RING`: the seven lifecycle cases `7 passed, 1 deselected` in **three fresh processes** at 224.28 s, 225.96 s and 226.15 s, with the `[gap2]` output **byte-identical across all three** - every PCC to seven decimals and the sealed weight addresses. That is the anti-aliasing statement: unlike both 2026-08-25 root causes, nothing here depends on residual L1 or allocator history. Case 8 is deselected by node ID in the file runs so its abort cannot wedge the mesh mid-file, and run separately. No `tt-smi -glx_reset` was needed between any of the three runs.
- Regression gates all green: prefetcher/galaxy/MLP host suites `78 passed`; pre-commit clean; MLP2D device `4 passed`; RMSNorm2D device `8 passed`; **Attention2D device `2 passed` in 75.36 s** - the one outstanding gate, since `test_attention_2d_wh_galaxy.py` imports the refactored `_wh_galaxy_hardware.py`. 75.36 s sits inside the 74.90-76.15 s recorded on 08-25, so the shared-helper split is behaviour-preserving for its third consumer too.
- Pre-existing noise recorded so it is not misread as new: each whole-file run emits 128 `matmul_multi_core_reuse_mcast_1d_optimized_helper: program_config.allowed_worker_cores not populated; auto-populating ... will become a hard error in a future release` warnings. They come from the MLP2D ring matmul reaching the factory through a path that bypasses `ttnn::prim::matmul()` and so never calls `normalize_program_config`; `dev01_mlp_regression.log` has 64 of them. Benign today because `gather_in0` takes its cores from the input shard grid, but the prefetch-fed ring matmul is the only prefetch consumer qualified anywhere, so it wants attention before the deprecation lands.
- Not established, deliberately: the payload is MLP2D geometry only, so the contexts are qualified for that consumer shape (3 weights, the 24-core ring, `global_cb_size = 728*1088`) - and Test 5 is the concrete demonstration that "a different consumer" is not a formality; Attention2D consuming prefetched weights, which is not reachable from its current decode configuration at all; capture/replay of a transition (the plan names it; `Prefetcher2D` has no trace mode and `context("trace")` is rejected, which case 7 asserts); failure injection anywhere other than the `_dram_prefetch_start` seam the module exposes; any mesh shape other than `(8,4)`. The foreign-mesh registration rejection *was* observed working on device in the scratch probe, but the probe process then aborted at interpreter shutdown with `MeshDevice cq ID 0 is in use by parent mesh ID 0`, which is why case 7 documents it as out of scope on a reserved 32-device Galaxy.
- Evidence and the full argument in `tttv2_milestone_a_gap2_evidence/REPORT.md`, including drafted-but-not-applied replacement rows for `MILESTONE_A_STATUS.md`'s `Prefetcher2D` and `Galaxy CCL/resources` entries and for `modules/README.md:212`, whose "not yet qualified on hardware" caveat can now be dropped - provided the ownership sentence before it is weakened at the same time, since that is the claim F1 disproves.

## Verification checkpoint: Milestone A committed, full device matrix and host gate re-run 2026-08-26

- Committed the whole Milestone A change set and pushed it: `cf803f23647` (four defect fixes, two new device suites, shared HF-reference plumbing) and `bf403d93fed` (evidence packages, gap briefs, work-log checkpoints). Branch `gongyu/tttv2_wh_glx_2d_modules`, from `de4c8f4e659`.
- The repo's `prefer-expect-error` hook blocked the first commit on twelve `pytest.raises` blocks in `test_rmsnorm_2d.py` and `test_sampling_2d.py` - eleven pre-existing, one added by the D1 contract test. Converted all twelve to the sanctioned `expect_error` fixture rather than suppressing the hook; both suites still pass (`46 passed`). The gap-1 report had flagged these as knowingly left standing; they are now fixed rather than carried.
- `.gitignore`'s `*.log` rule excludes every raw pytest log in the three evidence packages. Committed the `REPORT.md`/`ENVIRONMENT.md` analysis and left the logs on the host; each report names the log behind each claim. The 839 KB agent jsonl in the 08-24 package was excluded deliberately.
- **Integrated host gate re-run at the committed tree: `1263 passed, 1 skipped, 9 warnings in 265.06s`**, up from the `1259 passed` of 2026-08-19. The four extra tests are the new RMSNorm2D stats-placement/head-local contracts and the Sampling2D reciprocal-temperature contract.
- First host-gate attempt was invalid and is retained as `host01_integrated_gate_ABORTED_bad_selection.log`. Passing `models/common/tests/modules/prefetcher` as a *directory* collected that module's `*_wh_galaxy.py` device suite alongside its host suite, so a host-only gate opened the mesh and ran the terminal-failing `attention_decode_with_active_prefetch`. It aborted with the L3 `TT_FATAL: Programs must be executed on a single sub-device` and hung in the un-drainable teardown; killed and reset (`reset01_after_bad_selection.log`, `Re-initialized 32 boards`). Host-only selections need `--ignore-glob="*_wh_galaxy*.py"`; now documented in `modules/README.md`.
- **Full device matrix re-run as one sweep at `bf403d93fed`: 37 of 37 cases passed, 14:39:07Z -> 14:55:02Z (15 m 55 s), no `tt-smi -glx_reset` needed at any point.** Per group: embedding `2 passed` 57.32s, rope `2 passed` 7.31s, rmsnorm `8 passed` 33.27s, mlp `4 passed` 116.40s, lm_head `2 passed` 65.68s, sampling greedy `1 passed` 7.46s, sampling stochastic `9 passed` 29.96s, attention `2 passed` 74.61s, prefetcher `7 passed, 1 deselected` 227.64s. Every group logged `Cluster destructor completed`; all 32 devices closed normally in all nine.
- The one deselection is `attention_decode_with_active_prefetch`, excluded on purpose: terminal FAILED by construction (L3) and its abort leaves the mesh un-drainable, so including it would cost a reset and contaminate the sweep. Diagnosis stands in `tttv2_milestone_a_gap2_evidence/REPORT.md` §6.
- This is the artifact the 2026-08-24 run could not produce: one coherent green matrix at one tree, covering the original 21 cases plus the 9 Sampling2D stochastic and 7 Prefetcher2D cases added since. Reproducible with `bash tttv2_milestone_a_final_evidence/run_device_matrix.sh`.
- `modules/README.md` updated: the 1D-vs-2D section now records the hardware qualification, both ownership limits (global-CB lifetime, Attention2D/prefetch grid incompatibility), the host-only-vs-device suite split with the `--ignore-glob` guidance, and the shared `_hf_reference.py` / `_mlp_2d_galaxy.py` plumbing. `MILESTONE_A_STATUS.md` rewritten against the committed diff; modularity scorecard re-audited, with `git diff --stat de4c8f4e659..HEAD` confirming zero changed `models/common/modules/**/*_1d.py` and zero changed `models/common/llm_runtime/` files.
- **Exit gate: eleven of twelve lines met.** The outstanding one is the 1D regression matrix (attention and MLP 1D hardware selections after the `_hf_reference.py` refactor), being run on separate hardware and deliberately not on this Galaxy host.
