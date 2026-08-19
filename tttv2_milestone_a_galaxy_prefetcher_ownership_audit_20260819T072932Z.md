# Milestone A Galaxy Ownership and Prefetcher2D Integration Audit

## Scope

- Dedicated goal: audit Milestone A Galaxy resource ownership and `Prefetcher2D` integration.
- Constraints: no TT hardware execution and no production-file edits. Host tests are permitted.
- Audit output: this file is the only planned modification.

## Checkpoint 1: Static ownership boundary

- `Prefetcher2D` is the sole owner of subdevice managers, the global circular buffer, weight-address metadata, active prefetch results, and stop sentinels. Registered model weights are explicitly borrowed.
- `GalaxyResources` owns only factory-created CCL semaphores and persistent/intermediate tensors. It borrows a sealed `Prefetcher2DResourceOwner`, validates exact mode/subdevice compatibility before CCL allocation, and does not clean up that borrowed owner.
- The required root cleanup order is therefore `GalaxyResources.cleanup()` followed by `Prefetcher2D.cleanup()`. The hardware-test adapter performs this order by deduplicating `owner`, `ccl`, and `prefetcher` identities.
- Existing unit coverage tests both components independently, but `test_resources.py` uses `FakePrefetcher`; there is no host composition test using a real sealed `Prefetcher2D` with real `GalaxyResources`.
- `GalaxyResources` has private `_synchronize(mode)` only. The MLP hardware test consequently reads `resources.ccl.context(mode).worker_sub_device_id` and calls `ttnn.synchronize_device` itself before host readback. This is concrete evidence that the public lifecycle API is incomplete for executor/test callers.

## Checkpoint 2: Host baseline and adequacy assessment

- Host command: `pytest -q models/common/tests/models/galaxy/test_ccl.py models/common/tests/models/galaxy/test_resources.py models/common/tests/modules/prefetcher/test_prefetcher_2d.py`.
- Result: `38 passed in 5.97s`. No TT hardware command or fixture was run.
- The ownership split is adequate and should be preserved: the prefetcher owns manager/prefetch resources, Galaxy owns CCL resources, model weights remain borrowed, and neither lower-level owner should destroy the other.
- The public synchronization API is not adequate. Mode transitions and Galaxy cleanup synchronize internally, but an executor cannot explicitly wait for the current mode before readback, output handoff, trace boundary work, or deterministic teardown without calling TTNN directly and reconstructing subdevice policy from a borrowed CCL context.
- `GalaxyResources.activate()` also intentionally does not synchronize repeated same-mode activation. That is acceptable only if the executor has an explicit operation-boundary wait; without a public wait method, repeated decode/prefill correctness depends on ad hoc caller behavior.
- `Prefetcher2DResourceOwner` omits `cleanup()` appropriately because `GalaxyResources` borrows rather than owns it. Cleanup composition belongs at the model/executor root and should remain explicit and ordered.

## Checkpoint 3: Exact missing integration tests

Add a host-only composition file, preferably `models/common/tests/models/galaxy/test_prefetcher_integration.py`, which uses the concrete `Prefetcher2D` and concrete `GalaxyResources` with injected fake TTNN bindings. The minimum missing cases are:

1. `test_concrete_owners_repeat_transition_and_cleanup_in_root_order`
   - Initialize and seal a real prefetcher, construct real Galaxy resources, then execute `decode -> decode -> prefill -> prefill -> decode`.
   - Assert exact prefetch start/stop ordering, published active modes, worker-scoped synchronization, independent CCL cycle state, and final root cleanup order.
   - After Galaxy cleanup, assert prefetch metadata/session/managers are still owned. After prefetch cleanup, assert those resources are released once and registered weights are never deallocated.
2. `test_concrete_transition_failure_restores_both_owners`
   - Inject a prefetch start, stop, or stall-group failure during a mode transition.
   - Assert `GalaxyResources.active_mode`, the concrete prefetcher's mode/session, and the Galaxy CCL context remain on the previous mode; then assert both owners remain cleanly releasable.
3. `test_ccl_allocation_failure_leaves_concrete_prefetcher_for_root_cleanup`
   - Fail a CCL persistent-buffer allocation after the real prefetcher is sealed.
   - Assert Galaxy rolls back only CCL allocations, the prefetcher still owns its resources and borrows its weights, and root cleanup subsequently releases prefetch resources without touching weights.
4. `test_public_synchronize_is_worker_scoped_and_terminal_after_cleanup`
   - Activate each mode, call the public wait API, and assert the injected TTNN synchronization receives only that mode's worker subdevice ID.
   - Assert synchronization is rejected after cleanup and does not include the persistent decode sender subdevice.
5. `test_root_cleanup_continues_to_prefetcher_after_galaxy_release_error`
   - At the root/adapter layer, inject one CCL release failure and use `try/finally` cleanup.
   - Assert prefetch stop, metadata release, and manager removal still occur exactly once while the first Galaxy error is retained.

Current WH MLP tests are structurally valuable integration coverage: they build both concrete owners, pass sealed contexts into `MLP2D`, repeat decode or prefill, and clean resources in the correct order. The recorded full matrix passed four cases. However, each test owns only one execution mode sequence, so it does not cover decode-to-prefill, prefill-to-decode, transition failure, or cleanup failure composition in one owner. This audit did not rerun those hardware tests.

## Checkpoint 4: Bounded implementation recommendation

1. Add `GalaxyResources.synchronize(mode: GalaxyMode) -> None` as a public, open-state-checked method. Synchronize exactly `context(mode).worker_sub_device_id`, not the whole mesh and not all decode subdevices, because the decode sender runs a persistent prefetch kernel.
2. Route `GalaxyResources.activate()` transition waits and `cleanup()` waits through that method (or one shared worker-scoped implementation). Keep repeated same-mode waiting explicit at the executor operation boundary rather than adding an unconditional hidden wait to every activation.
3. Add `GalaxyHardwareResources.synchronize(mode)` and make it require/delegate to the production owner. Replace the MLP test's direct `ttnn.synchronize_device` call and reach-through to `ccl.context(...).worker_sub_device_id` with this adapter method.
4. Add the five host integration cases above. Reuse existing fakes or move the common fake mesh/tensor/bindings into a small test helper; do not add production abstractions solely for tests.
5. Preserve the current ownership boundary. Do not make `GalaxyResources.cleanup()` destroy the borrowed prefetcher, and do not add `cleanup()` to the borrowing protocol. The model/executor root must continue to clean Galaxy first and prefetcher second under `try/finally`.

This is a narrow API and test change: one public synchronization method, one adapter forwarding method, one hardware-call-site cleanup, and a host composition suite. It does not require module hot-path changes, CCL resource redesign, or a combined owner that obscures ownership.

## Final status

- Ownership API: adequate and explicit, with correct borrowed-versus-owned separation.
- Synchronization API: inadequate for public executor/readback use until worker-scoped synchronization is exposed.
- Integration coverage: individual owners are well covered and the MLP hardware path provides successful concrete composition evidence, but host composition, cross-mode transitions, transition failure, and root cleanup failure remain untested.
- Verification: focused host suites passed `38 passed in 5.97s`.
- Constraints honored: no TT hardware was run and no production file was edited.
