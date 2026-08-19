# Milestone A Distributed RMSNorm2D Audit

Timestamp: 2026-08-19 07:31:16 UTC

## Scope and constraints

- Audited only the Milestone A distributed final/residual `RMSNorm2D` implementation, its WH Galaxy hardware tests, shared Galaxy test plumbing, and prior worklogs.
- Did not run TT hardware.
- Did not edit production or test implementation files. This audit file is the only repository write.

## Executive finding

The current WH hardware file is not safe to launch as written. The prior failures do not yet prove a defect in the RMS arithmetic. They are explained by test/resource lifecycle mismatches that occur before a trustworthy numerical readback:

1. Decode registers the row-major RMS gamma tensor as the sole `Prefetcher2D` payload. The prior integrated attempt segfaulted inside `ttnn.dram_prefetcher` on exactly this payload, before the norm kernel. The qualified Galaxy prefetch path registers DRAM-sharded projection weights, not norm gamma.
2. `_invoke` calls whole-mesh `ttnn.synchronize_device(mesh_device)` while decode has a persistent prefetch kernel running on the sender subdevice. The qualified MLP hardware path synchronizes only `resources.ccl.context(mode).worker_sub_device_id`; its worklog established that whole-device synchronization waits on the intentionally persistent sender.
3. Fused decode receives a semaphore allocated on the mode-wide worker core set. The proven `fused_rms_minimal` unit paths allocate their semaphore on the norm input shard grid. Earlier RMS work briefly made that correction, then restored full-worker semaphore allocation to satisfy the separate standalone async all-gather protocol. The current resource model/test plan therefore conflates fused and two-phase semaphore placement.
4. The decode test uses `fabric_config=True` and `Topology.Linear`. On this 6U Galaxy, the qualified integrated prefetch path uses explicit `FABRIC_1D_RING`; the legacy 6U RMS path also resolves Ring. Any fused retry sharing the qualified projection prefetch queue must use a fabric/topology pair consistent with that queue.

## Implementation audit

### Fused residual path

`RMSNorm2D._decode_fused_residual_norm` has the intended high-level contract:

- dynamic residual input;
- static `FUSED_DECODE` policy;
- direct `ttnn.fused_rms_minimal` invocation;
- exact sequence-keyed persistent stats buffer;
- scalar semaphore adaptation;
- returned `(normalized_output, residual_tensor)` pair, relying on the fused primitive's in-place residual accumulation.

The host test verifies argument plumbing and returned identity, but cannot verify that the residual tensor was updated to `x + residual`. That remains a real-device requirement.

The highest-risk unresolved detail is semaphore placement. `GalaxyModePlan.semaphore_cores` is mode-wide, while fused RMS and persistent `all_gather_async` have different demonstrated requirements. A fused resource should not silently inherit the full-worker semaphore allocation used by the standalone CCL path.

### Two-phase decode/prefill path

The decomposed path now has the expected operation order and ownership:

- optional residual add;
- local RMS statistics;
- axis-1 persistent async all-gather;
- post-all-gather RMS;
- borrowed gathered stats are not deallocated by the module.

The prefill statistics shape correctly preserves `(N, C, S, 32)`. The prior readback segfault followed by a blocking explicit synchronization is consistent with an incomplete async collective, not a host composition-only failure. Subsequent fixes aligned the 2D overload, explicit mesh, adjacent semaphore window, no barrier, and canonical worker partition, but this path has not been re-qualified numerically.

## Hardware-test audit

The current file has six cases and collects cleanly, but the distributed decode cases are unsafe:

- `prefetch_weights=(("norm.weight", device_weight),)` recreates the exact invalid gamma prefetch payload recorded in the worklog.
- Whole-device synchronization at `_invoke` line 130 can block on the persistent sender even if RMS worker kernels complete.
- Both Llama and Qwen decode cases repeat twice, increasing recovery cost before the first invocation is known good.
- The same all-gather resource abstraction is used for fused decode and two-phase prefill despite their different semaphore cardinality/placement contracts.

The distributed prefill case is safer than decode because `Prefetcher2D.activate("prefill")` does not start the persistent DRAM prefetch kernel. However, the existing test immediately runs 128 and 2048 twice for both models, which is larger than necessary for the first diagnostic.

## Smallest safe next serialized hardware experiment

Run one Llama distributed prefill invocation at sequence length 128 with a residual tensor, under a hard external timeout. This is the smallest useful experiment because it validates residual addition, local stats, the corrected axis-1 async all-gather protocol, post-gather RMS, worker completion, composition, PCC, and teardown without starting the persistent decode prefetch kernel.

Before running it, make test-only changes so the case can execute exactly one sequence and one invocation, and synchronize only the mode's worker subdevice. Keep the existing Linear/neighbor-exchange pair for this isolated prefill diagnostic; do not mix in the ring projection-prefetch queue yet.

Suggested bounded command after those test-only changes:

```bash
timeout --signal=TERM --kill-after=30s 300s \
  pytest -q models/common/tests/modules/rmsnorm/test_rmsnorm_2d_wh_galaxy.py \
  -k 'final_norm_prefill and llama and smoke_128'
```

Success criteria:

- worker-subdevice synchronization returns;
- output PCC is at least 0.99 against RMSNorm of `x + residual`;
- pytest fixture closes all 32 devices without reset;
- no whole-device synchronization occurs while a persistent sender is active.

If it times out or faults, terminate the bounded process and use `tt-smi -glx_reset` before any later hardware case.

## Required follow-up before fused decode

Do not run the current fused decode test unchanged. Prepare a test-only integrated Llama case based on the already-qualified MLP decode resource setup:

- register the known-good DRAM-sharded `w1`, `w3`, `w2` projection queue, never gamma;
- add the RMS stats resource to that plan;
- use explicit `FABRIC_1D_RING` and matching Ring RMS topology for the 6U queue;
- synchronize only the worker subdevice before readback;
- begin with one Llama invocation and a hard timeout;
- allocate/select the fused semaphore on the norm input shard grid, independently from any standalone async all-gather semaphore set.

That last requirement may need a production resource-schema change, such as a per-collective semaphore-core override or a distinct fused-RMS resource key. It should be covered by host allocation tests before hardware. Reusing the current mode-wide full-worker semaphore allocation is not a controlled experiment.

## Host-only changes recommended

1. Change RMS hardware `_invoke` to worker-scoped synchronization, matching the qualified MLP test.
2. Add a host assertion/test that decode readback requests only `worker_sub_device_id` when a prefetch session is active.
3. Remove norm gamma from decode prefetch registration; construct fused decode with a known-good projection queue.
4. Add a one-invocation/one-sequence prefill smoke case or parameter boundary for serialized diagnosis.
5. Add host coverage for fused semaphore core placement distinct from standalone async all-gather placement.
6. Add a host contract that the fused returned residual is the primitive-mutated tensor; numerical `x + residual` semantics remain hardware-only.

## Verification performed

- Host-only focused suite: `53 passed in 7.90s` across RMSNorm2D, Galaxy CCL/resources, and Prefetcher2D.
- WH hardware collection only: `6 tests collected`; no device fixture execution.
- `git diff --check`: passed.
- TT hardware commands executed: none.

## Conclusion

Distributed final/residual RMSNorm remains unqualified. The next serialized run should be the single Llama prefill-128 residual smoke, not fused decode. Fused decode should follow only after invalid gamma prefetch, whole-device readback synchronization, fabric/topology pairing, and fused semaphore placement are corrected or isolated by host-tested resource configuration.
