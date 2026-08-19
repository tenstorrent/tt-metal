# MLP2D Async CCL and Readback Audit

Timestamp: 2026-08-19 06:35:05 UTC

## Scope

Read-only audit of the clean WH Galaxy MLP2D decode timeout against the legacy
`TtLlamaMLP` and `TT_CCL` decode path. No TT hardware was used. The audit covers
async collective semaphore order, subdevice synchronization, and tensor
deallocation/lifetimes.

## Conclusion

The timeout was not caused by a different MLP collective launch order, a
semaphore-cycle mismatch, or an early tensor deallocation. It occurred at host
readback: `ttnn.to_torch`/mesh composition could wait on the whole mesh while the
decode prefetch sender subdevice still had its persistent program active. A
worker-subdevice synchronization before composition makes the same clean Llama
decode pass in 22.93 seconds, as recorded in
`tttv2_2d_modules_work_log.md` lines 721-725.

The prior diagnostic synchronization most likely to have masked the timeout was
the worker-subdevice synchronization immediately after the final
`all_reduce_async` returned. The work log records that boundary at lines 439-448.
It guarantees the returned MLP output is complete before diagnostic readback, so
the subsequent host conversion does not need to discover completion by waiting
on the sender subdevice. Earlier stage synchronizations after fused
reduce-scatter, W3 reduce-scatter, multiply, or all-gather were broader than
necessary and are not required to explain the observed fix.

## Async launch order

The current decode path in `models/common/modules/mlp/mlp_2d.py` launches:

1. `llama_rs_matmul` for W1/W3 plus W1 axis-1 reduce-scatter.
2. `llama_reduce_scatter` for W3 on axis 1.
3. gated `ttnn.mul`.
4. persistent `all_gather_async` on axis 1.
5. W2 `ttnn.linear`.
6. `all_reduce_async` on axis 0.

This matches the legacy order in
`models/demos/llama3_70b_galaxy/tt/llama_mlp.py` lines 175-293 and its CCL
adapters in `llama_ccl.py` lines 907-956, 1114-1151, 1260-1284, and 773-865.

The common CCL owner cycles semaphores per fully qualified resource key. With
the test plans' two slots:

- fused W1 reduce-scatter consumes reduce-scatter slot 0;
- W3 reduce-scatter consumes reduce-scatter slot 1;
- all-gather receives the two-handle window beginning at its slot 0;
- final all-reduce consumes its independent axis-0 slot 0;
- subsequent invocations rotate those same keyed pools deterministically.

Legacy uses shared per-axis `gather_idx` state rather than per-operation keys,
but increments it after each corresponding launch. Both implementations enqueue
the dependent operations on the worker subdevice in program order. There is no
evidence that the clean timeout is a semaphore reuse hazard: the identical
pipeline completes once only the final worker/readback boundary is added.

## Subdevice behavior

Decode uses two subdevices: sender/prefetch subdevice 0 and worker subdevice 1.
Activation starts `dram_prefetcher` while both are stalled, then changes the
stall group to worker subdevice 1. MLP matmuls and all CCL operations explicitly
target worker subdevice 1.

The clean test originally called mesh composition directly after `module()`.
That host conversion can impose a broader device wait than the CCL dependency
chain requires. Since the sender program is intentionally active, a whole-mesh
wait can remain blocked even though every worker operation and the output tensor
are complete. Synchronizing only `worker_sub_device_id` before composition is the
correct qualification boundary and leaves the sender program undisturbed.

Legacy establishes the same sender/worker partition and worker-only stall group
in `prefetcher_common.py` lines 88-98 and the legacy MLP test lines 95-130. Its
MLP implementation contains no active per-stage synchronization; the commented
syncs in `llama_ccl.py` lines 955 and 1007 confirm that the production pipeline
is intended to remain asynchronous.

## Tensor lifetimes

Current and legacy code deallocate the raw W1 projection immediately after
`llama_rs_matmul`, the raw W3 projection immediately after launching its
reduce-scatter, the gated input immediately after launching all-gather, and W2's
partial output immediately after launching all-reduce. Those deallocations occur
after the consuming operation has been enqueued and match the legacy ownership
pattern. The diagnostic run and the clean worker-synchronized run both complete,
which rules these sites out as the timeout trigger.

Two lifetime differences are worth tracking separately from this timeout:

- Current decode does not deallocate `w1_out` and `w3_out` after `ttnn.mul`, while
  legacy does. This is an L1 retention/leak risk across repeated invocations.
- Current decode does not deallocate gathered `w2_in` after W2 linear, while the
  normal legacy WH path also leaves that tensor allocated. This is not evidence
  for the first-invocation readback timeout, but ownership should be made
  explicit before long trace/repeat qualification.

Neither difference can explain a stall at clean first-result readback that is
removed solely by worker-scoped synchronization.

## Recommended follow-up

Keep synchronization out of the production MLP hot path. The hardware test and
other host-readback call sites should synchronize the CCL worker subdevice before
`ttnn.to_torch` while decode prefetch is active. Add a focused host contract for
that qualification helper if the synchronization remains test-owned. Audit and
then explicitly release the reduced W1/W3 intermediates after `ttnn.mul` to avoid
repeat-run L1 growth; validate that ownership change separately on hardware.
