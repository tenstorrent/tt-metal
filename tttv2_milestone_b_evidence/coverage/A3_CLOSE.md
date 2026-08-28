
## What attempt 3 committed

Tests and evidence. **No implementation file, in any package**, either invocation.

```text
models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py   +3 cases
models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py     +5 cases
tttv2_milestone_b_evidence/coverage/                                             logs2/, logs3/, this section
```

The five test-level changes, and why each is a measurement rather than an
accommodation:

1. **`test_qwen_prefix_cached_and_plain_requests_mixed_across_slots`** and
   **`test_qwen_per_slot_heterogeneous_sampling_controls`** — two cases the brief
   asks for by name that existed only for Llama. Written by attempt 3's first
   invocation.
2. **`test_{qwen,llama}_paged_pool_logits_are_recorded_for_cross_process_comparison`**
   and **`test_{qwen,llama}_two_paged_pools_agree_across_processes`** — area 1's
   headline claim, split across processes because D-C7 says a process gets one
   model. Same PCC threshold, same claim, one fewer model per process. The
   comparison **fails** rather than skips when a recording is absent, and refuses
   to run if both recordings report the same `max_num_blocks` — the exact
   tautology D-C4 created the first time this case was written. Both guards were
   exercised: `logs3/a3_h10_pool_compare_missing_guard.log`.
3. **`test_{qwen,llama}_device_sampling_claims_behind_dc5_with_interleaved_logits`**
   — the diagnostic that found D-C8. It removes D-C5 *at the call site, in the
   test*, and does not touch the product: area 4 stays reported as BLOCKED
   whatever it says. It is the reason this job can distinguish "device sampling is
   blocked by one memory-layout precondition" from "device sampling is blocked by
   a memory-layout precondition **and** a sub-device core-set violation behind
   it", which is a different conversation with whoever fixes it.

Nothing here relaxes a threshold, a tolerance or a parametrization, and no test
was deleted or `xfail`ed. Two tests were **added that fail**, on purpose, because
the thing they measure is broken.

## What Milestone C inherits from this job

Ranked by what a human has to decide, not by severity.

1. **Device sampling does not work end to end on this hardware, and it is two
   defects deep.** D-C5 (the selector matmul requires an interleaved input B;
   both models' decode logits are width-sharded, from the *shared* recipe) and
   D-C8 (with that satisfied, the matmul builds a program over cores outside the
   loaded decode sub-device). Both are in `models/common/models/galaxy/`, both are
   shared code, and the fix needs a program config the selector currently has no
   way to accept. Every claim in the brief's area 4 is behind them.
2. **L1 has two signatures and only one of them is an ordering problem.** The
   address clash is Llama-only at this tree and might yield to teardown ordering.
   The capacity residue (**D-C7**) will not: the owner was closed, dereferenced
   and garbage-collected and its L1 did not come back, so the second model in a
   process cannot create its global circular buffer. That puts a hard *one model
   per process* bound on the stack — the same bound **D-C3** puts on it from the
   other direction, via a weight cache fingerprinted with `MeshDevice.id()`.
3. **D-C1** — decode's page-table validator cannot separate a prefill-shaped
   table from a legitimate L1-sharded repeat. Premise confirmed on silicon by
   three fresh processes. The fix changes a 2D-module expectation, which three
   attempts have now declined as a boundary violation. It needs a decision.
4. **D-C4** — `paged_attention_config=None` installs the default pool, not a
   contiguous cache. Either the adaptor grows a way to ask for a contiguous cache
   or the plan's area-1 wording changes to the two-pool form this job measured.
5. **D-C2** — is a sampling seed per-request or per-(request, slot)? A product
   decision about the serving contract, not a bug.
6. **F-C3** — one pre-existing `models.demos` import sits in
   `models/common/tests/modules/moe/`. Not Milestone B's, and `mb-signoff` should
   name it rather than assert a bare zero.
7. **G-C1, G-C2, G-C3, F-C2** — unchanged from attempt 1.
8. **Scheduling reality, for whoever plans Milestone C's device nights.** One node
   id per process is mandatory (D-C3), a warm Llama build is ~5.5 min and a cold
   weight set is 26 min and 138 GB, and a failing run costs its own wall clock
   plus a `tt-smi -glx_reset`. A 17-node-id file is a three-hour run. Plan around
   builds, not around tests.
