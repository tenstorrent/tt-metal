
## Findings, attempt 2

Attempt 1's seven (D-C1, D-C2, G-C1, G-C2, G-C3, F-C1, F-C2) plus what a live
mesh added. Only the changes are written out here; the unchanged ones keep
attempt 1's text above.

### F-C1 — **superseded, and it was the wrong way round**

See §A2's opening. Llama pads by 768 ids, Qwen by 1664. Attempt 1 recorded
Llama's padded-vocabulary gate as *vacuous*; it is live, and now has a device
case (`test_llama_no_padded_vocabulary_id_is_ever_sampled`, three policies).

### D-C1 — premise confirmed on silicon, verdict unchanged

Attempt 1 derived D-C1 from a host model of one `ttnn` fact and asked for one
line on a live mesh to settle it. That line is now a committed test,
`models/common/tests/models/galaxy/test_step7_page_table_placement_wh_galaxy.py`,
and it says attempt 1 read the fact correctly:

* a column-sharded decode table (`ShardTensor2dMesh(dims=(None, 0))`, mesh
  `(8, 4)`) has device-local shape **(8, 64)** — the shard shape, one mesh
  column's users;
* the replicated prefill table has device-local shape **(32, 64)**, and
  `32 % 8 == 0`.

So `_validate_decode_page_table`, which discriminates on the device-local row
count alone and accepts any positive multiple of `users_per_column`, cannot tell
the prefill layout from a legitimate four-core L1 repeat. **D-C1 stands exactly
as attempt 1 wrote it, and the worse variant it feared is ruled out.**

The test also records what attempt 1 could not check: **both tables are
DRAM-interleaved**, so `memory_config().is_sharded()` is false for both. A fix
therefore cannot be "reject unless sharded" applied to the 32-row case alone
without also deciding what a 32-row *interleaved* table means; the honest
discriminator is that a repeat is only legitimate when the tensor is L1
height-sharded over exactly `rows / users_per_column` cores, which makes the
existing 2D-module expectation
`test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats[16]`
and `[32]` — which pass a plain interleaved table — the thing that has to change.
That is the boundary attempt 1 declined to cross, and attempt 2 declines it for
the same reason: the brief says report it, do not edit the expectation.

### D-C2 — unchanged, and still a product decision

`_device_seed`/`_host_seed` are `blake2b("sampling2d:{seed}:{slot}")`, so a
request that migrates slots does not keep its stream. The step-7 gate asks for
the opposite. Attempt 2 measured only the half that holds — same seed, same slot,
same token across fresh runs — and did not assert the half that does not.

### D-C3 — the device weight cache is keyed by `MeshDevice.id()`, so every test after the first in a process re-stages every weight

**New, severity: test-infrastructure, and it costs hours and hundreds of GB.**

`LazyWeight._get_fingerprint` ends with

```python
device_id = self.device.id() if hasattr(self.device, "id") else "single"
parts.append(f"device_{device_id}")
```

`self.device` is the **`MeshDevice`**, and the `mesh_device` fixture builds a new
one per test, so its `.id()` is 0 for the first test in a pytest process, 1 for
the second, 2 for the third. The cache path therefore changes per test, and every
test after the first misses on **every** weight.

Measured, on this mesh, at this commit:

| | |
| --- | --- |
| whole-file run, `test_full_model_wh_galaxy.py` (8 node ids) | test 1: 240 cache hits, model built in ~6 min. Test 2: **965 misses**, 26 min of staging, **138 GB** written. Test 3: staging device_2's set again |
| the same test alone in its own process | **240 hits, 0 misses**, whole test 237 s |

A complete cache set is 138.5 GB for Llama-3.3-70B, so an 8-node-id file needs
**1.1 TB** of cache to run — on a filesystem that started this night with 1.0 TB
free and 95% used. This attempt paid 55 minutes and 277 GB of it before reading
the fingerprint, and then pruned the two duplicate sets.

**Consequence for anyone scheduling this hardware: one node id per pytest
process, always.** Every earlier job's harness happens to do that — `mb-qwen`'s
manifest format is one node id per line — but nothing in the tree says why, and
the cost of not knowing is a whole night.

The fix is a one-line change in shared 1D/2D code (`models/common/modules/lazy_weight.py`),
which is outside this job's mandate: a mesh of the same shape and mapper produces
the same tensor, so the fingerprint wants the mesh **shape**, not the instance id.
Reported, not changed.

### D-C4 — `from_pretrained` cannot build a contiguous KV cache, so area 1's headline gate is not expressible through the adaptor

**New, severity: contract gap. It also made one committed test a tautology.**

Both adaptors do

```python
paged = paged_attention_config or default_paged_attention_config(params)
```

so `paged_attention_config=None` does not mean "contiguous" - it means "give me
the default pool", `ceil(max_seq_len / 32) * max_batch_size` blocks. There is no
argument that yields `spec.paged_attention_config is None`, even though
`Attention2D`, `GalaxyPagedKVContract` and the model's own `kv_specs` all support
that state and the host suite exercises it.

Two consequences, both measured:

1. `test_*_paged_capacity_resolved_after_construction_serves_a_request` **failed**
   on `assert all(spec.paged_attention_config is None ...)` (`a2_02`). That is a
   true report of the gap, not a broken model.
2. `test_*_paged_and_contiguous_caches_agree` compared the default pool against
   an explicitly-constructed pool of **exactly the same geometry** - at
   `max_seq_len=2048`, batch 32, block 32, both are 2048 blocks. It would have
   passed at PCC 1.0 while proving nothing about paged addressing.

Attempt 2 rewrote both rather than leaving a green tautology:

* `test_*_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable` runs the
  same 32 requests through a 2048-block and a 4096-block pool - which gives every
  slot a different run of block ids - and compares prefill and decode logits per
  slot at PCC ≥ 0.99. It asserts `resolved is not None` with a message telling a
  future reader to restore the original comparison once D-C4 is fixed;
* the late-capacity case now asserts the *reachable* claim: the geometry
  installed at construction can still be replaced before anything is bound, is
  refused while bound, and can be replaced again after unbind.

**The gate line "paged fill during prefill, then decode reading the same blocks,
PCC ≥ 0.99 against the contiguous path" therefore cannot be met at this API**, and
that is the honest verdict rather than a green tick from a tautology.

**Where the contiguous path does exist**, for whoever fixes D-C4:
`models/common/tests/models/llama33_70b_galaxy/test_bringup_wh_galaxy.py` builds
one with `_contiguous_kv_cache(...)` and `model.set_kv_cache(...)` directly, and
`GalaxyDirectRunner` has a contiguous branch (`self.paged = False`, which then
requires `active_slots == max_batch_size`). So the missing piece is only an
adaptor argument — something like `paged=False` alongside
`paged_attention_config` — not a new mechanism.

### L1's remaining half is **Llama-specific at this tree**, not universal

**New, and it contradicts an inherited claim.** `mb-qwen` attempt 2's handoff
says of L1's remaining half — prefill after a decode — *"Untouched, inherited,
**identical for both models**."* Measured here, it is not:

| Test shape (two prefill phases with a decode between them) | Llama | Qwen |
| --- | --- | --- |
| `*_repeated_requests_and_deterministic_cleanup` | **FAIL**, `program 100` clashes on `[0-0 - 0-3]` (`a2_g6`) | **PASS**, no clash (`a2_g17`) |
| `*_batch32_slots_are_isolated` | **FAIL**, same signature (`a2_g7`) | **PASS**, no clash (`a2_g18`) |
| `demo.py::*_concat32_prefill_matches_sequential` | **FAIL**, `program 1552` clashes on `[0-0 - 6-9]` — the whole grid (`a2_g10`) | @@QWEN_CONCAT@@ |
| `demo.py::*_device_sampling_matches_host_greedy` | **FAIL**, `program 100` (`a2_g11`) | @@QWEN_SAMPLING@@ |

Both Qwen results were taken in fresh single-node-id processes and re-run to
three (`a2_L1_qwen_*_run2/3`); the Llama failures are four independent
reproductions in four different tests.

**Why this matters more than a green tick.** The clash is an address collision —
`L1 buffer allocated at 544832 and static circular buffer region ends at …` — and
Qwen's decode placements are narrower than Llama's (residual on 10 cores against
16, `local_dim` 1280 against 2048, and a 40-core LM-head reduction against 42).
So the failure is a function of *how much L1 the decode mode leaves below the
prefill program's static CB region*, not of the mechanism being absent. That
gives Milestone C something it did not have: **a working reference configuration
on the same silicon**, which turns "why does prefill-after-decode clash" from a
one-sided debugging problem into a differential one.

It also means the limitation cannot be stated as a property of the 2D modules. It
is a property of a *resolved geometry*, and the next model added to this stack may
land on either side of it with nothing in the contract to warn it.
