# Proposal: scope two DFB endpoint checks to Gen2

**Context:** Metal 2.0 port of `MatmulMultiCoreReuseMcast2DProgramFactory` (matmul port series,
parent issue #41908).
**Status:** blocking that port. The configuration below runs correctly on the legacy path today and
raises `TT_FATAL` at program build once the factory is on `ProgramSpec`.
**Arch:** reproduced on Blackhole.

## Ask

Two endpoint checks in `ValidateProgramSpec` reject a DFB topology that is legal on Gen1. Please
enforce them on Gen2 only, matching the idiom already used by the DM-self-loop rule at
`program_spec.cpp:1495`.

| | site | what it requires | proposed |
|---|---|---|---|
| **Rule 1** | `program_spec.cpp:1366-1377` | all `KernelSpec`s on one DFB role have the same kind (compute vs DM) | Gen2 only |
| **Rule 2** | `program_spec.cpp:1507-1522` | for a self-looped DFB, `producer_kernels == consumer_kernels` | Gen2 only |

**Gen2 behaviour is unchanged.** On Gen1 nothing becomes legal that the per-node census does not
already police — see §4, which is the core of the argument and does not depend on any claim about
masks.

---

## 1. Problem

The factory supports a configuration where `in0` is BLOCK_SHARDED and its shard grid is wider along
the multicast axis than the output needs in columns
(`in0_sender_num_cores_along_width > num_blocks_x`). K is split across the mcast axis, so no core
holds all of K and the `in0` broadcast is a rotating relay: on K-iteration `block`, whichever core
owns that slice multicasts it to the receiver row. When in0's K is spread over more cores than the
output's N needs columns, the surplus cores own K-slices but have no output block — they are pure
suppliers of the reduction dimension.

For those cores the `in0` DFB is self-looped by the sender; on the work cores the same DFB is
sender → compute:

| nodes | PRODUCER | CONSUMER |
|---|---|---|
| work grid | `in0_sender` (DM) | `compute` (compute) |
| no-work senders | `in0_mcast_no_work` (DM) | `in0_mcast_no_work` (**same kernel**) |

Spec-wide, the DFB's roles therefore hold:

- PRODUCER = { `in0_sender`, `in0_mcast_no_work` } — both DM
- CONSUMER = { `compute`, `in0_mcast_no_work` } — **one compute, one DM** → rule 1 rejects
- self-looped, and the two role sets differ → rule 2 rejects

**Every node has exactly one producer instance and one consumer instance.** The kinds differ *across*
nodes, never on a node.

**It has to be one DFB.** The sender multicasts to `.addr = dfb_in0.get_write_ptr()` — it derives one
destination address from its own cursor and broadcasts it to every receiver, so all instances must sit
at the same L1 offset. The self-pop on the no-work cores keeps that cursor in phase: `in0` is
double-buffered (`in0_num_entries *= MCAST_INPUT_BUFFERING_DEPTH`), on a work core compute's
`pop_front` advances it, and a core with no compute must do it itself or broadcast a stale slot
address. The kernel says so:

```
// If core does not produce output block work, free dfb::in0 immediately.
// This is necessary since mcast is in lockstep; this ensures write ptr addresses are synced
// properly for cores that only send and have no compute / writer active.
```

A private scratch DFB on those cores would decouple the address and break the multicast, and the
sender is the only kernel present there to drive the cursor. The topology is not a modelling choice.

## 2. It is legal on Gen1 — measured

The per-node census passes: every node has exactly one producer instance and one consumer instance,
which is the invariant `dataflow_buffer_spec.hpp:41-45` states. Only the two secondary rules fail.

Same reproducer, same device, same inputs:

| build | result |
|---|---|
| factory on `ProgramSpec` | `TT_FATAL` at `program_spec.cpp:1377` |
| factory on the legacy descriptor path | **succeeds, pcc 0.999881** |

The kernel placement is not new either: legacy issues the same two `CreateKernel` calls from the same
source on the same two ranges (`all_cores_with_work` and
`in0_mcast_cores_without_work_and_not_in_receiver_grid`), differing only in the
`(core_has_output_block_work, core_in_in0_receiver_mcast_grid)` compile-time pair. What Metal 2.0 adds
is the endpoint model — legacy's `CreateCircularBuffer(program, all_cores, cfg)` declared no roles, so
there was no uniformity rule to violate. This is a pre-existing, correct placement meeting a
validation vocabulary that did not previously exist.

## 3. Exactly one documented condition fails

Endpoint multi-binding (several `KernelSpec`s on one role, disjoint nodes) is an intended capability —
`advanced_options.hpp:182-186`: *"a DFBSpec (spanning multiple nodes) can have more than one
KernelSpec producer or more consumer bindings, as long as every node's DFB instance has one producer
and one consumer. (This enables, for example, a grid-spanning compute kernel to be fed data by
different producer kernels on different nodes.)"*

Its three conditions (`dataflow_buffer_spec.hpp:44-50`) are non-overlapping node coverage, same kernel
kind, and identical binding-site parameters. **This topology satisfies two and fails one — kind.**
Provable rather than asserted: `check_role_uniformity` tests `access_pattern`, then `num_threads`,
then kind in that order, and the observed failure is the third.

## 4. Why these checks are redundant on Gen1

This is the main argument, and it is independent of what the hardware does with masks.

**Without `allow_instance_multi_binding`, the per-node census already enforces "exactly one producer
instance and exactly one consumer instance" on every node** (`program_spec.cpp:1437-1456`:
`num_producers == 1 && num_consumers == 1`). If a node hosts exactly one producer and exactly one
consumer, then:

- **per-node kind uniformity is automatic** — a single kernel is trivially uniform with itself, so
  rule 1 can only ever reject *cross-node* kind differences;
- **per-node self-loop set equality is automatic** — if that node's one producer is also its one
  consumer, the node's producer and consumer sets are equal by construction, so rule 2 can only ever
  reject *cross-node* set differences.

**With the flag, both checks are skipped anyway** (rule 1 by the `if (!allow_multi)` guard at `:1380`;
the census relaxes to "at least one").

So on Gen1 the two rules add exactly one thing over the census: they forbid role heterogeneity
*across* nodes. And cross-node heterogeneity is precisely what the per-node invariant is documented to
permit, because each node gets its own DFB instance. The genuine hazards — two producers on one node,
or a self-looper sharing a node with an unrelated binder — are caught by the census either way.

**Supporting argument (masks).** Both rules state their rationale as guaranteeing a single per-role
processor mask (rule 1 at `:1334-1338`; rule 2 at `:1502-1506`). That mask is inert on Gen1 by the
runtime's own account. `MakeDataflowBufferConfig` takes a single representative mask from the first
record (`:2635-2641`), justified at `:2985-2987`:

> Instance-multi-binding (Gen1-only) intentionally binds same-role kernels on distinct RISCs … so
> their risc_masks differ by design and the uniform-mask requirement does not apply. **On Gen1 the DFB
> lowers to a plain circular buffer where the mask is inert (it never reaches the device blob)**, so
> the single representative mask `MakeDataflowBufferConfig` takes from the first binding is harmless.

Every consumer of the mask under `tt_metal/impl/dataflow_buffer/` is behind
`hal().has_tile_counter_registers()`, which is Quasar (asserted at `dataflow_buffer.cpp:284`): reads at
`:1214` (guarded `:1211`), `:1287-1290` (tile-counter path), `:1444-1446` (guarded `:1442`), and
`:1933-1939` (guarded `:1882`). So Step 2b already reasons *Gen1 + mask inert ⇒ drop the uniformity
requirement*; this proposal applies the same conclusion to the two checks that encode the same
requirement earlier in validation.

## 5. Proposed change

Both rules already have an arch-gated sibling adjacent to them — the DM self-loop restriction at
`:1495` is `!(is_gen2_arch(hal) && self_loop_kernel->is_data_movement_kernel())`.

**Rule 1** — gate only the kind check; leave `access_pattern` and `num_threads` unconditional:

```cpp
// Kind agreement guarantees one per-role processor mask, which is a Gen2 concern; on Gen1 the
// per-node census already implies per-node kind uniformity (one instance per role per node).
if (is_gen2_arch(hal)) {
    TT_FATAL(records[i].kernel->is_compute_kernel() == first_is_compute, ...);
}
```

**Rule 2** — same treatment, matching `:1495`:

```cpp
TT_FATAL(
    !is_gen2_arch(hal) || producer_kernels == consumer_kernels, ...);
```

`dataflow_buffer_spec.hpp:44-50` would want the matching qualification on its kind bullet so the
header and the validator continue to agree.

## 6. Test impact

I checked the existing coverage. `ProgramSpecTestGen1` and `ProgramSpecTestQuasar` in
`tests/tt_metal/tt_metal/api/metal2_host_api/test_program_spec.cpp` bracket both rules.

| test | fixture | effect |
|---|---|---|
| `CPU_DFBMixedKindProducersOnSameNodeFailsWithoutFlag` (`:3411`) | Gen1 | **expected message changes** — see below |
| `CPU_DFBMixedKindProducersOnSameNodeSucceedsWithFlag` (`:3440`) | Gen1 | unaffected (flag path, already skipped) |
| `CPU_DFBSelfLoopWithExtraProducerSideKernelFails` (`:834`) | Quasar | unaffected — Gen2 unchanged |
| `CPU_DFBMultiBindingSelfLoopWithMatchingSidesSucceeds` (`:801`) | Quasar | unaffected |
| `CPU_DMKernelSelfLoopOnGen1Succeeds` (`:3283`) | Gen1 | unaffected |

**One test needs a one-line update, and its spec stays rejected.**
`CPU_DFBMixedKindProducersOnSameNodeFailsWithoutFlag` puts two producers (one compute, one DM) on
node `{0,0}` with no flag. Today rule 1 fires first, so the test matches
`HasSubstr("mixing compute and data-movement kinds")`. With rule 1 scoped to Gen2 the same spec is
still rejected — by the census, which sees two producer instances on one node — so only the expected
substring needs to change to the malformed-node message (`:1456`). Arguably that is the more accurate
assertion for a Gen1 spec anyway, since two producers on one node is the actual violation.

**Rule 2 has no Gen1 test**, so scoping it to Gen2 breaks nothing. Its two tests are both Quasar
fixtures. Note that `CPU_DFBSelfLoopWithExtraProducerSideKernelFails` is structurally the same shape as
this port's topology (one node self-looped, another node producer→consumer with different kernels) —
on Gen2 that rejection is correct, which is why this proposal leaves Gen2 alone rather than
reformulating either rule per-node everywhere.

**A new Gen1 test should be added** for the topology this unblocks: one DFB, a DM producer plus a
compute consumer on one node, and a self-looping DM kernel on a second node, expecting no throw. Happy
to write it.

**CI context.** These tests run in the Sanity budget via `runtime_sim_cpp_unit_tests`
(`tests/pipeline_reorg/runtime_sanity_tests.yaml`, SKUs `sim_wh_n150` / `sim_bh_p150`), which runs
`unit_tests_api --gtest_filter='-ProgramRunArgsTestQuasar.*:ProgramSpecTestQuasar.*:ProgramSpecHWTest.*'`.
The leading `-` is an exclusion, so `ProgramSpecTestGen1` is gated there but the Quasar spec fixtures
are not. I found no other CI job that runs `ProgramSpecTestQuasar.*` — worth knowing if you are
relying on those Gen2 assertions.

## 7. Caveat worth your judgement: the emulator reads the mask on Gen1

"Inert" is accurate for the device blob, but `tt_metal/impl/emulation/emulated_program_runner.cpp`
reads the masks directly and handles all three architectures (`ARCH::QUASAR` / `WORMHOLE_B0` /
`BLACKHOLE` at `:1507-1511`), including explicit Gen1 handling — `:3601-3603` distinguishes Gen1 from
Gen2 masks by testing the high bits, and `:3454-3455` / `:3616-3617` derive producer/consumer identity
from them.

For a DFB whose kinds differ across nodes, the representative mask comes from whichever record is
first, so under emulation the modelled producer/consumer identity could depend on binding order. Two
notes:

- The exposure is **pre-existing**: the same representative-mask shortcut already applies to Gen1
  instance-multi-binding, whose "harmless" justification at `:2986` rests on the device blob and does
  not cover the emulation path.
- If it matters, the companion change belongs in the emulator (select the mask per node or per
  binding), not in the validation rules.

I cannot tell from outside whether the ttsim path needs Gen1 masks to be accurate. Flagging it so the
decision is yours rather than discovered later.

## 8. Why not `allow_instance_multi_binding`

It is a different feature and the wrong tool. Per `advanced_options.hpp:188-204` it means *a DFB
instance* having more than one producer and/or consumer — per-node multiplicity — and it is documented
as "unsafe", "discouraged", "for backwards compatibility with legacy APIs", and as forfeiting "the
protections of the FIFO synchronization mechanics".

The existing Gen1 test pair confirms the intent: `...SucceedsWithFlag` sets the flag precisely to put
two producers *on one node*. This program has one producer and one consumer on every node, so nothing
is instance-multi-bound. Setting the flag would (a) assert something untrue about the program,
(b) give up FIFO guarantees this kernel's correctness depends on — the lockstep write-pointer
behaviour in §1 is a FIFO-semantics argument, (c) opt into a discouraged legacy path, and (d) still be
rejected, since rule 2 sits outside the `if (!allow_multi)` guard.

## 9. Alternatives considered

- **Reformulate both rules per node on all architectures.** Rejected: it would make
  `CPU_DFBSelfLoopWithExtraProducerSideKernelFails` pass, and on Gen2 that spec should be rejected —
  the mask is live there and one per-role mask really must cover every instance.
- **A new option scoped to the kind condition.** If an opt-in is preferred to an arch gate, it should
  be its own flag rather than reuse of the instance-multi-binding one.
- **Refuse the configuration in the op.** It works on `main`, so this would be a user-visible
  regression introduced by an internal port. Noted for completeness; not proposed.

## 10. Reproducer

Fails at program build with the factory on `ProgramSpec`; passes on the legacy path.

```python
import torch, ttnn

device = ttnn.open_device(device_id=0)
M, K, N = 256, 512, 128

torch_a = torch.randn(M, K, dtype=torch.bfloat16)
torch_b = torch.randn(K, N, dtype=torch.bfloat16)

# BLOCK_SHARDED over a 4x2 grid -> shard [128, 128] = [4, 4] tiles
#   in0_sender_num_cores_along_width = 4   (shard grid x)
#   num_blocks_x = 1                       (N = 4 tiles / per_core_N = 4)
#   4 > 1  ->  in0 senders land on cores with no output work
a_mem = ttnn.create_sharded_memory_config(
    shape=(M, K), core_grid=ttnn.CoreGrid(x=4, y=2),
    strategy=ttnn.ShardStrategy.BLOCK, orientation=ttnn.ShardOrientation.ROW_MAJOR,
)
a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_mem)
b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)

out = ttnn.matmul(a, b, program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(4, 2), in0_block_w=4,
    out_subblock_h=1, out_subblock_w=1, out_block_h=4, out_block_w=4,
    per_core_M=4, per_core_N=4,
    transpose_mcast=False, fused_activation=None, fuse_batch=True,
))
print(ttnn.to_torch(out).shape)   # legacy path: (256, 128), pcc 0.9999
ttnn.close_device(device)
```

Observed failure:

```
TT_FATAL @ .../program_spec.cpp:1377: records[i].kernel->is_compute_kernel() == first_is_compute
DFB 'in0' has multiple CONSUMER KernelSpecs mixing compute and data-movement kinds
('in0_mcast_no_work' is a data-movement kernel; 'compute' is a compute kernel). All KernelSpecs
bound to the same DFB role must be of the same kind — the DFB's hardware config carries a single
processor mask per role.
```

With `allow_instance_multi_binding` set, rule 1 is skipped and rule 2 rejects instead.

`transpose_mcast=True` reaches the same shape with the axes swapped
(`in0_sender_num_cores_along_width` becomes the shard grid's `y`). No test in the factory's confirmed
test set covers this shape, which is why it took a hand-written probe to surface.
