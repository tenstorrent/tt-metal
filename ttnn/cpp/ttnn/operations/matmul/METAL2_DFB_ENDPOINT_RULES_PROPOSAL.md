# Proposal: scope two DFB endpoint checks to Gen2

**Context:** Metal 2.0 port of `MatmulMultiCoreReuseMcast2DProgramFactory` (matmul port series,
parent issue #41908).
**Status:** blocking that port. The configuration below runs correctly on the legacy path today and
raises `TT_FATAL` at program build once the factory is on `ProgramSpec`.
**Arch:** reproduced on Blackhole.

## Ask

Two endpoint checks in `ValidateProgramSpec` are enforced on all architectures, but both exist
solely to guarantee a single per-role processor mask — and on Gen1 that mask is inert by the
runtime's own account (`program_spec.cpp:2985-2987`). Please scope them to Gen2, matching the
idiom already used by the DM-self-loop rule at `program_spec.cpp:1495`.

| | site | what it requires |
|---|---|---|
| **Rule 1** | `program_spec.cpp:1366-1377` | all `KernelSpec`s on one DFB role have the same kind (compute vs DM) |
| **Rule 2** | `program_spec.cpp:1507-1522` | for a self-looped DFB, `producer_kernels == consumer_kernels` |

---

## 1. Problem

`MatmulMultiCoreReuseMcast2DProgramFactory` supports a configuration where `in0` is BLOCK_SHARDED and
its shard grid is wider along the multicast axis than the output needs in columns
(`in0_sender_num_cores_along_width > num_blocks_x`). K is split across the mcast axis, so no core
holds all of K and the `in0` broadcast is a rotating relay: on K-iteration `block`, whichever core
owns that slice multicasts it to the receiver row. When in0's K is spread over more cores than the
output's N needs columns, the surplus cores own K-slices but have no output block — they are pure
suppliers of the reduction dimension.

For those cores the `in0` DFB is **self-looped by the sender**, while on the work cores the same DFB
is sender → compute:

| nodes | PRODUCER | CONSUMER |
|---|---|---|
| work grid | `in0_sender` (DM) | `compute` (compute) |
| no-work senders | `in0_mcast_no_work` (DM) | `in0_mcast_no_work` (**same kernel**) |

So the DFB's roles hold, spec-wide:

- PRODUCER = { `in0_sender`, `in0_mcast_no_work` } — both DM
- CONSUMER = { `compute`, `in0_mcast_no_work` } — **one compute, one DM** → rule 1 rejects
- self-looped, and the two role sets differ → rule 2 rejects

**It has to be one DFB.** The sender multicasts to `.addr = dfb_in0.get_write_ptr()` — it derives one
destination address from its own cursor and broadcasts it to every receiver, so all instances must
sit at the same L1 offset. The self-pop on the no-work cores is what keeps that cursor in phase:
`in0` is double-buffered (`in0_num_entries *= MCAST_INPUT_BUFFERING_DEPTH`), on a work core compute's
`pop_front` advances the cursor, and a core with no compute must do it itself or broadcast a stale
slot address. The kernel says as much:

```
// If core does not produce output block work, free dfb::in0 immediately.
// This is necessary since mcast is in lockstep; this ensures write ptr addresses are synced
// properly for cores that only send and have no compute / writer active.
```

A private scratch DFB on those cores would decouple the address and break the multicast, and the
sender is the only kernel present there to drive the cursor. The topology is not a modelling choice.

## 2. The topology is legal on Gen1 — measured

The per-node census passes: every node has exactly one producer instance and one consumer instance,
which is the invariant `dataflow_buffer_spec.hpp:41-45` states. Only the two secondary rules fail.

Same reproducer, same device, same inputs:

| build | result |
|---|---|
| factory on `ProgramSpec` | `TT_FATAL` at `program_spec.cpp:1377` |
| factory on the legacy descriptor path | **succeeds, pcc 0.999881** |

The kernel placement is also not new: legacy issues the same two `CreateKernel` calls from the same
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
That is provable rather than asserted: `check_role_uniformity` tests `access_pattern`, then
`num_threads`, then kind in that order within the loop, and the observed failure is the third.

There is no option that relaxes the kind condition alone.

## 4. Proposed change

Both rules already have an arch-gated sibling immediately adjacent — the DM self-loop restriction at
`:1495` is `!(is_gen2_arch(hal) && self_loop_kernel->is_data_movement_kernel())`. The proposal is to
use the same idiom.

**Rule 1** — gate only the kind check (leave `access_pattern` and `num_threads` unconditional):

```cpp
// The kind condition exists to guarantee a single per-role processor mask; that mask is
// inert on Gen1 (see Step 2b), so enforce it only where it is live.
if (is_gen2_arch(hal)) {
    TT_FATAL(
        records[i].kernel->is_compute_kernel() == first_is_compute, ...);
}
```

**Rule 2** — same treatment, matching `:1495`:

```cpp
TT_FATAL(
    !is_gen2_arch(hal) || producer_kernels == consumer_kernels, ...);
```

`dataflow_buffer_spec.hpp:44-50` would want the matching qualification on its kind bullet, so the
header and the validator continue to agree.

## 5. Justification: on Gen1 the mask these rules protect is inert

Both rules state the same rationale. Rule 1 (`:1334-1338`): *"the DFB's hardware config carries a
single processor mask per role, and compute / DM masks live in disjoint bit ranges … mismatched kinds
cannot share a mask."* Rule 2 (`:1502-1506`): sharing *"would make the producer/consumer mask and
lowering semantics ambiguous."*

That mask is `DataflowBufferConfig::producer_risc_mask` / `consumer_risc_mask`
(`dataflow_buffer.hpp:40,43`). It is populated unconditionally, including for Gen1 kernel types
(`dataflow_buffer.cpp:846-893`, in `BindDataflowBufferToProducerConsumerKernels`). But **every
consumer of it inside `tt_metal/impl/dataflow_buffer/` is
behind `hal().has_tile_counter_registers()`**, which is Quasar — asserted outright at
`dataflow_buffer.cpp:284` (`TT_FATAL(hal.has_tile_counter_registers(), "compute_dfb_config_serialized_size requires Quasar")`):

| read | arch gate |
|---|---|
| `:1214` (`validate_ring_extent`) | early return at `:1211` |
| `:1287-1290` (`calculate_num_tile_counters`) | tile-counter path; called from `:2137-2138` |
| `:1444-1446` | guarded at `:1442` |
| `:1933-1939` (`finalize_dataflow_buffer_configs`) | returns for non-tile-counter archs at `:1882` |

And the lowering says it directly. `MakeDataflowBufferConfig` takes a *single representative* mask
from the first producer/consumer record (`:2635-2641`), which Step 2b justifies at `:2985-2987`:

> Instance-multi-binding (Gen1-only) intentionally binds same-role kernels on distinct RISCs … so
> their risc_masks differ by design and the uniform-mask requirement does not apply. **On Gen1 the DFB
> lowers to a plain circular buffer where the mask is inert (it never reaches the device blob)**, so
> the single representative mask `MakeDataflowBufferConfig` takes from the first binding is harmless.

So the runtime already reasons exactly this way — *Gen1 + mask inert ⇒ drop the uniformity
requirement* — for the mask-equality requirement in Step 2b. The request is to apply the same
reasoning to the two checks that encode the same requirement earlier in validation. On Gen1 the FIFO
pointers are shared L1 state any RISC can drive (`advanced_options.hpp:190-192`), which is why the
topology works on hardware today.

## 6. Caveat worth your judgement: the emulator does read the mask on Gen1

"Inert" is accurate for the device blob, but **`tt_metal/impl/emulation/emulated_program_runner.cpp`
reads the masks directly and handles all three architectures** (`ARCH::QUASAR` / `WORMHOLE_B0` /
`BLACKHOLE` at `:1507-1511`), including explicit Gen1 handling — `:3601-3603` distinguishes Gen1 from
Gen2 masks by testing the high bits, and `:3454-3455` / `:3616-3617` decide producer/consumer identity
from them.

For a mixed-kind DFB the representative mask comes from whichever record happens to be first, so under
emulation the modelled producer/consumer identity could depend on binding order. Two things to note:

- This is **not introduced by this proposal.** The same representative-mask exposure already exists for
  Gen1 instance-multi-binding, whose "harmless" justification at `:2986` rests on the device blob and
  does not cover the emulation path.
- If it matters, the companion change is in the emulator (select the mask per node or per binding)
  rather than in the validation rules.

I can't tell from outside whether the ttsim path needs Gen1 masks to be accurate. Flagging it so the
decision is yours rather than discovered later.

## 7. Why not `allow_instance_multi_binding`

It is a different feature and the wrong tool, not merely insufficient. Per
`advanced_options.hpp:188-204` it means *a DFB instance* having more than one producer and/or consumer
— i.e. per-node multiplicity — and it is documented as "unsafe", "discouraged", "for backwards
compatibility with legacy APIs", and as forfeiting "the protections of the FIFO synchronization
mechanics".

This program has exactly one producer and one consumer on every node, so nothing is
instance-multi-bound. Setting the flag would (a) assert something untrue about the program, (b) give up
FIFO guarantees this kernel's correctness depends on — the lockstep write-pointer behaviour in §1 is a
FIFO-semantics argument, (c) opt into a discouraged legacy path, and (d) still be rejected, since rule 2
sits outside the `if (!allow_multi)` guard at `:1380`.

Rule 2's exclusion from that guard may itself be an oversight, given its rationale is the mask
ambiguity the flag defines away. But moving it under the guard would not make the flag an acceptable
answer for this topology.

## 8. Alternatives

- **Evaluate the kind condition per node rather than per spec.** Most faithful to the hardware
  invariant — what matters is that a given node has one producer processor and one consumer processor,
  which holds here. Larger change; also resolves the emulator question, since a per-node mask is
  well-defined for this topology.
- **A new option scoped to the kind condition.** If an opt-in is preferred over an arch gate, it should
  be its own flag rather than reuse of the instance-multi-binding one.
- **Refuse the configuration in the op.** It works on `main`, so this would be a user-visible
  regression introduced by an internal port. Noted for completeness; not proposed.

## 9. Reproducer

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
