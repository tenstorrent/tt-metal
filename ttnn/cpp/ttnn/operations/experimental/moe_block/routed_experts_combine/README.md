# routed_experts_combine

`ttnn.experimental.moe_block.routed_experts_combine`: the flat routed experts and the sparse combine in **one program
per chip** (the MoE block's experts + combine stage). Every expert output row goes from the flat expert's y writers
through L1 into a fabric port's injection ring (or straight into a local accumulator's ring), home over the 2D fabric
(Y first, then X; multi-hop unicasts the routers forward) into a landing ring of the home accumulator, and is summed
there in fp32 (gate weight x row). With `combine_snap` (a power of two q) every term is rounded to a multiple of q
before its add, so the sum is bit-identical in any landing order.

Port of the Python exchange of `mstaletovic/moe-trim` (`combine_rf/sparse_combine_rf.py`, `sparse_combine_l1.py`,
`combine_plan.py`, `tt_k3_token_split_moe.py`). Kernels are mechanical ports: placeholders became compile-time args,
`ACC_LUT` / `__FUSED__` became the defines `REC_ACC_LUT` / `REC_FUSED`.

## Cells, per chip

* a **port** per (direction, link) next to its link's Ethernet core (`MoEBlockResources::ports`, `CombineTuning.place`):
  one fabric connection; injects its rows, returns credits for the rows its chip consumed, forwards ring layouts.
* **A accumulators** (o = token % A): a landing pool carved per call into one ring per source chip by that call's
  demand; reader (rings + local ring), fp32 accumulate compute, writer. Ports apart (per-chip roles: a chip's
  accumulators are the cells less its own ports, the senders find a home's accumulators in a DRAM LUT) or on the ports'
  cells (`acc_on_ports`), whichever `combine_auto` picks / the head planned (`CombineConfig.n_acc`, `.mode`).

## Program, arena and per-call zeroing (differs from the reference)

One Program per chip (a mesh-workload factory): the flat routed expert's fragment
(`flat_routed_expert/device/flat_program.hpp`: `add_flat_routed_expert_program`) on its cells, the combine
(`device/combine_program.hpp`: `add_combine_program`) on the cells it leaves, the y writers' report and L1 hand-off
args (`set_flat_combine_report`). With `experts_during_dispatch` the sparse 2D dispatch runs first in the combine's
port kernels (`sparse_dispatch_2d`'s `sa::read_phase` / `sa::send_phase`, its CBs 24..29 on the port cores, the manifest
first; the flat x relays wait for the ports' ready counts, `FlatLayoutOptions.x_ready`).

The arena is the call's L1 tensor over the whole worker grid (allocated by the host entry point, freed after the
launch): the flat expert binds its CBs to it, and main's device-wide L1 check (every program CB below the lowest L1
buffer on any core) rules out a combine CB arena next to it. A sharded tensor sits at the same address on every core of
every chip, which the fabric writes need. Its first `arena_bytes` a core are the combine's (the reference's offsets and
CB placement), then the y writers' words. No address is in the program hash: on a cache hit every buffer address (the
arena included) is re-applied to both halves' runtime args, arena CBs and the writers' report.

The reference zeroed the combine's state from another program (the dispatch's zero gate). Here each call zeroes it
in-program: the ports' senders (ports apart), the accumulators' readers and the flat y writers (their words; se_yrm.hpp
SE_Y_L1 args 5..7) zero their own L1 and count in on the chip's gate (accumulator 0), which releases the chip's combine
cores; the senders' existing cross-chip start barrier then orders all chips (`kernels/dataflow/combine_zero_gate.hpp`,
the persistent `bar` words).

The combine's L1 per core is the reference's (the same `combine_auto` sizing on the same budget: by default the
reference exchange's arena size, `exchange_arena_bytes`, from the largest free L1 block at the first call less the flat
program's static CBs and the dispatch's), so the rings and the device time should match.

## Resources (`MoEBlockResources`)

`routed_experts_combine/state` (plan: cells, layout, credit policy, LUT; computed once), `routed_experts_combine/bar`
(16 words on every worker core), `routed_experts_combine/hcache` (the senders' routed-header cache, DRAM),
`routed_experts_combine/lut` (per-chip roles), `routed_experts_combine/out` (y_own, persistent),
`routed_experts_combine/y` (the flat expert's DRAM y: never written with the L1 hand-off),
`routed_experts_combine/dispatch_sems` (the dispatch phase's semaphore words); the flat expert's via
`ensure_flat_routed_expert_resources` (plan, done / relay words). The program hash holds
the config, the cells / writers, the resources' `plan_key()` and the tensor specs, never an address; buffer addresses
are patched on a hit (BufferBinding).

## Tests

* `test_routed_experts_combine_host.py`: the credit policy and `combine_auto`'s invariants; no device.
* `test_routed_experts_combine.py`: device, bit-exact vs saved reference cases (GLM-5.3 / K3 at 8x4, chunk 5120,
  uniform routing, snap 2^-14; a miss and two hits).
