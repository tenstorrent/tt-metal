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

## Arena and per-call zeroing (differs from the reference)

The reference kept the combine's state in a per-call L1 tensor that had to land at one fixed address, zeroed by
another program (the dispatch's zero gate). Here the state lives in the program's **arena CB** (`kCbArena`), the first
CB of the program on every core, so it starts at the same address on every core and chip; every address exchanged is an
offset from it. The port's injection ring is the second CB on the port cores (at arena + `inj_off`). Each call zeroes
the state in-program: ports' senders (ports apart) and accumulators' readers zero their own state and count in on the
chip's gate (accumulator 0), which releases the chip's combine cores; the senders' existing cross-chip start barrier
then orders all chips (`kernels/dataflow/combine_zero_gate.hpp`; persistent `bar` words). The flat y writers take part
(zero their words, count in) once the flat builder is integrated.

The combine's L1 per core is the reference's (the same `combine_auto` sizing on the same budget: by default the
reference exchange's arena size, `exchange_arena_bytes`), so the rings and the device time should match; the CBs that
only local kernels use (`CB_TAB`, `CB_ROW`, ...) are ordinary program CBs after the arena instead of carved from it.

## Resources (`MoEBlockResources`)

`routed_experts_combine/state` (plan: cells, layout, credit policy, LUT; computed once), `routed_experts_combine/bar`
(16 words on every worker core), `routed_experts_combine/hcache` (the senders' routed-header cache, DRAM),
`routed_experts_combine/lut` (per-chip roles), `routed_experts_combine/out` (y_own, persistent). The program hash holds
the config, the cells / writers, the resources' `plan_key()` and the tensor specs, never an address; buffer addresses
are patched on a hit (BufferBinding).

## Status

* Builds: host math, the combine's program builder (`device/combine_program.{hpp,cpp}`), the composite factory with
  the flat builder's seam (`SEAM(flat)` in the factory / types / device test), the kernels.
* Not yet: the flat builder plugged in (the op refuses to run without y writers), the in-program dispatch phase
  (`experts_during_dispatch`: sparse_dispatch_2d's `sa::read_phase` / `sa::send_phase` called first in the combine's
  port reader / sender, its CBs and manifest on the port cores). Its manifest CB must then come after the arena CB on
  the port cores (or the arena would no longer start at the same address on every combine core).

## Tests

* `test_routed_experts_combine_host.py`: host math vs the reference (pieces, state layout, `combine_l1_layout`,
  `combine_auto`, credit policy, arena sizing, `plan_reference`); no device.
* `dump_ref_routed_experts_combine.py` (reference tree): GLM-5.3 / K3 at 8x4, chunk 5120, uniform routing, snap 2^-14.
* `test_routed_experts_combine.py`: device, bit-exact vs the references; `test_routed_experts_combine_host_perf.py`:
  us a call on a cache hit.
