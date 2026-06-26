# Metal 2.0 Host-Port Report — `experimental/quasar/reduction/generic`

## Status: CAPITULATED (blocked) — INTER-thread self-loop DFBs unsupported

Pre-port audit was GREEN (feasibility gates: ProgramDescriptor concept, Device 2.0, no UNSUPPORTED
host features). The audit does not analyze per-kernel compute-thread dataflow; the host-port attempt
surfaced a blocker the audit's gates don't cover.

### Blocker
Every reduction compute kernel fuses **tilize → reduce** (and accumulates) *inside a single compute
kernel*, recirculating tiles through intermediate CBs:
- `reduce_rm.cpp` (multi_core_h / multi_core_w, the **model's `pool_sum` hot path**): `tilize<…, cb_rm,
  cb_tile_in>` writes `cb_tile_in` (c_0) from the PACK thread; `compute_kernel_lib::reduce<…, cb_tile_in,
  …>` reads it from the UNPACK thread; partials accumulate through `cb_acc` (c_5) the same way
  (`Accumulate::at(cb_acc, chunk_idx)` — PACK writes, UNPACK reads next chunk).
- `reduce_hw_neg.cpp` / `reduce_{h,w}_neg.cpp` (MIN / negate paths): `cb_ineg` (c_5) and `cb_acc` (c_4)
  are pack-produced / unpack-consumed within the kernel.
- welford compute kernels: same fused-intermediate shape.

A CB that one compute thread produces and a *different* compute thread consumes is, in Metal 2.0, an
**INTER-thread self-loop DFB** (the compute KernelSpec binds the same DFB as both PRODUCER and CONSUMER).
Per `tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp`:

> Only the INTRA case is currently supported. INTER will trigger a validation error. There are
> currently no known use cases for an INTER-thread self-loop.

So the fused intermediate (`cb_tile_in`) and the accumulator (`cb_acc`) cannot be expressed in current
Metal 2.0. This blocks **all four factories** (multi_core_h, multi_core_w, single_core_hw, welford) —
including the non-negate Sum path the model needs — because the fused compute is shared across them.

### Why not work around it
- Splitting the fused compute into two compute KernelSpecs (a tilize kernel → normal DFB → a reduce
  kernel) is not viable: one core runs one compute kernel; this would be a semantic rewrite, not a
  faithful host-2.0 port, and is exactly the kind of fabrication the porting recipe forbids.
- The negate (MIN) HW path additionally relies on the same INTER self-loop (`cb_ineg`/`cb_acc`).

### Device-2.0 status
Kernels are already on Device 2.0 (CircularBuffer/Noc classes). Only the host-2.0 port is blocked.

### Re-attempt when
INTER-thread self-loop DFB support lands in Metal 2.0 (tracked alongside the "scratchpad" /
single-ended-producer feature that blocks reshape_view + tilize_with_val_padding sharded/block_interleaved).
At that point the port is mechanical: reader (data + `prepare_reduce_scaler` scaler DFB) → fused compute
(self-loop `cb_tile_in`/`cb_acc`) → writer, with `reduce_defines` → `compiler_options.defines`.
