# Metal 2.0 Host-Port Report — `experimental/quasar/reduction/generic`

## Status: PORTING (earlier INTER-self-loop capitulation was WRONG — see correction)

### Correction (supersedes the capitulation below)
The capitulation assumed the fused tilize→reduce / accumulator intermediate CBs (`cb_tile_in`, `cb_acc`,
`cb_ineg`) were **INTER**-thread self-loops (unsupported). That was wrong. The already-ported m2
`pool_generic` factory (`pool_multi_core_program_factory.cpp`) proves compute self-loop CBs are supported:
it declares its tilize intermediates (`DFB_PRE_TILIZE`/`DFB_FAST_TILIZE`) and `DFB_COMPUTE_TMP_IDX` as
**INTRA** self-loops and runs fine. Mechanism:
- Bind the DFB **twice** from the compute KernelSpec — same `accessor_name`, one `PRODUCER` + one `CONSUMER`.
- `compute.advanced_options.dfb_self_loop_connectivities.insert({DFB_NAME, DFBSelfLoopConnectivity::INTRA});`

So reduction is portable. Scope per request: port what the resnet50/quasar pytests reach + what is easily
portable. Model hot path = `pool_sum(canonical, dim=2=H)` → **ReduceMultiCoreH** (`reduce_rm`).

### Plan / progress
- **single_core_hw — PORTED (build pending).** Proving ground for the reduce m2 pattern. Forks:
  `reader_unary_reduce_universal_start_id_metal2` (data DFB + scaler DFB via `prepare_reduce_scaler<dfb::scaler,…>`),
  `writer_unary_interleaved_start_id_metal2`, `reduce_metal2` (non-negate, no self-loop),
  `reduce_hw_neg_metal2` (negate/MIN: `acc`/`ineg` bound PRODUCER+CONSUMER on the compute kernel +
  `dfb_self_loop_connectivities` INTRA). Reduce defines → `compiler_options.defines` on reader+compute.
  .hpp → create_program_artifacts (single_core_hw only; the other 2 factories stay legacy — framework
  auto-detects concept per factory). Self-audit clean. NOTE: not model-reachable (HW dim), ported as the
  "easily portable" piece + to de-risk the self-loop approach before multi_core_h.
- **multi_core_h — PORTED (build pending), SCOPED to the interleaved non-negate path (the resnet path).**
  Per user direction ("reachable + easy, defer rm"): only the interleaved, non-negate H-reduce path is on
  Metal 2.0 — this is what pool_sum/avg_pool2d reaches (Sum over H, INTERLEAVED in/out). It also covers
  Int32-MAX (reduce_metal2 routes Int32 to SFPU internally). New fork:
  `reader_unary_transpose_wh_universal_input_cols_partitioned_metal2`; reuses `reduce_metal2` +
  `writer_unary_interleaved_start_id_metal2` (the H path uses the same reduce.cpp/writer as single_core_hw,
  only REDUCE_DIM differs). 2 core groups (g1/g2), per-core RTAs mirror the legacy interleaved loop.
  - **Negate is now disabled at the host so the m2 H factory never receives it:** `h_reduce_negate_fits_in_l1`
    (common.cpp) gates fused in-kernel negate (reduce_h_neg) vs the external-negate fallback. Added
    `constexpr bool kFusedHNegateSupportedOnMetal2 = false;` → it always returns false → MIN H-reduce
    always takes the external fallback `-reduce(MAX, H, -x)` (regular negate=false reduce, which IS on
    Metal 2.0). So the negate TT_FATAL below is a defensive guard, not reachable in practice. (W-reduce
    negate still uses legacy multi_core_w / reduce_w_neg, which is fine.) Flip the bool once reduce_h_neg
    is ported.
  - **TT_FATAL (not yet ported):** rm_path (row_major_h_dense_path — perf-disabled upstream), width-sharding
    (re-routed to INTERLEAVED upstream; uses UnicastEndpoint + raw-L1 backed src1), and negate/MIN
    (dual SFPU/FPU with self-loop acc/ineg; tokens must bind even in the discarded if-constexpr branch).
    These proved NOT "easy" on inspection (hence narrowed from the originally-chosen interleaved+negate+width).
    They stay on the legacy ProgramDescriptor path (framework auto-detects concept per factory).
- multi_core_w / welford: still legacy. Port later if needed (multi_core_w mirrors multi_core_h's paths;
  welford is a separate DeviceOperation).

---
## (Superseded) earlier capitulation analysis — kept for history
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
