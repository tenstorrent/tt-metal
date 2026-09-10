# Quasar matmul: Scratchpad-backed K-spill partials — feasibility findings (2026-09-10)

**Branch:** `vsureshTT/quasar_uplift_round_2` (uncommitted; delete this file before merge)
**Op:** `ttnn/cpp/ttnn/operations/experimental/quasar/matmul` (1D mcast_in0 factory + `bmm_large_block_zm_fused_bias_activation_metal2.cpp`)
**Context:** `QUASAR_CRAQSIM_RUN_2026-09-10.md` blocker B1 — craq-sim aborts on the 2nd push into the intra-tensix partials
DFB `cb_intermed0` (`qsr_tile_counter_check_error: occupancy=2*capacity, posted=2*capacity, acked=0`).

## Stop reason

**Experiment stopped and fully reverted.** While the Scratchpad path was being validated, the coordinator reported that the
K=2048 K-spill probe that aborts craq-sim **passes on the ZEBU RTL emulator (PCC 0.9999)**. B1 is therefore a craq-sim bug
(intra-tensix tile-counter accounting), not an op or runtime defect, and no workaround in the op is needed. All Scratchpad
source edits were reverted by hand; `git diff` on this op copy again shows only the three earlier fixes (Semaphore::value()
in the in0 receiver, `pack_init` after every pack-destination switch, `separate_out_interm` forced on Quasar). The host
library was rebuilt (`qsr_rebuild` → REBUILD_OK) and the temporary graph-test copy was deleted. Nothing from the
experiment remains in the tree except this file.

## Feasibility findings (what exists on the Quasar LLK)

### Route 1 — id-free 2.0 compute API (`LLKOperand`): NOT available on Quasar
- `tt_metal/hw/inc/api/compute/experimental/2_0/llk_operand.h` hard-fails on any non-Blackhole build:
  `#ifndef ARCH_BLACKHOLE  #error "The experimental 2.0 compute API (LLKOperand) is Blackhole-only"`.
  `pack.h` / `tile_move_copy.h` in that directory are documented "Blackhole only".
- `tt_metal/hw/ckernels/quasar/metal/llk_api/` contains **zero** references to `LLKMemDescriptor` / `LLKOperand`
  (blackhole: 22 files; wormhole_b0: 0). No `experimental/2_0/` subdirectory exists for Quasar.
- `tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h` is Blackhole-only glue (indexes chlkc arrays by CB id).

### Route 2 — Quasar buffer-descriptor (BFD) entry point taking a raw L1 base: EXISTS
- `tt_metal/tt-llk/tt_llk_quasar/common/inc/llk_bfd_alloc.h` (the copy on the JIT include path; `tt_metal/third_party/tt_llk`
  is an untracked stale copy): `ckernel::trisc::bfd_alloc_and_program<BfdResource E, L1AccessMode MODE>(const TensorShape&,
  uint32_t l1_base_addr /*16B units*/, uint32_t l1_data_format)` — allocates a BFD id from the calling TRISC's partition
  (unpack [0,16), pack [16,24)), builds the descriptor (`construct_buf_desc`), programs the table entry
  (`_configure_buf_desc_table_`), and records it in `bfd_current<E>()`. No size/limit field: the descriptor is base + tile
  geometry + format; tile indices are unbounded offsets from the base.
- Every Quasar `llk_*_init` is a thin wrapper over it with a DFB lookup:
  `llk_pack_program_bfd(output_id)` = `bfd_alloc_and_program<Pack0>(get_output_tensor_shape(id), tc_slots[0].base_addr, pack_dst_format[id])`
  (`tt_metal/hw/ckernels/quasar/metal/llk_api/llk_pack_common_api.h`); `llk_unpack_program_bfd<E>(operand_id)` likewise
  (`llk_unpack_common_api.h`). The primitives then take the descriptor + a tile index, not a DFB:
  `_llk_pack_init_(bfd_id, TensorShape)`, `_llk_pack_(dst_tile, l1_tile_idx, TensorShape)` (`tt_llk_quasar/llk_lib/llk_pack.h`);
  `_llk_unpack_unary_operand_init_<UNP_A,...>(bfd_id, TensorShape, 1)`, `_llk_unpack_unary_operand_<UNP_A,...>(l1_tile_idx,
  TensorShape)` (`llk_lib/llk_unpack_unary_operand.h`).
- Consequence: a Program-scope `Scratchpad` (`tt_metal/hw/inc/api/scratchpad.h`, `get_base_address()` in bytes, host
  `ScratchpadSpec`/`ScratchpadBinding`) can be handed to the packer and unpacker through supported LLK calls without
  writing any BFD register from the op; it bypasses the DFB layer (no reserve/push/wait/pop, no tile counters), not the LLK.
  Formats/geometry must come from a bound DFB's chlkc entries (e.g. `cb_out`, valid only when the partials format equals
  the output format; the JIT emits `unpack_*`/`pack_*` arrays for all DFBs on the core, so both threads see `cb_out`).
- Host side works for compute kernels: `ProgramSpec::scratchpads` + `KernelSpec::scratchpad_bindings`;
  `ProgramImpl::allocate_scratchpads` (tt_metal/impl/program/program.cpp) has no DM-only restriction, aligns to the DRAM
  alignment (≥16 B, so `base >> 4` is exact) and delivers the base as a CRTA; TRISCs have `get_common_arg_addr`
  (`api/compute/common.h`). In-tree precedent is DM-only (data_movement/repeat, copy, paged_cache).

### Ordering primitive for a self-produced buffer without credits: EXISTS
- `tt_llk_quasar/common/inc/ckernel_trisc_common.h`: `semaphore::PACK_UNPACK = 7; // pack <-> unpack sync on L1 memory`
  (declared for exactly this, referenced nowhere else in tt-llk / metal LLK API); `semaphore::MATH_PACK = 1`,
  `UNPACK_MATH = 4` are taken. Helpers in `tt_llk_quasar/llk_lib/llk_sync.h`: `_llk_sync_init_(sem, max, init)`,
  `_llk_sync_wait_<StallRes, Condition>(sem...)` (SEMWAIT), `_llk_sync_get_<...>(sem)`, `_llk_sync_post_<WaitRes...>(sem)`
  (STALLWAIT(STALL_SYNC, WaitRes) + SEMPOST). The DEST handshake alone does not order PACK(k) before UNPACK(k+1) (SyncHalf
  lets MATH(k+1) start while PACK(k) is in flight), so an explicit PACK→UNPACK post/wait per reloaded K-block was required;
  the WAR direction is excluded by the MATH_PACK / SrcA-dvalid chain.

## For the record: what ran before the stop (all on craq-sim, host-tilized bf16 DRAM interleaved, 8x4 grid)

The Scratchpad path (kernel define `MM_PARTIALS_SCRATCHPAD`, one `ScratchpadSpec` of `out_block_tiles * tile_size`, PACK→UNPACK
handshake on `semaphore::PACK_UNPACK`, one post per K-block whose partials the next block reloads) was implemented, built and run
before the revert. These results are evidence only — the code is gone.

| Probe (1D mcast_in0) | DFB path (earlier run) | Scratchpad path |
|---|---|---|
| K=64 (no partials; stays on the DFB path by construction) | PASS 0.99999 | PASS 0.99999 (unchanged) |
| K=128, N=1024 (1 round trip) | PASS 0.99999 | PASS 0.99999 |
| K=2048, N=1024 (32 round trips) | sim abort `occupancy=2 > capacity=1` | PASS 0.99991 (19.4k cycles) |
| K=2048, N=2048 (32 round trips, 2-tile subblocks) | sim abort `occupancy=4 > capacity=2` | PASS 0.99991 |
| K=2048, N=2048, packer_l1_acc (model config) | sim abort | PASS 0.99997 |
| M=1024, K=2048, N=2048 through 1D (case-06 shape; 32 blocks x 8 subblocks) | — | PASS 0.99991 (371k cycles) |
| graph case 06 as captured (routes to the **2D** factory, out of scope) | sim abort `68 > 64` | sim abort `68 > 64` (unchanged) |

Not done because of the stop: the A/B run with the define forced off on the same build.

## Open issues / notes for the sim owner
- B1 remains a **craq-sim** defect: intra-tensix (PACK→UNPACK, same Neo) DFB tile counters never see the unpacker's
  `POP_TILES` ack in the simulator's check (`posted=2*capacity acked=0`), while the packer's `reserve_back` does see the
  space (the kernel would otherwise hang) and the RTL emulator runs the same program to PCC 0.9999. The runtime's own
  `TensixIntra*` gtests cannot run on this sim either (`UnimplementedFunctionality: tensix_execute_pacr_stride`).
- If a Scratchpad-backed partials path is ever wanted for its own sake, the recipe above is complete and was proven
  functionally on the sim; its costs are an explicit semaphore handshake in the kernel, a format-equality restriction, and
  per-block (not per-subblock) PACK→UNPACK granularity.
