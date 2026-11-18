# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/concat`

- **`ConcatDeviceOperation`** (`device/concat_device_operation.hpp:24`) — one DeviceOperation, seven factories in `program_factory_t` (`concat_device_operation.hpp:30-37`):
  - `ConcatProgramFactory` (`concat_program_factory.cpp`) — **already ported** (Metal 2.0 `create_program_artifacts`, #54651)
  - `ConcatS2SRMProgramFactory` (`concat_s2s_rm_program_factory.cpp`) — **already ported** (#54651)
  - `ConcatS2STiledProgramFactory` (`concat_s2s_tiled_program_factory.cpp`) — **already ported** (#54651)
  - `ConcatS2SMultiProgramFactory` (`concat_s2s_multi_program_factory.cpp`) — `descriptor`, audited here
  - `ConcatS2IProgramFactory` (`concat_s2i_program_factory.cpp`) — `descriptor`, audited here (dead code, see below)
  - `ConcatBlockShardedProgramFactory` (`concat_block_sharded_program_factory.cpp`) — `descriptor`, audited here
  - `ConcatTiledUnalignedProgramFactory` (`concat_tiled_unaligned_program_factory.cpp`) — `descriptor`, audited here (added 2026-08-31 in #53474)

This is a **first audit** (no prior `METAL2_PREPORT_AUDIT.md` in the directory). The op is **partially ported**: commit `a9be36158de` (#54651, 2026-09-21) moved 3 of 7 factories to Metal 2.0. This audit covers the **4 remaining `descriptor` factories**; the 3 ported ones are reported as done.

Kernels exercised by the 4 audited factories (all in `device/kernels/`):
- S2SMulti: `dataflow/reader_s2s_tensor_concat.cpp` (instantiated twice — reader + writer config)
- BlockSharded: `dataflow/reader_writer_block_sharded_concat.cpp` (instantiated twice — reader + writer config)
- S2I: `dataflow/reader_s2i_width.cpp` (**file does not exist**), `dataflow/writer_s2i_width.cpp`
- TiledUnaligned: `dataflow/reader_concat_tiled_unaligned.cpp`, `dataflow/writer_concat_tiled_unaligned.cpp`, `compute/concat_untilize_retilize.cpp`

Kernels used only by the already-ported factories (`reader_concat_*_interleaved_start_id.cpp`, `concat_reader_common.hpp`, `reader_height_sharded_width_concat_two_tensors*.cpp`, `writer_height_sharded_width_concat_two_tensors_tiled.cpp`, `compute/height_sharded_width_concat_two_tensors.cpp`) are already Metal 2.0 and out of scope.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

**Readiness sheet:** fetched fresh in this session, 2026-10-27.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/concat` |
| **Overall** | **RED** — 3/7 factories already ported; the remaining 4 are all blocked; no portable subset. **2026-11-12: GREEN (user waiver) for `ConcatS2SMultiProgramFactory`; 2026-11-18: GREEN (user waiver) for `ConcatBlockShardedProgramFactory`** — see the waiver notes below the table |
| **DOps / Factories** | `ConcatDeviceOperation` → done: `ConcatProgramFactory`, `ConcatS2SRMProgramFactory`, `ConcatS2STiledProgramFactory` · audited: `ConcatS2SMultiProgramFactory`, `ConcatS2IProgramFactory`, `ConcatBlockShardedProgramFactory`, `ConcatTiledUnalignedProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes for every kernel that exists. S2I's reader `reader_s2i_width.cpp` is missing from the tree — cannot be assessed (moot: S2I is dead code) |
| *Prereqs* — Cross-op escapes | Ok (in-family `common/kernels/common.hpp`, shared `kernel_lib` — no borrowed kernel files) |
| *Feature Support* — overall | GREEN (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | S2SMulti `no` · BlockSharded `no` · S2I `no` (all attributed to `Known op issues`) · TiledUnaligned **no row** (spreadsheet broken) · plus `Concept` and `Custom hash` conflicts on every row (spreadsheet broken) |
| *TTNN Readiness* — Concept (current) | Code: 4 × `descriptor` + 3 × `MetalV2`. Sheet: 6 × `descriptor` (the 3 ported factories are mislabelled) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | **Yes** in code: `concat_device_operation.cpp:102-136` (sheet says `no` — conflict) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (all 4) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | TiledUnaligned: `inputs[N]` Case 1, `output` Case 1. Others: skipped (see scoping) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (S2SMulti, BlockSharded, S2I) · TiledUnaligned: **no row — no value to read** |
| *Port work* — TensorAccessor 3rd arg | none — no accessor in the 4 factories passes a 3rd arg |
| *Port work* — CB endpoints | TiledUnaligned: all legal (4 × 1P+1C FIFO). Others: skipped |

> **Waiver (2026-11-12, Edwin):** `ConcatS2SMultiProgramFactory` proceeds to port as **GREEN (user waiver)**. Basis: the 2026-11-06 update below found every Metal 2.0 feature it needs on main, and the 2026-11-08 handoff (committed as `METAL2_PORT_BRIEF.md`) gives the redesign — a `TensorBindingSequence` over `LocalTensorAccessor`s replacing the runtime-indexed borrowed CBs. The other three audited factories stay RED. Concat changes between this audit's last check (`882c6e0953d`) and the port base (`0de8c7f3aad`): #63855 (`buffer()->alignment()` → `mesh_buffer().alignment()` in S2SMulti, cosmetic) and #64498 (validation matrix, outside the factory). Neither affects a finding.

> **Waiver (2026-11-18, Edwin):** `ConcatBlockShardedProgramFactory` proceeds to port as **GREEN (user waiver)**, on the same basis as S2SMulti: the 2026-11-06 update found the features it needs on main, and the Block-sharded handoff (appended to `METAL2_PORT_BRIEF.md`) gives the redesign — inputs addressed by shard id through `ShardView::ResolvedShard` (#62362, merged 2026-11-18 as `9490a5f3a2d`) instead of borrowed CBs and a NoC coordinate table. Port base `e38dfa66343`. Concat changes since the S2SMulti port base: #65249 (row-major mixed L1/DRAM page overflow, `ConcatProgramFactory`) and #63424 (Metal 2.0 headers moved out of `experimental/`). Neither touches this factory. S2I and TiledUnaligned stay RED.

## Result

**RED at op level; no portable subset** among the 4 unported factories. (3 of 7 factories are already on Metal 2.0.)

Every remaining factory is blocked at the **TTNN factory concept** gate:

1. **`ConcatS2SMultiProgramFactory`, `ConcatBlockShardedProgramFactory`** — sheet `Is able to port?` = `no`, attributed to `Known op issues` = `"DFB misuse; will need semi-manual port"`. → routed to whoever the cell names (ops team / Metal 2.0 port owner).
2. **`ConcatS2IProgramFactory`** — sheet `no`, attributed to `Known op issues` = `"... this factory is dead code"`. Confirmed in code (it cannot be reached, and its reader kernel file doesn't exist). → ops team: delete the factory rather than port it.
3. **`ConcatTiledUnalignedProgramFactory`** — **no row in the sheet** (factory added 2026-08-31, #53474). → readiness-sheet owner (Diego) to add the row.
4. **Every row** — the sheet is stale in two primary columns: `Custom hash` = `no` while the code has a `compute_program_hash` (re-added in the same #53474), and `Concept` = `descriptor` on the 3 factories already ported to Metal 2.0. → readiness-sheet owner to reconcile.

Every other gate (Device 2.0, Feature compatibility, Offset base pointers, TensorAccessor 3rd argument) is clear for the code that exists.

**This is not the "#57409 PD batch stale row" case.** #57409 (`f5093e705ae`, "Port More Ops to PD") did not touch concat. The sheet staleness here comes from #53474 (new factory + re-added hash) and #54651 (partial port). It is also not the only RED: three factories carry genuine `Known op issues` blocks. **No waiver is offered.**

**Path forward.** The closest-to-portable factory is **TiledUnaligned**: its only blockers are on the sheet side (missing row, hash column). Once Diego adds a row that reads `Is able to port? = yes`, a re-audit should clear it with no code change — its informational detail is recorded below for that reason. S2SMulti and BlockSharded need the "DFB misuse" resolved first (see Heads-ups for what the misuse looks like in code). S2I should be deleted.

> **Update 2026-11-06:** a follow-up check against `origin/main` (`882c6e0953d`) found that **everything S2SMulti and BlockSharded need already exists in Metal 2.0**. No new Metal 2.0 feature is needed, but the two factories need a redesign rather than a straight conversion. See [What the port needs](#what-the-port-needs-update-2026-11-06).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** RED for all 4 audited factories.

  | Factory | Sheet `Is able to port?` | Blocking column (verbatim) | Route |
  |---|---|---|---|
  | `ConcatS2SMultiProgramFactory` | `no` | `Known op issues`: `DFB misuse; will need semi-manual port` | owner named by the cell (ops / port owner) |
  | `ConcatBlockShardedProgramFactory` | `no` | `Known op issues`: `DFB misuse; will need semi-manual port` | same |
  | `ConcatS2IProgramFactory` | `no` | `Known op issues`: `... this factory is dead code` | ops team — delete |
  | `ConcatTiledUnalignedProgramFactory` | *(no row)* | **missing row** → spreadsheet broken | readiness-sheet owner |
  | `ConcatProgramFactory` / `S2SRM` / `S2STiled` | `yes` | — (already ported; `Concept` should read `MetalV2`) | readiness-sheet owner |

  **Spreadsheet-broken findings** (each rests on code evidence):
  - *Missing row:* `ConcatTiledUnalignedProgramFactory` is in `program_factory_t` (`concat_device_operation.hpp:37`) and selected at `concat_device_operation.cpp:49-53`; the sheet has 6 concat rows and none for it. Factory added in `7db779f57c2` (#53474, 2026-08-31).
  - *`Custom hash` conflict (all rows):* sheet says `no`; code defines `ConcatDeviceOperation::compute_program_hash` (`concat_device_operation.hpp:41`, body `concat_device_operation.cpp:102-136`), mixing in the selected factory index and an input-aliasing signature. It was dropped in `1b99a75e9a6` (#49867, 2026-07-30) and re-added in `7db779f57c2` (#53474). Not a pybound op, so the rename explanation doesn't apply.
  - *`Concept` conflict (3 ported rows):* sheet says `descriptor`; `ConcatProgramFactory`, `ConcatS2SRMProgramFactory`, `ConcatS2STiledProgramFactory` define `create_program_artifacts` returning `ProgramArtifacts` (`concat_program_factory.hpp:14`, `concat_s2s_rm_program_factory.hpp:14`, `concat_s2s_tiled_program_factory.hpp:14`) — i.e. `MetalV2`, ported in `a9be36158de` (#54651).

  **Primary columns that cross-check clean** (all rows): `Runtime-args update (get_dynamic_runtime_args)` = `no` ✓ (no hook in the device-op); `Override runtime args method?` = `no` ✓; `Pybind descriptor` = `no` ✓ (`concat_nanobind.cpp` has no `create_descriptor`); `Smuggled pointer` = `no` ✓ (all address args are `Buffer*` bindings or borrowed CBs); `Concept` = `descriptor` ✓ for S2SMulti / BlockSharded / S2I. No cross-column invariant violated.

- **Device 2.0 (every kernel used):** GREEN for every kernel that exists. All use `Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem`, endpoint structs; no raw `noc_async_*`, no addr-gens, no raw semaphores. `get_tile_size(cb)` in `reader_concat_tiled_unaligned.cpp:33` and `writer_concat_tiled_unaligned.cpp:49-50` is sanctioned. The in-family donor `tt_memmove` (`data_movement/common/kernels/common.hpp:217`) takes a `Noc` and is Device 2.0. **Unassessable:** `ConcatS2IProgramFactory` names `kernels/dataflow/reader_s2i_width.cpp` (`concat_s2i_program_factory.cpp:55-56`), which doesn't exist anywhere in the tree. Moot because the factory is dead; recorded so nobody tries to port it.

- **Feature compatibility:** GREEN — no gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer`, `global_circular_buffer`, `remote_index` anywhere in the op |
  | CBDescriptor `address_offset` (non-zero) | N/A | borrowed CBs set `.buffer` only; no `address_offset` |
  | GlobalSemaphore | N/A | no semaphores of any kind |

- **CB endpoints (GATE-free):** TiledUnaligned — all four CBs are plain 1P+1C FIFOs (detail in Port-work summary). S2SMulti / BlockSharded / S2I — skipped (see scoping note).
- **Offset base pointers:** GREEN. Every address the 4 factories hand a kernel is a clean base: `Buffer*` RTAs `t.buffer()` / `output.buffer()` (`concat_tiled_unaligned_program_factory.cpp:273,280`; `concat_s2i_program_factory.cpp:84`). S2SMulti and BlockSharded pass no addresses — they reach tensor memory through borrowed CBs, and their byte offsets (`input_read_offset`, `src_l1_offset`, `dst_offset`) are separate scalar RTAs added to the CB pointer **in the kernel**, not folded on the host. Concat isn't in the offset-base-pointer triage doc either; scan and doc agree.
- **TensorAccessor 3rd argument:** N/A — no accessor in the 4 factories passes a 3rd argument (`writer_concat_tiled_unaligned.cpp:48`, `writer_s2i_width.cpp:24` are two-arg; `make_tensor_accessor_tuple` in the TiledUnaligned reader builds two-arg accessors). Concat isn't in the 3rd-arg triage doc.

**Scoping of the informational subjects.** This is a whole-op RED with no portable subset, so per the recipe's Red scoping rule the seven informational subjects are judged per factory by *who clears the blocker*:
- **TiledUnaligned — run.** Its only blockers are sheet-side (missing row, stale hash column). They clear without touching the op's code, so today's detail survives to re-audit.
- **S2SMulti, BlockSharded — skipped** (op-code side: resolving "DFB misuse" rewrites the kernels' DFB handling, which is exactly what these subjects would describe; re-audit on unblock).
- **S2I — skipped** (factory to be deleted, not ported).

## What the port needs (update 2026-11-06)

This section was added after the audit. It answers what S2SMulti and BlockSharded need at run time, and whether Metal 2.0 provides it today. It was checked against `origin/main` at `882c6e0953d`. This checkout's `HEAD` (`3d23c7655d7`, 2026-10-27) is too old: it lacks the array arguments from #63805. **Port from current main.**

The concat sharded factories and kernels have not changed on main since the audit (only include/rounding cleanups in #60814 / #60730), so the findings above still hold.

### What the kernels need at run time

| Factory | Needs | Where |
|---|---|---|
| S2SMulti | Pick input buffer *i* inside a loop over inputs. The input count is already a compile-time arg (CTA 3). | `reader_s2s_tensor_concat.cpp:35,45` |
| S2SMulti | 5 numbers per input (pages per stick, sticks, write offset, read offset, block stride); the list length depends on the input count. | `:39-43` |
| BlockSharded | Pick input buffer *i* from a **runtime** arg; each core gets a different transfer list. | `reader_writer_block_sharded_concat.cpp:134,142` |
| BlockSharded | 1 count + 9 numbers per transfer, with counts that differ per core and per RISC. Capped by the 256-RTA budget. | `:129-140`; budget at `concat_block_sharded_program_factory.cpp:394-397` |
| BlockSharded | The L1 address of an input's shard **on another core**. Today it uses its own core's DFB read pointer, which only works if every core places the buffer at the same L1 address. | `:143,153` |
| Both | The output shard's local L1 address. | output DFB `get_write_ptr()` |

### What Metal 2.0 provides on main

| Feature | Status | Covers |
|---|---|---|
| **Tensor binding sequence + runtime-indexed accessor array.** `make_tensor_accessors(tensor::inputs)` then `make_abstract_tensor_accessor_wrappers(...)` gives a `std::array` of type-erased accessors, indexed as `accessor_array[i]` with a runtime `i`. Each one has `get_noc_addr(page_idx, offset, noc)`, which also resolves pages that live on other cores. | On main (`advanced_options.hpp:121-158`, `tensor_accessor.h:795-890`). Already present at the audit date; the audit under-weighted it. | Picking input *i* at run time, and remote shard addresses |
| **Array arguments** (compile-time, runtime, common runtime), read in-kernel as `get_arg(args::name)[i]` | On main since #63805 (2026-11-03) | Per-input / per-transfer number lists |
| RTA varargs | On main (already used by the ported `ConcatProgramFactory`) | Same, with variable per-core length |
| Borrowed DFBs (`DataflowBufferSpec::borrowed_from`) | On main (used by the ported S2SRM / S2STiled) | The output buffer |
| **A runtime-indexable array or sequence of DFBs** | **Not found** on main, and no open or merged PR found (searched 2026-11-06). Borrowed DFBs only have fixed names (`dfb::input_0`, ...). | Would allow a near-literal port; **not required** |

A runtime-indexable DFB/tensor-binding array was reported to be "already in". The tensor half is; the DFB half could not be found. If someone has a PR or file for DFB arrays, check whether it lets the kernels keep their current shape.

### Recommended design

- **Inputs:** stop binding inputs as DFBs. Declare one `TensorParameter` per input plus a `TensorBindingSequence` named `inputs`, as `ConcatProgramFactory` already does (`concat_program_factory.cpp:49-54,227-250`). In the kernel, build the accessor array once and index it with the input number.
- **Output:** keep a single borrowed output DFB; that already works.
- **Look up an address once per region, then add strides.** A shard is contiguous on its core, so call `get_noc_addr` once per input (S2SMulti) or once per transfer (BlockSharded), then keep the existing pointer arithmetic for rows and pages. This keeps the type-erased call off the per-page hot path.
- **S2SMulti:** each core needs the page id of its first page in each input's shard, which the host must add. Today every core receives **identical** RTAs (`concat_s2s_multi_program_factory.cpp:279-305`); this makes them per-core.
- **BlockSharded:** each transfer becomes "input *i*, first page id, offset, strides, size, rows" instead of "core x/y, DFB id, L1 offset, ...". The host-side transfer builder (`concat_block_sharded_program_factory.cpp:278-380`) changes too. This also removes the same-L1-address-on-every-core assumption.
- **`concat_reader::with_input`** (`concat_reader_common.hpp:20-23`) is an alternative to the accessor array (a compile-time switch over the input slots). It avoids the type-erased call but grows the code with the input count. Use it only if the accessor array measures too slow.
- **S2I:** delete; don't port.

### Open risks (not yet checked)

1. **Speed.** These are small, latency-sensitive copies; one kernel comment reports a few percent lost to a single extra RTA (`reader_s2s_tensor_concat.cpp:24-25`). Measure both factories against the current versions before landing.
2. **Remote shard pages in BlockSharded.** Confirm that `get_noc_addr` on a block-sharded L1 tensor gives the right remote address for the page ids the transfer builder will produce, including partial shards at grid edges.
3. **Runtime array arguments have a fixed declared size** (`kernel_spec.hpp`, `ArrayArgDecl::size`). BlockSharded's transfer count differs per core and per RISC, so either size the array to the maximum and pass a count, or keep RTA varargs.
4. **RTA budget.** Re-derive BlockSharded's `max_transfers_per_risc` for the new per-transfer arg count.

### Also on main, not covered by this audit

#62063 (2026-10-28) added a second device operation, `ConcatCodegenDeviceOperation` (`concat/codegen/`). It's a `descriptor`-style factory with its own `compute_program_hash`, and handles interleaved row-major concats of 2-8 inputs. It is **not** Metal 2.0 and was not audited. A full concat port needs a separate audit of it, and probably a readiness-sheet row.

## Port-work summary  *(TiledUnaligned only — for the re-audit / future brief)*

- **Tensor bindings:**
  - `inputs` (N tensors) — **Case 1**. Delivered as `Buffer*` RTAs 2..2+N-1 (`concat_tiled_unaligned_program_factory.cpp:272-274`), fed into `make_tensor_accessor_tuple(tensor_accessor_args, addr_base_idx)` (`reader_concat_tiled_unaligned.cpp:24-25`). → a tensor-binding sequence, as the already-ported `ConcatProgramFactory` does with `tensor::inputs` (`reader_concat_interleaved_start_id.cpp:53`).
  - `output` — **Case 1**. `Buffer*` RTA 0 (`concat_tiled_unaligned_program_factory.cpp:280`) → `TensorAccessor(dst_args, dst_addr)` (`writer_concat_tiled_unaligned.cpp:48`).
- **TensorParameter relaxation:** no sheet value (missing row). The sibling rows all read `none`; don't assume that — read the new row once it exists.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal, no flags:
  - `c_0` `cb_in` — reader FIFO-produces (`reader_concat_tiled_unaligned.cpp:42,54`), compute consumes via `untilize` → 1P+1C.
  - `c_1` `cb_rm` — compute produces via `untilize`, writer FIFO-consumes (`writer_concat_tiled_unaligned.cpp:62,109`, plus a `get_read_ptr` peek on its own binding) → 1P+1C.
  - `c_2` `cb_asm` — writer FIFO-produces (`:63,108`, plus a `get_write_ptr` peek), compute consumes via `tilize` → 1P+1C.
  - `c_16` `cb_out` — compute produces via `tilize`, writer FIFO-consumes (`:113,116`) → 1P+1C.

## Heads-ups  *(TiledUnaligned only)*

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none borrowed — all three kernels live in `concat/device/kernels/` and only this factory uses them. No `_metal2` fork question.
- **RTA varargs:**
  - Reader: the `wt[]` block — N values read in a loop over `num_tensors` via `get_arg_addr(addr_base_idx + num_tensors)` (`reader_concat_tiled_unaligned.cpp:27-31`) → RTA vararg. Its N address args become the tensor-binding sequence, not varargs.
  - Writer: `wt[]` and `w_bytes[]` — two N-long blocks read in one loop off `get_arg_addr(3)` (`writer_concat_tiled_unaligned.cpp:40-46`) → RTA varargs. `dst_addr`, `num_bands`, `band_start` (`:36-38`) are fixed fields → name them.
  - Reader CTAs: `make_tensor_accessor_args_tuple<num_tensors, 2>()` (`reader_concat_tiled_unaligned.cpp:18`) is the per-tensor accessor-args block; it goes away with the tensor-binding sequence rather than becoming CTA varargs.
- **Other:** the writer reads raw `uint32_t` arg pointers (`get_arg_addr`) rather than `get_arg_val`; the port's vararg accessor replaces both loops. Factory is gated off on Quasar at host (`concat_tiled_unaligned_program_factory.cpp:80-82`) — irrelevant to a Gen1 port but don't remove it.

## Team-only

- **Out-of-directory coupling & donor shape (TiledUnaligned):** roll-up **✓ clean**.

  | Op kernel | Donor | Class | Functions / shape | Status |
  |---|---|---|---|---|
  | `writer_concat_tiled_unaligned.cpp` | `data_movement/common/kernels/common.hpp` | in-family shared | `tt_memmove<…>(Noc, uint32_t dst, uint32_t src, uint32_t bytes)` — raw L1 addresses, no resource handles | ✓ |
  | `concat_untilize_retilize.cpp` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`, `tilize_helpers.hpp` | shared kernel lib | `untilize<…, cb_in, cb_rm>`, `tilize<…, cb_asm, cb_out, …>` — `uint32_t` CB ids as template params | ✓ |
  | all | `tt_metal/hw/inc/api/*` | LLK / HAL | — | ✓ |

  Borrowed kernel files: none.
- **Relaxation candidates (FALLIBLE):** the custom hash mixes in the selected factory index and an input-aliasing signature (`concat_device_operation.cpp:111-135`). Neither is a tensor property, so it suggests no relaxation. It does document that factory choice depends on live L1 occupancy (`can_use_tiled_unaligned_concat`, `concat_tiled_unaligned_program_factory.cpp:137`). The port leaves the hash as is.
- **TTNN factory analysis:** concept `descriptor` (4 audited factories); no op-owned tensors; no MeshWorkload; no pybound `create_descriptor`; custom hash present (`concat_device_operation.cpp:102`, not a gate); no `get_dynamic_runtime_args`; no `override_runtime_arguments` → target `ProgramSpecFactoryConcept` for all four.
- **What "DFB misuse" looks like in code** (S2SMulti, BlockSharded, also S2I's writer). Recorded for whoever clears the `Known op issues` block; not a full census:
  - `reader_s2s_tensor_concat.cpp:45` — `DataflowBuffer input_dfb(input_id)`, the DFB id taken from a **loop variable** over `num_input_tensors`.
  - `reader_writer_block_sharded_concat.cpp:44` — `DataflowBuffer src_dfb(src_dfb_id)`, the id taken from a **runtime arg**. `src_dfb.get_read_ptr()` (local) is then used as the address on a **remote** core (`:45,55`); that relies on every core placing the CB at the same L1 address.
  - `writer_s2i_width.cpp:28-29` — `DataflowBuffer input_shard_dfb(get_arg_val(...))`, id from a runtime arg.

  Metal 2.0 binds DFBs through compile-time `dfb::name` tokens, so a DFB chosen by runtime index has no direct translation. That's consistent with the sheet's "semi-manual port".

## Misc anomalies  *(team-only, non-gating)*

- **S2I is dead *and* broken.** `concat_s2i_program_factory.cpp:55-56` names `kernels/dataflow/reader_s2i_width.cpp`, which doesn't exist (only `writer_s2i_width.cpp` does). `select_program_factory` returns S2I for L1-sharded input + interleaved output (`concat_device_operation.cpp:57-59`), but `concat_impl` never sends that combination to `ttnn::prim::concat`: it either stages through a temporary sharded output (`concat_device_operation.cpp:549-562`) or unshards first. No other caller of `ttnn::prim::concat` exists in `ttnn/`. A direct `ttnn.prim.concat` call with that combination would fail at kernel compile. Recommend deleting the factory and `writer_s2i_width.cpp`.
- `reader_writer_block_sharded_concat.cpp:19-21` comment refers to `UpdateDynamicCircularBufferAddress`, a legacy imperative API the descriptor factory no longer uses (stale comment).
- `calculate_max_tensors_per_concat` (`concat_device_operation.cpp`, the `// Args breakdown` comment) describes the pre-port positional-arg layout of the interleaved factory (`src_addr[N]`, `TensorAccessorArgs[N]`), which no longer matches the ported `ConcatProgramFactory`. The 47 limit may now be conservative or wrong; worth re-measuring.

## Questions for the user

1. **Who owns the "DFB misuse" fix?** The `Known op issues` cell for S2SMulti / BlockSharded says "will need semi-manual port" but names no owner. Is that ops-team work before the port, or a hand-port the Metal 2.0 team will do? The answer decides whether those two factories ever come back through this audit. *(2026-11-06: no Metal 2.0 feature is missing — see [What the port needs](#what-the-port-needs-update-2026-11-06). The question is now only who does the hand rewrite.)*
2. **Do runtime-indexable DFB arrays exist?** *(2026-11-06)* Reported as "already in", but not found on main or in PRs. If they exist, check whether they allow a closer-to-literal port than the tensor-sequence redesign.

## Recipe notes

- **Partially-ported ops aren't anticipated.** The recipe treats `Concept == MetalV2` as one row's "already done" outcome, but here 3 of 7 factories are ported and the sheet still reads `descriptor` for them. I treated the 4 unported factories as the audit scope and the `Concept` mismatch on the 3 ported rows as a spreadsheet-broken finding. A sentence on scoping a partially-ported op would help.
- **The Red scoping rule is per-op, but blockers here differ per factory.** One factory's only blocker is sheet-side (→ "run"), others are op-code side (→ "skip"). I applied the rule per factory. The recipe could say explicitly that this is allowed.
- **Missing row → no relaxation value.** The TensorParameter relaxations subject assumes a cell exists. For a missing-row factory there's nothing to read; I recorded "no value" and didn't infer from sibling rows.
