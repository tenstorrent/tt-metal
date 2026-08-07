# Port Plan — `data_movement/bcast`

Port plan for `ttnn/cpp/ttnn/operations/data_movement/bcast`, ported from the legacy
`ProgramDescriptorFactoryConcept` (`create_descriptor` → `ProgramDescriptor`) to Metal 2.0
(`MetalV2FactoryConcept` → `create_program_artifacts` → `ProgramArtifacts`).

Written during the inventory and planning steps; committed alongside the port for review.

## Scope of this pass

`BcastDeviceOperation` has **five** program factories. **This second pass ports the fifth and final one, `BcastMultiCoreHWProgramFactory` (HW).** The other four were ported in the first pass:

- `BcastMultiCoreHProgramFactory` (H, interleaved) — **PORTED** (pass 1)
- `BcastMultiCoreWProgramFactory` (W, interleaved) — **PORTED** (pass 1)
- `BcastShardedHProgramFactory` (H, sharded) — **PORTED** (pass 1)
- `BcastShardedHOptimisedProgramFactory` (H, sharded, optimised) — **PORTED** (pass 1)
- **`BcastMultiCoreHWProgramFactory` (HW)** — **PORTED (this pass)** — see the HW-specific sections below.

**Why HW was deferred in pass 1 and is portable now.** HW binds the cross-family donor writer
`eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`. In pass 1 the only Metal 2.0
fork of that writer lived in the out-of-bounds `experimental/quasar/` tree, so porting HW would have meant
creating the fork itself — a cross-family write the older recipe did not sanction, hence the defer. Since
then the recipe's `Caution: Porting a shared kernel` rungs cover this, and a **real `_metal2` fork now
exists beside the original** (`writer_unary_interleaved_start_id_metal2.cpp`, committed by #51771). So HW's
two shared kernels are handled cleanly:
- **Donor writer** → **rung 1: reuse** the existing `writer_unary_interleaved_start_id_metal2.cpp` fork (read-only; already consumed by `copy/typecast` + `gelu_backward`).
- **Compute `bcast_hw.cpp`** → **rung 2: create** `bcast_hw_metal2.cpp` beside the original — `bcast_hw.cpp` is *lent* to `experimental/transformer/rotate_half` (legacy device-op concept), which keeps binding the legacy original (sunset list).

With HW ported, **all five factories are now on `MetalV2FactoryConcept`.** The `program_factory_t` variant
is unchanged in shape; all five alternatives now satisfy `MetalV2FactoryConcept`.

`BcastShardedHOptimised` (pass 1) initially hung on `in1_batch_size == 2` (`batch_b > 1`) / wide-shard
configs — a latent kernel buffer over-run surfaced then. It was root-caused and fixed on `main` by
**PR #51056** (`e09c6aea658`); the branch is rebased onto that fix. See `METAL2_PORT_REPORT.md`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — each factory has `create_descriptor(...) → tt::tt_metal::ProgramDescriptor`.
- Variants: `select_program_factory` picks one of the five factory structs (a variant, not a per-variant switch inside one factory).
- Custom `compute_program_hash`: **none** — `BcastDeviceOperation` uses the default reflection-based hash (confirmed in `bcast_device_operation.cpp/.hpp`). No deletion needed.
- Pybind: plain `ttnn::bind_function<"bcast">` (`bcast_nanobind.cpp:57`); **no** `create_descriptor`/`create_program_descriptor` pybind hook — nothing to remove.

*(Target concept `MetalV2FactoryConcept` chosen during the audit — carried forward below.)*

### Kernels (per ported factory)

**BcastMultiCoreH** (`bcast_multi_core_h_program_factory.cpp`):

| unique_id | source | core_ranges | CTAs (positional) | RTAs (kernel-read indices) | defines | config |
|---|---|---|---|---|---|---|
| reader | `reader_bcast_h_interleaved_input_rows_partitioned.cpp` | all_device_cores | `TensorAccessorArgs(src0)`, `TensorAccessorArgs(src1)` | 0=src0_addr, 3=src0_num_tiles, 4=src1_addr, 8=NCHtWt, 9=NC, 10=Ht, 11=Wt, 12=nc1, 13=start_id, 14=HtWt (idx 1,2,5,6,7 dead 0u) | — | ReaderConfigDescriptor{} |
| writer | `writer_unary_interleaved_input_cols_batched.cpp` | all_device_cores | `TensorAccessorArgs(dst)` | 0=dst_addr, 3=Ht, 4=Wt, 5=Wt_read, 6=Wt_skip, 7=NC, 8=HtWt (idx 1,2 dead 0u) | — | WriterConfigDescriptor{} |
| compute | `compute/bcast_h.cpp` | all_device_cores | none | 0=B, 1=Ht, 2=Wt | `bcast_op_utils::get_defines(H, math_op)` (BCAST_LLKOP/BCAST_DIM/BCAST_OP + math) | ComputeConfigDescriptor{} |

**BcastMultiCoreW** (`bcast_multi_core_w_program_factory.cpp`):

| unique_id | source | core_ranges | CTAs | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `reader_bcast_w_interleaved_input_cols_partitioned.cpp` | all_device_cores | `TensorAccessorArgs(src0)`, `TensorAccessorArgs(src1)` | 0=src0_addr, 3=src0_num_tiles, 4=src1_addr, 8=NCHtWt, 9=NC, 10=Ht, 11=Wt, 12=nc1, 13=start_id, 14=HtWt, 15=Wt_skip (idx 1,2,5,6,7 dead) | — | ReaderConfigDescriptor{} |
| writer | `writer_unary_interleaved_input_cols_batched.cpp` | all_device_cores | `TensorAccessorArgs(dst)` | 0=dst_addr, 3=Ht, 4=Wt, 5=Wt_read, 6=Wt_skip, 7=NC, 8=HtWt (idx 1,2 dead) | — | WriterConfigDescriptor{} |
| compute | `compute/bcast_w.cpp` | all_device_cores | none | 0=B, 1=Ht, 2=Wt | `get_defines(W, math_op)` | ComputeConfigDescriptor{} |

**BcastShardedH** (`bcast_sharded_h_program_factory.cpp`) — **no writer kernel**:

| unique_id | source | core_ranges | CTAs | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `reader_bcast_h_sharded.cpp` | all_cores (=shard grid) | `[0]=src0_cb_index (magic CB idx)`, `TensorAccessorArgs(src1)` | 0=src1_addr, 1=Ht, 2=Wt, 3=offset, 4=NC, 5=batch_offset | — | ReaderConfigDescriptor{} |
| compute | `compute/bcast_h.cpp` | all_cores | none | 0=B(NC), 1=Ht, 2=Wt | `get_defines(H, math_op)` | ComputeConfigDescriptor{} |

Dead writer CTA `{dst_is_dram}` is built then `(void)`-discarded — **not carried** (no writer kernel).

**BcastShardedHOptimised** (`bcast_sharded_h_optimised_program_factory.cpp`) — **no writer kernel**:

| unique_id | source | core_ranges | CTAs | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `reader_bcast_h_sharded_optimised.cpp` | all_cores | `[0]=src0_cb_index (magic CB idx)`, `TensorAccessorArgs(src1)` | 0=src1_addr, 1=Ht, 2=Wt, 3=offset, 4=batch_offset, 5=w_blk, 6=batch_b | — | ReaderConfigDescriptor{} |
| compute | `compute/bcast_h_sharded_optimised.cpp` | all_cores | none | 0=NC, 1=Ht, 2=Wt, 3=h_blk, 4=batch_b, 5=Ht_per_batch_b | `get_defines(H, math_op)` | ComputeConfigDescriptor{} |

Dead writer CTA `{dst_is_dram}` built + `(void)`-discarded — **not carried**.

### CBs (per ported factory)

Three CBs everywhere: `c_0` (src0/input_a), `c_1` (src1/input_b), `c_16` (output).

| factory | CB | index | entry_size | num_entries | data_format | borrowed? |
|---|---|---|---|---|---|---|
| H / W | c_0 | 0 | `src0_single_tile_size` | 2 | src0 df | no |
| H / W | c_1 | 1 | `src1_single_tile_size` | 2 | src1 df | no |
| H / W | c_16 | 16 | `dst_single_tile_size` | 2 | dst df | no |
| ShardedH | c_0 | 0 | `aligned_input_tile_nbytes` | `num_tile_per_core` | act df | **yes → src0** |
| ShardedH | c_1 | 1 | `input1_tile_size` | `num_input_tiles` | b df | no |
| ShardedH | c_16 | 16 | `aligned_input_tile_nbytes` | `num_tile_per_core` | out df | **yes → dst** |
| ShardedHOptimised | c_0 | 0 | `aligned_input_tile_nbytes` | `num_tile_per_core` | act df | **yes → src0** |
| ShardedHOptimised | c_1 | 1 | `input1_tile_size` | `num_input_tiles (=w_blk)` | b df | no |
| ShardedHOptimised | c_16 | 16 | `aligned_input_tile_nbytes` | `num_tile_per_core` | out df | **yes → dst** |

No `tile_format_metadata` set on any legacy CB (all default 32×32) → leave `tile_format_metadata` unset.
No GlobalCircularBuffer, no aliased CBs (single-element `format_descriptors` everywhere), no `address_offset`.

### Semaphores
none.

### Tensor accessors

| factory | host site | originating Tensor | kernel accessor |
|---|---|---|---|
| H / W | reader `TensorAccessorArgs(*src0_buffer)` | input_a | `TensorAccessor(tensor::src0)` |
| H / W | reader `TensorAccessorArgs(*src1_buffer)` | input_b | `TensorAccessor(tensor::src1)` |
| H / W | writer `TensorAccessorArgs(*dst_buffer)` | output | `TensorAccessor(tensor::dst)` |
| ShardedH / Opt | reader `TensorAccessorArgs(*src1_buffer)` | input_b | `TensorAccessor(tensor::src1)` |

Sharded input_a / output are **not** read via `TensorAccessor` — they are resident, backing borrowed DFBs (`c_0`/`c_16`).
All accessors are the 2-arg form (`TensorAccessor(args, addr)`); no 3rd page-size argument anywhere.
Addresses arrive via the `Buffer*`-binding form (`src0_buffer`, `b.buffer()`, …) → all **Case 1** (or clean borrowed). No `->address()` anywhere; no Case 2.

### Work split
- H: `split_work_to_cores(grid, Ht)` → group1 `Ht_per_core_group_1`, group2 `Ht_per_core_group_2`. **Per-group value carried as an RTA** (`Ht`), not a CTA — legacy already has **one** compute `KernelDescriptor`. No CTA multiplicity.
- W: `split_work_to_cores(grid, Wt)` → per-group `Wt` carried as RTA. One compute descriptor.
- Sharded: one work group over the shard grid (`all_cores`), per-core `offset` computed in a loop.
- **All three kernels are placed on `all_device_cores`** in H/W (idle cores get all-zero RTAs and no-op); sharded places on the shard grid only.

### Cross-op kernels
- **HW factory only** (deferred): `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`. Not touched this pass.
- The four ported factories use **only bcast-owned kernels** (verified: `writer_unary_interleaved_input_cols_batched.cpp` is used only by H+W; `bcast_h.cpp` only by H+ShardedH — both consumer sets fully inside this pass).

### Flags
- Unreferenced kernel files in the op dir (not audited, not touched): `reader_bcast_h_interleaved.cpp`, `reader_bcast_hw_interleaved.cpp`, `reader_bcast_scalar_interleaved_partitioned.cpp`, `reader_bcast_w_interleaved.cpp`.
- Dead kernel-side reads (`num_tiles = src0_num_tiles`, `NCHtWt`) in the H/W readers: kept faithfully as named args (kernel-logic cleanup is out of scope → report).

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `MetalV2FactoryConcept` (all four ported factories).
- **Custom `compute_program_hash`**: none — already default. No deletion.
- **Implementation notes**: shared kernel sources within the port set (`bcast_h.cpp` across H+ShardedH; `writer_unary_interleaved_input_cols_batched.cpp` across H+W) are ported **in place** — every consumer co-migrates this pass. Resource-name constants declared as function-local `const`s (avoids unity-build anon-namespace symbol collisions, per the catalog's unity-build hygiene pattern).

## Planned Spec Shape

Shared accessor-name convention (kernel-side handles) across the op:
`dfb::in0`=c_0, `dfb::in1`=c_1, `dfb::out`=c_16; `tensor::src0`=input_a, `tensor::src1`=input_b, `tensor::dst`=output.
Resource unique_ids (function-local): `IN0/IN1/OUT`, `INPUT_A/INPUT_B/OUTPUT`, `READER/WRITER/COMPUTE`.

**H / W** (one WorkUnitSpec over the full grid):
- KernelSpecs: `reader`, `writer`, `compute` (1 each; per-group value is an RTA, no multiplicity).
- DataflowBufferSpecs: `IN0`, `IN1`, `OUT` (plain, non-borrowed).
- TensorParameters: `INPUT_A`, `INPUT_B`, `OUTPUT`.
- WorkUnitSpecs: 1 (`{reader,writer,compute}` on `all_device_cores`).
- DFB roles: `IN0` reader-PRODUCER / compute-CONSUMER; `IN1` reader-PRODUCER / compute-CONSUMER; `OUT` compute-PRODUCER / writer-CONSUMER.

**ShardedH / ShardedHOptimised** (one WorkUnitSpec over the shard grid):
- KernelSpecs: `reader`, `compute` (no writer).
- DataflowBufferSpecs: `IN0` (borrowed_from `INPUT_A`), `IN1` (plain), `OUT` (borrowed_from `OUTPUT`).
- TensorParameters: `INPUT_A` (backs `IN0` via borrowed_from — no TensorBinding needed, per pad reference), `INPUT_B` (bound on reader via `tensor::src1`), `OUTPUT` (backs `OUT` via borrowed_from).
- WorkUnitSpecs: 1 (`{reader,compute}` on `all_cores`).
- DFB roles: `IN0` reader-PRODUCER / compute-CONSUMER (1P+1C); `IN1` reader-PRODUCER / compute-CONSUMER; `OUT` compute **self-loop** (PRODUCER+CONSUMER — resident output, nothing drains).

## Preserved Multiplicity

none — no work-split multiplicity in legacy. Each factory has exactly one `KernelDescriptor` per kernel; the per-core work-split value is passed as a **runtime arg** already (not a per-group CTA), so there is no multi-`KernelSpec` fan-out to preserve.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| H/W reader CTA `TensorAccessorArgs(src0)` + RTA idx 0 (`src0_addr`) | address RTA + accessor-args plumbing | `TensorParameter INPUT_A` + `TensorBinding`→`tensor::src0` |
| H/W reader CTA `TensorAccessorArgs(src1)` + RTA idx 4 (`src1_addr`) | address RTA + accessor-args | `TensorParameter INPUT_B` + `TensorBinding`→`tensor::src1` |
| H/W writer CTA `TensorAccessorArgs(dst)` + RTA idx 0 (`dst_addr`) | address RTA + accessor-args | `TensorParameter OUTPUT` + `TensorBinding`→`tensor::dst` |
| H/W reader RTA idx 1,2,5,6,7 | dead `0u` (never read by kernel) | dropped (route dead-arg cleanup to ops team) |
| H/W writer RTA idx 1,2 | dead `0u` | dropped |
| Sharded reader CTA `[0]=src0_cb_index` | magic CB index | `DFBBinding IN0`→`dfb::in0` |
| Sharded reader CTA `TensorAccessorArgs(src1)` + RTA idx 0 | address RTA + accessor-args | `TensorParameter INPUT_B` + `TensorBinding`→`tensor::src1` |
| Sharded dead writer CTA `{dst_is_dram}` | built then `(void)`-discarded | dropped (no writer kernel) |
| all kernels: positional CTAs | positional | named CTAs (there are none left after accessor-args drop; readers/writers keep only named RTAs) |
| all readers/writers: `get_tile_size(cb_id)` | free-fn on magic id | `dfb.get_tile_size()` (DFB object getter, whitelist §A) |

No semaphore-ID RTAs (op has no semaphores). No page-size 3rd-arg CTAs/RTAs.

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../shared/port_patterns.md): `OUT` (`c_16`) on ShardedH & ShardedHOptimised compute (PRODUCER+CONSUMER; resident borrowed output, nothing drains).
- Borrowed-memory DFBs (migration_guide `DataflowBufferSpec`): `IN0`←`INPUT_A`, `OUT`←`OUTPUT` in the sharded factories. borrowed_from reference satisfies the TensorParameter-binding validator rule (verified against `experimental/quasar/pad` sharded factory).
- [Pass DFB handles directly to LLKs](../shared/port_patterns.md): compute kernels pass `dfb::in0/in1/out` into `init_bcast` / `BCAST_OP` / `pack_tile`.
- Two-toucher 1P+1C (implicit): `IN0` in sharded is reader-PRODUCER + compute-CONSUMER — an ordinary 1:1, **not** self-loop and **not** multi-binding.

## Deferred / Flagged

- **HW factory deferred** (cross-op donor writer, see Scope above). No new structural surprises vs. the audit for the four ported factories.
- Dead host RTAs and dead kernel-side reads noted in the audit's Misc anomalies are **not** carried where the kernel never reads them (host idx 1,2,5,6,7 / writer 1,2); kernel-side dead reads (`num_tiles`, `NCHtWt`) are kept faithfully as named args (cleanup is out of scope). All routed to the port report.

---

# HW factory (pass 2) — `BcastMultiCoreHWProgramFactory`

Ported from `create_descriptor` → `ProgramDescriptor` to `create_program_artifacts` → `ProgramArtifacts`.
Unlike the H/W/Sharded factories, **HW handles both interleaved and HEIGHT-sharded configs in one factory**
(`validate` forces in0 and output to the same layout — so sharding is all-or-nothing, never mixed).

## Legacy Inventory (HW)

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor(...) → tt::tt_metal::ProgramDescriptor`.
- Custom `compute_program_hash`: none (device-op uses default hash). Pybind: plain `bind_function<"bcast">`. No device-op-class edits forced.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs (kernel-read idx) | defines | config / opt_level |
|---|---|---|---|---|---|---|
| reader | `reader_bcast_hw_interleaved_partitioned.cpp` (bcast-owned) | all_device_cores | `TensorAccessorArgs(src0)` (unless IN0_SHARDED), `TensorAccessorArgs(src1)` | 0=src0_addr, 1=src1_addr, 2=num_tiles, 3=HtWt, 4=base_start_id_HtWt, 5=curr_id_from_base, 6=bcast_id | `BCAST_SCALAR` (if bnc1), `IN0_SHARDED` (if src0 sharded) | ReaderConfigDescriptor{} / O2 |
| writer | `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` (**borrowed**) | all_device_cores | `[0]=cb_id_out`, `TensorAccessorArgs<1>(dst)` | 0=dst_addr, 1=num_pages, 2=start_id | `OUT_SHARDED` (if output sharded) | WriterConfigDescriptor{} / O2 |
| compute | `compute/bcast_hw.cpp` (**lent** → rotate_half) | all_device_cores | none | 0=B, 1=Ht, 2=Wt | `get_defines(HW, math_op)` + `BCAST_SCALAR` (if bnc1) | ComputeConfigDescriptor{} / **O3** (compute default) |

### CBs
Three CBs: `c_0` (src0/input_a), `c_1` (src1/input_b), `c_16` (output). Page size = `tt::tile_size` per format (the HW factory uses `src0/1/dst_single_tile_size` directly — **not** the `round_up_to_mul32` the ShardedH factory used; preserved verbatim).

| CB | index | entry_size | num_entries | data_format | borrowed? |
|---|---|---|---|---|---|
| c_0 | 0 | `src0_single_tile_size` | `src0_sharded ? num_tiles_per_shard : 2` | src0 df | **yes → INPUT_A** iff src0 sharded |
| c_1 | 1 | `src1_single_tile_size` | 2 | src1 df | no |
| c_16 | 16 | `dst_single_tile_size` | `output_sharded ? num_tiles_per_shard : 2` | dst df | **yes → OUTPUT** iff output sharded |

No `tile_format_metadata`, no GlobalCircularBuffer, no aliased CBs, no `address_offset`.

### Semaphores
none.

### Tensor accessors
| host site | originating Tensor | kernel accessor | notes |
|---|---|---|---|
| reader `TensorAccessorArgs(*src0_buffer)` | input_a | `TensorAccessor(tensor::src0)` | **conditional** — only when `!IN0_SHARDED` (sharded src0 is resident, backs borrowed c_0) |
| reader `TensorAccessorArgs(*src1_buffer)` | input_b | `TensorAccessor(tensor::src1)` | always (src1 always interleaved) |
| writer `TensorAccessorArgs(*dst_buffer)` | output | `TensorAccessor(tensor::dst)` | **conditional** — only when `!OUT_SHARDED` (sharded output resident, backs borrowed c_16) |

All 2-arg accessors (no 3rd page-size arg). Addresses arrive via `Buffer*`-binding form → all **Case 1** (or clean borrowed). No `->address()`; no Case 2.

### Work split
- `split_work_to_cores(grid, num_tensor_tiles = NC*Ht*Wt)` → per-core tile count carried as RTA `num_tiles` (one compute descriptor, no CTA multiplicity).
- If sharded: override `num_tiles_per_core_group_1 = num_tiles_per_shard`, `core_group_1 = all_cores = shard grid`.
- Legacy per-core loop: `for i in [0,num_cores_total): core = {i/num_cores_y, i%num_cores_y}`; idle cores get all-zero (reader 7, compute {1,1,0}, writer 3).

### Shared kernels (HW)
| kernel | class | `_metal2` fork | rung | remaining consumers (sunset) |
|---|---|---|---|---|
| `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | borrowed (cross-family) | **exists** (`..._metal2.cpp`, #51771) | **1 — reuse** (read-only) | ~31 other legacy binders of the *original* |
| `compute/bcast_hw.cpp` | lent (bcast-owned, bound by rotate_half) | **none** | **2 — create** `bcast_hw_metal2.cpp` | `experimental/transformer/rotate_half` (legacy device-op) |
| `reader_bcast_hw_interleaved_partitioned.cpp` | bcast-only (single binder) | n/a | convert in place | — |

**Reused-fork binding vocabulary** (`writer_unary_interleaved_start_id_metal2.cpp` — now the constraint on HW's writer `KernelSpec`): DFB `dfb::out` (CONSUMER), tensor `tensor::dst`, named args `num_pages` / `start_id`, defines `OUT_SHARDED` / `BACKWARDS` (BACKWARDS off for bcast), page size read via `dfb.get_entry_size()`.

### Flags
- Sharded HW is a **supported, reachable** config (validate: "HW bcast in0 supports Height Sharding or Interleaving"; in0 & out layouts must match). Primary CI coverage (C++ `test_bcast_op`, `test_binary_bcast.py -k test_bcast`) exercises **interleaved** HW; sharded-HW coverage is uncertain — flagged in the report.

## TTNN ProgramFactory (HW)
- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: single factory spans interleaved + HEIGHT-sharded. DFB `borrowed_from` and the two conditional tensor bindings (`tensor::src0`, `tensor::dst`) are toggled by `src0_sharded` / `output_sharded`; the kernels already `#ifdef`-gate the matching `IN0_SHARDED` / `OUT_SHARDED` paths, so this is the [conditional-binding pattern](../shared/port_patterns.md). `WorkUnitSpec::target_nodes` = `all_device_cores` (interleaved, idle cores zero-filled as legacy) or the **shard grid** (sharded — required for borrowed-DFB backing to resolve per shard core; matches the ShardedH sibling, and is behavior-preserving since legacy's non-shard cores were idle no-ops).

## Planned Spec Shape (HW)

Accessor-name convention (shared with the ported siblings): `dfb::in0`=c_0, `dfb::in1`=c_1, `dfb::out`=c_16; `tensor::src0`=input_a, `tensor::src1`=input_b, `tensor::dst`=output. unique_ids `IN0/IN1/OUT`, `INPUT_A/INPUT_B/OUTPUT`, `READER/WRITER/COMPUTE`.

- KernelSpecs: `reader`, `writer` (metal2 fork), `compute` (metal2 fork) — 1 each.
- DataflowBufferSpecs: `IN0` (borrowed_from INPUT_A iff src0 sharded), `IN1` (plain), `OUT` (borrowed_from OUTPUT iff output sharded).
- TensorParameters: `INPUT_A`, `INPUT_B`, `OUTPUT`.
- WorkUnitSpecs: 1 (`{reader,writer,compute}` on `target_nodes`).
- DFB roles (identical across configs): `IN0` reader-PRODUCER / compute-CONSUMER; `IN1` reader-PRODUCER / compute-CONSUMER; `OUT` compute-PRODUCER / writer-CONSUMER (**not** a self-loop — HW always has the writer, so `c_16` is a genuine 2-toucher 1P+1C even when borrowed/sharded).

## Preserved Multiplicity (HW)
none — one `KernelDescriptor` per kernel; per-core value is an RTA (`num_tiles`), not a per-group CTA.

## Dropped Plumbing (HW)
| legacy | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTA `TensorAccessorArgs(src0)` + RTA 0 (`src0_addr`) | address RTA + accessor-args | `TensorParameter INPUT_A` + conditional `TensorBinding`→`tensor::src0` (or borrowed_from when sharded) |
| reader CTA `TensorAccessorArgs(src1)` + RTA 1 (`src1_addr`) | address RTA + accessor-args | `TensorParameter INPUT_B` + `TensorBinding`→`tensor::src1` |
| writer CTA `[0]=cb_id_out` | magic CB index | `DFBBinding OUT`→`dfb::out` (in the reused fork) |
| writer CTA `TensorAccessorArgs<1>(dst)` + RTA 0 (`dst_addr`) | address RTA + accessor-args | `TensorParameter OUTPUT` + conditional `TensorBinding`→`tensor::dst` (or borrowed_from when sharded) |
| reader `get_tile_size(cb_id)` | free-fn on magic id | `dfb.get_tile_size()` (DFB getter, whitelist §A) |
| all kernels: positional CTAs/RTAs | positional | named (`args::`), CB ids → DFB bindings |

No semaphore-ID RTAs. No page-size 3rd-arg. The writer's page size moves from legacy `get_local_cb_interface(cb).fifo_page_size` to the fork's `dfb.get_entry_size()` (already in the reused fork — not this port's edit).

## Applied Patterns (HW)
- [Caution: Porting a shared kernel](../shared/port_patterns.md#caution-porting-a-shared-kernel): rung-1 reuse (donor writer fork) + rung-2 create (`bcast_hw_metal2.cpp` for the lent compute).
- [Conditional / optional DFB & tensor bindings](../shared/port_patterns.md#pattern-conditional--optional-dfb-bindings): `tensor::src0` (gated `IN0_SHARDED`) and `tensor::dst` (gated `OUT_SHARDED`), plus DFB `borrowed_from` toggled to match.
- Borrowed-memory DFBs: `IN0`←`INPUT_A`, `OUT`←`OUTPUT` under HEIGHT sharding.
- Two-toucher 1P+1C: `c_16` = compute-PRODUCER + writer-CONSUMER (writer `wait_front`s the resident output under OUT_SHARDED — a real consumer, not a self-loop).
- [Pass DFB handles directly to LLKs](../shared/port_patterns.md): compute passes `dfb::in0/in1/out` into `init_bcast` / `BCAST_OP` / `pack_tile`.

## Deferred / Flagged (HW)
- Compute `opt_level` set to **O3** explicitly (legacy `ComputeConfigDescriptor` default resolves O3; Metal 2.0 default is O2). Note: the four pass-1 factories omitted this — a latent perf drop on their compute — routed to the report as a follow-up (not fixed here; out of scope for this factory's port).
- Sharded-HW test coverage uncertain (see Flags) — routed to the report's Open items.
