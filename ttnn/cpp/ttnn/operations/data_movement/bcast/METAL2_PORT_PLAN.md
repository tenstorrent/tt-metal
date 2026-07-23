# Port Plan — `data_movement/bcast`

Port plan for `ttnn/cpp/ttnn/operations/data_movement/bcast`, ported from the legacy
`ProgramDescriptorFactoryConcept` (`create_descriptor` → `ProgramDescriptor`) to Metal 2.0
(`MetalV2FactoryConcept` → `create_program_artifacts` → `ProgramArtifacts`).

Written during the inventory and planning steps; committed alongside the port for review.

## Scope of this pass

`BcastDeviceOperation` has **five** program factories. This pass ports **three** (final result):

- `BcastMultiCoreHProgramFactory` (H, interleaved) — **PORTED**
- `BcastMultiCoreWProgramFactory` (W, interleaved) — **PORTED**
- `BcastShardedHProgramFactory` (H, sharded) — **PORTED**

**Two factories are deferred:**

- **`BcastMultiCoreHWProgramFactory` (HW)** — binds the cross-family donor writer
  `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`, shared by ~46 factories
  tree-wide. Porting it requires forking that writer (a change *outside* bcast's directory), which the
  brief flags for coordination and the recipe treats as its canonical stop signal. The invoker chose to
  defer HW.
- **`BcastShardedHOptimisedProgramFactory` (ShardedHOptimised)** — was fully ported and passed most
  configs, but **hangs reproducibly on `in1_batch_size == 2` (`batch_b > 1`) width-sharded configs**
  (legacy passes the same config). Root cause is a latent kernel over-run the new DFB L1 layout no
  longer tolerates (see `METAL2_PORT_REPORT.md` → Handoff points). Fixing it needs a kernel-logic
  change (out of scope), so the port **reverts this factory to legacy** and defers it.

Both deferred factories stay on `create_descriptor` (`ProgramDescriptorFactoryConcept`) and the op keeps
building and running (per-factory dispatch). Because the framework dispatches per factory, the device-op
`program_factory_t` variant is unchanged: three alternatives satisfy `MetalV2FactoryConcept`, two
(`BcastMultiCoreHW`, `BcastShardedHOptimised`) stay on `ProgramDescriptorFactoryConcept`.

*(The inventory / spec-plan sections below still describe ShardedHOptimised as it was constructed, since
that work informed the regression finding; treat it as reverted for the shipped diff.)*

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
