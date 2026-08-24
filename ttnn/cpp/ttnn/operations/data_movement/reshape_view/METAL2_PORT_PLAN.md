# Port Plan — reshape_view

Port plan for `data_movement/reshape_view`, ported from `ProgramDescriptor` (RM factory) and
`WorkloadDescriptor` (Tiled factory) to Metal 2.0 (`ProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

Single `DeviceOperation` (`ReshapeViewDeviceOperation`) with two program factories in its
`program_factory_t` variant. Both factories are conventional factory structs (not direct-descriptor —
a `program_factory_t` variant already exists), so no nested-struct creation is forced. Both port
together to `ProgramSpecFactoryConcept`; the tiled factory additionally carries one op-owned tensor.

## Legacy Inventory

### Legacy factory shape
- **RM factory** (`ReshapeViewRMProgramFactory`): `ProgramDescriptorFactoryConcept` —
  `create_descriptor` → `ProgramDescriptor` (`reshape_row_major_program_factory.hpp:15`). Method lives
  on the factory struct (conventional shape).
- **Tiled factory** (`ReshapeViewTiledProgramFactory`): `MeshWorkloadFactoryConcept` —
  `create_workload_descriptor` → `WorkloadDescriptor` (`reshape_tiled_program_factory.hpp:21`).
  Secretly SPMD: one `ProgramDescriptor` built once and replicated across `tensor_coords` ranges
  (`reshape_tiled_program_factory.cpp:471-488`) — collapses to single-program under Metal 2.0.
- **Variants**: single device-op with two factories, selected by input layout
  (`reshape_device_operation.cpp:12-18`): `ROW_MAJOR` → RM, else → Tiled.
- **Custom `compute_program_hash`**: present at `reshape_device_operation.cpp:48-63` — **left intact.**
  Shared by both factories. (Deliberately excludes `recreate_mapping_tensor` from the hash.)

*(Metal 2.0 target concept chosen during audit: `ProgramSpecFactoryConcept` for both factories,
tiled + op-owned tensors — carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

### Variant: RM (`ReshapeViewRMProgramFactory`)

#### Kernels
Single kernel source `device/device/rm_reshape_interleaved.cpp`, instantiated as **two**
`KernelDescriptor`s over the **same** `total_cores` (a dual-instance work-split with **disjoint** CBs):
a reader-config instance (CBs 0/1) and — only when `can_use_dual_kernel` — a writer-config instance
(CBs 2/3, via CTA swap `writer_compile_time_args[2]=src2_cb_index; [3]=src3_cb_index`,
`reshape_rm_program_factory.cpp:207-209`).

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader | rm_reshape_interleaved.cpp | total_cores | see below (0-8 + TA src, TA dst) | src_addr, dst_addr, source_read_size_bytes, read_start_page, read_end_page, write_start_page, write_start_offset, nop | none | (unset → O2, DM) | ReaderConfigDescriptor{} |
| writer (cond. `can_use_dual_kernel`) | rm_reshape_interleaved.cpp | total_cores | same as reader but slots [2]=src2, [3]=src3 | same | none | (unset → O2, DM) | WriterConfigDescriptor{} |

Positional CTAs (`reshape_rm_program_factory.cpp:162-173`):
0. `src_aligned_to_64` (`source_page_size_bytes % 64 == 0`)
1. `src_aligned_to_16` (`source_page_size_bytes % 16 == 0`)
2. `src0_cb_index` (CB idx → **DFB binding**)
3. `src1_cb_index` (CB idx → **DFB binding**)
4. `source_page_size_bytes`
5. `dest_page_size_bytes`
6. `num_dest_write_slots`
7. `dest_slot_size_bytes`
8. `dest_write_size_bytes`
9+. `TensorAccessorArgs(src)` then `TensorAccessorArgs(dst)` → **TensorBindings**

RTAs (`rm_reshape_interleaved.cpp:58-67`): slot 0 `src_addr` (→ TensorBinding), slot 1 `dst_addr`
(→ TensorBinding), 2 `source_read_size_bytes`, 3 `read_start_page`, 4 `read_end_page`,
5 `write_start_page`, 6 `write_start_offset`, 7 `nop`. All per-node (node-first loop).

#### CBs
| index | total_size | page_size (entry) | num_entries | data_format | tile |
|---|---|---|---|---|---|
| 0 (src0) | cb_size0 * 2 | cb_size0 | 2 | cb_data_format (input dtype) | (unset) |
| 1 (src1) | cb_size1 | cb_size1 | 1 | cb_data_format | (unset) |
| 2 (src2, cond) | cb_size0 * 2 | cb_size0 | 2 | cb_data_format | (unset) |
| 3 (src3, cond) | cb_size1 | cb_size1 | 1 | cb_data_format | (unset) |

All on `total_cores`. CBs 2/3 allocated only when `can_use_dual_kernel`.

#### Semaphores
none

#### Tensor accessors
| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| `rm_reshape_interleaved.cpp:87` `TensorAccessor(src_args, src_addr)` | input (`src`) | reader/writer RTA slot 0 |
| `rm_reshape_interleaved.cpp:88` `TensorAccessor(dst_args, dst_addr)` | output (`dst`) | reader/writer RTA slot 1 |

Both accessors constructed in the single kernel body → both bound by both KernelSpecs.

#### Work split
- Driver: manual `responsibility = ((input_log_shape[-2]-1)/num_cores_total)+1`, adjusted up until
  `(responsibility * source_page_size_bytes) % dest_page_size_bytes == 0`
  (`reshape_rm_program_factory.cpp:109-112`). Not `split_work_to_cores`.
- Kernel created on **all** cores of `total_cores`; cores past the input rows are marked idle with a
  trailing `nop=1` RTA and `0u` buffer slots (`reshape_rm_program_factory.cpp:219-227`); kernel
  early-returns on `nop==1` before building any `TensorAccessor` (`rm_reshape_interleaved.cpp:83-85`).

### Variant: Tiled (`ReshapeViewTiledProgramFactory`)

#### Kernels
`device/device/dataflow/reader_reshape_tiled.cpp` + `device/device/dataflow/writer_reshape_tiled.cpp`,
each once, over `all_cores` (from `split_work_to_cores`).

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader | reader_reshape_tiled.cpp | all_cores | mapping_page_size_bytes, input_tile_size_bytes, mapping_cb_idx(→DFB), input_cb_idx(→DFB), TA(map), TA(input) | input_addr(→TB), map_addr(→TB), start_output_page_idx, end_output_page_idx | none | (unset → O2, DM) | ReaderConfigDescriptor{} |
| writer | writer_reshape_tiled.cpp | all_cores | input_tile_size_bytes, max_map_entries, datum_size(out), mapping_cb_idx(→DFB), input_cb_idx(→DFB), output_cb_idx(→DFB), TA(output) | output_base_addr(→TB), start_output_page, end_output_page | none | (unset → O2, DM) | WriterConfigDescriptor{} |

#### CBs
| index | total_size | page_size (entry) | num_entries | data_format | tile |
|---|---|---|---|---|---|
| 0 (mapping) | mapping_page_size_bytes * 1 | mapping_page_size_bytes | 1 | mapping_dataformat | (unset) |
| 1 (input) | input_tile_size_bytes * 1 | input_tile_size_bytes | 1 | input_cb_data_format | (unset) |
| 2 (output/working) | output_tile_size_bytes | output_tile_size_bytes | 1 | output_cb_data_format | (unset) |

All on `all_cores`.

#### Semaphores
none

#### Tensor accessors
| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| `reader_reshape_tiled.cpp:36` `TensorAccessor(input_args, input_addr)` | input | reader RTA slot 0 |
| `reader_reshape_tiled.cpp:37` `TensorAccessor(map_args, map_addr)` | mapping (**op-owned**) | reader RTA slot 1 |
| `writer_reshape_tiled.cpp:30` `TensorAccessor(output_args, output_base_addr)` | output | writer RTA slot 0 |

#### Op-owned tensors
The host-computed input→output page-mapping tensor
(`compute_reshape_mapping_host_tensor(...).to_device(device)`,
`reshape_tiled_program_factory.cpp:444-446`), parked on `workload_descriptor.buffers` via a
`shared_ptr<Tensor>` owner (`:459-461`). Fully determined by the hashed input/output shapes.
→ Metal 2.0 `op_owned_tensors`.

#### Work split
- Driver: `split_work_to_cores(grid_or_sub_grid, num_output_pages)`
  (`reshape_tiled_program_factory.cpp:316-320`).
- `(num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2)`.
- Per-core `[page_idx_start, page_idx_end)` RTAs; cores in neither group are `continue`d (all_cores is
  the union, so this is defensive) (`:391-410`).

### Shared kernels
- **none.** All three kernels are owned by `reshape_view`; each factory instantiates only its own
  kernels. `grep -rl` confirms no other op binds them.
- The RM kernel is bound by two KernelSpecs of the **same** RM factory (reader + writer instances);
  they convert together in this port (atomic unit), so this is **not** a shared-kernel Caution case —
  no fork needed. The kernel source is identical for both instances (accessor names `in0`/`in1` map to
  different DFB specs per instance).
- All three kernels `#include ttnn/operations/data_movement/common/kernels/common.hpp` and call
  `enhanced_noc_async_read` / `enhanced_noc_async_write` / `tt_memmove` — all Device 2.0 native
  (`Noc`-first). Function-call escape is clean; no donor-side work.
- Shared host/device header `device/hostdevcommon/common.hpp` (`SegmentMapData`) — in scope, unchanged.

### Flags
- **RM `write_start_offset` (RTA 6) is a de-facto constant 0** — factory always passes `0u`
  (`reshape_rm_program_factory.cpp:266,270,281`), kernel still reads and folds it
  (`rm_reshape_interleaved.cpp:66,95,106`). Dead-valued RTA. **Kept as a named RTA** (faithful port,
  no behavior change); routed to report.
- **`recreate_mapping_tensor` op attribute accepted but unused** — ignored by the tiled factory
  (`:463-466`), excluded from the hash. Not a factory-body concern; leave as-is. Team note.
- **`source_read_size_bytes` (RM RTA 2) has the same value on every node** — a CRTA candidate, but
  RTA→CRTA changes dispatch semantics; **kept as a named RTA**, routed to report.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (both factories; tiled + op-owned tensors).
- **Custom `compute_program_hash`**: present at `reshape_device_operation.cpp:48-63` — **leave intact.**
- **Implementation notes**:
  - Both factory structs already exist in the `program_factory_t` variant → no direct-descriptor edit
    (exception 3 N/A). No pybound `create_descriptor` → no pybind removal (exception 1 N/A). No
    pybind-hook-only factory parameter (exception 2 N/A). No `override_runtime_arguments` → base concept.
  - The tiled factory's `create_workload_descriptor` takes `tensor_coords` and manually replicates the
    program across coord ranges; `create_program_artifacts` drops that parameter and the replication
    loop — the single-program adapter stamps the one spec across the mesh (the "secretly SPMD → single
    program" collapse).

## Planned Spec Shape

### Variant: RM

- **KernelSpecs**: `reader` (always), `writer` (only when `can_use_dual_kernel`). Same source
  `rm_reshape_interleaved.cpp`. Both are DM kernels.
- **DataflowBufferSpecs**: `SRC0`, `SRC1` (always); `SRC2`, `SRC3` (only when `can_use_dual_kernel`).
  - `SRC0`: entry_size = cb_size0, num_entries = 2, data_format = cb_data_format.
  - `SRC1`: entry_size = cb_size1, num_entries = 1, data_format = cb_data_format.
  - `SRC2` = SRC0 shape; `SRC3` = SRC1 shape.
  - **Self-loop** each (one toucher): reader binds SRC0/SRC1 PRODUCER+CONSUMER (accessor `in0`/`in1`);
    writer binds SRC2/SRC3 PRODUCER+CONSUMER (accessor `in0`/`in1`). DM self-loop, Gen1-legal.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `SRC` (input.tensor_spec()), `DST` (output.tensor_spec()). Both bound by both
  KernelSpecs (accessor `src`/`dst`).
- **WorkUnitSpecs**: one — kernels = {reader (, writer)}, target_nodes = total_cores.
- **Op-owned tensors**: none.

### Variant: Tiled

- **KernelSpecs**: `reader` (reader_reshape_tiled.cpp), `writer` (writer_reshape_tiled.cpp). Both DM.
- **DataflowBufferSpecs**: `MAPPING` (c_0), `INPUT` (c_1), `WORKING` (c_2).
  - `MAPPING`: entry_size = mapping_page_size_bytes, num_entries = 1, data_format = mapping_dataformat.
    reader PRODUCER (accessor `mapping`), writer CONSUMER (accessor `mapping`). Plain 1:1.
  - `INPUT`: entry_size = input_tile_size_bytes, num_entries = 1, data_format = input_cb_data_format.
    reader PRODUCER (accessor `input`), writer CONSUMER (accessor `input`). Plain 1:1.
  - `WORKING`: entry_size = output_tile_size_bytes, num_entries = 1, data_format = output_cb_data_format.
    writer PRODUCER+CONSUMER (accessor `working`). **Self-loop** (one toucher — writer-only scratch).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT_T` (input), `MAPPING_T` (op-owned mapping), `OUTPUT_T` (output).
  - reader binds `INPUT_T` (accessor `input`) and `MAPPING_T` (accessor `map`).
  - writer binds `OUTPUT_T` (accessor `output`).
- **WorkUnitSpecs**: one — kernels = {reader, writer}, target_nodes = all_cores.
- **Op-owned tensors**: the mapping tensor (1) → `op_owned_tensors`, bound as `MAPPING_T`.

## Preserved Multiplicity

RM factory:
```
Legacy KernelDescriptors [reader-config, writer-config] of source rm_reshape_interleaved.cpp
  → KernelSpecs [reader, writer] of same source
  → in WorkUnitSpec [main] (both over total_cores)
  → sharing upstream/downstream DFBs: NONE — disjoint CB sets (reader→SRC0/SRC1, writer→SRC2/SRC3),
    each self-looped by its owning instance. No shared DFB → no 1P+1C / multi-binding question.
```
This is a dual-instance work-split with **disjoint** CBs (not the shared-CB two-toucher case, and not
the disjoint-node CTA-demotion case). Both instances run on every node but touch disjoint DFBs.

Tiled factory: none — no work-split multiplicity (one reader + one writer over all_cores).

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| RM CTA slot 2 (`src0_cb_index`) | `src0_cb_index` | `DFBBinding(SRC0, "in0", P+C)` on reader; `SRC2` on writer |
| RM CTA slot 3 (`src1_cb_index`) | `src1_cb_index` | `DFBBinding(SRC1, "in1", P+C)` on reader; `SRC3` on writer |
| RM CTA 9+ (`TensorAccessorArgs(src)`, `(dst)`) | `TensorAccessorArgs(*src_buffer/*dst_buffer).append_to` | `TensorParameter` SRC/DST + `TensorBinding` |
| RM RTA slot 0 (`src_addr`) | `src_buffer` (Buffer* binding-form) | `TensorBinding(SRC)` → `TensorAccessor(tensor::src)` |
| RM RTA slot 1 (`dst_addr`) | `dst_buffer` (Buffer* binding-form) | `TensorBinding(DST)` → `TensorAccessor(tensor::dst)` |
| RM CTAs 0,1,4-8 (positional) | positional `compile_time_args` | named CTAs |
| Tiled reader CTA slot 2 (`mapping_cb_idx`) | `mapping_cb_idx` | `DFBBinding(MAPPING, "mapping", ...)` |
| Tiled reader CTA slot 3 (`input_cb_idx`) | `input_cb_idx` | `DFBBinding(INPUT, "input", ...)` |
| Tiled reader CTA 4+ (`TA(map)`, `TA(input)`) | `TensorAccessorArgs(*mapping_buffer/*input_buffer)` | `TensorParameter` MAPPING_T/INPUT_T + `TensorBinding` |
| Tiled reader RTA 0/1 (`input_addr`/`map_addr`) | `input_buffer`/`mapping_buffer` (Buffer*) | `TensorBinding(INPUT_T)`/`TensorBinding(MAPPING_T)` |
| Tiled writer CTA slots 3,4,5 (`mapping/input/output cb`) | cb indices | `DFBBinding(MAPPING/INPUT/WORKING, ...)` |
| Tiled writer CTA 6+ (`TA(output)`) | `TensorAccessorArgs(*output_buffer)` | `TensorParameter` OUTPUT_T + `TensorBinding` |
| Tiled writer RTA 0 (`output_base_addr`) | `output_buffer` (Buffer*) | `TensorBinding(OUTPUT_T)` |
| Tiled reader/writer positional CTAs | positional `compile_time_args` | named CTAs |

**TensorAccessor 3rd arg**: none — no accessor passes a 3rd argument. **Semaphore-ID RTAs**: none.

## Applied Patterns

- [Sync-free / single-ended CB → self-loop DFB](../shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  RM SRC0-3 (each a single-toucher scratch ring); Tiled WORKING (writer-only scratch page).
- [Self-loop DFB binding](../shared/port_patterns.md#pattern-self-loop-dfb-binding): mechanism the above use.
- [Conditional / optional DFB bindings](../shared/port_patterns.md#pattern-conditional--optional-dfb-bindings):
  RM SRC2/SRC3 + writer KernelSpec, all gated on `can_use_dual_kernel` — **host-side only**. The kernel
  source is identical for both instances (accessor names `in0`/`in1`), so **no kernel `#ifdef`** is
  needed: every instance that compiles the source binds `dfb::in0`/`dfb::in1`.
- [Multi-variant factories](../shared/port_patterns.md#pattern-multi-variant-factories): device-op selects
  RM vs Tiled by input layout (unchanged `select_program_factory`); each factory has its own
  `create_program_artifacts`.
- Op-owned tensor (tiled mapping) — first-exercise path per the recipe's Construct step.

## Deferred / Flagged
- No structural issues beyond the audit surfaced during planning.
- RM `write_start_offset` dead-valued RTA and `source_read_size_bytes` CRTA-candidate: routed to report,
  kept as-is (no behavior change).
