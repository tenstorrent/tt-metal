# Port Plan — `data_movement/slice`

Port plan for `ttnn/cpp/ttnn/operations/data_movement/slice`, ported from the TTNN
**ProgramDescriptor** API (`create_descriptor`) to **Metal 2.0** (`create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

**Inputs consumed:** `METAL2_PREPORT_AUDIT.md` (GREEN) and `METAL2_PORT_BRIEF.md`, both in this
directory.

## Tree state this port starts from — read first

The brief and the audit were written partly against **PR #55433 (`akertesz/slice-test`)**, where two
factories were already ported and three ops-team pre-port fixes had landed. **None of that is in this
tree.** Verified at `d1f66c276f2`:

| Item the brief says is "already done on #55433" | State in this tree |
|---|---|
| `SliceTileProgramFactory` ported (`8c8b9eea947`) | **not present** — still `create_descriptor` |
| `SliceTileTensorArgsProgramFactory` ported (`aafc364bc0c`) | **not present** |
| `TensorAccessor` 3rd args dropped (`87bd11a885e`) | **not present** — both sites still pass one |
| `check_accessor_page_size` `TT_FATAL` (`c65cafac4ee`) | **not present** |
| `IsSliceSpecFactory` bridge in `ccl/mesh_partition` (`8c8b9eea947`) | **not present** |

So this port runs the brief's *"a port that starts from an unpatched tree"* path throughout: it does
the 3rd-arg drop itself, and it must re-establish the `ccl/mesh_partition` bridge before any factory
can compile. See [Deferred / Flagged](#deferred--flagged).

> **Superseded in part — read this with the table above.** The port was later rebased onto the
> **remote** tip of `edwinlee/Port_Slice` (`6ebddf3088a`), which had diverged from the local branch
> this plan was written against and was ahead of it. Two rows above are true of the base this plan
> was written against but **not** of the base the port landed on: the remote already had the
> `TensorAccessor` 3rd-arg drop *and* a `check_accessor_page_size`. The upstream guard is the one that
> survived — it is stricter than the one written here, checking the interleaved case by rounding
> rather than skipping it. The remaining three rows are unchanged: no factory was ported upstream and
> the `ccl/mesh_partition` bridge was still absent. Full account in
> `METAL2_PORT_REPORT.md` → *Rebase onto the live branch*.

The op *does* carry `1b0d9d1258a [Bug Fix] Prepare Slice for Metal 2.0 Port (#55262)`, which is what
cleared the audit's offset-base-pointer gate (the W-begin fold is split out into `src_offset_bytes`).

---

## Legacy Inventory

### Legacy factory shape

- **Concept:** `ProgramDescriptorFactoryConcept` — all five factories declare
  `static tt::tt_metal::ProgramDescriptor create_descriptor(const SliceParams&, const SliceInputs&, Tensor&)`.
- **Where the methods live:** in five factory structs held by
  `SliceDeviceOperation::program_factory_t` (`device/slice_device_operation.hpp:36-41`). **Not** the
  direct-descriptor shape, so `ttnn_factory.md` exception 3 does not apply.
- **Variants:** five factories, selected by `select_program_factory`
  (`device/slice_device_operation.cpp:309-341`):

  | Config | Factory | File |
  |---|---|---|
  | `use_tensor_args == true` (TILE only) | `SliceTileTensorArgs` | `slice_program_factory_tile_tensor_args.cpp` |
  | RM, HEIGHT-sharded in **and** out, no step, W-begin L1-aligned | `SliceRmSharded` | `slice_program_factory_rm_sharded.cpp` |
  | RM, any `step != 1` | `SliceRmStride` (rank ≤ 4 and rank > 4 bind different kernels) | `slice_program_factory_rm_stride.cpp` |
  | RM, otherwise (interleaved **or** BLOCK/WIDTH-sharded) | `SliceRm` | `slice_program_factory_rm.cpp` |
  | TILE, otherwise | `SliceTile` | `slice_program_factory_tile.cpp` |

- **Custom `compute_program_hash`:** **present** — declared `device/slice_device_operation.hpp:51`,
  defined `device/slice_device_operation.cpp:343`. **Left intact.** It is deliberately over-keyed
  (comment at `:344-348`, issues `#53997` / `#47602` / `#45144`); several factory comments depend on
  that keying. No backdoor `attribute_values` / `to_hash` (grep: zero hits).
- **`override_runtime_arguments`:** present on all five, each a one-line delegation to the shared
  `patch_slice_program_addresses` (`slice_program_factory_rm_sharded.cpp:354-413`). This is the
  target-concept selector → `CustomProgramSpecFactoryConcept`.
- **`opt_level`:** `grep -n opt_level` over the five factory `.cpp`/`.hpp` → **zero hits**. Every
  kernel is data-movement, so every resolved level is `O2`, which is also Metal 2.0's
  `CompilerOptions` default. **No `opt_level` line is owed anywhere in this port** (there is not a
  single compute kernel in the op).
- **`defines`:** none set by any factory (confirms the donor's `OUT_SHARDED` / `BACKWARDS` and slice's
  own writer's identical pair never fire).

---

### Variant: `SliceTile`

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `slice/device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp` | `all_cores` | `[0]=num_dims`, then `TensorAccessorArgs(src0)` | `dfb_id_in = 0` (`:139`) | `[0]=start_id, [1]=num_tiles, [2..2+num_dims)=id_per_dim` | `[0]=src0_buffer (Buffer*)`, `[1..1+2·num_dims)` = `num_unpadded_tiles_per_dim` ‖ `num_padded_tiles_per_dim` | none | O2 (unset, DM) | `ReaderConfigDescriptor{}` |
| writer | `slice/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (slice's **own** copy) | `all_cores` | `TensorAccessorArgs(dst)` from slot 0 | `dfb_id_out = 0` (`:161`) | `[0]=dst_buffer (Buffer*), [1]=num_tiles, [2]=start_id` | none | none | O2 (unset, DM) | `WriterConfigDescriptor{}` |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| 0 | `2 · single_tile_size` | `all_cores` | `datatype_to_dataformat_converter(input.dtype())` | `single_tile_size` | unset |

#### Semaphores
none — the op declares no semaphores at all.

#### Tensor accessors
| host site | originating Tensor | arg slot (host) |
|---|---|---|
| `slice_program_factory_tile.cpp:65` (`TensorAccessorArgs(*src0_buffer)`) | `tensor_args.input` | reader **CRTA 0** (`:143`) |
| `slice_program_factory_tile.cpp:152` (`TensorAccessorArgs(*dst_buffer)`) | `tensor_return_value` (output) | writer RTA 0 (`:180`) |

#### Work split
`split_work_to_cores(sub_core_grids | compute_with_storage_grid_size, num_unpadded_tiles)`
→ `(num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2)`.
Cores enumerated with `corerange_to_cores(all_cores)`; cores in neither group are **no-op cores** that
still receive a zero-filled arg row.

---

### Variant: `SliceTileTensorArgs`

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `slice/device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | `all_cores` | `[0]=src0_cb_index, [1]=tensor_cb_index, [2]=num_dims, [3]=tile_width, [4]=tile_height`, then three chained `TensorAccessorArgs` (src, start, end) | none | `[0]=start_id, [1]=num_tiles, [2..2+num_dims)=id_per_dim` | `[0]=src_buffer, [1]=start_buffer, [2]=end_buffer` (all `Buffer*`), `[3..3+3·num_dims)` = `num_unpadded` ‖ `num_padded` ‖ `input_shape` | none | O2 | `ReaderConfigDescriptor{}` |
| writer | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**cross-family donor**) | `all_cores` | `[0]=src0_cb_index`, then `TensorAccessorArgs(dst)` | none | `[0]=dst_buffer, [1]=num_tiles, [2]=start_id` | none | none | O2 | `WriterConfigDescriptor{}` |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| 0 | `2 · single_tile_size` | `all_cores` | input dtype | `single_tile_size` | unset |
| 1 | `single_tile_size` | `all_cores` | input dtype | `single_tile_size` | unset |

#### Tensor accessors
| host site | originating Tensor | arg slot |
|---|---|---|
| `…_tile_tensor_args.cpp:82` | `tensor_args.input` | reader CRTA 0 (`:182`) |
| `…_tile_tensor_args.cpp:83` | `tensor_args.start_tensor` | reader CRTA 1 (`:183`) |
| `…_tile_tensor_args.cpp:84` | `tensor_args.end_tensor` | reader CRTA 2 (`:184`) |
| `…_tile_tensor_args.cpp:87` | output | writer RTA 0 (`:151`, `:168`) |

#### Work split
Same shape as `SliceTile`, over `num_unpadded_tiles`, with `start_offset` hard-wired to `0`
(`:129`) — the real offset is computed on-device from the start tensor.

---

### Variant: `SliceRm`

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `all_cores` | `TensorAccessorArgs(src0)` from slot 0 | none | `[0]=src0_buffer`, `[1]=reader_page_size`, `[2]=unpadded_row_size_bytes`, `[3]=unpadded_row_size_bytes_offset`, `[4]=num_dims`, `[5]=misalignment`, `[6]=start_id`, `[7]=num_sticks_per_core`, `[8]=num_sticks_per_core_read`, `[9]=num_read_per_barrier`, `[10]=chunk_size`, `[11]=num_chunks_per_stick`, `[12]=last_chunk_size`, `[13]=begins_bytes−misalignment`, `[14..]` = `num_unpadded_sticks_per_dim` ‖ `num_padded_sticks_per_dim` ‖ `id_per_dim` (3 × `num_dims`) | none | none | O2 | `ReaderConfigDescriptor{}` |
| writer | `slice/device/kernels/dataflow/slice_writer_unary_stick_layout_interleaved_start_id.cpp` | `all_cores` | `[0]=src0_cb_index`, then `TensorAccessorArgs(dst)` | none | `[0]=dst_buffer`, `[1]=stick_size`, `[2]=stick_size_offset`, `[3]=num_sticks_per_core`, `[4]=num_sticks_per_core_read`, `[5]=num_read_per_barrier`, `[6]=start_id`, `[7]=page_size_override`, `[8]=chunk_size`, `[9]=num_chunks_per_stick`, `[10]=last_chunk_size` | none | none | O2 | `WriterConfigDescriptor{}` |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| 0 | `num_read_per_barrier · 2 · cb_page_size` | `all_cores` | input dtype | `sizing.cb_page_size` | unset |

#### Tensor accessors
| host site | originating Tensor | arg slot |
|---|---|---|
| `slice_program_factory_rm.cpp:336` | `tensor_args.input` | reader RTA 0 (`:377`) |
| `slice_program_factory_rm.cpp:333` | output | writer RTA 0 (`:385`) |

Both accessors pass a **3rd (page-size) argument** — the two Class-2 drop sites the brief names.

#### Work split
`split_work_to_cores(…, num_unpadded_sticks)`; per-core `num_sticks_per_core` is `0` for cores in
neither group (still emitted, with a full arg row).

---

### Variant: `SliceRmStride`

**Runtime kernel-source selection** on `input_shape.rank()`:

| rank | reader source | writer source |
|---|---|---|
| ≤ 4 | `reader_multicore_slice_4d.cpp` | `writer_multicore_slice_4d.cpp` |
| > 4 | `reader_multicore_slice_nd.cpp` | `writer_multicore_slice_nd.cpp` |

Both pairs convert together — this factory is one atomic unit of four kernel entry points.

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader (4D) | `reader_multicore_slice_4d.cpp` | `all_cores` | `[0]=in_cb`, `[1]=element_size` (**dead**), then `TensorAccessorArgs(input)` | 25 fixed fields via `rt_args_idx++` — `src_addr`(Buffer*), `tensor_rank`, `input_w/h/d/n`, `output_w/h/d/n`, `slice_{start,end,step}_{w,h,d,n}`, `element_size`, `num_rows_for_this_core`, `start_row_for_this_core` | O2 | `ReaderConfigDescriptor{}` |
| writer (4D) | `writer_multicore_slice_4d.cpp` | `all_cores` | `[0]=in_cb`, `[1]=element_size` (**dead**), then `TensorAccessorArgs(output)` | 9 fixed fields — `dst_addr`(Buffer*), `tensor_rank`, `output_w/h/d/n`, `element_size`, `num_rows_for_this_core`, `start_row_for_this_core` | O2 | `WriterConfigDescriptor{}` |
| reader (ND) | `reader_multicore_slice_nd.cpp` | `all_cores` | same two + `TensorAccessorArgs(input)` | `src_addr`(Buffer*), `tensor_rank`, `element_size`, `num_rows_for_this_core`, `start_row_for_this_core`, then **five** `tensor_rank`-long blocks: `input_dims`, `output_dims`, `slice_starts`, `slice_ends`, `slice_steps` | O2 | `ReaderConfigDescriptor{}` |
| writer (ND) | `writer_multicore_slice_nd.cpp` | `all_cores` | same two + `TensorAccessorArgs(output)` | `dst_addr`(Buffer*), `tensor_rank`, `element_size`, `num_rows_for_this_core`, `start_row_for_this_core`, then one `tensor_rank`-long `output_dims` block | O2 | `WriterConfigDescriptor{}` |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| 0 | `2 · cb_page_size_aligned` | `all_cores` | input dtype | `round_up(input_w · element_size, max(src,dst) alignment)` | unset |

#### Work split
`split_work_to_cores(…, total_output_rows)`, but the per-core row counts are **not** taken from the
returned group counts: the factory re-derives them as `base_rows_per_core = total_output_rows /
num_cores` plus one extra row for the first `total_output_rows % num_cores` cores (`:101-102`,
`:120-124`). Preserve exactly.

---

### Variant: `SliceRmSharded`

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader | `slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp` | `all_cores_unpadded` | `[0]=stick_size_unpadded`, `[1]=num_sticks_unpadded`, `[2]=src_stride_bytes`, `[3]=dst_stride_bytes`, `[4]=begins_bytes` | `[0]=num_cores_read`, then the data-directed gather plan: `2·num_cores_read` interleaved NoC (x,y), `num_cores_read` chunk counts, and a `(start_id, length)` pair per chunk | O2 | `ReaderConfigDescriptor{}` |

**Only one kernel** in this factory — there is no writer.

#### CBs
| index | total_size | core_ranges | data_format | page_size | `.buffer` |
|---|---|---|---|---|---|
| 0 | `shard_height_padded · src_stride_bytes` | `all_cores_unpadded` | input dtype | `src_stride_bytes` | `input.buffer()` (**borrowed**) |
| `c_16` (16) | `shard_height_unpadded · dst_stride_bytes` | `all_cores_unpadded` | output dtype | `dst_stride_bytes` | `output.buffer()` (**borrowed**) |

Neither sets `address_offset`, and neither is a `GlobalCircularBuffer` — the plain borrowed-memory
pattern → `DataflowBufferSpec::borrowed_from`.

#### Tensor accessors
**None.** This factory constructs no `TensorAccessor` and passes no address arg; both tensors reach
the kernel only through the two borrowed DFBs.

#### Work split
Driven by the two shard specs, not `split_work_to_cores`: `num_cores_unpadded =
output.shard_spec().num_cores()`, with the core order recomputed from `row_major` and the unpadded
grid dimensions (`:335-341`).

---

### Shared kernels

| kernel | relationship | `_metal2` fork beside it? | Rung |
|---|---|---|---|
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **borrowed** by `SliceTileTensorArgs` (`…_tile_tensor_args.cpp:133`); ≥15 other binding factories | **yes** — `…/writer_unary_interleaved_start_id_metal2.cpp` | **Rung 1 — bind the existing fork, create nothing, edit nothing** |
| `slice/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (slice's own copy) | neither borrowed nor lent — `SliceTileProgramFactory` is its only binder | n/a | **Convert in place, no fork** |
| slice's other nine referenced kernels | slice-owned, single-binder | converted in place | none |

**Fork's binding vocabulary (now this port's constraint, not a free choice):** DFB `dfb::out`
(CONSUMER), tensor `tensor::dst`, named args `args::num_pages`, `args::start_id`; gated on
`#ifdef OUT_SHARDED` / `#ifdef BACKWARDS`, neither of which slice sets. That is exactly the
`{dst_buffer, num_tiles_per_core, num_tiles_written}` triple the factory supplies today.
`ttnn/cpp/ttnn/operations/kv_cache/device/fill_cache_multi_core_program_factory.cpp:208-220` is an
in-tree consumer of the same fork and confirms the shape.

Do **not** redirect `SliceTile` onto that fork: slice's own copy exists precisely because it takes its
DFB index from a **named** CTA so the fusion infrastructure can remap it. Collapsing them would be a
functional change.

### Flags

- **Unreferenced kernel files in the op directory, not audited and not touched:**
  `device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp`,
  `device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp`.
- **`ccl/mesh_partition` is an out-of-directory *host* consumer** of `create_descriptor` and
  `patch_slice_program_addresses`. See [Deferred / Flagged](#deferred--flagged).
- **Pre-existing anomalies the audit lists under *Misc anomalies*** — dead
  `compile_time_element_size` CTA in four stride kernels, dead RTAs in the 4D stride writer, the end
  tensor that `SliceTileTensorArgs` reads and discards, dead `#ifdef` branches in slice's own writer
  copy, `constexpr uint32_t start_offset = 0` in the tensor-args factory. **All preserved verbatim**;
  the port keeps binding and emitting them.

---

## TTNN ProgramFactory

- **Concept (inherited from audit):** **`CustomProgramSpecFactoryConcept`** — all five factories.
  Selected by *"Override runtime args method? == yes"*, confirmed by the readiness sheet's
  `Porting Target` cell. Each factory gains:
  ```cpp
  static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
      const SliceParams&, const SliceInputs&, Tensor&);
  static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
      const SliceParams&, const SliceInputs&, Tensor&,
      const std::optional<ttnn::MeshCoordinate>& = std::nullopt);
  ```
  and loses `create_descriptor` (mandatory: `ProgramDescriptorFactoryConcept` and
  `ProgramSpecFactoryConcept` are mutually exclusive by construction —
  `ttnn/api/ttnn/operation_concepts.hpp:120-136`).
- **Custom `compute_program_hash`:** present at `device/slice_device_operation.cpp:343` — **left
  exactly as it is.** Heads-up only; if a `TensorSpec` legality failure appears on the *second*
  invocation of a test, this hash is the named suspect and the correct response is stop-and-report.
- **Op-owned tensors:** **none.** No `create_workload_descriptor`, no `buffers` vector. The
  `op_owned_tensors` field of `ProgramArtifacts` is omitted throughout.
- **Implementation notes:**
  - The `CustomProgramSpecMeshWorkloadFactoryAdapter` cache-hit path
    (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:963-983`) calls **only**
    `UpdateProgramRunArgs` with what the override returns — it does **not** also run the base
    adapter's `UpdateTensorArgs`. So every `TensorParameter` bound to an io tensor must appear in the
    override's `tensor_args`, or its address freezes at the cache-miss value. All five overrides
    therefore return the factory's full `tensor_args` set.
  - `patch_slice_program_addresses` and `slice_tile_dynamic_args` are **deleted** once the last
    factory converts; their whole job (re-pointing addresses, re-emitting the tile factories' per-core
    scalars) is subsumed by `TensorArgument` bindings plus the translated overrides.
  - Spec resource names are declared **function-locally** in each factory: the five slice factory
    `.cpp`s land in one unity-build translation unit, where anonymous namespaces merge and
    file-scope `const KernelSpecName READER{"reader"}` in two files would collide. (This is the same
    hazard `pad_rm_sharded_height_only_program_factory.cpp:22-23` documents for its seven factories,
    which it solves with per-factory name prefixes.)

---

## Planned Spec Shape

### Variant: `SliceTile`

- **KernelSpecs (2):** `reader`, `writer` — 1:1 with the legacy `KernelDescriptor`s.
- **DataflowBufferSpecs (1):** `src0` ← legacy CB 0. `entry_size = single_tile_size`,
  `num_entries = 2`, `data_format_metadata = cb_data_format`.
- **SemaphoreSpecs:** none.
- **TensorParameters (2):** `input` ← `tensor_args.input.tensor_spec()`; `output` ←
  `tensor_return_value.tensor_spec()`. No relaxations.
- **WorkUnitSpecs (1):** `{reader, writer}` on `all_cores`.
- **Bindings:** `src0` — reader PRODUCER (`in0`), writer CONSUMER (`out`). Plain 1:1.
- **Varargs:** reader `num_runtime_varargs = num_dims` (`id_per_dim`);
  `num_common_runtime_varargs = 2 · num_dims` (`num_unpadded_tiles_per_dim` ‖ `num_padded_tiles_per_dim`).

### Variant: `SliceTileTensorArgs`

- **KernelSpecs (2):** `reader` (own source), `writer` (**donor `_metal2` fork**).
- **DataflowBufferSpecs (2):** `src0` ← CB 0 (`single_tile_size` × 2);
  `tensor_stage` ← CB 1 (`single_tile_size` × 1).
- **TensorParameters (4):** `input`, `start`, `end`, `output`.
- **WorkUnitSpecs (1):** `{reader, writer}` on `all_cores`.
- **Bindings:** `src0` — reader PRODUCER (`in0`), donor writer CONSUMER (`out`). `tensor_stage` —
  reader **self-loop** (PRODUCER *and* CONSUMER, both `tensor_stage`).
- **Varargs:** reader `num_runtime_varargs = num_dims`;
  `num_common_runtime_varargs = 3 · num_dims`.

### Variant: `SliceRm`

- **KernelSpecs (2):** `reader`, `writer`.
- **DataflowBufferSpecs (1):** `src0` ← CB 0. `entry_size = sizing.cb_page_size`,
  `num_entries = 2 · sizing.num_read_per_barrier` (legacy `total_size = nrpb · 2 · page_size`).
- **TensorParameters (2):** `input`, `output`.
- **WorkUnitSpecs (1):** `{reader, writer}` on `all_cores`.
- **Bindings:** `src0` — reader PRODUCER (`in0`), writer CONSUMER (`out0`). Plain 1:1.
- **Varargs:** reader `num_runtime_varargs = 3 · num_dims`; writer none.

### Variant: `SliceRmStride`

- **KernelSpecs (2):** `reader`, `writer` — each with a runtime-selected `source` and a
  rank-dependent `runtime_arg_schema` / vararg count.
- **DataflowBufferSpecs (1):** `in_dfb` ← CB 0. `entry_size = cb_page_size_aligned`, `num_entries = 2`.
- **TensorParameters (2):** `input`, `output`.
- **WorkUnitSpecs (1):** `{reader, writer}` on `all_cores`.
- **Bindings:** `in_dfb` — reader PRODUCER (`out`), writer CONSUMER (`in`). Plain 1:1 on both rank
  paths. *(The accessor names read "backwards" because they are the kernels' own vocabulary: the
  reader fills what it calls `dfb_out`, the writer drains what it calls `dfb_in`.)*
- **Varargs:** ND path only — reader `num_runtime_varargs = 5 · tensor_rank`, writer
  `num_runtime_varargs = tensor_rank`. 4D path: **zero** varargs; all 25 / 9 fields are named.

### Variant: `SliceRmSharded`

- **KernelSpecs (1):** `reader`.
- **DataflowBufferSpecs (2):** `in_shard` ← CB 0, `borrowed_from = input`;
  `out_shard` ← CB `c_16`, `borrowed_from = output`.
- **TensorParameters (2):** `input`, `output` — declared **only** to back the two borrowed DFBs; no
  kernel binds a `TensorAccessor` to either.
- **WorkUnitSpecs (1):** `{reader}` on `all_cores_unpadded`.
- **Bindings:** `in_shard` — reader **self-loop** (sync-free, raw `get_write_ptr()` peek only);
  `out_shard` — reader **self-loop** (locked producer, nothing drains it).
- **Varargs:** reader `num_runtime_varargs = max over nodes of (3·num_cores_read + 2·num_chunks)`,
  zero-filled per node to that maximum (the kernel walks the block by the counts it reads out of it,
  so the tail is never read).

---

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** No factory pushes the same `kernel_source` into two
`KernelDescriptor`s; every factory's reader and writer are distinct sources, and `SliceRmSharded` has
a single kernel. (Confirms the audit's *dual-instance work-split hunt (face c): none*.)

---

## Dropped Plumbing

### `SliceTile`
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `…_tile.cpp:143` (reader CRTA 0) | `reader_common.push_back(src0_buffer)` | `TensorBinding{input, "input"}` |
| `…_tile.cpp:65` | `TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args)` | binding mechanism |
| `…_tile.cpp:139` | `named_compile_time_args = {{"dfb_id_in", 0}}` | `DFBBinding{src0, "in0", PRODUCER}` |
| `…_tile.cpp:64` (reader CTA 0) | positional `{num_dims}` | named `compile_time_args = {{"num_dims", …}}` |
| `…_tile.cpp:180` (writer RTA 0) | `dst_buffer` in `emplace_runtime_args` | `TensorBinding{output, "dst"}` |
| `…_tile.cpp:152` | `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` | binding mechanism |
| `…_tile.cpp:161` | `named_compile_time_args = {{"dfb_id_out", 0}}` | `DFBBinding{src0, "out", CONSUMER}` |
| reader kernel `:14` | `constexpr auto src_args = TensorAccessorArgs<1>()` + `:15` `src_addr` CRTA read | `TensorAccessor(tensor::input)` |
| writer kernel `:20` | `constexpr auto dst_args = TensorAccessorArgs<0>()` + `:15` `dst_addr` RTA read | `TensorAccessor(tensor::dst)` |

### `SliceTileTensorArgs`
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `…_tensor_args.cpp:182,183,184` (reader CRTA 0/1/2) | three `Buffer*` pushes | `TensorBinding{input,"input"}`, `{start,"start"}`, `{end,"end"}` |
| `…_tensor_args.cpp:82,83,84` | three chained `TensorAccessorArgs(...).append_to(...)` | binding mechanism |
| reader kernel `:17-19` | `TensorAccessorArgs<5>()` → `next_compile_time_args_offset()` chain ×2 | three `TensorAccessor(tensor::…)` |
| `…_tensor_args.cpp:81` CTA `[0]`,`[1]` | `src0_cb_index`, `tensor_cb_index` | `DFBBinding`s (`in0`, `tensor_stage`) |
| `…_tensor_args.cpp:81` CTA `[2..4]` | positional `num_dims`, `tile_width`, `tile_height` | named CTAs |
| `…_tensor_args.cpp:86` CTA `[0]` | `src0_cb_index` for the donor writer | `DFBBinding{src0, "out", CONSUMER}` |
| `…_tensor_args.cpp:87` | `TensorAccessorArgs(*dst_buffer)` | `TensorBinding{output, "dst"}` (donor's name) |
| `…_tensor_args.cpp:151,168` (writer RTA 0) | `dst_buffer` | same `TensorBinding` |

### `SliceRm`
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `…_rm.cpp:377` (reader RTA 0) | `reader_args.push_back(src0_buffer)` | `TensorBinding{input, "input"}` |
| `…_rm.cpp:336` | `TensorAccessorArgs(*src0_buffer).append_to(...)` | binding mechanism |
| `…_rm.cpp:86` → reader RTA 1 | `reader_page_size` (`per_shard_page_size_bytes(...)`) — the accessor's **3rd argument** | **dropped**; the binding token supplies the aligned page size (Class 2) |
| `…_rm.cpp:385` (writer RTA 0) | `writer_args.push_back(dst_buffer)` | `TensorBinding{output, "dst"}` |
| `…_rm.cpp:333` | `TensorAccessorArgs(*dst_buffer).append_to(...)` | binding mechanism |
| `…_rm.cpp:155` → writer RTA 7 | `writer_page_size` — the accessor's **3rd argument** | **dropped** (Class 2) |
| `…_rm.cpp:332` writer CTA `[0]` | `src0_cb_index` | `DFBBinding{src0, "out0", CONSUMER}` |
| reader kernel `:45` | `constexpr uint32_t dfb_id_in0 = 0;` (hardcoded magic index) | `DFBBinding{src0, "in0", PRODUCER}` |
| reader kernel `:36`, writer `:30` | `TensorAccessorArgs<N>()` + address RTA reads | `TensorAccessor(tensor::…)` |
| every remaining reader / writer RTA | positional `get_arg_val<uint32_t>(N)` | named `get_arg(args::…)` |

### `SliceRmStride`
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `…_rm_stride.cpp:128,147` (reader RTA 0) | `input_buffer` | `TensorBinding{input, "src"}` |
| `…_rm_stride.cpp:136,160` (writer RTA 0) | `output_buffer` | `TensorBinding{output, "dst"}` |
| `…_rm_stride.cpp:80,83` | `TensorAccessorArgs(...).append_to(...)` ×2 | binding mechanism |
| `…_rm_stride.cpp:79,82` CTA `[0]` | `in_cb` | `DFBBinding`s |
| `…_rm_stride.cpp:79,82` CTA `[1]` | positional `element_size` (**dead** in all four kernels) | named CTA `compile_time_element_size` — **kept, still dead** (audit: preserve) |
| all remaining fixed RTAs | positional `get_arg_val<uint32_t>(rt_args_idx++)` | named `get_arg(args::…)` |

### `SliceRmSharded`
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `…_rm_sharded.cpp:290` | `CBDescriptor{.buffer = input.buffer()}` | `DataflowBufferSpec{.borrowed_from = input}` |
| `…_rm_sharded.cpp:302` | `CBDescriptor{.buffer = output.buffer()}` | `DataflowBufferSpec{.borrowed_from = output}` |
| `…_rm_sharded.cpp:364-366` | `apply_descriptor_runtime_args` over a CB-address-only descriptor, matched **positionally** | the two `TensorArgument`s in the override; the framework re-points each borrowed DFB from its backing tensor. **The positional CB order stops mattering** — the binding is by name. |
| reader kernel `:32,33` | `constexpr auto dfb_in0 = tt::CBIndex::c_0; … c_16` (magic indices) | `dfb::in_shard`, `dfb::out_shard` |
| reader kernel `:14-19` | five positional CTAs | five named CTAs |
| reader kernel `:25` | `get_arg_val<uint32_t>(0)` | `get_arg(args::num_cores_read)` |
| reader kernel `:26-30` | four aliased `get_arg_addr(...)` pointers into one interleaved block | one flat runtime-vararg block read through `get_vararg(base + i)` with explicit bases |

**Not dropped anywhere:** no semaphore-ID RTAs (the op has none), no CTA varargs (every
`get_compile_time_arg_val` index in the op is a literal), no Case-2 raw-pointer bindings.

---

## Applied Patterns

- **Self-loop DFB binding** — `SliceTileTensorArgs`'s `tensor_stage` (one toucher already holding
  both FIFO roles); `SliceRmSharded`'s `in_shard` (sync-free, raw peek only) and `out_shard` (locked
  producer, nothing drains it). Three sites, matching the audit's census exactly. Re-derived from the
  kernel-touch census rather than transcribed: `in_shard` — one toucher
  (`slice_reader_…_rm_sharded.cpp:41` `get_write_ptr`, no FIFO ops); `out_shard` — one toucher
  (`:40` `reserve_back`, `:42` `get_write_ptr`, `:89` `push_back`); `tensor_stage` — one toucher
  running `reserve_back`/`push_back`/`wait_front`/`pop_front` twice (`…_tensor_args.cpp:52-59,66,69-76,83`).
- **Borrowed-memory DFB** — `SliceRmSharded`'s two DFBs, `borrowed_from` the `input` / `output`
  `TensorParameter`s. This also preserves the invariant the kernel silently depends on: it takes
  `l1_read_addr = dfb_in.get_write_ptr()` — a *local* pointer — and uses it as the `.addr` of reads
  aimed at *other* cores, which is only correct because a sharded DFB lands at the same L1 offset on
  every core in the range.
- **Shared kernel, rung 1 (reuse the existing `_metal2` fork)** — `SliceTileTensorArgs`'s writer binds
  `eltwise/unary/…/writer_unary_interleaved_start_id_metal2.cpp`. No new file, no edit to the fork, no
  pointer comment added to the legacy original (it already has one).
- **Runtime kernel-source selection** — `SliceRmStride` picks its reader/writer pair by rank; all four
  sources convert in this change.
- **Removing a pybound legacy factory entry point** — `slice_nanobind.cpp:168-179` binds
  `SliceTileProgramFactory::create_descriptor`; the port makes that method vanish, so the binding is
  deleted. User-visible API change → its own port-report entry.
- **Multi-binding advanced option** — **not used anywhere.** Re-derived: no DFB in this op reaches ≥3
  touchers or doubles a FIFO role, and the hidden-co-filler face is structurally impossible (the op
  declares no semaphores, so nothing could coordinate a raw co-fill).

---

## Deferred / Flagged

1. **`ccl/mesh_partition` blocks the *first* factory port in this tree, and the fix is out-of-directory.**
   `ttnn/cpp/ttnn/operations/ccl/mesh_partition/device/mesh_partition_program_factory.cpp` calls
   `Factory::create_descriptor(...)` inside a `std::visit` over slice's `program_factory_t` (`:131`)
   and refreshes through `ttnn::prim::patch_slice_program_addresses` (`:155`). Porting **any** slice
   factory removes `create_descriptor` from that variant alternative and breaks the build. That op is
   `legacy (MeshWorkload)` / `Is able to port? = no`, so it cannot co-migrate.

   The audit closed this as *Questions* #3: **option (b) was chosen, implemented on `akertesz/slice-test`
   (`8c8b9eea947`), and the out-of-op-directory change was explicitly authorized by the invoker.** The
   design is specified there — a concept keyed on the *entry point*:
   ```cpp
   template <typename T>
   concept IsSliceSpecFactory = requires { &T::create_program_artifacts; };
   ```
   with both call sites branched on it. This port **re-implements that authorized change** (it is
   absent from this tree) rather than re-deciding it. Two caveats carried into the report: the edit is
   outside the op directory, and it is **not run-verified** — MeshPartition's tests are t3000/TG-only.

2. **`get_vararg()` is read-only, and three slice readers advance their `id_per_dim` block in place.**
   `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:80`,
   `reader_unary_unpad_dims_interleaved_start_id.cpp:45`,
   `…_tensor_args.cpp:124` all do `id_per_dim[j]++`. `genfiles.cpp:441` emits only
   `get_vararg(idx)` — a *value* getter — with no address form, so the host-seeded block cannot be
   written back. The two tile readers take `num_dims` from a **CTA**, so the block copies into a
   `uint32_t id_per_dim[num_dims]` local (the pattern `data_movement/pad` already established:
   `reader_pad_tiled.cpp:22-32`). The **RM** reader takes `num_dims` from an **RTA**, so a
   compile-time-sized local needs a bound: use `tensor_accessor::MAX_RANK` (8,
   `tt_metal/hw/inc/internal/tensor/const.h:11`) — the accessor's own rank ceiling, already in scope
   in these kernels — with a device `ASSERT`. This is the one place the port changes *where* a value
   lives rather than how it is spelled; it is behaviour-identical because nothing reads the block back
   from L1 after the kernel exits. Reported as a framework gap (a `get_vararg_addr(i)`).

3. **`SliceRmSharded`'s vararg block length varies per node.** `num_cores_read` and the chunk count
   are data-directed, but `KernelAdvancedOptions::num_runtime_varargs` is one number for the whole
   KernelSpec. Declare the longest block and zero-fill the shorter ones; the kernel walks the block by
   the counts it reads out of it, so the tail is never touched. (Same resolution
   `pad_rm_sharded_height_only_program_factory.cpp:360-367` reached.)

4. **Legacy cache-miss / cache-hit divergence in the tile factories, preserved.**
   `create_descriptor` gives a no-op core writer args `{0, 0, 0}` (`…_tile.cpp:176`), while
   `slice_tile_dynamic_args` re-emits that core's writer slot 2 as the *running* `num_tiles_written`
   (`:274`) rather than 0. Also, `patch_slot0` deliberately skips slots holding 0, so a no-op core's
   address slot keeps its zero across hits, whereas a `TensorBinding` patches every node uniformly.
   Both differences are inert (`num_pages == 0`, so the writer loop never runs). **Preserved as-is**
   and recorded in the port report — a port may not fix a latent inconsistency.

5. **No new findings that contradict the audit.** The CB census, the vararg census, the "no Case 2
   anywhere" finding, the `none` relaxation and the target concept all re-derived clean against this
   tree.
