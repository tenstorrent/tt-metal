# Port Plan — reshard (`NdReshardCopyPagesFactory`)

Port plan for `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard`, ported from the legacy
`ProgramDescriptor` (`descriptor` concept) API to Metal 2.0.

**Scope of this pass.** `ReshardDeviceOperation` holds **8 factory instantiations across 5 templates**
(see the audit). Per the recipe's atomic-unit rule, this pass ports **one factory**:
**`NdReshardCopyPagesFactory`** (DRAM↔DRAM page-by-page copy). It was selected first because it is the
cleanest: both its kernels live in the op's **own** directory (no shared-pool / cross-op coupling), both
tensor bindings are **Case 1** (plain `TensorAccessor`), it has **one legal `(1,1)` staging DFB** (no
multi-binding), and its args are **fixed-index** (no varargs). The other 7 instantiations are enumerated
under [Deferred / Flagged](#deferred--flagged) as remaining work — the variant's other factories stay on
the legacy `descriptor` concept meanwhile, and the op keeps building and running (per-factory dispatch).

Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor()` → `ProgramDescriptor`) — for all 8
  instantiations.
- Variants (this pass): `NdReshardCopyPagesFactory` — single, non-templated.
- Custom `compute_program_hash`: none — already default reflection-based hash (audit-confirmed; no
  `compute_program_hash` anywhere in the op).

*(Metal 2.0 target concept `MetalV2FactoryConcept` chosen by the audit — carried forward in
[TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels
Both kernel sources live in the op's **own** directory (`reshard/device/kernels/`).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `nd_reshard_copy_pages_reader.cpp` | `grid` (full compute-with-storage grid) | `[TensorAccessorArgs(input_buffer)…, cb_in0_idx (=c_0), aligned_page_size]` | none | `start_page`, `end_page` (per core) | `input_buffer` base addr (`Buffer*` binding) | none | `ReaderConfigDescriptor{}` |
| writer | `nd_reshard_copy_pages_writer.cpp` | `grid` | `[TensorAccessorArgs(output_buffer)…, cb_in0_idx (=c_0), aligned_page_size]` | none | `start_page`, `end_page` (per core) | `output_buffer` base addr (`Buffer*` binding) | none | `WriterConfigDescriptor{}` |

Kernel-side arg reads (both): `base_idx_cta`/`base_idx_crta` via `TensorAccessorArgs<0,0>().next_*_offset()`;
`dfb_id`, `page_size` CTAs; `bank_base_address_*` CRTA; `start_page`/`end_page` RTAs at fixed index 0/1.
Body: `TensorAccessor(args, base)` + `DataflowBuffer dfb(dfb_id)` + `Noc`, page loop with
`reserve_back`/`push_back` (reader) and `wait_front`/`pop_front` (writer).

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| c_0 (`cb_in0_idx`) | `aligned_page_size * num_tiles_in_cb` (num_tiles = 1) | `grid` | `input.dtype()` format | `aligned_page_size` | unset |

- `buffer = nullptr` → **not** borrowed. Plain staging FIFO.
- Endpoints: reader = FIFO **producer** (`reserve_back`/`push_back`), writer = FIFO **consumer**
  (`wait_front`/`pop_front`), on distinct kernels → **legal (1,1)** per the audit. No multi-binding.

### Semaphores
none — the op uses no semaphores.

### Tensor accessors
| host site (file:line) | originating Tensor | base delivery (host) |
|---|---|---|
| reader kernel `TensorAccessor(args_src, bank_base_address_src)` | `input` | CRTA (`emplace_common_runtime_args({input_buffer})`, `nd_reshard_program_factory_copy_pages.cpp:97`) |
| writer kernel `TensorAccessor(args_dst, bank_base_address_dst)` | `output` | CRTA (`…:98`) |

Both **Case 1** (base fed into a `TensorAccessor` constructor). Both accessors are **2-arg** (no 3rd
page-size arg). No `TensorParameter` relaxation.

### Work split
- Driver: **manual page-distribution loop** over `cores` (`nd_reshard_program_factory_copy_pages.cpp:107-116`),
  *not* `split_work_to_cores`.
- `num_dev_pages = input_buffer->buffer_distribution_spec()->tensor_shape_in_pages().volume()`, split
  evenly across all grid cores with a `remainder` spread over the first cores.
- **Single group**: every core runs the same (reader, writer) pair; only the per-core `[start_page, end_page]`
  RTA differs. **No per-group CTA variation** → no `KernelSpec` multiplicity.

### Cross-op kernels
none — both kernel sources are in the op's own directory.

### Flags
- The other **7 factory instantiations** (`ReshardSameWidthFactory<true/false>`,
  `ReshardSameHeightFactory<true/false>`, `ReshardGenericFactory`,
  `NdReshardCopyLocalShardFactory<true/false>`) are **not** ported in this pass — see
  [Deferred / Flagged](#deferred--flagged).
- Anonymous-namespace helper `push_reshard_copy_pages_cb` (`…:18`) builds a `CBDescriptor` — deleted by the
  port (replaced by a `DataflowBufferSpec`).

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `MetalV2FactoryConcept` (no op-owned tensors).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: factory method renames `create_descriptor` → `create_program_artifacts`,
  same three parameters (`const ReshardParams&`, `const ReshardInputs&`, `Tensor& output_tensor`), returns
  `ttnn::device_operation::ProgramArtifacts`. No device-op-class edits forced (no custom hash, no pybind
  `create_descriptor`, no pybind-hook-only parameter).

## Planned Spec Shape
- **KernelSpecs**: 2 — `reader` (`nd_reshard_copy_pages_reader.cpp`), `writer`
  (`nd_reshard_copy_pages_writer.cpp`). 1:1 with legacy `KernelDescriptor`s. No multiplicity.
- **DataflowBufferSpecs**: 1 — `STAGING` (plain, not borrowed). `entry_size = aligned_page_size`,
  `num_entries = num_tiles_in_cb (=1)`, `data_format_metadata = datatype_to_dataformat_converter(input.dtype())`,
  no `tile_format_metadata` (legacy `.tile` unset). reader binds PRODUCER, writer binds CONSUMER.
- **SemaphoreSpecs**: none.
- **TensorParameters**: 2 — `INPUT` (bound by reader), `OUTPUT` (bound by writer), each from
  `<tensor>.tensor_spec()`.
- **WorkUnitSpecs**: 1 — `main`, kernels `{reader, writer}`, `target_nodes = grid`.

## Preserved Multiplicity
none — no work-split multiplicity in legacy (single group; per-core variation is RTA-only).

## Dropped Plumbing
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTA (`…:69`) | `input_accessor_args.get_compile_time_args()` (`TensorAccessorArgs` plumbing) | `TensorParameter{INPUT}` + reader `TensorBinding{INPUT}` |
| reader CTA (`…:70`) | `cb_in0_idx` (magic CB index) | reader `DFBBinding{STAGING, PRODUCER}` |
| reader CRTA (`…:97`) | `emplace_common_runtime_args({input_buffer})` (base addr) | auto-injected by `TensorAccessor(tensor::input)` |
| writer CTA (`…:73`) | `output_accessor_args.get_compile_time_args()` | `TensorParameter{OUTPUT}` + writer `TensorBinding{OUTPUT}` |
| writer CTA (`…:74`) | `cb_in0_idx` | writer `DFBBinding{STAGING, CONSUMER}` |
| writer CRTA (`…:98`) | `emplace_common_runtime_args({output_buffer})` | auto-injected by `TensorAccessor(tensor::output)` |

**Retained (not dropped):**
- `aligned_page_size` CTA (`…:71,75`) → named CTA `page_size`. This is **not** a `TensorAccessor` 3rd-arg
  (the accessors are 2-arg); it is the transfer byte-count the kernel passes to `noc.async_read/write(...)`.
  It is a genuine scalar the kernel uses, so it becomes a named CTA — **not** dropped.
- `start_page`, `end_page` RTAs → named RTAs (fixed index; not varargs).

## Applied Patterns
- Standard reader → one-DFB → writer pipeline (migration guide *Complete Migration Example 1*): reader
  PRODUCER + writer CONSUMER of a plain staging DFB; derived DFB placement; `TensorAccessor` binding
  replacing the manual `TensorAccessorArgs` chain. No catalog anti-pattern applies.
- **hw_config**: `ReaderConfigDescriptor{}` resolves to the reader default `(RISCV_1, NOC_0, DM_DEDICATED_NOC)`
  → `ttnn::create_reader_datamovement_config(device->arch())`; `WriterConfigDescriptor{}` → the writer
  default → `ttnn::create_writer_datamovement_config(device->arch())`. Values-match, not name-match: these
  are the two DM default configs.

## Deferred / Flagged
- **Remaining factories (next passes):** each is a complete sub-port of its own.
  - `NdReshardCopyLocalShardFactory<true/false>` — own-dir kernel (`nd_reshard_copy_local_shards.cpp`,
    shared BRISC+NCRISC), Case 1, **no CB**. Next cleanest after this one.
  - `ReshardGenericFactory` — shared-pool kernels (`reshard_reader.cpp` / `reshard_reader_diff_width.cpp`),
    **Case 2** raw-pointer input + borrowed-DFB output (**multi-binding**), **varargs** RTA block.
  - `ReshardSameWidthFactory<true/false>`, `ReshardSameHeightFactory<true/false>` — shared-pool kernels,
    Case 2 remote + borrowed-DFB local (multi-binding), varargs; SameWidth`<true>` adds a scratch DFB
    (self-loop, multi-binding) when unaligned.
  - These 6 shared-pool kernels are a single in-place rewrite (Caution: shared dataflow kernel), but no
    other op instantiates those exact paths (audit-confirmed), so the port-together set stays this op.
- **New findings during planning:** none — the inventory matched the audit's dispositions exactly.
