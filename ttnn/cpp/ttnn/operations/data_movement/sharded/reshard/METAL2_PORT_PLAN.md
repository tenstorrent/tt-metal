# Port Plan — reshard (`NdReshardCopyPagesFactory`)

Port plan for `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard`, ported from the legacy
`ProgramDescriptor` (`descriptor` concept) API to Metal 2.0.

**Scope.** `ReshardDeviceOperation` holds **8 factory instantiations across 5 templates** (see the audit).
Per the recipe's atomic-unit rule, factories are ported one at a time. **Two are done so far:**

1. **`NdReshardCopyPagesFactory`** (DRAM↔DRAM page-by-page copy) — the cleanest (own-dir kernels, Case 1,
   plain `(1,1)` staging DFB, fixed-index args). Committed `cbc70d0efce`. **Hardware-verified.**
2. **`ReshardGenericFactory`** (general L1-sharded↔L1-sharded reshard) — the complex one: **Case 2**
   raw-pointer input, **borrowed-memory** output DFB, **multi-binding** (self-loop), **varargs**, and an
   **in-place shared-pool kernel rewrite** (`reshard_reader.cpp` / `reshard_reader_diff_width.cpp`).
   **Hardware-verified.** See [Factory 2](#factory-2--reshardgenericfactory).

The remaining 6 instantiations are enumerated under [Deferred / Flagged](#deferred--flagged); they stay on
the legacy `descriptor` concept and dispatch per-factory at runtime, so the op keeps building and running.

The sections below (Legacy Inventory … Applied Patterns) document **Factory 1**; Factory 2 has its own
[section](#factory-2--reshardgenericfactory).

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

## Factory 2 — `ReshardGenericFactory`

General L1-sharded↔L1-sharded reshard (selected for width-sharded-with-padding and the non-DRAM sharded
fallthrough). Single, non-templated factory. Runtime-selects one of two kernel sources by page-size match.

### Legacy Inventory (Generic)
- **Factory shape**: `descriptor` (`create_descriptor` → `ProgramDescriptor`); no custom hash.
- **Kernels** (runtime-selected, both **shared pool** `data_movement/sharded/device/kernels/dataflow/`,
  **not** the op's own dir): `reshard_reader.cpp` (page sizes equal) / `reshard_reader_diff_width.cpp`
  (page sizes differ). Cross-op check: **no other op** instantiates these exact paths — `experimental/quasar/reshard`
  keeps its *own* copies — so the port-together set is just this factory (Caution: shared dataflow kernel,
  rewrite in place). Each source is instantiated **twice** (kernel_desc_0 = `ReaderConfigDescriptor`,
  kernel_desc_1 = `WriterConfigDescriptor`) on the full `all_cores` grid, splitting output-page ranges
  across BRISC/NCRISC.
- **CTAs** (positional, both kernels): `{dst_cb_index(=16), grid.x, grid.y, page_size, unit_size}`.
- **CB** c_16: `total_size` (output shard bytes), `page_size = output_buffer->page_size()`, **borrowed**
  (`cb.buffer = output_buffer`). Used sync-free (`dfb.get_write_ptr()` only, no FIFO ops).
- **RTAs** (variable, per core): `[phys-x-coords(grid.x), phys-y-coords(grid.y), input_addr, num_output_pages,
  num_ranges/num_blocks, output_page_offset, <packed stride blocks…>]`. The `input_addr` slot at index
  `grid.x+grid.y` was **patched to an `input_buffer` `Buffer*` binding** (Case 2 base delivery). Kernel reads
  the tail sequentially *and* random-indexes the coord table — genuine **varargs**.
- **Semaphores**: none.

### Planned Spec Shape (Generic)
- **KernelSpecs**: 2 — `reader` (reader-DM-config), `writer` (writer-DM-config), same runtime-selected source.
- **DataflowBufferSpecs**: 1 — `shard` (**borrowed_from = OUTPUT**, `allow_instance_multi_binding = true`),
  `entry_size = output_buffer->page_size()`, `num_entries = min(total_size, packed_output_bytes)/page_size`.
- **TensorParameters**: 2 — `input` (bound by both kernels, Case 2 via `get_bank_base_address`), `output`
  (referenced only by the DFB `borrowed_from`; **no kernel TensorBinding needed** — the validator counts
  `borrowed_from` as a use).
- **WorkUnitSpecs**: 1 — `{reader, writer}` on `all_cores`.

### Dropped Plumbing (Generic)
| legacy | Metal 2.0 replacement |
|---|---|
| `dst_cb_index (=16)` CTA | `DFBBinding{shard}` |
| `input_addr` RTA slot (patched `Buffer*` at index `grid.x+grid.y`) | `TensorAccessor(tensor::input).get_bank_base_address()` (Case 2); host passes `input_addr=0` then `erase()`s the slot |
| positional CTAs | named CTAs (`num_x_cores`, `num_y_cores`, `page_size`, `unit_size`) |
| positional RTA tail via `emplace_runtime_args` | `num_runtime_varargs` (max length) + per-node zero-padded `runtime_varargs` |
| `push_reshard_generic_cb_pair` (builds `CBDescriptor`) | deleted; borrowed `DataflowBufferSpec` |

### Applied Patterns (Generic)
- [Sync-free / single-ended CB → self-loop DFB](../shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  the borrowed c_16 is an address-source only (`get_write_ptr`), touched by 2 co-resident instances → each
  self-loops it (PRODUCER + CONSUMER) with `allow_instance_multi_binding = true` (audit's disposition).
- [Kernel-side whitelist rule 5 — Case 2 raw pointer](../port/metal2_port.md#kernel-side-whitelist): input
  base via `get_bank_base_address`, raw `noc.async_read({.noc_x,.noc_y,.addr=base+off})` walk unchanged.
- Varargs (per [Caution: varargs](../shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)):
  legitimate here — the kernel loop-reads a runtime-count stride tail and random-indexes the coord table.
- hw_config: `ReaderConfigDescriptor{}`/`WriterConfigDescriptor{}` → `create_reader/writer_datamovement_config`.

## Deferred / Flagged
- **Remaining factories (next passes):** 6 instantiations, each a complete sub-port of its own.
  - `NdReshardCopyLocalShardFactory<true/false>` — own-dir kernel (`nd_reshard_copy_local_shards.cpp`,
    shared BRISC+NCRISC), Case 1, **no CB**. Next cleanest.
  - `ReshardSameWidthFactory<true/false>`, `ReshardSameHeightFactory<true/false>` — shared-pool kernels
    (`reshard_same_width_*` / `reshard_same_height_*`), Case 2 remote + borrowed-DFB local (multi-binding),
    varargs; SameWidth`<true>` adds a scratch DFB (self-loop, multi-binding) when unaligned. These reuse the
    Generic patterns proven here (borrowed multi-binding DFB, Case 2, varargs), so they should go faster.
- **New findings during the port:** none structural — the inventory matched the audit's dispositions. The
  audit dispositioned c_16 as multi-binding; realized as a self-loop pair on each of the 2 instances (see
  report). Borrowed-DFB size needed a clamp to the packed output size (padded shards); noted in the report.
