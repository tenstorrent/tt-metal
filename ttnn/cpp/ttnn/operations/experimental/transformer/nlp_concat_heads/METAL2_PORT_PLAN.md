# Port Plan — nlp_concat_heads

Port plan for `experimental/transformer/nlp_concat_heads`, ported from the TTNN
`ProgramDescriptor` (`create_descriptor`) concept to Metal 2.0
(`MetalV2FactoryConcept` / `create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `NLPConcatHeadsProgramFactory::create_descriptor`
  returns a `tt::tt_metal::ProgramDescriptor` (`nlp_concat_heads_program_factory.hpp:15`, `.cpp:19`).
- Variants: single factory, but with an internal `if (in_sharded)` branch selecting one of two
  code paths (**interleaved** vs **sharded**) — different kernels, CBs and bindings per path.
  Because the factory selects its kernel *sources* at runtime, both paths (all three kernel
  sources) convert together atomically.
- Custom `compute_program_hash`: none — already default reflection-based hash (audit confirmed).

*(Target Metal 2.0 concept chosen in the audit: `MetalV2FactoryConcept`. Carried forward below.)*

### Kernels

**Interleaved path**

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | defines | config |
|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` (in-op) | `all_cores` | `{in0_h_tiles, in0_w_tiles, in0_c, in0_HtWt}` + `TensorAccessorArgs(in0_buffer)`@4 | — | `{in0_buffer(addr), num_blocks, in0_h_dim, in0_tensor_tile_id}` | none | `ReaderConfigDescriptor{}` |
| writer | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**cross-op donor**) | `all_cores` | `{src0_cb_index=0}` + `TensorAccessorArgs(out_buffer)`@1 | — | `{out_buffer(addr), num_pages, start_id}` | none | `WriterConfigDescriptor{}` |

**Sharded path** — same source instantiated twice (dual-instance work-split)

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader (reader-cfg instance) | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` (in-op) | `all_cores` | `{src0_cb_index=0, out_cb_index=16, in0_h_tiles, head_dim_size_bytes, out_row_size_bytes, block_size}` | `{nheads_first_risc, 0, 0}` | none | `ReaderConfigDescriptor{}` |
| writer (writer-cfg instance) | same source | `all_cores` | same CTAs | `{nheads_second_risc, start_read_off, start_write_off}` | none | `WriterConfigDescriptor{}` |

### CBs

**Interleaved**: `cb_src0` (index 0), `total_size = per_tensor_tiles*2*single_tile_size`
(double-buffered), `data_format = cb_data_format`, `page_size = single_tile_size`, no borrowed buffer.
Output is interleaved → CB 16 not allocated.

**Sharded**:
- `cb_src0` (index 0): `total_size = per_tensor_tiles*single_tile_size` (sharded `per_tensor_tiles`),
  `.buffer = in0_buffer` (**borrowed memory**).
- `cb_out0` (index 16): `total_size = per_tensor_tiles*single_tile_size`, `.buffer = out_buffer`
  (**borrowed memory**), allocated under `if (out_sharded)`.

No `address_offset`, no `.tile` set on any format descriptor (default 32×32 tiles).

### Semaphores
none — the op declares no semaphores.

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| interleaved reader `reader_tm_tile_layout_nlp_concat_heads.cpp:31` | input | reader RTA slot 0 (`in0_buffer`, `program_factory.cpp:201`) |
| interleaved donor writer `writer_unary_interleaved_start_id.cpp:31` | output | writer RTA slot 0 (`out_buffer`, `program_factory.cpp:210`) |
| sharded path | — | none (input/output reached via borrowed CBs, no `TensorAccessor`) |

### Work split
- Interleaved: `split_work_to_cores(compute_with_storage_grid_size, num_blocks)` →
  `(num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2)`.
- Sharded: `all_cores = a.shard_spec().grid`; `num_blocks_per_core_group_1 = shard_h / padded_h`;
  per-core work split across the two RISCs via `nheads_first_risc = div_up(n,2)`,
  `nheads_second_risc = n - nheads_first_risc`.

### Cross-op kernels
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
  — **broadly-shared donor** (46 op `.cpp` files reference it). Caution case → **fork with `_metal2`
  suffix** (consumers cannot all co-migrate). Recorded in the report under "Open items for downstream".

### Flags
- The sharded kernel calls `cb_in0.reserve_back` / `cb_out0.reserve_back` with **no** matching
  `push_back` (one marked `// Redundant`, the other's `push_back` commented out). Audit: these are
  role-free no-ops on borrowed CBs; do **not** treat as producer-locking. Preserve verbatim (whitelist:
  `reserve_back`→`reserve_back`).
- Sharded `cb_out0` (index 16) is bound unconditionally by the kernel but allocated only under
  `if (out_sharded)` in the legacy factory. Audit confirmed sharded-in ⇒ sharded-out in practice
  (borrowed output CB requires L1-sharded output). See Deferred / Flagged.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: single factory struct `NLPConcatHeadsProgramFactory` keeps its single-type
  `program_factory_t = std::variant<NLPConcatHeadsProgramFactory>`; `create_descriptor` is replaced by
  `create_program_artifacts(const NlpConcatHeadsParams&, const Tensor& input, Tensor& output)`. The
  internal `if (in_sharded)` branch is preserved — each branch builds a complete `ProgramSpec` +
  `ProgramRunArgs` and returns `ProgramArtifacts`. No device-op-class edits are forced (no custom hash,
  no pybind `create_descriptor`, no pybind-hook-only parameter).

## Planned Spec Shape

### Interleaved path
- KernelSpecs: `reader` (in-op reader), `writer` (forked `_metal2` donor).
- DataflowBufferSpecs: `SRC0` (index-0 CB) — `entry_size=single_tile_size`,
  `num_entries=per_tensor_tiles*2`, `data_format_metadata=cb_data_format`. Reader PRODUCER,
  writer CONSUMER (plain 1:1). Not borrowed.
- SemaphoreSpecs: none.
- TensorParameters: `INPUT` (input, reader TensorBinding, Case 1), `OUTPUT` (output, writer
  TensorBinding, Case 1).
- WorkUnitSpecs: one — `{reader, writer}` on `all_cores`.

### Sharded path
- KernelSpecs: `reader` (reader-cfg instance) and `writer` (writer-cfg instance), **same source**,
  same accessor names, differing only in `hw_config` (reader vs writer default) and per-instance RTAs.
- DataflowBufferSpecs: `SRC0` (`borrowed_from=INPUT`, `entry_size=single_tile_size`,
  `num_entries=per_tensor_tiles`), `OUT0` (`borrowed_from=OUTPUT`, same sizes).
- SemaphoreSpecs: none.
- TensorParameters: `INPUT`, `OUTPUT` — **borrowed-only** (no `TensorBinding`; the borrowed DFB is the
  "use", and each is bound in `tensor_args`). Confirmed valid via the `binary_ng` Metal 2.0 port
  (borrowed operand needs no `TensorBinding`).
- WorkUnitSpecs: one — `{reader, writer}` on `all_cores` (both instances share the grid; on each node
  one reader instance + one writer instance).

### Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| sharded reader-cfg + writer-cfg (`reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`) | `reader`, `writer` | `main` (both on `all_cores`) | `SRC0`: reader=PRODUCER, writer=CONSUMER (1P+1C). `OUT0`: reader=PRODUCER, writer=CONSUMER (1P+1C). |

Interleaved path has no multi-`KernelDescriptor` work-split (distinct reader/writer sources).
The sharded 1P+1C is the two-toucher dual-instance work-split — **not** the `allow_instance_multi_binding`
flag (re-derived from the census: zero `push_back`/`pop_front` on either CB → role-free; label cosmetic
on Gen1).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| interleaved reader CTAs 0..3 (`program_factory.cpp:111-116`) | positional CTAs | named CTAs `in0_h_tiles,in0_w_tiles,in0_c,in0_HtWt` |
| interleaved reader CTA@4 (`program_factory.cpp:117`) | `TensorAccessorArgs(*in0_buffer)` | `TensorBinding(INPUT)` — kernel `TensorAccessor(tensor::in0)` |
| interleaved reader RTA slot 0 (`program_factory.cpp:201`) | `in0_buffer` (`Buffer*`→addr) | `TensorBinding(INPUT)` (auto-injected addr) |
| interleaved reader RTAs 1..3 | positional RTAs | named RTAs `num_blocks,in0_h_dim,in0_tensor_tile_id` |
| interleaved writer CTA 0 (`program_factory.cpp:118`) | `src0_cb_index=0` (magic CB idx) | `DFBBinding(SRC0, CONSUMER)` |
| interleaved writer CTA@1 (`program_factory.cpp:119`) | `TensorAccessorArgs(*out_buffer)` | `TensorBinding(OUTPUT)` — kernel `TensorAccessor(tensor::output)` |
| interleaved writer RTA slot 0 (`program_factory.cpp:210`) | `out_buffer` (`Buffer*`→addr) | `TensorBinding(OUTPUT)` |
| interleaved writer RTAs 1..2 | positional RTAs | named RTAs `num_pages,start_id` |
| sharded CTA 0/1 (`program_factory.cpp:88-89`) | `src0_cb_index=0`, `out_cb_index=16` (magic CB idxs) | `DFBBinding(SRC0)`, `DFBBinding(OUT0)` |
| sharded CTAs 2..5 | positional CTAs | named CTAs `in0_h_tiles,head_dim_size_bytes,out_row_size_bytes,block_size` |
| sharded RTAs 0..2 | positional RTAs | named RTAs `nheads,start_read_offset_bytes,start_write_offset_bytes` |

TensorAccessor 3rd argument: none (no accessor passes a 3rd arg). Semaphore-ID RTAs: none.

## Applied Patterns
- [Two-toucher DFB → assign 1P+1C](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md):
  sharded `SRC0` / `OUT0` (dual-instance work-split).
- [Borrowed-memory DFB]: sharded `SRC0` (`borrowed_from=INPUT`), `OUT0` (`borrowed_from=OUTPUT`).
- [Modifying a shared dataflow kernel → Fork with `_metal2`]: interleaved donor writer.
- Runtime kernel-source selection (`if (in_sharded)`) preserved inside `create_program_artifacts`.

## Deferred / Flagged
- **Sharded `cb_out0` unconditional bind vs conditional alloc.** The sharded kernel references `cb_out0`
  (→ `OUT0`, `borrowed_from=OUTPUT`) unconditionally; the legacy factory allocated CB 16 only under
  `if (out_sharded)`. The kernel has no `#ifdef` gate on `cb_out0`, so a faithful port allocates `OUT0`
  unconditionally within the sharded branch (the effective sharded-in ⇒ sharded-out invariant the audit
  confirmed). Introducing an `#ifdef` gate would be *more* than the legacy kernel did, and adding a host
  assert is out of the port's scope. Reproduced faithfully; flagged to the ops team in the report.
