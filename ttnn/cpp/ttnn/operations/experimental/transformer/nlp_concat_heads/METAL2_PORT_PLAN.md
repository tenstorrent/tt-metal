# Port Plan — nlp_concat_heads

Port plan for `nlp_concat_heads`, from legacy `ProgramDescriptorFactoryConcept` (`create_descriptor`) to
Metal 2.0 `ProgramSpecFactoryConcept` (`create_program_spec`).

**Outcome: PORTED (both paths).** On revisit, the interleaved path's cross-op writer was forked
(`writer_unary_interleaved_start_id_metal2.cpp`, in this op's own dir) and converted, so the whole factory
moved to `create_program_spec`. Sharded path = borrowed-memory fake-CB self-loops (reader=PRODUCER /
writer=CONSUMER); interleaved path = clean Case-1 `TensorAccessor` + the forked writer. Verified:
`test_nlp_concat_heads.py` → 217 passed on Blackhole p150b. See `METAL2_PORT_REPORT.md`. The inventory below
documents both paths.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`NLPConcatHeadsProgramFactory::create_descriptor`).
- Variants: single factory, but with **runtime kernel-source selection** on `input.is_sharded()` — two
  internal paths (sharded / interleaved), each selecting different kernel sources. Per the recipe, a single
  runtime-source-selecting factory + **all** sources it can select convert together; there is no
  "port one path only" sub-target.
- Custom `compute_program_hash`: none — already default reflection hash.

### Kernels

**SHARDED path** (selected when `input.is_sharded()`):

| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` (op-owned) | `all_cores` (= shard grid) | `{src0_cb_index, out_cb_index, in0_h_tiles, in0_w_tiles*tile_size, nbpcg1*in0_w_tiles*tile_size, nbpcg1*in0_HtWt}` | `{nheads_first_risc, 0, 0}` | ReaderConfigDescriptor |
| writer | **same source** | `all_cores` | same CTAs | `{nheads_second_risc, nheads_first_risc*in0_HtWt*tile_size, nheads_first_risc*in0_w_tiles*tile_size}` | WriterConfigDescriptor |

(`nbpcg1` = `num_blocks_per_core_group_1` = `shard.shape[0]/padded[-2]`; `nheads_first_risc = div_up(nbpcg1,2)`,
`nheads_second_risc = nbpcg1 - nheads_first_risc`.) Kernel CTA names: `cb_id_in0, cb_id_out0, in0_h_tiles,
head_dim_size_bytes, out_row_size_bytes, block_size`.

**INTERLEAVED path** (selected when `!input.is_sharded()`):

| unique_id | source | core_ranges | CTAs | RTAs | config |
|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` (op-owned) | `all_cores` | `{in0_h_tiles, in0_w_tiles, in0_c, in0_HtWt}` + `TensorAccessorArgs(in0_buffer)` | `{in0_buffer(addr), num_blocks, in0_h_dim, in0_tensor_tile_id}` | ReaderConfigDescriptor |
| writer | **`eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`** (**CROSS-OP**) | `all_cores` | `{src0_cb_index}` + `TensorAccessorArgs(out_buffer)` | `{out_buffer(addr), num_pages, start_id}` | WriterConfigDescriptor |

### CBs

| index | total_size | core_ranges | data_format | page_size | `.buffer` |
|---|---|---|---|---|---|
| 0 (src0) | `per_tensor_tiles * tile_size` (×2 if interleaved, double-buffer) | `all_cores` | `datatype_to_dataformat(input.dtype())` | `single_tile_size` | `in0_buffer` if sharded, else `nullptr` |
| 16 (out) | `per_tensor_tiles * tile_size` | `all_cores` | same | `single_tile_size` | `out_buffer` (only emitted when `out_sharded`) |

Both populated CBs are **borrowed-memory + address-source-only (fake) CBs** — read/written by base pointer
(`get_read_ptr`/`get_write_ptr`), no FIFO produce/consume.

### Semaphores
none

### Tensor accessors

| host site | originating Tensor | RTA slot | kernel |
|---|---|---|---|
| `create_descriptor` interleaved reader CTA append (`TensorAccessorArgs(*in0_buffer)`) | input | reader RTA slot 0 (`in0_buffer` addr) | op-owned reader — **Case 1, portable** |
| `create_descriptor` interleaved writer CTA append (`TensorAccessorArgs(*out_buffer)`) | output | writer RTA slot 0 (`out_buffer` addr) | **cross-op writer — BLOCKER** |

(Sharded path: no TensorAccessor — NoC-local addressing only.)

### Work split
- Sharded: `all_cores = input.shard_spec().grid`; `core_group_1 = all_cores`, `core_group_2` empty;
  `num_blocks_per_core_group_1 = shard.shape[0]/padded[-2]`. Single group.
- Interleaved: `split_work_to_cores(compute_with_storage_grid_size, num_blocks)` →
  `(num_cores, all_cores, core_group_1, core_group_2, nbpcg1, nbpcg2)`. Possible two groups.

### Cross-op kernels
- **`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`**
  — interleaved-path writer. **31 co-borrower factories** across the tree (typecast, bcast, concat, copy,
  permute, reshape, slice, tilize/untilize, transpose, embeddings, attn_matmul, matmul, reduce, prod, kv_cache,
  examples, gelu/tanh backward, nlp_concat_heads_boltz, …). Editing it is forbidden (breaks all of them);
  forking it is forbidden by the orchestrator's cross-op rule. This is the blocker.

### Flags
- The op directory has no unreferenced kernel files; both op-owned kernels are referenced.
- `nlp_concat_heads_boltz` is a sibling op that *also* borrows the same cross-op writer — same blocker will
  apply there.

## TTNN ProgramFactory
- **Concept (target)**: `ProgramSpecFactoryConcept` (`create_program_spec`).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: blocked — not realized. See report.

## Planned Spec Shape

*Not constructed — the port did not proceed.* The shape the sharded path *would* have taken (recorded so the
eventual port can pick it up):

- KernelSpecs: 2 (reader RISC + writer RISC of the one sharded source) — sharing the two borrowed DFBs.
- DataflowBufferSpecs: 2 borrowed-memory (`src0` `borrowed_from = INPUT`; `out` `borrowed_from = OUTPUT`),
  each bound as a **fake-CB self-loop** (PRODUCER+CONSUMER) — exactly like the reference `q_out`.
- SemaphoreSpecs: 0.
- TensorParameters: `INPUT` (+ `OUTPUT` for borrowing).
- WorkUnitSpecs: 1 (all_cores).

The interleaved path's spec shape is moot until the cross-op writer is Metal-2.0-ready.

## Preserved Multiplicity
Sharded path: two `KernelDescriptor`s of one source (reader RISC / writer RISC) → two `KernelSpec`s of that
source sharing both borrowed DFBs. (Not realized — blocked.)

## Dropped Plumbing
*(Planned, not realized.)*

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| sharded reader/writer CTA 0,1 | `src0_cb_index`, `out_cb_index` (magic CB indices) | `DFBBinding`s |
| interleaved reader CTA append | `TensorAccessorArgs(in0_buffer)` + addr RTA | `TensorBinding(INPUT)` (Case 1) |
| interleaved writer CTA + RTA | `TensorAccessorArgs(out_buffer)` + addr RTA + `cb` CTA | **cannot be replaced without editing the cross-op writer — BLOCKER** |

## Applied Patterns
- *(Would apply on the sharded path)* Borrowed-memory DFB; Fake CB → self-loop DFB; two `KernelDescriptor`s of
  one source → two `KernelSpec`s. None realized.

## Deferred / Flagged
- **Blocker (new finding during planning):** the single runtime-source-selecting factory's interleaved path
  binds the cross-op `writer_unary_interleaved_start_id.cpp`, which a Metal 2.0 factory cannot drive unmodified
  (positional CTA + `TensorAccessorArgs` in CTAs + buffer-address RTA). Grounded stop. See report.
