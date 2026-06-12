# Pre-Port Audit — nlp_concat_heads

Feasibility audit for porting `experimental/transformer/nlp_concat_heads` from the legacy
`ProgramDescriptorFactoryConcept` (`create_descriptor`) to the Metal 2.0 `ProgramSpecFactoryConcept`
(`create_program_spec`).

## Overall status: **GREEN — PORTED** (cross-op writer resolved by fork; see update note)

> **UPDATE (revisit):** Originally audited RED on the assumption the cross-op interleaved writer could be
> neither edited nor forked. The patterns catalog *does* sanction a `_metal2` fork for exactly this case, so on
> revisit the op was fully ported — the interleaved writer was forked into the op's own dir and converted, and
> the whole factory now satisfies `ProgramSpecFactoryConcept`. **Verified: `test_nlp_concat_heads.py` → 217
> passed on Blackhole p150b.** See `METAL2_PORT_REPORT.md`. Original RED analysis retained below for the record.

### Original analysis (superseded)

The op is structurally simple and the *sharded* path is cleanly portable in isolation. But the op has a
**single factory with runtime kernel-source selection** across two paths (sharded / interleaved), and the
**interleaved path binds a cross-op shared writer** (`eltwise/unary/.../writer_unary_interleaved_start_id.cpp`)
that cannot be driven by a Metal 2.0 factory unmodified, and which the porter may not edit or fork. Because a
single runtime-source-selecting factory must convert as a whole (the spec's bindings must match *every*
selectable source), the whole factory is blocked. Detail and the grounded-stop reasoning are in
`METAL2_PORT_REPORT.md` → *Successful failure*.

## Device-operation shape

- Device op: `NLPConcatHeadsDeviceOperation` (`device/nlp_concat_heads_device_operation.hpp`).
  - `operation_attributes_t = NlpConcatHeadsParams` (`{ MemoryConfig output_mem_config; }`).
  - `tensor_args_t = Tensor` (the single input tensor — *not* a struct).
  - `program_factory_t = std::variant<NLPConcatHeadsProgramFactory>` — **one** factory.
- One factory: `NLPConcatHeadsProgramFactory::create_descriptor(const NlpConcatHeadsParams&, const Tensor& input, Tensor& output)`
  in `device/nlp_concat_heads_program_factory.cpp`.
- **No custom `compute_program_hash`** — default reflection hash (`grep` in the device-op `.cpp` → none).
- **No pybound `create_descriptor` / factory-innards binding** — `nlp_concat_heads_nanobind.cpp` exposes only
  the user-facing op (`grep create_descriptor|create_program` → none).
- `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors` are standard and untouched
  by a port.

## Factory: runtime kernel-source selection (`if (in_sharded)`)

The factory selects its kernel **sources** at runtime on `input.is_sharded()`:

### SHARDED path (op-owned kernels only — portable in isolation)
- Reader `KernelDescriptor`: `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`
  (op-owned), `ReaderConfigDescriptor`.
- Writer `KernelDescriptor`: the **same** source, `WriterConfigDescriptor`. (One source, two descriptors —
  reader RISC / writer RISC of the same kernel.)
- CTAs (positional, 6): `src0_cb_index`, `out_cb_index`, `in0_h_tiles`, `in0_w_tiles*single_tile_size`,
  `num_blocks_per_core_group_1*in0_w_tiles*single_tile_size`, `num_blocks_per_core_group_1*in0_HtWt`.
  (The kernel reads CTAs 0..5 as `cb_id_in0`, `cb_id_out0`, `in0_h_tiles`, `head_dim_size_bytes`,
  `out_row_size_bytes`, `block_size`.)
- RTAs: reader `{nheads_first_risc, 0, 0}`; writer `{nheads_second_risc, read_off_bytes, write_off_bytes}`.
- CBs: `src0` (c_0, `.buffer = in0_buffer` — borrowed), `out` (c_16, `.buffer = out_buffer` — borrowed,
  emitted only when `out_sharded`). Both are *tensor-local-view* borrowed CBs (kernel reads/writes by base
  pointer via `get_read_ptr`/`get_write_ptr`; no FIFO produce/consume) → would become **borrowed-memory DFBs**
  bound as **fake-CB self-loops**, exactly like the `nlp_concat_heads_decode` reference's `q_out`.
- No TensorAccessor in the sharded kernel — all addressing is NoC-local (`my_x`/`my_y`, `src_ep`).

This path uses **only op-owned kernels** and maps onto documented patterns (borrowed-memory DFB + fake-CB
self-loop, two `KernelDescriptor`s of one source → two `KernelSpec`s). **It would be a clean port on its own.**

### INTERLEAVED path (cross-op writer — BLOCKED)
- Reader `KernelDescriptor`: `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` (op-owned),
  `ReaderConfigDescriptor`. CTAs `{in0_h_tiles, in0_w_tiles, in0_c, in0_HtWt}` + `TensorAccessorArgs(in0_buffer)`.
  RTA `{in0_buffer (address), num_blocks, in0_h_dim, in0_tensor_tile_id}`.
- Writer `KernelDescriptor`: **`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/`**
  **`writer_unary_interleaved_start_id.cpp`** — **NOT in the op directory; a cross-op shared kernel**
  (31 co-borrowers, see report). CTAs `{src0_cb_index}` + `TensorAccessorArgs(out_buffer)`.
  RTA `{out_buffer (address), num_pages, start_id}`.
- CBs: `src0` only (c_0, interleaved → `.buffer = nullptr`, double-buffered).

## TensorAccessor handling

- **Sharded path**: none. NoC-local addressing only. Clean.
- **Interleaved reader** (op-owned): legacy `TensorAccessorArgs<4>()` + `get_arg_val<uint32_t>(0)` address RTA.
  This is a **Case 1** binding — would re-express cleanly as a `TensorParameter` + `TensorBinding` and
  `TensorAccessor(ta::input)` kernel-side. Portable in isolation.
- **Interleaved writer** (cross-op `writer_unary_interleaved_start_id.cpp`): the **blocker**. The kernel reads
  `cb_id_out = get_compile_time_arg_val(0)` (positional CTA), `dst_args = TensorAccessorArgs<1>()`
  (CTA-baked accessor args), and `dst_addr = get_arg_val<uint32_t>(0)` (buffer-address RTA). A Metal 2.0
  factory emits **no positional CTAs**, **cannot bake `TensorAccessorArgs` into CTAs**, and **does not thread a
  buffer address through an RTA** — those are exactly the legacy plumbing patterns `TensorBinding` replaces.
  The implicit `dfb::name → uint32_t` conversion would rescue *only* the CB-id CTA; it does nothing for the
  `TensorAccessorArgs` / address-RTA dependence. So this kernel **cannot be used unmodified** by a Metal 2.0
  factory. Porting it would require editing (or forking) the shared kernel — out of scope by the cross-op /
  scope-boundary rules. **RED.**

## Semaphores / scratch tensors / per-execution CB updates

- No semaphores in either path.
- No op-owned scratch tensors.
- No per-execution CB size mutation (`UpdateCircularBuffer*`).
- No aliased CBs (each CB descriptor has a single `buffer_index`).

## Borrowed-memory CBs (FYI-P)

Both sharded CBs (`src0` `.buffer = in0_buffer`; `out` `.buffer = out_buffer`) and the interleaved-shared `out`
when sharded are borrowed-memory, address-source-only (fake) CBs — same family as the reference op's `q_out`.
Not gating; documented for the eventual scratchpad / local-`TensorAccessor` migration.

## TTNN ProgramFactory (target concept)

- Concept the port *would* target: `ProgramSpecFactoryConcept` (`create_program_spec`), per the reference port
  `nlp_concat_heads_decode`.
- Forced device-op edits: none (no custom hash, no pybound factory entry point).
- **Gate result: the port does not proceed** — the single runtime-source-selecting factory cannot convert as a
  whole without touching the cross-op writer. See `METAL2_PORT_REPORT.md`.
