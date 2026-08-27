# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/tilize`

> Config-scoped audit: all gates clear for **five of six factories**. Port the subset {Default, SingleCore, Sharded, ShardedRetile, Retile}. **Do not port `TilizeMultiCoreBlockProgramFactory`** — it is blocked by the readiness-sheet `Known op issues = "Per-node CB size"` (ops-team fix); keep its legacy path. The full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared (for the subset):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (5/6) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A)

**Recipe docs:** `7d5ddd43e0e 2026-08-27 docs(metal_2.0): a run in flight freezes the kernel sources` *(carry into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`):

- **Current concept:** `descriptor` (all five subset factories; `create_descriptor` returning `ProgramDescriptor`).
- **Op-owned tensors:** none.
- **Target concept:** **`CustomProgramSpecFactoryConcept`** — every factory defines `override_runtime_arguments`, so each translates into one method returning a `ProgramRunArgs` (per the recipe's *Translating `override_runtime_arguments`* step). Sites: `tilize_multi_core_default_program_factory.cpp:236`, `tilize_single_core_program_factory.cpp`, `tilize_multi_core_sharded_program_factory.cpp:165`, `tilize_multi_core_sharded_retile_program_factory.cpp`, `tilize_multi_core_retile_program_factory.cpp`. All currently delegate to the shared `patch_tilize_kernel_slot0` helper (`tilize_device_operation.cpp:372`); the two sharded factories additionally refresh borrowed-CB base addresses via a throwaway `cb_addr_only` `ProgramDescriptor` + `apply_descriptor_runtime_args` — see CB endpoints below.
- **Gate-cleared, confirmed absent** (each would have blocked): a non-`none` `TensorParameter relaxation`; `get_dynamic_runtime_args`. (A custom hash / pybound `create_descriptor` are also absent here, though neither would gate.)

## Construct — to do

**Tensor bindings** (per binding, per factory):

- **Default** — `input` **Case 1** (via `TensorAccessor` in `reader_unary_stick_layout_split_rows_multicore.cpp`) · `output` **Case 1** (`writer_unary_interleaved_start_id.cpp`).
- **SingleCore** — `input` **Case 1** (`…_singlecore.cpp`) · `output` **Case 1** (`writer_unary_interleaved_start_id.cpp`).
- **Retile** — `input` **Case 1** (`untilize/reader_unary_start_id.cpp`) · `output` **Case 1** (`writer_unary_interleaved_start_id.cpp`).
- **Sharded** — `input` **clean** (borrowed-memory: `cb_src0.buffer = src_buffer`, `reader_unary_sharded.cpp` only handshakes) · `output` **clean** for sharded output (`cb_output.buffer = dst_buffer`, `writer_unary_sharded.cpp`) / **Case 1** for INTERLEAVED output (local CB drained via `TensorAccessor` in `writer_unary_interleaved_start_id.cpp`). Per-config.
- **ShardedRetile** — same as Sharded: `input` clean (borrowed) · `output` clean (borrowed) / Case 1 (interleaved output).

For every Case-1 binding: express it as a `TensorParameter`/`TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`. This **replaces both** today's `Buffer*`-binding-form arg and the manual `patch_tilize_kernel_slot0` slot-0 re-point — the `Buffer*` slot, the `TensorAccessorArgs` plumbing, and the slot-0 patch all disappear (the framework refreshes the binding on cache hit). For every clean (borrowed-memory) binding: translate the buffer-backed CB (`cb.buffer = …`) to `DataflowBufferSpec::borrowed_from`; the sharded factories' `cb_addr_only` + `apply_descriptor_runtime_args` refresh disappears with it.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor passes a 3rd argument.

**CB endpoints** (classify per `(CB, config)`):

- **Default** — `c_0` 1:1 (reader PRODUCER, compute CONSUMER) · `c_16` 1:1 (compute PRODUCER, writer CONSUMER). Compute is two same-source kernels over **disjoint** full/cliff node sets → ordinary 1:1 per node (not multi-binding).
- **SingleCore** — `c_0` 1:1 · `c_16` 1:1.
- **Retile** — `c_0` 1:1 · `c_1` (`mid_cb`) **self-loop** (compute produces via `untilize` + `fill_zeros_pages`, consumes via `wait_front`/`pop_front`) · `c_2` (`mid_view_cb`) **self-loop** (compute-only; no FIFO producer, read cursor set by hand, consumed by `tilize_block`) · `c_16` 1:1.
- **Sharded** — `c_0` 1:1 (borrowed input) · `c_16` 1:1 (borrowed for sharded output, local for interleaved output).
- **ShardedRetile** — `c_0` 1:1 (borrowed) · `mid_cb` self-loop · `mid_view_cb` self-loop · `c_16` 1:1.

## Watch for

- **Aliased intermediate CB — confirm before building (Retile / ShardedRetile).** `mid_cb` (`c_1`) and `mid_view_cb` (`c_2`) are two `CBFormatDescriptor`s in **one** `CBDescriptor` (`tilize_multi_core_retile_program_factory.cpp:136-153`): one L1 allocation, two buffer indices, different tile/face geometry; `retile.cpp` drives `c_2`'s `fifo_rd_ptr` by hand into the shared region (`retile.cpp:108,119`). Not an Appendix A feature, so it does not block — but it is the one non-boilerplate construct here. **Confirm how a `DataflowBufferSpec` expresses two format descriptors over one allocation** (or the equivalent alias) before wiring it; don't assume. Both `mid` views are self-loops (compute is the only toucher).
- **Cross-op / shared kernels — reuse existing `_metal2` forks, do not re-fork:**
  - `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` → bind `tilize_metal2.cpp`
  - `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` → bind `…_metal2.cpp`
  - `untilize/.../reader_unary_start_id.cpp` → bind `…_metal2.cpp`
  - `eltwise/unary/.../reader_unary_sharded.cpp` → bind `…_metal2.cpp`
  - `sharded/.../writer_unary_sharded.cpp` → bind `…_metal2.cpp` (sunset: issue #52228)

  This op is one consumer on each fork's **sunset list** — that is a coordination/retire list, **not** authorization to convert the shared kernel in place. Op-owned kernels (`reader_unary_stick_layout_split_rows_multicore.cpp`, `…_singlecore.cpp`, `retile.cpp`) are ported in place.
- **RTA varargs:** none — name every runtime arg.
- **`retile.cpp` includes `kernel_lib/{tilize,untilize}_helpers.hpp`** (official shared library) — bridge the named tokens into the helper signatures; no donor-side change expected.
- **Do not touch `TilizeMultiCoreBlockProgramFactory`, its kernels (`tilize_wh.cpp`, `reader_unary_pad_multicore_both_dims.cpp`, `writer_unary_interleaved_start_id_wh.cpp`), or the dead `device/kernels/compute/tilize.cpp`.** The block factory stays on the legacy path; the dead file is not yours to port.
