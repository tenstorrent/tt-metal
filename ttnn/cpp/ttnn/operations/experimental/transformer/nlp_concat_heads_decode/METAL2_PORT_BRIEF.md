# Metal 2.0 Port Brief — `experimental/transformer/nlp_concat_heads_decode`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

Porting unit: one DeviceOperation (`NLPConcatHeadsDecodeDeviceOperation`), two factories — `NLPConcatHeadsDecodeProgramFactory` (full-grid) and `NLPConcatHeadsDecodeSubcoregridsProgramFactory` (`on_subcoregrids`). They share the DeviceOperation, the one output CB, and the reader/writer kernel structure; the findings below apply to **both** unless noted. Each factory has its own kernel file (`reader_tm_tile_layout_nlp_concat_heads_decode.cpp` vs. `..._subcoregrid.cpp`) — same shape, port both.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (both factories) — `create_descriptor()` returning a `ProgramDescriptor`.
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no`. No other migration-risky pybind (`Is safe to port? == yes`).

## Construct — to do

**Tensor bindings** (per binding, both factories):

- `input` — **Case 2** (raw pointer). Today the base is delivered via a `Buffer*` pushed into the RTA list (`in_buffer`), and the kernel uses the base raw: `q_start_addr = get_arg_val<uint32_t>(1)` → `qkv_read_addr = q_start_addr + in_tile_offset_by_head` → `noc.async_read(.addr = qkv_read_addr, .noc_x, .noc_y)`. **Do:** express `input` as a `TensorParameter` / `TensorBinding`; pull the base via `TensorAccessor::get_bank_base_address` and **keep the raw NoC arithmetic and the hand-rolled cross-core gather unchanged**. Do not convert the raw walk to `TensorAccessor` iteration. The `Buffer*` RTA and the base `get_arg_val(1)` both disappear.
  - Host: `device/nlp_concat_heads_decode_program_factory.cpp:130`; `device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp:137`.
  - Kernel: `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode.cpp:18,45,57,67`; `..._subcoregrid.cpp:16,44,56,65`.
- `output` — **clean** (borrowed-memory DFB). The output CB `c_16` is `.buffer = output.buffer()`; kernels write via `cb_q_out.get_write_ptr()`. **Do:** port via `DataflowBufferSpec::borrowed_from` (the borrowed-memory translation). No `get_bank_base_address` bridge here.
  - `device/nlp_concat_heads_decode_program_factory.cpp:46-55` (`.buffer` at `:54`); `device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp:56-65` (`.buffer` at `:64`).

**Note on the head offset (not a binding, not a fold):** the per-core byte offset `in_tile_offset_by_batch` is a *separate* scalar RTA (arg index 0), added to the base in the kernel. It stays an ordinary (named) runtime arg — this is the already-split-out shape, not an offset-base-pointer fold. Nothing to do beyond naming it.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no `TensorAccessor` in either kernel.

**CB endpoints:** `c_16` (both factories, both configs) → **assign 1P+1C**. The reader-config and writer-config instances (same kernel source over the same `q_cores`) both raw-write disjoint tile phases via `cb_q_out.get_write_ptr()`; nothing drains it (it is the output). Bind one instance PRODUCER, the other CONSUMER — cosmetic on Gen1. **Do not** reach for the multi-binding advanced option: there is no third toucher and no FIFO-role doubling.
  - `device/nlp_concat_heads_decode_program_factory.cpp:45-55` (CB), `:93-112` (reader+writer).
  - `device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp:55-65` (CB), `:100-119` (reader+writer).

## Watch for

- **CB endpoints (multi-binding):** none. The output CB is the dual-instance work-split (fully visible — no hidden second writer, no semaphore-gated co-fill). Resolve as 1P+1C, not multi-binding.
- **Cross-op / shared kernels:** none — both kernels are owned by this op; no out-of-directory `#include`s (only `api/*`), no file-path kernel borrows. No port-together coupling.
- **RTA varargs:** the two NoC-coordinate blocks `noc_x_coords` / `noc_y_coords` are **variable-count** (CTA-bounded by `in_num_cores_x`/`in_num_cores_y`, or `in_num_cores` in the subcoregrids factory) and vary with the input shard grid. The kernel reads them as L1 arrays via `get_arg_addr` and indexes `in0_mcast_noc_x[...]` / `in0_mcast_noc_y[...]`. **Port them as RTA varargs** (kernel-side vararg mechanism), not as individually-named args. The offset (arg 0) and the base (arg 1 → tensor binding) are the only named/bound leading args.
  - `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode.cpp:31-32`; `..._subcoregrid.cpp:31-32`.
