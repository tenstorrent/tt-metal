# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding`

> ## ⚠ SCOPED BRIEF — the op is RED at op level
>
> This is **not** an all-GREEN audit. It is the config-scoped-subset case: the Device 2.0 gate fails on **one** factory, and this brief covers only the four that are clear.
>
> **In scope — port these four factories:**
> - `UntilizeWithUnpaddingSingleCoreProgramFactory`
> - `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory`
> - `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory`
> - `UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory`
>
> **Out of scope — do not touch:**
> - `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` — **blocked on Device 2.0.** It instantiates two kernels still on Device 1.0 free-function idioms: `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` and `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp`. Migrating them is the Device 2.0 team's work and is **off your whitelist**. Leave that factory on the `ProgramDescriptor` path; it ports after a re-audit.
>
> **One open item before you start:** the per-factory readiness sheet could not be fetched during the audit (Google Drive connector unauthorized in a non-interactive session). Every code-checkable conjunct of the TTNN factory-concept gate was verified clean, but the sheet-owned **`Is safe to port?`** call is unread. Confirm that cell reads `yes` for these four factory rows before committing to the port. Details and full reasoning: `METAL2_PREPORT_AUDIT.md`.

**Gates cleared (for the four in-scope factories):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(code cross-check only — see above)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `66ac84052d4 2026-07-27 docs(metal_2.0): split the runtime-args porting gate into its two sheet columns` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — all five factories expose `static tt::tt_metal::ProgramDescriptor create_descriptor(const UntilizeWithUnpaddingParams&, const Tensor&, Tensor&)`. Single `DeviceOperation` (`ttnn::prim::UntilizeWithUnpaddingDeviceOperation`), `tensor_args_t = Tensor`, one output tensor.
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All verified directly in the code; `untilize_with_unpadding_nanobind.cpp` binds only the public `ttnn::untilize_with_unpadding` free function.
- **Mixed-concept transition state to expect:** you are converting four of five factories in a `std::variant` `program_factory_t` (`device/untilize_with_unpadding_device_operation.hpp:24-29`). The fifth stays on `ProgramDescriptor`. `select_program_factory` (`device/untilize_with_unpadding_device_operation.cpp:40-95`) is untouched by the port — its dispatch logic and every `TT_FATAL` in `validate_on_program_cache_miss` stay as they are.

## Construct — to do

**Tensor bindings** (per binding) — **every binding is Case 1**; there is no Case 2 anywhere in the subset, and no borrowed-memory DFB (all four in-scope factories use plain, non-`.buffer` CBs).

Today each tensor reaches its kernel as a `Buffer*` pushed into `KernelDescriptor::RTArgList` (the framework's interim `BufferBinding` hack), and every kernel feeds that base straight into a `TensorAccessor`. For each: express it as a `TensorParameter` / `TensorBinding`, have the kernel build `TensorAccessor(tensor::name)`, and delete both the `Buffer*` RTA and the `TensorAccessorArgs(...).append_to(...)` CTA plumbing.

| Factory | Binding | Host site to remove | Kernel site to convert |
|---|---|---|---|
| SingleCore | `input` | `..._single_core_program_factory.cpp:187` (RTA), `:130` (`TensorAccessorArgs`) | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp:15,25` |
| SingleCore | `output` | `..._single_core_program_factory.cpp:190` (RTA), `:137` | `.../writer_unary_unpad_dims_split_rows.cpp:38,44` |
| MultiCoreInterleaved | `input` | `..._multi_core_interleaved_program_factory.cpp:231` (RTA), `:93` | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp:15,25` |
| MultiCoreInterleaved | `output` | `..._multi_core_interleaved_program_factory.cpp:196` (RTA), `:119` | `.../writer_unary_stick_layout_split_rows_multicore.cpp:30,34` |
| MultiCoreBlockInterleaved | `input` | `..._multi_core_block_interleaved_program_factory.cpp:295` (RTA), `:185` | `eltwise/unary/.../reader_unary_interleaved_wh_multicore.cpp:21,27` |
| MultiCoreBlockInterleaved | `output` | `..._multi_core_block_interleaved_program_factory.cpp:300` (RTA), `:197` | `.../writer_unary_stick_layout_wh_multicore.cpp:21,25` |
| MultiCoreNDSharded | `input` | `..._multi_core_nd_sharded_program_factory.cpp:272` (reader RTA), `:275` (writer RTA arg 1), `:126` and `:187` (two `TensorAccessorArgs` appends) | `data_movement/sharded/.../reader_unary_nd_sharded_blocks.cpp:26,27` **and** `.../writer_..._nd_sharded.cpp:42,43` |
| MultiCoreNDSharded | `output` | `..._multi_core_nd_sharded_program_factory.cpp:275` (writer RTA arg 0), `:185` | `.../writer_..._nd_sharded.cpp:40,41` |

Two things to keep straight in `MultiCoreNDSharded`:

- The **`input` tensor is bound by two kernels** (reader and ND writer, the latter using `accessor_src.shard_pages(shard_id)` to walk its own shard's pages). One `TensorParameter`, two `TensorBinding` consumers — not a raw-pointer escape.
- That writer builds **two** accessors from a chained constexpr offset: `TensorAccessorArgs<17>()` for the destination, then `TensorAccessorArgs<dst_args.next_compile_time_args_offset()>()` for the source (`writer_..._nd_sharded.cpp:40,42`). Once both become `tensor::name` bindings, the chained offset arithmetic disappears with them — do not try to preserve it.

**TensorParameter relaxation:** none. The op has no custom hash, so no relaxation can be active. *(The sheet's column is unread — see the scope banner. If it turns out to name one, stop and re-check against the hash before applying it.)*

**TensorAccessor 3rd arg:** drop the redundant page-size argument at **`.../writer_unary_stick_layout_split_rows_multicore.cpp:34`** — `TensorAccessor(dst_args, dst_addr, writer_page_size)` becomes the 2-arg form. Classified **Class 2 (redundant/inert)**: the value resolves to the output buffer's logical page in every reachable config (full output row for interleaved/HEIGHT-sharded; shard-row bytes for WIDTH/BLOCK-sharded), and the sharded sub-case is exact because `compute_output_specs` rounds every shard width up to a tile multiple (≥64 B, in multiples of 64) — already conformant to the strictest Blackhole DRAM alignment. **Not** Class 1: it is a shard-geometry constant, not a per-shape varying page size, so do **not** set `dynamic_tensor_shape`.

Also drop the now-dead host-side plumbing that fed it: `writer_page_size` and its `if (out_mem_config.is_sharded() && …)` computation at `..._multi_core_interleaved_program_factory.cpp:108-113`, plus the CTA at `:118` and its kernel read at `writer_unary_stick_layout_split_rows_multicore.cpp:29`.

*(A second Class-2 site exists at `writer_unary_unpad_cross_sharded.cpp:35`, but it belongs to the out-of-scope `MultiCoreSharded` factory. Leave it.)*

**CB endpoints:** **all legal (1 producer + 1 consumer)** — nothing to self-loop, nothing to assign, no multi-binding flag, no dead CB to drop. Every in-scope factory is the same clean chain: reader → `c_0` → compute → `c_16` → writer.

The one thing worth *verifying* rather than assuming, because it is what a same-role doubling would look like: `MultiCoreInterleaved` emits 2 compute `KernelDescriptor`s and `MultiCoreBlockInterleaved` up to 4, all sharing `c_0`/`c_16`. Their core ranges are **disjoint by construction** — `split_blocks_for_tilize` and `split_blocks_for_tilize_wh` (`ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp:171` and nearby) build each sub-range from a monotonically advancing `core_index` over one enumerated core list, with `all_cores` defined as their union — so every node carries exactly one compute instance. Preserve that partitioning verbatim when you translate the descriptors; do not merge the cliff ranges into `all_cores`.

## Watch for

- **CB endpoints (multi-binding):** none. The op has **no semaphores at all**, which structurally rules out the hidden-second-writer face, and it never instantiates one kernel source twice over the same core range, which rules out the dual-instance work-split face. You can skip that hunt here.

- **Cross-op / shared kernels — the real coupling risk in this port.** Only 3 of the kernels you touch are owned by this op (`writer_unary_unpad_dims_split_rows.cpp`, `writer_unary_stick_layout_split_rows_multicore.cpp`, `writer_unary_stick_layout_wh_multicore.cpp`, plus `writer_..._nd_sharded.cpp`). The rest are **borrowed by file path, and a Metal 2.0 rewrite of any one of them is a single rewrite every co-borrower must adopt in the same change**:

  | Kernel you must rewrite | Owner | Co-borrowers |
  |---|---|---|
  | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | `eltwise/unary` | **~17 factories** (prod, typecast, `nlp_create_qkv_heads_falcon7b`, pad, …) |
  | `data_movement/untilize/device/kernels/compute/untilize.cpp` | `data_movement/untilize` | **~10 factories** (upsample, fold, untilize ×4, …) |
  | `data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` | `data_movement/untilize` | ~6 factories |
  | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp` | `eltwise/unary` | `data_movement/untilize` block factory |
  | `data_movement/untilize/device/kernels/compute/untilize_wh.cpp` | `data_movement/untilize` | `data_movement/untilize` block factory |
  | `data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` | `data_movement/sharded` | `data_movement/untilize` ND factory |

  `reader_unary_interleaved_start_id.cpp` is the sharp edge — a Case-1 conversion there ripples to ~17 factories across families. Raise it before you rewrite it rather than after.

  Function-call escapes are all clean and need no special handling: `tt::data_movement::common::noc_async_write_sharded(Noc, uint32_t l1_addr, TensorAccessor, …)` (`data_movement/common/kernels/common.hpp:294`) takes a Device 2.0 `Noc` and a `TensorAccessor` by value — pass `TensorAccessor(tensor::name)` straight in; and `compute_kernel_lib::untilize<block_w, input_dfb, output_dfb, …>` (`ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`) takes DFB identity as `uint32_t` NTTPs, which `dfb::name`'s constexpr cast covers.

- **RTA varargs — two genuine blocks. Use the vararg mechanism; do not try to name these.**

  1. **`writer_unary_stick_layout_split_rows_multicore.cpp:73-86` (RTA, `MultiCoreInterleaved`).** `n_block_reps` bounds a loop pulling a 5-tuple `{n_data, n_mixed, n_pads, times, repeat_count}` per group through `rt_arg_idx`, advanced *inside* the loop at `:82`. The group count varies per core with the block assignment (produced at `..._multi_core_interleaved_program_factory.cpp:195-226`). **But name the four leading args** — `dst_addr`, `padded_X_size`, `start_stick_id`, `n_block_reps` (`:19-22`) — they are fixed distinct fields; only the tail block is a vararg. (`dst_addr` disappears entirely, replaced by the `tensor::name` binding.)
  2. **`writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:100-105` (CRTA, `MultiCoreNDSharded`).** Two `get_common_arg_val<uint32_t>(i)` loops bounded by the `tensor_rank` CTA read the output shape dims then the input shape dims (produced at `..._multi_core_nd_sharded_program_factory.cpp:175-183`). A CTA-bounded count still varies across instantiations, so this is a **CRTA vararg**, not an unrolled name set — the kernel-side vararg mechanism supports common runtime args.

  **Non-signal — do not mis-flag:** `writer_unary_stick_layout_wh_multicore.cpp:65-70` re-reads args 2–7 inside the `third_dim` loop, but at **constant** indices. That is a fixed set of distinct fields read repeatedly, not a loop-indexed block. Name each of them.
