# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/copy`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A — no accessor passes a 3rd arg)

**Recipe docs:** `c16f21b8cb6 2026-08-18 docs(metal_2.0): unpack_modes -- the trigger is the buffer format, not the dtypes` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

One DeviceOperation — `CopyDeviceOperation` — with three factories, all porting to `ProgramSpecFactoryConcept`. These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`):

- **Current concept:** `descriptor` (all three factories: `SameMemoryConfig`, `DefaultRowMajor`, `DefaultTilized`). Each defines `create_descriptor()` returning a `ProgramDescriptor` (`device/copy_device_operation.hpp:24-42`).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (`Override runtime args? == no` → the framework refreshes tensor bindings on cache hit; the factory writes one method).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none` `TensorParameter relaxation`; `get_dynamic_runtime_args`. There is also **no** custom `compute_program_hash`, **no** `override_runtime_arguments`, and **no** pybound `create_descriptor` — none of those gate, and none is present here, so no hash to preserve, no override method to translate, no pybind binding to delete.

## Construct — to do

**Tensor bindings** — all **Case 1** (`Buffer*` base delivered via runtime arg → consumed through `TensorAccessor`). For each, replace the `Buffer*` runtime-arg with a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)` and both the address-via-RTA and the `TensorAccessorArgs(...).append_to(...)` CTA plumbing disappear.

- `SameMemoryConfig` — **input** (`src_buffer`, RTA `Buffer*` @ `copy_same_memory_config_program_factory.cpp:169,180`) and **output** (`dst_buffer` @ `:173,186`). Kernels: `reader_unary_start_id.cpp` / `reader_unary_stick_start_id.cpp` (reader) and `writer_unary_start_id.cpp` / `writer_unary_stick_start_id.cpp` (writer), each `TensorAccessor(args, addr)`.
- `DefaultRowMajor` — **input** (RTA `Buffer*` @ `copy_default_row_major_program_factory.cpp:172`, CTA args @ `:136`) and **output** (RTA @ `:173`, CTA args @ `:148`). Kernels: `redistribute_pages_row_major_reader.cpp:38` / `redistribute_pages_row_major_writer.cpp:31`.
- `DefaultTilized` — **input** (RTA `Buffer*` @ `copy_default_tilized_program_factory.cpp:144`, CTA args @ `:105`) and **output** (RTA @ `:145`, CTA args @ `:110`). Kernels: the `eltwise/unary` interleaved reader/writer (bind their `_metal2` forks — see Watch for).

No Case 2 (raw pointer) bindings, and no borrowed-memory DFB reads (the op calls `set_globally_allocated_address` nowhere).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor passes one; nothing to drop.

**CB endpoints:**
- **Self-loop** `DefaultRowMajor` c_0 (all configs): the reader alone touches it as an L1 scratchpad (`reserve_back`/`get_write_ptr`/`push_back`/`wait_front`/`pop_front` at `redistribute_pages_row_major_reader.cpp:42,61,183-185`); the writer never references it. Bind the reader as **both** PRODUCER and CONSUMER.
- **All other CBs are legal 1:1** — one producer, one consumer — across every config (see the per-`(CB, config)` table in `METAL2_PREPORT_AUDIT.md`). `SameMemoryConfig` c_0 (+ c_16 on convert-dtype), `DefaultRowMajor` c_1, `DefaultTilized` c_0 (+ c_16 on convert-dtype) need no special action.

## Watch for

- **CB endpoints (multi-binding):** none — no multi-binding, no hidden second writer to hunt.
- **Cross-op / shared kernels** — file-path borrows, with fork status. The `SameMemoryConfig` row-major-interleaved and `DefaultTilized` paths bind shared kernels; `DefaultRowMajor` borrows none (own kernels only).
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` — **`_metal2` fork already exists** (`reader_unary_interleaved_start_id_metal2.cpp`): **bind it, do not re-fork.**
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — **`_metal2` fork already exists** (`writer_unary_interleaved_start_id_metal2.cpp`): **bind it.**
  - `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp` — **no fork yet**; this port creates it beside the original. Sunset list (**not** authorization to convert in place): `embedding`, `data_movement/concat`.
  - `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` — **no fork yet**; create it. Sunset list: `embedding`, `data_movement/concat`.
  - `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` — **no fork yet**; create it. Sunset list: `sharded_to_interleaved`, `sharded_to_interleaved_partial`, `untilize_with_unpadding`.
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp` (in-family) — **no fork yet**; create it. Sunset list: `interleaved_to_sharded`, `interleaved_to_sharded_partial`.
- **RTA varargs:** none — every kernel reads a fixed set of named runtime args by constant index. Prefer named RTAs throughout.
- **Function-call escape:** `redistribute_pages_row_major_reader.cpp:11` calls `tt::data_movement::common::tt_memmove(Noc, …)` (`data_movement/common/kernels/common.hpp`) — Device 2.0-native signature (takes `Noc`), bridges cleanly; no donor change.
- **Dead CTA (optional cleanup, ops-team item — not porter work):** `redistribute_pages_row_major_reader.cpp:24` declares `num_output_pages_in_row` (CTA 2, emitted by the factory at `copy_default_row_major_program_factory.cpp:127`) but never uses it. Carry it forward as-is unless the ops team prunes it; do not change behavior.
