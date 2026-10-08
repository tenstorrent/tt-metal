# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user override: live readiness sheet not fetched; stale Sep 4 copy ignored at user direction; see the audit's Result)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`. `create_descriptor` is a static member of the device-op itself, `DeepseekMoEPostCombineTilizeDeviceOperation::create_descriptor` (`device/deepseek_moe_post_combine_tilize_device_operation.hpp:27`, defined at `device/deepseek_moe_post_combine_tilize_program_factory.cpp:23`). There is no separate factory struct and no `program_factory_t`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. There's no `override_runtime_arguments`, so the framework refreshes bindings on a cache hit.
- **Custom hash:** none. **Pybind `create_descriptor`:** none (`deepseek_moe_post_combine_tilize_nanobind.cpp:40` binds only the public function), so there's no user-visible binding removal.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer (stale sheet: `none`) · `get_dynamic_runtime_args` (absent in code).

## Construct — to do

**Tensor bindings** (per binding):

- `input` (interleaved DRAM/L1, ROW_MAJOR, bf16; one page = one row): **Case 1** (via `TensorAccessor`). Express it as a `TensorParameter` / `TensorBinding`; the reader uses `TensorAccessor(tensor::<name>)`. What goes away:
  - reader RTA[2] `input_tensor_address`, today pushed as a `Buffer*` (`program_factory.cpp:163`) and read at `reader.cpp:23`;
  - `TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args)` (`program_factory.cpp:93-94`, CTA offset 0);
  - `TensorAccessorArgs<0>()` / `TensorAccessor(args, addr)` (`reader.cpp:25-26`).
  Keep the access calls `noc.async_read(accessor, …, {.page_id = page_id, .offset_bytes = intra_row_byte_offset}, {})` (`reader.cpp:37-42`) unchanged.
- `output` (L1 ND-sharded, TILE): **clean** (a borrowed-memory DFB). The output CB `c_1` is backed by the output buffer (`.buffer = output_tensor.buffer()`, `program_factory.cpp:85`). Express it as a `TensorParameter` and make the `c_1` `DataflowBufferSpec` `borrowed_from` it. No kernel reads the output through an accessor or an address.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none. No accessor passes one.

**CB endpoints:** all legal, single config, plain 1P+1C on every node of `op_cores`:
- `c_0` / `tilize_input_cb_id` (`program_factory.cpp:62-71`; page `output_shard_width_bytes`, `TILE_HEIGHT` pages): reader **PRODUCER** (`reserve_back`/`push_back`, `reader.cpp:32, 49`), compute **CONSUMER** (`wait_front`/`pop_front`, `compute.cpp:25, 31`).
- `c_1` / `tilize_output_cb_id` (`program_factory.cpp:76-86`; borrowed from output; page `output_tile_page_size`, `output_shard_width_tiles` pages): compute **PRODUCER** (`reserve_back`/`push_back`, `compute.cpp:26, 30`), writer **CONSUMER** (`wait_front`/`pop_front`, `writer.cpp:15-16`).

No self-loops, no multi-binding flag, no dead CBs.

## Watch for

- **CB endpoints (multi-binding):** none. I hunted for a hidden second writer on `c_1`: none, and the op has no semaphores.
- **Cross-op / shared kernels:** none. All three kernels are op-owned and bound only by this op's factory (census excluding `experimental/quasar/`), so convert them in place in `device/kernels/`; there's no `_metal2` fork. Don't use anything under `experimental/quasar/` as a model.
- **RTA varargs:** none. The reader reads a fixed run of 3 args through `rt_args_idx++` (`reader.cpp:20-23`): name `intra_row_byte_offset` and `row_page_offset`; the third (the address) becomes the `input` tensor binding. The writer and compute have no RTAs. No CTA varargs.
- **Compute kernel's LLK calls take CB ids** (`compute_kernel_hw_startup`, `fast_tilize_init`, `fast_tilize_block`, `fast_tilize_uninit`; `compute.cpp:22, 23, 28, 33`). The ids are `constexpr` from named CTAs today (`compute.cpp:13-14`), so use `dfb::name`'s constexpr `uint32_t` conversion and keep them constexpr. The same `c_0`/`c_1` ids also feed the `CircularBuffer` objects (`compute.cpp:19-20`), which become `DataflowBuffer`s.
- **Include swap in all three kernels:** each includes `api/dataflow/circular_buffer.h` (`reader.cpp:9`, `writer.cpp:7`, and `compute.cpp:9`, which is easy to miss in a compute kernel). Swap per the whitelist.
- **Reader's CB write path is `CoreLocalMem<uint32_t>(cb_tilize_input.get_write_ptr())`** (`reader.cpp:33, 39`), with `l1_write_addr += bytes_to_read_per_row` per row. It's an existing Device 2.0 idiom: just swap to the DFB's `get_write_ptr()`, and don't rewrite it to a DFB destination.
- **Unread named CTA `input_row_page_size`** (`program_factory.cpp:105`): no kernel reads it. It isn't a dead-CB CTA, so the dead-CB drop rule doesn't apply. Handle it per the port recipe's rule for unread named args (no functional effect either way), and record the choice in the port report.
- **Per-core RTA order depends on shard orientation** (`program_factory.cpp:147-164`): the cores come from `corerange_to_cores(op_cores, std::nullopt, is_row_major_shard_orientation)`, and the offsets are computed from the index `i` in that order. Preserve the same core iteration when building `ProgramRunArgs`.
- **Kernel configs to carry verbatim:**
  - reader: `RISCV_1` / `NOC_0` / `DM_DEDICATED_NOC` / `O2` (`program_factory.cpp:108-113`)
  - writer: `RISCV_0` / `NOC_1` / `DM_DEDICATED_NOC` / `O2` (`:140-145`)
  - compute: default `ComputeConfigDescriptor{}` (`:127`)
- **Readiness sheet not live-verified:** the TTNN gate was cleared by user override. If a live sheet fetch becomes available before the PR lands and shows a `Known op issues` entry or a non-`none` relaxation, stop and report.
