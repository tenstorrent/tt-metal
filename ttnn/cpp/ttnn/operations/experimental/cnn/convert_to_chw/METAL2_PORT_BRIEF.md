# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

- **Device 2.0** cleared on a judgment call that is still open with the Device 2.0 team. The call concerns the pool helper's hand-built loopback endpoint (`METAL2_PREPORT_AUDIT.md` → Questions 2). If the team overturns it before you start, stop: the gate flips to RED and this brief no longer applies.
- **TensorAccessor 3rd arg** cleared with no sites at all, because the op has no `TensorAccessor`.

**Recipe docs:** ``c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`` *(carry this line into the port report's Provenance section)*

**Audited:** code at `4c3e12db692` (`origin/main` `d70d725e238`); readiness sheet fetched 2026-10-30 12:04 UTC.

Shorthand used below:
- `reader:` = `device/kernels/reader_convert_to_chw.cpp`
- `writer:` = `device/kernels/writer_convert_to_chw.cpp`
- `compute:` = `device/kernels/convert_to_chw.cpp`
- `.cpp:` / `.hpp:` = `device/convert_to_chw_program_factory.{cpp,hpp}`

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor`, via `ConvertToCHWProgramFactory::create_descriptor` (`.hpp:18-19`, `.cpp:42-148`).
  - The device-op already declares `program_factory_t = std::variant<ConvertToCHWProgramFactory>` (`device/convert_to_chw_device_operation.hpp:22`).
  - So this is a method swap inside the existing struct. The direct-descriptor exception in `ttnn_factory.md` does not apply.
- **Op-owned tensors:** none.
- **Target concept:** `CustomProgramSpecFactoryConcept`. The `override_runtime_arguments` to translate is at `.cpp:150-167` (declared at `.hpp:21-26`).
  - **What it refreshes today:** exactly two borrowed-CB addresses, `c_0` from the input buffer and `c_1` from the output buffer. It does this through a CB-only descriptor plus `apply_descriptor_runtime_args` (`.cpp:163-166`). The CBs are matched by position against the order `create_descriptor` pushes them (`tt_metal/impl/program/program_descriptors.cpp:231-243`).
  - **What it does not refresh:** any runtime args. `total_tiles_per_core` is keyed by the tensor spec.
  - **Return type:** translate it to return `ProgramRunArgs`, not `void`. A `void` return silently leaves the factory on the base concept (`ttnn_factory.md:104`).
  - **Contents:** `tensor_args` for **both** `INPUT` and `OUTPUT` (the borrow-only parameters behind `c_0` and `c_1`), and no `kernel_run_args`. Omitting either tensor arg freezes that borrowed DFB at its cache-miss address (`ttnn_factory.md:118-133`).
  - **Coordinate:** the ported-from override ignores `mesh_dispatch_coordinate` (`.cpp:155`), so the translated one can ignore `coord`.
- **Execution model:** `SPMD`. Nothing varies per coordinate.
- **Custom `compute_program_hash`:** none; the op uses the default reflection hash.
- **Pybind `create_descriptor`:** none, so there is no binding to remove. `convert_to_chw_nanobind.cpp:25-26` binds only the free function.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args` (deprecated hook). A custom hash, an `override_runtime_arguments`, and a pybound `create_descriptor` are **not** in this list: none of them gate, and any may be present on a cleared op.

## Construct — to do

**Tensor bindings** (per binding):

- `input` — **clean**: a borrowed-memory DFB, not Case 1 or Case 2.
  - Declare a `TensorParameter` from the input's `tensor_spec()`, and name it in `c_0`'s `DataflowBufferSpec::borrowed_from`.
  - Add **no** kernel `TensorBinding`. A borrow-only parameter counts as used (`migration_guide.md:366`).
- `output` — **clean**: a borrowed-memory DFB. Use the same pattern, backing `c_1`.

No kernel builds a `TensorAccessor`, and no address reaches a kernel through an RTA, CRTA or CTA. There is therefore no address argument to remove.

**TensorParameter relaxation:** `none`. Keep both `TensorParameter`s strict.

**TensorAccessor 3rd arg:** none.

**CB endpoints:**

- `c_1` `cb_out`: **self-loop**.
  - It is borrowed from the output (`.cpp:88-94`), with entry size = shard width × element size and `C` entries.
  - Bind the writer as both PRODUCER and CONSUMER, because it is the only kernel that touches this CB. That makes it a DM self-loop, which is legal on Gen1.
- `c_0` `cb_in`: plain 1:1.
  - It is borrowed from the input (`.cpp:82-86`), with entry size = one input tile and `total_tiles_per_core` entries.
  - The reader is PRODUCER (`push_back`, `reader:12`). Compute is CONSUMER (`wait_front` / `pop_front`, `compute:15, 22`).
- `c_2` `cb_in_transpose`: plain 1:1.
  - It uses its own L1 (`.cpp:96-106`), sized at 16 × `Float16_b` tile.
  - Compute is PRODUCER (`compute:24, 26, 29`). The writer is CONSUMER (`writer:35, 48, 52, 61`).

You need no multi-binding flag, no dead-CB drop and no conditional DFB, because the op has a single config.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none.
  - Only this factory binds the three kernel sources (checked by filename census), and no `_metal2` sibling exists. Convert them in place.
  - The shared-kernel caution in `port_patterns.md` does not apply.
  - Other binding ops: none (**sunset list: empty**).
- **RTA varargs:** none.
  - Each kernel has one named RTA, `total_tiles` (`reader:9`, `writer:13`, `compute:33`).
  - All CTAs are at fixed indices. `C` (writer CTA 2) is the only one that isn't a CB id.
- **Anything else:**
  - **Keep the pool-family helper as it is.**
    - The reader and writer include `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp`.
    - Keep calling `experimental::set_read_state` / `experimental::read_with_state` (`writer:29, 41, 56`) on raw `get_read_ptr()` / `get_write_ptr()` addresses. The whitelist keeps these peeks for helpers that take raw L1 (`cb_dfb_api_whitelist.md:138`).
    - Ported precedent: `data_movement/fold/device/kernels/dataflow/writer_dfb2s_row_major.cpp:7-9, 37-44, 60`.
    - Don't edit the header, and don't rewrite the hand-built loopback inside it (`local_addr`, `:37-39`). That is the open Device 2.0 question, not port work.
  - **`experimental::CB` is an alias for `CircularBuffer`** (`using CB = CircularBuffer`, pool header `:24`). Swap `reader:11` and `writer:24-25` to `DataflowBuffer` built from `dfb::` tokens, even though `CircularBuffer` never appears in those files.
  - **`writer:19` must stay a free-function call.** The line is `constexpr uint32_t in_transpose_tile_size = get_tile_size(cb_in_transpose);`.
    - Because it is `constexpr`, keep the free function and pass the token, `get_tile_size(dfb::<name>)`, instead of using the member getter (`cb_dfb_api_whitelist.md:93-107`).
    - The token form is Gen1-only, so record this site in the port report.
  - **The compute helper `transpose<BATCH_SIZE>(uint32_t cb_in, uint32_t cb_out)`** (`compute:11-30`) needs care:
    - Pass `dfb::` tokens straight to the LLKs (`port_patterns.md` → "Pass DFB handles directly to LLKs").
    - Its FIFO wrappers need `DataflowBuffer` objects. How the helper receives them is your call. Building them from a raw id uses the low-level constructor (`dataflow_buffer.h:155-156`).
    - Its local `cb_out` is `c_2`, not `c_1` (`compute:46, 50`).
  - **Swap the compute kernel's direct include** of `api/dataflow/circular_buffer.h` (`compute:9`), per the whitelist.
  - **Keep two unusual but intended patterns as they are:**
    - `reader:12` calls `push_back(total_tiles)` with no `reserve_back`, because the data is already resident.
    - `writer:29` peeks `get_read_ptr()` before its first `wait_front`, only to program the NoC read state.
  - **Leave `make_cnn_circular_buffer` untouched.** It lives in `ttnn/cpp/ttnn/operations/experimental/cnn/cnn_cb_descriptor.hpp`, and `convert_to_hwc` still calls it (`convert_to_hwc/device/convert_to_hwc_program_factory.cpp:313-355`). Stop calling it here, but don't edit or delete it.
  - **Both borrowed DFBs pass Metal 2.0's borrowed-memory check.**
    - `ValidateBorrowedMemory` (`tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp:34`) requires an L1 backing tensor that is not per-core-allocated, with a per-bank slice of at least `entry_size × num_entries`.
    - The op documents its input as L1 height-sharded (`convert_to_chw_nanobind.cpp:19`). For that input, `c_0` is at most the input shard and `c_1` equals the output shard.
  - **Carry the kernel configs over:**
    - Reader: `ReaderConfigDescriptor{}` (`.cpp:118`). Writer: `WriterConfigDescriptor{}` (`.cpp:126`).
    - Compute: `HiFi4`, `fp32_dest_acc_en = false`, `dst_full_sync_en = false`, `math_approx_mode = false` (`.cpp:134-139`).
    - `enable_32_bit_dest` stays false, so no `unpack_modes` entries are needed (`migration_guide.md:302`).
  - **No build-file edit is needed.** Kernel files are picked up by the family glob (`ttnn/cpp/ttnn/operations/experimental/cnn/CMakeLists.txt:16-20`).
  - **Cache-hit coverage already exists:** `tests/ttnn/unit_tests/operations/conv/data_movement/test_convert_to_chw.py::test_convert_to_chw_with_program_cache` (`:94-116`).
    - It runs eight iterations over two specs and two dtypes. A dummy L1 allocation shifts addresses on each iteration, and the test expects 4 cache entries.
    - That exercises exactly the refresh the translated override owns.
