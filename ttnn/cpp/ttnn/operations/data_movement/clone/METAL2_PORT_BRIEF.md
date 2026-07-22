# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/clone`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4af0db51e3c 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — one `create_descriptor` returning a `ProgramDescriptor` (`clone_device_operation.hpp:29`), with four build-time code-paths ({tilized, row-major} × {interleaved, sharded}) and an optional compute kernel when `convert_dtype`.
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no` on this op.

## Construct — to do

**Tensor bindings** (two bindings, `input` and `output`; classification splits by config branch):

- **`input`**
  - **Interleaved paths** (`read_kernel.cpp`, `read_kernel_rm.cpp`) — **Case 1** (via `TensorAccessor`): express as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(args, input_buffer_address)`. The address-via-RTA and the `TensorAccessorArgs` compile-time plumbing both disappear.
  - **Sharded paths** (`read_kernel_sharded.cpp`, `read_kernel_rm_sharded.cpp`) — **Case 2** (raw pointer): bind the tensor as a `TensorParameter`, pull the base via `get_bank_base_address`, and leave the existing raw local-L1 walk unchanged (`noc.async_read(UnicastEndpoint{}, …, {.addr = local_l1_read_addr})`, `local_l1_read_addr += tile_size`). Do **not** rewrite the raw walk into accessor iteration.
- **`output`**
  - **Interleaved paths** (`write_kernel.cpp`, `write_kernel_rm.cpp`) — **Case 1** (via `TensorAccessor`), same treatment as `input`.
  - **Sharded paths** (`write_kernel_sharded.cpp`, `write_kernel_rm_sharded.cpp`) — **Case 2** (raw pointer), bridge via `get_bank_base_address`, raw local-L1 walk unchanged.

Delivery today is the `Buffer*`-binding form (`emplace_runtime_args` with `input_buffer` / `output_buffer`, `clone_program_factory.cpp:231-243`) — correct on cache hits under the framework's interim `BufferBinding` patching, not the silent-wrong `->address()`-in-RTA hazard. Replace it with the typed `TensorParameter` binding regardless.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor passes a page-size argument; nothing to drop.

**CB endpoints:** all legal — every CB is a plain 1P+1C FIFO on every node in every config (`src_cb` = `c_4`; `dst_cb` = `c_20`, present only when `convert_dtype`). No self-loop, 1P+1C assignment, multi-binding flag, or dead-CB drop needed.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none — clone owns all nine kernel files; no shared-kernel port-together coupling.
- **RTA varargs:** none — all RTAs are fixed distinct fields; name each (`input_buffer_address`/`output_buffer_address`, `stick_size`/unit size, `num_units`/`num_sticks`/`num_tiles`, `start_id`).
- **Minor (from the team doc, not blocking):** the row-major *interleaved* kernels carry a dead compile-time arg at CTA index 1 (`input_unit_size`/`output_unit_size`, skipped because `TensorAccessorArgs<2>` starts at index 2; the kernel reads the stick size from RTA index 1 instead). It's inert; the ops team owns cleanup. Don't let it confuse the CTA-index bookkeeping when you rewire compile-time args.
