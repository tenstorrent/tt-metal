# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/bcast_to`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.
>
> **One gate cleared by user override:** the readiness sheet row for this op is stale. It predates the PD migration in #57409 and still says `legacy device-op` / `BcastToTileFactory`. The user directed proceeding on the code-side evidence; see the audit's *Result*. Record this in the port report's Provenance section.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (user override; sheet stale) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A, no sites)

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`, in the **direct-descriptor** shape. `BcastToOperation` declares `static ProgramDescriptor create_descriptor(...)` as its own member (`device/bcast_to_device_operation.hpp:40-43`, defined at `device/bcast_to_program_factory.cpp:135`), with **no** `program_factory_t` and no factory struct. → `ttnn_factory.md` §3 "Give a direct-descriptor op a conventional program factory" applies. Introduce a factory struct + `using program_factory_t = std::variant<…>;` and record it under Handoff points. Replacing `create_descriptor` with `create_program_artifacts` in place would drop `HasDirectDescriptor` (`ttnn/api/ttnn/operation_concepts.hpp:158`) and leave the op an invalid device operation.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. There is no `override_runtime_arguments` (removed in #57409), so the framework refreshes bindings on cache hit.
- **Custom hash:** none. The default hash covers `operation_attributes_t{output_shape, memory_config, subtile_broadcast_type}` + `tensor_args_t{input, optional output}`. Every non-address RTA comes from that hashed state (comment at `device/bcast_to_device_operation.hpp:37-39`).
- **Pybind `create_descriptor`:** none. `bcast_to_nanobind.cpp:43` binds only `ttnn.experimental.broadcast_to`, so there is no user-visible API removal.
- **Gate-cleared, confirmed absent:** a `TensorParameter relaxation` that is neither `none` nor an analysis pointer (sheet: `none`) · `get_dynamic_runtime_args` (absent).

## Construct — to do

**Tensor bindings:**

- `input` — **Case 1** (via `TensorAccessor`) → express it as a `TensorParameter` / `TensorBinding`, and have the kernel use `TensorAccessor(tensor::<name>)`. Legacy plumbing that goes away:
  - the `Buffer*` RTA slot 0 in the reader (`device/bcast_to_program_factory.cpp:84`)
  - `TensorAccessorArgs(input.buffer()).append_to(...)` (`:173`)
  - kernel-side `src_addr = get_arg_val(0)` + `TensorAccessorArgs<1>()` + `TensorAccessor(src_args, src_addr)` (`reader_interleaved_*_bcast_to.cpp:13,28,31`)
- `output` — **Case 1** → same treatment. Legacy plumbing that goes away:
  - the `Buffer*` RTA slot 0 in the writer (`:100`)
  - the `TensorAccessorArgs` CTA (`:186`)
  - kernel-side `dst_addr` / `dst_args` / `TensorAccessor(dst_args, dst_addr)` (`writer_interleaved_*_bcast_to.cpp:13,31-33`)

  The output may be a caller-supplied preallocated tensor (`device/bcast_to_device_operation.cpp:154-156`); bind it as the output either way.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none. Every accessor is the 2-arg form.

**CB endpoints:** two CBs, `c_0` and `c_1`, both `num_tiles_per_cb = 2` × the input tile size in the input data format (`device/bcast_to_program_factory.cpp:156-167`). They are allocated over the full compute grid (`all_device_cores`).

- `c_0` — **1P+1C in every config**, with roles fixed by FIFO ops:
  - `NONE`: reader PRODUCER → writer CONSUMER (`writer_cb_id = c_0` at `:183-184`)
  - `ROW` / `COL` / `SCALAR`: reader PRODUCER → compute CONSUMER
- `c_1` — make its DFB spec **conditional on `subtile_broadcast_type != NONE`**. It is dead under `NONE` (zero touchers) and live under `ROW` / `COL` / `SCALAR` (compute PRODUCER → writer CONSUMER). **Do not drop it.** The legacy host allocates both CBs unconditionally in one loop (`:157`), so this condition is new host-side structure. Note it in the port report.

## Watch for

- **CB endpoints (multi-binding):** none. Under `NONE`, the compute kernel (`compute_interleaved_no_bcast_to.cpp`, empty `kernel_main`) touches **no** DFB, even though the legacy factory hands it CTAs `{c_0, c_1}` (`device/bcast_to_program_factory.cpp:209`). **Don't bind `c_0` to compute under `NONE`.** It would become a phantom third toucher of `c_0` (reader + writer + compute) and wrongly force multi-binding. Under `NONE` the compute KernelSpec binds no DFBs.
- **Cross-op / shared kernels:** none. All 12 kernel sources are private to this op. The filename census finds no binder outside `experimental/bcast_to/`, and each config binds its own distinct kernels, so there are no intra-op shares either. No `_metal2` fork exists or is needed: convert these kernels **in place**. Ignore anything under `experimental/quasar/`.
- **RTA varargs:** none. Every kernel reads a fixed `arg_index++` run, so name each arg. After the address slot moves to the binding: reader 12 named args, writer 13 (adds the trailing `start_tile_id`), compute 12. Several named args are read but never used kernel-side (e.g. `n_stride` / `c_stride` in writers and compute; see the audit's *Misc anomalies*). **Carry them as-is.** Trimming them is an ops-team change, not port work.
- **Idle-core RTAs:** cores outside `core_group_1` / `core_group_2` get all-zero vectors of length 13 / 14 / 12 (`device/bcast_to_program_factory.cpp:65-70`), and the kernels exit on `num_tiles == 0`. Keep supplying zeros for every named arg on idle cores, so the per-core schema stays uniform.
- **`unpack_to_dest_mode` is keyed by CB index:** for 32-bit formats (`Float32` / `Int32` / `UInt32`), `c_0` gets `UnpackToDestFp32`, alongside `fp32_dest_acc_en = is_32bit_format` (`device/bcast_to_program_factory.cpp:197-214`). Carry it onto the compute config for the DFB that replaces `c_0`. Confirm the Metal 2.0 form rather than swapping blind.
- **Compute kernels go through `kernel_lib/eltwise`** (`ckl::input(uint32_t cb_id, …)` / `ckl::output(uint32_t cb_id, …)` as NTTPs, `kernel_lib/eltwise/api/chain.hpp:356`), plus `compute_kernel_hw_startup` / `unary_bcast_init<…>` taking the ids. `dfb::name`'s constexpr `uint32_t` conversion covers both the runtime and template-parameter positions, so no donor change is needed. `kernel_lib` is out of porter scope; don't touch it. The compute CTAs are already named `dfb_id_src_id` / `dfb_id_dst_id`, so the compute-side change is the binding layer only.
- **Dataflow kernels** include `api/dataflow/circular_buffer.h` and construct `CircularBuffer cb_*(get_compile_time_arg_val(0))`, using `cb.get_tile_size()` for the transfer size. That gives the normal CB→DFB swap with `dfb::name`. `get_tile_size()` moves onto the `DataflowBuffer` object (kernel-side whitelist rule 7).
- **Regression check:** `tests/ttnn/nightly/unit_tests/operations/experimental/test_bcast_to.py`. In particular, `test_bcast_to_program_cache` (`:137`) moves tensor addresses across cache hits and asserts a single cache entry, which directly exercises the binding refresh. The row / col / scalar int tests (`:57`, `:86`, `:115`) cover the three non-`NONE` kernel sets. Also run `tests/ttnn/unit_tests/operations/eltwise/test_broadcast_to.py` (bf16 / bf8_b / preallocated-output / invalid-sharding cases).
- **The `NONE` config has no active test coverage.** Every same-H/W shape pair in `tests/ttnn/unit_tests/operations/eltwise/test_broadcast_to.py:10-17` is commented out (e.g. `((1, 1, 64, 64), (1, 310, 64, 64))`, `((1, 1, 32, 32), (7, 17, 32, 32))`), and the nightly file has none either. This is the one config where your conditional `c_1` spec and the reader→writer `c_0` pairing differ from the other three. Verify it explicitly, e.g. with an ad-hoc run of an N/C-only broadcast like `(1, 1, 32, 32) → (2, 3, 32, 32)` before and after the port. Note the gap in the port report; adding a test is outside the port diff unless your invoker asks for it.
