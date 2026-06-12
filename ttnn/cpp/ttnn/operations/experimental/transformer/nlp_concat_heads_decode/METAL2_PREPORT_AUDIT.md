# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode`

- **`NLPConcatHeadsDecodeDeviceOperation`**
  - `NLPConcatHeadsDecodeProgramFactory` (`device/nlp_concat_heads_decode_program_factory.cpp`)
  - `NLPConcatHeadsDecodeSubcoregridsProgramFactory` (`device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp`)

Both factories instantiate kernels owned by this op. There is one DeviceOperation with two ProgramFactories selected by the `on_subcoregrids` flag.

Kernels referenced:
- `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode.cpp` — used by `NLPConcatHeadsDecodeProgramFactory` (both reader and writer KernelDescriptors point to this same file, instantiated twice with different CTAs)
- `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp` — used by `NLPConcatHeadsDecodeSubcoregridsProgramFactory` (same dual-instantiation pattern)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode` |
| **Overall** | GREEN |
| **DOps / Factories** | `NLPConcatHeadsDecodeDeviceOperation` → `NLPConcatHeadsDecodeProgramFactory`, `NLPConcatHeadsDecodeSubcoregridsProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | present: CB `c_16` on both factories — `get_write_ptr()` used as address source, no LLK FIFO producer or consumer (workaround) |

## Result

**GREEN → brief issued.** All gates pass. The port can proceed with the work items below. Primary port-time decisions: (a) re-express the input tensor binding from raw-address RTA to `TensorParameter` / `TensorBinding` (Case 1 assumed — see open question about sub-tile face access granularity); (b) apply the fake-CB workaround for the output CB.

## Gate detail

- **ProgramDescriptor:** GREEN — both `NLPConcatHeadsDecodeProgramFactory::create_descriptor` and `NLPConcatHeadsDecodeSubcoregridsProgramFactory::create_descriptor` return a `ProgramDescriptor` populated with `CBDescriptor`, `KernelDescriptor`, and `KernelDescriptor::RTArgList`. No imperative `host_api.hpp` calls (`CreateKernel`, `CreateCircularBuffer`, `SetRuntimeArgs`, etc.) appear in either factory.

- **Device 2.0 (every kernel used):** GREEN — both kernel files use the Device 2.0 API throughout:
  - `Noc noc;`, `noc.async_read(...)`, `noc.async_read_barrier()` — Device 2.0 `Noc` wrapper
  - `CircularBuffer cb_q_out(cb_id_q_out);`, `cb_q_out.get_write_ptr()` — Device 2.0 member form
  - `UnicastEndpoint src_ep;` with `{.noc_x, .noc_y, .addr}` args struct — Device 2.0 `UnicastEndpoint`
  - `CoreLocalMem<uint32_t>(...)` — Device 2.0 local-memory destination
  - `get_arg_val<uint32_t>(...)`, `get_arg_addr(...)` — Device 2.0 `dataflow_api.h` free functions
  - No legacy `InterleavedAddrGen`, `ShardedAddrGen`, `noc_async_read(addr, ...)` free-function calls, or `get_noc_addr_from_bank_id` with manual bank arithmetic.

- **Feature compatibility:** See table below. No UNSUPPORTED feature fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Op uses no `GlobalCircularBuffer` |
  | Dynamic CircularBuffer (borrowed memory) | N/A (superseded by fake-CB finding) | `.buffer = output.buffer()` fires the recognition signal, but the CB has no LLK FIFO producer or consumer — it is a fake CB, not a genuine borrowed-memory DFB. Recorded under Fake CBs. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Neither factory sets `.address_offset` on the `CBDescriptor`. |
  | Aliased Circular Buffers | N/A | Both factories use single-element `format_descriptors = {{CBFormatDescriptor{...}}}` — not aliased. |
  | GlobalSemaphore | N/A | Op uses no semaphores. |
  | Non-zero semaphore initial value | N/A | Op uses no semaphores. |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `TensorAccessorArgs` at all; no `ArgConfig::Runtime*` tokens. |
  | `UpdateCircularBuffer*` | N/A | Neither factory nor any override hook calls `UpdateCircularBuffer*`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | CTAs are fixed-count arrays (9 elements in the base factory, 10 in the subcoregrid factory). `tensor_args_t` carries no `std::vector<Tensor>`. |

## Port-work summary *(mirrors the brief)*

- **Tensor bindings:**
  - `input` (`in_buffer`) — **Case 1** (re-express via `TensorParameter` / `TensorBinding`). The factory pushes `in_buffer` as a `Buffer*` RTA (`rt_args.push_back(in_buffer)` at `nlp_concat_heads_decode_program_factory.cpp:130` and `nlp_concat_heads_decode_subcoregrids_program_factory.cpp:137`). The framework auto-registers this as a `BufferBinding`, so it is patched on cache hits today, but the kernel consumes a raw `uint32_t` base address (`q_start_addr = get_arg_val<uint32_t>(1)`) and does manual address arithmetic. Port: bind input as `TensorParameter` and build the access via `TensorAccessor(ta::input)` — **but see the open question** about sub-tile face-granularity access, which may require confirming Case 1 vs. Case 2 with the user.
  - `output` (fake CB) — no `TensorParameter` binding as such; the output CB is resolved via the fake-CB workaround (see Heads-ups). Not a standalone `TensorParameter` / `TensorBinding` item until the porter decides the workaround shape.

- **Custom hash:** none — no `compute_program_hash` to delete.

## Heads-ups *(mirrors the brief)*

- **Fake CBs (address-only):** CB `c_16` in both factories is declared with `.buffer = output.buffer()`, which fires the Dynamic CircularBuffer recognition signal. However, applying the causal-link litmus — does the CB have a producer AND a consumer? — reveals it does not: the kernels call `cb_q_out.get_write_ptr()` to obtain a destination L1 address and then write there directly via `noc.async_read(...)` (DMA into the write pointer), but no LLK `cb_push_back`/`reserve_back` produces into the CB and no `cb_wait_front`/`pop_front` consumes it. The CB is used purely as an address source — a fake CB. A Metal 2.0 DFB requires ≥1 producer and ≥1 consumer, so this cannot be expressed as a `DataflowBufferSpec::borrowed_from`. The port resolves this with the **fake-CB workaround** (see the porting recipe). This is an FYI-P heads-up, not a gate.
  - `(CB c_16, write-ptr endpoint)` — `nlp_concat_heads_decode_program_factory.cpp:46–55` (CBDescriptor), `reader_tm_tile_layout_nlp_concat_heads_decode.cpp:49` (get_write_ptr)
  - `(CB c_16, write-ptr endpoint)` — `nlp_concat_heads_decode_subcoregrids_program_factory.cpp:56–65` (CBDescriptor), `reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp:48` (get_write_ptr)

- **Cross-op / shared kernels:** None. All kernels are owned by this op and instantiated only from this factory. The only `#include`s are Device 2.0 framework headers (`api/dataflow/*`, `api/core_local_mem.h`) — no cross-op donor function calls.

- **RTA varargs:** Not present (no `num_runtime_varargs`, no counted loop over `get_arg_val`). The `get_arg_addr(2)` / `get_arg_addr(2 + num_x)` pattern reads the noc-coord arrays by casting the arg-area pointer, but the array sizes are compile-time-known (`num_x`, `num_y` / `in_num_cores` are CTAs), so this is a fixed-layout multi-arg, not true RTA varargs.

- **TTNN factory analysis (porter-relevant):**
  - Pybind `create_descriptor`: No — `nlp_concat_heads_decode_nanobind.cpp` uses `bind_function<"nlp_concat_heads_decode", "ttnn.experimental.">` to bind the user-facing function only; no `nb::class_<...ProgramFactory>` binding exists.
  - Other risky pybind: None — the nanobind file exposes only the user-facing function and its arguments.
  - Custom `override_runtime_arguments`: No — neither factory defines this hook.

## Team-only

### TensorAccessor convertibility

- `input` binding — tentatively Case 1. The access pattern is: given the input shard's base address and an offset `in_tile_offset_by_head`, the kernel walks through shard memory reading `SUBTILE_LINE_BYTES`-size chunks at sub-tile face-row offsets (reading at `q_start_addr + in_tile_offset_by_head`, `+ tile_size`, `+ 2*tile_size`, etc., with face-row arithmetic). This is **sub-tile face-row granularity** reading from a HEIGHT-SHARDED L1 shard on a remote core. This access granularity is unusual for `TensorAccessor` (which typically operates at tile or page granularity). Whether the pattern is genuinely exotic (Case 2) or merely awkward-but-convertible (Case 1 with an enhanced `TensorAccessor` iteration) should be confirmed with the user. The recipe requires assuming Case 1; the open question is surfaced below.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** `✓ clean` — no cross-family donor function calls; all includes are Device 2.0 framework headers.

**Summary table:**

| Op kernel | Include / donor | Donor class | Notes |
|---|---|---|---|
| `reader_tm_tile_layout_nlp_concat_heads_decode.cpp` | `api/dataflow/dataflow_api.h` | `tt_metal/*` (HAL/firmware) | No concern |
| `reader_tm_tile_layout_nlp_concat_heads_decode.cpp` | `api/dataflow/noc.h` | `tt_metal/*` | No concern |
| `reader_tm_tile_layout_nlp_concat_heads_decode.cpp` | `api/dataflow/circular_buffer.h` | `tt_metal/*` | No concern |
| `reader_tm_tile_layout_nlp_concat_heads_decode.cpp` | `api/dataflow/endpoints.h` | `tt_metal/*` | No concern |
| `reader_tm_tile_layout_nlp_concat_heads_decode.cpp` | `api/core_local_mem.h` | `tt_metal/*` | No concern |
| `reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp` | same five headers | `tt_metal/*` | No concern |

**Per-call detail:** Omitted — all includes are `✓` framework headers with no cross-op function calls.

**Borrowed kernel files (file-path kernel instantiation):** None. Both kernel `.cpp` files are owned by this op and are not known to be instantiated by any other op family.

### Relaxation candidates

None — no custom `compute_program_hash` to mine.

### TTNN factory analysis

1. **Op-owned tensors?** No. The `create_device_tensor(...)` call at `nlp_concat_heads_decode_device_operation.cpp:118` allocates the op's declared output tensor in `create_output_tensors` — this is the standard output-allocation pattern, not an intermediate/scratch tensor the factory owns. No factory-owned intermediate tensors were observed.

2. **MeshWorkload concept needed?** No. No `create_mesh_workload`, `create_workload_descriptor`, or `cached_mesh_workload_t` in any file. Single-program op.

3. **Pybind `create_descriptor`?** No. `nlp_concat_heads_decode_nanobind.cpp` contains only `bind_function<"nlp_concat_heads_decode", "ttnn.experimental.">` binding the user-facing function. No `nb::class_<...ProgramFactory>(...).def_static("create_descriptor", ...)` or similar.

4. **Other migration-risky pybind?** No. The nanobind file exposes nothing from `DeviceOperation` internals or factory/param classes.

5. **Custom hash?** No. Neither `NLPConcatHeadsDecodeDeviceOperation` nor the factory structs define `compute_program_hash`.

6. **Custom `override_runtime_arguments`?** No. Neither `NLPConcatHeadsDecodeProgramFactory` nor `NLPConcatHeadsDecodeSubcoregridsProgramFactory` defines this static method.

## Misc anomalies *(non-gating, team-only)*

- **Dual kernel instantiation with the same file path.** Both factories instantiate the *same* kernel `.cpp` file twice — once as `ReaderConfigDescriptor` and once as `WriterConfigDescriptor` — with identical runtime args but different compile-time arg `[6]` (phase 1 vs. phase 2). The kernel dispatches on `PHASES_TO_READ` at compile time. This is intentional (parallelizes face-0 vs face-2 reading across RISC0/RISC1) but unusual; the port should preserve it carefully when translating to `KernelSpec`.
- **`cb_q_out.get_write_ptr()` called before any writes to the CB.** The CB is used as an address-only anchor: `const uint32_t cb_write_ptr_base = cb_q_out.get_write_ptr()` at kernel line 49 / 48 serves as the L1 base address of the output shard. This is the fake-CB pattern documented under Heads-ups. No data flows through the CB FIFO; the port's fake-CB workaround should account for the `get_write_ptr()` call becoming a direct `DFB::get_base_address()` or equivalent.

## Questions for the user

1. **Input tensor access granularity — Case 1 or Case 2?** The kernels read from the input tensor at sub-tile face-row granularity: `SUBTILE_LINE_BYTES`-size chunks at face-row offsets from a raw shard base address, walking across multiple input cores via raw NoC coordinates. `TensorAccessor` typically iterates at tile/page granularity. Is this access pattern genuinely unsupported by `TensorAccessor`'s iteration model (→ Case 2: bridge via `get_bank_base_address`), or is it expressible via a `TensorAccessor` enhancement that should be filed? The recipe requires assuming Case 1 until you confirm otherwise. Note: *"The use of `TensorAccessor` is an ergonomic choice on Gen1 architectures. It has meaningful performance implications on Gen2 architectures. Ideally, `TensorAccessor` should be updated to support the required iteration pattern; consider filing an issue requesting that support."*

## Recipe notes

- The Dynamic CircularBuffer (LANDED) recognition fires when `.buffer` is set non-null, but the causal-link gate then immediately supersedes the finding with the fake-CB determination. The recipe handles this flow correctly (the litmus is in the TensorAccessor subject, not Feature compatibility), but the Feature compatibility table ends up with a "N/A (superseded by fake-CB finding)" entry for Dynamic CB — the recipe doesn't explicitly give a table status for this case. I used `N/A` with a parenthetical note, which felt right: the LANDED feature's presence signal fired but the causal-link gate redirected it. Recipe maintainer may want to clarify the recommended table entry when a Dynamic CB fires but fails the producer+consumer litmus.
