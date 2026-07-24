# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode`

One DeviceOperation, three program factories:

- **`NLPCreateQKVHeadsDecodeDeviceOperation`**
  - `NLPCreateQKVHeadsDecodeInterleavedProgramFactory` (`nlp_create_qkv_heads_decode_interleaved_program_factory.cpp`)
  - `NLPCreateQKVHeadsDecodeShardedProgramFactory` (`nlp_create_qkv_heads_decode_sharded_program_factory.cpp`)
  - `NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory` (`nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp`)

Kernels owned and instantiated by this op:

- `device/kernels/reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` — used by `InterleavedProgramFactory` (both reader RISC0 and writer RISC1)
- `device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` — used by `ShardedProgramFactory`
- `device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp` — used by `ShardedSubcoregridProgramFactory`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode` |
| **Overall** | RED |
| **DOps / Factories** | `NLPCreateQKVHeadsDecodeDeviceOperation` → `NLPCreateQKVHeadsDecodeInterleavedProgramFactory`, `NLPCreateQKVHeadsDecodeShardedProgramFactory`, `NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | No — `reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` calls into `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` which uses Device 1.0 idioms (`noc_async_read`, raw integer addresses). Sharded and subcoregrid kernels: Device 2.0 throughout. |
| *Prereqs* — Cross-op escapes | issue — `common.hpp` (data_movement family, Device 1.0) is a function-call donor from the interleaved kernel. Sharded/subcoregrid kernels have no cross-op escapes. |
| *Feature Support* — overall | GREEN (no UNSUPPORTED features in use) |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | present: `(c_16, q_output writer)` · `(c_17, k_output writer)` · `(c_18, v_output writer)` — all three factories. (workaround) |

---

## Result

**RED — blocked on Device 2.0 prerequisite**, routed to the Device 2.0 migration team.

Primary blocker: the interleaved factory's kernel (`reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp`, line 13) includes `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` and calls `tt::data_movement::common::tt_memmove<...>` (line 116), which internally invokes `noc_async_read` — a Device 1.0 free function. The donor (`common.hpp`) is not on Device 2.0.

**Scoped subset:** The sharded factory (`NLPCreateQKVHeadsDecodeShardedProgramFactory`) and subcoregrid factory (`NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory`) are both Device 2.0 compliant — their kernels use `Noc`, `CircularBuffer`, `TensorAccessor` consistently with no Device 1.0 holdovers and no cross-op donor issues. A scoped port of the sharded + subcoregrid paths (omitting the interleaved path) is feasible once the gates are otherwise cleared.

Path forward for the full port: the `data_movement/common/kernels/common.hpp` donor needs its `tt_memmove` function migrated to Device 2.0 idioms (or the interleaved kernel needs to replace its use of `tt_memmove` with a Device 2.0 equivalent). That is Device 2.0 track work and must precede the Metal 2.0 port of the interleaved path.

---

## Gate detail

### ProgramDescriptor

**GREEN.** All three factories declare and return `tt::tt_metal::ProgramDescriptor`. Their `.hpp` files include `<tt-metalium/program_descriptors.hpp>` and the `create_descriptor` signatures return `ProgramDescriptor`. Factory bodies use `CBDescriptor`, `KernelDescriptor`, `ReaderConfigDescriptor`, `WriterConfigDescriptor`, `KernelDescriptor::emplace_runtime_args`, etc. — the full `ProgramDescriptor` vocabulary.

### Device 2.0 (every kernel used)

**RED (GATE) — interleaved factory kernel; GREEN for sharded and subcoregrid kernels.**

#### Violation: `reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp`

This kernel's own code is Device 2.0 (`Noc noc`, `noc.async_read(qkv_reader, ...)`, `CircularBuffer cb_q_out(...)`, `TensorAccessor qkv_reader = TensorAccessor(qkv_args, q_start_addr)`, etc.) — the op-local code is clean. The problem is the cross-family donor call at line 116:

```
tt::data_movement::common::tt_memmove<false, true, true, SUBTILE_LINE_BYTES>(write_addr, scratch_read_offset, SUBTILE_LINE_BYTES);
```

This is declared in `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` (included at line 13). Inside `tt_memmove`, the call chain is:

```
tt_memmove → enhanced_noc_async_read → noc_async_read_one_packet / noc_async_read<N>
```

`noc_async_read` is a Device 1.0 free function taking raw integer source/destination addresses — the exact pre-Device-2.0 idiom that gates the port. The donor file `common.hpp` broadly uses Device 1.0 idioms (`noc_async_read`, `noc_async_write`, `noc_async_read_one_packet`, `noc_async_write_one_packet`, raw `uint64_t` NOC-encoded addresses, `get_noc_addr(src_l1_addr)`).

| File | Line | Call | Issue |
|---|---|---|---|
| `device/kernels/reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` | 13 | `#include "ttnn/operations/data_movement/common/kernels/common.hpp"` | donor include |
| `device/kernels/reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` | 116 | `tt::data_movement::common::tt_memmove<false,true,true,SUBTILE_LINE_BYTES>(...)` | calls into Device 1.0 donor |
| `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | 30 | `noc_async_read_one_packet(...)` | Device 1.0 |
| `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | 32 | `noc_async_read<...>(...)` | Device 1.0 |

**Route to:** Device 2.0 migration team, `data_movement` family. The donor `common.hpp` (or at minimum the `tt_memmove` path used here) must be migrated to Device 2.0 idioms before the Metal 2.0 port of the interleaved path can proceed.

#### Clean: sharded and subcoregrid kernels

Both `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` and `reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp` are Device 2.0 throughout:

- `Noc noc;` (Device 2.0 NOC wrapper)
- `noc.async_read(src_ep, ...)` with `UnicastEndpoint` (Device 2.0)
- `CircularBuffer cb_q_out(cb_id_q_out)`, `cb_q_out.get_write_ptr()` (Device 2.0)
- `TensorAccessor(index_args, batch_offset_tensor_addr)` (Device 2.0)
- No `InterleavedAddrGen`, `ShardedAddrGen`, raw `noc_async_read`, raw sem addresses.

No cross-op donor includes in either kernel.

### Feature compatibility

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer`, or `global_circular_buffer` field on any `CBDescriptor` in the op. |
| Dynamic CircularBuffer (borrowed memory) | GREEN | All three factories set `CBDescriptor::buffer = output[N].buffer()` for the q/k/v output CBs (c_16/c_17/c_18). LANDED — port uses `DataflowBufferSpec::borrowed_from`. See Fake CBs below for an important litmus caveat. |
| CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` field set to non-zero in any `CBDescriptor`. |
| Aliased Circular Buffers | N/A | All `CBDescriptor::format_descriptors` initializers are single-element (`{{CBFormatDescriptor{...}}}`). No aliased CBs. |
| GlobalSemaphore | N/A | No semaphores of any kind (`SemaphoreDescriptor`, `CreateSemaphore`, `GlobalSemaphore`) in any factory. |
| Non-zero semaphore initial value | N/A | No semaphores used. |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` tokens in any factory. All `TensorAccessorArgs` calls use the single-argument form `TensorAccessorArgs(buffer)`. |
| `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize`, `UpdateCircularBufferPageSize`, or `UpdateDynamicCircularBufferAddressAndTotalSize` calls anywhere. |
| Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed struct (`NlpCreateQkvHeadsDecodeInputs`), not a variable-count container. No `get_compile_time_arg_val(i)` in a loop with runtime-varying `i`. |

---

## Port-work summary *(mirrors the brief when the brief is issued)*

**Tensor bindings (per binding):**

All three factories share the same binding structure:

- **`input_tensor` (sharded factories)** — `Buffer*`-binding form: `rt.push_back(in_buffer)` at `nlp_create_qkv_heads_decode_sharded_program_factory.cpp:254` and `nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp:252`. Kernel reads as `q_start_addr = get_arg_val<uint32_t>(0)` and uses it for unicast NOC address arithmetic. **Case 1** — re-express via `TensorParameter` / `TensorBinding`. (The sharded kernel reads tensor data via raw `q_start_addr + in_tile_offset_by_batch` arithmetic through `UnicastEndpoint` NoC reads. A reviewer should confirm Case 1 vs. Case 2 given the sub-tile-line read pattern; assume Case 1 per recipe guidance.)
- **`input_tensor` (interleaved factory)** — `Buffer*`-binding form and `TensorAccessorArgs(in_buffer).append_to(compile_time_args)`: `nlp_create_qkv_heads_decode_interleaved_program_factory.cpp:152,186-187`. Kernel reads as `q_start_addr = get_arg_val<uint32_t>(1)` and constructs `TensorAccessor(qkv_args, q_start_addr)`. **Case 1** — re-express via `TensorParameter` / `TensorBinding`.
- **`batch_offset` tensor (sharded + subcoregrid factories)** — `Buffer*`-binding form: `rt.push_back(batch_offset_buffer)` at `nlp_create_qkv_heads_decode_sharded_program_factory.cpp:255` and `nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp:243`. `TensorAccessorArgs(batch_offset_buffer).append_to(compile_time_args)` at sharded line 189, subcoregrid line 185. Kernel reads as `batch_offset_tensor_addr = get_arg_val<uint32_t>(1)` and constructs `TensorAccessor(index_args, batch_offset_tensor_addr)`. **Case 1** — re-express via `TensorParameter` / `TensorBinding`. (Optional tensor; both present-and-absent cases must be handled.)

**Custom hash:** None — no `compute_program_hash` override found. No port work required.

---

## Heads-ups *(mirrors the brief when the brief is issued)*

### Notable LANDED constructs

**Dynamic CircularBuffer (borrowed memory) — with Fake CB caveat:**

All three factories create output CBs (c_16, c_17, c_18) with `CBDescriptor::buffer = output[N].buffer()` — the borrowed-memory CB pattern. This is LANDED (`DataflowBufferSpec::borrowed_from`). However, per the Fake CB litmus test, these CBs are **fake CBs** (address-only): the kernels access them exclusively via `get_write_ptr()` with no `push_back` / `pop_front` calls. There is no FIFO producer+consumer pair. A Metal 2.0 DFB requires ≥1 producer and ≥1 consumer, so these cannot be expressed as DFBs. The port resolves this with the sanctioned fake-CB workaround (see the porting recipe). See the Fake CBs section below.

### Fake CBs (address-only)

The Q/K/V output CBs in all three factories are fake CBs: they are set up with `CBDescriptor::buffer = output[N].buffer()` (borrowed-memory form) but no kernel ever calls `push_back` or `pop_front` on them — they are used purely as address sources via `get_write_ptr()`. The kernels write tensor data directly to those pointers without the FIFO protocol.

Edges:

| Factory | CB | kernel:endpoint | Evidence |
|---|---|---|---|
| Interleaved | `c_16` (q_output) | `reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp`:reader+writer | `cb_q_out.get_write_ptr()` only; no `push_back` / `pop_front` |
| Interleaved | `c_17` (k_output) | same kernel | `cb_k_out.get_write_ptr()` only |
| Interleaved | `c_18` (v_output) | same kernel | `cb_v_out.get_write_ptr()` only |
| Sharded | `c_16` (q_output) | `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp`:reader+writer | `cb_q_out.get_write_ptr()` only |
| Sharded | `c_17` (k_output) | same kernel | `cb_k_out.get_write_ptr()` only |
| Sharded | `c_18` (v_output) | same kernel | `cb_v_out.get_write_ptr()` only |
| Subcoregrid | `c_16` (q_output) | `reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp`:reader+writer | `cb_q_out.get_write_ptr()` only |
| Subcoregrid | `c_17` (k_output) | same kernel | `cb_k_out.get_write_ptr()` only |
| Subcoregrid | `c_18` (v_output) | same kernel | `cb_v_out.get_write_ptr()` only |

The batch_offset CBs (c_14 and c_15 in sharded+subcoregrid factories) are **not** fake — they have `reserve_back(1)` + `push_back(1)` calls in the kernels and are therefore legitimate DFBs.

The port resolves fake CBs with the sanctioned fake-CB workaround (see `port_op_to_metal2_recipe.md`). This does **not** gate the port; it is a FYI-P heads-up.

### Cross-op / shared kernels

**Interleaved factory — cross-family function-call donor:**

`reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` line 13 includes:
```
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
```

The kernel calls `tt::data_movement::common::tt_memmove<false,true,true,SUBTILE_LINE_BYTES>(...)` (line 116). The donor is in the `data_movement` family and uses Device 1.0 idioms internally. This is the **Device 2.0 gate violation** detailed above; it also appears here as a cross-op coupling signal. The donor's Device 2.0 migration must complete before the interleaved kernel can port.

Sharded and subcoregrid kernels have no cross-op includes.

### RTA varargs

The sharded factory kernel (`reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp`) and subcoregrid factory kernel (`reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp`) both read variable-count NoC coordinate arrays from the RT arg region:

```cpp
tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(3 + num_x)); // sharded
// or: get_arg_addr(3 + in_num_cores) // subcoregrid
```

These are counted arrays appended at the end of each core's RT arg list (`rt.append(noc_x_coords)`, `rt.append(noc_y_coords)`) in the program factories. The count is known at factory-construction time (from the core-grid shape). Metal 2.0 supports RTA varargs; the port will use named RTAs for the per-core index argument and the vararg mechanism for the NoC coordinate arrays. The porter should confirm whether named RTAs (one per array element, when the max length is bounded) or the Metal 2.0 vararg mechanism (for genuinely loop-retrieved variable-count) is preferred here.

Recognition sites:
- `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:42-43`
- `reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp:42-43`

### TTNN factory analysis (porter-relevant)

- **Pybind `create_descriptor`:** None. The `nlp_create_qkv_heads_decode_nanobind.cpp` binds only the op function via `bind_function<"nlp_create_qkv_heads_decode">` — no `ProgramFactory.create_descriptor` binding.
- **Other risky pybind:** None detected.
- **Custom `override_runtime_arguments`:** None — no `override_runtime_arguments` declaration in any of the three factory `.hpp` or `.cpp` files.

---

## Team-only

### TensorAccessor convertibility (Case 2 annotations)

No Case-2 bindings classified — all bindings are Case 1 or the user should confirm after reviewing the sharded access pattern.

The sharded input access (`q_start_addr + in_tile_offset_by_batch`, navigating across shards via a precomputed NoC coordinate table) could be argued exotic. However, per recipe guidance, this is assumed Case 1 until the user confirms otherwise. The porter should evaluate at port time whether a `TensorAccessor` page-by-page iteration can express the sub-tile-line reads via unicast NoC coordinates, or whether `get_bank_base_address` bridge (Case 2) is needed. If the user confirms Case 2, surface the following:

> The use of `TensorAccessor` is an ergonomic choice on Gen1 architectures. It has meaningful performance implications on Gen2 architectures. Ideally, `TensorAccessor` should be updated to support the required iteration pattern; consider filing an issue requesting that support.

### Out-of-directory coupling and donor shape analysis

**Op-level roll-up: ⭐ blocked** — interleaved kernel has a Device-1.0 donor (Shape 4). Sharded and subcoregrid kernels are ✓ clean.

**Summary table:**

| Op kernel | Donor file | Shape | Status |
|---|---|---|---|
| `reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | Shape 4 (pre-Device-2.0) | ⭐ Device 2.0 GATE |
| `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` | (none) | — | ✓ clean |
| `reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp` | (none) | — | ✓ clean |

**Per-call detail — `common.hpp`:**

Functions called by the interleaved kernel:

| Function | Signature shape | Status |
|---|---|---|
| `tt::data_movement::common::tt_memmove<bool,bool,bool,uint32_t>(dst_l1_addr, src_l1_addr, bytes)` | uint32_t L1 addresses, internally calls raw `noc_async_read` | ⭐ Shape 4 — pre-Device-2.0 donor |

The `common.hpp` donor also declares `enhanced_noc_async_read`, `enhanced_noc_async_write`, `noc_async_write_sharded`, `noc_async_read_sharded` — all using Device 1.0 `noc_async_read` / `noc_async_write` free functions. Only `tt_memmove` is called by this op.

**Borrowed kernel files (file-path kernel instantiation):**

All three kernel `.cpp` files are owned by this op's directory. No file-path borrowing from external families.

### Relaxation candidates

No custom `compute_program_hash` to mine. No relaxation candidates.

### TTNN factory analysis (six questions)

1. **Op-owned tensors?** No. `create_output_tensors` calls `create_device_tensor` three times for the Q/K/V outputs, but these are the declared output tensors (`tensor_return_value_t = std::vector<Tensor>`), not intermediate scratch. `nlp_create_qkv_heads_decode_device_operation.cpp:181-185`.

2. **MeshWorkload concept needed?** No. Single-program op; no `create_mesh_workload` / `create_workload_descriptor` / `cached_mesh_workload_t` anywhere. No op-owned tensors (Q1 is No), so no plumbing-artifact MeshWorkload path either.

3. **Pybind `create_descriptor`?** No. `nlp_create_qkv_heads_decode_nanobind.cpp` only binds the op function itself: `ttnn::bind_function<"nlp_create_qkv_heads_decode", "ttnn.experimental.">(mod, ...)`. No `nb::class_<...ProgramFactory>(...).def_static("create_descriptor", ...)`.

4. **Other migration-risky pybind?** No. No `nb::class_<>` wrapping any DeviceOperation or factory type. The nanobind file is exclusively an op-function binding.

5. **Custom hash?** No. No `compute_program_hash` member or override in `NLPCreateQKVHeadsDecodeDeviceOperation` or in any factory class.

6. **Custom override-runtime-args?** No. No `override_runtime_arguments` declaration in any factory `.hpp` or `.cpp` file.

---

## Misc anomalies *(team-only, non-gating)*

- **Subcoregrid factory: `v_shard_spec` uses Q shard spec.** `nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp:114`: `const auto v_shard_spec = output[0].shard_spec().value();` — this takes the Q shard spec rather than the V shard spec (`output[2]`). The sharded factory does the same thing at line 73-75 but uses `v_shard_spec.grid` from `output[2].shard_spec()`. The subcoregrid factory then at line 115 uses `const auto v_cores = q_shard_spec.grid` (not `v_shard_spec.grid`), so the CB and grid ultimately derive from Q regardless. This appears intentional (Q and V share the same core grid per the op's design), but the naming is misleading. Not a Metal 2.0 port concern; flagged for op-owner awareness.

- **Interleaved factory: `v_cores` assigned from `q_shard_spec.grid` rather than `v_shard_spec.grid`.** `nlp_create_qkv_heads_decode_interleaved_program_factory.cpp:74`: `auto v_cores = q_shard_spec.grid;` (not `v_shard_spec.grid`). Same as the subcoregrid note — intentional per design but potentially confusing.

---

## Questions for the user

1. **Sharded input Case 1 vs. Case 2 confirmation:** The sharded and subcoregrid kernels read the input tensor (`in_buffer`) via raw base-address + per-batch offset arithmetic with unicast NoC reads across a precomputed coordinate table (e.g., `qkv_read_addr = q_start_addr + in_tile_offset_by_batch`). The access navigates shard boundaries by iterating over a NoC-x/y coordinate table rather than via `TensorAccessor` page IDs. Is this expressible as a `TensorAccessor` page-by-page iteration (Case 1), or does the sub-tile-line granularity and the coordinate-table navigation make it genuinely exotic (Case 2)? Relevant files: `device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:79-243` and `device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp:79-237`.

---

## Recipe notes

- The recipe's Device 2.0 gate section focuses on the op's own kernels plus donor kernels it calls. In this op, the op's own kernel code is fully Device 2.0 compliant, but a helper function it calls (`tt_memmove` from `common.hpp`) is implemented in Device 1.0 idioms inside a `kernel_helper_functions`-class donor (class 4 in the out-of-directory coupling taxonomy). The gate fired correctly here, but the recipe could be clearer that a class-4 donor (utility header) with Device 1.0 internals is equivalent to a class-6 cross-family donor for Device 2.0 gating purposes — the text currently emphasizes "cross-family" and "shared kernel library" as the typical concern, which could cause an auditor to miss a utility-header donor.

- The "Fake CB litmus" rule in the audit and the "Dynamic CircularBuffer / borrowed-memory DFB" feature entry are closely related but require the auditor to check both in sequence: a `CBDescriptor::buffer` set to non-null fires the Dynamic CB LANDED feature entry, then the Fake CB litmus determines whether it's a real DFB or just an address lookup. In this op, all three output CBs fire the LANDED feature entry first, then fail the litmus (no push/pop), making them fake CBs. The audit recipe covers this correctly, but the sequencing is easy to miss — a reader doing a quick feature-compatibility scan might mark "Dynamic CB: GREEN" and overlook the follow-on fake-CB determination.
