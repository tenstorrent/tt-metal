# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/eltwise/binary_ng`

**Identifying section:**

- **`BinaryNgDeviceOperation`** (single device operation)
  - `ProgramFactory` (`binary_ng_program_factory.cpp`)
    - Covers all broadcast sub-types: NONE, ROW_A/B, COL_A/B, SCALAR_A/B, ROW_A_COL_B, ROW_B_COL_A
    - Two layout paths: tile (default) and row-major (inputs_row_major)
    - Two broadcast implementation layers: `kernels/` (legacy ng kernels) and `kernels_ng/` (new ng kernels); the factory uses both via `KernelName` dispatch (see `binary_ng_utils.cpp:get_kernel_file_path`)

Kernel files used by the factory (via `get_kernel_file_path`):

Tile path — own kernels:
- `device/kernels/dataflow/reader_interleaved_no_bcast.cpp` (`ReaderNoBcast`)
- `device/kernels/dataflow/writer_interleaved_scalar.cpp` (`WriterScalar`, `WriterNoBcastNg` overrides via kernels_ng)
- `device/kernels/compute/eltwise_binary_no_bcast.cpp`, `eltwise_binary.cpp`, `eltwise_binary_scalar.cpp`
- `device/kernels/compute/eltwise_binary_sfpu*.cpp`, `eltwise_where*.cpp`

Row-major path — kernels_ng:
- `device/kernels_ng/dataflow/reader_interleaved_rm_no_bcast.cpp` (`ReaderRmNoBcastNg`)
- `device/kernels_ng/dataflow/reader_interleaved_rm_row_bcast.cpp` (`ReaderRmRowBcastNg`)
- `device/kernels_ng/dataflow/reader_interleaved_rm_col_bcast.cpp` (`ReaderRmColBcastNg`)
- `device/kernels_ng/dataflow/reader_interleaved_rm_row_col_mixed_bcast.cpp` (`ReaderRmRowBColABcastNg`)
- `device/kernels_ng/dataflow/reader_interleaved_rm_scalar_bcast.cpp` (`ReaderRmScalarBcastNg`)
- `device/kernels_ng/dataflow/reader_interleaved_rm_scalar_op.cpp` (`ReaderRmScalarOpNg`)
- `device/kernels_ng/dataflow/writer_interleaved_rm_no_bcast.cpp` (`WriterRmNoBcastNg`)

Tile-path kernels_ng (broadcast sub-types with LLK bcast):
- `device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` (`ReaderNoBcastNg`)
- `device/kernels_ng/dataflow/reader_interleaved_row_bcast.cpp` (`ReaderRowBcastNg`)
- `device/kernels_ng/dataflow/reader_interleaved_col_bcast.cpp` (`ReaderColBcastNg`)
- `device/kernels_ng/dataflow/reader_interleaved_row_col_mixed_bcast.cpp` (`ReaderRowBColABcastNg`)
- `device/kernels_ng/dataflow/reader_interleaved_scalar_bcast.cpp` (`ReaderScalarBcastNg`)
- `device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` (`WriterNoBcastNg`)
- `device/kernels_ng/compute/eltwise_binary_{row,col,scalar,row_col}_bcast.cpp`
- `device/kernels_ng/compute/eltwise_binary_sfpu_{row,col,scalar,row_col}_bcast.cpp`
- `device/kernels_ng/compute/eltwise_where_sfpu_{row,row_col}_bcast.cpp`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng` |
| **Overall** | YELLOW |
| **DOps / Factories** | `BinaryNgDeviceOperation` → `ProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes-with-holdovers (YELLOW — fix on D2.0 track first) |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None |

---

## Result

**YELLOW — isolated Device 2.0 holdovers in RM-path dataflow kernels, blocked until those are cleaned on the Device 2.0 track.** All other gates clear. The port is structurally feasible.

The blocking holdovers are `noc_async_read` / `noc_async_write` free function calls in 7 row-major-path dataflow kernels under `device/kernels_ng/dataflow/`. All 7 kernels otherwise consistently use Device 2.0 wrappers (`Noc noc;`, `CircularBuffer`, `TensorAccessor`). These are isolated Device 2.0 cleanup items — they are NOT part of the Metal 2.0 port scope. They must be cleaned first, on the Device 2.0 track, before the port begins.

Once the holdovers are fixed, no re-audit is needed — proceed with the porter brief.

---

## Gate detail

### ProgramDescriptor

**GREEN.** The op uses `ProgramDescriptor`, `KernelDescriptor`, `CBDescriptor`, `CBFormatDescriptor`, `ReaderConfigDescriptor`, `WriterConfigDescriptor`, `ComputeConfigDescriptor`. No `CreateProgram` / `CreateKernel` / `SetRuntimeArgs` / `CreateCircularBuffer` imperative API present.

- `binary_ng_program_factory.cpp:384` — `ProgramDescriptor BinaryNgDeviceOperation::ProgramFactory::create_descriptor(...)`
- `binary_ng_program_factory.cpp:562` — `desc.cbs.push_back(CBDescriptor{...})`
- `binary_ng_program_factory.cpp:704` — `KernelDescriptor writer_desc;`

### Device 2.0 (every kernel used)

**YELLOW — isolated holdovers in RM-path dataflow kernels; structurally Device 2.0 everywhere.**

All tile-path kernels (`kernels/dataflow/`, `kernels_ng/dataflow/reader_interleaved_*.cpp` excluding RM variants, `kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`) are fully Device 2.0 compliant: they use `Noc noc;`, `CircularBuffer cb(id)`, `TensorAccessor(args, addr)`, and the member-form CB operations throughout.

The 7 RM-path kernels below are otherwise structurally Device 2.0 (they declare `Noc noc;` and `CircularBuffer` objects; binding tokens will attach) but contain isolated `noc_async_read` / `noc_async_write` free function calls that should be replaced with `noc.async_read(...)` / `noc.async_write(...)`.

**Holdovers to fix on the Device 2.0 track (NOT in the port diff):**

| File | Line | Call | Device 2.0 object in scope |
|---|---|---|---|
| `kernels_ng/dataflow/reader_interleaved_rm_no_bcast.cpp` | 139 | `noc_async_read(addr_a, curr_l1_a, current_read_len_a)` | `Noc noc` (line 47) |
| `kernels_ng/dataflow/reader_interleaved_rm_no_bcast.cpp` | 148 | `noc_async_read(addr_b, curr_l1_b, current_read_len_b)` | `Noc noc` (line 47) |
| `kernels_ng/dataflow/writer_interleaved_rm_no_bcast.cpp` | 84 | `noc_async_write(l1_read_addr, dst_noc_addr, current_write_len)` | `Noc noc` (line 33) |
| `kernels_ng/dataflow/reader_interleaved_rm_row_bcast.cpp` | 144, 150, 160, 165 | `noc_async_read(...)` × 4 | `Noc noc` (line ~48) |
| `kernels_ng/dataflow/reader_interleaved_rm_col_bcast.cpp` | 175, 186, 197, 210 | `noc_async_read(...)` × 4 | `Noc noc` (in scope) |
| `kernels_ng/dataflow/reader_interleaved_rm_scalar_bcast.cpp` | 170, 180, 192, 201 | `noc_async_read(...)` × 4 | `Noc noc` (in scope) |
| `kernels_ng/dataflow/reader_interleaved_rm_row_col_mixed_bcast.cpp` | 175, 183, 197, 205 | `noc_async_read(...)` × 4 | `Noc noc` (in scope) |
| `kernels_ng/dataflow/reader_interleaved_rm_scalar_op.cpp` | 127 | `noc_async_read(addr_a, curr_l1_a, current_read_len)` | `Noc noc` (line 43) |

Each holdover replaces as: `noc_async_read(src_addr, dst_l1, size)` → `noc.async_read(src_ep, dst_l1, size, {.addr=...}, {})` and `noc_async_write(src_l1, dst_addr, size)` → `noc.async_write(src_l1, dst_ep, size, {}, {.addr=...})`. These are mechanical 1-to-1 substitutions.

**Note on compute kernels:** `device/kernels/compute/eltwise_utils.hpp` and the compute kernel `.cpp` files use `cb_wait_front`, `cb_reserve_back`, `cb_pop_front` etc. These are compute/LLK-side CB primitives that are architecturally distinct from the dataflow Device 2.0 scope. They are not dataflow kernels and do not perform NoC or TensorAccessor operations; they are left as-is and are not holdovers in the Device 2.0 dataflow sense.

### Feature compatibility

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | Not in use — no `GlobalCircularBuffer` type or `global_circular_buffer` CBDescriptor field set |
| Dynamic CircularBuffer (borrowed memory) | GREEN | CBDescriptor `.buffer` set to non-null for sharded tensors — port uses `DataflowBufferSpec::borrowed_from` |
| CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` not set on any CBDescriptor |
| Aliased Circular Buffers | N/A | All CBDescriptors have single-element `format_descriptors` — no aliasing |
| GlobalSemaphore | N/A | No semaphores used |
| Non-zero semaphore initial value | N/A | No semaphores used |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | GREEN | `ArgConfig::RuntimeTensorShape` used for a, b, c buffers — LANDED, but UNSAFE opt-in; FYI-P heads-up below |
| `UpdateCircularBuffer*` | N/A | None of the Update* free functions appear in op code or override hooks |
| Variable-count compile-time arguments (CTA varargs) | N/A | Fixed input count (tensor_args_t has named tensors, not a vector) |

**Dynamic TensorAccessor detail (FYI-P heads-up):**
- `binary_ng_program_factory.cpp:700` — `TensorAccessorArgs(*c_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape).append_to(writer_compile_time_args, writer_common_runtime_args)` — c binding
- `binary_ng_program_factory.cpp:855` — `TensorAccessorArgs(*a_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape).append_to(reader_compile_time_args, reader_common_runtime_args)` — a binding
- `binary_ng_program_factory.cpp:858` — `TensorAccessorArgs(b_buffer != nullptr ? *b_buffer : *a_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape).append_to(...)` — b binding (falls back to a_buffer when b is scalar)

Metal 2.0 supports this via `TensorParameterAdvancedOptions::dynamic_tensor_shape = true`, but that opt-in is marked **UNSAFE** in the framework header and has structural implications for per-dispatch caching. The default is strict; applying the relaxation is an explicit user-OK decision (per `port_op_to_metal2_ttnn_factory.md`).

**Dynamic CircularBuffer detail (FYI-P heads-up):**
- `binary_ng_program_factory.cpp:570` — `CBDescriptor{..., .buffer = a_sharded ? a_buffer : nullptr}` (CB c_0, a-tensor backing when a is sharded)
- `binary_ng_program_factory.cpp:602` — `CBDescriptor{..., .buffer = b_sharded ? b_buffer : nullptr}` (CB c_1, b-tensor backing when b is sharded)
- `binary_ng_program_factory.cpp:660` — `CBDescriptor{..., .buffer = c_sharded ? c_buffer : nullptr}` (CB c_2, c-tensor backing when c is sharded)

When the buffer pointer is non-null (sharded path), these are borrowed-memory DFBs. The port declares these `DataflowBufferSpec` entries with `borrowed_from = <tensor_parameter_name>`. When null (interleaved path), they are plain static DFBs.

---

## Port-work summary *(mirrors the brief)*

- **Tensor bindings** — all three bindings are Case 1 (re-express):
  - `a` — Case 1: `a.buffer()->address()` flows into per-core reader RTAs (e.g. `reader_runtime_args[0]` non-RM path, `reader_runtime_args[0]` RM path); `TensorAccessorArgs` also set up via common_runtime_args. Re-express via `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(ta::name)` and the per-core buffer-address RTA disappears.
  - `b` — Case 1: same pattern. `b->buffer()->address()` at index 15 of non-RM reader RTAs; `b_addr = b->buffer()->address()` at index 7 of RM reader RTAs (`binary_ng_program_factory.cpp:1220`). Note: when b is scalar (no b tensor), b's TensorAccessorArgs uses a_buffer as fallback — the port would need a conditional TensorParameter for b.
  - `c` — Case 1: `c.buffer()->address()` at index 0 of writer RTAs for both RM and non-RM paths (`binary_ng_program_factory.cpp:1117`, `1133`, `1183`, `1198`). Re-express via `TensorParameter` / `TensorBinding`.

- **Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception). Location: `binary_ng_device_operation.cpp:487`.

---

## Heads-ups *(mirrors the brief)*

- **Device 2.0 holdovers (BLOCKER — fix on D2.0 track BEFORE porting):** 8 RM-path dataflow kernel files contain isolated `noc_async_read` / `noc_async_write` free function calls that must be converted to `noc.async_read(...)` / `noc.async_write(...)` before the port begins. See Gate detail → Device 2.0 for the full table. This is Device 2.0 cleanup, not port scope; fix first, then port.

- **Notable LANDED constructs:**
  - Dynamic CircularBuffer (borrowed memory): `CBDescriptor::buffer` set for sharded tensors at lines 570, 602, 660 of `binary_ng_program_factory.cpp`. Port uses `DataflowBufferSpec::borrowed_from`. Confirm the borrowed-CB and TensorParameter are consistently co-bound at port time.
  - Dynamic TensorAccessor (`ArgConfig::RuntimeTensorShape`): 3 sites in `binary_ng_program_factory.cpp` (lines 700, 855, 858). Metal 2.0 path is `TensorParameterAdvancedOptions::dynamic_tensor_shape = true` — UNSAFE opt-in with caching implications. Confirmed-applicable here per the example in the audit recipe.

- **Cross-op / shared kernels:** All out-of-directory `#include` escapes from `kernels_ng/` are into `kernels/` within the same `binary_ng` family — in-family escapes only. Headers consumed: `eltwise_utils.hpp`, `eltwise_utils_common.hpp`, `eltwise_utils_sfpu.hpp`, `fill_tile_utils.hpp`. The `kernels/` header content (inline LLK helpers, fill utilities) has no Device 2.0 or Metal 2.0 surface; it is pure compute/data-manipulation logic. No port-together coupling beyond the binary_ng family itself.

- **RTA varargs:** None. All RTA layouts are fixed-count per code path.

- **TTNN factory analysis (porter-relevant):**
  - Pybind `create_descriptor`: None — no `nb::class_<...ProgramFactory>(...)` or `.def_static("create_descriptor", ...)` found in binary_ng. (No nanobind source file exists for binary_ng itself.)
  - Other risky pybind: None.
  - Custom `override_runtime_arguments`: None — no `static void ProgramFactory::override_runtime_arguments(...)` found.

---

## Team-only

### TensorAccessor convertibility

All three bindings (a, b, c) are Case 1. None are genuinely exotic:

- **a binding:** page-by-page read with `TensorAccessor::get_noc_addr(page_id)` call pattern (line 106 in `reader_interleaved_no_bcast.cpp`: `noc.async_read(src, cb_src, ..., {.page_id = tile_offset + tw}, ...)`). Standard interleaved accessor; fully convertible.
- **b binding:** same pattern — standard interleaved accessor.
- **c binding:** same pattern for write path — standard interleaved accessor.
- RM path (reader_interleaved_rm_no_bcast.cpp): uses `src.get_noc_addr(row_idx_a)` then `noc_async_read(addr_a + offset, ...)`. The `get_noc_addr` call is from TensorAccessor (correct), and the `+ current_chunk_offset` is a sub-page byte offset added to the returned NoC address. This is an inline byte-offset pattern on top of TensorAccessor; it's awkward but not exotic — the accessor provides the page base, and the offset is added on the host arithmetic. Still Case 1 (once the noc_async_read holdovers are fixed, the binding is clean).

**Note on b-as-scalar fallback:** When `b` is a scalar (no b tensor), the factory passes `*a_buffer` to `TensorAccessorArgs` for b (line 858). At port time, the `b` TensorParameter should be `std::optional<TensorParameter>` or conditioned on `b.has_value()`. This is standard port-time judgment, not an exotic pattern.

### Out-of-directory coupling and donor shape analysis

**Op-level roll-up:** `✓ clean` — no cross-family coupling at all.

**Summary table (op kernel → donor file):**

| Op kernel | Donor file | Class |
|---|---|---|
| All `kernels_ng/compute/*.cpp` files that include utilities | `kernels/compute/eltwise_utils_common.hpp` | In-family shared |
| Some `kernels_ng/compute/*.cpp` | `kernels/compute/eltwise_utils.hpp` | In-family shared |
| Some `kernels_ng/compute/*.cpp` | `kernels/compute/eltwise_utils_sfpu.hpp` | In-family shared |
| Several `kernels_ng/dataflow/*.cpp` | `kernels/dataflow/fill_tile_utils.hpp` | In-family shared |
| `kernels/dataflow/writer_interleaved_scalar.cpp` | `kernels/dataflow/fill_tile_utils.hpp` | In-family (same `kernels/` dir) |

**Per-call detail:** Not needed — all `✓` escapes within the same op family.

**Borrowed kernel files (file-path kernel instantiation):**

All kernel `.cpp` files instantiated by the `ProgramFactory` live within the `binary_ng` directory itself (`device/kernels/` and `device/kernels_ng/`). The op owns all of its kernels. There is no borrowing from shared pools or other op families.

No file-path coupling beyond the binary_ng family.

### Relaxation candidates (mined from custom hash — FALLIBLE)

The custom `compute_program_hash` (in `binary_ng_device_operation.cpp:487`) keys on:
- `attributes` (via `operation_attributes_t::to_hash()`, which includes `memory_config`, op type, kernel config, dtype, broadcast type, etc., but **not** tensor shapes)
- `input_tensor_a.dtype()`, `input_tensor_a.memory_config()`
- `input_tensor_b->dtype()`, `input_tensor_b->memory_config()` (when b is a tensor)
- `shard_volumes` (derived from tensor specs — implicitly captures shard tile counts)

**`to_hash()` intentionally omits `post_activations` for where/quant ops** (`binary_ng_device_operation.cpp:234`): `(is_where_op || is_quant_op) ? SmallVector<...>{} : post_activations`. This means two WHERE or QUANT ops with different post_activations will hit the same hash bucket. This looks like an intentional optimization (zero-point value comes in via RTA, not CTA), but it is a deviation from the default strict hash and could be a correctness concern for the default Metal 2.0 hash if post_activations affect the compiled program. Flag to the op owner.

**Relaxation candidates (FALLIBLE — verify before applying):**
- The hash does not explicitly include tensor shapes beyond what `shard_volumes` captures. This suggests the op may tolerate different tensor shapes that produce the same shard volumes. Candidate for `dynamic_tensor_shape` relaxation — but note that `ArgConfig::RuntimeTensorShape` is already used, which implies shape does vary at runtime; the relaxation is consistent with this. The default Metal 2.0 hash is strict (keys on full `TensorSpec`); the relaxation to `dynamic_tensor_shape=true` would align with the existing custom hash's behavior.
- `input_layout_a`, `input_layout_b`, `output_layout` are in `to_hash()` — layout is considered part of program identity. No relaxation candidate here.

### TTNN factory analysis (full six-question record)

**Q1 — Op-owned tensors:** No. The factory (`create_descriptor`) constructs only CBDescriptors and KernelDescriptors from the provided `tensor_args`. The `create_output_tensors` method calls `create_device_tensor(compute_output_specs(...), ...)` but that is the declared output tensor, not a factory-owned intermediate. No `create_device_tensor` or `allocate_tensor_on_device` calls inside the program factory itself. The standard device-op output allocation is not an "op-owned tensor" in the MeshWorkload sense.

**Q2 — MeshWorkload concept needed:** No. The device operation does not define `create_mesh_workload` or `cached_mesh_workload_t`. Q1 is No, so the MeshWorkload-path artifact doesn't apply either. Single-program, single-device op.

**Q3 — Pybind `create_descriptor`:** No. No `nb::class_<...ProgramFactory>(...)` binding found. The op does not appear in any nanobind `.cpp` file under `ttnn/cpp/ttnn-nanobind/`. The normal op-function / program-config surface is exposed elsewhere (binary family-level bindings), but `create_descriptor` itself is not exposed.

**Q4 — Other migration-risky pybind:** None found. No nanobind file targets `BinaryNgDeviceOperation` or `ProgramFactory` internals.

**Q5 — Custom hash:** Yes. `binary_ng_device_operation.cpp:487`. Treatment: delete and revert to default TTNN hash per the Custom program hash subject. See also the relaxation candidate note above — the existing hash's behavior (shape-tolerant, `post_activations`-omitting for certain ops) should be noted in the port plan.

**Q6 — Custom override-runtime-args:** No. No `override_runtime_arguments` static method found in ProgramFactory.

---

## Misc anomalies *(team-only, non-gating)*

1. **`b_buffer` TensorAccessorArgs fallback (`binary_ng_program_factory.cpp:857-859`):** When `b` is a scalar (no tensor), the factory passes `*a_buffer` to `TensorAccessorArgs` for the b slot. This means the b-slot CTA/CRTA will encode a_buffer's bank layout — but the RM scalar-op reader (`reader_interleaved_rm_scalar_op.cpp`) skips the b TensorAccessorArgs offset entirely (`constexpr auto src_args = TensorAccessorArgs<0>();` — only one accessor). In the non-RM scalar case (`ReaderNoBcast`), the b CB is filled with the scalar value by the writer, so no tensor b is read at all. The fallback is harmless but mildly confusing. Route to op owner to document or guard with `TT_ASSERT(b_buffer != nullptr)` before accessing.

2. **`operation_attributes_t::to_hash()` omits `post_activations` for where/quant ops (`binary_ng_device_operation.cpp:234`):** As noted in relaxation candidates, two WHERE or QUANT ops with different post_activations (different zero-point values) will collide in the hash. The zero-point is plumbed as an RTA, not a CTA, so the compiled program is the same regardless — this is intentional. The omission is correct for the current design. However, if future changes promote zero-point to a CTA (e.g. for performance), this hash omission would become a silent correctness bug. Op owner should comment this intent.

3. **`eltwise_where_sfpu_scalar.cpp` referenced via `KernelName::ComputeScalar` / `is_where_op`:** In `get_kernel_file_path`, `ComputeScalar` with `is_where_op` produces `"eltwise_where_sfpu_scalar"` (missing `.cpp` extension at `binary_ng_utils.cpp:126`). If this path is actually exercised at runtime it would fail to find the kernel file. Route to op owner to verify whether this code path is reachable and fix the extension if so.

---

## Questions for the user

1. **RM-kernel `noc_async_read`/`noc_async_write` holdover classification:** The recipe's YELLOW carve-out for isolated Device 2.0 holdovers describes "the CB-index-keyed free-function family" — e.g. `get_read_ptr(cb_id)` → `cb_obj.get_read_ptr()`. The RM-path kernels use `noc_async_read` / `noc_async_write` (raw NoC free functions, not CB-index functions) alongside a declared `Noc noc;` object. The Device 2.0 replacement exists (`noc.async_read(...)` / `noc.async_write(...)`), the wrappers ARE in scope, and the calls are isolated within otherwise-Device-2.0 kernels. This audit classifies these as YELLOW holdovers by analogy, but the recipe's definition is strictly about CB-index free functions. Please confirm: should this be treated as YELLOW (isolated holdover, structurally feasible) or RED (pre-Device-2.0 idiom, blocked)?

---

## Recipe notes

1. **YELLOW carve-out scope for holdovers:** The recipe (Check 2) defines YELLOW holdovers as "isolated legacy holdovers from the **CB-index-keyed free-function family**: free functions taking a `uint32_t` CB index where the corresponding Device-2.0 wrapper object is already in scope." The `noc_async_read`/`noc_async_write` calls in the RM kernels do NOT fit this definition — they are NoC-level free functions, not CB-index functions. But the structural condition (Device 2.0 wrapper in scope, isolated instances, mechanically replaceable) is fully met. The recipe doesn't explicitly address this case; an auditor must either classify as RED (strict reading of the holdover definition) or YELLOW (intent-based reading). This audit chose YELLOW by intent, but the recipe should clarify whether the CB-index language is intentionally exclusive or illustrative.

2. **`noc_async_read` with `TensorAccessor::get_noc_addr` output:** The RM kernels call `src.get_noc_addr(row_idx)` (Device 2.0 TensorAccessor member), then pass the result to `noc_async_read(addr, l1, size)` (Device 1.0 free function). The mixed usage — Device 2.0 for address computation, Device 1.0 for the actual transfer — is an unusual pattern. The recipe's guard "do NOT flag `accessor.get_noc_addr(page_id)` outputs" in TensorAccessor handling guards against false-positive TensorAccessor classification, not against Device 2.0 classification of `noc_async_read`. This audit treats the `noc_async_read` calls as holdovers (not the `get_noc_addr` calls). Recipe may benefit from an explicit note here.
