# Metal 2.0 Pre-Port Audit — sdpa_decode

**Op:** `ttnn/cpp/ttnn/operations/transformer/sdpa_decode`
**Audited:** 2026-06-11
**Recipe version:** port_op_to_metal2_audit.md (3f5d01812f3)
**Overall:** YELLOW — feasible; Device 2.0 holdover cleanup required first (on Device 2.0 track, not in the port)

---

## Subject 1 — Prerequisites

### 1a. ProgramDescriptor API (Gate)

**Result: GREEN**

`device/sdpa_decode_device_operation.hpp` includes `<tt-metalium/program_descriptors.hpp>` and
declares `create_descriptor`. The factory at `device/sdpa_decode_program_factory.cpp:508` opens
with `ProgramDescriptor desc;` and builds the program entirely with `CBDescriptor`, `KernelDescriptor`,
`SemaphoreDescriptor`, and `KernelDescriptor::emplace_runtime_args`. No legacy `CreateProgram` /
`CreateKernel` / `SetRuntimeArgs` calls present.

### 1b. Device 2.0 Kernel APIs (Gate — isolated holdovers permitted)

**Result: YELLOW (isolated holdovers in two files)**

#### Kernel coverage

| Kernel | Status |
|---|---|
| `device/kernels/dataflow/reader_decode_all.cpp` | Clean Device 2.0 — `Noc`, `CircularBuffer`, `TensorAccessor`, `UnicastEndpoint` throughout |
| `device/kernels/dataflow/writer_decode_all.cpp` | Mostly Device 2.0; five isolated holdovers (see below) |
| `device/kernels/compute/sdpa_flash_decode.cpp` | Out of scope — compute kernel consuming/producing CBs only |

#### Out-of-directory kernel files

| File | Status |
|---|---|
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp` | Clean Device 2.0 — `Noc`, `CircularBuffer`, `TensorAccessor`, `CoreLocalMem` |
| `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | Device 1.0 throughout — `cb_reserve_back`, `get_write_ptr`, `cb_push_back` (see below) |
| `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` | Clean Device 2.0 |
| `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | Clean Device 2.0 |

#### Local dataflow_common.hpp

`device/kernels/dataflow/dataflow_common.hpp` includes the sdpa family header and adds local
helper functions with Device 1.0 free-function holdovers:

- Line 37: `get_write_ptr(cb_id)` inside `fill_tile<>` (Device 2.0 `CircularBuffer cb(cb_id)` is
  in scope on the other branch of the same function — consistent wrapper already present)
- Lines 58, 60: `get_write_ptr(cb_id)` in `fill_tile_partial`
- Lines 120, 122: `get_write_ptr(cb_id)` in `fill_tile_partial_sliding_window`

#### writer_decode_all.cpp holdovers

- Line 32: `uint32_t reducer_semaphore_addr = get_semaphore(reducer_semaphore_id);`
- Line 33: `uint32_t output_semaphore_addr = get_semaphore(get_compile_time_arg_val(12));`
- Line 201: `get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, reducer_semaphore_addr)` — passes
  the raw semaphore address obtained above
- Line 431: `noc_semaphore_wait(output_self_semaphore_addr_ptr, num_reducers_to_wait)` — pointer
  cast from `output_semaphore_addr`
- Lines 487-489: `get_noc_addr(output_core_noc_x, output_core_noc_y, output_semaphore_addr)` +
  `noc_semaphore_inc(output_core_semaphore_noc_addr, 1)` — classic D1.0 remote semaphore increment

  Device 2.0 `Semaphore<>` objects and `noc.async_write(...)` / `.up()` / `.wait_min()` are used
  elsewhere in the same file (e.g., line 390: `Semaphore<>(reducer_semaphore_id).up(noc, ...)`),
  so the wrappers are in scope. These holdovers are a D2.0-track cleanup item.

#### generate_bcast_scalar.hpp holdovers

The writer calls `generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed)` from
`ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp`. The function body uses:

- Line 15 (approx): `cb_reserve_back(cb_id, 1)`
- Line 16 (approx): `get_write_ptr(cb_id)` for pointer cast
- Line 22 (approx): `cb_push_back(cb_id, 1)`

No `CircularBuffer` wrapper objects are in scope inside that function. Parameter shape
(`uint32_t cb_id`) is compatible with Metal 2.0 (`dfb::name` constexpr-casts to `uint32_t`);
the body is the D2.0-track concern.

---

## Subject 2 — Feature Compatibility

| Feature | Verdict | Evidence |
|---|---|---|
| `GlobalCircularBuffer` | N/A — not used | No include or usage found |
| `GlobalSemaphore` | N/A — not used | No include or usage found |
| `UpdateCircularBufferTotalSize` | N/A — not used | Absent from factory |
| `UpdateCircularBufferPageSize` | N/A — not used | Absent from factory |
| Dynamic CBs | OK (LANDED) | `factory.cpp:508` opens `ProgramDescriptor desc`; Metal 2.0 supports dynamic sizes |
| CTA varargs (`std::vector` in CTAs) | N/A | All CTAs are `std::vector<uint32_t>`; no vararg capture in `tensor_args_t` |
| Borrowed-memory CBs | PORT WORK | Four CBs use `cb.buffer = ...` (see Subject 3) |

### Borrowed-memory CBs

All four borrowed-memory CBs use `CBDescriptor::buffer` (the Metal 2.0 PD mechanism). At port
time each translates to `DataflowBufferSpec::borrowed_from`. No `address_offset` field is set on
any of them, so the simple form applies.

| CB | Buffer source | Condition |
|---|---|---|
| `c_0` (`factory.cpp:538`) | `q_buffer` | `q_locally_available` |
| `c_8` (`factory.cpp:563`) | `cur_pos_buffer` | `is_cur_pos_tensor_sharded` |
| `c_9` (`factory.cpp:574`) | `page_table_buffer` | `is_page_table_sharded` |
| `c_20` (`factory.cpp:616`) | `out_buffer` | `is_output_sharded` |

---

## Subject 3 — TensorAccessor Bindings (Case 1 / Case 2 Classification)

**All bindings: Case 1**

The factory passes every buffer address as a raw `uint32_t` via RTAs (`->address()`), alongside
`TensorAccessorArgs` appended to CTAs. This is the standard PD-era pattern. Each is a Case 1
re-expression:

| Tensor | CTA append (`factory.cpp`) | RTA pass (lines 909-915 / 825-828 / 936) |
|---|---|---|
| `q` | Line 684 | `q_buffer->address()` at line 909 |
| `k` | Line 685 | `k_buffer->address()` at line 910 |
| `v` | Line 686 | `v_buffer->address()` at line 911 |
| `attn_mask` | Line 687 | `attn_mask_addr` at line 914 |
| `cur_pos_tensor` | Line 688 | `pos_addr` at line 912 |
| `page_table_tensor` | Line 690 | `page_table_addr` at line 913 |
| `attention_sink` | Lines 692-695 | `attention_sink_addr` at line 915 |
| `output` | Line 728 | `out_buffer->address()` at line 936 |

At port time: each becomes a `TensorParameter<...>` in the program spec; the factory receives a
`TensorBinding` and calls `.address()` for the RTA and passes the binding itself (or its layout
descriptor) in place of `TensorAccessorArgs`. No `get_bank_base_address` or exotic lookup needed.

**Stale comment:** Factory comment at lines 822-824 reads "Mandatory buffers are passed as
Buffer*" — this is incorrect; the code at lines 909-911 uses `->address()` (raw uint32_t).
The comment is vestigial; ignore at port time.

---

## Subject 4 — Out-of-Directory Coupling

### In-family donor: `sdpa/device/kernels/dataflow/dataflow_common.hpp`

Included transitively via `sdpa_decode`'s local `dataflow_common.hpp`. File is Clean Device 2.0.
No port work required in this file for Metal 2.0 gate purposes; D2.0 track may have minor
cleanup elsewhere in the sdpa family but that is out of scope here.

### Shared pool: `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp`

Device 1.0 function body (see Subject 1b). This file is shared across ops; D2.0-track cleanup
should be coordinated with other consumers.

### Shared kernel_lib: `l1_helpers.hpp`, `reduce_helpers_dataflow.hpp`

Clean Device 2.0. No port-blocking issues.

---

## Subject 5 — Custom `compute_program_hash`

**Result: PORT WORK — custom hash present; delete at port time**

`device/sdpa_decode_device_operation.cpp:500-555` defines a custom `compute_program_hash`. The
hash supplements the default (which covers `logical_shape + tensor_layout`) with:

1. **Explicit padded shapes** for Q, K, V — comment at line 502 explains: "TensorSpec hashing
   uses logical_shape + tensor_layout only, not cached_padded_shape_". This is a correctness
   guard: the factory is sensitive to padded shapes (stride computation, tile layout).
2. **Explicit `logical_shape`** for `cur_pos_tensor` — the tensor itself is hashed for
   layout/memory/placement, but shape is hashed separately to ensure rank/extents always
   contribute.
3. **Encoded `share_cache`** optional bool as a ternary uint8 tag to distinguish unset / false /
   true in the hash.
4. All operation_attributes fields including `block_size_override`, `num_kv_heads_override`,
   `cache_position_modulo`, `sliding_window_size`.

**Relaxation analysis:** The explicit padded_shape hashing for Q/K/V is substantive — the
factory uses padded shapes in stride and tile-count computation. Deleting the hash without
providing an equivalent mechanism risks spurious cache hits when padded shapes change while
logical shapes do not. At port time, verify whether Metal 2.0 hash infrastructure covers
padded shapes via `TensorSpec`; if not, the padded-shape sensitivity must be expressed another
way. This is a non-trivial port concern.

---

## Subject 6 — Variable-Length RT Arg Arrays

**Result: FYI (heads-up, not a gate)**

The factory uses `RTArgList::append()` to extend RT arg arrays with variable-length physical
coordinate vectors per core:

- `reader_rt_args.append(output_core_physical_xs)` / `append(output_core_physical_ys)` (lines 929-930)
- `writer_rt_args.append(reduce_core_physical_xs)` / `append(reduce_core_physical_ys)` (lines 964-965)
- `writer_rt_args.append(output_core_physical_xs)` / `append(output_core_physical_ys)` (lines 966-967)

The count varies per program instantiation based on `num_output_cores`, `num_reducer_cores`, and
`num_cores_per_head`. These are NOT CTA varargs (tensor_args_t is a fixed struct). The Metal 2.0
`emplace_runtime_args` mechanism can accommodate variable-length per-core RTA vectors; this is
not a gate. Flag for porter to verify the correct `emplace_runtime_args` overload is used when
array sizes vary across cores.

Also note: the writer kernel additionally appends two groups of `num_cores_per_head` reduction
group coordinate pairs inline via a `for` loop (lines 957-963) before the `append` calls. This
is a fixed-count loop per core but the count itself is compile-time-determined; same
considerations apply.

---

## Subject 7 — TTNN Factory Analysis

| Question | Answer | Evidence |
|---|---|---|
| Q1: Op-owned tensors forcing MeshWorkload? | No | No `MeshWorkload` or per-device tensor creation in factory |
| Q2: Genuine MeshWorkload need? | No | Single-device operation; no multi-device dispatch |
| Q3: `create_descriptor` exposed in pybind? | No | `sdpa_decode_nanobind.cpp` binds four op-surface functions via `ttnn::bind_function<...>()`; no DeviceOperation internals exposed |
| Q4: Other risky pybind patterns? | No | `nb::class_<...ProgramFactory>` absent; no raw descriptor exposure |
| Q5: Custom `compute_program_hash`? | Yes | `sdpa_decode_device_operation.cpp:500` — PORT WORK (see Subject 5) |
| Q6: `override_runtime_arguments`? | No | Not declared or used |

---

## FYI Items (non-gating)

### Fake CB — c_9 (page_table)

CB `c_9` is a fake CB: `reader_decode_all.cpp` calls `cb_page_table.reserve_back(...)` and
`cb_page_table.push_back(...)` (lines 260 and 276) to claim storage and record the write pointer,
but no kernel calls `wait_front`/`pop_front` on c_9 — it is used purely as an address store for
the page table data. No FIFO producer+consumer relationship exists.

Metal 2.0 DFB validator requires at least one producer AND one consumer. At port time, the
sanctioned fake-CB workaround applies (allocate L1 buffer directly; pass address via RTA).
This is FYI-P: a heads-up for the porter, resolved with an established pattern.

---

## Summary of Port Work

| Item | Priority | File(s) |
|---|---|---|
| Delete `compute_program_hash` | Required | `device/sdpa_decode_device_operation.cpp:500-555` |
| Translate TensorAccessorArgs → TensorParameter/Binding (8 tensors) | Required | `device/sdpa_decode_program_factory.cpp:684-728` |
| Translate borrowed-memory CBs → `DataflowBufferSpec::borrowed_from` (4 CBs) | Required | `device/sdpa_decode_program_factory.cpp:538,563,574,616` |
| Padded-shape hash sensitivity — verify Metal 2.0 TensorSpec covers it | Required (verify) | `device/sdpa_decode_device_operation.cpp:502-513` |
| Fake CB c_9 — apply sanctioned workaround | FYI-P | `reader_decode_all.cpp:258-276`, `factory.cpp:566-575` |
| Variable-length RTA arrays — verify `emplace_runtime_args` overload | FYI heads-up | `device/sdpa_decode_program_factory.cpp:929-967` |

## Pre-conditions (D2.0 track, not in the port)

The following Device 1.0 holdovers must be resolved before the Metal 2.0 port begins:

- `writer_decode_all.cpp:32-33` — `get_semaphore(...)` calls
- `writer_decode_all.cpp:201` — `get_noc_addr(..., reducer_semaphore_addr)`
- `writer_decode_all.cpp:431` — `noc_semaphore_wait(...)`
- `writer_decode_all.cpp:487-489` — `get_noc_addr(...)` + `noc_semaphore_inc(...)`
- `device/kernels/dataflow/dataflow_common.hpp:37,58,60,120,122` — `get_write_ptr(cb_id)` in fill helpers
- `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` — entire function body of `generate_bcast_col_scalar` and `generate_bcast_row_scalar`
