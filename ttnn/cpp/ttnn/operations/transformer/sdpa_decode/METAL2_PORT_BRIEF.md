# Metal 2.0 Port Brief — sdpa_decode

**Op:** `ttnn/cpp/ttnn/operations/transformer/sdpa_decode`
**Audit date:** 2026-06-11
**Gate status:** YELLOW — all Metal 2.0 gates clear; Device 2.0 holdover cleanup required first

> **Blocked until** the Device 2.0 holdovers listed at the end of this brief are resolved
> on the D2.0 track. The port may not begin until those are done.

---

## Scope

The factory (`device/sdpa_decode_program_factory.cpp`) is already on the ProgramDescriptor API.
The Metal 2.0 port consists of:

1. Replacing `TensorAccessorArgs` + raw `->address()` RTAs with `TensorParameter` / `TensorBinding`
2. Translating borrowed-memory CBs (`CBDescriptor::buffer`) to `DataflowBufferSpec::borrowed_from`
3. Deleting the custom `compute_program_hash` (and verifying Metal 2.0 hash covers padded shapes)
4. Applying the fake-CB workaround for c_9 (page_table)

---

## Step-by-Step Port Instructions

### Step 1 — Tensor Bindings (TensorParameter / TensorBinding)

Eight tensors currently pass layout info via `TensorAccessorArgs` appended to CTAs, with raw
addresses via RTAs. All are Case 1 (re-express via TensorParameter).

For each tensor:
- Replace `TensorAccessorArgs(tensor.buffer()).append_to(cta_vec)` with a `TensorParameter`
  declaration in the program spec.
- Replace `tensor.buffer()->address()` (or the ternary `use_x ? x.buffer()->address() : 0`) in
  the RTA with the address from the bound `TensorBinding`.
- In the kernel, the `TensorAccessorArgs<offset>()` CTA-decode block becomes `TensorAccessor`
  constructed from the Metal 2.0 descriptor; the `addr` argument (currently the RTA uint32_t)
  becomes the binding address.

| # | Tensor | CTA append (factory.cpp) | RTA (factory.cpp) | Kernel consumer |
|---|---|---|---|---|
| 1 | `q` | line 684 | line 909 | `reader_decode_all.cpp` |
| 2 | `k` | line 685 | line 910 | `reader_decode_all.cpp` |
| 3 | `v` | line 686 | line 911 | `reader_decode_all.cpp` |
| 4 | `attn_mask` | line 687 | line 914 | `reader_decode_all.cpp` |
| 5 | `cur_pos_tensor` | line 688 | line 912 | `reader_decode_all.cpp` |
| 6 | `page_table_tensor` | line 690 | line 913 | `reader_decode_all.cpp` |
| 7 | `attention_sink` | lines 692-695 | line 915 | `reader_decode_all.cpp` |
| 8 | `output` | line 728 | line 936 | `writer_decode_all.cpp` |

**Note on optional tensors (4-7):** The current code uses ternary `use_x ? x.buffer()->address() : 0`
patterns. Metal 2.0 optional tensor parameters handle this via a null/inactive binding; consult
the Metal 2.0 optional TensorParameter API for the correct idiom.

### Step 2 — Borrowed-Memory CBs

Replace `CBDescriptor::buffer = ptr` with `DataflowBufferSpec::borrowed_from(ptr)` at port time.
No `address_offset` is set on any of these; use the simple (non-offset) form.

| CB | Buffer | Factory location | Condition |
|---|---|---|---|
| `c_0` | `q_buffer` | `factory.cpp:538` | `q_locally_available` |
| `c_8` | `cur_pos_buffer` | `factory.cpp:563` | `is_cur_pos_tensor_sharded` |
| `c_9` | `page_table_buffer` | `factory.cpp:574` | `is_page_table_sharded` |
| `c_20` | `out_buffer` | `factory.cpp:616` | `is_output_sharded` |

### Step 3 — Fake CB (c_9, page_table)

CB `c_9` is a fake CB: the reader (`reader_decode_all.cpp:260,276`) calls `reserve_back` /
`push_back` to claim L1 storage, but no kernel calls `wait_front`/`pop_front` — it is used purely
as an address-indexed L1 scratch region.

Metal 2.0 DFB validator rejects CBs with zero producers or zero consumers. Apply the sanctioned
fake-CB workaround:
- Allocate the L1 region as a plain `DataflowBuffer` (not a DFB with FIFO semantics).
- Pass the base address via RTA or a named token.
- Remove the `reserve_back` / `push_back` / `get_write_ptr` calls; use the raw address directly.

The borrowed-memory path (`is_page_table_sharded ? page_table_buffer : nullptr`) must also be
updated consistently.

### Step 4 — Delete Custom `compute_program_hash`

Delete `device/sdpa_decode_device_operation.cpp:500-555` in full.

Before deleting, verify that Metal 2.0 hash infrastructure covers padded shapes for Q/K/V. The
comment at line 502 explains why the custom hash was introduced: default `TensorSpec` hashing
covers `logical_shape + tensor_layout` but not `cached_padded_shape_`. If Metal 2.0 hash does not
cover padded shapes, the porter must use another mechanism (e.g., include padded shapes in
operation_attributes, or use a Metal 2.0 hash extension) — otherwise spurious cache hits will
occur when padded shapes change without logical shape changes.

The `cur_pos_tensor` logical shape explicit hashing (lines 517-520) and the `share_cache` ternary
encoding (lines 522-525) should likewise be verified against the default Metal 2.0 hash before
deleting.

### Step 5 — Variable-Length RTA Arrays (Heads-Up)

The factory appends variable-length physical coordinate vectors to reader and writer RT args
(factory.cpp lines 929-967). The count depends on `num_output_cores`, `num_reducer_cores`, and
`num_cores_per_head`. Verify the `emplace_runtime_args` overload being used correctly handles
arrays of varying length across cores.

---

## Blocked-Until — Device 2.0 Track Prerequisites

The following Device 1.0 holdovers must be resolved on the D2.0 track before this port begins.
They are NOT Metal 2.0 port work; they belong to whoever is cleaning D1.0 holdovers from these
files.

### `writer_decode_all.cpp`

| Line(s) | Holdover | D2.0 replacement |
|---|---|---|
| 32 | `get_semaphore(reducer_semaphore_id)` | `Semaphore<>(reducer_semaphore_id)` — no separate addr needed |
| 33 | `get_semaphore(get_compile_time_arg_val(12))` | `Semaphore<>(get_compile_time_arg_val(12))` |
| 201 | `get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, reducer_semaphore_addr)` | Use `Semaphore<>(reducer_semaphore_id).up(noc, ...)` directly |
| 431 | `noc_semaphore_wait(output_self_semaphore_addr_ptr, num_reducers_to_wait)` | `Semaphore<>(output_semaphore_id).wait_min(num_reducers_to_wait)` |
| 487-489 | `get_noc_addr(output_core_noc_x, output_core_noc_y, output_semaphore_addr)` + `noc_semaphore_inc(output_core_semaphore_noc_addr, 1)` | `Semaphore<>(output_semaphore_id).up(noc, output_core_noc_x, output_core_noc_y, 1)` |

### `device/kernels/dataflow/dataflow_common.hpp` (sdpa_decode-local)

| Line(s) | Holdover | D2.0 replacement |
|---|---|---|
| 37 | `get_write_ptr(cb_id)` in `fill_tile<>` | `CircularBuffer cb(cb_id); cb.get_write_ptr()` (D2.0 object already on other branch of same function) |
| 58, 60 | `get_write_ptr(cb_id)` in `fill_tile_partial` | `CircularBuffer cb(cb_id); cb.get_write_ptr()` |
| 120, 122 | `get_write_ptr(cb_id)` in `fill_tile_partial_sliding_window` | `CircularBuffer cb(cb_id); cb.get_write_ptr()` |

### `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` (shared pool)

The writer calls `generate_bcast_col_scalar(cb_col_identity, ...)`. The function body uses
`cb_reserve_back`, `get_write_ptr`, `cb_push_back` — fully D1.0. Parameter shape
(`uint32_t cb_id`) is compatible with Metal 2.0. Coordinate cleanup of the function body with
other consumers of this shared file.

---

## Files to Touch During Port

**Factory / host-side:**
- `device/sdpa_decode_program_factory.cpp` — Steps 1, 2, 3, 4, 5
- `device/sdpa_decode_device_operation.cpp` — Step 4 (delete `compute_program_hash`)
- `device/sdpa_decode_device_operation.hpp` — Step 4 (remove `compute_program_hash` declaration)

**Kernel-side (after D2.0 track clears):**
- `device/kernels/dataflow/reader_decode_all.cpp` — Step 3 (fake CB c_9 usage)
- `device/kernels/dataflow/writer_decode_all.cpp` — Step 1 (output TensorAccessor), post-D2.0
- `device/kernels/dataflow/dataflow_common.hpp` — post-D2.0 (local fill helpers)
- `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` — post-D2.0 (coordinate with other consumers)
