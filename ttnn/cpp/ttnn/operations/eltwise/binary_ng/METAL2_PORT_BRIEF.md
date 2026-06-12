# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/eltwise/binary_ng`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ▲ (holdovers — see Blocked-until) · Features ✓

> ⚠ **BLOCKED until Device 2.0 cleanup.** This port **cannot begin** until the following isolated Device 2.0 holdovers are fixed — *separately, on the Device 2.0 track; never in the port diff*:
>
> - `kernels_ng/dataflow/reader_interleaved_rm_no_bcast.cpp:139,148` — `noc_async_read(...)` → `noc.async_read(...)`
> - `kernels_ng/dataflow/writer_interleaved_rm_no_bcast.cpp:84` — `noc_async_write(...)` → `noc.async_write(...)`
> - `kernels_ng/dataflow/reader_interleaved_rm_row_bcast.cpp:144,150,160,165` — `noc_async_read(...)` × 4 → `noc.async_read(...)`
> - `kernels_ng/dataflow/reader_interleaved_rm_col_bcast.cpp:175,186,197,210` — `noc_async_read(...)` × 4 → `noc.async_read(...)`
> - `kernels_ng/dataflow/reader_interleaved_rm_scalar_bcast.cpp:170,180,192,201` — `noc_async_read(...)` × 4 → `noc.async_read(...)`
> - `kernels_ng/dataflow/reader_interleaved_rm_row_col_mixed_bcast.cpp:175,183,197,205` — `noc_async_read(...)` × 4 → `noc.async_read(...)`
> - `kernels_ng/dataflow/reader_interleaved_rm_scalar_op.cpp:127` — `noc_async_read(...)` → `noc.async_read(...)`
>
> All 7 files already declare `Noc noc;` and otherwise consistently use Device 2.0 APIs. Once these holdovers are cleaned, proceed with this brief as-is — **no re-audit needed.**

---

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** none
- **MeshWorkload:** not needed
- **Pybind `create_descriptor`:** none
- **Other risky pybind:** none
- **Custom `override_runtime_arguments`:** none

---

## Construct — to do

**Tensor bindings** (per binding):

- `a` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. The kernel builds `TensorAccessor(ta::a)`. Remove `a.buffer()->address()` from per-core reader RTAs (positions 0 in non-RM path, position 0 in RM path). Note: when b is a scalar (no b tensor), the b-slot `TensorAccessorArgs` currently uses `*a_buffer` as a fallback (`binary_ng_program_factory.cpp:857-859`) — handle this conditional at port time.

- `b` — **Case 1** → re-express via `TensorParameter` / `TensorBinding` (conditional — only when `b.has_value()`). The kernel builds `TensorAccessor(ta::b)`. Remove `b->buffer()->address()` from per-core reader RTAs (position 15 non-RM, position 7 RM path at `binary_ng_program_factory.cpp:1220`).

- `c` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. The kernel builds `TensorAccessor(ta::c)`. Remove `c.buffer()->address()` from per-core writer RTAs (positions 0 in both RM and non-RM writer paths: `binary_ng_program_factory.cpp:1117`, `1133`, `1183`, `1198`).

**Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception). Location: `binary_ng_device_operation.cpp:487`.

---

## Watch for

- **Dynamic CircularBuffer (borrowed memory):** `CBDescriptor::buffer` is set to `a_buffer`, `b_buffer`, `c_buffer` when the respective tensor is sharded (`binary_ng_program_factory.cpp:570`, `602`, `660`). Port declares these `DataflowBufferSpec` entries with `borrowed_from = <tensor_parameter>`. The `TensorParameter` for each tensor must be consistently co-declared with the corresponding `DataflowBufferSpec` that borrows from it.

- **Dynamic TensorAccessor (`ArgConfig::RuntimeTensorShape`):** Three sites in `binary_ng_program_factory.cpp` (lines 700, 855, 858). Metal 2.0 path is `TensorParameterAdvancedOptions::dynamic_tensor_shape = true` — **UNSAFE** opt-in. The default is strict; confirm with the user before applying. This is consistent with the op's existing behavior (the `RuntimeTensorShape` flag already allows shape variation at runtime).

- **Cross-op / shared kernels:** All includes are in-family (kernels_ng/ → kernels/). No cross-family port-together coupling.

- **RTA varargs:** None — all RTA layouts are fixed-count per path.
