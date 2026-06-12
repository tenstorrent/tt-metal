# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/pool/generic`

The directory contains a single DeviceOperation class (`Pool2D`) with one program factory (`MultiCore`). The factory is implemented as a static `create_workload_descriptor` method (WorkloadDescriptor path).

- **`Pool2D`**
  - `Pool2D::MultiCore` (`pool_multi_core_program_factory.cpp`)
    - Selects kernel at runtime: `reader_pool_2d.cpp` (normal pool) or `reader_mpwi.cpp` (max-pool with indices)
    - Selects compute kernel at runtime: `compute_pool_2d.cpp` (normal pool) or `compute_mpwi.cpp` (max-pool with indices)

Kernels owned by this op (file-path instantiated):
- `device/kernels/dataflow/reader_pool_2d.cpp`
- `device/kernels/dataflow/reader_mpwi.cpp`
- `device/kernels/compute/compute_pool_2d.cpp`
- `device/kernels/compute/compute_mpwi.cpp`

In-family shared kernel headers included by the above:
- `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp`
- `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/pool/generic` |
| **Overall** | YELLOW |
| **DOps / Factories** | `Pool2D` → `MultiCore` (`pool_multi_core_program_factory.cpp`) |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes-with-holdovers (YELLOW — fix on D2.0 track first) |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | Yes: `MultiCore::create_workload_descriptor` — `reader_indices_tensor_owner` (always) and `scalar_config_tensor_owner` (avg-pool non-trivial-scalar path) — `pool_multi_core_program_factory.cpp:1150–1212` |
| *TTNN Readiness* — MeshWorkload needed | No (op-owned-tensor artifact only): the `WorkloadDescriptor` path is used solely to keep op-owned buffer tensors alive in the cached workload; `Pool2D` has no genuine multi-program or cross-device-coordination need |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash): `pool_op.cpp:168–185` |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | Present: `raw_in_cb_id` (input shard, `pool_multi_core_program_factory.cpp:456`) and `in_reader_indices_cb_id` L1-sharded path (`pool_multi_core_program_factory.cpp:486–491`) and `config_cb_id` L1-sharded path (`pool_multi_core_program_factory.cpp:754`) — all three are address-only (no FIFO); workaround applies |

**Fake CBs** = CBs used purely as an address source. **Litmus: does the CB have a producer *and* a consumer?** The input shard CB (`raw_in_cb_id`) is backed by the input tensor's L1 shard; the kernel reads it via `in_shard_cb.get_read_ptr()` with no `wait_front`/`pop_front` — no FIFO pair. Similarly the reader-indices and scalar-config CBs (L1-sharded path) are accessed only via `get_read_ptr()`. These cannot be expressed as Metal 2.0 DFBs; the port resolves them with the sanctioned fake-CB workaround (see the porting recipe) — FYI-P heads-up, not a gate.

---

## Result

**YELLOW** — blocked until an isolated Device 2.0 holdover is cleaned on the D2.0 track, after which the port can proceed. One isolated free-function holdover in `reader_pool_2d.cpp:88`: `get_write_ptr(in_cb_id)` where the Device 2.0 wrapper `in_cb` is in scope; member form is `in_cb.get_write_ptr()`. Fix must land on the Device 2.0 track first; do not fold into the port diff.

No UNSUPPORTED feature gates fire. All other subjects are clear.

---

## Gate detail

### ProgramDescriptor

**GREEN.** The factory creates a `tt::tt_metal::ProgramDescriptor` at `pool_multi_core_program_factory.cpp:385` and populates it with `CBDescriptor` and `KernelDescriptor` objects throughout. The program is then wrapped in a `WorkloadDescriptor` (to keep op-owned buffer tensors alive) and returned via `create_workload_descriptor`. The factory does NOT use the imperative `host_api.hpp` builder API (`CreateProgram`, `CreateKernel`, `CreateCircularBuffer`, `SetRuntimeArgs`, etc.).

### Device 2.0 (every kernel used)

**YELLOW — isolated holdover.** All four kernels (`reader_pool_2d.cpp`, `reader_mpwi.cpp`, `compute_pool_2d.cpp`, `compute_mpwi.cpp`) and the in-family shared headers (`pool_kernels_common.hpp`, `experimental_device_api.hpp`) consistently use Device 2.0 wrappers — `experimental::CB`, `Noc`, `UnicastEndpoint`, `TensorAccessor`/`TensorAccessorArgs` — for all substantive operations. One isolated holdover remains:

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `device/kernels/dataflow/reader_pool_2d.cpp` | 88 | `get_write_ptr(in_cb_id)` (free-function form) | `experimental::CB in_cb(in_cb_id)` at line 53 |

The member replacement is `in_cb.get_write_ptr()`. This is a 1-line mechanical fix. The holdover is isolated within a kernel that is otherwise consistently Device 2.0. Route to the Device 2.0 effort to fix first; do not fold into the port diff.

Note: `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` appear in `pool_kernels_common.hpp` (lines 45–47, 61, 75–76, 129) and are sanctioned Device 2.0 free functions — not flagged.

### Feature compatibility

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | Not used |
| Dynamic CircularBuffer (borrowed memory) | GREEN | `CBDescriptor::buffer` set for input shard (`raw_in_cb_id`), reader-indices, scalar-config, and output CBs — all are borrowed-memory DFBs; port uses `DataflowBufferSpec::borrowed_from` |
| CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` not set in any `CBDescriptor` |
| Aliased Circular Buffers | GREEN | `pre_tilize_cb_id` / `fast_tilize_cb_id` aliased pair at `pool_multi_core_program_factory.cpp:657–684`; port uses `advanced_options.alias_with` |
| GlobalSemaphore | N/A | No semaphores used |
| Non-zero semaphore initial value | N/A | No semaphores used |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` in host code |
| `UpdateCircularBuffer*` | N/A | Not used |
| Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` has a single fixed `const Tensor& input_tensor_`; no variable-count CTA loop |

---

## Port-work summary *(mirrors the brief)*

**Tensor bindings** (per binding):

- `input_tensor_` — **fake CB** (address-only; no FIFO producer+consumer): `raw_in_cb_id` at `pool_multi_core_program_factory.cpp:456`. The input shard is accessed kernel-side via `in_shard_cb.get_read_ptr()` with no `wait_front`. Port resolves with the sanctioned fake-CB workaround (see the porting recipe). See Heads-ups.
- `reader_indices_buffer` (op-owned) — **Case 1 (CTA-baked address)**: `reader_indices_buffer->address()` baked at CTA slot 35 (`pool_multi_core_program_factory.cpp:793`); used kernel-side as base for `TensorAccessor(config_tensor_args, reader_dram_addr)` in `load_config_tensor_if_in_dram`. Port: re-express as `TensorParameter`/`TensorBinding`; the CTA-baked address and `TensorAccessorArgs` wiring disappear. (Also available as a sharded-CB in the L1 path — that becomes `borrowed_from` DFB.)
- `scalar_config_buffer` (op-owned, avg-pool non-trivial-scalar path only) — **Case 1 (CTA-baked address)**: `config_buffer->address()` baked at CTA slot 33 (`pool_multi_core_program_factory.cpp:791`); same pattern as `reader_indices_buffer`. Port analogously.
- `outputs[0]` (output tensor) — **clean**: sharded output CB (`out_cb_id`, `pool_multi_core_program_factory.cpp:691–698`) with `CBDescriptor::buffer = outputs[0].buffer()`; compute kernel produces via `out_cb.reserve_back/push_back` — real FIFO producer. Port via `DataflowBufferSpec::borrowed_from`.
- `outputs[1]` (index tensor, `return_indices=true` path only) — **clean**: same pattern as `outputs[0]` (`out_idx_cb_id`, `pool_multi_core_program_factory.cpp:707–712`); compute/reader writes into it. Port via `DataflowBufferSpec::borrowed_from`.

**Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception). See Custom program hash section.

---

## Heads-ups *(mirrors the brief)*

- **Notable LANDED constructs:**
  - **Borrowed-memory DFBs:** Multiple `CBDescriptor::buffer = ...` sets — input shard, reader-indices (L1 path), scalar-config (L1 path), output(s). Port via `DataflowBufferSpec::borrowed_from`. Affected sites: `pool_multi_core_program_factory.cpp:456`, `486–491`, `691–698`, `707–712`, `754`.
  - **Aliased CBs:** `pre_tilize_cb_id` / `fast_tilize_cb_id` multi-format pair at `pool_multi_core_program_factory.cpp:657–684` — two `CBFormatDescriptor` elements in one `CBDescriptor`. Port via `DataflowBufferSpec::advanced_options.alias_with`. Note: these CBs are also conditionally allocated (`has_pre_tilize = true` path only, i.e. TILED output).

- **Fake CBs (address-only):**
  - `raw_in_cb_id` (input shard, `pool_multi_core_program_factory.cpp:456`): backed by `input.buffer()`; kernel reads base via `in_shard_cb.get_read_ptr()`, no `wait_front`. No producer+consumer FIFO pair from the pool kernel's perspective. **Fake CB.**
  - `in_reader_indices_cb_id` L1-sharded path (`pool_multi_core_program_factory.cpp:486–491`): backed by `reader_indices_buffer`; kernel reads base via `reader_indices_cb.get_read_ptr()`, no `wait_front`. **Fake CB.**
  - `config_cb_id` L1-sharded path (`pool_multi_core_program_factory.cpp:754`): backed by `config_buffer`; kernel reads base via `config_cb.get_read_ptr()`, no `wait_front`. **Fake CB.**
  - All three resolve with the sanctioned fake-CB workaround in the porting recipe; none gate the port.

- **Cross-op / shared kernels:** All out-of-directory includes resolve to either LLK/HAL (`api/dataflow/`, `api/compute/`) or in-family pool shared headers (`pool/device/kernels/pool_kernels_common.hpp`, `pool/device/kernels/experimental_device_api.hpp`). No cross-family donor coupling. Kernel files are owned by this op's directory (no file-path borrowing from other families).

- **RTA varargs:** None.

- **TTNN factory analysis (porter-relevant):**
  - Pybind `create_descriptor`: **None.** The nanobind file (`generic_pools_nanobind.cpp`) exposes the op functions (`max_pool2d`, `avg_pool2d`) via `bind_function<>` — normal op-surface pybind only.
  - Other risky pybind: **None.**
  - Custom `override_runtime_arguments`: **None.**

---

## Team-only

### TensorAccessor convertibility

- `reader_indices_buffer` (Case 1): straightforwardly convertible — page-by-page read of the reader-indices lookup table from DRAM. `TensorAccessor` iteration is the natural fit. Not exotic.
- `scalar_config_buffer` (Case 1): straightforwardly convertible — page-by-page read of the scalar config table from DRAM. Not exotic.

### Out-of-directory coupling & donor shape analysis

**Op-level roll-up: ✓ clean.** All cross-directory includes are LLK/HAL or in-family pool shared; no cross-family donor; all in-family shared code is Device 2.0 compliant.

**Summary table:**

| Op kernel | Donor file | Class | Shape | Status |
|---|---|---|---|---|
| `reader_pool_2d.cpp` | `api/dataflow/dataflow_api.h` | LLK/HAL | — | ✓ no concern |
| `reader_pool_2d.cpp` | `pool/device/kernels/pool_kernels_common.hpp` | In-family shared | `experimental::CB` / `Noc` / `TensorAccessorArgs` args | ✓ Device 2.0 native |
| `reader_mpwi.cpp` | `api/dataflow/dataflow_api.h` | LLK/HAL | — | ✓ no concern |
| `reader_mpwi.cpp` | `pool/device/kernels/pool_kernels_common.hpp` | In-family shared | `experimental::CB` / `Noc` args | ✓ Device 2.0 native |
| `compute_pool_2d.cpp` | `api/compute/*.h` | LLK/HAL | — | ✓ no concern |
| `compute_pool_2d.cpp` | `pool/device/kernels/experimental_device_api.hpp` | In-family shared | `experimental::CB` alias | ✓ Device 2.0 native |
| `compute_mpwi.cpp` | `api/compute/*.h` | LLK/HAL | — | ✓ no concern |
| `compute_mpwi.cpp` | `pool/device/kernels/experimental_device_api.hpp` | In-family shared | `experimental::CB` alias | ✓ Device 2.0 native |

**Per-call detail:** omitted — all donors are ✓.

**Borrowed kernel files:** All four kernel files (`reader_pool_2d.cpp`, `reader_mpwi.cpp`, `compute_pool_2d.cpp`, `compute_mpwi.cpp`) are owned by this op directory. No file-path borrowing from other op families. The in-family shared headers (`pool_kernels_common.hpp`, `experimental_device_api.hpp`) are also within the `pool/` family. No cross-family port-together coupling obligation.

### Relaxation candidates (FALLIBLE — candidates to verify, default strict)

From the custom hash at `pool_op.cpp:168–185`:

The hash keys on: `sliding_window_config_.get_hash()`, `pool_type_`, `output_layout_`, `memory_config_`, `compute_kernel_config_`, `divisor_override_`, `count_include_pad_`, `return_indices_`, `config_tensor_in_dram`, `input_mem_config`, `in_dtype`, `out_dtype`.

`TensorSpec` is not explicitly hashed — the factory relies on `input_mem_config` + `in_dtype` as a proxy. This is a known omission pattern. **Potential relaxation signal:** If the `input_mem_config` fully determines the shard spec (and hence the `TensorSpec`), the default hash may be safely strict here. **But this is a candidate to verify**, not a conclusion — the default is strict; a relaxation requires explicit user OK.

### TTNN factory analysis (six-question answers)

1. **Op-owned tensors? Yes.** `create_workload_descriptor` allocates two device tensors:
   - `reader_indices_tensor_owner` (always): halo lookup table uploaded to device, parked in `workload_descriptor.buffers[0]` (`pool_multi_core_program_factory.cpp:1150–1152`).
   - `scalar_config_tensor_owner` (avg-pool non-trivial-scalar path: `!one_scalar_per_core`): per-output-stick scalar config tensor, parked in `workload_descriptor.buffers.push_back(...)` (`pool_multi_core_program_factory.cpp:1210–1212`).
   Both tensors live for the lifetime of the cached workload.

2. **MeshWorkload concept needed? No (op-owned-tensor artifact only).** The factory exposes `create_workload_descriptor` and returns a `WorkloadDescriptor` (which the framework materializes into a MeshWorkload). However, the code comment at `pool_multi_core_program_factory.cpp:1217–1220` makes the intent explicit: "Single-device op: the kernel program is structurally identical for every coord in `tensor_coords` (Pool2D doesn't depend on cluster position). Build the per-coord ProgramDescriptor ONCE and copy it." The `WorkloadDescriptor` path is used solely to keep the op-owned buffer tensors alive across cache hits; it is not needed for cross-device or multi-program coordination. Q1 (op-owned tensors) is the cause.

3. **Pybind `create_descriptor`? No.** `generic_pools_nanobind.cpp` exposes `max_pool2d` and `avg_pool2d` via `bind_function<"max_pool2d">` / `bind_function<"avg_pool2d">` — standard op-surface pybind, not ProgramFactory internals.

4. **Other migration-risky pybind? None.** The nanobind file wraps only the top-level op functions and does not expose `Pool2D`, `Pool2D::MultiCore`, `operation_attributes_t`, `tensor_args_t`, or any device-operation or factory class.

5. **Custom hash? Yes.** `Pool2D::compute_program_hash` at `pool_op.cpp:168–185`. Treatment: delete and revert to default (PORT WORK — see Port-work summary).

6. **Custom override-RTA? No.** No `override_runtime_arguments` in the op directory.

---

## Misc anomalies *(team-only, non-gating)*

- **`pool_multi_core_program_factory.cpp:791–793`:** `config_buffer->address()` and `reader_indices_buffer->address()` are baked as CTAs (slots 33 and 35). Slots 33/35 feed the DRAM path only (`config_in_dram` CTA 32); the TensorAccessorArgs appended at lines 816–818 also carry these buffers for the Device 2.0 binding path. The dual-encoding (CTA-baked raw address AND TensorAccessorArgs) means the kernel has two representations of the same binding — the TensorAccessorArgs one is the right Metal 2.0 ancestor; the raw CTA-baked address slot can be removed in the port. Flagged for op owner awareness.
- **`pool_multi_core_program_factory.cpp:985–1012`:** Post-allocation L1 validation (`actual_local_cb_size == cb_sizes.local_cb_total()` etc.) queries the device allocator directly. This is a debug-time consistency check but couples the factory to the device allocator's state at build time — may require adjustment in the Metal 2.0 port if the allocator interface changes.

---

## Questions for the user

1. **Fake CB for `raw_in_cb_id` (input shard):** The input tensor is placed into L1 by the halo op upstream and accessed kernel-side purely via `in_shard_cb.get_read_ptr()` — no FIFO `wait_front`/`pop_front` in the pool reader. This fits the fake-CB pattern (address-only). The port would apply the sanctioned fake-CB workaround. **Please confirm** this interpretation is correct before the port proceeds, as the halo infra is involved and there may be synchronization machinery not visible in the pool op's kernel code.

---

## Recipe notes

- The audit recipe's TensorAccessor section focuses on user-facing `TensorParameter` bindings. This op also has CTA-baked addresses for **op-owned internal buffers** (`reader_indices_buffer`, `scalar_config_buffer`). These don't cleanly map to "Case 1 re-express via TensorParameter" for a user-facing input, but rather to what would become op-owned `TensorParameter`s in Metal 2.0. The recipe does not explicitly address op-owned tensor bindings in the TensorAccessor handling subject — classified as Case 1 in this report for consistency, but the porter should expect these to be handled via the op-owned-tensor path in the factory-selection doc rather than the standard `TensorParameter` user-binding path. Recipe maintainer may want to clarify this case.
