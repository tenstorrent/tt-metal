# Pre-port audit: `ttnn/cpp/ttnn/operations/reduction/generic/`

Audit performed against the four program factories under `ttnn/cpp/ttnn/operations/reduction/generic/device/`. Two `DeviceOperation` types share the directory; both audited together because they share kernel sources (`reader_unary_reduce_universal_start_id.cpp`, `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`, `writer_unary_interleaved_start_id.cpp`).

- **`ReduceDeviceOperation`** (in `reduce_op_device_operation.hpp`)
  - `ReduceSingleCoreHwProgramFactory` (`reduce_op_single_core_hw_program_factory.cpp`)
  - `ReduceMultiCoreHProgramFactory` (`reduce_op_multi_core_h_program_factory.cpp`)
  - `ReduceMultiCoreWProgramFactory` (`reduce_op_multi_core_w_program_factory.cpp`)
- **`WelfordReduceDeviceOperation`** (in `welford_reduce_device_operation.hpp`)
  - `WelfordReduceProgramFactory` (`welford_reduce_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**GREEN** — all four factories are on the `ProgramDescriptor` API, all kernels are Device 2.0 compliant with isolated `get_tile_size(cb_id)` holdovers (fold into port-time cleanup), all kernels that touch tensor memory use `TensorAccessor`, and no Appendix A `UNSUPPORTED` feature signal fires. The H factory's `use_width_sharding` branch uses borrowed-memory CBs (`CBDescriptor::buffer = ...`); this is **LANDED** in Metal 2.0 as `DataflowBufferSpec::borrowed_from`.

Handoff to the recipe doc is appropriate after explicit user go-ahead.

### Yellow side-issues

- **Isolated Device 2.0 free-function holdovers** (`get_tile_size(cb_id)`) in five kernels. Default recommendation per Check 2: fold into the Metal 2.0 port as port-time cleanup. See [Device 2.0 DM](#device-20-dm-yellow-isolated-holdovers) below for the file:line list.

## Porting prerequisites

### ProgramDescriptor API: **GREEN**

All four factory implementations populate a `ProgramDescriptor` and use `KernelDescriptor` / `CBDescriptor` directly. No `CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs` calls present in any factory.

Note: the op's header (`reduce_op_device_operation.hpp:39-44`) currently declares `ReduceMultiCoreWProgramFactory::create_program_spec` returning `ttnn::device_operation::ProgramArtifacts`, while the matching `.cpp` still defines `create_descriptor` returning `ProgramDescriptor`. This is residual from a previous (now-reverted) port attempt; the implementation file is on the `ProgramDescriptor` API as required. The port will resolve the mismatch by reimplementing the W factory on `create_program_spec`.

### Device 2.0 DM: **YELLOW (isolated holdovers)**

All kernels broadly use Device 2.0 wrappers (`CircularBuffer`, `Noc`, `noc.async_read(...)`, `TensorAccessor`). The CB-index-keyed free function `get_tile_size(cb_id)` appears in the shape the audit identifies as the holdover family — the corresponding `CircularBuffer cb(cb_id)` wrapper is already in scope at the call site:

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | 26 | `get_tile_size(cb_id_in0)` | `CircularBuffer cb_in0(cb_id_in0);` (line 31) |
| `kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | 37 | `get_tile_size(cb_id_in0)` | `CircularBuffer cb_in0(cb_id_in0);` (line 40) |
| `kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | 34 | `get_tile_size(cb_id_in0)` | `CircularBuffer cb_in0(cb_id_in0);` (line 44) |
| `kernels/dataflow/writer_welford_hw.cpp` | 55 | `get_tile_size(cb_partial)` | `CircularBuffer cb_partial_obj(cb_partial);` (line 59) |
| `kernels/dataflow/writer_welford_hw.cpp` | 56 | `get_tile_size(cb_out)` | `CircularBuffer cb_out_obj(cb_out);` (line 61) |

Cross-op kernel `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` also uses `get_local_cb_interface(cb_id_out).fifo_page_size` (line 18) — same holdover family, but the kernel lives outside the op directory; it falls under the [shared-dataflow-kernel Caution](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel) and will be touched only if the port replaces it.

**Default recommendation:** Fold the in-directory holdovers into the Metal 2.0 port as 1-line mechanical replacements (e.g. `get_tile_size(cb_id_in0)` → `cb_in0.get_tile_size()`). No separate prereq PR.

### TensorAccessor usage: **GREEN**

Every kernel that directly reads or writes tensor memory uses `TensorAccessor`:

- `reader_unary_reduce_universal_start_id.cpp:28` — `TensorAccessor(tensor_args, src_addr)`
- `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp:41` — `TensorAccessor(tensor_args, src_addr)`
- `writer_welford_hw.cpp:63` — `TensorAccessor(dst_args, dst_addr)`
- Cross-op `writer_unary_interleaved_start_id.cpp` and `writer_unary_sharded.cpp` — `TensorAccessor` and CB-only respectively.

**Causal-link gate (verified).** `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` does **not** use `TensorAccessor`; it reads from CB `c_1` via `cb_in1.get_write_ptr()` after `reserve_back(num_tiles)`. CB `c_1` is a borrowed-memory CB built on the input tensor's buffer (`reduce_op_multi_core_h_program_factory.cpp:111` — `CBDescriptor{..., .buffer = a.buffer()}`). The kernel's lack of `TensorAccessor` is intended — the borrowed-memory CB **is** the tensor access — and the port handles it via `DataflowBufferSpec::borrowed_from`. Not classified as RED.

Compute kernels (`reduce.cpp`, `reduce_{w,h,hw}_neg.cpp`, `welford_reduce_{w,h,hw}.cpp`) only consume from / produce to CBs; they are out of scope for Check 3 per the doc's compute-kernel exclusion.

## Feature compatibility check

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` references; no `CBDescriptor::global_circular_buffer` set on any descriptor. |
| Dynamic CircularBuffer (CB on borrowed memory) | GREEN | LANDED — H factory's `use_width_sharding` branch uses this construct twice (input and output). Port uses `borrowed_from`. See detail below. |
| CBDescriptor `address_offset` (non-zero) | GREEN | No `.address_offset` field set on any `CBDescriptor` in any factory. |
| Aliased Circular Buffers | GREEN | Every `format_descriptors` initializer is single-element across all four factories. |
| GlobalSemaphore | N/A | No `GlobalSemaphore` references; no semaphore use at all. |
| Non-zero semaphore initial value | N/A | No `SemaphoreDescriptor` in any factory. |
| `ArgConfig::Runtime*` tensor-accessor flavors | GREEN | No `ArgConfig::Runtime*` in op host code or kernels. All `TensorAccessorArgs` are the standard `TensorAccessorArgs(buffer)` / `TensorAccessorArgs<N>()` form. |
| `UpdateCircularBuffer*` | GREEN | No `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` calls; no `override_runtime_arguments`-shaped path. |

All non-N/A signals are either GREEN or LANDED. No gating issue.

### Dynamic CircularBuffer (CB on borrowed memory): **GREEN (LANDED)**

**Signal:** `CBDescriptor::buffer` field set to a non-null `Buffer*`.

**Sites (H factory's `use_width_sharding` branch only):**

- `reduce_op_multi_core_h_program_factory.cpp:111` — input CB `c_1` borrowed from `a.buffer()` (input tensor).
- `reduce_op_multi_core_h_program_factory.cpp:148` — output CB `c_3` borrowed from `output.buffer()` (output tensor).

**Expected resolution:** LANDED in Metal 2.0 — declare the affected `DataflowBufferSpec`s with `borrowed_from = INPUT` / `borrowed_from = OUTPUT` naming the `TensorParameter`s backing them. The DFB handles resolve to the borrowed L1 address at runtime via the corresponding `TensorArg`. Reader kernel `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` continues to read from the DFB wrapper as today.

## Path forward

GREEN audit. Proceed to the port via [`port_op_to_metal2_recipe.md`](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/port_op_to_metal2_recipe.md). The port will:

1. Reimplement all four factories as `ProgramSpecFactoryConcept` factories returning `ttnn::device_operation::ProgramArtifacts`.
2. Drop legacy plumbing per the recipe: buffer-address RTAs, magic CB indices in CTAs, the `TensorAccessorArgs(buffer).append_to(cta)` chains, the kernel-side `TensorAccessorArgs<N>()` offset lines.
3. Apply the borrowed-memory DFB pattern to the H factory's width-sharded path.
4. Mechanically clean up the listed `get_tile_size(cb_id)` holdovers as part of the kernel modifications.
5. Preserve the multi-`KernelDescriptor` work-split (per-group `Ht`/`Wt` CTAs) across all four factories — each has a `compute_desc_g1` plus an optional `compute_desc_g2`, which must port to two `KernelSpec`s, not collapse to one (per the [Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) anti-pattern).

## Questions for the user

None. Audit is GREEN; only the YELLOW side-issue is the `get_tile_size(cb_id)` holdover family, and the default recommendation (port-time cleanup) is appropriate per the audit doc.
