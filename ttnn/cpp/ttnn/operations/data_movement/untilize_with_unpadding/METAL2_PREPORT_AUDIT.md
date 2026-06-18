# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/`

Single device-operation directory. One `DeviceOperation`, six `ProgramFactory`s:

- **`UntilizeWithUnpaddingDeviceOperation`**
  - `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory` (`untilize_with_unpadding_multi_core_interleaved_program_factory.cpp`) — **already ported to Metal 2.0** (`create_program_artifacts`); out of audit scope, retained as the reference port.
  - `UntilizeWithUnpaddingSingleCoreProgramFactory` (`untilize_with_unpadding_single_core_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory` (`untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory` (`untilize_with_unpadding_multi_core_col_interleaved_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` (`untilize_with_unpadding_multi_core_sharded_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory` (`untilize_with_unpadding_multi_core_nd_sharded_program_factory.cpp`)

**Audit scope:** the **five not-yet-ported factories** listed above. The interleaved factory is already on the Metal 2.0 host API and serves as the in-repo reference for the patterns this port will follow (vararg RTAs, named DFB/TensorParameter bindings, the `_metal2` kernel-fork convention).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/` |
| **Overall** | **GREEN** (all 5 remaining factories) |
| **DOps / Factories** | `UntilizeWithUnpaddingDeviceOperation` → SingleCore, MultiCoreBlockInterleaved, MultiCoreColInterleaved, MultiCoreSharded, MultiCoreNDSharded (MultiCoreInterleaved already ported) |
| *Prereqs* — ProgramDescriptor | **Yes** (all 5 factories use `create_descriptor` → `ProgramDescriptor`) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** (no holdovers; all dataflow kernels use `Noc` / `CircularBuffer` / `TensorAccessor` wrappers; compute kernels are CB-index LLK with no wrapper-in-scope holdovers) |
| *Prereqs* — Cross-op escapes | Ok (shared/donor kernels all Device 2.0; induces a port-together set — see Team-only) |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — Variadic-CTA | Ok (op takes a single `Tensor`; no variable-count input list) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind binds only the op function) |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | Present (1, sharded factory): output CB `c_17` is producer-only (out-sharded path) → port via `borrowed_from`, fake-CB workaround if validator requires a consumer |

**Fake CBs** = CBs used purely as an address source (no producer *and* consumer pair). Granularity is the (CB, endpoint) edge. The one instance here is the borrowed-memory output CB and is **FYI-P, not a gate**.

## Result

**GREEN → brief issued.** All five not-yet-ported factories clear every gate: each is on the `ProgramDescriptor` API, every kernel they exercise is Device 2.0 compliant (no holdovers), and no UNSUPPORTED Metal 2.0 feature is in use. The only LANDED feature requiring a non-trivial construct is the **borrowed-memory DFB** (sharded factory's resident input/output CBs → `DataflowBufferSpec::borrowed_from`). The port is **mechanical**: per-binding `TensorAccessor` conversions (all Case 1), plus `_metal2` forks of the shared compute/reader/writer kernels (same convention the interleaved port already established). No factory needs to be omitted — there is no localized blocked code path.

## Gate detail

- **ProgramDescriptor:** **GREEN.** All five factories define `static ProgramDescriptor create_descriptor(...)` and populate `desc.cbs` / `desc.kernels` with `CBDescriptor` / `KernelDescriptor` (e.g. single core `...single_core_program_factory.cpp:25`, block `...block_interleaved...:56`, col `...col_interleaved...:23`, sharded `...sharded...:25`, nd `...nd_sharded...:26`). No imperative `host_api.hpp` builder calls (`CreateProgram` / `CreateKernel` / `SetRuntimeArgs`) in any factory.

- **Device 2.0 (every kernel used):** **GREEN — no holdovers.** Every dataflow kernel referenced by the five factories uses the Device 2.0 wrappers (`Noc`, `CircularBuffer`, `TensorAccessor`, `CoreLocalMem`, `UnicastEndpoint`); the only free functions present are the **sanctioned** `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`. No Device 1.0 idioms (`noc_async_read`, `InterleavedAddrGen`, `ShardedAddrGen`, raw bank/sem addresses) anywhere. Compute kernels pass CB indices to `compute_kernel_lib::untilize` / LLK calls with **no `CircularBuffer` wrapper in scope**, so there is nothing to flag as a holdover; the CB-index → `dfb::name` swap is the routine Metal 2.0 binding rewrite (done via fork), not a Device 2.0 fix.

  Kernel-by-kernel Device 2.0 status (per factory):

  | Factory | Kernel | Path | Status |
  |---|---|---|---|
  | SingleCore | reader | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | ✓ D2.0 (`Noc`,`CircularBuffer`,`TensorAccessor`; `get_local_cb_interface` sanctioned) |
  | SingleCore | writer | own `writer_unary_unpad_dims_split_rows.cpp` | ✓ D2.0 |
  | SingleCore | compute | `untilize/.../compute/untilize.cpp` | ✓ CB-index LLK (fork to `dfb::`) |
  | BlockInterleaved | reader | `eltwise/unary/.../reader_unary_interleaved_wh_multicore.cpp` | ✓ D2.0 (`get_tile_size` sanctioned) |
  | BlockInterleaved | writer | own `writer_unary_stick_layout_wh_multicore.cpp` | ✓ D2.0 |
  | BlockInterleaved | compute | `untilize/.../compute/untilize_wh.cpp` | ✓ CB-index LLK (fork) |
  | ColInterleaved | reader | `eltwise/unary/.../reader_unary_interleaved_col_multicore.cpp` | ✓ D2.0 (`get_tile_size` sanctioned) |
  | ColInterleaved | writer | own `writer_unary_stick_layout_col_multicore.cpp` | ✓ D2.0 |
  | ColInterleaved | compute | `untilize/.../compute/untilize_w.cpp` | ✓ CB-index LLK (fork) |
  | Sharded | reader | `eltwise/unary/.../reader_unary_sharded.cpp` | ✓ D2.0 (CB fake-push) |
  | Sharded | writer (out-sharded) | own `writer_unary_unpad_batch_rows_sharded.cpp` | ✓ D2.0 (`Noc`,`UnicastEndpoint`,`CoreLocalMem`) |
  | Sharded | writer (W=16) | own `writer_unary_unpad_width_16_sharded.cpp` | ✓ D2.0 (`async_read_with_state`; `get_tile_size` sanctioned) |
  | Sharded | writer (interleaved-out) | shared `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | ✓ D2.0 |
  | Sharded | compute | `untilize/.../compute/untilize.cpp` or shared `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | ✓ CB-index/CircularBuffer LLK (fork) |
  | NDSharded | reader | `data_movement/sharded/.../reader_unary_nd_sharded_blocks.cpp` | ✓ D2.0 (`TensorAccessor.shard_pages`; `get_tile_size` sanctioned) |
  | NDSharded | writer | own `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` | ✓ D2.0 (`TensorAccessor.shard_pages`, `CoreLocalMem`) |
  | NDSharded | compute | `untilize/.../compute/untilize_variable_num_blocks.cpp` | ✓ CB-index LLK (fork) |

- **Feature compatibility:** Appendix A, in order:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | not used |
  | Dynamic CircularBuffer (borrowed memory) | **GREEN** | sharded factory: input CB `.buffer = a.buffer()` (`...sharded...:97`) and out-sharded CB `c_17` `.buffer = output.buffer()` (`...sharded...:126`) → port uses `borrowed_from` |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset` set anywhere |
  | Aliased Circular Buffers | N/A | every `format_descriptors` initializer is single-element |
  | GlobalSemaphore | N/A | no semaphores used |
  | Non-zero semaphore initial value | N/A | no semaphores used |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | all `TensorAccessorArgs(*buffer)` are the single-arg static form |
  | `UpdateCircularBuffer*` | N/A | none |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = Tensor` (single input); no variable-count CTA loops |

  No UNSUPPORTED feature fires. The only feature in active use is the LANDED borrowed-memory DFB.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — **all Case 1 or clean; no Case 2, no compute-kernel blocker:**

  | Factory | Binding | Case | Evidence |
  |---|---|---|---|
  | SingleCore | `input` | Case 1 (`Buffer*` RTA → TA) | factory:187 (`emplace_runtime_args` `src0_buffer`); `reader_unary_interleaved_start_id.cpp:25` |
  | SingleCore | `output` | Case 1 (`Buffer*` RTA → TA) | factory:188; `writer_unary_unpad_dims_split_rows.cpp:44` |
  | BlockInterleaved | `input` | Case 1 (**`->address()` RTA** → TA) | factory:308; `reader_unary_interleaved_wh_multicore.cpp:27` |
  | BlockInterleaved | `output` | Case 1 (**`->address()` RTA** → TA) | factory:295; `writer_unary_stick_layout_wh_multicore.cpp:24` |
  | ColInterleaved | `input` | Case 1 (**`->address()` RTA** → TA) | factory:178; `reader_unary_interleaved_col_multicore.cpp:35` |
  | ColInterleaved | `output` | Case 1 (**`->address()` RTA** → TA) | factory:169; `writer_unary_stick_layout_col_multicore.cpp:26` |
  | Sharded | `input` | clean (borrowed-memory DFB) | factory:97; `reader_unary_sharded.cpp` (fake-push + compute consumer) |
  | Sharded | `output` (out-sharded) | clean (borrowed-memory DFB; producer-only → fake-CB workaround possible) | factory:126 |
  | Sharded | `output` (interleaved-out) | Case 1 (`Buffer*` RTA → TA) | factory:296; `writer_unary_stick_layout_interleaved_blocks.cpp:74` |
  | NDSharded | `input` | Case 1 (`Buffer*` RTA → TA; bound on **reader and writer**) | factory:272,275; `reader_unary_nd_sharded_blocks.cpp:28`, nd-writer:44 |
  | NDSharded | `output` | Case 1 (`Buffer*` RTA → TA) | factory:275; nd-writer:42 |

  All bindings translate to `TensorParameter` / `TensorBinding`, with the kernel constructing `TensorAccessor(ta::name)` (Case 1) — the legacy address-via-RTA disappears in every case. The block/col-interleaved bindings push `buffer->address()` *directly* into an RTA list and are therefore the **silent-wrong-on-cache-hit hazard** the recipe describes; single-core / nd / sharded-interleaved-out push the `Buffer*` object (`emplace_runtime_args`), which the framework auto-registers as a `BufferBinding` and patches on cache hits — correct today, still Case 1 port work.

- **Custom hash:** none (no `compute_program_hash` in the device-operation).

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:** borrowed-memory DFB in `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` — input CB `c_0` (`...sharded...:97`) and out-sharded CB `c_17` (`...sharded...:126`) → `DataflowBufferSpec::borrowed_from = <input>/<output>`. The `c_17` edge is **producer-only** (writer pushes; no in-kernel consumer — it is the resident output); if the spec validator rejects a producerless/consumerless DFB, apply the sanctioned fake-CB workaround (see porting recipe).
- **Fake CBs (address-only):** out-sharded `c_17` only, as above. FYI-P, not a gate.
- **Cross-op / shared kernels (fork on port):** the five factories instantiate kernels they do not own. The interleaved port already forked its shared kernels with the `_metal2` suffix (`untilize_compute_metal2.cpp`, `writer_unary_stick_layout_split_rows_multicore_metal2.cpp`, and reused `reader_unary_start_id.cpp`); this port follows the same convention for the legacy compute kernels (`untilize.cpp`, `untilize_wh.cpp`, `untilize_w.cpp`, `untilize_variable_num_blocks.cpp` — note `untilize_variable_num_blocks_metal2.cpp` already exists) and the eltwise/unary readers, because those sources are still shared by un-ported ops. See Team-only for the full port-together set.
- **RTA varargs:** none in the five remaining factories (the already-ported interleaved factory uses them; the rest use fixed RTAs / CRTAs only — nd writer uses `get_common_arg_val`).
- **TTNN factory analysis (porter-relevant):** no pybind `create_descriptor`, no other migration-risky pybind, no custom `override_runtime_arguments`. Nothing for the porter to delete on the host side.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up:** ✓ clean (no Device-2.0 donor gate; all donor/shared kernels are Device 2.0). Coupling is purely the **port-together set** below.
  - **Function-call escapes (`#include` outside op dir):**
    - `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` — official shared kernel library (compute kernels). No concern.
    - `ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp` — cross-family (ccl) include in the ND reader (`reader_unary_nd_sharded_blocks.cpp:10`) and ND writer (`...nd_sharded.cpp:12`). **No symbol from it is used** — both kernels do sharded access via `TensorAccessor.shard_pages`. Appears to be a stale/dead include (see Misc anomalies). No donor shape to bridge.
    - `ttnn/operations/ccl/sharding_addrgen_helper.hpp` — host-side include in the nd_sharded factory. No concern.
  - **Borrowed kernel files (file-path instantiation) — port-together set:**
    | Kernel | Owning pool/family | Broadly shared? |
    |---|---|---|
    | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | eltwise/unary | yes (eltwise/unary, untilize family, others) |
    | `eltwise/unary/.../reader_unary_interleaved_wh_multicore.cpp` | eltwise/unary | yes |
    | `eltwise/unary/.../reader_unary_interleaved_col_multicore.cpp` | eltwise/unary | yes |
    | `eltwise/unary/.../reader_unary_sharded.cpp` | eltwise/unary | yes |
    | `data_movement/untilize/.../compute/untilize.cpp` | data_movement/untilize | yes (untilize + untilize_with_unpadding; pool/upsample, fold per fork comments) |
    | `data_movement/untilize/.../compute/untilize_wh.cpp` | data_movement/untilize | shared with untilize |
    | `data_movement/untilize/.../compute/untilize_w.cpp` | data_movement/untilize | shared with untilize |
    | `data_movement/untilize/.../compute/untilize_variable_num_blocks.cpp` | data_movement/untilize | shared with untilize (`_metal2` fork already present) |
    | `data_movement/sharded/.../reader_unary_nd_sharded_blocks.cpp` | data_movement/sharded | shared with sharded ops |
    | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | shared pool (`ttnn/cpp/ttnn/kernel`) | yes |
    | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | shared pool | yes |

    A Metal 2.0 CB→DFB / named-token rewrite of any shared kernel breaks its co-borrowers unless the rewrite is forked. The interleaved port already adopted the `_metal2` fork convention precisely for this reason; this port should fork (not rewrite-in-place) every shared kernel it touches, leaving the legacy source for un-ported co-borrowers.
- **Relaxation candidates:** none mined — there is no custom `compute_program_hash` to read. Default strict spec matching applies.
- **TTNN factory analysis (six questions):**
  1. **Op-owned tensors?** **No.** `create_output_tensors` builds only the declared output (`untilize_with_unpadding_device_operation.cpp:254`); no `create_device_tensor`/`allocate_tensor_on_device` for scratch/intermediate tensors in any factory.
  2. **MeshWorkload concept needed?** **No.** Each factory builds a single program; no `create_mesh_workload`/`create_workload_descriptor`, no `cached_mesh_workload_t` on the device-op. Not on the MeshWorkload path at all.
  3. **Pybind `create_descriptor`?** **No.** `untilize_with_unpadding_nanobind.cpp:39` binds only `bind_function<"untilize_with_unpadding">`; no `nb::class_<...ProgramFactory>`.
  4. **Other migration-risky pybind?** **No.** Nanobind exposes only the user-facing op function; no `DeviceOperation`/factory internals.
  5. **Custom hash?** **No** (cross-ref Custom program hash — none present).
  6. **Custom override-runtime-args?** **No.** Factory headers declare only `create_descriptor`; no `override_runtime_arguments`.

## Misc anomalies  *(team-only, non-gating)*

- **Dead struct:** `UntilizeWithUnpaddingMultiCoreSharedVariables` (`device/factories/untilize_with_unpadding_multi_core_shared_variables.hpp:12`) is defined but referenced nowhere in the directory — a leftover from the legacy cached-program style. Routes to the op owner; not port work.
- **Stale include:** `#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"` in `reader_unary_nd_sharded_blocks.cpp:10` and `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:12` — no symbol from the header is used (sharded access goes through `TensorAccessor.shard_pages`). Likely removable.
- **Validation comments:** `untilize_with_unpadding_device_operation.cpp:49` ("I am not sure it is correct to ever use the shard_spec here") and `:115` ("// What else?") are author uncertainty markers in `validate_on_program_cache_miss` — not port work.

## Recipe notes

- **Compute-kernel Device 2.0 framing.** The recipe's Device 2.0 gate is written around dataflow (NoC) idioms and the "wrapper-in-scope → CB-index free-function holdover" test. Pure compute kernels here (`untilize*.cpp`) pass raw CB indices to `compute_kernel_lib::untilize` with **no** `CircularBuffer` wrapper in scope, so the holdover test cannot fire, yet they still require a CB-index → `dfb::name` rewrite at port time (handled by fork). It was momentarily ambiguous whether a CB-index-only compute kernel should read as a Device-2.0 holdover (it should not — there is no wrapper to key off, and `dfb::name` casts to a CB index in template position per the donor-shape table). A one-line note in Check 2 that "compute kernels with no DM-wrapper in scope are not Device-2.0 holdovers; their CB-index→token swap is the routine Metal 2.0 binding rewrite" would remove the ambiguity.
