# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/sliding_window/halo/`

**Device operations / factories in this directory:**

- **`HaloDeviceOperation`** (`device/halo_device_operation.hpp` / `.cpp`)
  - `UntilizeWithHaloProgramFactory` (`device/untilize_with_halo_program_factory.cpp`, 517 lines; `.hpp` declaration)

Single device-operation, single program factory — audited as one unit.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/sliding_window/halo/` |
| **Overall** | **RED** |
| **DOps / Factories** | `HaloDeviceOperation` → `UntilizeWithHaloProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes (uses `ProgramDescriptor` + `KernelDescriptor` + `CBDescriptor`, via `create_workload_descriptor`) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok (shared header is Device-2.0; lib-team owned) |
| *Feature Support* — overall | GREEN (no UNSUPPORTED feature fires) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | **Yes** — `UntilizeWithHaloProgramFactory::create_workload_descriptor` allocates & parks 4 config tensors (`untilize_with_halo_program_factory.cpp:436-487`) |
| *TTNN Readiness* — MeshWorkload needed | No (op-owned-tensor artifact only — morally single-program; the factory copies one identical `ProgramDescriptor` to every coord, `:496-513`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No (no `compute_program_hash` on `HaloDeviceOperation`) |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None (all CBs have producer+consumer; see notes) |

## Result

**RED — blocked on framework work (op-owned device resources), routed to the Metal 2.0 framework / TTNN-factory team.**

The single Metal 2.0 factory concept available on `main` is `ProgramSpecFactoryConcept`, which requires **single-program with no op-owned device resources** — every tensor referenced must be reachable from `tensor_args` / `tensor_return_value` (`port_op_to_metal2_ttnn_factory.md`, "Feasibility gate"). This op's factory **allocates and owns four intermediate config `Tensor`s on device** (`pad_config0/1`, `gather_config0/1`) and parks them on the `WorkloadDescriptor` (`WorkloadDescriptor::buffers`) so their backing buffers outlive the cached programs. These are device resources the factory owns — they are **not** in `tensor_args` (which is the single input `Tensor`) nor in `tensor_return_value` (the single output `Tensor`). Today a `TensorArgument` that does not reference an input/output tensor `TT_FATAL`s, and the framework adapter has an explicit TODO for op-owned resources.

The op is **morally single-program**: it is a single-device op (halo doesn't depend on cluster position), builds one `ProgramDescriptor` and copies it identically to every mesh coord. It is on the `WorkloadDescriptor` / MeshWorkload path **only** because that path is the current vehicle for parking op-owned intermediate buffers — exactly the "legacy `MeshWorkload` is a resource workaround, not a genuine multi-program need" case the factory doc's heads-up describes. So this is **blocked-on-framework (op-owned-resource support), a resource-workaround unwind — not genuine per-coord variation.**

**Path forward:** unblocks once the reworked factory-concept design (op-owned device resources / caching-strategy axis) lands on `main`. Nothing porter-resolvable today.

All other audit subjects clear — the rest of the report is forward-looking context for when the framework gap closes.

## Gate detail

- **ProgramDescriptor:** GREEN. The factory builds a `ProgramDescriptor desc` (`untilize_with_halo_program_factory.cpp:146`), populating `desc.cbs` via `CBDescriptor` (`add_cb`, `:68-77`) and `desc.kernels` via `KernelDescriptor` (`ComputeConfigDescriptor` / `DataMovementConfigDescriptor`, `:204-222`, `:340-381`). No imperative `host_api.hpp` builder calls (`CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs`). The factory's entry point is `create_workload_descriptor` returning a `tt::tt_metal::WorkloadDescriptor` (`:388`).

- **Device 2.0 (every kernel used):** GREEN. Two kernels referenced:
  - `device/kernels/dataflow/halo_gather.cpp` — uses Device-2.0 wrappers throughout: `experimental::Noc noc` (`:282`), `experimental::CB` objects (`:283-288`), `noc.async_write` / `noc.async_read` / `noc.async_*_barrier`, `experimental::set_read_state` / `read_with_state`, `cb.get_read_ptr()` / `get_write_ptr()` / `wait_front` / `pop_front` / `reserve_back` / `push_back`. No Device-1.0 addr-gen (`InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_read(`); grep clean).
  - `device/kernels/compute/pack_untilize.cpp` — Device-2.0 compute via `compute_kernel_lib::untilize*` from `ttnn/kernel_lib/untilize_helpers.hpp`.

  The CB-index CTAs the kernels still read via `get_compile_time_arg_val(...)` and feed into the `experimental::CB(uint32_t)` ctor (`halo_gather.cpp:258-288`; `pack_untilize.cpp:15-19`) are **not** Device-1.0 holdovers — the wrappers are already in scope; these are exactly the legacy-CB-index → `dfb::name` swaps the [kernel-side whitelist] performs at port time. So the Device 2.0 gate is clean (not even YELLOW). *(These are routine port work, surfaced only because the port is blocked upstream by the op-owned-resource gate.)*

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `global_circular_buffer` field set on any `CBDescriptor`; `add_cb` never sets it (`:68-77`). |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | LANDED + in use. `add_cb`'s `.buffer` is set non-null for `src_cb` (input buffer, `:152`), `out_cb` (output buffer, `:166`), and for the four config CBs when `!config_tensors_in_dram` (`:238,248,258,268`). Port path: `DataflowBufferSpec::borrowed_from`. (The config-CB borrows are from the *op-owned* tensors — which is itself the RED blocker above.) |
  | CBDescriptor `address_offset` (non-zero) | N/A | `address_offset` is never set; `add_cb` omits it (defaults 0). |
  | Aliased Circular Buffers | N/A | Every `format_descriptors` initializer is single-element (`{{CBFormatDescriptor{...}}}`, `:71-75`). |
  | GlobalSemaphore | N/A | Op uses no semaphores. |
  | Non-zero semaphore initial value | N/A | Op uses no semaphores. |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | `TensorAccessorArgs(buffer)` is the single-arg form (`:319-323`); no `ArgConfig::Runtime*`. |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer*` / `UpdateDynamicCircularBufferAddressAndTotalSize` calls. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = Tensor` (single fixed tensor); kernels read CTAs at fixed indices, no runtime-varying-index CTA loop. |

  No UNSUPPORTED entry fires. Feature support is GREEN.

## Port-work summary  *(mirrors the brief — forward-looking; no brief issued while RED)*

- **Tensor bindings** (per binding):
  - `input` (src_cb) — **Case 1** re-express via `TensorParameter`/`TensorBinding`; it is read through a borrowed-memory CB (`src_buffer`, `:152`) so it is effectively a clean borrowed-memory DFB read (`borrowed_from`). Routine.
  - `output` (out_cb) — borrowed-memory DFB (`dst_buffer`, `:166`); clean via `borrowed_from`. Routine.
  - `pad_config0/1`, `gather_config0/1` — these are the **op-owned** config tensors. In the `!config_tensors_in_dram` path they back borrowed-memory CBs (`:238,248,258,268`); in the `config_tensors_in_dram` path they are read via `TensorAccessor` from DRAM (`halo_gather.cpp:296-308`), with **their addresses + `TensorAccessorArgs` baked into compile-time args** on the host (`untilize_with_halo_program_factory.cpp:307-323`). Either way these bindings cannot be expressed under `ProgramSpecFactoryConcept` because the tensors aren't in `tensor_args`/`tensor_return_value` — **this is the RED blocker, not routine port work.**
- **Custom hash:** none (already default reflection-based).

## Heads-ups  *(forward-looking)*

- **Notable LANDED constructs:** borrowed-memory DFB in use (input, output, and config CBs on the L1 path) → port via `DataflowBufferSpec::borrowed_from`. No aliased CB, no dynamic TA, no non-zero sem init.
- **Fake CBs (address-only):** none observed. Every CB has a producer and a consumer — e.g. `src_cb` is fake-pushed by the reader (`reserve_back`/`push_back`, `halo_gather.cpp:317-318`) and waited on by the compute untilize / `skip_untilize` path; the config CBs are produced (borrowed memory / DRAM read) and consumed (`get_read_ptr`). Litmus passes.
- **Cross-op / shared kernels (file-path):** none — neither `halo_gather.cpp` nor `pack_untilize.cpp` is file-path-instantiated by any other op (grep clean). No port-together set on the file-path axis.
- **Shared header coupling (function-call escape):** `halo_gather.cpp` `#include`s the pool family's `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp` (`:9`) — a Device-2.0 convenience header (`experimental::CB`, `Noc`, endpoints) consumed broadly across the tree (conv2d, fold, pool/upsample/grid_sample, reduction, cnn, conv3d, padded_slice, slice_write, …). It supplies only Device-2.0 type aliases/wrappers, so it crosses cleanly and does not gate; flagged because it is an out-of-directory `#include` from another op family that a porter will see. `pack_untilize.cpp` `#include`s `ttnn/kernel_lib/untilize_helpers.hpp` (official shared kernel lib — lib-team owned, no concern). *Note: `pack_untilize.cpp` includes `untilize_helpers.hpp` twice (`:7` and `:12`, one bare, one full-path) — a harmless duplicate, recorded under Misc anomalies.*
- **RTA varargs:** none. RTAs are fixed-count (compute: single `total_blocks`; reader: single config-read index in the DRAM path).
- **TTNN factory analysis (porter-relevant):** no pybind `create_descriptor`, no other risky pybind, no custom `override_runtime_arguments`. (Nanobind `sliding_window_nanobind.cpp` binds only `ParallelConfig` and `Op2DSliceConfig` — normal op surface, not a finding.)

## Team-only

- **TensorAccessor convertibility:** N/A — no Case-2 bindings. The only `TensorAccessor` usage (config tensors in the DRAM path, `halo_gather.cpp:296-308`) is a standard page-read pattern (Case 1 re-express), not exotic.
- **Out-of-directory coupling & donor shape:**
  - Op-level roll-up: **✓ clean** (no gating donor coupling).
  - Summary table:

    | Op kernel | Donor `#include` | Donor class | Status |
    |---|---|---|---|
    | `halo_gather.cpp` | `pool/device/kernels/experimental_device_api.hpp` | cross-family header, but Device-2.0 alias/wrapper convenience header | ✓ |
    | `pack_untilize.cpp` | `ttnn/kernel_lib/untilize_helpers.hpp` | official shared kernel lib (class 2) | ✓ |

  - Per-call detail: omitted — all rolls ✓ (no ⚠/✗/⭐). The pool header provides only Device-2.0 wrapper *types* (`experimental::CB` alias, `Noc`, endpoints), not addr-gen donor functions; `untilize_helpers.hpp` provides Device-2.0 `compute_kernel_lib::untilize*` templates.
  - Borrowed kernel files (file-path instantiation): none — the factory instantiates only its own `halo_gather.cpp` and `pack_untilize.cpp` (`:206`, `:275`). No file-path borrow in or out.
- **Relaxation candidates:** none mined (no custom hash to mine).
- **TTNN factory analysis — six questions:**
  1. **Op-owned tensors? — YES.** `create_workload_descriptor` calls `sliding_window::generate_halo_kernel_config_tensors` then `construct_on_host_config_tensor` (`:427-434`) and `move_config_tensor_to_device` for all four configs (`:436-459`), wraps each in `std::make_shared<Tensor>` and pushes `{owner, buffer}` onto `workload_descriptor.buffers` (`:473-487`). These are device tensors the factory allocates and owns, outside `tensor_args` (single input `Tensor`) and `tensor_return_value` (single output `Tensor`). **This is the primary RED blocker.**
  2. **MeshWorkload concept needed? — NO (op-owned-tensor artifact only).** The factory provides `create_workload_descriptor` and the op is on the MeshWorkload path, but the program is structurally identical across coords — built once and copied (`:496-513`), explicitly commented as a single-device op whose program doesn't depend on cluster position (`:492-495`). The MeshWorkload path is used **solely** to carry the op-owned config buffers (Q1). Morally single-program; needs op-owned-resource support, not multi-program support.
  3. **Pybind `create_descriptor`? — NO.** No `nb::class_<...ProgramFactory>` / `def_static("create_descriptor", ...)` anywhere; `sliding_window_nanobind.cpp` binds only config structs.
  4. **Other migration-risky pybind? — NO.** No `DeviceOperation`/factory-internals bindings exposed to Python.
  5. **Custom hash? — NO.** `HaloDeviceOperation` (`halo_device_operation.hpp:19-31`) declares no `compute_program_hash`; default reflection-based hash applies.
  6. **Custom override-runtime-args? — NO.** No `override_runtime_arguments` on the factory.

## Misc anomalies  *(team-only, non-gating)*

- `pack_untilize.cpp:7` and `:12` both `#include` `untilize_helpers.hpp` (one as `"ttnn/kernel_lib/untilize_helpers.hpp"`, one as the full `"ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"`) — a redundant duplicate include. Header-guarded, so harmless; routes to the op owner, not the port.
- `halo_gather.cpp:277` has `static_assert(!remote_read, ...)` — the kernel hard-rejects `remote_read`, yet `remote_read` is a live op attribute threaded all the way through the factory into `common_reader_ct_args[10]` (`untilize_with_halo_program_factory.cpp:287`). The host accepts `remote_read=true` but the kernel would fail to compile; effectively a dead/contradictory path. Team/op-owner FYI, not port work.

## TTNN ProgramFactory

### Concept
**BLOCKED** — op-owned device resources (the four halo config tensors). Not `ProgramSpecFactoryConcept`-compatible on `main`.

### Fit
- Single vs multi-program: **single** — one `ProgramDescriptor` stamped identically across the mesh (built once, copied per coord, `untilize_with_halo_program_factory.cpp:496-513`).
- Op-owned device resources: **present — BLOCKED.** `pad_config0/1`, `gather_config0/1` allocated in the factory and parked on `WorkloadDescriptor::buffers` (`:436-487`).
- Tensor-arg matching: strict (default; no relaxation observed or warranted).
- Legacy-to-Metal-2.0 shape: **legacy `WorkloadDescriptor`/MeshWorkload is a resource workaround** — morally single-program (see the factory doc's heads-up); the MeshWorkload path exists only to carry the op-owned config buffers, not for genuine per-coord variation.

### Custom compute_program_hash
None — already default reflection-based hash.

### Stop signals
**BLOCKED.** Missing framework capability: **op-owned device resources** on the single-program factory concept (the reworked factory-concept design — op-owned resources / caching-strategy axis — is not yet on `main`). Overall audit result is **RED**. Path forward: unblocks when that framework work lands; this is a resource-workaround unwind, not a multi-program need.

## Recipe notes

None — the audit recipe and the factory-concept feasibility gate covered this op's shape (the "legacy MeshWorkload as op-owned-resource workaround" heads-up in `port_op_to_metal2_ttnn_factory.md` matched exactly).
