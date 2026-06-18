# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/sliding_window/halo`

Single device-operation, single program factory:

- **`HaloDeviceOperation`** (`device/halo_device_operation.hpp` / `.cpp`)
  - `UntilizeWithHaloProgramFactory` (`device/untilize_with_halo_program_factory.cpp`) — sole factory, on the `program_factory_t` variant.

Kernels referenced by `KernelDescriptor::kernel_source`:

- `device/kernels/dataflow/halo_gather.cpp` — reader (instantiated twice: RISCV_0 / RISCV_1 split reader).
- `device/kernels/compute/pack_untilize.cpp` — compute (instantiated only when `!skip_untilize`, i.e. tiled input).

Donor headers pulled in by the referenced kernels (in scope, followed across boundaries):

- `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp` (pool family) — included by `halo_gather.cpp`.
- `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` (`kernel_lib` shared pool) — included by `pack_untilize.cpp`.

No unreferenced kernel files in the directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/sliding_window/halo` |
| **Overall** | GREEN |
| **DOps / Factories** | `HaloDeviceOperation` → `UntilizeWithHaloProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | Yes: `UntilizeWithHaloProgramFactory::create_workload_descriptor` — 4 halo config tensors @ `untilize_with_halo_program_factory.cpp:436-459`, parked @ `:473-487` |
| *TTNN Readiness* — MeshWorkload needed | No (op-owned tensors — carried natively, single-program; see Q1/Q2) |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | present: `out_cb` (`:166`), `pad_cb0/1` (`:172-174`), and in the L1-config path `padding_config0/1` + `gather_config0/1` (`:230-268`) — fake-CB workaround |

**Fake CBs** = CBs used purely as an address source. **Litmus: does the CB have a producer *and* a consumer?** No producer–consumer pair → fake: the port resolves it with the sanctioned fake-CB workaround (see the porting recipe) — **FYI-P heads-up, not a gate**.

## Result

**GREEN → brief issued.** All gates clear: the op is on the `ProgramDescriptor` API (via `create_workload_descriptor` → `WorkloadDescriptor` of `ProgramDescriptor`s built from `KernelDescriptor` / `CBDescriptor`), every kernel it uses (own + both donor headers) is Device 2.0, and no UNSUPPORTED Appendix A feature fires. Port work is the routine borrowed-memory-DFB + fake-CB-workaround translation plus a single Case-1 tensor binding on the DRAM-config code path. No subset carve-out needed — the whole op is portable.

## Gate detail

- **ProgramDescriptor:** GREEN. `UntilizeWithHaloProgramFactory::create_workload_descriptor` (`untilize_with_halo_program_factory.cpp:388`) returns a `tt::tt_metal::WorkloadDescriptor`; the per-coord program is assembled in `build_halo_program` (`:83`) entirely from `ProgramDescriptor desc` (`:146`), `CBDescriptor` (`add_cb`, `:68`), and `KernelDescriptor` (`:204`, `:340`, `:351`). No imperative `host_api.hpp` builder calls (`CreateProgram`/`CreateKernel`/`CreateCircularBuffer`/`SetRuntimeArgs`) appear. The `#include <tt-metalium/host_api.hpp>` at `:12` brings in transitive types only; no imperative builder is invoked.

- **Device 2.0 (every kernel used):** GREEN.
  - `halo_gather.cpp` is fully on Device 2.0 wrappers: `Noc noc;` (`:282`) with `noc.async_write`/`noc.async_read`/`noc.async_read_barrier`/`noc.async_write_barrier`; `experimental::CB` objects (`:283-288`) with member-form `wait_front`/`pop_front`/`reserve_back`/`push_back`/`get_read_ptr()`/`get_write_ptr()`. The DRAM path uses `TensorAccessor` / `TensorAccessorArgs` (`:296-308`). No raw `noc_async_read`, no `InterleavedAddrGen`/`ShardedAddrGen`, no CB-index free-function holdovers (every `get_read_ptr`/`get_write_ptr` is `cb_obj.method()` form — `:105,108,205,330,335,343`).
  - `pack_untilize.cpp` uses the `compute_kernel_lib::untilize*` helpers (`kernel_lib/untilize_helpers.hpp`) keyed by CB index NTTPs — sanctioned compute-side LLK usage, no Device 1.0 data-movement idioms.
  - Donor `pool/.../experimental_device_api.hpp` is itself a Device 2.0 convenience header (`experimental::CB`, `Noc`, `set_async_read_state`/`async_read_with_state`); no pre-D2.0 idioms.
  - Donor `kernel_lib/untilize_helpers.hpp` is shared-lib class (lib team owned) and is Device 2.0.

  No holdover rows.

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `CBDescriptor.global_circular_buffer` field set, no `remote_index`/`remote_cb_*` idioms. |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `CBDescriptor.buffer` set non-null on `src_cb` (`:152`, input buffer) and `out_cb` (`:166`, output buffer), and on the four config CBs in the L1 path (`:238,248,258,268`, `buffer` arg = `config_tensors_in_dram ? nullptr : <cfg_buffer>`). Port uses `DataflowBufferSpec::borrowed_from`. (`out_cb` and the L1 config CBs additionally lack a FIFO producer/consumer pair → also fake-CB; see Heads-ups.) |
  | CBDescriptor `address_offset` (non-zero) | N/A | `CBDescriptor.address_offset` never set (default 0); `add_cb` never assigns it. |
  | Aliased Circular Buffers | N/A | Every `format_descriptors` initializer in `add_cb` (`:71-75`) is single-element. No multi-`buffer_index` config. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore`, no `CreateGlobalSemaphore`. |
  | Non-zero semaphore initial value | N/A | Op uses no semaphores at all (no `CreateSemaphore` / `SemaphoreDescriptor`). |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | Only the single-arg `TensorAccessorArgs(buffer)` form (`:319-323`); no `ArgConfig::Runtime*` token. |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer*` / `UpdateDynamicCircularBufferAddressAndTotalSize` calls. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = Tensor` (single fixed input, `halo_device_operation.hpp:22`), not a `std::vector<Tensor>`. Both kernels read CTAs at fixed literal indices (`get_compile_time_arg_val(0..21)`); no runtime-varying CTA loop. (Reader runtime args are the routine `config_read_index`/`core_index` scalars — RTA, not CTA, and not varargs.) |

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - **input tensor → `src_cb`** (`:152`, `.buffer = src_buffer`): **clean** — borrowed-memory DFB. Producer `src_cb.reserve_back`/`push_back` (`halo_gather.cpp:317-318`, reader_0 only); consumers = compute `untilize` (reads `src_cb_id`) and the reader skip-untilize path `src_cb.wait_front` (`:340`). This is the canonical sharded-reader-fake-push-satisfying-compute-consumer case the causal-link gate marks clean. Port: `DataflowBufferSpec::borrowed_from = <input TensorParameter>`.
  - **output tensor → `out_cb`** (`:166`, `.buffer = dst_buffer`): **fake CB** — read only via `out_cb.get_write_ptr()` (`halo_gather.cpp:108,205`) as a NoC base for hand-rolled `noc.async_write`; no `push_back`/`pop_front`/`wait_front` on `out_cb`. No producer/consumer FIFO → cannot be a DFB; port applies the fake-CB workaround over the borrowed output buffer. (FYI-P, not a gate.)
  - **config tensors (DRAM path, `config_tensors_in_dram == true`) → `padding_config*` / `gather_config*`**: **Case 1** (via `TensorAccessor`). `buffer->address()` is baked into the reader's **compile-time** args (`:307-317`) and consumed kernel-side by `TensorAccessor(padding_config_tensor_args, padding_config_dram_addr)` / `TensorAccessor(gather_config_tensor_args, gather_config_dram_addr)` (`halo_gather.cpp:300-301`). CTA-baked-address form → classify Case 1: express each config tensor as a `TensorParameter` and the CTA-baked address + kernel-side NTTP base both disappear. (CTA-baked, not the silent-wrong RTA hazard.)
  - **config tensors (L1 path, `config_tensors_in_dram == false`) → same CBs** (`.buffer` set, `:238,248,258,268`): **fake CB** on borrowed memory — read only via `gather_config_cb.get_read_ptr()` / `padding_config_cb.get_read_ptr()` (`halo_gather.cpp:105,343`) with no FIFO push in L1 mode. Per-factory split with the DRAM Case-1 above: same binding, fake-CB in L1, Case-1 in DRAM. Port applies the fake-CB workaround for the L1 path.
- **Custom hash:** none.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:**
  - Borrowed-memory DFB (`DataflowBufferSpec::borrowed_from`): `src_cb` @ `:152`, `out_cb` @ `:166`, and the four config CBs in the L1 path @ `:238,248,258,268`.
- **Fake CBs (address-only):** at the `(CB, endpoint)` edge —
  - `out_cb` (`:166`) — written by raw NoC via `get_write_ptr()`, no producer FIFO (`halo_gather.cpp:108,205`).
  - `pad_cb0` / `pad_cb1` (`:172-174`) — scratch immediate-value buffers, only `get_write_ptr()`/`get_read_ptr()` (`halo_gather.cpp:330,335`), no FIFO; not borrowed-memory.
  - `padding_config0/1`, `gather_config0/1` in the **L1** path (`:230-268` with `buffer` non-null) — read-by-pointer only (`halo_gather.cpp:105,343`). (In the DRAM path these are real DFB endpoints fed by `noc.async_read`; see Case-1 binding above.)
  - Resolved with the sanctioned fake-CB workaround; does **not** gate.
- **Cross-op / shared kernels:** see Team-only (no function-call escapes; two donor *header* includes, both Device 2.0).
- **RTA varargs:** none. Reader RTAs are single fixed scalars (`core_index` / `config_read_index`, `:366-373` / `halo_gather.cpp:303`); compute RTA is a single `total_blocks` (`pack_untilize.cpp:21`).
- **TTNN factory analysis (porter-relevant):** no pybind `create_descriptor`, no other migration-risky pybind, no custom `override_runtime_arguments`.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** No function-call escapes into other ops' helpers; the only out-of-directory `#include`s are two convenience/library *headers* that contribute inline types and templates, both already Device 2.0. No `Semaphore`/`TensorAccessor`/addr-gen donor *function signatures* are crossed.

Includes outside the op directory (per kernel):

| Op kernel | Include | Donor class |
|---|---|---|
| `halo_gather.cpp` | `api/dataflow/dataflow_api.h` | `tt_metal/*` (LLK) — no concern |
| `halo_gather.cpp` | `ttnn/operations/pool/device/kernels/experimental_device_api.hpp` | Cross-family header (pool), but it is a thin Device 2.0 alias/wrapper header (`experimental::CB`, `Noc`, `read_with_state`); contributes inline templates, not a stateful donor signature. ✓ |
| `pack_untilize.cpp` | `ttnn/kernel_lib/untilize_helpers.hpp` (+ `api/compute/pack_untilize.h`) | `kernel_lib` shared pool — lib team owned. ✓ |

Per-call detail: omitted (all rolls ✓; no ⚠/✗/⭐ donor function signatures).

**Borrowed kernel files (file-path kernel instantiation).** The op owns both of the kernel `.cpp` files it instantiates (`device/kernels/dataflow/halo_gather.cpp`, `device/kernels/compute/pack_untilize.cpp`) — no file-path borrow of another op's `.cpp`. No port-together set from file-path coupling.

Coupling that *does* exist is at the **shared-header** level (not a gate, not file-path kernel coupling): `experimental_device_api.hpp` is broadly included (conv2d, pool generic/upsample/grid_sample/rotate, fold, conv3d, padded_slice, slice_write, convert_to_chw/hwc, reduction generic, and halo); `untilize_helpers.hpp` is broadly included (untilize, conv_bmm_tilize, groupnorm, paged_cache, rotary_embedding, kv_cache, sdpa_decode, attn_matmul, conv3d, halo). A Metal 2.0 rewrite *inside* either shared header would ripple to all consumers — but halo neither owns nor is expected to modify these; they are lib/shared-pool surface. No action for the halo port.

### Relaxation candidates (mined from custom hash)

None — the op has no custom `compute_program_hash`. (Caching keys on the default reflection hash over `HaloParams` + the single input `Tensor`'s `TensorSpec`.)

### TTNN factory analysis — six-question answers

1. **Op-owned tensors? — Yes.** `create_workload_descriptor` builds four intermediate halo-config device tensors — `pad_config_device_tensor0/1`, `gather_config_device_tensor0/1` via `sliding_window::move_config_tensor_to_device` (`untilize_with_halo_program_factory.cpp:436-459`) — and parks each (wrapped in a `std::make_shared<Tensor>`) on `workload_descriptor.buffers` (`:473-487`) so the backing device memory outlives the cached workload. These are neither inputs (`tensor_args_t = Tensor`, the single activation) nor the declared output. `MetalV2FactoryConcept::op_owned_tensors` carries these natively. (`create_device_tensor` @ `halo_device_operation.cpp:96` allocates the *declared* output, not an owned intermediate.)

2. **MeshWorkload concept needed? — No** (op-owned tensors only — carried natively, single-program). The factory provides `create_workload_descriptor` and returns a `WorkloadDescriptor`, which is the MeshWorkload-path signature — but the in-code rationale (`:492-495`) states this is a single-device op whose per-coord `ProgramDescriptor` is structurally identical for every coord (built once at `:496`, copied/moved into each coord-range entry at `:508-513`). The op sits on the WorkloadDescriptor path *only* to carry its four op-owned config tensors (Q1) past the framework's single-program plumbing — the false-positive trap the recipe names. No genuine cross-program / cross-device coordination. Ports cleanly as single-program with `op_owned_tensors`.

3. **Pybind `create_descriptor`? — No.** No `*_nanobind.cpp` in the op directory; grep for `halo` in `sliding_window/*nanobind*` returns nothing. (The op is exposed via `ttnn::halo` / `prim::halo` free functions, called from conv2d/pool — no factory-innards binding.)

4. **Other migration-risky pybind? — None.** No `nb::class_<>` wrapping `HaloDeviceOperation` or `UntilizeWithHaloProgramFactory`, no pybound device-op methods.

5. **Custom hash? — No.** `HaloDeviceOperation` (`halo_device_operation.hpp:19-31`) declares only `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`; no `compute_program_hash`.

6. **Custom override-runtime-args? — No.** `UntilizeWithHaloProgramFactory` declares only `create_workload_descriptor`; no `override_runtime_arguments`.

## Misc anomalies

- **Unused `HaloParams` field `output_memory_config`** (`halo_device_operation_types.hpp:21`): set nowhere in `halo()` (`halo_device_operation.cpp:135-145` omits it) and read nowhere in the factory; it is still part of the attributes struct fed to the default reflection hash, so it participates in the cache key as a default-constructed `MemoryConfig`. Harmless (constant), but dead. Team-only; not porter-actionable.
- **Duplicate include in `pack_untilize.cpp`** (`pack_untilize.cpp:7` and `:12` both include `untilize_helpers.hpp`, via different path spellings). Header-guarded, harmless; cosmetic.
- **`remote_read` is plumbed but unsupported in the kernel:** the factory threads `remote_read` into CTA index 10 (`untilize_with_halo_program_factory.cpp:287`), but `halo_gather.cpp:277` hard-asserts `static_assert(!remote_read, ...)`. Any caller passing `remote_read == true` is a compile-time kernel failure, not a runtime guard. Pre-existing; not port scope.

## Recipe notes

- The **multi-consumer borrowed-memory DFB** here (`src_cb` borrowed from the input shard, produced by reader_0's fake-push, consumed by both the compute `untilize` and the readers' skip-untilize path across three kernel instances) is exactly the "sharded reader's fake-push satisfying a waiting compute consumer" case the causal-link gate (§TensorAccessor handling) names as the canonical legit DFB. The recipe's litmus (producer + consumer) marks it clean, which is what I did. Flagging only because the *split-reader + 3-kernel-consumer* fan-out on a single borrowed DFB is more elaborate than the two-endpoint example the recipe sketches — a porter should confirm the framework's `borrowed_from` + multi-binding spec accepts one borrowed DFB bound to three kernels (2 dataflow + 1 compute) before relying on it. This is a confirm-at-port-time note, not a gate per the recipe as written.
