# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/pool/generic`

Single device-operation in this directory:

- **`Pool2D`** (`device/pool_op.hpp`, `device/pool_op.cpp`)
  - **`Pool2D::MultiCore`** — the only program factory, `create_workload_descriptor` (`device/pool_multi_core_program_factory.cpp`). Builds the `ProgramDescriptor` via the free function `pool2d_multi_core_sharded_with_halo_v2_impl_new` in the same file.

Pybind: `generic_pools_nanobind.cpp` (binds only the user-facing `max_pool2d` / `avg_pool2d` functions). Host orchestration: `generic_pools.cpp` / `.hpp`.

Referenced kernels (via `KernelDescriptor::kernel_source`):
- Dataflow: `device/kernels/dataflow/reader_pool_2d.cpp` (non-indices path), `device/kernels/dataflow/reader_mpwi.cpp` (`return_indices` path).
- Compute: `device/kernels/compute/compute_pool_2d.cpp` (non-indices path), `device/kernels/compute/compute_mpwi.cpp` (`return_indices` path).
- Cross-op donor header (in scope): `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp` (one level up, `pool/` family), which itself includes `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/pool/generic` |
| **Overall** | GREEN |
| **DOps / Factories** | `Pool2D` → `Pool2D::MultiCore` (`create_workload_descriptor`) |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok (donor header is Device 2.0) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | Yes: `Pool2D::MultiCore::create_workload_descriptor` — reader-indices tensor (`pool_multi_core_program_factory.cpp:1133-1152`) and avg-pool scalar-config tensor (`:1192-1212`), both parked in `workload_descriptor.buffers` |
| *TTNN Readiness* — MeshWorkload needed | No (op-owned tensors — carried natively, single-program; see Q1/Q2) |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) — `device/pool_op.cpp:168`, `device/pool_op.hpp:68` |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | present: `raw_in_cb` (input shard, reader read-ptr), and sharded-path `in_reader_indices_cb` / `config_cb` — see Heads-ups (workaround) |

**Fake CBs** = CBs used purely as an address source (producer/consumer litmus below).

## Result

**GREEN → brief issued.** All gates clear: the op is on the `ProgramDescriptor` API, every kernel it exercises (own + the cross-op `pool_kernels_common.hpp` donor) is Device 2.0 compliant, and no UNSUPPORTED Appendix A feature is in use. Port work is substantial but mechanical: per-binding tensor wiring (one Case-2 raw-pointer input via the fake-CB workaround, two Case-1 TensorAccessor configs that vary per code path), deletion of the custom `compute_program_hash`, and the aliased-CB / fake-CB constructs. No clean-subset carve-out needed — the whole op clears.

## Gate detail

- **ProgramDescriptor:** GREEN. The factory populates a `tt::tt_metal::ProgramDescriptor` (`pool_multi_core_program_factory.cpp:385`) with `CBDescriptor` (`:394`, `:414`, `:657`), `KernelDescriptor` (`:828`, `:840`, `:922`), `DataMovementConfigDescriptor` / `ComputeConfigDescriptor`, and returns a `tt::tt_metal::WorkloadDescriptor` (`:1114`). No imperative `host_api.hpp` builder calls (`CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs`) appear. (`<tt-metalium/host_api.hpp>` is `#include`d at `:19` but only for ancillary types; the build is descriptor-based.)

- **Device 2.0 (every kernel used):** GREEN. All four referenced kernels and the donor header consistently use the Device 2.0 surface — `experimental::CB`, `Noc`, `UnicastEndpoint`, `noc.async_read(...)`, `experimental::local_addr(...)`, `TensorAccessor` / `TensorAccessorArgs`. No Device 1.0 idioms (`InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_read` / `get_noc_addr_from_bank_id` / raw sem addresses) appear in any of them. The only CB-index free functions in the donor are `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` (`pool_kernels_common.hpp:45-47`, `:61`, `:75-76`, `:129`) — both **sanctioned** by the Device 2.0 migration guide (kept as free functions in its migrated examples), so they are NOT holdovers. No YELLOW holdovers found.

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer`, `.global_circular_buffer` field, `remote_index`, or `remote_cb` anywhere. |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `add_sharded_cb` sets `CBDescriptor::buffer` to a real `Buffer*` for input (`:456`), reader-indices sharded path (`:486`), config sharded path (`:754`), output (`:691`) and output-indices (`:707`). Port uses `borrowed_from`. (Caveat: several of these are *address-only* / fake CBs — see Heads-ups.) |
  | CBDescriptor `address_offset` (non-zero) | N/A | `add_sharded_cb` / `add_local_cb` never set `.address_offset`; no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`. |
  | Aliased Circular Buffers | GREEN | One multi-element `format_descriptors` CBDescriptor: `pre_tilize_cb_id` + `fast_tilize_cb_id` share one L1 allocation with two `CBFormatDescriptor` views (`:657-675`). Port uses `advanced_options.alias_with`. Active only for TILED output (`has_pre_tilize`). |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` / `CreateGlobalSemaphore` / `global_semaphore.hpp`. |
  | Non-zero semaphore initial value | N/A | The op creates **no semaphores at all** (no `CreateSemaphore`, no `SemaphoreDescriptor`). |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` token. `TensorAccessorArgs(reader_indices_buffer)` / `(config_buffer)` (`:816`, `:818`) are the single-argument static form. |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` / `UpdateDynamicCircularBufferAddressAndTotalSize`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` carries a single `const Tensor&` (`pool_op.hpp:42`); the `std::vector<Tensor>` is only the fixed 1-or-2 output return value. No kernel reads `get_compile_time_arg_val(i)` on a runtime-varying `i`. Output count is fixed by the `return_indices` CTA, known at factory-build time. |

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - **input** (`raw_in_cb` / `in_shard_cb_id`) — **Case 2 (raw pointer)**. The kernel pulls the shard base via `in_shard_cb.get_read_ptr()` (`reader_pool_2d.cpp:267`, `reader_mpwi.cpp:483`) and does explicit address arithmetic in `read_kernel_with_top_left_index` (`reader_pool_2d.cpp:94-101`). No producer pushes into this CB and nothing waits on it → it is a **fake CB**, not a genuine borrowed-memory DFB. Port: bind the input as a `TensorParameter`, pull the base via the `get_bank_base_address` bridge, keep the raw walk; resolve the address-only CB with the sanctioned fake-CB workaround. *(Dataflow kernels — bridge available; not the compute-kernel-blocked case.)*
  - **reader_indices** (`in_reader_indices_cb_id`) — **per-path split.** *DRAM path* (`config_tensor_in_dram`): **Case 1** — the kernel reads via `TensorAccessor` in `load_config_tensor_if_in_dram` (`pool_kernels_common.hpp:82-89`), with `TensorAccessorArgs(reader_indices_buffer)` appended to CTAs (`:816`) and the DRAM base baked into CTA #35 (`:793`). Express as `TensorParameter`; the CTA-baked address + `TensorAccessorArgs` plumbing disappear. *Sharded path*: address-only fake CB read via `get_read_ptr()` (`reader_pool_2d.cpp:280`) → fake-CB workaround. Record via Per-DeviceOperation/per-path attribution.
  - **scalar config** (`config_cb_id`, avg-pool with non-trivial scalar layout) — same per-path split: **Case 1** in the DRAM path (TensorAccessor via `load_config_tensor_if_in_dram`, CTA-baked address #33 at `:791`, `TensorAccessorArgs(config_buffer)` at `:818`); address-only fake CB in the sharded path (`reader_pool_2d.cpp:294`).
  - **output(s)** (`out_cb_id`, and `out_idx_cb_id` when `return_indices`) — **clean** genuine borrowed-memory DFBs: produced (`push_back`) by compute (`compute_pool_2d.cpp:226`, `:256`) or by the MPWI reader (`reader_mpwi.cpp:347-348`) and consumed by sharded writeback. Port via `borrowed_from`.
- **Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception). `device/pool_op.cpp:168-185`, declared `device/pool_op.hpp:68`.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:**
  - **Aliased CB** @ `pool_multi_core_program_factory.cpp:657-675` — `pre_tilize_cb_id` / `fast_tilize_cb_id` two `CBFormatDescriptor` views over one L1 allocation (producer stick-packed view + consumer full-tile view). Port: one `DataflowBufferSpec` per index, mutually declared via `advanced_options.alias_with`; do not split. TILED-output path only.
  - **Borrowed-memory DFB** — output CBs (`:691`, `:707`) via `DataflowBufferSpec::borrowed_from`.
- **Fake CBs (address-only):** the same CB can be a real LLK operand on one edge and address-only on another; record each address-only `(CB, endpoint)`:
  - `raw_in_cb` (input shard) — reader endpoints, read by base pointer only, no producer/consumer FIFO. `reader_pool_2d.cpp:267`, `reader_mpwi.cpp:483`.
  - `in_reader_indices_cb` (sharded path) — read by base pointer at `reader_pool_2d.cpp:280` / `reader_mpwi.cpp:497`; in the **DRAM path** the same CB *is* a genuine DFB (producer `push_back` in `load_config_tensor_if_in_dram`, consumer `wait_front`), so only the sharded edge is fake.
  - `config_cb` (sharded path) — read by base pointer at `reader_pool_2d.cpp:294`; genuine DFB in the DRAM path.
  - Port resolves each address-only edge with the sanctioned fake-CB workaround; does NOT gate.
- **Cross-op / shared kernels:** all four kernels `#include` the donor header `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp` (`pool/` family, one level above this op). The donor is Device 2.0 (✓). Its functions take resource handles in Device-2.0-native shapes (`experimental::CB`, `Noc`, `TensorAccessor` / `TensorAccessorArgs<N>` NTTP). See Team-only for the by-shape detail. No file-path kernel instantiation of *another op's* `.cpp` (every kernel `.cpp` instantiated is owned by this op directory).
- **RTA varargs:** none. Kernels read fixed positional RTAs (`get_arg_val<uint32_t>(0..3)`); no counted loop over a runtime-varying arg index.
- **TTNN factory analysis (porter-relevant):** no pybind `create_descriptor`; no other migration-risky pybind; no custom `override_runtime_arguments`. (Custom `compute_program_hash` present — carried as PORT WORK above.)

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** One donor header, fully Device 2.0; all consumed functions present resource handles in Device-2.0-native or constexpr-castable shapes. No ⭐/✗ scheduling blockers.

**Summary table** (one row per op-kernel × donor-file pair):

| Op kernel | Donor file | Roll-up |
|---|---|---|
| `reader_pool_2d.cpp` | `pool/device/kernels/pool_kernels_common.hpp` | ✓ |
| `reader_mpwi.cpp` | `pool/device/kernels/pool_kernels_common.hpp` | ✓ |
| `compute_pool_2d.cpp` | `pool/device/kernels/experimental_device_api.hpp` (alias-only header) | ✓ |
| `compute_mpwi.cpp` | `pool/device/kernels/experimental_device_api.hpp` (alias-only header) | ✓ |

`pool_kernels_common.hpp` itself includes `experimental_device_api.hpp` (the `experimental::CB` alias + Device-2.0 includes). Both are in the `pool/` family (in-family escape — does not gate the Metal 2.0 syntax rewrite; port the family together).

**Per-call detail.** Functions the op's kernels call from `pool_kernels_common.hpp`, with handle shapes:
- `fill_with_val(uint32_t addr, ...)` — raw L1 address, scalar. Not a tensor/CB/sem handle.
- `clear_out_tiles<cb_id, clear_value_cb_id>(Noc, experimental::CB, experimental::CB[, num_tiles])` — `Noc` (✓ D2.0), `experimental::CB` (✓), CB index as NTTP (✓ `dfb::name` constexpr cast).
- `zero_out_tiles<cb_id>(Noc, experimental::CB)` / `zero_out_page(Noc, experimental::CB)` — ✓.
- `load_config_tensor_if_in_dram<config_dram_addr, config_page_size, tensor_args_index, cb_reader_index>(Noc, experimental::CB, uint32_t)` — uses `TensorAccessorArgs<tensor_args_index>()` (Shape 2, ✗ "not OK" by the table — porter passes `ta::name.args`, workable but suboptimal) plus a CTA-baked `config_dram_addr` NTTP (the address the Case-1 binding eliminates). This is the donor entry point for the DRAM reader-indices / config TensorAccessor reads. Not a blocker.
- `fill_scalar<...>(experimental::CB, ...)` — ✓ `experimental::CB` + scalars.

No `Semaphore`/`uint32_t sem_addr`, no old-style addr-gen (Shape 4), no `CircularBuffer&` (raw-ref) shapes in the donor surface → no ⭐ entries.

**Borrowed kernel files (file-path instantiation):** none from another op. Every kernel `.cpp` the factory instantiates lives in this op's own `device/kernels/` directory. The shared *header* coupling (above) induces a port-the-pool-family-together consideration for `pool_kernels_common.hpp` / `experimental_device_api.hpp` (consumed by other `pool/` ops — conv/halo/upsample/grid_sample neighbours), but that's a header rewrite, not a kernel-`.cpp` co-borrow.

### Relaxation candidates (mined from the custom hash before deletion — FALLIBLE, default strict)

`compute_program_hash` (`pool_op.cpp:172-184`) keys on `sliding_window_config_.get_hash()`, `pool_type_`, `output_layout_`, `memory_config_`, `compute_kernel_config_`, `divisor_override_`, `count_include_pad_`, `return_indices_`, `config_tensor_in_dram`, the input `memory_config()`, input `dtype`, and output `dtype`. **Notably it omits `TensorSpec` / tensor shapes** beyond what `sliding_window_config_.get_hash()` encodes (the input H/W/C/N are carried inside the SlidingWindowConfig, so a same-shape-different-storage cache hit is plausible-but-untracked here) — this is the canonical reason to delete it and revert to the default (which keys on `TensorSpec`). No safe relaxation candidate stands out; the op's correctness depends tightly on shape via the precomputed reader-indices/scalar-config tensors, so treat as strict. Candidates fallible — verify before any relaxation.

### TTNN factory analysis (six questions)

1. **Op-owned tensors? — Yes.** `create_workload_descriptor` materializes a host-built reader-indices config tensor, moves it to device, wraps it in a `shared_ptr<Tensor>`, and parks it in `workload_descriptor.buffers` (`pool_multi_core_program_factory.cpp:1133-1152`). For avg-pool variants needing per-stick scalars it additionally builds and parks a scalar-config tensor (`:1192-1212`). Both are intermediates beyond the declared input/output. (`MetalV2FactoryConcept` carries these via `op_owned_tensors`.) Separately, `generic_pools.cpp:909` creates a DRAM output tensor, but that is in the host-level DRAM-slicing orchestration wrapper (`run_sliced_op`), outside the `Pool2D` DeviceOperation factory — note only.
2. **MeshWorkload concept needed? — No (op-owned-tensor artifact).** The factory provides `create_workload_descriptor` and returns a `WorkloadDescriptor`, but the program is structurally identical for every coord — built **once** and copied/moved into each coord-range entry (`:1217-1280`), with an explicit comment "Single-device op … Pool2D doesn't depend on cluster position." This is the false-positive trap: it sits on the workload path only because the legacy framework couldn't carry op-owned tensors on the single-program path. No cross-program or cross-device coordination → ports cleanly as morally single-program with `op_owned_tensors`. (Cause: Q1.)
3. **Pybind `create_descriptor`? — No.** `generic_pools_nanobind.cpp` binds only `bind_function<"max_pool2d">` / `<"avg_pool2d">` (the user-facing functions). No `nb::class_<...ProgramFactory>` and no `def_static("create_descriptor", ...)`.
4. **Other migration-risky pybind? — None.** No `nb::class_<>` wrapping the `Pool2D` DeviceOperation or its factory; no `compute_program_hash` / `compute_output_specs` / `select_program_factory` exposed to Python; no factory parameter existing only to drive a pybind hook.
5. **Custom hash? — Yes.** `Pool2D::compute_program_hash`, `device/pool_op.cpp:168` (declared `device/pool_op.hpp:68`). Treatment in the Custom-program-hash subject (delete → default).
6. **Custom override-runtime-args? — No.** No `static void Pool2D::MultiCore::override_runtime_arguments(...)`. Per-core runtime args are emitted inline during program build (`:937-974`); the framework patches them by kernel index on cache hit (the deterministic kernel-push order at `:976-983` is relied on for that).

## Misc anomalies  *(team-only, non-gating)*

- **Large flat CTA arg list shared across two kernel pairs.** `reader0_ct_args` (`:757-814`, 55 entries) and `compute_ct_args` (`:857-899`, 39 entries) are single flat positional lists where args past a comment boundary are "MPWI-only" and ignored by the pool2d kernels (e.g. reader CTAs 37-54, compute CTAs 17-37). Functionally fine today, but a sizeable bank of position-coupled CTAs the port will translate to named CTAs; note the index `reader_tensor_args_index = 55` hardcoded in both readers (`reader_pool_2d.cpp:212`, `reader_mpwi.cpp:427`) must stay in lockstep with the host CTA count. Route to op owner as a maintainability note.
- **`out_c` parameter passed to the impl (`:271`) but used only in debug logging** (`:1037`) — effectively dead beyond logging. Minor.
