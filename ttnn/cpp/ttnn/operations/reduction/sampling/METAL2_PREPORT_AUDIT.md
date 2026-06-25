# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/sampling/`

Single device-operation directory:

- **`SamplingDeviceOperation`** (`device/sampling_device_operation.{hpp,cpp}`)
  - `SamplingProgramFactory` (`device/sampling_program_factory.cpp`) — sole ProgramFactory; `create_descriptor` only.

Kernels the factory instantiates:

- `device/kernels/dataflow/reader_values_indices_tensor.cpp` (reader, whole grid)
- `device/kernels/dataflow/writer_interleaved.cpp` (writer, per-core)
- `device/kernels/compute/sampling.cpp` (compute, per-core; bitonic top-k + softmax + RNG)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

**Recipe docs:** `776151aeca8 2026-06-24 docs(metal2): clarify SPSC face-(b) producer-as-consumer, aliased-vs-same-FIFO, no-portable-subset`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/sampling/` |
| **Overall** | RED |
| **DOps / Factories** | `SamplingDeviceOperation` → `SamplingProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | No — donor `generate_bcast_scalar.hpp` is Device 1.0 (RED gate) |
| *Prereqs* — Cross-op escapes | Issue — 1 donor (`generate_bcast_scalar.hpp`) on Device 1.0; others clean |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *Ops readiness* — Sync-free CBs (address-only) | Present: `k` (c_14), `p` (c_15), `output` (c_13) — self-loop workaround |

**Sync-free CBs** = CBs used purely as an address source (the kernel grabs the base pointer and walks the memory, no FIFO producer+consumer pair). The port resolves these with the sanctioned self-loop workaround; FYI-P, not a gate.

## Result

**RED — blocked on the Device 2.0 prerequisite of a donor kernel.** The op itself is on the `ProgramDescriptor` API and its three own kernels are Device-2.0-clean, but the writer kernel calls `generate_bcast_unary_scalar` from the shared-pool donor `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp`, which is entirely on Device 1.0 data-movement idioms (raw `cb_reserve_back(cb_id)` / `get_write_ptr(cb_id)` free functions + `tt_l1_ptr` pointer, no `CircularBuffer` wrapper). Per Prerequisites Check 2, a donor consumed by the op on pre-Device-2.0 idioms blocks the Metal 2.0 port until the donor migrates. **No portable subset** — the donor call is unconditional (every config runs the writer), so the whole op is RED until the donor's Device 2.0 migration lands.

Path forward: route the donor's Device 2.0 migration to the shared-kernel-library / Device 2.0 effort (the donor is broadly shared — see Team-only). Once it is on the `CircularBuffer` wrapper, the op is otherwise feasible and a re-audit will clear all other gates. This is a normal prereq RED, not a permanent blocker.

## Gate detail

- **ProgramDescriptor:** GREEN. `SamplingProgramFactory::create_descriptor` (`sampling_program_factory.cpp:20`) populates a `ProgramDescriptor` with `CBDescriptor`, `KernelDescriptor`, `TensorAccessorArgs`, `emplace_runtime_args` — no imperative `host_api.hpp` builder calls (`CreateProgram` / `CreateCircularBuffer` / `SetRuntimeArgs`).

- **Device 2.0 (every kernel used):** RED — the op's own kernels are clean, but one shared-pool donor function is on Device 1.0.

  Own kernels — clean (Device 2.0 wrappers throughout):
  - `reader_values_indices_tensor.cpp` — `Noc`, `CircularBuffer`, `CoreLocalMem`, `cb.get_write_ptr()`, `noc.async_read(...)`. Uses `cb.get_tile_size()` (member form). Clean.
  - `writer_interleaved.cpp` — `Noc`, `CircularBuffer cb_*`, `CoreLocalMem`, `cb.get_write_ptr()`/`get_read_ptr()`, `TensorAccessor`, `use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out)`. Clean *except* the donor call below.
  - `compute/sampling.cpp` — compute (TRISC) kernel; uses `CircularBuffer` wrappers + compute LLK APIs; data-movement idioms N/A on the compute read path.

  Donor violation (RED gate):

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 43–49 (`generate_bcast_unary_scalar`) | `cb_reserve_back(cb_id, 1)` / `get_write_ptr(cb_id)` / `cb_push_back(cb_id, 1)` + raw `tt_l1_ptr` write — entire function on Device 1.0 | none (donor takes a bare `uint32_t cb_id`; no `CircularBuffer` object) |

  Call site in the op: `writer_interleaved.cpp:127` — `generate_bcast_unary_scalar(cb_id_temp, temp_packed);`.

  This is **not** the YELLOW isolated-holdover carve-out: that applies only to a single-CB-index free function where the Device-2.0 wrapper is *already in scope at the call site* and a member-form replacement exists. Here the donor *file* itself is wholly Device 1.0 (no wrapper object exists in the donor), so the Metal 2.0 binding tokens have nothing to attach to in the donor — a hard prereq, routed to the Device 2.0 / shared-kernel-library effort. Donor class: `ttnn/cpp/ttnn/kernel/` (shared pool).

  > **Other donors are clean** (Device 2.0): `generate_mask` in `sdpa_decode/.../dataflow_common.hpp` (uses `Noc`, `CircularBuffer`, `get_tile_size(cb_id)` [sanctioned free function], `fill_tile`/`copy_tile` Device-2.0 helpers); `calculate_and_prepare_reduce_scaler` in `kernel_lib/reduce_helpers_dataflow.{hpp,inl}` (`dfb.get_write_ptr()`, `Noc`, `get_tile_size(dfb_id)` [sanctioned]); `compute_kernel_lib::reduce` in `kernel_lib/reduce_helpers_compute.hpp` (compute-side). See Out-of-directory coupling.

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer`, no `.global_circular_buffer` field, no `remote_*`/`.remote_index`. |
  | Dynamic CircularBuffer (borrowed memory) | N/A | No `CBDescriptor::.buffer` set; all 18 CBs are plain static CBs (`.buffer` unset). |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set on any `CBDescriptor`. |
  | Aliased Circular Buffers | N/A | Every `format_descriptors` initializer is single-element (one `CBFormatDescriptor`). |
  | GlobalSemaphore | N/A | No `GlobalSemaphore`; the op uses **no semaphores at all**. |
  | Non-zero semaphore initial value | N/A | No semaphores. |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | All `TensorAccessorArgs(tensor)` are the single-argument static form; no `ArgConfig::Runtime*`. |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer*` calls (no override hook). |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Fixed-count tensor args (`SamplingInputs` has 5 named tensors + 1 optional output); CTAs are fixed-shape; no `get_compile_time_arg_val(i)` with runtime-varying `i`. |

  No UNSUPPORTED feature fires. Feature compatibility is GREEN (no gate).

- **DFB endpoint legality (SPSC):** GREEN. Every CB sits in the legal `(1 producer, 1 consumer)` window per node, or is a port-handled sync-free/single-ended CB. The reader+writer+compute kernels all co-reside on each active core (one core per user), so the census is per-node across all three. No CB has ≥2 producers or ≥2 consumers on a node; no hidden second writer (no `get_write_ptr`/`fifo_wr_ptr` co-fill gated by side-channel semaphores — the op uses no semaphores); no dead CBs. Per-CB census:

  | CB (index) | Producer | Consumer | Verdict |
  |---|---|---|---|
  | `input_values` c_0 | reader (push) | compute (wait/pop) | legal (1,1) |
  | `cb_local_vals` c_1 | compute (push) | writer (wait/pop) | legal (1,1) |
  | `index`/`cb_intermed` c_2 | reader (`generate_index_tile` push) | compute (wait/pop) | legal (1,1) |
  | `scaler_max` c_3 | writer (`calculate_and_prepare_reduce_scaler`) | compute (reduce) | legal (1,1) |
  | `topk_mask` c_4 | writer (`generate_mask`) | compute (wait) | legal (1,1) |
  | `input_transposed` c_5 | compute | compute (in-place) | legal (1,1) same kernel |
  | `index_transposed` c_6 | compute | compute (in-place) | legal (1,1) same kernel |
  | `values_cb` c_7 | compute (push) | compute (in-place add/mul/reduce) | legal (1,1) same kernel |
  | `output_ind` c_8 | compute (push) | writer (`cb_local_indices` wait/pop) | legal (1,1) |
  | `cb_cur_max` c_9 | compute | compute | legal (1,1) same kernel |
  | `cb_cur_sum` c_10 | compute | compute | legal (1,1) same kernel |
  | `rand` c_11 | compute (`generate_rand_tile` push) | writer (`cb_rand` wait/pop) | legal (1,1) |
  | `final_indices_rm` c_12 | reader (streams `input_indices` tensor, push ×num_users) | writer (`cb_final_indices` wait/pop) | legal (1,1) |
  | `output` c_13 | writer (raw `get_write_ptr` → NoC write; no FIFO) | — | **single-ended / sync-free** → self-loop |
  | `k` c_14 | writer (reserve/push then raw `get_write_ptr` read; no FIFO consumer) | — | **single-ended / sync-free** → self-loop |
  | `p` c_15 | writer (reserve/push then raw read; no FIFO consumer) | — | **single-ended / sync-free** → self-loop |
  | `temp` c_16 | writer (`generate_bcast_unary_scalar` push; also raw `get_write_ptr` overlay) | compute (`mul_block_bcast_scalar_inplace` wait) | legal (1,1) |
  | `scaler_sum` c_17 | writer (`calculate_and_prepare_reduce_scaler`) | compute (reduce) | legal (1,1) |

  Note `c_12` mapping: the factory passes `final_indices_rm_cb_index` as reader CTA[1] and the reader kernel reads CTA[1] under the local name `input_indices_cb_index` — i.e. the input-indices tensor is streamed *into* the `final_indices_rm` CB (`final_indices_rm_cb_index`), which the writer later consumes. Not a separate binding (see Misc anomalies).

## Port-work summary  *(mirrors the brief)*

> No brief is issued — the op is RED. These items are recorded for when the Device 2.0 donor prereq clears and the op is re-audited.

- **Tensor bindings** (per binding) — all Case 1 (via `TensorAccessor`), all clean of the silent-wrong hazard once bound:
  - `input_values` (reader) — Case 1. `TensorAccessor(s0_args, values_addr)`; `values_addr` is RTA 0.
  - `input_indices` (reader) — Case 1. `TensorAccessor(s1_args, indices_addr)`; `indices_addr` is RTA 1.
  - `output` (writer) — Case 1. `TensorAccessor(dst_args, dst_addr)`; `dst_addr` is RTA 0.
  - `temp` (writer) — Case 1. `TensorAccessor(temp_args, temp_addr)`; `temp_addr` is RTA 1.
  - `k` (writer) — Case 1. `TensorAccessor(k_args, k_addr)`; `k_addr` is RTA 2.
  - `p` (writer) — Case 1. `TensorAccessor(p_args, p_addr)`; `p_addr` is RTA 3.
  - No raw-pointer (Case 2) tensor reads; no buffer-address smuggled as a non-`TensorAccessor` RTA. The RTAs are buffer base addresses fed straight into `TensorAccessor` constructors → the legacy address-via-RTA + `TensorAccessorArgs` plumbing disappear at port time.
- **Custom hash:** none — the device-op defines no `compute_program_hash`. Nothing to delete.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:** none (no aliased CB, no borrowed-mem DFB, no dynamic TA, no non-zero sem init).
- **Sync-free CBs (address-only):** three CBs are pure address sources with no FIFO producer+consumer pair, each resolved by the self-loop workaround (FYI-P, not a gate):
  - `output` c_13 — `writer_interleaved.cpp:147` (`cb_out.get_write_ptr()`); writer NoC-writes the result; no `reserve_back`/`push_back`/`wait_front`. Single-ended (1 raw writer, 0 readers).
  - `k` c_14 — `writer_interleaved.cpp:94-102`; writer FIFO-produces (`reserve_back`/`push_back`) then reads its own buffer by raw pointer (`k_ptr[core_id]`); no FIFO consumer. Single-ended.
  - `p` c_15 — `writer_interleaved.cpp:105-113`; same shape as `k`. Single-ended.
  - (`temp` c_16 is **not** sync-free — it is a real `(1,1)` FIFO: `generate_bcast_unary_scalar` produces, compute `mul_block_bcast_scalar_inplace` consumes — though note the writer also raw-writes `cb_temp.get_write_ptr()` at line 117 with the `reserve_back`/`push_back` commented out, before the FIFO producer runs. Same kernel, so still one producer endpoint; legal. See Misc anomalies.)
- **Dead CBs (zero endpoints):** none — every CB index is referenced by at least one kernel.
- **Cross-op / shared kernels:** see Team-only. The RED donor (`generate_bcast_scalar.hpp`) and the clean donors (`sdpa_decode` `dataflow_common.hpp`, `kernel_lib/reduce_helpers_*`) are all out-of-directory; each induces a port-the-family-together coupling.
- **RTA varargs:** none. The reader/writer read fixed positional RTAs; no `num_runtime_varargs` and no runtime-varying-index RTA loop.
- **TTNN factory analysis (porter-relevant):** none — no pybind `create_descriptor`, no other migration-risky pybind, no custom `override_runtime_arguments`.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ⭐ blocked — one donor (`generate_bcast_scalar.hpp`) is pre-Device-2.0 (Shape 4-class: the donor-side Device 2.0 gate). All other donors are ✓ clean.

**Summary table** (op kernel → donor file):

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `writer_interleaved.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | shared pool (`ttnn/cpp/ttnn/kernel/`) | ⭐ ✗ Device 1.0 — gates (Prereq Check 2) |
| `writer_interleaved.cpp` | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` | cross-family donor (transformer/sdpa_decode) | ✓ clean (Device 2.0) |
| `writer_interleaved.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` (+`.inl`) | shared kernel library | ✓ clean (Device 2.0) |
| `compute/sampling.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | shared kernel library | ✓ clean (compute-side) |

**Per-call detail (donors with ✗/⭐):**

- `generate_bcast_scalar.hpp` → `generate_bcast_unary_scalar(uint32_t cb_id, uint32_t scalar)` — body uses `cb_reserve_back(cb_id, 1)`, `get_write_ptr(cb_id)` (Device 1.0 free function — has a `cb_obj.get_write_ptr()` member-form replacement), raw `tt_l1_ptr uint32_t*` write, `cb_push_back(cb_id, 1)`. Pre-Device-2.0. The sibling functions in the same file (`generate_bcast_col_scalar`, `generate_bcast_row_scalar`) are likewise Device 1.0; the whole file needs migration. **Broadly shared** — this donor is included by ~10 ops, so its Device 2.0 rewrite is one shared change across all co-borrowers:
  - `experimental/transformer/fused_distributed_rmsnorm` (2 readers)
  - `experimental/transformer/dit_layernorm_post_all_gather`, `..._pre_all_gather`
  - `experimental/ccl/rms_allgather` (writer)
  - `normalization/layernorm_distributed` (3 kernels)
  - `normalization/softmax` (reader)
  - `reduction/sampling` (this op, writer)

**Borrowed kernel files (file-path kernel instantiation):** the op instantiates **only its own three kernel files** (all under `device/kernels/`); it borrows no kernel `.cpp` from a shared pool. The coupling above is via `#include` (function-call escape), not file-path instantiation.

### TTNN factory analysis (six questions)

1. **Op-owned tensors?** **No.** `SamplingProgramFactory::create_descriptor` allocates only CBs; it creates no device tensors. The output is produced via `create_output_tensors` (`sampling_device_operation.cpp:183`) using `create_device_tensor` of the declared output spec, or the preallocated output — the standard output, not an op-owned intermediate.
2. **MeshWorkload concept needed?** **No.** Single-program op; no `create_mesh_workload` / `cached_mesh_workload_t`. No op-owned tensors either (so not even the artifact case).
3. **Pybind `create_descriptor`?** **No.** `sampling_nanobind.cpp` binds only the user-facing function via `ttnn::bind_function<"sampling">` (the expected, carved-out surface). No `nb::class_<...ProgramFactory>`.
4. **Other migration-risky pybind?** **None.** No `DeviceOperation`/factory/param class bound to Python.
5. **Custom hash?** **No.** `SamplingDeviceOperation` defines no `compute_program_hash` (`sampling_device_operation.hpp:19-28`) — uses the default reflection-based hash. (Cross-ref: Custom program hash subject — nothing to delete.)
6. **Custom override-runtime-args?** **No.** No `override_runtime_arguments` on the factory.

### Relaxation candidates

None — no custom hash to mine.

## Misc anomalies  *(team-only, non-gating)*

- **`out_stick_size` is a dead CTA in the writer.** `writer_interleaved.cpp:59` documents CTA `args_base + 8` (`aligned_out0_unit_size`, passed from the factory at `sampling_program_factory.cpp:395`) as "unused in kernel." Harmless; op-owner may drop it.
- **`temp` CB raw-write overlay with commented-out FIFO ops.** `writer_interleaved.cpp:116-121`: `cb_temp.reserve_back(1)` and `cb_temp.push_back(1)` are commented out, yet the writer raw-writes via `cb_temp.get_write_ptr()` (NoC read of the temp chunk) before `generate_bcast_unary_scalar(cb_id_temp, ...)` (line 127) does the real FIFO `reserve_back`/`push_back`. The raw overlay relies on the FIFO write-ptr position being unchanged across the two writes. Functionally fine on Gen1 (same kernel, sequential), but the dual-write-into-one-CB shape is subtle; an op-owner may want to clean it up. Not a port concern.
- **Reader CB-name mismatch (documented above).** Reader CTA[1] is `final_indices_rm_cb_index` (c_12) but the kernel reads it under the local name `input_indices_cb_index`. Intentional reuse (input-indices tensor streamed into the final-indices CB), but the misleading name is worth a comment. Not a defect.

## Recipe notes

- The YELLOW isolated-holdover carve-out (audit §Prerequisites Check 2) is framed around a free function "where the Device-2.0 wrapper object is already in scope at the call site." For a **donor** that takes a bare `uint32_t cb_id` and is itself wholly Device 1.0 (no wrapper anywhere in the donor body), the carve-out clearly does not apply and the donor-side gate (Check 2 RED) governs — but the recipe could state explicitly that a *donor function whose signature is `uint32_t cb_id` and whose body is Device 1.0* is a Check-2 RED, not a YELLOW holdover, since the surface shape (`get_write_ptr(cb_id)`) superficially resembles the holdover pattern. The disambiguator used here: a holdover needs the wrapper *already in scope in the same kernel*; a donor on `uint32_t cb_id` constructs no wrapper, so the binding tokens have nothing to attach to.

---

## ⚠️ Post-port-attempt correction (2026-06-25) — self-loop workaround does NOT apply (compute-only)

The Device-2.0 donor prerequisite (`generate_bcast_scalar.hpp`) was resolved (**PR #48148**, validated). The port was then attempted and **gated**: this audit prescribed "the sanctioned self-loop workaround" for the writer's sync-free CBs **`c_13`** (output staging: `cb_out.get_write_ptr()` → `noc.async_write`, no FIFO partner) and **`c_14`/`c_15`** (writer `reserve_back`/`push_back` + raw `CoreLocalMem` self-read, no consumer). That prescription is **wrong**: the self-loop workaround is **compute-kernel-only**; binding it on the (DM) writer FATALs, and Metal 2.0 has no scratch/sync-free DFB. `c_14`/`c_15` could be salvaged by hoisting their DRAM read writer→reader (PRODUCER→CONSUMER), but **`c_13` has no escape**.

**Corrected status: RED, framework-blocked** (wait-for-feature: sync-free/scratch DFB or DM-kernel self-loop). See [[metal2-port-portability-predictor]]. The #48148 prereq is necessary but not sufficient.

---

## 🔄 Revision (2026-06-25, supersedes the correction above) — workaround found; NOT framework-blocked

The "framework-blocked / wait-for-feature" verdict above is **overstated**. A workaround exists with **no framework change**: the **cross-kernel DFB bridge**. Only a DM-kernel *self*-loop FATALs; a DM kernel paired *cross-kernel* with a different co-located kernel (DM↔DM or DM↔compute) is fully legal. **Proven in shipped code:** the landed JointSDPA port (PR #48175, 160 passed/0 failed) binds `mask`/`scale`/`col_identity` as PRODUCER on the **writer (DM)** → CONSUMER on **compute** (`joint_sdpa_program_factory.cpp:359-451`); the SPSC validator accepts and runs them.

**This op:** `cb_k`/`cb_p` are **read-staging** → relocate to the reader (reader PRODUCER → writer CONSUMER; proven-class). `cb_out` is **write-staging** (writer assembles output, DMAs to DRAM) → bridge PRODUCER-on-writer + a TERMINAL no-op CONSUMER on compute. The no-op-consumer variant is high-confidence (validator only counts endpoints; terminal consume after producer push → no deadlock) but **not yet hardware-verified** — confirm by completing the port. **PORTABLE via cross-kernel bridge**; no framework feature needed.
