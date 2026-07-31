# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward`

One device operation, one program factory:

- **`GeluBackwardDeviceOperation`** (`device/gelu_backward_device_operation.{hpp,cpp}`)
  - `GeluBackwardProgramFactory` (`device/gelu_backward_program_factory.cpp`) — the op's only factory.

Kernels the factory instantiates (all by `SourceType::FILE_PATH`):

| Role | Path | Ownership |
|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | **borrowed** (eltwise/binary) |
| Writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **borrowed** (eltwise/unary) |
| Compute (`approximate == "tanh"`) | `device/kernels/compute/eltwise_bw_gelu_approx_tanh.cpp` | op-owned |
| Compute (otherwise) | `device/kernels/compute/eltwise_bw_gelu_poly.cpp` | op-owned |

No unreferenced kernel files in the op directory — both compute kernels are live, selected by the `approximate` attribute (`gelu_backward_program_factory.cpp:119-127`). No `experimental/quasar/` copy of this op exists.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
*(This working checkout has no `docs/.../host_apis/metal_2.0/` tree — the provenance command prints nothing here. The hash above is from the doc-branch checkout the recipe was read from, `/localdev/edwinlee/Port_Recipe`, same-content-verified against the recipe file used for this run.)*

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `GeluBackwardDeviceOperation` → `GeluBackwardProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all four kernels (2 borrowed DM + 2 op-owned compute) are Device 2.0 |
| *Prereqs* — Cross-op escapes | Ok — no function-call escape outside `tt_metal/*`; two borrowed *kernel files* (coordination cost, non-gating) |
| *Feature Support* — overall | **GREEN** (all four Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | N/A — every CTA read is at a constexpr offset |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No (sheet also: `Backdoor custom hash` = no; `Formerly custom hashed?` = yes — historical, now removed) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none — no address arithmetic reaches any kernel arg |
| *Port work* — Tensor bindings (per binding) | `grad_output` Case 1 · `input` Case 1 · `output` Case 1 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — every `TensorAccessor` is 2-arg |
| *Port work* — CB endpoints | all legal (3 CBs, each a plain 1 producer + 1 consumer) |

**CB endpoints** are dispositions, not gates. Here every CB is already a genuine 1:1 FIFO on every node, so no self-loop, no 1P+1C assignment, no multi-binding flag, and no dead-CB drop is needed. The op has a single instantiation shape (interleaved + TILE, enforced by validation), and the `approximate` branch changes only the compute-kernel *source*, not the CB topology — so there is no per-config flip to record.

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓, Offset base pointers ✓, TensorAccessor 3rd argument ✓. The porter brief is at `METAL2_PORT_BRIEF.md` in this directory.

This is a small, textbook interleaved elementwise op: three CBs, three kernels, no semaphores, no sharding, no multicast, no offset arithmetic. The only non-trivial port consideration is that both dataflow kernels are **borrowed and broadly shared**, so the port creates the first `_metal2` fork of each (see Heads-ups).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet (fetched fresh this run; row `experimental/unary_backward/gelu_backward` / `GeluBackwardDeviceOperation` / `GeluBackwardProgramFactory`) reports `Is able to port? = yes`, with every conjunct `no`/clear: `Concept = descriptor`, `Custom hash = no`, `Runtime-args update (get_dynamic_runtime_args) = no`, `Override runtime args method? = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`.

  Cross-check against the code — clean on every cheaply-checkable column:

  | Column | Sheet | Code evidence |
  |---|---|---|
  | `Concept` | `descriptor` | `create_descriptor()` returning `tt::tt_metal::ProgramDescriptor` @ `device/gelu_backward_program_factory.hpp:13`, defined @ `device/gelu_backward_program_factory.cpp:17` |
  | `Custom hash` | `no` | no `compute_program_hash` anywhere in the op directory (grep over all 11 files) |
  | `get_dynamic_runtime_args` | `no` | no such hook on `GeluBackwardDeviceOperation` (`device/gelu_backward_device_operation.hpp:20-31` declares only `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`) |
  | `override_runtime_arguments` | `no` | absent — same declaration list |
  | `Pybind descriptor` | `no` | `gelu_backward_nanobind.cpp:64-73` binds only the user-facing `ttnn.experimental.gelu_bw`; no `create_descriptor` binding, no `nb::class_` of the device op |
  | `Op-owned tensors?` | (blank) | `create_descriptor` returns a `ProgramDescriptor`, not a `WorkloadDescriptor` — structurally cannot carry op-owned tensors |
  | Factory-set match | 1 row | 1 factory in code (`program_factory_t = std::variant<GeluBackwardProgramFactory>` @ `device/gelu_backward_device_operation.hpp:25`) — one-to-one, no phantom or missing row |

  Cross-column invariants hold (`get_dynamic_runtime_args = no` on a `descriptor` concept; no op-owned tensors on a `descriptor` row). The sheet's `Factory definition path` and `Declared in` cells both point at the real files.

- **Device 2.0 (every kernel used):** **GREEN** — no violations, so the violations table is empty. All four kernels are structurally Device 2.0:

  | Kernel | Evidence |
  |---|---|
  | `reader_binary_interleaved_start_id.cpp` (borrowed, eltwise/binary) | `Noc noc;` + `DataflowBuffer dfb0/dfb1` objects (`:37-39`); `noc.async_read(s0, dfb0, …)` / `noc.async_read_barrier()` (`:67,92,100`); `dfb0.reserve_back` / `dfb0.push_back` (`:91,103`). No `noc_async_read`, no `InterleavedAddrGen`/`ShardedAddrGen` family, no raw sem addresses. |
  | `writer_unary_interleaved_start_id.cpp` (borrowed, eltwise/unary) | `Noc noc;` + `DataflowBuffer dfb` (`:21-22`); `noc.async_write(dfb, s, …)` / `noc.async_writes_flushed()` / `noc.async_write_barrier()` (`:41,42,45`); `dfb.wait_front` / `dfb.pop_front` (`:40,43`). |
  | `eltwise_bw_gelu_approx_tanh.cpp` (op-owned) | `CircularBuffer` wrapper objects for every FIFO op (`:30-32`, then `.reserve_back`/`.wait_front`/`.pop_front`/`.push_back` @ `:45-47,116-118`). |
  | `eltwise_bw_gelu_poly.cpp` (op-owned) | Same shape (`:28-30`, `:37-39`, `:55-57`). |

  Two classes of CB-index free function appear and neither is a holdover:
  - **Sanctioned per the recipe's Green bullet** — `get_tile_size(cb_id_in0/in1)` @ `reader_binary_interleaved_start_id.cpp:45,52` and `get_local_cb_interface(cb_id_out).fifo_page_size` @ `writer_unary_interleaved_start_id.cpp:19`. Both are explicitly on the sanctioned list.
  - **Compute LLK APIs, outside the Device 2.0 data-movement boundary** — `copy_tile(cb_grad_out, 0, 0)`, `pack_tile(0, cb_grad_in)`, `unary_op_init_common(cb_grad_out, cb_grad_in)` in both compute kernels. These take a CB index by design and have **no wrapper-method replacement**: `tt_metal/hw/inc/api/dataflow/circular_buffer.h` exposes no `copy_tile`/`pack_tile` member (its method set is `reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_tile_address`/`read_tile_value`/`get_tile_size`/`get_tile_hw`/`get_dataformat`/`get_write_ptr`/`get_read_ptr`/`pages_*`/`scoped_lock`). The Green bullet's test — *wrapper-method replacement exists* — is not met, so these are not holdovers. (See Recipe notes #4: the recipe would benefit from saying this outright.)

  Note for the reader: the two borrowed DM kernels are already **ahead** of Device 2.0 — they use `DataflowBuffer` (the Metal 2.0 DFB object from `api/dataflow/dataflow_buffer.h`) constructed from a raw index, rather than the Device 2.0 `CircularBuffer` wrapper. That is strictly a superset of compliance (see Recipe notes #5), and at port time the raw-index construction becomes `DataflowBuffer(dfb::name)`.

- **Feature compatibility:** every Appendix A entry scanned against host code, factory, descriptors, and all four kernels. No entry fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Three `CBDescriptor` literals (`gelu_backward_program_factory.cpp:49,59,69`) set only `total_size`, `core_ranges`, `format_descriptors`. No `.global_circular_buffer` field, no `experimental::CreateCircularBuffer(…, global_cb)`, no `.remote_index(`, no `remote_cb_*` identifier, no `global_circular_buffer.hpp` include anywhere in the op. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` is never set on any of the three descriptors (defaults to 0). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` call. |
  | GlobalSemaphore | N/A | The op uses **no semaphores at all** — no `GlobalSemaphore`, no `CreateGlobalSemaphore`, no `global_semaphore.hpp`, and no plain `SemaphoreDescriptor` either. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | *Op-level signal absent*: `tensor_args_t` is a fixed three-field struct (`GeluBackwardInputs` @ `device/gelu_backward_device_operation_types.hpp:18-22` — `grad_output`, `input`, `std::optional<Tensor> preallocated_input_grad`), no variable-count container. *Kernel-level signal (the decider) absent*: the reader reads `get_compile_time_arg_val(0)` and two `TensorAccessorArgs<N>` at constexpr offsets (`reader_binary_interleaved_start_id.cpp:27,29,30` — the second offset is `src0_args.next_compile_time_args_offset()`, a constexpr); the writer reads `get_compile_time_arg_val(0)` and `TensorAccessorArgs<1>` (`writer_unary_interleaved_start_id.cpp:15,16`); the compute kernels read **no** CTAs. No runtime-varying CTA index exists. |

- **CB endpoints (GATE-free):** every CB is a plain 1:1 FIFO — 1 locked producer + 1 locked consumer on every node. No disposition needed. All three CBs are declared over `all_cores` (`gelu_backward_program_factory.cpp:51,61,71`) and all three kernels run over `all_cores` (`:93,106,132`), so every node's census is identical.

  | CB | Config | Touchers on a node | Roles | Verdict |
  |---|---|---|---|---|
  | `c_0` (grad_output tiles) | all (single shape) | reader, compute | reader = **locked producer** (`dfb0.reserve_back`/`push_back` @ reader `:91,103`); compute = **locked consumer** (`cb_grad_out_cb.wait_front`/`pop_front` @ tanh `:46,116` / poly `:38,55`) | plain 1:1 — legal |
  | `c_1` (input tiles) | all | reader, compute | reader = **locked producer** (`dfb1.reserve_back`/`push_back` @ reader `:96,107`); compute = **locked consumer** (`cb_input_cb.wait_front`/`pop_front` @ tanh `:47,117` / poly `:39,56`) | plain 1:1 — legal |
  | `c_2` (grad_in / output tiles) | all | compute, writer | compute = **locked producer** (`cb_grad_in_cb.reserve_back`/`push_back` @ tanh `:45,118` / poly `:37,57`); writer = **locked consumer** (`dfb.wait_front`/`pop_front` @ writer `:40,43`) | plain 1:1 — legal |

  Hidden-second-writer hunt (face (a)) run on all three CBs: **no** kernel raw-writes a CB it does not FIFO-produce. There is no `get_write_ptr()` / `fifo_wr_ptr` write anywhere in the four kernels, and no semaphore exists to coordinate one. Multi-reader hunt (face (b)): the only non-FIFO accesses are the compute kernels' `copy_tile(cb, …)` reads of `c_0`/`c_1` and `pack_tile(0, cb)` write of `c_2` — all by the kernel that already holds the matching binding, so each is a peek on an existing endpoint, not a new toucher. Dual-instance work-split (face (c)): absent — each of the three `kernel_source` values is pushed into exactly one `KernelDescriptor` (`:161-163`); no source is instantiated twice.

  No borrowed-memory (buffer-backed) CB exists: no `set_globally_allocated_address`, and no `CBDescriptor` carries a `buffer` — the three CBs are plain L1 FIFOs. No dead CB: every `buffer_index` (0, 1, 2) is referenced by at least two kernels.

- **Offset base pointers:** **GREEN.** The op is not in the `2026-07-19_offset_base_pointers.md` triage tables, *and* an independent scan of every pointer-bearing arg confirms no fold. There is **no `->address()` call anywhere in the op** — the factory passes `Buffer*` objects directly into the runtime-arg lists (`gelu_backward_program_factory.cpp:152,156`, sourced from the bare `grad_output.buffer()` / `input.buffer()` / `output.buffer()` @ `:79-81`), with no arithmetic at any point. So no Type 1 (raw offset arg) and no Type 2 (accessor-fed offset arg). Type 3 (`address_offset`) is N/A per the Appendix A row above; Type 4 (`ttnn::narrow`) does not appear. Every address arg reaches TensorParameter analysis as a clean base.

- **TensorAccessor 3rd argument:** **GREEN.** The op is not in the `2026-07-06_tensor_accessor_3rd_arg_triage.md` table, and an independent scan of all four kernels finds **three** `TensorAccessor` constructions, all **two-argument**:
  - `reader_binary_interleaved_start_id.cpp:46` — `TensorAccessor(src0_args, src0_addr)`
  - `reader_binary_interleaved_start_id.cpp:53` — `TensorAccessor(src1_args, src1_addr)`
  - `writer_unary_interleaved_start_id.cpp:31` — `TensorAccessor(dst_args, dst_addr)`

  No explicit page size is passed at any site, so the subject does not fire — there is nothing to classify and nothing to drop. (For completeness: `src0_tile_bytes` / `src1_tile_bytes` / `page_bytes` in those kernels are the *transfer size* argument to `noc.async_read` / `noc.async_write`, not the accessor's optional 3rd argument.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, all Case 1 — express as `TensorParameter` / `TensorBinding`, kernel builds `TensorAccessor(tensor::name)`):
  - `grad_output` — **Case 1**. Host: `Buffer*` pushed as reader RTA slot 0 (`gelu_backward_program_factory.cpp:152`, `src0_buffer` from `:79`) — the `Buffer*`-binding form, which the framework auto-registers as a `BufferBinding` and patches on cache hit, so this is *routine port work, not a correctness hazard*. Kernel: `src0_addr = get_arg_val<uint32_t>(0)` → `TensorAccessor(src0_args, src0_addr)` (`reader_binary_interleaved_start_id.cpp:17,46`). The CTA-appended `TensorAccessorArgs(*src0_buffer)` (`:85`) disappears with the port.
  - `input` — **Case 1**. Host: `Buffer*` as reader RTA slot 1 (`:152`, `src1_buffer` from `:80`). Kernel: `src1_addr = get_arg_val<uint32_t>(1)` → `TensorAccessor(src1_args, src1_addr)` (`reader…:18,53`). CTA plumbing @ `:86`.
  - `output` — **Case 1**. Host: `Buffer*` as writer RTA slot 0 (`:156`, `dst_buffer` from `:81`). Kernel: `dst_addr = get_arg_val<uint32_t>(0)` → `TensorAccessor(dst_args, dst_addr)` (`writer…:11,31`). CTA plumbing @ `:99`.

  No Case 2 (raw-pointer) binding and no borrowed-memory-DFB "clean" binding exists — all three tensors are accessed exclusively through a `TensorAccessor`. This matches the sheet's `Smuggled pointer = no`.
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation = none`; no custom hash to reconcile against).
- **TensorAccessor 3rd arg:** none — no site passes one.
- **CB endpoints:** all legal. `c_0`, `c_1`, `c_2` each bind one PRODUCER and one CONSUMER exactly as the legacy topology dictates.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. The hidden-second-writer and multi-reader hunts both came back empty (see Gate detail).
- **Cross-op / shared kernels:** both dataflow kernels are borrowed and **broadly shared**, and **neither has a `_metal2` fork yet** — this port creates the first one for each, beside the original (rung 2 of *Caution: Porting a shared kernel*). A tree-wide check confirms **zero** `_metal2` files exist outside `experimental/quasar/**`, so there is nothing to reuse anywhere.
  - `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` — no sibling fork. Other binders (**sunset list**, verified by path): `eltwise/unary_backward/gelu_bw`, `eltwise/unary_backward/tanh_bw`, plus `tests/ttnn/unit_tests/gtests/test_generic_op.cpp`. Note the file's own header comment (`:5-7`) calls it a temporary copy expected to be deleted or refactored — worth knowing before investing in the fork.
  - `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — no sibling fork. Very broadly shared: **~34** non-quasar factories bind this exact path (tilize ×7, tilize_with_val_padding ×3, reduction/generic ×4, reduction/prod ×2, transpose ×2, slice, concat, copy, permute, reshape_on_device, bcast, typecast, embedding, examples ×2, attn_matmul, nlp_concat_heads ×2, kv_cache, layernorm-adjacent, `gelu_bw`, `tanh_bw`, and this op). Count is approximate — derived by intersecting files containing the donor directory path with files naming the kernel. Beware same-named **private copies** under `matmul/`, `data_movement/slice/`, `point_to_point/`, and the layernorm families: those are different files and are *not* on this sunset list.
  - Because the borrowed forks will be named by *this* port and inherited by every later consumer, name their bindings from the **kernel's own** role vocabulary (`tensor::src0`/`src1`, `tensor::dst`, `dfb::in0`/`in1`, `dfb::out`), not from gelu-backward's locals.
  - **Negative pointer:** `ttnn/cpp/ttnn/operations/experimental/quasar/` holds a copy of the reader (`quasar/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`) and several `*_metal2.cpp` writer variants (`quasar/reduction/generic/…`, `quasar/tilize_with_val_padding/…`). That tree is out of bounds — do not bind them, do not count them as forks, do not copy their binding names.
  - The two **compute** kernels are op-owned and have **no** other consumer: `gelu_bw` binds its *own* same-named private copies under `eltwise/unary_backward/gelu_bw/device/kernels/compute/`, not these files. They port in place, no fork.
- **RTA varargs:** none. Every kernel reads its runtime args as distinct fields at fixed constant indices — reader `get_arg_val<uint32_t>(0…6)` (`:17-23`), writer `(0…2)` (`:11-13`), compute `(0)` (tanh `:24`, poly `:22`). No counted loop over args, no data-selected index. All args are nameable; the port should name each.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up: ✓ clean** (function-call escape). Every non-local `#include` in all four kernels resolves to `tt_metal/*` (donor class 1 — LLK / HAL / firmware, no concern): the DM kernels include `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h`; the compute kernels include `api/compute/*` and `api/dataflow/circular_buffer.h`. **No include resolves into `ttnn/cpp/ttnn/kernel_lib/`, `ttnn/cpp/ttnn/kernel/`, `operations/kernel_helper_functions/`, an in-family shared header, or a cross-family donor.** There is therefore no per-call shape analysis to perform — no donor function is called across an op boundary, so none of the ⚠/✗/⭐ signature shapes (`uint32_t sem_id`, `TensorAccessorArgs<N>`, Shape-3 NTTP, old-style addr-gen, `CircularBuffer&`) arises.
  - **Summary table:** empty — zero (op kernel, donor file) pairs outside `tt_metal/*`.
  - **Per-call detail:** omitted (all rolls ✓).
  - **Borrowed kernel files (file-path instantiation):** the two dataflow kernels above. Full detail — path, owning family, co-binder set, fork status — is in Heads-ups. This is a coordination/sunset cost, not a gate.
- **Relaxation candidates:** none. No custom hash exists to mine, and the sheet proposes `none`.
- **TTNN factory analysis:** `Concept = descriptor` → target `ProgramSpecFactoryConcept` (the sheet's own `Porting Target` column independently says `ProgramSpecFactoryConcept`; `Execution Model = SPMD`). No op-owned tensors (structurally impossible on a `descriptor` factory, and the code confirms — `create_descriptor` returns a bare `ProgramDescriptor`). No MeshWorkload need. No pybind of internals: `gelu_backward_nanobind.cpp` exposes only `ttnn.experimental.gelu_bw`. No custom hash, no backdoor custom hash, no `get_dynamic_runtime_args`, no `override_runtime_arguments`. The sheet flags `Formerly custom hashed? = yes` — a historical state; no hash override remains in the code today, so it does not bear on this port.

## Misc anomalies  *(team-only, non-gating)*

- **Three dead reader RTAs.** `gelu_backward_program_factory.cpp:152` passes `block_height = 0u`, `block_width = 0u` (slots 4, 5) and `num_cores_y` (slot 6). The reader consumes all three **only** inside `if constexpr (block_or_width_sharded)` (`reader_binary_interleaved_start_id.cpp:60-87`), and this factory hardcodes that CTA to `0` (`reader_compile_time_args = {0}` @ `:84`), so the branch is dead for this op and all three values are unused. Passing the real `num_cores_y` alongside two literal zeros is also misleading — it reads as if it mattered. Routes to the ops team. *Note for whoever acts on it: the port must **not** drop these — the kernel is shared and reads them unconditionally at the top of `kernel_main`.*
- **Dead sharded branch in `compute_output_specs`.** `gelu_backward_device_operation.cpp:109-111` selects `output_layout = tensor_args.input.layout()` when `args.output_memory_config.is_sharded()`, but `validate_on_program_cache_miss` rejects sharded input (`:50`) and requires input and output memory layouts to match (`:44-48`) with input `INTERLEAVED` (`:58-62`). The sharded branch is therefore unreachable.
- **Dead `DataType::INVALID` path.** `GeluBackwardParams::output_dtype` defaults to `INVALID` and both the validator (`:20-22`) and `compute_output_specs` (`:113-116`) handle it, but the only caller — `gelu_backward.cpp:16` — always sets `output_dtype = input_tensor.dtype()`. Combined with the validator's hard requirement that output dtype equal input dtype (`:29-33`), the attribute carries no information yet still participates in the program-cache key.
- **Near-duplicate compute kernel across two ops.** `device/kernels/compute/eltwise_bw_gelu_poly.cpp` and `eltwise/unary_backward/gelu_bw/device/kernels/compute/eltwise_bw_gelu_poly.cpp` differ only in the SPDX line and three comment lines — the code is identical. Two op directories carrying the same kernel under the same filename is a maintenance trap (a fix to one silently misses the other), and it is exactly the shape that makes a filename-based shared-kernel census dangerous. Not port work; flagged for the ops team.
- **Redundant `tanh_tile_init()`.** Called once during setup (`eltwise_bw_gelu_approx_tanh.cpp:41`) and again inside the per-tile loop (`:72`). Harmless but likely unintended.

## Recipe notes

1. **Readiness-sheet column rename breaks the doc's stability guarantee.** `analyses/ttnn_op_porting_readiness.md:57` and the audit's derivation formula both name the column **`Override runtime args method? (PD and legacy)`**; the live sheet's header (fetched 2026-07-31) reads **`Override runtime args method?\n(PD only)`**. The readiness doc states as a standing guarantee that *"existing column names never change, and no column is ever deleted"* — this rename contradicts it. Harmless here (the value is `no` either way), but a strict name-match lookup would have missed the column entirely. Suggest either updating both docs to the current header or softening the guarantee to "columns are renamed rarely; match on the leading phrase."

2. **The live sheet carries gate-adjacent columns the derivation formula doesn't mention.** The current header includes `Backdoor custom hash (attribute_values / to_hash)`, `Formerly custom hashed?`, `Pointer patching perf issue?`, `Known op issues`, `Execution Model`, and `Porting Target`. `Backdoor custom hash` in particular looks like it belongs in the `Is able to port?` conjunction next to `Custom hash` — the audit's *TTNN factory concept prerequisite* formula lists only `Custom hash`, so an auditor can't tell whether the backdoor column is already folded into `Is able to port?` or is an independent signal they should be checking. All were `no`/benign for this op, so nothing turned on it. Please state each new column's gate status.

3. **`Porting Target` duplicates the *TTNN porting shape* subject's whole job.** The sheet now supplies the target concept directly (`ProgramSpecFactoryConcept` here), which is exactly what that subject derives by hand from `Concept` + `Op-owned tensors?`. Worth saying whether the sheet's cell is now authoritative (subject becomes a lookup + confirm) or whether the hand derivation remains the source of truth and the column is a cross-check.

4. **The Device 2.0 gate has no explicit ruling on compute-LLK CB-index free functions.** The Green bullet's sanctioned list names exactly two functions, both dataflow-side (`get_tile_size`, `get_local_cb_interface`). This op's compute kernels call `copy_tile(cb_id, …)`, `pack_tile(0, cb_id)`, and `unary_op_init_common(cb_id, cb_id)` — same *shape* as a holdover (single CB-index argument, a `CircularBuffer` object in scope at the call site). I resolved it GREEN via the bullet's second condition (*"and a wrapper-method replacement exists"* — `circular_buffer.h` has no such member), and via the gate being scoped to *Data Movement* migration. But that took reading the wrapper's header to settle, and a hurried auditor could plausibly RED a perfectly clean op over `pack_tile`. Suggest one sentence in the Green bullet: *compute LLK APIs (`copy_tile`, `pack_tile`, `*_init*`, …) take CB indices by design, are outside the Device 2.0 data-movement boundary, and are never holdovers.*

5. **`DataflowBuffer`-with-raw-index isn't in the Device 2.0 recognition vocabulary.** Both borrowed DM kernels use `DataflowBuffer dfb(cb_id)` (from `api/dataflow/dataflow_buffer.h`) rather than the Device 2.0 `CircularBuffer` wrapper that `device_api_migration_guide.md` prescribes. This is *ahead* of Device 2.0, not behind it — but the audit's Device 2.0 subject and the *CB endpoints* precondition (*"the recognition signals assume Device-2.0 kernel idioms — `get_write_ptr` methods, `get_local_cb_interface`, `Semaphore` objects"*) both describe compliance in `CircularBuffer` terms only. An auditor pattern-matching literally could get stuck. Suggest naming `DataflowBuffer` as an accepted (indeed preferred) Device-2.0-or-better idiom in both places.

6. **Provenance command assumes the docs and the op live in one checkout.** `git log -1 … -- docs/source/.../metal_2.0/` printed nothing from the op's working checkout because the metal_2.0 doc tree isn't in it — the recipe was read from a separate doc-branch checkout. The recipe's fallback (*"record that instead, since the version can't be pinned"*) would have discarded a hash that was in fact available a directory over. Suggest: *if the docs live in a different checkout, run the command there and say so* — which is what this report does.

7. **Minor, working-as-designed:** a stale `analyses/ttnn_op_porting_readiness.csv` was sitting in the doc checkout with a materially different column set (24 columns live vs. the stale copy's fewer, and a different `Op Classification` spelling). The refetch-every-run rule caught it exactly as intended; noting only as confirmation that the rule earns its keep.
