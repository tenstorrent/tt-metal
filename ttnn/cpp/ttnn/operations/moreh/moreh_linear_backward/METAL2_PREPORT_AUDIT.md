# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward`

One device operation lives in this directory. The op's *other* two gradients are not device
operations at all — `ttnn::moreh_linear_backward` (`moreh_linear_backward.cpp:138-167`) composes them
out of `ttnn::moreh_matmul` and `ttnn::moreh_sum`, which are separate ops in their own directories
and out of scope here.

- **`MorehBiasAddBackwardOperation`** (the bias-gradient path only; `ttnn::prim::moreh_bias_add_backward`)
  - `SingleCoreProgramFactory` (`device/moreh_linear_backward_single_core_program_factory.cpp`) — selected when `bias_grad` is a scalar
  - `MultiCoreProgramFactory` (`device/moreh_linear_backward_multi_core_program_factory.cpp`) — selected otherwise (1-D `bias_grad`)

  Selection: `select_program_factory` (`device/moreh_linear_backward_device_operation.cpp:28-35`), on `is_scalar(bias_grad)`.

Kernels (all five live in this op's own directory; none is borrowed, none is lent):

| Kernel | Bound by |
|---|---|
| `device/kernels/reader_moreh_bias_backward_hw.cpp` | SingleCore |
| `device/kernels/moreh_bias_backward_single_core_hw.cpp` | SingleCore |
| `device/kernels/reader_moreh_bias_backward_h.cpp` | MultiCore |
| `device/kernels/moreh_bias_backward_multi_core_h.cpp` | MultiCore ×2 (per core group) |
| `device/kernels/writer_moreh_bias_backward.cpp` | **both factories** |

No unreferenced kernel files. No semaphores in either factory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `9440205cf62 2026-08-19 docs(metal_2.0): have the porter prove the legality checks are running`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehBiasAddBackwardOperation` → `SingleCoreProgramFactory`, `MultiCoreProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all five kernels structurally Device 2.0; donors Device-2.0 native |
| *Prereqs* — Cross-op escapes | Ok — every donor signature is ✓; no borrowed/lent kernel files |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — Variadic-CTA | Ok — no kernel reads any compile-time arg |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factory rows); cross-check clean |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — concept is `descriptor`, not `WorkloadDescriptor` |
| *TTNN Readiness* — Custom hash | No — default reflection-based hash; no backdoor `attribute_values` / `to_hash` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `moreh_linear_backward_nanobind.cpp` binds only the user-facing function |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (both factories) |
| *Port work* — Offset base pointer | none — cleared |
| *Port work* — Tensor bindings (per binding) | `output_grad` **Case 1**; `bias_grad` **Case 1** (identical in both factories) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **N/A** — no accessor in the op passes a 3rd argument |
| *Port work* — CB endpoints | 4× legal 1:1 · 2× compute self-loop · 1× conditional DFB (SingleCore `c_2`) |

**CB endpoints** are dispositions, not gates: every out-of-window CB here has a port-time resolution
(two compute self-loops, one conditional DFB). Nothing in this subject blocks a Gen1 port. Recorded
per `(CB, factory)` below; both factories were classified separately.

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear: Device 2.0, Feature compatibility,
TTNN factory concept, Offset base pointers, TensorAccessor 3rd argument. `METAL2_PORT_BRIEF.md` is
written alongside this file.

The port is small and self-contained: two factories (200 and 264 lines), five kernels, no semaphores,
no borrowed memory, no op-owned tensors, no custom hash. **Both factories should be ported in one
change** — they share `writer_moreh_bias_backward.cpp`, so converting one alone breaks the other (see
[Out-of-directory coupling](#out-of-directory-coupling--shared-kernels)).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet carries two rows for
  this op, one per factory, both `Is able to port? = yes`, both `Concept = descriptor`,
  `Porting Target = ProgramSpecFactoryConcept`, `Execution Model = SPMD`,
  `TensorParameter relaxation = none`, `Known op issues` empty, `Diego validation = yes`.
  `Op Classification = PD Op (pointer-patching)` with `Pointer patching perf issue? = OK` and
  `Smuggled pointer = no` — consistent with what the code does (see Tensor bindings below).

  Lightweight cross-check against the code — clean on every checkable column:

  | Column | Sheet | Code | Agree |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor()` returning `ProgramDescriptor` (`device/moreh_linear_backward_device_operation.hpp:33,40`) | ✓ |
  | `Custom hash` | `no` | no `compute_program_hash` anywhere in the op dir | ✓ |
  | Backdoor custom hash | `no` | no `attribute_values` / `to_hash` | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | hook absent from the device op | ✓ |
  | `Override runtime args method?` | `no` | no `override_runtime_arguments` | ✓ |
  | `Pybind descriptor` | `no` | `moreh_linear_backward_nanobind.cpp:18-35` binds only `ttnn::moreh_linear_backward`; no `nb::class_` of the device op, no `create_descriptor` binding | ✓ |
  | `Smuggled pointer` | `no` | RTAs carry `Buffer*` objects, which the framework auto-registers and patches — not an un-annotated raw address | ✓ |
  | `Op-owned tensors?` | (blank) | factories allocate no device tensors of their own | ✓ |

  **Factory-set match:** the sheet's two factory rows correspond one-to-one with the code's two
  factories (`SingleCoreProgramFactory`, `MultiCoreProgramFactory`) in the `program_factory_t` variant
  (`device/moreh_linear_backward_device_operation.hpp:46`). No phantom row, no missing row.

  **Cross-column invariants:** `get_dynamic_runtime_args == no` on a `descriptor` concept — consistent.
  `Op-owned tensors?` empty on a `descriptor` concept — consistent (the `descriptor` form cannot carry
  op-owned tensors).

- **Device 2.0 (every kernel used):** **GREEN.** All five kernels are structurally Device 2.0 — `Noc`
  with `noc.async_read` / `noc.async_write` and object barriers, `DataflowBuffer` FIFO methods, and
  `TensorAccessor`. Scans for Device 1.0 idioms (`InterleavedAddrGen`, `ShardedAddrGen`,
  `InterleavedPow2AddrGen*`, bare `noc_async_read(` / `noc_async_write(`, `noc_semaphore*`,
  `get_noc_addr(`, `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`) return **zero
  hits over 5 kernel files**. No cursor surgery (`evil_set*`, `fifo_wr_ptr` / `fifo_rd_ptr`,
  `tiles_acked` / `tiles_received`) and no `pages_reservable` / `pages_available`.

  Three CB-index free-function call sites exist and are **not** holdovers — `get_tile_size(cb_id)` is
  explicitly **sanctioned** by the Device 2.0 surface:

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/reader_moreh_bias_backward_hw.cpp` | 43 | `get_tile_size(cb_id_in0)` | `dfb_in0` (`DataflowBuffer`) |
  | `device/kernels/reader_moreh_bias_backward_h.cpp` | 39 | `get_tile_size(cb_id_in0)` | `dfb_in0` (`DataflowBuffer`) |
  | `device/kernels/writer_moreh_bias_backward.cpp` | 24 | `get_tile_size(cb_id_out)` | `dfb_out` (`DataflowBuffer`) |

  Per the Green bullet these stay sanctioned regardless of what object is in scope, so a
  `DataflowBuffer` being present at the call site does **not** make them violations. They move onto the
  object at *port* time (kernel-side whitelist rule 7), which is a Metal 2.0 change, not a Device 2.0
  one. **No routing to the Device 2.0 track.**

  Donor kernels the op calls into are Device-2.0 native as well — see
  [Out-of-directory coupling](#out-of-directory-coupling--shared-kernels).

- **Feature compatibility:** every Appendix A entry, in order. All absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `.global_circular_buffer` field on any `CBDescriptor`, no `global_cb` parameter, no `remote_cb_*` / `remote_index()` / `remote_circular_buffer.h` idiom. All 12 `CBDescriptor`s (6 per factory) are plain. |
  | CBDescriptor `address_offset` (non-zero) | **N/A** | No `.address_offset`, no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. No borrowed-memory CBs at all (no `set_globally_allocated_address`, no `CBDescriptor::buffer` set). |
  | GlobalSemaphore | **N/A** | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`, no `global_semaphore.hpp` include. The op declares **no semaphores of any kind** — `grep -i semaphore` over the whole op directory returns zero hits. |

  A clean scan is all-`N/A`; the subject verdict is GREEN — no gate fired.

- **CB endpoints (GATE-free):** classified per `(CB, factory)`. **No dead CB, no multi-binding
  anywhere, and no `allow_instance_multi_binding` needed.** Census below; endpoint counts are per node.

  **`MultiCoreProgramFactory`** (`c_0..c_25`, all over `all_cores`):

  | CB | Role | Touchers on a node | Verdict |
  |---|---|---|---|
  | `c_0` | output_grad in | reader (FIFO produce, `reader_..._h.cpp:45,48`) + compute (FIFO consume — `pop_front` at `..._multi_core_h.cpp:88` and inside `compute_kernel_lib::reduce<…, cb_in0, …>`) | **legal 1:1** |
  | `c_1` | scaler | reader (produces via `calculate_and_prepare_reduce_scaler<cb_id_scaler,…>`, `reader_..._h.cpp:27-28`) + compute (`wait_front`, `..._multi_core_h.cpp:32`, never popped) | **legal 1:1** |
  | `c_2` | mask_h_w | reader (`generate_mask_h_w`, `reader_..._h.cpp:31-32`) + compute (`wait_front` + `copy_tile`, `..._multi_core_h.cpp:35,63,72`) | **legal 1:1** |
  | `c_16` | bias_grad out | compute (produced as the `reduce<…, cb_out0>` output) + writer (`wait_front` / `pop_front`, `writer_...cpp:28,31`) | **legal 1:1** |
  | `c_24` | intermed0 | compute **only** — `reserve_back` / `push_back` (`..._multi_core_h.cpp:80,85`) *and* consumed as the `reduce<…, cb_intermed0, …>` input | **1 toucher → compute self-loop** |
  | `c_25` | intermed1 (accumulator) | compute **only** — written as the `reduce` output and read back via `Accumulate::at(cb_intermed1, …)` (`..._multi_core_h.cpp:93,99,107`) | **1 toucher → compute self-loop** |

  **`SingleCoreProgramFactory`** — same six CBs, same dispositions, with one difference:

  - `c_2` is allocated **conditionally**, behind `if (in2_t > 0)` where
    `in2_t = (do_mask_h || do_mask_w) ? 2 : 0`
    (`..._single_core_program_factory.cpp:43,86-96`). `do_mask_h` / `do_mask_w` are host-time values
    derived from `output_grad.logical_shape()`, so the condition is known when the spec is built. →
    **conditional DFB**, not a drop: live when either mask applies, absent otherwise.

  Both compute self-loops are the **legitimate accumulator/staging** case: they are supported on Gen2
  as well as Gen1 (the DM-self-loop restriction does not apply), so they carry no Quasar debt.

- **Offset base pointers:** **GREEN — cleared.** Every address argument in both factories is a clean
  base. The factories push the `Buffer*` **object** into the runtime-arg list —
  `reader_desc.emplace_runtime_args(core, {output_grad_buf, …})` and
  `writer_desc.emplace_runtime_args(core, {bias_grad_buf, …})`
  (`..._multi_core_program_factory.cpp:217-229`; `..._single_core_program_factory.cpp:180-189`) — with
  **no host arithmetic folded in**: no `->address() + <expr>` anywhere in the op. Kernel-side, each base
  goes straight into a `TensorAccessor` constructor and is never used as a raw NoC address.
  `moreh_linear_backward` does not appear in the Type-1 or Type-2 tables of
  `2026-07-19_offset_base_pointers.md`, and the scan agrees with that silence — the *"no fold, op not in
  the tables"* outcome. No Type 3 (`address_offset` absent) and no Type 4 (`ttnn::narrow` absent).

- **TensorAccessor 3rd argument:** **N/A — no accessor in the op passes a 3rd argument.** All three
  construction sites are 2-arg: `TensorAccessor(src_args, src_addr)`
  (`reader_..._hw.cpp:39`), `TensorAccessor(src0_args, src0_addr)` (`reader_..._h.cpp:35`),
  `TensorAccessor(dst_args, dst_addr)` (`writer_...cpp:20`). The subject never fires, so there is
  nothing to classify — this is *no sites*, not *sites found and judged redundant*.
  `moreh_linear_backward` is absent from `2026-07-06_tensor_accessor_3rd_arg_triage.md`, consistent.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, identical in both factories):
  - `output_grad` — **Case 1** (via `TensorAccessor`). Delivered today by the `Buffer*`-binding form
    (`emplace_runtime_args(core, {output_grad_buf, …})`); the kernel feeds the resulting `uint32_t`
    base into `TensorAccessor(src_args, src_addr)` and accesses only through the accessor
    (`{.page_id = …}`). Becomes a `TensorParameter` + `TensorBinding`.
  - `bias_grad` — **Case 1**, same shape on the output side (`TensorAccessor(dst_args, dst_addr)`).
  - `bias` is **not** a binding: `tensor_args.bias` is read only on the host, for the output spec
    (`device/moreh_linear_backward_device_operation.cpp:47-51`). No kernel touches it.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none — subject N/A.
- **CB endpoints:** self-loop `c_24` and `c_25` (compute, both factories) · make `c_2`'s
  `DataflowBufferSpec` **conditional** in `SingleCoreProgramFactory` (live when `do_mask_h || do_mask_w`)
  · all four remaining CBs are legal 1:1 in both factories · no dead CB, no multi-binding flag.
- **`unpack_modes` — the two factories differ, deliberately.** Detail in the brief; this is the
  highest-risk item in the port.
- **Compute `opt_level`:** no `opt_level` is set anywhere in the op (`grep -n opt_level` → zero hits),
  so all three compute descriptors resolve to the legacy `ComputeConfigDescriptor` default **O3**.
  Metal 2.0's `CompilerOptions` defaults to O2, so every compute `KernelSpec` needs an explicit
  `KernelBuildOptLevel::O3`. The DM kernels need nothing (legacy O2 = Metal 2.0 O2).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. The hidden-second-writer and multi-reader
  faces were hunted and did not fire: no kernel takes a raw pointer into a CB another kernel writes,
  no CB has two FIFO producers or two FIFO consumers, and neither factory uses the dual-instance
  work-split shape (the multi-core factory's two compute descriptors cover **disjoint** core groups,
  so each node sees exactly one compute instance).
- **Shared kernel:** `writer_moreh_bias_backward.cpp` is bound by **both** factories — an *intra-op*
  shared kernel. Nothing outside this op binds it, and no `_metal2` fork exists.
- **RTA varargs:** none needed. Every kernel walks a **fixed** set of distinct fields via `ArgFetcher`.
- **Per-group compute multiplicity** in `MultiCoreProgramFactory` must be preserved (two `KernelSpec`s,
  two `WorkUnitSpec`s) — and its per-group CTA is **dead**, which makes the shape easy to mis-simplify.

## Team-only

### Out-of-directory coupling & shared kernels

**Op-level roll-up: ✓ clean.** Every donor function the op's kernels call has a Metal-2.0-crossable
signature. No donor needs a change, and no donor blocks the port.

Summary table — one row per (op kernel, donor file):

| Op kernel | Donor file | Class | Roll-up |
|---|---|---|---|
| `reader_..._hw.cpp`, `reader_..._h.cpp`, `writer_...cpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 3 — second shared-kernel pool | ✓ |
| `reader_..._h.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | 2 — official shared kernel library | ✓ |
| `moreh_bias_backward_single_core_hw.cpp`, `moreh_bias_backward_multi_core_h.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | 2 — official shared kernel library | ✓ |
| `moreh_bias_backward_single_core_hw.cpp`, `moreh_bias_backward_multi_core_h.cpp` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | 3 — second shared-kernel pool | ✓ |
| all five | `tt_metal/hw/inc/api/**` (`noc.h`, `dataflow_buffer.h`, `noc_traits.h`, `api/compute/*` LLKs) | 1 — LLK / HAL / firmware | ✓ no concern |

Per-call detail (all ✓, so this section is informational rather than a blocker list):

| Donor function | Signature shape | Status |
|---|---|---|
| `fill_cb_with_value(DataflowBuffer cb, uint32_t value, int32_t = 1024)` — `moreh_common.hpp:98` | `DataflowBuffer` **by value** | ✓ excellent — donor already migrated to DFB. Porter builds a named local from `dfb::scaler` and passes it. |
| `generate_mask_h_w(DataflowBuffer cb_mask_h_w, uint32_t, uint32_t, uint32_t = 2048)` — `moreh_common.hpp:262` | `DataflowBuffer` by value | ✓ excellent |
| `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<uint32_t dfb_id, PoolType, ReduceDim, reduce_factor>` — `reduce_helpers_dataflow.hpp:83` | DFB id as a `uint32_t` **NTTP** | ✓ OK — `dfb::name`'s conversion is `constexpr`, valid in template-argument position. The parameter is already *named* `dfb_id`. |
| `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, in, scaler, out>(…)` — `reduce_helpers_compute.hpp:392` | three CB ids as `uint32_t` NTTPs | ✓ OK |
| `compute_kernel_lib::Accumulate::at(uint32_t cb, uint32_t iter, uint32_t dst = 0)` — `reduce_helpers_compute.hpp:193` | CB id as a runtime `uint32_t` argument | ✓ OK |
| `ArgFetcher` — `moreh_common.hpp:44` (dataflow) / `:128` (compute) | no resource handle; wraps `get_arg_val<T>(arg_idx++)` | ✓ not a boundary case. It **disappears** from the ported kernels as every read becomes `get_arg(args::name)`; the donor header is untouched. |
| LLKs: `copy_tile`, `copy_tile_to_dst_init_short`, `mask_tile`, `mask_tile_init`, `pack_tile`, `reconfig_data_format_srca`, `pack_reconfig_data_format`, `compute_kernel_hw_startup` | `uint32_t` CB ids | ✓ OK — implicit `DFBAccessor → uint32_t` conversion |

**No `sem::` or `tensor::` handle ever crosses a donor boundary.** The op has no semaphores at all, and
every `TensorAccessor` is constructed and consumed inside the op's own kernels. The recipe's
boundary-rule assumption holds with room to spare — no assumption-violation risk in this port.

**Borrowed kernel files: none.** All five kernel `.cpp` files live in this op's own `device/kernels/`.

**Lent kernel files: none.** `grep -rln <filename> ttnn/ tests/ models/` for each of the five returns
only this op's own two factories (plus `ttnn.egg-info/SOURCES.txt`, a packaging manifest, not a
consumer). Nothing outside `moreh_linear_backward` binds any of them, so this port creates no
cross-op coordination cost and no sunset list.

**Intra-op sharing — the one real coupling.** `writer_moreh_bias_backward.cpp` is bound by *both*
factories (`..._single_core_program_factory.cpp:143`, `..._multi_core_program_factory.cpp:147`). No
`_metal2` sibling exists in `device/kernels/`. Porting one factory alone would Metal-2.0-ify that
writer and break the other factory, so the port either co-converts both factories (recommended — the
op is small) or lands a `writer_moreh_bias_backward_metal2.cpp` fork. Recommendation and rungs are in
the brief.

### Relaxation candidates

None. The op has no custom `compute_program_hash`, so there is no hash to mine for the tensor
properties the op actually depends on. `TensorParameter relaxation = none` on both sheet rows, and
nothing in the code suggests otherwise.

### TTNN factory analysis

Sheet-derived facts, with `file:line` evidence:

- **Op-owned tensors:** none. Neither factory allocates a device tensor of its own; the only tensor the
  op creates is the output, and it is created by the framework hook
  `create_output_tensors` (`device/moreh_linear_backward_device_operation.cpp:54-62`) — which is the
  op's declared return value, not an op-owned workspace.
- **MeshWorkload need:** none. `Concept = descriptor` (not `WorkloadDescriptor`), so the
  secretly-SPMD question does not arise; `Execution Model = SPMD` on both rows.
- **Pybind `create_descriptor`:** absent. `moreh_linear_backward_nanobind.cpp:18-35` binds only
  `ttnn::moreh_linear_backward` through `ttnn::bind_function`. **No pybind deletion is forced by this
  port** — device-op-class exception 1 does not apply.
- **Other risky pybind:** none. No `nb::class_` of the device operation or its factories.
- **Custom hash:** none (default reflection-based hash). No backdoor `attribute_values` / `to_hash`.
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** absent — which is what selects the **base**
  `ProgramSpecFactoryConcept` rather than `CustomProgramSpecFactoryConcept`.
- **Direct-descriptor shape:** does **not** apply. The op already has a `program_factory_t` variant
  (`device/moreh_linear_backward_device_operation.hpp:46`) with both factories as nested structs, so
  device-op-class exception 3 is not triggered; the port is a method swap inside the existing structs.
- **Target concept:** `ProgramSpecFactoryConcept`, matching the sheet's `Porting Target` column.

**Consequence: the port forces no device-operation-class edit at all**, beyond the two
`create_descriptor` → `create_program_artifacts` signature changes in the header. None of the three
sanctioned exceptions applies.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

1. **Dead compile-time arg on both multi-core compute descriptors.**
   `..._multi_core_program_factory.cpp:171` and `:188` set
   `compile_time_args = {num_cols_per_core_group_N}`, but `moreh_bias_backward_multi_core_h.cpp`
   reads **no** compile-time argument at all — `grep -n 'get_compile_time_arg_val'` over all five
   kernels returns zero hits. The per-core count the kernel actually uses arrives as RTA slot 2
   (`Wt_per_core`, `..._multi_core_h.cpp:14`), fed from the same
   `num_cols_per_core` value (`:236`, `:245`). So the value is passed twice and the CTA copy is unread.
   Dropping it would collapse the two compute descriptors into one, which is a structural change and
   an owner decision — not port work. *(The port preserves it verbatim; see the brief.)*

2. **Misleading kernel-side variable name: `batch_num` holds `batch_num * Ht`.**
   The multi-core factory passes `num_tiles = batch_num * Ht` (`..._multi_core_program_factory.cpp:40`)
   as reader RTA slot 1, and the reader unpacks it into a local named `batch_num`
   (`reader_..._h.cpp:13`), then loops `for (b = 0; b < batch_num; ++b)` striding by `Wt`
   (`:44-50`). The *value* is correct — a full column spans `batch_num * Ht` tiles — but the name says
   something the value is not. The same misnomer appears in the multi-core compute kernel
   (`..._multi_core_h.cpp:12`, then `num_tiles = batch_num * Ht` at `:38`, i.e. `batch_num * Ht * Ht`
   by the name's own reading). Not a bug; a naming wart that will now be baked into a *named* Metal 2.0
   argument. Suggest the ops team rename the kernel-side locals.

3. **The single-core factory's `dfb_mask_h_w_obj` is constructed on a CB that may not exist.**
   `moreh_bias_backward_single_core_hw.cpp:21-22` constructs `DataflowBuffer dfb_mask_h_w_obj(cb_mask_h_w)`
   **unconditionally**, but `SingleCoreProgramFactory` allocates CB index 2 only when
   `do_mask_h || do_mask_w` (`..._single_core_program_factory.cpp:43,86-96`). On Gen1 this is benign —
   constructing the wrapper only computes an L1 interface address and the object is never touched on
   the no-mask path — but it is a latent wart, and it is precisely why the Metal 2.0 port needs the
   conditional-binding pattern there (a token for an undeclared DFB does not exist).

4. **The multi-core factory allocates the mask CB unconditionally.**
   `..._multi_core_program_factory.cpp:64` sets `const uint32_t in2_t = 2;` with no
   `do_mask_h || do_mask_w` guard, so two tiles of L1 for `c_2` are reserved on every core even when no
   masking is needed — unlike the single-core factory, which guards the same allocation. Wasted L1 in
   the common aligned-shape case. The port preserves this faithfully (an unconditional DFB in the
   multi-core factory, conditional in the single-core one); tightening it is an owner decision.

## Per-DeviceOperation attribution

Not applicable — the directory holds a single device operation, `MorehBiasAddBackwardOperation`. All
findings above are already attributed per factory where the two differ (`c_2` conditionality,
`unpack_modes`, kernel set).

## Questions for the user

None. Every gate resolved from the code plus the supplied sheet rows; no ambiguity required a
conservative call.

## Recipe notes

1. **The audit has no subject for a composite op whose siblings are other ops.**
   `moreh_linear_backward` computes three gradients, but only the bias one is a device operation; the
   input- and weight-gradient paths call `ttnn::moreh_matmul` / `ttnn::moreh_sum` from op-level host
   code (`moreh_linear_backward.cpp:138-167`). The *"Multiple device-operations in one op directory"*
   rule covers the opposite shape (several DOps in one directory) and the *"Resolving the op you were
   given"* step only confirms the directory *is* an op. Neither tells you what to do when the op name
   you were handed is a composite whose other limbs are separately-owned ops. I treated them as out of
   scope and said so in the identifying section, which seems clearly right — but a one-line rule would
   save the next auditor the judgement call, and would also prompt them to note (as I do here) that
   both delegates are *already* ported to `ProgramSpecFactoryConcept` on `main`, so the composite's
   full gradient path lands entirely on Metal 2.0 once this op ports.

2. **`Op Classification = "PD Op (pointer-patching)"` is not in the audit's column vocabulary.**
   The *TTNN factory concept prerequisite* subject in `metal2_audit.md` describes `Op Classification` only as *"a derived summary of an op's overall state,
   including whether it reads as broken"*, and the blocking table keys on a *smuggled* RTA pointer.
   This op reads `PD Op (pointer-patching)` with `Smuggled pointer = no` and `Pointer patching perf
   issue? = OK`. I read that as the benign `Buffer*`-binding form (which the TensorParameter-analysis
   subject documents well as framework-patched and *not* the silent-wrong hazard), and the code agrees.
   Worth stating the mapping explicitly — `PD Op (pointer-patching)` + `Smuggled pointer = no`
   ⇒ `Buffer*`-binding form ⇒ enumerate and classify Case 1/2, do not gate — because the phrase
   "pointer-patching" reads alarming next to a subject whose whole point is that a *different* pointer
   pattern is a correctness hazard.

3. **Minor: the sanctioned-free-function list and the port's rule 7 pull in opposite directions on the
   same three lines, and the audit says so well.** Recording only that the Green bullet's *"the list is
   the whole test, and it does not turn on what object is in scope"* paragraph was exactly what I
   needed — a `DataflowBuffer` is in scope at all three `get_tile_size(cb_id)` sites here, which is the
   configuration the doc warns misfires hardest. It fired correctly. No change requested.
