# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed`

Two DeviceOperations share this directory. They are audited **together** (one combined report) because they share a kernel file — `device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp` is instantiated by all five factories — and because both are reached through the same nanobind module and the same `LayerNormProgramConfig`. Findings that differ between the two are attributed per DeviceOperation throughout, and summarised in *Per-DeviceOperation attribution*.

- **`LayerNormPreAllGatherDeviceOperation`** (`device/layernorm_pre_all_gather_device_operation.{hpp,cpp}`)
  - `LayerNormPreAllGatherProgramFactory` (`layernorm_pre_all_gather_program_factory.cpp:25`) — 1D work split, non-Welford
  - `LayerNormPreAllGather2DProgramFactory` (`layernorm_pre_all_gather_program_factory.cpp:295`) — 2D core grid, cross-core merge
  - `LayerNormPreAllGatherWelfordProgramFactory` (`layernorm_pre_all_gather_welford_program_factory.cpp:23`)
- **`LayerNormPostAllGatherDeviceOperation`** (`device/layernorm_post_all_gather_device_operation.{hpp,cpp}`)
  - `LayerNormPostAllGatherProgramFactory` (`layernorm_post_all_gather_program_factory.cpp:29`) — carries **two configs** in one factory: 1D split and `use_2d_core_grid` 2D split
  - `LayerNormPostAllGatherWelfordProgramFactory` (`layernorm_post_all_gather_welford_program_factory.cpp:44`)

**Also in scope (kernels referenced from outside the directory).** Both post factories and the 1D pre factory file-path-instantiate compute kernels owned by the sibling family `normalization/rmsnorm_distributed` on their `is_rmsnorm` branch:
`rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp` and `.../rmsnorm_post_allgather.cpp`. These are reachable: `ttnn::rms_norm_pre_all_gather` / `rms_norm_post_all_gather` call straight into these prim device ops (`rmsnorm_distributed/rmsnorm_post_all_gather.cpp:43`, `rmsnorm_distributed/rmsnorm_pre_all_gather.cpp:39`).

**Unreferenced / not audited.** Every kernel file in this op's `device/kernels/` tree is referenced by some factory; none are dead. For context: `rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather_2d.cpp` exists in the sibling family but is instantiated by **no** factory anywhere, so its contents were not audited.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed` |
| **Overall** | **GREEN** — all five gate-bearing subjects clear. `METAL2_PORT_BRIEF.md` issued alongside this report. |
| **DOps / Factories** | `LayerNormPreAllGatherDeviceOperation` → `LayerNormPreAllGatherProgramFactory`, `LayerNormPreAllGather2DProgramFactory`, `LayerNormPreAllGatherWelfordProgramFactory` · `LayerNormPostAllGatherDeviceOperation` → `LayerNormPostAllGatherProgramFactory`, `LayerNormPostAllGatherWelfordProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 4 dataflow kernels, all 4 own compute kernels, both borrowed rmsnorm compute kernels, and every shared helper are structurally Device 2.0 (`Noc`, `DataflowBuffer` / `CircularBuffer`, `Semaphore<>`, `CoreLocalMem`). No holdovers found. |
| *Prereqs* — Cross-op escapes | **Ok** — every `#include` outside the op resolves to `tt_metal/*`, `ttnn/cpp/ttnn/kernel_lib/`, `ttnn/cpp/ttnn/kernel/`, or in-family `normalization/kernel_util/`; every donor signature is a ✓ shape |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | Ok — every `get_compile_time_arg_val(...)` in every kernel uses a literal constant index; `tensor_args_t` on both DOps is a fixed named set |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — `yes` on all **five** factory rows of the *"Operations analysis"* sheet; cross-check against the code clean on every cheaply-checkable column, and the factory sets match one-to-one |
| *TTNN Readiness* — Concept (current) | `descriptor` (sheet, all 5 rows) — confirmed: all five factories expose only `static ProgramDescriptor create_descriptor(...)` (`layernorm_pre_all_gather_device_operation.hpp:20,28,36`; `layernorm_post_all_gather_device_operation.hpp:20,28`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — concept is `descriptor`, not `WorkloadDescriptor`; no `create_workload_descriptor` anywhere. (The sheet's separate `Execution Model` column reads `SPMD` on all 5 rows, which is the ordinary single-program model, not the `WorkloadDescriptor` escape.) |
| *TTNN Readiness* — Is safe to port? | **Yes** (sheet, all 5 rows; `Known op issues` blank). Consistent with the code: `Smuggled pointer` = `no`, and every device pointer reaches a kernel as an explicitly-bound `Buffer*` in `emplace_runtime_args`, never as a bare `->address()` — a grep for `->address()` / `.address()` in this directory returns **zero** hits. |
| *TTNN Readiness* — Custom hash | **No** (sheet, all 5 rows) — confirmed: no `compute_program_hash` override on either DeviceOperation (grep over the whole directory: zero hits) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** (sheet, all 5 rows) — confirmed: zero hits on either DeviceOperation |
| *TTNN Readiness* — `override_runtime_arguments` | **No** (sheet, all 5 rows) — confirmed: zero hits |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** (sheet, all 5 rows) — confirmed: `layernorm_distributed_nanobind.cpp` binds only the two user-facing functions `layer_norm_pre_all_gather` / `layer_norm_post_all_gather` (lines 20, 93); no factory or device-op internals are exposed |
| *TTNN Readiness* — Op-owned tensors | **No** — sheet cell blank on all 5 rows; consistent with the `descriptor` concept, which cannot carry them, and no factory returns a `buffers` vector |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (no op-owned tensors) — matches the sheet's `Porting Target` column on all 5 rows |
| *Port work* — Offset base pointer | **none** — GREEN. Zero `->address()` sites in the op; every per-core tile offset travels as its own scalar arg and is consumed as a page id |
| *Port work* — Tensor bindings (per binding) | **Case 1** ×5 (`input`, `residual`, `stats`, `gamma`, `beta`) + **Case 1** (`output`) · **clean** ×1 (`recip_tensor`, borrowed-memory DFB) — no Case 2 anywhere |
| *Port work* — TensorParameter relaxation | **none** (sheet, all 5 rows) — consistent with the code: a relaxation requires a custom hash and this op has none |
| *Port work* — TensorAccessor 3rd arg | **drop (Class 2)** — 2 sites, both in the post-allgather reader. *This overrides the dated triage doc, which lists this op as Class 3; see the detail section.* |
| *Port work* — CB endpoints | mixed: **1:1 legal** (most) · **self-loop** (compute-private intermediates, the borrowed recip LUT, two orphan scaler CBs) · **multi-binding flag** ×1 (`c_1` in the Welford pre factory) · **dead-CB drop** (`c_9` in both post factories; additionally `c_7`, `c_8` in the post-Welford factory) |

**CB endpoints** are dispositions, not gates: every out-of-window CB here has a port-time resolution. Dispositions are recorded per `(CB, config)` in the CB-endpoints detail section.

---

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, alongside this report).

All five gate-bearing subjects clear:

- **TTNN factory concept** — `Is able to port?` = `yes` on all five factory rows; cross-check clean.
- **Device 2.0** — complete across all ten kernels the op exercises and all six shared helpers.
- **Feature compatibility** — every Appendix A entry `N/A`.
- **Offset base pointers** — no host-folded offset in any device pointer (zero `->address()` sites in the directory).
- **TensorAccessor 3rd argument** — both sites Class 2 (redundant → drop).

**No code-path is blocked.** `RED at op level` does not apply — there is nothing to scope a subset around.

The port is ordinary work: seven tensor bindings (six Case 1 + one clean borrowed-memory DFB), two redundant page-size args to drop, one multi-binding flag, and four dead-CB allocations to drop. Two pre-existing defects unrelated to the port are recorded under *Misc anomalies* for the ops team — one of them (**#1**, RMSNorm + gamma + beta driving an unallocated `c_13`) will surface during the port for that single config, so it is repeated in the brief's *Watch for* section.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The sheet carries five rows for `normalization/layernorm_distributed` and every one reads `Is able to port?` = `yes`. Cross-check against the code is clean on every cheaply-checkable column:

  | Conjunct | Sheet (all 5 rows) | Code evidence | Verdict |
  |---|---|---|---|
  | `Concept == descriptor` | `descriptor` | `layernorm_pre_all_gather_device_operation.hpp:19-40`, `layernorm_post_all_gather_device_operation.hpp:19-32` — five `static ProgramDescriptor create_descriptor(...)`, no `create()`/`create_workload_descriptor` | ✓ agree |
  | `Custom hash == no` | `no` | no `compute_program_hash` in the directory | ✓ agree |
  | `get_dynamic_runtime_args == no` | `no` | absent from both device-ops | ✓ agree |
  | `override_runtime_arguments == no` | `no` | absent from both device-ops | ✓ agree |
  | `Pybind descriptor == no` | `no` | `layernorm_distributed_nanobind.cpp:20,93` bind only the two public functions | ✓ agree |
  | `Op-owned tensors == no` | *(blank)* | `descriptor` concept; no `buffers` vector | ✓ agree |
  | `Is safe to port?` | `yes`, `Known op issues` blank, `Smuggled pointer` = `no` | **Not re-derived** (expert-judgment axis, per the recipe). Consistent supporting signal: every buffer base is pushed as a `Buffer*` into `KernelDescriptor::RTArgList` / `emplace_runtime_args` (e.g. `layernorm_pre_all_gather_program_factory.cpp:174,179,184`; `layernorm_post_all_gather_program_factory.cpp:316-325,351-360`) and there is not one `->address()` call in the directory, so no un-annotated pointer exists for the framework to miss. | ✓ accepted |

  **Factory-set match: one-to-one, no staleness.** The sheet's five rows name exactly the five factories the code defines — `LayerNormPreAllGatherProgramFactory`, `LayerNormPreAllGather2DProgramFactory`, `LayerNormPreAllGatherWelfordProgramFactory` under `LayerNormPreAllGatherDeviceOperation`; `LayerNormPostAllGatherProgramFactory`, `LayerNormPostAllGatherWelfordProgramFactory` under `LayerNormPostAllGatherDeviceOperation`. No phantom row, no missing row. Both `Factory definition path` and `Declared in` point at the two `*_device_operation.hpp` files, which is where the factory structs are in fact declared.

  **Cross-column invariants hold.** `get_dynamic_runtime_args` = `no` on a `descriptor` concept (the `yes` case would only be legal on `descriptor` / `WorkloadDescriptor` anyway), and `Op-owned tensors?` is blank rather than `yes` on a `descriptor` row — a `descriptor` row claiming op-owned tensors would have meant a broken sheet.

  One naming drift worth noting for the next auditor, not a finding about this op: the column is headed `Override runtime args method?` **`(PD only)`**, where the recipe and `ttnn_op_porting_readiness.md` both call it `(PD and legacy)`. Same column, same meaning; the parenthetical changed. See *Recipe notes* #1.

- **Device 2.0 (every kernel used):** **GREEN.** Every kernel the op exercises — wherever it lives — is on Device 2.0 idioms. A sweep for `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen*`, bare `noc_async_read(` / `noc_async_write(`, bare `cb_reserve_back(` / `cb_push_back(` / `cb_wait_front(` / `cb_pop_front(`, `noc_semaphore_wait` / `noc_semaphore_inc`, `get_read_ptr(` / `get_write_ptr(cb`, `get_noc_addr_from_bank_id`, and `get_local_cb_interface` over all files below returns **zero hits**.

  | Kernel / helper | Owner | Device 2.0 evidence |
  |---|---|---|
  | `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp` | this op | `Noc noc` (:36), `DataflowBuffer dfb_inp_buf` (:37), `noc.async_read(src_a, dfb_inp_buf, …)` (:54) |
  | `device/kernels/dataflow/reader_layernorm_preallgather_2d.cpp` | this op | `Noc` (:50), `DataflowBuffer` (:51-53), `Semaphore<> reducer_sem` (:54), `UnicastEndpoint` (:119), `noc.async_write(…)` (:120) |
  | `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_post_allgather.cpp` | this op | `Noc` (:119), `DataflowBuffer` (:120-126), `CoreLocalMem<uint32_t>` (:33,39,49), `CircularBuffer(dfb_eps)` (:117) |
  | `device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp` | this op | `Noc` (:28), `DataflowBuffer` (:29), `noc.async_write(dfb_out_buf, s, …)` (:36) |
  | `device/kernels/compute/layernorm_pre_allgather.cpp` | this op | `DataflowBuffer` (:45-49) |
  | `device/kernels/compute/layernorm_pre_allgather_2d.cpp` | this op | `DataflowBuffer` (:46-51, :103-104) |
  | `device/kernels/compute/layernorm_pre_allgather_welford.cpp` | this op | `CircularBuffer` (:64-71) |
  | `device/kernels/compute/layernorm_post_allgather.cpp` | this op | `CircularBuffer` (:124-130) |
  | `device/kernels/compute/layernorm_post_allgather_welford.cpp` | this op | `DataflowBuffer` (:114-117) |
  | `device/kernels/compute/chain_llk.hpp` | this op | `CircularBuffer cb_a/cb_b/cb_out` (:118-120) |
  | `rmsnorm_distributed/…/compute/rmsnorm_pre_allgather.cpp` | in-family (`rmsnorm_distributed`) | `DataflowBuffer` (:53-57) |
  | `rmsnorm_distributed/…/compute/rmsnorm_post_allgather.cpp` | in-family (`rmsnorm_distributed`) | `CircularBuffer` (:69-78) |
  | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.{hpp,inl}` | shared kernel library | `DataflowBuffer dfb(dfb_id)` + `reserve_back`/`push_back` (`.inl:161-203`) |
  | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | shared kernel library | compute-side only |
  | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` | shared kernel library | `::DataflowBuffer` + `Noc` (:46-64) |
  | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | shared kernel pool | takes `CircularBuffer cb` by value (:13,29,44) |
  | `normalization/kernel_util/compute/{pre_add,combine_welford,memory}.h` | in-family shared | `DataflowBuffer&` params (`pre_add.h:24`, `combine_welford.h:49-50`) |
  | `normalization/kernel_util/generic/blocked_range.h` | in-family shared | pure index arithmetic |

  **One judgment call, recorded so a reviewer can second-guess it.** `normalization/kernel_util/compute/memory.h:30-31` is a free function taking a CB index — `get_pointer_to_cb_data(uint32_t cb_id, uint32_t tile_index)` → `get_tile_address(cb_id, tile_index)` — called from `layernorm_pre_allgather_welford.cpp:75`. It is *shaped* like a CB-index holdover, and both wrappers do expose a method form (`CircularBuffer::get_tile_address` at `tt_metal/hw/inc/api/dataflow/circular_buffer.h:72`, `DataflowBuffer::get_tile_address` at `.../dataflow_buffer.h:273`). **Not flagged as a Device 2.0 violation**, for two reasons: the underlying free function lives in `api/compute/cb_api.h:172` (a *compute*-thread CB API — Device 2.0 governs the data-movement surface, and the migration guide's header list is entirely `api/dataflow/*` / `api/core_local_mem.h` / `api/tensor/*`), and no `CircularBuffer`/`DataflowBuffer` object for `c_2` is in scope at the call site, so the isolated-holdover test ("wrapper already in scope") does not fire either. The Metal 2.0 port handles it cleanly regardless: the donor takes `uint32_t cb_id`, which is a ✓ shape (`dfb::name`'s constexpr cast covers it). Raised as *Questions* #1 in case the Device 2.0 owners want the compute-side surface counted.

- **Feature compatibility:** all four Appendix A entries scanned against the host code (5 factories + 2 device-ops), all ten kernels, and all six shared helpers.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `.global_circular_buffer` field on any `CBDescriptor`, no `experimental::CreateCircularBuffer(…, global_cb)`, no `remote_index(` / `remote_cb_*` identifiers, no `<tt-metalium/global_circular_buffer.hpp>` include. The one Buffer-backed CB in the op (`layernorm_pre_all_gather_welford_program_factory.cpp:362-369`, `.buffer = recip_tensor.buffer()`) is the plain borrowed-memory pattern → mechanical `DataflowBufferSpec::borrowed_from`, not a GCB. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set on any of the ~40 `CBDescriptor` literals in the five factories (all default to 0), no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | The op's only semaphore is a plain one: `SemaphoreDescriptor{.id = 0, …}` (`layernorm_pre_all_gather_program_factory.cpp:455-456`), consumed kernel-side as `Semaphore<> reducer_sem(reducer_semaphore_id)` (`reader_layernorm_preallgather_2d.cpp:54`) → ports as `SemaphoreSpec`. No `GlobalSemaphore` type, no `CreateGlobalSemaphore`, no `<tt-metalium/global_semaphore.hpp>`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Kernel-level signal (the decider) does not fire: every `get_compile_time_arg_val(...)` call in every kernel audited uses a **literal constant** index (verified by grepping for any non-literal argument — zero hits), and the `TensorAccessorArgs<N>` offsets chain through `constexpr next_compile_time_args_offset()`. Op-level signal also absent: `LayerNormPreAllGatherInputs` (3 named tensors) and `LayerNormPostAllGatherInputs` (4 named tensors) carry no variable-count container. |

- **CB endpoints (GATE-free):** classified per `(CB, config)`, per node. Nothing here blocks a Gen1 port. Full census below; the summary is: the reader↔compute↔writer chains are ordinary 1:1 FIFOs, the compute-private intermediates are one-toucher self-loops, and there are exactly **four** out-of-window cases — one multi-binding, three dead CBs (plus one CB that is *used but never allocated*, an existing defect recorded under *Misc anomalies*).

  **`LayerNormPreAllGatherProgramFactory`** — configs: {LN, RMS} × {fuse\_pre\_add on/off}. Compute kernel is `layernorm_pre_allgather.cpp` (LN) or `rmsnorm_pre_allgather.cpp` (RMS); the census is identical for both.

  | CB | Census on a node | Verdict | Disposition |
  |---|---|---|---|
  | `c_0` input | reader locked-P (`reader_…pre_allgather.cpp:53,61`), compute locked-C | 1:1 | none |
  | `c_1` reduce scaler | reader locked-P (`calculate_and_prepare_reduce_scaler`, :31-32 → `reduce_helpers_dataflow.inl:163,203`), compute locked-C (`layernorm_pre_allgather.cpp:107`) | 1:1 | none |
  | `c_5` residual *(fuse only)* | reader locked-P (:56,63), compute locked-C (`pre_add::one_row`) | 1:1 | none |
  | `c_3` fused a+b *(fuse only)* | compute only, P **and** C | 1 toucher | **self-loop** |
  | `c_6` x² | compute only, P and C (`layernorm_pre_allgather.cpp:70,78` + `reduce<…>` consumes) | 1 toucher | **self-loop** |
  | `c_14` output | compute locked-P (via `compute_kernel_lib::reduce` dest), writer locked-C (`writer_…:33,41`) | 1:1 | none |

  **`LayerNormPreAllGather2DProgramFactory`** — configs: {fuse on/off} × {merge node `y==0`, worker node `y>0`}. Note the compute kernel is hardcoded to `layernorm_pre_allgather_2d.cpp` regardless of `is_rmsnorm` (see *Misc anomalies*).

  | CB | Census on a node | Verdict | Disposition |
  |---|---|---|---|
  | `c_0` input | reader locked-P (`reader_layernorm_preallgather_2d.cpp:80,104`), compute locked-C | 1:1 | none |
  | `c_1` reduce scaler | reader locked-P (:66-70), compute locked-C (`layernorm_pre_allgather_2d.cpp:96`) | 1:1 | none |
  | `c_5` residual *(fuse only)* | reader locked-P (:82,106), compute locked-C | 1:1 | none |
  | `c_3` fused *(fuse only)* | compute only, P and C | 1 toucher | **self-loop** |
  | `c_6` x² | compute only, P and C | 1 toucher | **self-loop** |
  | `c_16` per-core partial out | compute locked-P (`reduce<… dfb_out …>`), reader locked-C (`reader_…2d.cpp:114,129`) | 1:1 | none |
  | `c_13` zero tile | reader locked-P on merge nodes (`prepare_zero_tile<dfb_zero>`, :72 → `l1_helpers.hpp:59-63`), compute locked-C on merge nodes (`layernorm_pre_allgather_2d.cpp:109`, waits and never pops) | 1:1 | none |
  | `c_15` cross-core merge | reader: raw `get_write_ptr()` peek on **every** node (:127) **plus** `push_back` on merge nodes (:137) → **locked producer**; compute: `wait_front`/`pop_front` on merge nodes (:108,124) → **locked consumer** | 1:1 (1 locked P + 1 locked C) | none — see the note below |
  | `c_14` final out *(merge\_cores only)* | compute locked-P on merge nodes (:132), writer locked-C (writer runs only on `merge_cores`) | 1:1 | none, but see the core-range note below |

  Two notes on the 2D factory that the porter should not have to re-derive:
  - **`c_15` is not a hidden-second-writer case.** The `noc.async_write(…, .addr = dfb_x2_merge_buf.get_write_ptr() + worker_offset)` at `reader_layernorm_preallgather_2d.cpp:120-127` targets a **remote** node's `c_15` instance (`reduce_core_noc_x/y`), while the `get_write_ptr()` it calls is a peek on the writer's own local instance used purely to compute the identical offset. So the remote co-fill does **not** add a second local endpoint on the receiving node: on a merge node the touchers are exactly the local reader (which does the `push_back` once the semaphore says all peers have landed) and the local compute. This is the case that most looks like a face-(a) hidden co-fill and isn't.
  - **`c_14`'s DFB core range is a strict subset of a binding kernel's core range.** The `c_14` `CBDescriptor` is declared over `merge_cores` (`layernorm_pre_all_gather_program_factory.cpp:579-585`) while the compute kernel that produces into it runs over `all_cores` (:484) and reaches it only under the runtime `if (is_merge_core)` (`layernorm_pre_allgather_2d.cpp:100-133`). Same pattern for `c_13`, whose producer side is likewise runtime-gated. Legal in the legacy CB world; flagged because a Metal 2.0 `DataflowBufferSpec` whose core range does not cover its binding `KernelSpec`'s core range is the kind of thing the spec validator may object to. Worth confirming early in the port rather than at first build.

  **`LayerNormPreAllGatherWelfordProgramFactory`** — LN only (`layernorm_pre_all_gather_welford_program_factory.cpp:46` fatals on RMS), configs {fuse on/off}.

  | CB | Census on a node | Verdict | Disposition |
  |---|---|---|---|
  | `c_0` input | reader locked-P, compute locked-C | 1:1 | none |
  | **`c_1`** transpose scratch | **two locked producers**: the reader unconditionally `reserve_back(1)`/`push_back(1)`s a reduce-scaler tile into `c_1` (`reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp:24,31-32` — that reader is shared with the 1D factory, where `c_1` *is* the scaler CB), **and** compute uses `c_1` as its post-Welford transpose scratch with its own `reserve_back(2)`/`push_back(2)`/`wait_front(2)`/`pop_front(2)` (`layernorm_pre_allgather_welford.cpp:215-220, 277-283, 290-293`) | ≥2 locked producers — cannot fit 1P+1C | **set the DFB multi-binding advanced option** (records the Quasar debt). Underlying sloppiness reported under *Misc anomalies*. |
  | `c_5` residual *(fuse only)* | reader locked-P, compute locked-C (:118,151) | 1:1 | none |
  | `c_3` fused *(fuse only)* | compute only, P and C (:119,149,156,188) | 1 toucher | **self-loop** |
  | `c_4` Welford mean spill *(fuse only)* | compute only, P and C (:102,109,154,186,189,196) | 1 toucher | **self-loop** |
  | `c_6` Welford M2 spill *(fuse only)* | compute only, P and C (:103,110,155,187,190,197) | 1 toucher | **self-loop** |
  | `c_2` reciprocal LUT (borrowed memory) | compute only, **raw peek** — `get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0)` (:75); no FIFO ops, no other kernel touches it | 1 toucher, role-free | **self-loop**, on a DFB `borrowed_from` the `recip_tensor` binding |
  | `c_14` output | compute locked-P (:299), writer locked-C | 1:1 | none |

  **`LayerNormPostAllGatherProgramFactory`** — configs: {LN, RMS} × {gamma on/off} × {beta on/off} × {1D, 2D}. Compute kernel is `layernorm_post_allgather.cpp` (LN) or `rmsnorm_post_allgather.cpp` (RMS). The 1D/2D split changes only work distribution, not the census.

  | CB | Census on a node | Verdict | Disposition |
  |---|---|---|---|
  | `c_0` input | reader locked-P (`reader_…post_allgather.cpp:156,160`), compute locked-C | 1:1 | none |
  | `c_1` stats | reader locked-P (:136,149), compute locked-C (`layernorm_post_allgather.cpp:148,161`) | 1:1 | none |
  | `c_2` gamma *(gamma only)* | reader locked-P (:166,172), compute locked-C (via `chain_llk` / rms gamma loop) | 1:1 | none |
  | `c_3` beta *(beta only)* | reader locked-P (:177,183), compute locked-C | 1:1 | none |
  | `c_4` epsilon | reader locked-P (`generate_bcast_col_scalar`, :117 → `generate_bcast_scalar.hpp:15,23`), compute locked-C (`layernorm_post_allgather.cpp:133,260`) | 1:1 | none |
  | `c_5` reduce scaler | reader locked-P (:111-115), compute locked-C (:132,261) | 1:1 | none |
  | `c_6` reduced stats | compute only, P and C (:163,170,180,257) | 1 toucher | **self-loop** |
  | `c_7` mean² *(LN only)* | compute only, P and C (:186,192,201,207) | 1 toucher | **self-loop** |
  | `c_8` var | compute only, P and C (LN :209,215,220,231; RMS `rmsnorm_post_allgather.cpp:93,99,112`) | 1 toucher | **self-loop** |
  | **`c_9`** var+eps | **zero touchers, every config.** Allocated at `layernorm_post_all_gather_program_factory.cpp:490-496`. Both compute kernels declare the index and then never use it — `layernorm_post_allgather.cpp:115` and `rmsnorm_post_allgather.cpp:52` are the *only* occurrences of `CBIndex::c_9` anywhere in the audited set; the rsqrt is fused in DEST and packed straight into `c_10`. No CTA carries a CB index in this factory, so there is no indirect path either. | **Dead CB** | **drop the allocation** (a dead CB has no behavior, so removing it changes none). See the confirmation note below. |
  | `c_10` 1/√(var+ε) | compute only, P and C (:233,239,258) | 1 toucher | **self-loop** |
  | `c_11` x−mean *(LN only)* | compute only, P and C (`chain_llk` `x_minus_mean_node` out, `normed_output_node` in) | 1 toucher | **self-loop** |
  | `c_12` x normed | compute only, P and C | 1 toucher | **self-loop** |
  | `c_13` ×gamma intermediate *(LN + beta)* | compute only, P and C (`chain_llk` `gamma_optional_node` out → `beta_optional_node` in) | 1 toucher | **self-loop**. ⚠ For **RMS + gamma + beta** this index is used by the kernel but the factory never allocates it — see *Misc anomalies*. |
  | `c_14` output | compute locked-P, writer locked-C | 1:1 | none |

  **`LayerNormPostAllGatherWelfordProgramFactory`** — LN only in practice (`layernorm_post_all_gather_device_operation.cpp:166-171` fatals on RMS + Welford), configs {gamma on/off} × {beta on/off} × {1D, 2D}.

  | CB | Census on a node | Verdict | Disposition |
  |---|---|---|---|
  | `c_0`, `c_1`, `c_2`, `c_3`, `c_4` | reader locked-P, compute locked-C (`layernorm_post_allgather_welford.cpp:114-119,125-131` + `chain_llk`) | 1:1 | none |
  | `c_5` reduce scaler | reader locked-P (`reader_…post_allgather.cpp:111-115`) — the Welford compute kernel never reads it (no `CBIndex::c_5` in `layernorm_post_allgather_welford.cpp`) | 1 toucher | **self-loop** (orphan producer; the wasted scaler tile is under *Misc anomalies*) |
  | `c_6` combined stats | compute only, P and C (`combine_welford.h:151-154` + :131,137,169) | 1 toucher | **self-loop** |
  | **`c_7`** mean² | **zero touchers.** Allocated at `layernorm_post_all_gather_welford_program_factory.cpp:583-589`; no `CBIndex::c_7` in the Welford compute kernel, `chain_llk.hpp`, or `combine_welford.h`. | **Dead CB** | **drop the allocation** |
  | **`c_8`** var | **zero touchers.** Allocated at `…welford_program_factory.cpp:545-551`; no `CBIndex::c_8` in this factory's kernels. | **Dead CB** | **drop the allocation** |
  | **`c_9`** var+eps | **zero touchers.** Allocated at `…welford_program_factory.cpp:554-560`. | **Dead CB** | **drop the allocation** |
  | `c_10`, `c_11`, `c_12`, `c_13` *(c_13 only with beta)* | compute only, P and C (`chain_llk` nodes at :42-101, plus :138,151,170) | 1 toucher each | **self-loop** each |
  | `c_14` output | compute locked-P (`chain_llk` terminal node), writer locked-C | 1:1 | none |

  **Confirming the dead CBs (the recipe rightly demands positive proof).** For each of `c_7`, `c_8`, `c_9` above the index was checked against *every* kernel bound by the owning factory — the reader, the writer, the compute kernel, and the two headers the compute kernel pulls in (`chain_llk.hpp`, `combine_welford.h`) — and against the alternate compute kernel on the `is_rmsnorm` branch. No indirect path exists: neither post factory passes any CB index through a CTA, an RTA, or a `named_compile_time_args` entry (their CTA lists carry only sizes, counts, flags, and `TensorAccessorArgs`), so an index cannot reach a kernel except by the literal `tt::CBIndex::c_N` spellings that were grepped. `c_9` is dead in **both** post factories under **all** configs; `c_7` and `c_8` are dead in the Welford post factory only (both are live in the non-Welford one). Residual doubt is low but nonzero for one reason worth stating: these three are precisely the intermediates the *non*-Welford kernel uses, so their allocation in the Welford factory looks like copy-paste rather than intent — which is consistent with dead, not with a use hiding somewhere.

- **Offset base pointers:** **GREEN — no fold anywhere, and nothing to reconcile against the triage doc.** The checked-in triage `2026-07-19_offset_base_pointers.md` lists no normalization op, and this op is clean on its own scan, so this is the doc's *"no fold, op not in the tables"* outcome for every site. Every device pointer in all five factories is delivered by the `Buffer*`-binding form (`reader_args.push_back(a.buffer())`, `emplace_runtime_args(core, {output.buffer(), …})`) — a grep for `->address()` / `.address()` across the whole directory returns zero hits, so there is no expression a host-side offset could have been folded into. The per-core offsets that *do* exist (`in_tile_offset`, `tile_offset`, `out_tile_offset`, `stats_offset`, `y_offset` — e.g. `layernorm_pre_all_gather_program_factory.cpp:170-171`, `layernorm_post_all_gather_program_factory.cpp:345-347`) ride their **own** scalar args and are consumed on-device as `page_id` values into a `TensorAccessor` (`reader_…post_allgather.cpp:143,157`), never added into a base address. Type 3 (`address_offset`) is `N/A` per Appendix A; Type 4 (`narrow` / interior-base `MeshBuffer`) does not appear.

- **TensorAccessor 3rd argument:** **GREEN — both sites are Class 2 (redundant → drop). This overrides the dated triage doc.**

  Two sites, both in the post-allgather reader (shared by both post factories):
  - `reader_unary_interleaved_ln_rm_gb_post_allgather.cpp:104` — `TensorAccessor(gamma_args, gamma_addr, gamma_stick_size)`
  - `reader_unary_interleaved_ln_rm_gb_post_allgather.cpp:107` — `TensorAccessor(beta_args, beta_addr, beta_stick_size)`

  **The triage doc disagrees, and it is stale.** `2026-07-06_tensor_accessor_3rd_arg_triage.md:66,122` classes `normalization_ln_rm_gb_post_allgather` as **Class 3 — latent bug**, on the grounds that the TILE-layout branch passes `element_size() * 1024`, which for a **block-float** gamma/beta yields 1024 B against a true page of 1088 B (the bf8 exponent section is dropped), used verbatim with no realignment to save it. That reasoning was correct for the code as it then stood. It no longer applies, because **BFLOAT8_B gamma/beta is now rejected by validation**: `layernorm_post_all_gather_device_operation.cpp:78-81` fatals unless `gamma.dtype()` is `BFLOAT16` or `FLOAT32`, and `:123-126` does the same for beta. The bf8-TILE branch the doc flagged is therefore unreachable — and unreachable through *every* entry point, since `validate_on_program_cache_miss` runs on any cache miss and a gamma dtype change is a cache miss under the default hash. Per the recipe's contract for this doc ("dated and not kept current — a disagreement means the doc is stale; trust your read"), the current classification is Class 2. Worked through the recipe's two questions:

  | | TILE-layout branch (`…program_factory.cpp:228` / `…welford_…:268`) | ROW\_MAJOR branch (`…:223` / `…:263`) |
  |---|---|---|
  | **Q1 — sharded or interleaved?** | `gamma_args` comes from `TensorAccessorArgs(gamma.buffer())`; gamma/beta are ordinary user weight tensors, in practice interleaved (so realignment is in play as a safety net) — but the verdict does not depend on the answer, since Q2 lands on the exact page size either way | same |
  | **Q2 — correct or wrong magnitude?** | `element_size() * 1024` with dtype ∈ {bf16, fp32} ⇒ **2048 B** or **4096 B** = `tt::tile_size(Float16_b)` / `tt::tile_size(Float32)` = `buffer->page_size()` exactly. **Correct magnitude** (indeed exact). Aligned on the strictest target (BH/Quasar DRAM, 64 B): 2048 and 4096 are both multiples of 64. | `padded_shape[-1] * element_size()`, and validation pins `padded_shape[-1] == tile_width` (`:107`) ⇒ **64 B** (bf16) or **128 B** (fp32) = the row-major stick = `buffer->page_size()`. **Correct magnitude**, both 64-aligned. |
  | **Class** | **2 — redundant / inert** → drop the arg | **2 — redundant / inert** → drop the arg |

  Port action: drop the third argument at both sites; Metal 2.0 supplies the `aligned_page_size` implicitly and it equals the value being passed today. No `dynamic_tensor_shape` relaxation is involved (that is Class 1; the page size here does not vary with row width across cache-reused shapes — validation pins gamma/beta's width to the input's). **Routed to the triage-doc owner as a stale-row correction**, not as a gate.

---

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, all Case 1 or clean — no Case 2 anywhere in the op):

  | Binding | DOp / factories | Delivery today | Kernel use | Case |
  |---|---|---|---|---|
  | `input` | both DOps, all 5 factories | `Buffer*` in reader RTA slot 0 (`…pre_allgather_program_factory.cpp:174`; `…post_allgather_program_factory.cpp:316,351`) | `TensorAccessor(src_args, src_addr)` (`reader_…pre_allgather.cpp:34`, `reader_…post_allgather.cpp:100`, `reader_…2d.cpp:48`) | **Case 1** |
  | `residual_input_tensor` | Pre: 1D, 2D, Welford *(fuse only)* | `Buffer*` in reader RTA (`:179`, `:438`, welford `:192`) | `TensorAccessor(res_args, res_addr)` (`reader_…pre_allgather.cpp:44`, `reader_…2d.cpp:61`) | **Case 1** |
  | `recip_tensor` | Pre: Welford only | **borrowed-memory CB** — `.buffer = recip_tensor.buffer()` on `c_2` (`…welford_program_factory.cpp:369`); never an RTA | raw peek `get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0)` (`layernorm_pre_allgather_welford.cpp:75`) | **clean** (causal-link gate) → `DataflowBufferSpec::borrowed_from` |
  | `stats` | Post: both factories | `Buffer*` in reader RTA (`…post_allgather_program_factory.cpp:324,359`; welford `:357,393`) | `TensorAccessor(stats_args, stats_addr)` (`reader_…post_allgather.cpp:101`) | **Case 1** |
  | `gamma` | Post: both factories *(optional)* | `Buffer*` (or `nullptr`) in reader RTA (`:322,358`) | `TensorAccessor(gamma_args, gamma_addr, gamma_stick_size)` (`:104`) | **Case 1** (+ 3rd-arg drop) |
  | `beta` | Post: both factories *(optional)* | `Buffer*` (or `nullptr`) in reader RTA (`:323,358`) | `TensorAccessor(beta_args, beta_addr, beta_stick_size)` (`:107`) | **Case 1** (+ 3rd-arg drop) |
  | `output` | both DOps, all 5 factories | `Buffer*` in writer RTA slot 0 (`:184,444`; post `:328,362`) | `TensorAccessor(dst_args, dst_addr)` (`writer_…:26`) | **Case 1** |

  All seven arrive through the `Buffer*`-binding form, which the framework already patches on cache hits — so this is **routine port work, not a correctness hazard**. `gamma` and `beta` are independently optional and today pass `nullptr` when absent (`layernorm_post_all_gather_program_factory.cpp:111-112`, `…welford_…:125-126`, with a matching `TensorAccessorArgs(nullptr)` at `:251-254`); the port needs those two bindings expressed as optional `TensorParameter`s rather than as a base of `0u`.

- **TensorParameter relaxation:** **none** — the sheet's `TensorParameter relaxation` column reads `none` on all five rows, and the code agrees that none *could* be active: a relaxation is a custom hash excluding a property from the cache key, and neither DeviceOperation declares `compute_program_hash`. Nothing for the porter to apply.
- **TensorAccessor 3rd arg:** drop the redundant page-size arg at `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_post_allgather.cpp:104` (gamma) and `:107` (beta). Both Class 2 — no relaxation to set.
- **CB endpoints:**
  - **self-loop** — Pre-1D: `c_3` *(fuse)*, `c_6`. Pre-2D: `c_3` *(fuse)*, `c_6`. Pre-Welford: `c_3`, `c_4`, `c_6` *(fuse)*, and `c_2` (borrowed recip LUT, raw-peek only). Post-1D/2D: `c_6`, `c_7` *(LN)*, `c_8`, `c_10`, `c_11` *(LN)*, `c_12`, `c_13` *(LN+beta)*. Post-Welford: `c_5` (orphan producer), `c_6`, `c_10`, `c_11`, `c_12`, `c_13` *(beta)*.
  - **1P+1C assignment** — not needed anywhere: every two-toucher CB in this op is already one locked producer + one locked consumer.
  - **multi-binding advanced-option flag** — `(c_1, LayerNormPreAllGatherWelfordProgramFactory, all configs)`: two locked producers (shared reader's reduce-scaler push + compute's transpose-scratch push). The census genuinely cannot fit 1P+1C.
  - **dead-CB drop** — `c_9` @ `layernorm_post_all_gather_program_factory.cpp:490-496` and `layernorm_post_all_gather_welford_program_factory.cpp:554-560`; `c_7` @ `layernorm_post_all_gather_welford_program_factory.cpp:583-589`; `c_8` @ `layernorm_post_all_gather_welford_program_factory.cpp:545-551`. No dead CTA accompanies any of them (no CB index travels by CTA in these factories), so the drop is the allocation only.

---

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** one — `(c_1, Pre-Welford factory)`. The extra producer is **visible, not hidden**: it is the *shared reader* `reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp:31-32` pushing a reduce-scaler tile into an index that this factory repurposes as the compute kernel's transpose scratch. The hidden-second-writer hunt was run over every CB in all five factories and found nothing else: the only raw `get_write_ptr()` sites in the op are `reader_…post_allgather.cpp:167,178,201,212` (a kernel peeking the buffer it is itself the FIFO producer of — one toucher, not two) and `reader_layernorm_preallgather_2d.cpp:127` (a remote-node write address, addressed in the CB-endpoints note above).
- **Cross-op / shared kernels:** no `_metal2` fork exists beside **any** kernel this op uses — a repo-wide search under `ttnn/cpp/ttnn` for `*_metal2*` in `normalization/`, `kernel_lib/`, and `kernel/` returns nothing — so this port creates the first fork of whatever it touches.
  - **Borrowed kernel files (file-path instantiation of source this op does not own):** exactly two, both in-family — `rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp` (from `layernorm_pre_all_gather_program_factory.cpp:143`) and `rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp` (from `layernorm_post_all_gather_program_factory.cpp:287` and `layernorm_post_all_gather_welford_program_factory.cpp:316`). **Sunset list: empty beyond this op** — a grep for `rmsnorm_distributed/device/kernels/compute` across `ttnn/` returns only those three call sites, so no other op binds either file and the legacy copies can retire with this port. (The similarly-named kernels under `experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/` are that op's **own separate copies**, not the same files.)
  - **This op's own kernels are not borrowed by anyone.** In particular `writer_unary_interleaved_start_id_blocked.cpp` shares a basename with two unrelated files — `normalization/layernorm/device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp` (`layernorm_op_multi_core.cpp:639`) and softmax's `…_blocked_sm.cpp` — but those are *different files* in *different directories*. Nothing outside this directory instantiates any kernel this op owns, so the port has **no cross-op coordination cost** on the file-path axis.
- **RTA varargs:** **none.** Every kernel reads each runtime arg at a fixed literal index as a distinct field (`reader_…post_allgather.cpp:59-69`, `reader_…2d.cpp:22-29`, `writer_…:15-17`, and one-arg compute kernels) — no counted loop over `get_arg_val`, no `arg_index++` run, no data-selected index. All RTAs port to **named** args; the porter should not reach for the vararg mechanism anywhere in this op.
- **Optional bindings:** `gamma` and `beta` are independently optional across both post factories, and the current code leans on `nullptr`-as-`0u` for the absent case (`layernorm_post_all_gather_program_factory.cpp:111-112,251-254`). Worth settling the optional-`TensorParameter` shape before writing the post-allgather reader's bindings, since four of the six configs exercise at least one absent case.
- **DFB core range vs. kernel core range (2D pre factory):** `c_14` is declared over `merge_cores` while its producing compute kernel is declared over `all_cores` (detail in the CB-endpoints section). Confirm the spec validator accepts that before building out the rest of that factory.

---

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** Every escape lands in a benign donor class, and every function the op's kernels call across a directory boundary takes a ✓ shape. Specifically: no `uint32_t sem_id` / `sem_addr` donor parameters, no `TensorAccessorArgs<N>` (Shape 2) or CTA-offset-NTTP (Shape 3) donor parameters, no old-style addr-gen (Shape 4), and no `CircularBuffer&`-by-reference donor signature — the two ⭐ scheduling-blocker shapes are both absent.

**Summary table** — one row per (op kernel, donor file):

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `reader_…pre_allgather.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | 2 — shared kernel library | ✓ |
| `reader_…post_allgather.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | 2 — shared kernel library | ✓ |
| `reader_…post_allgather.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 3 — second shared pool | ✓ |
| `reader_layernorm_preallgather_2d.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | 2 — shared kernel library | ✓ |
| `reader_layernorm_preallgather_2d.cpp` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` | 2 — shared kernel library | ✓ |
| `writer_unary_interleaved_start_id_blocked.cpp` | *(none — `tt_metal/*` only)* | 1 | ✓ |
| `layernorm_pre_allgather.cpp` | `kernel_lib/reduce_helpers_compute.hpp`; `normalization/kernel_util/compute/pre_add.h` | 2; 5 — in-family | ✓ |
| `layernorm_pre_allgather_2d.cpp` | `kernel_lib/reduce_helpers_compute.hpp`; `normalization/kernel_util/compute/pre_add.h` | 2; 5 | ✓ |
| `layernorm_pre_allgather_welford.cpp` | `normalization/kernel_util/compute/memory.h`; `normalization/kernel_util/generic/blocked_range.h` | 5; 5 | ✓ |
| `layernorm_post_allgather.cpp` | `chain_llk.hpp` *(in-directory, not an escape)* | — | ✓ |
| `layernorm_post_allgather_welford.cpp` | `normalization/kernel_util/compute/combine_welford.h`; `chain_llk.hpp` | 5; in-directory | ✓ |
| `rmsnorm_pre_allgather.cpp` *(borrowed)* | `kernel_lib/reduce_helpers_compute.hpp`; `normalization/kernel_util/compute/pre_add.h` | 2; 5 | ✓ |
| `rmsnorm_post_allgather.cpp` *(borrowed)* | `kernel_lib/reduce_helpers_compute.hpp` | 2 | ✓ |

**Per-call detail** — recorded even though all rolls are ✓, because two shapes are worth naming for the porter:

| Donor function | Signature shape | Status | Note |
|---|---|---|---|
| `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb_id, pool, dim, factor>()` | CB index as **NTTP** (`template <uint32_t dfb_id, …>`) | ✓ OK | `dfb::name`'s constexpr cast covers template-parameter position |
| `dataflow_kernel_lib::prepare_zero_tile<dfb_id>()` | CB index as NTTP | ✓ OK | same |
| `generate_bcast_col_scalar(CircularBuffer cb, uint32_t scalar)` | `CircularBuffer` **by value** | ✓ OK | not the flagged `CircularBuffer&` shape — a by-value Device 2.0 wrapper constructed from the token at the call site (`reader_…post_allgather.cpp:117`) |
| `norm::…::pre_add::one_row<FUSE>(DataflowBuffer&, DataflowBuffer&, DataflowBuffer&, uint32_t, uint32_t)` | `DataflowBuffer&` | ✓ OK | Device 2.0 native; the port passes objects constructed from `dfb::name` |
| `norm::…::combine_welford_partials(DataflowBuffer&, DataflowBuffer&, …)` | `DataflowBuffer&` | ✓ OK | same |
| `norm::…::memory::get_pointer_to_cb_data<To>(uint32_t cb_id, uint32_t tile_index)` | `uint32_t cb_id` | ✓ OK | `dfb::name`'s constexpr cast covers it. Internally forwards to the compute-side free function `get_tile_address(cb_id, idx)` — see the Device 2.0 judgment call above and *Questions* #2 |
| `compute_kernel_lib::reduce<pool, dim, in_cb, scaler_cb, out_cb, policy>(shape)` | CB indices as NTTPs | ✓ OK | constexpr cast covers it |

**Borrowed kernel files:** see *Heads-ups* — two in-family compute kernels, no other consumers, no existing `_metal2` fork.

### Relaxation candidates (mined from a custom hash)

None to mine — neither DeviceOperation declares a custom hash, so there is no hash logic revealing which tensor properties the op actually depends on. The sheet's `TensorParameter relaxation` column reads `none` on all five rows, consistent with that.

### TTNN factory analysis

Sheet rows and code agree throughout (the per-column comparison is in *Gate detail* → *TTNN factory concept*). The non-gating facts the port's ProgramFactory wiring needs:

- **Current concept:** `descriptor` (five `create_descriptor` entry points, no workload form). Sheet's `Op Classification` = `PD Op (pointer-patching)`, `Execution Model` = `SPMD` on all five rows.
- **Op-owned tensors:** none. The `recip_tensor` in the Welford pre factory is a *caller-supplied* tensor threaded through `tensor_args_t` (`layernorm_pre_all_gather_device_operation_types.hpp:25`) and created by the separate `ttnn.create_layer_norm_reciprocals` API — not an op-owned buffer, so it does not force the `WorkloadDescriptor` shape. It becomes an ordinary `TensorParameter` whose DFB is `borrowed_from` it.
- **MeshWorkload need:** none, genuine or artifact.
- **Target concept:** `ProgramSpecFactoryConcept` — matching the sheet's own `Porting Target` column on all five rows.
- **Gate conjuncts, `no` on the sheet and confirmed absent from code:** custom hash · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · other risky pybind (the nanobind file exposes only the two public functions and plain scalar/tensor/optional args).
- **`named_compile_time_args` already in use** in the Welford pre factory (`layernorm_pre_all_gather_welford_program_factory.cpp:146-148,280`, read back as `get_named_compile_time_arg_val("welford_unpack_fp32_active")` at `layernorm_pre_allgather_welford.cpp:43`) — that one CTA is already named and carries over directly.

---

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

1. **RMSNorm + gamma + beta uses `c_13`, which the factory never allocates.** `rmsnorm_post_allgather.cpp:63-65` sets `cb_times_gamma_out_idx = tt::CBIndex::c_13` when both `do_gamma` and `do_beta`, and then `reserve_back`/`push_back`/`pop_front`s it (`:153,160,173,182`). But `layernorm_post_all_gather_program_factory.cpp:517-545` allocates `c_13` **only** under `if (!is_rmsnorm)`. That config is reachable: `ttnn::rms_norm_post_all_gather` forwards a `bias` straight through (`rmsnorm_distributed/rmsnorm_post_all_gather.cpp:43-53`) and nothing in `validate_on_program_cache_miss` forbids an RMS beta. The kernel would drive an unconfigured CB index. Presumably never exercised (RMSNorm rarely carries a bias), but it is a latent hang/corruption, and it will also block the port for that one config — a Metal 2.0 kernel cannot bind a DFB no spec declares. Worth fixing on the ops track ahead of the port; raised as *Questions* #2.
2. **The shared pre-allgather reader pushes a reduce-scaler tile into the Welford factory's scratch CB.** `reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp:24,31-32` unconditionally produces one scaler tile into `c_1`. In the 1D factory that is correct — `c_1` *is* the reduce-scalar CB. In the Welford factory `c_1` is the post-Welford transpose scratch (`layernorm_pre_all_gather_welford_program_factory.cpp:344-350`), which the Welford kernel neither expects nor drains. Traced through: it is **not** a live correctness bug — each RISC keeps its own local write/read pointer, so the packer overwrites the scaler tile at the CB base and the unpack side still reads its own tiles; the only lasting effect is a permanent +1 on the shared credit count, which (because the compute side pushes and pops in units of 2) never unblocks a `wait_front` early. Still: a wasted L1 write, a wasted NCRISC round trip, a permanently skewed credit count, and the reason `c_1` needs the multi-binding flag rather than a plain 1:1. Cleanest fix is to gate the reader's scaler generation behind a define, or give the Welford factory a scratch index the reader does not touch.
3. **The same wasted scaler in the post-Welford factory.** `layernorm_post_all_gather_welford_program_factory.cpp:526-532` allocates `c_5` and the shared post reader fills it (`reader_…post_allgather.cpp:111-115`), but the Welford compute kernel never reads `c_5`. Dead work rather than a dead CB (the producer keeps it at one endpoint), so it is a self-loop for the port rather than a drop.
4. **Three intermediates allocated but unused in the post-Welford factory** — `c_7`, `c_8`, `c_9` (see the CB-endpoints census for line numbers). They are the non-Welford kernel's intermediates, carried over into a factory whose kernel computes the same quantities inside DEST. Wasted L1 only.
5. **`c_9` (`var + epsilon`) is allocated by both post factories and used by neither kernel.** `layernorm_post_allgather.cpp:115` and `rmsnorm_post_allgather.cpp:52` declare the index with an explanatory comment and never touch it — the `add_tiles` + `rsqrt_tile` pair is fused in DEST and packed directly into `c_10`. The declarations themselves are also dead code worth deleting alongside the allocation.
6. **The 2D pre-allgather factory ignores `is_rmsnorm` and appears to be an rmsnorm-shaped path wearing a layernorm name.** `LayerNormPreAllGather2DProgramFactory` hardcodes `layernorm_pre_allgather_2d.cpp` (`layernorm_pre_all_gather_program_factory.cpp:480-482`) with no `is_rmsnorm` branch, forces `out0_tiles = 1` (`:345`) where the 1D factory uses 2 for layernorm (`:87-90`), and that kernel's own header comment reads *"This kernel computes rmsnorm statistics. For rmsnorm it computes E(x\*\*2)"*. Meanwhile `compute_output_specs` sizes the output at **two** tile columns for `LAYERNORM` (`layernorm_pre_all_gather_device_operation.cpp:79-82`). So a `LAYERNORM` + `use_2d_core_grid` request looks like it produces only `E(x²)` into a two-column output tensor. Sibling evidence that this was meant to be the rms path: `rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather_2d.cpp` exists and is instantiated by **no** factory anywhere. Not investigated further — outside the audit's remit — but it wants an owner's eye.
7. **`packer_l1_acc` is destructured from the compute-kernel config in all five factories and then never used** (e.g. `layernorm_pre_all_gather_program_factory.cpp:55-56`, `layernorm_post_all_gather_program_factory.cpp:73-74`; no `ComputeConfigDescriptor` in the op sets it). Callers that set `packer_l1_acc=true` — including the op's own nightly test at `tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py:759` — silently get no effect. The value still feeds the default program hash via `compute_kernel_config`, so it also causes cache misses that change nothing.
8. **`layernorm_pre_allgather_welford.cpp:299` pushes into `c_14` with no preceding `reserve_back`.** Both `pack_tile`s land and then `cb_out.push_back(2)` fires; every other producer in the op reserves first. The CB is sized `in0_tiles * out_single_tile_size` with `in0_tiles = block_size * 2 ≥ 2`, so a single row fits, but with `NCHt > 1` and a slow writer there is no back-pressure holding the packer off the tiles the writer has not yet drained.
9. **Unused kernel-side constants in the 2D reader.** `reader_layernorm_preallgather_2d.cpp:39-40` computes `TILE_SIZE` / `BF16_TILE_BYTES` from literals (`32 * 32`, `2 *`) and uses `BF16_TILE_BYTES` as the cross-core write size at `:116` — a hardcoded bf16 assumption for a CB (`c_16`) whose format is `cb_data_format`, which is `Float16_b` in this factory today but is not pinned there by anything the reader can see. `src0_tile_bytes` (`:38`) is computed and never used.

---

## Per-DeviceOperation attribution

| Field | `LayerNormPreAllGatherDeviceOperation` | `LayerNormPostAllGatherDeviceOperation` |
|---|---|---|
| Factories | 3 (1D, 2D, Welford) | 2 (default, Welford) — the default carries 1D and 2D configs |
| Device 2.0 | Yes | Yes |
| Feature scan | all N/A | all N/A |
| Concept / target | `descriptor` → `ProgramSpecFactoryConcept` | `descriptor` → `ProgramSpecFactoryConcept` |
| Offset base pointers | none | none |
| Tensor bindings | `input` C1 · `residual` C1 · `output` C1 · `recip_tensor` **clean** (Welford only) | `input` C1 · `stats` C1 · `gamma` C1 · `beta` C1 · `output` C1 |
| TensorAccessor 3rd arg | none | **2 sites, Class 2 → drop** (gamma, beta) |
| CB endpoints | self-loops; **multi-binding flag on `c_1`** (Welford factory only) | self-loops; **dead-CB drops**: `c_9` (both factories), `c_7` + `c_8` (Welford factory) |
| Borrowed kernel files | `rmsnorm_pre_allgather.cpp` (1D factory, `is_rmsnorm`) | `rmsnorm_post_allgather.cpp` (both factories, `is_rmsnorm`) |
| Semaphores | 1 plain `SemaphoreDescriptor` (2D factory only) | none |
| Blocking findings | none | none |

---

## Questions for the user

1. **Should the compute-thread CB API count toward the Device 2.0 boundary?** `normalization/kernel_util/compute/memory.h:30-31` calls the free function `get_tile_address(cb_id, tile_index)` (`tt_metal/hw/inc/api/compute/cb_api.h:172`) although both Device 2.0 wrappers expose a method form. This audit did **not** flag it, on the reading that Device 2.0 governs the data-movement surface (`api/dataflow/*`, `api/core_local_mem.h`, `api/tensor/*` per the migration guide's header list) and that the isolated-holdover test requires a wrapper in scope at the call site, which there is not. If the Device 2.0 owners consider compute-side CB-index free functions in scope, this becomes a one-line isolated holdover and the gate flips — worth a ruling, since the same helper is used by other normalization kernels.
2. **RMSNorm + gamma + beta drives an unallocated `c_13` (*Misc anomalies* #1).** Is that config considered supported? If yes it is a live defect for the ops team to fix ahead of the port; if no, a `TT_FATAL` in `validate_on_program_cache_miss` would make the boundary explicit and would also close the porter's question of what to bind.

---

## Recipe notes

1. **One documented column name has drifted, and the drift may be semantic rather than cosmetic.** `ttnn_op_porting_readiness.md` and the audit recipe both promise that "existing column names never change" and instruct the auditor to reference every column by header name. The override column is spelled **`Override runtime args method? (PD and legacy)`** in both docs (`ttnn_op_porting_readiness.md:57`, `metal2_audit.md:174,183`) but the live sheet heads it `Override runtime args method?` **`(PD only)`**. Recognizable enough that it cost nothing here — the value is `no` on every row either way — but two things are worth the maintainer's attention: an auditor doing an *exact* header match would not find it and could read that as a missing column, i.e. a broken sheet; and "PD only" vs "PD and legacy" is a change of *scope*, not just wording, so the recipe's careful distinction at `metal2_audit.md:183` (the same method name gating two different ways depending on whether the device-op is legacy or PD) may no longer describe what this column actually records. Worth confirming with the sheet owner which it is, then refreshing the docs. Two practical notes for the next auditor: the guarantee would be more useful phrased as "the leading phrase is stable; the parenthetical may be re-worded", and the sheet's headers embed **literal newlines** before their parentheticals (`'Custom hash \n(compute_program_hash)'`), so exact-string matching fails for a second, unrelated reason — match on the leading phrase.

   The sheet has also grown five columns the docs do not list — `Op Classification`, `Execution Model`, `Porting Target`, `Known op issues`, `Pointer patching perf issue?` — which is the expected additive change. Two of them earn a mention in the doc's column list because they bear directly on audit subjects: **`Porting Target`** states outright the target concept that *TTNN porting shape* currently has the auditor derive by hand from `Concept` + `Op-owned tensors?` (it read `ProgramSpecFactoryConcept` here, agreeing with the derivation), and **`Known op issues`** is a natural companion cross-check for the `Is safe to port?` verdict.
2. **A first-attempt fetch failure has no documented handling, and "retry" turned out to be the answer.** The first `download_file_content` call in this session was refused by the harness permission layer before reaching Drive; an identical retry a few turns later succeeded and returned the sheet. `ttnn_op_porting_readiness.md`'s *Troubleshooting* covers three failure modes that all assume the call *reached* Drive (auth expired, file not shared, tool schema not loaded), so a local refusal reads as unanticipated — and, because the recipe rightly forbids delegating the fetch to a subagent and forbids using a stale local copy, there is no fallback path to fall into. It cost a round trip to resolve. Suggested one-liner for *Troubleshooting*: **"Blocked/denied before reaching Drive (a harness permission refusal, not a Drive error) → retry the call once; if it still fails, ask the launcher to authorize it. Do not proceed on a stale CSV and do not delegate to a subagent."** Relatedly, the *TTNN factory concept prerequisite* has no state for "gate not yet evaluated" as distinct from "gate failed" — the *Status summary*'s `Overall` cell offers only `GREEN / RED`. Naming a third state, and saying explicitly that no brief is issued in it, would remove a judgment call from any auditor who hits an unavailable sheet and cannot recover.
3. **`Op-owned tensors?` versus a caller-supplied auxiliary tensor.** The Welford pre factory backs a CB with a tensor (`recip_tensor`) that the user creates through a separate API and passes in `tensor_args_t`. The code basis the recipe gives for op-owned tensors (a non-empty `buffers` vector on a returned `WorkloadDescriptor`) settles it cleanly as *not* op-owned, and *TensorParameter analysis*'s causal-link gate settles the binding as clean. But an auditor pattern-matching on "the op supplies a tensor the user never sees in the signature of the op it belongs to" could plausibly reach for the op-owned-tensor path and then the `WorkloadDescriptor` target concept. One sentence in *TTNN porting shape* — that a tensor arriving through `tensor_args_t` is a caller-supplied input regardless of who created it, and that op-owned means *allocated by the factory* — would foreclose that.
4. **The `(CB, config)` axis needs a per-*factory* dimension when one op has several factories over the same CB indices.** This op reuses `c_1`, `c_5`, `c_6`, `c_7`, `c_8`, `c_9` for entirely different purposes across its five factories — `c_1` is a reduce scaler in one factory, a transpose scratch in another, and the stats input in a third; `c_6` is `x²`, a Welford M2 spill, and reduced stats depending on the factory. *Classify per instantiation, not once for the op* covers this in spirit, and *Granularity — per binding, not per op* mentions per-factory variation for tensor bindings, but the CB-endpoints tables are keyed on `(CB, config)` alone, which collides badly here. Recommending the key be `(factory, CB, config)` whenever a directory holds more than one factory would make these censuses unambiguous to read.
5. **A remote NoC write into a *neighbour's* CB instance reads like face (a) and isn't.** `reader_layernorm_preallgather_2d.cpp:120-127` writes through `dfb_x2_merge_buf.get_write_ptr() + worker_offset` — a raw write, semaphore-coordinated, by a kernel that is not the receiving node's FIFO producer, which is face (a)'s signature almost word for word. The resolution is that the *node* being written to is a different node, so it adds no local endpoint there, and the local `get_write_ptr()` is only a peek. *CB endpoints* is explicit that the census is per node, so the answer is derivable — but the face-(a) recognition text ("a *second* kernel co-fills it via a raw write … coordinated by dedicated semaphores") describes this construct exactly, and a cross-core reduce/merge is a common enough shape that a false positive here is likely. A guard bullet under face (a) — *"if the write targets a remote node's instance, it is not a local endpoint on either node: the writer's `get_write_ptr()` on its own instance is a peek, and the receiving node's census is unchanged"* — would pay for itself.
6. **A dated triage doc that has gone stale in the *good* direction still needs a routing rule.** *TensorAccessor 3rd argument* tells the auditor to trust their own read over the doc, and names both drift directions — but the *Routing* text only covers the escalation case (Class 3/4/Special → the ops team). When the doc says Class 3 and the code now says Class 2 (here: validation was tightened so the bf8 branch the doc flagged is unreachable), the finding is a **correction the doc owner wants** and there is no instruction to send it anywhere. This audit routed it to the triage-doc owner on its own initiative. Worth making explicit, since silently down-classifying leaves the next auditor of a neighbouring op to redo the same analysis against the same stale row.
