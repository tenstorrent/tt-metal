# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/eltwise/binary_ng`

Audited against `origin/main` @ `fcbc2e31a65` (2026-08-03). All `file:line` references below are
line numbers **on `origin/main`**, not on the currently checked-out branch (which carries unrelated
Quasar work and has local drift in `binary_ng_device_operation.hpp`, `binary_ng_program_factory.cpp`,
and `ttnn/api/ttnn/operation_concepts.hpp`).

Identifying section — one DeviceOperation, one ProgramFactory:

- **`BinaryNgDeviceOperation`** (`device/binary_ng_device_operation.hpp:34`, `device/binary_ng_device_operation.cpp`)
  - `ProgramFactory` (`device/binary_ng_program_factory.cpp`) — the sole alternative in
    `program_factory_t = std::variant<ProgramFactory>` (`binary_ng_device_operation.hpp:145`)

Supporting host file: `device/binary_ng_utils.cpp` / `.hpp` (kernel-name → file-path table, `OpConfig`,
shard-volume helpers). No `*_nanobind.cpp` in the op directory (the op is reached through
`eltwise/binary`'s pybinds).

**Circular buffers declared (7 indices, ≤6 live in any one config):** `c_0` input a, `c_1` input b /
scalar tile, `c_2` output c, `c_3` a-intermediate (LHS activations only), `c_4` b-intermediate (RHS
activations only), `c_5` a-broadcast/`llk_post`, `c_6` b-broadcast/`llk_post` (`c_5`/`c_6` are mutually
exclusive per config). `c_0`/`c_1`/`c_2` become **borrowed-memory** CBs (`CBDescriptor::buffer`) when the
corresponding tensor is L1-sharded (`binary_ng_program_factory.cpp:1031`, `:1062`, `:1122`).

**Unreferenced kernel file (mentioned so it does not confuse a reader; contents not audited):**
`device/kernels/compute/eltwise_where_sfpu_scalar.cpp` is unreachable on `origin/main` — it is selected
only by `KernelName::ComputeScalar` **and** `is_where_op` (`binary_ng_utils.cpp:123-126`), but
`ComputeScalar` requires `!input_tensor_b.has_value()` (`binary_ng_program_factory.cpp:1135-1139`) while
every `WHERE_TTS`/`WHERE_TST` dispatch goes through the two-tensor `ttnn::prim::binary_ng` overload
(`eltwise/binary/binary.cpp:949-972`), and the tensor-scalar overload hard-codes `is_where_op = false`
(`binary_ng_device_operation.cpp:728`).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
*(The mandated provenance command — `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
— printed **nothing** from this checkout root: the recipe docs are not on the checked-out branch. The
line above is the same command run against the pinned recipe commit `4386dc456a1` on
`origin/akertesz/op-porting-recipe`, which is the guidance this audit ran against.)*

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng` |
| **Overall** | **RED** |
| **DOps / Factories** | `BinaryNgDeviceOperation` → `ProgramFactory` (single) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 33 reachable kernels are structurally Device 2.0 (`Noc`, `CircularBuffer`/`DataflowBuffer`, `TensorAccessor`); zero Device-1.0 idioms; zero unsanctioned CB-index free-function holdovers (see Gate detail for the `get_tile_hw(cb_id)` sanction-boundary note) |
| *Prereqs* — Cross-op escapes | Ok — every kernel `#include` resolves to `tt_metal/hw/inc/api/**` (donor class 1) or to binary_ng's own directory. Op owns all its kernels; borrows none |
| *Feature Support* — overall | GREEN — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **No: `override_runtime_arguments`** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes *(sheet's expert-judgment axis; not re-derived, per recipe)* |
| *TTNN Readiness* — Custom hash | No — no `compute_program_hash` override anywhere in the op directory |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes (gate → Metal 2.0 side; not yet supported)**: `ProgramFactory` — declared `device/binary_ng_device_operation.hpp:137`, defined `device/binary_ng_program_factory.cpp:1356` |
| *TTNN Readiness* — Pybind `create_descriptor` | No — no `create_descriptor` binding under `ttnn/cpp/ttnn-nanobind/` or `eltwise/` |
| *TTNN Readiness* — Op-owned tensors | No (`descriptor` concept cannot carry them; no `buffers` vector) |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` per the recipe's mapping — **but see Questions #1**: the op's `override_runtime_arguments` shape corresponds to `CustomProgramSpecFactoryConcept`, which the recipe does not cover |
| *Port work* — Offset base pointer | **none** — GREEN |
| *Port work* — Tensor bindings (per binding) | 3 bindings, config-dependent: `input_tensor_a` / `input_tensor_b` / output `c` are **Case 1** on every interleaved config and **clean** (borrowed-memory DFB) on their respective L1-sharded configs |
| *Port work* — TensorParameter relaxation | `OTHER(interleaved elementwise: shape rides runtime work-split)` — confirmed against the backdoor hash; **`dynamic_tensor_shape` is necessary but not sufficient** (rank invariant) — see Gate detail / Questions #2 |
| *Port work* — TensorAccessor 3rd arg | **Class 1** (dynamic page size) at 12 sites across 7 row-major kernels — drop the override **and** set `dynamic_tensor_shape`; no Class 3/4/Special anywhere → not a gate |
| *Port work* — CB endpoints | **deferred** — see the skipped-subject ledger below |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution
(self-loop, 1P+1C assignment, the multi-binding advanced option, or a dead-CB drop). This subject was
**not run** — see the ledger immediately below.

### Purely-informational subject ledger

The op is a whole-op RED with no portable subset, so `metal2_audit.md`'s **Red** outcome scoping rule
applies to the seven non-gating subjects. Disposition of each:

| Subject | Status |
|---|---|
| TTNN porting shape | run (cheap; recorded above and in Team-only) |
| TensorParameter relaxations | **run in full** — explicitly requested by the invoker, overriding the scoping rule (the sheet flags this op's relaxation as needing study) |
| TensorParameter analysis (tensor bindings) | **run in full** — explicitly requested by the invoker, overriding the scoping rule |
| CB endpoints | **skipped — whole-op RED, no portable subset; re-audit on unblock.** The per-`(CB, config)`-per-node census over 7 CB indices × 3 kernel roles × {interleaved, height/width/block-sharded, RM, tensor-scalar, ±LHS/RHS activation, 9 subtile-broadcast types} is exactly the acute case the recipe names, and it would be re-derived against changed code at re-audit. Not a clean result — unassessed |
| Out-of-directory coupling | run (turned out cheap: no borrowed kernels, all donors class 1) |
| RTA varargs | run — **none** |
| Incidental anomalies | run (opportunistic capture; see Misc anomalies) |

## Result

**RED → blocked on `override_runtime_arguments`**, routed to the **Metal 2.0 side**
(`FactoryConcept` + recipe TODO) — *not* TTNN.

`RED at op level; no portable subset.` The blocker is not one branch among siblings: the op has exactly
one program factory, and `override_runtime_arguments` is an unconditional static member of it. It is
also structurally load-bearing rather than incidental — `tensor_args_t::to_hash()`
(`binary_ng_device_operation.hpp:118-125`) hashes only the a/b **dtype and memory config**, deliberately
excluding every tensor **shape**, so one cached program is shared across arbitrarily-shaped calls and
*every* per-core work-split argument (`c_start_id`, per-core tile counts, D/N/C/Ht/Wt, `compute_tiles`,
`freq`/`counter`, strides, the packed scalar, the row-major page sizes) has to be re-derived per
dispatch. That is precisely what `override_runtime_arguments` does, from the same shared builder
`create_descriptor()` uses (`binary_ng_program_factory.cpp:406-418`, `:1356-1425`). Remove the hook
without replacing the mechanism and the op silently computes on a stale work-split; there is no
subset of configs for which it is unnecessary.

Every other gate is clear: Device 2.0 ✓ · Features ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓.
So this is a single-blocker RED against an otherwise-ready op — the gate lifts on the Metal 2.0 side
alone, with no prereq-migration work queued behind it.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED — failing conjunct: `override_runtime_args == "no"`.**
  Column `Override runtime args method? (PD and legacy)` == `yes` on a `descriptor`/PD op → routed to
  the **Metal 2.0 side** (`FactoryConcept` + recipe TODO), not to TTNN. Site: declared
  `device/binary_ng_device_operation.hpp:137`, defined `device/binary_ng_program_factory.cpp:1356`:

  ```cpp
  static void override_runtime_arguments(
      tt::tt_metal::Program& program,
      const operation_attributes_t& operation_attributes,
      const tensor_args_t& tensor_args,
      tensor_return_value_t& c,
      const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
  ```

  Its body does two things Metal 2.0 must absorb: (1) re-applies **all** per-core RTAs for **all**
  `num_cores_total` cores (work *and* noop) via `GetRuntimeArgs`, patching `Buffer*` slots to their
  current `->address()` (`:1386-1403`); (2) re-points the tensor-backed (globally-allocated) CBs by
  `CBIndex` via `UpdateDynamicCircularBufferAddress` (`:1412-1424`). The in-code rationale at `:1369-1371`
  is worth carrying to whoever owns the gate — it states the framework's own binding-address inference
  (`resolve_bindings`' `std::find`) is wrong for an in-place alias (`input_a == output`) or a mixed
  in-place/out-of-place reuse of one cache entry, and that re-deriving each slot from the tensor it
  belongs to is what makes those cases correct. Any Metal 2.0 replacement has to preserve that property,
  not just the arg refresh.

  All other conjuncts pass — see the cross-check table below. No conflict with the sheet, so no
  "spreadsheet is broken" routing.

  **Lightweight cross-check (sheet row relayed by the main session vs. code):**

  | Column (header name) | Sheet | Code evidence | |
  |---|---|---|---|
  | `Op` / `Device operation` / `Factory (variant)` | `eltwise/binary_ng` / `BinaryNgDeviceOperation` / `ProgramFactory` | `binary_ng_device_operation.hpp:34`, `:128` | ✓ |
  | **Factory-set match** | 1 row | `program_factory_t = std::variant<ProgramFactory>` (`hpp:145`) — exactly one factory in code, one row on the sheet | ✓ |
  | `Concept` | `descriptor` | `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`hpp:129-132`, `program_factory.cpp:833`) | ✓ |
  | `Custom hash (compute_program_hash)` | `no` | no `compute_program_hash` in the op directory (only comment references at `hpp:135`, `program_factory.cpp:409`, `:1363`) | ✓ |
  | `Backdoor custom hash (attribute_values/to_hash)` | `yes (tensor_args_t)` | `tensor_args_t::to_hash()` (`hpp:118-125`) + a masking `attribute_values()` (`hpp:89-110`) | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | no such hook on the device-op | ✓ |
  | `Override runtime args method? (PD and legacy)` | `yes` | `hpp:137` / `program_factory.cpp:1356` | ✓ → **GATE** |
  | `Pybind descriptor` | `no` | no `create_descriptor` binding under `ttnn-nanobind/` or `eltwise/`; no `*_nanobind.cpp` in the op dir | ✓ |
  | `Smuggled pointer` | `no` | `Buffer*` RTA entries are explicit `std::variant<uint32_t, Buffer*>` slots (`program_factory.cpp:401-403`), i.e. annotated, not smuggled | ✓ |
  | `Is safe to port?` | `yes` | **not verified** — expert-judgment axis, per recipe | — |
  | `Is able to port?` | `no` | derivation reproduced: every conjunct `yes`/`no`-correct except `override_runtime_args` | ✓ |
  | `TensorParameter relaxation` | `OTHER(interleaved elementwise: shape rides runtime work-split)` | confirmed — shape excluded from the key, work-split re-derived per dispatch | ✓ (with the rank caveat below) |
  | `Op-owned tensors?` | — | No: `descriptor` concept, no `WorkloadDescriptor`/`buffers` vector | ✓ |
  | `Secretly SPMD Workload?` | — | N/A (`Concept != WorkloadDescriptor`) | ✓ |

  **Cross-column invariants:** `get_dynamic_runtime_args == no` is consistent with a `descriptor`
  concept; `Op-owned tensors == no` is consistent with a `descriptor` concept. No inconsistency found.

- **Device 2.0 (every kernel used):** **GREEN.**

  Every kernel the factory can bind is structurally Device 2.0. Positive evidence, not just absence:
  data movement goes through `Noc noc;` + `noc.async_read(accessor, dfb_or_CoreLocalMem, len,
  {.page_id = …, .offset_bytes = …}, {})` / `noc.async_write(...)`; CBs through `CircularBuffer` /
  `DataflowBuffer` objects (`cb.reserve_back` / `.push_back` / `.wait_front` / `.pop_front` /
  `.get_write_ptr()` / `.get_read_ptr()` / `.get_tile_size()` / `.get_entry_size()`); addressing through
  `TensorAccessor` + `TensorAccessorArgs`. Zero hits across all 33 reachable kernels + 4 headers for:
  free-function `noc_async_read(` / `noc_async_write(` / `cb_reserve_back(` / `cb_push_back(` /
  `cb_wait_front(` / `cb_pop_front(` / `get_write_ptr(cb_id)` / `get_read_ptr(cb_id)` /
  `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*` /
  `get_local_cb_interface` / `evil_set_*_ptr` / raw semaphore addresses (the op uses **no** semaphores
  at all). The only free functions taking a CB index are `get_tile_size(cb_id)` (7 sites — explicitly
  **sanctioned** by the recipe's Green bullet) and `get_tile_hw(cb_id)` (7 sites, table below).

  **Not flagged as holdovers — `get_tile_hw(cb_id)`, 7 sites.** The recipe's holdover test is two-part:
  the Device-2.0 wrapper is in scope at the call site **and** a wrapper-method replacement exists. The
  first holds (a `CircularBuffer`/`DataflowBuffer` for the same index is constructed a few lines above);
  the second **does not at these call sites**. Every one is a `constexpr` initializer, and the wrapper
  method `CircularBuffer::get_tile_hw()` (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:114`) is a
  non-`constexpr` member on a non-`constexpr` object, so `constexpr uint32_t tile_hw = cb_src.get_tile_hw();`
  is ill-formed — the "1-line mechanical replacement" is not available. The free function
  (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:290`) is `constexpr inline`, which is why it is used
  here. The same kernels use the *method* form wherever the value is not needed at compile time (e.g.
  `cb_src.get_tile_size()` in the tiled readers), so the split is deliberate, not a missed migration.
  Recorded with `file:line` for the Device 2.0 team's awareness, and the sanction-list gap is logged in
  Recipe notes — **not** counted as a gate.

  | File (under `device/kernels_ng/dataflow/`) | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `reader_interleaved_rm_no_bcast.cpp` | 54 | `get_tile_hw(cb_id_src)` | `CircularBuffer cb_src` (:50) |
  | `reader_interleaved_rm_row_bcast.cpp` | 55 | `get_tile_hw(cb_id_src)` | `CircularBuffer cb_src` |
  | `reader_interleaved_rm_col_bcast.cpp` | 77 | `get_tile_hw(cb_id_src)` | `CircularBuffer cb_src` |
  | `reader_interleaved_rm_row_col_mixed_bcast.cpp` | 77 | `get_tile_hw(cb_id_src)` | `CircularBuffer cb_src` |
  | `reader_interleaved_rm_scalar_bcast.cpp` | 77 | `get_tile_hw(cb_id_src)` | `CircularBuffer cb_src` |
  | `reader_interleaved_rm_scalar_op.cpp` | 50 | `get_tile_hw(cb_id_src)` | `CircularBuffer cb_src` |
  | `writer_interleaved_rm_no_bcast.cpp` | 37 | `get_tile_hw(cb_id_out)` | `DataflowBuffer dfb_out` (:36) |

  **Kernel enumeration — every kernel the factory can bind (33 reachable files).** The factory sets
  `KernelDescriptor::kernel_source` at three sites — reader (`program_factory.cpp:1329`), writer
  (`:1166`), compute (`:1308`) — each through `get_kernel_file_path(KernelName, is_sfpu, is_where_op)`
  (`binary_ng_utils.cpp:79-147`). Selection: reader by
  `get_reader_kernel_name_and_defines` / `get_reader_rm_kernel_name_and_defines`
  (`program_factory.cpp:227-293`, dispatched at `:1148-1156`); writer by the same branch; compute by
  `BinaryNgKernelConfig` (`binary_ng_utils.cpp:34-70`) then possibly rewritten by
  `overwrite_compute_kernel_name_and_defines` (`program_factory.cpp:295-322`) when `use_llk_bcast`
  (`:1220-1289`). Everything in the op's `kernels/` + `kernels_ng/` trees is reachable except the one
  unreferenced file named in the identifying section.

  **Readers — 12 files** (`device/kernels_ng/dataflow/` unless noted; the RM group requires row-major
  a, b and c and forbids sharding, `program_factory.cpp:1128`):

  | Selector | File |
  |---|---|
  | `ReaderNoBcastNg` (tiled, b present, `NONE`) | `reader_interleaved_no_bcast.cpp` |
  | `ReaderRowBcastNg` (`ROW_A`/`ROW_B`) | `reader_interleaved_row_bcast.cpp` |
  | `ReaderColBcastNg` (`COL_A`/`COL_B`) | `reader_interleaved_col_bcast.cpp` |
  | `ReaderRowBColABcastNg` (`ROW_A_COL_B`/`ROW_B_COL_A`) | `reader_interleaved_row_col_mixed_bcast.cpp` |
  | `ReaderScalarBcastNg` (`SCALAR_A`/`SCALAR_B`) | `reader_interleaved_scalar_bcast.cpp` |
  | `ReaderRmNoBcastNg` | `reader_interleaved_rm_no_bcast.cpp` |
  | `ReaderRmRowBcastNg` | `reader_interleaved_rm_row_bcast.cpp` |
  | `ReaderRmColBcastNg` | `reader_interleaved_rm_col_bcast.cpp` |
  | `ReaderRmRowBColABcastNg` | `reader_interleaved_rm_row_col_mixed_bcast.cpp` |
  | `ReaderRmScalarBcastNg` | `reader_interleaved_rm_scalar_bcast.cpp` |
  | `ReaderRmScalarOpNg` (RM, tensor-scalar) | `reader_interleaved_rm_scalar_op.cpp` |
  | `ReaderNoBcast` (tiled, tensor-scalar) | `device/kernels/dataflow/reader_interleaved_no_bcast.cpp` |

  **Writers — 3 files:** `kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` (`WriterNoBcastNg`,
  tiled + b), `kernels_ng/dataflow/writer_interleaved_rm_no_bcast.cpp` (`WriterRmNoBcastNg`, all RM),
  `kernels/dataflow/writer_interleaved_scalar.cpp` (`WriterScalar`, tiled tensor-scalar).

  **Compute — 18 reachable files.** Each `KernelName` fans out over `is_where_op` / `is_sfpu`:

  | Selector | where | sfpu | fpu |
  |---|---|---|---|
  | `ComputeNoBcast` (`NONE`/`ROW_A`/`ROW_B`, or tensor-scalar) | `kernels/compute/eltwise_where_no_bcast.cpp` | `kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` | `kernels/compute/eltwise_binary_no_bcast.cpp` |
  | `ComputeBcast` (subtile bcast, software path) | `kernels/compute/eltwise_where_sfpu.cpp` | `kernels/compute/eltwise_binary_sfpu.cpp` | `kernels/compute/eltwise_binary.cpp` |
  | `ComputeScalar` (tensor-scalar) | *(unreachable — see identifying section)* | `kernels/compute/eltwise_binary_sfpu_scalar.cpp` | `kernels/compute/eltwise_binary_scalar.cpp` |
  | `ComputeRowBcastNg` (LLK row bcast) | `kernels_ng/compute/eltwise_where_sfpu_row_bcast.cpp` | `kernels_ng/compute/eltwise_binary_sfpu_row_bcast.cpp` | `kernels_ng/compute/eltwise_binary_row_bcast.cpp` |
  | `ComputeColBcastNg` (LLK col bcast; never where — `program_factory.cpp:1231-1237`, `:310`) | — | `kernels_ng/compute/eltwise_binary_sfpu_col_bcast.cpp` | `kernels_ng/compute/eltwise_binary_col_bcast.cpp` |
  | `ComputeScalarBcastNg` (LLK scalar bcast; never where) | — | `kernels_ng/compute/eltwise_binary_sfpu_scalar_bcast.cpp` | `kernels_ng/compute/eltwise_binary_scalar_bcast.cpp` |
  | `ComputeRowColBcastNg` (LLK mixed bcast) | `kernels_ng/compute/eltwise_where_sfpu_row_col_bcast.cpp` | `kernels_ng/compute/eltwise_binary_sfpu_row_col_bcast.cpp` | `kernels_ng/compute/eltwise_binary_row_col_bcast.cpp` |

  **Headers pulled in (all in-directory, all audited):**
  `kernels/dataflow/fill_tile_utils.hpp`, `kernels/compute/eltwise_utils.hpp`,
  `kernels/compute/eltwise_utils_common.hpp`, `kernels/compute/eltwise_utils_sfpu.hpp`.

  **Kernel-side generation note (not a gate, useful to the porter):** the migration is already partly
  done — 13 of the 33 kernels use the Metal 2.0 `DataflowBuffer` type, the other 20 still use
  `CircularBuffer`, and both `api/dataflow/circular_buffer.h` (17 includes) and
  `api/dataflow/dataflow_buffer.h` (13 includes) appear across the tree. That mix is legal at the
  Device 2.0 stage; a Metal 2.0 port converts the remainder per the kernel-side whitelist.

- **Feature compatibility:** GREEN — no gate fired. All four Appendix A entries are absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Zero hits for the type `experimental::GlobalCircularBuffer`, `CreateGlobalCircularBuffer`, the `CBDescriptor::global_circular_buffer` **field** (the arcane signal — every `CBDescriptor` literal, `program_factory.cpp:1023`/`:1040`/`:1054`/`:1071`/`:1086`/`:1100`/`:1114`, sets only `total_size`, `core_ranges`, `format_descriptors`, `buffer`), the 4-arg `experimental::CreateCircularBuffer(..., global_cb)`, `CircularBufferConfig::remote_index`, any `remote_cb_*` identifier, `remote_circular_buffer.h`, and both spellings of the GCB include. No `std::optional<const GlobalCircularBuffer>&` factory parameter |
  | CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` is never set on any `CBDescriptor` (default 0); no `set_address_offset`; no `cb_descriptor_from_sharded_tensor` / `cb_descriptor_from_overlapped_tensor` call. The three `UpdateDynamicCircularBufferAddress` calls (`program_factory.cpp:1418`, `:1420`, `:1422`) are the **three-argument** `(program, cb_id, Buffer&)` form — explicitly covered by this entry's false-positive guard, not the 4-arg offset overload. No runtime-team consultation triggered |
  | GlobalSemaphore | N/A | The op creates and uses **no semaphores of any kind** — zero hits for `Semaphore`, `semaphore`, `GlobalSemaphore`, `CreateGlobalSemaphore`, `global_semaphore.hpp` across host and kernel code |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent: `tensor_args_t` (`hpp:113-126`) is a fixed trio — `const Tensor&`, `std::optional<Tensor>`, `std::optional<Tensor>` — no `std::vector<Tensor>`. Kernel-level decider absent: every `get_compile_time_arg_val` index is a literal or a **constexpr** expression (`get_compile_time_arg_val(src_args.next_compile_time_args_offset())`, 9 sites) — args at constexpr offsets, even computed ones, are fixed-count. No runtime-varying CTA index, no kernel templated over a CTA-derived count |

- **CB endpoints (GATE-free):** **not assessed — skipped per the Red outcome scoping rule** (whole-op
  RED, no portable subset; re-audit on unblock). Nothing here could have blocked a Gen1 port, so no
  gate information is lost. The CB inventory is recorded in the identifying section so the re-audit has
  a starting point; the per-`(CB, config)`-per-node census, the hidden-second-writer hunt, and the
  dead-CB determination are all **unperformed**. Do not read this row as "all legal".

- **Offset base pointers:** **GREEN** — no address RTA folds a host-side offset into its base, and no
  fold has been split out either (there was never one).

  Scan performed over every address-valued arg, per the recipe (the doc's tables are a dated prior, not
  the authority): binary_ng is **not** in `2026-07-19_offset_base_pointers.md`'s Type-1/2/4 tables, and
  the code confirms clean — the fourth reconcile outcome, *no fold, op not in the tables*.

  - There is exactly **one** `->address()` expression in the entire op's host code:
    `program_factory.cpp:1393`, `static_cast<uint32_t>(std::get<tt::tt_metal::Buffer*>(slot)->address())`
    — a bare base with **no arithmetic**, inside the `apply` lambda that patches RTA slots on a cache hit.
  - Buffers enter the RTA lists as bare `Buffer*` values (`a.buffer()` at `:764`/`:792`, `b->buffer()` at
    `:756`/`:807`, `c.buffer()` at `:674`/`:690`/`:721`/`:738`) — pointer objects, never `address() + offset`.
  - Kernel side, each of `src_addr` / `src_addr_b` / `dst_addr` flows **directly** into a
    `TensorAccessor` constructor as its base, with no addition. Type 1 (raw NoC `.addr`) does not occur:
    no kernel uses an RTA-derived value as a raw NoC address. Type 2 (accessor-fed *offset* base) does
    not occur: the accessor bases are clean.
  - Type 3 (`address_offset`) — see the Appendix A row: never non-zero. Type 4 (`narrow` /
    `MeshBuffer::create(…, parent_base + offset)`) — no occurrence.

  All three bindings therefore reach `TensorParameter analysis` as clean bases.

- **TensorAccessor 3rd argument:** **GREEN as a gate** (no Class 3/4/Special) — but **12 Class-1 sites**
  are real port work, and the classification is *not* the usual "redundant, drop it".

  The op passes a 3rd argument at 12 accessor constructions in 7 **row-major** kernels; the 13 tiled-path
  accessors are 2-arg and unaffected.

  | File (`device/kernels_ng/dataflow/`) | Line(s) | Accessor / 3rd arg |
  |---|---|---|
  | `reader_interleaved_rm_no_bcast.cpp` | 69, 70 | `src` / `page_size_a`, `src_b` / `page_size_b` |
  | `reader_interleaved_rm_row_bcast.cpp` | 75, 76 | `src` / `page_size_a`, `src_b` / `page_size_b` |
  | `reader_interleaved_rm_col_bcast.cpp` | 99, 100 | `src` / `page_size_a`, `src_b` / `page_size_b` |
  | `reader_interleaved_rm_row_col_mixed_bcast.cpp` | 99, 100 | `src` / `page_size_a`, `src_b` / `page_size_b` |
  | `reader_interleaved_rm_scalar_bcast.cpp` | 99, 100 | `src` / `page_size_a`, `src_b` / `page_size_b` |
  | `reader_interleaved_rm_scalar_op.cpp` | 63 | `src` / `page_size_a` |
  | `writer_interleaved_rm_no_bcast.cpp` | 44 | `dst` / `full_page_size` |

  **Question 1 — sharded or interleaved? Interleaved, definitively.** The RM path asserts
  `TT_FATAL(!has_sharding, "Row-major binary_ng path does not support sharded tensors yet")`
  (`program_factory.cpp:1128`), so the realignment safety net is in play and only magnitude matters.

  **Question 2 — correct or wrong magnitude? Correct.** Each value is
  `align(buffer->aligned_page_size(), buffer->alignment())` — the host passes
  `a.buffer()->aligned_page_size()` / `b->buffer()->aligned_page_size()` /
  `c.buffer()->aligned_page_size()` as per-core RTAs (`program_factory.cpp:757-758`, `:783-784`,
  `:684`, `:731`) alongside the matching `alignment`, and the kernel re-aligns idempotently (e.g.
  `reader_interleaved_rm_no_bcast.cpp:64-65`). `aligned_page_size()` is explicitly a correct-magnitude
  source. Evaluated against Blackhole DRAM (64 B), the strictest target: no mis-addressing, so **no
  Class 3 and no Class 4**. Nothing here is a sharded raw-pack page or a sub-page base offset, so **no
  Special** either — the gate is clear.

  **Why Class 1 and not Class 2.** The values are `== aligned_page_size`, which reads like a clean
  Class-2 drop — but they are **load-bearing across cache hits**, which is the Class-1 condition. The
  op's cache key excludes every tensor shape (`hpp:118-125`), so one compiled program is reused across
  arbitrary row widths, while the compile-time `TensorAccessorArgs<...>::AlignedPageSize` that a 2-arg
  accessor would fall back to is frozen at the first-miss shape. The kernels say so in-source — the
  identical comment appears above each site, e.g. `reader_interleaved_rm_no_bcast.cpp:67-68`:

  > `// Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on program cache hits.`

  Dropping the override without a relaxation would strand a stale page size — a wrong magnitude the
  interleaved realignment cannot repair. This agrees with `2026-07-06_tensor_accessor_3rd_arg_triage.md`,
  which lists `binary_ng (RM readers)` as Class 1; the writer site (`writer_interleaved_rm_no_bcast.cpp:44`)
  is the same class and belongs on that row too (the table says "RM readers"). **Port action:** set
  `dynamic_tensor_shape` on the row-major `TensorParameter`s **and** drop all 12 overrides —
  cross-referenced to the relaxation subject below, whose caveat applies here as well.

## Port-work summary  *(mirrors the brief — no brief issued; RED)*

- **Tensor bindings** (3 bindings; classification varies **per config**, so recorded per config rather
  than flattened):

  | Binding | CB | Interleaved config | L1-sharded config |
  |---|---|---|---|
  | `input_tensor_a` | `c_0` | **Case 1** — `TensorAccessor(src_args, src_addr[, page_size])`, all access via the accessor | **clean** — borrowed-memory DFB (`CBDescriptor::buffer = a_buffer`, `program_factory.cpp:1031`); reader only `reserve_back`/`push_back`, constructs no accessor |
  | `input_tensor_b` | `c_1` | **Case 1** | **clean** — borrowed-memory DFB (`:1062`) |
  | output `c` | `c_2` | **Case 1** | **clean** — borrowed-memory DFB (`:1122`) |

  Evidence for the split, in one place: `kernels_ng/dataflow/reader_interleaved_no_bcast.cpp:46-59` —
  under `#if SRC_SHARDED` the reader does only `dfb_src.reserve_back(...)` / `dfb_src.push_back(...)`;
  under `#else` it builds `TensorAccessor(src_args, src_addr)`. Same shape for `SRC_SHARDED_B` and, in
  the writers, `DST_SHARDED` (`writer_interleaved_no_bcast.cpp:32-36`). The `SRC_SHARDED`/`DST_SHARDED`
  defines are set from `a_sharded`/`b_sharded`/`c_sharded` at `program_factory.cpp:1142-1147`.

  **No Case 2 anywhere** — no kernel takes an RTA-sourced base and does hand-rolled NoC arithmetic with
  it; every base goes through a `TensorAccessor`, so no `get_bank_base_address` bridge is needed. The
  raw `cb.get_write_ptr()` values that feed `fill_tile_utils.hpp` are **CB** pointers, not tensor bases —
  correctly outside this subject. The 18 compute kernels are out of scope (CB-only).

  **Delivery mechanism note for the porter:** the base addresses arrive by the recipe's `Buffer*`-binding
  form, not by `->address()`-in-an-RTA — the factory pushes `std::variant<uint32_t, Buffer*>` slots
  through `emplace_runtime_args` (`program_factory.cpp:401-403`, `:1341-1346`). Per the recipe that shape
  is *correct on cache hit today* (the framework registers `BufferBinding`s and patches them), so this is
  routine port work, **not** the silent-wrong hazard. The op additionally re-applies the same slots
  itself in `override_runtime_arguments` because the framework's address inference mis-handles the
  in-place alias case (`:1369-1371`) — that motivation belongs with the gate, above.

- **TensorParameter relaxation:** `OTHER(interleaved elementwise: shape rides runtime work-split)` —
  confirmed, and this is the sheet's `Porting Target: TBD (study relaxations)` cell. The study result:

  1. **The relaxation the sheet describes is real and exactly located.** The properties excluded from
     the cache key are the **tensor shapes** — of a, of b, and of the output. `tensor_args_t::to_hash()`
     (`hpp:118-125`) hashes only `input_tensor_a.dtype()`, `input_tensor_a.memory_config()`, and b's
     dtype/memory-config; `attribute_values()` (`hpp:89-110`) carries `memory_config`, `dtype`,
     `input_layout_a/b`, `output_layout`, `subtile_broadcast_type` and the three shard volumes, but no
     shape. So the hash excludes shape and *only* shape (plus the deliberately-runtime `scalar`, `rtol`,
     `atol` — all delivered as RTAs). That matches the listed relaxation: **no mismatch, the relaxation
     is not contradicted by the hash.**
  2. **`dynamic_tensor_shape` is necessary — including for the 12 Class-1 page-size sites.** It is the
     loosest relaxation that exists (`tt_metal/impl/metal2_host_api/program_run_args.cpp:46-61`,
     `:96-104`), and on an **interleaved row-major** `TensorParameter` it does exactly what the
     row-major kernels need: `tt_metal/impl/metal2_host_api/program_spec.cpp:2248-2261` sets
     `ArgConfig::RuntimePageSize` for `dynamic_tensor_shape && !is_sharded && layout == ROW_MAJOR`,
     moving the page size to a per-enqueue CRTA word. That is the framework-side replacement for the
     manual 3rd argument. On the **sharded** path it instead emits `rank` shape-in-pages CRTA words
     (`:2247`), and on the interleaved **tiled** path it is a pure host-side validation loosening
     (`:2243-2246`) — which is all the tiled path needs, since its page size is dtype-pinned and dtype
     *is* hashed.
  3. **But `dynamic_tensor_shape` is not sufficient: it pins the logical-shape rank, and binary_ng does
     not.** `program_run_args.cpp:55-61` FATALs when
     `runtime_spec.logical_shape().rank() != expected_spec.logical_shape().rank()` — "the rank must
     remain constant". binary_ng's cache key contains no rank: neither `attribute_values()` nor
     `to_hash()` carries it, `validate_on_program_cache_hit` explicitly tolerates differing and varying
     ranks (`binary_ng_device_operation.cpp:353-376` computes `larger_rank` from the *runtime* tensors),
     and the kernels take D/N/C/Ht/Wt as per-core RTAs precisely so rank can vary. Two dispatches that
     differ only in rank (e.g. `[32,32]` then `[1,1,32,32]`, same dtypes, same DRAM-interleaved memory
     config, both `SubtileBroadcastType::NONE`) share one cache entry today and would trip that FATAL
     after the port.

  **So the relaxation cannot be delivered by any existing `TensorParameter` flag alone.** This is a
  finding for the ops team **and** the framework/relaxation owner, and it is the substance behind the
  sheet's `TBD (study relaxations)`. Either a relaxation looser than `dynamic_tensor_shape` (rank-varying)
  is needed on all three bindings, or the op must start pinning rank in its cache key — a behaviour and
  program-cache-population change, and therefore ops-team work, not port work. Note this is a *second,
  independent* Metal-2.0-side need beyond the `override_runtime_arguments` gate: they are two faces of
  the same design (shape out of the key ⇒ per-dispatch work-split ⇒ per-dispatch args **and** a
  shape-loose binding), and clearing only one leaves the port blocked on the other.

- **TensorAccessor 3rd arg:** drop the redundant-looking-but-load-bearing page-size argument at the 12
  sites tabulated above, **and** set `dynamic_tensor_shape` on the row-major `TensorParameter`s (Class 1);
  the caveat in the relaxation item above gates this.

- **CB endpoints:** not assessed (skipped — see the ledger).

## Heads-ups  *(mirrors the brief — no brief issued; RED)*

- **CB endpoints (multi-binding shapes to watch):** not assessed (skipped). Flagging one thing the
  re-audit should not miss, since it is visible from the CB inventory: `c_5` / `c_6` are the
  `llk_post` intermediates written and read by the **compute** kernel alone in the LLK-broadcast
  configs (`kernels_ng/compute/eltwise_binary_row_bcast.cpp:50-60`: `exp_cb_llk_post.reserve_back(...)`
  then `pack_tile(0, cb_llk_post)` then consumed in the same kernel) — a one-toucher / self-loop
  candidate. That is a pointer for the census, **not** a census result.
- **Cross-op / shared kernels:**
  - **Borrowed kernel files: none.** binary_ng owns all 33 reachable kernel `.cpp` files; the factory
    instantiates nothing from a shared pool or another family.
  - **No `_metal2` fork exists** anywhere under `ttnn/cpp/ttnn/operations/eltwise/`. A port would create
    the first one for any file it forks.
  - **Reverse coupling — binary_ng is itself a donor, and this is the real coordination cost.**
    `eltwise/ternary` includes three of binary_ng's kernel headers from **13** of its own kernel files:
    `kernels/compute/eltwise_utils_common.hpp`, `kernels/compute/eltwise_utils_sfpu.hpp`, and
    `kernels/dataflow/fill_tile_utils.hpp`. Consumers, as a **sunset and coordination list — not
    authorization to convert anything in place**: `ternary/device/kernels/compute/ternary_sfpu_{no_bcast_ttt,
    no_bcast_tts_tst,row_bcast_ttt,col_scalar_bcast_ttt,col_scalar_bcast_tts_tst}.cpp` and
    `ternary/device/kernels/dataflow/{ternary_reader_colbcast_ttt,ternary_reader_row_col_bcast_ttt,
    ternary_reader_rowbcast_ttt,ternary_reader_scalar_ttt,tst_tts_reader_scalar_bcast,
    tts_tst_reader_col_bcast,tts_tst_reader_row_bcast,tts_tst_reader_row_col_bcast}.cpp`. A Metal 2.0
    rewrite of any of those three headers in place would break `ternary`.
  - **Negative pointer (a wrong turn worth saving the porter):** `ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/`
    contains copies of this op's kernels and a `binary_ng_metal_v2_factory.cpp`. Per the recipe that
    directory is **out of bounds** — those are hacky whole-op pre-port copies, they are **not** an
    existing `_metal2` fork to reuse, and they are not a naming or precedent source. They were not read
    for this audit and contributed no finding.
- **RTA varargs:** **none.** Every dataflow kernel reads a **fixed** run of RTAs via a running `index++`
  at the top of `kernel_main` (e.g. `reader_interleaved_rm_no_bcast.cpp:14-42`, 26 reads; the tiled
  readers use literal indices, `reader_interleaved_no_bcast.cpp:13-33`), which the recipe's non-signal
  rule covers explicitly — legacy positional plumbing that dissolves into named args. Zero counted loops
  over `get_arg_val` / `get_common_arg_val`, zero data-selected reads (no `get_arg_val(k)` where `k` came
  from another argument). Compute kernels read fixed indices or define-supplied constant indices
  (`ISCLOSE_RTOL_RT_ARG_IDX` = `"3"`, `ISCLOSE_ATOL_RT_ARG_IDX` = `"4"`,
  `QUANT_ZERO_POINT_RT_ARGS_IDX` = `"3"`, set at `program_factory.cpp:911-912`, `:979`). The porter can
  name every RTA.

## Team-only

- **Out-of-directory coupling & donor shape:** op-level roll-up **`✓ clean`** — no per-call detail table
  is owed. Every `#include` in every audited kernel resolves either into
  `tt_metal/hw/inc/api/**` (`api/dataflow/{dataflow_api,noc,circular_buffer,dataflow_buffer}.h`,
  `api/tensor/noc_traits.h`, `api/alignment.h`, `api/core_local_mem.h`, `api/compute/**`) — **donor class
  1**, LLK/HAL, no concern — or into binary_ng's own directory. There is **no**
  `ttnn/cpp/ttnn/kernel_lib/`, `ttnn/cpp/ttnn/kernel/`, `operations/kernel_helper_functions/`, in-family
  or cross-family kernel donor. Consequently no shape falls in the ⚠/✗/⭐ rows: no `uint32_t sem_id`, no
  `uint32_t`/`uint64_t` sem address, no `TensorAccessorArgs<N>` or CTA-offset-NTTP donor parameter, no
  old-style addr-gen (Shape 4), no `CircularBuffer&` donor parameter. The `uint32_t cb_id` arguments the
  compute kernels hand to LLK APIs (`binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out)`,
  `unary_bcast_init<...>(cb_bcast, cb_llk_post)`, `pack_tile(0, cb_llk_post)`,
  `pack_reconfig_data_format(...)`) are the `✓ OK` row — `dfb::name`'s constexpr cast covers both runtime
  and template-parameter position. Borrowed kernel files: none (see Heads-ups). *Host-side* out-of-directory
  dependencies exist and are ordinary: `eltwise/binary/common/binary_op_utils.hpp`,
  `eltwise/unary/common/unary_op_utils.hpp`, `ttnn/operations/cb_utils.hpp` — not kernel coupling.
- **Relaxation candidates** (mined from the backdoor hash on a gated op) — **FALLIBLE — candidates to
  verify, default strict:**
  - **Rank-varying shape relaxation** (a new flag, looser than `dynamic_tensor_shape`) — the concrete
    need established above. Highest-value candidate here because it is *required*, not merely enabled.
  - `equal_nan` is hashed only for `ISCLOSE` (`hpp:106`) and `post_activations` is masked for
    where/quant ops (`hpp:94`) — narrow, deliberate exclusions that a relaxation model of this op should
    know about. The `post_activations` mask has a soundness question; see Misc anomalies.
- **TTNN factory analysis** (sheet-derived facts, with `file:line` evidence):
  - Current concept **`descriptor`** — `create_descriptor` returning `ProgramDescriptor`
    (`hpp:129-132`, `program_factory.cpp:833`). Single factory (`hpp:145`).
  - **Op-owned tensors: none** — structurally impossible on the `descriptor` concept; no
    `create_workload_descriptor`, no `buffers` vector.
  - **MeshWorkload need: none.** Note the op *does* already carry a
    `const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate` parameter on
    `override_runtime_arguments` (`hpp:142`), unnamed and unused in the definition
    (`program_factory.cpp:1361`) — mesh-uniform behaviour, i.e. SPMD, consistent with the sheet's
    `Execution model: SPMD`.
  - **Pybind `create_descriptor`: no.** No other risky pybind of internals (no `*_nanobind.cpp` in the
    op directory at all).
  - **Custom hash: no** (`compute_program_hash` is never overridden) — **but the backdoor is active**:
    `tensor_args_t::to_hash()` (`hpp:118-125`) plus a value-masking `attribute_values()` (`hpp:94`,
    `:106`) achieve the same effect. `ttnn_factory.md`'s "Delete a custom `compute_program_hash`" step
    has no analogue for this shape; a port that deletes nothing still inherits a shape-loose key. Flagged
    to the recipe maintainer in Recipe notes.
  - **`get_dynamic_runtime_args`: no.**
  - **`override_runtime_arguments`: yes** — the gate; detail above.
  - **Target concept:** the recipe maps `descriptor` → `ProgramSpecFactoryConcept`, and
    `ttnn_factory.md` describes only that concept. That mapping is unreachable for this op:
    `ProgramSpecFactoryConcept` is defined with `&& !detail::HasSpecRuntimeArgsOverride<T>`
    (`ttnn/api/ttnn/operation_concepts.hpp:119-121`), i.e. it *excludes* a factory with a
    `ProgramRunArgs`-returning `override_runtime_arguments`. Routed as Questions #1 — **surfaced, not
    resolved.**

## Misc anomalies  *(team-only, non-gating; opportunistic capture — routes to the ops team, the port does not act on these)*

1. **Dead RTA slot in both tiled writers.** `build_per_core_runtime_args` appends a trailing `0u` to the
   tiled writer arg list — `program_factory.cpp:690` (b present) and `:748` (tensor-scalar) — that no
   kernel reads: `kernels_ng/dataflow/writer_interleaved_no_bcast.cpp:13-22` consumes exactly 10 args
   (indices 0-9), and `kernels/dataflow/writer_interleaved_scalar.cpp:14-23` exactly 11 (0-10). The slot
   is nonetheless counted in the hardcoded noop-core length `writer_len` (`:616`), so removing it means
   updating both.
2. **Read of an uninitialized enum member (value discarded).** `BinaryNgKernelConfig`'s constructor
   (`binary_ng_utils.cpp:34-70`) assigns `reader_kernel` and `writer_kernel` **only** in the
   `SubtileBroadcastType::NONE` case; for the other eight cases both members stay indeterminate.
   `program_factory.cpp:1137` then reads `kernel_config.writer_kernel` whenever
   `input_tensor_b.has_value()` — indeterminate for any non-`NONE` broadcast. The value is harmlessly
   overwritten at `:1151` or `:1155` before use, so there is no observable misbehaviour, but it is an
   indeterminate-value read (UB; MSan-visible). Relatedly, `BinaryNgKernelConfig::writer_kernel` is
   **never** load-bearing — every path that reads it overwrites it — and `reader_kernel` is load-bearing
   only for tiled tensor-scalar. The struct's two kernel-name members are effectively dead weight.
3. **Duplicated, unchecked runtime-arg-length invariant.** The noop-core arg lengths are hardcoded
   literals — `reader_len = row_major_inputs ? 26 : 21`, `writer_len = row_major_inputs ? 14 : (b ? 11 : 12)`,
   `compute_len = ISCLOSE ? 5 : 4` (`program_factory.cpp:615-617`) — and must match the work-core lists
   built 60-200 lines below. All five currently match (verified by hand), but nothing enforces it; a
   future arg addition that misses these constants changes the per-kernel slot count when a core flips
   between work and noop across cache hits, which is exactly the failure the surrounding comment
   (`:611-614`) exists to prevent. A `static`/`TT_ASSERT` cross-check, or deriving the lengths from the
   builder itself, would close it.
4. **Candidate program-cache hash hole: output dtype from a preallocated `output_tensor`.**
   `create_output_tensors` returns a caller-supplied `output_tensor` verbatim
   (`binary_ng_device_operation.cpp:482-483`), and `tensor_args_t::to_hash()` (`hpp:118-125`) does not
   hash the optional output tensor at all — the output dtype enters the key only through
   `attributes.dtype` (via `get_dtype()`, `:227-229`, which falls back to `input_dtype`). The
   consistency check `*attributes.dtype == output_tensor->dtype()` exists but is guarded by
   `attributes.dtype.has_value()` (`:291-293`) and lives in `validate_on_program_cache_miss` only —
   so it is skipped entirely when `output_dtype` is omitted while `output_tensor` is supplied, and it
   is never re-run on a cache hit. Meanwhile `create_descriptor` derives the `c_2` CB data format,
   `fp32_dest_acc_en` (`program_factory.cpp:1178-1185`) and the typecast post-activation
   (`:953-958`) from `c.dtype()`. Layout and memory config are safe by comparison — both are taken
   *from* the output tensor at invoke time and hashed (`:559`/`:618`, `:705`/`:703`). Worth confirming
   whether any public wrapper can reach the unguarded combination.
5. **Masked-but-define-affecting attribute.** `attribute_values()` substitutes an empty
   `post_activations` into the key when `is_where_op || is_quant_op` (`hpp:94`), yet
   `create_descriptor` still feeds the real `post_activations` to `add_activation_defines(...,
   "POST", ...)` (`program_factory.cpp:982`, `:985`) on paths where the single-activation fast path
   at `:972-980` does not apply (e.g. `op_config.postprocess` prepended at `:946` making
   `post_activations.size() == 2`). For quant ops the factory constrains the shape
   (`TT_FATAL` at `:649-653`); for **where** ops nothing forbids a non-empty `post_activations`
   (no validation in `validate_on_program_cache_miss`), so two where dispatches differing only in
   `post_activations` would hash identically while needing different compiled kernels. Latent —
   reachable only if a caller passes post-activations to a where op — but the masking and the define
   generation should agree.
6. **Inconsistent include spelling in a downstream consumer** (noticed while inventorying the reverse
   coupling): `eltwise/ternary/device/kernels/compute/ternary_sfpu_row_bcast_ttt.cpp:16-17` includes
   binary_ng's headers via `ttnn/cpp/ttnn/operations/...` while its twelve siblings use
   `ttnn/operations/...`. Cosmetic, but it will make a grep-based sunset sweep of the shared headers
   miss that file.

## Questions for the user

1. **`CustomProgramSpecFactoryConcept` exists on `origin/main` and looks shaped for exactly this op, but
   the recipe does not know about it — surfacing, not clearing.** `ttnn/api/ttnn/operation_concepts.hpp:132-134`
   defines a Metal 2.0 factory concept whose doc comment (`:124-131`) reads: *"Spec factory that
   additionally re-applies per-dispatch runtime args on every cache hit: its `override_runtime_arguments`
   returns a `ProgramRunArgs` applied via `UpdateProgramRunArgs` (the spec-path analog of the
   ProgramDescriptor path's `get_dynamic_runtime_args`)"* — with a signature
   `(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&, const std::optional<ttnn::MeshCoordinate>& = std::nullopt)`
   that differs from binary_ng's PD-era hook only in returning `ProgramRunArgs` instead of `void` and
   taking no `Program&`. The variant validator knows it (`:181-182`), and framework plumbing exists
   (`ttnn/api/ttnn/device_operation.hpp:243`, `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:955`,
   `ttnn/api/ttnn/metal_v2_artifacts.hpp:18`). **However:** it has **zero** op-side consumers — no file
   under `ttnn/cpp/ttnn/operations/` references it — and the recipe corpus mentions it **zero** times
   (`ttnn_factory.md` states flatly that "A Metal 2.0 op factory satisfies **`ProgramSpecFactoryConcept`**",
   which by its own definition at `operation_concepts.hpp:119-121` *excludes* a factory with this hook).
   Per the recipe's operating principle I have **not** treated this as support and have **not** cleared
   the gate. Routing the question to the Metal 2.0 side: is `CustomProgramSpecFactoryConcept` the
   intended landing place for this op, and if so, does the recipe TODO reduce to documenting it?
2. **Which relaxation should binary_ng target, given that `dynamic_tensor_shape` pins rank?** See the
   relaxation item in Port-work summary. The op's cache key does not pin logical-shape rank and its
   cache-hit validation explicitly tolerates rank variation (`binary_ng_device_operation.cpp:353-376`),
   while `dynamic_tensor_shape` FATALs on a rank change (`program_run_args.cpp:55-61`). This needs the
   relaxation owner (framework/Audrey) and the ops team: a rank-varying relaxation, or a hash change on
   the op side. It is a **second** Metal-2.0-side prerequisite, independent of the
   `override_runtime_arguments` gate; clearing one alone will not unblock the port.

## Recipe notes

- **The report was written outside the recipe's mandated location, at the invoker's explicit direction.**
  `metal2_audit.md` → *Output: the two documents* requires `METAL2_PREPORT_AUDIT.md` in the **op's root
  directory** (`ttnn/cpp/ttnn/operations/eltwise/binary_ng/`). The invoker specified
  `/workspace/.link_to_claude/plans/METAL2_PREPORT_AUDIT.md` instead, and this audit was explicitly
  analysis-only (no source-tree modification). Disclosed because a downstream consumer scanning op
  directories for audit artifacts will not find this one.
- **The Red-outcome scoping rule was partially overridden, also at the invoker's direction.** The rule
  says to skip all seven purely-informational subjects on a whole-op RED with no portable subset. The
  invoker explicitly required **TensorParameter analysis** and **TensorParameter relaxations**, so those
  were run in full; only **CB endpoints** was skipped. Full ledger under the status summary. Worth the
  maintainer's attention: for *this* op the override was clearly right — the relaxation subject produced
  the audit's second-most-load-bearing finding (Questions #2), and it is *only* reachable by running a
  subject the rule would have skipped. The rule's premise ("no brief will be issued, so the detail is
  unread") does not hold when a relaxation study is itself an input to clearing the gate. Consider
  carving relaxation study out of the skip set when the sheet's `Porting Target` is `TBD (study
  relaxations)`.
- **Device 2.0 sanctioned-free-function list is incomplete in a way that forced a judgement call.** The
  Green bullet names exactly two sanctioned CB-index free functions (`get_tile_size(cb_id)`,
  `get_local_cb_interface(cb_id)`). This op uses a third, `get_tile_hw(cb_id)`, at 7 sites — structurally
  identical to `get_tile_size` (both `constexpr inline` in `dataflow_api.h`, both mirrored as pure
  forwarding wrapper methods at `circular_buffer.h:113-114`). By the *letter* of the two-part holdover
  test it would be a holdover; I resolved it as **not** a holdover on the ground that all 7 sites are
  `constexpr` initializers where the non-`constexpr` wrapper method is not a legal replacement, so the
  test's second clause ("a wrapper-method replacement exists") fails. Two suggestions: (a) add
  `get_tile_hw(cb_id)` and `get_dataformat(cb_id)` to the sanctioned list, or state the sanction as a
  rule ("any pure forwarder to a wrapper method") rather than an enumeration; (b) add an explicit
  **constexpr carve-out** to the holdover test — "a wrapper-method replacement does not exist where the
  call site requires a constant expression" — because that reasoning is not currently written down and a
  hurried auditor would RED the gate here.
- **The recipe's custom-hash model does not cover the `tensor_args_t::to_hash()` backdoor, and the
  relaxation subject's stated precondition is therefore wrong for this op.** *TensorParameter
  relaxations* asserts: "A relaxation-bearing op **has a custom hash** (the relaxation *is* the hash
  excluding the relaxed property from the cache key), and the TTNN factory concept prerequisite
  currently gates custom-hash ops — so today a real relaxation value co-occurs with a gate and this
  subject rarely fires." binary_ng is a counterexample: `Custom hash == no` (no `compute_program_hash`
  override at all), yet it carries a real, substantial relaxation implemented through
  `tensor_args_t::to_hash()` plus value-masking in `attribute_values()`. The readiness sheet already
  models this — it carries a `Backdoor custom hash (attribute_values/to_hash)` column, `yes
  (tensor_args_t)` for this row — but neither `metal2_audit.md` nor `ttnn_op_porting_readiness.md`'s
  column legend mentions that column, and `ttnn_factory.md`'s "Delete a custom `compute_program_hash`"
  step has no analogue for it. Concretely: the recipe's `Custom hash` cross-check ("grep the device-op
  for a `compute_program_hash` override") returns a clean `no` here while the op's cache key is in fact
  shape-loose. Suggest (a) documenting the backdoor column in the readiness legend, (b) having the
  relaxation subject key on *either* hash mechanism, and (c) adding the `to_hash()`/`attribute_values()`
  masking shape to the cross-check bullet so it is not missed on an op where the sheet lacks the
  backdoor column.
- **Relayed column label differed from the checked-in legend (minor).** The sheet row relayed to me
  labelled the gate column `Override runtime args method? (PD only)`; the legend in
  `ttnn_op_porting_readiness.md` names it `Override runtime args method? (PD and legacy)`. Values agree
  and the derivation is unaffected, so this is not treated as a sheet-broken conflict — but since the
  standing rule is "reference every column by header name", one of the two spellings should be corrected.
  I could not check which is live: the recipe forbids fetching the sheet from a subagent, and I did not.
- **Two things worked well and are worth keeping.** (a) The instruction to run every *gate*-bearing
  subject to completion after the first RED paid off directly here: the gate that failed is a
  one-line-of-code fact, and everything expensive (Device 2.0 over 33 kernels, the offset scan, the
  12-site 3rd-arg classification) came back **clean** — so this RED reports as "one blocker, Metal 2.0
  side, nothing queued behind it" instead of "unknown depth". Short-circuiting would have lost that.
  (b) The *TensorAccessor 3rd argument* subject's warning that the drop "is not the mechanical drop it
  looks like" was exactly right for this op: all 12 values are literally `aligned_page_size()`, which
  reads as a textbook Class-2 no-op drop, and only the cache-key analysis reveals them as Class-1
  load-bearing. The in-kernel comments made it decidable; without the subject's framing an auditor would
  plausibly have written "Class 2, drop" and handed the porter a silent mis-addressing bug.
