# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward`

Single device-operation directory, single program factory:

- **`MorehNllLossBackwardDeviceOperation`** (`device/moreh_nll_loss_backward_device_operation.{hpp,cpp}`)
  - `Factory` (`device/moreh_nll_loss_backward_program_factory.cpp:691`) — the only factory; `program_factory_t = std::variant<Factory>`.

**One factory, three rank-dispatched code paths.** `Factory::create_descriptor` branches on `input_grad.logical_shape().rank()` into three free functions in the same file — `moreh_nll_loss_backward_impl_2d` (`:46`), `_impl_3d` (`:259`), `_impl_4d` (`:474`). These are **configs of one factory**, not separate factories: the sheet carries one row, and one port converts all three. Findings below are attributed per rank path where they differ — and two do.

All **5** kernel files in `device/kernels/` are referenced; none is dead. The writer and the compute kernel are shared by all three rank paths:

| Rank path | Reader | Writer | Compute |
|---|---|---|---|
| 2d (`:46`) | `reader_moreh_nll_loss_backward_2d.cpp` | `writer_moreh_nll_loss_backward.cpp` | `moreh_nll_loss_backward_kernel.cpp` |
| 3d (`:259`) | `reader_moreh_nll_loss_backward_3d.cpp` | *(same)* | *(same)* |
| 4d (`:474`) | `reader_moreh_nll_loss_backward_4d.cpp` | *(same)* | *(same)* |

Config axes that matter below: rank ∈ {2d, 3d, 4d} × `WEIGHT` (optional `weight_tensor`) × `DIVISOR` (optional `divisor_tensor`) × `fp32_dest_acc_en` (formats only).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `38da2cdbd29 2026-08-13 docs(metal_2.0): stop gating on a pybound create_descriptor, and drop a stale reference port`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward` |
| **Overall** | **GREEN** — every gate-bearing subject cleared |
| **DOps / Factories** | `MorehNllLossBackwardDeviceOperation` → `Factory` (3 rank paths: 2d / 3d / 4d) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — own kernels and both shared-pool donors are structurally Device 2.0 (`Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem`, `UnicastEndpoint`). One judgment call recorded below (`get_dataformat(cb_id)`). |
| *Prereqs* — Cross-op escapes | **Ok** — `✓ clean`; no borrowed kernel files, no donor needing conversion, no external borrower |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | **Ok** — the single CTA read is at literal offset `0` |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (launcher-supplied; cross-check clean — see *Gate detail*) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — not a `WorkloadDescriptor` op |
| *TTNN Readiness* — Custom hash | **No** (not a gate either way; none present) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | **No** (not a gate; selects the target concept — base concept applies) |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** (not a gate; nothing for the port to delete) |
| *TTNN Readiness* — Op-owned tensors | **No** — `descriptor` concept; no `buffers` vector |
| *TTNN Readiness* — TensorParameter relaxation | **`none`** (launcher-supplied, verbatim) → clears |
| *TTNN Readiness* — `Known op issues` | **Empty** (launcher-supplied) → no blocker |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | **none** — no address RTA at all; bases arrive as `Buffer*` bindings, zero host arithmetic |
| *Port work* — Tensor bindings (per binding) | **Case 1 × 5** — `target`, `output_grad`, `weight`, `divisor`, `input_grad`; two are conditional |
| *Port work* — TensorAccessor 3rd arg | **none** — all 13 construction sites are 2-arg |
| *Port work* — CB endpoints | **legal + self-loop + 3 dead-CB drops** — 4 plain 1P+1C, 4 self-loops, and **three config-scoped dead CBs** (`c_8` in 2d; `c_25` / `c_26` when `DIVISOR` is off). No multi-binding. |

**CB endpoints** are dispositions, not gates. Record the disposition per `(CB, config)`, and classify per instantiation — on this op the disposition genuinely flips with config, which is where the three dead CBs come from.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, same directory).

Every gate clears: Device 2.0, feature compatibility, the TTNN factory-concept prerequisite (`Is able to port?` = `yes`), the `TensorParameter relaxation` gate (`none`), offset base pointers, and the `TensorAccessor` 3rd argument. The kernels are already fully Device 2.0, all five tensors reach their kernels through a `TensorAccessor` built from a framework-patched `Buffer*` binding, there are no semaphores, and no Appendix A feature is touched.

The port is more involved than the gate tally suggests, and the reason is **optionality**. Two of the five tensors are optional, and the factory expresses their absence by *skipping the CB allocation* and passing `nullptr`/placeholder args. That interacts with Metal 2.0's structural rule that a DFB must have at least one binding: **three CBs are allocated in configs where no kernel touches them**, which a Metal 2.0 spec cannot express. Two of them (`c_25`, `c_26`) need their allocation made conditional on `divisor_has_value`; the third (`c_8`) is unconditionally vestigial and is dropped. The brief carries the specifics. Nothing here blocks — it is all port work, but it is the part a porter working one file at a time would most easily get wrong.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The sheet's cell reads `yes` (launcher-supplied), which clears **this prerequisite and nothing else** — the remaining gate-bearing subjects are assessed independently below.

  The readiness sheet could not be fetched from this session: the claude.ai Google Drive connector is unauthorized and the session is non-interactive, so `mcp__claude_ai_Google_Drive__download_file_content` is not present to call, and the recipe forbids delegating the fetch to a subagent. Because the recipe now makes two further cells decisive and explicitly non-derivable — `TensorParameter relaxation` ("Read the cell; do not re-derive it") and the free-text `Known op issues` — I asked the launcher for all three rather than inferring any of them. Answers: `Is able to port?` = `yes`, `TensorParameter relaxation` = `none`, `Known op issues` = empty. This is **not** the recipe's *spreadsheet-broken* case: the sheet is neither wrong nor silent for this op, only unreachable from here.

  Per the current recipe I do **not** reproduce or recompute the sheet's derivation. The lightweight cross-check of the cheaply-checkable columns is clean:

  | Column | Sheet (supplied / expected) | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | `static ProgramDescriptor create_descriptor(...)` on `Factory`, `device/moreh_nll_loss_backward_device_operation.hpp:38-43`; defined at `device/moreh_nll_loss_backward_program_factory.cpp:691` | ✓ |
  | `Custom hash` | `no` | No `compute_program_hash`, and no backdoor `attribute_values` / `to_hash`, anywhere in the op | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | Hook absent from `MorehNllLossBackwardDeviceOperation` (`...device_operation.hpp:46-52` declares only `validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`) | ✓ |
  | `Override runtime args method?` | `no` | Method absent from device-op and factory → target concept is the base `ProgramSpecFactoryConcept`, not `CustomProgramSpecFactoryConcept` | ✓ |
  | `Pybind descriptor` | `no` | `moreh_nll_loss_backward_nanobind.cpp:23-36` — a single `ttnn::bind_function<"moreh_nll_loss_backward">`; no `create_descriptor` binding, no factory/device-op internals. Nothing for the port to delete, so no user-visible API change from this column. | ✓ |
  | `Secretly SPMD Workload?` | N/A | Not a `WorkloadDescriptor` op | — |
  | `Op-owned tensors?` | `no` | `descriptor` concept; output is an ordinary TTNN tensor (`create_output_tensors`, `...device_operation.cpp:90-98`) | ✓ |
  | **Factory-set match** | 1 row expected | Code has exactly one factory (`program_factory_t = std::variant<Factory>`, `...hpp:45`). The three rank impls are internal branches of that one factory, **not** additional factories — a sheet carrying three rows here would be the staleness signal, not a match. | *see Questions* |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no`, and `Op-owned tensors?` is `no` as the `descriptor` concept requires. Per the recipe I did **not** verify `Is safe to port?` — the readiness doc now records that column as stale and states the audit does not read it.

- **TensorParameter relaxation: GREEN.** The cell reads **`none`** verbatim (launcher-supplied) — the only value that clears, and the only value that reaches a porter brief. Not re-derived: the recipe assigns that analysis to the ops team, and this op has no custom hash from which a candidate could even be mined.

- **Device 2.0 (every kernel used): GREEN.** All 5 kernels are structurally Device 2.0, and so is every donor function they call.

  - **Readers** (`reader_moreh_nll_loss_backward_{2d,3d,4d}.cpp`) — `TensorAccessor` for all tensor addressing; `DataflowBuffer` objects with `reserve_back`/`push_back`/`wait_front`/`pop_front` methods; `CoreLocalMem<volatile T>` typed L1 pointers seeded from `dfb.get_read_ptr()` / `dfb.get_write_ptr()` **methods**; NoC traffic delegated to the donor's `Noc`-based `read_tile` / `read_line`. No `noc_async_read`, no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen*`, no `cb_reserve_back`/`cb_push_back`, no raw semaphore addresses.
  - **Writer** (`writer_moreh_nll_loss_backward.cpp`) — `Noc noc;` + `noc.async_write(dfb, accessor, bytes, {.offset_bytes = 0}, {.page_id = i})` + `noc.async_write_barrier()`; `dfb_input_grad_obj.wait_front`/`pop_front`.
  - **Compute** (`moreh_nll_loss_backward_kernel.cpp`) — `DataflowBuffer` objects for all FIFO traffic; LLK compute calls take raw CB ids by design.
  - **Donors** — `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (`read_tile`, `read_line`, `fp32_dest_acc_cast`, `get_tilized_idx`) and `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` (`copy_tile_init_with_dt`, `pack_tile_with_dt`, `mul_bcast_scalar_init_with_dt`): every `get_write_ptr`/`get_read_ptr` in these files is a **method on the DFB object**; `read_line`'s local L1 copy uses a Device 2.0 `UnicastEndpoint` with `my_x`/`my_y` and `noc.get_noc_id()`. No legacy addr-gen types, no raw NoC calls.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | — | — | *(no violations)* | — |

  **One judgment call at the RED/GREEN boundary — recorded, and logged as a recipe note.** Each reader calls the CB-index free function **`get_dataformat(cb_id)`** three times (`reader_..._2d.cpp:34,36,38`; `_3d.cpp:35,37,39`; `_4d.cpp:35,37,39`). It is not on the recipe's sanctioned list (`get_tile_size(cb_id)`, `get_local_cb_interface(cb_id)`), and a wrapper-method replacement does exist (`DataflowBuffer::get_dataformat()`, `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:275`). I judge it **GREEN**, on four grounds:

  1. It is **structurally identical to the sanctioned `get_tile_size(cb_id)`** — a `constexpr` metadata lookup keyed by CB index — and the Device 2.0 `CircularBuffer` wrapper merely *forwards* to it (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:115`: `DataFormat get_dataformat() const { return ::get_dataformat(cb_id_); }`). That forwarding relationship is precisely the reasoning the recipe gives for keeping `get_tile_size(cb_id)` sanctioned.
  2. It is **not a data-movement API**, and the Device 2.0 gate is about data-movement migration. The recipe hands tile/format metadata lookups to the *Metal 2.0 port* under kernel-side whitelist rule 7, explicitly noting this "does not move the Device 2.0 boundary."
  3. The **holdover cue does not fire**: it requires the wrapper object to be in scope *at the call site*, and it is not — the DFB objects for `c_2` / `c_3` / `c_0` are constructed later in each reader (2d: lines 52, 66, 70).
  4. **All nine results are dead** — `weight_data_format`, `divisor_data_format` and `output_grad_data_format` are assigned once and never read in any of the three readers (verified by grep). Nothing behavioural depends on them, so there is no Device 2.0 semantic to migrate; the correct fix is deletion, which is an ops-team cleanup, not a Device 2.0 migration item.

  Calling this RED would route an op with zero data-movement debt to the Device 2.0 team over nine dead lines — the too-conservative RED the recipe warns misroutes work. It is instead routed as a **Misc anomaly** (deletion) plus a **porter heads-up**, because two of the three calls name CBs that *do not exist* in the non-`WEIGHT` / non-`DIVISOR` configs — so they must not be naively rewritten as `DataflowBuffer(dfb::weight).get_dataformat()`. See *Heads-ups* and *Recipe notes*.

- **Feature compatibility: GREEN.** Every Appendix A entry is absent. A grep of the op directory for `GlobalCircularBuffer`, `global_circular_buffer`, `CreateGlobalCircularBuffer`, `remote_index`, `remote_cb`, `GlobalSemaphore`, `global_semaphore`, `address_offset`, `set_address_offset`, `UpdateDynamicCircularBufferAddress`, `cb_descriptor_from_sharded_tensor` and `set_globally_allocated_address` returns **zero** hits. The op declares no semaphores (no `.semaphores` on any `ProgramDescriptor`; no `Semaphore`/`semaphore` token anywhere in the directory) and no Buffer-backed CBs.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No GCB type; no `.global_circular_buffer` field on any `CBDescriptor` (all are built by the local `push_cb` helper, `program_factory.cpp:23-42`); no `remote_*` idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | Field never set; no imperative `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | Op uses no semaphores of any kind |
  | Variable-count compile-time arguments (CTA varargs) | N/A | See below |

  On **CTA varargs**: the op-level cue is absent (`tensor_args_t` is five named tensors — `target_tensor`, `output_grad_tensor`, and three `std::optional<Tensor>`, `...device_operation.hpp:26-32`; no `std::vector<Tensor>`), and the deciding kernel-level signal is absent — the op contains exactly **one** `get_compile_time_arg_val` call, at literal offset `0` (`moreh_nll_loss_backward_kernel.cpp:12`). The readers' `TensorAccessorArgs<0>()` / `TensorAccessorArgs<…next_compile_time_args_offset()>()` offsets are `constexpr`. Fixed-count CTAs.

  *Adjacent but not this entry:* the optional tensors give the CTA layout a **conditional shape** — an absent optional still contributes a 2-word placeholder block (`args_config_.raw()` plus `aligned_page_size = 0`, `tt_metal/impl/buffers/tensor_accessor_args.cpp:196-205`), so the count is fixed for a given config and varies only *across* configs, which is ordinary per-config CTA binding, not a runtime-varying index. Recorded under *Port-work summary* because the placeholder mechanism disappears in the port.

- **CB endpoints (GATE-free): 4 plain 1P+1C, 4 self-loops, and 3 config-scoped dead CBs.** Nothing needs the multi-binding advanced option. Each node runs exactly one reader, one writer and one compute instance — the two compute `KernelDescriptor`s cover **disjoint** core groups (`core_group_1` / `core_group_2`).

  Both faces of the multi-binding hunt were run and came back empty. **(a) Hidden second writer:** the readers *do* raw-write CB memory via `CoreLocalMem<...>(dfb_tmp_weight_obj.get_write_ptr())`, but that write is bracketed by the reader's own `reserve_back` … `push_back` on `c_24` — it is the FIFO producer's own peek, not a second toucher — and there are no semaphores to coordinate a co-fill. Likewise the readers' `get_read_ptr()` peeks on `c_1` / `c_2` are on CBs they already produce. **(b) Multiple readers:** no borrowed-memory / tensor-view CB exists (no Buffer-backed CB in the op), and no CB is touched by two co-resident kernels other than the four legal producer/consumer pairs. **(c) Dual-instance work-split:** absent — reader and writer are single instances over `all_cores`, and the two compute instances are group-disjoint.

  | CB | Role | Toucher(s) | Census | Disposition | Configs |
  |---|---|---|---|---|---|
  | `c_0` | `output_grad` | reader produces (`read_tile`); compute consumes (`wait_front`, holds — never pops) | 1P + 1C | plain 1:1 ✓ | all |
  | `c_1` | `target` | **reader only** — produces (`read_tile`), consumes (`wait_front`/`pop_front`), peeks (`get_read_ptr`) | **1 toucher** | **self-loop** | all |
  | `c_2` | `weight` | **reader only** — produces (`read_line`), then `wait_front` + `get_read_ptr` (never pops) | **1 toucher** | **self-loop** | `WEIGHT` only |
  | `c_3` | `divisor` | reader produces (`read_tile`); compute consumes (`wait_front`/`pop_front`) | 1P + 1C | plain 1:1 ✓ | `DIVISOR` only |
  | `c_7` | `weight_scratch` | **reader only**, inside `read_line` — **sync-free**: NoC-written and `get_write_ptr()`-read, with *no* FIFO ops at all | **1 toucher** | **self-loop** | `WEIGHT` only |
  | `c_8` | *(intended output_grad scratch)* | **nobody** | **0 touchers** | **DEAD → drop** | 2d only |
  | `c_16` | `input_grad` | compute produces; writer consumes | 1P + 1C | plain 1:1 ✓ | all |
  | `c_24` | `tmp_weight` | reader produces (`reserve_back` + raw write + `push_back`); compute consumes | 1P + 1C | plain 1:1 ✓ | all |
  | `c_25` | `tmp1` | compute only — produces and consumes | 1 toucher / **0** | **self-loop** / **DEAD** | `DIVISOR` / **non-`DIVISOR`** |
  | `c_26` | `tmp2` | compute only — produces and consumes | 1 toucher / **0** | **self-loop** / **DEAD** | `DIVISOR` / **non-`DIVISOR`** |

  `c_7` is a textbook *sync-free single-ended* CB: `read_line` NoC-reads DRAM into it and copies the valid bytes out via a local L1 unicast read from `cb_scratch.get_write_ptr()`, with no `reserve_back`/`push_back` on the scratch at any point (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, `read_line` body). One toucher, no FIFO → self-loop.

  Two consumers legitimately never pop: `c_0` (compute waits one tile at `moreh_nll_loss_backward_kernel.cpp:52` and holds it for the whole loop) and `c_2` (the reader holds the whole weight line). A held 1:1 FIFO is still a plain pair.

  ### The three dead CBs — confirmation, because a dead CB should be rare

  The recipe is right that one dead CB is surprising; three warrants showing the work. `c_8` is unconditionally vestigial; `c_25` and `c_26` are dead only in the non-`DIVISOR` config.

  **`c_8` — dead in the only config that allocates it (2d).** Allocated at `program_factory.cpp:107`, with the comment *"Need another scratch CB for output_grad reading data from DRAM into L1."* No kernel references it. I ruled out every indirect path the recipe names:
  - *Direct reference:* grep of `device/kernels/` for `c_8` and for `scratch` returns only `cb_weight_scratch = tt::CBIndex::c_7` in the three readers. Zero hits for `c_8`.
  - *A CTA carrying the index:* the readers' only compile-time args are the four `TensorAccessorArgs` blocks — no CB-index CTA exists to thread it through.
  - *A helper hiding it:* the only donor helpers taking a CB are `read_tile` and `read_line`, and every call site passes an explicitly named DFB object (`c_0`, `c_1`, `c_2`, `c_3`, `c_7`, `c_24`). The 2d reader's output_grad read is the **3-argument** `read_tile(dfb_output_grad_obj, addrg_output_grad, 0)` — that overload takes no scratch CB at all.
  - *A computed / aliased index:* every CB handle in these kernels is a `constexpr` initialised from a named `tt::CBIndex::c_N`; none is arithmetic.
  - *An unchecked config:* c_8 exists only in the 2d path, and the 2d reader was checked under both the `WEIGHT` and `DIVISOR` arms, along with the writer and the compute kernel.

  The comment is stale rather than aspirational: `read_line` needs a scratch because it reads `FACE_WIDTH`-sized slices that fall under the DRAM read alignment, whereas a full-tile `read_tile` is naturally aligned and reads straight into its destination. So `c_8` is a **confirmed** dead CB — a full tile of L1 burned in the 2d path. Dropping it changes L1 footprint and nothing else.

  A structural reassurance for the reviewer, since a wrongly-dropped *live* CB is the worst outcome this port can produce: on this op the error would be **loud, not silent**. Every CB handle in these kernels is a named `constexpr`, with no CTA-carried or computed index anywhere — so if `c_8` were somehow live, the kernel would name a DFB that has no binding and fail to build, rather than mis-addressing at runtime.

  **`c_25` / `c_26` — dead when `divisor_tensor` is absent.** Both are allocated *unconditionally* (`program_factory.cpp:96-97`, `:312-313`, `:529-530`) in all three rank paths, but every **use** sits inside `#if defined(DIVISOR)`: `dfb_tmp1_obj` at `moreh_nll_loss_backward_kernel.cpp:36,46,49,75,78` and `dfb_tmp2_obj` at `:57,67,70,74,78,87` — verified against the kernel's preprocessor structure (guards open at `:34` and `:55`, `#else` at `:89`, `#endif` at `:50`/`:109`). The non-`DIVISOR` branch (`:90-108`) touches only `tmp_weight`, `output_grad` and `input_grad`. A grep of the whole kernels directory finds `c_25` / `c_26` / `tmp1` / `tmp2` **only** in the compute kernel — the readers and writer never mention them. So with no divisor, both CBs are allocated and touched by nobody.

  **The fix is conditional allocation, not deletion** — and it is required, not cosmetic: a DFB with neither a producer nor a consumer binding is rejected by the spec validator, so the non-`DIVISOR` spec cannot be built while these two are allocated unconditionally. The porter gates their `DataflowBufferSpec` on `divisor_has_value`, exactly as the factory already gates `c_3`.

  **One subtlety the raw census misses.** `DataflowBuffer dfb_tmp1_obj(cb_tmp1)` and `dfb_tmp2_obj` are *constructed* unconditionally at `moreh_nll_loss_backward_kernel.cpp:23,25` — outside the `DIVISOR` guard that wraps every use. A bare construction is not a "touch" (no FIFO op, no pointer access), so it does not enter the endpoint count; but in Metal 2.0 the `dfb::` token must exist for the construction to compile. So the porter must move those two constructions **inside** the existing `#if defined(DIVISOR)` guard as well — otherwise the non-`DIVISOR` build names two DFBs the spec no longer declares. Same shape as the dead `get_dataformat` calls above. Logged as a recipe note.

  **Config-dependence summary.** The census flips on two axes: `WEIGHT` decides whether `c_2` / `c_7` exist at all, and `DIVISOR` decides whether `c_3` exists and whether `c_25` / `c_26` are self-loops or dead. Rank (2d/3d/4d) affects only `c_8` (2d-only) — the other nine CBs and all four producer/consumer pairs are identical across the three rank paths. `fp32_dest_acc_en` changes only the intermediates' `data_format` / `page_size`, never who touches them.

- **Offset base pointers: GREEN.** No fold exists to split out — and, more strongly, **there is no address RTA in this op at all.** Every base is delivered through the descriptor-API **`Buffer*`-binding form**: `reader_desc.emplace_runtime_args(core, {target_buf, output_grad_buf, weight_buf, divisor_buf, ignore_index, …})` and `writer_desc.emplace_runtime_args(core, {input_grad_buf, units_per_core, tile_offset})` (`program_factory.cpp:218-233`, `:432-448`, `:649-665`). The pushed values are `Buffer*` objects — the factory even comments the intent (`:197-198`, `:411-412`, `:628-629`: *"Pass `Buffer*` (not a raw address) so the program-cache fast hit path re-patches the binding"*). There is no `->address()` call anywhere in the op, hence no expression into which an offset could be folded, and no host arithmetic on any base. Type 3 (`address_offset`) is absent per the Appendix A row; Type 4 (`ttnn::narrow`) does not appear. The offset-base-pointer triage analysis (`analyses/2026-07-19_offset_base_pointers.md`, a dated prior) contains no `nll` entry — consistent with the scan, and the scan is what decides it.

  *Not to be confused with a fold:* the donor's `read_tile` / `read_value` / `read_line` pass `{.page_id = …, .offset_bytes = …}` to `noc.async_read`. Those are **framework-level accessor page offsets computed on-device**, not a host-folded base — and the op always passes `offset = 0` to `read_tile` anyway. `read_line`'s `noc_offset` and `cb_offset` are likewise kernel-side.

- **TensorAccessor 3rd argument: GREEN — N/A, nothing to drop.** All **13** `TensorAccessor(` construction sites across the readers and writer take exactly **two** arguments (`TensorAccessor(target_args, target_addr)`, `(output_grad_args, output_grad_addr)`, `(weight_args, weight_addr)`, `(divisor_args, divisor_addr)`, `(input_grad_args, input_grad_addr)`), and neither donor header constructs a `TensorAccessor`. The page-size override is never used, so no site needs classifying and the usual Class-2 "drop the arg" step does not apply. Note the distinction that makes this unambiguous: the `aligned_page_size` these accessors use is baked *inside* the `TensorAccessorArgs` CTA block by the framework (`tt_metal/impl/buffers/tensor_accessor_args.cpp:198-205`), which is the standard path — not the manual 3rd constructor argument this subject is about. The 3rd-arg triage analysis (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, a dated prior) lists `moreh_fold` and `moreh_getitem` but no `nll` op — again consistent.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — five bindings, all **Case 1**, identical across the three rank paths:
  - `target` — **Case 1**. `Buffer*` at reader RTA index 0 → `TensorAccessor(target_args, target_addr)`; all access through the accessor (via the donor's `read_tile`).
  - `output_grad` — **Case 1**. `Buffer*` at reader RTA index 1 → `TensorAccessor(output_grad_args, output_grad_addr)`.
  - `weight` — **Case 1**, **conditional** (`WEIGHT`). `Buffer*` at reader RTA index 2, or `nullptr` when absent. Accessor built only inside `#if defined(WEIGHT)`.
  - `divisor` — **Case 1**, **conditional** (`DIVISOR`). `Buffer*` at reader RTA index 3, or `nullptr` when absent. Accessor built only inside `#if defined(DIVISOR)`.
  - `input_grad` — **Case 1**. `Buffer*` at writer RTA index 0 → `TensorAccessor(input_grad_args, input_grad_addr)`.

  **No Case 2 — and this is the one place the op invites a misread.** The readers use raw typed pointers heavily (`CoreLocalMem<volatile int32_t> target_l1_ptr(dfb_target_obj.get_read_ptr())`, `CoreLocalMem<volatile uint16_t> weight_l1_ptr(...)`, `CoreLocalMem<volatile FP32_DEST_ACC_FTYPE> tmp_weight_l1_ptr(dfb_tmp_weight_obj.get_write_ptr())`). Every one of those is a pointer into **CB/L1 memory obtained from a DFB method** — not a tensor base address obtained from an RTA. No kernel performs address arithmetic on a tensor base, so no binding needs the `get_bank_base_address` bridge. Case 1 throughout.

  Delivery-mechanism note: because these are `Buffer*` entries rather than `->address()` values, the framework already registers them as `BufferBinding`s and patches them on cache hits (`tt_metal/api/tt-metalium/program_descriptors.hpp:114-118`, `170-176`) — routine port work, not a latent correctness hazard. The typed Metal 2.0 binding supersedes the interim mechanism, and the `TensorAccessorArgs` placeholder chain (below) goes with it.

- **Conditional bindings and the placeholder CTA chain.** The factory appends **four** `TensorAccessorArgs` blocks to the reader's CTAs unconditionally, passing `nullptr` for an absent optional (`program_factory.cpp:111-114`, `:325-328`, `:542-545`). A null block still emits two words — `args_config_.raw()` and `aligned_page_size = 0` — which is what keeps the kernel's offset chain (`weight_args.next_compile_time_args_offset()` → `divisor_args` → `output_grad_args`) aligned across configs. Under Metal 2.0 the framework builds accessor args from the `TensorParameter` bindings, so **both the placeholder blocks and the whole offset chain disappear**; the port expresses `weight` and `divisor` as conditional bindings instead. Correspondingly, `push_cb` skips allocation entirely when `num_tiles == 0` (`program_factory.cpp:29-31`), which is why `c_2` / `c_3` are absent rather than zero-sized — the same conditionality the DFB specs must carry.

- **TensorParameter relaxation:** `none` — nothing to apply.
- **TensorAccessor 3rd arg:** none — no site passes one.
- **CB endpoints:**
  - **self-loop** — `c_1` (all configs), `c_2` and `c_7` (`WEIGHT`), `c_25` and `c_26` (`DIVISOR`).
  - **plain 1P+1C, no action** — `c_0`, `c_16`, `c_24` (all configs), `c_3` (`DIVISOR`).
  - **dead-CB drop** — `c_8` @ `program_factory.cpp:107` (2d path; confirmed unreferenced).
  - **conditional allocation** — `c_25` @ `:96`/`:312`/`:529` and `c_26` @ `:97`/`:313`/`:530` must be gated on `divisor_has_value`, and their kernel-side constructions at `moreh_nll_loss_backward_kernel.cpp:23,25` moved inside the existing `#if defined(DIVISOR)` guard.
  - **multi-binding advanced option** — not needed anywhere.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. Both faces were hunted and came back empty (detail above); the porter need not re-run the hunt. The reader's raw `get_write_ptr()` write into `c_24` is the producer's own peek, not a hidden second writer.
- **Cross-op / shared kernels:** **none — no cross-op coupling of any kind.** All five kernel sources live in this op's own `device/kernels/` and are bound only by this op's single factory; the only outside reference to the directory is the family CMake glob (`ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:42`). No op borrows these files and this op borrows none. No `_metal2` fork exists beside any of them. The writer and compute kernel are each bound at three call sites, but all three are the rank branches of the *same* factory — so one port converts every binder at once, and no fork question arises. The practical consequence is narrower than the shared-kernel caution: any edit to `writer_moreh_nll_loss_backward.cpp` or `moreh_nll_loss_backward_kernel.cpp` must satisfy all three rank paths, which supply different reader RTA layouts (2d has 10 reader args, 3d/4d have 11).
- **RTA varargs:** none. Every arg is read at a fixed index. The readers use a running `i++` counter (`get_arg_val<uint32_t>(i++)` × 10 or 11 at the top of the kernel) — a fixed run over a fixed set, which the recipe classifies as ordinary positional plumbing that dissolves into named args, **not** a loop. No counted loop over arg indices, no data-selected read.
- **Several args and one CTA are dead — do not invent names for them.** Naming is port work, so the porter needs to know which args have no meaning to name: reader `element_size` (index 9 in 2d, 10 in 3d/4d) is read into an unused local in all three readers; compute RTA index 0 is never read at all; compute RTA index 1 (`tile_offset`) is read into an unused local; and compute CTA index 1 (`divisor_has_value`) is never read, the kernel branching on the `DIVISOR` define instead. Details and `file:line` under *Misc anomalies*; removal routes to the ops team, not the port.
- **The dead `get_dataformat(cb_id)` locals must not become DFB metadata accesses.** Nine dead calls across the three readers (`_2d.cpp:34,36,38` and equivalents). Two of the three per reader name `c_2` (weight) and `c_3` (divisor) — CBs that are **not allocated** in the non-`WEIGHT` / non-`DIVISOR` configs, and whose `dfb::` tokens therefore will not exist. Rewriting them as `DataflowBuffer(dfb::weight).get_dataformat()` under whitelist rule 7 would name an unbound DFB. They are provably dead, so deleting them is behaviour-preserving; see *Questions* for the ops-team confirmation.
- **Per-core-group compute pair:** each rank path emits two compute `KernelDescriptor`s from the same source over disjoint core groups, differing only in the leading CTA (`{units_per_core_group_1, divisor_has_value}` vs `{units_per_core_group_2, …}`, `program_factory.cpp:169`/`:185` and the 3d/4d equivalents). Preserve them as two `KernelSpec`s in two `WorkUnitSpec`s — demoting that CTA to an RTA to collapse them into one spec is a named anti-pattern with a real kernel-perf cost.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: `✓ clean`.** No entry is ⚠, ✗ or ⭐.

- No **file-path kernel instantiation** escape: all 5 bound `kernel_source` paths are under this op's own `device/kernels/`. The op borrows no kernel file and lends none.
- Every **function-call** escape resolves to a shape that bridges cleanly today: donors take a `DataflowBuffer` object by value, or a `TensorAccessor` as a template parameter.
- No donor is on pre-Device-2.0 idioms, so the Device 2.0 gate has no donor component.
- No `Semaphore`-shaped escape of any kind (the op has no semaphores), so the `uint32_t sem_id` / `sem_addr` problem rows cannot arise.

**Summary table** — one row per (op kernel, donor file):

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `reader_moreh_nll_loss_backward_{2d,3d,4d}.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 3 — second shared-kernel pool | ✓ |
| `moreh_nll_loss_backward_kernel.cpp` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | 3 — second shared-kernel pool | ✓ |
| all 5 kernels | `tt_metal/hw/inc/api/{dataflow,compute,tensor}/*`, `api/core_local_mem.h` | 1 — LLK / HAL / firmware | ✓ no concern |

**Shape notes** (per-call detail otherwise omitted, all rolls being ✓ — but two shapes are worth recording):

- `read_tile(DataflowBuffer cb, AddrGen addrgen, uint32_t noc_id, …)` and `read_line(DataflowBuffer cb, DataflowBuffer cb_scratch, AddrGen addrgen, uint32_t num_tiles, …)` (`kernel/dataflow/moreh_common.hpp:666`, `:739`) take **`DataflowBuffer` by value** plus the accessor as a **template parameter** (`AddrGen`, instantiated with `TensorAccessor`). Both are ✓: `DataflowBuffer` has a **non-explicit** converting constructor from `DFBBindingToken` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:106`), so `dfb::name` converts implicitly at the call site; and the `AddrGen` template accepts a `TensorAccessor` built from `tensor::name` directly (Shape 1, ✓ excellent). This is **not** the ⚠-flagged `CircularBuffer&` row — the donors are already on the DFB type. The recipe's shape table still has no `DataflowBuffer` row; see *Recipe notes*.
- `copy_tile_init_with_dt(DataflowBuffer, uint32_t)`, `pack_tile_with_dt(uint32_t, DataflowBuffer)`, `mul_bcast_scalar_init_with_dt(DataflowBuffer, DataflowBuffer)` (`kernel/compute/moreh_common.hpp:35`, `:28`, `:121`) — same `DataflowBuffer`-by-value shape, same ✓.
- LLK compute entry points the compute kernel calls with raw CB ids — `init_sfpu`, `copy_tile`, `mul_tiles_bcast_scalar`, `recip_tile`, `negative_tile` — take `uint32_t` by design, with no object-based replacement. `DFBBindingToken`'s `constexpr operator uint32_t()` (`dataflow_buffer.h:89`) covers them.

**Borrowed kernel files (file-path instantiation):** none. The op owns every kernel source it binds. No `_metal2` fork exists in `device/kernels/`.

**Negative pointer (saves a wrong turn):** there is **no** `experimental/quasar/` copy of this op or of any of its kernels — nothing in that out-of-bounds tree to mistake for prior art or for a fork to reuse.

### Relaxation candidates

None. The sheet's cell reads `none`, and the op carries no custom hash from which a candidate could be mined.

### TTNN factory analysis

The sheet-derived facts, with `file:line` evidence, in the form the TTNN ProgramFactory wiring consumes. The gate conjuncts among them (a non-`none` relaxation, `get_dynamic_runtime_args`, genuine multi-program) are recorded in *Gate detail*; the rest are non-gating facts:

- **Op-owned tensors:** none. `descriptor` concept; no factory returns a `WorkloadDescriptor`, so no `buffers` vector exists. The output is either the caller's preallocated `input_grad_tensor` or an ordinary `create_device_tensor` (`...device_operation.cpp:90-98`).
- **MeshWorkload need:** none — not a `WorkloadDescriptor` op, so neither genuinely multi-program nor an op-owned-tensor artifact.
- **Custom hash:** absent — the framework default hash applies. (Non-gating either way under the current recipe; nothing for the porter to preserve.)
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** absent → the port targets the **base `ProgramSpecFactoryConcept`**, not `CustomProgramSpecFactoryConcept`. No method to translate.
- **Pybind `create_descriptor`:** absent (`moreh_nll_loss_backward_nanobind.cpp:23-36`) → nothing for the port to delete, so this port carries **no** user-visible API change from that column.
- **Other risky pybind:** none. The nanobind surface is one `bind_function` with plain tensor / bool / int / optional-config arguments. No enums, no internals.

One factory-shape note for the wiring: the single `Factory` fans out to three rank impls, each emitting its own kernel set and CB set. All three land on the same target concept; the rank dispatch stays host-side in `create_descriptor`, exactly as today.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

- **`reduction_mean` is accepted, hashed, and completely unused.** It is a public argument (`moreh_nll_loss_backward.hpp:16`), stored in `operation_attributes_t` (`...device_operation.hpp:20`), threaded through `Factory::create_descriptor` (`program_factory.cpp:703`) — and then **ignored by all three impls**, each of which takes it as `const bool /*reduction_mean*/` (`:52`, `:265`, `:480`). Since the op has no custom hash, the attribute still participates in the **default program hash**, so two otherwise-identical calls differing only in `reduction_mean` occupy two program-cache entries that compile to identical programs. This is the recipe's "attribute forced or ignored in the factory yet still fed to `compute_program_hash`" shape. (The `divisor_tensor` argument is presumably how the caller actually communicates mean-reduction, which would make `reduction_mean` genuinely vestigial — worth the ops team confirming before removing it from the public signature.)
- **Dead CB `c_8` plus a stale comment.** `program_factory.cpp:106-107` allocates a full tile in the 2d path with the comment *"Need another scratch CB for output_grad reading data from DRAM into L1"*, but the 2d reader's output_grad read is a full-tile `read_tile` that needs no scratch. Confirmed unreferenced (see *Gate detail → CB endpoints*). The port drops the allocation; the ops team may want to drop the comment's premise too.
- **Dead compute CTA.** `compile_time_args = {units_per_core_group_N, static_cast<uint32_t>(divisor_has_value)}` (`program_factory.cpp:169`, `:185`, `:383`, `:399`, `:600`, `:616`) — index 1 is never read by `moreh_nll_loss_backward_kernel.cpp`, which branches on the `DIVISOR` **define** instead (also supplied, `:129`). The same fact is passed twice, one copy unused.
- **Both compute RTAs are dead.** The factory passes `{units_per_core, tile_offset}` (`program_factory.cpp:236`, `:451`, `:668`). The kernel never reads index 0 — its tile count comes from CTA 0 — and reads index 1 into `const uint32_t tile_offset` (`moreh_nll_loss_backward_kernel.cpp:14`) which is never used. Note the kernel reads index **1** while never reading index 0, so the two are not merely redundant: the arg vector exists only to position `tile_offset`, which is itself unused.
- **Dead reader RTA `element_size`.** Computed on the host as `weight.value().element_size()` or `0` (`program_factory.cpp:205`, `:419`, `:636`), passed as the last reader arg, and read into an unused local in all three readers (`_2d.cpp:22`, `_3d.cpp:23`, `_4d.cpp:23`).
- **Nine dead `get_dataformat(cb_id)` locals.** `weight_data_format`, `divisor_data_format`, `output_grad_data_format` are assigned and never read in each of the three readers (`_2d.cpp:34,36,38`; `_3d.cpp:35,37,39`; `_4d.cpp:35,37,39`). Two of the three per reader query CBs that are not allocated in the non-`WEIGHT` / non-`DIVISOR` configs. Harmless today only because the values are unused — see *Heads-ups* for why the porter cannot simply modernise them in place.
- **Dead local `n` in the 2d reader.** `uint32_t n = nt * TILE_HEIGHT + h;` (`reader_moreh_nll_loss_backward_2d.cpp:90`) is computed in the innermost loop and never used; the row within the tile is addressed via `get_tilized_idx(0, h)` instead. (The 3d reader does use its `n`.)
- **`TILE_HEIGHT` used to tile the channel dimension in the 2d reader.** `Ct = (C + TILE_HEIGHT - 1) / TILE_HEIGHT` (`reader_..._2d.cpp:72`) while the same quantity is computed host-side as `div_up(channel_size, tt::constants::TILE_WIDTH)` (`program_factory.cpp:83`), and the derived index uses `c = ct * TILE_WIDTH + w` (`:91`). For `input_grad: (N, C)` the channel is a width dimension, so `TILE_WIDTH` is the semantically correct divisor. Numerically identical today (both 32) and therefore invisible, but it would break silently if the tile dimensions ever diverged.
- **Unreachable output-allocation path.** `compute_output_specs` unconditionally `TT_FATAL`s when `input_grad_tensor` is absent (`...device_operation.cpp:87`, with a comment explaining that the channel size cannot be recovered from `target` and `output_grad`), and `create_output_tensors` reaches its `create_device_tensor` call only on that same absent-optional path (`:96-97`). So the `create_device_tensor` line is dead: the op always requires a preallocated `input_grad_tensor`. Intentional as a guard, but the dead allocation call reads as if the op could allocate its own output, and the pybind default (`input_grad_tensor = None`, `moreh_nll_loss_backward_nanobind.cpp:32`) advertises exactly the shape that fatals.
- **Inconsistent assertion macro for the same condition.** The compute-runtime-args branch uses `TT_FATAL(false, "Core not in specified core ranges.")` in the 2d impl (`program_factory.cpp:243`) but `TT_ASSERT(false, …)` in the 3d and 4d impls (`:458`, `:675`). `TT_ASSERT` compiles out in release builds, so the 3d/4d paths would silently skip assigning compute runtime args for such a core rather than failing. In practice unreachable — the earlier `units_per_core` dispatch (`:214`, `:428`, `:645`) already `TT_THROW`s for a core in neither group — so this is a consistency and defence-in-depth point, not a live bug.

## Questions for the user

1. **Readiness-sheet row count (the one cross-check I could not run):** the sheet is unreachable from this session (connector unauthorized, non-interactive), and you supplied the three decisive cells — thank you. The remaining gap is the **factory-set match**. The code has exactly **one** factory: `MorehNllLossBackwardDeviceOperation::Factory` (`program_factory_t = std::variant<Factory>`, `device/moreh_nll_loss_backward_device_operation.hpp:45`). The three rank impls (`_impl_2d` / `_impl_3d` / `_impl_4d`) are internal branches of that one factory, **not** separate factories — so I expect exactly one sheet row here, and *three* rows would be the staleness signal rather than a match. Worth a glance while you have the sheet open; a mismatch routes to the readiness-sheet owner and would not change the verdict, since there is only one factory to gate.
2. **Confirm the nine dead `get_dataformat` locals can be deleted (ops team).** They are provably unused, so removing them is behaviour-preserving — but two of the three per reader query CBs that do not exist in the non-`WEIGHT` / non-`DIVISOR` configs, and under Metal 2.0 they cannot be carried forward as DFB metadata accesses (there is no binding to read from). I would rather the ops team delete them on their own track than have the porter decide unilaterally. If they should instead stay, the port needs a different answer for those two configs and I would want to know before the port starts.
3. **Is `reduction_mean` genuinely vestigial?** It is public, hashed, and unused by every impl (see *Misc anomalies*). Removing it is an API change and squarely the ops team's call, not the porter's — but if it *should* have had an effect, that is a functional bug worth knowing about independently of this port.

## Recipe notes

1. **`get_dataformat(cb_id)` sits exactly on the Device 2.0 RED/GREEN boundary, and the sanctioned list is closed-form.** The Green bullet names two sanctioned CB-index free functions (`get_tile_size`, `get_local_cb_interface`); `get_dataformat(cb_id)` is not among them, yet it is structurally the same thing — a `constexpr` metadata lookup keyed by CB index, which the Device 2.0 `CircularBuffer` wrapper merely forwards to (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:115`), which is the very argument the recipe uses to sanction `get_tile_size`. The migration guide's examples use `get_tile_size(cb_id)` and never mention `get_dataformat`, so "check the current Device 2.0 surface" does not settle it either. I called it GREEN (reasoning in *Gate detail*), but that was a judgment call on a hard gate, made three times over nine call sites. **Suggest naming `get_dataformat(cb_id)` explicitly** — either adding it to the sanctioned list beside `get_tile_size`, or stating that only the two listed functions are sanctioned and every other CB-index metadata free function is a holdover. Either resolution is fine; the ambiguity is the problem, and it will recur because the metadata accessor set is broad (`get_tile_r_dim`, `get_tile_c_dim`, … all have the same shape).

2. **The dead-CB resolution has no branch for "dead in one config, live in another."** *Dead CB (0, 0)* says a confirmed dead CB "**must** be dropped" because a bindingless DFB is structurally inexpressible, and frames the fix as the porter dropping the allocation. On this op, `c_25` / `c_26` are live under `DIVISOR` and dead without it, while the *allocation* is unconditional — so "drop it" is wrong (it would break the divisor path) and "keep it" is also wrong (the non-divisor spec would not validate). The actual fix is **make the allocation conditional**, so the DFB exists exactly in the configs where it has endpoints — which is what the factory already does for `c_3` via `push_cb`'s `num_tiles == 0` early return. *Classify per instantiation* tells me to notice the flip, but the resolution list (self-loop / 1P+1C / multi-binding / drop) has no entry for it. **Suggest a fifth disposition — "config-conditional: gate the `DataflowBufferSpec` on the same predicate the legacy factory gates the allocation on"** — since an op with optional tensors will hit this routinely, and the two wrong answers are both plausible-looking.

3. **A DFB can be *named* without being *touched*, and only the touch enters the census.** The endpoint census counts FIFO ops and raw pointer access. But in Metal 2.0 merely **constructing** `DataflowBuffer(dfb::x)` requires the binding to exist, and this op's compute kernel constructs `dfb_tmp1_obj` / `dfb_tmp2_obj` unconditionally (`moreh_nll_loss_backward_kernel.cpp:23,25`) while every *use* is inside `#if defined(DIVISOR)`. A census that counts only touches correctly reports 0 endpoints — yet the kernel still names the DFB, so dropping or conditionalising the spec without also moving those two constructions inside the guard leaves a build that names an undeclared binding. The same trap sits in the readers, where `get_dataformat(cb_weight)` / `(cb_divisor)` sit outside the `WEIGHT` / `DIVISOR` guards that wrap every real use of those CBs. **Suggest a line in the dead-CB / config-dependence guidance:** when a CB's disposition is config-scoped, also check for *unguarded mentions* of it — bare constructions and metadata lookups — since those are what the porter must relocate, and a touch-only census is silent about them.

4. **The sheet is now harder to work around when unreachable, and the recipe still has no fallback.** Since my previous run against `bace43c8fb5`, the gate moved onto cells that are *by design* not derivable from code: `TensorParameter relaxation` ("Read the cell; do not re-derive it") is now a gate, and `Known op issues` is free text that "names its own owner." Combined with `ttnn_op_porting_readiness.md`'s standing constraints — the human authorizes the connector, it cannot be done in-session, and the fetch may not be delegated to a subagent — a non-interactive session cannot obtain three of the values the audit now turns on. The only documented outcome for lacking sheet data is *spreadsheet-broken → GATE*, which is wrong here: the sheet is fine, it is simply unreachable, and REDing on that would misroute a clean op to the readiness-sheet owner. I asked the launcher for the three cells directly and recorded the provenance. **Suggest documenting that as the sanctioned path:** *when the sheet is unreachable rather than wrong, ask the launcher for `Is able to port?`, `TensorParameter relaxation` (verbatim) and `Known op issues`, cross-check every code-checkable column, and disclose the factory-set check as unrun.* The distinction worth stating plainly is **unreachable ≠ broken**.

5. **Minor — the donor shape table still has no `DataflowBuffer` row.** Carried over from my previous audit and hit again here, so repeating it briefly: every donor this op calls takes `DataflowBuffer` **by value**, which is neither the `⭐ ⚠` `CircularBuffer&` row nor the `✓` `uint32_t cb_id` row. It is unambiguously fine (implicit `DataflowBuffer(DFBBindingToken)`, `dataflow_buffer.h:106`), but its resemblance to the starred row invites flagging a clean donor as a cross-team blocker. The shared pools appear to be converging on this shape.
