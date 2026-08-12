# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward`

Single device-operation directory. One `DeviceOperation`, five program factories — all `create_descriptor` returning a `tt::tt_metal::ProgramDescriptor`:

- **`MorehSoftmaxBackwardOperation`** (`device/moreh_softmax_backward_device_operation.{hpp,cpp}`)
  - `MorehSoftmaxBackwardWSmallFactory` (`device/softmax_backward_w_small/softmax_backward_w_small.cpp`)
  - `MorehSoftmaxBackwardWLargeFactory` (`device/softmax_backward_w_large/softmax_backward_w_large.cpp`)
  - `MorehSoftmaxBackwardHSmallFactory` (`device/softmax_backward_h_small/softmax_backward_h_small.cpp`)
  - `MorehSoftmaxBackwardHLargeFactory` (`device/softmax_backward_h_large/softmax_backward_h_large.cpp`)
  - `MorehSoftmaxBackwardCLargeFactory` (`device/softmax_backward_c_large/softmax_backward_c_large.cpp`)

Factory selection is by `dim` position plus an L1-fit heuristic (`get_parallelization_strategy`, `device/moreh_softmax_backward_device_operation.cpp:119`). The op serves three math variants through `defines` on one kernel set — `SOFTMAX` / `SOFTMIN` / `LOGSOFTMAX` (`LOG`) — plus `FP32_DEST_ACC_EN`.

**13 kernel files are referenced.** Two files in `device/kernels/` are referenced by **no** factory anywhere in the repo and are therefore **out of audit scope** (contents not audited):

- `device/kernels/writer_moreh_softmax_backward_h.cpp` — unreferenced
- `device/kernels/writer_moreh_softmax_backward_w.cpp` — unreferenced

They are called out because they are near-duplicates of the writers the op *does* bind (`writer_moreh_softmax_h.cpp` / `writer_moreh_softmax_w.cpp`, same directory, name differing only by the word `backward`) — an easy mis-read for anyone scanning this directory. See *Misc anomalies*.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `bace43c8fb5 2026-08-12 docs(metal_2.0): stop the port from deleting the op's custom program hash`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward` |
| **Overall** | **GREEN** — all five gates cleared, all five factories |
| **DOps / Factories** | `MorehSoftmaxBackwardOperation` → `WSmall`, `WLarge`, `HSmall`, `HLarge`, `CLarge` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — own kernels and both shared-pool donors are structurally Device 2.0 (`Noc`, `DataflowBuffer` objects, `TensorAccessor`). Only sanctioned `get_tile_size(cb_id)` free-function use. |
| *Prereqs* — Cross-op escapes | **Ok** — `✓ clean`; no borrowed kernel *files*, no donor needing conversion |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | **Ok** — CTAs read only at literal constexpr offsets `0`/`1` |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (launcher-supplied; every conjunct independently confirmed in code — see *Gate detail*) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 5 factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — not a `WorkloadDescriptor` op |
| *TTNN Readiness* — Is safe to port? | **Yes** (launcher-supplied; readiness-sheet owner's axis, not re-derived) |
| *TTNN Readiness* — Custom hash | **No** — no `compute_program_hash` override anywhere in the op |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | **No** — method absent |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `moreh_softmax_backward_nanobind.cpp` binds only the three public functions + two enums |
| *TTNN Readiness* — Op-owned tensors | **No** — `descriptor` concept; no `buffers` vector |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | **none** — no address RTA at all; bases arrive as `Buffer*` bindings, zero host arithmetic |
| *Port work* — Tensor bindings (per binding) | **Case 1 × 3** — `output`, `output_grad`, `input_grad`, all fed straight to a `TensorAccessor` |
| *Port work* — TensorParameter relaxation | **none** (no custom hash ⇒ no relaxation to apply) |
| *Port work* — TensorAccessor 3rd arg | **none** — all 15 construction sites are 2-arg; nothing to drop |
| *Port work* — CB endpoints | **legal + self-loop** — every reader/writer↔compute CB is a plain 1P+1C; the compute-local intermediates (3–4 per factory) need self-loops. No multi-binding, no dead CB. |

**CB endpoints** are dispositions, not gates. Record the disposition per `(CB, config)`, and classify per instantiation. For this op the census is stable across every config axis (`LOG` vs `SOFTMAX`/`SOFTMIN`, `fp32_dest_acc_en`, `has_core_group_2`) — see *Gate detail → CB endpoints* for why.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, same directory).

All five gates clear on all five factories. This is an unusually clean op for a Metal 2.0 port: the kernels are already fully Device 2.0 (they construct `DataflowBuffer` objects and use the `Noc` wrapper), every tensor reaches its kernel through a `TensorAccessor` built from a framework-patched `Buffer*` binding, no semaphores exist, no Appendix A feature is touched, and no kernel file is shared with another op. The port work is the mechanical core of the recipe — three `TensorParameter`/`TensorBinding` conversions, self-loops on the compute-local intermediates, and named RTAs — with two structural cautions for the porter (the per-core-group compute-CTA pair, and the two writers each bound by two of this op's own factories). No portable-subset scoping is needed; nothing is blocked.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The launcher supplied the two decisive sheet columns — `Is safe to port?` = `yes` and `Is able to port?` = `yes` — and `Is able to port?` *is* the gate. The readiness sheet could not be fetched in this session: the claude.ai Google Drive connector is unauthorized and the session is non-interactive, so `mcp__claude_ai_Google_Drive__download_file_content` is not even present to call (the recipe's own note is that authorization "cannot be done from inside the session," and the fetch may not be delegated to a subagent). This is **not** the recipe's *spreadsheet-broken* case — the sheet is neither wrong nor silent for this op, merely unreachable from here — so it is recorded as a limitation, not routed as a gate. To keep the verdict grounded I ran the full cross-check from the code instead, and every conjunct of the `Is able to port?` derivation agrees with the supplied `yes`:

  | Conjunct | Sheet (supplied / derived) | Code evidence | Agrees |
  |---|---|---|---|
  | `Is safe to port?` == `yes` | `yes` (supplied) | *Not re-derived* — the readiness-sheet owner's correctness axis, per recipe | — |
  | `Concept` == `descriptor` | `descriptor` | Five `static ProgramDescriptor create_descriptor(...)` declarations via the `DEFINE_SOFTMAX_BACKWARD_FACTORY` macro, `device/moreh_softmax_backward_device_operation.hpp:50-63`; each defined in its own `device/softmax_backward_*/` file | ✓ |
  | `Custom hash` == `no` | `no` | No `compute_program_hash` (or renamed variant) in the op directory | ✓ |
  | `get_dynamic_runtime_args` == `no` | `no` | Hook absent from `MorehSoftmaxBackwardOperation` (`...device_operation.hpp:72-78` lists only `select_program_factory`, `validate_*`, `compute_output_specs`, `create_output_tensors`, `get_parallelization_strategy`) | ✓ |
  | `override_runtime_arguments` == `no` | `no` | Method absent from device-op and all five factories | ✓ |
  | `Pybind descriptor` == `no` | `no` | `moreh_softmax_backward_nanobind.cpp:18-63` — three `ttnn::bind_function` calls + two `export_enum`; no `create_descriptor` binding, no factory/device-op internals exposed | ✓ |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no` (its `yes` is only possible on `descriptor`/`WorkloadDescriptor`, so no conflict either way), and `Op-owned tensors?` is `no` as the `descriptor` concept requires.

  **The one cross-check that could not be run** is the *factory-set match* (does the sheet carry exactly five rows for this op, one per factory, with no phantom and no missing row?). The code side is settled — five factories, enumerated above, matching the `program_factory_t` variant at `...device_operation.hpp:65-70` — but the sheet side is unreadable from here. Raised in *Questions for the user*; a five-row check by eye closes it. It cannot change the verdict for any factory that *does* have a row, since all five factories are structurally identical on every gate conjunct.

- **Device 2.0 (every kernel used): GREEN.** All 13 referenced kernels are structurally Device 2.0, and so is every donor function they call. Evidence, by kernel class:

  - **Readers** (`reader_moreh_softmax_backward_{c,h,h_large,w,w_large}.cpp`) — `Noc noc;` + `noc.async_read(<accessor>, <dfb>, bytes, {.page_id = …}, {.offset_bytes = 0})` + `noc.async_read_barrier()`; `DataflowBuffer dfb_*_obj(cb_*)` with `reserve_back`/`push_back` methods; `TensorAccessor` for all tensor addressing. No `noc_async_read`, no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen*`, no `cb_reserve_back`/`cb_push_back`, no raw semaphore addresses.
  - **Writers** (`writer_moreh_softmax_{h,w}.cpp`, `writer_moreh_softmax_backward_c.cpp`) — same idioms on the write path: `noc.async_write(<dfb>, <accessor>, …)`, `dfb_out_obj.wait_front`/`pop_front`.
  - **Compute** (`moreh_softmax_backward_{c_large,h,h_large,w,w_large}.cpp`) — `DataflowBuffer` objects throughout; all FIFO traffic via object methods.
  - **Donors** — `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_{dataflow,compute}.{hpp,inl}`: every `get_write_ptr` in these files is a **method on the DFB object** (`cb.get_write_ptr()`, `cb_scaler.get_write_ptr()`, `dfb.get_write_ptr()`) — not a CB-index free function. No legacy addr-gen types, no raw NoC calls.

  **No holdovers to report.** The only CB-index free function in play is `get_tile_size(cb_id)` (e.g. `reader_moreh_softmax_backward_c.cpp:33-34`, `writer_moreh_softmax_w.cpp:26`), which the Green bullet lists as **sanctioned** — Device 2.0 keeps it, so it does not knock the op out of Green. Confirmed against the current surface: `DataflowBuffer::get_tile_size()` exists as a member at `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:201`, so moving these lookups onto the object is *Metal 2.0 port* work under kernel-side whitelist rule 7 — not a Device 2.0 change and not a gate.

  Two LLK-facing raw-id passes are likewise **not** holdovers, because no wrapper-method replacement exists: `compute_kernel_hw_startup(uint32_t, uint32_t, uint32_t)` (`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:54`) and `copy_tile(maskcb.get_id(), …)` inside the donor — both are compute-LLK entry points that take a raw index by design.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | — | — | *(no violations)* | — |

- **Feature compatibility: GREEN.** Every Appendix A entry is absent. A repo-scoped grep of the op directory for `GlobalCircularBuffer`, `global_circular_buffer`, `CreateGlobalCircularBuffer`, `remote_index`, `remote_cb`, `GlobalSemaphore`, `global_semaphore`, `address_offset`, `set_address_offset`, `UpdateDynamicCircularBufferAddress` and `cb_descriptor_from_sharded_tensor` returns **zero** hits. The op declares no semaphores at all (no `.semaphores` on any `ProgramDescriptor`, no `Semaphore`/`semaphore` token in the directory) and no Buffer-backed CBs (no `set_globally_allocated_address`, no `.buffer` field on any `CBDescriptor`).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No GCB type, no `.global_circular_buffer` field on any of the 6–9 `CBDescriptor`s per factory, no `remote_*` idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | Field never set on any `CBDescriptor`; no imperative `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | Op uses no semaphores of any kind |
  | Variable-count compile-time arguments (CTA varargs) | N/A | See below |

  On **CTA varargs** specifically, both signals were checked: the op-level cue is absent (`tensor_args_t` is three named tensors — `output_tensor`, `output_grad_tensor`, `std::optional<Tensor> input_grad_tensor`, `...device_operation.hpp:41-45`; no `std::vector<Tensor>`), and the deciding kernel-level signal is absent too — every `get_compile_time_arg_val` in the op reads a **literal** index `0` or `1` (10 sites across the five compute kernels), and the reader/writer `TensorAccessorArgs<0>()` / `TensorAccessorArgs<y_args.next_compile_time_args_offset()>()` offsets are `constexpr`. Fixed-count CTAs, not varargs.

- **CB endpoints (GATE-free): every CB is either a plain legal 1:1 or a one-toucher self-loop.** Nothing needs the multi-binding advanced option, and there is no dead CB. The census is per node: each node runs exactly one reader, one writer and one compute instance (the two compute `KernelDescriptor`s cover **disjoint** core groups — `core_group_1` / `core_group_2` — so no node sees two compute instances).

  Both faces of the multi-binding hunt were run and came back empty. **(a) Hidden second writer:** no kernel writes a CB it does not FIFO-own — every raw `get_write_ptr()` in play is inside a donor helper (`generate_bcast_scaler`, `generate_mask_h/w`, `prepare_reduce_scaler`) that brackets it with `reserve_back(1)` … `push_back(1)` on the *same* kernel that calls it, so the raw write is the FIFO producer's own peek, not a second toucher; and there are no semaphores to coordinate a co-fill with. **(b) Multiple readers:** no borrowed-memory/tensor-view CB exists (no Buffer-backed CB anywhere in the op), and no CB is read by two co-resident kernels. **(c) Dual-instance work-split:** absent — the reader and writer are single instances over `all_cores`, and the two compute instances are group-disjoint (this is the *demoting-per-group-CTA* shape, not the same-grid work-split; see *Heads-ups*).

  Per-CB dispositions, uniform across all configs:

  | CB | Role | Producer (locked) | Consumer (locked) | Census | Disposition |
  |---|---|---|---|---|---|
  | `c_0` | `output` / `y` | reader — `reserve_back`/`push_back` | compute — `wait_front`/`pop_front` | 1P + 1C | plain 1:1 ✓ |
  | `c_1` | `output_grad` / `dy` | reader — `reserve_back`/`push_back` | compute — `wait_front`/`pop_front` | 1P + 1C | plain 1:1 ✓ |
  | `c_2` | `scaler` | reader — via `generate_bcast_scaler` (H) / `calculate_and_prepare_reduce_scaler` (W) | compute — `scaler_dfb.wait_front(1)` inside `compute_kernel_lib::reduce` (`reduce_helpers_compute.inl:387`) | 1P + 1C | plain 1:1 ✓ |
  | `c_3` | `mask` | reader — via `generate_mask_h` / `generate_mask_w` | compute — `maskcb.wait_front(mtile+1)` inside `mask_tile_to_cb`/`mul_tiles_and_mask_tile_to_cb` | 1P + 1C | plain 1:1 ✓ |
  | `c_16` | `input_grad` / `dx` | compute — `ocb.reserve_back`/`push_back` | writer — `wait_front`/`pop_front` | 1P + 1C | plain 1:1 ✓ |
  | `c_24` | `y*dy` (`cb_exp`/`cb_inter0` under `LOG`) | compute | compute | **1 toucher** | **self-loop** |
  | `c_25` | `reduce`/`sum` (`cb_inter1` under `LOG`) | compute | compute | **1 toucher** | **self-loop** |
  | `c_26` | `dy - sum` (`cb_inter2` under `LOG`) | compute | compute | **1 toucher** | **self-loop** |
  | `c_27` | `add(y*dy)` — **`WLarge` / `HLarge` only** | compute | compute | **1 toucher** | **self-loop** |

  Two consumers legitimately never `pop_front`: `c_2` (the scaler tile is waited once and held) and `c_3` (`popm=0` at every call site). A held 1:1 FIFO is still a plain 1P+1C — no special action.

  Per-factory self-loop set: `WSmall` = {`c_24`, `c_25`, `c_26`}; `HSmall` = {`c_24`, `c_25`, `c_26`}; `WLarge` = {`c_24`, `c_25`, `c_26`, `c_27`}; `HLarge` = {`c_24`, `c_25`, `c_26`, `c_27`}; `CLarge` = {`c_24`, `c_25`, `c_26`} (this factory allocates no `c_2`/`c_3` — the C-dim path needs no scaler or mask).

  **Config-dependence: none.** The census was re-derived under each config axis and does not flip. Under `LOG` the reader's first loop skips `y` (`#ifndef LOG`) but its second loop still reads `y`, so `c_0` keeps the reader as producer; `fp32_dest_acc_en` changes only the intermediates' `data_format`/`page_size`, not who touches them; `has_core_group_2` adds a second compute `KernelDescriptor` over a *disjoint* node set, so per-node counts are unchanged. The `LOG` path also re-aliases `c_24`/`c_25`/`c_26` under second names (`cb_exp`, `cb_inter0/1/2`) — same indices, same single toucher.

  **No dead CB.** Every allocated `buffer_index` is referenced by a kernel in *both* the `LOG` and non-`LOG` paths of the factory that allocates it — verified index-by-index per factory, including the aliased intermediates above. `c_27` exists only in the two `*Large` W/H factories, and both their compute kernels reference it on both paths.

- **Offset base pointers: GREEN.** No fold exists to split out — and, more strongly, **there is no address RTA in this op at all.** Every factory delivers tensor bases through the descriptor-API **`Buffer*`-binding form**: `reader_desc.emplace_runtime_args(core, {output.buffer(), output_grad.buffer(), num_tiles_per_core, …})` and `writer_desc.emplace_runtime_args(core, {input_grad.buffer(), …})` (e.g. `softmax_backward_w_small.cpp:233-243`, `softmax_backward_c_large.cpp:219-230`). The pushed value is the `Buffer*` object, not `->address()`; there is no `->address()` call anywhere in the op directory, hence no expression into which an offset *could* be folded, and no host arithmetic on any base. Type 3 (`address_offset`) is absent per the Appendix A row above; Type 4 (`ttnn::narrow`) does not appear. The offset-base-pointer triage analysis (`analyses/2026-07-19_offset_base_pointers.md`, a dated prior) lists no moreh or softmax op — consistent with the scan, and the scan is what decides it.

- **TensorAccessor 3rd argument: GREEN — N/A, nothing to drop.** All **15** `TensorAccessor(` construction sites in the op's kernels take exactly **two** arguments (`TensorAccessor(y_args, y_addr)`, `TensorAccessor(dy_args, dy_addr)`, `TensorAccessor(out_args, dst_addr)`), and none of the donor headers the op calls constructs a `TensorAccessor` at all. The page-size override is never used, so no site needs classifying, no `dynamic_tensor_shape` relaxation follows, and the usual Class-2 "drop the arg" port step does not apply here. The 3rd-arg triage analysis (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, a dated prior) lists `moreh_fold` and `moreh_getitem` but not this op — again consistent, and again the scan decides.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — three bindings, all **Case 1**, all uniform across the five factories:
  - `output` (`y`) — **Case 1**. Base arrives as a `Buffer*` binding at reader RTA index 0; kernel feeds it straight into `TensorAccessor(y_args, y_addr)` and does every access through the accessor (`noc.async_read(y_in, …, {.page_id = …})`). Port: express as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::…)`; the RTA and its `TensorAccessorArgs` CTA plumbing (`TensorAccessorArgs(*output.buffer()).append_to(reader_ct_args)`) both disappear.
  - `output_grad` (`dy`) — **Case 1**, identical shape at reader RTA index 1 / second `TensorAccessorArgs` block.
  - `input_grad` (`dx`, the output) — **Case 1**, writer RTA index 0 / sole `TensorAccessorArgs` block.

  None is a raw-pointer Case 2 (no kernel does hand-rolled address arithmetic on a base), and none is a borrowed-memory-DFB clean case (the op has no Buffer-backed CB). Note the delivery mechanism: because these are `Buffer*` entries rather than `->address()` values, the framework already registers them as `BufferBinding`s and patches them on cache hits (`tt_metal/api/tt-metalium/program_descriptors.hpp:114-118`, `170-176`) — so this is **routine port work, not a latent correctness hazard**. The typed Metal 2.0 binding supersedes the interim mechanism.
- **TensorParameter relaxation:** **none.** The op has no custom hash, and a relaxation is the hash excluding a property from the cache key — with no hash there is nothing to confirm and nothing to apply. (The sheet's `TensorParameter relaxation` cell could not be read directly; see *Questions*. A non-`none` value there would contradict the absent hash and would itself be the finding.)
- **TensorAccessor 3rd arg:** **none** — no site passes one.
- **CB endpoints:** self-loop `c_24`, `c_25`, `c_26` in every factory, plus `c_27` in `WLarge` and `HLarge`. All other CBs are plain 1P+1C and need no action. No 1P+1C *assignment* decisions, no multi-binding flag, no dead-CB drop.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** **none.** Both hidden-writer and multi-reader faces were hunted and came back empty (detail above). The porter does not need to re-run the hunt.
- **Cross-op / shared kernels:** **no cross-op coupling — but two of the op's kernel files are bound by two of its own factories each.** `writer_moreh_softmax_w.cpp` is bound by `WSmall` (`softmax_backward_w_small.cpp:156`) and `WLarge` (`softmax_backward_w_large.cpp:187`); `writer_moreh_softmax_h.cpp` by `HSmall` (`softmax_backward_h_small.cpp:156`) and `HLarge` (`softmax_backward_h_large.cpp:187`). This is the **intra-op** shape of the shared-kernel caution: a whole-op port converts every binder in the same change and may convert in place, but a factory-at-a-time port must fork instead. No `_metal2` fork exists beside any of the op's kernels — this port would create the first, if it forks at all.

  **Filename trap, resolved:** `writer_moreh_softmax_h.cpp` and `writer_moreh_softmax_w.cpp` also exist as **separate private copies** under `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/`, and those copies are what `moreh_softmax`'s own factories and `normalization/softmax`'s general factories bind (via `SOFTMAX_KERNEL_PATH_GENERAL`, which resolves to the **forward** op's kernel directory — `softmax_operation_types.hpp:39-40`). A filename grep makes those look like co-borrowers of *this* op's files; checking the bound **path** shows they are not. This op's kernel directory has **no external borrower** — the only outside reference to it is the family CMake glob (`ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:53`).
- **RTA varargs:** **none.** Every RTA is read at a fixed literal index (`get_arg_val<uint32_t>(0)` … `(7)`); no counted loop over arg indices, no running `arg_index++`, no data-selected element. All args are nameable — the preferred case.
- **Per-core-group compute CTA pair:** each factory emits two compute `KernelDescriptor`s from the same source, differing only in the per-group tile count (`{num_tiles_per_core_group_1, Wt}` vs `{num_tiles_per_core_group_2, Wt}`) over disjoint core groups. Preserve them as two `KernelSpec`s in two `WorkUnitSpec`s — demoting that CTA to an RTA to collapse them into one spec is a named anti-pattern with a real kernel-perf cost.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: `✓ clean`.** No entry is ⚠, ✗ or ⭐.

- No **file-path kernel instantiation** escape: all 13 referenced `kernel_source` paths are under this op's own `device/kernels/`. The op borrows no kernel file, and lends none (see the filename trap above).
- Every **function-call** escape resolves to a shape that bridges cleanly today: donors take either a `DataflowBuffer` object by value or a `uint32_t` CB id as an NTTP.
- No donor is on pre-Device-2.0 idioms, so the Device 2.0 gate has no donor component.
- No `Semaphore`-shaped escape of any kind (the op has no semaphores), so the `uint32_t sem_id` / `sem_addr` problem rows cannot arise.

**Summary table** — one row per (op kernel, donor file):

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `reader_moreh_softmax_backward_h.cpp`, `reader_moreh_softmax_backward_h_large.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 3 — second shared-kernel pool | ✓ |
| `reader_moreh_softmax_backward_w.cpp`, `reader_moreh_softmax_backward_w_large.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 3 — second shared-kernel pool | ✓ |
| `reader_moreh_softmax_backward_w.cpp`, `reader_moreh_softmax_backward_w_large.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | 2 — official kernel library | ✓ |
| `moreh_softmax_backward_{c_large,h,h_large,w,w_large}.cpp` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | 3 — second shared-kernel pool | ✓ |
| `moreh_softmax_backward_{h,h_large,w,w_large}.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | 2 — official kernel library | ✓ |
| all 13 kernels | `tt_metal/hw/inc/api/{dataflow,compute,tensor}/*` | 1 — LLK / HAL / firmware | ✓ no concern |

**Shape notes** (per-call detail is otherwise omitted, all rolls being ✓ — but the shapes are worth recording, since one of them is not in the recipe's table):

- `generate_bcast_scaler(DataflowBuffer, uint32_t)`, `generate_mask_h(DataflowBuffer, uint32_t)`, `generate_mask_w(DataflowBuffer, uint32_t)` (`kernel/dataflow/moreh_common.hpp:72`, `:183`, `:223`) and the whole `*_to_cb` compute family — `mul_tiles_to_cb`, `add_tiles_to_cb`, `sub_tiles_to_cb`, `copy_tile_to_cb`, `exp_tile_to_cb`, `mask_tile_to_cb`, `mul_tiles_and_negative_to_cb`, `mul_tiles_and_mask_tile_to_cb`, `mul_tiles_bcast_{rows,cols}_to_cb`, `sub_tiles_bcast_{rows,cols}_to_cb` (`kernel/compute/moreh_common.hpp:139` onward) — all take **`DataflowBuffer` by value**. This is **✓ excellent**, and it is *not* the ⚠-flagged `CircularBuffer&` row: `DataflowBuffer` has a **non-explicit** converting constructor from `DFBBindingToken` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:106`), so a `dfb::name` token converts implicitly at the call site with no bridge, no cast, and no donor-side change. The recipe's shape table has no `DataflowBuffer` row — see *Recipe notes*.
- `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<uint32_t dfb_id, PoolType, ReduceDim, uint32_t reduce_factor>()` (`kernel_lib/reduce_helpers_dataflow.hpp:83`) and `compute_kernel_lib::reduce<PoolType, ReduceDim, uint32_t input_dfb_id, uint32_t scaler_dfb_id, uint32_t output_dfb_id, …>()` (`kernel_lib/reduce_helpers_compute.hpp:381-392`) take CB ids as **`uint32_t` NTTPs** — the ✓ `uint32_t cb_id` row; `DFBBindingToken`'s `constexpr operator uint32_t()` (`dataflow_buffer.h:89`) covers template-parameter position.
- `compute_kernel_hw_startup(uint32_t, uint32_t, uint32_t)` — LLK entry point taking raw ids; same constexpr conversion applies.

**Borrowed kernel files (file-path instantiation):** none. The op owns every kernel source it binds. No `_metal2` fork exists anywhere in `device/kernels/`. Recorded for completeness: an unrelated `_metal2` header fork exists in the same shared pool one of the donors lives in (`ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp`), but it forks a **different** header that this op does not include — there is no `moreh_common_metal2.hpp`, and none is needed, the donor's `DataflowBuffer`-by-value signatures already bridging cleanly.

**Negative pointer (saves a wrong turn):** there is **no** `experimental/quasar/` copy of this op or of any of its kernels — nothing in that out-of-bounds tree to be tempted by, and nothing to mistake for a fork to reuse.

### Relaxation candidates

None. Mining candidates from a custom hash presupposes a custom hash; this op has none, and the ops it would be mined for are the gated ones. Nothing to route to the relaxation roadmap.

### TTNN factory analysis

The sheet-derived facts, with `file:line` evidence. Gate conjuncts are recorded in *Gate detail* above; repeated here in the form the TTNN ProgramFactory wiring consumes:

- **Op-owned tensors:** none. `descriptor` concept throughout; no factory returns a `WorkloadDescriptor`, so no `buffers` vector exists. Output allocation is ordinary TTNN (`create_output_tensors` → `create_device_tensor`, or the caller's preallocated `input_grad_tensor`, `...device_operation.cpp:108-117`).
- **MeshWorkload need:** none — not a `WorkloadDescriptor` op, so neither genuinely multi-program nor an op-owned-tensor artifact. Target concept is the plain `ProgramSpecFactoryConcept`.
- **Pybind `create_descriptor`:** absent (`moreh_softmax_backward_nanobind.cpp:18-63`).
- **Other risky pybind:** none. The nanobind surface is three `bind_function` entries plus `export_enum` of `MorehSoftmaxBackwardOp` and `MorehSoftmaxBackwardOpParallelizationStrategy`. The two exported enums are plain value enums reaching the op only through `operation_attributes_t` — no factory or descriptor internals are exposed, so nothing here creates a hand-port dependency.
- **Custom hash:** absent — the framework default hash applies.
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** absent.

One factory-shape note for the wiring: the compute kernel is emitted **twice per factory** (`compute_desc_1` over `core_group_1`, `compute_desc_2` over `core_group_2`, the second guarded by `has_core_group_2`), differing only in the leading CTA. That maps to two `KernelSpec`s in two `WorkUnitSpec`s over disjoint `target_nodes`, not to one spec with a demoted RTA.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

- **Dead RTA — `scaler` is passed to the W readers and never read.** Both W factories push `std::bit_cast<uint32_t>(scaler)` at RTA **index 5** (`softmax_backward_w_small.cpp:240`, `softmax_backward_w_large.cpp:256`, with the value fixed at `1.0f` two lines above each), but `reader_moreh_softmax_backward_w.cpp:19` and `reader_moreh_softmax_backward_w_large.cpp:19` read `mask_w` from **index 6** and never touch index 5 — the reader obtains its scaler on-device from `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<…>()` instead. The host value, and the local `float scaler` computing it, are dead plumbing. (The H factories' `scaler` at index 6 *is* live: `reader_moreh_softmax_backward_h.cpp:19` reads it and passes it to `generate_bcast_scaler`.)
- **Two unreferenced kernel files.** `device/kernels/writer_moreh_softmax_backward_h.cpp` and `device/kernels/writer_moreh_softmax_backward_w.cpp` are bound by no factory in the repo; they are near-duplicates of the writers that *are* bound (`writer_moreh_softmax_h.cpp` / `writer_moreh_softmax_w.cpp`, same directory). They are still installed by the family CMake glob (`ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:53`). Candidates for deletion by the ops team; out of port scope.
- **Hardcoded 512 KiB L1 budget in the small-path heuristics.** `#define L1_512KB (512 * 1024)` (`device/moreh_softmax_backward_device_operation.cpp:11`) bounds both `is_moreh_softmax_backward_w_small_available` and `..._h_small_available` (`:31`, `:52`). WH and BH both have ~1.5 MiB of L1, so the check under-selects the faster small path on tensors that would in fact fit.
- **The same heuristic under-counts the intermediate CBs.** It sizes *every* CB with the **data-format** tile size (`tile_size(data_format)`, `:19`/`:40`), but the factories allocate `c_24`/`c_25`/`c_26` at the **intermediate** format size — `tt::DataFormat::Float32` whenever `fp32_dest_acc_en` (e.g. `softmax_backward_w_small.cpp:53-55`, `:108-136`). With fp32 dest accumulation on, the estimate is ~2× low on those CBs, so the availability check can admit a configuration that does not actually fit. The two errors push in opposite directions, which may be why neither has surfaced.
- **Tautological validation guard.** `TT_FATAL(dim >= 0 && dim < rank, …)` (`device/moreh_softmax_backward_device_operation.cpp:87`) — `dim` is `uint32_t` (`...hpp:34`), so `dim >= 0` is always true; a negative `dim` reaching here has already wrapped to a large positive and is caught only by the `< rank` half.
- **Unused kernel locals.** `uint32_t l1_write_addr_in;` is declared and never used in all four H/W readers (e.g. `reader_moreh_softmax_backward_h.cpp:27`, `reader_moreh_softmax_backward_w.cpp:26`); `constexpr uint32_t onetile = 1;` is declared but unused in the writers, which loop on `blk` instead (e.g. `writer_moreh_softmax_w.cpp:17` vs `:22`).
- **`log_info(tt::LogTest, …)` on the production path.** All five factories log `"Small tensor algorithm selected"` / `"Large tensor algorithm selected"` on the `LogTest` channel at every cache miss (e.g. `softmax_backward_w_small.cpp:21`, `softmax_backward_c_large.cpp:20`). Wrong channel for shipped op code.

## Questions for the user

1. **Readiness-sheet row count (the one cross-check I could not run):** the sheet could not be fetched from this session — the claude.ai Google Drive connector is unauthorized and the session is non-interactive, so the download tool is not available to call, and the recipe forbids delegating the fetch to a subagent. Your supplied `Is safe to port?` = `yes` / `Is able to port?` = `yes` covers the gate, and I confirmed every other conjunct from the code (see *Gate detail*). The remaining gap is the **factory-set match**: does the sheet carry exactly **five** rows for `moreh/moreh_softmax_backward` — one each for `WSmall`, `WLarge`, `HSmall`, `HLarge`, `CLarge` (per `device/moreh_softmax_backward_device_operation.hpp:58-70`) — with no phantom row for a renamed/deleted factory and no factory missing a row? A five-row check by eye closes it. A mismatch would mean the sheet is stale for this op and route to the readiness-sheet owner; it would not change any individual factory's verdict, since all five are structurally identical on every gate conjunct.
2. **`TensorParameter relaxation` cell (same fetch limitation, low stakes):** I recorded `none`, inferred from the absent custom hash. If the sheet actually proposes a relaxation for any of the five factories, that contradicts the missing hash and is itself a finding for the ops team — worth a glance while you are in the sheet.

## Recipe notes

1. **The shape table has no `DataflowBuffer` row, and the nearest row is the ⚠-flagged one.** *Out-of-directory coupling → Per-call shape analysis* lists `CircularBuffer` / `CircularBuffer&` / `const CircularBuffer&` as `⭐ ⚠ flag` ("DFB-replaces-CB on the consumer side leaves no clean per-op story today") and `uint32_t cb_id` as `✓ OK`. Every donor this op calls takes **`DataflowBuffer` by value** — `generate_bcast_scaler(DataflowBuffer, uint32_t)`, the whole `*_to_cb` compute family — which is neither row. It is unambiguously fine (`DataflowBuffer` has a non-explicit ctor from `DFBBindingToken`, `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:106`, so `dfb::name` converts implicitly at the call site), but the resemblance to the ⭐ row is close enough that a hurried auditor could flag a clean donor as a cross-team blocker. Suggest a `DataflowBuffer` / `DataflowBuffer&` row: `✓ excellent — implicit conversion from the binding token; the donor is already on the DFB type, so the ⭐ concern does not apply`. Donors in the shared pools appear to be migrating to this shape, so it will keep coming up.

2. **No procedure for "readiness sheet unfetchable, launcher supplied the decisive columns."** `ttnn_op_porting_readiness.md` says the human authorizes the Drive connector "once" and that it "cannot be done from inside the session," and the audit adds *"Do not fetch or cross-check in a subagent."* In a non-interactive session both doors are shut, and the audit's only documented outcome for lacking sheet data is *spreadsheet-broken → GATE, route to the readiness-sheet owner* — which would produce a RED on an op whose every gate conjunct I could verify directly from the code, exactly the too-conservative RED the *"be specific"* preamble warns misroutes work. I resolved it by treating the launcher-supplied `Is able to port?` / `Is safe to port?` as the sheet's verdict, re-deriving every other conjunct from the code, and disclosing the one uncheckable item (factory-set row match) as a question — but that is improvisation, not guidance. Worth a short explicit branch: *when the sheet is unreachable rather than wrong, and the launcher supplies `Is able to port?` (± `Is safe to port?`), verify all remaining conjuncts in code, record the fetch failure and the unrunnable factory-set check, and proceed on the supplied verdict rather than gating.* The distinction that matters is **unreachable ≠ broken** — the spreadsheet-broken gate exists for data we can't *trust*, not data we can't *reach*.

3. **The borrowed-kernel-files inventory excludes the intra-op case by its own wording, yet that is where the brief needs it.** *Out-of-directory coupling → Borrowed kernel files* says to "list every kernel `.cpp` file the op's program factory instantiates whose source it does **not** own." This op's `writer_moreh_softmax_w.cpp` is bound by two of its *own* factories (`WSmall`, `WLarge`) — the **Intra-op** shape that `port_patterns.md`'s *Caution: Porting a shared kernel* explicitly governs ("two factories of *your own* op bind the same kernel and you are porting one of them") — and it is precisely the fact the brief's shared-kernel line has to carry, since it decides fork-vs-convert-in-place. Following the instruction literally, the audit would report "no borrowed kernel files" and say nothing about it. Suggest widening the bullet to *"every kernel `.cpp` bound by more than one factory — whether the other binders are other ops or other factories of this op"*, which also matches the Caution's own three-shape framing (borrowed / lent / intra-op) rather than just the first.

4. **Minor — the CB-endpoint precondition names pre-DFB idioms.** *CB endpoints → Precondition* describes Device-2.0-intact kernels as using "(`get_write_ptr` methods, `get_local_cb_interface`, `Semaphore` objects)." This op's kernels are a step further along: they construct `DataflowBuffer` objects and do all FIFO traffic through `reserve_back`/`push_back`/`wait_front`/`pop_front` **methods**. The mapping is obvious and cost me nothing, but naming the DFB-object idiom in that list would make the precondition read as satisfied rather than merely analogous — and the same vocabulary gap appears in the census bullets, which spell the raw-access signals as `get_write_ptr` / `get_read_ptr` / `get_local_cb_interface(<cb>).fifo_*_ptr` without the object-method spelling that a DFB-era kernel actually uses.
