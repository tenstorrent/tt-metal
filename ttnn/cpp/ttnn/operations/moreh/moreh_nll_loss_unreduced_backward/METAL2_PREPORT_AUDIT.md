# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward`

> **Re-audit.** The previous audit was RED at op level: all three readers constructed `DataflowBuffer` objects on `c_7` (and the 2d reader also on `c_8`) that no code path in the factory allocated. That fix is merged — `047fecfec7f Prep moreh_nll_loss_unreduced_backward for Metal 2.0 Port (#53534)`, issue #53527 — and is verified below. This audit re-runs every subject against the current tree, including the seven informational subjects the previous RED deferred.

Single device-operation directory, single program factory:

- **`MorehNllLossUnreducedBackwardDeviceOperation`** (`device/moreh_nll_loss_unreduced_backward_device_operation.{hpp,cpp}`)
  - `Factory` (`device/moreh_nll_loss_unreduced_backward_program_factory.cpp:443`) — the only factory; `program_factory_t = std::variant<Factory>`.

**One factory, three rank-dispatched code paths.** `Factory::create_descriptor` branches on `input_grad.logical_shape().rank()` into three free functions in the same file — `moreh_nll_loss_unreduced_backward_impl_2d` (`:46`), `_impl_3d` (`:182`), `_impl_4d` (`:318`). These are **configs of one factory**, not separate factories: the sheet carries one row, and one port converts all three.

**No compute kernel.** The op runs reader + writer only — the readers compute `input_grad` directly into the output CB with `CoreLocalMem` scalar writes. All 4 kernel files are referenced; none is dead. The writer is shared by all three rank paths.

| Rank path | Reader | Writer |
|---|---|---|
| 2d (`:46`) | `reader_moreh_nll_loss_unreduced_backward_2d.cpp` | `writer_moreh_nll_loss_unreduced_backward.cpp` |
| 3d (`:182`) | `reader_moreh_nll_loss_unreduced_backward_3d.cpp` | *(same)* |
| 4d (`:318`) | `reader_moreh_nll_loss_unreduced_backward_4d.cpp` | *(same)* |

Config axes: rank ∈ {2d, 3d, 4d} × **`WEIGHT`** (optional `weight_tensor`). Dtypes are fixed by validation — `target` INT32, `output_grad` / `weight` / `input_grad` all BFLOAT16 (`..._device_operation.cpp:24`, `:34`, `:49`, `:64`).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `f6033c9ec2d 2026-08-19 docs(metal_2.0): a direct-descriptor op converts to a real program factory`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward` |
| **Overall** | **GREEN** — every gate cleared; the previous blocker is resolved |
| **DOps / Factories** | `MorehNllLossUnreducedBackwardDeviceOperation` → `Factory` (3 rank paths: 2d / 3d / 4d) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 4 own kernels and both donors are structurally Device 2.0 |
| *Prereqs* — Cross-op escapes | **Ok** — `✓ clean`; no borrowed kernel *files*, two header donors, both ✓ shape |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (launcher-supplied this session; cross-check clean) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | **No** (not a gate either way) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** |
| *TTNN Readiness* — `override_runtime_arguments` | **No** (not a gate; selects the base target concept) |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** (not a gate; nothing to delete) |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — TensorParameter relaxation | **`none`** (launcher-supplied, verbatim) → clears |
| *TTNN Readiness* — `Known op issues` | **Empty** (launcher-supplied) → no blocker |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | **none** — no `->address()` anywhere in the op; every base is a `Buffer*` binding, zero host arithmetic |
| *Port work* — Tensor bindings (per binding) | **Case 1 × 4** — `target`, `output_grad`, `weight` (conditional), `input_grad` |
| *Port work* — TensorAccessor 3rd arg | **N/A** — no accessor in the op passes a 3rd argument, so the subject never fires |
| *Port work* — CB endpoints | **5 self-loops + 1 plain 1P+1C** (2d); 4 + 1 (3d/4d). Two conditionally-declared DFBs. No multi-binding, no dead CB. |

**CB endpoints** are dispositions, not gates. Recorded per `(CB, config)` below.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, same directory).

**The previous blocker is resolved, and the fix is exactly scoped.** `047fecfec7f` added 17 lines to one file — nothing else in the op changed — bringing CB allocation into exact correspondence with kernel usage:

| CB | Allocated (after fix) | Used by | Match |
|---|---|---|---|
| `c_7` weight scratch | `if (weight_has_value)` in **all three** rank impls (`:90`, `:226`, `:363`) | all three readers, inside `#if defined(WEIGHT)` | ✓ exact, including the guard |
| `c_8` output_grad scratch | unconditionally, **2d only** (`:93`) | 2d reader only, unconditionally | ✓ exact, including the rank scope |

Nothing is now over-allocated (no dead CB) and nothing under-allocated. The commit also confirms the Blackhole failure mode the previous audit predicted: run under `TT_METAL_WATCHER=1` on BH, `main` tripped a Watcher NOC error; after the fix, 28 passed / 16 skipped (the 16 being the unconditionally-skipped `bfloat8_b` cases, no arch-related skips). That is a stronger verification than the audit asked for.

Every gate clears on the current tree: Device 2.0 across all 4 own kernels and both donors, all three Appendix A entries `N/A`, the TTNN factory-concept gate, the `TensorParameter relaxation` gate (`none`), offset base pointers (no address RTA exists at all), and the `TensorAccessor` 3rd argument (no site passes one).

The port itself is small and unusually clean. There is **no compute kernel**, so the whole op is one reader plus one writer: four Case-1 tensor bindings, a CB census that is almost entirely reader-local self-loops, no CTAs anywhere, no cross-op kernel sharing, and no semaphores. The one structural thing to get right is conditionality — `c_2`/`c_7` exist only under `WEIGHT` and `c_8` only on the 2d path, so their DFB specs must be declared conditionally, mirroring the guards the factory already has.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The sheet's cell reads `yes`, which clears **this prerequisite and nothing else**.

  The readiness sheet cannot be fetched from this session: the claude.ai Google Drive connector is unauthorized and the session is non-interactive, so `mcp__claude_ai_Google_Drive__download_file_content` is not present to call, and the recipe forbids delegating the fetch to a subagent. The three non-derivable cells were supplied by the launcher **earlier in this same session** for this same op — `Is able to port?` = `yes`, `TensorParameter relaxation` = `none`, `Known op issues` = empty — and are reused here under the recipe's *"once per session is enough"* fetch rule. Disclosed so a reviewer can check it in one glance: the only code change since those values were read is `047fecfec7f`, which adds CB allocations and cannot alter what any of the three cells mean. This is **not** one of the four *spreadsheet-broken* triggers — the sheet is neither wrong nor silent for this op, only unreachable from here.

  Per the recipe I do **not** reproduce or recompute the derived verdict. The cross-check of the **primary** (factual) columns against the code is clean:

  | Column | Sheet (supplied / expected) | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` on `Factory`, `..._device_operation.hpp:35-40`; defined at `..._program_factory.cpp:443` | ✓ |
  | `Custom hash` | `no` | No `compute_program_hash`, and no backdoor `attribute_values` / `to_hash`, anywhere in the op | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | Hook absent from the device-op (`..._device_operation.hpp:43-49` declares only `validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`) | ✓ |
  | `Override runtime args method?` | `no` | Absent → target concept is the base `ProgramSpecFactoryConcept`, not `CustomProgramSpecFactoryConcept` | ✓ |
  | `Pybind descriptor` | `no` | `moreh_nll_loss_unreduced_backward_nanobind.cpp` has no `create_descriptor` binding → nothing for the port to delete, so no user-visible API change from this column | ✓ |
  | `Secretly SPMD Workload?` | N/A | Not a `WorkloadDescriptor` op | — |
  | `Op-owned tensors?` | `no` | `descriptor` concept; output is the caller's `input_grad_tensor` or an ordinary `create_device_tensor` | ✓ |
  | **Factory-set match** | 1 row | Code has exactly one factory (`program_factory_t = std::variant<Factory>`, `..._device_operation.hpp:42`); the three rank impls are internal branches of it, **not** additional factories | ✓ |

  Cross-column invariants hold. Per the recipe I did **not** verify `Is safe to port?` — the readiness doc records that column as stale and states the audit does not read it.

- **TensorParameter relaxation: GREEN.** `none`, verbatim — the only value that clears. Not re-derived; that analysis belongs to the ops team, and this op has no custom hash from which a candidate could be mined.

- **Device 2.0 (every kernel used): GREEN.** All 4 own kernels and both donors are structurally Device 2.0.

  - **Readers** (`reader_..._{2d,3d,4d}.cpp`) — `TensorAccessor` for all tensor addressing; `DataflowBuffer` objects with `reserve_back`/`push_back`/`wait_front`/`pop_front` methods; `CoreLocalMem<volatile T>` typed L1 pointers seeded from `dfb.get_read_ptr()` / `dfb.get_write_ptr()` **methods**; NoC traffic delegated to the donor's `Noc`-based `read_tile` / `read_line`.
  - **Writer** — `Noc noc;` + `noc.async_write(dfb, accessor, bytes, {.offset_bytes = 0}, {.page_id = i})` + `noc.async_write_barrier()`; `wait_front`/`pop_front`; the sanctioned `get_tile_size(cb_id)` at `:26`.
  - **Donors** — `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (class 3, second shared-kernel pool): every `get_write_ptr`/`get_read_ptr` in that file is a **method** on the DFB object, and `read_line`'s local L1 copy uses a Device 2.0 `UnicastEndpoint` with `my_x`/`my_y` and `noc.get_noc_id()`. `tt_metal/hw/inc/api/numeric/bfloat16.h` (class 1) contributes only scalar conversions. No legacy addr-gen types, no raw NoC calls, no raw `cb_*` FIFO calls anywhere.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | — | — | *(no violations)* | — |

  **`get_dataformat(cb_id)` is port work, not a holdover.** Each reader calls it twice (`_2d.cpp:34,36`; `_3d.cpp:33,35`; `_4d.cpp:33,35`). It is not on the audit's sanctioned list (still exactly `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`), but the port recipe's **kernel-side whitelist rule 7** names it explicitly among the compile-time tile/format metadata free functions the **port** moves onto the object or the binding token. So it is a port-stage conversion, cleanly on the GREEN side of this gate. All six results are **dead** in any case — `weight_data_format` and `output_grad_data_format` are assigned once and never read in any reader — and two of the six query `c_2`, which is not allocated when `weight_tensor` is absent. See *Heads-ups*.

- **Feature compatibility: GREEN.** Every Appendix A entry is absent. A source-only grep of the op directory for `GlobalCircularBuffer`, `global_circular_buffer`, `remote_index`, `remote_cb`, `GlobalSemaphore`, `global_semaphore`, `address_offset`, `set_address_offset`, `UpdateDynamicCircularBufferAddress`, `cb_descriptor_from_sharded_tensor`, `set_globally_allocated_address`, `.buffer =`, `Semaphore` and `address()` returns **zero** hits across every `.cpp`/`.hpp`.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No GCB type; no `.global_circular_buffer` field on any `CBDescriptor` (all built by the local `push_cb` helper, `:23-42`); no `remote_*` idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | Field never set; no `.buffer` on any `CBDescriptor` at all, so no borrowed-memory CBs either |
  | GlobalSemaphore | N/A | Op uses no semaphores of any kind |

  *CTA varargs* is no longer an Appendix A entry in the current recipe — it ports onto `KernelAdvancedOptions::compile_time_varargs` as ordinary port work. Moot here regardless: the op's kernels contain **no** `get_compile_time_arg_val` call at all, and the only compile-time args are the `TensorAccessorArgs` blocks at `constexpr` offsets.

- **Offset base pointers: GREEN.** No fold exists — and, more strongly, **there is no address RTA in this op at all** (`address()`: zero hits). Every base reaches its kernel through the descriptor-API **`Buffer*`-binding form**: `reader_desc.emplace_runtime_args(core, {target_buf, output_grad_buf, weight_buf, ignore_index, …})` and `writer_desc.emplace_runtime_args(core, {input_grad_buf, units_per_core, tile_offset})`, with the factory's own comment recording the intent (*"Pass `Buffer*` (not a raw address) so the program-cache fast hit path re-patches the binding when the tensor is reallocated; nullptr is fine for an absent optional"*). There is no expression into which an offset could be folded and no host arithmetic on any base. Type 3 (`address_offset`) is absent per the Appendix A row; Type 4 (`ttnn::narrow`) does not appear. The offset-base-pointer triage analysis (a dated prior) carries no `nll` entry — consistent with the scan, and the scan decides it.

- **TensorAccessor 3rd argument: N/A — the subject never fires.** All **10** `TensorAccessor(` construction sites take exactly **two** arguments (three readers × three accessors — `target`, `output_grad`, `weight` — plus the writer's `input_grad`), and neither donor constructs a `TensorAccessor`. No accessor passes a page-size override, so there is nothing to classify and no Class-1/2 drop to schedule. *(Stated as "no sites", not "every site is Class 2" — different findings.)*

- **CB endpoints (GATE-free): 5 self-loops + 1 plain 1P+1C.** Each node runs exactly two kernels — one reader, one writer. There is no compute kernel, and neither dataflow kernel is instantiated more than once, so there is no per-core-group or same-source pair anywhere in this op.

  Both faces of the multi-binding hunt were run and came back empty. **(a) Hidden second writer:** the reader's raw `get_write_ptr()` write into `c_16` is bracketed by its own `reserve_back` … `push_back` — the producer's own peek, not a second toucher — and the op has no semaphores to coordinate a co-fill. **(b) Multiple readers:** no borrowed-memory CB exists (no `.buffer` anywhere), and no CB is touched by two co-resident kernels except `c_16`'s legal pair. **(c) Dual-instance work-split:** absent.

  | CB | Role | Toucher(s) | Census | Disposition | Configs |
  |---|---|---|---|---|---|
  | `c_0` | `target` | **reader only** — produces via `read_tile`, consumes (`wait_front`/`pop_front`), peeks (`get_read_ptr`) | 1 toucher | **self-loop** | all |
  | `c_1` | `output_grad` | **reader only** — 2d: produces via `read_line(Nt)`, `wait_front(Nt)` + peek, never pops. 3d/4d: produces via `read_tile` per iteration, `wait_front`/`pop_front` + peek | 1 toucher | **self-loop** | all |
  | `c_2` | `weight` | **reader only** — produces via `read_line(Ct)`, `wait_front(Ct)` + peek, never pops | 1 toucher | **self-loop** | `WEIGHT` only |
  | `c_7` | `weight_scratch` | **reader only**, inside `read_line` — **sync-free**: NoC-written and `get_write_ptr()`-read, with *no* FIFO ops at all | 1 toucher | **self-loop** | `WEIGHT` only |
  | `c_8` | `output_grad_scratch` | **reader only**, inside `read_line` — sync-free, same shape | 1 toucher | **self-loop** | **2d only** |
  | `c_16` | `input_grad` | reader produces (`reserve_back` + raw `get_write_ptr` write + `push_back`); writer consumes (`wait_front`/`pop_front`) | 1P + 1C | plain 1:1 ✓ | all |

  `c_7` and `c_8` are textbook *sync-free single-ended* CBs: `read_line` NoC-reads DRAM into the scratch and copies the valid bytes out via a local L1 unicast read from `cb_scratch.get_write_ptr()`, with no `reserve_back`/`push_back` on the scratch at any point. One toucher, no FIFO → self-loop.

  Two consumers legitimately never pop: `c_1` on the 2d path (the whole `Nt`-tile row is held for the loop) and `c_2` (the whole weight line is held). A held single-toucher CB is still a self-loop.

  **No dead CB, and no dead-CB-derived conditional.** Every allocated `buffer_index` is referenced by a kernel in the configuration that allocates it — that is exactly what `047fecfec7f` established. `c_2`/`c_7`/`c_8` are **conditionally declared** rather than conditionally dead: the factory already gates their allocation (`weight_has_value`, and the 2d-only placement), so the port carries that conditionality into the DFB specs. Per-config disposition set: **2d** = self-loop {`c_0`, `c_1`, `c_2`\*, `c_7`\*, `c_8`} + plain {`c_16`}; **3d/4d** = self-loop {`c_0`, `c_1`, `c_2`\*, `c_7`\*} + plain {`c_16`} (\* = `WEIGHT` only).

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — four, all **Case 1**, uniform across the three rank paths:
  - `target` — `Buffer*` at reader RTA 0 → `TensorAccessor(target_args, target_addr)`, consumed via the donor's `read_tile`.
  - `output_grad` — `Buffer*` at reader RTA 1 → `TensorAccessor(output_grad_args, output_grad_addr)`; via `read_line` on 2d, `read_tile` on 3d/4d.
  - `weight` — `Buffer*` at reader RTA 2 (or `nullptr` when absent), **conditional**: the accessor is built only inside `#if defined(WEIGHT)`.
  - `input_grad` — `Buffer*` at writer RTA 0 → `TensorAccessor(input_grad_args, input_grad_addr)`.

  **No Case 2.** The readers use raw typed pointers heavily (`CoreLocalMem<volatile uint16_t> input_grad_l1_ptr(dfb_input_grad_obj.get_write_ptr())` and friends), but every one is a pointer into **CB/L1 memory obtained from a DFB method** — never a tensor base from an RTA. No kernel does address arithmetic on a tensor base, so no binding needs the `get_bank_base_address` bridge. No **clean** (borrowed-memory DFB) bindings either, since the op has no Buffer-backed CB.

  Delivery note: the bases arrive as `Buffer*` entries rather than `->address()` values, so the framework already registers `BufferBinding`s and patches them on cache hits — routine conversion, not a stale-pointer repair.

- **Conditional bindings and the placeholder CTA chain.** The factory appends **three** `TensorAccessorArgs` blocks to the reader's CTAs unconditionally, passing `nullptr` for an absent `weight` (`:90-92` and the 3d/4d equivalents). A null block still emits two words, which is what keeps the kernel's offset chain (`target_args` → `output_grad_args` → `weight_args`) aligned across configs. Under Metal 2.0 the framework builds accessor args from the bindings, so **both the placeholder block and the offset chain disappear**; express `weight` as a conditional binding instead.
- **TensorParameter relaxation:** `none` — nothing to apply.
- **TensorAccessor 3rd arg:** none — no site passes one.
- **CB endpoints:** self-loop `c_0`, `c_1` (all configs), `c_2` and `c_7` (`WEIGHT`), `c_8` (2d only); `c_16` is a plain 1P+1C needing no action. Declare `c_2`/`c_7`/`c_8`'s DFB specs conditionally, mirroring the factory's existing guards. No 1P+1C assignment decisions, no multi-binding flag, no dead-CB drop.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. Both faces hunted, both empty.
- **Cross-op / shared kernels:** **none.** All four kernel sources live in this op's own `device/kernels/` and are bound only by this op's single factory (one binder each, verified repo-wide); the op borrows no kernel *file*. No `_metal2` fork exists beside any of them, and there is no `experimental/quasar/` copy of this op — so no fork question arises at all. The only sharing is at the header level (two donors, both ✓ — see *Team-only*).
- **RTA varargs:** none. Each reader reads 9 args and the writer 3, all through a fixed `i++` run at the top of `kernel_main` — a fixed run over a fixed set, which the recipe classifies as ordinary positional plumbing that dissolves into named args, **not** a loop. No counted loop over arg indices, no data-selected read. Names are legible from the kernel locals: `target_addr`, `output_grad_addr`, `weight_addr`, `ignore_index`, `num_tiles_per_core`, `start_id`, then per rank `Nt`/`C`/`Ct` (2d), `C`/`Ct`/`Wt` (3d), `num_inner_tile`/`C`/`Ct` (4d); writer `input_grad_addr`, `num_tiles_per_core`, `start_id`.
- **The direct-descriptor exception does *not* apply to this op.** The recipe's newest sanctioned exception — *"Give a direct-descriptor op a conventional program factory"* — fires only when the device-operation declares `create_descriptor` as its own static member **with no `program_factory_t`**. This op already has the nested `Factory` struct and `using program_factory_t = std::variant<Factory>` (`..._device_operation.hpp:35-42`), so the exception is closed and the port is a **method swap inside the existing struct**, with the device-operation class untouched. Flagged because a moreh op whose factory struct is named bare `Factory` superficially resembles the shape that exception targets. (The recipe's `<OpName>ProgramFactory` naming convention applies to the *conversion* case; renaming an existing struct is not port work.)
- **The dead `get_dataformat(cb_id)` locals must not become DFB metadata accesses.** Six dead calls across the three readers (`_2d.cpp:34,36`; `_3d.cpp:33,35`; `_4d.cpp:33,35`). Two of the three per reader query `c_2` (weight), which is **not allocated** in the non-`WEIGHT` config — so its `dfb::` token will not exist, and converting these under whitelist rule 7 would name an unbound DFB. They are provably dead, so deleting them is behaviour-preserving; see *Questions* for the ops-team confirmation.
- **Anything else the porter needs:** the whole compute-kernel-config path is vestigial — the op has no compute kernel, and its one derived `FP32_DEST_ACC_EN` define is read by nothing. Do not go looking for a compute kernel to configure, and do not carry that define into the port. Detail under *Misc anomalies*.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: `✓ clean`.** No entry is ⚠, ✗ or ⭐.

- **No file-path kernel instantiation escape.** All four bound `kernel_source` paths are under this op's own `device/kernels/`. The op borrows no kernel file and lends none — each of its four kernels has exactly one binder repo-wide.
- **Function-call escape: two header donors, both ✓.**

| Op kernel | Donor file | Donor class | Functions called | Shape | Status |
|---|---|---|---|---|---|
| all 3 readers | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 3 — second shared-kernel pool | `read_tile(DataflowBuffer, AddrGen, uint32_t, …)` (`:666`), `read_line(DataflowBuffer, DataflowBuffer, AddrGen, uint32_t, …)` (`:739`), `get_tilized_idx(uint32_t, uint32_t)` (`:618`) | `DataflowBuffer` **by value** + accessor as a **template parameter** | ✓ excellent |
| all 3 readers | `tt_metal/hw/inc/api/numeric/bfloat16.h` | 1 — LLK / HAL | `bf16_to_fp32(uint16_t)` (`:20`), `fp32_to_bf16_truncate(float)` (`:44`) | plain scalars | ✓ no concern |
| writer | `tt_metal/hw/inc/api/{dataflow,tensor}/*` | 1 — LLK / HAL | — | — | ✓ no concern |

**Shape note.** `read_tile` / `read_line` take `DataflowBuffer` **by value**, which is the recipe's newly-added `DataflowBuffer` row — **✓ excellent, no donor-side change and no fork**: build a named `DataflowBuffer` from the token and pass it, which is exactly what these readers already do (`DataflowBuffer dfb_weight_obj(cb_weight); read_line(dfb_weight_obj, dfb_weight_scratch_obj, addrg_weight, Ct);`). Note this is the ✓ case and *not* the adjacent ⭐-flagged `CircularBuffer&` row — the donor has already migrated to the DFB type, so the port changes only where the handle comes from. The accessor parameter is Shape 1 (`TensorAccessor` as a template argument), also ✓: the porter passes `TensorAccessor(tensor::name)` straight through.

**Negative pointer (saves a wrong turn):** there is **no** `experimental/quasar/` copy of this op or of any of its kernels — nothing in that out-of-bounds tree to mistake for prior art or for a fork to reuse.

### Relaxation candidates

None. The cell reads `none`, and the op carries no custom hash from which a candidate could be mined.

### TTNN factory analysis

Sheet-derived facts with `file:line` evidence, in the form the TTNN ProgramFactory wiring consumes:

- **Op-owned tensors:** none. `descriptor` concept; output is the caller's `input_grad_tensor` or an ordinary `create_device_tensor`.
- **MeshWorkload need:** none — not a `WorkloadDescriptor` op.
- **Custom hash:** absent (non-gating either way; nothing for the porter to preserve).
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** absent → the port targets the **base `ProgramSpecFactoryConcept`**; no method to translate.
- **Pybind `create_descriptor`:** absent → nothing for the port to delete; no user-visible API change from that column.
- **Other risky pybind:** none — the nanobind surface is a single `bind_function` with plain tensor / int / optional arguments.

Factory-shape notes: the single `Factory` fans out to three rank impls, each emitting its own kernel and CB set; all three land on the same target concept, and the rank dispatch stays host-side in `create_descriptor`. The op already satisfies `HasProgramFactoryType`, so the direct-descriptor conversion exception does not apply (see *Heads-ups*). Per-core work division rides an **RTA** (`units_per_core`) rather than a per-group CTA — there are no CTAs at all — so the *demoting-per-group-CTA* anti-pattern has no purchase here.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

- **The entire compute-kernel-config path is vestigial, and it feeds the program hash.** The op has **no compute kernel** (`ComputeConfigDescriptor` appears nowhere), yet `operation_attributes_t` carries a `DeviceComputeKernelConfig` (`..._device_operation.hpp:22`), each rank impl calls `get_compute_kernel_config_args` and destructures all five values, and four of them — `math_fidelity`, `math_approx_mode`, `packer_l1_acc`, `dst_full_sync_en` — are never used. The fifth, `fp32_dest_acc_en`, is used only to emplace an `FP32_DEST_ACC_EN` define that **no kernel references** (grep: zero hits for `FP32_DEST_ACC*` under `device/kernels/`). With no custom hash, the attribute still participates in the **default program hash**, so calls differing only in `compute_kernel_config` occupy separate cache entries that compile to identical programs.
- **`writer_defines` is declared, never populated, and moved in empty** — all three rank impls. Only `reader_defines` ever receives entries.
- **Six dead `get_dataformat` locals.** `weight_data_format` and `output_grad_data_format` are assigned and never read in each of the three readers. Two of the six query `c_2`, which is not allocated when `weight_tensor` is absent — harmless only because the values are unused. See *Heads-ups* for why the porter cannot simply modernise them in place.

## Questions for the user

1. **Confirm the six dead `get_dataformat` locals can be deleted (ops team).** They are provably unused, so removing them is behaviour-preserving — but two per reader query a CB that does not exist in the non-`WEIGHT` config, and under Metal 2.0 they cannot be carried forward as DFB metadata accesses (no binding to read from). Better deleted on the ops track than decided unilaterally by the porter. If they should stay, the port needs a different answer for that config and I would want to know before it starts.
2. **Is the `compute_kernel_config` attribute worth removing?** It is public, hashed, and drives nothing (no compute kernel; its one derived define is unread). An API change and squarely the ops team's call — but it is currently multiplying program-cache entries for no effect.

## Recipe notes

1. **The re-audit-after-an-op-code fix is cheap, and the scoping rule's skip is what made it cheap — worth saying so.** The previous audit deferred the seven informational subjects under the **Red** scoping rule (whole-op RED, blocker cleared on the op-code side). That call paid off exactly as the rule predicts: the fix changed the CB inventory, which is the input to the CB-endpoints census, so a census produced last time would have been redone anyway. What the rule does not say, and could: **on a re-audit, the gate-bearing subjects still need a full re-run, but the diff since the previous audit is a legitimate shortcut for scoping it** — here `git show` on one commit established that only CB allocations changed, which let me re-verify the five gates by targeted grep instead of re-deriving them. A sentence pointing re-auditors at the previous report plus the intervening diff would make that explicit rather than something each auditor improvises.

2. **`TensorParameter relaxations` is still listed among the seven never-gating subjects while carrying finding role GATE.** Unchanged since I raised it on the previous audit of this op: the **Red** scoping rule enumerates it in *"These seven never gate"*, and the subject ends with *"**Finding role: GATE** (routed to the ops team)"*. It did not bite this time because the op is GREEN, but on a RED it forces the same improvisation as before (read the cell because it gates; skip the candidate-mining because it is informational). **Suggest moving it into the gate-bearing list** — it is a one-cell read.

3. **The unreachable-sheet workaround is still undocumented, and the new once-per-session wording interacts with it.** The recipe now says *"Pull a fresh copy of the sheet — once per session is enough, so a session auditing several ops fetches once"*, which is helpful, but it presumes a session that **can** fetch. This session cannot (connector unauthorized, non-interactive, subagent delegation forbidden), so the launcher supplies the three non-derivable cells. That combination raised a question the recipe does not answer: **on a re-audit of the same op in the same session, may previously-supplied cell values be reused?** I judged yes — same session, and the only intervening code change cannot alter what those cells mean — and disclosed the reuse so a reviewer can check it. **Suggest documenting both halves:** the ask-the-launcher path when the sheet is unreachable, and that launcher-supplied values inherit the same once-per-session lifetime as a real fetch. *(Raised on the previous three audits; repeating because the re-audit case is a new wrinkle on it.)*

4. **Minor — the `Anything else the porter needs` bullet earned its keep on the first op I hit it with.** The bullet is new and described as *"open by design; the list is not closed."* It was the right home for this op's vestigial compute-kernel-config path: not a gate, not a CB/binding/vararg finding, but something a porter would otherwise waste time looking for a compute kernel to configure. No change requested — recording that the open bullet worked as intended, since a note saying a new affordance was used as designed is presumably as useful to the maintainer as one reporting friction.
