# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/eltwise/unary`

One device operation, one program factory:

- **`UnaryDeviceOperation`** (`device/unary_device_operation.{hpp,cpp}`)
  - `ProgramFactory` (`device/unary_program_factory.cpp`) — `std::variant<ProgramFactory>`, no `select_program_factory`

Host-facing entry point is `unary.cpp`; it is a pure dispatch layer — `prim::unary` at `unary.cpp:68` — and carries no device constructs.

**The whole directory was swept, and it holds exactly one auditable porting unit.** A grep for `struct *DeviceOperation` / `program_factory_t` / `create_descriptor` / `create_workload_descriptor` / `create_program_artifacts` across every `.cpp` and `.hpp` in the directory returns only the three lines above. Two things in the directory that might look like additional scope but are not:

- **`device/unary_composite_op.{cpp,hpp}`** is host-side op *composition*, not a device operation. It builds its results by calling **other** TTNN ops — `ttnn::square`, `ttnn::sum`, `ttnn::bcast`, `ttnn::slice`, `ttnn::reshape_on_device`, `ttnn::lgamma`, `ttnn::add`, `ttnn::subtract` (`device/unary_composite_op.cpp:41-116`). Each of those carries its own readiness row and its own audit; none is part of this porting unit, and nothing here gates on them.
- **`alias.hpp`** (12 lines) and **`unary_composite.hpp`** (64 lines) are declaration-only surface with no device constructs.

**Kernels this factory references** (the audit scope):

| Role | File |
|---|---|
| Reader | `device/kernels/dataflow/reader_unary.cpp` |
| Writer | `device/kernels/dataflow/writer_unary.cpp` |
| Compute (9, selected by `get_compute_kernel_path`, `common/unary_op_utils.cpp:1190`) | `compute/eltwise_sfpu.cpp` (default), `eltwise_identity_kernel.cpp`, `logit_kernel.cpp`, `hardswish_kernel.cpp`, `logsigmoid_kernel.cpp`, `where_tss_kernel.cpp`, `mac_tss_kernel.cpp`, `lgamma_kernel.cpp`, `lgamma_fast_kernel.cpp` |

**Unreferenced kernel files in the same directory — out of scope, and a trap.** `device/kernels/dataflow/` also holds 9 kernels no factory in *this* op references. Six are heavily shared with other ops and three already carry `_metal2` forks. They are flagged here only because their names and their existing forks are easy to mistake for this op's — see *Out-of-directory coupling* below.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `86bccbc58a5 2026-09-17 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

> **Note on one input.** `analyses/relaxations/eltwise_unary.md` was **revised while the first pass of this audit was in progress.** Its verdict did not move — validity check 1 still fails, so the relaxation is still UNCONFIRMED — but three things in it did: the prescribed resolution is now a specific one-line hash change (previously it argued for rejecting the domain), `match_page_size` has been **dropped** from the §2 declaration, and the non-`32x32` tile bug is now explicitly **not** a reason to hold the port. All five validity checks were re-run against the revised text, and its new load-bearing claims were independently verified in code (see the relaxation detail section). This report reflects the revised document. Note the doc is **untracked** in git, so it has no commit to pin and can move again without any signal — see *Recipe notes* item 4.

> **Re-verification, second pass.** This audit was re-run against the op **directory** as the stated target, having first been run against `unary.cpp`; per the recipe's scope rules both resolve to the same porting unit, so this is one audit, re-verified rather than duplicated. Nothing material moved, and everything was re-checked rather than carried over:
>
> - **Op code unchanged** — last commit touching the directory is `32e9f88020c 2026-09-10`, and `git status` shows no modifications (only this untracked report).
> - **All five relaxation validity checks re-run** against the code: check 1 still **FAILS** (both hash branches still carry `dtype()` / `layout()` / `memory_config()` with no page config and no `Alignment`, `device/unary_device_operation.cpp:193-209`); checks 2, 3, 4, 5 still **pass**.
> - **All gate scans re-run** — Appendix A signals, `get_dynamic_runtime_args`, and `->address()` sites all return exactly the earlier results (still precisely two bare address sites, `device/unary_program_factory.cpp:585-586`).
> - **Relaxation analysis doc re-read in full** and is byte-for-byte the revised version above.
> - **Audit recipe content is unchanged** — `git diff 32d64b21d06 HEAD -- ai/audit/metal2_audit.md` is empty. Only the *path's* head commit advanced (`32d64b21d06` → `86bccbc58a5`, same subject, a rebase), which is why the provenance line above differs from the first pass while the guidance behind it does not.
> - **The readiness sheet is still unfetchable** — `ToolSearch` for the Google Drive connector still returns no match, so blocker 2 stands unchanged.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/eltwise/unary` |
| **Overall** | **RED** |
| **DOps / Factories** | `UnaryDeviceOperation` → `ProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — GREEN, all 11 referenced kernels clean |
| *Prereqs* — Cross-op escapes | Ok — `✓ clean` (only `tt_metal/api/*` and `ttnn/cpp/ttnn/kernel_lib/`) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries `N/A`) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **UNVERIFIED — gate input unavailable.** The readiness sheet could not be fetched in this session (no Google Drive connector). Neither this cell nor `Known op issues` was read. |
| *TTNN Readiness* — Concept (current) | `descriptor` — `create_descriptor` returns a `ProgramDescriptor` (`device/unary_device_operation.hpp:44`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `compute_program_hash` @ `device/unary_device_operation.cpp:171`, plus the backdoor `operation_attributes_t::to_hash()` @ `device/unary_device_operation.cpp:16` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (grep clean across the op) |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects `CustomProgramSpecFactoryConcept`): `ProgramFactory::override_runtime_arguments` @ `device/unary_program_factory.cpp:570` |
| *TTNN Readiness* — Pybind `create_descriptor` | No (grep clean in `unary_nanobind.cpp`) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `CustomProgramSpecFactoryConcept` *(conditional on the readiness gate clearing)* |
| *Port work* — Offset base pointer | **none** — GREEN |
| *Port work* — Tensor bindings (per binding) | `input` → Case 1 (interleaved) / clean (native-L1 sharded) · `output` → Case 1 (interleaved) / clean (native-L1 sharded) |
| *TTNN Readiness* — TensorParameter relaxation | **UNCONFIRMED** → **GATE** → ops team. Analysis doc `analyses/relaxations/eltwise_unary.md` exists and clears on its own terms, but its **validity check 1 fails** — independently confirmed in code. Prescribed fix is one line in `compute_program_hash`. |
| *Port work* — TensorAccessor 3rd arg | **none** — no accessor in the op passes a 3rd argument, so the subject never fires |
| *Port work* — CB endpoints | `c_0` legal (1:1) · `c_1` **self-loop** + conditional DFB (LOGIT only) · `c_2` legal (1:1) |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. Recorded per `(CB, config)` below.

## Result

**RED — blocked on two findings, routed to two different owners:**

1. **`TensorParameter relaxation` verdict is UNCONFIRMED** → **ops team; the fix is one line.** The op's `compute_program_hash` does not pin `Tile` or `Alignment`, which the relaxation the port must declare requires to be exactly equal. This is validity check 1 of `analyses/relaxations/eltwise_unary.md`, and it fails; per that document's contract the verdict is UNCONFIRMED and no relaxation declaration may be written. Declaring one against a key this loose turns a working `ttnn.relu` into a hard `TT_FATAL` on the second dispatch.
2. **The TTNN factory-concept gate could not be read** → **re-run this subject in an authorized interactive session.** The readiness sheet is fetched through the claude.ai Google Drive connector, which authorizes only in an interactive main session; this session is non-interactive and the tool is not even present. So `Is able to port?` and `Known op issues` are unread. This is *not* a "spreadsheet is broken" finding — I hold no conflicting evidence, just no cell.

**RED at op level; no portable subset.** The relaxation blocker lives in `compute_program_hash`, which is op-wide and unconditional — one factory, no branch to carve a clean subset out of.

**The path forward is short, and the relaxation fix is one line.** Neither blocker is structural. The analysis doc prescribes the minimal resolution: in `compute_program_hash`, hash `input_tensor.tensor_spec().tensor_layout()` in place of the present `dtype()` / `layout()` / `memory_config()` triple (`device/unary_device_operation.cpp:195-197`, `:205-207`). Verified in code — `TensorLayout::attribute_values()` is exactly `(DataType, PageConfig, MemoryConfig, Alignment)` (`tt_metal/api/tt-metalium/tensor/spec/layout/tensor_layout.hpp:75`), which is precisely what `tensorspecs_match_with_relaxation` requires to be equal, and `TensorSpec::attribute_values()` is `(logical_shape_, tensor_layout_)` (`tt_metal/api/tt-metalium/tensor/spec/tensor_spec.hpp:97`), so the two are cleanly separable and the deliberate omission of shape survives untouched. Because it is a strict superset of what the key carries today, it can only *split* cache entries that are currently shared — never merge ones that are currently separate.

**The non-`32x32` tile bug does not gate this port** (see *Misc anomalies* item 1). It is real and confirmed on silicon, but it is orthogonal to Metal 2.0 and routes to the eltwise team on its own track. Once the hash fix lands and the sheet is read, every other gate in this audit is already GREEN, and the port work below is small and fully inventoried.

## Gate detail

- **TTNN factory concept (`Is able to port?`): UNVERIFIED — input unavailable.** The readiness sheet (Diego's *"Operations analysis"*, file ID `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`) could not be fetched: the `claude.ai Google Drive` MCP connector is unauthorized in this session and `mcp__claude_ai_Google_Drive__download_file_content` is not available at all (`ToolSearch` returns no match). No local CSV exists, and `analyses/.gitignore` correctly keeps one from being checked in — so there was no stale copy to (wrongly) fall back on. **`Is able to port?` and `Known op issues` are therefore unread, and this gate is neither cleared nor failed.**

  Two things sharpen this. First, `analyses/relaxations/eltwise_unary.md` §1 states that the sheet's **`Known op issues` cell for this op is "a second, independent block"** not cleared by that document — so there is positive reason to expect an entry in a cell I could not read. Second, everything code-visible in this subject *was* cross-checked and is recorded below, so the re-run is a lookup, not a re-audit.

  | Column | Code-side value | Evidence |
  |---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` returns `ProgramDescriptor`, `device/unary_device_operation.hpp:44` |
  | `Custom hash` | yes (+ backdoor `to_hash`) | `device/unary_device_operation.cpp:171`; `:16` |
  | `Runtime-args update (get_dynamic_runtime_args)` | no | grep clean across the op directory |
  | `Override runtime args method?` | yes | `device/unary_program_factory.cpp:570` |
  | `Pybind descriptor` | no | grep clean in `unary_nanobind.cpp` |
  | `Op-owned tensors?` | no | `descriptor` concept; no `WorkloadDescriptor` |
  | `Secretly SPMD Workload?` | N/A | not a `WorkloadDescriptor` |
  | Factory-set match | expect exactly **1** row: `UnaryDeviceOperation` / `ProgramFactory` | `std::variant<ProgramFactory>`, `device/unary_device_operation.hpp:59`; no `select_program_factory` anywhere in the op |

  Cross-column invariants hold on what is visible: `get_dynamic_runtime_args == no` is consistent with any concept, and `Op-owned tensors? == no` is required on a `descriptor` row.

- **Device 2.0 (every kernel used): GREEN.** All 11 referenced kernels are structurally Device 2.0; there are no violations to route.

  - **Dataflow kernels are Device 2.0-native.** `reader_unary.cpp` and `writer_unary.cpp` use `Noc noc;` + `noc.async_read` / `noc.async_write` / `noc.async_read_barrier` / `noc.async_writes_flushed` / `noc.async_write_barrier`, and `DataflowBuffer dfb_src(cb_id_src)` with DFB methods (`reserve_back`, `push_back`, `wait_front`, `pop_front`). Includes are the current `api/dataflow/{dataflow_api.h,noc.h,dataflow_buffer.h}` + `api/tensor/noc_traits.h` — notably **not** the stale `api/dataflow/circular_buffer.h`.
  - **A targeted scan for Device 1.0 idioms came back empty** across all 11 kernels: no `noc_async_read`/`noc_async_write` free functions, no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front`, no `get_write_ptr(`/`get_read_ptr(` free functions, no raw semaphore addresses, no `get_noc_addr_from_bank_id`, no `evil_set_*`.
  - **The only CB-index free-function calls are sanctioned**, so they are not holdovers and do not knock the op out of Green:

    | File | Line | Call | Wrapper in scope | Disposition |
    |---|---|---|---|---|
    | `device/kernels/dataflow/reader_unary.cpp` | 57 | `get_local_cb_interface(cb_id_src).fifo_page_size` | `DataflowBuffer dfb_src` | **sanctioned** — not a violation |
    | `device/kernels/dataflow/writer_unary.cpp` | 60 | `get_local_cb_interface(cb_id_dst).fifo_page_size` | `DataflowBuffer dfb_dst` | **sanctioned** — not a violation |

    Per the Green bullet, `get_local_cb_interface(cb_id)` stays sanctioned regardless of what object is in scope at the call site, and the Device 2.0 surface still grounds it: the Device 2.0 `CircularBuffer` wrapper implements `get_write_ptr()` / `get_read_ptr()` *by calling* `get_local_cb_interface` (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:125,131,137`). A DFB-side equivalent does exist (`DataflowBuffer::get_tile_size()`, `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:200`), which makes this a **port-stage** move, not a Device 2.0 one — carried to *Heads-ups* below, with the caveat that `fifo_page_size` and `get_tile_size()` are not the same quantity by definition even though this factory happens to make them equal.
  - **Compute kernels are outside the Device 2.0 DM surface.** The nine compute kernels use compute-side LLK APIs (`compute_kernel_hw_startup`, `copy_tile`, `pack_tile`, `tile_regs_*`, `fill_tile`) and `compute_kernel_lib`; the Device 2.0 migration guide's scope is data movement — `Noc`, `CircularBuffer`, `CoreLocalMem<T>`, `Semaphore`, `Endpoints` (`docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md`). Two of them (`eltwise_sfpu.cpp:23-24`, `mac_tss_kernel.cpp:25-26`) additionally hold real `DataflowBuffer` objects for their FIFO ops. Nothing to flag.
  - **Donor kernels carry no Device 2.0 debt** — see *Out-of-directory coupling*; the only donors are `tt_metal` `api/*` and `ttnn/cpp/ttnn/kernel_lib/`.

- **Feature compatibility:** every Appendix A entry, in order. A clean scan is all-`N/A`.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `experimental/global_circular_buffer.hpp` include, no `global_cb` factory parameter. The arcane signal is also absent: neither `CBDescriptor` literal (`device/unary_program_factory.cpp:420`, `:432`, `:444`) sets `.global_circular_buffer`. No `remote_index(`, no `remote_cb_*` identifiers, no `remote_circular_buffer.h`. |
  | CBDescriptor `address_offset` (non-zero) | **N/A** | `.address_offset` is never set on any of the three `CBDescriptor` literals (default zero). No `set_address_offset`, no 4-argument `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. The sharded CBs are plain buffer-backed (`.buffer = src_buffer` @ `:428`, `.buffer = dst_buffer` @ `:452`) — the borrowed-memory pattern, which is a mechanical porting-recipe translation via `DataflowBufferSpec::borrowed_from`, not this entry. |
  | GlobalSemaphore | **N/A** | The op uses **no semaphores of any kind** — no `GlobalSemaphore`, no `CreateGlobalSemaphore`, no `global_semaphore.hpp`, and no plain `CreateSemaphore` / `SemaphoreDescriptor` either. |

- **CB endpoints (GATE-free):** three CBs, all with a clean disposition; nothing blocks a Gen1 port. Census is per CB, per node, per config. There is **no hidden second writer anywhere in this op** — the face-(a) hunt is decisive rather than merely negative, because the op contains no raw `get_write_ptr`/`fifo_wr_ptr` writes *and* no semaphores at all, and semaphore-gated co-fill is the mechanism that face exists to catch. Neither dataflow kernel is dual-instantiated (three `KernelDescriptor`s, three distinct sources — `device/unary_program_factory.cpp:563-565`), so face (c) does not apply, and no CB is read by two co-resident kernels, so face (b) does not either.

  | CB | Config | Touchers on a node | Census | Disposition |
  |---|---|---|---|---|
  | `c_0` (src0, `:420`) | interleaved (TILE or RM) | reader (`reserve_back`/`push_back`, `reader_unary.cpp:43,53,59,62`) · compute (`wait_front`/`pop_front`) | 2 — 1 locked producer + 1 locked consumer | **plain 1:1 — legal**, no action |
  | `c_0` (src0, `:420`) | native-L1 sharded (`.buffer = src_buffer`) | reader (`reserve_back`/`push_back`, `reader_unary.cpp:21-22`) · compute (`wait_front`/`pop_front`) | 2 — 1 locked producer + 1 locked consumer | **plain 1:1 — legal**; the CB is borrowed-memory → `DataflowBufferSpec::borrowed_from` |
  | `c_1` (tmp0, `:431-441`) | **LOGIT only** | `logit_kernel.cpp` alone — it packs into `dfb_tmp0_id` (`:41-45`) and copies back out of it (`:49-56`) | **1** | **self-loop** — bind the compute kernel PRODUCER *and* CONSUMER. Legal on Gen1 for a compute kernel. Also a **conditional DFB**: the allocation itself is gated on `needs_tmp0_cb` (`:75`, `:431`). |
  | `c_2` (output, `:444`) | interleaved (TILE or RM) | compute (`pack_tile` + `reserve_back`/`push_back`) · writer (`wait_front`/`pop_front`, `writer_unary.cpp:45,55,62,65`) | 2 — 1 locked producer + 1 locked consumer | **plain 1:1 — legal**, no action |
  | `c_2` (output, `:444`) | native-L1 sharded (`.buffer = dst_buffer`) | compute (push) · writer (`wait_front`/`pop_front`, `writer_unary.cpp:23-24`) | 2 — 1 locked producer + 1 locked consumer | **plain 1:1 — legal**; borrowed-memory → `borrowed_from` |

  **`c_1` is not a dead CB, and must not be treated as one.** Its allocation and its liveness are perfectly coupled: `needs_tmp0_cb(t)` returns true exactly for `LOGIT` (`device/unary_program_factory.cpp:75`), and `get_compute_kernel_path(LOGIT)` returns `logit_kernel.cpp` (`common/unary_op_utils.cpp:1203`), which is the one kernel that touches `c_1`. So the CB exists in every config where a kernel references it and in no other. What the port needs is a **conditional** `DataflowBufferSpec` on the LOGIT config carrying a self-loop binding — expect this to be new host-side structure, since the legacy factory expresses the condition as a conditional `push_back` rather than something translatable in place.

  One near-miss worth recording so it is not re-derived: `unpack_to_dest_mode[tmp0_cb_index]` is set at `device/unary_program_factory.cpp:403` for *every* op type when `preserve_fp32_precision` is on, including the eight where `c_1` is never allocated. That is a write into a `NUM_CIRCULAR_BUFFERS`-wide array, not a CB reference, so it does not make `c_1` live outside LOGIT — see *Misc anomalies* item 5.

- **Offset base pointers: GREEN.** No address RTA folds a host-side offset into its base.

  The op has exactly **two** `->address()` sites in the entire directory, both in `override_runtime_arguments`, both bare:

  - `device/unary_program_factory.cpp:585` — `const uint32_t src_addr = input.buffer()->address();` → written to `r[0]` (`:613`), consumed as the base of `TensorAccessor(src_args, src_addr)` (`reader_unary.cpp:26`).
  - `device/unary_program_factory.cpp:586` — `const uint32_t dst_addr = output.buffer()->address();` → written to `wr[0]` (`:616`), consumed as the base of `TensorAccessor(dst_args, dst_addr)` (`writer_unary.cpp:28`).

  Neither has any arithmetic. The cache-miss path never takes an address at all — it hands the framework the `Buffer*` itself (`emplace_runtime_args(w.core, {input.buffer(), …})` @ `:531`, `:536`, `:555` and the output mirrors @ `:532`, `:546`, `:557`), which is likewise offset-free. Where the kernels *do* offset, they do it on the device side through the accessor's own page/offset fields (`{.page_id = base_page + r, .offset_bytes = j * chunk_size}`, `reader_unary.cpp:49`) — a relocatable per-access offset, not a folded base.

  Reconciled against the triage prior: `eltwise/unary` is **not** in `analyses/2026-07-19_offset_base_pointers.md`. Fold absent + op unlisted → the fourth outcome, **clean**; both addresses hand off to *TensorParameter analysis* as clean bases. Type 3 (`address_offset`) is `N/A` per the Appendix A row above; Type 4 (`narrow` / `MeshBuffer::create` interior base) does not appear.

- **TensorAccessor 3rd argument: N/A — no accessor in the op passes a 3rd argument, so the subject never fires.** Both accessor constructions are two-argument: `TensorAccessor(src_args, src_addr)` @ `reader_unary.cpp:26` and `TensorAccessor(dst_args, dst_addr)` @ `writer_unary.cpp:28`. There is no manual page-size override to classify, redundant or otherwise — this is the *no sites* outcome, not *sites found and classified Class 2*. Consistent with `eltwise/unary` being absent from `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, though the read above stands on its own.

  Where the kernels need a page size they take it from the CB rather than the accessor — `get_local_cb_interface(cb_id).fifo_page_size` (`reader_unary.cpp:57`, `writer_unary.cpp:60`) on the TILE path, and per-core `chunk_size` / `last_chunk_size` RTAs on the row-major path. Worth pairing with the relaxation analysis's `match_page_size` reasoning: because the row-major kernels never touch the accessor's page size, unary *would* be entitled to set that flag — yet the analysis declines it anyway, on precedent grounds. Independently verified: **no shipped factory in the tree sets `match_page_size`, and none declares any `.relaxations` at all** (grep over `ttnn/cpp/ttnn/operations/` excluding `experimental/quasar/` returns nothing).

### Why the informational subjects were run despite the RED

The **Red** outcome scoping rule skips the seven purely-informational subjects on a whole-op RED with no portable subset, and asks which side the blocker clears on. I judged this **the Exception case — re-audit will read the same code — and ran all seven.** Stating the reasoning so it can be overruled:

- The relaxation blocker is cleared by a **one-line change to `compute_program_hash`** in `device/unary_device_operation.cpp` (see the relaxation detail). **Every line the seven subjects describe is untouched** by it — the factory's CB and kernel construction, both dataflow kernels, all nine compute kernels, the donor set, the RTA layout. The rule's justification is *staleness*, and there is none here.
- The second blocker (an unfetchable readiness sheet) clears entirely off the op's code — squarely the Exception.
- The rule's *test* says "op-code side → skip", and its exemplars (Device 2.0 migration, an offset split, a page-size fix, a PD migration) all rewrite factory or kernel code. A one-line hash change is op-code by location but not by that description. Where test and justification diverge I followed the justification; this is logged in *Recipe notes*.
- **The revised analysis doc closes the one branch where this judgment could have gone wrong.** An earlier revision left open the possibility that the gap would be resolved by fixing `create_descriptor`'s page sizing — which *would* have changed the factory and invalidated the CB and binding line numbers below. The current text rules that out explicitly: the non-`32x32` tile bug is a separate, non-gating, eltwise-team issue (*Misc anomalies* item 1), and the relaxation fix is confined to the hash. So the detail below should survive to port time intact.

## Port-work summary  *(no brief issued — RED)*

- **Tensor bindings** (per binding, per config — the classification splits by config on both):
  - `input` — **Case 1** (via `TensorAccessor`) in the interleaved configs: the base reaches the kernel as a `Buffer*` on the miss path (`device/unary_program_factory.cpp:531,536,555`) and as a raw address on the hit path (`:585` → `r[0]` @ `:613`), and the kernel feeds it straight into `TensorAccessor(src_args, src_addr)` (`reader_unary.cpp:26`), doing all access through the accessor. Express as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::input)` and both the address arg and the `TensorAccessorArgs<0,0>()` plumbing disappear. — **clean** in the native-L1-sharded config: `c_0` is a borrowed-memory CB (`.buffer = src_buffer` @ `:428`) and the reader's accessor is compiled out entirely by `#if SRC_SHARDED` (`reader_unary.cpp:20-26`), so the DFB *is* the tensor access → `DataflowBufferSpec::borrowed_from`, not Case 1 or 2.
  - `output` — the exact mirror: **Case 1** interleaved (`:532,546,557`; `:586` → `wr[0]` @ `:616`; `TensorAccessor(dst_args, dst_addr)` @ `writer_unary.cpp:28`), **clean** native-L1-sharded (`.buffer = dst_buffer` @ `:452`; accessor under `#if DST_SHARDED` @ `writer_unary.cpp:20-28`).
  - Neither binding is the silent-wrong hazard today. The miss path uses the `Buffer*` form, which the framework auto-registers as a `BufferBinding` and patches on cache hits; the hit path rewrites the address explicitly in `override_runtime_arguments`. Both are routine port work, and the typed binding supersedes both mechanisms.
  - Op-level roll-up: **⚠ port work** (two Case-1 bindings in the dominant configs).
- **TensorParameter relaxation:** **cannot be written — UNCONFIRMED.** See the gate above and the detail below. The analysis doc's §2 declaration is, on **both** slots (input and output), unconditionally:

  ```cpp
  .relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true},
  ```

  Recorded here for the team's benefit **only**; it must not be carried into a port while validity check 1 fails. Note `match_page_size` is **not** set (declined on precedent grounds — see the 3rd-arg section), and `match_padded_shape_only` is not either (strictly weaker than `dynamic_tensor_shape`).
- **TensorAccessor 3rd arg:** none — no site to drop.
- **CB endpoints:** `c_0` legal 1:1 (both configs) · `c_1` **self-loop** + **conditional DFB** on the LOGIT config · `c_2` legal 1:1 (both configs). No multi-binding flag, no dead-CB drop.

### TensorParameter relaxation — the UNCONFIRMED verdict in detail

The sheet's `TensorParameter relaxation` cell was not readable (see the gate above), but the analysis doc it would point at exists — `analyses/relaxations/eltwise_unary.md`, which names this exact sheet row (`eltwise/unary` · `UnaryDeviceOperation` · `ProgramFactory`) and states that the cell points there. That document is **authoritative but perishable**, and it carries five validity checks that must all pass before anything in it may be applied. I ran all five against the code:

| # | Check | Result | Evidence |
|---|---|---|---|
| **1** | **The cache key pins `tensor_layout`** — no two tensors differing in dtype, page config *including* `Tile`, memory config, or `Alignment` may share a cache entry. (The check is on the *property*, not any one implementation: hashing the field, normalising it, or rejecting the divergent domain all satisfy it.) | **FAIL** | `compute_program_hash` carries `input_tensor.layout()` (`device/unary_device_operation.cpp:196`, `:206`) — the `Layout` **enum** (`TILE`/`ROW_MAJOR`), *not* the page config. `TensorLayout::attribute_values()` is `(DataType, PageConfig, MemoryConfig, Alignment)` (`tt_metal/api/tt-metalium/tensor/spec/layout/tensor_layout.hpp:75`), so **`Tile` is unpinned** and **`Alignment` is absent entirely**. The other three components are present: `dtype()` (`:195`,`:205`), `memory_config()` (`:197`,`:207`). |
| 2 | The dataflow kernels compile the accessor away when the slot is sharded | **PASS** | `reader_unary.cpp`: `TensorAccessorArgs<0,0>()` @ `:25` and `TensorAccessor` @ `:26` both sit inside the `#else` of `#if SRC_SHARDED` (`:20`…`:65`). `writer_unary.cpp` mirrors it exactly (`:20`…`:69`, accessor @ `:27-28`). The sharded accessor payload is emitted by the host and never read by the device. |
| 3 | `has_sharding` is itself pinned by the key | **PASS** | `dst_shard_vol` is set for exactly `shard_specs.has_value()` (`device/unary_device_operation.cpp:180-187`) and `src_shard_vol` for `shard_specs.has_value() && is_sharded()`; both optionals are hashed on both branches (`:199-200`, `:208-209`). The factory's `has_sharding` is the same `get_shard_specs(...).has_value()` (`device/unary_program_factory.cpp:364`, `:587`), so the compile-time `SRC_SHARDED` / `DST_SHARDED` / `RM_INTERLEAVED` defines and the 3-vs-8 per-core arg-slot count cannot flip on a cache hit. |
| 4 | The op still has exactly one factory | **PASS** | `using program_factory_t = std::variant<ProgramFactory>;` (`device/unary_device_operation.hpp:59`); no `select_program_factory` anywhere in the op. The declaration would be unconditional across factory choice. |
| 5 | The TILE-path key still omits shape, and the override still re-applies the whole split | **PASS** | `padded_shape()` is hashed only on the `ROW_MAJOR` branch (`device/unary_device_operation.cpp:192-201`); the TILE return (`:203-209`) omits it. `override_runtime_arguments` re-enumerates through the same `enumerate_core_rt_args` the miss path uses (`device/unary_program_factory.cpp:596` vs. `:521`). |

**Four of five pass; check 1 fails, and it is the load-bearing one.** Per the document's contract — *"If any validity check fails, do not substitute your own judgement. Stop, and report the relaxation verdict as UNCONFIRMED"* — and per the audit recipe's matching instruction, the verdict is **UNCONFIRMED** and the port is blocked. I did not derive a substitute relaxation; deriving one is precisely what the recipe forbids here.

**Why this is a hard block rather than a caution.** Every relaxation in the framework requires exact `tensor_layout` equality, and no flag reaches inside it. A declaration written against a key that does not pin `Tile` throws on the first cache hit at a different tile. The analysis doc reports this confirmed on silicon: two `bfloat16` TILE tensors with identical padded shape and memory config but tiles `32x32` and `16x32` shared one cache entry.

**Route: ops team. The prescribed fix is one line.** In `compute_program_hash`, replace the `dtype()` / `layout()` / `memory_config()` triple on **both** branches (`device/unary_device_operation.cpp:195-197` and `:205-207`) with `input_tensor.tensor_spec().tensor_layout()`. Three independent checks say this is the minimal correct change:

- It pins **exactly** what the match requires and nothing more — `TensorLayout::attribute_values()` is `(DataType, PageConfig, MemoryConfig, Alignment)` (`tensor_layout.hpp:75`), the four fields validity check 1 names.
- It leaves the deliberate omission of shape intact — `TensorSpec::attribute_values()` is `(logical_shape_, tensor_layout_)` (`tensor_spec.hpp:97`) and `TensorLayout` carries no shape, so the two separate cleanly. `relax_logical_rank` is still required afterwards, because rank lives in `logical_shape`, on the other side of that split.
- It is **safe by construction**: a strict superset of today's key can only split entries that are currently shared, never merge entries that are currently separate. No existing cache behaviour is lost.

**A useful framing for the fix, verified:** every already-ported sibling pins `Tile` and `Alignment` *for free*, because none of them defines a custom `compute_program_hash` and the framework default hashes the whole `TensorSpec`. Confirmed for `copy/typecast` — grep for `compute_program_hash` across `ttnn/cpp/ttnn/operations/copy/typecast/` returns nothing. **Unary's gap exists precisely because its custom hash drops fields the default would have kept.** That is also why this is not a framework deficiency to wait on.

Once check 1 passes, the verdict is `dynamic` and §2's declaration applies as written; nothing else in the analysis document changes.

## Heads-ups  *(recorded for the eventual brief; no brief issued now)*

- **CB endpoints (multi-binding shapes to watch):** **none.** No CB in this op has a ≥3-toucher or FIFO-doubling census, and the hidden-second-writer face is structurally excluded (no raw `get_write_ptr`/`fifo_wr_ptr` writes and no semaphores anywhere in the op). The two dispositions that *are* needed — `c_1`'s self-loop and its conditional DFB — are in *Port-work* above, not here.
- **Cross-op / shared kernels:** **none for the kernels this port touches.** `reader_unary.cpp` and `writer_unary.cpp` are referenced by exactly one file in the tree — `device/unary_program_factory.cpp` — so they are op-owned, not shared, and **no `_metal2` fork exists for either.** No coordination cost, no sunset list.
- ⚠ **Do not mistake the neighbouring forks for this op's.** `device/kernels/dataflow/` contains three `_metal2` files — `reader_unary_interleaved_start_id_metal2.cpp`, `reader_unary_sharded_metal2.cpp`, `writer_unary_interleaved_start_id_metal2.cpp` — and they are forks of *other, shared* kernels that this factory does not reference. `reader_unary_interleaved_start_id_metal2.cpp` is **not** a fork of `reader_unary.cpp`. A porter reaching for "the existing fork beside my kernel" would bind the wrong file. Details in *Out-of-directory coupling*.
- ⚠ **Also out of bounds: `ttnn/cpp/ttnn/operations/experimental/quasar/`.** That tree holds deliberately hacky shortcut copies, and two of them (`quasar/untilize/...`, `quasar/reduction/...`) reference the shared kernels sitting in this op's directory — so a grep from here lands in it easily. Nothing there is a precedent, a naming source, or evidence that a construct is portable.
- **RTA varargs:** **none — every argument is nameable.** Reader and writer read slots 0-2 at constant indices (`reader_unary.cpp:11-13`, `writer_unary.cpp:11-13`) and slots 3-7 at constant indices under `#if RM_INTERLEAVED` (`reader_unary.cpp:30-34`, `writer_unary.cpp:32-36`); every compute kernel reads slots 0-2 at constant indices. No counted loop over `get_arg_val`, no running `arg_index++`, no data-selected index. **No CTA varargs either**: the only compile-time reads are `get_compile_time_arg_val(0)` / `(1)` at constexpr indices (`hardswish_kernel.cpp:14-15`, `logit_kernel.cpp:16`). The CRTAs are the framework-built `TensorAccessorArgs` payload (`device/unary_program_factory.cpp:462-463`, `:481-482`), which the typed binding replaces wholesale.
- **Device 2.0 → Metal 2.0 breadcrumb — confirm, don't swap blind.** `get_local_cb_interface(cb_id).fifo_page_size` (`reader_unary.cpp:57`, `writer_unary.cpp:60`) is a sanctioned Device 2.0 call that kernel-side whitelist rule 7 would move onto the DFB object. The tempting target is `DataflowBuffer::get_tile_size()` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:200`), and it happens to be numerically right here — the factory sets both CB page sizes to `tile_size(df)` (`device/unary_program_factory.cpp:356,358,371-372`) — but *page size* and *tile size* are distinct quantities that coincide only because of that choice. Confirm the DFB exposes a page-size accessor before substituting a tile-size one.
- ⚠ **This port would be the first shipped op in the tree to declare *any* `TensorSpecRelaxations` — budget for that.** Independently verified: a grep for `.relaxations` across `ttnn/cpp/ttnn/operations/` excluding `experimental/quasar/` returns **zero** hits. The only in-tree precedent is five factories under `experimental/quasar/transpose` setting `.relaxations = {.dynamic_tensor_shape = true}` on both slots — and that tree is out of bounds as a source, so it is cited here only as the analysis doc cites it, not as something to copy from. Unary matches that shape and adds `relax_logical_rank`, which **no** factory in the tree sets yet. Practical consequence for the porter: **treat a validation throw during the port as a plausible framework-side gap, not automatically a mistake in the declaration** — there is no shipped op to diff against.
- **The row-major and TILE paths differ only in compile-time defines, not in kernel identity.** One reader source and one writer source serve all four regimes via `SRC_SHARDED` / `DST_SHARDED` / `RM_INTERLEAVED`. Whatever the port does to the binding layer applies to every regime at once — convenient, but it means a mistake is not config-localized.
- **`constexpr`, not `const`, on every CB handle.** `constexpr auto cb_id_src = tt::CBIndex::c_0;` (`reader_unary.cpp:15`), and likewise in the writer and all nine compute kernels. That is the form that admits the token-style `dfb::name` binding rather than forcing a member-getter — worth noting because it is already correct and should stay that way.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: `✓ clean`.** No donor needs work; no coupling sequences this port.

- No pre-Device-2.0 donor (no Shape 4 old-style addr-gen anywhere).
- No `CircularBuffer&` donor signature (the ⭐ friction case) — the shared library this op calls into is DFB-era.
- No semaphore-shaped donor parameters of any kind (the op has no semaphores).
- No `TensorAccessor<DSpec>` / `TensorAccessorArgs<N>` / CTA-offset-NTTP donor parameters.

**Summary table** — one row per (op kernel, donor file), collapsed by donor class since every entry in a class shares a shape:

| Op kernel(s) | Donor | Class | Shape | Status |
|---|---|---|---|---|
| `reader_unary.cpp`, `writer_unary.cpp` | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h` | 1 — `tt_metal/*` LLK/HAL/firmware | n/a | ✓ |
| all 9 compute kernels | `api/compute/*` (`common.h`, `compute_kernel_hw_startup.h`, `tile_move_copy.h`, `eltwise_unary/*`, `eltwise_binary*.h`, `mul_int_sfpu.h`), `api/dataflow/dataflow_buffer.h` | 1 — `tt_metal/*` LLK/HAL/firmware | n/a | ✓ |
| `eltwise_identity_kernel.cpp`, `logsigmoid_kernel.cpp`, `logit_kernel.cpp`, `hardswish_kernel.cpp`, `where_tss_kernel.cpp`, `lgamma_kernel.cpp`, `lgamma_fast_kernel.cpp` | `ttnn/cpp/ttnn/kernel_lib/eltwise/**` — `api/chain.hpp`, `api/convenience.hpp`, `unary/{math,misc,activations,special,trig,rounding,scalar,predicates}.hpp`, `binary/sfpu/{basic,extended}.hpp`, `generators/fill.hpp`, `core/optional.hpp` | 2 — official shared kernel library | `uint32_t cb_id` | ✓ |

**Per-call detail.** Included although every roll-up is `✓`, because the shared-library shape is worth pinning: the `compute_kernel_lib` entry points (`ckl::eltwise_chain`, `ckl::copy`, and the `ckl::input(...)` / `ckl::output(...)` descriptors they take) receive the CB identity as a `tt::CBIndex` value — e.g. `ckl::input(dfb_input_id, ckl::WaitPolicy::PerTile, …)` at `logsigmoid_kernel.cpp:25`, `where_tss_kernel.cpp:41-42`, `lgamma_kernel.cpp:33`. That is the `uint32_t cb_id` row: ✓ OK, since `dfb::name`'s constexpr cast serves both runtime and template-parameter position — and these *are* template-parameter positions (`ckl::CopyTile<ckl::input(...), ckl::Dst::D0>`), which is the case that needs the constexpr cast specifically. The handles are declared `constexpr` in every kernel, so the cast is available. No donor-side change, no fork.

**Borrowed kernel files (file-path instantiation): none.** The factory instantiates only `reader_unary.cpp`, `writer_unary.cpp` and one of its own nine compute kernels — all owned by this op, none referenced by any other op in the tree.

**The inverse coupling is the live one, and it runs *out* of this directory.** Six kernels sitting in `device/kernels/dataflow/` are this op's *property* but not its *dependencies*, and other ops lean on them heavily. They are out of audit scope (no factory here references them) and out of port scope, but a porter working in this directory will trip over them:

| Unreferenced file (in this op's dir) | Other ops referencing it | `_metal2` fork beside it? |
|---|---|---|
| `writer_unary_interleaved_start_id.cpp` | 14 | **yes** — `writer_unary_interleaved_start_id_metal2.cpp` |
| `reader_unary_interleaved_start_id.cpp` | 10 | **yes** — `reader_unary_interleaved_start_id_metal2.cpp` |
| `reader_unary_sharded.cpp` | 7 | **yes** — `reader_unary_sharded_metal2.cpp` |
| `reader_unary_interleaved_wh_multicore.cpp` | 2 | no |
| `writer_unary_interleaved_start_id_wh.cpp` | 2 | no |
| `reader_unary_interleaved_col_multicore.cpp` | **0** — dead in-tree | no |

Consumers of the three forked ones, for planners: `copy/typecast` (both program factories), `data_movement/untilize` (4 factories), `data_movement/tilize` (2), `data_movement/transpose` (`transpose_wh_sharded`), `data_movement/sharded/sharded_to_interleaved`, `data_movement/copy` (`copy_default_tilized`). **This is a sunset list for those kernels, not authorization to convert anything in place** — and it is not this op's list at all, since this op uses none of them. `reader_unary_interleaved_col_multicore.cpp` having zero in-tree references is noted as a plain observation; it is nobody's port work.

### Relaxation candidates (noticed in the custom hash)

**FALLIBLE — candidates to verify; default strict. The ops team owns the real analysis.** These fall out of reading `compute_program_hash` and are recorded for the relaxation roadmap only; they never reach a porter.

- The TILE branch omits `padded_shape` entirely (`device/unary_device_operation.cpp:203-209`) while the `ROW_MAJOR` branch hashes it (`:198`). The in-code `TODO` at `:189-191` proposes narrowing the row-major key to the **last dimension only**, so height-only differences share an entry. That would widen the row-major regime from "shape barely varies within an entry" to something closer to the TILE regime, which changes what `dynamic_tensor_shape` is doing on rows 2 and 5 of the analysis doc's table. Worth pairing with any revisit of that document.
- `worker_grid` is hashed via `to_hash()` (`:25`) while also being derived from the input's shard grid and the device's sub-device set (`get_worker_grid`, `common/unary_utils.cpp:132-191`). It is therefore partly redundant with `memory_config`, but not wholly — `sub_core_grids` and the sub-device lookup both feed it. Not obviously a relaxation candidate; recorded because it is the one key component whose independence from the others is not evident by inspection.
- **The general lesson this op illustrates, worth carrying to the roadmap.** Unary needs a relaxation *and* has a cache-key gap for the same underlying reason: it hand-writes `compute_program_hash`. Every ported sibling checked (`typecast`, and per the analysis doc also `tilize`, `untilize`, `transpose`) defines **no** custom hash, so the framework default hashes the whole `TensorSpec` and they pin `Tile` and `Alignment` for free — at the cost of also pinning shape, which is exactly what unary's custom hash deliberately drops to get cache reuse. So "custom hash" and "needs a relaxation" are not independent signals on the readiness sheet; the second is largely a consequence of the first. Any op with a custom hash is a candidate to re-check against this rule (*an op's `compute_program_hash` must pin at least everything `tensorspecs_match_with_relaxation` requires to be exactly equal*) **before** its relaxation analysis is queued, since a key-vs-declaration mismatch is cheaper to find than to debug.

### TTNN factory analysis

Sheet-derived facts are unavailable this session (see the gate). The code-visible counterparts, with evidence, are tabulated in *Gate detail* above and summarised here for the re-run:

- **Op-owned tensors:** none. `create_descriptor` returns a bare `ProgramDescriptor` (`device/unary_device_operation.hpp:44`); there is no `create_workload_descriptor` and no `buffers` vector.
- **MeshWorkload need:** none — not a `WorkloadDescriptor` concept at all.
- **Pybind `create_descriptor`:** absent. `unary_nanobind.cpp` (2177 lines) binds no factory or device-op internals. Nothing for the port to delete, and no user-visible API change on that axis.
- **Custom hash:** present, and the port leaves it alone — `compute_program_hash` @ `device/unary_device_operation.cpp:171`, with the backdoor `operation_attributes_t::to_hash()` @ `:16` hashing `(op_chain, output_dtype, memory_config, fp32_dest_acc_en, preserve_fp32_precision, bfp8_pack_precise, sub_core_grids, worker_grid)`. **Note the coupling:** it does not gate here, but it is the same code the relaxation gate turns on — the relaxation fix will edit this method, and the port must still not otherwise touch it.
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** present @ `device/unary_program_factory.cpp:570` → target concept `CustomProgramSpecFactoryConcept`. The porter translates the method into one returning a `ProgramRunArgs`, rather than deleting it. It re-applies *everything* the miss path writes — per-core args via the shared `enumerate_core_rt_args` (`:596`), the accessor common args (`:640-653`), and the sharded CB addresses (`:657-665`) — because the TILE key omits shape. Non-trivial but self-contained.
- **Target concept:** `CustomProgramSpecFactoryConcept`, no op-owned tensors — *conditional on the readiness gate clearing.*

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

1. ⭐ **Live correctness bug — non-`32x32` tiles silently return wrong data. Family-wide, and it does *not* gate this port.** `create_descriptor` sizes both CBs with `tile_size(cb_data_format)` (`device/unary_program_factory.cpp:356`, `:358`, used as the page sizes at `:371-372` and the totals at `:421`, `:445`), and `tile_size(const DataFormat&)` (`tt_metal/api/tt-metalium/tt_backend_api_types.hpp:121`) takes **only** a `DataFormat` — so it hard-assumes `32x32`. Meanwhile `enumerate_core_rt_args` reads the **real** tile for the work split (`output.tensor_spec().tile().get_height()` / `get_width()`, `:167-168`, and `tile().get_tile_hw()` at `:386`), and so does the hash (`device/unary_device_operation.cpp:181`, `:185`). The two disagree and nothing in `eltwise/` guards the tile. The relaxation analysis reports this confirmed on silicon — `ttnn.relu` on a `16x32`-tile `bfloat16` tensor returns wrong data *in isolation*, with a clean `from_torch`/`to_torch` round-trip at the same tile, so it is an op bug rather than a caching artifact.

   **Why it does not gate, and why the relaxation fix should not try to absorb it.** `ttnn.typecast` fails identically, and `copy/typecast` is **already ported** — verified: all four of its factories expose `create_program_artifacts` (`typecast_program_factory.hpp:13,18`, `typecast_sharded_program_factory.hpp:13`, `typecast_rm_chunked_program_factory.hpp:13`). A ported sibling therefore already ships this exact defect, which settles that it is orthogonal to Metal 2.0 rather than something a port must clear. Fixing it inside the unary port would additionally violate the zero-functional-change invariant, since it moves behaviour the port's sentinels are meant to hold fixed. And pinning `Tile` in the hash does not mask it either — it just gives each tile its own equally-wrong entry, which is fine: the hash fix and this bug are independent, and the hash fix is still the right minimal change.

   Real support elsewhere is narrower than one might assume: `data_movement/tilize` is the only nearby op that engages deliberately — it sizes buffers from the *real* tile and `TT_FATAL`s that the tile **width** is 32 while permitting a smaller height. `untilize` has unary's own split personality (real tile for work partitioning, `tile_size(DataFormat)` for buffers). So "the ported ops support non-`32x32` tiles" is not the state of the tree.

   → **eltwise team**, as one issue spanning at least `unary` and `typecast`. The same gap is recorded against `eltwise/binary_ng`.
2. **Dead compile-time arg.** `compute_desc.compile_time_args.push_back(static_cast<uint32_t>(cb_data_format));` (`device/unary_program_factory.cpp:506`) is read by **none** of the nine compute kernels. `hardswish_kernel.cpp:14-15` reads CTAs 0 and 1 and `logit_kernel.cpp:16` reads CTA 0 — in both cases the ones set immediately above at `:499-505` — so the appended data format lands at index 2, 1, or 0 respectively and is never consumed. The data format reaches the kernels a different way entirely: baked into the `SFPU_OP_CHAIN_0` define strings (`fill_tile_int<DataFormat::{}>`, `mac_tile<DataFormat::{}>`, `where_tile<DataFormat::{}>`, … `common/unary_op_utils.cpp:173,672,678-679,803,826-837`), or via `INP_*` defines that `where_tss_kernel.cpp:14-23` tests with `#if defined`.
3. **Unreachable branch.** `get_shard_specs`'s `!input_sharded` fallthrough (`common/unary_utils.cpp:105-107`) is dead. Line `:66` returns `nullopt` unless `is_native_L1_sharding` is true, and that function returns `false` whenever the input is not sharded (`:37-39`) — so `input_sharded` is necessarily true by the time control reaches `:99`, and the `if` at `:99` always takes its first branch. A useful corollary for anyone reading the sharding logic: `src_sharded`, `dst_sharded` and `has_sharding` in the factory are all the *same* predicate (`is_native_L1_sharding` also requires the output sharded, `:34-36`), so `SRC_SHARDED` and `DST_SHARDED` can never disagree.
4. **Dead runtime args in the op's dominant regime.** On the interleaved-TILE path the factory writes 8 per-core slots per dataflow kernel and zero-fills slots 3-7 (`device/unary_program_factory.cpp:555-557`, re-zeroed on the hit path at `:629-630`), but the kernel code those slots feed is compiled out — `#if RM_INTERLEAVED` is `0` there (`reader_unary.cpp:29`, `writer_unary.cpp:31`). That is 5 dead slots per core per kernel in the regime that carries almost all of unary's traffic. Harmless, and the uniform count keeps `kInterleavedDataMovementArgs` (`:520`) simple; noted because the slot count is already config-dependent (3 sharded vs. 8 interleaved), so a third value for interleaved-TILE would not have been a new kind of complexity.
5. **`unpack_to_dest_mode` written for a CB that often does not exist.** `unpack_to_dest_mode[tmp0_cb_index] = UnpackToDestFp32;` (`device/unary_program_factory.cpp:403`) runs whenever `preserve_fp32_precision` is set, including for the eight op types where the `c_1` CB is never allocated (`needs_tmp0_cb` is LOGIT-only, `:75`). Harmless — the vector is `NUM_CIRCULAR_BUFFERS` wide (`:397-398`) — but it reads as if `c_1` were unconditional, and it is the kind of line that makes a CB-endpoint census look ambiguous when it is not.
6. **Truncating refresh loops in the override.** The accessor common-arg refresh is bounded by the *cached* buffer length on both kernels: `for (uint32_t i = 0; i < common_args.size() && i < reader_common.size(); ++i)` (`device/unary_program_factory.cpp:643`, and the writer's twin at `:651`). A fresh tensor needing *more* accessor words than the cached program allocated would be silently truncated rather than caught. The relaxation analysis flags this as **not established to be reachable** — under `dynamic_tensor_shape` the word count varies with rank for sharded slots, but the distribution spec squeezes rank before that point, which may well preclude it — and records it precisely because the `&&` guard would mask the mismatch either way. Recorded here with the code site so it is not re-derived.

## Questions for the user

1. **The readiness sheet needs an authorized session.** The TTNN factory-concept gate (`Is able to port?`) and the `Known op issues` cell could not be read here — the claude.ai Google Drive connector authorizes only in an interactive main session, and the download tool is not present in this one. Re-running *just that subject* in an authorized session would settle it; every other code-visible column is already cross-checked and tabulated above. Worth doing in the same pass as the relaxation fix, since `analyses/relaxations/eltwise_unary.md` §1 warns that `Known op issues` carries a **second, independent block** for this op that the relaxation document does not clear — so there may be a third blocker behind that cell.
2. **Which configurations would a port actually cover?** The relaxation analysis carries one explicit stop condition: row 5, *input tensor sharded but the op took the interleaved code path*, where the accessor is live over a sharded buffer and the declaration's geometry term does real work instead of pinning dead code. I confirmed that state is **plainly reachable** — `get_shard_specs` returns `nullopt` on three separate fallbacks (`common/unary_utils.cpp`): `is_native_L1_sharding` failing on DRAM (`:43-46`) or mismatched in/out grids (`:47-53`) or an uneven input (`:40-42`); an uneven *output* (`:66`); and a `ROW_MAJOR` shard whose element count is not tile-aligned (`:73-92`), which even emits a `log_warning` (`:85-88`). So "probably dead in practice" is not available as a defence here. If a port proceeds after the gates clear, the porter will need to know whether row-5 configurations are in scope.

## Recipe notes

1. **The relaxation subject does not say whether UNCONFIRMED is a GATE.** Its table maps the *sheet cell* to clears/GATE, and a separate paragraph says "If it fails, report the relaxation verdict as UNCONFIRMED" — but never states the routing consequence of UNCONFIRMED itself. I treated it as a GATE, on two grounds: the analysis doc's own contract says "**Stop**", and its §1 spells out the failure mode (a hard `TT_FATAL` on the second dispatch), which is port-blocking by any reading. The brief template's *"the only two values that reach a brief"* corroborates it, but only by implication. One sentence in the relaxation subject — "an UNCONFIRMED verdict is a GATE, routed to the ops team" — would remove the inference.
2. **There is no routing row for a gate whose authoritative input cannot be fetched.** The TTNN-factory-concept subject enumerates four outcomes (`yes`; an attributable `no`; an unattributed `no`; spreadsheet-broken) and all four presuppose a cell in hand. An *unfetchable* sheet is none of them: the four "spreadsheet is broken" triggers each require independent conflicting evidence, which I do not have, and an "unattributed `no`" requires a `no`, which I also do not have. I reported the gate as **UNVERIFIED — input unavailable** and routed it as a re-run rather than a defect. A fifth row would make that a rule rather than an improvisation. Related: `ttnn_op_porting_readiness.md` correctly says the connector cannot be authorized from inside a session, and `metal2_audit.md` correctly forbids delegating the fetch to a subagent — but neither says what the audit *outputs* when the fetch is simply impossible, which is the common case for a non-interactive run.
3. **The Red-outcome scoping rule's test and its justification diverge here.** The rule justifies skipping the seven informational subjects by *staleness*, then operationalises it as "op-code side → skip", with exemplars (Device 2.0 migration, offset split, page-size fix, PD migration) that all rewrite factory or kernel code. This op's blocker is cleared in `compute_program_hash` or `validate_on_program_cache_miss` — op-code by location, but it leaves every line the seven subjects describe untouched. I followed the justification and ran them, and flagged the one branch where that is wrong (if the ops team instead fixes `create_descriptor`'s page sizing). Rewording the test to name *the code the seven subjects describe* — the factory, the kernels, the donors — rather than "op-code side" would make this mechanical.
4. **A relaxation analysis doc can change under the auditor, and the recipe has no instruction for that.** `analyses/relaxations/eltwise_unary.md` was revised mid-audit: the verdict held, but the prescribed resolution, the §2 declaration (`match_page_size` dropped), and the tile bug's gating stance all moved. The recipe rightly calls these docs "authoritative but perishable" and tells me not to re-derive them — but perishability cuts both ways, and there is no guidance on re-reading before writing the report, nor on recording *which revision* a report was written against. A commit stamp is deliberately rejected by the doc itself (reasonably — it would fire on every unrelated commit), so the gap is real rather than an oversight. My mitigation was to re-run all five checks against the new text, verify its new load-bearing claims in code, and disclose the change at the top of this report. A one-line instruction — *"re-read the analysis doc immediately before writing the relaxation sections, and say in the report that you did"* — would make that reliable instead of lucky.

   **One concrete aggravating detail worth knowing:** `analyses/relaxations/eltwise_unary.md` is **untracked in git** (`git status` reports it `??`, unlike its committed siblings `eltwise_binary_ng.md` and `transformer_sdpa.md`). So it has no commit to pin *even in principle*, it does not travel with the docs branch, and a second auditor on another checkout may not have it at all — in which case the recipe's *"if the cell says `see analysis` and no such doc exists, STOP"* rule would fire spuriously on an analysis that does exist, just not in their tree. That is a distribution problem rather than a recipe problem, but it is the recipe's STOP rule that it lands on, so it belongs here: either commit the doc, or have the STOP rule distinguish *"no analysis was written"* from *"the analysis is not in your checkout."*
5. **Minor, and in the recipe's favour:** the `get_local_cb_interface` sanctioning bullet anticipated this op exactly — a DFB in scope, a DFB-side `get_tile_size()` available, and the free function still correct to keep. Without the "the list is the whole test, and it does not turn on what object is in scope" sentence I would have flagged `reader_unary.cpp:57` as an isolated holdover and RED'd the Device 2.0 gate. Worth keeping; the sub-point that `fifo_page_size` is not the same quantity as `get_tile_size()` might be worth adding to the breadcrumb, since the obvious port-stage substitution is not actually equivalent by definition.
