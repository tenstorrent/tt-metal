# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1`

One device operation, one program factory:

- **`MorehNllLossStep1DeviceOperation`** (`device/moreh_nll_loss_step1_device_operation.{hpp,cpp}`)
  - **`Factory`** — `create_descriptor` in `device/moreh_nll_loss_step1_program_factory.cpp`

The factory instantiates **three** kernels, all owned by this op, all referenced (no dead kernel files in the directory):

| Kernel | File | Selected when |
|---|---|---|
| reader (small) | `device/kernels/reader_moreh_nll_loss_step1.cpp` | `use_large_algorithm == false` |
| reader (large) | `device/kernels/reader_moreh_nll_loss_step1_large.cpp` | `use_large_algorithm == true` |
| writer | `device/kernels/writer_moreh_nll_loss_step1.cpp` | always |

There is **no compute kernel** — both readers compute the loss mask in-place in L1 with a scalar loop. This
is load-bearing for the CB-endpoint findings below.

The op's user-facing entry points (`moreh_nll_loss.cpp`, `moreh_nll_loss_nanobind.cpp`) live one level up in
`moreh_nll_loss/`, shared with `moreh_nll_loss_step2`. `step2` is a **separate device operation with its own
factory and its own kernels** — it shares no factory and no kernel file with `step1`, so the two are audited
separately and this report covers `step1` only.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources`

**Readiness-sheet values — provenance.** This session could not fetch the *"Operations analysis"* sheet
itself: the claude.ai Google Drive MCP connector needs an interactive OAuth authorization that cannot be
performed from inside a session, and a direct CSV export of the sheet returns `HTTP 401`. The three cells
below were **read from the sheet by the user and relayed to this audit**, not retrieved by the auditor:
`Is able to port?` = **`yes`**, `Is ready to port?` = **`yes`**, `TensorParameter relaxation` = **`none`**.
Recorded plainly because a downstream reader should know which facts in this report rest on the auditor's own
evidence (everything else) and which are relayed (these three). The code-side cross-check of the *primary*
columns was performed independently and agrees with them in every particular — see
[Gate detail](#gate-detail).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1` |
| **Overall** | **GREEN** — every gate cleared. `METAL2_PORT_BRIEF.md` issued alongside this report. |
| **DOps / Factories** | `MorehNllLossStep1DeviceOperation` → `Factory` (single factory) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — GREEN. All three op kernels and every donor function they call are structurally Device 2.0 (`Noc`, `DataflowBuffer`, `CoreLocalMem`, `TensorAccessor`, `UnicastEndpoint`). No holdovers. |
| *Prereqs* — Cross-op escapes | **Ok** — one donor header (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`), all consumed signatures ✓ excellent |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | Ok — no CTA read at a varying index |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — gate cleared (value relayed by the user; see the provenance note above). The sheet's companion `Is ready to port?` is likewise `yes`. Primary code-side cross-check performed independently and **clean on every column**. |
| *TTNN Readiness* — Concept (current) | `descriptor` — `Factory::create_descriptor` returns `tt::tt_metal::ProgramDescriptor` (`moreh_nll_loss_step1_device_operation.hpp:34`, `..._program_factory.cpp:17`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — concept is `descriptor`, not `WorkloadDescriptor` |
| *TTNN Readiness* — Custom hash | **No** — no `compute_program_hash`, no backdoor `attribute_values` / `to_hash` anywhere under `moreh_nll_loss/` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | **No** — method absent |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `moreh_nll_loss_nanobind.cpp` binds only the user-facing `ttnn::moreh_nll_loss`; no factory/device-op internals exposed |
| *TTNN Readiness* — Op-owned tensors | **No** — the concept is `descriptor`, which cannot carry them |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (`descriptor` + no op-owned tensors + no `override_runtime_arguments`) |
| *Port work* — Offset base pointer | **none** — GREEN. No address RTA folds a host-side offset; the factory passes `Buffer*` handles, never `->address() + <expr>`. |
| *Port work* — Tensor bindings (per binding) | **Case 1 ×3** — `target`, `weight` (optional), `output`; all fed to a `TensorAccessor` |
| *TTNN Readiness* — TensorParameter relaxation | **`none`** — clears (value relayed by the user). The only value that reaches a brief. |
| *Port work* — TensorAccessor 3rd arg | **none** — no accessor in the op passes a 3rd argument (all three are 2-arg `TensorAccessor(args, addr)`) |
| *Port work* — CB endpoints | **self-loop ×3 · legal 1:1 ×1 · dead-CB drop ×1 · conditional DFB ×1** — see [CB endpoints](#cb-endpoints-gate-free) |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution — a **self-loop**
(one toucher), a **1P+1C assignment** (two touchers), the **multi-binding advanced option** (a census that cannot
fit 1P+1C), or a **dead-CB drop** (zero endpoints). Recorded per `(CB, config)` below; two of this op's CBs have
**zero** endpoints in at least one config, and both are **must-fix** port work — a bindingless DFB is rejected by
the spec validator, so the port cannot build without acting on them.

## Result

**GREEN — every gate cleared. Brief issued.**

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ ·
TensorAccessor 3rd arg ✓ (N/A — no site)

The op is a clean, small port with no blockers on any track. Nothing routes to the ops team, the Device 2.0
team, the readiness-sheet owner, or the framework. No code-path scoping applies: the op has a **single**
factory, so there is no clean-vs-blocked subset question to answer.

The porter's actionable input is `METAL2_PORT_BRIEF.md`, written alongside this report. Two items in it are
must-fix rather than mechanical — both are zero-endpoint CBs that a Metal 2.0 spec validator will reject
outright, so the port cannot build without acting on them:

- **`c_24` — drop it** (dead in every config), while leaving the `cb_usage` arithmetic that reads its size
  byte-for-byte intact. That second half is the highest-risk line in this port.
- **`c_7` — make its DFB spec conditional** (dead under the large algorithm, live under the small one), not
  dropped.

Everything else is the standard mechanical work: three Case-1 tensor bindings, three self-loops, one legal
1:1 FIFO, no multi-binding flag anywhere, no relaxation, no 3rd-arg drops, no vararg blocks, and no shared
kernel to fork or coordinate.

**A note on how this GREEN was reached, since it bears on the audit trail.** The TTNN-factory-concept gate is
a *lookup* — the recipe has the auditor read `Is able to port?`, not re-derive it — and this session could not
reach the sheet (see the provenance note at the top). The gate was cleared on values the user read from the
sheet and relayed: `Is able to port?` = `yes`, `Is ready to port?` = `yes`,
`TensorParameter relaxation` = `none`. The auditor's independent contribution here is the *cross-check*, which
was performed in full against the code and agrees with those values on every primary column
([Gate detail](#gate-detail)). One cross-check remains unperformed and is called out there rather than
papered over: the **factory-set match**, which compares the sheet's row set against the code's factory set,
needs the sheet's rows and so could not be run. The code side is unambiguous — exactly one factory — and the
relayed values describe a single factory, which is consistent; but a reader tracking sheet staleness for this
op should know that particular check is outstanding.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** — the cell reads **`yes`**, and the sheet's
  companion `Is ready to port?` reads `yes` too. Both values were relayed by the user rather than fetched by
  this session (provenance note at the top). The recipe's lightweight cross-check was performed in full
  against the code, independently of those values, and **every primary column came back clean, mutually
  consistent, and in agreement with the relayed verdict**:

  | Column | Code-side finding | Evidence |
  |---|---|---|
  | `Concept` | `descriptor` | `Factory::create_descriptor` returns `ProgramDescriptor` — `moreh_nll_loss_step1_device_operation.hpp:34`, `moreh_nll_loss_step1_program_factory.cpp:17`. No `create()`+`override_runtime_arguments()` pair, no mesh-workload return, not already `MetalV2`. |
  | `Custom hash` | absent | `grep -rn "compute_program_hash\|attribute_values\|to_hash"` over `moreh_nll_loss/` → no hits. (No pybound `create_descriptor`, so the rename caveat doesn't apply here.) |
  | `Runtime-args update (get_dynamic_runtime_args)` | absent | No `get_dynamic_runtime_args` hook on `MorehNllLossStep1DeviceOperation` (`..._device_operation.hpp:41-47` is the full static-method set). |
  | `Override runtime args method?` | absent | No `override_runtime_arguments` anywhere under `moreh_nll_loss/`. The device-op declares only `validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`. |
  | `Pybind descriptor` | absent | `moreh_nll_loss_nanobind.cpp:24-37` binds `ttnn::moreh_nll_loss` only — no factory or device-op internals. |
  | `Smuggled pointer` | absent | The factory passes `Buffer*` through the **sanctioned** `emplace_runtime_args(core, initializer_list<variant<uint32_t, Buffer*>>)` overload (`..._program_factory.cpp:203`, `:217`), which auto-registers a `BufferBinding` — not an un-annotated pointer. See [Tensor bindings](#tensorparameter-analysis--tensor-bindings). |
  | `Secretly SPMD Workload?` | N/A | Concept is `descriptor`, not `WorkloadDescriptor`. |
  | `Op-owned tensors?` | No | Structurally impossible on `descriptor`; the factory returns a bare `ProgramDescriptor` with no `buffers` vector (`..._program_factory.cpp:225`). |
  | **Factory-set match** | **outstanding** | The one cross-check that *needs* the sheet's rows, since it compares the sheet's row set against the code's factory set — relayed cell values don't supply it. The code side is unambiguous: **exactly one** factory, `Factory`, via `using program_factory_t = std::variant<Factory>` (`..._device_operation.hpp:40`), and the relayed values describe a single factory, which is consistent. Not treated as a finding — no phantom or missing row was observed, because no row set was observed. |

  No primary-column conflict, no violated cross-column invariant, no missing row — **nothing
  "spreadsheet-broken" to report.** The sheet and the code agree.

  Per the recipe, a `yes` here **clears this prerequisite and nothing else** — the sheet reasons about
  TTNN-side considerations only, and it never sees Device 2.0, feature compatibility, offset base pointers,
  or the 3rd-argument question. This op is GREEN because *all five* gate-bearing subjects cleared
  independently, not because this cell reads `yes`.

- **TensorParameter relaxation:** **`none`** — clears. `none` is the only value that reaches a brief, and the
  port never applies a relaxation. Read, not re-derived: the recipe is explicit that judging whether the op
  "really" needs one is the ops team's analysis, not the auditor's. Consistent with the code, for what it is
  worth as corroboration only — there is no custom hash from which a relaxation candidate could be mined,
  `compute_output_specs` derives the output spec directly from `target_tensor.logical_shape()`
  (`..._device_operation.cpp:35-42`), and no accessor passes a dynamic page size (the Class-1
  `dynamic_tensor_shape` shape is absent).

- **Device 2.0 (every kernel used):** **GREEN.** All three of the op's kernels, and every donor function they
  call, are structurally Device 2.0. No violations table — there are no violations.

  | Kernel / donor | Device 2.0 idioms observed |
  |---|---|
  | `kernels/reader_moreh_nll_loss_step1.cpp` | `DataflowBuffer` objects (`:49-57`), `CoreLocalMem<T>` (`:61`, `:72-73`), `TensorAccessor` (`:34`, `:37`), method-form `get_read_ptr()` / `get_write_ptr()` on the DFB |
  | `kernels/reader_moreh_nll_loss_step1_large.cpp` | same set (`:51-54`, `:65-66`, `:82`, `:34`, `:39`) |
  | `kernels/writer_moreh_nll_loss_step1.cpp` | `Noc noc` + `noc.async_write(dfb, accessor, …)` (`:23`, `:30-31`), `DataflowBuffer` (`:24`), `TensorAccessor` (`:19`) |
  | donor `moreh_common.hpp` — `read_tile` (`:666`), `read_value` (`:695`), `read_line` (`:739`), `get_tilized_idx` (`:618`), `union Scalar` (`:39`) | `DataflowBuffer` by value, `Noc` object, `UnicastEndpoint`, template `AddrGen` deduced to `TensorAccessor<DSpec>` |

  **Two idioms I checked explicitly and am *not* flagging**, with the reasoning, so a reader can re-derive
  the call rather than take it on trust:

  1. **`get_tile_size(cb_id)`** — `writer_moreh_nll_loss_step1.cpp:25`,
     `reader_moreh_nll_loss_step1_large.cpp:37`, and inside the donors at `moreh_common.hpp:683`, `:709`,
     `:753` (there via `cb.get_id()`). This is **sanctioned** by the Device 2.0 Green bullet, and this op is
     precisely the case the recipe calls out as where the holdover cue misfires hardest: the call sites hold
     `DataflowBuffer` objects that *do* expose their own `get_tile_size()`, and sanctioned still means
     sanctioned. Confirmed against the Device 2.0 surface itself — the migration guide keeps
     `uint32_t tile_size = get_tile_size(cb_id);` verbatim inside its own **migrated** example
     (`docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md:630`).
     Moving these onto the object is a *port*-stage change (kernel-side whitelist rule 7), not a Device 2.0 one.
  2. **`my_x[noc.get_noc_id()]` / `my_y[...]`** in the donor `read_line` (`moreh_common.hpp:795-796`), used to
     build a `UnicastEndpoint` for the local-L1 loopback copy. Not on the Device 1.0 violation list, and not a
     CB-index-keyed free function. Decisively: the Device 2.0 `Noc` class implements its **own**
     `is_local_bank()` with exactly this expression — `virtual_x == my_x[noc_id_] && virtual_y == my_y[noc_id_]`
     (`tt_metal/hw/inc/api/dataflow/noc.h:150-152`) — so this is Device 2.0's own idiom, not a holdover from
     before it. The legacy shape the migration guide replaces is `get_noc_addr(my_x[...], ...)` +
     `noc_async_read(...)`; neither free function appears anywhere in this op or its donor path.

  Also confirmed absent across all three kernels and the donor path: `noc_async_read` / `noc_async_write` /
  `noc_semaphore_*` free functions, `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`,
  `get_local_cb_interface`, `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`,
  `InterleavedPow2AddrGen*`, and raw semaphore addresses. The includes are the current
  `api/dataflow/dataflow_buffer.h` — **not** the stale `api/dataflow/circular_buffer.h`.

- **Feature compatibility:** every Appendix A entry, in order. Every entry is UNSUPPORTED, so an absent
  feature is `N/A`, not a vacuous GREEN. **Subject verdict: GREEN — no gate fired.**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | No `experimental::GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `<tt-metalium/global_circular_buffer.hpp>` include, no factory parameter of that type. Checked the arcane descriptor-API signal specifically: **all five `CBDescriptor` literals** (`..._program_factory.cpp:75`, `:91`, `:104`, `:115`, `:129`) leave `.global_circular_buffer` unset (default `nullptr`, `program_descriptors.hpp:82`). Also checked the "remote CB" idiom — no `remote_cb_*` identifier, no `CircularBufferConfig::remote_index(`, no `remote_circular_buffer.h`. |
  | CBDescriptor `address_offset` (non-zero) | **N/A** | No `CBDescriptor` literal in the factory sets `.address_offset` at all (all five leave it at its default `0`). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. Nothing leaks in via the donor header. No runtime-team consultation is triggered. |
  | GlobalSemaphore | **N/A** | The op uses **no semaphores of any kind** — `grep -rni semaphore` over the op directory returns zero hits, so neither the `Global` variant nor the plain `SemaphoreSpec` path is in play. |

- **CB endpoints (GATE-free):** see the dedicated section below — five CBs, four dispositions, two of them
  must-fix. Not deferred: the Device 2.0 gate is GREEN, so the Device-2.0 idioms the recognition signals key
  on are intact and the census is trustworthy.

- **Offset base pointers:** **GREEN.** The op has three address-bearing runtime args, and **none** folds a
  host-side offset into its base. The factory does not compute an address at all — it hands the framework the
  `Buffer*` itself (`..._program_factory.cpp:183-186`, consumed at `:203-215` and `:217`), and
  `emplace_runtime_args_impl` emits the clean `buf->address()` with a registered binding
  (`tt_metal/impl/program/program_descriptors.cpp:251-252`). There is no `->address() + <expr>` expression
  anywhere in the op, so neither Type 1 (raw offset arg) nor Type 2 (accessor-fed offset arg) can fire.
  Type 3 (`address_offset`) is `N/A` per Appendix A above; Type 4 (`ttnn::narrow`) does not appear.

  Reconciled against the dated triage prior `analyses/2026-07-19_offset_base_pointers.md`: **no fold, op not
  in the tables** → clean, and the three RTAs pass to ordinary tensor-binding port work. Recorded explicitly
  rather than inferred from the doc's silence: the recognition scan was run on **every** address RTA, per the
  rule that "not in the tables" must never stand in for "scanned and clean."

  One nuance worth naming, since it is the kind of arithmetic that *looks* like a fold and is not: the donor
  `read_line` and `read_value` do compute byte offsets (`moreh_common.hpp:709-710`, `:775-779`, `:797`), but these
  are **kernel-side** `offset_bytes` / `.addr` fields on `noc.async_read`, applied *after* the accessor
  resolves a page — the accessor's own base stays the clean bound address. That is the supported shape, not a
  host-folded base.

- **TensorAccessor 3rd argument:** **N/A — no accessor in the op passes a 3rd argument.** Stating that
  rather than "every site is Class 2", because *no sites* and *sites found and classified redundant* are
  different findings. All three constructions are the 2-arg form:
  `TensorAccessor(target_args, target_addr)` (`reader_moreh_nll_loss_step1.cpp:34`,
  `reader_moreh_nll_loss_step1_large.cpp:34`), `TensorAccessor(weight_args, weight_addr)` (`:37` / `:39`), and
  `TensorAccessor(output_args, output_addr)` (`writer_moreh_nll_loss_step1.cpp:19`). The subject never fires,
  so no sharded-vs-interleaved / magnitude classification is needed. Consistent with the dated prior
  `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, which lists `moreh_fold` and `moreh_getitem` but
  not `moreh_nll_loss` — and, per the staleness rule, the finding rests on my own scan, not on that silence.

  *One related observation, deliberately kept out of the finding:* the donor `read_value` derives an element
  size as `get_tile_size(cb.get_id()) / 1024` (`moreh_common.hpp:709`), and the large reader computes the same
  quantity at `reader_moreh_nll_loss_step1_large.cpp:38`. That `/1024` is the "drops the block-float exponent"
  shape the 3rd-arg subject flags as wrong-magnitude — but it is **not** a `TensorAccessor` 3rd argument (it
  feeds a NoC byte offset, not an accessor stride), and it is **correct here** because the weight tensor is
  asserted `BFLOAT16` (`..._device_operation.cpp:26`), for which `tile_size / 1024 == 2` exactly. Recorded
  under [Misc anomalies](#misc-anomalies) as a latent fragility, not as a gate.

### CB endpoints — the census

Five `CBDescriptor`s, but the set allocated depends on config, so the census is per `(CB, config)`. The three
configs, and why there are only three:

| Config | Meaning | Reachable? |
|---|---|---|
| **A** — small, no weight | `use_large_algorithm == false`, `weight_has_value == false` | yes |
| **B** — small, with weight | `use_large_algorithm == false`, `weight_has_value == true` | yes |
| **C** — large, with weight | `use_large_algorithm == true`, `weight_has_value == true` | yes |
| *(large, no weight)* | — | **unreachable** — see below |

**Why "large without weight" cannot occur**, since two findings depend on it: `use_large_algorithm` is
`cb_usage >= available_L1` (`..._program_factory.cpp:70`), and with no weight `weight_num_tile == 0`
(`:64`), so `cb_usage` reduces to one target tile + one intermed tile + one output tile (`:67-68`) —
at most `4096 + 4096 + 4096 = 12 KiB` (int32 target tile, worst-case fp32 intermed and output). `available_L1`
is a full core's L1 less the allocator base (`:60-61`), on the order of 1 MiB. So `use_large_algorithm`
implies `weight_has_value`, and config C is the only large config.

The census (an endpoint is **any** kernel that touches the CB — FIFO produce, FIFO consume, or raw-pointer
access; role-free means a raw peek that locks neither FIFO role):

| CB | Config | Touchers on a node | Verdict | Port-time resolution |
|---|---|---|---|---|
| `c_0` target (`:75`) | A, B, C | **1** — the reader only. `read_tile` does `reserve_back`+`push_back` (`moreh_common.hpp:678`, `:690`); the reader body does `wait_front` / `get_read_ptr` / `pop_front` (`reader_...step1.cpp:70`, `:73`, `:97`). Both roles, one kernel. | single-ended | **self-loop** — bind the reader PRODUCER **and** CONSUMER |
| `c_1` weight (`:91`) | A | **not allocated** — `weight_cb_tiles == weight_num_tile == 0`, so the `if (weight_cb_tiles > 0)` at `:90` skips it. The reader's `c_1` references are all inside `#if defined(WEIGHT)`, which is undefined here (`:151-153`), so nothing dangles. | n/a | no DFB in this config |
| `c_1` weight | B | **1** — the reader only. `read_line` does `reserve_back`+`push_back` (`moreh_common.hpp:750`, `:806`); the reader body does `wait_front` + `get_read_ptr` (`reader_...step1.cpp:60-61`). | single-ended | **self-loop** |
| `c_1` weight | C | **1** — the large reader only. `read_value` does `reserve_back`+`push_back` (`moreh_common.hpp:706`, `:717`); the body does `wait_front` / `get_read_ptr` / `pop_front` (`reader_...step1_large.cpp:81-86`). | single-ended | **self-loop** |
| `c_7` weight scratch (`:129`) | A | **not allocated** — guarded by `if (weight_has_value)` at `:125`. | n/a | no DFB in this config |
| `c_7` weight scratch | B | **1, role-free** — the reader only, entirely sync-free: `read_line` uses it as an `async_read` destination and reads `cb_scratch.get_write_ptr()` (`moreh_common.hpp:782-784`, `:797`), with **no** FIFO ops on it at all. | single-ended / sync-free | **self-loop** (the label is cosmetic on Gen1 — no FIFO machinery is invoked) |
| `c_7` weight scratch | **C** | **0** — allocated, **never touched.** The `:125` guard is `weight_has_value` only, so C allocates it; but C instantiates `reader_..._large.cpp`, which **never names `c_7`** (its CB constants are `c_0`, `c_1`, `c_16` at `:23-26`) and calls only `read_tile` / `read_value`, neither of which takes a scratch buffer. | **dead in this config, live in B** | **conditional DFB** — tighten the guard to `weight_has_value && !use_large_algorithm`. **Do not drop it** — B needs it. |
| `c_16` output (`:115`) | A, B, C | **2** — one locked producer + one locked consumer. Reader FIFO-produces (`reserve_back` `:69` / `push_back` `:95`); writer FIFO-consumes (`wait_front` `:29` / `pop_front` `:32`). The reader's `get_write_ptr()` (`:72`) is a peek on its own PRODUCER binding, not a third endpoint. | **plain 1:1 — legal** | none — bind reader PRODUCER, writer CONSUMER. No flag. |
| `c_24` intermed (`:104`) | **A, B, C** | **0** — allocated unconditionally, **referenced by no kernel in any config.** | **dead CB** | **drop** the allocation |

**No CB in this op needs the multi-binding advanced option.** The maximum census on any node is 2
(`c_16`), and that pair is exactly one locked producer + one locked consumer. All three multi-toucher faces
were hunted and none applies:

- **(a) hidden second writer** — none. Every CB write in this op is either a FIFO `push_back` or the single
  reader's own raw peek. The face requires a *second* kernel co-filling via `get_write_ptr()`/`fifo_wr_ptr`
  gated by a semaphore pair, and **this op has no semaphores at all** (zero `grep -i semaphore` hits), so the
  coordination mechanism the face depends on cannot exist here.
- **(b) multiple readers** — none. No CB's read sites span two co-resident kernels; `c_16`'s only reader is
  the writer kernel.
- **(c) dual-instance work-split** — does not apply. The factory pushes **two distinct `kernel_source`
  values** into two `KernelDescriptor`s (reader and writer, `..._program_factory.cpp:168` vs `:176`), not the
  same source twice under a Reader/Writer config pair. There is no work-split-by-offset pairing.

**The two zero-endpoint findings are must-fix, and the confidence behind each differs — read them separately.**

**`c_24` — confirmed dead in every config → drop.** The recipe rightly says to distrust a `(0, 0)` result and
treat it as more likely a gap in my analysis than a real dead CB, so here is the positive confirmation rather
than an absence of hits:
- `grep -rn "c_24\|CBIndex" ` over the entire op directory returns the `c_24` allocation site
  (`..._program_factory.cpp:108`) and **nothing else** — no kernel names it.
- The indirect paths are ruled out individually. **No CTA carries a CB index**: the reader's compile-time args
  are exactly `{weight_has_value}` followed by two `TensorAccessorArgs` blocks (`:141-143`), and the writer's
  are one `TensorAccessorArgs` block (`:145-146`). **No index is computed, offset, or aliased** from another
  value in any kernel — all four CB constants in each kernel are literal `tt::CBIndex::c_N` initializers.
  **No config hides a reference** — all three configs were checked, and the only per-config variation in the
  kernels is the `#if defined(WEIGHT)` blocks, none of which mentions `c_24`.
- The structural reason is visible and corroborating: `c_24` is the conventional *intermediate* index for a
  **compute** kernel, and **this op instantiates no compute kernel** (`desc.kernels` receives only the reader
  and the writer, `:222-223`). It reads as a leftover from a version that had one.
- Framing for the porter, deliberately narrow: a dead CB has no behavior, so removing its allocation changes
  L1 footprint and nothing else — this is a rule the porter executes, **not** a sanctioned exception to
  "don't modify behavior." Metal 2.0 makes it mandatory besides: a DFB with neither a producer nor a consumer
  binding is rejected by the spec validator, so `c_24` cannot be carried across at all.
- **The one-line trap that comes with it, and it is a real hazard:** `c_24`'s size is *not* dead — it feeds
  `cb_usage` (`:67-68`), which decides `use_large_algorithm` (`:70`), which selects **which reader kernel
  file** is compiled (`:158-162`). A porter who drops the CB and "tidily" drops the
  `intermed_num_tile * intermed_tile_size` term from `cb_usage` would shift the small/large threshold and
  change which kernel runs for some shapes — a **functional** change, out of scope for the port.
  **Drop the allocation; leave the `cb_usage` arithmetic byte-for-byte intact,** dead term included. Also
  called out under [Heads-ups](#heads-ups-mirrors-the-brief), because this is the single easiest way to break
  this port.

**`c_7` — dead in config C, live in config B → conditional DFB, not a drop.** Naming both configs explicitly,
because "dead CB" plus a drop instruction is precisely how a porter deletes a buffer another config still
needs. This is the "expect new structure" case: the legacy factory allocates `c_7` on `weight_has_value`
alone and gates its *use* by which kernel file gets compiled, so there is **no existing host-side conditional
to translate** — the porter writes a new one. The `use_large_algorithm` value is already in scope at the
allocation site (computed at `:70`, allocation at `:125`), so the change is a one-line guard tightening.
Safe by inspection: `c_7` does **not** appear in the `cb_usage` sum (`:67-68`), so tightening its guard cannot
feed back into algorithm selection the way `c_24`'s size does.

*A nuance I checked and am deliberately **not** reporting as a finding:* within config B, the `c_7` touch sits
on the `else` branch of a **runtime** comparison in `read_line` (`moreh_common.hpp:770-802`) — the branch is
taken only where DRAM read alignment exceeds the valid-element byte count, so on some architectures the
scratch path may never execute at runtime. But `noc_read_size_bytes` derives from a runtime
`get_tile_size(cb.get_id())` lookup (`:753`), so the branch is **not** compile-time eliminated and the
reference is genuinely present in the compiled kernel. Per the rule to bias hard toward caution — a wrongly
dropped live CB silently mis-addresses, with nothing to catch it — `c_7` counts as **touched** in config B.
Recorded here so a future reader knows the question was asked and answered, not overlooked.

### TensorParameter analysis — tensor bindings

**Op-level roll-up: `⚠ port work`** — three bindings, all **Case 1**, none clean-via-borrowed-DFB, none
Case 2. All are **PORT WORK**; nothing here gates.

The causal-link gate was run first and does not fire for any binding: no CB in this op is a borrowed-memory
CB (no `set_globally_allocated_address` anywhere, and no `CBDescriptor` in the factory carries a `buffer`
field), so no binding is "clean via borrowed DFB" and none is deferred to the CB-endpoint subject on that
basis.

Every address here is a **clean base** — the [Offset base pointers](#gate-detail) gate cleared first, so no
Case-1/2 verdict below can be silently swallowing an offset.

| Binding | Delivery (host) | Consumption (kernel) | Case |
|---|---|---|---|
| `target` | `Buffer*` at reader RTA idx **0** (`..._program_factory.cpp:206`, from `target_buf` `:183`) | `get_arg_val<uint32_t>(0)` → `TensorAccessor(target_args, target_addr)`; all access via `read_tile(dfb, addrg_target, page)` (`reader_...step1.cpp:13`, `:34`, `:67`; large `:13`, `:34`, `:60`) | **Case 1** |
| `weight` (optional) | `Buffer*` **or `nullptr`** at reader RTA idx **1** (`:207`, from `weight_buf` `:186`) | `get_arg_val<uint32_t>(1)` → `TensorAccessor(weight_args, weight_addr)` under `#if defined(WEIGHT)`; access via `read_line` (small, `:58`) or `read_value` (large, `:79`) | **Case 1** |
| `output` | `Buffer*` at writer RTA idx **0** (`:217`, from `output_buf` `:187`) | `get_arg_val<uint32_t>(0)` → `TensorAccessor(output_args, output_addr)`; written via `noc.async_write(dfb_out, output_addrg, …)` (`writer_...step1.cpp:11`, `:19`, `:30`) | **Case 1** |

**All three are the `Buffer*`-binding form, and that shape is *not* the silent-wrong hazard.** The factory
pushes the `Buffer*` object itself — never `->address()` — through
`emplace_runtime_args(core, initializer_list<variant<uint32_t, Buffer*>>)`
(`tt_metal/api/tt-metalium/program_descriptors.hpp:194`). The framework auto-registers each as a
`BufferBinding` and patches it on cache hits
(`tt_metal/impl/program/program_descriptors.cpp:251-252`; `program_descriptors.hpp:110-118`), so this op is
**already correct on cache hits today** — it is on the framework's interim fix for the stale-pointer hole,
and the Metal 2.0 typed binding supersedes it. The factory's own comment says as much
(`..._program_factory.cpp:184-185`). Enumerated in full per the recipe, but **do not over-state the urgency**:
this is routine port work, not a correctness bug. Port work per binding is the standard Case-1 mechanical
swap — express as `TensorParameter` / `TensorBinding`, the kernel builds `TensorAccessor(tensor::name)`, and
the address RTA plus its `TensorAccessorArgs` plumbing both disappear.

**The optional `weight` binding is the one non-mechanical wrinkle**, and it is porter-facing rather than a
gate. Today the absence of a weight tensor is expressed in **three coordinated places**:
1. host: `weight_buf = nullptr` (`..._program_factory.cpp:186`), which the framework turns into a literal
   `0u` with **no** binding registered — `program_descriptors.cpp:245-250`, whose comment states the intent
   ("nullptr Buffer* represents an absent optional tensor … so the fast cache-hit path is not invalidated by
   optional inputs");
2. host: `TensorAccessorArgs(nullptr).append_to(...)` still appends a **placeholder args block** for the
   absent weight (`..._program_factory.cpp:143`), which is what keeps
   `TensorAccessorArgs<target_args.next_compile_time_args_offset()>()` at a fixed CTA offset in the kernel
   (`reader_...step1.cpp:32`) whether or not weight exists;
3. kernel: the `WEIGHT` define (`..._program_factory.cpp:151-153`) compiles the weight accessor and its DFB
   in or out entirely.

The port must keep all three consistent. Flagged under [Heads-ups](#heads-ups-mirrors-the-brief).

### RTA varargs

**None — no vararg block in either kernel.** Both readers read a **fixed run of nine** args through a running
`uint32_t i = 0; … get_arg_val<uint32_t>(i++)` at the top of the kernel
(`reader_...step1.cpp:12-21`, `reader_...step1_large.cpp:12-21`), and the writer reads three at constant
indices (`writer_...step1.cpp:11-13`). Per the recipe's non-signal rule, a sequential counter over a **fixed**
set is legacy positional plumbing, not a loop — it dissolves into named args, and every one of the twelve has
an obvious name from its own declaration (`target_addr`, `weight_addr`, `ignore_index`, `num_units_per_core`,
`start_id`, `C`, `weight_num_tile`, `element_size`, `target_element_size`; `output_addr`,
`num_units_per_core`, `start_id`). Neither recognition shape fires: **no** count-bounded loop advances an
`arg_index` inside its body, and **no** read's index is unpacked from another argument. Two of the nine are
dead and should not be carried across at all — see [Misc anomalies](#misc-anomalies).

**CTA varargs: none either.** All compile-time reads are at constexpr offsets — `get_compile_time_arg_val(0)`
plus the two/one `TensorAccessorArgs<N>` blocks (`..._program_factory.cpp:141-146`; kernels `:30-32` / `:17`).
No `get_compile_time_arg_val(i)` in a count-driven loop, so `KernelAdvancedOptions::compile_time_varargs` is
not needed. Named CTAs stay the default.

## Port-work summary  *(mirrors the brief)*

*(Mirrored into `METAL2_PORT_BRIEF.md`, which is the porter's copy.)*

- **Tensor bindings** (per binding): `target` **Case 1** · `weight` **Case 1** (optional — see the
  three-place consistency note above) · `output` **Case 1**. All three are the sanctioned `Buffer*` delivery
  form, correct-on-cache-hit today; each becomes a `TensorParameter` / `TensorBinding` with the kernel
  building `TensorAccessor(tensor::name)`, and the address RTA + `TensorAccessorArgs` plumbing disappears.
- **TensorParameter relaxation:** **`none`** — nothing to apply.
- **TensorAccessor 3rd arg:** **none** — no accessor passes one.
- **CB endpoints:**
  - **self-loop** `c_0` (all configs) · `c_1` (configs B, C) · `c_7` (config B — sync-free, label cosmetic)
  - **legal 1:1, no action** `c_16` (all configs) — reader PRODUCER, writer CONSUMER
  - **dead-CB drop** `c_24` @ `..._program_factory.cpp:104-112` — dead in **every** config; **leave the
    `cb_usage` term at `:67-68` untouched** (see the trap above). No dead CTA carries its index, so there is
    nothing further to remove.
  - **conditional DFB** `c_7` @ `..._program_factory.cpp:125-138` — **dead under config C** (large + weight),
    **live under config B** (small + weight). Tighten the guard from `weight_has_value` to
    `weight_has_value && !use_large_algorithm`; do **not** drop it.
  - **multi-binding advanced option:** not needed anywhere in this op.
- **Target concept:** `ProgramSpecFactoryConcept` — no `override_runtime_arguments` to translate, no
  op-owned tensors to carry.

## Heads-ups  *(mirrors the brief)*

- **`cb_usage` is load-bearing even though `c_24` is dead — the highest-risk line in this port.** Dropping
  the dead `c_24` allocation is required; dropping its `intermed_num_tile * intermed_tile_size` term from
  `cb_usage` (`..._program_factory.cpp:67-68`) would move the `use_large_algorithm` threshold (`:70`) and
  change **which reader kernel file** is compiled (`:158-162`) for some shapes. That is a functional change.
  Drop the allocation only; leave the arithmetic byte-for-byte.
- **`c_7` is dead in one config and live in another.** Named in both directions above precisely so it is not
  mistaken for a second dead-CB drop.
- **The optional `weight` tensor is expressed in three coordinated places** (host `nullptr` `Buffer*`, the
  placeholder `TensorAccessorArgs(nullptr)` that pins the kernel's CTA offset, and the `WEIGHT` define). The
  port must keep them consistent; the CTA-offset placeholder is the one easiest to lose.
- **CB endpoints (multi-binding shapes to watch):** **none.** No CB in this op needs the flag; max census on
  any node is 2, and that pair is a genuine 1P+1C FIFO. Recorded as a positive finding — the hidden-2nd-writer
  face in particular cannot apply here, since the op has no semaphores to coordinate one.
- **Cross-op / shared kernels:** the op **owns all three** of its kernel `.cpp` files and **no other op
  instantiates them** (verified by grepping the two reader paths and the writer path across `ttnn/` — the only
  hits are this factory's own `:158-165`). So there is **no `_metal2` fork to reuse, no fork to create, and no
  sunset list** — the shared-kernel coordination cost is zero. The single donor is a *header* (function-call
  escape), not a borrowed kernel file: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, all consumed
  signatures ✓ excellent (detail in [Team-only](#team-only)).
- **The donors take `DataflowBuffer` by value, which is the easy case.** `read_tile` / `read_value` /
  `read_line` all take `DataflowBuffer` parameters, and the kernels already pass named DFB locals
  (`dfb_target_obj`, `dfb_weight_obj`, `dfb_weight_scratch_obj`). Construct those locals from the tokens
  (`DataflowBuffer dfb_target_obj(dfb::target);`) and every call site is unchanged — **no donor-side change,
  no fork.**
- **These kernels are already part-modernized, so the port is a binding-layer change, not an idiom rewrite.**
  All three are on `DataflowBuffer` / `Noc` / `CoreLocalMem` / `TensorAccessor` with the current
  `api/dataflow/dataflow_buffer.h` include. Expect to touch bindings and arg names, not control flow.
- **Two Device 2.0 → Metal 2.0 breadcrumbs to confirm rather than swap blind:** the `get_tile_size(cb_id)`
  free-function calls at `writer_moreh_nll_loss_step1.cpp:25` and
  `reader_moreh_nll_loss_step1_large.cpp:37` are sanctioned *today* but are exactly what kernel-side whitelist
  rule 7 moves onto the DFB object at port time. The donor's internal ones (`moreh_common.hpp:683`, `:709`,
  `:753`) are in a **shared header** — changing them there reaches every moreh op, so leave them alone.
- **`constexpr` vs `const` on the CB handles:** all CB indices are declared `constexpr uint32_t`
  (`reader_...step1.cpp:23-26` etc.), which is the form that admits the token / constexpr-cast path. Worth a
  glance before assuming member-getter form.
- **No quasar copy of this op exists** (`ttnn/cpp/ttnn/operations/experimental/quasar/` has no `nll` or
  `moreh_nll` entry) — so there is no shortcut-port lookalike to be misled by. A negative pointer, recorded to
  save a wrong turn.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: `✓ clean`.** One donor file, one donor class, five consumed symbols, every shape ✓. No
⚠ / ✗ / ⭐ entries, so the per-call detail section is omitted per the report format.

Inventory of every `#include` in the op's kernels resolving outside the op directory:

| Op kernel | Include | Resolved donor | Class | Status |
|---|---|---|---|---|
| both readers | `ttnn/kernel/dataflow/moreh_common.hpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | **3** — `ttnn/cpp/ttnn/kernel/` (singular), the second shared-kernel pool; treat as shared-lib | ✓ |
| both readers, writer | `api/dataflow/dataflow_buffer.h` | `tt_metal/hw/inc/api/…` | 1 — LLK/HAL/firmware | ✓ no concern |
| both readers | `api/core_local_mem.h` | `tt_metal/hw/inc/api/…` | 1 | ✓ no concern |
| both readers, writer | `api/tensor/noc_traits.h` | `tt_metal/hw/inc/api/…` | 1 | ✓ no concern |
| writer | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h` | `tt_metal/hw/inc/api/…` | 1 | ✓ no concern |

**Per-call shape analysis — `moreh_common.hpp`** (only the symbols this op actually calls; the header is
large and the rest of it is not exercised here):

| Function | Signature shape | Status |
|---|---|---|
| `read_tile(DataflowBuffer cb, AddrGen addrgen, uint32_t noc_id, …)` (`:666`) | `DataflowBuffer` **by value**; `AddrGen` template deduced to `TensorAccessor<DSpec>` (Shape 1) | ✓ excellent — both rows are the ✓ case |
| `read_value(DataflowBuffer cb, AddrGen addrgen, …)` (`:695`) | same | ✓ excellent |
| `read_line(DataflowBuffer cb, DataflowBuffer cb_scratch, AddrGen addrgen, …)` (`:739`) | **two** `DataflowBuffer` by value + `TensorAccessor` | ✓ excellent |
| `get_tilized_idx(uint32_t h, uint32_t w)` (`:618`) | plain scalars, no resource handles | ✓ n/a |
| `union Scalar` (`:39`) | a POD union, no handles | ✓ n/a |

**This is the `DataflowBuffer` row, not the `CircularBuffer` row** — the two are opposite verdicts and sit
adjacent in the shape table, so: checked, and the signatures name `DataflowBuffer`. The donor has **already
migrated to DFB**, so it needs no donor-side change and no fork, and nothing here routes cross-team work.
No consumed signature takes a `uint32_t sem_id`, a sem address, a `TensorAccessorArgs<N>`, a CTA-offset NTTP,
an old-style addr-gen, or a `CircularBuffer` — the five shapes that would have been ⚠ / ✗ / ⭐.

**Borrowed kernel files (file-path kernel instantiation): none.** All three `kernel_source` paths
(`..._program_factory.cpp:158-165`) point inside this op's own `device/kernels/`. No `_metal2` fork exists
beside any of them, and none is needed — the files have exactly one consumer. Nothing to sunset, nothing to
coordinate.

### Relaxation candidates

**None to mine.** The candidate source is a custom `compute_program_hash` revealing which tensor properties
the op actually depends on, and this op has no custom hash — it uses the framework default over
`operation_attributes_t` and `tensor_args_t`. So there is nothing fallible to record for the relaxation
roadmap here.

### TTNN factory analysis

The sheet-derived facts, with `file:line` evidence. **Two cells could not be read** (`Is able to port?`,
`TensorParameter relaxation`) — see [Result](#result); everything below is my own code evidence, which is the
cross-check, not a substitute for those two cells.

- **Current concept:** `descriptor` — `Factory::create_descriptor` returns `tt::tt_metal::ProgramDescriptor`
  (`..._device_operation.hpp:34`; body at `..._program_factory.cpp:17-226`, returning `desc` at `:225`).
- **Factory set:** exactly one — `using program_factory_t = std::variant<Factory>`
  (`..._device_operation.hpp:40`). No per-factory divergence is possible in this op, so no
  per-DeviceOperation attribution section is needed.
- **Op-owned tensors:** none — structurally impossible on the `descriptor` concept, and the returned
  `ProgramDescriptor` carries no `buffers` vector.
- **MeshWorkload need:** none — not a `WorkloadDescriptor` op, so neither the genuine-multi-program nor the
  op-owned-tensor-artifact case arises.
- **Custom hash:** absent (gate-irrelevant either way — the port would leave one intact).
- **`get_dynamic_runtime_args`:** absent. Would have been a gate → TTNN.
- **`override_runtime_arguments`:** absent → target is the base `ProgramSpecFactoryConcept`, not
  `CustomProgramSpecFactoryConcept`. Nothing for the porter to translate.
- **Pybind `create_descriptor`:** absent. `moreh_nll_loss_nanobind.cpp:24-37` binds only the user-facing
  `ttnn::moreh_nll_loss` (via `ttnn::bind_function<"moreh_nll_loss">`), so the port removes **no** user-visible
  Python API and the port report needs no entry for it.
- **Other risky pybind:** none — no factory or device-op internals are exposed.
- **Target concept:** **`ProgramSpecFactoryConcept`** (`descriptor` + no op-owned tensors +
  `Override runtime args method? == no`). A clean mapping onto the common target; no recipe gap.

## Misc anomalies  *(team-only, non-gating)*

Latent issues noticed while auditing. These route to the **ops team**; the port does not act on them, and none
is a gate. (The dead `c_24` CB is *not* listed here — it is real port work, recorded above.)

1. **Two dead RTAs, passed on every dispatch to every core.** `element_size` (index 7) and
   `target_element_size` (index 8) are set by the factory (`..._program_factory.cpp:213-214`, with
   `element_size` computed at `:201`) and read by **both** readers
   (`reader_...step1.cpp:20-21`, `reader_...step1_large.cpp:20-21`) — then never used in either kernel. Nine
   RTAs are shipped where seven are live.
2. **A dead local in the large reader, computing a value the donor already computes.**
   `reader_moreh_nll_loss_step1_large.cpp:37-38` computes `weight_tile_bytes = get_tile_size(cb_weight)` and
   `weight_element_size = weight_tile_bytes / 1024`; neither is used. `read_value` derives the same quantity
   internally (`moreh_common.hpp:709`). Likely the residue of an inlining that was later factored into the
   donor.
3. **The `/1024` element-size derivation is correct here but fragile.** `moreh_common.hpp:709` (and the dead
   `:38` above) computes element size as `tile_size / 1024`, which is wrong for block-float formats — bf8
   gives 1024, not 1088. It is **safe in this op** only because `validate_inputs` hard-asserts the weight
   tensor is `BFLOAT16` (`..._device_operation.cpp:26`). If that assertion is ever relaxed to admit a
   block-float weight, `read_value`'s byte offsets mis-address silently. Worth a comment at the donor, or a
   `static_assert`-style guard.
4. **The `FP32_DEST_ACC_EN` define is dead.** The factory defines it on the **reader** when
   `fp32_dest_acc_en` (`..._program_factory.cpp:155-157`), but neither reader — nor any donor function they
   call — consumes it. Its only in-header consumer, `fp32_dest_acc_cast` (`moreh_common.hpp:23-31`), is never
   called from this op. It reads as another leftover from a version with a compute kernel.
5. **`compute_kernel_config` reaches the program only through a dead CB's size.** Follow the chain:
   `fp32_dest_acc_en` (`:48-49`) → `intermed_data_format` (`:54`) → `intermed_tile_size` (`:58`) → both the
   **dead `c_24`** allocation (`:104-112`) *and* `cb_usage` (`:67-68`) → `use_large_algorithm` (`:70`). So the
   op's only *functional* use of `compute_kernel_config` is that it perturbs the small/large algorithm
   threshold via a buffer nothing reads — for an op that has **no compute kernel** and so no fp32 dest
   accumulation to configure at all. Both halves are latent bugs of a kind: the threshold responds to a
   parameter that should be irrelevant to it, and a user changing `compute_kernel_config` can silently change
   which reader kernel runs. Worth an ops-team look independent of the port. *(This is the flip side of the
   porter warning above: the coupling is wrong, but it is also **live**, so the port must preserve it exactly
   and the fix belongs on the ops track.)*
6. **`reduction` is an unused-but-hashed attribute.** `operation_attributes_t::reduction`
   (`..._device_operation.hpp:16`) is set from the caller (`..._device_operation.cpp:65`) but read **nowhere**
   in `step1` — not in the factory, not in validation, not in output-spec computation. It nonetheless feeds
   the default `compute_program_hash` over `operation_attributes_t`, so two otherwise identical invocations
   differing only in `reduction` miss the program cache and compile a second, byte-identical program.
   (`reduction` is genuinely meaningful to `step2`; it appears to have been carried into `step1`'s attribute
   struct for symmetry.)
7. **A declared-but-unused CTA mirror.** Both readers declare
   `constexpr bool weight_has_value = get_compile_time_arg_val(0) == 1;`
   (`reader_...step1.cpp:30`, `reader_...step1_large.cpp:30`) and then never use it — the weight paths are
   selected by `#if defined(WEIGHT)` instead. The CTA **slot** is not dead (it is positional padding that
   keeps `TensorAccessorArgs<1>` at its fixed offset, `:31`), but the same fact is plumbed twice, by two
   different mechanisms, from one host boolean (`..._program_factory.cpp:141` and `:151-153`). Harmless today;
   a trap for anyone who removes one of the two.

## Questions for the user

1. **`c_24` — one last confirmation before the porter deletes it.** My analysis says it is dead in every
   config (evidence in [CB endpoints](#cb-endpoints--the-census)), and the recipe rightly asks for caution
   here since a wrongly-dropped live CB mis-addresses silently. The corroborating structural fact is that this
   op instantiates **no compute kernel**, which is where a `c_24` intermediate would normally be consumed. If
   anyone on the ops team knows of a planned or reverted compute kernel for `step1`, that would be worth
   saying before the drop lands. (A genuinely dead CB resurfaces loudly at the spec validator if I am wrong in
   the *safe* direction; the unsafe direction has no safety net.)
2. **`compute_kernel_config` influencing algorithm selection through a dead buffer** (Misc anomaly 5) — is
   that intended? It is not a port question (the port preserves it), but it looks like a real bug and I would
   rather surface it than leave it in a team-only list.

## Recipe notes

Friction with the audit recipe itself, not findings about the op.

1. **The recipe has no outcome for "the readiness sheet is unreachable," and it is a reachable state.**
   *(This audit ended GREEN — the note describes friction hit on the way there, and the paragraph at the end
   describes how it was resolved. The verdict is not in question.)* The
   *TTNN factory concept prerequisite* section enumerates five routings — `yes`, an attributed `no`, an
   unattributed `no`, spreadsheet-broken, and `MetalV2` — all of which presuppose that the cell was **read**.
   `ttnn_op_porting_readiness.md` states plainly that the connector "authorizes only in the main interactive
   session" and that "You **cannot** authorize it from inside a session," so a **non-interactive** session
   (a scheduled run, a CI-driven audit, an SDK invocation — this one) can never satisfy the fetch step. The
   recipe is emphatic that the four spreadsheet-broken triggers are "the *only*" ones and that a derived cell
   you cannot explain is not among them — correctly — but neither branch covers a cell you could not
   *retrieve*. I judged this to be materially different from an unattributed `no`: there, the sheet spoke and
   I could not explain it; here, the sheet did not speak. I therefore reported the gate as **indeterminate**,
   withheld the brief (an unread gate is not a cleared gate), and routed it as a data-availability question
   rather than alleging any defect. **Suggested addition:** a sixth routing — *"sheet unreachable → GATE
   indeterminate; report the code-side cross-check in full, withhold the brief, route to the launcher to
   re-run with Drive access"* — plus a line in the *Reference data* preamble noting that the fetch is
   impossible in a non-interactive session, so an auditor meets this before doing the work rather than at the
   end.

   **How it actually resolved, which is itself the useful data point:** the user read the three cells from the
   sheet and relayed them, and the audit went GREEN on that. So the practical unblock is cheap — but note what
   it does to the audit trail. The recipe's design has the auditor hold the sheet values and the code evidence
   *together*, which is what makes a disagreement detectable and a "spreadsheet-broken" verdict defensible.
   With the values relayed, the auditor can still cross-check the primary columns (I did), but two things
   change: the **factory-set match** cannot be run at all, since it needs the row *set* rather than named
   cells; and any disagreement would now be between the code and a human's reading of the sheet rather than
   between the code and the sheet. Neither cost bit here — the cross-check was clean and the op has a single
   factory. **A second suggested addition:** if relaying cells is to be the sanctioned fallback, the recipe
   should say so and say what it costs — *"a relayed value clears the gate; record the provenance, and note
   that the factory-set staleness check is not satisfiable this way."* Right now an auditor in this position
   is improvising both the routing and the disclosure, which is why this report carries a bespoke provenance
   note at the top.
2. **The Red-outcome scoping rule's exception resolved cleanly, and the reading paid off.** *(Recorded from
   the first pass, when the audit stood at RED-indeterminate; the call it describes is why nothing had to be
   re-derived when the cells arrived and the audit went GREEN.)* The rule
   asks which side a RED clears on: op-code side → skip the seven informational subjects; elsewhere → run
   them. A sheet-availability blocker clears with the op's code **untouched**, so I ran all seven. That felt
   unambiguous and the "Say in the report which case you judged it to be" instruction was exactly the right
   nudge — noting it only because the exception's examples are all *content* changes (an unattributed verdict,
   a feature landing, an unreleased capability) rather than *access* problems, so a future auditor may not
   immediately see that it applies.
3. **A small but real gap in the dead-CB guidance: a dead CB's *size* can be live.** The recipe's framing —
   "a dead CB has no behavior, so removing its allocation changes L1 footprint and nothing else" — is exactly
   right about the allocation, and it is the framing I passed to the porter. But in this op the dead CB's
   `tile_size` also feeds a host-side `cb_usage` sum that selects **which kernel file** is compiled
   (`..._program_factory.cpp:67-70`, `:158-162`). A porter who reads "this CB has no behavior" and tidies away
   the whole computation makes a silent functional change of exactly the kind the port forbids. The
   *Dead CB* subsection warns thoroughly about the mirror error (marking a live CB dead) but not about this
   one. **Suggested addition:** one line under *Dead CB* — *"check whether the dead CB's size or format feeds
   host-side control flow (an L1-budget sum, a threshold, a variant choice); if so, drop only the allocation
   and leave the arithmetic intact."*
4. **The sheet carries an `Is ready to port?` column that the readiness doc's legend doesn't list.** The user
   relayed it alongside `Is able to port?`, both `yes`. `ttnn_op_porting_readiness.md`'s *Reading the CSV*
   section enumerates the columns the audit reads and names `Is able to port?` as "the derived verdict, and
   the cell the audit reads," with no mention of a `ready` companion. The recipe is right that column lists go
   stale and that everything should be resolved against the header row you just fetched — so this is not a
   defect, just an observation from the field: there are apparently **two** derived verdict columns, and the
   recipe is silent on whether `Is ready to port?` is a gate conjunct, a scheduling signal, or informational.
   I treated `Is able to port?` as the gate (as the recipe directs) and recorded `Is ready to port?` as
   corroborating context. Worth one line in the legend saying which it is, since an auditor who meets a
   `yes`/`no` split between the two would have nothing to go on.
5. **The CB-endpoint census could say what to do with a runtime-conditional touch.** `c_7`'s only touch in
   config B sits on the `else` branch of a runtime comparison inside a donor
   (`moreh_common.hpp:770-802`) that may never execute on some architectures. The subject's guidance is
   framed around *config*-dependence (`(CB, config)`) and around compile-time indirection (a CTA-carried
   index, an `#ifdef`), not around a branch that is present in the compiled kernel but possibly never taken.
   I applied the "bias hard toward caution" principle and counted it as touched, which I am confident is
   right — a DFB binding for a branch that never runs costs nothing, while dropping a live one is the worst
   outcome available. But the principle is stated for the *dead-CB* decision specifically, and it took a
   deliberate step to carry it over to a within-config runtime branch. **Suggested addition:** a sentence in
   the census — *"a touch on a runtime branch counts as a touch; only compile-time-eliminated references do
   not"* — would make that a rule rather than an inference.
