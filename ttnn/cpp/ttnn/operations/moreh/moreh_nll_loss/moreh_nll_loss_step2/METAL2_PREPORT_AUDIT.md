# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2`

One device operation, one program factory — but the factory **dispatches on input rank** into three
separately-built programs, and that three-way split is the axis most findings below are scoped along:

- **`MorehNllLossStep2DeviceOperation`** (`device/moreh_nll_loss_step2_device_operation.{hpp,cpp}`)
  - **`Factory`** — `create_descriptor` in `device/moreh_nll_loss_step2_program_factory.cpp:701`, a thin
    rank dispatcher over three file-local `impl` builders:
    - `moreh_nll_loss_step2_impl_2d` (`:45`) — `rank == 2`
    - `moreh_nll_loss_step2_impl_3d` (`:258`) — `rank == 3`
    - `moreh_nll_loss_step2_impl_4d` (`:471`) — `rank >= 4` (the fallthrough)

**This is one factory, not three.** `program_factory_t` is `std::variant<Factory>`
(`..._device_operation.hpp:41`), so the readiness sheet carries one row and the TTNN gate is a single
verdict. The three `impl`s are internal code paths of that one factory. They matter here because each
builds its **own** kernel set and CB set, so the CB-endpoint census and several anomalies differ per path —
recorded per path throughout.

Seven kernels, all owned by this op, all referenced (no dead kernel files in the directory):

| Kernel | File | Used by |
|---|---|---|
| reader 2d | `device/kernels/reader_moreh_nll_loss_step2_2d.cpp` | `impl_2d` |
| reader 3d | `device/kernels/reader_moreh_nll_loss_step2_3d.cpp` | `impl_3d` |
| reader 4d | `device/kernels/reader_moreh_nll_loss_step2_4d.cpp` | `impl_4d` |
| writer 2d | `device/kernels/writer_moreh_nll_loss_step2_2d.cpp` | `impl_2d` |
| writer 3d | `device/kernels/writer_moreh_nll_loss_step2_3d.cpp` | `impl_3d` |
| writer 4d | `device/kernels/writer_moreh_nll_loss_step2_4d.cpp` | `impl_4d` |
| compute | `device/kernels/moreh_nll_loss_step2_kernel.cpp` | **all three** paths |

Unlike `step1`, this op **has a compute kernel**, and it is the *same source* for all three rank paths,
instantiated **twice** per path — once per `split_work_to_cores` core group, with the per-group unit count
baked as a CTA. That shape drives two of the most important findings below (the per-node census, and the
demoting-per-group-CTA trap).

The op's user-facing entry points (`moreh_nll_loss.cpp`, `moreh_nll_loss_nanobind.cpp`) live one level up in
`moreh_nll_loss/`, shared with `moreh_nll_loss_step1`. **`step1` is a separate device operation** with its own
factory and its own kernels; it shares no factory and no kernel file with `step2`, so the two are audited
separately. (`step1` was audited separately and cleared GREEN. Its `METAL2_PREPORT_AUDIT.md` / `METAL2_PORT_BRIEF.md`
were written on branch `anasuya/metal2_port_moreh_nll_loss` and are **not present in this checkout** — this
branch is `anasuya/metal2_port_moreh_nll_loss_step2`, so recover them from that branch if you need them. The
two ops share the *donor header* `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` and repeat one
CB-allocation defect in the same shape — noted where relevant.)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `28c1b0b4224 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources`

**Readiness-sheet values — provenance.** This session could not fetch the *"Operations analysis"* sheet
itself: the claude.ai Google Drive MCP connector needs an interactive OAuth authorization that cannot be
performed from inside a session, and a direct CSV export of the sheet returns `HTTP 401`. The two gate cells
below were **read from the sheet by the user and relayed to this audit**, not retrieved by the auditor:
`Is able to port?` = **`yes`**, `TensorParameter relaxation` = **`none`**. Recorded plainly because a
downstream reader should know which facts in this report rest on the auditor's own evidence (everything else)
and which are relayed (these two). The code-side cross-check of the *primary* columns was performed
independently and agrees with them in every particular — see [Gate detail](#gate-detail).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2` |
| **Overall** | **GREEN** — every gate cleared. `METAL2_PORT_BRIEF.md` issued alongside this report. |
| **DOps / Factories** | `MorehNllLossStep2DeviceOperation` → `Factory` (single factory; three internal rank paths) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — GREEN. All seven kernels and every donor function they call are structurally Device 2.0. The `get_dataformat(cb_id)` calls are **port-stage** work, not Device 2.0 holdovers — reasoning in [Gate detail](#gate-detail). |
| *Prereqs* — Cross-op escapes | **Ok** — two donor headers (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`), all consumed signatures ✓ excellent |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | Ok — no CTA read at a varying index |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — gate cleared (value relayed by the user; see the provenance note above). Primary code-side cross-check performed independently and **clean on every column**. |
| *TTNN Readiness* — Concept (current) | `descriptor` — `Factory::create_descriptor` returns `tt::tt_metal::ProgramDescriptor` (`..._device_operation.hpp:35`, `..._program_factory.cpp:701`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — concept is `descriptor` |
| *TTNN Readiness* — Custom hash | **No** — no `compute_program_hash`, no backdoor `attribute_values` / `to_hash` anywhere under `moreh_nll_loss/` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | **No** — method absent |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `moreh_nll_loss_nanobind.cpp` binds only the user-facing `ttnn::moreh_nll_loss` |
| *TTNN Readiness* — Op-owned tensors | **No** — the `descriptor` concept cannot carry them |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (`descriptor` + no op-owned tensors + no `override_runtime_arguments`) |
| *Port work* — Offset base pointer | **none** — GREEN. The factory contains **no** `->address()` expression at all; every tensor reaches a kernel as a `Buffer*` handle. |
| *Port work* — Tensor bindings (per binding) | **Case 1 ×5** — `input`, `target`, `weight` (optional), `divisor` (optional), `output`; all fed to a `TensorAccessor` |
| *TTNN Readiness* — TensorParameter relaxation | **`none`** — clears (value relayed by the user). The only value that reaches a brief. |
| *Port work* — TensorAccessor 3rd arg | **none** — all 15 accessor constructions in the op are the 2-arg form |
| *Port work* — CB endpoints | **dead-CB drop ×2 · self-loop ×6 · legal 1:1 ×4 · 2 must-fix DFB-declaration guards** — no multi-binding flag anywhere. See [CB endpoints](#cb-endpoints--the-census). |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution — a
**self-loop** (one toucher), a **1P+1C assignment** (two touchers), the **multi-binding advanced option**
(a census that cannot fit 1P+1C), or a **dead-CB drop** (zero endpoints). Recorded per `(CB, config)` below.
This op has **four** must-fix items in this subject, more than any other part of the port: two dead-CB drops
and two unconditional DFB declarations that will not compile in the configs where their CB is unallocated.

## Result

**GREEN — every gate cleared. Brief issued.**

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ ·
TensorAccessor 3rd arg ✓ (N/A — no site)

No gate blocks this port, and nothing routes to the ops team, the Device 2.0 team, the readiness-sheet owner,
or the framework as a blocker. No code-path scoping applies: the op has a **single** factory, so there is no
clean-vs-blocked subset question — and the three rank paths, being internal code paths of that one factory,
all port together.

The porter's actionable input is `METAL2_PORT_BRIEF.md`, written alongside this report.

**This is a substantially harder port than `step1`, and the difficulty is concentrated in one subject.** Not
in any gate — in the CB-endpoint work: eleven CBs across three rank paths and four `(WEIGHT, DIVISOR)`
combinations, carrying **four must-fix items**. Two of those are ordinary drops. The other two are
**compile-blockers that only fire when an optional tensor is absent** — configs a porter exercising the
default path will never build:

- **`c_3`** — the compute kernel constructs its DFB unconditionally (`compute:15`) for a CB that isn't
  allocated without a divisor, so `dfb::divisor` won't resolve.
- **`c_24`** — referenced unconditionally at `compute:18` and `compute:34`, so its spec must survive into the
  no-weight configs even though nothing FIFO-touches it there. It must **not** be dropped.

Plus one trap that fails the opposite way — quietly, at runtime rather than at compile time: four **dead CB
declarations** that must be *deleted*, not mechanically converted to `dfb::` handles. Converting the readers'
`cb_output` would fabricate a third endpoint on the output DFB and push a porter into a spurious
multi-binding flag.

Everything outside that subject is routine: five Case-1 tensor bindings, no relaxation, no 3rd-arg drops, no
vararg blocks, no multi-binding flag anywhere, and no shared kernel to fork or coordinate.

**A note on how this GREEN was reached, since it bears on the audit trail.** The TTNN-factory-concept gate is
a *lookup* — the recipe has the auditor read `Is able to port?`, not re-derive it — and this session could not
reach the sheet (see the provenance note at the top). The gate was cleared on values the user read from the
sheet and relayed: `Is able to port?` = `yes`, `TensorParameter relaxation` = `none`. The auditor's
independent contribution here is the *cross-check*, performed in full against the code and in agreement with
those values on every primary column ([Gate detail](#gate-detail)). One cross-check remains unperformed and is
called out there rather than papered over: the **factory-set match**, which compares the sheet's row set
against the code's factory set, needs the sheet's rows and so could not be run. The code side is unambiguous —
exactly one factory — but a reader tracking sheet staleness for this op should know that check is outstanding,
and specifically that **the sheet should carry one `step2` row, not three**; the rank paths are not factories.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** — the cell reads **`yes`**. The value was relayed
  by the user rather than fetched by this session (provenance note at the top). The recipe's lightweight
  cross-check was performed in full against the code, independently of that value, and **every primary column
  came back clean, mutually consistent, and in agreement with the relayed verdict**:

  | Column | Code-side finding | Evidence |
  |---|---|---|
  | `Concept` | `descriptor` | `Factory::create_descriptor` returns `ProgramDescriptor` — `..._device_operation.hpp:35`, `..._program_factory.cpp:701`. No `create()`+`override_runtime_arguments()` pair, no mesh-workload return, not already `MetalV2`. |
  | `Custom hash` | absent | `grep -rn "compute_program_hash\|attribute_values\|to_hash"` over `moreh_nll_loss/` → no hits. (No pybound `create_descriptor`, so the rename caveat doesn't apply.) |
  | `Runtime-args update (get_dynamic_runtime_args)` | absent | No hook on `MorehNllLossStep2DeviceOperation` (`..._device_operation.hpp:42-48` is the full static-method set). |
  | `Override runtime args method?` | absent | No `override_runtime_arguments` anywhere under `moreh_nll_loss/`. The device-op declares only `validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`. |
  | `Pybind descriptor` | absent | `moreh_nll_loss_nanobind.cpp:24-37` binds `ttnn::moreh_nll_loss` only — no factory or device-op internals. |
  | `Smuggled pointer` | absent | The factory passes `Buffer*` through the **sanctioned** `emplace_runtime_args(core, initializer_list<variant<uint32_t, Buffer*>>)` overload (`..._program_factory.cpp:210`, `:225`, `:421`, `:437`, `:651`, `:669`), which auto-registers a `BufferBinding`. See [Tensor bindings](#tensorparameter-analysis--tensor-bindings). |
  | `Secretly SPMD Workload?` | N/A | Concept is `descriptor`. |
  | `Op-owned tensors?` | No | Structurally impossible on `descriptor`; each `impl` returns a bare `ProgramDescriptor` with no `buffers` vector (`:255`, `:468`, `:698`). |
  | **Factory-set match** | **outstanding** | The one cross-check that *needs* the sheet's rows, since it compares the sheet's row set against the code's factory set — relayed cell values don't supply it. The code side is unambiguous: **exactly one** factory, `Factory`, via `using program_factory_t = std::variant<Factory>` (`..._device_operation.hpp:41`). Not treated as a finding — no phantom or missing row was observed, because no row set was observed. Worth confirming on a future fetch that the sheet carries **one** `step2` row and **has not been split into three per rank path**, which would be a phantom-row finding since the rank paths are not factories. |

  No primary-column conflict, no violated cross-column invariant, no missing row — **nothing
  "spreadsheet-broken" to report.** The sheet and the code agree.

  Per the recipe, a `yes` here **clears this prerequisite and nothing else** — the sheet reasons about
  TTNN-side considerations only, and never sees Device 2.0, feature compatibility, offset base pointers, or
  the 3rd-argument question. This op is GREEN because *all five* gate-bearing subjects cleared independently,
  not because this cell reads `yes`.

- **TensorParameter relaxation:** **`none`** — clears. `none` is the only value that reaches a brief, and the
  port never applies a relaxation. Read, not re-derived: the recipe is explicit that judging whether the op
  "really" needs one is the ops team's analysis, not the auditor's. Consistent with the code, as corroboration
  only — there is no custom hash from which a relaxation candidate could be mined, and no accessor passes a
  dynamic page size (the Class-1 `dynamic_tensor_shape` shape is absent). One nuance worth handing the ops
  team if they ever revisit this column: `compute_output_specs` has a **pass-through branch** — when
  `reduction == NONE` and an output tensor was supplied it returns `tensor_args.output_tensor->tensor_spec()`
  verbatim (`..._device_operation.cpp:61-63`) rather than a spec derived from the input. That is a spec the op
  does not itself construct, which is the kind of thing a strict-match analysis wants to know about. The cell
  says `none`, so this is not a port concern; recorded because it is cheap to note and expensive to rediscover.

- **Device 2.0 (every kernel used):** **GREEN.** All seven kernels, and every donor function they call, are
  structurally Device 2.0. No violations table — there are no violations.

  | Kernel / donor | Device 2.0 idioms observed |
  |---|---|
  | readers 2d / 3d / 4d | `DataflowBuffer` objects (2d `:56-67`, 3d `:59-72`, 4d `:60-73`), `CoreLocalMem<T>` (2d `:81-88`, 3d `:94-101`, 4d `:86-109`), `TensorAccessor` (2d `:47-54`, 3d `:50-57`, 4d `:51-58`), method-form `get_read_ptr()` / `get_write_ptr()` on the DFB |
  | writers 2d / 3d / 4d | `Noc noc` + `noc.async_write(dfb, accessor, …)` (2d `:23`/`:30`, 3d `:31`/`:46`, 4d `:25`/`:32`), `DataflowBuffer` (2d `:24`, 3d `:32`, 4d `:26`), `TensorAccessor` |
  | compute | `DataflowBuffer` objects for **every** FIFO operation (`:15-29`; `reserve_back` / `push_back` / `wait_front` / `pop_front` are all DFB methods, e.g. `:37`, `:46-51`, `:55`, `:65`) |
  | donor `dataflow/moreh_common.hpp` — `read_tile` (`:666`), `read_value` (`:695`), `read_line` (`:739`), `get_tilized_idx` (`:618`), `get_noc_offset` (`:635`), `fp32_dest_acc_cast` (`:23-31`) | `DataflowBuffer` by value, `Noc` object, `UnicastEndpoint`, template `AddrGen` deduced to `TensorAccessor<DSpec>` |
  | donor `compute/moreh_common.hpp` — `copy_tile_init_with_dt` (`:35`), `pack_tile_with_dt` (`:28`), `mul_tiles_init_with_dt` (`:100`) | all three take `DataflowBuffer` **by value** |

  **Three idiom families I checked explicitly and am *not* flagging.** Each is a judgment the porter would
  otherwise have to re-derive, so the reasoning is recorded in full:

  1. **`get_dataformat(cb_id)` — port work, not a Device 2.0 holdover.** All three readers compute three
     data-format locals this way: `reader_..._2d.cpp:36`, `:38`, `:40`; `..._3d.cpp:37`, `:39`, `:41`;
     `..._4d.cpp:40`, `:42`, `:44`. This is the one call in the op that genuinely looks like a holdover, and
     it is the RED/GREEN boundary for this op, so here is the full reasoning:
     - It is **not** on the audit's sanctioned list (`get_tile_size(cb_id)`, `get_local_cb_interface(cb_id)`),
       and "the list is the whole test" — so the sanctioned bullet does not cover it directly.
     - But it also **fails the holdover test**, which requires *both* that a wrapper-method replacement exist
       *and* that the wrapper object be **already in scope at the call site**. A replacement does exist
       (`DataflowBuffer::get_dataformat()`, `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:279`) — however at
       every one of these nine call sites **no DFB object is in scope yet**: the `DataflowBuffer`s are
       constructed 20+ lines later (2d `:62-67`, 3d `:67-72`, 4d `:64-73`).
     - **What settles it** is the port recipe, which the audit's Green bullet cross-references for exactly
       this family: kernel-side whitelist **rule 7** names `get_dataformat(cb_id)` *explicitly, by name,
       alongside `get_tile_size(cb_id)*, as compile-time tile/format metadata that the **port** moves onto the
       DFB object. The audit's own breadcrumb says the same in so many words — "a Metal 2.0 **port** moves
       these lookups onto the object (kernel-side whitelist rule 7) — a port-stage change that does not move
       the Device 2.0 boundary here."
     - Corroborating, and the reason the sanctioned list's silence reads as incompleteness rather than a
       deliberate distinction: `get_dataformat` sits in the **same three-line block** as the sanctioned
       `get_tile_size` inside the Device 2.0 `CircularBuffer` wrapper, with identical grounding — the wrapper
       implements its method *by calling* the free function
       (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:113-115`). That is precisely the grounding the audit
       gives for sanctioning `get_local_cb_interface`.
     - **Disposition:** not a gate. Recorded as port work — and, because all nine values are **unused**
       (see [Misc anomalies](#misc-anomalies) 1), the port should **delete** these lines rather than convert
       them to getters. Logged as a recipe note, since the sanctioned list's silence on `get_dataformat` is
       exactly the ambiguity that should be surfaced rather than silently resolved.
  2. **`get_tile_size(cb_id)`** — `writer_..._2d.cpp:25`, `writer_..._4d.cpp:27`, and inside the donor at
     `dataflow/moreh_common.hpp:683`, `:709`, `:753`. **Sanctioned** by the Green bullet, and this op is the
     case the recipe flags as where the cue misfires hardest: the call sites hold `DataflowBuffer` objects
     that expose their own `get_tile_size()`, and sanctioned still means sanctioned. Confirmed against the
     Device 2.0 surface itself, which keeps the free function verbatim in its own *migrated* example
     (`docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md:630`).
  3. **Raw `uint32_t` CB indices passed to compute LLK primitives** — `compute_kernel_hw_startup(cb_tmp_weight,
     cb_tmp_input, cb_output)` (`compute:34`), `copy_tile(cb_divisor, 0, dst0)` (`:41`),
     `mul_tiles(cb_tmp1, cb_tmp_weight, …)` (`:80`), `reconfig_data_format` (`:97`, `:129`),
     `mul_bcast_scalar_init` (`:99`, `:131`), `mul_tiles_bcast_scalar` (`:100`, `:132`). **Not holdovers**, for
     a structural reason: the holdover test requires that *a wrapper-method replacement exists*, and for the
     compute LLK surface **none does** — `grep -rl DataflowBuffer tt_metal/hw/inc/api/compute/` returns
     **nothing**, i.e. the entire compute API is `uint32_t cb_id`-based by design. The donor's own migrated
     `*_with_dt` helpers confirm the intent: they accept a `DataflowBuffer` and immediately call `.get_id()`
     to feed the raw LLK (`compute/moreh_common.hpp:28-40`). These indices become `dfb::name` handles at port
     time via the whitelist rule-2 implicit conversion, which is port work, not Device 2.0 work. Separately,
     the Device 2.0 gate is scoped to **data-movement** migration; the compute LLK is not that surface, and
     this kernel's data-movement surface — its FIFO operations — is fully on `DataflowBuffer`.

  Also confirmed absent across all seven kernels and both donor paths: `noc_async_read` / `noc_async_write` /
  `noc_semaphore_*` free functions, `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`,
  `get_local_cb_interface`, `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`,
  `InterleavedPow2AddrGen*`, and raw semaphore addresses. Includes are the current
  `api/dataflow/dataflow_buffer.h` — **not** the stale `api/dataflow/circular_buffer.h`.

- **Feature compatibility:** every Appendix A entry, in order. Every entry is UNSUPPORTED, so an absent
  feature is `N/A`, not a vacuous GREEN. **Subject verdict: GREEN — no gate fired.**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | No `experimental::GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `<tt-metalium/global_circular_buffer.hpp>` include, no factory parameter of that type. Checked the arcane descriptor-API signal specifically: **every** `CBDescriptor` in this op is built by the single local helper `push_cb` (`..._program_factory.cpp:22-41`), whose literal at `:32-40` leaves `.global_circular_buffer` unset (default `nullptr`, `program_descriptors.hpp:82`) — so one read of one helper clears all 33 CB allocation sites. Also checked the "remote CB" idiom — no `remote_cb_*` identifier, no `CircularBufferConfig::remote_index(`, no `remote_circular_buffer.h`. |
  | CBDescriptor `address_offset` (non-zero) | **N/A** | Same single-helper argument: `push_cb`'s `CBDescriptor` literal never sets `.address_offset` (default `0`). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. Nothing leaks in via either donor header. No runtime-team consultation is triggered. |
  | GlobalSemaphore | **N/A** | The op uses **no semaphores of any kind** — `grep -rni semaphore` over the op directory returns zero hits, so neither the `Global` variant nor the plain `SemaphoreSpec` path is in play. |

- **CB endpoints (GATE-free):** see the dedicated section below. Eleven CBs, four must-fix items, no
  multi-binding flag. Not deferred: the Device 2.0 gate is GREEN, so the idioms the recognition signals key
  on are intact and the census is trustworthy.

- **Offset base pointers:** **GREEN**, and unusually cleanly so: the factory contains **no `->address()`
  expression anywhere**. A `grep -nE 'address\(\)|->address'` over the whole op directory returns zero hits
  outside `TensorAccessorArgs(*tensor.buffer())` construction and `TT_FATAL(... .buffer() != nullptr ...)`
  validation. Every tensor reaches its kernel as a `Buffer*` handle
  (`..._program_factory.cpp:190-196`, `:401-407`, `:631-637`), and the framework emits the clean
  `buf->address()` with a registered binding (`tt_metal/impl/program/program_descriptors.cpp:251-252`).
  So there is no expression in which an offset *could* be folded — neither Type 1 (raw offset arg) nor Type 2
  (accessor-fed offset arg) can fire. Type 3 (`address_offset`) is `N/A` per Appendix A above; Type 4
  (`ttnn::narrow`) does not appear.

  Reconciled against the dated triage prior `analyses/2026-07-19_offset_base_pointers.md`: **no fold, op not
  in the tables** → clean. Recorded explicitly rather than inferred from the doc's silence — the recognition
  scan was run on every address-bearing RTA, per the rule that "not in the tables" must never stand in for
  "scanned and clean."

  One nuance worth naming, since it is the arithmetic that most resembles a fold and is not: the 3d path
  computes NoC byte offsets host-side-flavoured but **kernel-side** — `get_noc_offset(n, w, element_size,
  target_offset)` feeding `read_tile`'s `offset` parameter (`reader_..._3d.cpp:84-90`) and the writer's
  `.offset_bytes` (`writer_..._3d.cpp:44-51`). These are page-relative offsets applied *after* the accessor
  resolves a page; the accessor's base stays the clean bound address. That is the supported shape.

- **TensorAccessor 3rd argument:** **N/A — no accessor in the op passes a 3rd argument.** Stating that rather
  than "every site is Class 2", because *no sites* and *sites found and classified redundant* are different
  findings. All **15** constructions across the seven kernels are the 2-arg form `TensorAccessor(args, addr)`
  — enumerated: readers 2d `:47`, `:48`, `:49`, `:54`; 3d `:50`, `:51`, `:52`, `:57`; 4d `:51`, `:52`, `:53`,
  `:58`; writers 2d `:19`, 3d `:24`, 4d `:21`. The subject never fires, so no
  sharded-vs-interleaved / magnitude classification is needed. Consistent with the dated prior
  `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, which lists `moreh_fold` and `moreh_getitem` but
  not `moreh_nll_loss` — and, per the staleness rule, the finding rests on my own scan, not that silence.

  *One related observation, deliberately kept out of the finding:* the donor `read_value` derives an element
  size as `get_tile_size(cb.get_id()) / 1024` (`dataflow/moreh_common.hpp:709`) — the "drops the block-float
  exponent" shape the 3rd-arg subject flags as wrong-magnitude. It is **not** a `TensorAccessor` 3rd argument
  (it feeds a NoC byte offset, not an accessor stride), and it is **correct here** because `validate_inputs`
  hard-asserts input and weight are `BFLOAT16` (`..._device_operation.cpp:23`, `:39`), for which
  `tile_size / 1024 == 2` exactly. Recorded under [Misc anomalies](#misc-anomalies) as a latent fragility.

### CB endpoints — the census

**The config space.** Three rank paths × weight present/absent × divisor present/absent = **12 configs**,
and all 12 are reachable (nothing here is arithmetically excluded the way `step1`'s *large-without-weight*
was). The census is organised per CB with config qualifiers rather than by enumerating all 12.

**Per-node kernel population.** On any node: **1 reader + 1 writer + 1 compute instance**. The compute kernel
is instantiated **twice** per path — `compute_desc_1` over `core_group_1` and `compute_desc_2` over
`core_group_2` (`:159-188`, `:370-399`, `:600-629`) — but those core ranges are **disjoint**, so each node
sees exactly one. This matters for the census and is the single easiest thing to get wrong here:

> **Three `KernelSpec`s will bind some of these DFBs, yet the per-node census is still 2.** For `c_25`,
> `c_16`, `c_24`, `c_3`: the reader/writer spec plus **both** compute specs reference the DFB, so a porter
> counting *bindings* would see three and reach for the multi-binding flag. The census is **per CB, per
> node**, and because the two compute specs cover disjoint node sets each node has one producer and one
> consumer → **ordinary 1:1**. The framework validates the non-overlapping coverage; this is explicitly the
> legal single-role case, *not* `allow_instance_multi_binding`.

An endpoint is any kernel that touches the CB — FIFO produce, FIFO consume, or raw-pointer access. Role-free
means a raw peek that locks neither FIFO role.

| CB | Config | Touchers on a node | Verdict | Port-time resolution |
|---|---|---|---|---|
| `c_0` input | all 12 | **1** — reader only. `read_value` does `reserve_back`+`push_back` (`dataflow/moreh_common.hpp:706`, `:717`); reader body does `wait_front` / `get_read_ptr` / `pop_front` (2d `:100-104`, 3d `:112-117`, 4d `:125-130`). Compute never references `c_0`. | single-ended | **self-loop** — bind the reader PRODUCER **and** CONSUMER |
| `c_1` target | all 12 | **1** — reader only. `read_tile` does `reserve_back`+`push_back` (`:678`, `:690`); body does `wait_front` / `get_read_ptr` / `pop_front` (2d `:85-124`, 3d `:98-139`, 4d `:85-140`). | single-ended | **self-loop** |
| `c_2` weight | weight **absent** (6 configs) | **not allocated** — `push_cb` returns early on `num_tiles == 0` (`:28-30`), and the readers' `c_2` FIFO uses are inside `#if defined(WEIGHT)`. | n/a | no DFB in these configs |
| `c_2` weight | weight present, 2d / 3d | **1** — reader only (`read_value` reserve+push, then `wait_front` / `get_read_ptr` / `pop_front`: 2d `:112-117`, 3d `:122-128`). | single-ended | **self-loop** |
| `c_2` weight | weight present, 4d | **1** — reader only (`reserve_back` `:71`, `read_line` reserve+push, `wait_front` `:76`, `get_read_ptr` `:77`, `pop_front` `:143`). | single-ended | **self-loop** |
| `c_3` divisor | divisor present (6 configs) | **2** — one locked producer + one locked consumer. Reader FIFO-produces via `read_tile` (2d `:57`, 3d `:60`, 4d `:61`); compute FIFO-consumes (`wait_front` `:37`, `pop_front` `:46`). | **plain 1:1 — legal** | none — bind reader PRODUCER, compute CONSUMER. No flag. |
| `c_3` divisor | divisor **absent** (6 configs) | **not allocated** — but the compute kernel **unconditionally constructs a DFB for it** (`compute:15`). | ⚠ **must-fix** | **guard the declaration** — see *Must-fix 3* below |
| `c_7` weight scratch | **2d path, weight present** | **0** — allocated at `:102`, and the 2d reader **never names `c_7`** (its CB constants are `c_0`, `c_1`, `c_2`, `c_3`, `c_24`, `c_25`, `c_16` at `:25-33`); it reads weight via `read_value`, which takes no scratch. | **dead in every 2d config** | **drop** the allocation at `:102` |
| `c_7` weight scratch | **3d path, weight present** | **0** — allocated at `:313`; the 3d reader likewise never names `c_7` (`:26-34`) and uses `read_value`. | **dead in every 3d config** | **drop** the allocation at `:313` |
| `c_7` weight scratch | **4d path, weight present** | **1, role-free** — the 4d reader only, entirely sync-free: `read_line` uses it as an `async_read` destination and reads `cb_scratch.get_write_ptr()` (`dataflow/moreh_common.hpp:782-784`, `:797`), with **no** FIFO ops. Declared at `reader_..._4d.cpp:32`, `:73`; passed to `read_line` at `:74`. | single-ended / sync-free | **self-loop** (label cosmetic on Gen1) — allocation at `:543` is correct as written |
| `c_16` output | all 12 | **2** — compute FIFO-produces (`reserve_back` / `push_back`, e.g. `:104-108`, `:110-114`, `:137-141`, `:143-147`); writer FIFO-consumes (`wait_front` / `pop_front`: 2d `:29`/`:32`, 3d `:36`/`:54`, 4d `:31`/`:34`). Both compute instances produce, but on disjoint nodes. | **plain 1:1 — legal** | none — bind compute PRODUCER, writer CONSUMER. **See the *dead-declaration trap* below.** |
| `c_24` tmp_weight | weight present (6 configs) | **2** — reader FIFO-produces (`reserve_back` + `get_write_ptr` + `push_back`: 2d `:80-122`, 3d `:93-137`, 4d `:89-105`); compute FIFO-consumes (`wait_front` `:76`, `mul_tiles` `:80`, `pop_front` `:83`). | **plain 1:1 — legal** | none — bind reader PRODUCER, compute CONSUMER |
| `c_24` tmp_weight | weight **absent** (6 configs) | **1, role-free** — allocated unconditionally (`:90`, `:301`, `:531`); no FIFO op anywhere, but compute **references the index unconditionally** at `compute:34` (`compute_kernel_hw_startup(cb_tmp_weight, …)`) and constructs a DFB for it at `compute:18`. **Not dead.** | single-ended | **self-loop** — the spec must exist in every config; see *Must-fix 4* |
| `c_25` tmp_input | all 12 | **2** — reader FIFO-produces (`reserve_back` + `get_write_ptr` + `push_back`: 2d `:84-120`, 3d `:97-135`, 4d `:108-139`); compute FIFO-consumes (`wait_front` `:55`, `copy_tile` `:59`, `pop_front` `:65`). Unconditional on both sides. | **plain 1:1 — legal** | none — bind reader PRODUCER, compute CONSUMER |
| `c_26` tmp1 | weight **or** divisor present (9 configs) | **1** — compute only, both roles: `reserve_back`/`push_back` then `wait_front`/`pop_front` (`:68-84` under WEIGHT; `:118-135` under DIVISOR-without-WEIGHT). | single-ended | **self-loop** |
| `c_26` tmp1 | neither present (3 configs) | **1, role-free** — no FIFO op (the `#else`/`#else` arm at `:143-147` uses only `c_16`), but the DFB is constructed unconditionally at `compute:22`. | single-ended | **self-loop** — spec unconditional |
| `c_27` divisor_recip | divisor present (6 configs) | **1** — compute only, both roles: `reserve_back`/`push_back` (`:47-51`), `wait_front` (`:94` or `:124`), `pop_front` (`:153`). | single-ended | **self-loop** |
| `c_27` divisor_recip | divisor absent (6 configs) | **1, role-free** — DFB constructed unconditionally at `compute:24`; no FIFO op. | single-ended | **self-loop** — spec unconditional |
| `c_28` tmp3 | weight **and** divisor present (3 configs) | **1** — compute only, both roles: `reserve_back`/`push_back` (`:87-91`), `wait_front` (`:93`), `pop_front` (`:102`). | single-ended | **self-loop** |
| `c_28` tmp3 | otherwise (9 configs) | **1, role-free** — DFB constructed unconditionally at `compute:26`; no FIFO op. | single-ended | **self-loop** — spec unconditional |

**No CB in this op needs the multi-binding advanced option.** The maximum per-node census is 2, and every
such pair is one locked producer + one locked consumer. All three multi-toucher faces were hunted and none
applies:

- **(a) hidden second writer** — none. Every CB write is either a FIFO `push_back` or the writing kernel's
  own raw peek on its own binding. The face requires a semaphore-gated raw co-fill, and this op has **no
  semaphores at all** (zero `grep -i semaphore` hits), so the coordinating mechanism cannot exist here.
- **(b) multiple readers** — none. No CB's read sites span two co-resident kernels.
- **(c) dual-instance work-split** — **does not apply, and this is the important negative.** The two compute
  `KernelDescriptor`s are the same source over **disjoint** node sets (`core_group_1` / `core_group_2`), which
  is the *disjoint-node* case the recipe explicitly separates from the same-grid work-split: each node sees
  one instance, so each node's DFB is an ordinary 1:1 with no assignment question. The op has no
  Reader-config/Writer-config pair of one source over one grid.

#### The four must-fix items

Ordered by how quietly they fail. The last two are **compile-blockers in configs a porter testing only the
default path would never build**, which is why they lead the brief.

**Must-fix 1 — drop the dead `c_7` allocation in `impl_2d` (`:102`).** Zero endpoints in *every* 2d config.
A DFB with neither a producer nor a consumer binding is rejected by the spec validator, so it cannot be
carried across; and a dead CB has no behavior, so removing the allocation changes L1 footprint and nothing
else. Positively confirmed rather than inferred from absent grep hits: the 2d reader's CB constant list
(`:25-33`) has no `c_7`; no CTA carries a CB index (the reader's compile-time args are four
`TensorAccessorArgs` blocks and nothing else, `:107-110`); no index is computed, offset, or aliased in any
kernel; and the 2d path's kernel set is fixed at build time, so there is no unexamined config.

**Must-fix 2 — drop the dead `c_7` allocation in `impl_3d` (`:313`).** Identical reasoning; the 3d reader's
constant list is `:26-34`, compile-time args `:318-321`.

> **Why `c_7` is *not* a conditional DFB here, unlike in `step1`.** This is the same defect shape `step1`
> carries — a scratch CB allocated on `weight_has_value` alone while only *one* reader variant uses it — but
> the disposition differs, so don't copy `step1`'s answer across. In `step1` both variants live behind one
> host flag inside **one** program, so the CB had to become a *conditional* DFB. Here the three rank paths are
> **three separate `impl` functions building three separate `ProgramDescriptor`s**, so in `impl_2d` and
> `impl_3d` the allocation is dead unconditionally → a straight drop, with no conditional to write.
> `impl_4d`'s allocation (`:543`) is already correctly scoped and needs no change.

**Must-fix 3 — guard the compute kernel's unconditional `c_3` DFB declaration (`compute:15`).**
`DataflowBuffer dfb_divisor_obj(cb_divisor);` sits **outside** `#if defined(DIVISOR)`, while every *use* of it
is inside (`:36-52`, `:94`, `:124`, `:153`) and `c_3` is **not allocated at all** when divisor is absent
(`:88`, `:299`, `:529` with `divisor_has_value ? 1 : 0`). Today this is latent: the constructor merely records
the id on the MATH thread, and on the unpack/pack threads it eagerly reads
`get_local_cb_interface(3)` to feed a NoC-debug tracker
(`tt_metal/hw/inc/internal/tt-1xx/dataflow_buffer.inl:31-39`) — reading a stale interface entry for a CB the
program never created, harmless only because that tracker is normally compiled out. **At port time it is a
hard failure:** the DFB must be constructed from a `dfb::divisor` binding token, which does not exist when
there is no `DataflowBufferSpec` for `c_3` — so the no-divisor configs will not compile. Move the declaration
inside `#if defined(DIVISOR)`. (Note the readers already do this correctly — 2d `:56`, 3d `:59`, 4d `:60` are
all inside the guard. Only the compute kernel is inconsistent.)

**Must-fix 4 — keep the `c_24` spec unconditional and self-loop it when weight is absent.** `c_24` is
allocated unconditionally in all three paths (`:90`, `:301`, `:531`) but FIFO-touched only under `WEIGHT`.
It is **not** a dead CB and must **not** be dropped: the compute kernel references its index unconditionally
at `compute:34` (`compute_kernel_hw_startup(cb_tmp_weight, cb_tmp_input, cb_output)`) and constructs a DFB for
it at `compute:18` — so, exactly as in Must-fix 3, the `dfb::tmp_weight` token must exist in **every** config
or the no-weight configs will not compile. Keep the spec unconditional; bind it self-loop where no FIFO
producer/consumer pair exists.

> **The one judgment call in this census, flagged rather than buried.** Whether
> `compute_kernel_hw_startup(dfb::tmp_weight, …)` constitutes an endpoint *binding* — as opposed to a
> format/hardware-configuration reference that needs the token but not a binding — is a framework question the
> audit recipe does not settle, and I could not settle it from the recipe alone. It changes the *label*
> (1 role-free toucher → self-loop, versus 0 touchers → conditional DFB) but **not the instruction**: either
> way the spec must exist in every config, because the token must resolve for `compute:18` and `compute:34` to
> compile. The recommendation above is therefore robust to the ambiguity, and it is the cautious side — the
> recipe's warning is that a wrongly-dropped live CB mis-addresses silently while a wrongly-kept one is
> harmless. Raised as [Question 2](#questions-for-the-user).

#### The dead-declaration trap

Three kernels declare a CB constant they never use. Today these are inert dead locals (listed under
[Misc anomalies](#misc-anomalies)). **At port time they are a trap**, because the mechanical conversion of a
CB constant is to a `dfb::name` binding — and a binding is an endpoint:

| Dead declaration | If mechanically converted | Consequence |
|---|---|---|
| `cb_output` = `c_16` in **all three readers** (2d `:33`, 3d `:34`, 4d `:37`) | adds a reader binding on `c_16` | per-node census 2 → **3** ⇒ a porter would wrongly set the multi-binding flag on the output DFB |
| `cb_weight` = `c_2` in the **compute** kernel (`:13`) | adds a compute binding on `c_2` | per-node census 1 → **2** ⇒ turns a clean self-loop into a spurious 1P+1C |

**Delete these four declarations; do not convert them.** They carry no behavior, so deleting them is
zero-functional-change — the same reasoning as a dead-CB drop.

### TensorParameter analysis — tensor bindings

**Op-level roll-up: `⚠ port work`** — five bindings, all **Case 1**, none clean-via-borrowed-DFB, none
Case 2. All are **PORT WORK**; nothing here gates.

The causal-link gate was run first and does not fire for any binding: no CB in this op is a borrowed-memory
CB (no `set_globally_allocated_address` anywhere, and `push_cb`'s `CBDescriptor` literal carries no `buffer`
field), so no binding is "clean via borrowed DFB".

Every address here is a **clean base** — the [Offset base pointers](#gate-detail) gate cleared first, so no
Case-1/2 verdict below can be silently swallowing an offset.

**Scope note:** the **compute kernel is out of scope** for this subject, per the recipe — it only consumes
from and produces to circular buffers and never touches tensor memory (it constructs no `TensorAccessor` and
reads no address RTA; it reads **no runtime args at all**). All tensor access happens in the readers and
writers. This also means the op has **no Case-2-in-a-compute-kernel** problem, which is the one shape that
would have blocked the port here.

| Binding | Delivery (host) | Consumption (kernel) | Case |
|---|---|---|---|
| `input` | `Buffer*` at reader RTA idx **0** (`:213`, `:424`, `:654`) | `get_arg_val<uint32_t>(0)` → `TensorAccessor(input_args, input_addr)`; access via `read_value(dfb_input_obj, addrg_input, …)` (2d `:47`/`:98`, 3d `:50`/`:110`, 4d `:51`/`:123`) | **Case 1** |
| `target` | `Buffer*` at reader RTA idx **1** (`:214`, `:425`, `:655`) | `get_arg_val<uint32_t>(1)` → `TensorAccessor(target_args, target_addr)`; access via `read_tile` (2d `:48`/`:77`, 3d `:51`/`:85`, 4d `:52`/`:83`) | **Case 1** |
| `weight` (optional) | `Buffer*` **or `nullptr`** at reader RTA idx **2** (`:215`, `:426`, `:656`) | `get_arg_val<uint32_t>(2)` → `TensorAccessor(weight_args, weight_addr)`; access via `read_value` (2d `:112`, 3d `:122`) or `read_line` (4d `:74`) under `#if defined(WEIGHT)` | **Case 1** |
| `divisor` (optional) | `Buffer*` **or `nullptr`** at reader RTA idx **3** (`:216`, `:427`, `:657`) | `get_arg_val<uint32_t>(3)` → `TensorAccessor(divisor_args, divisor_addr)` **inside** `#if defined(DIVISOR)`; access via `read_tile` (2d `:57`, 3d `:60`, 4d `:61`) | **Case 1** |
| `output` | `Buffer*` at writer RTA idx **0** (`:228`, `:440`, `:672`) | `get_arg_val<uint32_t>(0)` → `TensorAccessor(output_args, output_addr)`; written via `noc.async_write(dfb_out, output_addrg, …)` (2d `:19`/`:30`, 3d `:24`/`:46`, 4d `:21`/`:32`) | **Case 1** |

**All five are the `Buffer*`-binding form, and that shape is *not* the silent-wrong hazard.** The factory
pushes the `Buffer*` object itself — never `->address()` — through
`emplace_runtime_args(core, initializer_list<variant<uint32_t, Buffer*>>)`
(`tt_metal/api/tt-metalium/program_descriptors.hpp:194`). The framework auto-registers each as a
`BufferBinding` and patches it on cache hits (`program_descriptors.cpp:251-252`;
`program_descriptors.hpp:110-118`), so this op is **already correct on cache hits today** — it is on the
framework's interim fix, and the Metal 2.0 typed binding supersedes it. The factory's own comments say as
much (`:192-193`, `:403-404`, `:633-634`). Enumerated in full per the recipe, but **do not over-state the
urgency**: routine port work, not a correctness bug.

**The two optional bindings differ from each other, and the asymmetry is porter-relevant.** Both `weight` and
`divisor` are `std::optional<Tensor>` delivered as a possibly-null `Buffer*`, but the kernels treat them
differently:

- **`divisor`** — its `TensorAccessor` is constructed **inside** `#if defined(DIVISOR)` (2d `:54`, 3d `:57`,
  4d `:58`). Consistent; only the compute kernel's DFB declaration breaks the pattern (*Must-fix 3*).
- **`weight`** — its `TensorAccessor` is constructed **unconditionally** (2d `:49`, 3d `:52`, 4d `:53`),
  *outside* the `WEIGHT` guard, even though every *use* is inside it. When weight is absent this builds an
  accessor over a null-buffer `TensorAccessorArgs` and simply never dereferences it. Harmless today; at port
  time a `tensor::weight` binding must exist for that line to compile in the no-weight configs, so either the
  binding is declared optional-but-present or the construction moves inside the guard. Flagged in
  [Heads-ups](#heads-ups-mirrors-the-brief).

Common to both, and to `step1`: the absent-optional plumbing is three coordinated pieces —
host `nullptr` `Buffer*` → framework emits `0u` with **no** binding registered
(`program_descriptors.cpp:245-250`); host `TensorAccessorArgs(nullptr).append_to(...)` still appends a
**placeholder args block** (`:109-110`, `:320-321`, `:550-551`), which is what keeps the *following*
accessors' CTA offsets fixed via the `next_compile_time_args_offset()` chain (2d `:42-45`); and the
`WEIGHT` / `DIVISOR` defines (`:119-126`, `:330-337`, `:560-567`). **The placeholder is load-bearing here in a
way it was not in `step1`**: `step1` had one optional tensor last in its chain, whereas here `weight` and
`divisor` sit at positions 3 and 4 of a four-accessor chain, so dropping a placeholder shifts every
downstream offset.

### RTA varargs

**None — no vararg block in any kernel.** Every argument is nameable, and the counts are fixed per kernel:

| Kernel | RTA reads | Form |
|---|---|---|
| reader 2d | **10**, `i++` run at `:13-23` | fixed sequential counter |
| reader 3d | **11**, `i++` run at `:13-24` | fixed sequential counter |
| reader 4d | **13**, `i++` run at `:12-25` | fixed sequential counter |
| writer 2d | **3**, constant indices `:11-13` | distinct constant indices |
| writer 3d | **5**, `i++` run at `:13-18` | fixed sequential counter |
| writer 4d | **3**, `i++` run at `:12-15` | fixed sequential counter |
| compute | **0** — reads no runtime args at all | — |

Per the recipe's non-signal rule, a sequential counter over a **fixed** set is legacy positional plumbing,
not a loop, and dissolves into named args. Neither recognition shape fires: **no** count-bounded loop advances
an `arg_index` inside its body, and **no** read's index is unpacked from another argument. Names come straight
off the declarations. Note that **four** of these args are dead and should not be carried across — see
[Misc anomalies](#misc-anomalies) 2-4.

**CTA varargs: none either.** All compile-time reads are at constexpr offsets — the readers' four
`TensorAccessorArgs<N>` blocks, the writers' one, and the compute kernel's single
`get_compile_time_arg_val(0)` (`compute:11`). No `get_compile_time_arg_val(i)` in a count-driven loop, so
`KernelAdvancedOptions::compile_time_varargs` is not needed. Named CTAs stay the default.

## Port-work summary  *(mirrors the brief)*

*(Mirrored into `METAL2_PORT_BRIEF.md`, which is the porter's copy.)*

- **Tensor bindings** (per binding): `input` **Case 1** · `target` **Case 1** · `weight` **Case 1**
  (optional; accessor built unconditionally) · `divisor` **Case 1** (optional; accessor built under the
  guard) · `output` **Case 1**. All five are the sanctioned `Buffer*` delivery form, correct-on-cache-hit
  today; each becomes a `TensorParameter` / `TensorBinding` with the kernel building
  `TensorAccessor(tensor::name)`, and the address RTA + `TensorAccessorArgs` plumbing disappears.
- **TensorParameter relaxation:** **`none`** — nothing to apply.
- **TensorAccessor 3rd arg:** **none** — no accessor passes one.
- **CB endpoints** — four must-fix items, then the routine dispositions:
  - **dead-CB drop** `c_7` @ `..._program_factory.cpp:102` (`impl_2d`) and `:313` (`impl_3d`) — dead in every
    config of those paths. No dead CTA carries the index, so nothing further to remove.
  - **guard the DFB declaration** `c_3` @ `moreh_nll_loss_step2_kernel.cpp:15` — move inside
    `#if defined(DIVISOR)`; otherwise the no-divisor configs cannot compile.
  - **keep the spec unconditional + self-loop** `c_24` under no-`WEIGHT` — referenced unconditionally at
    `compute:18` and `compute:34`; must **not** be dropped.
  - **self-loop** `c_0` (all) · `c_1` (all) · `c_2` (weight present) · `c_7` (4d + weight) · `c_26` (all) ·
    `c_27` (all) · `c_28` (all)
  - **legal 1:1, no action** `c_3` (divisor present) · `c_16` (all) · `c_24` (weight present) · `c_25` (all)
  - **multi-binding advanced option:** not needed anywhere in this op.
  - **delete, don't convert** the four dead CB declarations — readers' `cb_output` (2d `:33`, 3d `:34`,
    4d `:37`) and compute's `cb_weight` (`:13`) — per the [dead-declaration trap](#the-dead-declaration-trap).
- **`get_dataformat` lines:** delete the nine dead data-format locals (readers 2d `:36`/`:38`/`:40`,
  3d `:37`/`:39`/`:41`, 4d `:40`/`:42`/`:44`) rather than converting them to DFB getters — the values are
  unused. (If any were live, whitelist rule 7 would apply: `const` declarations take the member getter;
  `constexpr` ones keep the free-function form with the token. All nine here are `const`.)
- **Target concept:** `ProgramSpecFactoryConcept` — no `override_runtime_arguments` to translate, no
  op-owned tensors to carry.

## Heads-ups  *(mirrors the brief)*

- **Do not collapse the two compute `KernelDescriptor`s into one spec with a runtime arg.** This op is a
  textbook instance of the **demoting-per-group-CTA anti-pattern**: `split_work_to_cores` plus two same-source
  compute descriptors carrying different per-group CTAs (`units_per_core_group_1` / `_2` at `:163`/`:179`,
  `:374`/`:390`, `:604`/`:620`). The correct port is **two `KernelSpec`s of the same source in two
  `WorkUnitSpec`s**, one per core group — Metal 2.0 supports that, and the demotion costs compile-time loop
  unrolling on `per_core_tile_cnt` (`compute:11`, the loop bound at `:54`), a measurable kernel-perf
  regression the port is not entitled to make.
  **This op sets the trap unusually well:** the factory *already* populates a per-core compute RTA carrying
  exactly that value (`:235-243`, `:448-456`, `:678-686`) — and the compute kernel **never reads it**
  (zero `get_arg_val`). A porter who notices the RTA and "simplifies" toward it lands precisely on the
  anti-pattern. The dead RTA should be **deleted**, not adopted.
- **Two compile-blockers live in absent-optional configs.** Must-fix 3 (`c_3` DFB declaration) and Must-fix 4
  (`c_24` spec) both fail only when an optional tensor is **absent** — configs a porter exercising the default
  path will not build. Build all four `(weight, divisor)` combinations, on all three rank paths, before
  calling the port done.
- **Three `KernelSpec`s binding one DFB is legal here.** For `c_25`, `c_16`, `c_24`, `c_3` the reader/writer
  spec plus **both** compute specs reference the DFB. The per-node census is still 2, because the compute
  specs cover disjoint core groups. **Do not** reach for `allow_instance_multi_binding` on that count.
- **Four dead CB declarations must be deleted, not converted** — see the
  [dead-declaration trap](#the-dead-declaration-trap). Converting the readers' `cb_output` would fabricate a
  third endpoint on the output DFB and push a porter toward a spurious multi-binding flag.
- **`c_7` repeats `step1`'s defect shape but takes a different resolution.** Same mis-scoped scratch-CB guard;
  here it is a straight drop in two of three rank paths rather than a conditional DFB, because the paths are
  separate programs. Don't carry `step1`'s answer across.
- **The optional-accessor placeholder chain is load-bearing.** `TensorAccessorArgs(nullptr)` for an absent
  `weight` / `divisor` (`:109-110`, `:320-321`, `:550-551`) keeps the downstream accessors' CTA offsets fixed
  through the `next_compile_time_args_offset()` chain (2d `:42-45`). Unlike `step1`, the optionals here are
  **not** last in the chain, so dropping a placeholder shifts every offset after it.
- **`weight`'s accessor is built outside its `WEIGHT` guard** (2d `:49`, 3d `:52`, 4d `:53`) while
  `divisor`'s is built inside its own (2d `:54`). Harmless today; at port time the `tensor::weight` binding
  must resolve in the no-weight configs, so either declare it there or move the construction inside the guard.
- **Cross-op / shared kernels: nothing to coordinate.** The op **owns all seven** of its kernel `.cpp` files
  and **no other op instantiates them** (verified by grepping all seven paths across `ttnn/` — the only hits
  are this factory's own `kernel_source` assignments). So: no `_metal2` fork to reuse, none to create, **no
  sunset list**. The two out-of-directory dependencies are *headers*, not borrowed kernel files.
- **Both donors take `DataflowBuffer`, which is the easy case — no donor-side change, no fork.** The
  dataflow donor's `read_tile` / `read_value` / `read_line` and the compute donor's `copy_tile_init_with_dt` /
  `pack_tile_with_dt` / `mul_tiles_init_with_dt` all take `DataflowBuffer` **by value**, and the kernels
  already pass named DFB locals. Construct those locals from the tokens
  (`DataflowBuffer dfb_input_obj(dfb::input);`) and every call site is unchanged.
- **Compute LLK call sites take the rule-2 implicit conversion, not a getter.** `copy_tile`, `mul_tiles`,
  `mul_tiles_bcast_scalar`, `mul_bcast_scalar_init`, `reconfig_data_format`, and
  `compute_kernel_hw_startup` all take `uint32_t cb_id` and have **no** `DataflowBuffer` overload anywhere in
  `tt_metal/hw/inc/api/compute/`. Pass `dfb::name` directly and let the conversion fire; do **not** extract
  `.get_id()` and do **not** hunt for a getter that doesn't exist.
- **`get_tile_size(cb_id)` breadcrumbs — confirm, don't swap blind.** Sanctioned today; whitelist rule 7 moves
  them onto the object at port time: `writer_..._2d.cpp:25`, `writer_..._4d.cpp:27`. Both are declared
  `const auto`, so the member-getter form applies (not the `constexpr` carve-out). **Leave the donor's own
  internal ones alone** (`dataflow/moreh_common.hpp:683`, `:709`, `:753`) — a shared header; changing it
  reaches every moreh op.
- **These kernels are already part-modernized.** All seven are on `DataflowBuffer` / `Noc` / `CoreLocalMem` /
  `TensorAccessor` with the current `api/dataflow/dataflow_buffer.h` include. Expect a binding-layer change,
  not an idiom rewrite.
- **`constexpr` vs `const` on the CB handles.** All CB indices are declared `constexpr uint32_t` — the form
  that admits the token / constexpr-cast path. The data-format and tile-size locals are `const auto`. Worth a
  glance before assuming a form.
- **No quasar copy of this op exists** (`ttnn/cpp/ttnn/operations/experimental/quasar/` has no `nll` entry),
  so there is no shortcut-port lookalike to be misled by. A negative pointer, to save a wrong turn.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: `✓ clean`.** Two donor files, one donor class between them, eight consumed symbols, every
shape ✓. No ⚠ / ✗ / ⭐ entries, so the per-call detail section is omitted per the report format.

Inventory of every `#include` in the op's kernels resolving outside the op directory:

| Op kernel | Include | Resolved donor | Class | Status |
|---|---|---|---|---|
| all three readers; writers 3d, 4d | `ttnn/kernel/dataflow/moreh_common.hpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | **3** — `ttnn/cpp/ttnn/kernel/` (singular), the second shared-kernel pool; treat as shared-lib | ✓ |
| compute | `ttnn/kernel/compute/moreh_common.hpp` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | **3** — same pool | ✓ |
| all seven | `api/dataflow/dataflow_buffer.h` | `tt_metal/hw/inc/api/…` | 1 — LLK/HAL/firmware | ✓ no concern |
| readers | `api/core_local_mem.h`, `api/tensor/noc_traits.h` | `tt_metal/hw/inc/api/…` | 1 | ✓ no concern |
| writers | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/tensor/noc_traits.h` | `tt_metal/hw/inc/api/…` | 1 | ✓ no concern |
| compute (transitively, via the donor) | `api/compute/*.h` — `compute_kernel_api.h`, `bcast.h`, `eltwise_binary.h`, `eltwise_unary/*.h`, `tile_move_copy.h`, … | `tt_metal/hw/inc/api/compute/` | 1 | ✓ no concern |

**Per-call shape analysis** (only the symbols this op actually calls; both headers are large and the rest is
not exercised here):

| Donor | Function | Signature shape | Status |
|---|---|---|---|
| dataflow | `read_tile(DataflowBuffer cb, AddrGen addrgen, uint32_t noc_id, …)` (`:666`) | `DataflowBuffer` **by value**; `AddrGen` deduced to `TensorAccessor<DSpec>` (Shape 1) | ✓ excellent |
| dataflow | `read_value(DataflowBuffer cb, AddrGen addrgen, …)` (`:695`) | same | ✓ excellent |
| dataflow | `read_line(DataflowBuffer cb, DataflowBuffer cb_scratch, AddrGen addrgen, …)` (`:739`) | **two** `DataflowBuffer` by value + `TensorAccessor` | ✓ excellent |
| dataflow | `get_tilized_idx(uint32_t, uint32_t)` (`:618`) | plain scalars | ✓ n/a |
| dataflow | `get_noc_offset(uint32_t, uint32_t, uint32_t, uint32_t&)` (`:635`) | plain scalars + out-param | ✓ n/a |
| dataflow | `fp32_dest_acc_cast(…)` / `FP32_DEST_ACC_FTYPE` (`:23-31`) | scalar overloads + a typedef | ✓ n/a |
| compute | `copy_tile_init_with_dt(DataflowBuffer, uint32_t)` (`:35`) | `DataflowBuffer` by value | ✓ excellent |
| compute | `pack_tile_with_dt(uint32_t, DataflowBuffer)` (`:28`) | `DataflowBuffer` by value | ✓ excellent |
| compute | `mul_tiles_init_with_dt(DataflowBuffer, DataflowBuffer)` (`:100`) | two `DataflowBuffer` by value | ✓ excellent |

**These are the `DataflowBuffer` rows, not the `CircularBuffer` rows** — the two are opposite verdicts and sit
adjacent in the shape table, so: checked, and every consumed signature names `DataflowBuffer`. Both donors
have **already migrated to DFB**, so neither needs a donor-side change or a fork, and nothing here routes
cross-team work. No consumed signature takes a `uint32_t sem_id`, a sem address, a `TensorAccessorArgs<N>`,
a CTA-offset NTTP, an old-style addr-gen, or a `CircularBuffer`.

Everything else the compute kernel calls — `compute_kernel_hw_startup`, `copy_tile`, `mul_tiles`,
`mul_tiles_bcast_scalar`, `mul_bcast_scalar_init`, `reconfig_data_format`, `recip_tile{,_init}`,
`negative_tile{,_init}`, `tile_regs_*` — resolves to **class 1** `tt_metal` LLK compute headers, pulled in
transitively by the donor (`compute/moreh_common.hpp:20-24` and siblings). No concern; these take CB indices
by design (see the Device 2.0 gate detail, item 3).

**Borrowed kernel files (file-path kernel instantiation): none.** All seven `kernel_source` paths point inside
this op's own `device/kernels/`. No `_metal2` fork exists beside any of them, and none is needed — every file
has exactly one consumer. Nothing to sunset, nothing to coordinate.

### Relaxation candidates

**None to mine.** The candidate source is a custom `compute_program_hash` revealing which tensor properties
the op actually depends on, and this op has no custom hash — it uses the framework default over
`operation_attributes_t` and `tensor_args_t`. Nothing fallible to record for the relaxation roadmap.

### TTNN factory analysis

The sheet-derived facts, with `file:line` evidence. **Two cells could not be read** (`Is able to port?`,
`TensorParameter relaxation`) — see [Result](#result); everything below is my own code evidence, which is the
cross-check, not a substitute for those cells.

- **Current concept:** `descriptor` — `Factory::create_descriptor` returns `tt::tt_metal::ProgramDescriptor`
  (`..._device_operation.hpp:35`; body at `..._program_factory.cpp:701-732`).
- **Factory set:** exactly one — `using program_factory_t = std::variant<Factory>`
  (`..._device_operation.hpp:41`). The three rank paths are **internal code paths, not factories** — a point
  worth carrying into the sheet cross-check, since a sheet that split them into three rows would be a
  phantom-row finding.
- **Op-owned tensors:** none — structurally impossible on `descriptor`; each `impl` returns a bare
  `ProgramDescriptor` with no `buffers` vector.
- **MeshWorkload need:** none — not a `WorkloadDescriptor` op.
- **Custom hash:** absent (gate-irrelevant either way — the port would leave one intact).
- **`get_dynamic_runtime_args`:** absent. Would have been a gate → TTNN.
- **`override_runtime_arguments`:** absent → target is the base `ProgramSpecFactoryConcept`. Nothing for the
  porter to translate.
- **Pybind `create_descriptor`:** absent. `moreh_nll_loss_nanobind.cpp:24-37` binds only the user-facing
  `ttnn::moreh_nll_loss`, so the port removes **no** user-visible Python API and the port report needs no
  entry for it.
- **Other risky pybind:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** (`descriptor` + no op-owned tensors +
  `Override runtime args method? == no`). A clean mapping onto the common target; no recipe gap.

## Misc anomalies  *(team-only, non-gating)*

Latent issues noticed while auditing. These route to the **ops team**; the port does not act on them, and none
is a gate. The dead CBs and dead CB *declarations* are excluded — they are real port work, recorded above.
This is a notably long list for an op this size; several items look like residue from an earlier revision.

1. **Nine dead data-format locals.** Each reader computes three `get_dataformat` values and uses none:
   `input_data_format`, `weight_data_format`, `divisor_data_format` at
   `reader_..._2d.cpp:36`/`:38`/`:40`, `..._3d.cpp:37`/`:39`/`:41`, `..._4d.cpp:40`/`:42`/`:44`. Note
   `divisor_data_format` is computed **unconditionally** on `c_3`, a CB that does not exist when divisor is
   absent — the same category of latent read as Must-fix 3, though `get_dataformat` only indexes a JIT
   descriptor array rather than a live interface, so it is benign today.
2. **The entire per-core compute RTA vector is dead.** All three `impl`s populate `compute_desc_1.runtime_args`
   / `compute_desc_2.runtime_args` with `{units_per_core}` (`:235-243`, `:448-456`, `:678-686`), and the
   compute kernel reads **no runtime args at all** (zero `get_arg_val`; it takes the value from
   `get_compile_time_arg_val(0)` at `compute:11`). Every dispatch writes per-core args nothing reads.
   *(Also the bait for the anti-pattern — see [Heads-ups](#heads-ups-mirrors-the-brief).)*
3. **A dead writer RTA in the 2d path.** `impl_2d` passes four writer args — `output_buf`, `units_per_core`,
   `tile_offset`, `origin_N` (`:225-232`) — and `writer_..._2d.cpp` reads only the first three (`:11-13`).
   `origin_N` is never read. The 3d writer (5 passed, 5 read) and 4d writer (3 passed, 3 read) are consistent.
4. **Two dead reader `element_size` RTAs.** `input.element_size()` is passed as the last reader arg in all
   three paths (`:222`, `:434`, `:666`). The **3d** reader uses it (`:89`); the **2d** (`:23`) and **4d**
   (`:25`) readers read it and never use it.
5. **A redundant double `reserve_back` in the 4d reader.** `reader_..._4d.cpp:71` calls
   `dfb_weight_obj.reserve_back(weight_num_tile)`, then `:74` calls `read_line(...)`, whose `do_reserve`
   parameter defaults to `true` and reserves the same count again
   (`dataflow/moreh_common.hpp:745`, `:749-751`). Harmless — the second wait is satisfied immediately since
   nothing was pushed in between — but it is dead work on every core, and it reads as a misunderstanding of
   `read_line`'s default. *(`step1`'s small reader calls `read_line` with no preceding `reserve_back`, which is
   the correct usage; the two ops disagree.)*
6. **The `/1024` element-size derivation is correct here but fragile.** `dataflow/moreh_common.hpp:709`
   computes element size as `tile_size / 1024`, which is wrong for block-float formats (bf8 gives 1024, not
   1088). Safe in this op only because `validate_inputs` hard-asserts input and weight are `BFLOAT16`
   (`..._device_operation.cpp:23`, `:39`). If those assertions are ever relaxed to admit a block-float dtype,
   `read_value`'s byte offsets mis-address silently. Worth a comment or a guard at the donor. *(Same finding as
   `step1`'s anomaly 3 — it lives in the shared header, so it is one fix for both ops.)*
7. **A hardcoded `target_element_size` in the 3d reader.** `reader_..._3d.cpp:48` sets
   `uint32_t target_element_size = 4;  // sizeof(int32)` rather than deriving it or receiving it as an arg.
   Correct today — `validate_inputs` requires `target_tensor.dtype() == DataType::INT32`
   (`..._device_operation.cpp:28`) — but it silently couples the kernel to that assertion, and the 2d/4d
   readers do not need the value at all. Note the 2d and 4d paths *do* pass an unused `element_size`
   (anomaly 4) while the 3d path hardcodes the *other* element size it needs: the arg plumbing and the
   hardcoding are inconsistent across the three readers.
8. **An unused include in the 4d writer.** `writer_..._4d.cpp:6` includes
   `ttnn/kernel/dataflow/moreh_common.hpp` but uses nothing from it (no `get_noc_offset`, no `read_*`, no
   `Scalar`, no `fp32_dest_acc_cast`) — it needs only `Noc`, `DataflowBuffer`, `TensorAccessor`, and
   `get_tile_size`. The 3d writer's identical include *is* used (`get_noc_offset` at `:44`).
9. **`c_28`'s comment does not match its use.** The factory labels `c_28` `// tmp3` (`:95`, `:306`, `:536`),
   and the compute kernel names it `cb_tmp3` (`:25`) — but `c_27`, labelled `// tmp2` in the factory
   (`:94`, `:305`, `:535`), is named `cb_divisor_recip` in the kernel (`:23`) and holds `1/divisor`. The
   factory-side comment is stale relative to the kernel's actual use. Cosmetic, but it is the kind of drift
   that misleads a reader trying to follow the divisor path.
10. **`reduction` is read but only partially used.** Unlike `step1` — where `reduction` is entirely unused —
    `step2`'s device-op **does** consult it (`..._device_operation.cpp:61`, `:95`), but the *factory* takes it
    as a parameter and every `impl` marks it unused (`const std::string& /*reduction*/` at `:51`, `:264`,
    `:477`) while `create_descriptor` still copies it out of the attributes (`:713`). So the program never
    varies on `reduction`, yet it participates in the default program hash — two invocations differing only in
    `reduction` (with `NONE`'s output-tensor branch not taken) miss the cache and compile a byte-identical
    program.

## Questions for the user

1. **Does `compute_kernel_hw_startup(dfb::x, …)` constitute an endpoint binding?** This is the one judgment
   call in the CB census I could not settle from the recipe (`moreh_nll_loss_step2_kernel.cpp:34`). It decides
   whether `c_24` under no-`WEIGHT` is a 1-toucher self-loop or a 0-toucher conditional DFB. **It does not
   change the porter's instruction** — either way the spec must exist in every config, because
   `dfb::tmp_weight` must resolve for `compute:18` and `compute:34` to compile — so the port is not blocked on
   the answer. But the framework team's answer would let a future audit of any op with a
   `compute_kernel_hw_startup` call classify this cleanly rather than reasoning around it.
2. **`compute_kernel_config` for an op whose compute kernel ignores most of it.** `fp32_dest_acc_en` correctly
   drives the intermediate CBs' data format (`:83`, `:294`, `:517`) and the `FP32_DEST_ACC_EN` define — that
   part is live and used. But `math_fidelity`, `math_approx_mode`, `dst_full_sync_en`, and
   `unpack_to_dest_mode` are threaded into both `ComputeConfigDescriptor`s (`:165-171`, `:376-382`,
   `:606-612`) for a kernel whose only math is `negative`, `recip`, `mul_tiles`, and
   `mul_tiles_bcast_scalar`. `packer_l1_acc` is destructured from the config and never used at all (`:75`,
   `:286`, `:509`). Not a port question; flagging it because the ops team may want to know the knob surface is
   wider than the kernel's actual sensitivity.
3. **`c_7`'s mis-scoped guard appears in both `step1` and `step2`** (this audit's Must-fix 1/2, and `step1`'s
   conditional-DFB finding). Same author, same shape, two ops. Worth a single ops-team pass over the
   `moreh_nll_loss` family rather than two independent fixes — and worth checking whether other `moreh` ops
   that call `read_line` share it.

## Recipe notes

Friction with the audit recipe itself, not findings about the op. *(Notes 1-2 restate friction already logged
in `step1`'s audit, since it recurred here; notes 3-5 are new to this op.)*

1. **The recipe has no outcome for "the readiness sheet is unreachable," and it is a reachable state.**
   *(This audit ended GREEN — the note describes friction hit on the way there. The verdict is not in
   question.)* Recurred verbatim here. The *TTNN factory concept prerequisite* section enumerates five routings — `yes`,
   an attributed `no`, an unattributed `no`, spreadsheet-broken, and `MetalV2` — all of which presuppose the
   cell was **read**, while `ttnn_op_porting_readiness.md` states that the connector "authorizes only in the
   main interactive session" and "You **cannot** authorize it from inside a session." A non-interactive session
   can therefore never satisfy the fetch step. I again reported the gate as **indeterminate**, withheld the
   brief, and routed it as a data-availability question rather than alleging a defect. **Suggested addition:**
   a sixth routing — *"sheet unreachable → GATE indeterminate; report the code-side cross-check in full,
   withhold the brief, route to the launcher to re-run with Drive access"* — plus a line in the *Reference
   data* preamble noting the fetch is impossible in a non-interactive session, so an auditor meets this
   before doing the work rather than at the end. **The recurrence is the point:** this is not a one-off, it is
   deterministic for a whole class of sessions, and a second op has now paid the cost — in both cases the
   audit was complete except for two cells, and in both cases the user unblocked it by reading them manually.
2. **A second suggested addition, if relaying cells is to be the sanctioned fallback.** For `step1` the user
   read the cells from the sheet and relayed them, which cleared the gate. That works, but it silently changes
   what the auditor can verify: the **factory-set match** cross-check needs the row *set*, not named cells, so
   it cannot be run at all — and any disagreement would be between the code and a human's reading rather than
   between the code and the sheet. The recipe should say a relayed value clears the gate, that its provenance
   must be recorded, and that the factory-set staleness check is not satisfiable that way.
3. **The sanctioned-free-function list omits `get_dataformat(cb_id)`, and the omission is load-bearing.** This
   was the single hardest call in this audit and it decided RED vs GREEN. The Device 2.0 Green bullet says
   "the list is the whole test" and names exactly two functions; `get_dataformat` is not one, which on a
   literal reading makes nine call sites in this op candidate holdovers and REDs the gate. What actually
   resolves it is **outside** the audit doc: port-recipe kernel-side whitelist **rule 7** names
   `get_dataformat(cb_id)` explicitly, next to `get_tile_size(cb_id)`, as port-stage metadata work. The audit's
   own breadcrumb gestures at this ("a port-stage change that does not move the Device 2.0 boundary here") but
   does not name the function, so an auditor must follow the cross-reference into the port recipe to get it
   right — and the audit elsewhere says not to pre-load that document. Corroborating evidence that the
   omission is incompleteness rather than intent: `get_dataformat` sits in the *same three-line block* as the
   sanctioned `get_tile_size` inside the Device 2.0 `CircularBuffer` wrapper, with identical grounding
   (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:113-115`). **Suggested fix:** add `get_dataformat(cb_id)`
   — and ideally the rest of rule 7's metadata family, `get_tile_hw(cb_id)` and the `chlkc_descriptors.h`
   array lookups — to the sanctioned list, with a one-line pointer to rule 7 saying these are port-stage
   moves, not Device 2.0 holdovers.
4. **The Device 2.0 gate does not say how compute kernels are scoped, and the holdover cue misfires on them.**
   The gate is titled *"Device 2.0 **Data Movement** migration"* and its violation examples are all
   data-movement idioms, but its instruction is "every kernel this op exercises," which includes compute
   kernels. A compute kernel legitimately passes raw `uint32_t` CB indices to every LLK primitive it calls
   (`copy_tile`, `mul_tiles`, `compute_kernel_hw_startup`, …) — matching the holdover *shape* exactly, with a
   `DataflowBuffer` in scope at the call site. What saved me is the second conjunct: no wrapper-method
   replacement exists, because **nothing** in `tt_metal/hw/inc/api/compute/` takes a `DataflowBuffer`. That is
   a real test but an indirect one, and an auditor who keys on the shape (CB index + wrapper in scope) will
   RED every compute kernel in the codebase. **Suggested addition:** one sentence under the Device 2.0 gate —
   *"the compute LLK surface is CB-index-based by design and has no DFB overloads; raw indices at compute LLK
   call sites are never holdovers. A compute kernel is Device 2.0 when its **FIFO/data-movement** operations
   are on `DataflowBuffer`."*
5. **The census needs a rule for a CB *declaration* without an access, and for hardware-config references.**
   This op has five DFB objects constructed unconditionally in the compute kernel (`:15`, `:18`, `:22`, `:24`,
   `:26`) whose FIFO uses are all `#ifdef`-guarded, plus a `compute_kernel_hw_startup(cb_tmp_weight, …)` that
   references a CB without producing, consuming, or raw-accessing it. The census defines an endpoint as FIFO
   produce / FIFO consume / raw-pointer access — none of which these are — yet in Metal 2.0 each still
   requires a binding *token* to exist, so treating them as zero-toucher dead CBs would delete specs the
   kernel cannot compile without. I resolved it by counting them as role-free touchers and self-looping, which
   is safe in both directions, and flagged the ambiguity as [Question 2](#questions-for-the-user).
   **Suggested addition:** extend the endpoint definition with a fourth category — *"or requires a binding
   token to compile (a DFB constructed unconditionally, a CB id passed to a hardware-config primitive)"* —
   and note that such a reference makes the CB **not dead** even with no FIFO or pointer access. Without it,
   the recipe's strongest warning (never drop a live CB) points the wrong way for exactly this shape.
