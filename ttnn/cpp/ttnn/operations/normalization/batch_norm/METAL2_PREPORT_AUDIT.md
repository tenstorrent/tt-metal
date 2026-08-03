# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/normalization/batch_norm`

Two **independent** device-operations live under this op directory. They share **no** factories and
**no** kernels with each other; they are bundled into one report because they are sequenced by a
single user-facing entry point (`ttnn::batch_norm` → `batch_norm.cpp`, which calls
`ttnn::prim::running_statistics` then `ttnn::prim::batch_norm`) and the audit was requested at
directory scope. Per-DeviceOperation attribution is retained throughout; **both clear every gate**.

- **`BatchNormOperation`** — `device/batch_norm_device_operation.{hpp,cpp}`
  - `BatchNormFactory` (`device/batch_norm_program_factory.cpp:140`), sole alternative of
    `program_factory_t`
  - kernels: `dataflow/reader_batch_norm.cpp`, `dataflow/writer_batch_norm.cpp`, and **two**
    runtime-selected compute sources — `compute/batch_norm_kernel.cpp`,
    `compute/batch_norm_sfpu_kernel.cpp`
- **`RunningStatistics`** — `device/running_statistics_device_operation.{hpp,cpp}`
  - `RunningStatisticsProgramFactory` (`device/running_statistics_program_factory.cpp:137`), sole
    alternative of `program_factory_t`
  - kernels: `dataflow/reader_running_statistics.cpp`, `dataflow/writer_running_statistics.cpp`, and
    **two** runtime-selected compute sources — `compute/running_statistics_kernel.cpp`,
    `compute/running_statistics_sfpu_kernel.cpp`

**Unreferenced kernel files:** none — all eight kernels are instantiated by their device-op's factory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

**Audited at:** branch `virdhatchani/BN_Metal_port`, HEAD `4386dc456a1`, merge-base with `main`
`c8fab99f9ac`.

> ### Readiness-sheet data provenance
>
> The claude.ai Google Drive connector is **not authorized in this session**, so the *"Operations
> analysis"* sheet could not be fetched directly. The two rows for this op were **supplied verbatim by
> the invoker** (revised set, superseding an earlier one) and are treated as the sheet of record.
> Every *cheaply-checkable* column was cross-checked against the code — see
> [TTNN readiness cross-check](#ttnn-readiness-cross-check), which records **one disagreement** on a
> new, non-gating column. `Is safe to port?` was **not** verified, per recipe (expert-judgment axis).

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/normalization/batch_norm/` |
| **Overall** | **GREEN — both factories cleared; brief issued for both.** Invoker has scoped both into a single PR |
| **DOps / Factories** | `BatchNormOperation` → `BatchNormFactory` · `RunningStatistics` → `RunningStatisticsProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 8 kernels + all 3 donor headers are structurally Device 2.0. No holdovers |
| *Prereqs* — Cross-op escapes | Ok — 3 donor headers, all `uint32_t cb_id` / raw-L1-address shapes (✓ crossing) |
| *Feature Support* — overall | **GREEN** — all four Appendix A entries `N/A` |
| *Feature Support* — Variadic-CTA | Ok — every `get_compile_time_arg_val` index is a source literal or a `constexpr` accessor offset |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both) — cross-check clean on every gate conjunct |
| *TTNN Readiness* — Concept (current) | `descriptor` (both) — confirmed in code |
| *TTNN Readiness* — Op Classification / Execution Model | `PD Op (pointer-patching)` / `SPMD` (both) — corroborated: pointers ride RTAs as `Buffer*` objects, one program stamped mesh-wide |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — neither op is a `WorkloadDescriptor` (sheet cell correctly blank) |
| *TTNN Readiness* — Is safe to port? | **Yes** (both) |
| *TTNN Readiness* — Custom hash (`compute_program_hash`) | **No** (both) — absent from the directory |
| *TTNN Readiness* — Backdoor custom hash (`attribute_values` / `to_hash`) | **`BatchNormOperation`: yes** (`to_hash` @ `batch_norm_device_operation.cpp:121`) · **`RunningStatistics`: NO — sheet says `yes`, code says none.** Non-gating; see cross-check |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (both) — hook absent from both device-ops |
| *TTNN Readiness* — `override_runtime_arguments` | No (both) — absent from both factories |
| *TTNN Readiness* — Pybind descriptor (`nb::class_` of device op) | No — `batch_norm_nanobind.cpp` exposes only the user-facing `ttnn::batch_norm` |
| *TTNN Readiness* — Smuggled pointer | No (both) — confirmed: zero `->address()` sites; the `Buffer*` form is registered, not smuggled |
| *TTNN Readiness* — Op-owned tensors | No (both) — no factory-allocated device tensors beyond declared io |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (both), no op-owned tensors — matches the sheet's `Porting Target` |
| *Port work* — Offset base pointer | **none** — zero `->address()` sites in the op; every tensor address rides the `Buffer*` binding form with no arithmetic |
| *Port work* — Tensor bindings (per binding) | **Case 1** for all 11 bindings (every address feeds a `TensorAccessor`) |
| *Port work* — TensorParameter relaxation | **none** (both) — matches the sheet; no `ArgConfig::Runtime*` in the op or its donors |
| *Port work* — TensorAccessor 3rd arg | **none** — all 11 `TensorAccessor(...)` sites are 2-arg |
| *Port work* — CB endpoints | 1P+1C mostly; **5 compute self-loops**; **3 config-flipping CBs**; no multi-binding; no dead CBs |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves at port time with a
**self-loop** (one toucher) — no CB needs the multi-binding advanced option and none is dead. Three
CBs' dispositions **flip with config**; recorded per `(CB, config)` below.

## Result

**GREEN — brief issued for both factories.**

Every gate is cleared for both `BatchNormOperation::BatchNormFactory` and
`RunningStatistics::RunningStatisticsProgramFactory`: Device 2.0 compliance on all eight kernels and
all three donor headers, all four Appendix A features absent, `Is able to port? = yes` with a clean
cross-check, no offset base pointers anywhere, and no `TensorAccessor` third argument anywhere.
Nothing in this op requires a Metal 2.0 feature that does not exist, an ops-team functional fix, or a
Device 2.0 migration.

The invoker has scoped **both factories into a single PR**. That is well-supported here — the two
device-ops share no factories and no kernels, so neither constrains the other's conversion, and each
is a complete sub-port on its own. `METAL2_PORT_BRIEF.md` covers both, ordered so that either can be
finished and committed independently if the pass has to stop early.

One **non-gating** discrepancy is recorded for the readiness-sheet owner: the new
`Backdoor custom hash` column reads `yes` for the `RunningStatistics` row, but that device-op has no
`to_hash()` and no `attribute_values` in code. Details and the reasoning for not gating on it are in
the cross-check below.

---

## Gate detail

### TTNN readiness cross-check

Sheet rows as supplied (revised set). Both rows are identical in every cell:

| Column | Both rows |
|---|---|
| `Concept` | `descriptor` |
| `Op Classification` | `PD Op (pointer-patching)` |
| `Execution Model` | `SPMD` |
| `Porting Target` | `ProgramSpecFactoryConcept` |
| `Custom hash (compute_program_hash)` | no |
| `Backdoor custom hash (attribute_values / to_hash)` | **yes** |
| `Runtime-args update (get_dynamic_runtime_args)` | no |
| `Override runtime args method? (PD only)` | no |
| `Pybind descriptor (nb::class_ of device op)` | no |
| `Smuggled pointer (raw buffer addr in RTA/CRTA)` | no |
| `Known op issues` | *(blank)* |
| `Is safe to port?` | **yes** |
| `Is able to port?` | **yes** |
| `TensorParameter relaxation` | none |
| `Op-owned tensors?` / `Secretly SPMD Workload?` | *(blank)* |
| `Pointer patching perf issue?` | `suspect perf regression (+ fixed latent bug)` |
| `Formerly custom hashed?` | `yes (to_hash, still present)` |

Cross-check of the cheaply-checkable columns:

| Column | Sheet | Code evidence | Agrees |
|---|---|---|---|
| `Concept` | `descriptor` | `create_descriptor` returning `tt::tt_metal::ProgramDescriptor`: `batch_norm_device_operation.hpp:39`, `running_statistics_device_operation.hpp:36`. Single-alternative `program_factory_t` in both (`:45`, `:42`) | ✓ |
| `Custom hash (compute_program_hash)` | no | no `compute_program_hash` in the directory (grep clean) | ✓ |
| `Backdoor custom hash` | yes (both) | **`BatchNormOperation`: yes** — `to_hash()` declared `batch_norm_device_operation.hpp:22`, defined `batch_norm_device_operation.cpp:121`. **`RunningStatistics`: no** — grep for `to_hash` and `attribute_values` over `running_statistics_device_operation.{hpp,cpp}` is empty; its `operation_attributes_t` (`:15-23`) declares only `get_dtype()` | **✗ on the RunningStatistics row** |
| `Runtime-args update (get_dynamic_runtime_args)` | no | hook absent from both device-ops | ✓ |
| `Override runtime args method?` | no | absent from both factories | ✓ |
| `Pybind descriptor` | no | `batch_norm_nanobind.cpp:77` binds only `ttnn::batch_norm`; no `nb::class_` of either device-op, no factory internals | ✓ |
| `Smuggled pointer` | no | **zero** `->address()` sites in the directory. Addresses reach kernels as `Buffer*` objects pushed into `KernelDescriptor::runtime_args`, which the framework auto-registers and patches — registered, not smuggled | ✓ |
| `Op Classification` / `Execution Model` | `PD Op (pointer-patching)` / `SPMD` | consistent with the `Buffer*` delivery form above, and with one program stamped mesh-wide (no `MeshWorkload`, no per-coord variation) | ✓ |
| `TensorParameter relaxation` | none | no `ArgConfig::Runtime*` in the op or its three donors | ✓ |
| `Op-owned tensors?` | *(blank)* | no factory-allocated tensors beyond declared io; no `WorkloadDescriptor` | ✓ |
| Factory-set match | 2 rows | exactly 2 factories in code, 1:1 with the rows; no phantom, no missing | ✓ |

Cross-column invariants hold: `get_dynamic_runtime_args = no` is compatible with `descriptor`;
`Op-owned tensors?` is blank on a `descriptor` concept (which cannot carry them); `Secretly SPMD
Workload?` is blank because it applies only to `Concept == WorkloadDescriptor`. Applying the
documented derivation to these rows yields `Is able to port? = yes`, matching the sheet.

#### The `Backdoor custom hash` discrepancy — why it does not gate

`RunningStatistics` has no attribute-hash customization of any kind, but its row reads `yes`. Since
every other cell in the two rows is byte-identical, this looks like a copy-down from the
`BatchNormOperation` row. I am **not** gating on it, for three reasons:

1. **It is not a gate conjunct.** The recipe's `Is able to port?` derivation composes
   `Is safe to port?`, `Custom hash`, the two runtime-args columns, `Pybind descriptor` and `Concept`.
   `Backdoor custom hash` is not among them — and the sheet itself confirms this by reporting
   `Is able to port? = yes` *alongside* `Backdoor custom hash = yes`. A mismatch here cannot move the
   gate verdict.
2. **The recipe's spreadsheet-broken GATE is scoped to the gate-feeding columns and the factory-set
   match.** This column is new and post-dates the recipe's cross-check list, so there is no
   instruction to verify it at all; I checked it opportunistically. Gating the port on an
   informational-column mismatch would be exactly the "too-conservative RED misroutes work to a team
   that doesn't need it" failure the recipe warns against.
3. **The direction of the error is harmless.** The sheet over-reports customization for
   `RunningStatistics`; it does not hide any. There is no hash for the port to mishandle.

Routed to the **readiness-sheet owner** as a data-quality fix, not a blocker.

> **`to_hash()` — present on `BatchNormOperation` only, and the port does not delete it.**
> The `Custom hash (compute_program_hash)` cell is `no`, so the recipe's sanctioned custom-hash
> deletion — which is scoped to a `compute_program_hash` override — **does not trigger**. `to_hash()`
> is the **ttsl attribute-hash protocol**: it customizes how `operation_attributes_t` hashes *within*
> the framework's default reflection hash, rather than replacing the cache key. Tensor args are still
> hashed separately, so `TensorSpec` remains folded in and the `UpdateTensorArgs`-on-cache-hit hazard
> from the migration guide's troubleshooting table **does not apply**. Deleting it would change the
> cache key — a functional change outside the port's scope.
>
> I checked what it omits, since `Formerly custom hashed? = yes (to_hash, still present)` marks it as
> residual debt. `to_hash()` (`batch_norm_device_operation.cpp:122`) hashes `eps`, `memory_config`,
> `get_dtype()` and `compute_kernel_config`; of the five `operation_attributes_t` fields, only
> `input_dtype` and `dtype` are not hashed independently. That is **safe today**: `input_dtype` is
> assigned `input.dtype()` at `batch_norm_device_operation.cpp:143`, so it always mirrors a property
> of a tensor the default hash already folds in, and `dtype` reaches the key through `get_dtype()`.
> Whether the port should nonetheless retire the backdoor is a question for the owner —
> see [Q1](#questions-for-the-user).

> **`Pointer patching perf issue? = suspect perf regression (+ fixed latent bug)`.** Worth surfacing
> as a *motivation* rather than a finding: this op is classified `PD Op (pointer-patching)` precisely
> because its tensor addresses ride RTAs as `Buffer*` objects that the framework must patch on every
> cache hit. **The Metal 2.0 port removes that mechanism entirely** — a typed `TensorBinding` carries
> the base address on the framework's own channel, so the pointer-patching path this suspicion attaches
> to no longer exists post-port. If the regression is real and attributable to patching, this port is
> plausibly its fix. Recommend measuring before/after rather than assuming; recorded so the porter
> knows the perf comparison is worth more attention here than on a typical port.

### Device 2.0 (every kernel used) — GREEN

All eight in-scope kernels are structurally Device 2.0: `Noc` + `noc.async_read` / `async_write` with
endpoint args, `DataflowBuffer` objects (already the DFB *type*, though still constructed from
magic-number CB-index CTAs — that binding swap is the Metal 2.0 port work), `TensorAccessor`,
`CoreLocalMem`, and **member** metadata getters (`dfb.get_entry_size()`).

Targeted holdover scan across all 8 kernels **and** all 3 donor headers returned zero hits for:
`get_local_cb_interface`, `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen*`,
`cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, free-function
`get_read_ptr(cb)` / `get_write_ptr(cb)` / `get_tile_size(cb)`, `get_semaphore`,
`get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`, `read_tile_value`,
`get_pointer_to_cb_data`, raw `noc_async_read(` / `noc_async_write(`, `get_noc_addr`.

Two shapes were examined closely and are **not** violations — recorded transparently so a reader can
form their own view:

1. **`fill_cb_with_value(dfb_id_one, one_u)`** — `reader_running_statistics.cpp:56`. A free function
   taking a `uint32_t` CB index, from the shared pool
   `ttnn/cpp/ttnn/kernel/dataflow/cb_fill_helpers.hpp`. It is **not** an isolated CB-index holdover
   under the gate's definition, which requires *both* that a Device-2.0 wrapper object be in scope at
   the call site *and* that a wrapper-method replacement exist. Neither holds: the reader constructs
   no `DataflowBuffer` for the `one` CB, and `DataflowBuffer` has no fill method — this is a library
   *orchestrator* (it does `reserve_back` / format-aware fill / `push_back` internally), not a CB
   accessor. The helper is itself Device 2.0 inside (`CircularBuffer` + `CoreLocalMem`). Port-side it
   becomes `fill_cb_with_value(dfb::one, one_u)` via the constexpr `DFBAccessor::operator uint32_t`.
2. **`dfb.get_write_ptr()` fed to raw-L1-address helpers** — 16 sites across the four dataflow
   kernels (e.g. `writer_batch_norm.cpp:89`, `reader_batch_norm.cpp:50`). These are **member** public
   peeks, which the CB→DFB whitelist explicitly sanctions for "interop with helpers that still take a
   raw L1 address". The port recipe further directs that such transfer/peek idioms **stay as-is**
   (minimal diff; "cleanup debt, not evil").

Donor headers are Device 2.0 as well — `dest_format_helpers.hpp` states it in its own header comment
("CB operations migrated to Device 2.0 `CircularBuffer` method form"); `fill_tile_utils.hpp` contains
no CB or NoC API at all (pure L1 pointer writes); `cb_fill_helpers.hpp` uses `CircularBuffer` +
`CoreLocalMem`.

### Feature compatibility — GREEN (all N/A)

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` / `CreateGlobalCircularBuffer` / `.global_circular_buffer` field / `remote_cb_*` / `.remote_index(` / `remote_circular_buffer.h`. All 23 CBs are plain `CBDescriptor`s with a single `CBFormatDescriptor` |
| CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset`, `set_address_offset`, 4-arg `UpdateDynamicCircularBufferAddress`, or `cb_descriptor_from_sharded_tensor`. No borrowed-memory CBs at all (no `set_globally_allocated_address`) |
| GlobalSemaphore | N/A | The op uses **no semaphores of any kind** — grep for `[Ss]emaphore` across the directory is empty |
| Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent: both `tensor_args_t` are fixed-count structs of named tensors (no `std::vector<Tensor>`). Kernel-level decider absent: every `get_compile_time_arg_val` index is a source literal or a `constexpr` `TensorAccessorArgs<…>::next_compile_time_args_offset()` — no runtime-varying CTA index |

### Offset base pointers — GREEN

**Zero `->address()` sites in the entire op directory** (grep clean, including
`(*buffer).address()` / `tensor.buffer()->address()` spellings). Every tensor address reaches its
kernel through the **`Buffer*`-binding form**: the factory pushes the `Buffer*` object itself into
`KernelDescriptor::runtime_args` via `emplace_runtime_args`
(`batch_norm_program_factory.cpp:90,111,112,115`; `running_statistics_program_factory.cpp:88,109,112`).
With no `->address()` expression anywhere there is no site at which a host-side offset *could* be
folded, so all four offset types are excluded by construction:

- **Type 1 / Type 2** — not present. No address arithmetic on the host, and no kernel consumes an
  offset base (each address flows straight into a `TensorAccessor` constructor).
- **Type 3** — not present (no `address_offset`; see Appendix A row above).
- **Type 4** — not present (no `ttnn::narrow`, no explicit-address `MeshBuffer::create`).

`normalization/batch_norm` does not appear in the `2026-07-19_offset_base_pointers.md` triage tables,
and my scan independently confirms clean — *"no fold, op not in the tables"*. Every address RTA is
handed to [TensorParameter analysis](#tensorparameter-analysis) as a clean base.

### TensorAccessor 3rd argument — GREEN

**All 11 `TensorAccessor(...)` construction sites are 2-arg** (`args`, `addr`) — no explicit
page-size third argument anywhere:

`reader_batch_norm.cpp:38` · `writer_batch_norm.cpp:53,57,61,65,69` ·
`reader_running_statistics.cpp:39` · `writer_running_statistics.cpp:52,55,58,61`

The subject cannot fire — there is no override to classify, so no Class 1/2 drop and no Class
3/4/Special gate. `normalization/batch_norm` is correspondingly absent from the
`2026-07-06_tensor_accessor_3rd_arg_triage.md` lookup table, and the syntactic scan above confirms
that absence is genuine rather than stale.

### TensorParameter analysis

11 bindings across the two factories, **all Case 1** — every address flows into a `TensorAccessor`
constructor and all memory access goes through the accessor. No kernel in this op does raw address
arithmetic, so **no `get_bank_base_address` bridge is needed anywhere** and no Case 2 exists.

| Factory | Binding | Bound on | Legacy delivery |
|---|---|---|---|
| `BatchNormFactory` | `input` | reader | `Buffer*`, reader RTA slot 1 |
| | `batch_mean` | writer | `Buffer*`, writer RTA slot 0 |
| | `batch_var` | writer | `Buffer*`, writer RTA slot 1 |
| | `weight` † | writer | `Buffer*`, writer RTA slot 2 — literal `0u` when absent |
| | `bias` † | writer | `Buffer*`, writer RTA slot 3 — literal `0u` when absent |
| | `output` | writer | `Buffer*`, writer RTA slot 4 |
| `RunningStatisticsProgramFactory` | `batch_mean` | reader | `Buffer*`, reader RTA slot 1 |
| | `batch_var` | writer | `Buffer*`, writer RTA slot 0 |
| | `running_mean` †‡ | writer | `Buffer*`, writer RTA slot 1 — literal `0u` when absent |
| | `running_var` †‡ | writer | `Buffer*`, writer RTA slot 2 — literal `0u` when absent |
| | `output` | writer | `Buffer*`, writer RTA slot 3 |

† optional — conditional `TensorBinding` required (see the brief).
‡ **read *and* written in place.** `writer_running_statistics.cpp:87-92,103-110` and `:116-121,132-139`
read each old running statistic and write the updated value back to the *same* tensor. These are
`tensor_args` the framework sees only as inputs; the declared output receives a duplicate of one stat
instead. One `TensorParameter` with one `TensorBinding` serves both directions — no special port
handling — but it is an unusual data-flow shape worth knowing about.

Delivery today is the **`Buffer*` binding form**, which the framework auto-registers as a
`BufferBinding` and patches on cache hits. Per the audit's own guidance this is *correct-on-cache-hit*
and **not** the silent-wrong stale-address hazard — routine port work only. It is also exactly what
`Op Classification = PD Op (pointer-patching)` refers to, and what the port eliminates.

**Op-level roll-up:** `⚠ port work` (11 Case-1 bindings, no Case 2, no clean/borrowed-DFB cases).

### CB endpoints (GATE-free)

Census counted **per CB, per node, per config**, over distinct kernel instances that *touch* the CB
(FIFO produce, FIFO consume, or raw-pointer access). All three kernels of each factory are placed on
`all_device_cores` (a single node set), so every node sees exactly one reader, one writer and one
compute instance — there is **no** dual-instance work-split and **no** same-source multiplicity
anywhere in this op.

**Result: no multi-binding, no dead CBs.** Dispositions are 1P+1C or self-loop.

#### `BatchNormFactory` — 9 CBs (10 when `needs_output_typecast`)

| CB | role | producer | consumer | disposition |
|---|---|---|---|---|
| `c_0` | input | reader | compute | **1P+1C** |
| `c_1` | batch_mean | **writer** | compute | **1P+1C** |
| `c_2` | compute output / FP32 staging | compute | writer *or* compute | **config-flip** — see below |
| `c_3` | batch_var | **writer** | compute | **1P+1C** |
| `c_4` | eps | reader | compute | **1P+1C** |
| `c_5` | weight | writer | compute | **1P+1C** (bind unconditionally — see below) |
| `c_6` | bias | writer | compute | **1P+1C** (bind unconditionally — see below) |
| `c_7` | den = `1/sqrt(var+eps)` | compute | compute | **self-loop** |
| `c_8` | temp_1 | compute | compute | **self-loop** |
| `c_9` | writer-facing output | compute | writer | **1P+1C** — *exists only when* `needs_output_typecast` |

Note the writer kernel is a reader **and** a writer: it fills `c_1`/`c_3`/`c_5`/`c_6` from tensor
memory as well as draining `c_2`/`c_9` to the output tensor. Its producer role on four CBs is easy to
miss from the kernel's name alone.

#### `RunningStatisticsProgramFactory` — 12 CBs (up to 14 with typecast)

| CB | role | producer | consumer | disposition |
|---|---|---|---|---|
| `c_0` | batch_mean | reader | compute | **1P+1C** |
| `c_1` | batch_var | **writer** | compute | **1P+1C** |
| `c_2` | out0 | compute | writer | **1P+1C** (see anomaly A3 — `push_back` with no `reserve_back` in the non-SFPU kernel) |
| `c_3` | old_running_mean | writer | compute | **1P+1C** (bind unconditionally) |
| `c_4` | old_running_var | writer | compute | **1P+1C** (bind unconditionally) |
| `c_5` | momentum | reader | compute | **1P+1C** |
| `c_6` | one | reader (via `fill_cb_with_value`) | compute | **1P+1C** |
| `c_7` | updated_m / FP32 staging | compute | writer *or* compute | **config-flip** |
| `c_8` | updated_v / FP32 staging | compute | writer *or* compute | **config-flip** |
| `c_9` | tmp1 | compute | compute | **self-loop** |
| `c_10` | tmp2 | compute | compute | **self-loop** |
| `c_11` | tmp3 | compute | compute | **self-loop** |
| `c_12` | writer-facing updated_m | compute | writer | **1P+1C** — *only when* `needs_mean_typecast` |
| `c_13` | writer-facing updated_v | compute | writer | **1P+1C** — *only when* `needs_var_typecast` |

#### The three config-flipping CBs (classify per instantiation)

`c_2` (batch_norm) and `c_7`/`c_8` (running_statistics) are the **typecast staging** buffers, and
their disposition flips:

| Config | Data flow | Disposition |
|---|---|---|
| typecast **off** (`writer_*_cb == staging cb`) | compute packs → writer drains | **1P+1C** |
| typecast **on** (`writer_*_cb` = `c_9` / `c_12` / `c_13`) | compute packs → **compute** re-reads to typecast → writes the writer-facing CB | **self-loop** on the staging CB |

The typecast path is reachable **only through the SFPU compute source**: `needs_output_typecast` /
`stat_format_needs_typecast` both require `interm_data_format == Float32`, which requires
`any_float32`, which forces the `*_sfpu_kernel.cpp` selection. Confirmed from the other side too —
the non-SFPU kernels never read the typecast CTAs (see anomaly A2).

#### Why the optional-tensor CBs are *not* dead CBs

`c_5`/`c_6` (weight/bias) and `c_3`/`c_4` (old_running_mean/var) are allocated by the host
**unconditionally**, but only *accessed* at runtime when the corresponding optional tensor is
present. They are nonetheless **referenced in compiled code in every config** — the kernels construct
`DataflowBuffer` objects for them outside the conditional (e.g. `writer_batch_norm.cpp:49-50`,
`running_statistics_sfpu_kernel.cpp:72-73`), and the `batch_norm` compute kernels gate on a
**runtime** `if` (not `if constexpr`), so both branches are compiled.

The dead-CB test — *"the `buffer_index` is unreferenced by every kernel in every config"* — therefore
**fails**, and these must **not** be dropped. Dropping them would also change the op's L1 footprint
relative to legacy, i.e. a functional change the port is forbidden from making. Disposition:
ordinary **1P+1C with unconditional bindings**. This is called out prominently in the brief because
the natural instinct ("the tensor is absent, so skip the DFB") is wrong here in two independent ways.

### Out-of-directory coupling

**Function-call escape — roll-up: ✓ clean.** Three donor headers, all with crossing-friendly
signature shapes.

| Op kernel(s) | Donor file | Class | Functions called | Shape | Status |
|---|---|---|---|---|---|
| all 4 dataflow kernels | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp` | 6 — cross-family donor | `fill_with_val<>`, `fill_with_val_bfloat16`, `fill_tile_with_first_element<>`, `fill_tile_with_first_element_bfloat16` | `uint32_t l1_write_ptr` (raw L1 address) | ✓ — fed by a member `dfb.get_write_ptr()` peek; no CB id, no `sem::`/`tensor::` handle |
| `reader_running_statistics.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/cb_fill_helpers.hpp` | 3 — shared kernel pool | `fill_cb_with_value` | `uint32_t cb_id` | ✓ — `dfb::name`'s constexpr cast covers it |
| all 4 compute kernels | `ttnn/cpp/ttnn/kernel/compute/dest_format_helpers.hpp` | 3 — shared kernel pool | `pack_tile_with_dt`, `copy_tile_init_with_dt`, `{add,sub,mul}_tiles_init_with_dt`, `{mul,add,sub}_tiles_to_cb` | `uint32_t icb` | ✓ — same constexpr cast |

No donor takes a `Semaphore`, a `uint32_t sem_id`, a `TensorAccessorArgs<N>`, a tensor CTA offset as
an NTTP, a `CircularBuffer&`, or a legacy addr-gen type. **The `sem::` / `tensor::` boundary
assumption holds** — no out-of-op call site requires either handle.

**Borrowed kernel files (file-path instantiation): none.** All eight kernel sources live in this op's
own `device/kernels/` tree.

**Lent kernels: none.** `grep -rl` per kernel filename across `ttnn/cpp/ttnn/operations/` returns
only this op's own factories. The four compute sources return *no* hits at all because their paths are
assembled with `fmt::format` (`batch_norm_program_factory.cpp:388`,
`running_statistics_program_factory.cpp:438`) — I confirmed by reading the format strings, not by
trusting the grep. **No `_metal2` fork exists anywhere relevant, and none is needed** — this op has no
shared-kernel Caution case in either direction, intra-op included (the two device-ops share no kernel,
so porting both in one PR introduces no shared-kernel coupling either).

### RTA varargs — none

Every runtime arg in all eight kernels is read **once, at a source-literal index**, as a distinct
field (`get_arg_val<uint32_t>(0)` … `(11)`). No counted loop over RTAs, no `arg_index++` run, no
data-selected index, no sentinel-terminated scan. All RTAs port to **named** runtime args; the vararg
mechanism is not needed anywhere in this op.

---

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** — 11 bindings, **all Case 1**; full table in
  [TensorParameter analysis](#tensorparameter-analysis). Four are optional and need conditional
  `TensorBinding`s; two (`running_mean` / `running_var`) are read-and-written in place.
- **TensorParameter relaxation:** none (matches the sheet). Keep strict `TensorSpec` matching.
- **TensorAccessor 3rd arg:** none — all sites already 2-arg.
- **CB endpoints:** self-loop `c_7`/`c_8` (batch_norm) and `c_9`/`c_10`/`c_11` (running_statistics);
  assign 1P+1C everywhere else; **config-flip to self-loop** on `c_2` (batch_norm) and `c_7`/`c_8`
  (running_statistics) when the typecast path is selected. No multi-binding flag anywhere; no
  dead-CB drop.
- **Compute `opt_level`:** `grep -n opt_level` over both factories returns **nothing** — the field is
  absent from every `KernelDescriptor`. Absent on a `ComputeConfigDescriptor` resolves to **`O3`**,
  so **each** ported compute `KernelSpec` (one per factory) needs
  `compiler_options.opt_level = KernelBuildOptLevel::O3` set explicitly. The DM kernels' absent field
  resolves to `O2`, which is Metal 2.0's default — nothing to do there.
- **`unpack_modes`:** both factories build a computed `std::vector<UnpackToDestMode>` indexed by CB id
  (`batch_norm_program_factory.cpp:352-368` — 8 CBs, +1 conditional;
  `running_statistics_program_factory.cpp:394-411` — 12 CBs) — `UnpackToDestFp32` on the listed set
  when `fp32_dest_acc_en`, `Default` elsewhere. Requires reindex-to-DFB-name **and** value
  translation, plus the newly-required explicit entries for consumed Float32 DFBs. Per-factory detail
  in the brief.
- **Device-op-class edits:** one per factory — retarget `create_descriptor` →
  `create_program_artifacts` in each device-op header, and swap the `program_descriptors.hpp` include
  for `ttnn/metal_v2_artifacts.hpp`. **No custom-hash deletion** (neither op has a
  `compute_program_hash`; `to_hash()` stays). **No pybind removal** (no factory entry point is
  exposed).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No hidden second writer and no multi-reader
  shape exists in this op — every CB has at most one producing and one consuming kernel per node. The
  hidden-co-filler hunt was run against all four dataflow and all four compute kernels and came back
  empty (no raw `get_write_ptr` write by a non-producer, no semaphore-gated co-fill — the op has no
  semaphores).
- **Runtime-selected compute source (×2 per factory):** each port unit is *the factory plus **both***
  of its compute sources — 4 files per factory, 8 across the PR. Sized in the brief.
- **Unity build:** both factories' `.cpp` files land in the same translation unit and both already use
  `namespace { namespace CMAKE_UNIQUE_NAMESPACE { … } }`. With both ported in one PR, keep the new
  spec-name constants **function-local** — this is the one collision risk the single-PR scoping adds.
- **Cross-op / shared kernels:** none in either direction; no `_metal2` fork exists or is needed.
- **RTA varargs:** none — name every runtime arg.
- **Perf comparison is worth extra attention** on this op — see the `Pointer patching perf issue?`
  note in the cross-check.

## Team-only

- **Out-of-directory coupling & donor shape:** full inventory above. All three donors are
  `uint32_t cb_id` or raw-L1-address shapes; no scheduling blocker (no ⭐ entries), no donor is
  pre-Device-2.0.
- **Relaxation candidates (FALLIBLE — candidates to verify):** none mined. `to_hash()`'s omission of
  the `input_dtype` field is **not** a relaxation candidate — see the `to_hash()` note in the
  cross-check for why it is inert.
- **Readiness-sheet data quality:** the `Backdoor custom hash` cell for the `RunningStatistics` row
  reads `yes` but that device-op has no `to_hash()` / `attribute_values`. Non-gating; likely a
  copy-down from the `BatchNormOperation` row (the two rows are otherwise byte-identical).
- **TTNN factory analysis:** both ops are plain single-program `descriptor` factories with no
  op-owned tensors, no `MeshWorkload`, no custom `compute_program_hash`, no
  `get_dynamic_runtime_args`, no `override_runtime_arguments`, and no pybound internals. Target
  concept `ProgramSpecFactoryConcept` for both, matching the sheet's `Porting Target`.
  `select_program_factory` is absent from both device-ops — correct, since each `program_factory_t`
  has a single alternative and the framework returns it automatically.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

- **A1 — Dead RTA slots (all four dataflow kernels, plus the RunningStatistics compute kernels).**
  Each factory emits more runtime args than its kernels read.
  - `batch_norm_program_factory.cpp:87-99` emits 11 reader args; `reader_batch_norm.cpp` reads
    indices 0–8 → slots 9, 10 (`cHt`, `cWt`) dead.
  - `batch_norm_program_factory.cpp:109-124` emits 14 writer args; `writer_batch_norm.cpp` reads
    0–11 → slots 12, 13 dead.
  - `running_statistics_program_factory.cpp:85-97` emits 11 reader args; reads 0–8 → 9, 10 dead.
  - `running_statistics_program_factory.cpp:107-121` emits 13 writer args; reads 0–10 → 11, 12 dead.
  - `running_statistics_program_factory.cpp:126` emits 3 compute args
    (`num_tiles_per_core`, `freq`, `counter`); **both** RunningStatistics compute kernels read only
    index 0 (`running_statistics_kernel.cpp:12`, `running_statistics_sfpu_kernel.cpp:45`) → slots 1, 2
    dead. (The `batch_norm` compute kernels read all three.)
  - The zero-fill path for idle cores hardcodes the same inflated counts
    (`num_reader_args = 11`, `num_writer_args = 14`/`13`, `num_kernel_args = 3`).
- **A2 — Dead CTA slots in the non-SFPU compute kernels.** The shared host CTA list is built for the
  SFPU kernel's superset. `batch_norm_kernel.cpp` reads CTAs 0–10 of 15 (11–14 —
  `writer_output_cb`, `needs_output_typecast`, `tc_in_fmt`, `tc_out_fmt` — unread);
  `running_statistics_kernel.cpp` reads 0–13 of 19 (14–18 unread). Harmless, but a porter should
  **not** "clean these up": the sibling SFPU source, selected from the same factory, does read them.
- **A3 — `push_back` with no matching `reserve_back` on `c_2` (out0).**
  `running_statistics_kernel.cpp:57-59` packs into `dfb_out0` and calls `dfb_out0_obj.push_back(1)`,
  but nothing ever calls `dfb_out0_obj.reserve_back(...)` in that kernel — the producer posts credits
  without reserving space in a 2-entry FIFO. The SFPU sibling *does* reserve
  (`running_statistics_sfpu_kernel.cpp:95`), so this is an asymmetry between the two sources rather
  than a deliberate idiom. Routed to the ops team: it is a legacy FIFO-protocol defect, and the port
  will carry it forward unchanged (the recipe forbids adding a `reserve`/`pop` to "balance" a FIFO).
- **A4 — Duplicated `extract_shape_dims` and `populate_runtime_arguments`.** The two factories carry
  near-identical private copies (`batch_norm_program_factory.cpp:19-134` vs
  `running_statistics_program_factory.cpp:19-131`), both inside
  `namespace { namespace CMAKE_UNIQUE_NAMESPACE { … } }`. Not port work; noted for the ops team. With
  both factories converted in one PR the duplication will be visible in a single diff — resist
  de-duplicating it there, and route any proposal to the port report.
- **A5 — `TensorAccessorArgs(nullptr)` for absent optionals.** `batch_norm_program_factory.cpp:322,324`
  and `running_statistics_program_factory.cpp:366,368` build accessor args from a null `Buffer*` so
  the CTA layout stays fixed, and the kernels then construct a `TensorAccessor` over address `0`
  unconditionally (`writer_batch_norm.cpp:65,69`; `writer_running_statistics.cpp:58,61`) while gating
  only the *uses*. It works today but is a latent footgun; the Metal 2.0 port removes the idiom
  entirely (conditional `TensorBinding` + `#ifdef`), so no ops-team action is needed unless the legacy
  path is expected to live on.
- **A6 — `RunningStatistics` mutates its inputs in place.** `writer_running_statistics.cpp:103-110,132-139`
  writes the updated statistics back into the `running_mean` / `running_var` **input** tensors, while
  the declared output receives a duplicate of one stat
  (`running_statistics_sfpu_kernel.cpp:179-180,278-279`). Legal and ported as-is, but it means the op's
  true output set is wider than its declared one — relevant to anyone reasoning about aliasing or
  caching. Noted for the ops team; no port action.

## Per-DeviceOperation attribution

| Field | `BatchNormOperation` / `BatchNormFactory` | `RunningStatistics` / `RunningStatisticsProgramFactory` |
|---|---|---|
| **Overall** | **✅ GREEN — brief issued** | **✅ GREEN — brief issued** |
| Device 2.0 | Yes (4 kernels) | Yes (4 kernels) |
| Feature support | GREEN (all N/A) | GREEN (all N/A) |
| `Is able to port?` | yes (cross-check clean) | yes (cross-check clean) |
| `Is safe to port?` | yes | yes |
| Custom hash (`compute_program_hash`) | none | none |
| Backdoor custom hash | **yes** — `to_hash()` @ `batch_norm_device_operation.cpp:121`; port keeps it | **no in code** (sheet says `yes` — non-gating discrepancy) |
| Offset base pointers | none | none |
| TensorAccessor 3rd arg | none (6 sites, all 2-arg) | none (5 sites, all 2-arg) |
| Tensor bindings | 6, all Case 1 (2 optional) | 5, all Case 1 (2 optional, 2 written in place) |
| CBs | 9 (+1 typecast) — 2 self-loop, 1 config-flip | 12 (+2 typecast) — 3 self-loop, 2 config-flip |
| Compute sources | 2 (`batch_norm_kernel`, `batch_norm_sfpu_kernel`) | 2 (`running_statistics_kernel`, `running_statistics_sfpu_kernel`) |
| Semaphores | none | none |
| Op-owned tensors | none | none |
| Device-op-class edits forced | 1 (entry-point retarget + include swap) | 1 (entry-point retarget + include swap) |

## Questions for the user

1. **Does the port own retiring `BatchNormOperation::to_hash()`?** *(non-blocking)* The sheet now
   tracks it as a `Backdoor custom hash` and flags `Formerly custom hashed? = yes (to_hash, still
   present)`, which reads like residual debt someone intends to clear. Under the current recipe the
   answer is **no**: the sanctioned deletion is scoped to `compute_program_hash`, the
   `Custom hash` cell is `no`, and `to_hash()` is inert today (it omits only `input_dtype`, which
   always mirrors the hashed input tensor's dtype). So the brief tells the porter to leave it. If the
   owner wants it removed, that is a separate PR — please confirm, because the port must not touch it
   otherwise.
2. **Is a perf comparison in scope for this PR?** `Pointer patching perf issue? = suspect perf
   regression (+ fixed latent bug)` attaches to the exact mechanism this port removes. A before/after
   measurement would either confirm the port as the fix or rule pointer-patching out — but it is
   beyond the "no behavior change" verification the recipe asks for. Say the word if you want the
   porter to gather numbers rather than just green tests.

## Recipe notes

1. **The audit's cross-check list needs to grow as the sheet gains columns.** The revised sheet carries
   several columns the recipe's cross-check does not mention — `Op Classification`, `Execution Model`,
   `Porting Target`, `Backdoor custom hash (attribute_values / to_hash)`, `Smuggled pointer`,
   `Pointer patching perf issue?`, `Formerly custom hashed?`. Two of these are directly checkable
   against code (`Backdoor custom hash`, `Smuggled pointer`) and one restates a decision the audit
   otherwise derives (`Porting Target`). The recipe's standing rule ("reference every column by header
   name; no column is ever deleted") anticipates growth, but the *cross-check* section enumerates a
   fixed list, so a new checkable column has no defined handling. I checked the two new checkable ones
   opportunistically and found a mismatch on one. Suggest either naming them or stating a general rule:
   *verify any column whose claim is decidable from code; gate only on the derivation's conjuncts.*
2. **State explicitly that a mismatch on a non-conjunct column does not gate.** Related to note 1:
   the recipe's spreadsheet-broken GATE is written for "cross-check conflicts with the sheet", with no
   qualifier. Read literally, my `Backdoor custom hash` mismatch would RED a fully portable op. I
   reasoned from the derivation (the column is not a conjunct, and the sheet's own `able = yes` proves
   it) plus the "too-conservative RED misroutes work" warning — but that reasoning had to be
   constructed rather than looked up. One sentence scoping the gate to the conjuncts and the
   factory-set match would remove the ambiguity.
3. **`to_hash()` / `attribute_values` deserves a false-positive guard in the `Custom hash`
   cross-check.** The recipe says to "grep the device-op for a `compute_program_hash` override". A
   `to_hash()` member is a different mechanism (the ttsl attribute-hash protocol feeding the *default*
   reflection hash), but it greps and reads like a custom hash, and misclassifying it is severe in both
   directions — a spurious gate, or a port that deletes a load-bearing attribute hash under the
   custom-hash exception. The sheet now has a dedicated column for it, which makes the distinction
   more visible but also more likely to be conflated with the gate conjunct next to it.
4. **The "isolated CB-index holdover" test does not cleanly classify library *orchestrators*.** The
   Device 2.0 gate's two-part test (wrapper in scope ∧ wrapper-method replacement exists) is written
   for CB *accessors* like `get_read_ptr(cb_id)`. `fill_cb_with_value(cb_id, value)` is a shared-pool
   helper that internally reserves, fills and pushes — no wrapper method could replace it, so it
   passes the test by failing both conjuncts, which feels like the right answer for the wrong reason.
   A clarifying clause ("a shared-pool orchestrator taking a CB id is not a holdover; it is a donor
   call, classified under Out-of-directory coupling") would make this call reproducible rather than
   judgment-dependent.
5. **CB-endpoint census needs a rule for "referenced in compiled code but never touched at
   runtime".** The dead-CB test keys on whether a `buffer_index` is *referenced* by any kernel, while
   the endpoint census keys on whether a kernel *touches* the CB. This op has four CBs that are
   referenced-but-untouched in one config (unconditionally-constructed `DataflowBuffer` objects for
   absent optional tensors). Those two tests point opposite ways, and the difference decides between
   "drop the CB" and "bind it unconditionally" — i.e. between a functional change and a faithful
   port. I resolved it via the dead-CB rule's "in every config" wording, but the census section could
   say directly that an unconditional `DataflowBuffer` construction is a binding requirement even
   without an access.
