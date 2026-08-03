# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/normalization/batch_norm`

Two **independent** device-operations live under this op directory. They share **no** kernels and **no**
factories with each other, but they are sequenced by a single user-facing entry point
(`ttnn::batch_norm`, `batch_norm.cpp:127` and `:132`), so they are bundled into one report at directory
scope. **Per-DeviceOperation attribution is retained throughout**; findings are near-identical but the
CB inventories and conditional-binding sets differ.

- **`BatchNormOperation`** — `device/batch_norm_device_operation.hpp:14`
  - `BatchNormFactory` (`device/batch_norm_program_factory.cpp:140`) — sole member of `program_factory_t`
  - kernels (4, all runtime- or config-selected together): `kernels/dataflow/reader_batch_norm.cpp`,
    `kernels/dataflow/writer_batch_norm.cpp`, and **one of** `kernels/compute/batch_norm_kernel.cpp` /
    `kernels/compute/batch_norm_sfpu_kernel.cpp` (selected at runtime — see Runtime kernel-source selection)
- **`RunningStatistics`** — `device/running_statistics_device_operation.hpp:14`
  - `RunningStatisticsProgramFactory` (`device/running_statistics_program_factory.cpp:137`) — sole member of `program_factory_t`
  - kernels (4): `kernels/dataflow/reader_running_statistics.cpp`,
    `kernels/dataflow/writer_running_statistics.cpp`, and **one of**
    `kernels/compute/running_statistics_kernel.cpp` / `kernels/compute/running_statistics_sfpu_kernel.cpp`

**Unreferenced kernel files:** none — all eight kernels are instantiated by their device-op's factory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `a38e7b405db 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

**Audited at:** branch `virdhatchani/BN_Porting`, HEAD `a38e7b405db`, merge-base with `main` `f6e166da2c1` (2026-08-03).

> ### Readiness-sheet provenance
>
> The claude.ai Google Drive connector is **not authorized in this session**, so the *"Operations analysis"*
> sheet could not be fetched programmatically. **The user supplied both rows directly**; they are transcribed
> verbatim in the *TTNN factory concept* gate section below. Both read **`Is able to port? = yes`**, and every
> cheaply-checkable column matches the `file:line` evidence this audit derived independently. The factory set
> matches one-to-one (2 sheet rows ↔ 2 DeviceOperations, one factory each). One **non-gate-conjunct** column
> (`Backdoor custom hash` / `Formerly custom hashed?`) does **not** match for the `RunningStatistics` row — see
> Question 1; it does not affect the verdict.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/normalization/batch_norm` |
| **Overall** | **GREEN** — both device-operations clear every gate; brief issued for both |
| **DOps / Factories** | `BatchNormOperation` → `BatchNormFactory` · `RunningStatistics` → `RunningStatisticsProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 8 kernels structurally Device 2.0; no holdovers, no sanctioned-free-function reliance |
| *Prereqs* — Cross-op escapes | Ok — 3 donor headers, all ✓ shapes (raw-address and `uint32_t cb_id`); no borrowed *kernel files* |
| *Feature Support* — overall | **GREEN** — all Appendix A entries N/A |
| *Feature Support* — Variadic-CTA | Ok — every CTA read is at a `constexpr` offset |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both) — confirmed: `create_descriptor` returning `ProgramDescriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — neither is a `WorkloadDescriptor` op |
| *TTNN Readiness* — Is safe to port? | Yes (both) — sheet-owner judgment, not re-derived |
| *TTNN Readiness* — Custom hash | **No** (both) — confirmed: no `compute_program_hash` anywhere in the directory |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (both) — confirmed absent |
| *TTNN Readiness* — `override_runtime_arguments` | No (both) — confirmed absent |
| *TTNN Readiness* — Pybind `create_descriptor` | No (both) — confirmed: `batch_norm_nanobind.cpp` binds only the user-facing op |
| *TTNN Readiness* — Op-owned tensors | No (both) — neither factory allocates a device tensor beyond its io |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (both), no op-owned tensors |
| *Port work* — Offset base pointer | none — every address arg is a clean `Buffer*` base or literal `0u`; no host-side fold |
| *Port work* — Tensor bindings (per binding) | 11 bindings total (6 BatchNorm + 5 RunningStatistics), **all Case 1** |
| *Port work* — TensorParameter relaxation | none (sheet) — confirmed: no `ArgConfig::Runtime*` anywhere |
| *Port work* — TensorAccessor 3rd arg | none — every `TensorAccessor` is the 2-arg form |
| *Port work* — CB endpoints | 1P+1C / self-loop only; **no multi-binding flag, no dead CB** — 4 CBs flip disposition with config |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves as a **self-loop** (one
toucher) or a **1P+1C assignment** (two touchers). No CB anywhere in either factory reaches ≥3 distinct
touchers or doubles a FIFO role, so the multi-binding advanced option is **not** needed. Four CBs change
disposition with config and are recorded per `(CB, config)` below.

## Result

**GREEN → brief issued for both device-operations.** No gate fired for either. The port targets
`ProgramSpecFactoryConcept` for both factories, with no sanctioned device-op-class edits required (no custom
`compute_program_hash`, no pybound `create_descriptor`, no pybind-hook-only factory parameter).

The two device-operations are **independent porting units** that happen to share a directory and a public
entry point. They can be ported in one change (as requested) or separately; nothing couples them
structurally. Bundling them is the better choice here because `test_batch_norm_program_cache.py` — the
highest-value no-regression test — exercises both in a single `ttnn.batch_norm(training=True)` call and
cannot isolate either.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** for both rows. Sheet cells transcribed verbatim
  from the user-supplied rows:

  | Column | `BatchNormOperation` / `BatchNormFactory` | `RunningStatistics` / `RunningStatisticsProgramFactory` |
  |---|---|---|
  | `Concept` | `descriptor` | `descriptor` |
  | `Op Classification` | PD Op (pointer-patching) | PD Op (pointer-patching) |
  | `Execution Model` | SPMD | SPMD |
  | `Porting Target` | `ProgramSpecFactoryConcept` | `ProgramSpecFactoryConcept` |
  | `Custom hash (compute_program_hash)` | `no` | `no` |
  | `Backdoor custom hash (attribute_values / to_hash)` | `yes` | `yes` |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | `no` |
  | `Override runtime args method? (PD only)` | `no` | `no` |
  | `Pybind descriptor (nb::class_ of device op)` | `no` | `no` |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` | `no` |
  | `Known op issues` | *(blank)* | *(blank)* |
  | `Is safe to port?` | `yes` | `yes` |
  | `Is able to port?` | **`yes`** | **`yes`** |
  | `TensorParameter relaxation` | `none` | `none` |
  | `Op-owned tensors?` | *(blank)* | *(blank)* |
  | `Secretly SPMD Workload?` | *(blank)* | *(blank)* |
  | `Pointer patching perf issue?` | suspect perf regression (+ fixed latent bug) | suspect perf regression (+ fixed latent bug) |
  | `Formerly custom hashed?` | `yes (to_hash, still present)` | `yes (to_hash, still present)` |

  Cross-check against the code — every gate-relevant column confirmed:

  | Column | Sheet | Code evidence | Match |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` → `tt::tt_metal::ProgramDescriptor` at `batch_norm_device_operation.hpp:39`, `running_statistics_device_operation.hpp:36` | ✓ |
  | `Custom hash` | `no` | `grep compute_program_hash` over the directory: zero hits | ✓ |
  | `get_dynamic_runtime_args` | `no` | zero hits | ✓ |
  | `override_runtime_arguments` | `no` | zero hits | ✓ |
  | `Pybind descriptor` | `no` | `batch_norm_nanobind.cpp` binds only `ttnn::batch_norm`; no `nb::class_` of a device op, no `create_descriptor` | ✓ |
  | `Secretly SPMD Workload?` | *(blank)* | N/A — no `WorkloadDescriptor` / `create_workload_descriptor` anywhere | ✓ |
  | `Op-owned tensors?` | *(blank)* | neither factory allocates a device tensor; both take io only | ✓ |
  | Factory-set match | 2 rows | 2 DeviceOperations × 1 factory each; no phantom or missing row | ✓ |

  `Is safe to port?` was **not** re-derived (expert-judgment axis, per the recipe). Cross-column invariants
  hold: neither row is `legacy device-op`, and neither claims op-owned tensors on a `descriptor` concept.

  The one non-matching column — `Backdoor custom hash` / `Formerly custom hashed?` = `yes` on the
  **`RunningStatistics`** row — is not a conjunct of `Is able to port?` and is not in the recipe's
  cheaply-checkable cross-check list, so it does not trigger the spreadsheet-broken gate. Recorded as
  Question 1 for the sheet owner.

- **Device 2.0 (every kernel used):** **GREEN.** All eight kernels are structurally Device 2.0: `Noc` for
  every transfer (`noc.async_read` / `async_write` with DFB endpoints and `{.page_id = …}` /
  `{.offset_bytes = …}` args), `DataflowBuffer` wrapper objects for every buffer, `TensorAccessor` for every
  tensor walk, and wrapper **methods** for every metadata read (`dfb.get_entry_size()`). Notably the kernels
  already use `DataflowBuffer` rather than `CircularBuffer` — the object-type swap of kernel-side whitelist
  rule 1 is already done; only the *id source* changes in the port.

  Scans returning **zero** hits across `device/kernels/`: `get_tile_size(`, `get_write_ptr(<id>`,
  `get_read_ptr(<id>`, `get_dataformat(`, `get_local_cb_interface`, `cb_reserve_back` / `cb_push_back` /
  `cb_wait_front` / `cb_pop_front`, `get_pointer_to_cb_data`, `get_cb_tiles_acked_ptr` /
  `get_cb_tiles_received_ptr`, `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen`, bare
  `noc_async_read(` / `noc_async_write(`, `noc_semaphore*`, `get_noc_addr(`, `CircularBuffer`.

  No violation table — there are no violations. The op does not even rely on the two *sanctioned*
  CB-index free functions (`get_tile_size(cb_id)`, `get_local_cb_interface(cb_id)`); all metadata comes off
  the object already.

  One donor detail, called out because it looks like a holdover and is not:
  `reader_running_statistics.cpp:56` calls `fill_cb_with_value(dfb_id_one, one_u)` — a free function taking
  a raw `uint32_t` CB id, from the shared pool at `ttnn/cpp/ttnn/kernel/dataflow/cb_fill_helpers.hpp`. It is
  **not** a Device 2.0 holdover: it is a TTNN kernel helper, not a Device 2.0 API; it has no
  `DataflowBuffer` member-method equivalent; and no wrapper object for `one_cb` is in scope at the call site
  (the reader never constructs one). Its body is itself Device 2.0-clean (`CircularBuffer` wrapper +
  `CoreLocalMem` + wrapper methods). Its `uint32_t cb_id` parameter is the ✓ donor shape — `dfb::one`
  crosses on the `DFBAccessor` constexpr conversion. See *Out-of-directory coupling* for the Quasar-facing
  note.

- **Feature compatibility:** all four Appendix A entries **N/A** — a clean scan.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | zero hits for `GlobalCircularBuffer`, `CreateGlobalCircularBuffer`, `global_circular_buffer`, `remote_cb*`, `remote_index`, and no `CBDescriptor.global_circular_buffer` field set |
  | CBDescriptor `address_offset` (non-zero) | N/A | zero hits for `address_offset`, `set_address_offset`, `UpdateDynamicCircularBufferAddress`; no borrowed-memory CBs at all (`set_globally_allocated_address`: zero hits) |
  | GlobalSemaphore | N/A | zero hits; neither factory declares **any** semaphore (no `SemaphoreDescriptor`) |
  | Variable-count compile-time arguments (CTA varargs) | N/A | neither `tensor_args_t` carries a variable-count container (both are fixed named-tensor structs); every kernel `get_compile_time_arg_val` index is a literal or a `constexpr` `next_compile_time_args_offset()` expression — fixed-count, explicitly guarded against by the entry's false-positive rule |

- **CB endpoints (GATE-free):** every CB in both factories resolves to **1P+1C** or a **self-loop**; none
  needs the multi-binding advanced option, and none is dead. Full per-`(CB, config)` census in
  *Port-work summary* below. The hidden-second-writer hunt (face (a)) was run over all eight kernels: there
  is **no** semaphore-gated raw co-fill anywhere (the op declares no semaphores at all), and no CB is touched
  by more than two distinct kernel instances on a node. The dual-instance work-split shape (face (c)) does
  not occur — each factory instantiates each kernel source exactly once, over one `core_ranges`.

- **Offset base pointers:** **GREEN.** Both factories deliver tensor addresses by pushing a `Buffer*`
  (not `->address()`) into `KernelDescriptor::runtime_args` — `batch_norm_program_factory.cpp:90, 111-115`
  and `running_statistics_program_factory.cpp:88, 109-112`. There is no `->address()` expression anywhere in
  the directory, hence no host-side arithmetic to fold an offset into. The only non-`Buffer*` value in an
  address slot is the literal `0u` placeholder for an absent optional tensor
  (`batch_norm_program_factory.cpp:101-108`, `running_statistics_program_factory.cpp:99-106`) — an absent
  binding, not an offset. Type 3 (`address_offset`) and Type 4 (`narrow`) do not appear.

- **TensorAccessor 3rd argument:** **GREEN — N/A.** Every `TensorAccessor` construction in the directory is
  the 2-arg form `TensorAccessor(args, addr)` (7 sites across the four dataflow kernels). No site passes an
  explicit page size, so the subject does not fire and no class assignment is needed.

## Port-work summary  *(mirrors the brief)*

### Tensor bindings — 11 bindings, all Case 1

Every tensor address reaches its kernel through a `TensorAccessor`, so all eleven are **Case 1** (the
mechanical, low-risk case): declare a `TensorParameter`, bind it, and the kernel collapses to
`TensorAccessor(tensor::<name>)`. **No Case 2 anywhere** — no kernel does raw base-pointer arithmetic, so
the `get_bank_base_address` bridge is not needed, and the compute-kernel Case-2 block cannot arise (the
compute kernels construct no `TensorAccessor` at all).

All eleven arrive in the **`Buffer*`-binding form** (the factory pushes the `Buffer*` object, not
`->address()`). Per the recipe this is *correct-on-cache-hit today* — the framework auto-registers these as
`BufferBinding`s and patches them — so it is **not** the silent-wrong hazard; it is routine port work. The
sheet's `Smuggled pointer = no` is consistent. The sheet's `Pointer patching perf issue? = suspect perf
regression` refers to exactly this mechanism; the Metal 2.0 typed binding replaces it.

**`BatchNormOperation`** — 6 bindings:

| Binding | Origin | Kernel + accessor site | Case |
|---|---|---|---|
| input | `tensor_args.input` | `reader_batch_norm.cpp:38` | 1 |
| batch_mean | `tensor_args.batch_mean` | `writer_batch_norm.cpp:53` | 1 |
| batch_var | `tensor_args.batch_var` | `writer_batch_norm.cpp:61` | 1 |
| weight *(optional)* | `tensor_args.weight` | `writer_batch_norm.cpp:65` | 1 |
| bias *(optional)* | `tensor_args.bias` | `writer_batch_norm.cpp:69` | 1 |
| output | `tensor_return_value` | `writer_batch_norm.cpp:57` | 1 |

**`RunningStatistics`** — 5 bindings:

| Binding | Origin | Kernel + accessor site | Case |
|---|---|---|---|
| batch_mean | `tensor_args.batch_mean` | `reader_running_statistics.cpp:39` | 1 |
| batch_var | `tensor_args.batch_var` | `writer_running_statistics.cpp:52` | 1 |
| running_mean *(optional, in-place RW)* | `tensor_args.running_mean` | `writer_running_statistics.cpp:58` | 1 |
| running_var *(optional, in-place RW)* | `tensor_args.running_var` | `writer_running_statistics.cpp:61` | 1 |
| output | `tensor_return_value` | `writer_running_statistics.cpp:55` | 1 |

`running_mean` / `running_var` are **read and written through the same accessor** — the writer reads the old
value (`writer_running_statistics.cpp:87`) and writes the update back to the same pages
(`:103`). One `TensorParameter` per stat covers both directions; nothing special is required, but the
porter should not be surprised to see a "writer" reading, nor mistake the in-place stat for the op's
`tensor_return_value` (the created output tensor is separate, and its value is discarded by the caller —
`batch_norm.cpp:131`).

### Conditional (optional) tensor bindings — 4, all needing a preprocessor gate

Four of the eleven bindings are absent in some configurations, and in **every** case the kernel constructs
the `TensorAccessor` **unconditionally** and gates only its *uses*. Metal 2.0 emits `tensor::<name>` only
when the host actually binds it, so the construction must move behind an `#ifdef` — the mandatory-gate case
for optional tensors.

| Binding | Absent when | Unconditional construction to gate | Gate condition already a CTA |
|---|---|---|---|
| BatchNorm `weight` | `!weight.has_value()` | `writer_batch_norm.cpp:64-65` | CTA 0 (`weight_has_value`) |
| BatchNorm `bias` | `!bias.has_value()` | `writer_batch_norm.cpp:68-69` | CTA 1 (`bias_has_value`) |
| RS `running_mean` | `!running_mean.has_value()` | `writer_running_statistics.cpp:57-58` | CTA 0 (`old_running_mean_has_value`) |
| RS `running_var` | `!running_var.has_value()` | `writer_running_statistics.cpp:60-61` | CTA 1 (`old_running_var_has_value`) |

Each is a **promote-a-CTA-gate-to-a-define** case: the condition exists today as a CTA driving
`if constexpr`, and must become a `KernelSpec::compiler_options.defines` entry so the preprocessor removes
the `tensor::` reference. The promotion is needed **only on the two writer kernels** — the compute kernels
gate on the same conditions but never reference a `tensor::` token, so their copies stay named CTAs.

RS additionally guarantees at least one stat is present (`running_statistics_device_operation.cpp:42-44`,
mirrored by a `static_assert` in both compute kernels), so the port never faces both absent.

### Conditional DFB specs — 3, path-dependent handle aliases

Three CBs are allocated only in the typecast configuration, and in each case a *second* CTA name resolves
to either the conditional CB or the unconditional staging CB. That is the **same-FIFO aliasing,
path-dependent variant** — one `#ifdef`-gated `constexpr` alias, not a second binding.

| Conditional CB | Exists when | Aliased name in kernel | Alias resolves to |
|---|---|---|---|
| BatchNorm `c_9` (writer-facing output) | `needs_output_typecast` (`batch_norm_program_factory.cpp:227-239`) | `dfb_output_final`, CTA 11 (`batch_norm_sfpu_kernel.cpp:219`) | `c_9` when typecast, else `c_2` |
| RS `c_12` (writer-facing updated mean) | `needs_mean_typecast` (`running_statistics_program_factory.cpp:285-297`) | `dfb_writer_updated_mean`, CTA 14 (`running_statistics_sfpu_kernel.cpp:64`) | `c_12` when typecast, else `c_7` |
| RS `c_13` (writer-facing updated var) | `needs_var_typecast` (`running_statistics_program_factory.cpp:298-310`) | `dfb_writer_updated_var`, CTA 15 (`running_statistics_sfpu_kernel.cpp:65`) | `c_13` when typecast, else `c_8` |

Recommended shape: emit `NEEDS_OUTPUT_TYPECAST` / `NEEDS_MEAN_TYPECAST` / `NEEDS_VAR_TYPECAST` as host
defines (computing the RS pair host-side rather than re-deriving
`has_value && stat_needs_typecast` in the kernel), and gate each alias:

```cpp
#ifdef NEEDS_OUTPUT_TYPECAST
constexpr auto dfb_output_final = dfb::writer_out;
#else
constexpr auto dfb_output_final = dfb::out;
#endif
```

A convenient property: `needs_*_typecast` implies `interm_data_format == Float32`, which implies
`any_float32`, which selects the **SFPU** compute source. So the conditional CBs only ever exist on the SFPU
path, and the non-SFPU compute kernels never read those CTAs at all (verified: `batch_norm_kernel.cpp` reads
CTAs 0–10 only; `running_statistics_kernel.cpp` reads CTAs 0–13 only).

**Every other DFB is bound unconditionally**, which is the *faithful* choice rather than a shortcut: the
legacy factory allocates `weight`/`bias`/`old_running_mean`/`old_running_var` CBs unconditionally
(`batch_norm_program_factory.cpp:260-279`, `running_statistics_program_factory.cpp:222-241`) even when the
corresponding tensor is absent, so keeping their DFBs unconditional reproduces the legacy L1 footprint
exactly. The kernels also reference those handles outside their `if constexpr` guards
(e.g. `writer_batch_norm.cpp:49, 64`), so unconditional binding is required as well as faithful.

### CB endpoints — per `(CB, config)`

**`BatchNormOperation`** (10 CBs, one conditional):

| CB | Role | Touchers | Disposition |
|---|---|---|---|
| `c_0` input | reader P → compute C | 2 | 1P+1C |
| `c_1` batch_mean | writer P → compute C | 2 | 1P+1C |
| `c_3` batch_var | writer P → compute C | 2 | 1P+1C |
| `c_4` eps | reader P (fills via `get_write_ptr`) → compute C | 2 | 1P+1C |
| `c_5` weight | writer P → compute C | 2 | 1P+1C *(both bind in all configs; FIFO ops are gated)* |
| `c_6` bias | writer P → compute C | 2 | 1P+1C *(same)* |
| `c_7` den | compute produces + consumes | 1 | **self-loop** |
| `c_8` temp_1 | compute produces + consumes | 1 | **self-loop** |
| `c_2` output_0 | **config-dependent** | 1 or 2 | **no typecast:** compute P → writer C = 1P+1C · **typecast:** compute P + compute C = **self-loop** |
| `c_9` writer_out | compute P → writer C | 2 | 1P+1C *(only allocated when `needs_output_typecast`)* |

**`RunningStatistics`** (14 CBs, two conditional):

| CB | Role | Touchers | Disposition |
|---|---|---|---|
| `c_0` batch_mean | reader P → compute C | 2 | 1P+1C |
| `c_1` batch_var | writer P → compute C | 2 | 1P+1C |
| `c_2` output | compute P → writer C | 2 | 1P+1C |
| `c_3` old_running_mean | writer P → compute C | 2 | 1P+1C *(both bind in all configs)* |
| `c_4` old_running_var | writer P → compute C | 2 | 1P+1C *(same)* |
| `c_5` momentum | reader P → compute C | 2 | 1P+1C |
| `c_6` one | reader P (via `fill_cb_with_value`) → compute C | 2 | 1P+1C |
| `c_9`/`c_10`/`c_11` tmp1/2/3 | compute produces + consumes | 1 each | **self-loop** ×3 |
| `c_7` updated_m | **config-dependent** | 1 or 2 | **no mean typecast:** compute P → writer C = 1P+1C · **mean typecast:** compute P + compute C (in `maybe_typecast_stat`) = **self-loop** |
| `c_8` updated_v | **config-dependent** | 1 or 2 | same shape as `c_7`, keyed on `needs_var_typecast` |
| `c_12` writer_updated_m | compute P → writer C | 2 | 1P+1C *(only when `needs_mean_typecast`)* |
| `c_13` writer_updated_v | compute P → writer C | 2 | 1P+1C *(only when `needs_var_typecast`)* |

The four config-dependent rows (`c_2` on BatchNorm; `c_7`, `c_8` on RS) are the *classify per instantiation*
case: the same CB is a cross-kernel FIFO in one configuration and a compute self-loop in another, because
the typecast stage inserts a compute-side consumer and hands the writer the new CB instead. A single
disposition applied across configs would mis-bind one of them.

### Hardware configuration and `opt_level`

Not gates, but the two silent-regression surfaces, and both factories are identical in shape here.

- **Compute config is Style A** — both factories resolve a TTNN `DeviceComputeKernelConfig` through
  `batch_norm::utils::resolve_compute_kernel_config` (`batch_norm_utils.cpp:14-38`) and destructure it with
  `get_compute_kernel_config_args` (`batch_norm_program_factory.cpp:349`,
  `running_statistics_program_factory.cpp:391`). So `to_compute_hardware_config(device->arch(), config)` is
  the right translation. Note the op's **non-standard defaults**: `default_fp32_acc = true` and, on
  Wormhole, `default_fp32_acc_math_fidelity = HiFi3` (working around hardware bug #38306). Because
  `fp32_dest_acc_en` defaults to **true**, `enable_32_bit_dest = true` is the *common* case here, not a
  corner — which makes `unpack_modes` load-bearing on the default path.
- **`unpack_modes` is the dangerous field, and it is populated on the default path.** Legacy builds a
  `vector<UnpackToDestMode>` indexed by CB id and sets `UnpackToDestFp32` for a specific list, only when
  `fp32_dest_acc_en` (`batch_norm_program_factory.cpp:352-368`,
  `running_statistics_program_factory.cpp:394-411`). The port must re-key that list by `DFBSpecName`. The
  exact legacy sets:
  - BatchNorm: `input`, `batch_mean`, `batch_var`, `eps`, `den`, `weight`, `temp_1`, `bias` — **plus**
    `output_0` (`c_2`) when `needs_output_typecast`. `writer_out` (`c_9`) gets **no** entry.
  - RS: `batch_mean`, `batch_var`, `output`, `old_running_mean`, `old_running_var`, `updated_m`,
    `updated_v`, `momentum`, `one`, `tmp1`, `tmp2`, `tmp3`. `writer_updated_m`/`_v` (`c_12`/`c_13`) get
    **no** entry.
  I traced these against the validator (`tt_metal/impl/metal2_host_api/program_spec.cpp:921-1073`) and both
  sets are **legal and complete** as-is:
  - Every `UnpackToDest` entry sits under `enable_32_bit_dest == true` (the `if (fp32_dest_acc_en)` guard),
    which the validator accepts unconditionally (`program_spec.cpp:1011-1013`). In particular the entries
    on ≤16-bit DFBs — reachable when `fp32_dest_acc_en` is true but `any_float32` is false, so
    `interm_data_format` is `Float16_b` — are **not** rejected. They would be rejected only with
    `enable_32_bit_dest == false`, which this op never pairs with a populated list.
  - The validator's *newly required* explicit entry (a consumed **Float32** DFB under
    `enable_32_bit_dest == true`, `program_spec.cpp:1051-1073`) is already satisfied: the legacy list covers
    every DFB the compute kernel consumes in every configuration. The two omitted CBs (`c_9`, and
    `c_12`/`c_13`) are **producer-only** for the compute kernel, so no entry is required for them.
  So this is a faithful re-key, not a value change — but it is the one place where a copy-paste between the
  two factories, or a dropped entry, flips a precision/perf tradeoff with no compile or test signal.
- **DM configs are both defaults** — `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` on all four
  dataflow kernels, so `ttnn::create_reader_datamovement_config(arch)` /
  `create_writer_datamovement_config(arch)` reproduce them exactly. No custom `(processor, noc, noc_mode)`
  triple, no `DM_DYNAMIC_NOC`.
- **`opt_level`: not set anywhere** (`grep -n opt_level` over the directory: zero hits). Resolved levels are
  therefore `O2` for the four DM kernels (no action) and **`O3` for the compute kernels** — which Metal 2.0
  does *not* default to. Both compute `KernelSpec`s need an explicit
  `compiler_options.opt_level = KernelBuildOptLevel::O3`.

### Runtime kernel-source selection — the true size of the port

Each factory selects its compute kernel **source file** at runtime:
`batch_norm_program_factory.cpp:388-390` and `running_statistics_program_factory.cpp:438-440` both
`fmt::format` the path on `(fp32_dest_acc_en || any_float32) ? "sfpu_kernel" : "kernel"`. So each factory
binds **two** possible compute sources and the atomic porting unit is *factory + 3 kernel entry points*
(reader, writer, and **both** compute sources) — 4 files per factory, 8 in total for the bundled change.
There is no "port the common path only" sub-target that builds.

Selection here runs on a **single** axis (the fp32/SFPU flavor) — it does not fan out further. The two
sources differ in CTA arity (SFPU reads 5 more CTAs) and in the presence of the typecast stage, so their
`KernelSpec`s will differ in `compile_time_args`, `defines`, and conditional DFB bindings.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No CB in either factory reaches ≥3 touchers or
  doubles a FIFO role. The hidden-second-writer hunt found nothing (the op declares no semaphores, and no
  raw co-fill exists). Self-loop and 1P+1C dispositions are recorded above.
- **Cross-op / shared kernels:** **no borrowed or lent *kernel files*.** All eight kernel sources live in
  this directory and are bound only by this directory's factories (verified by
  `grep -rl <filename> ttnn/cpp/ttnn/operations/` for each of the eight — every hit is inside
  `normalization/batch_norm/`). So the shared-kernel Caution does not apply and no `_metal2` fork is needed.
  Coupling is limited to *function-call escape* via three donor headers — see *Out-of-directory coupling*.
- **RTA varargs:** none. Every kernel reads each runtime arg exactly once as a distinct field at a literal
  index (`get_arg_val<uint32_t>(0..12)`); there is no counted loop over arg indices, no `arg_index++` run,
  and no data-selected index anywhere. All RTAs port to **named** args. Arities: BatchNorm reader 9 /
  writer 12 / compute 3; RS reader 9 / writer 11 / compute 1.
- **Zero-work cores are explicit, not skipped.** Both factories place their kernels on **all** device cores
  and hand the cores outside both work groups an all-zero RTA vector
  (`batch_norm_program_factory.cpp:72-79`, `running_statistics_program_factory.cpp:71-78`), rather than
  restricting `core_ranges` to the working set. The port must therefore supply named-RTA values for **every**
  node the kernels run on (`SetProgramRunArgs` requires it), with `0` on the idle nodes — the same values,
  minus the address slot, which the `TensorBinding` now injects. The kernels already handle it
  (`batch_norm_sfpu_kernel.cpp:205-207` returns early on `num_tiles == 0`).
- **Work split does not produce per-group CTA multiplicity.** Both factories call `split_work_to_cores` but
  use the result only to compute **per-core RTA values**; the two core groups get the *same* CTAs and a
  single `KernelDescriptor` per kernel over `all_device_cores`. So there is no preserved-multiplicity case
  here and no `WorkUnitSpec` split is needed — one work unit over all device cores per factory. (Do not
  invent a two-group split; and note this is the *opposite* of the demoting-per-group-CTA anti-pattern —
  there is nothing to preserve.)

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** Three donor headers, all ✓ shapes; no ⚠ / ✗ / ⭐ entries; no borrowed kernel
files. No donor gates the port, and none needs donor-side work for this port to proceed.

| Op kernel | Donor file | Donor class |
|---|---|---|
| `reader_batch_norm.cpp`, `writer_batch_norm.cpp`, `reader_running_statistics.cpp`, `writer_running_statistics.cpp` | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp` | 6 — cross-family donor |
| `reader_running_statistics.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/cb_fill_helpers.hpp` | 3 — shared-kernel pool |
| `running_statistics_kernel.cpp`, `running_statistics_sfpu_kernel.cpp` | `ttnn/cpp/ttnn/kernel/compute/dest_format_helpers.hpp` | 3 — shared-kernel pool |

Per-call detail (all ✓, so listed only for the record):

| Donor | Functions called | Handle shape | Status |
|---|---|---|---|
| `fill_tile_utils.hpp` | `fill_with_val<N,T>`, `fill_with_val_bfloat16`, `fill_tile_with_first_element<T>`, `fill_tile_with_first_element_bfloat16` | raw `uint32_t l1_write_ptr` | ✓ — no resource handle in the signature; nothing to bridge. Donor body is pure L1 pointer writes, no addr-gen, no CB idioms |
| `cb_fill_helpers.hpp` | `fill_cb_with_value(uint32_t cb_id, uint32_t value, int32_t)` | `uint32_t cb_id` | ✓ OK — `dfb::one` crosses on the constexpr `DFBAccessor → uint32_t` conversion |
| `dest_format_helpers.hpp` | `pack_tile_with_dt`, `copy_tile_to_dst_init_short_with_dt`, `copy_tile_init_with_dt` | `uint32_t` cb ids | ✓ OK — same conversion |

One Quasar-facing note (not a Gen1 concern, not porter work): `fill_cb_with_value` constructs a legacy
`CircularBuffer` from the id internally (`cb_fill_helpers.hpp`, `#include "api/dataflow/circular_buffer.h"`).
On Gen1 a DFB lowers to a plain circular buffer so this is byte-for-byte correct, and the helper is out of
porter scope (shared pool, kernel-side whitelist rule 9). It is worth the kernel-lib owners knowing that a
Metal 2.0 caller now reaches it with a DFB id, so the eventual Quasar uplift of any caller will need this
helper converted. Not a blocker for this port.

### Relaxation candidates

None. The sheet says `TensorParameter relaxation = none` for both rows and the code agrees: zero
`ArgConfig::Runtime*` uses, and no `TensorAccessor` third argument, so neither of the two signals that
normally motivate `dynamic_tensor_shape` / `match_padded_shape_only` is present. Keep strict matching.

Worth noting *why* this op does not need the relaxation despite being shape-parameterized: all shape
information travels as ordinary scalar RTAs (`HtWt`, `n_stride`, `c_stride`, `N`, `C`, `cHt`, `cWt`), and
the tensors are tile-layout interleaved, so the accessor configuration does not vary with shape. A shape
change invalidates the cache entry, which is the correct (strict) behavior.

### TTNN factory analysis

- **Op-owned tensors:** none. Neither factory allocates a device tensor beyond its io; both take their
  buffers from `tensor_args` / `tensor_return_value`. So `ProgramArtifacts::op_owned_tensors` stays
  defaulted, and the op does not exercise that first-use path.
- **MeshWorkload need:** none — neither op constructs a `MeshWorkload`; both are plain `descriptor`
  single-program ops. Genuinely SPMD, not a resource-workaround unwind.
- **Pybind surface:** `batch_norm_nanobind.cpp` binds only the user-facing `ttnn::batch_norm` operation. No
  device-op `nb::class_`, no `create_descriptor` binding, no pybind-hook-only factory parameter. So **none**
  of the three sanctioned device-op-class edits applies: nothing to delete, nothing to unwind.
- **Custom hash:** no `compute_program_hash`. But see the `to_hash` backdoor immediately below — it is a
  *different* mechanism and the port must leave it alone.

### The `to_hash` backdoor — analysed; leave it alone

`BatchNormOperation::operation_attributes_t` defines `to_hash()`
(`batch_norm_device_operation.hpp:22`, `batch_norm_device_operation.cpp:121-123`), hashing
`(eps, memory_config, get_dtype(), compute_kernel_config)`. This is the sheet's `Backdoor custom hash`
column and it is **confirmed present** for `BatchNormOperation`. It is *not* a
`compute_program_hash`, so the recipe's custom-hash deletion rule does not reach it, and `to_hash` lives on
the device-op class — off-limits under host-side scope discipline, and not one of the three sanctioned
exceptions. **The port must not touch it.**

I checked whether it nonetheless creates the hazard the custom-hash rule exists to prevent (a cache key that
omits `TensorSpec`, producing `UpdateTensorArgs` legality failures on the second and later dispatches). It
does not:

- The framework's default key is `hash_objects_with_default_seed(type_hash, operation_attributes,
  tensor_args)` (`ttnn/api/ttnn/device_operation.hpp:65-67`) plus an exact canonical encoding
  (`compute_mesh_workload_canonical_key`, `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1006-1022`).
  `to_hash` is honored by **both** — `append_canonical` treats a `to_hash`-supporting type as a lossy leaf
  (`tt_stl/tt_stl/reflection.hpp:1554-1557`) — so the narrowing it performs is real, not resolved by the
  canonical tiebreak.
- But `to_hash` narrows only the **attributes**: it collapses `(input_dtype, dtype)` into
  `get_dtype() = dtype.value_or(input_dtype)`. `tensor_args` is hashed and canonicalized separately and
  exactly, so **`TensorSpec` remains in the key**.
- And the collapse is currently inert: `input_dtype` is always initialized from `input.dtype()`
  (`batch_norm_device_operation.cpp:143`) — already covered exactly by the `tensor_args` half of the key —
  and `dtype` is never set by any caller. Neither field is read by the factory, which takes its data formats
  from the tensors themselves (`batch_norm_program_factory.cpp:156-163`).

So: no hazard today, nothing for the port to do. The fragility worth recording for the ops team is that the
inertness depends on `input_dtype` never diverging from `tensor_args.input.dtype()`; if a future caller sets
it independently, two attribute sets with the same effective dtype would share a program.

`RunningStatistics::operation_attributes_t` defines **no** `to_hash` — see Question 1.

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **`packer_l1_acc` is resolved and then dropped.** Both factories destructure `packer_l1_acc` from
  `get_compute_kernel_config_args` (`batch_norm_program_factory.cpp:349`,
  `running_statistics_program_factory.cpp:391`) and never use it, while
  `resolve_compute_kernel_config` deliberately defaults it to `true` (`batch_norm_utils.cpp:28`). The
  descriptor API has nowhere to put it — `ComputeConfigDescriptor` (`program_descriptors.hpp:99`) has no
  `packer_l1_acc` field — so packer-L1-accumulation the op asks for is silently not applied. This looks like
  a drop introduced by the op's earlier ProgramDescriptor migration. **Metal 2.0 cannot restore it either**:
  `ComputeGen1Config` has no equivalent field. So it is neither port work nor a port regression — it is a
  pre-existing gap for the ops team, and one the porter should be careful *not* to "fix".
- **Dead CTAs on the non-SFPU compute paths.** `batch_norm_program_factory.cpp:370-385` emits 15 CTAs but
  `batch_norm_kernel.cpp` reads only 0–10 (CTAs 11–14 — `writer_output_cb`, `needs_output_typecast`,
  `tc_in_fmt`, `tc_out_fmt` — are unread on that path). Same shape in RS: 19 CTAs emitted
  (`running_statistics_program_factory.cpp:416-435`), `running_statistics_kernel.cpp` reads only 0–13.
  Harmless today (the values are consistent — `needs_*_typecast` is necessarily false whenever the non-SFPU
  source is selected), and the natural Metal 2.0 shape drops them per-source since each source gets its own
  `KernelSpec`. Recorded so the drop reads as intentional rather than as lost plumbing.
- **`b_num_tiles_per_cb` is a redundant alias.** Both factories declare
  `uint32_t b_num_tiles_per_cb = num_tiles_per_cb;` (`batch_norm_program_factory.cpp:192`,
  `running_statistics_program_factory.cpp:189`) and use the two interchangeably. Presumably a vestige of a
  time when the batch-stat CBs had a different depth. Cosmetic.
- **Single-entry CBs are double-buffered.** `eps` / `momentum` / `one` hold exactly one tile for the whole
  kernel lifetime but are allocated with `num_entries = 2`. A small L1 saving is available; changing it is a
  functional change (L1 footprint) and explicitly out of port scope.

## Per-DeviceOperation attribution

| Field | `BatchNormOperation` | `RunningStatistics` |
|---|---|---|
| Overall | GREEN | GREEN |
| Factory | `BatchNormFactory` | `RunningStatisticsProgramFactory` |
| Kernel entry points to convert | 4 (reader, writer, 2 compute sources) | 4 (reader, writer, 2 compute sources) |
| CBs → DFBs | 10 (1 conditional: `c_9`) | 14 (2 conditional: `c_12`, `c_13`) |
| Tensor bindings | 6 (2 conditional: weight, bias) | 5 (2 conditional: running_mean, running_var) |
| Semaphores | none | none |
| Self-loop DFBs | `den`, `temp_1`, + `output_0` when typecast | `tmp1`, `tmp2`, `tmp3`, + `updated_m`/`updated_v` when typecast |
| Multi-binding flag | not needed | not needed |
| Dead CBs | none | none |
| `to_hash` backdoor | **present** (`batch_norm_device_operation.cpp:121`) — leave alone | **absent** (sheet says present — Question 1) |
| Custom `compute_program_hash` | none | none |
| `opt_level` | compute → explicit `O3`; DM → default | same |

## Questions for the user

1. **Sheet vs. code on the `RunningStatistics` backdoor hash:** both supplied rows carry
   `Backdoor custom hash (attribute_values / to_hash) = yes` and
   `Formerly custom hashed? = yes (to_hash, still present)`. For `BatchNormOperation` that is confirmed
   (`batch_norm_device_operation.cpp:121-123`). For **`RunningStatistics` I find no `to_hash` and no
   `attribute_values`** — its `operation_attributes_t` (`running_statistics_device_operation.hpp:15-23`)
   declares only `get_dtype()`, and a directory-wide grep for `to_hash` / `attribute_values` returns just
   the two BatchNorm hits. Since neither column is a conjunct of `Is able to port?` nor one of the recipe's
   cheaply-checkable cross-check columns, I did **not** treat this as spreadsheet-broken and did not gate on
   it. Worth a note to the sheet owner in case the row was filled by op-directory rather than by
   DeviceOperation.

2. **Confirm the no-regression test set** (the audit discovers tests; the porter must have them signed off
   before relying on them). The op's coverage does not live under a `normalization/` test slug — it is under
   **`fused/`**:
   - `tests/ttnn/unit_tests/operations/fused/test_batch_norm.py` — primary functional coverage; exercises
     both device-ops (`training=True` drives `RunningStatistics`), both dtypes, weight/bias present-absent,
     and the mean/var optional combinations.
   - `tests/ttnn/unit_tests/operations/fused/test_batch_norm_program_cache.py` — **the highest-value
     baseline for this port.** It pins program-cache keying *and* the running-statistics in-place side
     effect across cache hits, which is exactly the surface Metal 2.0's `UpdateTensorArgs` cache-hit path
     touches.
   - `tests/sweep_framework/sweeps/normalization/batch_norm/batch_norm.py` — a sweep. **Recommend excluding
     it from the baseline:** sweep-framework modules are currently unimportable (the `tests/` packaging
     breaks `tests.ttnn.*` imports), so it cannot be run as a regression gate.
   - `tests/tt_eager/python_api_testing/unit_testing/fallback_ops/test_batch_norm2d.py` — **excluded, not
     this op.** It tests `tt_lib.fallback_ops.BatchNorm2d` (a torch fallback) and never calls
     `ttnn.batch_norm`.
   - No C++ gtests and no nightly variants exist for this op.

   Is that the complete set, and is `test_batch_norm.py` + `test_batch_norm_program_cache.py` the baseline
   you want the port held to?

## Recipe notes

1. **The audit's `override_runtime_arguments` gate conjunct looks stale.** The recipe routes
   `Override runtime args method? == yes` to "the Metal 2.0 side — the `FactoryConcept` and this recipe
   don't support it **yet**". But `CustomProgramSpecFactoryConcept` — a spec factory whose
   `override_runtime_arguments` returns a `ProgramRunArgs` applied via `UpdateProgramRunArgs` — is on `main`
   now (`ttnn/api/ttnn/operation_concepts.hpp:132`), and `ttnn/api/ttnn/metal_v2_artifacts.hpp` names it in
   its header comment. It did not affect this audit (both rows are `no`), but the derivation and the routing
   text will mis-gate the next op that trips it.

2. **The gate derivation does not mention the `Backdoor custom hash` column, and the recipe's custom-hash
   rule is `compute_program_hash`-shaped.** An op can narrow its cache key via `to_hash` on the attributes
   struct with `Custom hash = no`, which is exactly this op. The analysis needed to clear it was
   non-trivial (does the canonical key resolve the narrowing? does `TensorSpec` survive? does the factory
   read the collapsed fields?) and the answers are not in the recipe. Two additions would help the next
   auditor: (a) state that `to_hash` on `operation_attributes_t` narrows *only* the attributes half of the
   key — `tensor_args`, hence `TensorSpec`, is hashed separately, so the troubleshooting table's
   `UpdateTensorArgs` failure mode does **not** apply; and (b) state that `to_hash` is off-limits to the
   port (it is a device-op-class member and not one of the three sanctioned exceptions), so a porter who
   pattern-matches it to the custom-hash deletion rule is wrong.

3. **`unpack_modes` guidance could name the `enable_32_bit_dest == true` shortcut.** The recipe warns that
   the validator rejects "a ≤16-bit format with `UnpackToDest` — rejected on Gen1 as a pure perf loss",
   which reads as a live hazard for any op that sets `UnpackToDestFp32` on a bf16 CB — this op does, on the
   `fp32_dest_acc_en && !any_float32` path. Reading the validator resolves it: the rejection only applies
   with `enable_32_bit_dest == false` (`program_spec.cpp:1011-1013` short-circuits first), and a legacy op
   that populates `unpack_to_dest_mode` under an `if (fp32_dest_acc_en)` guard can never hit that
   combination. One clause — "the ≤16-bit rejection applies only when `enable_32_bit_dest` is false; a
   legacy list populated under an `fp32_dest_acc_en` guard is always in the accepted branch" — would save
   the next auditor a validator read.

4. **No guidance on all-cores placement with zero-work RTA padding.** Both factories place kernels on every
   device core and pad the idle cores with all-zero RTAs rather than narrowing `core_ranges`. The recipe's
   `ProgramRunArgs` completeness rule covers the *consequence* (every named RTA must be set on every node),
   but a porter meeting this shape for the first time may reasonably wonder whether to narrow the work unit
   to the working cores instead — which would be a behavior change (kernel placement). A one-line note that
   the legacy placement is preserved verbatim, padding included, would remove the ambiguity.
