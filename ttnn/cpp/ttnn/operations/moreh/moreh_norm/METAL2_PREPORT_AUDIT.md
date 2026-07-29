# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_norm`

One `DeviceOperation`, three program factories nested inside it:

- **`ttnn::operations::moreh::moreh_norm::MorehNormOperation`** (`device/moreh_norm_device_operation.{hpp,cpp}`)
  - `MorehNormOperation::ProgramFactoryWOther` (`device/ord_other/moreh_norm_program_factory_w_other.cpp`) — reduce over the last dim
  - `MorehNormOperation::ProgramFactoryHOther` (`device/ord_other/moreh_norm_program_factory_h_other.cpp`) — reduce over the second-to-last dim
  - `MorehNormOperation::ProgramFactoryNCOther` (`device/ord_other/moreh_norm_program_factory_nc_other.cpp`) — reduce over any outer dim

Dispatch is by reduced-dim position only (`device/moreh_norm_device_operation.cpp:43-54`): `dim == rank-1` → W, `dim == rank-2` → H, otherwise NC. Interleaved-only; no sharded path, no semaphore, no Buffer-backed (borrowed-memory) CB anywhere in the op.

**The device op is reached only for `p ∈ {0, +INF, -INF}`.** The host wrapper (`moreh_norm.cpp:29-60`) routes every other `p` through `moreh_abs_pow` + `moreh_sum` instead, so those three values are the whole reachable attribute space for the ported factories. This is what the `IS_ZERO` / `MINUS_INF` / `REDUCE_OP` defines encode.

> ### ⚠ Three unreferenced kernel files share basenames with live ones — do not edit the wrong file
>
> The op directory contains **two parallel kernel trees**, and only one is live:
>
> | Path | Status |
> |---|---|
> | `device/ord_other/moreh_norm_{w,h,nc}/kernels/` (9 files) | **LIVE** — every `kernel_source` in all three factories points here |
> | `device/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp` | **unreferenced** |
> | `device/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp` | **unreferenced** |
> | `device/moreh_norm_other/kernels/moreh_norm_other_kernel.cpp` | **unreferenced** |
>
> No `KernelDescriptor::kernel_source` in the op names any of the three, and no other op or test references them; the only thing that still touches them is the kernel-install glob at `ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:44-46`. **Two of the three share a basename with a live kernel** (`moreh_norm_h_kernel.cpp`, `moreh_norm_w_kernel.cpp` exist under *both* trees), and they were swept by the Device 2.0 migrations (`af9c372a48c`, `51184417c2b`) as though live, so they look current. Their contents are **out of scope** for this audit per the recipe, and they must stay out of scope for the port. See Misc anomaly 1.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`. No copy of this op exists under `ttnn/cpp/ttnn/operations/experimental/quasar/` and there are no `*_metal2.*` lookalikes in the moreh tree, so the out-of-bounds-directory hazard does not arise here (checked by path listing only).

**Recipe docs:** `9bba65ffd6b 2026-07-29 docs(metal_2.0): follow main's MetalV2FactoryConcept -> ProgramSpecFactoryConcept rename`

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_norm` |
| **Overall** | **GREEN** — all gates cleared; brief issued for all three factories. One pre-port verification item outstanding (Q1, readiness sheet unfetchable) |
| **DOps / Factories** | `MorehNormOperation` → `ProgramFactoryWOther` · `ProgramFactoryHOther` · `ProgramFactoryNCOther` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 9 referenced kernels and every donor function they call are DFB/Device-2.0-native. Only free functions in use are 6 × **sanctioned** `get_tile_size(cb_id)`. (All three readers call `fill_cb_with_value`, whose last holdover was fixed by `ca4bc15ffb8` — see *Gate detail*) |
| *Prereqs* — Cross-op escapes | Ok — every function-call escape ✓ (donors take `DataflowBuffer` by value, or `uint32_t` CB ids as NTTPs). **Zero borrowed kernel files** — the 9 live kernels are op-exclusive, so no port-together set |
| *Feature Support* — overall | **GREEN** — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok — the only CTAs in the op are `TensorAccessorArgs`; compute kernels declare `compile_time_args = {}`. `tensor_args_t` is a fixed `{const Tensor&, const std::optional<Tensor>&}` pair |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes on code evidence** — all six cheaply-checkable conjuncts verified clean. **The readiness sheet could not be fetched this run** (claude.ai Google Drive MCP connector unauthorized; non-interactive session, so the OAuth flow cannot run), so the sheet-owned `Is safe to port?` axis and `TensorParameter relaxation` are unread → Q1 |
| *TTNN Readiness* — Concept (current) | `descriptor` — all three factories expose `static ProgramDescriptor create_descriptor(...)` (`device/moreh_norm_device_operation.hpp:34, 41, 48`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — not a `WorkloadDescriptor` op |
| *TTNN Readiness* — Is safe to port? | **Unread** (sheet). Code-side note: **no `->address()`-in-RTA smuggled pointer exists** (grep over `device/` returns nothing); every buffer rides the framework's typed `Buffer*` binding form, which is the *marked* shape — the most common source of a `no` is affirmatively absent |
| *TTNN Readiness* — Custom hash | No — no `compute_program_hash` in the op |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | No — method absent from every factory and from the device-op |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `moreh_norm_nanobind.cpp:38-49` binds only the user-facing `ttnn::moreh_norm` free function |
| *TTNN Readiness* — Op-owned tensors | No — `descriptor` concept, no `buffers` vector |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | **none — cleared.** No `->address()` anywhere; both tensors reach their kernels as bare `Buffer*` RTAs with no host arithmetic, and `tile_offset` travels as its own scalar |
| *Port work* — Tensor bindings (per binding) | **Case 1** for both bindings in all three factories (both consumed only through a `TensorAccessor`); no Case 2, no borrowed-DFB |
| *Port work* — TensorParameter relaxation | Expected **none** (no custom hash exists, so none can be active); sheet column unread → Q1 |
| *Port work* — TensorAccessor 3rd arg | **none** — all six `TensorAccessor` constructions are the 2-arg form; the subject does not fire |
| *Port work* — CB endpoints | legal 1:1 for `c_0`/`c_1`/`c_2`/`c_16`; **self-loop** ×8 — `c_24`, `c_25`, `c_26` (W, H) and `c_24`, `c_25` (NC). **No config-dependence** |

**CB endpoints** are dispositions, not gates. Recorded per `(CB, config)` below. Unusually for a masked reduction, **no disposition here flips with config** — the mask branches are plain runtime `if`s in both reader and compute, never `#ifdef`-elided, so every CB has the same census in every instantiation.

---

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`), covering all three factories.

`moreh_norm` is a clean, structurally uniform three-factory `descriptor`-concept op: no gate hits, no sharding, no semaphores, no borrowed memory, and one shape repeated three times (reader fills a `one` tile and streams input tiles → compute applies `f(x)`, accumulates along the reduced dim, reduces → writer drains). The port work is ordinary: two Case-1 tensor bindings per factory, eight self-loop assignments on the compute-private intermediates, and positional→named arg conversion.

Worth noting for sequencing: **this op would have RED'd the Device 2.0 gate a day ago.** All three readers call `fill_cb_with_value`, which until commit `ca4bc15ffb8` (*"Prepare moreh_norm for Device 2.0 port (#51402)"*, landed for `moreh_mean`) reached back through `cb.get_id()` to a free function. That one-line fix in the shared moreh dataflow pool cleared this op's gate too, on all three factories at once — the shared-donor upside of the isolated-holdover routing.

**One item to settle before the porter commits:** the readiness-sheet lookup could not be run (Q1). Six of the seven `Is able to port?` conjuncts were derived from the code and pass; `Is safe to port?` is the sheet owner's judgment, which the recipe forbids re-deriving. There is no code evidence pointing at a problem.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **cleared on code evidence; the sheet cell is unread.** The sheet could not be fetched — the claude.ai Google Drive MCP connector is unauthorized and this session is non-interactive, so the OAuth flow cannot run (`ToolSearch` finds no `mcp__claude_ai_Google_Drive__*` tool). No stale local CSV exists (`metal_2.0/analyses/` holds only the two triage `.md`s), and the recipe forbids relying on one. Conjunct by conjunct:
  - `Concept == descriptor` ✓ — all three factories are `static tt::tt_metal::ProgramDescriptor create_descriptor(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&)` (`device/moreh_norm_device_operation.hpp:34-52`), each building and returning a `ProgramDescriptor` in place (`..._w_other.cpp:82, 290`; `..._h_other.cpp:75, 268`; `..._nc_other.cpp:80, 260`). They are **nested structs inside `MorehNormOperation`**, named in the `program_factory_t` variant at `:54`. No mesh-workload return, no `create()`/`override_runtime_arguments()` pair, not already `MetalV2`.
  - `Custom hash == no` ✓ — no `compute_program_hash` in the op.
  - `get_dynamic_runtime_args == no` ✓ — absent; `device/moreh_norm_device_operation.hpp:56-60` is the complete static-hook set (`validate_inputs`, `select_program_factory`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`).
  - `override_runtime_arguments == no` ✓ — same evidence.
  - `Pybind descriptor == no` ✓ — `moreh_norm_nanobind.cpp:38-49` binds `&ttnn::moreh_norm` via `ttnn::bind_function<"moreh_norm">`; no `create_descriptor` binding, no factory/device-op internals exposed.
  - `Op-owned tensors == no` ✓ — a `descriptor`-concept op cannot carry them; no `buffers` vector exists.
  - `Is safe to port?` — **not verifiable from code by recipe rule** (expert-judgment axis). What *is* checkable is the signal that most often drives a `no`: a grep for `address()` over `device/` returns **nothing**. Both tensors are delivered as typed `Buffer*` entries into `KernelDescriptor::emplace_runtime_args` (`..._w_other.cpp:262-278`, `..._h_other.cpp:244-256`, `..._nc_other.cpp:236-248`), which binds the `std::initializer_list<std::variant<uint32_t, Buffer*>>` overload (`tt_metal/api/tt-metalium/program_descriptors.hpp:191`) and auto-registers a `BufferBinding` the framework patches on cache hits. That is the *marked* pointer form.
  - **Factory-set match** — not runnable (sheet unread). The code side is three factories; the sheet should carry three rows. Folded into Q1.

  Routing if the lookup contradicts the above: a `Concept` / `Custom hash` / `get_dynamic_runtime_args` / `Pybind` disagreement → readiness-sheet owner (the `file:line` evidence here is the counter-claim); `Is safe to port? == no` → readiness-sheet owner.

- **Device 2.0 (every kernel used):** **GREEN.** All nine referenced kernels are structurally Device 2.0: `Noc noc;` + `noc.async_read`/`async_write`/`*_barrier`, `DataflowBuffer` objects with method-form FIFO ops, `TensorAccessor` for all tensor addressing. A scan of `device/ord_other/` for `noc_async_read`, `noc_async_write`, `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`, `get_read_ptr(`, `get_write_ptr(`, `get_local_cb_interface`, `InterleavedAddrGen`, `ShardedAddrGen`, `get_semaphore`, `noc_semaphore` returns **zero hits**. The op uses no semaphores.

  The sweep was run by **shape** — every free function whose argument is a CB/DFB id — plus the high-signal `wrapper.get_id()`-inside-a-free-call cue, not against a name list. The only hits are the **sanctioned** free function:

  | File | Line | Call | Status |
  |---|---|---|---|
  | `device/ord_other/moreh_norm_w/kernels/reader_moreh_norm_w.cpp` | 45 | `get_tile_size(cb_id_input)` | sanctioned |
  | `device/ord_other/moreh_norm_w/kernels/writer_moreh_norm_w.cpp` | 30 | `get_tile_size(cb_id_output)` | sanctioned |
  | `device/ord_other/moreh_norm_h/kernels/reader_moreh_norm_h.cpp` | 44 | `get_tile_size(cb_id_input)` | sanctioned |
  | `device/ord_other/moreh_norm_h/kernels/writer_moreh_norm_h.cpp` | 27 | `get_tile_size(cb_id_output)` | sanctioned |
  | `device/ord_other/moreh_norm_nc/kernels/reader_moreh_norm_nc.cpp` | 34 | `get_tile_size(cb_id_input)` | sanctioned |
  | `device/ord_other/moreh_norm_nc/kernels/writer_moreh_norm_nc.cpp` | 27 | `get_tile_size(cb_id_output)` | sanctioned |

  **Donor bodies checked, all clean.** The functions this op actually calls:
  - `fill_cb_with_value` (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:98`) — called by **all three** readers (`reader_moreh_norm_w.cpp:30`, `reader_moreh_norm_h.cpp:31`, `reader_moreh_norm_nc.cpp:30`). Its body now reads `cb.get_dataformat()` (the wrapper method) as of commit `ca4bc15ffb8`; before that it was `get_dataformat(cb.get_id())`, a CB-index-keyed holdover with the wrapper in scope, which would have RED'd this gate on **every** factory of this op. Re-verified against the committed content.
  - `generate_mask_w` / `generate_mask_h` (`moreh_common.hpp:223, 183`) — `cb_mask.reserve_back()` / `.get_write_ptr()` / `.push_back()`, method form throughout.
  - `compute_kernel_lib::reduce<…>` (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`) — no CB-index free function in the instantiated path (`reduce_helpers_compute.inl` has no `get_tile_size(` / `get_dataformat(` / `get_local_cb_interface` call at all).
  - The `*_with_dt` helpers (`ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp:28, 35, 42`) — take `DataflowBuffer` by value.

  **Note this op does *not* reach `reduce_helpers_dataflow.inl`.** Its readers build the `one` tile with `fill_cb_with_value`, not `calculate_and_prepare_reduce_scaler`, so the constexpr `get_dataformat(dfb_id)` / `get_tile_r_dim<dfb_id>()` observation recorded against `moreh_mean`'s H path has **no analogue here** — there is no residual Device 2.0 question on this op.

  **Not flagged — compute-side LLK calls.** The three compute kernels and the `*_with_dt` donors pass raw CB ids to Gen1 **compute** LLKs (`copy_tile(cb_x, 0, dst0)`, `add_tiles`, `binary_max_tile`, `mask_tile`, `abs_tile`, `unary_ne_tile`, `negative_tile`, `pack_tile`, `binary_op_init_common`, `reconfig_data_format` via `icb.get_id()`). These are **compute** APIs, outside the scope of a *data-movement* migration, and `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:47-57` documents `DFBAccessor`'s implicit constexpr `operator uint32_t()` as existing precisely so a Metal 2.0 kernel can feed them. Not holdovers.

- **Feature compatibility:** all four Appendix A entries scanned; every one `N/A` (a clean scan is all-`N/A`).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` / `CreateGlobalCircularBuffer` / `.global_circular_buffer` field / `remote_index` / `remote_cb_*`. All 19 `CBDescriptor`s across the three factories are plain `{total_size, core_ranges, format_descriptors}` literals. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset`, no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. No CB is Buffer-backed (no `set_globally_allocated_address`). |
  | GlobalSemaphore | N/A | No `GlobalSemaphore`, no `CreateGlobalSemaphore`, no `global_semaphore.hpp` — the op declares no semaphores of any kind. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent: `tensor_args_t` is a fixed pair (`device/moreh_norm_device_operation.hpp:25-28`), no `std::vector<Tensor>`. Kernel-level decider absent: the *only* CTAs in the whole op are the `TensorAccessorArgs` blocks (`..._w_other.cpp:158-161`, `..._h_other.cpp:151-154`, `..._nc_other.cpp:138-141`), consumed as `constexpr auto args = TensorAccessorArgs<0>()`; all three compute kernels set `compile_time_args = {}` and read none. No CTA loop, no runtime-varying CTA index. |

- **Offset base pointers:** **GREEN — cleared.** Every runtime-arg site in all three factories was resolved. There is **no `buffer()->address()` expression anywhere in the op** (grep over `device/`: zero hits), so there is no host arithmetic into which an offset could be folded — tensor bases are delivered as typed `Buffer*` entries. `tile_offset` travels as its **own scalar** and is consumed only as a page index, never as an address: W `..._w_other.cpp:268` → `reader_moreh_norm_w.cpp:16, 41, 49`; H `..._h_other.cpp:249` → `reader_moreh_norm_h.cpp:15, 46, 51`; NC `..._nc_other.cpp:241` → `reader_moreh_norm_nc.cpp:15, 36, 42`; writers `..._w_other.cpp:278` / `..._h_other.cpp:256` / `..._nc_other.cpp:248` → `writer_*.cpp` `tile_offset` → `{.page_id = …}`. The W writer's `start_tile_idx = tile_offset / Wt` (`writer_moreh_norm_w.cpp:26`) is *page-index* arithmetic on a tile counter, not address arithmetic. Type 1: none. Type 2: none. Type 3: none. Type 4: none. Cross-check against the dated prior `analyses/2026-07-19_offset_base_pointers.md`: `moreh_norm` is **not** in its tables — the "no fold, not in the tables" outcome, i.e. agreement.

- **TensorAccessor 3rd argument:** **GREEN — the subject does not fire.** All six constructions pass exactly two arguments: `reader_moreh_norm_w.cpp:25`, `writer_moreh_norm_w.cpp:24`, `reader_moreh_norm_h.cpp:26`, `writer_moreh_norm_h.cpp:23`, `reader_moreh_norm_nc.cpp:25`, `writer_moreh_norm_nc.cpp:23`. No page-size override → no Class 1/2 drop, no Class 3/4/Special gate. Cross-check against the dated prior `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`: `moreh_norm` is absent from its op→class table — agreement.

- **CB endpoints (GATE-free):** classified per `(CB, config)`, per node. **No dead CB, no multi-binding, no ≥3-toucher CB anywhere.** In each factory the two compute `KernelDescriptor`s cover **disjoint** core sets (`core_group_1` / `core_group_2` from `tt::tt_metal::split_work_to_cores`), so **each node sees exactly one compute instance** — not a dual-instance work-split, and there is no co-fill or co-read. The hidden-second-writer hunt found no raw `get_write_ptr()`/`fifo_wr_ptr` write by a non-producer (and no semaphores to coordinate one); the multiple-reader hunt found no borrowed-memory CB and no CB whose read sites span two kernels.

  **`ProgramFactoryWOther`** (7 CBs at `..._w_other.cpp:84-146`):

  | CB | Touchers on a node | Verdict | Disposition |
  |---|---|---|---|
  | `c_0` input | reader locked-P (`reader_moreh_norm_w.cpp:50, 53`) · compute locked-C (`moreh_norm_w_kernel.cpp:57, 91`) | plain 1:1 | none |
  | `c_1` one | reader locked-P (`fill_cb_with_value` → `moreh_common.hpp:99, 106`, via `reader_moreh_norm_w.cpp:30`) · compute locked-C (`moreh_norm_w_kernel.cpp:43, 164`; also the `reduce<>` scaler at `:140`, same kernel) | plain 1:1 | none |
  | `c_2` mask_w | reader locked-P (`generate_mask_w`, via `reader_moreh_norm_w.cpp:36-39`) · compute locked-C (`moreh_norm_w_kernel.cpp:50, 166`) | plain 1:1 | none |
  | `c_16` output | compute locked-P (`moreh_norm_w_kernel.cpp:146, 161`) · writer locked-C (`writer_moreh_norm_w.cpp:34, 37`) | plain 1:1 | none |
  | `c_24` val — `f(x)` | compute only — P `moreh_norm_w_kernel.cpp:58, 92`; C `:97, 108, 112, 134` | 1 toucher | **self-loop** |
  | `c_25` cal — accumulator | compute only — P `:98, 109, 114, 136`; C `:113, 135`, plus the `reduce<>` input at `:140` | 1 toucher | **self-loop** |
  | `c_26` reduce | compute only — P as the `reduce<>` output DFB `:140`; C `:145, 160` | 1 toucher | **self-loop** |

  **`ProgramFactoryHOther`** (7 CBs at `..._h_other.cpp:77-139`) — structurally identical to W, with `mask_h` for `mask_w` and the row/col loops transposed:

  | CB | Touchers on a node | Verdict | Disposition |
  |---|---|---|---|
  | `c_0` input | reader locked-P (`reader_moreh_norm_h.cpp:53, 56`) · compute locked-C (`moreh_norm_h_kernel.cpp:56, 91`) | plain 1:1 | none |
  | `c_1` one | reader locked-P (via `reader_moreh_norm_h.cpp:31`) · compute locked-C (`moreh_norm_h_kernel.cpp:43, 165`; also the `reduce<>` scaler at `:141`) | plain 1:1 | none |
  | `c_2` mask_h | reader locked-P (`generate_mask_h`, via `reader_moreh_norm_h.cpp:37-40`) · compute locked-C (`moreh_norm_h_kernel.cpp:50, 167`) | plain 1:1 | none |
  | `c_16` output | compute locked-P (`moreh_norm_h_kernel.cpp:147, 162`) · writer locked-C (`writer_moreh_norm_h.cpp:31, 34`) | plain 1:1 | none |
  | `c_24` val | compute only — P `:57, 92`; C `:97, 108, 113, 135` | 1 toucher | **self-loop** |
  | `c_25` cal | compute only — P `:98, 109, 115, 137`; C `:114, 136`, plus the `reduce<>` input at `:141` | 1 toucher | **self-loop** |
  | `c_26` reduce | compute only — P as the `reduce<>` output `:141`; C `:146, 161` | 1 toucher | **self-loop** |

  **`ProgramFactoryNCOther`** (5 CBs at `..._nc_other.cpp:82-126`) — no mask CB and no `reduce<>` call; the accumulator is drained straight to the output:

  | CB | Touchers on a node | Verdict | Disposition |
  |---|---|---|---|
  | `c_0` input | reader locked-P (`reader_moreh_norm_nc.cpp:44, 47`) · compute locked-C (`moreh_norm_nc_kernel.cpp:43, 66`) | plain 1:1 | none |
  | `c_1` one | reader locked-P (via `reader_moreh_norm_nc.cpp:30`) · compute locked-C (`moreh_norm_nc_kernel.cpp:37, 137`) — **the tile's data is never used**; see Misc anomaly 4 | plain 1:1 | none |
  | `c_16` output | compute locked-P (`moreh_norm_nc_kernel.cpp:120, 135`) · writer locked-C (`writer_moreh_norm_nc.cpp:31, 34`) | plain 1:1 | none |
  | `c_24` val | compute only — P `:44, 67`; C `:72, 83, 88, 110` | 1 toucher | **self-loop** |
  | `c_25` cal | compute only — P `:73, 84, 90, 112`; C `:89, 111, 119, 134` | 1 toucher | **self-loop** |

  **Dead-CB check — deliberately distrusted; nothing is dead.** W and H each allocate 7 CBs and touch all 7; NC allocates 5 and touches all 5. **The mask CB is *not* a config-dependent dead-CB candidate here** — unlike some masked reductions, both the reader's `generate_mask_*` (`reader_moreh_norm_w.cpp:36-39`, `reader_moreh_norm_h.cpp:37-40`) and the compute's `wait_front`/`pop_front` sit behind plain **runtime** `if (do_mask_*)` branches, never `#ifdef`, so both touchers are compiled in every instantiation and the census is config-invariant. **No CB in this op should be dropped.**

---

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, identical in all three factories):
  - `input` — **Case 1** (via `TensorAccessor`). Base arrives as RTA index 0 (typed `Buffer*`), consumed only by the accessor. W: `..._w_other.cpp:159` (accessor CTAs), `:264` (RTA) → `reader_moreh_norm_w.cpp:12, 24, 25`. H: `..._h_other.cpp:152`, `:246` → `reader_moreh_norm_h.cpp:12, 25, 26`. NC: `..._nc_other.cpp:139`, `:238` → `reader_moreh_norm_nc.cpp:12, 24, 25`.
  - `output` — **Case 1**. W: `..._w_other.cpp:161`, `:274` → `writer_moreh_norm_w.cpp:14, 23, 24`. H: `..._h_other.cpp:154`, `:256` → `writer_moreh_norm_h.cpp:14, 22, 23`. NC: `..._nc_other.cpp:141`, `:248` → `writer_moreh_norm_nc.cpp:14, 22, 23`.
  - **No Case 2** and **no borrowed-DFB/clean bindings** anywhere in the op.
  - Urgency note: the `Buffer*` form is the framework's interim marked-pointer mechanism, patched correctly on cache hits today → **routine port work, not a live correctness hazard.**
- **TensorParameter relaxation:** none expected — no custom hash exists, so none can be active. Confirm the sheet column at Q1.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop 8 — W `c_24`/`c_25`/`c_26`, H `c_24`/`c_25`/`c_26`, NC `c_24`/`c_25`. All other CBs are legal 1:1, in every config. No 1P+1C assignment, no multi-binding flag, no dead-CB drop.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — both hunts came back empty; no semaphores, no borrowed-memory CBs, no config-dependent census.
- **Two parallel kernel trees; three of the twelve files are unreferenced** — see the banner above and Misc anomaly 1. The live tree is `device/ord_other/…` throughout; two unreferenced files share a basename with a live one.
- **Every kernel derives its CB ids from a runtime counter, not a named constant.** All six dataflow kernels open with `uint32_t cb_id{0};` (readers) or `uint32_t cb_id{16};` (writers) and then `const auto cb_id_x = cb_id++;` (e.g. `reader_moreh_norm_w.cpp:19-22`, `writer_moreh_norm_w.cpp:20-21`), and the **NC compute kernel** does the same with `std::uint8_t input_id{tt::CB::c_in0}; const auto cb_x = input_id++;` (`moreh_norm_nc_kernel.cpp:12-29`). These are non-`constexpr` values, so a `dfb::name` accessor token cannot be substituted at the declaration site — the ids stay runtime values feeding the low-level `DataflowBuffer(uint16_t)` ctor. (The W and H compute kernels use `constexpr auto` for the same idiom — `moreh_norm_w_kernel.cpp:14-35` — and *can* take named tokens directly.)
- **Same-source compute pair over disjoint core groups — do not demote the per-group RTA/CTA.** Each factory emits two compute `KernelDescriptor`s from one source over disjoint node sets (`..._w_other.cpp:200-229`, `..._h_other.cpp:192-221`, `..._nc_other.cpp:176-205`). Ordinary 1:1 — keep two `KernelSpec`s. Note that here the per-group work count is already an **RTA**, not a CTA (`compile_time_args = {}` on both), so the *demoting-per-group-CTA* anti-pattern cannot arise; just don't collapse the two specs.
- **Cross-op / shared kernels:** no borrowed kernel *file* — the 9 live kernels are op-exclusive, so nothing to co-port. All donor headers cross cleanly (detail below).
- **RTA varargs:** **none.** Every kernel reads its args through a running `i++` over a **fixed** run at the top of `kernel_main` — the recipe's explicit non-signal case. All are nameable; e.g. `reader_moreh_norm_nc.cpp:11-18` → `input_addr`, `input_is_dram`, `num_output_tiles_per_core`, `tile_offset`, `outer_stride`, `num_inner_tiles`, `num_reduced_tiles_along_dim`. No count-bounded RTA loop, no data-selected index. (Note the `*_is_dram` args are dead — Misc anomaly 2 — but the port preserves them unless the ops team removes them first.)

---

## Team-only

### Out-of-directory coupling & donor shape analysis

**Op-level roll-up: ✓ clean.** Every escape lands in a shared kernel pool (never a cross-family op donor), and every function called takes either a `DataflowBuffer` by value or a `uint32_t` CB id as an NTTP. No `CircularBuffer&`, no `uint32_t sem_id`/`sem_addr`, no `TensorAccessorArgs<N>` parameter, no CTA-offset NTTP, no old-style addr-gen.

| Op kernel | Donor file | Class | Functions / types used | Shape | Status |
|---|---|---|---|---|---|
| `reader_moreh_norm_w.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 3 — shared pool (`ttnn/cpp/ttnn/kernel/`) | `fill_cb_with_value` (`:98`), `generate_mask_w` (`:223`), `Scalar` union (`:39`) | `DataflowBuffer` by value | ✓ |
| `reader_moreh_norm_h.cpp` | same | 3 | `fill_cb_with_value`, `generate_mask_h` (`:183`), `Scalar` | `DataflowBuffer` by value | ✓ |
| `reader_moreh_norm_nc.cpp` | same | 3 | `fill_cb_with_value`, `Scalar` | `DataflowBuffer` by value | ✓ |
| `moreh_norm_w_kernel.cpp`, `moreh_norm_h_kernel.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | 2 — official kernel lib | `compute_kernel_lib::reduce<op, dim, in, scaler, out>` + `ReduceInputBlockShape::single()` | `uint32_t` CB-id NTTPs | ✓ |
| `moreh_norm_{w,h,nc}_kernel.cpp` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | 3 | `copy_tile_init_with_dt` (`:35`), `pack_tile_with_dt` (`:28`), `add_tiles_init_with_dt` (`:42`); also re-exports the compute LLKs the kernels use (`mask_tile`, `mask_posinf_tile`, `abs_tile`, `unary_ne_tile`, `negative_tile`, `binary_max_tile`) | `DataflowBuffer` by value | ✓ (bodies feed compute LLKs raw ids via `icb.get_id()` — out of the data-movement gate's scope) |
| all 6 dataflow kernels | `tt_metal/hw/inc/api/**` (`dataflow/noc.h`, `dataflow/dataflow_buffer.h`, `dataflow/dataflow_api.h`, `tensor/noc_traits.h`) | 1 — LLK / HAL | — | — | ✓ no concern |

No per-call detail section is needed (all rolls ✓).

**Borrowed kernel files (file-path instantiation):** **none.** All nine live `kernel_source` paths point inside `device/ord_other/…`, and a repo-wide grep for `ord_other` outside this op returns nothing — the op is its own port-together set, with no shared-kernel rewrite to coordinate. (The three *unreferenced* files are likewise op-local; see Misc anomaly 1.)

### Relaxation candidates

None to mine — the op has no custom hash.

### TTNN factory analysis

Sheet-derived facts unread this run; the code-side equivalents are in *Gate detail*. Non-gating facts that inform the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`): current concept `descriptor` for all three factories; **no** op-owned tensors; **no** MeshWorkload need; **no** pybind of internals and no other migration-risky pybind; **no** custom hash; **no** `get_dynamic_runtime_args`; **no** `override_runtime_arguments`. Target concept: `ProgramSpecFactoryConcept`. All three factories in the `program_factory_t` variant (`device/moreh_norm_device_operation.hpp:54`) convert together; `validate_inputs`, `select_program_factory`, `compute_output_specs`, and `create_output_tensors` are untouched by the port.

---

## Misc anomalies  *(team-only, non-gating — route to the ops team; the port does not act on these)*

1. **Three unreferenced kernel files, two of them basename-colliding with live ones.** `device/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp`, `device/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp`, and `device/moreh_norm_other/kernels/moreh_norm_other_kernel.cpp` are named by **no** `kernel_source` in the op, by no other op, and by no test — only by the kernel-install glob at `ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:44-46`, which still copies them into the install tree. They are almost certainly the pre-`ord_other` implementation left behind when the op moved to the current three factories. Two hazards: (a) the basename collision invites editing the wrong file (the audit's identifying banner warns the porter), and (b) the Device 2.0 sweeps (`af9c372a48c`, `51184417c2b`) migrated them as though live, so they read as current code and will keep absorbing maintenance. Recommend deleting the three files and their CMake glob lines.
2. **Six dead `*_is_dram` RTAs.** Every dataflow kernel reads a DRAM flag it never uses — `reader_moreh_norm_w.cpp:13`, `reader_moreh_norm_h.cpp:13`, `reader_moreh_norm_nc.cpp:13`, `writer_moreh_norm_w.cpp:15`, `writer_moreh_norm_h.cpp:15`, `writer_moreh_norm_nc.cpp:15` (each name occurs exactly once in its file, at the read). The `TensorAccessorArgs` CTAs already carry the DRAM/L1 distinction, so the flag is redundant; the host still computes and passes it (`is_dram(input)` / `is_dram(output)` at `..._w_other.cpp:265, 275`, `..._h_other.cpp:247, 256`, `..._nc_other.cpp:239, 248`). Dropping it would remove one RTA per kernel across all three factories.
3. **`get_floored_p_and_decimal_and_p_is_negative` is dead in this op.** Defined at `device/moreh_norm_device_operation.cpp:14-22` and declared at `device/moreh_norm_device_operation.hpp:14`, but called nowhere in `moreh_norm` — the live copy is `moreh_abs_pow`'s byte-identical function (`moreh_abs_pow/device/moreh_abs_pow_device_operation.cpp:14`, used at `moreh_abs_pow_program_factory.cpp:55`). No ODR clash (different namespaces), just a stale duplicate left over from when `moreh_norm` computed fractional `p` itself; the host wrapper now delegates those cases to `moreh_abs_pow` (`moreh_norm.cpp:33-35, 57-59`).
4. **NC: the `one` tile is produced and consumed but never used.** `reader_moreh_norm_nc.cpp:27-30` fills `c_1` with `1.0f` and `moreh_norm_nc_kernel.cpp:37, 137` waits on and pops it, but no compute op in that kernel ever reads it — the NC path accumulates with `add_tiles`/`binary_max_tile` and has no `reduce<>` call, so it needs no scaler. The result is a wasted L1 tile plus a full tile-fill per core per program. (In the W and H paths the same CB *is* load-bearing, as the `reduce<>` scaler — `moreh_norm_w_kernel.cpp:140`, `moreh_norm_h_kernel.cpp:141` — which is likely why the NC copy was never noticed.) The CB is genuinely live in the endpoint census (1P+1C), so this is a waste finding, not a drop candidate for the port.
5. **Deprecated `tt::CB` enum in the compute kernels.** All three use `tt::CB::c_in0` / `c_out0` / `c_intermed0` (`moreh_norm_w_kernel.cpp:14, 22, 25`; `moreh_norm_h_kernel.cpp:14, 22, 25`; `moreh_norm_nc_kernel.cpp:12, 18, 22`) while the factories use `tt::CBIndex::c_N`. Both resolve to the same values (`tt_metal/hostdevcommon/api/hostdevcommon/kernel_structs.h:112` puts `c_intermed0 = 24`), so there is no bug — only an inconsistency, and one that makes the CB↔factory mapping harder to read than it needs to be.
6. **Inconsistent constexpr-ness of the CB-id idiom.** W and H compute use `constexpr auto cb_x = input_id + 0;` while NC compute uses a mutable `std::uint8_t input_id{…}; const auto cb_x = input_id++;` (`moreh_norm_nc_kernel.cpp:12-29`), and all six dataflow kernels use the mutable form. Same effect, three spellings; the constexpr form is the one that will accept a `dfb::name` token directly at port time.

## Per-DeviceOperation attribution

Single DeviceOperation (`MorehNormOperation`) — no bundling. All three factories carry the same GREEN verdict. Findings that differ between them (NC's missing mask CB and missing `reduce<>`, NC's unused `one` tile, NC compute's runtime CB ids) are attributed per factory throughout.

## Questions for the user

1. **Readiness-sheet lookup (needed before the port commits).** The sheet could not be fetched — the claude.ai Google Drive connector is unauthorized and this session is non-interactive, so the OAuth flow cannot run; it must be authorized from claude.ai connector settings, or the fetch run from an interactive session. Please pull `metal_2.0/analyses/ttnn_op_porting_readiness.csv` per `ttnn_op_porting_readiness.md` and confirm, for the `moreh/moreh_norm` rows: (a) there are **three** rows, one per factory — `ProgramFactoryWOther`, `ProgramFactoryHOther`, `ProgramFactoryNCOther` (a phantom or missing row means the sheet is stale for this op; note the *factory names* here do not follow the `Moreh…Factory` convention that `moreh_mean` uses, so a name-based lookup may need care); (b) `Is able to port? == yes` on all three; (c) `Is safe to port? == yes` — the one conjunct I am forbidden to re-derive (see *Gate detail* for the code evidence that no smuggled pointer exists); (d) `TensorParameter relaxation == none`.
2. **May the three unreferenced kernel files be deleted (Misc anomaly 1)?** They are dead by every reference test I can run, but "unreferenced in this repo" is not the same as "safe to delete" if anything out-of-tree consumes the installed kernel directory. Deleting them before the port would remove the basename-collision hazard entirely; if they must stay, the port simply ignores them (the brief says so explicitly).
3. **NC's unused `one` tile (Misc anomaly 4) — remove before the port, or after?** Removing it deletes a CB, a reader fill, and a compute wait/pop from the NC factory. Doing it *before* the port shrinks that factory's spec by one DFB; doing it *after* keeps the port a pure translation. Either is fine — the port preserves current behavior verbatim if it is left alone — but the two changes are cleaner reviewed separately.

## Recipe notes

1. **The "unreferenced kernel files" rule works, but the basename-collision case deserves a sentence.** The recipe says unreferenced kernels are out of scope and to "mention them in the identifying section as unreferenced" if their presence could confuse a reader. Here the confusion risk is unusually sharp: `moreh_norm_h_kernel.cpp` and `moreh_norm_w_kernel.cpp` each exist **twice** in the op, once live and once dead, differing only by a `ord_other/` path segment — and the dead copies have been kept superficially current by the Device 2.0 sweeps. A porter working from a basename (or an editor's fuzzy file-open) can land in the dead file and produce a diff that compiles, installs, and changes nothing. Suggest the recipe promote this from "mention it" to "call it out prominently *and* state the live path explicitly" whenever a dead kernel shares a basename with a live one — which is what I did in the banner.
2. **A `p`-style float attribute whose reachable value set is fixed by the *host wrapper*, not by validation, is worth a named slot.** `moreh_norm`'s device op only ever sees `p ∈ {0, +INF, -INF}` because `moreh_norm.cpp:29-60` routes all other values through `moreh_abs_pow`/`moreh_sum`; the device op itself never asserts this. That fact is load-bearing for reading the factories (it is why the `IS_ZERO`/`MINUS_INF`/`REDUCE_OP` define matrix is exhaustive) but it lives one file *above* the audit's usual scope, and no subject asks for it. A line in the identifying-section guidance — *"record any attribute whose reachable range is narrowed by the host wrapper rather than by validation"* — would make the next auditor look for it.
3. **The CB-endpoint subject's config-dependence guidance assumes `#ifdef`; the `runtime if` case deserves equal billing.** For `moreh_mean` I had to reason carefully about `#ifdef`-elided reader accesses creating config-dependent one-toucher CBs. `moreh_norm` is the *contrast* case: the identical masked-reduction shape, but with plain runtime `if (do_mask_*)` guards on both sides, so the census is config-invariant and the mask CB is a boring 1:1 in every instantiation. Both spellings are common in this family and they produce different dispositions from otherwise-identical code. Suggest the *Classify per instantiation* paragraph say so directly: **the question is whether the access is compiled, not whether it executes** — `#ifdef` removes it, `if`/`if constexpr` does not.
4. **Worth recording as a positive:** the isolated-holdover routing paid a dividend across ops here. `moreh_norm` would have RED'd on all three factories via `fill_cb_with_value`; the one-line fix routed out of the `moreh_mean` audit (`ca4bc15ffb8`) cleared it before this audit ran. That is an argument for the recipe's insistence on reporting donor-body holdovers with a precise `file:line` and owning family rather than scoping them to the op under audit — the fix lands once and unblocks every co-borrower.
5. **Minor, carried over:** the *Feature compatibility* per-row label rule (`N/A` when absent, never `GREEN`) is stated well *after* the status-summary template, whose *Feature Support — overall* row shows `GREEN / RED`. Reading top-to-bottom invites filling the per-feature rows with `GREEN`. A pointer at the template would help.
