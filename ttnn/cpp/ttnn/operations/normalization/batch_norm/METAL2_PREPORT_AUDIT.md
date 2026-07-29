# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/normalization/batch_norm`

The directory holds **two independent DeviceOperations** behind one user-facing op (`ttnn::batch_norm`), audited together in this one report (see *Bundling decision* below):

- **`BatchNormOperation`** (`device/batch_norm_device_operation.{hpp,cpp}`)
  - `BatchNormFactory` (`device/batch_norm_program_factory.cpp`)
    - kernels: `dataflow/reader_batch_norm.cpp`, `dataflow/writer_batch_norm.cpp`, `compute/batch_norm_kernel.cpp`, `compute/batch_norm_sfpu_kernel.cpp`
- **`RunningStatistics`** (`device/running_statistics_device_operation.{hpp,cpp}`)
  - `RunningStatisticsProgramFactory` (`device/running_statistics_program_factory.cpp`)
    - kernels: `dataflow/reader_running_statistics.cpp`, `dataflow/writer_running_statistics.cpp`, `compute/running_statistics_kernel.cpp`, `compute/running_statistics_sfpu_kernel.cpp`

Each factory selects one of **two** compute-kernel source files at descriptor-build time
(`fmt::format(... (fp32_dest_acc_en || any_float32) ? "sfpu_kernel" : "kernel")`,
`batch_norm_program_factory.cpp:388-390`, `running_statistics_program_factory.cpp:438-440`).
Both variants of both pairs are in scope and were audited. **The SFPU variant is the default path** —
`resolve_compute_kernel_config` sets `default_fp32_acc = true` (`device/batch_norm_utils.cpp:31`), so
`fp32_dest_acc_en` is true unless the caller explicitly overrides it.

No unreferenced kernel files in the directory. **The op owns all 8 kernels and no other op binds any of them**
(see *Out-of-directory coupling*).

**Bundling decision.** The two DeviceOperations share **no** factory and **no** kernel, which by the letter of
the recipe's shared-code test argues for separate audits. They are bundled here because they share a host
helper (`device/batch_norm_utils.hpp`), the same three donor headers, the same structural shape
(reader/writer/compute over `all_device_cores`, same RTA layout, same optional-tensor idiom), and are
co-invoked from a single `ttnn::batch_norm` call — so they form one porting unit in practice, and there is
one report location. Confirmed with the owner. **Findings are attributed per DeviceOperation throughout**; see
*Per-DeviceOperation attribution*. Every finding below is identical for both unless stated.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `40b61b016a1 2026-07-29 docs(metal_2.0): fix stale API symbol names across the porting docs`

> ### ↺ Re-audit — what moved since the previous revision of this report
>
> This is a re-audit against a rebased branch. Every finding below was re-derived from the current tree; the
> two code deltas are both material, and one recipe delta retires a standing flag.
>
> 1. **All 8 kernels were migrated `CircularBuffer` → `DataflowBuffer`** by
>    `bed70038e18 (#49173) — "[Cleanup] Migrate MM/Fused/Reduce Kernels from CircularBuffer to DataflowBuffer"`.
>    519 lines changed across all 8 files: `CircularBuffer` → `DataflowBuffer`, `#include
>    "api/dataflow/circular_buffer.h"` → `"api/dataflow/dataflow_buffer.h"`, `cb_*` → `dfb_*` naming, and —
>    notably — **the last CB-index free function in the op's own kernels is gone**: `get_tile_size(cb_id)` is now
>    `dfb.get_entry_size()`. **Every kernel `file:line` in the previous revision of this report was invalidated;
>    all have been re-derived.** No gate verdict changed, and the endpoint census is structurally identical
>    (same touchers, same roles, same self-loops).
> 2. **The `unpack_to_dest_mode` fix landed** — `c8fab99f9ac (#51313) — "batch_norm: add output CB to
>    UnpackToDestFp32 list for typecast path"`. This was Misc anomaly 6 of the previous revision; it is now
>    **resolved in code** (`batch_norm_program_factory.cpp:365-367`) and moved to *Resolved* below.
> 3. **The recipe caught up on the factory-concept rename** — `4d77c80a2f7 — "docs(metal_2.0): follow main's
>    MetalV2FactoryConcept -> ProgramSpecFactoryConcept rename"`. The previous revision carried this as a
>    standing divergence (docs vs. code) and told the porter to ignore the docs. **That flag is retired**: the
>    docs and the code now agree on `ProgramSpecFactoryConcept`.
>
> Framework headers were re-checked and are **unchanged** since the previous revision
> (`tt_metal/api/tt-metalium/experimental/metal2_host_api/`, `ttnn/api/ttnn/operation_concepts.hpp` — empty diff),
> so the target-concept and `ProgramRunArgs` findings carry over verbatim.
>
> **`experimental/quasar/` was not consulted.** The recipe now puts that tree out of bounds for the audit as
> well as the port. There is no quasar copy of `batch_norm` in any case (the directory holds ~28 other ops), and
> no `_metal2` file anywhere in this op's or its donors' directories, so nothing from that tree reached this
> report.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/normalization/batch_norm/` |
| **Overall** | **GREEN — every gate cleared. `METAL2_PORT_BRIEF.md` issued.** |
| **DOps / Factories** | `BatchNormOperation` → `BatchNormFactory` · `RunningStatistics` → `RunningStatisticsProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes — GREEN.** All 8 kernels are now DFB-native (#49173); **zero** CB-index free functions remain in the op's own kernels. All 3 donor headers clear. |
| *Prereqs* — Cross-op escapes | Ok — 3 header donors (function-call escape only), all `uint32_t cb_id` / raw-L1-pointer shapes. **No borrowed kernel files**, so no `_metal2` fork and no sunset list. |
| *Feature Support* — overall | **GREEN** — all four Appendix A entries `N/A` |
| *Feature Support* — Variadic-CTA | Ok — every CTA index is a literal or a `constexpr` chain |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes — GREEN, both factory rows.** Sheet rows supplied by the owner; cross-check clean on every checkable column. |
| *TTNN Readiness* — Concept (current) | `descriptor` — sheet and code agree (both factories expose `create_descriptor`, satisfying `ProgramDescriptorFactoryConcept`) |
| *TTNN Readiness* — Op Classification (sheet) | `PD (pointer-patching)` — both rows |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (`descriptor` form, not `create_workload_descriptor`) |
| *TTNN Readiness* — Is safe to port? | **Yes** (sheet), `Smuggled pointer = no`. Consistent with `ab578509470 2026-07-07 (#49136)` and with my scan finding zero `->address()` RTAs. |
| *TTNN Readiness* — Custom hash | **No** — sheet and code agree. (`BatchNormOperation::operation_attributes_t::to_hash()` at `device/batch_norm_device_operation.cpp:121-123` is *not* scored as a custom hash — the column tracks `compute_program_hash` only. See Recipe note 4.) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — sheet and code agree |
| *TTNN Readiness* — `override_runtime_arguments` | No — sheet and code agree |
| *TTNN Readiness* — Pybind `create_descriptor` | No — sheet and code agree (`batch_norm_nanobind.cpp:70-84` binds only `&ttnn::batch_norm`) |
| *TTNN Readiness* — Op-owned tensors | No (blank on both rows; not expressible on the `descriptor` form) |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** — code (`ttnn/api/ttnn/operation_concepts.hpp:119`) and recipe docs now agree |
| *Port work* — Offset base pointer | **none — GREEN.** Zero `->address()` call sites op-wide; every buffer reaches the RTA list as a `Buffer*`. No host-folded offsets. |
| *Port work* — Tensor bindings (per binding) | **Case 1 ×11** (all via `TensorAccessor`). No Case 2, no borrowed-memory DFBs. |
| *Port work* — TensorParameter relaxation | **none** — `none` on both sheet rows; no custom hash to reconcile |
| *Port work* — TensorAccessor 3rd arg | **none — GREEN.** All 11 `TensorAccessor` constructions are 2-arg; no page-size override anywhere. |
| *Port work* — CB endpoints | **legal 1:1 ×13–15** · **self-loop ×5–8** · **cosmetic 1P+1C ×0–4** · **multi-binding ×0** · **dead-CB drop ×0** (config-dependent). Full census below. |

**CB endpoints** are dispositions, not gates. Recorded per `(CB, config)` below.

## Result

**GREEN at op level — every gate cleared. `METAL2_PORT_BRIEF.md` issued alongside this report.**

All five gate-bearing subjects pass:

- **Device 2.0** — all 8 own kernels DFB-native after #49173; all 3 donor headers clear; no holdovers.
- **Feature compatibility** — all four Appendix A entries `N/A`; the op uses none of them.
- **TTNN factory concept** — `Is able to port? = yes` on **both** factory rows, cross-check clean.
- **Offset base pointers** — zero `->address()` sites op-wide; no host-folded offsets to split.
- **TensorAccessor 3rd argument** — zero 3rd-arg sites; nothing to classify or drop.

The port work ahead is entirely mechanical: 11 Case-1 tensor bindings, a CB census with no multi-binding and no
dead CB, and four hardware-config / kernel-shape translation points (`unpack_to_dest_mode` re-keying, the
`dst_full_sync_en` polarity inversion, the two config-selected compute-kernel source files, and two
ternary-selected DFB handles in the compute kernels). No relaxation applies.

**#49173 made this port materially easier.** With the kernels already on `DataflowBuffer` and `dfb_*` naming,
the Metal 2.0 kernel-side delta shrinks to swapping the DFB *ids* for `dfb::name` binding tokens and the
address RTAs for `tensor::name` — the wrapper-object rewrite is already done, and the whitelist rule-7
metadata move (`get_tile_size(cb_id)` → a member getter) has already happened as `dfb.get_entry_size()`.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN — `yes` on both factory rows.** The readiness sheet is not
  fetchable in this session (the `claude.ai Google Drive` connector is unauthorized and `ToolSearch` finds no
  `mcp__claude_ai_Google_Drive__download_file_content`); the two rows were supplied directly by the owner and are
  recorded verbatim below. **The lightweight cross-check is clean: my independent code-derived verdict matches
  the sheet on every cheaply-checkable column, and the factory sets match one-to-one** (2 sheet rows ↔ 2
  factories in code, 1 per DeviceOperation — no phantom row, no missing row).

  | Column | `BatchNormOperation` / `BatchNormFactory` | `RunningStatistics` / `RunningStatisticsProgramFactory` | My code-derived cross-check |
  |---|---|---|---|
  | `Concept` | `descriptor` | `descriptor` | ✓ agrees — both factories expose `create_descriptor` returning a `ProgramDescriptor` (`batch_norm_device_operation.hpp:39-42`, `running_statistics_device_operation.hpp:36-39`); both DOps declare `program_factory_t` |
  | `Op Classification` | `PD (pointer-patching)` | same | ✓ consistent — every buffer rides the framework-patched `Buffer*` / `BufferBinding` channel |
  | `Custom hash (compute_program_hash)` | `no` | `no` | ✓ agrees — none in the directory (`RunningStatistics`'s was removed in `975decf0ac2` / #49871). See Recipe note 4 on `to_hash`. |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | `no` | ✓ agrees — hook absent from both device-ops |
  | `Override runtime args method? (PD and legacy)` | `no` | `no` | ✓ agrees — absent from both factories and both device-ops |
  | `Pybind descriptor` | `no` | `no` | ✓ agrees — `batch_norm_nanobind.cpp:70-84` binds `&ttnn::batch_norm` only |
  | `Smuggled pointer` | `no` | `no` | ✓ agrees — zero `->address()` sites; #49136 already swept this family |
  | `Is safe to port?` | **`yes`** | **`yes`** | (expert axis — not re-derived, per the recipe) |
  | **`Is able to port?`** | **`yes`** | **`yes`** | **gate cleared** |
  | `Model` | `other` | `other` | not a gate conjunct; recorded for completeness |
  | `TensorParameter relaxation` | `none` | `none` | ✓ consistent — no custom hash, so no relaxation is available or needed |
  | `Op-owned tensors?` | *(blank)* | *(blank)* | ✓ consistent — the `descriptor` form can't carry them |
  | `Secretly SPMD Workload?` | *(blank)* | *(blank)* | ✓ N/A — only meaningful when the factory returns a `WorkloadDescriptor` |

  **Cross-column invariants hold:** `get_dynamic_runtime_args == yes` is impossible on a `legacy device-op`
  (both are `descriptor`, both `no`); `Op-owned tensors? == yes` is impossible on a `descriptor` row (both
  blank). **Derivation confirmed** on each row: `safe=yes` ∧ `hash=no` ∧ `dynamic=no` ∧ `override=no` ∧
  `pybind=no` ∧ `concept=descriptor` ⇒ `Is able to port? = yes`.

  *Terminology note (recipe-current):* both descriptor entry points — `create_descriptor` and
  `create_workload_descriptor` — live under the single umbrella `ProgramDescriptorFactoryConcept`; there is no
  separate workload-descriptor *factory* concept. This op uses the plain `create_descriptor` form.

- **Device 2.0 (every kernel used): GREEN — and cleaner than at the previous audit.** After #49173 all eight
  own kernels are DFB-native: `Noc noc;`, `DataflowBuffer` wrapper objects, `TensorAccessor`, `CoreLocalMem`.
  Re-verified on the current tree: no `noc_async_read`/`noc_async_write` free calls, no `InterleavedAddrGen` /
  `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no raw semaphore addresses (the op
  uses **no semaphores at all**), no `get_read_ptr(cb_id)` / `get_write_ptr(cb_id)` free-function holdovers.

  **The op's own kernels now contain zero CB-index free functions.** The previous revision recorded
  `get_tile_size(cb_id)` as a sanctioned free function in all four dataflow kernels; #49173 replaced every site
  with the member getter `dfb.get_entry_size()` (`reader_batch_norm.cpp:37`, `writer_batch_norm.cpp:52,56,60,64,68`,
  `reader_running_statistics.cpp:38`, `writer_running_statistics.cpp:51,54,57,60`). Per the CB→DFB whitelist that
  is the sanctioned member for *"fifo_page_size / entry size"*; the kernels read it as a tile-byte count, which
  on Gen1 tile-formatted DFBs is the same value. **Already-made choice, not port work** — carry it as-is.

  Remaining free functions taking a buffer index, checked individually and **all cleared**:

  | Call | Site | Verdict |
  |---|---|---|
  | `fill_cb_with_value(dfb_id_one, one_u)` | `reader_running_statistics.cpp:56` | ✓ **not a holdover.** Donor (`ttnn/cpp/ttnn/kernel/dataflow/cb_fill_helpers.hpp:18-43`) builds its own wrapper internally and uses `cb.get_dataformat()` / `CoreLocalMem`. No wrapper for `dfb_id_one` is in scope at the call site and no member replacement exists, so neither holdover condition is met. The `uint32_t cb_id` parameter shape is the donor table's `✓ OK` row — `dfb::name`'s constexpr cast covers it. |
  | `sub_tiles_to_cb` / `mul_tiles_to_cb` / `add_tiles_to_cb` | `running_statistics_kernel.cpp:44-53` | ✓ same reasoning — donor (`ttnn/cpp/ttnn/kernel/compute/dest_format_helpers.hpp`) takes `uint32_t icb0/icb1/ocb` |
  | LLK compute primitives taking a buffer id (`add_tiles`, `mul_tiles`, `sub_tiles`, `pack_tile`, `copy_tile`, `binary_op_init_common`, `unary_op_init_common`, `pack_reconfig_data_format`, `reconfig_data_format*`, `typecast_tile`) | all 4 compute kernels | ✓ LLK compute surface, out of Device 2.0's data-movement scope |

  **Donor headers — all three clear this gate.** Note a new asymmetry worth recording: the op's kernels are now
  **DFB-native while two donors are still `CircularBuffer`-native internally**
  (`cb_fill_helpers.hpp:19` constructs `CircularBuffer cb(cb_id)`; `dest_format_helpers.hpp:14` still includes
  `api/dataflow/circular_buffer.h` and builds `CircularBuffer` objects at `:78-80,115-117,…`). **This is not a
  Device 2.0 violation** — `CircularBuffer` *is* the Device 2.0 kernel-side wrapper; CB→DFB is a Metal 2.0
  concern, and it is confined to the donors' *internals*. Both donors take plain `uint32_t` ids across the
  boundary, so the call sites are unaffected either way and **no donor edit is needed for this port**.
  `fill_tile_utils.hpp` is Device-2.0-**neutral** (44 sites, every one a bare `uint32_t l1_write_ptr` — no
  buffer handle, no addr-gen).

- **Feature compatibility: GREEN — every Appendix A entry `N/A`.** Re-scanned host code, factory code, and all
  kernel code on the current tree; no signal fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer` field on any `CBDescriptor`, no `remote_index` / `remote_cb_*` / `remote_circular_buffer.h`, no 4-arg `CreateCircularBuffer(..., global_cb)`. All 20 CB sites are plain `CBDescriptor` + `CBFormatDescriptor` literals. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Never set on any of the 20 `CBDescriptor` literals (defaults to 0). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. No borrowed-memory CBs at all — no `set_globally_allocated_address` and no `.buffer` on any descriptor. |
  | GlobalSemaphore | N/A | The op uses **no semaphores of any kind** — zero matches for `Semaphore` in the directory. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent: neither `tensor_args_t` carries a variable-count container (both are fixed named-tensor structs; `std::optional<Tensor>` is 0-or-1, not variadic). Kernel-level decider absent: every `get_compile_time_arg_val` index is a literal or a `constexpr` chain of `TensorAccessorArgs<N>::next_compile_time_args_offset()` (`reader_batch_norm.cpp:28`, `writer_batch_norm.cpp:42-43`, `reader_running_statistics.cpp:30`, `writer_running_statistics.cpp:41`) — no runtime-varying CTA index anywhere. |

- **CB endpoints (GATE-free): no multi-binding, no confirmed dead CB.** All three kernels of each factory run
  over the *same* `all_device_cores` range, so every node hosts all three and the census is uniform across
  nodes. Full per-`(CB, config)` inventory in the Port-work section.

- **Offset base pointers: GREEN.** `grep -rn "address()"` over the op directory returns **zero hits** on the
  current tree — there is no address expression to fold an offset into. Every tensor reaches the runtime-args
  list as a `Buffer*` through the `emplace_runtime_args(core, std::initializer_list<std::variant<uint32_t,
  Buffer*>>)` overload (`batch_norm_program_factory.cpp:87-124`, `running_statistics_program_factory.cpp:85-121`),
  which the framework auto-registers as `BufferBinding`s (`tt_metal/api/tt-metalium/program_descriptors.hpp:113,
  164, 191`). No host arithmetic on any base. No `ttnn::narrow`, no interior-base `MeshBuffer::create`.

  Reconciled against the dated triage `2026-07-19_offset_base_pointers.md`: `batch_norm` / `normalization` do not
  appear in its tables. That is the *"no fold, op not in the tables"* outcome — clean, handed straight to
  TensorParameter analysis. (Recorded explicitly so the clean result reads as *scanned*, not as *unlisted*.)

- **TensorAccessor 3rd argument: GREEN — the subject does not fire.** All **eleven** `TensorAccessor`
  constructions in the op are the **two-argument** form `TensorAccessor(args, addr)`:
  `reader_batch_norm.cpp:38`; `writer_batch_norm.cpp:53,57,61,65,69`; `reader_running_statistics.cpp:39`;
  `writer_running_statistics.cpp:52,55,58,61`. No explicit page size is passed anywhere, so there is no site to
  classify and nothing for the port to drop.

  Reconciled against the dated triage `2026-07-06_tensor_accessor_3rd_arg_triage.md`: the only `normalization`
  entry is `normalization_ln_rm_gb_post_allgather` (a layernorm post-allgather factory, Class 3) — a **different
  op**. `batch_norm` is correctly absent, matching my scan.

## Port-work summary  *(mirrors the brief)*

### Tensor bindings — all **Case 1**, all mechanical

Every binding follows the same shape: the factory pushes a `Buffer*` into the per-core RTA list; the kernel reads
that slot as a `uint32_t` and feeds it straight into a `TensorAccessor` built from a `TensorAccessorArgs<N>` CTA
block; all memory access then goes through the accessor. That is textbook **Case 1** — express as a
`TensorParameter` / `TensorBinding`, build `TensorAccessor(tensor::name)` in the kernel, and the RTA slot plus its
`TensorAccessorArgs` CTA plumbing both disappear. **No Case 2** (no kernel does hand-rolled NoC arithmetic on a
base pointer) and **no borrowed-memory / clean bindings** (the op declares no buffer-backed CBs).

`BatchNormOperation` — 6 bindings:

| Binding | Host site (RTA slot) | CTA block | Kernel accessor | Case |
|---|---|---|---|---|
| `input` | `batch_norm_program_factory.cpp:90` (reader RTA 1) | `:307` | `reader_batch_norm.cpp:38` | 1 |
| `batch_mean` | `:111` (writer RTA 0) | `:319` | `writer_batch_norm.cpp:53` | 1 |
| `batch_var` | `:112` (writer RTA 1) | `:321` | `writer_batch_norm.cpp:61` | 1 |
| `weight` *(optional)* | `:113` (writer RTA 2) | `:322-323` | `writer_batch_norm.cpp:65` | 1 |
| `bias` *(optional)* | `:114` (writer RTA 3) | `:324` | `writer_batch_norm.cpp:69` | 1 |
| `output` | `:115` (writer RTA 4) | `:320` | `writer_batch_norm.cpp:57` | 1 |

`RunningStatistics` — 5 bindings:

| Binding | Host site (RTA slot) | CTA block | Kernel accessor | Case |
|---|---|---|---|---|
| `batch_mean` | `running_statistics_program_factory.cpp:88` (reader RTA 1) | `:351` | `reader_running_statistics.cpp:39` | 1 |
| `batch_var` | `:109` (writer RTA 0) | `:364` | `writer_running_statistics.cpp:52` | 1 |
| `running_mean` *(optional, **read-modify-write in place**)* | `:110` (writer RTA 1) | `:366-367` | `writer_running_statistics.cpp:58` | 1 |
| `running_var` *(optional, **read-modify-write in place**)* | `:111` (writer RTA 2) | `:368-369` | `writer_running_statistics.cpp:61` | 1 |
| `output` | `:112` (writer RTA 3) | `:365` | `writer_running_statistics.cpp:55` | 1 |

Two shapes the porter should have in hand up front:

- **Optional bindings are delivered as a literal `0u`, not as an absent arg.** When `weight`/`bias`/`running_mean`/
  `running_var` is absent the factory pushes `std::variant<uint32_t, Buffer*> arg = 0u`
  (`batch_norm_program_factory.cpp:101-108`, `running_statistics_program_factory.cpp:99-106`) and pairs it with
  `TensorAccessorArgs(nullptr)` (`batch_norm_program_factory.cpp:322-324`,
  `running_statistics_program_factory.cpp:366-369`). The kernel still *constructs* the accessor unconditionally
  and simply never uses it, guarded by a `..._has_value` CTA. Because presence is a compile-time branch keyed by
  that CTA, the port should simply **not declare the `TensorParameter` in the absent configuration** rather than
  trying to bind a null tensor.
- **`running_mean` / `running_var` are read *and* written through the same binding.**
  `writer_running_statistics.cpp:86-99` reads and `:102-110` writes back to the same `TensorAccessor`
  (in-place update; `batch_norm.cpp:124-128` documents the ordering constraint this creates). One
  `TensorParameter` covers both directions; do not split it into an in-binding plus an out-binding.

### TensorParameter relaxation

**none.** `TensorParameter relaxation = none` on both sheet rows, and there is no custom `compute_program_hash`
to reconcile against.

### TensorAccessor 3rd arg

**none.** No site exists.

### CB endpoints — per `(CB, config)`

All CBs are allocated over `all_device_cores` and all three kernels of the owning factory run over that same
range, so the census is identical on every node. **No CB anywhere in this op reaches ≥3 touchers or doubles a
FIFO role — the multi-binding advanced option is never needed.**

**`BatchNormFactory`** (`batch_norm_program_factory.cpp`) — R = reader, W = writer, C = compute. Compute sites
cite the SFPU variant (the default); the non-SFPU variant has the same touchers at
`batch_norm_kernel.cpp:46,47,60,61,63,64,66,69,73,74,89,90,112,113,124,125,128,129,131,134,170,205`:

| CB | Index | Touchers | Disposition |
|---|---|---|---|
| `input_tensor_cb` | `c_0` | R produces (`reader_batch_norm.cpp:66,69`), C consumes as `dfb_other` (`batch_norm_sfpu_kernel.cpp:89,114`) | **legal 1:1** |
| `batch_mean_tensor_cb` | `c_1` | W produces (`writer_batch_norm.cpp:85,93`; its `get_write_ptr()` at `:89/:91` is a same-kernel peek covered by the PRODUCER binding), C consumes as `dfb_bcast` (`batch_norm_sfpu_kernel.cpp:80,187`) | **legal 1:1** |
| `output_tensor_cb` | `c_2` | *no-typecast config:* C produces (`batch_norm_sfpu_kernel.cpp:142,159`), W consumes (`writer_batch_norm.cpp:133,137`) | **legal 1:1** |
| `output_tensor_cb` | `c_2` | *typecast config* (`needs_output_typecast`, SFPU only): W is redirected to `c_9`, so **C is the only toucher** — it both produces (`:142,159`) and consumes (`:164,183`) | **self-loop** |
| `writer_cb` | `c_9` | *typecast config only.* C produces (`batch_norm_sfpu_kernel.cpp:166,184`), W consumes (`writer_batch_norm.cpp:133,137`) | **legal 1:1** |
| `batch_var_tensor_cb` | `c_3` | W produces (`writer_batch_norm.cpp:96,105`), C consumes (`batch_norm_sfpu_kernel.cpp:58,78`) | **legal 1:1** |
| `eps_cb` | `c_4` | R produces (`reader_batch_norm.cpp:46,54`), C consumes (`batch_norm_sfpu_kernel.cpp:235,274`) | **legal 1:1** |
| `weight_tensor_cb` | `c_5` | *weight present:* W produces (`writer_batch_norm.cpp:108,116`), C consumes (`batch_norm_sfpu_kernel.cpp:83,190`) | **legal 1:1** |
| `weight_tensor_cb` | `c_5` | *weight absent:* allocated unconditionally (`:260-269`) and **named** by both W (`writer_batch_norm.cpp:49` wrapper + `:64` `get_entry_size()`) and C (`batch_norm_sfpu_kernel.cpp:49` wrapper; in the non-SFPU variant the FIFO calls are compiled in under a *runtime* `if`), but **no FIFO or pointer access executes** | **assign cosmetic 1P+1C** (W PRODUCER, C CONSUMER) — **confirmed by the owner**; see *Decisions taken* 2 |
| `bias_tensor_cb` | `c_6` | symmetric to `c_5` (`:270-279`; W `:50`/`:68`; C `:50`) | as above |
| `den_cb` | `c_7` | C only — produces (`batch_norm_sfpu_kernel.cpp:57,77`) and consumes (`:81,188`) | **self-loop** |
| `temp_1_cb` | `c_8` | C only — reached as `dfb_affine_or_out` / `dfb_scaled_output` / `dfb_tmp_1` (`batch_norm_sfpu_kernel.cpp:42-43,90,115,118,137,141,160`) | **self-loop** (also when weight *and* bias are both absent, where it is named but unused) |

**`RunningStatisticsProgramFactory`** (`running_statistics_program_factory.cpp`) — compute sites cite the SFPU
variant; the non-SFPU variant's touchers are at `running_statistics_kernel.cpp:36,37,44-53,57,59,62,63`:

| CB | Index | Touchers | Disposition |
|---|---|---|---|
| `batch_mean_tensor_cb` | `c_0` | R produces (`reader_running_statistics.cpp:73,76`), C consumes (`running_statistics_sfpu_kernel.cpp:94,196`; non-SFPU via `mul_tiles_to_cb`'s internal `pop_front`) | **legal 1:1** |
| `batch_var_tensor_cb` | `c_1` | W produces (`writer_running_statistics.cpp:79,82`), C consumes (`running_statistics_sfpu_kernel.cpp:219,237`) | **legal 1:1** |
| `output_tensor_cb` | `c_2` | C produces (`running_statistics_sfpu_kernel.cpp:95,294`; non-SFPU `running_statistics_kernel.cpp:59` — see Misc anomaly 1), W consumes (`writer_running_statistics.cpp:144,148`) | **legal 1:1** |
| `old_running_mean_tensor_cb` | `c_3` | *present:* W produces (`writer_running_statistics.cpp:86,99`), C consumes (`running_statistics_sfpu_kernel.cpp:138,156`) | **legal 1:1** |
| `old_running_mean_tensor_cb` | `c_3` | *absent:* named by W (`:46` wrapper, `:57` `get_entry_size()`) and by the SFPU C (`:72` wrapper) but never accessed; in the **non-SFPU** variant C's references are `if constexpr`-elided entirely | **assign cosmetic 1P+1C** — see *Decisions taken* 2 |
| `old_running_var_tensor_cb` | `c_4` | symmetric to `c_3` (W `:47`/`:60`; C `:73`) | as above |
| `momentum_cb` | `c_5` | R produces (`reader_running_statistics.cpp:59,67`), C consumes (`running_statistics_sfpu_kernel.cpp:86,296`) | **legal 1:1** |
| `one_cb` | `c_6` | R produces via `fill_cb_with_value` (reserve+push inside the donor, `cb_fill_helpers.hpp:20,42`), C consumes (`running_statistics_sfpu_kernel.cpp:87,297`) | **legal 1:1** |
| `updated_m_cb` | `c_7` | *no mean typecast:* C produces (`running_statistics_sfpu_kernel.cpp:162,183`), W consumes (`writer_running_statistics.cpp:102,110`) | **legal 1:1** |
| `updated_m_cb` | `c_7` | *mean typecast:* W is redirected to `c_12`; **C is the only toucher** — produces (`:162,183`) and consumes inside `maybe_typecast_stat` (`:20,39`) | **self-loop** |
| `updated_v_cb` | `c_8` | symmetric to `c_7` (`:264,281`) | **legal 1:1** / **self-loop** |
| `wm_cb` | `c_12` | *mean-typecast config only.* C produces (`running_statistics_sfpu_kernel.cpp:22,40`), W consumes (`writer_running_statistics.cpp:102,110`) | **legal 1:1** |
| `wv_cb` | `c_13` | symmetric to `c_12` (W `:131,139`) | **legal 1:1** |
| `tmp1_cb` | `c_9` | C only (`running_statistics_sfpu_kernel.cpp:99,115,137,157`) | **self-loop** |
| `tmp2_cb` | `c_10` | C only (`:118,134,160,193`) | **self-loop** |
| `tmp3_cb` | `c_11` | C only (`:139,154,161,192`) | **self-loop** |

**Roll-up:** legal 1:1 ×13–15 · self-loop ×5–8 (config-dependent) · **cosmetic 1P+1C ×0–4** (only the
optional-tensor-absent configs) · **multi-binding ×0** · **dead-CB drop ×0**.

**No dead CB is reported.** Every CB is at minimum *named* by a kernel (wrapper construction and/or a
`get_entry_size()` metadata read), so none is truly zero-referenced; the ambiguous class is resolved as a
cosmetic 1P+1C assignment per *Decisions taken* 2, and nothing is dropped.

### Additional port-time shapes worth carrying forward

- **`unpack_to_dest_mode` is a CB-id-indexed vector** (`batch_norm_program_factory.cpp:352-368`,
  `running_statistics_program_factory.cpp:394-411`): a `std::vector<UnpackToDestMode>` of length
  `NUM_CIRCULAR_BUFFERS` with selected slots set to `UnpackToDestFp32`. Metal 2.0's
  `ComputeHardwareConfig::unpack_modes` re-keys this from CB id to **DFB name** and flips the sense of the value —
  a known silent-failure surface. Translate it deliberately, entry by entry. **Note the conditional entry added
  by #51313** (`:365-367`): `output_tensor_cb` joins the set only when `needs_output_typecast`, so the ported
  config must stay conditional too.
- **`dst_full_sync_en` inverts.** Both factories set `ComputeConfigDescriptor::dst_full_sync_en` from
  `get_compute_kernel_config_args` (`batch_norm_program_factory.cpp:397`,
  `running_statistics_program_factory.cpp:447`). The Metal 2.0 field is `double_buffer_dest` with the **opposite**
  polarity; a straight copy is silently wrong.
- **The compute-kernel *source file* is config-selected** via `fmt::format` on `(fp32_dest_acc_en || any_float32)`.
  Both source files must be ported together; they are not variants of one file.
- **Two compute-kernel DFB handles are chosen by a runtime ternary.**
  `auto dfb_affine_or_out = (weight_has_value || bias_has_value) ? dfb_tmp_1 : dfb_output_0;` and
  `auto dfb_scaled_output = (bias_has_value) ? dfb_tmp_1 : dfb_output_0;`
  (`batch_norm_kernel.cpp:31-32`, `batch_norm_sfpu_kernel.cpp:42-43`) — inside `batchnorm_bcast_tiles`, whose
  `weight_has` / `bias_has` parameters are plain runtime `uint32_t` even though every caller passes a `constexpr`.
  This is expressible in Metal 2.0 (`dfb::name`'s `constexpr operator uint32_t()` makes both ternary arms
  `uint32_t`), but it means **the compute kernel must bind both `temp_1_cb` and `output_tensor_cb`
  unconditionally** — which the census above already assumes. Flagged so the porter does not narrow the bindings
  to whichever arm the config selects.
- **A local compute helper takes `DataflowBuffer&`:** `maybe_typecast_stat(DataflowBuffer& src_obj, ...)`
  (`running_statistics_sfpu_kernel.cpp:15-18`). This is an in-file `ALWI` helper, **not** a donor, so it does not
  hit the donor table's `CircularBuffer&` ⭐ flag — the port updates the signature alongside the kernel. (#49173
  already changed this parameter from `CircularBuffer&`.)

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** **none.** I ran the three-face hunt explicitly and found no
  hidden second writer (the op has no semaphores at all, so the semaphore-gated raw co-fill face cannot occur),
  no multi-reader CB, and no dual-instance work-split (each factory instantiates each kernel source exactly
  once; the reader/writer/compute triple is three *distinct* sources over one core range). The one raw-pointer
  write pattern present — `fill_tile_with_first_element*(dfb_*.get_write_ptr())` in
  `writer_batch_norm.cpp:89,91,101,103,112,114,124,126` and `writer_running_statistics.cpp:95,97,124,126` — is
  performed by the *same* kernel that FIFO-produces that CB, so it is a same-binding peek and adds no toucher.
- **Cross-op / shared kernels: none — no fork, no sunset list.** The op **owns all 8 kernel sources and no other
  op binds any of them**, so none of the shared-kernel rungs apply (not borrowed, not lent, not intra-op). The
  locational `_metal2` check was run on all four relevant directories and found no sibling fork. Three donor
  headers are consumed by `#include` (function-call escape, a different mechanism) and need no edit — detail in
  *Team-only*.
- **RTA varargs:** **none.** Every `get_arg_val<uint32_t>` in all 8 kernels uses a literal constant index
  (reader `0..8`, batch-norm writer `0..11`, running-stats writer `0..10`, compute `0..2` / `0`). No counted loop
  over args, no running `arg_index++`, no data-selected index. All args are nameable — the port should name every
  one.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** No ⭐ entries, no ✗ entries, no scheduling blocker. Three header donors, and
every consumed function's signature is a shape the Metal 2.0 binding tokens cross without donor-side work.

**Borrowed kernel files (file-path instantiation): none.** Every `KernelDescriptor::kernel_source` in both
factories points inside this op's own directory, and the census (`grep -rl <kernel-filename>
ttnn/cpp/ttnn/operations/`) returns **no external binder** for any of the 8 — the only outside hits are
`ttnn/ttnn.egg-info/SOURCES.txt`, a build artifact the recipe's disambiguation rule discards. So there is **no
`_metal2` fork to reuse or create, and no sunset list to report.** The locational fork check (`ls` for a
same-stem `_metal2` sibling) was run on `device/kernels/dataflow/`, `device/kernels/compute/`,
`ttnn/cpp/ttnn/kernel/{dataflow,compute}/` and `eltwise/binary_ng/device/kernels/dataflow/` — none found.

| Op kernel | Donor file | Donor class |
|---|---|---|
| `reader_batch_norm.cpp` | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp` | 6 — cross-family |
| `writer_batch_norm.cpp` | same | 6 — cross-family |
| `reader_running_statistics.cpp` | same | 6 — cross-family |
| `writer_running_statistics.cpp` | same | 6 — cross-family |
| `reader_running_statistics.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/cb_fill_helpers.hpp` | 3 — shared kernel pool (`ttnn/kernel/`) |
| `batch_norm_kernel.cpp` | `ttnn/cpp/ttnn/kernel/compute/dest_format_helpers.hpp` | 3 — shared kernel pool |
| `batch_norm_sfpu_kernel.cpp` | *(none — LLK only)* | — |
| `running_statistics_kernel.cpp` | `ttnn/cpp/ttnn/kernel/compute/dest_format_helpers.hpp` | 3 — shared kernel pool |
| `running_statistics_sfpu_kernel.cpp` | same | 3 — shared kernel pool |

Per-call shape analysis (only functions this op actually calls):

| Donor | Function | Signature shape | Status |
|---|---|---|---|
| `fill_tile_utils.hpp` | `fill_with_val_bfloat16(uint32_t l1_write_ptr, uint32_t)` | raw L1 pointer | ✓ no resource handle to translate |
| `fill_tile_utils.hpp` | `fill_with_val<Elems, ScalarT>(uint32_t l1_write_ptr, ScalarT)` | raw L1 pointer | ✓ |
| `fill_tile_utils.hpp` | `fill_tile_with_first_element<T>(uint32_t l1_write_ptr)` | raw L1 pointer | ✓ |
| `fill_tile_utils.hpp` | `fill_tile_with_first_element_bfloat16(uint32_t l1_write_ptr)` | raw L1 pointer | ✓ |
| `cb_fill_helpers.hpp` | `fill_cb_with_value(uint32_t cb_id, uint32_t, int32_t)` | `uint32_t cb_id` | ✓ OK — `dfb::name`'s constexpr cast covers it |
| `dest_format_helpers.hpp` | `pack_tile_with_dt(uint32_t, uint32_t icb)` | `uint32_t cb_id` | ✓ OK |
| `dest_format_helpers.hpp` | `add_tiles_init_with_dt` / `sub_tiles_init_with_dt` / `mul_tiles_init_with_dt` | `uint32_t cb_id` ×2 | ✓ OK |
| `dest_format_helpers.hpp` | `ckernel::{add,sub,mul}_tiles_to_cb(uint32_t icb0, uint32_t icb1, uint32_t ocb, ...)` | `uint32_t cb_id` ×3 | ✓ OK |

No `Semaphore` / `uint32_t sem_id` / `sem_addr` shapes (the op has no semaphores). No `TensorAccessor<DSpec>`,
`TensorAccessorArgs<N>`, or NTTP-CTA-offset donor shapes — accessors are constructed and consumed entirely inside
the op's own kernels. No old-style addr-gen (Shape 4). No `CircularBuffer` **parameter** on any donor the op
calls (the one `DataflowBuffer&` signature in the op is a same-file helper — see Port-work).

**Coordination note, not a blocker.** `fill_tile_utils.hpp` is *broadly* shared — ~35 kernel files across
`eltwise/binary_ng`, `eltwise/ternary`, `experimental/quasar/binary_ng`, and this op include it. But because
every consumed function takes a bare L1 pointer rather than a buffer or tensor handle, **a Metal 2.0 port of
batch_norm requires no edit to this donor**, so it induces no coordination cost.
`cb_fill_helpers.hpp` (1 consumer — this op) and `dest_format_helpers.hpp` (4 consumers) are likewise
`uint32_t cb_id`-shaped and need no donor change. Recorded for the team: both `ttnn/cpp/ttnn/kernel/` donors are
still `CircularBuffer`-native internally while their callers here are now DFB-native — a tidy-up for the
kernel-pool owners, invisible at the call boundary and **not** port work.

### Relaxation candidates

The sheet lists `TensorParameter relaxation = none` for both factories, and there is no custom
`compute_program_hash` to mine — so there is **no relaxation for the port to apply**, and the
`Hash consistent with TP relaxation?` / `Dynamic args consistent with TP relaxation?` columns are correctly blank.
One roadmap candidate surfaced anyway, recorded for the team:

**FALLIBLE — a candidate to verify, default strict.** `BatchNormOperation::operation_attributes_t::to_hash()`
(`device/batch_norm_device_operation.cpp:121-123`) hashes `{eps, memory_config, get_dtype(), compute_kernel_config}`.
The reflection default would additionally distinguish the raw `input_dtype` and `dtype` members separately; the
hand-written version collapses them to `get_dtype() == dtype.value_or(input_dtype)`. As written this looks benign
— `input_dtype` is always set from `input.dtype()` (`:143`), which the tensor-args hash already covers — but it
*is* a deliberate narrowing of the attribute key and belongs on the relaxation roadmap rather than being assumed
correct. `RunningStatistics::operation_attributes_t` has **no** `to_hash`, so the two DOps hash their attributes
by different mechanisms despite near-identical attribute structs — an asymmetry worth a second look.

### TTNN factory analysis

Sheet rows are transcribed in *Gate detail*; both read `Is able to port? = yes`. The facts that feed the port's
TTNN ProgramFactory wiring, with code evidence:

- **Op-owned tensors:** none (sheet blank; confirmed in code). Neither factory returns a `WorkloadDescriptor`;
  both return a plain `ProgramDescriptor`. `BatchNormOperation` supports a caller-supplied preallocated `output`
  (`batch_norm_device_operation.cpp:113-118`), but that is an ordinary optional output tensor, **not** an
  op-owned tensor in the `WorkloadDescriptor::buffers` sense.
- **MeshWorkload need:** none — single-program, SPMD by construction. `Secretly SPMD Workload?` correctly blank.
- **Pybind `create_descriptor`:** `no`. `batch_norm_nanobind.cpp:70-84` binds only `&ttnn::batch_norm`.
- **Other risky pybind:** none observed; `Is safe to port? = yes` with no `warning`.
- **Custom hash:** `no`. See the `to_hash` caveat above (Recipe note 4).
- **`get_dynamic_runtime_args`:** `no`. **`override_runtime_arguments`:** `no`.
- **Target concept:** **`ProgramSpecFactoryConcept`** (base form — cache hit is `UpdateTensorArgs` only),
  `ttnn/api/ttnn/operation_concepts.hpp:119`. The recipe docs now name the same concept, so there is no
  divergence to work around (see *Resolved* 3).
- **The concept flip is atomic.** `ProgramSpecFactoryConcept` requires `!ProgramDescriptorFactoryConcept`
  (`:116-119`) and `all_factories_valid` (`:176-182`) permits exactly one of the five concepts per factory. Both
  DOps declare a `program_factory_t` variant holding a single factory struct, so `create_descriptor` must be
  removed from each factory in the same change that adds `create_program_artifacts`.

## Misc anomalies  *(team-only, non-gating, not porter work)*

1. **`running_statistics_kernel.cpp:59` — `push_back` on `dfb_out0` with no matching `reserve_back`.**
   The non-SFPU compute kernel packs and pushes the output tile (`:57-59`) without ever calling
   `dfb_out0_obj.reserve_back(1)`. The writer does `wait_front`/`pop_front` on that CB
   (`writer_running_statistics.cpp:144,148`), so the FIFO has a consumer but no producer-side back-pressure: with
   `num_tiles` greater than the CB depth (2 tiles, `running_statistics_program_factory.cpp:212-221`) the producer
   can outrun the consumer and overwrite unread data. The SFPU sibling does it correctly
   (`running_statistics_sfpu_kernel.cpp:95` reserves, `:294` pushes). **Masked today** because
   `default_fp32_acc = true` (`batch_norm_utils.cpp:31`) makes the SFPU variant the default; the buggy kernel is
   reached only when a caller passes `fp32_dest_acc_en = false` *and* all tensors are bf16. *(Survived #49173 —
   that change was a mechanical CB→DFB rename and did not touch the missing reserve.)*
2. **`running_statistics_kernel.cpp:40-58` — nested `tile_regs_acquire()`.** The outer
   `tile_regs_acquire()` (`:40`) / `tile_regs_commit()` (`:55`) / `tile_regs_wait()` (`:56`) /
   `tile_regs_release()` (`:58`) bracket wraps calls to `sub_tiles_to_cb` / `mul_tiles_to_cb` /
   `add_tiles_to_cb` (`:44-53`), **each of which performs its own full acquire→commit→wait→release cycle**
   (`dest_format_helpers.hpp:85-99` and siblings). The `pack_tile(0, dfb_out0)` at `:57` therefore packs whatever
   DST reg 0 holds after the last inner release, not a value this bracket produced.
3. **`running_statistics_kernel.cpp:57` — output packed from an undefined DST when both optionals are absent.**
   With `old_running_mean_has_value == false` and `old_running_var_has_value == false`, both `if constexpr` blocks
   (`:43-54`) are elided, yet `pack_tile(0, dfb_out0)` still runs and pushes a tile of uninitialized DST content
   to the op's output tensor. The SFPU variant has the same structural hole (`running_statistics_sfpu_kernel.cpp:95`
   reserves and `:294` pushes with nothing packed in between). This is reachable: `ttnn::batch_norm` calls
   `ttnn::prim::running_statistics` whenever `training == true` (`batch_norm.cpp:130-133`) and the nanobind
   docstring makes both running-stat tensors optional in training mode.
4. **Eight dead RTAs — `cHt` and `cWt` are pushed to every dataflow kernel and never read.**
   Reader RTAs 9 and 10 vs. a kernel reading only `0..8`: `batch_norm_program_factory.cpp:98-99` /
   `reader_batch_norm.cpp:15-23`, and `running_statistics_program_factory.cpp:96-97` /
   `reader_running_statistics.cpp:16-24`. Batch-norm writer RTAs 12 and 13 vs. a kernel reading `0..11`:
   `batch_norm_program_factory.cpp:123-124` / `writer_batch_norm.cpp:14-25`. Running-stats writer RTAs 11 and 12
   vs. a kernel reading `0..10`: `running_statistics_program_factory.cpp:120-121` /
   `writer_running_statistics.cpp:14-24`. The `num_reader_args = 11` / `num_writer_args = 14` / `13` constants used
   to zero-fill idle cores (`batch_norm_program_factory.cpp:61-62`, `running_statistics_program_factory.cpp:60-61`)
   encode the same inflated counts.
5. **Unguarded `0 / 0` on idle cores in both readers.** Cores in neither work group receive an all-zero RTA
   block (`batch_norm_program_factory.cpp:73-79`). The readers have no `num_tiles == 0` early-out (the
   batch-norm compute kernels do — `batch_norm_kernel.cpp:145`, `batch_norm_sfpu_kernel.cpp:205` — and the
   running-statistics compute kernels have none either), so `tiles_per_batch = HtWt * C` is `0` and
   `start_tile_id / tiles_per_batch` (`reader_batch_norm.cpp:41`, `reader_running_statistics.cpp:42`) is an
   integer divide by zero. Benign on RISC-V (unsigned `div` by zero yields all-ones, and the subsequent loop
   bound `n < N == 0` is immediately false), but it is a landmine that a target-ISA change or a UB-sensitive
   toolchain would expose. Idle cores also still fill and push the `eps` / `momentum` / `one` CBs.
6. **`b_num_tiles_per_cb` is a pointless alias.** Initialized to `num_tiles_per_cb` and never reassigned in either
   factory (`batch_norm_program_factory.cpp:192`, `running_statistics_program_factory.cpp:189`), yet used for a
   subset of CBs — implying an intent to differ that was never implemented.
7. **`packed_scalar_eps` / `packed_scalar_momentum` are loop-invariant but recomputed per core.**
   `batch_norm_program_factory.cpp:83-85`, `running_statistics_program_factory.cpp:82-84`. Host-side only,
   trivial cost; noted for tidiness.
8. **Dead CTAs in the non-SFPU compute variants.** Each factory pushes one CTA list used by both compute
   variants: `batch_norm_kernel.cpp` reads indices `0..10` of 15 pushed
   (`batch_norm_program_factory.cpp:370-385`), and `running_statistics_kernel.cpp` reads `0..13` of 19
   (`running_statistics_program_factory.cpp:416-435`). Harmless (unread trailing CTAs), but the port will want to
   size the compile-time-arg schema per variant rather than carrying the union.

## Per-DeviceOperation attribution

| Field | `BatchNormOperation` | `RunningStatistics` |
|---|---|---|
| Factory | `BatchNormFactory` | `RunningStatisticsProgramFactory` |
| Device 2.0 | GREEN (DFB-native after #49173) | GREEN (DFB-native after #49173) |
| Appendix A | all `N/A` | all `N/A` |
| `Is able to port?` (sheet) | **yes** | **yes** |
| `Is safe to port?` (sheet) | yes | yes |
| Concept (current → target) | `descriptor` → `ProgramSpecFactoryConcept` | `descriptor` → `ProgramSpecFactoryConcept` |
| Custom `compute_program_hash` | no | no (removed in `975decf0ac2` / #49871) |
| Custom attribute `to_hash()` | **yes** (`batch_norm_device_operation.cpp:121-123`) — *not* scored as a custom hash by the sheet | no |
| `get_dynamic_runtime_args` / `override_runtime_arguments` / pybind descriptor | no / no / no | no / no / no |
| TensorParameter relaxation | none | none |
| Offset base pointers | GREEN (0 `->address()`) | GREEN (0 `->address()`) |
| TensorAccessor 3rd arg | GREEN (6 sites, all 2-arg) | GREEN (5 sites, all 2-arg) |
| Tensor bindings | 6 × Case 1 | 5 × Case 1 (2 of them read-modify-write in place) |
| CBs | 9–10 (config-dependent) | 12–14 (config-dependent) |
| CB dispositions | legal 1:1 ×6–7 · self-loop ×2–3 · cosmetic 1P+1C ×0–2 · multi-binding ×0 | legal 1:1 ×7–8 · self-loop ×3–5 · cosmetic 1P+1C ×0–2 · multi-binding ×0 |
| Borrowed kernel files / `_metal2` fork | none / none | none / none |
| Misc anomalies | 4, 5, 6, 7, 8 | 1, 2, 3, 4, 5, 6, 7, 8 |
| Verdict | **GREEN — all gates cleared** | **GREEN — all gates cleared** |

## Resolved  *(kept for the record — nothing here is open)*

1. **Readiness sheet unfetchable; rows supplied by the owner — gate GREEN.** The `claude.ai Google Drive`
   connector is unauthorized in this session and the OAuth flow cannot be run from a non-interactive session, so
   the sheet could not be pulled. The owner supplied both rows; they are transcribed in *Gate detail* and every
   conjunct reads clean. **The access gap itself is unresolved** — see Recipe note 1.
2. **Named-but-unaccessed CBs — cosmetic 1P+1C, confirmed by the owner.** `weight_tensor_cb` (`c_5`),
   `bias_tensor_cb` (`c_6`), `old_running_mean_tensor_cb` (`c_3`) and `old_running_var_tensor_cb` (`c_4`) under
   their optional-absent configurations are bound **writer PRODUCER, compute CONSUMER**; the roles are cosmetic
   on Gen1 and cost nothing at runtime. **Nothing is dropped as dead.** The recipe gap this exposed is Recipe
   note 2.
3. **Factory-concept rename — the recipe has caught up; no divergence remains.** PR #50942 (squash-merged
   `78cff43bb04`, 2026-07-24) renamed `MetalV2FactoryConcept` → `ProgramSpecFactoryConcept` and added
   `CustomProgramSpecFactoryConcept`, but changed no file under `metal_2.0/`, so the doc set lagged. The previous
   revision of this report carried that as a standing divergence and told the porter to ignore the docs.
   `4d77c80a2f7 — "docs(metal_2.0): follow main's MetalV2FactoryConcept -> ProgramSpecFactoryConcept rename"`
   has since landed: **docs and code now agree**, and the porter needs no special instruction. The
   corresponding recipe note is retired.
4. **Bundling — one combined report, confirmed by the owner.** Both DOps stay in a single audit/brief pair with
   per-DeviceOperation attribution retained.
5. **`unpack_to_dest_mode` inconsistency — fixed in code (#51313).** Was Misc anomaly 6 of the previous
   revision. `output_tensor_cb` (`c_2`) is now added to the `UnpackToDestFp32` set conditionally on
   `needs_output_typecast` (`batch_norm_program_factory.cpp:365-367`). That was the substantive half: `c_2` holds
   `Float32` in the typecast path *and* is unpacked there (`copy_tile(dfb_output_0, …)`,
   `batch_norm_sfpu_kernel.cpp:171`), so its omission cost precision in the typecast step. `c_9` was correctly
   **not** added — it is only ever a pack destination (`pack_tile`, `:178`), so an entry would be inert. The
   port now simply translates the list as it stands, conditional entry included.

## Recipe notes  *(omit-if-none section; four remain)*

1. **The readiness-sheet dependency still has no documented fallback.** The TTNN factory concept prerequisite
   says *"Pull a fresh copy of the sheet every run"* and *"Do not fetch or cross-check in a subagent — the Drive
   connector authorizes only in the main session"*, but says nothing about the connector being unavailable in the
   **main** session too. The nearest rule is *"the op has no row → spreadsheet is broken → GATE → readiness-sheet
   owner"*, which is about the sheet being *wrong* — a different situation with a different fix. **This op is the
   worked example of the cost:** an earlier revision of this report shipped RED purely because a spreadsheet
   could not be read, while every conjunct was in fact `yes` — exactly the *"too-conservative RED misroutes work
   to a prereq team that doesn't need it"* failure the recipe warns about. Suggest (a) an explicit
   **UNRESOLVED / blocked-on-data** verdict distinct from RED, and (b) **moving the sheet fetch to the opening
   steps** so a fetch failure prompts the launcher for the rows *before* the analysis runs.
2. **The endpoint census still has no row for "named but not accessed."** The census defines a toucher as
   FIFO-produce, FIFO-consume, or raw-pointer memory access, then says a `(0, 0)` CB *"cannot be carried into
   Metal 2.0 at all"* and **must** be dropped. Optional-tensor ops break that dichotomy:
   `writer_batch_norm.cpp:49,64` constructs `DataflowBuffer dfb_weight(dfb_id_weight)` and calls
   `dfb_weight.get_entry_size()` unconditionally, while every FIFO op on that buffer sits under
   `if constexpr (weight_has_value)`. Zero touchers by the letter of the rule, yet undroppable in practice — and
   now *more* likely to recur, since #49173-style migrations replace `get_tile_size(cb_id)` with a **member**
   getter, which is exactly the metadata-read shape that produces this state. The owner has ruled on this
   instance (cosmetic 1P+1C, never drop — *Resolved* 2); suggest writing that in as the standing rule.
3. **The census's "per node" framing needs a companion "per compile-time branch" axis.** *Classify per
   instantiation, not once for the op* covers config-driven variation (sharding, split-reader), which reads as
   *host-side* configuration. This op's variation is driven by `if constexpr` on presence CTAs *inside* one
   kernel, and by a `fmt::format`-selected kernel **source file** — both change the census, and neither is what
   "instantiation" evokes. Worth naming compile-time-branch variation explicitly.
4. **The `Custom hash` cross-check is specified narrowly enough to miss a real hash customization.** The rule is
   *"grep the device-op for a `compute_program_hash` override"*, and it names one escape hatch (the
   pybind-renamed hook). `operation_attributes_t::to_hash()` is a second, independent mechanism —
   `tt_stl/tt_stl/reflection.hpp:285-288, 1314-1318, 1555-1556` dispatches to it in place of the reflection
   default — and it is a plain public method on the attributes struct, not the device-op. The sheet scored this
   op `no` and I agree, so nothing was mis-gated; but the rule as written wouldn't have caught it either way.
   Suggest widening to *"grep for `compute_program_hash` **or** a `to_hash()` on `operation_attributes_t`"* and
   saying which one the sheet's column tracks.

*Two notes from the previous revision are retired as done:* the `MetalV2FactoryConcept` doc-staleness note
(actioned by `4d77c80a2f7`), and the *"add an opening staleness check"* note — this re-audit ran exactly that
check first, and it is what caught #49173 invalidating every kernel `file:line` in the prior revision. Also
retired: the *"donor-shape table has no raw-L1-pointer row"* note and the *"kernel source selected by
`fmt::format`"* note are folded into note 3 rather than repeated.
