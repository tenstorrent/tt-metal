# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard`

Single device operation, `descriptor` concept, eight program-factory variants (five factory types):

- **`ReshardDeviceOperation`** (`device/reshard_device_operation.{hpp,cpp}`)
  - `ReshardSameWidthFactory<local_is_output=true>` (`reshard_program_factory_same_width.cpp`)
  - `ReshardSameWidthFactory<local_is_output=false>` (`reshard_program_factory_same_width.cpp`)
  - `ReshardSameHeightFactory<local_is_output=true>` (`reshard_program_factory_same_height.cpp`)
  - `ReshardSameHeightFactory<local_is_output=false>` (`reshard_program_factory_same_height.cpp`)
  - `ReshardGenericFactory` (`reshard_program_factory_generic.cpp`)
  - `NdReshardCopyPagesFactory` (`nd_reshard_program_factory_copy_pages.cpp`)
  - `NdReshardCopyLocalShardFactory<local_is_input=true>` (`nd_reshard_program_factory_copy_local.cpp`)
  - `NdReshardCopyLocalShardFactory<local_is_input=false>` (`nd_reshard_program_factory_copy_local.cpp`)

**Kernels exercised (all in scope):**
- **Op-owned** (`reshard/device/kernels/`): `nd_reshard_copy_local_shards.cpp`, `nd_reshard_copy_pages_reader.cpp`, `nd_reshard_copy_pages_writer.cpp`
- **In-family shared** (`data_movement/sharded/device/kernels/dataflow/`, file-path instantiated): `reshard_reader.cpp`, `reshard_reader_diff_width.cpp`, `reshard_same_width_reader.cpp`, `reshard_same_width_writer.cpp`, `reshard_same_height_reader.cpp`, `reshard_same_height_writer.cpp`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** provenance could not be pinned — `git log -1 … metal_2.0/` printed nothing (the docs are not from a tracked doc-branch checkout). This audit ran against the standalone recipe file `/localdev/edwinlee/metal2_audit.md` (last modified 2026-07-23).

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard` |
| **Overall** | **RED** |
| **DOps / Factories** | `ReshardDeviceOperation` → SameWidth×2, SameHeight×2, Generic, NdCopyPages, NdCopyLocal×2 |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes (GREEN)** — all 9 kernels are Device 2.0 native |
| *Prereqs* — Cross-op escapes | Ok — no function-call escapes; file-path coupling only (in-family shared pool + Quasar co-borrower) |
| *Feature Support* — overall | **GREEN** — all Appendix A entries N/A |
| *Feature Support* — Variadic-CTA | Ok (N/A) — CTAs read at constexpr offsets; variable-count loops are all **RTA**-driven |
| *TTNN Readiness* — `Is able to port?` (the gate) | **No — sheet is broken for this op** (see Gate detail) |
| *TTNN Readiness* — Concept (current) | `descriptor` (verified in code — all 8 factories are `create_descriptor`) |
| *TTNN Readiness* — Secretly SPMD | N/A (not a WorkloadDescriptor op) |
| *TTNN Readiness* — Is safe to port? | Sheet: **no** on 5 rows / yes on 3 → routed to readiness-sheet owner (correctness axis; not re-derived) |
| *TTNN Readiness* — Custom hash | Sheet: **yes on 5 rows / no on 3** — **conflicts with code** (no `compute_program_hash` exists) and is internally inconsistent |
| *TTNN Readiness* — Runtime-args update | No (verified — no `override_runtime_arguments` / `get_dynamic_runtime_args`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (verified — `reshard_nanobind.cpp` binds only the `reshard` function) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (would apply once the sheet is reconciled) |
| *Port work* — Offset base pointer | **none (GREEN)** — every base is a clean `Buffer*` binding; offsets are added kernel-side |
| *Port work* — Tensor bindings (per binding) | Case 1 / Case 2 / clean (borrowed-DFB) — see Port-work summary |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd argument |
| *Port work* — CB endpoints | 1P+1C (dual-instance work-split) on most CBs; one legal 1:1 (NdCopyPages); no dead CBs, no multi-binding flags |

## Result

**RED — blocked on the TTNN factory-concept gate, which cannot be cleared because the readiness sheet is untrustworthy for this op.** Routed to the **readiness-sheet owner** to reconcile.

The obstruction is a **data-quality problem in the sheet, not a structural blocker in the code.** The `Custom hash` column marks 5 of the 8 reshard factory rows `yes` and 3 `no` — but (a) custom hash is a per-`DeviceOperation` property and cannot legitimately differ across factory rows of one DeviceOperation, and (b) `ReshardDeviceOperation` has **no `compute_program_hash` override at all** (`reshard_device_operation.hpp:20-48`; confirmed by repo-wide grep — only the sibling `interleaved_to_sharded` op defines one). Because the sheet's `Is able to port?` verdict is derived from `Custom hash == no`, the `no` verdicts on the five "yes-hash" rows are corrupted. Per the recipe's trust-but-verify cross-check, a sheet↔code conflict on a factual column is a **"spreadsheet is broken"** GATE: stop rather than proceed on data we can't trust.

**No trustworthy portable subset can be carved from a broken sheet.** As written, the sheet lists `ReshardSameHeightFactory` (both variants) and one `ReshardSameWidthFactory` variant as `Is able to port? = yes`; treat that as *unverified* until the sheet is reconciled.

**Every other gate is GREEN** (Device 2.0 ✓, Feature compatibility ✓, Offset base pointers ✓, TensorAccessor 3rd arg ✓), and the code is a freshly-migrated `descriptor` op (ProgramDescriptor migration #43840, DataflowBuffer kernel migration #49392) using the sanctioned `Buffer*`-binding fast-path pattern throughout. **The code evidence strongly suggests the entire op is portable**, and a re-audit is likely to GREEN once the sheet reflects the current code (expected corrected state: `Custom hash = no`, and `Is safe to port?` re-confirmed — see Gate detail). Because the code is stable and a port is plausibly imminent after reconciliation, the informational subjects were **run in full for the whole op** (not skipped) so a re-audit/port has everything ready — see the Recipe note on this judgment.

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED — spreadsheet broken → readiness-sheet owner.**
  Sheet rows for `data_movement/sharded/reshard` / `ReshardDeviceOperation` (columns referenced by header name):

  | Factory (variant row) | Concept | Custom hash | RT-upd (dyn) | RT-upd (PD) | Pybind | Smuggled | Is safe? | **Is able?** |
  |---|---|---|---|---|---|---|---|---|
  | NdReshardCopyLocalShardFactory (row 1) | descriptor | **yes** | no | no | no | no | no | **no** |
  | NdReshardCopyLocalShardFactory (row 2) | descriptor | **yes** | no | no | no | no | no | **no** |
  | NdReshardCopyPagesFactory | descriptor | **yes** | no | no | no | no | no | **no** |
  | ReshardGenericFactory | descriptor | **yes** | no | no | no | no | no | **no** |
  | ReshardSameHeightFactory (row 1) | descriptor | no | no | no | no | no | yes | yes |
  | ReshardSameHeightFactory (row 2) | descriptor | no | no | no | no | no | yes | yes |
  | ReshardSameWidthFactory (row 1) | descriptor | **yes** | no | no | no | no | no | **no** |
  | ReshardSameWidthFactory (row 2) | descriptor | no | no | no | no | no | yes | yes |

  **Conflicts driving the RED:**
  1. **`Custom hash` — sheet vs. code.** Sheet says `yes` on 5 rows; code has **no** `compute_program_hash` anywhere in the op. Verified: `reshard_device_operation.hpp:20-48` (the `ReshardDeviceOperation` struct declares `select_program_factory`, `validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`, `create_op_performance_model` — and **no** `compute_program_hash`); repo-wide `grep compute_program_hash` under `sharded/` matches only `interleaved_to_sharded`. This op is not pybound-`create_descriptor`, so the renamed-hash exception does not apply.
  2. **`Custom hash` — internal inconsistency.** The column splits 5-`yes` / 3-`no` across factory rows of the *same* `ReshardDeviceOperation`. Custom hash is device-op-level (a `compute_program_hash` method on the DeviceOperation, or its absence), so it must be uniform across all 8 rows. The split is impossible → the sheet is internally inconsistent for this op.
  - Confirmed-consistent conjuncts (sheet matches code): `Concept == descriptor` ✓, `Runtime-args update (dyn/PD) == no` ✓ (no `override_runtime_arguments` / `get_dynamic_runtime_args`), `Pybind descriptor == no` ✓, `Op-owned tensors == no` ✓.
  - **`Is safe to port? = no` on the 5 rows** is the readiness-sheet owner's correctness axis and is **not re-derived here.** Note for the owner: it warrants re-confirmation, because (i) `Smuggled pointer = no` on all reshard rows, so the `no` is not attributable to a smuggled pointer per the sheet's own columns, and (ii) the op was freshly migrated to the sanctioned `Buffer*`-binding fast-path pattern (`emplace_runtime_args`/`emplace_common_runtime_args` with `Buffer*`), which the framework patches on cache hits — i.e. the historical stale-pointer hazard is already handled. `Is safe to port? = no` may be stale relative to the current code.
  - **Route:** readiness-sheet owner to reconcile the `Custom hash` and `Is safe to port?` columns for all 8 reshard rows. **Expected corrected state:** `Custom hash = no` (uniform) and `Is able to port? = yes` for every factory, pending the owner's `Is safe to port?` re-confirmation. The gate then clears with no code change.

- **Device 2.0 (every kernel used): GREEN.** All nine kernels are structurally Device 2.0 native — `Noc` object (`noc.async_read/async_write/async_*_barrier`), `DataflowBuffer` object (`reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_write_ptr`/`get_read_ptr` as **methods**, never CB-index free functions), `CoreLocalMem<uint32_t>`, `AllocatorBank<bank_type>` with `{.bank_id, .addr}` addressing (the Device 2.0 replacement for `get_noc_addr_from_bank_id`), `UnicastEndpoint{}`, and `TensorAccessor`/`TensorAccessorArgs`. **No** legacy Device 1.0 idioms found: no raw `noc_async_read`, no `InterleavedAddrGen`/`ShardedAddrGen`/`InterleavedAddrGenFast`/`InterleavedPow2AddrGen*`, no raw semaphore addresses, no CB-index-keyed free-function holdovers. Firmware primitives `my_x[]`/`my_y[]` indexed by `noc.get_noc_id()` (`reshard_same_width_reader.cpp:66-67`) and `get_arg_addr()` for vararg unpacking are sanctioned, not holdovers. The in-family shared host helper `compute_width_sharding_reshard_segments` (`sharded_common.{hpp,cpp}`) is host code, not a kernel — no Device 2.0 concern.

- **Feature compatibility:** clean scan — all entries N/A.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | All CBs are plain `CBDescriptor`; `.buffer` set to a `Buffer*` (borrowed memory) or `nullptr`. No `.global_circular_buffer` field, no `.remote_index`, no `remote_cb`, no 4-arg `CreateCircularBuffer`, no `experimental::` GCB type. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `address_offset` (default 0). Borrowed-memory CBs set `.buffer` only. |
  | GlobalSemaphore | N/A | The op uses **no semaphores at all** (`grep Semaphore` under reshard → none). |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = ReshardInputs{input, preallocated_output}` — fixed count, no `std::vector<Tensor>`. All kernels read CTAs at **constexpr** offsets (`base_idx_cta + N`). The variable-count loops (`num_ranges`, `num_blocks`, `num_reads`, `num_segments`) are all **RTA**-driven → supported (see Heads-ups → RTA varargs). |

- **CB endpoints (GATE-free):** Device 2.0 idioms intact, so classified per `(CB, config)`, per node. Every CB is either a legal 1:1 or resolves at port time; nothing gates.
  - **`ReshardGenericFactory` — output CB (`dst_cb_index = 16`, bound to `output_buffer`):** dual-instance work-split — one `kernel_source` (`reshard_reader.cpp` or `reshard_reader_diff_width.cpp`) instantiated twice (ReaderConfig + WriterConfig) over the same `all_cores`, both raw-writing **disjoint** output page ranges via `dfb.get_write_ptr() + output_page_offset*page_size` (`reshard_reader.cpp:30`, `reshard_reader_diff_width.cpp:30`); output resident, nothing drains. Two role-free touchers → **1P+1C** (bind one PRODUCER, one CONSUMER; cosmetic on Gen1). *This is the verified example named in the recipe's CB-endpoints section — do not mis-slot it as multi-binding.*
  - **`ReshardSameWidthFactory` — local CB (`c_0`, bound to `local_buffer`):** same source instantiated twice (reader/writer config), both touch via `shard_dfb.get_write_ptr()` (reader path) / `get_read_ptr()` (writer path), work split by local-unit range. Two role-free touchers → **1P+1C**.
  - **`ReshardSameWidthFactory` — scratch CB (`c_1`, `buffer=nullptr`):** present **only when `unaligned && local_is_output`**. Both same-source instances touch it (`dfb_scratch.get_write_ptr()`/`get_read_ptr()`, `reshard_same_width_reader.cpp:39-41`). Two role-free touchers → **1P+1C**. *(Config-dependent existence — note per `(CB, config)`.)*
  - **`ReshardSameHeightFactory` — local CB (`c_0`, bound to `local_buffer`):** dual-instance work-split (reader/writer config), both touch via `get_write_ptr`/`get_read_ptr`. Two role-free touchers → **1P+1C**.
  - **`NdReshardCopyPagesFactory` — CB (`c_0`, `buffer=nullptr`):** genuine FIFO — reader (`nd_reshard_copy_pages_reader.cpp`) `reserve_back`/`push_back` (locked producer), writer (`nd_reshard_copy_pages_writer.cpp`) `wait_front`/`pop_front` (locked consumer). One producer + one consumer → **plain 1:1 legal**, no action.
  - **`NdReshardCopyLocalShardFactory`:** **no CBs** — kernel copies L1↔L1/DRAM directly via `TensorAccessor` + `CoreLocalMem`. Nothing to classify.
  - No dead CBs; no CB reaches ≥3 touchers or ≥2 locked same-role, so **no multi-binding flag** is needed anywhere.

- **Offset base pointers: GREEN — cleared.** Every tensor base delivered to a kernel is a **clean base** bound as a `Buffer*` (the framework's `Buffer`/`CommonBuffer` binding), with any offset added **kernel-side**, never host-folded into the delivered pointer:
  - `ReshardGenericFactory`: `input_buffer->address()` is computed into the RTA vector (`get_runtime_args_for_given_ranges*`) but the factory **overwrites** arg position `grid.x+grid.y` with the raw `input_buffer` pointer (`reshard_program_factory_generic.cpp:781-798`); the kernel adds the offset (`input_shard_addr + addr_offset`, `reshard_reader.cpp:67`). Clean base.
  - `ReshardSameWidthFactory` / `ReshardSameHeightFactory`: `remote_buffer` bound as a `Buffer*` arg (`same_width.cpp:165`, `same_height.cpp:127/134`); offsets (`src_offset`, `read_offset`, `write_offset`) are separate args added kernel-side. Clean base.
  - `NdReshardCopyPagesFactory` / `NdReshardCopyLocalShardFactory`: `input_buffer`/`output_buffer` bound as `Buffer*` common runtime args; fed as the clean base into `TensorAccessor`. Clean base.
  - Not Type 1/2/3/4 anywhere. Cross-referenced against the (dated) offset-base-pointer triage: **not applicable** — no fold present.

- **TensorAccessor 3rd argument: GREEN — no site fires.** Every `TensorAccessor` construction is 2-arg (`TensorAccessor(args, base_addr)`): `nd_reshard_copy_local_shards.cpp:44-45`, `nd_reshard_copy_pages_reader.cpp:26`, `nd_reshard_copy_pages_writer.cpp:26`. No explicit page-size 3rd argument anywhere. The same-width/same-height/generic kernels use no `TensorAccessor` (raw NoC / `AllocatorBank`). Cross-referenced against the (dated) 3rd-arg triage: nothing to classify.

## Port-work summary  *(for the eventual port, once the gate clears)*

- **Tensor bindings** (per binding, per factory — all delivered today via the `Buffer*`-binding form, which the framework patches on cache hits; routine port work, not a correctness hazard):

  | Factory | Binding | Case | Note |
  |---|---|---|---|
  | `ReshardGenericFactory` | output CB (`output_buffer`) | **clean** | borrowed-memory DFB (`cb.buffer = output_buffer`); port via `DataflowBufferSpec::borrowed_from` |
  | `ReshardGenericFactory` | input tensor | **Case 2** | raw NoC addressing (`.addr = input_shard_addr + addr_offset` with explicit `noc_x/noc_y`) → bind as `TensorParameter`, bridge base via `get_bank_base_address`, keep raw walk |
  | `ReshardSameWidthFactory` | local CB (`local_buffer`) | **clean** | borrowed-memory DFB |
  | `ReshardSameWidthFactory` | remote tensor | **Case 2** | raw `AllocatorBank` addressing (`{.bank_id, .addr = src_addr + offset}`) → Case 2 bridge |
  | `ReshardSameHeightFactory` | local CB (`local_buffer`) | **clean** | borrowed-memory DFB |
  | `ReshardSameHeightFactory` | remote tensor | **Case 2** | raw `AllocatorBank` addressing → Case 2 bridge |
  | `NdReshardCopyPagesFactory` | input tensor | **Case 1** | fed into `TensorAccessor(args_src, base)` → express as `TensorParameter`, kernel uses `TensorAccessor(tensor::name)` |
  | `NdReshardCopyPagesFactory` | output tensor | **Case 1** | fed into `TensorAccessor(args_dst, base)` → Case 1 |
  | `NdReshardCopyLocalShardFactory` | input tensor | **Case 1** | fed into `TensorAccessor(args_src, base)` → Case 1 |
  | `NdReshardCopyLocalShardFactory` | output tensor | **Case 1** | fed into `TensorAccessor(args_dst, base)` → Case 1 |

  Op-level roll-up: **⚠ port work** (Case-1 and Case-2 bindings present; borrowed-DFB bindings clean).
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** 1P+1C assign — Generic output CB, SameWidth local CB, SameWidth scratch CB (`unaligned && local_is_output` only), SameHeight local CB; legal 1:1 — NdCopyPages CB; no CBs — NdCopyLocal. No dead-CB drops, no multi-binding flags.

## Heads-ups  *(for the eventual porter)*

- **CB endpoints (shapes to watch):** all two-toucher CBs here are the **dual-instance work-split** shape (same `kernel_source` in a ReaderConfig + WriterConfig pair over one core range, splitting work by disjoint ranges) → assign **1P+1C**, do **not** reach for the multi-binding flag. No hidden-second-writer (semaphore-gated co-fill) shapes — the op uses no semaphores. No multi-reader ≥3-toucher shapes.
- **Cross-op / shared kernels (port-together set):** the six shared reshard kernels live in `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/` (one level up from the op — in-family shared pool, file-path instantiated by the Generic/SameWidth/SameHeight factories). The `reshard_same_width_*` and `reshard_same_height_*` kernels are **also instantiated by `ttnn/cpp/ttnn/operations/experimental/quasar/reshard/`** — a Quasar (Gen2) port, out of scope here, but a co-borrower: any Metal 2.0 CB→DFB / named-token rewrite of these shared kernels must be coordinated so the Quasar reshard is not broken. Port the shared kernels + both consuming reshard ops as one unit.
- **RTA varargs (prefer the kernel-side vararg mechanism, do not try to name each):**
  - `reshard_reader.cpp:35` — `for (range_id < num_ranges)` with `arg_index++` reads inside the loop; also data-selected reads `get_arg_val(start_x_index)` / `get_arg_val(core_id_x_index)` (`:41-42,60-61`). Variable-count + data-selected → varargs.
  - `reshard_reader_diff_width.cpp:35` — `for (block_id < num_blocks)` with nested `current_pattern_arg_index++` reads. Variable-count → varargs.
  - `reshard_same_width_reader.cpp:30,42` / `reshard_same_width_writer.cpp:29,45` / `reshard_same_height_reader.cpp:25,33` / `reshard_same_height_writer.cpp:25,33` — a `get_arg_addr(N)` pointer walked as `args[args_idx++]` in a runtime-count loop (`num_reads` / `num_segments`). Variable-count → varargs. The leading scalars (args 0-4) are nameable.
  - The `nd_reshard_*` kernels read only fixed RTAs (`get_arg_val<uint32_t>(0)`/`(1)`) + common args at constexpr offsets → nameable, no varargs.

## Team-only

- **Out-of-directory coupling & donor shape:** Op-level roll-up **✓ clean** — no function-call escapes. All kernels `#include` only `api/*` (tt_metal LLK/HAL, donor class 1 — no concern). No cross-family donor functions, no `CircularBuffer&`/`Semaphore`/addr-gen donor signatures. The only out-of-directory coupling is **file-path kernel instantiation** of the in-family shared pool `data_movement/sharded/device/kernels/dataflow/` (class 5, in-family) — captured as the port-together set in Heads-ups. Host-side, the SameHeight factory calls `ttnn::operations::data_movement::detail::compute_width_sharding_reshard_segments` from `sharded_common.{hpp,cpp}` (in-family host helper) — no gating implication.
- **Relaxation candidates:** none mined (no custom hash exists).
- **TTNN factory analysis (sheet-derived + code cross-check):** Concept `descriptor` (verified); custom hash **absent in code** (sheet disagrees — see Gate detail); no `override_runtime_arguments` / `get_dynamic_runtime_args`; no pybind `create_descriptor` (nanobind binds only `ttnn::reshard`); no op-owned tensors; not a WorkloadDescriptor. Target concept on reconciliation: `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **Dead / unreachable code in `is_valid_for_legacy_reshard`** (`reshard_device_operation.cpp:34-50`): the `return` at line 39 is unconditional (executed whenever the line-34 `if` is false), so the entire `if (input_tensor.layout() == Layout::ROW_MAJOR) { … }` block at lines 41-50 is **unreachable**. Likely a logic bug introduced during editing (the intent was probably to reach the row-major shard-width checks). Route to the ops team.
- **Live `DPRINT` left in a shipping kernel** (`reshard_same_width_reader.cpp:46`): `DPRINT("addr: {}\n", addr);` in the unaligned reader path (plus commented-out `print_bf16_pages` at `:52-53,70`). Harmless when DPRINT is compiled out, but stray debug output. Route to the ops team.
- **Dead RTA read** in `reshard_reader.cpp:24` and `reshard_reader_diff_width.cpp:24`: `num_output_pages` is unpacked from an RTA but never referenced in either kernel. Minor.

## Recipe notes

- **Scoping judgment on the informational subjects under a "spreadsheet-broken" RED.** The recipe's Red-outcome scoping rule says to skip the seven purely-informational subjects on a "whole-op RED with no portable subset," on the rationale that they would be produced against "soon-to-change legacy code." That rationale does **not** fit this RED: the code is a freshly-migrated, settled `descriptor` op, and the RED is a *sheet data-quality* failure, not a structural/legacy blocker — a re-audit will run against the *same* code, not changed code. I judged the informational census cheap (everything was already in context) and its downstream value high (a port is plausibly imminent once the sheet is reconciled), so I ran all subjects for the whole op and recorded them. Flagging in case the maintainer wants an explicit branch in the scoping rule for "gate failed on untrustworthy sheet, code is stable" vs. "gate failed on genuine legacy/structural blocker."
- **Cross-column invariant for `Custom hash`.** The recipe lists cross-column invariants for `Runtime-args update` and `Op-owned tensors?` but not for `Custom hash`. This op exercised exactly such an invariant: `Custom hash` is device-op-level (`compute_program_hash` on the DeviceOperation), so it must be uniform across all factory rows of one DeviceOperation — the reshard rows violate this (5-yes/3-no). Suggest adding `Custom hash` (and `Pybind descriptor`, likewise device-op-level) to the enumerated cross-column-invariant list so a future auditor keys on the inconsistency directly.
