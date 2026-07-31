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

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` — this audit ran against `/localdev/edwinlee/metal2_audit.md`, which is byte-identical to the tracked `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/metal2_audit.md` in the `Port_Recipe` checkout (provenance pinned there; the prior audit could not pin it because the git command was run from the reshard checkout, which does not carry the docs tree).

---

## Change from the prior audit (2026-07-23): RED → **GREEN**

The previous audit RED'd on a single blocker — the **TTNN factory-concept gate could not be cleared because the readiness sheet was internally inconsistent and conflicted with the code** (the `Custom hash` column split 5-`yes` / 3-`no` across factory rows of one DeviceOperation, while the code has no `compute_program_hash` at all). That was a **"spreadsheet is broken"** GATE, routed to the readiness-sheet owner. **Every other gate was already GREEN.**

The readiness sheet has since been **reconciled**. A fresh fetch (this run) shows all 8 `data_movement/sharded/reshard` / `ReshardDeviceOperation` factory rows now uniform and consistent with the code: `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args? = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, **`Is able to port? = yes`**, relaxation `none`, op-owned tensors empty. The sole blocker is cleared, and the lightweight cross-check matches the sheet on every factual column. **The op now clears every gate → GREEN → `METAL2_PORT_BRIEF.md` is issued alongside this file.**

No op code changed between the two audits (same files, same commit); only the sheet was corrected.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `ReshardDeviceOperation` → SameWidth×2, SameHeight×2, Generic, NdCopyPages, NdCopyLocal×2 |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes (GREEN)** — all 9 kernels are Device 2.0 native |
| *Prereqs* — Cross-op escapes | Ok — no function-call escapes; file-path coupling only (in-family shared pool + Quasar co-borrower) |
| *Feature Support* — overall | **GREEN** — all Appendix A entries N/A |
| *Feature Support* — Variadic-CTA | Ok (N/A) — CTAs read at constexpr offsets; variable-count loops are all **RTA**-driven |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes (GREEN)** — all 8 factory rows `yes`; cross-check clean |
| *TTNN Readiness* — Concept (current) | `descriptor` (verified in code — all 8 factories are `create_descriptor`) |
| *TTNN Readiness* — Secretly SPMD | N/A (not a WorkloadDescriptor op) |
| *TTNN Readiness* — Is safe to port? | Yes (sheet, uniform on all 8 rows) |
| *TTNN Readiness* — Custom hash | No (sheet uniform `no`; **matches code** — no `compute_program_hash` in the op) |
| *TTNN Readiness* — Runtime-args update | No (verified — no `override_runtime_arguments` / `get_dynamic_runtime_args`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (verified — `reshard_nanobind.cpp` binds only the `reshard` function) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | **`MetalV2FactoryConcept`** (no op-owned tensors) |
| *Port work* — Offset base pointer | **none (GREEN)** — every base is a clean `Buffer*` binding; offsets are added kernel-side |
| *Port work* — Tensor bindings (per binding) | Case 1 / Case 2 / clean (borrowed-DFB) — see Port-work summary |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd argument |
| *Port work* — CB endpoints | 1P+1C (dual-instance work-split) on most CBs; one legal 1:1 (NdCopyPages); no dead CBs, no multi-binding flags |

## Result

**GREEN → brief issued.** All five gates clear: **TTNN factory concept ✓** (readiness sheet reconciled — `Is able to port? = yes` on all 8 factory rows, cross-check clean), **Device 2.0 ✓** (all 9 kernels Device 2.0 native), **Feature compatibility ✓** (all Appendix A entries N/A), **Offset base pointers ✓** (clean bases, kernel-side offsets), **TensorAccessor 3rd arg ✓** (no site fires).

The op is a freshly-migrated `descriptor` op (ProgramDescriptor migration #43840, DataflowBuffer kernel migration #49392) using the sanctioned `Buffer*`-binding fast-path pattern throughout. Target concept: **`MetalV2FactoryConcept`** for all 8 factories. Port work is routine: Case-1/Case-2 tensor bindings, self-loop/1P+1C CB dispositions, no relaxations, no 3rd-arg drops. See `METAL2_PORT_BRIEF.md`.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN — cleared.**
  Sheet rows for `data_movement/sharded/reshard` / `ReshardDeviceOperation` (columns referenced by header name), fetched fresh this run:

  | Factory (variant row) | Concept | Custom hash | RT-upd (dyn) | Override RTA | Pybind | Smuggled | Is safe? | **Is able?** |
  |---|---|---|---|---|---|---|---|---|
  | NdReshardCopyLocalShardFactory (row 1) | descriptor | no | no | no | no | no | yes | **yes** |
  | NdReshardCopyLocalShardFactory (row 2) | descriptor | no | no | no | no | no | yes | **yes** |
  | NdReshardCopyPagesFactory | descriptor | no | no | no | no | no | yes | **yes** |
  | ReshardGenericFactory | descriptor | no | no | no | no | no | yes | **yes** |
  | ReshardSameHeightFactory (row 1) | descriptor | no | no | no | no | no | yes | **yes** |
  | ReshardSameHeightFactory (row 2) | descriptor | no | no | no | no | no | yes | **yes** |
  | ReshardSameWidthFactory (row 1) | descriptor | no | no | no | no | no | yes | **yes** |
  | ReshardSameWidthFactory (row 2) | descriptor | no | no | no | no | no | yes | **yes** |

  **Lightweight cross-check (trust-but-verify) — every cheaply-checkable column matches the code:**
  - `Concept == descriptor` ✓ — all 8 factories expose `create_descriptor()` returning a `ProgramDescriptor` (verified: `create_descriptor` present in all five factory `.cpp`/`.hpp` under `device/`).
  - `Custom hash == no` ✓ — `ReshardDeviceOperation` declares no `compute_program_hash` (`reshard_device_operation.hpp:20-48` lists only `select_program_factory`, `validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`, `create_op_performance_model`). Repo-wide grep under `sharded/` matches only the sibling `interleaved_to_sharded`. **This is the column that was corrected since the last audit** — it now agrees with the code.
  - `Runtime-args update == no` ✓ — no `override_runtime_arguments` / `get_dynamic_runtime_args` anywhere in the op.
  - `Pybind descriptor == no` ✓ — `reshard_nanobind.cpp` binds only `ttnn::reshard` (a plain `bind_function<"reshard">`); no `create_descriptor` / `nb::class_` of the device op.
  - `Op-owned tensors == no` ✓ — not a `WorkloadDescriptor` op; no `buffers` vector.
  - Cross-column invariants hold: `Runtime-args update == no` (consistent with a `descriptor` concept); `Op-owned tensors == no` (consistent — a `descriptor` op cannot carry op-owned tensors).
  - `Is safe to port? == yes` (uniform on all 8 rows) is the readiness-sheet owner's correctness axis and is **not re-derived here**; it is consistent with `Smuggled pointer == no` on every row.
  - **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors) for all 8 factories.

- **Device 2.0 (every kernel used): GREEN.** All nine kernels are structurally Device 2.0 native. Verified by idiom grep across all 9 kernels: **no** legacy Device 1.0 idioms (no `InterleavedAddrGen`/`ShardedAddrGen`/`InterleavedAddrGenFast`/`InterleavedPow2AddrGen*`, no raw `noc_async_read`/`noc_async_write`, no `noc_semaphore*`, no `CircularBuffer&`, no `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front` free functions, no `get_noc_addr_from_bank_id`). Device 2.0 idioms present throughout: `Noc` object (`noc.async_read`/`async_write`), `DataflowBuffer` object with `reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_write_ptr`/`get_read_ptr` as **methods** (`dfb.get_write_ptr()`, `shard_dfb.get_read_ptr()`, `dfb_scratch.get_write_ptr()` — never CB-index free functions), `CoreLocalMem<uint32_t>`, `AllocatorBank<bank_type>` with `{.bank_id, .addr}` addressing (the Device 2.0 replacement for `get_noc_addr_from_bank_id`), `UnicastEndpoint{}`, and `TensorAccessor`/`TensorAccessorArgs`. **No CB-index-keyed free-function holdovers.** Firmware primitives `my_x[]`/`my_y[]` indexed by `noc.get_noc_id()` (`reshard_same_width_reader.cpp:66-67`) and `get_arg_addr()` for vararg unpacking are sanctioned, not holdovers. The in-family shared host helper `compute_width_sharding_reshard_segments` (`sharded_common.{hpp,cpp}`) is host code, not a kernel — no Device 2.0 concern.

- **Feature compatibility:** clean scan — all entries N/A.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | All CBs are plain `CBDescriptor`; `.buffer` set to a `Buffer*` (borrowed memory) or `nullptr`. No `.global_circular_buffer` field, no `.remote_index`, no `remote_cb`, no 4-arg `CreateCircularBuffer`, no `experimental::` GCB type. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `address_offset` (default 0). Borrowed-memory CBs set `.buffer` only. |
  | GlobalSemaphore | N/A | The op uses **no semaphores at all** (`grep Semaphore` under reshard → none). |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = ReshardInputs{input, preallocated_output}` — fixed count, no `std::vector<Tensor>`. All kernels read CTAs at **constexpr** offsets. The variable-count loops (`num_ranges`, `num_blocks`, `num_reads`, `num_segments`) are all **RTA**-driven → supported (see Heads-ups → RTA varargs). |

- **CB endpoints (GATE-free):** Device 2.0 idioms intact, so classified per `(CB, config)`, per node. Every CB is either a legal 1:1 or resolves at port time; nothing gates.
  - **`ReshardGenericFactory` — output CB (`dst_cb_index = 16`, bound to `output_buffer`):** dual-instance work-split — one `kernel_source` (`reshard_reader.cpp` or `reshard_reader_diff_width.cpp`, chosen by page-size equality at `reshard_program_factory_generic.cpp:701-705`) instantiated twice (`kernel_desc_0` + `kernel_desc_1` over the same `all_cores`, `:712/:719`), both raw-writing **disjoint** output page ranges via `dfb.get_write_ptr() + output_page_offset*page_size` (`reshard_reader.cpp:30`, `reshard_reader_diff_width.cpp:30`); output resident, nothing drains. Two role-free touchers → **1P+1C** (bind one PRODUCER, one CONSUMER; cosmetic on Gen1). *This is the verified example named in the recipe's CB-endpoints section — do not mis-slot it as multi-binding.*
  - **`ReshardSameWidthFactory` — local CB (`c_0`, bound to `local_buffer`):** same source instantiated twice (reader/writer config, `reshard_program_factory_same_width.cpp:91-92,135,142`), both touch via `shard_dfb.get_write_ptr()` (reader path) / `get_read_ptr()` (writer path), work split by local-unit range. Two role-free touchers → **1P+1C**.
  - **`ReshardSameWidthFactory` — scratch CB (`c_1`, `buffer=nullptr`):** present **only when `unaligned && local_is_output`**. Both same-source instances touch it (`dfb_scratch.get_write_ptr()`/`get_read_ptr()`, `reshard_same_width_reader.cpp:40-41`). Two role-free touchers → **1P+1C**. *(Config-dependent existence — note per `(CB, config)`.)*
  - **`ReshardSameHeightFactory` — local CB (`c_0`, bound to `local_buffer`):** dual-instance work-split (reader/writer config, `reshard_program_factory_same_height.cpp:83-84,88,95`), both touch via `get_write_ptr` (`reshard_same_height_reader.cpp:31`) / `get_read_ptr` (`reshard_same_height_writer.cpp:31`). Two role-free touchers → **1P+1C**.
  - **`NdReshardCopyPagesFactory` — CB (`c_0`, `buffer=nullptr`):** genuine FIFO — reader `reserve_back`/`push_back` (`nd_reshard_copy_pages_reader.cpp:34,38`, locked producer), writer `wait_front`/`pop_front` (`nd_reshard_copy_pages_writer.cpp:34,38`, locked consumer). One producer + one consumer → **plain 1:1 legal**, no action.
  - **`NdReshardCopyLocalShardFactory`:** **no CBs** — kernel copies L1↔L1/DRAM directly via `TensorAccessor` + `CoreLocalMem`. Nothing to classify.
  - No dead CBs; no CB reaches ≥3 touchers or ≥2 locked same-role, so **no multi-binding flag** is needed anywhere.

- **Offset base pointers: GREEN — cleared.** Every tensor base delivered to a kernel is a **clean base** bound as a `Buffer*`, with any offset added **kernel-side**, never host-folded into the delivered pointer:
  - `ReshardGenericFactory`: `input_buffer->address()` is computed into the RTA vector by the `get_runtime_args_for_given_ranges*` helpers (`reshard_program_factory_generic.cpp:748,754,769,775`), but the factory then **overwrites** arg position `grid.x + grid.y` with the raw `input_buffer` pointer before emplacing (`:783-798` — `if (i == grid.x + grid.y) rt_args.push_back(input_buffer);`), so the framework registers a `BufferBinding` and delivers the clean base. The kernel adds the offset (`.addr = input_shard_addr + addr_offset`, `reshard_reader.cpp:67`). Clean base.
  - `ReshardSameWidthFactory` / `ReshardSameHeightFactory`: `remote_buffer` bound as a `Buffer*` arg; offsets (`src_offset`, `read_offset`, `write_offset`) are separate args added kernel-side (`reshard_same_width_reader.cpp:37`, `reshard_same_width_writer.cpp:36`). Clean base.
  - `NdReshardCopyPagesFactory` / `NdReshardCopyLocalShardFactory`: `input_buffer`/`output_buffer` bound as `Buffer*` common runtime args; fed as the clean base into `TensorAccessor`. Clean base.
  - Not Type 1/2/3/4 anywhere. Cross-referenced against the (dated) offset-base-pointer triage `2026-07-19_offset_base_pointers.md`: reshard is **not applicable** — no fold present.

- **TensorAccessor 3rd argument: GREEN — no site fires.** Every `TensorAccessor` construction is 2-arg (`TensorAccessor(args, base_addr)`): `nd_reshard_copy_pages_reader.cpp:26`, `nd_reshard_copy_pages_writer.cpp:26`, `nd_reshard_copy_local_shards.cpp:44-45`. No explicit page-size 3rd argument anywhere. The same-width/same-height/generic kernels use no `TensorAccessor` (raw NoC / `AllocatorBank`). Cross-referenced against the (dated) 3rd-arg triage `2026-07-06_tensor_accessor_3rd_arg_triage.md`: nothing to classify.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory — all delivered today via the `Buffer*`-binding form, which the framework patches on cache hits; routine port work, not a correctness hazard):

  | Factory | Binding | Case | Note |
  |---|---|---|---|
  | `ReshardGenericFactory` | output CB (`output_buffer`) | **clean** | borrowed-memory DFB (`cb.buffer = output_buffer`); port via `DataflowBufferSpec::borrowed_from` |
  | `ReshardGenericFactory` | input tensor | **Case 2** | raw NoC addressing (`.addr = input_shard_addr + addr_offset` with explicit `noc_x/noc_y`, `reshard_reader.cpp:60-67`) → bind as `TensorParameter`, bridge base via `get_bank_base_address`, keep raw walk |
  | `ReshardSameWidthFactory` | local CB (`local_buffer`) | **clean** | borrowed-memory DFB |
  | `ReshardSameWidthFactory` | remote tensor | **Case 2** | raw `AllocatorBank` addressing (`{.bank_id, .addr = src_addr + offset}`) → Case 2 bridge |
  | `ReshardSameHeightFactory` | local CB (`local_buffer`) | **clean** | borrowed-memory DFB |
  | `ReshardSameHeightFactory` | remote tensor | **Case 2** | raw `AllocatorBank` addressing → Case 2 bridge |
  | `NdReshardCopyPagesFactory` | input tensor | **Case 1** | fed into `TensorAccessor(args_src, base)` (`nd_reshard_copy_pages_reader.cpp:26`) → express as `TensorParameter`, kernel uses `TensorAccessor(tensor::name)` |
  | `NdReshardCopyPagesFactory` | output tensor | **Case 1** | fed into `TensorAccessor(args_dst, base)` (`nd_reshard_copy_pages_writer.cpp:26`) → Case 1 |
  | `NdReshardCopyLocalShardFactory` | input tensor | **Case 1** | fed into `TensorAccessor(args_src, base)` (`nd_reshard_copy_local_shards.cpp:44`) → Case 1 |
  | `NdReshardCopyLocalShardFactory` | output tensor | **Case 1** | fed into `TensorAccessor(args_dst, base)` (`nd_reshard_copy_local_shards.cpp:45`) → Case 1 |

  Op-level roll-up: **⚠ port work** (Case-1 and Case-2 bindings present; borrowed-DFB bindings clean).
- **TensorParameter relaxation:** none (sheet: `none` on all 8 rows).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** 1P+1C assign — Generic output CB, SameWidth local CB, SameWidth scratch CB (`unaligned && local_is_output` only), SameHeight local CB; legal 1:1 — NdCopyPages CB; no CBs — NdCopyLocal. No dead-CB drops, no multi-binding flags.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (shapes to watch):** all two-toucher CBs here are the **dual-instance work-split** shape (same `kernel_source` in a ReaderConfig + WriterConfig pair over one core range, splitting work by disjoint ranges) → assign **1P+1C**, do **not** reach for the multi-binding flag. No hidden-second-writer (semaphore-gated co-fill) shapes — the op uses no semaphores. No multi-reader ≥3-toucher shapes.
- **Cross-op / shared kernels (port-together set):** the six shared reshard kernels live in `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/` (one level up from the op — in-family shared pool, file-path instantiated by the Generic/SameWidth/SameHeight factories). The `reshard_same_width_*`, `reshard_same_height_*`, `reshard_reader*` kernels are **also instantiated by `ttnn/cpp/ttnn/operations/experimental/quasar/reshard/`** — a Quasar (Gen2) port, out of scope here, but a co-borrower. Per the readiness sheet, the Quasar reshard's `NdReshardCopyPagesFactory` and `ReshardGenericFactory` are already `MetalV2`; its SameWidth/SameHeight/CopyLocal factories are still `legacy device-op`. Any Metal 2.0 CB→DFB / named-token rewrite of these shared kernels must be coordinated so the Quasar reshard is not broken. Port the shared kernels + both consuming reshard ops as one unit.

  > **⚠ CORRECTION (port, 2026-07-31) — this heads-up did not hold at port time.**
  > Two facts in it were stale, both traceable to the readiness sheet rather than to the code:
  > 1. **Quasar is not a co-borrower.** It has private copies of all nine kernels under
  >    `experimental/quasar/reshard/device/kernels/` and instantiates only those paths. A
  >    repo-wide grep for consumers of the `data_movement/sharded/device/kernels/dataflow/reshard_*`
  >    paths returns only this op's three factories, so the shared kernels were modified **in
  >    place** with no Quasar coordination and no scope expansion.
  > 2. **All five Quasar reshard factories are already on `create_program_artifacts`**, not just
  >    `NdReshardCopyPagesFactory` and `ReshardGenericFactory`.
  >
  > The original text is preserved above as the audit-time record; only this note is added. The
  > correction changes no gate and no other conclusion in this audit — the shared-kernel item was
  > non-gating coordination advice. **Process note for the readiness-sheet owner:** a sibling op
  > forking its kernels silently invalidates the sheet's co-borrower answer, so a future audit
  > should derive the consumer list from a grep at audit time rather than from the sheet. See
  > `METAL2_PORT_PLAN.md` → *Cross-op kernels* and `METAL2_PORT_REPORT.md` → *Confusion*.
- **RTA varargs (prefer the kernel-side vararg mechanism, do not try to name each):**
  - `reshard_reader.cpp:35` — `for (range_id < num_ranges)` with `arg_index++` reads inside the loop; also data-selected reads `get_arg_val(start_x_index)` / `get_arg_val(core_id_x_index)` (`:41-42,60-61`). Variable-count + data-selected → varargs.
  - `reshard_reader_diff_width.cpp:35` — `for (block_id < num_blocks)` with nested `current_pattern_arg_index++` reads. Variable-count → varargs.
  - `reshard_same_width_reader.cpp:30,42` / `reshard_same_width_writer.cpp:29,45` / `reshard_same_height_reader.cpp:25,33` / `reshard_same_height_writer.cpp:25,33` — a `get_arg_addr(N)` pointer walked as `args[args_idx++]` in a runtime-count loop (`num_reads` / `num_segments`). Variable-count → varargs. The leading scalars (args 0-4) are nameable.
  - The `nd_reshard_*` kernels read only fixed RTAs (`get_arg_val<uint32_t>(0)`/`(1)`) + common args at constexpr offsets → nameable, no varargs.

## Team-only

- **Out-of-directory coupling & donor shape:** Op-level roll-up **✓ clean** — no function-call escapes. All kernels `#include` only `api/*` (tt_metal LLK/HAL, donor class 1 — no concern): `api/tensor/tensor_accessor.h`, `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`. No cross-family donor functions, no `CircularBuffer&`/`Semaphore`/addr-gen donor signatures. The only out-of-directory coupling is **file-path kernel instantiation** of the in-family shared pool `data_movement/sharded/device/kernels/dataflow/` (class 5, in-family) — captured as the port-together set in Heads-ups. Host-side, the SameHeight factory calls `ttnn::operations::data_movement::detail::compute_width_sharding_reshard_segments` from `sharded_common.{hpp,cpp}` (in-family host helper) — no gating implication.
- **Relaxation candidates:** none mined (no custom hash exists).
- **TTNN factory analysis (sheet-derived + code cross-check):** Concept `descriptor` (verified); custom hash **absent in code** (sheet now agrees, `no`); no `override_runtime_arguments` / `get_dynamic_runtime_args`; no pybind `create_descriptor` (nanobind binds only `ttnn::reshard`); no op-owned tensors; not a WorkloadDescriptor. Target concept: `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **Dead / unreachable code in `is_valid_for_legacy_reshard`** (`reshard_device_operation.cpp:34-50`): the `return out_mem_config.buffer_type() == BufferType::L1;` at line 39 is reached unconditionally whenever the line-34 `if` is false, so the entire `if (input_tensor.layout() == Layout::ROW_MAJOR) { … }` block at lines 41-50 is **unreachable**. Likely a logic bug introduced during editing (the intent was probably to reach the row-major shard-width checks). Route to the ops team.
- **Live `DPRINT` left in a shipping kernel** (`reshard_same_width_reader.cpp:46`): `DPRINT("addr: {}\n", addr);` in the unaligned reader path (plus commented-out `print_bf16_pages` at `:52-53,70`). Harmless when DPRINT is compiled out, but stray debug output. Route to the ops team.
- **Dead RTA read** in `reshard_reader.cpp:24` and `reshard_reader_diff_width.cpp:24`: `num_output_pages` is unpacked from an RTA but never referenced in either kernel. Minor.

## Recipe notes

- **Cross-column invariant for `Custom hash` (still stands from the prior audit).** The recipe lists cross-column invariants for `Runtime-args update` and `Op-owned tensors?` but not for `Custom hash`. `Custom hash` is device-op-level (`compute_program_hash` on the DeviceOperation), so it must be uniform across all factory rows of one DeviceOperation. The *prior* sheet violated this (5-yes/3-no) and that inconsistency was the entire RED; the *current* sheet is uniform `no`. Suggest adding `Custom hash` (and `Pybind descriptor`, likewise device-op-level) to the enumerated cross-column-invariant list so a future auditor keys on such an inconsistency directly — it is exactly the signal that caught the prior sheet bug.
- **A "spreadsheet-broken" RED on stable code is worth a fast re-audit path.** This op's prior RED was a pure sheet data-quality failure on code that was already fully GREEN on every structural gate; the fix was the owner correcting the sheet, and the re-audit (this run) flipped straight to GREEN with zero code change. The recipe's reassuring-framing section could note that a "spreadsheet-broken" RED is often the cheapest kind to clear — a good candidate for the maintainer to call out as an explicitly transient state distinct from a genuine structural/legacy blocker.
