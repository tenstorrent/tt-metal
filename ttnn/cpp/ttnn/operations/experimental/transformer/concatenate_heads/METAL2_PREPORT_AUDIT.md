# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/concatenate_heads`

- **`ConcatenateHeadsDeviceOperation`** (`device/concatenate_heads_device_operation.hpp:18`) is the only DeviceOperation in the directory.
  - **Direct-descriptor factory.** `create_descriptor` is a static member of the device-op itself, and there is no `program_factory_t` (`device/concatenate_heads_device_operation.hpp:27-28`; body at `device/concatenate_heads_program_factory.cpp:20-143`). The framework wraps it in the `MeshDeviceOperationAdapter::DirectDescriptorFactory` shim (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170`), which is selected by `HasDirectDescriptor` (`ttnn/api/ttnn/operation_concepts.hpp:158`).
  - Two kernels, both owned by the op, both data movement:
    - reader `device/kernels/dataflow/reader_tm_tile_layout_concat_heads.cpp` (`ReaderConfigDescriptor`)
    - writer `device/kernels/dataflow/writer_tm_tile_layout_concat_heads.cpp` (`WriterConfigDescriptor`)

    Both run over one `all_cores` rectangle of `(H/32) × B` cores (12 × 7–9 for the only legal shape). There is no compute kernel.

The op is a BERT-large TM. It reshuffles a `[B, 16, 384, 64]` TILE tensor into `[B, 1, 384, 1024]`, with B in 7–9 (`device/concatenate_heads_device_operation.cpp:18,27`). Its Python entry point is `ttnn.experimental.concatenate_heads` (`concatenate_heads_nanobind.cpp:28`). It is unrelated to `experimental/transformer/nlp_concat_heads`, which has its own kernels.

**Scope:** TTNN op, Gen1 (WH/BH) target. This is within the scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
*(The recipe tree isn't in this checkout (`Metal_Ports`, branch `edwinlee/PD_Metal_Ports`). The hash comes from the `Port_Recipe` checkout (branch `akertesz/op-porting-recipe`). This audit followed `/localdev/edwinlee/metal2_audit.md`, which is a symlink to that checkout's `ai/audit/metal2_audit.md`.)*

**Readiness sheet:** fetched live on 2026-10-01 via the Google Drive connector (`download_file_content`, CSV). One row for this op.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/concatenate_heads` |
| **Overall** | **GREEN (user waiver).** Every code-side gate is clean. The only RED, the stale readiness-sheet row, was waived by the user on 2026-10-01 (see Result). |
| **DOps / Factories** | `ConcatenateHeadsDeviceOperation` → direct `create_descriptor` (no factory struct). The sheet still lists `ConcatenateHeadsProgramFactory`, which was deleted in #57409. |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** Both kernels use `Noc`, `CircularBuffer`, `TensorAccessor` and `CoreLocalMem`. The only CB-index free function is the sanctioned `get_tile_size(cb_id)`. |
| *Prereqs* — Cross-op escapes | Ok. Includes are `tt_metal/hw/inc/*` only, and no kernel is borrowed. |
| *Feature Support* — overall | GREEN (all N/A) |
| *Feature Support* — Variadic-CTA | Ok. All CTAs are at fixed indices. |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet says `yes (with PD step)`, but the row is **stale**: `Concept` conflicts with the code, and the factory row is a phantom. Treated as **spreadsheet-broken**, which makes this a GATE routed to the readiness-sheet owner. |
| *TTNN Readiness* — Concept (current) | Code: **`descriptor`** (direct-descriptor shape, since PR #57409 on 2026-09-25). Sheet: `legacy device-op`. |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No. The default hash is used. Sheet agrees (`no` / backdoor `no`). |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No. Sheet agrees. |
| *TTNN Readiness* — `override_runtime_arguments` | No: the code has none. The sheet says `n/a`, which fits its stale `legacy` concept. |
| *TTNN Readiness* — Pybind `create_descriptor` | No. The only binding is the user entry point (`concatenate_heads_nanobind.cpp:28`). Sheet agrees. |
| *TTNN Readiness* — Op-owned tensors | No. Sheet agrees. |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept`. The sheet's `Porting Target` column agrees. The port must first introduce a factory struct (see Heads-ups). |
| *Port work* — Offset base pointer | none. Both address args are bare `Buffer*`, and tile offsets travel as separate scalars. |
| *Port work* — Tensor bindings (per binding) | `input` → **Case 1**. `output` → **Case 1**. Both use the `Buffer*`-binding form, and each address feeds a `TensorAccessor`. |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none. No accessor passes a 3rd argument. |
| *Port work* — CB endpoints | legal. CB 0 is a plain 1:1: reader FIFO-produces it and writer FIFO-consumes it. |

**CB endpoints** are dispositions, not gates. The op has one CB in one config, and it is a legal producer/consumer FIFO.

## Result

**GREEN by user waiver → brief issued** (`METAL2_PORT_BRIEF.md`). As audited, this was RED on the TTNN factory concept gate (spreadsheet-broken). On 2026-10-01 the user waived that gate ("if the sheet is just out of date, please proceed"): the blocker is only the sheet being out of date, not anything in the op. The sheet refresh is still owed to the readiness-sheet owner as housekeeping; it is no longer a port blocker. The original finding is kept below.

**Original finding (pre-waiver):** RED → blocked on the TTNN factory concept gate (spreadsheet-broken), routed to the **readiness-sheet owner** (Diego, `dgomez@tenstorrent.com`).

The live sheet still describes the op as it was before PR #57409, *[Cleanup] Port More Ops to PD* (`f5093e705ae`, 2026-09-25). That PR deleted the `ConcatenateHeadsProgramFactory` struct and its header `device/concatenate_heads_program_factory.hpp`, and moved `create_descriptor` onto the device-op. The sheet still shows:

- `Concept` = `legacy device-op`
- `Factory (variant)` = `ConcatenateHeadsProgramFactory`
- `Factory definition path` = that now-deleted `.hpp`

**No other gate fires.** Device 2.0, Feature compatibility, Offset base pointers and TensorAccessor 3rd argument are all clean on the code.

**This RED is cleared outside the op's code.** The sheet owner updates the row; nothing in the op changes. Per the recipe's exception, I ran all the informational subjects. The detail below should survive re-audit unchanged, so once the row is refreshed (`Concept` = `descriptor`, factory row renamed or removed, `Is able to port?` re-derived), the re-audit should be a quick re-read and issue the brief.

The op has a single code path. Whole-op RED, with no subset distinction (there is nothing to split).

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED (spreadsheet-broken)**, routed to the readiness-sheet owner to reconcile. The sheet row is `Op` = `experimental/transformer/concatenate_heads`, `Device operation` = `ConcatenateHeadsDeviceOperation`, `Factory (variant)` = `ConcatenateHeadsProgramFactory`.
  - **Primary-column conflict, `Concept`.** The sheet says `legacy device-op`. The code is `descriptor`: `static ProgramDescriptor create_descriptor(...)` on the device-op (`device/concatenate_heads_device_operation.hpp:27-28`), with no `create()` and no `override_runtime_arguments`. Before #57409 the device-op declared `using program_factory_t = std::variant<ConcatenateHeadsProgramFactory>;` (`git show f5093e705ae^:…/concatenate_heads_device_operation.hpp`). That line is gone now.
  - **Phantom factory row.** `ConcatenateHeadsProgramFactory` no longer exists. `Factory definition path` points at `device/concatenate_heads_program_factory.hpp`, which #57409 deleted (the diffstat shows `concatenate_heads_program_factory.hpp | 33 ------`). There is also a **missing row**: the code's only factory is the device-op's direct `create_descriptor` (wrapped by `DirectDescriptorFactory`), and no row matches it.
  - `Is able to port?` = `yes (with PD step)`. That PD step has since landed. This is a derived cell, so it is read, not vetted. It is moot here, because the conflicts above already make the row spreadsheet-broken.
  - **The rest of the primary cross-check is clean:**
    - `Custom hash` = `no` and backdoor = `no`. No `compute_program_hash`, `attribute_values` or `to_hash` exists in the op.
    - `Runtime-args update (get_dynamic_runtime_args)` = `no`. There is no hook.
    - `Override runtime args method?` = `n/a`. The code has none.
    - `Pybind descriptor` = `no`. There is no `create_descriptor` binding.
    - `Op-owned tensors?` = `no`.
    - `Smuggled pointer` = `no`. Both addresses use the `Buffer*`-binding form (see Port-work summary).

    No cross-column invariant is violated. `Known op issues` is empty.
  - **Path forward:** the sheet owner refreshes the row. Re-audit after that is expected to go GREEN, since there are no code-side blockers.

- **Device 2.0 (every kernel used): GREEN.**
  - **Reader** (`reader_tm_tile_layout_concat_heads.cpp`):
    - `Noc noc` (`:13`)
    - `CircularBuffer cb_in0(cb_id_in0)` with `reserve_back` / `push_back` / `get_write_ptr` (`:30-35,50`)
    - `TensorAccessor(in0_args, in0_tensor_addr)` (`:28`)
    - `noc.async_read(s0, CoreLocalMem<uint32_t>(…), size, {.page_id=…}, {})` and `noc.async_read_barrier()` (`:39-49`)
  - **Writer** (`writer_tm_tile_layout_concat_heads.cpp`): the mirror image.
    - `CircularBuffer cb_out0` with `wait_front` / `pop_front` / `get_read_ptr` (`:31-36,48`)
    - `noc.async_write(...)` and `noc.async_write_barrier()` (`:39-47`)
  - **The one CB-index free function** is `get_tile_size(cb_id)` (reader `:27`, writer `:28`), which is **sanctioned**.
  - No raw `noc_async_*`, no `*AddrGen*`, no `get_noc_addr`, no raw semaphores.
  - The kernels were moved to Device 2.0 in #45843 (2026-06-03).

- **Feature compatibility: GREEN** (no gate fired).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | There are no `GlobalCircularBuffer`, `global_circular_buffer` or `remote_*` signals. The one `CBDescriptor` (`concatenate_heads_program_factory.cpp:109-117`) leaves `.global_circular_buffer` unset. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` is not set (default 0), and the CB is not buffer-backed. |
  | GlobalSemaphore | N/A | The op has no semaphores at all. |

- **CB endpoints (GATE-free): legal.** See Port-work summary.

- **Offset base pointers: GREEN.**
  - The reader RTA 0 `in0_tensor_addr` is `in0_buffer` (`concatenate_heads_program_factory.cpp:127`).
  - The writer RTA 0 `out_tensor_addr` is `out_buffer` (`:133`).

  Both are bare `Buffer*` with no host arithmetic. The per-core offsets are separate scalars:

  - reader RTA 1 `in0_tensor_tile_id`, computed at `:122`
  - writer RTA 1 `out_tensor_tile_id`, computed at `:134`

  Each feeds `{.page_id = …}` on the accessor. The op is not in the `2026-07-19_offset_base_pointers.md` tables, so this is the "no fold, not in tables" outcome: clean.

- **TensorAccessor 3rd argument: N/A.** The subject never fires: both accessors are 2-arg (reader `:28`, writer `:29`). The op is not in the `2026-07-06_tensor_accessor_3rd_arg_triage.md` table either.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `input` (`tensor_args.input`) is **Case 1**. The host passes `in0_buffer` as `Buffer*` (`concatenate_heads_program_factory.cpp:127`) and its accessor args as CTAs from index 3 (`:80`; kernel `TensorAccessorArgs<3>()` at reader `:24`). The kernel builds `TensorAccessor(in0_args, in0_tensor_addr)` (reader `:28`) and reads only through it (`:39-44`).
  - `output` is **Case 1**. The host passes `out_buffer` as `Buffer*` (`:133`) and its accessor args as CTAs from index 2 (`:86`; writer `:25`). The writer builds `TensorAccessor(out_args, out_tensor_addr)` (writer `:29`) and writes only through it (`:39-40`).
  - The `Buffer*`-binding form is correct on cache hits today. The framework patches it, and the device-op comment at `concatenate_heads_device_operation.hpp:24-26` says the same. So this is routine work, not a hazard.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal, with one config.

  | CB (index) | Node set | Touchers | Disposition |
  |---|---|---|---|
  | `src0_cb_index` = 0 (`concatenate_heads_program_factory.cpp:107-117`; reader `cb_id_in0`, writer `cb_id_out0`) | every core in `all_cores` | reader: **locked producer** (`reserve_back` `:35`, `push_back` `:50`; its `get_write_ptr` `:31` is a peek on its own binding). Writer: **locked consumer** (`wait_front` `:36`, `pop_front` `:48`; its `get_read_ptr` `:32` is a peek). | **plain 1:1.** Reader is PRODUCER, writer is CONSUMER. No flag. |

  I checked for a hidden second writer: neither kernel raw-writes into the other's side, and the op has no semaphores.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** none. Both kernel files are owned by the op and are instantiated only by this factory (`concatenate_heads_program_factory.cpp:89-91,98-100`; there are no other in-tree references outside `experimental/quasar/`). No `_metal2` fork exists. The port converts them in place.
- **RTA varargs:** none. Each kernel reads exactly two fixed RTAs (reader `:16-17`, writer `:18-19`), which name directly:
  - reader: `in0_tensor_tile_id` (and the address, which becomes a binding)
  - writer: `out_tensor_tile_id`

  CTAs are also fixed-index (reader `0..2`, writer `0..1`, then the accessor args), so there are no CTA varargs.
- **Direct-descriptor shape: the port must introduce a factory struct.** The op declares `create_descriptor` on the device-op with no `program_factory_t` (`concatenate_heads_device_operation.hpp:27-28`). The `DirectDescriptorFactory` shim (`mesh_device_operation_adapter.hpp:170`, gated by `HasDirectDescriptor`, `operation_concepts.hpp:158`) is keyed on the name `create_descriptor` and has no `create_program_artifacts` counterpart. Follow `ttnn_factory.md` §"3. Give a direct-descriptor op a conventional program factory":
  - nest a factory struct (e.g. `ConcatenateHeadsProgramFactory`, the pre-#57409 name) with `create_program_artifacts`
  - add `using program_factory_t = std::variant<...>;`
  - remove the device-op-level `create_descriptor`

  Keep the body in `device/concatenate_heads_program_factory.cpp`, and record the change under Handoff points. *(Check again at port time in case TTNN has added a `program_factory_t` since this audit.)*
- **Stale kernel include.** Both kernels include `api/dataflow/circular_buffer.h` (reader `:8`, writer `:9`). When the port swaps `CircularBuffer` → `DataflowBuffer`, that include goes too (port-recipe kernel-side whitelist).
- **`get_tile_size(cb_id)`** (reader `:27`, writer `:28`) is a Device 2.0-sanctioned free function. The port moves it onto the DFB object (`dfb.get_tile_size()`, whitelist rule 7). It's a port-stage swap, not a Device 2.0 fix.
- **Pointer captured once, walked linearly.** The reader takes `get_write_ptr()` once before its loop (`:31`) and advances it linearly across all 16 `reserve_back` rounds without wrapping. The writer does the same with `get_read_ptr()` (`:32`). This works only because the CB is 64 tiles and the kernel writes 32 tiles per core (see Misc anomalies). Keep the access pattern exactly as is: on Gen1 a DFB lowers to the same circular buffer, so the pointer semantics are identical.

## Team-only

- **Out-of-directory coupling & donor shape: ✓ clean.**
  - Function-call escapes: every include resolves under `tt_metal/hw/inc/`:
    - `api/dataflow/dataflow_api.h`
    - `api/dataflow/noc.h`
    - `api/dataflow/circular_buffer.h`
    - `api/core_local_mem.h`
    - `api/tensor/noc_traits.h`
    - `tensix_types.h`, plus `<array>` in the writer

    That makes them all donor class 1 (LLK/HAL), with no concern.
  - No donor functions are called, so there is no per-call table.
  - Borrowed kernel files: none.
- **Relaxation candidates:** none (no custom hash to mine).
- **TTNN factory analysis:**
  - op-owned tensors: none
  - MeshWorkload need: none (plain `ProgramDescriptor`)
  - pybind `create_descriptor`: none
  - other risky pybind: none (only `bind_function` of the user API)
  - custom hash: none
  - `get_dynamic_runtime_args`: none
  - `override_runtime_arguments`: none
  - target concept: **`ProgramSpecFactoryConcept`**

## Misc anomalies  *(team-only, non-gating)*

- **Writer pops 2 more tiles than were pushed.** `out_num_tiles_read` starts at `in0_w_tiles` (2), and after each of the `in0_c` (16) rounds it is incremented by `+= in0_w_tiles` (`writer_tm_tile_layout_concat_heads.cpp:33,44`). It therefore ends at 34. The final `cb_out0.pop_front(out_num_tiles_read)` (`:48`) then pops 34 tiles, while the reader pushes only 16 × 2 = 32. The writer's cumulative `wait_front(2, 4, …, 32)` (`:36`) is correct. Only the trailing pop count is off by one round. It sits at kernel exit, so it is probably harmless, but it leaves the CB's acked count past its received count. Route to the ops team.
- **The "double buffer" CB is never double-buffered.** `cb0_tiles = per_core_tiles * 2` (`concatenate_heads_program_factory.cpp:108`) allocates 64 tiles. Both kernels walk a single linear 32-tile region and never reuse it (see Heads-ups). Half the CB is unused L1. The comment is also misleading: the linear walk *depends on* capacity ≥ 32, not on double-buffering.
- **The batch-vs-grid check is debug-only.** `num_cores_y = ashape[0]` (batch, 7–9) is checked against `compute_with_storage_grid_size.y` only by `TT_ASSERT` (`concatenate_heads_program_factory.cpp:50-51`), which compiles out in release. `validate_on_program_cache_miss` checks only that the requested grid fits the device (`concatenate_heads_device_operation.cpp:38-42`). A caller-supplied grid with `y < B` is not rejected.
- **`compute_with_storage_grid_size` only feeds asserts, yet it is hashed.** The default hash covers the whole `ConcatenateHeadsParams` (`concatenate_heads_device_operation_types.hpp:12-15`). The program is sized purely from the input shape (`concatenate_heads_program_factory.cpp:48-49`), so calls that differ only in this argument produce identical programs under separate cache entries.
- **Input layout isn't validated.** The kernels treat the input as tiles (`TensorAccessor` page = tile). `validate_on_program_cache_miss` checks shape and dtype (`concatenate_heads_device_operation.cpp:22-27`) but not `Layout::TILE`.
- **Nanobind doc understates dtype support.** The doc says BFLOAT8_B only (`concatenate_heads_nanobind.cpp:18,21`), but validation also accepts BFLOAT16 (`concatenate_heads_device_operation.cpp:22-25`).

## Recipe notes

- **Stale-sheet RED after a recent PD migration (repeat).** This is the same shape as `experimental/test/hang_device` from PD-migration batch #57409: a `Concept` conflict, a phantom factory row, and `Is able to port?` = `yes (with PD step)`. The `yes (with …)` tag still isn't covered by the routing rules. I treated it as moot, as there. Every other ex-#57409 op should be expected to hit this until the sheet is refreshed. A batch sheet refresh, or a rule for issuing a provisional brief when the *only* RED is "PD step landed after the sheet was derived", would save a re-audit round-trip per op.
- **Provenance in a separate checkout** (repeat): the provenance `git log` prints nothing in `Metal_Ports`. The hash above comes from the sibling `Port_Recipe` checkout.
