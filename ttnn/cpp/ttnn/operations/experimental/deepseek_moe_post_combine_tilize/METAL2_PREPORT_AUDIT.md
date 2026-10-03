# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize`

- **`DeepseekMoEPostCombineTilizeDeviceOperation`** (`device/deepseek_moe_post_combine_tilize_device_operation.{hpp,cpp}`)
  - One `ProgramDescriptor` factory, defined **directly on the device-op** as `DeepseekMoEPostCombineTilizeDeviceOperation::create_descriptor` (`device/deepseek_moe_post_combine_tilize_program_factory.cpp:23`). There is no separate factory struct and no `program_factory_t` variant. That shape dates from the PD migration in `f5093e705ae` (#57409, 2026-09-25), which deleted the old `DeepseekMoEPostCombineTilizeProgramFactory` struct and its `.hpp`.
  - Kernels (all op-owned, all referenced by the factory):
    - reader — `device/kernels/deepseek_moe_post_combine_tilize_reader.cpp` (RISCV_1 / NOC_0)
    - compute — `device/kernels/deepseek_moe_post_combine_tilize_compute.cpp`
    - writer — `device/kernels/deepseek_moe_post_combine_tilize_writer.cpp` (RISCV_0 / NOC_1)
  - No unreferenced kernel files in the op directory.

**Scope:** TTNN op, Gen1 (WH/BH) target, within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
*(This came from the `/localdev/edwinlee/Port_Recipe` checkout, branch `akertesz/op-porting-recipe`. The recipe I was given, `/localdev/edwinlee/metal2_audit.md`, is byte-identical to that checkout's `ai/audit/metal2_audit.md`. In the op's own checkout, `/localdev/edwinlee/Metal_Ports`, the provenance command prints nothing because that tree has no `metal_2.0/` docs.)*

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize` |
| **Overall** | **GREEN, by user override.** The TTNN readiness gate couldn't be checked against a live sheet. The user directed the audit to ignore the stale copy and treat the gate as cleared, based on the code cross-check (see Result). Every other gate is clear on the evidence. |
| **DOps / Factories** | `DeepseekMoEPostCombineTilizeDeviceOperation` → its own `create_descriptor` (single factory) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes**. All 3 kernels are Device 2.0 compliant. |
| *Prereqs* — Cross-op escapes | **Ok**. No out-of-directory kernel includes beyond `tt_metal` API headers and `tt-metalium/constants.hpp`. No borrowed kernel files. |
| *Feature Support* — overall | **GREEN**. No Appendix A entry fires. |
| *Feature Support* — Variadic-CTA | **Ok**. No CTA read at a varying index. *(Appendix A has no Variadic-CTA entry; see Recipe notes.)* |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes (user override; not verified live).** The live sheet couldn't be fetched (no Google Drive connector in this session). The only local copy (Sep 4) predates the op's PD migration and says `yes (with PD step)`; the PD step has since landed (#57409). The user directed the audit to ignore the stale copy and clear the gate. See Gate detail. |
| *TTNN Readiness* — Concept (current) | **`descriptor`** in code (`program_factory.cpp:23`, `device_operation.hpp:27`). The stale sheet copy says `legacy device-op`. |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | **No**. There's no `compute_program_hash`, `attribute_values` or `to_hash` in the device-op. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** |
| *TTNN Readiness* — `override_runtime_arguments` | **No**. It was removed by #57409; the comment at `device_operation.hpp:24-26` records that. |
| *TTNN Readiness* — Pybind `create_descriptor` | **No**. `deepseek_moe_post_combine_tilize_nanobind.cpp:40` binds only the user-facing function through `ttnn::bind_function`. |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (no op-owned tensors) |
| *Port work* — Offset base pointer | **none**. The only address argument is a clean `Buffer*` base. The intra-row offset travels as a separate scalar. |
| *Port work* — Tensor bindings (per binding) | `input` → **Case 1** · `output` → **clean** (borrowed-memory CB) |
| *TTNN Readiness* — TensorParameter relaxation | `none` in the stale copy. **Not confirmed live.** |
| *Port work* — TensorAccessor 3rd arg | **none**. No accessor passes a 3rd argument. |
| *Port work* — CB endpoints | **legal**. Both CBs are plain 1P+1C FIFOs. |

## Result

**GREEN → brief issued (`METAL2_PORT_BRIEF.md`), with a user override on the TTNN factory concept gate.**

> **Override record (2026-10-01).** The first pass of this audit came back RED, because the TTNN gate couldn't be verified against a live readiness sheet (details below). After reviewing that result, the user directed: *"ignore the out of date sheet and proceed."* The gate is therefore treated as cleared on the code cross-check alone: the op is `descriptor`, there's no `get_dynamic_runtime_args`, no `Known op issues` in the last known row, and the relaxation is `none`. The readiness sheet itself was **not** confirmed. If a later live fetch disagrees, for example a `Known op issues` entry or a non-`none` relaxation added after Sep 4, the sheet wins and the port should stop. The text below is the original RED analysis, kept for the record.

In plain terms: every gate this audit can evaluate from the code is clear. That covers Device 2.0, feature compatibility, offset base pointers and the TensorAccessor 3rd argument. The one gate taken from the readiness sheet couldn't be read live. The live fetch needs the claude.ai Google Drive connector (`mcp__claude_ai_Google_Drive__download_file_content`), and it isn't available in this session. The recipe forbids relying on a copy this session didn't pull itself.

The only copy on disk (`Port_Recipe/.../analyses/ttnn_op_porting_readiness.csv`, fetched 2026-09-04) is stale for this op in exactly the way you'd expect. It predates the op's PD migration (#57409, landed 2026-09-25). So it still lists `Concept = legacy device-op` and a `DeepseekMoEPostCombineTilizeProgramFactory` row whose factory and `.hpp` no longer exist.

**Path forward (expected to be cheap):** fetch the live sheet in a session that has the Drive connector, then re-check this one subject.
- If the live row shows `Concept = descriptor` and `Is able to port? = yes`, and the factory row matches the code, the audit goes GREEN. The brief can then be issued from this report's port-work and heads-up sections unchanged, because none of them depend on the sheet.
- If the live row still shows the pre-migration state, it's a **spreadsheet-broken** finding (primary-column conflict plus phantom factory row). It routes to the readiness-sheet owner (Diego) to reconcile.

This RED clears **without touching the op's code**. So, per the scoping-rule exception, every informational subject was run in full below, and that detail stays valid for the re-check.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED, unverifiable.** I could not fetch a live copy. `ToolSearch select:mcp__claude_ai_Google_Drive__download_file_content` returned "No matching deferred tools found". For the record, here is what the stale 2026-09-04 copy says for `experimental/deepseek_moe_post_combine_tilize`:
  - `Device operation` = `DeepseekMoEPostCombineTilizeDeviceOperation`; `Factory (variant)` = `DeepseekMoEPostCombineTilizeProgramFactory`
  - `Concept` = `legacy device-op`; `Op Classification` = `Legacy Op`; `Porting Target` = `ProgramSpecFactoryConcept`
  - `Custom hash` = `no`; `Backdoor custom hash` = `no`; `Runtime-args update (get_dynamic_runtime_args)` = `no`; `Override runtime args method? (PD only)` = `n/a`; `Pybind descriptor` = `no`; `Smuggled pointer` = `no`
  - `Known op issues` = *(blank)*; `TensorParameter relaxation` = `none`; `Op-owned tensors?` = `no`
  - `Is able to port?` = `yes (with PD step)`
  - `Factory definition path` = `…/device/deepseek_moe_post_combine_tilize_program_factory.hpp` (**this file no longer exists**); `Declared in` = `…/device/deepseek_moe_post_combine_tilize_device_operation.hpp`

  I cross-checked those columns against the current code:

  | Column | Stale sheet | Code (evidence) | Match? |
  |---|---|---|---|
  | `Concept` | `legacy device-op` | `descriptor`. `static ProgramDescriptor create_descriptor(...)` at `device_operation.hpp:27`, defined at `program_factory.cpp:23`; no `create()`/`override_runtime_arguments()` | **conflict** (explained by #57409) |
  | Factory-set | `DeepseekMoEPostCombineTilizeProgramFactory` | No such struct. The factory is the device-op's own `create_descriptor`, and #57409 deleted `program_factory_t` and the factory `.hpp` | **phantom row** (explained by #57409) |
  | `Custom hash` | `no` | no `compute_program_hash`, `attribute_values` or `to_hash` | ✓ |
  | `get_dynamic_runtime_args` | `no` | absent | ✓ |
  | `Override runtime args method?` | `n/a` | absent (now meaningful as `no`, since the op is PD) | ✓ in substance |
  | `Pybind descriptor` | `no` | `nanobind.cpp:40` binds only `ttnn::experimental::deepseek_moe_post_combine_tilize` | ✓ |
  | `Op-owned tensors?` | `no` | `ProgramDescriptor` return, so no `buffers` vector | ✓ |

  Cross-column invariants hold in the stale copy. Both conflicts come from one event, the PD migration, after the snapshot. I'm not filing them as a broken sheet, because the copy I hold isn't the live sheet. **On re-fetch:** if the row reflects the migration and reads `yes`, the gate clears. If it doesn't, file it as spreadsheet-broken (owner: Diego) for the `Concept` conflict and the phantom factory row.

- **Device 2.0 (every kernel used):** **GREEN.** All three kernels use Device 2.0 idioms throughout. There are no CB-index-keyed free-function holdovers, no addr-gens and no raw NoC calls.
  - reader: `Noc noc;` (`reader.cpp:15`), `TensorAccessor(input_tensor_accessor_args, input_tensor_address)` (`:26`), `CircularBuffer cb_tilize_input(...)` with `.reserve_back`/`.get_write_ptr()`/`.push_back` (`:30-33`, `:49`), `noc.async_read(accessor, CoreLocalMem<uint32_t>(l1_write_addr), size, {.page_id, .offset_bytes}, {})` (`:37-42`), `noc.async_read_barrier()` (`:48`).
  - writer: `CircularBuffer` with `.wait_front`/`.pop_front` (`writer.cpp:13-16`).
  - compute: `CircularBuffer` wrappers for FIFO sync (`compute.cpp:19-31`). The compute LLK calls `compute_kernel_hw_startup`, `fast_tilize_init`, `fast_tilize_block` and `fast_tilize_uninit` (`:22, 23, 28, 33`) take CB indices. They're compute-API LLK signatures, not Device 2.0 data-movement free functions, so they aren't holdovers.

  Table of violations: *(none)*

- **Feature compatibility:** GREEN. No gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `global_circular_buffer` field, no `remote_index`/`remote_cb`. The output CB's `.buffer = output_tensor.buffer()` (`program_factory.cpp:85`) is the ordinary borrowed-memory CB, which is supported. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Neither `CBDescriptor` (`program_factory.cpp:63-71`, `:77-86`) sets `.address_offset`, so it defaults to 0. No `cb_descriptor_from_sharded_tensor` or 4-arg `UpdateDynamicCircularBufferAddress`. |
  | GlobalSemaphore | N/A | No semaphores of any kind. |

- **CB endpoints (GATE-free):** **legal**. Both CBs are plain 1P+1C on every node in `op_cores`. There's a single config, with no `#ifdef`s or branches in the kernels.
  - `c_0` (`tilize_input_cb_id`, `program_factory.cpp:62-71`): the reader is the locked producer (`reserve_back`/`push_back`, `reader.cpp:32, 49`; its `get_write_ptr` at `:33` is a peek on its own producer binding). Compute is the locked consumer (`wait_front`/`pop_front`, `compute.cpp:25, 31`). The writer never touches `c_0`.
  - `c_1` (`tilize_output_cb_id`, `program_factory.cpp:76-86`, borrowed from the output buffer): compute is the locked producer (`reserve_back`/`push_back`, `compute.cpp:26, 30`). The writer is the locked consumer (`wait_front`/`pop_front`, `writer.cpp:15-16`). The reader never touches `c_1`. I hunted for a hidden second writer: no kernel other than compute calls `get_write_ptr`/`fifo_wr_ptr` on `c_1`, and the op has no semaphores.

- **Offset base pointers:** **GREEN.** There's one address RTA: reader RTA[2] `input_tensor_address`, delivered as a `Buffer*` (`input_tensor.buffer()`, `program_factory.cpp:163`), with no host-side arithmetic. The per-core offsets `intra_row_byte_offset` (RTA[0]) and `row_page_offset` (RTA[1]) are separate scalars (`program_factory.cpp:154-162`). The kernel applies them through the accessor's `{.page_id, .offset_bytes}` (`reader.cpp:35, 41`), which means the offset is already split out. The op isn't in `2026-07-19_offset_base_pointers.md`, and the scan agrees. Type 3 and Type 4 are absent.

- **TensorAccessor 3rd argument:** **N/A.** No accessor in the op passes a 3rd argument (`reader.cpp:26` is the 2-arg form), so this subject doesn't fire. The op isn't in `2026-07-06_tensor_accessor_3rd_arg_triage.md`; the sibling ops `deepseek_moe_post_combine_reduce` and `deepseek_moe_fast_reduce_nc_fused` are listed there, but not this one.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `input` (interleaved, ROW_MAJOR, bf16; one page = one row): **Case 1 (`TensorAccessor`)**. The host passes `Buffer*` as reader RTA[2] (`program_factory.cpp:163`; the `Buffer*`-binding form, which the framework patches on cache hit today, so it's not a live hazard). `TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args)` sits at CTA offset 0 (`program_factory.cpp:94`). The kernel builds `TensorAccessor(TensorAccessorArgs<0>(), input_tensor_address)` (`reader.cpp:25-26`), and every access goes through `noc.async_read(accessor, …, {.page_id, .offset_bytes})`. Port: `TensorParameter`/`TensorBinding`, `TensorAccessor(tensor::<name>)`. The RTA[2] address, the `TensorAccessorArgs` CTA plumbing and `TensorAccessorArgs<0>` all go away.
  - `output` (L1 ND-sharded, TILE): **clean**. It's a borrowed-memory CB (`c_1` with `.buffer = output_tensor.buffer()`, `program_factory.cpp:85`) and is never accessed through an address RTA or an accessor. Port via `DataflowBufferSpec::borrowed_from` the output `TensorParameter`.
- **TensorParameter relaxation:** `none` (stale copy; re-confirm on live fetch).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal. `c_0`: reader PRODUCER, compute CONSUMER. `c_1` (borrowed from output): compute PRODUCER, writer CONSUMER. No self-loops, no multi-binding, no dead CBs.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** none. The census (`grep -rl <kernel-filename> ttnn/cpp/ttnn/operations/`, excluding `experimental/quasar/`) finds each of the three kernels bound only by this op's factory. So there's no `_metal2` fork question, and the kernels convert in place inside the op's own directory. Don't look under `experimental/quasar/` for a model.
- **RTA varargs:** none. The reader reads a fixed run of 3 args through `rt_args_idx++` (`reader.cpp:20-23`), so name each one (`intra_row_byte_offset`, `row_page_offset`, plus the address that becomes the tensor binding). No CTA varargs. The writer and compute have no RTAs.
- **Compute kernel passes CB indices to LLK.** `compute_kernel_hw_startup`, `fast_tilize_init`, `fast_tilize_block` and `fast_tilize_uninit` (`compute.cpp:22, 23, 28, 33`) take `uint32_t` CB ids. They're `constexpr` from named CTAs today (`:13-14`), so `dfb::name`'s constexpr `uint32_t` conversion covers them. Keep them as constexpr token-form uses.
- **Compute kernel includes `api/dataflow/circular_buffer.h`** (`compute.cpp:9`). It's the CB→DFB include swap like the reader and writer (`reader.cpp:9`, `writer.cpp:7`). Note it so the compute kernel isn't missed.
- **Reader writes into the CB through `CoreLocalMem<uint32_t>(cb.get_write_ptr())`** (`reader.cpp:33, 39`) rather than passing the CB as the destination. That's an existing Device 2.0 idiom: keep it as written (DFB `get_write_ptr()` swap only). Rewriting it to a CB/DFB destination isn't port scope.
- **Unread named CTA `input_row_page_size`** (`program_factory.cpp:105`). The factory passes it, but no kernel reads it (see Misc anomalies). It isn't a dead *CB* CTA, so the dead-CB drop rule doesn't cover it. Carry it or not per the port recipe's guidance on unread named args, and record whichever you choose.
- **NoC/RISC assignment is swapped from the usual convention** (reader on RISCV_1/NOC_0, writer on RISCV_0/NOC_1, both `DM_DEDICATED_NOC`; `program_factory.cpp:109-113`, `:141-145`). Carry it verbatim into the `KernelSpec` configs.
- **Per-core RTAs depend on shard orientation** (`program_factory.cpp:147-164`). The core list comes from `corerange_to_cores(op_cores, std::nullopt, is_row_major_shard_orientation)`, and the offsets are computed per index in that order. Keep the same core ordering when building `ProgramRunArgs`.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.**
  - Op kernels' out-of-directory includes are all class 1 (`tt_metal/*`): `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`, `api/compute/cb_api.h`, `api/compute/tilize.h`, plus `tt-metalium/constants.hpp`. There are no kernel_lib, `kernel/`, `kernel_helper_functions/`, in-family or cross-family donors.

    | Op kernel | Donor file | Class | Status |
    |---|---|---|---|
    | reader / writer / compute | `tt_metal` API headers, `tt-metalium/constants.hpp` | 1 | ✓ |

  - Borrowed kernel files: none. All three kernels are op-owned and bound only here.
- **Relaxation candidates:** none. There's no custom hash to mine.
- **TTNN factory analysis:**
  - Current concept: `descriptor`. `create_descriptor` lives on the device-op itself (`device_operation.hpp:27`, `program_factory.cpp:23`).
  - Op-owned tensors: none. Not a MeshWorkload.
  - Custom hash: none. `get_dynamic_runtime_args`: none. `override_runtime_arguments`: none.
  - Pybind `create_descriptor`: none. Other risky pybind: none; `nanobind.cpp:40-46` binds only the public function.
  - Target concept: `ProgramSpecFactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Dead named CTA:** `{"input_row_page_size", input_row_page_size}` is passed to the reader (`program_factory.cpp:105`), but `reader.cpp` never calls `get_named_compile_time_arg_val("input_row_page_size")`. The kernel gets page geometry from `TensorAccessorArgs` instead. It's harmless, but because it's a CTA it bakes the input row width into the compiled kernel binary.
- **Unused include:** `#include "ttnn/operations/moreh/moreh_helper_functions.hpp"` (`device_operation.cpp:10`). Nothing from moreh is used, so this is a cross-family header dependency on the host side with no use.
- **Output spec uses `padded_shape()` as the shape** (`device_operation.cpp:90-96`). The output `TensorSpec` is built from the input's padded shape, so any logical/padded distinction on the input is lost in the output. It's probably intentional for this DeepSeek-specific op, which validates tile-aligned shard geometry. Noting it for the owners.
- **`validate_on_program_cache_hit` checks only storage and buffer presence** (`device_operation.cpp:14-19`). That's normal for TTNN, since the default hash covers the tensor spec. Noting it only because the factory's correctness depends on the shard-grid/shape invariants checked on the miss path.

## Questions for the user

1. **Readiness sheet fetch (resolved by override):** I asked for a live fetch or a fresh CSV, because the stale copy predates #57409 (2026-09-25). The user chose to ignore the stale sheet and proceed, so the gate was cleared by override and the brief was issued. A live re-check is still recommended before the port PR lands, to catch any `Known op issues` or relaxation entry added after Sep 4.

## Recipe notes

- **No fallback when the Drive connector is missing.** `ttnn_op_porting_readiness.md` says to fetch live every session, never to reuse an earlier copy, and never to delegate the fetch. It doesn't say what the audit verdict should be when the connector isn't loadable at all ("No matching deferred tools found", as opposed to an auth error the Troubleshooting section covers). I treated it as an unverifiable gate, RED and routed as a question, and still recorded the stale copy's values and cross-check as a labelled prior. A one-line rule would help here: either "RED-unverifiable, cleared by re-fetch only" or "the human may supply a CSV".
- **Stale-copy conflicts vs "spreadsheet is broken".** The routing rules assume the copy in hand *is* the live sheet. When the only available copy is known-stale and the conflict is fully explained by a dated commit after the snapshot (here #57409), filing it as spreadsheet-broken would be an allegation against the live sheet based on old data. It might be worth stating that spreadsheet-broken requires a copy fetched this session.
- **Template has a `Feature Support — Variadic-CTA` row, but Appendix A has no Variadic-CTA entry.** The RTA varargs subject says CTA varargs port via `KernelAdvancedOptions::compile_time_varargs` and "don't gate either". So the status-summary row reads like a leftover from when it was an Appendix A entry. I filled it as `Ok` (no CTA varargs present).
- **Factory defined on the device-op.** Post-PD-migration ops like this one put `create_descriptor` directly on the DeviceOperation, with no factory struct. The sheet's `Factory (variant)` naming, and the factory-set match check, could note how such ops should appear, for example with the device-op name as the factory. Otherwise every op converted this way will hit the phantom-row trigger until the sheet catches up.
- **Provenance command location.** The recipe says to run `git log … -- docs/source/.../metal_2.0/` "from the checkout root". When the op checkout and the recipe-doc checkout differ, as here (the recipe was handed over as a standalone file outside both), the command run in the op checkout prints nothing. I recorded the recipe-doc checkout's hash and confirmed the given file is byte-identical to it.
