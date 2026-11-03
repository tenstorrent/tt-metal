# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm`

- **`MoEGateMMDeviceOperation`** (`device/moe_gate_mm_device_operation.hpp`)
  - Single program builder: `MoEGateMMDeviceOperation::create_descriptor` (declared `device/moe_gate_mm_device_operation.hpp:22`, defined `device/moe_gate_mm_program_factory.cpp:70`). There is **no** separate program-factory struct. The PD migration (`dc788a6a7ff`, 2026-09-29, #57521) deleted the old `MoEGateMMProgramFactory` / `moe_gate_mm_program_factory.hpp` and put `create_descriptor` on the device-op.
  - Kernels (all owned by the op, all under `device/kernels/`): `dm0.cpp` (`ReaderConfigDescriptor`, weight DRAM reader), `dm1.cpp` (`WriterConfigDescriptor`, ring/collector NoC traffic), `compute.cpp` (matmul + SFPU top-k pipeline; includes the local headers `bias_bcast_sfpu.h`, `top2_sum_sfpu.h`, `top4_sfpu.h`, `top8_sfpu.h`, `top8_merge_sfpu.h`).
- The op runs on **Wormhole only**: `TT_FATAL(num_cores == 12, …)` at `device/moe_gate_mm_program_factory.cpp:78-85` (the ring is hardcoded for 12 DRAM-aligned workers). This audit is static. Nothing was run on hardware (the host has Blackhole cards only).

**Scope:** TTNN op, Gen1 (WH/BH) target, within the scope of `audit/metal2_audit.md`.

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`

**Readiness data provenance:** The main (launching) session fetched the readiness sheet ("Operations analysis", file ID `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`) on 2026-11-03 with the Google Drive MCP `download_file_content` as `text/csv`. It base64-decoded the result into `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/analyses/ttnn_op_porting_readiness.csv`. This audit (a subagent) read that local copy and did not re-fetch.

**Worktree:** `/localdev/iwrosz/tt-metal-moegate-audit`, branch `iwrosz/moe_gate_mm_2.0`. This is recipe branch `origin/akertesz/op-porting-recipe` (`c41f7f4ef0a`) with `origin/main` merged in (`1de7db85c37`).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm` |
| **Overall** | **RED** |
| **DOps / Factories** | `MoEGateMMDeviceOperation` → `create_descriptor` (single program, on the device-op; the sheet still names it `MoEGateMMProgramFactory`) |
| *Prereqs* — Device 2.0 (every kernel used) | **No.** RED, routed to the Device 2.0 track. **Broad Device 1.0** in `dm0.cpp` and `dm1.cpp`: all of their NoC and semaphore traffic is raw legacy API. The CB access already uses `CircularBuffer` wrappers, and one `Noc`/`Semaphore<>` block exists in dm1. `compute.cpp` is clean. |
| *Prereqs* — Cross-op escapes | Ok. No donor kernels and no out-of-directory includes beyond `tt_metal/*` LLK/API headers. The full inventory was skipped (see below). |
| *Feature Support* — overall | GREEN. No Appendix A entry fires. |
| *Feature Support* — Variadic-CTA | Ok (not in Appendix A). |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet: `yes (with PD step)`. **RED: spreadsheet is broken.** The sheet's `Concept` (`legacy device-op`) contradicts the code (`descriptor`), and the row names a factory and definition file that no longer exist. Routed to the readiness-sheet owner. |
| *TTNN Readiness* — Concept (current) | Code: **`descriptor`**. Sheet: `legacy device-op` (stale). |
| *TTNN Readiness* — Execution Model | `SPMD` (sheet). Single `ProgramDescriptor`, not mesh-shaped, so the `Secretly SPMD` columns don't apply. |
| *TTNN Readiness* — Custom hash | No (sheet `no`; code agrees: no `compute_program_hash` and no `attribute_values`/`to_hash`) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet `no`; code agrees) |
| *TTNN Readiness* — `override_runtime_arguments` | No. The code has none, because the PD migration removed the legacy one. The sheet says `n/a`, which follows from its stale `legacy device-op` concept. On a `descriptor` row it should read `no`. |
| *TTNN Readiness* — Pybind `create_descriptor` | No (sheet `no`; `moe_gate_mm_nanobind.cpp:13` binds only the free function `ttnn::experimental::deepseek::moe::moe_gate_mm`) |
| *TTNN Readiness* — Op-owned tensors | No (sheet `no`; `create_descriptor` returns a plain `ProgramDescriptor`) |
| *TTNN Readiness* — Target concept | Sheet `Porting Target`: `ProgramSpecFactoryConcept`. This is derived from a stale row, so re-read it after the sheet is reconciled. It is consistent with the code having no `override_runtime_arguments`. |
| *Port work* — Offset base pointer | none. GREEN: all three address RTAs are clean `Buffer*` bases. |
| *Port work* — Tensor bindings (per binding) | skipped (whole-op RED; see Skipped subjects) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none. N/A: no kernel constructs a `TensorAccessor` at all. |
| *Port work* — CB endpoints | deferred (re-evaluate after Device 2.0 migration) |

## Result

**RED: blocked on two gates. RED at op level; no portable subset** (the op has a single program builder, so there is no subset).

1. **Device 2.0 (route: Device 2.0 team).** `dm0.cpp` and `dm1.cpp` broadly use legacy Device 1.0 NoC and semaphore idioms: raw `noc_async_read_*`/`noc_async_write_*` stateful and trid calls, `get_noc_addr*`, `get_noc_addr_from_bank_id`, `get_semaphore` plus raw `volatile` pointers, `noc_semaphore_wait_min`/`noc_semaphore_inc`, and `noc_inline_dw_write_*`. This needs a real migration of both DM kernels. The fix is on the op-code side.
2. **TTNN factory concept (route: readiness-sheet owner).** The sheet row for this op is stale. It still describes the pre-ProgramDescriptor op (`Concept` = `legacy device-op`, factory `MoEGateMMProgramFactory`, definition `…/moe_gate_mm_program_factory.hpp`). The PD migration landed on `main` on 2026-09-29 (`dc788a6a7ff`), and the code now has `create_descriptor` on the device-op. That is a primary-column conflict plus a phantom factory row, so the sheet is "broken" for this op and the gate holds until the owner reconciles the row. **Path forward:** once the row is updated, re-read `Is able to port?` and `Porting Target`. The other primary columns all cross-check clean, so no other TTNN-side blocker is visible from here. That derived cell is not this audit's to predict, though.

Feature compatibility (all Appendix A entries N/A), Offset base pointers (GREEN) and TensorAccessor 3rd argument (N/A) are clear. After both blockers are resolved, re-audit: Device 2.0 rewrites the very kernels the porter-detail subjects describe.

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED. Spreadsheet is broken; route to the readiness-sheet owner to reconcile.**
  - Sheet row (one row): `Op` = `experimental/deepseek/moe/moe_gate_mm`, `Device operation` = `MoEGateMMDeviceOperation`, `Factory (variant)` = `MoEGateMMProgramFactory`, `Concept` = `legacy device-op`, `Op Classification` = `Legacy Op`, `Is able to port?` = `yes (with PD step)`, `Porting Target` = `ProgramSpecFactoryConcept`, `Execution Model` = `SPMD`, `Custom hash (compute_program_hash)` = `no`, `Backdoor custom hash` = `no`, `Runtime-args update (get_dynamic_runtime_args)` = `no`, `Override runtime args method? (PD only)` = `n/a`, `Pybind descriptor` = `no`, `Smuggled pointer` = `no`, `Known op issues` = *(empty)*, `TensorParameter relaxation` = `none`, `Op-owned tensors?` = `no`, `Diego validation` = `yes`, `Factory definition path` = `…/moe_gate_mm/device/moe_gate_mm_program_factory.hpp`, `Declared in` = `…/moe_gate_mm/device/moe_gate_mm_device_operation.hpp`.
  - **Conflict 1 (primary column `Concept`).** The code is `descriptor`. `MoEGateMMDeviceOperation::create_descriptor(...)` returns `tt::tt_metal::ProgramDescriptor` (`device/moe_gate_mm_device_operation.hpp:22-25`, `device/moe_gate_mm_program_factory.cpp:70`). The code has no `create()` / `override_runtime_arguments()` pair.
  - **Conflict 2 (factory-set match, phantom row).** `MoEGateMMProgramFactory` and `device/moe_gate_mm_program_factory.hpp` do not exist in the tree (a repo-wide grep finds no hits). Both existed before `dc788a6a7ff`: `git show dc788a6a7ff^:…/moe_gate_mm_program_factory.hpp` shows `struct MoEGateMMProgramFactory { create(...); override_runtime_arguments(...); }`. So the row describes the pre-migration op. The code's single program builder (`create_descriptor` on the device-op) has no row of its own.
  - The remaining primary columns cross-check **clean**: custom hash `no`, `get_dynamic_runtime_args` `no`, pybind `create_descriptor` `no`, and op-owned tensors `no`. The cross-column invariants hold given the stated concept.
  - A literal read of the cell (`yes (with PD step)`) would also RED, via the legacy-API short-circuit. That reading is wrong here, because the PD step is already done. The applicable route is spreadsheet-broken. This RED clears on the sheet side without touching the op's code.
- **Device 2.0 (every kernel used): RED. Broad Device 1.0** in two of three kernels; route to the Device 2.0 team. Both kernels are op-owned (`ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/`). There are no donor kernels. The kernels are already partly modernized: every CB sync and pointer access goes through `CircularBuffer` wrappers, and dm1 has one `Noc`/`Semaphore<>` block (`dm1.cpp:387-395`). **All** other NoC and semaphore traffic is legacy, so this is a full DM migration of both kernels, not an isolated-holdover cleanup. `get_tile_size(cb_id)` (`dm0.cpp:73-75`, `dm1.cpp:129-131`) is sanctioned and **not** flagged. `compute.cpp` is a compute kernel using `CircularBuffer` wrappers and LLK CB-index calls, with no DM idioms. It is clean.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/dm0.cpp` | 101-102 | `get_noc_addr_from_bank_id</*DRAM=*/true>(dram_bank_id, w_addr)` (raw bank addressing; D2.0 → `BankView` / `AllocatorBank`) | `CircularBuffer cb_r2c_w` only; no `Noc` |
  | `device/kernels/dm0.cpp` | 125 | `noc_async_read_one_packet_set_state<true>(dram_noc_addr, w_bytes_per_txn, vchannel)` (→ `noc.set_async_read_state<…CUSTOM_VC…>`) | none |
  | `device/kernels/dm0.cpp` | 143, 169 | `noc_async_read_set_trid(trid)` | none |
  | `device/kernels/dm0.cpp` | 145, 171 | `noc_async_read_one_packet_with_state_with_trid<false, true>(…)` (→ `noc.async_read_with_state<TXN_ID>`) | none |
  | `device/kernels/dm0.cpp` | 155, 176, 182 | `noc_async_read_barrier_with_trid(trid)` (→ `noc.async_read_barrier<TXN_ID>`) | none |
  | `device/kernels/dm1.cpp` | 143-144 | `get_semaphore(partial_semaphore)` + raw `volatile tt_l1_ptr uint32_t*` | none (a `Semaphore<>` is only built later, at 388) |
  | `device/kernels/dm1.cpp` | 147, 150 | `get_noc_addr(x, y, semaphore_addr)`: raw remote-sem NoC address | none |
  | `device/kernels/dm1.cpp` | 154, 155, 168 | `get_noc_addr(x, y, l1_addr)`: raw remote L1 NoC address | `CircularBuffer` (for the local ptr only) |
  | `device/kernels/dm1.cpp` | 190-194 | `get_semaphore(collector_semaphore)` + raw pointer + `get_noc_addr(…)` | none |
  | `device/kernels/dm1.cpp` | 223-224 | `get_noc_multicast_addr(…)` (result unused; see Misc anomalies) | none |
  | `device/kernels/dm1.cpp` | 232, 255, 298-299, 328-329 | `noc_async_write_one_packet_set_state</*posted=*/true>(…, noc=1, vchannel)` (→ `noc.set_async_write_state<POSTED\|CUSTOM_VC>`) | none |
  | `device/kernels/dm1.cpp` | 238, 261, 302-305, 332 | `noc_async_write_one_packet_with_state</*posted=*/true>(…)` | none |
  | `device/kernels/dm1.cpp` | 241-248, 264-271 | `noc_inline_dw_write_set_state<posted, set_val>` + `noc_inline_dw_write_with_state<…>` (a remote semaphore *set* via stateful inline write) | none |
  | `device/kernels/dm1.cpp` | 251, 274 | `noc_async_posted_writes_flushed()` | none |
  | `device/kernels/dm1.cpp` | 287-288, 319-320 | `noc_semaphore_wait_min(my_semaphore_ptr, 1); *my_semaphore_ptr = 0;` (→ `Semaphore<>::wait_min` / `set`) | none |
  | `device/kernels/dm1.cpp` | 308, 311, 335, 338 | `noc_async_write_set_trid(trid, noc=1)` | none |
  | `device/kernels/dm1.cpp` | 309, 336 | `noc_semaphore_inc</*posted=*/true>(collector_semaphore_noc_addr, 1, 1, vchannel)` (→ `Semaphore<>::up`) | none |
  | `device/kernels/dm1.cpp` | 310, 337 | `noc_async_write_flushed_with_trid(trid, noc=1)` | none |
  | `device/kernels/dm1.cpp` | 367, 402 | `noc_semaphore_wait_min(my_collector_semaphore_ptr, …)` | none |
  | `device/kernels/dm1.cpp` | 380-385 | `noc_async_write(local_group_masks_addr, get_noc_addr(rx, ry, …), 2048, 1, vchannel)` | `Noc noc_obj(1)` is declared at 387, right after this loop |

  *Sizing notes for the Device 2.0 team:*
  - The stateful inline-dw-write pair at `dm1.cpp:241-248` / `264-271` has no stateful counterpart on the Device 2.0 `Noc` class. `tt_metal/hw/inc/api/dataflow/noc.h:648` offers a one-shot `inline_dw_write<POSTED|CUSTOM_VC>`, and `set_async_write_state` does not accept `TXN_ID` (`noc.h:545`). Expect a small perf/behavior conversation there.
  - The posted, trid-tagged `noc_semaphore_inc` + `noc_async_write_flushed_with_trid` pairs (`dm1.cpp:308-311`, `335-338`) need a confirmed `Semaphore<>::up` + `Noc::async_writes_flushed<TXN_ID>` mapping.
  - The kernels are perf-tuned (trid pipelining, VC selection, posted writes on NoC 1), so the migration should be checked against `tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_mm_perf.py` (Wormhole).

- **Feature compatibility: GREEN (no gate fired).** Scanned the host factory, device-op, nanobind and all kernels.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `.global_circular_buffer` field, no `remote_index`/`remote_cb` idiom. CBs are built by the local `make_cb` (`moe_gate_mm_program_factory.cpp:48-66`). |
  | CBDescriptor `address_offset` (non-zero) | N/A | `make_cb` never sets `address_offset` (default 0). The borrowed CBs `c_1` (input) and `c_4` (output) set only `.buffer` (`moe_gate_mm_program_factory.cpp:64, 124-125`). |
  | GlobalSemaphore | N/A | Only plain `SemaphoreDescriptor`s (`moe_gate_mm_program_factory.cpp:250-262`). |

- **CB endpoints (GATE-free): deferred. Re-evaluate after the Device 2.0 migration.** The Device 2.0 gate is broadly RED, and dm1's raw NoC co-writes into peers' CBs (e.g. `cb_w2c_in2` / `cb_w2c_in4` / `cb_w2c_in6` written remotely by other cores' dm1, coordinated by semaphores) are exactly the hidden-writer shapes whose idioms the rewrite will change.
- **Offset base pointers: GREEN.** There are three address RTAs, all pushed as `Buffer*` objects with no host-side arithmetic: `input_buffer`, `weight_buffer`, `output_buffer` (`moe_gate_mm_program_factory.cpp:293-295`, slots 2/3/4, given identically to dm0, dm1 and compute at 324-326).
  - The only one consumed on-device is the weight base `w_addr`, in `dm0.cpp:47`. It becomes the `bank_address_offset` of `get_noc_addr_from_bank_id<true>(dram_bank_id, w_addr)` (`dm0.cpp:101-102`), and the per-transaction offset is added **kernel-side** (`w_dram_read_offset`, `dm0.cpp:139-147`). That is a clean base, not a host fold.
  - `in_addr` / `out_addr` are read but never used in any kernel. Input and output reach the kernels through the borrowed CBs `c_1` / `c_4`.
  - The op is not in the offset-base-pointer triage (`2026-07-19_offset_base_pointers.md`). That agrees with the scan: no fold present, op not in the tables, so clean.
- **TensorAccessor 3rd argument: N/A.** No kernel constructs a `TensorAccessor`; `in_args`/`w_args`/`out_args` are declared and never used (`dm0.cpp:38-40`, `dm1.cpp:83-85`), so the subject never fires. The op is not in `2026-07-06_tensor_accessor_3rd_arg_triage.md`, which agrees.

## Port-work summary  *(mirrors the brief)*

No brief is issued (RED). The porter-detail subjects were skipped, so this section is empty by design:

- **Tensor bindings:** skipped (re-audit after unblock). *(Preview from the gate scan, not a census: input/output are borrowed-memory CBs → likely **clean**. Weight is a raw bank-addressed DRAM read → likely **Case 2**. Confirm at re-audit.)*
- **TensorParameter relaxation:** sheet `none`.
- **TensorAccessor 3rd arg:** none (no sites).
- **CB endpoints:** deferred.

## Heads-ups  *(mirrors the brief)*

Skipped. Whole-op RED, no portable subset; re-audit on unblock.

## Skipped subjects

The Device 2.0 blocker is cleared on the **op-code side**: dm0 and dm1 get rewritten. Per the Red outcome scoping rule, the seven purely-informational subjects are therefore skipped. The TTNN-sheet blocker alone would clear elsewhere (sheet side), which would justify running them, but the Device 2.0 rewrite makes any detail produced now stale.

- **TTNN porting shape:** skipped (whole-op RED, no portable subset; re-audit on unblock). The sheet's `Porting Target` is recorded in the status table for reference only.
- **TensorParameter relaxations:** skipped (whole-op RED, no portable subset; re-audit on unblock). Sheet cell `none`.
- **TensorParameter analysis:** skipped (whole-op RED, no portable subset; re-audit on unblock).
- **CB endpoints:** skipped and deferred (re-evaluate after Device 2.0 migration).
- **RTA varargs:** skipped (whole-op RED, no portable subset; re-audit on unblock). *(Pointer for the re-auditor: `dm1.cpp:377-379` / `390-392` read a trailing receiver-coordinate block at `argidx + 2*receiver`, collector core only, appended at `moe_gate_mm_program_factory.cpp:305-322`. It looks like a CTA-bounded vararg block.)*
- **Out-of-directory coupling:** skipped (whole-op RED, no portable subset; re-audit on unblock). The Device 2.0 gate already confirmed there are no donor kernels: every kernel and every non-`tt_metal` include is op-local.
- **Incidental anomalies:** the systematic pass was skipped. The items below were noticed incidentally while doing gate work and are recorded because they cost nothing to keep (see Recipe notes).

## Team-only

- **Out-of-directory coupling & donor shape:** skipped (see above). Known from the gate scan: there are no borrowed kernel files; the factory instantiates only its own `dm0.cpp`, `dm1.cpp` and `compute.cpp` by `FILE_PATH`. Every `#include` outside the op directory resolves under `tt_metal/` (LLK / API headers).
- **Relaxation candidates:** none. There is no custom hash to mine.
- **TTNN factory analysis (facts with evidence):**
  - Op-owned tensors: none. `create_descriptor` returns a `ProgramDescriptor`, not a `WorkloadDescriptor`.
  - Execution model: single program, SPMD.
  - Pybind `create_descriptor`: none. `moe_gate_mm_nanobind.cpp:13-31` binds only the free function.
  - Custom hash: none (default hash over `operation_attributes_t{layer_id, column_id}` and the tensor args).
  - `get_dynamic_runtime_args`: none.
  - `override_runtime_arguments`: none. The cache-hit refresh relies on `Buffer*` RTA bindings and `CBDescriptor::buffer` patching (`moe_gate_mm_program_factory.cpp:114`).
  - Output: preallocated, passed in by the caller and returned as-is (`moe_gate_mm_device_operation.cpp:19-30`).

## Misc anomalies  *(team-only, non-gating; noticed incidentally, not a full pass)*

- **`layer_id` named CTA is unused by every kernel** (`dm0.cpp:31`, `dm1.cpp:75`, `compute.cpp:24`). It is still a compile-time arg (`moe_gate_mm_program_factory.cpp:196`) and part of the hashed attributes, so each distinct `layer_id` compiles and caches its own identical program.
- **Dead runtime args:**
  - `in_addr` and `out_addr` are read and never used in all three kernels (`dm0.cpp:46,48`, `dm1.cpp:91,93`, `compute.cpp:36,38`).
  - `w_addr` is unused in `dm1.cpp:92` and `compute.cpp:37`.
  - `compute.cpp:33-46` reads the whole 13-slot block, though it only uses `is_send_core` and `core_id`.
  - The factory gives the identical RTA list, including the collector's trailing receiver coordinates, to all three kernels (`moe_gate_mm_program_factory.cpp:324-326`).
- **Dead `TensorAccessorArgs` CTAs:** the factory appends `TensorAccessorArgs` for all three buffers (`moe_gate_mm_program_factory.cpp:131-134`), but no kernel constructs a `TensorAccessor`. The `in_args`/`w_args`/`out_args` objects are unused (`dm0.cpp:38-40`, `dm1.cpp:83-85`).
- **Dead code in `dm1.cpp`:**
  - `gather_raw_scores(...)` (`dm1.cpp:22-71`) is never called.
  - `raw_scores_noc_addr` (`dm1.cpp:223-224`, computed via `get_noc_multicast_addr`), `local_raw_scores_dst_addr` (`dm1.cpp:221`) and `local_gathered_scores_src_addr` (`dm1.cpp:218`) are computed and never used.
  - `#include "api/debug/dprint_pages.h"` (`dm1.cpp:14`) has no visible use.
- **CB `c_9` (`cb_w2c_in7`) looks unused.** Its only access found is `cb_w2c_in7.get_write_ptr()` at `dm1.cpp:220`, which feeds only the dead values above. It may be a dead CB (an 8 KB L1 allocation, `moe_gate_mm_program_factory.cpp:123`). **Not confirmed:** the CB census was deferred. The re-auditor should verify across all core roles before anyone drops it.
- **Collector-ID divergence risk:** `compute.cpp:97` hardcodes `COLLECTOR_CORE_ID = 7`, while `dm1.cpp:164` derives it from the `num_mask_receivers` CTA (which the factory computes, `moe_gate_mm_program_factory.cpp:178-189, 229`). They agree only for the 12-core Wormhole ring.
- **Hardcoded tile-size constants:**
  - `dm0.cpp:88` / `compute.cpp:86` compute `w_tiles_per_txn = noc_packet_size / 2048` instead of using `w_tile_size`.
  - `dm1.cpp:383` writes a hardcoded `2048` bytes.
  - All assume bf16 32×32 tiles. Nothing validates the dtype (next item).
- **No validation:** `validate_on_program_cache_miss` is empty (`moe_gate_mm_device_operation.cpp:17`). Dtype, shape, sharding and memory config of `input_tensor` / `w_tensor` / `output_tensor` are never checked; only null buffers and the 12-core count are `TT_FATAL`ed in the factory.

## Questions for the user

1. **Readiness sheet is stale for this op. Confirm with the sheet owner (Diego).** The `experimental/deepseek/moe/moe_gate_mm` row still describes the pre-PD op (`legacy device-op`, `MoEGateMMProgramFactory`, `moe_gate_mm_program_factory.hpp`), though `dc788a6a7ff` (#57521, 2026-09-29) moved it to `create_descriptor`. Could you ask the owner to refresh the row? *(Incidental: the same PR also migrated `deepseek_moe_gate` / `generalized_moe_gate` / `mla/matmul_wo`, and the `generalized_moe_gate` row seen in the same lookup also still reads `legacy device-op`. I did not audit it; flagging only because it is the same PR.)*
2. **Device 2.0 migration ownership.** dm0 and dm1 need a full DM Device 2.0 migration. The stateful inline-dw-write and posted trid semaphore-increment idioms have no 1:1 `Noc`-class equivalent (see the sizing notes). Should this go to the Device 2.0 team, or to the DeepSeek ops owners, given the perf-tuned NoC usage and Wormhole-only test coverage?

## Recipe notes

- **Two `Is able to port?` routing rows overlap for a stale legacy row.** When the sheet says `yes (with PD step)` but the code is already `descriptor`, both the "`yes (with PD step)` → legacy short-circuit" row and the "primary-column conflict → spreadsheet is broken" row apply. I chose spreadsheet-broken because it rests on code evidence and routes to the right owner (the legacy short-circuit would misroute to a PD migration that has already landed). The routing list could state that a code-vs-sheet `Concept` conflict takes precedence over reading the cell.
- **Factory-set match with a device-op-level `create_descriptor`.** This op has no factory struct, so "every factory in the code has a row" is ambiguous. I treated the sheet's `Factory (variant)` naming a struct that no longer exists as a phantom row. A line on how to name and match a factory-less `create_descriptor` device-op would help.
- **Incidental anomalies on a whole-op RED.** The scoping rule says to skip Incidental anomalies, but that subject is defined as "just note what you happen to see", and the gate-bearing work (Device 2.0 and offset scans read every line of dm0/dm1) surfaces those items for free. I recorded them and labelled them incidental. Consider exempting the opportunistic capture from the skip, since it costs nothing and is not re-derived from changed code in a way that loses value.
- **"Isolated holdover" vs "broad Device 1.0" for mixed kernels.** dm1 already uses `CircularBuffer` for every CB and has a small `Noc`/`Semaphore<>` block, yet all of its NoC traffic is legacy. I called it **broad**, because the legacy surface is the bulk of the kernel's data movement. The two-bucket wording ("uses `Noc`, `CircularBuffer`, etc. for the bulk of operations" vs "broadly uses legacy idioms") doesn't name the "CB-wrapped, NoC-legacy" middle case, which is common among ops that were part-migrated.
