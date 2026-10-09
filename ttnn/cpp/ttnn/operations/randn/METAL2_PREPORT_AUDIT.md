# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/randn`

- **`RandnDeviceOperation`** (`device/randn_device_operation.hpp:14`)
  - `RandnDeviceOperation::ProgramFactory` (`device/randn_program_factory.cpp`; declared `randn_device_operation.hpp:30`)
- Kernels instantiated (both owned by this op): `device/kernels/writer_standard_normal.cpp` (writer, DM) and `device/kernels/compute_standard_normal.cpp` (compute).
- The op has **no input tensors** (`tensor_args_t {}`, `randn_device_operation.hpp:25`). Its only tensor is the output.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

**Audit type:** first audit (no prior `METAL2_PREPORT_AUDIT.md`). Readiness sheet fetched live 2026-10-07 in the auditing session. Code audited at `edwinlee/PD_Metal_Ports` @ `f5c6264bf64`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/randn` |
| **Overall** | **GREEN (user waiver).** Every code-side gate is clean. The only RED, the stale readiness-sheet row from PD batch #57409, was waived by the user on 2026-10-07 (see Result). |
| **DOps / Factories** | `RandnDeviceOperation` → `ProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok — none |
| *Feature Support* — overall | GREEN (all N/A) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet: `yes (with PD step)` — but **RED: spreadsheet broken** (`Concept` and `Override runtime args method?` conflict with code; see Gate detail). **Waived by the user 2026-10-07.** |
| *TTNN Readiness* — Concept (current) | Code: `descriptor`. Sheet: `legacy device-op` (stale) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `RandnDeviceOperation::compute_program_hash`, `randn_device_operation.cpp:47-52` — hashes all attributes except `seed` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects `CustomProgramSpecFactoryConcept`): `ProgramFactory::override_runtime_arguments`, `randn_program_factory.cpp:146-172`. Sheet says `n/a` (stale) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`randn_nanobind.cpp` binds no device-op class) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `CustomProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | `output` Case 1 |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd arg |
| *Port work* — CB endpoints | legal (`c_0` 1:1) |

## Result

**GREEN by user waiver → brief issued** (`METAL2_PORT_BRIEF.md`). As audited, this was RED on the TTNN factory concept gate (spreadsheet-broken). On 2026-10-07 the user waived that gate ("Waive"): the blocker is only the sheet being out of date, not anything in the op. The sheet refresh (`Concept` = `descriptor`, `Override runtime args method?` = `yes`) is still owed to the readiness-sheet owner as housekeeping; it no longer blocks the port. Because the sheet's `Porting Target` reads `TBD (study op structure)` and the row is not owner-validated (`Diego validation` = `no`), the target concept `CustomProgramSpecFactoryConcept` rests on this audit's code reading alone. The original finding is kept below.

**Original finding (pre-waiver):** **RED → blocked on the TTNN factory concept gate: the readiness sheet is out of date for this op.** Route to the **readiness-sheet owner**. The sheet lists `ProgramFactory` as `Concept` = `legacy device-op`, with `Override runtime args method?` = `n/a`. In the code, the factory has been on the `ProgramDescriptor` API (`create_descriptor`) since PD batch **#57409** (`f5093e705ae`, 2026-09-25), and it carries a PD-style `override_runtime_arguments`. **This is the only RED.** Device 2.0, feature compatibility, offset base pointers, and the TensorAccessor 3rd argument all clear, and the relaxation cell is `none`. The block clears without touching the op's code once the sheet is reconciled (or the user waives it). That is why every informational subject was run below. RED at op level. The op has a single factory, so there is no portable subset.

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED — spreadsheet broken (primary-column conflicts).** Sheet row `randn` / `RandnDeviceOperation` / `ProgramFactory`:
  - `Concept` = `legacy device-op` (`Op Classification` = `Legacy Op`). Code: `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` (`randn_device_operation.hpp:31`, defined `randn_program_factory.cpp:42`), with no legacy `create()`. **Conflict.**
  - `Override runtime args method?\n(PD only)` = `n/a`. Code: `override_runtime_arguments(Program&, …)` (`randn_device_operation.hpp:38`, defined `randn_program_factory.cpp:146`). On a `descriptor` op, this is the target-concept signal. **Conflict.** It shares the same root cause: the sheet still treats the op as legacy, where the column doesn't apply.
  - `git log`: the PD migration landed in **#57409** (`f5093e705ae [Cleanup] Port More Ops to PD`, 2026-09-25), which touched `randn_device_operation.hpp` and `randn_program_factory.cpp`.
  - Cell quoted verbatim: `Is able to port?` = `yes (with PD step)`. The "PD step" is already done. `Porting Target` = `TBD (study op structure)`. This is a derived cell; for the answer, see TTNN porting shape below. `Diego validation` = `no`.
  - Columns that agree with the code: `Custom hash` `yes` (`randn_device_operation.cpp:47`), backdoor hash `no`, `get_dynamic_runtime_args` `no`, `Pybind descriptor` `no`, `Smuggled pointer` `no` (the address reaches the writer as a `Buffer*` binding, and the override's raw re-patch is the override mechanism, not an un-annotated pointer), `Op-owned tensors?` `no`, `Known op issues` empty. The factory set matches (one row ↔ one factory). No invariant is violated.
  - **Path forward:** the readiness-sheet owner updates the row to `Concept` = `descriptor` and `Override runtime args method?` = `yes`. Nothing in the op needs to change. This is the same #57409 lag already seen on other ops from that batch.
- **Device 2.0 (every kernel used): GREEN.** The writer uses `Noc`, `CircularBuffer`, `TensorAccessor`, and `CoreLocalMem` (`writer_standard_normal.cpp:24-66`). It reads the page size via `cb_dst.get_tile_size()` (`:29`), a wrapper method. The compute kernel uses the `CircularBuffer` wrapper for FIFO ops (`compute_standard_normal.cpp:22-25,83,109`). Its `cb_dst.get_cb_id()` / `dst_cb_id` are passed only to compute LLK calls (`pack_tile`, `pack_reconfig_data_format`, `compute_kernel_hw_startup`, `copy_init`), which take a CB index by design. No legacy idioms.
- **Feature compatibility: GREEN — no gate fired.**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | — |
  | CBDescriptor `address_offset` (non-zero) | N/A | The single CB (`randn_program_factory.cpp:68-76`) sets neither `.buffer` nor `.address_offset` |
  | GlobalSemaphore | N/A | No semaphores |

- **CB endpoints (GATE-free):** `c_0` legal. See Port-work summary.
- **Offset base pointers: GREEN.** The one address RTA is the writer's `output.buffer()` (`Buffer*` form, `randn_program_factory.cpp:135`), with no folded offset. `start_id` is a separate scalar tile index. The override re-patches a bare `output.buffer()->address()` (`:156`, `:169`), also with no offset. The op is not in `2026-07-19_offset_base_pointers.md`.
- **TensorAccessor 3rd argument: N/A.** The only accessor is `TensorAccessor(dst_args, dst_addr)` (`writer_standard_normal.cpp:25`), with two arguments. The op is not in `2026-07-06_tensor_accessor_3rd_arg_triage.md`.

## Port-work summary  *(mirrors what the brief would carry)*

- **TTNN porting shape (FYI-P):** current concept `descriptor`, no op-owned tensors, `override_runtime_arguments` present → **`CustomProgramSpecFactoryConcept`**. The porter translates `override_runtime_arguments` (`randn_program_factory.cpp:146-172`) into one returning a `ProgramRunArgs`. What it refreshes on a cache hit:
  1. **Per-core compute seed** (compute RTA 0), redrawn from `seeded_rng` (seeded from `operation_attributes.seed`) or the persistent `default_rng` (`:158-165`). This is the reason the op needs the custom concept: the seed is excluded from the hash, so it must be re-supplied every dispatch. The draw order must match `create_descriptor` (core `i` = `randn_core(i, grid)`, column-major, `:36`) so that a seeded call stays reproducible across cache miss and cache hit.
  2. **Writer output address** (writer RTA 0, `:167-170`). After the port this becomes the `output` `TensorBinding` in the returned `ProgramRunArgs`; the raw `runtime_args[0] = output_addr` patch goes away.
  - The custom hash (`randn_device_operation.cpp:47-52`) stays exactly as is.
- **Tensor bindings:** `output` — **Case 1**. `Buffer*` RTA (`randn_program_factory.cpp:135`) → writer `get_arg_val(0)` (`writer_standard_normal.cpp:18`) → `TensorAccessor(dst_args, dst_addr)` (`:25`). Accessor args are CTAs at offset 1 (`randn_program_factory.cpp:80`).
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** `c_0` (`randn_program_factory.cpp:67-76`, over `all_cores`): compute FIFO-produces (`reserve_back`/`push_back`, `compute_standard_normal.cpp:25,83`). The writer FIFO-consumes (`wait_front`/`pop_front`, `writer_standard_normal.cpp:33,52,57,68`), plus raw `get_read_ptr` peeks on its own consumer binding. **Legal 1:1.** No dead CBs.

## Heads-ups  *(mirrors what the brief would carry)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** none. Both kernels are owned by and used only by this op, and there are no function-call escapes.
- **RTA varargs:** none. Compute RTAs `seed`, `num_tiles` (`compute_standard_normal.cpp:104-105`) and writer RTAs `dst_addr`, `start_id`, `num_tiles` (`writer_standard_normal.cpp:18-20`) are fixed distinct fields → name them. Keep `seed` a **named per-node RTA**: it is the value the custom override rewrites on every cache hit.
- **Two core groups, one kernel each:** `split_work_to_cores` gives two groups that differ only in `units_per_core`, which travels as an RTA (`randn_program_factory.cpp:121-133`). One `KernelSpec` per role over `all_cores` is enough, and no per-group CTA split is needed.
- **Core-walk invariant the override depends on:** `randn_num_cores` / `randn_core` (`randn_program_factory.cpp:33-36`) mirror `split_work_to_cores`' column-major order, guarded by a `TT_FATAL` (`:52-57`). Keep both helpers and the guard. The translated override walks cores the same way.
- **Compute config:** `fp32_dest_acc_en` is **forced `true`** regardless of the kernel config (`randn_program_factory.cpp:108-109`, with a precision-range comment), and `OUTPUT_DTYPE_BFLOAT16` is a define set only for BF16 (`:92-96`). Carry both over verbatim. Legacy left `opt_level` unset on `ComputeConfigDescriptor`, and recent ports restate O3 explicitly because a Metal 2.0 `KernelSpec` defaults to O2.
- **Global RNG state:** the unseeded path draws from a file-scope `std::mt19937 default_rng` (`randn_program_factory.cpp:24`) shared by `create_descriptor` and the override. Carry it unchanged. Don't move it into the factory or make it per-call; unseeded calls rely on it advancing across dispatches.
- **Verification:** `tests/ttnn/nightly/unit_tests/operations/rand/test_randn.py`. Check its grid needs against the 8×8 card before porting.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean**. Every `#include` in both kernels resolves to `tt_metal/*` (`api/dataflow/*`, `api/compute/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`, `tt-metalium/constants.hpp`), donor class 1. There are no borrowed kernel files.
- **Relaxation candidates (FYI-U, fallible):** the custom hash excludes only `seed`. Everything else is hashed, including `MeshDevice* device` (`randn_device_operation.hpp:20`). No relaxation candidate stands out.
- **TTNN factory analysis:** `descriptor` (`randn_device_operation.hpp:31`); op-owned tensors none; no MeshWorkload; no pybound `create_descriptor`; custom hash `randn_device_operation.cpp:47`; no backdoor hash; no `get_dynamic_runtime_args`; `override_runtime_arguments` `randn_program_factory.cpp:146`. Target: `CustomProgramSpecFactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Global, non-thread-safe RNG state:** `std::uniform_int_distribution distribution` (`randn_program_factory.cpp:20`, namespace scope with external linkage) and `std::mt19937 default_rng` (`:24`) are mutable file-scope globals. They are touched on every cache miss and every cache hit, with no synchronization. Concurrent unseeded `randn` calls from multiple host threads would race on them.
- **`fp32_dest_acc_en` forced `true`, yet the full `compute_kernel_config` is hashed:** `randn_program_factory.cpp:108` overrides the config's `fp32_dest_acc_en`, while `compute_program_hash` still hashes `compute_kernel_config` (`randn_device_operation.cpp:51`). Two configs differing only in that flag build identical programs under different cache keys. This is harmless (it only wastes cache entries), but it is the dead-but-hashed shape.
- **`compute_program_hash` hashes a raw `MeshDevice*`** (`randn_device_operation.hpp:20`, via `hash_objects_with_default_seed(cached_operation_attributes, …)`, `randn_device_operation.cpp:51`). Pointer identity in a program-cache key is unusual. It is harmless in practice (the cache is per-device), but it is worth a look by the ops team.

## Questions for the user

1. **Waiver:** the only RED is the stale #57409 sheet row (`Concept` = `legacy device-op` and `Override runtime args method?` = `n/a`, versus `create_descriptor` + `override_runtime_arguments` in code). Waive it, and issue the brief? — **Resolved 2026-10-07: the user waived it ("Waive"); brief issued.**

## Recipe notes

- *TTNN porting shape:* the sheet's derived `Porting Target` reads `TBD (study op structure)` for this op, while the recipe's mapping gives `CustomProgramSpecFactoryConcept` once the stale `Override runtime args method?` cell is corrected. I recorded the recipe-derived target and noted the derived cell rather than vetting it.
