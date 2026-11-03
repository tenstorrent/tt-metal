# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/untilize`

**Audit scope: the one remaining unported factory.** `UntilizeDeviceOperation` declares **eight** factories in its `program_factory_t` variant (`device/untilize_device_operation.hpp:40-48`). Seven are already `MetalV2` (`create_program_artifacts`). This audit covers the eighth:

- **`UntilizeDeviceOperation`**
  - `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` (`device/factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.{hpp,cpp}`) — **in scope**, still `create_descriptor` (`.hpp:14`)
  - Already `MetalV2` (out of scope, listed for disambiguation): `UntilizeSingleCoreProgramFactory`, `UntilizeMultiCoreSubCoreGridsProgramFactory`, `UntilizeMultiCoreBlockProgramFactory`, `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory` (the 2D sibling of the in-scope factory), `UntilizeMultiCoreParallelizeColumnProgramFactory`, `UntilizeMultiCoreProgramFactory`, `UntilizeMultiCoreNDShardInputProgramFactory`
- **`UntilizeCodegenDeviceOperation`** (`codegen/`): an independent device-op with its own factory and kernels, sharing no factory or kernel with the in-scope factory. **Not audited here.** Its sheet row is `data_movement/untilize/codegen`, `descriptor`, `Is able to port?` = `yes`. Audit it separately if wanted.

**Selection path:** the factory is chosen at `device/untilize_device_operation.cpp:332-339` when input and output are both ND-sharded with identical `nd_shard_spec`, **and the arch is not Quasar**. On Quasar the same inputs fall through to the (ported, non-zero-copy) `UntilizeMultiCoreNDShardInputProgramFactory`. So this factory only runs on Gen1.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `ed3eb002115 2026-11-03 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`

**Code state:** audited on branch `akertesz/op-porting-recipe`. The `untilize/` tree is byte-identical to `origin/main` (`a0e6a42ec92`, 2026-11-03).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/untilize` |
| **Overall** | **GREEN (user-waived sheet gate).** The recipe REDs this because the readiness sheet has **no row** for the in-scope factory. The user waived that gate on 2026-11-03 ("Don't worry about the missing spreadsheet row. Proceed"). All other gates are GREEN on code evidence. Brief issued. |
| **DOps / Factories** | `UntilizeDeviceOperation` → `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` (7 sibling factories already `MetalV2`) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** All three kernels are already on `DataflowBuffer` objects, with no CB-index free functions. |
| *Prereqs* — Cross-op escapes | **Ok.** Two borrowed kernel files, both with a `_metal2` fork already in tree. The own compute kernel also has a fork. |
| *Feature Support* — overall | **GREEN** (all Appendix A rows N/A) |
| *Feature Support* — Variadic-CTA | Ok (none) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **No row → "spreadsheet is broken" (missing row).** Per the recipe this is a GATE routed to the readiness-sheet owner; **waived by the user**. The code cross-check is clean (below). The row request still stands for the sheet owner. |
| *TTNN Readiness* — Concept (current) | `descriptor` (from code: `create_descriptor` returning `ProgramDescriptor`, `.hpp:14`; no sheet row to compare) |
| *TTNN Readiness* — Execution Model | `SPMD` (single `ProgramDescriptor`, no mesh shape; no sheet row to compare) |
| *TTNN Readiness* — Custom hash | No. No `compute_program_hash` / `attribute_values` / `to_hash` anywhere under `untilize/device/`. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (absent from the device-op) |
| *TTNN Readiness* — `override_runtime_arguments` | No (absent from the factory and device-op) |
| *TTNN Readiness* — Pybind `create_descriptor` | No. `untilize_nanobind.cpp` binds no `create_descriptor` and no `nb::class_` of the device-op. |
| *TTNN Readiness* — Op-owned tensors | No (`descriptor` concept; no `buffers`) |
| *TTNN Readiness* — Target concept | **Unavailable (no row).** The code is consistent with `ProgramSpecFactoryConcept` (no `override_runtime_arguments`; all 7 siblings read `ProgramSpecFactoryConcept`), but the recipe reads this off the sheet, so it is unconfirmed until the row exists. |
| *Port work* — Offset base pointer | none (the factory passes no address RTAs) |
| *Port work* — Tensor bindings (per binding) | `input`: clean (borrowed DFB) · `output`: clean (borrowed DFB) |
| *TTNN Readiness* — TensorParameter relaxation | **Unavailable (no row).** All 7 sibling rows read `none`; not derived here. |
| *Port work* — TensorAccessor 3rd arg | none — no kernel constructs a `TensorAccessor` |
| *Port work* — CB endpoints | legal: `c_0` 1:1 and `c_16` 1:1 in every config |

## Result

**GREEN → brief issued (`METAL2_PORT_BRIEF.md`), on a user waiver of the sheet gate.** After reviewing the first version of this report (which read RED, below), the user said: *"Don't worry about the missing spreadsheet row. Proceed."* The waiver covers only the missing-row gate. Two sheet-derived values are therefore taken from code evidence instead of the sheet, and are marked as such in the brief:

- the target concept: `ProgramSpecFactoryConcept`;
- `TensorParameter relaxation`: `none`.

Both match all seven sibling rows and the code (no `override_runtime_arguments`, no custom hash). The row request to Diego still stands as a sheet-hygiene item.

*Original verdict, kept for the record:* **RED → blocked on the readiness sheet (missing row), routed to the readiness-sheet owner (Diego).** The sheet has no `data_movement/untilize` row for `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory`. Under the recipe's factory-set match, a factory that exists in code with no sheet row is a missing row, which makes the sheet "broken" for this op. That gates the port until the row exists.

**This is the only blocker, and it clears without touching the op's code.** Every other gate is GREEN on independent code evidence:

- Device 2.0
- Feature compatibility
- Offset base pointers
- TensorAccessor 3rd argument
- the code-side cross-check of every primary sheet column

So the seven informational subjects were run in full, per the recipe's "RED that clears elsewhere" exception: re-audit will read the same code. When the row is added and reads `Is able to port?` = `yes` with `TensorParameter relaxation` = `none`, a re-audit should be a quick confirmation that issues the brief. The port itself looks small: a near-copy of the already-ported 2D sibling, binding three existing `_metal2` forks.

~~No port brief is issued (recipe: no brief on any RED).~~ Superseded by the user waiver above; the brief is issued.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED — spreadsheet is broken (missing row)** → readiness-sheet owner.
  - **Evidence.** A fresh CSV export (fetched this session) has seven `data_movement/untilize` rows, all `MetalV2` / `N/A (done)`, one per sibling factory. **None** is for the ND-identical factory.
  - **Where the factory name does appear.** It appears exactly once, under `Op` = `experimental/quasar/untilize`, with `Factory definition path` = `ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/factories/…`. That row describes the out-of-bounds Quasar copy, not this factory, and is **not** evidence about it (recipe: `experimental/quasar/` is out of bounds).
  - **Code-side cross-check of the primary columns (all clean, ready for the owner to fill the row):**
    - `Concept`: `descriptor`
    - `Custom hash`: no
    - `Backdoor custom hash`: no
    - `Runtime-args update (get_dynamic_runtime_args)`: no
    - `Override runtime args method?`: no
    - `Pybind descriptor`: no
    - `Smuggled pointer`: no. No buffer address reaches any RTA/CRTA; the only RTAs are tile/block counts at `.cpp:154-156`. Both tensors reach the kernels through borrowed CBs (`.cpp:81`, `.cpp:95`).
  - **Factory-set match:** 8 factories in code vs 7 rows. There is no phantom row (every row maps to an existing factory) and **one missing row** (this factory).
- **Device 2.0 (every kernel used):** **GREEN.** All three kernels construct `DataflowBuffer` wrappers and use only member FIFO ops. There are no CB-index free functions and no raw NoC / addr-gen idioms.
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp:20-21`: `DataflowBuffer dfb(cb_id_in0); dfb.push_back(n)`. Cross-family donor (eltwise/unary).
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp:28-33`: `DataflowBuffer dfb_out(...)`, `wait_front` / `pop_front`. In-family donor (data_movement/sharded).
  - `untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`: a compute kernel calling `compute_kernel_lib::untilize<…>` (`ttnn/cpp/ttnn/kernel_lib/untilize_helpers.{hpp,inl}`), which drives `in_dfb.wait_front` / `pop_front` and `out_dfb.reserve_back` / `push_back`. This is compute, not data movement, and introduces no Device 1.0 idioms.
- **Feature compatibility:** GREEN — no gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `global_circular_buffer` field on either `CBDescriptor` (`.cpp:73-97`) |
  | CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` never set on either `CBDescriptor` (defaults to 0) |
  | GlobalSemaphore | N/A | no semaphores at all |

- **CB endpoints (GATE-free):** both CBs are a legal 1P+1C in every config. See the Port-work summary.
- **Offset base pointers:** **GREEN.** The factory passes no address RTA, CRTA, or CTA. Both tensors reach the kernels as borrowed CBs (`cb.buffer = src0_buffer` / `dst_buffer`, `.cpp:81` / `.cpp:95`) with no `address_offset`. There is no `->address()` expression in the factory, so no fold is possible. Untilize is not listed in the `2026-07-19_offset_base_pointers.md` tables, which agrees with this scan.
- **TensorAccessor 3rd argument:** **N/A.** No kernel used by this factory constructs a `TensorAccessor`, so the subject never fires. This agrees with `2026-07-06_tensor_accessor_3rd_arg_triage.md:114`, which lists the main `data_movement/untilize` op as having no 3rd arg.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `input`: **clean** — borrowed-memory DFB read (legacy `cb_src0.buffer = src0_buffer`, `.cpp:81`). Port as `DataflowBufferSpec{.borrowed_from = INPUT}`.
  - `output`: **clean** — borrowed-memory DFB (legacy `cb_output.buffer = dst_buffer`, `.cpp:95`). Port as `DataflowBufferSpec{.borrowed_from = OUTPUT}`.
  - Both are declared as `TensorParameter`s (needed for `borrowed_from`) and supplied in `ProgramRunArgs::tensor_args`. No Case 1 / Case 2 work.
- **TensorParameter relaxation:** expected `none` (all siblings), but **unconfirmed** — must be read from the missing sheet row.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal (census below). No self-loop, multi-binding, dead-CB drop, or conditional DFB.

  | CB | Touchers on each node | Disposition |
  |---|---|---|
  | `c_0` (input, borrowed) | reader: locked PRODUCER (`push_back` only, the borrowed-shard readiness handshake) · compute: locked CONSUMER (`wait_front` / `pop_front` inside `compute_kernel_lib::untilize`) | 1:1 legal |
  | `c_16` (output, borrowed) | compute: locked PRODUCER (`reserve_back` / `push_back`) · writer: locked CONSUMER (`wait_front` / `pop_front`, handshake only) | 1:1 legal |

  **Config check (uneven ND sharding).** Grid cores outside `page_mapping.all_cores` get `num_tiles_to_process = num_blocks_to_process = 0` (`.cpp:141-150`). On those cores:

  - the reader does `push_back(0)` and the writer does `wait_front(0)` / `pop_front(0)`;
  - compute early-returns (`untilize_variable_num_blocks.cpp:20-22`).

  The bindings are static, so the census is still 2 touchers per CB per node. No disposition change.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** all three kernels already have a `_metal2` fork beside them. **Bind the forks; create none.**
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp`
    - Interface: `dfb::in` (PRODUCER), `args::num_tiles_per_core` (RTA).
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp`
    - Interface: `dfb::out` (CONSUMER), `args::num_units` (RTA). Tracked by issue #52228.
  - `untilize/device/kernels/compute/untilize_variable_num_blocks_metal2.cpp`
    - Interface: `dfb::src` (CONSUMER), `dfb::out` (PRODUCER), `args::per_core_block_cnt` (**RTA**, `get_arg` non-constexpr, `:18`), `args::per_core_block_tile_cnt` (**CTA**, constexpr, `:27`).
    - This matches the legacy factory exactly: per-core RTA `num_blocks_to_process` (`.cpp:155`) and CTA `num_tiles_per_block`.
    - Already bound by the ported `UntilizeMultiCoreNDShardInputProgramFactory` (`.cpp:173-183`), which is a working reference for the compute `KernelSpec`.
  - **Template:** the ported 2D sibling `untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp` is a near-exact template. It uses the same borrowed DFB pair and the same reader/writer forks. The differences are:
    - (a) compute binds `untilize_variable_num_blocks_metal2.cpp`, with `per_core_block_cnt` as a per-node RTA instead of a CTA;
    - (b) DFB `num_entries` uses ND geometry: `num_tiles_per_shard * num_shards_per_core_in_group_1`;
    - (c) per-node RTAs come from `distribution_spec.compute_page_mapping()`, with 0 for unmapped grid cores.
  - **Sunset list** (legacy consumers still binding the original after this port; *not* authorization to edit any original in place):
    - `reader_unary_sharded.cpp`: `experimental/slice_write` (tiled + RM sharded-input factories), `data_movement/sharded_partial/sharded_to_interleaved_partial`.
    - `writer_unary_sharded.cpp`: `data_movement/sharded_partial/interleaved_to_sharded_partial`. This port retires one of its last two legacy consumers.
    - `untilize_variable_num_blocks.cpp`: `experimental/deepseek_prefill/update_padded_kv_cache`. This port removes the original's last in-directory consumer.
- **RTA varargs:** none. Each kernel reads one fixed RTA (`get_arg_val<uint32_t>(0)`) and fixed CTAs, all nameable.
- **Borrowed-DFB size check.** `ValidateBorrowedMemory` (`tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp:63-88`) requires `entry_size * num_entries <= TensorSpec::compute_consumed_memory_bytes_per_bank`. The legacy CB `total_size` (`.cpp:74`, `.cpp:88`) is `num_tiles_per_shard * num_shards_per_core_in_group_1 * tile_size`. Group 1 holds the maximum shards per core, so this should equal the per-bank allocation. This is the **first borrowed-DFB-over-ND-shard use in this op**: the ND-input sibling allocates its own DFBs. Confirm the check passes on an uneven, multi-shard-per-core case, not just a 1-shard-per-core case. A rejection is loud (`TT_FATAL`), not silent.
- **Compute config carry-over** (matches both ported siblings):
  - `fp32_dest_acc_en` → `ComputeHardwareConfig{.enable_32_bit_dest}`.
  - The legacy `UnpackToDestFp32` on `c_0` (`.cpp:122`) → `unpack_modes.insert({SRC0, UnpackMode::UnpackToDest})`.
  - Set `opt_level = KernelBuildOptLevel::O3`. Both siblings do, and legacy compute was built at -O3; Metal 2.0 defaults compute to -O2.
- **Do not touch the Quasar exclusion** at `device/untilize_device_operation.cpp:335-339`. It is selection logic, not factory code, and lifting it is a behaviour change outside the port.
- **Work unit:** one `WorkUnitSpec` over `nd_shard_spec.grid` (all three kernels on every grid core, as legacy `.cpp:104/113/129` do). Cores with no shards still get the kernels and zero-count RTAs, as in legacy.

## Team-only

- **Out-of-directory coupling & donor shape — op-level roll-up: ✓ clean.**
  - Function-call escapes: the compute kernel includes `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`. This is the shared-lib class, and it takes CB ids as `uint32_t` template parameters (✓; the `_metal2` fork already passes `dfb::src` / `dfb::out`). The reader and writer include only `tt_metal` headers (`api/dataflow/*`).

  | Op kernel | Donor | Class | Shape | Status |
  |---|---|---|---|---|
  | `untilize_variable_num_blocks.cpp` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | shared kernel lib | `uint32_t cb_id` (template) | ✓ |
  | `reader_unary_sharded.cpp` (borrowed) | `tt_metal` dataflow API | tt_metal | — | ✓ |
  | `writer_unary_sharded.cpp` (borrowed) | `tt_metal` dataflow API | tt_metal | — | ✓ |

  - Borrowed kernel files: the reader is owned by eltwise/unary (cross-family) and the writer by data_movement/sharded (in-family). Fork status and the sunset list are in Heads-ups.
- **Relaxation candidates:** none (no custom hash).
- **TTNN factory analysis:**
  - concept `descriptor`
  - SPMD
  - no op-owned tensors
  - no custom hash
  - no `get_dynamic_runtime_args`
  - no `override_runtime_arguments`
  - no pybound `create_descriptor`
  - target concept unconfirmed pending the sheet row (code-consistent with `ProgramSpecFactoryConcept`)

## Misc anomalies  *(team-only, non-gating)*

- `.cpp:54-55`: `total_shards` and `num_cores` are computed only for a `log_debug`. Harmless.
- `.cpp:119`, `.cpp:131`: `compute_kernel_defines` is an always-empty vector moved into `compute_desc.defines`. Harmless leftover.
- `.cpp:138`: `orientation` only orders the `corerange_to_cores` walk. Each core gets its own RTAs, so the order has no effect. Harmless.

## Questions for the user

1. **Missing sheet row: likely a name collision with the Quasar copy.** The in-scope factory's `(Device operation, Factory (variant))` pair is identical to the `experimental/quasar/untilize` copy's, and the sheet has that pair only under the Quasar `Op`. If the sheet builder keys or dedups on class names, the main-tree row may have been collapsed into the Quasar one.
   - Worth raising with Diego along with the row request. Other ops with Quasar copies may have the same gap.
   - Not investigated further: cross-row analysis is out of audit scope.
2. **Factory count.** The request said "6 of 7 factories" are ported. The variant has **8** factories with **7** ported. The remaining one is the ND-identical factory either way, so this audit's scope is unaffected. Maybe the count included the codegen device-op, or missed the 2D-identical factory.

## Recipe notes

- **Name collision with `experimental/quasar/` copies vs the sheet lookup.** The fetch doc says to grep the CSV by op path. A broader grep on the factory class name turns up the Quasar copy's row, which looks like a ready answer (`MetalV2`, `N/A (done)`) and is easy to misattribute if the auditor greps by factory name. The recipe could say explicitly: match on `Op` **and** confirm `Factory definition path`, never on the factory name alone.
- **Missing row with a fully clean code cross-check.** The recipe treats a missing row as "spreadsheet is broken" → GATE. Here every primary column is code-verifiable and clean, and every other gate is GREEN, so the only remaining unknowns are the two *derived* cells (`Porting Target`, `TensorParameter relaxation`). The outcome is a RED whose informational detail is fully produced but whose brief is withheld. That may be intended, but the recipe could consider a "provisional brief, pending sheet row" for this case, to save the re-audit pass.
