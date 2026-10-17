# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half`

- **`RotateHalfDeviceOperation`** (`device/rotate_half_device_operation.hpp:15`)
  - No separate factory struct. `create_descriptor` is a static method on the DeviceOperation itself, defined in `device/rotate_half_program_factory.cpp:16`. It builds one single-core program on core `(0,0)`.
  - Kernels bound:
    - reader: `device/kernels/dataflow/reader_rotate_half_interleaved_start_id.cpp` (own)
    - writer: `device/kernels/dataflow/writer_rotate_half_interleaved_start_id.cpp` (own)
    - compute: `ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw.cpp` (borrowed from `data_movement/bcast`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`. First audit of this op; no prior `METAL2_PREPORT_AUDIT.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

**Readiness sheet:** fetched live 2026-10-07 in the auditing session (Drive connector, CSV export).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half` |
| **Overall** | **GREEN (user waiver).** Every code-side gate is clean. The only RED, the stale readiness-sheet row from PD batch #57409, was waived by the user on 2026-10-07 (see Result). |
| **DOps / Factories** | `RotateHalfDeviceOperation` → `create_descriptor` on the device-op (no factory struct) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok. Borrows `bcast_hw.cpp`, and a `_metal2` fork already exists (see Heads-ups) |
| *Feature Support* — overall | GREEN (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Waived by user 2026-10-07.** Sheet says `yes (with PD step)`, but the row is **stale**: a primary-column conflict (`Concept`) plus a phantom factory row → **spreadsheet-broken GATE** |
| *TTNN Readiness* — Concept (current) | Code: `descriptor`. Sheet: `legacy device-op` (conflict) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No (sheet `no`; matches) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet `no`; matches) |
| *TTNN Readiness* — `override_runtime_arguments` | No (sheet `n/a`; the code has none) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (sheet `no`; `rotate_half_nanobind.cpp:18` binds only the function) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | `src` Case 1 · `dst` Case 1 |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd arg |
| *Port work* — CB endpoints | all legal (four 1P+1C FIFOs) |

## Result

**GREEN by user waiver → brief issued** (`METAL2_PORT_BRIEF.md`). As audited, this was RED on the TTNN factory concept gate (spreadsheet-broken). On 2026-10-07 the user waived that gate ("waive"): the blocker is only the sheet being out of date, not anything in the op. The sheet refresh is still owed to the readiness-sheet owner as housekeeping; it no longer blocks the port. The original finding is kept below.

**Original finding (pre-waiver):** **RED → blocked on the TTNN factory concept gate (spreadsheet-broken)**, routed to the **readiness-sheet owner**. The sheet still describes the op as it was before PD-migration batch #57409 (`f5093e705ae`, 2026-09-25). That batch moved `rotate_half` onto the `ProgramDescriptor` API and deleted `RotateHalfProgramFactory`. The sheet still shows `Concept` = `legacy device-op` with a `RotateHalfProgramFactory` row. **This is the only RED.** Every other gate clears: Device 2.0, Feature compatibility, Offset base pointers and TensorAccessor 3rd argument. Against the code, the op is otherwise a clean `ProgramSpecFactoryConcept` port. RED at op level; there is one program and no subset to split off. The block clears once the sheet row is reconciled, or by user waiver.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** RED — **spreadsheet-broken**, routed to the readiness-sheet owner (Diego) to reconcile. The sheet row for `experimental/transformer/rotate_half` is:

  | Column | Sheet value | Code evidence |
  |---|---|---|
  | `Device operation` | `RotateHalfDeviceOperation` | matches |
  | `Factory (variant)` | `RotateHalfProgramFactory` | **phantom row**: no such type exists. `device/rotate_half_program_factory.hpp` (the sheet's `Factory definition path`) was deleted in `f5093e705ae`. Before that commit, the device-op declared `using program_factory_t = std::variant<RotateHalfProgramFactory>;` (`f5093e705ae^:…/rotate_half_device_operation.hpp:19`). |
  | `Concept` | `legacy device-op` | **conflict**: `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` on the device-op (`device/rotate_half_device_operation.hpp:24`, defined at `device/rotate_half_program_factory.cpp:16`) → `descriptor` |
  | `Op Classification` | `Legacy Op` | derived; follows the stale `Concept` |
  | `Is able to port?` | `yes (with PD step)` | derived; the "PD step" has already landed |
  | `Custom hash` / `Backdoor custom hash` | `no` / `no` | matches (no `compute_program_hash`, `attribute_values` or `to_hash`) |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | matches |
  | `Override runtime args method?` | `n/a` | none in code |
  | `Pybind descriptor` | `no` | matches |
  | `Smuggled pointer` | `no` | matches. Addresses travel as `Buffer*` RTAs, which the framework patches (`device/rotate_half_program_factory.cpp:109,119`) |
  | `TensorParameter relaxation` | `none` | clears |
  | `Known op issues` | *(empty)* | — |
  | `Op-owned tensors?` | `no` | matches |

  Two of the recipe's four spreadsheet-broken triggers fire: a primary-column conflict (`Concept`) and a phantom factory row. This is the same staleness the sheet shows for other #57409 ops. **The sheet's verdict (`yes`) and this audit's code reading agree that the op can port.** The RED is purely that the sheet row can't be trusted as written. Path forward: the sheet owner refreshes the row (`Concept` → `descriptor`; factory → the device-op's own `create_descriptor`; `Factory definition path` → `device/rotate_half_program_factory.cpp`), then re-audit. Alternatively, the user waives this gate.
- **Device 2.0 (every kernel used):** GREEN.
  - The reader and writer are fully Device 2.0 (migrated in `ae414c18e46`, #47571). They use `Noc`, `CircularBuffer` objects, `TensorAccessor`, `CoreLocalMem` and `noc.async_read`/`async_write` with `{.page_id = …}`. There are no raw `noc_async_*` calls, no addr-gen and no CB-index holdovers.
  - `get_tile_size(cb_id)` at `reader_…:27` and `writer_…:25` is **sanctioned**, so it is not flagged.
  - The borrowed compute kernel `bcast_hw.cpp` has no data-movement idioms. It drives CBs `c_0`, `c_1` and `c_16` through `kernel_lib` `eltwise_chain`.
- **Feature compatibility:** GREEN — no gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `global_circular_buffer` field, type or remote-CB idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | the four `CBDescriptor`s (`device/rotate_half_program_factory.cpp:42,55,67,79`) leave it unset |
  | GlobalSemaphore | N/A | no semaphores at all |

- **CB endpoints (GATE-free):** all legal. See the census under Port-work.
- **Offset base pointers:** GREEN. Both address RTAs are bare `Buffer*` values: `src_buffer` at `:109` and `dst_buffer` at `:119`, from `input.buffer()` and `output.buffer()` (`:92-93`), with no folded offset. Page offsets travel separately as a `start_id` scalar (always `0u`). The op is not in `2026-07-19_offset_base_pointers.md`, and the scan confirms it is clean.
- **TensorAccessor 3rd argument:** N/A — no accessor in the op passes a 3rd argument. The reader has `TensorAccessor(src_args, src_addr)` at `reader_…:30` and the writer has `TensorAccessor(dst_args, dst_addr)` at `writer_…:28`. The redundant page-size arg was already removed in `69eb869a7bf` (#41902). The op is not in the 3rd-arg triage table.

## Port-work summary  *(mirrors the brief)*

This subject ran despite the RED. Per the scoping rule's exception, the blocker clears **without touching the op's code** (a sheet-row reconcile or a waiver), so today's detail survives to the port.

- **Tensor bindings** (per binding):
  - `src` (input) — **Case 1**. The `Buffer*` RTA at `device/rotate_half_program_factory.cpp:109` (reader RTA 0) feeds `TensorAccessor(src_args, src_addr)` at `reader_…:30`, and every read goes through the accessor.
  - `dst` (output) — **Case 1**. The `Buffer*` RTA at `:119` (writer RTA 0) feeds `TensorAccessor(dst_args, dst_addr)` at `writer_…:28`.
  - Both are `Buffer*`-binding form: correct on cache hit today, so this is routine work, not a hazard.
- **TensorParameter relaxation:** none (sheet cell `none`).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal on the single core `(0,0)`. Nothing is dead, there are no self-loops and no multi-binding.

  | CB | Producer | Consumer | Disposition |
  |---|---|---|---|
  | `c_0` (`src_mul`, `:40`) | reader FIFO (`cb_in_mul.reserve_back/push_back`, `reader_…:57-61`) | compute (`bcast_hw.cpp` lhs, PerTile wait/pop) | 1P+1C |
  | `c_1` (`src_scalar`, `:53`) | reader FIFO, one tile (`reader_…:37-42`) | compute (rhs; waits upfront and never pops under `BCAST_SCALAR`) | 1P+1C |
  | `c_2` (`src_no_mul` = `output_no_mul`, `:66,88`) | reader FIFO (`reader_…:49-54`) | writer FIFO (`writer_…:38-43`) | 1P+1C. Compute never touches it: the un-multiplied half goes DM→DM |
  | `c_16` (`output_mul`, `:77`) | compute (pack, PerTile reserve/push) | writer FIFO (`writer_…:46-51`) | 1P+1C |

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No raw co-fill (the reader's `get_write_ptr` on `c_1` is its own FIFO producer peek), and no dual-instance shape.
- **Cross-op / shared kernels:** `data_movement/bcast/device/kernels/compute/bcast_hw.cpp` is borrowed (cross-family).
  - **A `_metal2` fork already exists beside it**: `bcast_hw_metal2.cpp`, created by the bcast port `6e46b12f678` (#50901). This is **rung 1: bind the fork; don't re-fork and don't edit it.**
  - Fork interface:
    - DFBs: `dfb::in0` (CONSUMER), `dfb::in1` (CONSUMER), `dfb::out` (PRODUCER)
    - named RTAs: `B`, `Ht`, `Wt`
    - defines: it gates on `BCAST_OP`, `BCAST_LLKOP`, `BCAST_DIM` and `BCAST_SCALAR`
  - **Define mismatch:** rotate_half today passes `CHAIN_BCAST_OP`, `CHAIN_BCAST_DIM` and `BCAST_SCALAR` (`device/rotate_half_program_factory.cpp:125-128`). Those match the *legacy* `eltwise_chain` body, not the fork. The porter must supply the fork's vocabulary for Mul + HW-scalar instead. `bcast_op_utils::get_defines(BcastOpDim::HW, BcastOpMath::MUL)` (`data_movement/bcast/bcast_types.cpp:9`) yields exactly `BCAST_OP=mul_tiles_bcast`, `BCAST_LLKOP=EltwiseBinaryType::ELWMUL` and `BCAST_DIM=BroadcastType::SCALAR`, plus the `CHAIN_*` pair that the fork ignores. Add `BCAST_SCALAR=1` on top. This mirrors the bcast HW factory (`bcast_multi_core_hw_program_factory.cpp:159-163`).
  - **Drift:** the fork lags its original. `8cf12808550` (#55584) moved `bcast_hw.cpp` to `kernel_lib` `eltwise_chain` *after* the fork was cut, and did not touch the fork. So binding the fork moves rotate_half's compute from the `eltwise_chain` body to the fork's hand-rolled `bcast_init` / `mul_tiles_bcast` loop. It is the same math (scalar-broadcast multiply; the rhs is waited once and never popped in both), and it is what rung 1 prescribes. It is still a code-path change, so check numerics with `test_rotate_half` (both test shapes are single-core and run on an 8×8 grid).
  - **opt_level:** the bcast port pins the fork at `KernelBuildOptLevel::O3` to keep the legacy compute default. rotate_half's legacy `ComputeConfigDescriptor{}` has the same O3 default, so do the same.
  - **Sunset list (not authorization to convert in place):** rotate_half is the **only remaining binder** of the legacy `bcast_hw.cpp`. The bcast factory's own comment says so (`bcast_multi_core_hw_program_factory.cpp:220`), and the census confirms it. Once this port lands, the legacy copy has no consumers. Record that in the port report for sunset; deleting it is not the porter's job.
- **RTA varargs:** none. Reader and writer each read four fixed-index RTAs (`src_addr`/`dst_addr`, `num_rows`, `half_row_size`/`half_row_width`, `start_id`). Compute reads `B`, `Ht`, `Wt`. All of these are nameable. CTAs are fixed-index (CB ids, `scalar_value`, then `TensorAccessorArgs<4>` / `<2>`).
- **Anything else:**
  - The reader writes the scalar through a raw `uint16_t*` into `cb_in_scalar.get_write_ptr()` (`reader_…:38-41`). This is a producer-side peek, so port it as-is on the DFB.
  - The scalar `-1.0` reaches the reader as a CTA (`static_cast<uint32_t>(bfloat16_scalar)`, `:90,96`) and is read as `constexpr uint16_t` (`reader_…:23`). Keep it a named CTA.
  - CB `c_2` has two names on the host (`src_no_mul_cb_index` and `output_no_mul_cb_index`, `:66,88`) and is bound by reader and writer. Declare it as one `DataflowBufferSpec` with reader PRODUCER and writer CONSUMER.

## Team-only

- **Out-of-directory coupling & donor shape:** ✓ clean.

  | Op kernel | Donor | Class | Shape |
  |---|---|---|---|
  | reader / writer | `tt_metal` dataflow APIs (`dataflow_api.h`, `noc.h`, `circular_buffer.h`, `core_local_mem.h`, `noc_traits.h`) | `tt_metal/*` | no concern |
  | `bcast_hw.cpp` (borrowed) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp` | kernel_lib | uses `uint32_t cb_id` / `CBIndex` → ✓ OK |

  - Borrowed kernel file: `data_movement/bcast/device/kernels/compute/bcast_hw.cpp`, owned by `data_movement/bcast`. Its only binder is rotate_half. The `_metal2` fork exists and is bound by `bcast_multi_core_hw_program_factory.cpp:222`.
  - **Drift finding for the bcast owners and the recipe maintainer:** the fork no longer matches the original's implementation (see Heads-ups). Rotate_half is the last legacy consumer, so the original retires after this port. After that, the fork's raw-LLK body is the only copy, and it does not carry #55584's `eltwise_chain` migration. Whoever owns the `kernel_lib` migration may want to re-apply it to the fork after sunset.
- **Relaxation candidates:** none (no custom hash).
- **TTNN factory analysis:**
  - op-owned tensors: none (no `WorkloadDescriptor`)
  - no MeshWorkload need
  - no pybind `create_descriptor` (`rotate_half_nanobind.cpp:18` binds `ttnn::experimental::rotate_half` only)
  - no other risky pybind
  - no custom hash: the default hash covers `RotateHalfParams{output_mem_config}` plus the input tensor. The header comment at `device/rotate_half_device_operation.hpp:21-23` relies on this.
  - no `get_dynamic_runtime_args`
  - no `override_runtime_arguments`
  - **target concept: `ProgramSpecFactoryConcept`**
  - Program-cache coverage exists: `test_rotate_half_program_cache` (`tests/tt_eager/python_api_testing/unit_testing/misc/test_rotate_half.py:33`).

## Misc anomalies

- **Scalar CB format vs. page size mismatch** (`device/rotate_half_program_factory.cpp:55-63`). CB `c_1` is declared with `.data_format = cb_data_format` (the *input* dtype) but `.page_size` / `.total_size` from `Float16_b` (`scalar_single_tile_size`, `:32-33`). The reader always writes a bfloat16 bit pattern (`reader_…:38-41`).
  - For bf16 input this is consistent.
  - For `float32` input, compute would unpack a 4096-byte Float32 tile from a 2048-byte CB.
  - For `bfloat8_b` input, the bf16 bits would be read as BFP8.
  - `validate_on_program_cache_miss` (`device/rotate_half_device_operation.cpp:16-29`) does not restrict dtype, and the unit tests cover only bf16. This is likely a latent bug for non-bf16 inputs. Route to the ops team. The port preserves it as-is.
- **Constant `start_id` RTA:** reader and writer RTA 3 is always `0u` (`:109,119`) because the op is single-core. This is harmless dead plumbing.
- **Misleading comments:** `reader_…:36` says "Fill tile with zeros" but writes the single `-1.0` scalar into element 0. `writer_…:35` says "read a ublock of tiles from src to CB" in the writer.

## Recipe notes

- This is the seventh #57409-batch op whose only RED is the stale-sheet "spreadsheet-broken" trigger. While the batch remains unreflected in the sheet, the recipe might name this case explicitly: phantom factory row plus `Concept` conflict, where the derived `Is able to port?` already says `yes (with PD step)`. The finding is mechanical and recurring, and the RED (with waiver) is arguably heavier than needed when the sheet's *verdict* agrees with the code.
- The *Caution: Porting a shared kernel* rung-1 fit check lists defines as something to match. It doesn't anticipate the case found here: the fork's defines differ because the **original** drifted after the fork was cut (not because the porter's need differs). The drift-discipline paragraph covers the obligation but not how an auditor or porter should treat a fork that is now behind its original. A sentence saying "a stale fork is still rung 1; translate defines to the fork's vocabulary and flag the drift" would settle it.
