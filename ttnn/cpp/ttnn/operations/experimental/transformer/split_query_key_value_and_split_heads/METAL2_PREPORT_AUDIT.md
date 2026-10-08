# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads`

- **`SplitFusedQKVAndSplitHeadsDeviceOperation`** (`device/split_query_key_value_and_split_heads_device_operation.hpp:38`)
  - `SplitFusedQKVAndSplitHeadsProgramFactory` — interleaved (`device/split_query_key_value_and_split_heads_program_factory.cpp`, declared `device_operation.hpp:22`)
  - `SplitFusedQKVAndSplitHeadsShardedProgramFactory` — block-sharded (`device/split_query_key_value_and_split_heads_sharded_program_factory.cpp`, declared `device_operation.hpp:31`)
- Factory selection: `select_program_factory` → sharded iff `input_tensor.is_sharded()` (`device_operation.cpp:12-18`).
- Kernels instantiated:
  - Interleaved: `kernels/dataflow/reader_tm_tile_layout_create_qkv_heads.cpp`, `kernels/dataflow/writer_tm_tile_layout_create_qkv_heads.cpp`, and the shared-pool compute `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`.
  - Sharded: `kernels/dataflow/reader_tm_tile_layout_create_qkv_heads_sharded.cpp`, `kernels/dataflow/writer_tm_tile_layout_create_qkv_heads_sharded.cpp`, `kernels/compute/transpose_wh_sharded.cpp`.
- **Not instantiated by this op:** `kernels/compute/transpose_wh_sharded_metal2.cpp` — the Metal 2.0 fork of `transpose_wh_sharded.cpp`, created by the `create_qkv_heads_from_separate_tensors` port (`f5c6264bf64`) and bound by that op only.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

**Audit type:** first audit (no prior `METAL2_PREPORT_AUDIT.md`). Readiness sheet fetched live 2026-10-07 in the auditing session. Code audited at `edwinlee/PD_Metal_Ports` @ `f5c6264bf64`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads` |
| **Overall** | **GREEN (user waiver).** Every code-side gate is clean. The only RED, the stale readiness-sheet row from PD batch #57409, was waived by the user on 2026-10-07 (see Result). |
| **DOps / Factories** | `SplitFusedQKVAndSplitHeadsDeviceOperation` → `SplitFusedQKVAndSplitHeadsProgramFactory`, `SplitFusedQKVAndSplitHeadsShardedProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok — no function-call escapes; two borrowed/shared kernel files, both with an existing `_metal2` fork |
| *Feature Support* — overall | GREEN (all N/A) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet: `yes (with PD step)` — but **RED: spreadsheet broken** (`Concept` conflicts with code; see Gate detail). **Waived by the user 2026-10-07.** |
| *TTNN Readiness* — Concept (current) | Code: `descriptor` (both factories). Sheet: `legacy device-op` (stale) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No (sheet `no`; no `compute_program_hash` in the device-op — confirmed) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet `no`; confirmed) |
| *TTNN Readiness* — `override_runtime_arguments` | No (sheet `n/a`; no method on either factory — confirmed) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (sheet `no`; `split_query_key_value_and_split_heads_nanobind.cpp` binds no device-op class — confirmed) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | Interleaved: `input`, `q`, `k`, `v` Case 1. Sharded: `input`, `q`, `k`, `v` clean (borrowed-memory DFBs) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd arg |
| *Port work* — CB endpoints | Interleaved: all legal 1:1. Sharded: `c_0` 1P+1C assignment (two role-free readers); `c_16`, `c_17`, `c_18` self-loop; `c_24` legal |

## Result

**GREEN by user waiver → brief issued** (`METAL2_PORT_BRIEF.md`). As audited, this was RED on the TTNN factory concept gate (spreadsheet-broken). On 2026-10-07 the user waived that gate ("Waive"): the blocker is only the sheet being out of date, not anything in the op. The sheet refresh (both rows to `Concept` = `descriptor`, plus the stale `Factory definition path` cells) is still owed to the readiness-sheet owner as housekeeping; it no longer blocks the port. The original finding is kept below.

**Original finding (pre-waiver):** **RED → blocked on the TTNN factory concept gate: the readiness sheet is out of date for this op.** Route to the **readiness-sheet owner**. The sheet still lists both factories as `Concept` = `legacy device-op`. In the code, both factories have been on the `ProgramDescriptor` API (`create_descriptor`) since PD batch **#57409** (`f5093e705ae`, 2026-09-25). **This is the only RED.** Device 2.0, feature compatibility, offset base pointers, and the TensorAccessor 3rd argument all clear, and the relaxation cell is `none`. The block clears without touching the op's code once the sheet is reconciled (or the user waives it). That is why every informational subject was run below. RED at op level. Both factories are affected by the same stale row, so no subset is cleaner than the whole op.

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED — spreadsheet broken (primary-column conflict).** Both sheet rows (`SplitFusedQKVAndSplitHeadsProgramFactory`, `SplitFusedQKVAndSplitHeadsShardedProgramFactory`) read `Concept` = `legacy device-op` and `Op Classification` = `Legacy Op`. The code says `descriptor`:
  - `device/split_query_key_value_and_split_heads_device_operation.hpp:23` and `:32` — each factory declares `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`, with no `create()` and no `override_runtime_arguments()`.
  - `git log` shows the PD migration landed in **#57409** (`f5093e705ae [Cleanup] Port More Ops to PD`, 2026-09-25). That commit also **deleted** the per-factory headers `device/split_query_key_value_and_split_heads_program_factory.hpp` and `device/split_query_key_value_and_split_heads_sharded_program_factory.hpp`, which the sheet's `Factory definition path` column still names. The factories themselves still exist under the same names, so the **factory set matches one-to-one** (no phantom or missing factory row). Only the path cell is stale.
  - Cell quoted verbatim: `Is able to port?` = `yes (with PD step)`. The "PD step" it refers to is already done.
  - Other cross-checked columns agree with the code: `Custom hash` `no`, backdoor hash `no`, `get_dynamic_runtime_args` `no`, `Pybind descriptor` `no`, `Smuggled pointer` `no`, `Op-owned tensors?` `no`, `Known op issues` empty. No cross-column invariant is violated.
  - **Path forward:** the readiness-sheet owner updates the two rows to `descriptor` and the factory-definition path. Nothing in the op needs to change. This is the same #57409 lag already seen on other ops from that batch.
- **Device 2.0 (every kernel used): GREEN.** All six kernels are on Device 2.0 idioms: `Noc`, `CircularBuffer` wrappers, `TensorAccessor`, `UnicastEndpoint`, and `CoreLocalMem`. No `InterleavedAddrGen*`, raw `noc_async_*`, or raw semaphore addresses. Free-function sites are sanctioned only: `get_tile_size(cb_id)` at reader `:27`, writer `:38`, sharded reader `:30`, and sharded writer `:30`. The sharded kernels read their own NoC coords via `my_x[noc_id]` / `my_y[noc_id]` (sharded reader `:37-38`, sharded writer `:36-37`) to build a `UnicastEndpoint` self-loopback read. This is not a Device 2.0 holdover: `Noc` exposes no own-coords accessor, and `tt_metal/hw/inc/api/dataflow/noc.h:167` itself reads `my_x[noc_id_]`. The kernels were moved to Device 2.0 in #45843 / #58858.
- **Feature compatibility: GREEN — no gate fired.**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `global_circular_buffer` field, or `remote_index` |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set; the sharded factory's borrowed CBs use `.buffer =` only |
  | GlobalSemaphore | N/A | No semaphores at all |

- **CB endpoints (GATE-free):** see Port-work summary. Nothing blocks.
- **Offset base pointers: GREEN.** Address RTAs exist only in the interleaved factory: reader `in0_buffer` (`program_factory.cpp:187`) and writer `q_buffer` / `k_buffer` / `v_buffer` (`:193-195`). All are pushed as bare `Buffer*` (BufferBinding form), with no host-side `+ offset`. Tile offsets travel as separate scalar RTAs (`:188`, `:196-197`). The sharded factory has no RTAs. Its kernel-side `+ tensor_stride_size_bytes` offsets are added to a CB read pointer on-device, not folded into a host address. Neither op appears in `2026-07-19_offset_base_pointers.md`. My own scan is clean.
- **TensorAccessor 3rd argument: N/A.** No accessor in the op passes a 3rd argument: reader `:28` `TensorAccessor(in0_args, in0_tensor_addr)`, writer `:39-41` with two arguments each. Neither op appears in `2026-07-06_tensor_accessor_3rd_arg_triage.md`.

## Port-work summary  *(mirrors what the brief would carry)*

- **TTNN porting shape (FYI-P):** current concept `descriptor`, no op-owned tensors, `Override runtime args method?` = no → **`ProgramSpecFactoryConcept`** for both factories. This matches the sheet's `Porting Target` column. No custom hash.
- **Tensor bindings** (per binding, per factory):
  - *Interleaved factory:*
    - `input`: **Case 1**. `Buffer*` RTA (`program_factory.cpp:187`) → reader `get_arg_val(0)` (`reader_tm_tile_layout_create_qkv_heads.cpp:16`) → `TensorAccessor(in0_args, …)` (`:28`). The accessor args are CTAs at offset 2 (`program_factory.cpp:99`).
    - `q`, `k`, `v`: **Case 1**. `Buffer*` RTAs (`program_factory.cpp:193-195`) → writer `get_arg_val(0..2)` (`writer_tm_tile_layout_create_qkv_heads.cpp:17-19`) → `TensorAccessor(q_args/k_args/v_args, …)` (`:39-41`). The accessor args are chained CTAs starting at 7 (`program_factory.cpp:111-113`).
  - *Sharded factory:* `input` (`c_0`, `sharded_program_factory.cpp:133`), `q` (`c_16`, `:154`), `k` (`c_17`, `:165`), `v` (`c_18`, `:176`): all **clean**, borrowed-memory DFBs via `.buffer =` → `DataflowBufferSpec::borrowed_from`.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (per `(CB, factory)`, per node; every CB spans the factory's full core range):
  - *Interleaved* (`program_factory.cpp:150-178`):
    - `c_0` (K pre-transpose): reader FIFO-produces (`reserve_back`/`push_back` via `cb_cur`, reader `:46-61`, when `out_tensor == 1`). Compute `transpose_wh.cpp` FIFO-consumes. **Legal 1:1.**
    - `c_1` (Q and V tiles): reader FIFO-produces. Writer FIFO-consumes (`wait_front` `:57/:65/:113/:121`, `pop_front` `:140`). **Legal 1:1.**
    - `c_16` (K transposed): compute FIFO-produces. Writer FIFO-consumes (`:86/:93`, `pop_front` `:139`). **Legal 1:1.**
  - *Sharded* (`sharded_program_factory.cpp:125-178`):
    - `c_0` (input shard, borrowed): no FIFO ops anywhere. Raw `get_read_ptr` by the sharded **reader** (`:43`) and the sharded **writer** (`:41`) — two role-free touchers → **1P+1C assignment** (bind one PRODUCER, the other CONSUMER; cosmetic on Gen1). Not multi-binding.
    - `c_24` (K intermediate): sharded writer FIFO-produces (`reserve_back` `:49`, `push_back` `:63`). Compute `transpose_wh_sharded.cpp` FIFO-consumes. **Legal 1:1.**
    - `c_16` (Q output shard, borrowed): only the sharded reader touches it (`reserve_back` `:42`, `get_write_ptr` `:44`, `push_back` `:62`). Nothing drains it → **self-loop** (DM).
    - `c_18` (V output shard, borrowed): only the sharded reader touches it (`:65`, `:68`, `:86`) → **self-loop** (DM).
    - `c_17` (K output shard, borrowed): only compute touches it (`reserve_back`/`pack_tile`/`push_back`) → **self-loop** (compute). Precedent: the `create_qkv_heads_from_separate_tensors` port binds this same fork's `out` as PRODUCER + CONSUMER on its K shard.
  - No dead CBs. Every allocated index is touched in its factory. The kernels' unused `constexpr` index names (e.g. the sharded reader's `cb_im0` / `cb_out1`, and the sharded writer's `cb_out0..2`) are declarations only. They do not make a CB dead.

## Heads-ups  *(mirrors what the brief would carry)*

- **CB endpoints (multi-binding shapes to watch):** none. No CB needs the multi-binding flag. I hunted for a hidden second writer on every sharded CB: no `get_write_ptr` co-fill by a non-producer, and no semaphores.
- **Cross-op / shared kernels:**
  - Interleaved compute `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` (shared pool `ttnn/kernel/`). A `_metal2` fork **already exists** at `ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp` (bound today by `nlp_create_qkv_heads`) — bind it, don't re-fork. Its interface: `dfb::in` / `dfb::out` plus a **named CTA** `NHtWt`; legacy passes `num_tiles_per_tensor` as positional CTA 0 (`program_factory.cpp:138`). Other legacy binders (a **sunset list, not authorization to convert the kernel in place**): `nlp_create_qkv_heads_boltz`, `nlp_create_qkv_heads_vit`.
  - Sharded compute `kernels/compute/transpose_wh_sharded.cpp` (owned by this op, borrowed by others). A `_metal2` fork **already exists beside it**: `kernels/compute/transpose_wh_sharded_metal2.cpp` (bound by `create_qkv_heads_from_separate_tensors`) — bind it, don't re-fork. Its interface: `dfb::in` / `dfb::out` plus a **named CTA** `num_tiles`; legacy passes positional CTA 0 (`sharded_program_factory.cpp:119`). Other legacy binder (sunset list): `experimental/transformer/create_qkv_heads` (`create_qkv_heads_program_factory.cpp:167-169`). Do **not** confuse it with `data_movement/transpose/.../transpose_wh_sharded_metal2.cpp`, which is a different file.
  - The four dataflow kernels are instantiated by this op only (`create_qkv_heads` uses its own `reader_create_qkv_heads_sharded.cpp`). They can be converted in place or forked per the shared-kernel caution — no co-borrower.
- **RTA varargs:** none. Interleaved reader RTAs (`in0_tensor_addr`, `in0_tensor_tile_id`) and writer RTAs (`q/k/v_tensor_addr`, `out_tensor_tile_id`, `out_tensor_tile_id_with_transpose`) are fixed distinct fields → name them. The sharded factory has no RTAs.
- **Interleaved writer CB counters (don't "fix" during the port):** the writer waits with **cumulative** counts and pops once at the end (`writer_tm_tile_layout_create_qkv_heads.cpp:48-49`, `:139-140`). The counters start at `block_size` (1), so the final `pop_front` pops N+1 pages after N were pushed (see Misc anomalies). Carry it over byte-for-byte as legacy behavior. Gen1 DFB lowers to the same CB.
- **Sharded writer order quirk:** `cb_im0_obj.get_write_ptr()` is read *before* `reserve_back` (`writer_tm_tile_layout_create_qkv_heads_sharded.cpp:48-49`). This is legacy behavior, so preserve it.
- **Verification on this workspace:** the interleaved unit test (`tests/tt_eager/python_api_testing/unit_testing/misc/test_split_query_key_value_and_split_heads.py:23-33`, batch 8, `seq_len = 32*min(12, grid.x)`) skips only if `grid.x < 8` or `grid.y < batch`, so it runs on the 8×8 Wormhole. Find a sharded-path test before porting; the BERT-large sharded configs historically need a ≥12-column grid.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean** for function-call escapes. Every kernel `#include` resolves to `tt_metal/*` (`api/dataflow/*`, `api/compute/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`), donor class 1, no concern. Borrowed kernel files are listed under Heads-ups. Summary:

  | Op kernel | Donor | Class | Roll-up |
  |---|---|---|---|
  | all six kernels | `tt_metal/hw/inc/api/*` | 1 (LLK/HAL) | ✓ |
  | (file-path) interleaved compute | `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` | 3 (shared pool) | ✓ — fork exists |

- **Relaxation candidates:** none (no custom hash).
- **TTNN factory analysis:** `descriptor` (`device_operation.hpp:23`, `:32`); op-owned tensors none; no MeshWorkload; no pybound `create_descriptor` (`split_query_key_value_and_split_heads_nanobind.cpp`); no custom or backdoor hash; no `get_dynamic_runtime_args`; no `override_runtime_arguments`. Target: `ProgramSpecFactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Hard-coded head dim 64 in the interleaved factory:** `program_factory.cpp:48` sets `out_w = 64` and derives `out_c` (heads per tensor) from it (`:50`). `num_heads` (in `operation_attributes`, hashed) is never read by this factory. Meanwhile `compute_output_specs` derives the head dim as `width / 3 / num_heads` (`device_operation.cpp:84`). Any `num_heads` that doesn't give a 64-wide head would make the program and output spec disagree. Validation (`batch ∈ [7,9]`, interleaved) doesn't guard it. Related to #57158 ("interleaved writer head count").
- **Interleaved writer pops one page too many:** in `writer_tm_tile_layout_create_qkv_heads.cpp`, `out_num_tiles_read_out0/1` start at `block_size` (`:48-49`) and are incremented after each `wait_front`. The final `pop_front(out_num_tiles_read_out0)` / `pop_front(out_num_tiles_read_out1)` (`:139-140`) therefore pops 33 / 65 pages after 32 / 64 were pushed (with `num_tiles_per_tensor = 32`). This is benign at kernel exit on Gen1, but it is wrong as written.
- **Grid-fit checks are debug-only:** `TT_ASSERT(num_cores_x <= grid.x)` / `TT_ASSERT(num_cores_y <= grid.y)` (`program_factory.cpp:61-62`) compile out in release. An oversized input silently builds a core range past the grid. The sharded path validates its bbox with `TT_FATAL` (`device_operation.cpp:38-45`); the interleaved path has no equivalent.
- **Hard-coded `num_heads_per_tensor = 2` in the sharded factory:** `sharded_program_factory.cpp:43` fixes two heads per tensor per core and derives `out_block_w` from it (`:49`). `num_heads` is not read. `compute_output_specs` instead derives per-core heads as `num_heads / num_M_cores` (`device_operation.cpp:94`). The program and output spec agree only when that quotient is 2.
- **Dead locals in the sharded kernels:** `block_wt` (CTA 2) and the `cb_im0` / `cb_out1` constexprs in the sharded reader, and `cb_out0..2` in the sharded writer, are unused.

## Questions for the user

1. **Waiver:** the only RED is the stale #57409 sheet row (`Concept` = `legacy device-op` vs. `create_descriptor` in code). Waive it, and issue the brief? — **Resolved 2026-10-07: the user waived it ("Waive"); brief issued.**

## Recipe notes

- *TTNN factory concept → cross-check → Factory-set match:* PD batches like #57409 delete per-factory `*_program_factory.hpp` files and fold the declarations into the device-op header, while the factory **names** survive. The sheet's `Factory definition path` then points at a deleted file even though the factory-set match is exact. The recipe doesn't say whether a stale path cell counts as a "phantom row". I treated it as part of the same stale-row finding rather than as a separate phantom-factory trigger, because the factory still exists.
