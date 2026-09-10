# Quasar Uplift Report — `data_movement/slice`

*Uplift audit per `quasar_porting.md` + `ai/audit/quasar_audit.md` (+ `cb_dfb_quasar_audit_helper.md`),
run against the Metal 2.0 state of PR #55216 (`edwinlee/Port_Slice`, cherry-picked onto
`vsureshTT/quasar_uplift_round_2`). No build and no device run were performed in either session — the
user runs those; the parity and Quasar verdicts below are argued structurally (recipe §9, "Auditing
without a device run") and the exact commands to confirm them are at the end.*

Date: 2026-09-10 · Branch: `vsureshTT/quasar_uplift_round_2` · Op dir: `ttnn/cpp/ttnn/operations/data_movement/slice`

**Session 2 (same day): the deferred in-op items were applied.** Section A0 below lists every file
changed and why; section D records what was applied vs what stays deferred; sections F/G carry the
updated parity argument and test commands. The session-1 audit text is otherwise left as written.

---

## Status: **GREEN** on the Quasar test path (`SliceTileProgramFactory`, unchanged) · **2 deferred items applied off-path** · **1 owner decision still open** (`SliceRmSharded::out_shard`)

- The Quasar model test (`models/experimental/llama32_1b_quasar/tests/graph_ops/test_slice.py`) reaches
  **`SliceTileProgramFactory`**, which **is** Metal 2.0 (`create_program_artifacts` → `ProgramArtifacts`,
  `CustomProgramSpecFactoryConcept`), and both kernels it binds are on the device-2.0 kernel API. So the
  RED "not Metal 2.0 yet" condition does **not** apply to the test path.
- Every §7/§8/§11 gotcha was checked against that factory and its two kernels; **none fires**, so per §2
  ("already-M2 → often a no-op"; fixes are reactive) **the test-path factory and its kernels are untouched**.
- **Files changed (session 2): 4**, all off the Quasar test path, all in already-Metal-2.0 factories, all
  WH/BH-behaviour-preserving by the recipes' own statements — see §A0. This report is the only new file
  (uncommitted, delete before merge).
- **Off-path Gen2 debt, after session 2:**
  - `SliceTileTensorArgs` — the `tensor_stage` DM self-loop is **converted to a `Scratchpad`**
    (`dm_self_loop_dfbs.md`). That factory no longer has a construct the Gen2 validator rejects.
  - `SliceRmSharded` — `in_shard` (sync-free, borrowed) is **converted to a `LocalTensorAccessor`**
    (`sync_free_dfbs.md`). `out_shard` remains a **synchronized single-ended DM producer** self-looped on
    the reader: the audit helper classifies that as **STOP → owner decision** and no recipe gives a
    mechanical fix, so it is **left in place and still flagged (D2)**. If a Quasar caller reaches this
    factory it is **RED** until the owner resolves `out_shard`; the validator will now name `out_shard`
    (previously it named `in_shard` first).

### RED-stop conditions checked (recipe §1)

| RED condition | Test path (`SliceTile`) | Rest of op |
|---|---|---|
| Not Metal 2.0 on Gen1 yet | No — ported (see §A) | No — all 5 factories ported by #55216 |
| Required capability missing from sanctioned Quasar API (`evil_set_*`, etc.) | No — no `evil_*`, no cursor surgery anywhere in the op | No |
| Construct needing an owner decision (non-zero-init semaphore / DM self-loop / open HW bug) | No — zero semaphores, no self-loop, no mcast | **Yes, one, factory-scoped**: `SliceRmSharded::out_shard` (single-ended DM producer, self-looped) — see D2. `SliceTileTensorArgs::tensor_stage` and `SliceRmSharded::in_shard` were resolved in session 2 (D1, D2). |
| Only fix would change WH/BH un-guarded / require `experimental/quasar/` copy | No fix needed | n/a |
| LLK the op needs is a stub | No compute kernel in the op at all | No |

---

## A0. Session 2 — changes applied (every file, one reason each)

All four edits are in factories that were already `create_program_artifacts` / `ProgramArtifacts`
(Metal 2.0); no legacy factory was touched, no file outside the op directory was touched, and no
`_metal2` fork was needed (both kernels are slice-owned single-binder kernels — the only other
directory naming them binds its *own* same-named copy under `experimental/quasar/slice/`, which is
out of bounds and not a consumer of these files).

| File | Change | Recipe | Why it is WH/BH-neutral |
|---|---|---|---|
| `device/slice_program_factory_tile_tensor_args.cpp` | `tensor_stage`: `DFBSpecName` → `ScratchpadSpecName`; the `DataflowBufferSpec` (1 entry × `single_tile_size`) → `ScratchpadSpec{.size_per_node = single_tile_size * num_tensor_stage_entries}`; the reader's two `DFBBinding`s (PRODUCER+CONSUMER) → one `ScratchpadBinding{"tensor_stage"}`; `spec.scratchpads = {tensor_stage_scratch}`. `src0`, all tensor parameters, run args and `override_runtime_arguments` untouched. | `dm_self_loop_dfbs.md` (host side) | Same L1 bytes reserved per node (entry_size × num_entries); allocation *order* may shift (recipe-stated, nothing depends on it). |
| `device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | `DataflowBuffer dfb_tensor(dfb::tensor_stage)` → `Scratchpad<volatile uint32_t> tensor_stage(scratch::tensor_stage)`; the two staging rounds lose `reserve_back/push_back/wait_front/pop_front` and the `get_write_ptr()` + C-style `uint32_t → volatile tt_l1_ptr uint32_t*` cast, and read `tensor_stage[i]` directly; the two NOC reads take the scratchpad as destination with `{.offset_bytes = 0}`. Added `#include "api/scratchpad.h"`. Everything else (the `src0` loop, `id_per_dim`, the dead `end_indices` read, `old_src_tile_id`) untouched. | `dm_self_loop_dfbs.md` (kernel side) | Recipe translation: `num_entries == 1`, so every FIFO index is `0` at every read (each `push_back(1)` wrapped to 0) — no stride, no wrap, indices dropped (cleanup item 1); `T = volatile uint32_t` read off the old cast (volatile carried across); same NOC reads, same sizes, same barriers, same words read back. |
| `device/slice_program_factory_rm_sharded.cpp` | `in_shard` removed: its `DFBSpecName`, its borrowed `DataflowBufferSpec`, and its two self-loop `DFBBinding`s; the reader gains `TensorBinding{names.input, "input"}`; `spec.dataflow_buffers = {out_shard_dfb}`. `out_shard` (spec, bindings, comment) kept verbatim. | `sync_free_dfbs.md`, borrowed → `LocalTensorAccessor` | The kernel only ever took the DFB's base address; the tensor binding hands it the same input-shard L1 base address via CRTA. The input `TensorParameter` and `run_args.tensor_args` are unchanged, so the per-dispatch re-pointing is identical. Validator already required a borrowed tensor to be L1 (`program_spec.cpp:1607`), so the LTA's `!is_dram` static_assert adds no new constraint. |
| `device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp` | `DataflowBuffer dfb_in(dfb::in_shard)` + `dfb_in.get_write_ptr()` → `LocalTensorAccessor<uint8_t> in_shard(tensor::input)` + `in_shard.get_bank_base_address()`. Added `#include "api/tensor/local_tensor_accessor.h"`. All NOC reads, `dfb_out` handling and barriers untouched. | `sync_free_dfbs.md` | Same base address (a sync-free DFB's `get_write_ptr()` is its base for the whole kernel); `T = uint8_t` because the kernel never dereferences the region, only hands the address to `UnicastEndpoint` reads. On Quasar this is if anything *more* correct: post-#52769 `get_write_ptr()` returns an uncached alias, which the kernel was using as a remote NOC address. |

**Repaired because the change falsified it** (per `pass_procedure.md` Step 3):
- `slice_program_factory_rm_sharded.cpp`: the `tt::DataFormat dfb_data_format` local (input dtype) lost its only consumer (`in_shard_dfb.data_format_metadata`) → deleted (would otherwise be an unused variable). `dst_dfb_data_format` stays.
- `slice_program_factory_rm_sharded.cpp`: the comment block over the two DFB specs described `in_shard` as a borrowed self-loop DFB → rewritten to describe the tensor binding + LTA and the unchanged `out_shard`.
- `slice_program_factory_rm_sharded.cpp::override_runtime_arguments`: "Both now resolve by name from their backing tensor" clarified to say *how* each resolves (tensor binding vs borrowed DFB); the claim itself stays true.
- `slice_program_factory_tile_tensor_args.cpp`: the `TensorArgsSpecNames::tensor_stage` comment ("single-entry staging buffer the reader fills and drains") and the spec comment ("one toucher that already runs both halves of the handshake") → rewritten for the scratchpad.
- `reader_..._tensor_args.cpp`: the two "Complete the producer/consumer handshake … so the scratch DFB is left balanced" comments described calls that no longer exist → removed; the `start_buffer_l1_addr` / `end_buffer_l1_addr` locals whose only consumers were the deleted casts → removed (cleanup item 4).
- `slice_reader_unary_unpad_dims_rm_sharded.cpp`: the two comments calling `l1_read_addr` a pointer into "this core's own borrowed input DFB" → now "this core's own input shard".

**`data_format_metadata` on `tensor_stage`** — established inert before dropping (recipe host-side step): every use of the DFB handle in the reader was a FIFO call, `get_write_ptr()`, or a NOC destination; the one `get_entry_size()` in that kernel is on `dfb_in0`, not `dfb_tensor`. Nothing consulted the format.

**Survey evidence for D1 (`dm_self_loop_dfbs.md` Step 2):** the only binder of `tensor_stage` was the reader `KernelSpec` (PRODUCER + CONSUMER, a DM kernel); `borrowed_from` unset; no `dfb_run_overrides` anywhere in the factory; every use of the handle was on the covered list (`reserve_back`, `get_write_ptr`, NOC dst, `push_back`, `wait_front`, `pop_front`, twice each); no `pages_*`, no `async_write_zeros`, no multicast, no helper/RAII/template use of the handle or id; `entry_size (tile_size) % sizeof(uint32_t) == 0`; both `get_write_ptr()` captures were taken at index 0 (1-entry buffer, wrap-to-0 after each push) so no stride is needed and no CTA had to be invented.

**Survey evidence for D2-`in_shard` (`sync_free_dfbs.md` Step 2):** one binder (the reader, a DM kernel); grep for the six credit methods on `dfb_in` → zero hits (the four FIFO calls in that file are all on `dfb_out`); the handle / id is passed nowhere (no helper, guard, or template); one configuration path (the kernel's `can_coalesce` branch only changes the NOC transfer shape, both branches take the same address); `borrowed_from = names.input` → LTA end-state; one kernel per node → no shared-scratch feature request.

---

## A. Which factory the Quasar test reaches, and its Metal 2.0 state

**Test case** (`test_slice.py`, id `00_1024x2048_bf16_int-dram`): input `[1,1,1024,2048]` BF16, TILE,
DRAM interleaved; `begins=(0,0,480,0)`, `ends=(1,1,512,2048)`, default step; output `[1,1,32,2048]`
TILE DRAM interleaved. Golden: torch slice (`graph_case.py:567 _ref_slice`), PCC ≥ 0.999.

**Host path trace** (`slice.cpp`):
1. `ttnn::slice<T>` — not a no-op (`starts_zero` false), rank 4, `no_step` true.
2. `check_handled_tile_alignment()` (`slice.cpp:226-230`): `480 % 32 == 0`, `0 % 32 == 0` → true;
   `one_dimensional` false → `rm_only = false` (`:237`). No to_layout, no composite hop (`rm_in_bad`,
   `rm_out_bad`, `out_no_spec` all false — input is interleaved).
3. `padded_ends` = ends (already tile multiples); `ttnn::prim::slice(..., use_tensor_args=false, ...)`
   (`:398-410`).
4. `SliceDeviceOperation::select_program_factory` (`slice_device_operation.cpp:314-346`):
   `use_tensor_args` false → not `SliceTileTensorArgs`; layout TILE → **`SliceTileProgramFactory{}`** (`:345`).
5. Post-op: `ttnn::experimental::view` (metadata only), `ret_adjustment` → `to_memory_config` (same
   config, no-op) + `to_layout(TILE)` (no-op). No second device op on the path.

**Factory** — `device/slice_program_factory_tile.cpp`:
- `create_program_artifacts` (`:114-261`) returns `ProgramArtifacts{spec, run_params}`;
  `override_runtime_arguments` (`:263-297`) returns `ProgramRunArgs` (custom concept, re-supplies both
  tensor bindings + per-node scalars on every cache hit).
- Kernels: `reader` = `device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp`,
  `writer` = `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (slice's own copy; only
  binder — the `experimental/quasar/slice` fork binds its *own* duplicate under its own directory, so the
  mainline kernel is not lent).
- One DFB `src0` (`:135-140`): `entry_size = tile_size(bf16)`, `num_entries = 2`,
  `data_format_metadata = Float16_b` (valid; not `Invalid`, not bf8, not uint16/uint32). Reader
  PRODUCER (`:179-183`), writer CONSUMER (`:209-213`) — a plain cross-kernel FIFO, **not** a self-loop,
  no `borrowed_from`, no `alias_with`, no `allow_instance_multi_binding`, no `dfb_run_overrides`.
- Tensor parameters `input`/`output` (`:143-144`), bound as `tensor::input` / `tensor::dst`.
- `hw_config` = `ttnn::create_reader_datamovement_config(device->arch())` /
  `create_writer_datamovement_config(...)` (`:191`, `:220`) — the arch-agnostic helper
  (`ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp:27-43`), which
  returns `DataMovementGen2Config{}` on `ARCH::QUASAR` with `disable_dfb_implicit_sync_for_all = false`.
  This is `gen2_hardware_configs.md` shape 1 — "no work; do not touch".
- `compiler_options` absent → `O2`, which equals the legacy DM default (no compute kernel exists, so
  the O3 trap of §4 does not arise).
- No `SemaphoreSpec`, no multicast, no `TensorAccessor` 3rd argument, no varargs beyond the two
  legitimate indexed blocks (`id_per_dim` per-node RTA varargs, `num_dims*2` common varargs).

**Kernels** (both DM; the op has no compute kernel):
- `reader_unary_unpad_dims_interleaved_start_id.cpp`: `#include "api/dataflow/noc.h"`,
  `"api/dataflow/dataflow_buffer.h"`, `"api/tensor/noc_traits.h"`, `"experimental/kernel_args.h"`;
  `get_arg(args::…)`, `get_vararg`/`get_common_vararg`, `TensorAccessor(tensor::input)`,
  `DataflowBuffer dfb_in0(dfb::in0)`, `dfb_in0.get_entry_size()` (`:39` — §5 rule satisfied, no
  `fifo_page_size`), `reserve_back(1)` → `noc.async_read(s0, dfb_in0, tile_size, {.page_id}, {.offset_bytes=0})`
  → `async_read_barrier()` → `push_back(1)` (`:45-48`).
- `writer_unary_interleaved_start_id.cpp`: same headers; `DataflowBuffer dfb_out(dfb::out)`,
  `get_entry_size()` (`:23`), `TensorAccessor(tensor::dst)`, `wait_front(1)` →
  `noc.async_write(dfb_out, s, page_bytes, …)` → `async_writes_flushed()` → `pop_front(1)` (`:42-45`),
  `async_write_barrier()` (`:47`). `#ifdef OUT_SHARDED` / `#ifdef BACKWARDS` are dead (no defines set).
- No legacy `dataflow_api.h` free-function NOC calls, no `CircularBuffer`, no `cb_*`, no
  `get_local_cb_interface`, no `get_cb_tiles_*_ptr`, no `read_tile_value`, no `get_pointer_to_cb_data`,
  no `evil_*`, no `TensorAccessorArgs<N>`, no positional args (scan output in §E).

**Verdict on the Metal 2.0 gate (§1 step 1):** done and green on WH per the PR's own
`METAL2_PORT_REPORT.md` (448 passed / 38 skipped on `test_slice.py`; 321 passed nightly). This session
did not re-run it.

---

## B. Quasar-uplift audit of the test path (`quasar_audit.md` + `cb_dfb_quasar_audit_helper.md`)

### Check 1 — device-side CB/DFB redesign (kernel audit)

| Buffer | Class | Kernel(s) | 1xx status | 2xx (Quasar) status | Notes |
|---|---|---|---|---|---|
| `src0` (`dfb::in0` / `dfb::out`) | **1 — linear FIFO** | reader (PRODUCER), writer (CONSUMER) | Portable (`1xx_port: already_DFB`) | **Portable** | Canonical explicit reserve/push → wait/pop; Noc from `noc.h` with the DFB as endpoint; no pointer peeks, no surgery. |

GATE hits: none. Blocked-on-runtime: none. **Rollup: GREEN.**

### Check 2 — non-zero-init semaphores

None: `SliceTileProgramFactory` declares no `SemaphoreSpec` (nor does any slice factory — grep in §E).
Gen2 validator check at `program_spec.cpp:1756-1762` cannot fire.

### Gen2 spec-validator checks (`tt_metal/impl/metal2_host_api/program_spec.cpp`) walked for this spec

| Check | Line | Fires? |
|---|---|---|
| DM `hw_config` must hold `DataMovementGen2Config` on Quasar | 884-889 | No — helper returns Gen2 config on Quasar |
| Compute config must be Gen2 | 902-907 | n/a — no compute kernel |
| Thread counts (`num_threads` default 1) | 818-846 | No |
| `allow_instance_multi_binding` rejected on Gen2 | 1298-1300 | No — flag not set |
| **DM self-loop rejected on Gen2** | 1486-1500 | **No for `SliceTile`** (reader and writer are distinct kernels). **Yes for `SliceRmSharded` and `SliceTileTensorArgs`** — see Deferred. |
| Non-zero semaphore init | 1756-1762 | No — no semaphores |
| Per-WU engine/DM-core budget | 1804-1812 | No — 2 DM kernels, 0 compute |
| DFB slots per node | 1264 | No — 1 DFB |

### Explicit vs implicit sync on Quasar — why the kernels' explicit pattern is correct as written

The recipe (§7, §12) says implicit sync is the Gen2 default and must not be disabled. Slice's two
kernels keep the **explicit** pattern *and* pass the DFB straight to `Noc::async_read/async_write`.
I verified from the runtime that this is the sanctioned explicit mode, not a double-count:

- Implicit sync is engaged **only** by the `NocOptions::TXN_ID` overloads that take a `DataflowBuffer&`
  (`tt_metal/hw/inc/api/dataflow/noc.h:824-852`, implemented at
  `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:595-639` via `prepare_implicit_read/write` →
  `commit_implicit_read/write`). Those are the only sites that advance the implicit shadow state
  (`ptiles_read_` / `ctiles_written_`, `dataflow_buffer.h:420-428`). Slice calls the plain
  `async_read(src, dfb, size, …)` / `async_write(dfb, dst, size, …)` overloads — no `TXN_ID` — so the
  DFB is just an address endpoint (`noc_traits`, cached view via `l1_cached_view`, `noc.h:112-121`).
- The explicit calls on Quasar DM are pure tile-counter operations:
  `reserve_back_impl` polls `llk_intf_get_free_space` (`tt-2xx/dataflow_buffer.inl:140-172`),
  `push_back_impl` → `llk_intf_inc_posted` + wr_ptr advance (`:174-207`), `wait_front_impl` polls
  occupancy (`:209-227`), `pop_front_impl` → `llk_intf_inc_acked` (`:229-250`). None consults the
  implicit state, so nothing double-counts. This matches `DataflowBuffer.md` Part A "Explicit sync",
  which shows exactly this DM reader/writer shape.
- `finish()` is not called by either kernel; on the explicit path `finish_impl` (`:252-`) only
  drains `posted == acked` and `handle_final_credits` is skipped when the implicit counters are 0, so
  its absence is not a correctness gap for this op (noted, not changed — adding it would be an
  unguarded WH/BH kernel change and the WH kernels never called it either).

`disable_dfb_implicit_sync_for_all` is **not** set anywhere in the op (§7 / §10 checkbox satisfied).

---

## C. §7 / §8 / §11 gotchas — applied vs considered-and-rejected

**Applied: none.** (No symptom can fire without a device run, and none is structurally present.)

**Considered and rejected for the test path** (reason in parentheses):

| Gotcha | Verdict |
|---|---|
| §7 `disable_dfb_implicit_sync_*` must not be set | Not set. Helper called with default `false`. |
| §7 `compute_kernel_hw_startup` once / re-`*_init` on DFB-id change / bare `wait_front→pop_front` TEN-4746 / tilize pack config / no DEST wrap / `matmul_partials` borrow | **n/a — the op has no compute kernel.** TEN-4746 is a TRISC hazard; the recipe explicitly excludes DM kernels ("their CB credits go through the NOC overlay tile counters"). |
| §7 Quasar has Int32, no uint16/uint32 | No dtype-specific branch in either kernel or the factory (`tile_size(dfb_data_format)` forwards the dtype). Test is BF16. Nothing to guard. Op-level limitation (uint16/uint32 slices) lives at the format layer — flag only. |
| §7 RM shard width 16-byte aligned | n/a — TILE path, interleaved. |
| §7 non-zero-init semaphores | None. |
| §7 "don't invent Quasar-only interfaces" / `evil_set_*` missing on Gen2 | Op uses no `evil_*` (grep §E). |
| §5 `fifo_page_size` stale on Quasar | Both kernels already use `get_entry_size()`. |
| §4 `unpack_modes` for FP32 | n/a — no compute kernel. |
| §4 designated-initializer order | Already compiles on WH (PR CI); no new fields added here. |
| §4 `hw_config` carry values not names | Both kernels resolve to the reader/writer defaults; helper reproduces them on Gen1 and emits `DataMovementGen2Config{}` on Gen2 (`gen2_hardware_configs.md` shape 1). |
| §4 `opt_level` | Absent = O2 = legacy DM default; no compute kernel. Not an uplift edit anyway. |
| §6 DM self-loop rejected on Gen2 | **Not on the test path** (distinct producer/consumer kernels). Off-path: `tensor_stage` and `in_shard` converted in session 2; `out_shard` remains (D2, owner decision). |
| §6 borrowed-DFB capacity | No `borrowed_from` on the test path (interleaved DRAM in/out). |
| §6 local self-copy on emulator | No L1→L1 loopback; reads DRAM→L1, writes L1→DRAM. |
| §8.1 DM kernel including `common.hpp` (ckernel symbols) | Neither test-path kernel includes `ttnn/.../common/kernels/common.hpp` (the RM kernels do, and that header already carries the `ARCH_QUASAR` guard on main). |
| §8.1 `-Werror=int-to-pointer-cast` | No integer→pointer cast in either test-path kernel. The off-path C-style casts in `…_tensor_args.cpp` were removed by the D1 scratchpad conversion (D4 moot). |
| §8.1 `MEM_ZEROS_BASE` / `flush_l2_cache_range` / `blank.cpp` / `REDUCE_OP` / `sfpu_reduce` | Not referenced. |
| §8.2 implicit-sync double-count / credit stall | Explicit path only; see §B. If a `WFW`/`RBW` hang appears on the emulator anyway, report it to the runtime team — do **not** flip `disable_dfb_implicit_sync_for_all`. |
| §8.2 "Not done phys cores" 8-bit capacity | `num_entries = 2`. |
| §8.3 value inflation via `fifo_page_size` | Not used. |
| §8.3 stale L2 on reused L1 (`get_read/write_ptr` cached) | Kernels never dereference a DFB pointer; NOC endpoint only. Post-#52769 no manual flush is wanted anyway. |
| §8.3 buggy default program hash reusing a stale kernel | Slice has a custom `compute_program_hash` (`slice_device_operation.cpp:348-432`) folding shapes/layout/dtype/mem-config; left alone (off-limits, and already vetted by the PR's audit). |
| §11 multicast rectangle / NOC0-NOC1 tricks / degenerate-grid clamp | No multicast anywhere in the op; `split_work_to_cores` over `compute_with_storage_grid_size()` adapts to the emulator's small grid (the test's 32 output tiles split across whatever cores exist; inactive nodes get `num_tiles = 0` rows, which both kernels loop zero times). |
| §11 pad W-dim tail | n/a — tile-granular reads. |
| §11 `async_write_zeros` | Not used. |
| §12 tile-counter remapper / intra-tensix aliasing | Runtime-owned; one DM↔DM DFB per node — nothing at op level. |
| Quasar no Bfp8 (`ValidateProgramSpec` rejects bf8 DFBs) | Test is BF16. A BF8 slice on Quasar would be rejected by the validator — an op-level dtype limitation, not a kernel change (`data_format_metadata` merely forwards the dtype). Flag only. |

---

## D. Deferred / follow-up items (off the Quasar test path) — status after session 2

**D1 — `SliceTileTensorArgsProgramFactory`: DM self-loop `tensor_stage`. ✅ APPLIED (session 2).**
Converted per `ai/post_port/semantic/dm_self_loop_dfbs.md` — see §A0 for the file-by-file change and
the survey evidence. The Gen2 validator check at `program_spec.cpp:1494-1500` can no longer fire for
this factory (its only remaining DFB, `src0`, is a plain reader→writer FIFO). **Not yet verified by a
run** (this session builds and tests nothing); the sentinel set for it is the tensor-args subset of
`tests/ttnn/unit_tests/operations/data_movement/test_slice.py` — `test_slice_tensor_args`,
`test_slice_tensor_args_device_path`, `test_slice_tensor_args_upper_dim_offset`,
`test_slice_subcores` (the `args_as_tensor=True` ids), plus `test_slice_tensor_args_mesh_device` and
the `ccl/mesh_partition` tests where a multi-device box is available (the port PR could not run those
either). The full-file runs in §G cover all of them.

**D2 — `SliceRmShardedProgramFactory`: `in_shard` ✅ APPLIED, `out_shard` ⛔ STILL OPEN (owner decision).**
- `in_shard` (sync-free, borrowed, address-only) → `LocalTensorAccessor<uint8_t>` over `tensor::input`
  per `sync_free_dfbs.md` — see §A0. The "local base address used as a remote NOC address" behaviour
  is preserved exactly: the LTA's `get_bank_base_address()` is the input shard's L1 base on this core,
  which the sharded allocation places at the same offset on every core in the range.
- `out_shard`: **synchronized single-ended DM producer** into the resident output shard
  (`reserve_back(num_sticks)` … NOC reads land in it … `push_back(num_sticks)`; nothing drains). Still
  bound PRODUCER+CONSUMER on the reader (`slice_program_factory_rm_sharded.cpp`, `.dfb_bindings`) and
  still `borrowed_from = names.output`. `cb_dfb_quasar_audit_helper.md` ("DM · single-ended producer
  (real reserve/push, no consumer) → **STOP** — surface to API owner. Do not self-loop; prefer writing
  the tensor directly") and `dm_self_loop_dfbs.md` (a `borrowed_from` DFB is a stop for that pass; a
  borrowed buffer with real FIFO calls is "a combination nothing in this suite has examined") both
  decline it, and `sync_free_dfbs.md` does not apply (it has real credit calls). **Not edited.**
  - **Exact symptom on Quasar** (any HEIGHT-sharded RM in *and* out, no step, L1-aligned W-begin —
    `select_program_factory` routes there): `MakeProgramFromSpec` → `TT_FATAL` from
    `program_spec.cpp:1494-1500`: *"DataflowBuffer 'out_shard' is self-looped by data-movement kernel
    'reader' (bound as both PRODUCER and CONSUMER). Self-loop DFBs are not supported for data-movement
    kernels on Gen2 architectures. Consider using a scratchpad or LocalTensorAccessor instead."*
  - **What the owner has to decide:** the reader's `reserve_back`/`push_back` on a buffer nobody
    consumes is credit bookkeeping with no counterpart; the natural end-state the helper names is to
    write the output shard directly (i.e. a `LocalTensorAccessor<…>` over `tensor::output` as the NOC
    destination, `l1_write_addr = out.get_bank_base_address()`), dropping the two credit calls. That
    deletes real (if pointless) synchronization and is exactly the change the recipes say not to make
    unilaterally. Sentinels for whoever does: `test_slice_rm_sharded_with_program_cache`,
    `test_slice_override_addr_change_rm_height_sharded`, `test_slice_rm_height_sharded_override_cache_hit_is_o1`,
    `test_slice_rm_height_sharded_cache_hit_correctness`, `test_slice_override_alternating_factories_cache_hit`
    in `test_slice.py`.
- Until then **this factory is RED on Quasar**; nothing on the model test path reaches it.

**D3 — `SliceRm`, `SliceRmStride`: unchanged, no Gen2 structural blocker found, not exercised.**
Both use one plain reader→writer FIFO, arch-agnostic DM configs, no semaphores, no `evil_*`. Their
kernels include `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` (already
`ARCH_QUASAR`-guarded on main: `common.hpp:17`, `:121`, `:151`) and call
`tt::data_movement::common::noc_async_read_sharded` / `noc_async_write_sharded` — the shared helpers
are outside this op's writeable surface; any Quasar build skew there is a
`data_movement/common` item, not slice's. RM sticks are sub-tile NOC transfers: the helper header's own
comment (`datamovement_kernel_config.hpp:21-24`) says such kernels can stall implicit credit
accounting — irrelevant here because slice keeps explicit sync (no `TXN_ID`). Verify with an RM slice
on the emulator when one is on a model path; nothing to change pre-emptively.

**D4 — C-style `uint32_t → volatile tt_l1_ptr uint32_t*` casts in the tensor-args reader. ✅ MOOT.**
Both casts (`:63`, `:80` pre-edit) were the raw-address reads of the staging DFB and were removed by
the D1 scratchpad conversion (`tensor_stage[i]` through the bounds-checked `Scratchpad::operator[]`).
No int→pointer cast remains in any referenced slice kernel.

**D5 — Two unreferenced legacy kernels** remain in the directory and were **not** audited or touched:
`device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp`,
`device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp` (no factory binds them; still
`TensorAccessorArgs<N>` / positional-CTA). Already flagged by the PR's port report for deletion; deleting
them is not an uplift edit.

**D6 — Dtype limitations to flag, not fix:** a BF8 or uint16/uint32 slice on Quasar is rejected /
unsupported at the format layer (Quasar has no Bfp8 and no uint16/uint32 device format). Slice only
forwards `input.dtype()`; per §7 that is a format/LLK-layer limitation and no op edit is warranted.

**D7 — Recipe feedback (`quasar_audit.md` "more checks land here"):** the DM-self-loop check is now
a hard validator error on Gen2 (`program_spec.cpp:1494-1500`); worth adding as check 3 with that
message as the grep signature, since it is exactly what D1/D2 hit. Two further observations from
applying the passes:
- `sync_free_dfbs.md`'s borrowed→LTA branch has **no in-tree op precedent** (the only
  `LocalTensorAccessor` users outside `experimental/quasar/` are metal2 gtests and the minimal-matmul
  dataflow headers); the cited `layernorm_pre_all_gather_welford` example is not in this tree. The
  recipe's transformation was still unambiguous for an address-only use, but a worked op example would
  help the next porter.
- The audit helper's "DM · single-ended producer → STOP" row and `dm_self_loop_dfbs.md`'s
  "`borrowed_from` set → stop" both land on `out_shard`; a recipe for "single-ended borrowed DM
  producer → write the tensor directly through an LTA" would turn this whole class from an owner
  decision into a mechanical pass. It is the shape every sharded-output DM-only op will have.

**Noticed, not done (pre-existing, outside the uplift's remit):**
- `reader_..._tensor_args.cpp` still stages and reads the *end* tensor into `[[maybe_unused]] end_indices`
  (port report "Pre-existing findings" #4) and keeps the dead `old_src_tile_id` local (#5). Preserved.
- The `tensor_stage` scratchpad is sized by the *input* dtype's tile size while it receives a page of
  the start/end (integer) tensors and is read as `uint32_t` words — pre-existing sizing carried across
  verbatim (the DFB was sized identically). A `num_dims`-word region would suffice; an owner decision.
- `slice_program_factory_rm_sharded.cpp` still `#include`s `<tt-metalium/hal.hpp>` for the alignment
  check and computes `src_stride_bytes` for the reader CTA — both still used; nothing became dead
  beyond the `dfb_data_format` local repaired above.

---

## E. Scan evidence (whole op directory, referenced kernels only)

- `get_local_cb_interface|fifo_page_size|fifo_num_pages|fifo_wr_ptr|fifo_rd_ptr|get_cb_tiles_*_ptr|read_tile_value|get_tile_address|get_pointer_to_cb_data|evil_set|push_back_hold|CircularBuffer|cb_id|CBIndex|get_compile_time_arg_val|get_arg_val|TensorAccessorArgs|MEM_ZEROS_BASE|invalidate_l2|flush_l2|multicast|mcast|Semaphore|sem::|ARCH_QUASAR|disable_dfb_implicit|finish()` over `device/kernels/dataflow/*.cpp` (excluding the two unreferenced `strided_slice_*` files): **0 hits** except comments/calls to the shared `noc_async_read_sharded`/`noc_async_write_sharded` helpers in the RM kernels (D3).
- Factory scan for `SemaphoreSpec|allow_instance_multi_binding|Gen1Config|Gen2Config|std::get<|std::get_if<|holds_alternative|opt_level|disable_dfb_implicit|ARCH::QUASAR|alias_with|dfb_run_overrides|compiler_options`: **0 hits** in all five factories; `hw_config` is the arch-agnostic helper at every one of the 8 DM `KernelSpec`s; `borrowed_from` only in `rm_sharded.cpp:312,319`.
- Self-loop bindings (same DFB, PRODUCER and CONSUMER, one KernelSpec) — session 1: `rm_sharded.cpp:352-371` (2), `tile_tensor_args.cpp:141-150` (1). **After session 2: `rm_sharded.cpp` (1, `out_shard`) only.** `tile.cpp`, `rm.cpp`, `rm_stride.cpp`, `tile_tensor_args.cpp`: none.
- `git status --porcelain -- ttnn/cpp/ttnn/operations/data_movement/slice` before session 1: clean. After session 2: exactly the four files in §A0 modified + this report untracked; nothing outside the op directory.

---

## F. Parity claim (WH/BH)

**Test path (`SliceTileProgramFactory` + its two kernels): zero-diff.** Byte-identical to the branch as
cherry-picked from PR #55216. The one behaviour Quasar gets that WH/BH do not — a
`DataMovementGen2Config{}` from the helper — was already present in the ported factory and is selected
at runtime by `device->arch()`, never on Gen1.

**Off-path edits (`SliceTileTensorArgs`, `SliceRmSharded`): behaviour-preserving on WH/BH by
construction, unguarded on purpose.** Neither recipe applied is Quasar-specific; both are Gen1
post-port passes whose stated property is "results, numerics and observable behaviour are identical"
(`sync_free_dfbs.md`) / "the converted kernel performs the same reads and writes, in the same order, to
and from the same remote addresses" (`dm_self_loop_dfbs.md`). Concretely:
- `tensor_stage`: same two NOC reads (same source accessor, page 0, `tile_size` bytes) into a
  per-node private L1 region of the same size, same `async_read_barrier()`s, same `num_dims` words read
  back as `volatile uint32_t`. What disappears is credit traffic on a stream register that nobody else
  read (a 1-entry buffer polled and posted by its sole toucher). The only observable difference is
  *where* in L1 the region lands (scratchpads and DFBs are allocated from the same region, order may
  shift) — nothing in the op depends on the address.
- `in_shard`: the address the kernel hands to `UnicastEndpoint` reads is the input shard's L1 base on
  this core in both versions (a sync-free DFB's `get_write_ptr()` never moves off its base; the borrowed
  DFB's base *is* the tensor's buffer address; the tensor binding's CRTA *is* the same address). The
  input `TensorParameter` and the per-dispatch `run_args.tensor_args` entry are unchanged, so cache-hit
  re-pointing is identical. One DFB id per node is freed; nothing consumed it.
- There is **no `ARCH_QUASAR` guard to audit** because there is no Quasar-only edit: the same source
  runs on WH, BH and Quasar. The `LocalTensorAccessor` `static_assert(!is_dram)` cannot fire on any
  configuration that previously ran, because the borrowed DFB it replaces was already validator-limited
  to L1 tensors (`program_spec.cpp:1607`).

Confirm by running the WH/BH suites below and comparing against the PR's recorded baseline
(`METAL2_PORT_REPORT.md`: `test_slice.py` 448 passed / 38 skipped; nightly pair 321 passed / 4 skipped).
Purge the JIT cache (or force JIT) before the post-edit runs: both edited kernels are compiled from the
working tree at program-cache miss time, and a stale binary would mask the kernel edits.

---

## G. Commands for the user (order per §9: BH → WH → Quasar)

Force JIT once so no stale kernel binary masks anything, and run with Watcher on:

```bash
export TT_METAL_WATCHER=10
export TT_METAL_FORCE_JIT_COMPILE=1
```

Build first (host factories changed): `./build_metal.sh` (or your usual `ninja` target + install step),
then `rm -rf ~/.cache/tt-metal-cache` so no pre-edit kernel binary survives.

**Blackhole** (parity — the op's own suite; note the Quasar model harness skips on BH by design,
`models/experimental/llama32_1b_quasar/tests/conftest.py:154`):
```bash
# fast focus on the two edited factories first
pytest tests/ttnn/unit_tests/operations/data_movement/test_slice.py -v \
       -k "tensor_args or subcores or rm_sharded or rm_height_sharded or alternating_factories"
# then the full parity set
pytest tests/ttnn/unit_tests/operations/data_movement/test_slice.py -v
pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_slice_for_conv.py \
       tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_slice.py -v
```

**Wormhole** (parity; expected 448 passed / 38 skipped and 321 passed / 4 skipped per the PR report; on a
multi-card box restrict with `TT_VISIBLE_DEVICES=0` as the PR did):
```bash
pytest tests/ttnn/unit_tests/operations/data_movement/test_slice.py -v \
       -k "tensor_args or subcores or rm_sharded or rm_height_sharded or alternating_factories"
pytest tests/ttnn/unit_tests/operations/data_movement/test_slice.py -v
pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_slice_for_conv.py \
       tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_slice.py -v
# multi-device only (t3000 / TG): the tensor-args mesh path and its ccl consumer
pytest tests/ttnn/unit_tests/operations/data_movement/test_slice.py -v -k tensor_args_mesh_device
pytest tests/ttnn/unit_tests/operations/ccl/ -v -k mesh_partition
```

**Quasar emulator** (the model-level test; run once with LLK asserts on, once off — §9). The test path
is unchanged, so these runs re-confirm session 1's GREEN; the edited factories are not reached by the
model test. To exercise the D1 conversion on Quasar directly, run a tensor-args slice
(`ttnn.slice(t, start_tensor, end_tensor)`, TILE, BF16, DRAM interleaved) — e.g. `pytest
tests/ttnn/unit_tests/operations/data_movement/test_slice.py -v -k "tensor_args_device_path"` if that
suite runs on the emulator harness; it now builds a spec with one FIFO DFB + one scratchpad and should
pass `ValidateProgramSpec`. A HEIGHT-sharded RM slice will still stop at the `out_shard` validator
error quoted in D2.
```bash
# asserts on
TT_METAL_LLK_ASSERTS=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_slice.py -v -k 00_1024x2048
# asserts off (DPRINT-compatible)
unset TT_METAL_LLK_ASSERTS
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_slice.py -v -k 00_1024x2048
# shape/dtype/finiteness only, if the golden is ever in doubt
TTNN_GRAPH_OPS_NO_GOLDEN=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_slice.py -v
```
Notes: the case's activation is 1024 rows, above the local conftest's `_EMU_MAX_ROWS = 128`, so it is
**not** tagged `emulator` — run it by file/-k, not with `-m emulator`. It is a `(1,1)` mesh with a DRAM
interleaved 4 MB input, so `graph_case.build_memory_config` has no shard grid to reject. The work split
follows `compute_with_storage_grid_size()`, so the small emulator grid is handled by the factory
(32 output tiles over however many cores exist). If the reader hangs at `RBW`/`WFW` or the writer at
`WFW`, that is a runtime implicit/explicit-credit issue to report to the runtime team (recipe §8.2),
not something to fix by setting `disable_dfb_implicit_sync_for_all`.

---

## H. Definition-of-done checklist (recipe §10) — test path

- [x] Uplifted in place, existing directory/namespace; nothing copied from or into `experimental/quasar/`.
- [x] Factory is `create_program_artifacts`/`ProgramArtifacts`; kernels use `dfb::`/`args::`/`tensor::`; no `CBIndex::c_`, positional `get_arg_val`, or `TensorAccessorArgs`.
- [x] `opt_level` matches legacy (absent → O2 on both DM kernels; no compute kernel).
- [x] Every DFB has valid `data_format_metadata`; kernels read sizes via `get_entry_size()`.
- [x] No sync-free / DM self-loop DFB **on the test path**. Off-path: `tensor_stage` → `Scratchpad`, `in_shard` → `LocalTensorAccessor` (done); `out_shard` single-ended DM producer left for the owner (D2).
- [x] No `disable_dfb_implicit_sync_for_all` / `disable_implicit_sync_for`.
- [x] No borrow-with-offset (no `borrowed_from` on the test path).
- [x] No multicast → nothing to normalize.
- [x] No compute kernel → no `*_init`-per-execute or TEN-4746 pairs to audit.
- [x] No non-zero-init semaphore.
- [ ] BH and WH pass unchanged — **user to run** (§G); test path zero-diff, off-path edits behaviour-preserving per §F.
- [ ] Quasar builds and runs — **user to run** (§G); no Quasar-specific change exists to guard (both edits are unguarded, arch-neutral recipes).
- [x] No DIAG/debug leftovers (the four edits contain no logging or scaffolding).
- [x] Missing core-LLK deps: none for this op (no LLK used).
- [x] This report written; RED-stop conditions checked (top of report); every changed file listed with its reason (§A0).

---

## craq-sim run 2026-09-10

First device run of this uplift, on the Quasar functional simulator (craq-sim). **Result: slice is GREEN
on Quasar on every reachable factory; no source fix was needed — the working tree is byte-identical to
the session-2 state above (`git diff --stat` = the same 4 files, 49+/80−).** The one open item is
unchanged: `SliceRmSharded::out_shard` (D2, owner decision), now confirmed by its exact validator message.

### Environment

- Tree: `/localdev/vsuresh/tt-metal` @ `vsureshTT/quasar_uplift_round_2` (uncommitted session-2 edits);
  host libs built/installed from this tree; kernels JIT-compiled from the working tree
  (`TT_METAL_FORCE_JIT_COMPILE=1`, private cache `TT_METAL_CACHE=/localdev/vsuresh/tt-metal-cache-qsr-slice`).
- Simulator: `TT_METAL_SIMULATOR=/localdev/vsuresh/qsr-sim/libttsim.so`, `soc_descriptor.yaml` arch
  `QUASAR`, **8×4 = 32 functional workers** (smaller than the N150 capture grid), 2 DRAM channels,
  `TT_METAL_SLOW_DISPATCH_MODE=1`. Sim speed ≈ 1–3 kHz; every slice case here finished in **≈ 6–12 s**
  (≈ 7.5 k sim cycles — slice reads only the 64 output tiles, so seq 128/512/1024 cost the same).
- Runs were serialized through the shared lock (`qsr_test`); a private conftest device-lock name
  (`TT_DEVICE_LOCK_PATH=tt_device_slice.lock`) kept other agents' concurrent sim runs from timing out the
  60 s `tt_device_lock`.
- Debug pass (every passing case re-run once): `TT_METAL_WATCHER=5 TT_METAL_WATCHER_DUMP_ALL=1
  TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1`.
  Watcher dumps confirm `k_id 5/6` = slice's `reader_unary_unpad_dims_interleaved_start_id.cpp` /
  `writer_unary_interleaved_start_id.cpp` on all 32 cores; no assert, fault, or stalled waypoint.

### Per-test results

| # | Test / case | Factory reached | Result | PCC | Notes |
|---|---|---|---|---|---|
| 1 | `tests/ops/test_slice.py::test_slice[seq128_last127]` | `SliceTile` | **PASS** (via twin, see helper-op note) | 1.0 | 7.6 k cycles, 6.4 s |
| 1 | `…[seq512_last200]` | `SliceTile` | **PASS** (twin) | 1.0 | 7.3 s |
| 1 | `…[seq1024_last1023]` | `SliceTile` | **PASS** (twin) | 1.0 | 7.2 s |
| 1 | same file, **as checked in** | — | **FAIL before slice runs** | — | `ttnn.from_torch(device=mesh, layout=TILE)` → on-device `ttnn::tilize` (legacy `TilizeDeviceOperation`, `Program(ProgramDescriptor)` → `CreateDataMovementKernel`) → `TT_FATAL kernel.hpp:450: DataMovementKernel is not supported on Quasar. Use QuasarDataMovementKernel instead.` Helper op, not slice. |
| 2 | `tests/graph_ops/test_slice.py::test_slice[00_1024x2048_bf16_int-dram]` | `SliceTile` | **PASS** (twin) | 1.0 | 7.7 k cycles, 7.2 s — did **not** need the long timeout. As checked in it hits the same `from_torch` tilize failure through `graph_case.build_tensor`. |
| 3a | tensor-args tiled slice, `[1,1,64,64]` bf16 TILE DRAM, start/end `uint32` device tensors, `slice_dim=2, num_devices=2` (scratch script) | `SliceTileTensorArgs` (**D1 Scratchpad conversion**) | **PASS** | 1.0 | Passes `ValidateProgramSpec` (1 FIFO DFB + 1 scratchpad); the `uint32` start/end tensors are tensor parameters, not DFBs, so the no-uint32-format rule does not fire. |
| 3b | RM HEIGHT-sharded slice, `[1,1,64,16]` bf16 RM L1 on 2 cores, out `[1,1,48,16]` height-sharded (scratch script) | `SliceRmSharded` (**D2**) | **FAIL — expected, stopped here** | — | `TT_FATAL program_spec.cpp:1500: DataflowBuffer 'out_shard' is self-looped by data-movement kernel 'reader' (bound as both PRODUCER and CONSUMER). Self-loop DFBs are not supported for data-movement kernels on Gen2 architectures. Consider using a scratchpad or LocalTensorAccessor instead.` — the validator now names `out_shard`, not `in_shard`, i.e. the D2 `in_shard`→`LocalTensorAccessor` conversion cleared its own error as predicted. |
| — | LLK-asserts + watcher re-run of rows 1 (×3), 2, 3a | | **PASS** (5/5) | 1.0 | no watcher findings |

`SliceRm` / `SliceRmStride` (D3) were not exercised — no Quasar model path reaches them and they were
out of this run's scope. No case skipped for grid size: `split_work_to_cores` adapted to the 32-core sim
grid (2 output tiles per core for the model shapes).

### Fixes applied this run

**None in the op.** Every symptom that fired was either outside the op (helper `tilize`) or the known
owner-decision blocker (`out_shard`). Test-side workaround only, in **temp copies (deleted)**
`tests/ops/test_tmp_slice_qsr.py` and `tests/graph_ops/test_tmp_slice_qsr.py`: tilize on host
(`ttnn.from_torch(x, dtype=bf16, layout=TILE, mesh_mapper=replicate)`) then
`ttnn.to_device(t, mesh, memory_config=DRAM)`; the graph_ops twin monkeypatched `graph_case.build_tensor`
the same way. Readback (`ttnn.to_torch`) untilizes on host and needed nothing.

### Open blockers

| Blocker | Symptom | Where | Owner |
|---|---|---|---|
| `from_torch(device=…, layout=TILE)` runs the legacy on-device tilize | `kernel.hpp:450 DataMovementKernel is not supported on Quasar` from `ttnn::tilize` → `TilizeDeviceOperation` | `ttnn/core/tensor/py_to_tt_tensor.cpp:134-190` (chooses device tilize), `ttnn/cpp/ttnn/operations/data_movement/tilize/` | tilize op owner / tensor-construction owner. **Affects every Quasar test that builds a TILE tensor with `device=`**, incl. both checked-in slice tests and `graph_case.build_tensor`. Not a slice issue. |
| `SliceRmSharded::out_shard` single-ended DM producer self-loop | `program_spec.cpp:1500` message quoted in row 3b | `device/slice_program_factory_rm_sharded.cpp` (`.dfb_bindings` PRODUCER+CONSUMER on the reader, `borrowed_from = names.output`), `device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp` (`reserve_back`/`push_back` on `dfb_out`) | slice / Metal-2.0 API owner — decision recorded in D2 (write the output shard directly through a `LocalTensorAccessor` over `tensor::output`, dropping the two credit calls). Off the model path. |
| Quasar rejects Bfp8_b / UInt16 / UInt32 DFB formats | not hit (all cases bf16) | format layer | model-level dtype decision (D6) |

### Reproduce

```bash
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh slice
export TT_DEVICE_LOCK_PATH=tt_device_slice.lock TT_DEVICE_LOCK_TIMEOUT=7200
# (1) as checked in — fails in from_torch's device tilize, before slice:
qsr_test timeout 2400 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/ops/test_slice.py -k seq128_last127 -v -s
# (1)/(2) with the host-tilize twin: copy the test file, replace U.to_tt(...) (or patch graph_case.build_tensor) with
#   t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh))
#   t = ttnn.to_device(t, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
qsr_test timeout 2400 ./python_env/bin/python -m pytest <twin of tests/ops/test_slice.py> -v -s
qsr_test timeout 2400 ./python_env/bin/python -m pytest <twin of tests/graph_ops/test_slice.py> -v -s
# (3a) tensor-args factory: TILE bf16 [1,1,64,64] on device (host-tilized), start/end = from_torch(torch.tensor([0,0,32,0]/[1,1,64,64]), device=mesh),
#      ttnn.slice(t, st, en, slice_dim=2, num_devices=2)  -> PCC 1.0
# (3b) RM sharded: [1,1,64,16] bf16 ROW_MAJOR, HEIGHT_SHARDED L1 over cores (0,0)-(1,0) shard (32,16);
#      ttnn.slice(t, (0,0,0,0), (1,1,48,16), memory_config=<height-sharded (24,16)>) -> out_shard validator TT_FATAL
# debug pass: prepend TT_METAL_WATCHER=5 TT_METAL_WATCHER_DUMP_ALL=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1
```
Logs: `/tmp/claude-1211407858/-localdev-vsuresh-tt-metal/47c32300-1ca5-43ee-a917-81ae38742e1d/scratchpad/qsr_slice/` (`t1_*.log`, `t2_graph_ops*.log`, `t3a_*.log`, `t3b_rm_sharded.log`, `watcher_t*.log`, driver scripts `tensor_args_small.py`, `rm_sharded_small.py`).

### Definition-of-done delta (recipe §10)

- [x] Quasar builds and runs — confirmed on craq-sim for `SliceTile` (model path) and `SliceTileTensorArgs` (D1); no Quasar-specific change exists, so nothing to `ARCH_QUASAR`-guard.
- [ ] BH and WH parity — still user-run (§G); the tree is unchanged by this run.
