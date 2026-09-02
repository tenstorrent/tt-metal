# QUASAR_UPLIFT_REPORT — `ttnn/cpp/ttnn/operations/experimental/paged_cache/`

Date: 2026-09-01
Branch: `vsuresh/quasar-porting-recipe` (based on `origin/main`)
Recipe executed: `docs/source/ttnn/ttnn/ai/quasar_porting.md` + `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/quasar_audit.md` (Quasar-uplift gate), with the Metal-2.0-or-not gate per `quasar_porting.md` §1 and the RED-stop conditions.

This report is left **uncommitted** for review and should be **deleted before merge** (per the recipe's deliverable rules).

---

## Scope: what the target tests exercise

| Graph-trace test | ttnn op | prim | Device op | Program factory hit by the test case |
|---|---|---|---|---|
| `models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_fill_cache.py` | `ttnn.experimental.paged_fill_cache` | `ttnn::prim::paged_fill_cache` (`paged_cache.cpp:83`) | `PagedFillCacheDeviceOperation` (`device/fill_cache/`) | `PagedFillCacheProgramFactory` / `PagedFillCacheMeshWorkloadFactory` (both wrap `build_paged_fill_cache_descriptor`) |
| `models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_update_cache.py` | `ttnn.experimental.paged_update_cache` | `ttnn::prim::paged_update_cache` (`paged_cache.cpp:28`) | `PagedUpdateCacheDeviceOperation` (`device/update_cache/`) | `PagedUpdateCacheProgramFactory` / `PagedUpdateCacheMeshWorkloadFactory` (index-tensor + interleaved INT32 page-table path) |
| *(not exercised by these tests, but in-directory — audited because the directory is owned as a unit)* | `ttnn.experimental.paged_fused_update_cache` | `ttnn::prim::paged_fused_update_cache` (`paged_cache.cpp:55`) | `PagedFusedUpdateCacheDeviceOperation` (`device/fused_update_cache/`) | `PagedTiledFusedUpdateCacheProgramFactory` / `PagedRowMajorFusedUpdateCacheProgramFactory` (+ their MeshWorkload wrappers) |

Test-case specifics: `paged_fill_cache` case = BFLOAT8_B TILE interleaved-DRAM cache `[128,8,32,64]` + input `[1,8,512,64]` + INT32 RM page table `[1,16]`, `batch_idx=0`. `paged_update_cache` case = same cache, BFLOAT16 height-sharded L1 input `[1,1,8,64]`, INT32 `update_idxs_tensor` `[1]`, INT32 interleaved RM `page_table` `[1,128]` (so the **interleaved / uint32-entry** page-table branch of the kernels, not the sharded uint16 branch — see `paged_update_cache_device_operation.cpp:153-157`).

---

## Verdict summary

| Device op / factory | Status | Reason |
|---|---|---|
| `PagedFillCacheDeviceOperation` (both factories) | **RED** | **Not Metal 2.0 on Gen1 yet** — factory is `create_descriptor` → `ProgramDescriptor` |
| `PagedUpdateCacheDeviceOperation` (both factories) | **RED** | **Not Metal 2.0 on Gen1 yet** — factory is `create_descriptor` → `ProgramDescriptor` |
| `PagedFusedUpdateCacheDeviceOperation` (all four factories) | **RED** | **Not Metal 2.0 on Gen1 yet** — factories are `create_descriptor` → `ProgramDescriptor` |

Per `quasar_porting.md` "RED status — STOP the uplift": *"Not Metal 2.0 on Gen1 yet — factory still `create_descriptor`/`ProgramDescriptor`. Do the Metal 2.0 port first."* The RED result is the audit succeeding — it stops a premature uplift. **No Quasar uplift was performed and no source file was modified.**

---

## Gate evidence (per op)

The gate (`quasar_porting.md` §1) requires BOTH: (a) host factory on `create_program_artifacts` → `ProgramArtifacts` with `dfb::`/`args::`/`tensor::`/`scratch::` bindings, and (b) kernels on the Metal 2.0 binding surface (named args via `experimental/kernel_args.h`, `DataflowBuffer` objects, `TensorAccessor(tensor::…)`), not CB-index/positional-arg idioms.

### 1. `paged_fill_cache` — RED

- Host: `device/fill_cache/paged_fill_cache_program_factory.cpp:74-76` — `ProgramDescriptor build_paged_fill_cache_descriptor(...)`; `:340` `PagedFillCacheProgramFactory::create_descriptor`, `:348` mesh variant. Header `paged_fill_cache_program_factory.hpp:17,37` declares `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`. No `create_program_artifacts` / `ProgramArtifacts` / `KernelSpec` / `DataflowBufferSpec` anywhere in the directory.
- Kernels (`reader_fill_cache_interleaved.cpp`, `writer_fill_cache_interleaved.cpp`): positional `get_arg_val<uint32_t>(0..3)` (reader `:18-21`), CB indices via `get_compile_time_arg_val` + `CircularBuffer cb_in(cb_id_in)` (`:15,:31`), buffer-address RTA + `TensorAccessorArgs<2>()` (`:27-29`). No `dfb::`/`args::`/`tensor::` tokens, no `kernel_args.h`.

### 2. `paged_update_cache` — RED

- Host: `device/update_cache/paged_update_cache_program_factory.cpp:89` — `ProgramDescriptor PagedUpdateCacheProgramFactory::create_descriptor(...)`; `:443` mesh variant; `CBDescriptor`/`SemaphoreDescriptor`/RTA-vector idiom throughout. Header `:17,:35`.
- Kernels (`reader_update_cache_interleaved_start_id.cpp`, `writer_update_cache_interleaved_start_id.cpp`, `compute/update_cache.cpp`): positional `get_arg_val<uint32_t>(0..5)` (reader `:16-21`), CB ids from CTAs (`:23-46`), address-RTA + `TensorAccessorArgs<19>()` chain (`:48-50,:69`), `CircularBuffer` wrappers by index (`:54-57`).

### 3. `paged_fused_update_cache` — RED

- Host: `device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:79` and `paged_row_major_fused_update_cache_program_factory.cpp:79` — `ProgramDescriptor …::create_descriptor(...)`; mesh variants at `:539` / `:538`. Headers `:29,:55` in each.
- Kernels (`reader/writer_paged_fused_update_cache_interleaved_start_id.cpp`, `reader/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp`, `compute/paged_fused_update_cache.cpp`, `compute/paged_row_major_fused_update_cache.cpp`): same Metal 1.x binding model (positional `get_arg_val`, CTA-indexed CBs, `TensorAccessorArgs<N>` + address RTAs).

**Nuance worth recording for the downstream port team:** the kernels are already **Device 2.0** (they include `api/dataflow/*` and use the `Noc`, `CircularBuffer`, `TensorAccessor`, `CoreLocalMem`, `Semaphore<>` wrapper objects — no legacy `noc_async_*` free functions, no top-level `dataflow_api.h`). So the Device-2.0 prerequisite of `metal2_audit.md` looks satisfied; what is missing is the **Metal 2.0 layer itself** (ProgramArtifacts factory + named-binding kernels). That makes the eventual Metal 2.0 port the standard mechanical shape rather than a two-migration job. (Note: `api/dataflow/circular_buffer.h` + CB-index naming is exactly the idiom the Metal 2.0 port replaces with `DataflowBuffer`/`dfb::` — flagged by the canonical audit as the tell-tale of a pre-M2 kernel.)

---

## Files changed

**None.** Zero source modifications. The only file added is this report (uncommitted, to be deleted before merge).

The op's directory and namespace were never in question — nothing tempted a move or rename, nothing was copied from or compared against `ttnn/cpp/ttnn/operations/experimental/quasar/` (that tree was not read).

---

## §7–§8 gotchas: applied vs considered

**Applied: none.** The M2 gate REDs the ops before the uplift begins, and the recipe's §7–§8 fixes are reactive (no device run happened in this session), so no fix was legitimate to apply.

Considered — informational pre-scan done anyway so the eventual port/uplift inherits the data:

| Check | Finding |
|---|---|
| Non-zero-init semaphores (`quasar_audit.md` check 2) | **Clean.** All three semaphore sites are `initial_value = 0`: `update_cache/paged_update_cache_program_factory.cpp:247-252`, `fused_update_cache/paged_row_major_…:256-261`, `fused_update_cache/paged_tiled_…:260-265`. No Quasar blocker here. |
| DM self-loop / sync-free CB debt (`quasar_audit.md` check 1) | Not formally classified (moot pre-M2 — the CB→DFB inventory is the M2 port's job), but no reader-and-writer-on-one-kernel CB pattern was observed in the kernels read; CBs are conventional reader→compute→writer FIFOs plus small DM staging CBs for index/page-table sticks (the staging CBs are `Scratchpad` candidates at uplift time). |
| `fifo_page_size` / `get_local_cb_interface` (§8.3 value inflation) | **Absent** — kernels use `CircularBuffer::get_tile_size()`. |
| `evil_set_read_ptr`/`evil_set_write_ptr` ring rewind (§7 RED trigger) | **Absent.** |
| `disable_dfb_implicit_sync_*` (§7) | **Absent.** |
| `MEM_ZEROS_BASE` (§8.1) | **Absent.** |
| uint16/uint32 device formats (§7: Quasar has Int32, no uint16/uint32) | **Present — future uplift item, not a change now.** `fill_cache/paged_fill_cache_program_factory.cpp:127,218` declares a CB with `tt::DataFormat::UInt32` (batch-idx/page-table staging), and the fused/update kernels parse the sharded page table as `uint16_t` (e.g. `reader_paged_fused_update_cache_interleaved_start_id.cpp:122-131`; gated host-side by `dtype()==UINT16` for *sharded* page tables only — the interleaved path the target tests use is INT32/uint32). These CBs are DM staging buffers (never a compute unpack operand), so they may be benign on Quasar, but the format declarations must be re-derived when the DFB `data_format_metadata` is written during the M2 port — decide then whether an `ARCH_QUASAR` Int32 mapping is needed. |
| `compute_kernel_hw_startup` placement (§7) | Compute kernels enter through `ttnn/kernel_lib/{tilize,untilize}_helpers.hpp`; startup discipline to be verified at M2-port time. |
| Custom `compute_program_hash` (§8.3 stale-kernel row) | Both update ops have hash-relevant comments around `…device_operation.cpp:324/:382` (`share_cache` affects program structure/semaphore setup) — carry any custom hash through the M2 port verbatim. |

---

## Deferred / follow-up items

1. **Metal 2.0 port of all three device ops** — the blocking prerequisite. Route to the Metal 2.0 port track: run `ai/audit/metal2_audit.md` (→ `METAL2_PREPORT_AUDIT.md` / `METAL2_PORT_BRIEF.md`), then `ai/port/metal2_port.md`, then the post-port passes, and only then re-run this Quasar-uplift audit. Note for the M2 auditor: Device 2.0 already done; five ProgramDescriptor factories + MeshWorkload wrappers with `override_runtime_arguments` patch paths (readiness sheet: one row per factory); 11 kernels, none shared outside the directory.
2. **UInt32/uint16 format usage** (table above) — re-examine during the M2 port's `data_format_metadata` assignment; possible `ARCH_QUASAR` Int32 substitution or none needed (DM-only staging).
3. Nothing required an out-of-op-directory edit in this session; no shared-file changes were deferred because none were attempted.

---

## WH/BH parity claim (argued structurally — no device run in this session)

**The diff against the base branch is empty for every source file in this directory** (the only addition is this report, which is not compiled or shipped). A zero diff cannot change WH/BH behavior; parity is exact by construction. No `ARCH_QUASAR` guards were added because no uplift was performed.

## Test commands (user runs all builds/tests — recipe §9)

BH/WH parity (mainline suites for these ops; run on both archs, expect identical results to base branch):

```bash
# update_cache + fill_cache (covers paged + non-paged decode/prefill variants)
pytest tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py
# fused_update_cache
pytest tests/ttnn/unit_tests/operations/transformers/test_paged_fused_update_cache.py
# flexible-geometry / mask coverage in the same family
pytest tests/ttnn/unit_tests/operations/transformers/test_paged_cache_flexible_geometry.py
pytest tests/ttnn/unit_tests/operations/transformers/test_paged_cache_mask.py
```

Quasar graph-trace tests (the targets of this task; on the Quasar emulator env, expected to require the Metal 2.0 port first):

```bash
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_fill_cache.py
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_update_cache.py
```

If any kernel is ever edited before re-testing: `TT_METAL_FORCE_JIT_COMPILE=1` (and purge `~/.cache/tt-metal-cache` between eras).
