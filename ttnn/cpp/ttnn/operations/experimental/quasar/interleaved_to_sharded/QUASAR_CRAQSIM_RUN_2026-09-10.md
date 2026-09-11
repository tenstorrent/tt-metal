# Quasar craq-sim run — `ttnn.experimental.quasar.interleaved_to_sharded`

**Date:** 2026-09-10
**Op:** `ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded` (namespace `ttnn::prim::qsr`,
single Metal 2.0 factory `create_program_artifacts`, Python `ttnn.experimental.quasar.interleaved_to_sharded`)
**Verdict:** GREEN on the simulator for every llama32_1b_quasar shape. **No source edit to the op was needed.**
This file is the only artifact (uncommitted, delete before merge).

---

## 1. Environment

| Item | Value |
|---|---|
| Repo / branch | `/localdev/vsuresh/tt-metal`, `vsureshTT/quasar_uplift_round_2` (HEAD `5c482a80f26`) |
| Host libs | built + installed from this tree (`build_Release/lib/_ttnn.so` == `ttnn/ttnn/_ttnn.so`, 2026-09-10 18:48); no rebuild was needed |
| Simulator | Quasar craq-sim `/localdev/vsuresh/qsr-sim/libttsim.so` (2026-09-10 18:27), `soc_descriptor.yaml`: 32 functional workers (compute grid **8x4**), 2 DRAM channels, 4 MiB L1/core; ~1–2 kHz |
| Env (`source /localdev/vsuresh/qsr-sim/env.sh i2s`) | `TT_METAL_SIMULATOR`, `TT_METAL_SLOW_DISPATCH_MODE=1`, `TT_METAL_FORCE_JIT_COMPILE=1`, private `TT_METAL_CACHE=/localdev/vsuresh/tt-metal-cache-qsr-i2s`, shared-lock `qsr_test` wrapper |
| Debug env (second run of every passing case) | `TT_METAL_WATCHER=5 TT_METAL_WATCHER_DUMP_ALL=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` |
| Quasar HAL constants that pick the reader path | `DRAM_ALIGNMENT=64`, `L1_ALIGNMENT=16` (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:494,499`) |
| Logs | `/tmp/claude-1211407858/-localdev-vsuresh-tt-metal/47c32300-1ca5-43ee-a917-81ae38742e1d/scratchpad/qsr_i2s/` (pytest logs, three watcher logs, the two probe scripts, archived temp test copies) |

## 2. Per-case results

All PASS rows were run twice: plain, and with the debug env above (watcher + LLK asserts + lightweight
kernel asserts). Watcher logs from the debug runs contain **0** assert/fault lines. PCC is against the
torch input (identity golden).

| # | Suite / case | Input → output | Factory path exercised | Cores | Result | PCC |
|---|---|---|---|---|---|---|
| 1 | prototype_ops `rope-cos-sin-32x64` | `[1,1,32,64]` bf16 TILE DRAM-interleaved → HEIGHT_SHARDED L1 shard `[32,64]` | TILE reader + `writer_unary_sharded` (output DFB borrowed on the shard), no compute | 1 | **PASS** | 1.0 |
| 2 | prototype_ops `lm-head-input-32x2048` | `[1,1,32,2048]` → WIDTH_SHARDED shard `[32,2048]` | same | 1 | **PASS** | 1.0 |
| 3 | prototype_ops `kv-width-32x512` | `[1,1,32,512]` → WIDTH_SHARDED shard `[32,512]` | same | 1 | **PASS** | 1.0 |
| 4 | graph_ops `00_1x64_bf16_int-dram` (40 model calls) | `[1,1,1,64]` bf16 TILE → HEIGHT_SHARDED grid `[[0,0,0,0]]` shard `[32,64]` | same; partial shard (1 logical row in a 32-row tile) | 1 | **PASS** | ≥ 0.999 asserted by `graph_case.run_case` (identity golden) + shape/dtype/layout/memory-config/finiteness checks |
| 5 | graph_ops `01_32x2048_bf16_int-dram` (2 model calls) | `[1,1,32,2048]` → WIDTH_SHARDED grid `[[0,0,7,3]]` shard `[32,64]` | same, 32 cores, last-core `padded_offset` path (`num_units_per_shard_width_last` == full width here) | **32 (fits 8x4 exactly — not skipped)** | **PASS** | ≥ 0.999 asserted (as above) |
| 6 | extra: ROW_MAJOR `rm_h32_w64_1core` | `[1,1,32,64]` bf16 RM (128 B rows) → HEIGHT shard `[32,64]` | stick-layout reader, `aligned=true` branch (128 % 64 == 0) | 1 | **PASS** | 1.0 (bit-exact) |
| 7 | extra: ROW_MAJOR `rm_h64_w64_2core` | `[1,1,64,64]` RM → HEIGHT shard `[32,64]` on 2 cores | stick-layout reader, `aligned=true` | 2 | **PASS** | 1.0 (bit-exact) |
| 8 | extra: ROW_MAJOR `rm_h64_w40_2core` | `[1,1,64,40]` RM (80 B rows: 16 B-aligned, not 64 B DRAM-aligned) → HEIGHT shard `[32,40]` on 2 cores | stick-layout reader **`aligned=false` branch**: `Scratchpad` TRID staging + same-core NoC loopback read (`padded_offset_bytes=80`, `80 % 64 != 0` ⇒ unaligned on both cores) | 2 | **PASS** (also with debug env) | 1.0 (bit-exact) |
| 9 | extra: `convert_df` bf16 → **fp32**, `[1,1,32,64]` TILE → HEIGHT shard | reader → `eltwise_copy` compute → writer (input DFB + borrowed output DFB) | 1 | **BLOCKED (simulator)** — process aborted inside craq-sim: `[pid] ERROR: UnimplementedFunctionality: qsr_convert_pack_value: qsr pack in_format=5 out_format=0` (bf16→fp32 packer conversion not modelled). Not an op fault. | — |
| 10 | extra: `convert_df` **fp32 → bf16**, `[1,1,32,64]` TILE → HEIGHT shard | same compute path, supported direction | 1 | **PASS** (LLK asserts on) | 0.9999985 (bf16 rounding, expected) |

**Not run (off the model path):** DRAM-sharded destinations (`writer_unary_sharded_blocks_start_id.cpp`,
`writer_unary_sharded_stick_layout_start_id.cpp`), BLOCK_SHARDED, COL_MAJOR orientation, Bfp8_b/Bfp4_b
(rejected on Quasar at `ValidateProgramSpec` anyway). Rows 4–5 are the only shapes the captured model
issues; rows 1–3 are the prototype suite's single-core equivalents.

### Runs of the tests *as written in the repo*

Both repo tests **FAIL before reaching the op** on the simulator, at input upload:

```
TT_FATAL @ tt_metal/impl/kernels/kernel.hpp:450: MetalContext::instance(context_id_).get_cluster().arch() != ARCH::QUASAR
DataMovementKernel is not supported on Quasar. Use QuasarDataMovementKernel instead.
  raised from ttnn/ttnn/operations/core.py:376 (ttnn.Tensor(... device=..., layout=TILE ...))
```

`op_utils.to_tt` (prototype_ops) and `graph_case.build_tensor` (graph_ops) call
`ttnn.from_torch(..., layout=TILE_LAYOUT, device=mesh)`, which runs the legacy on-device tilize. The
temp copies below tilize on host and `ttnn.to_device` instead; with that single change the op passes.
Log of the as-written failure: `proto_rope.log`.

## 3. Fixes

**Inside the op: none.** `git diff -- ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded`
is empty. No rebuild was performed; kernels were JIT-compiled from the working tree.

Test-harness adjustments — made in **temporary copies only** (both deleted after the run; archived in
the scratch dir as `archived_test_tmp_i2s_proto_qsr.py` / `archived_test_tmp_i2s_qsr.py`):

| Temp file (deleted) | Change | Reason |
|---|---|---|
| `models/experimental/llama32_1b_quasar/tests/prototype_ops/test_tmp_i2s_proto_qsr.py` | `from_torch(layout=TILE)` on host + `ttnn.to_device(..., DRAM_MEMORY_CONFIG)` instead of `U.to_tt`; prints the PCC | on-device tilize is `DataMovementKernel`-based → fatal on Quasar |
| `models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_i2s_qsr.py` | `_OP = ttnn.experimental.quasar.interleaved_to_sharded`; imports the generated `CASES` unchanged; wraps `G.build_tensor` so interleaved inputs are host-tilized then `to_device` | same; CASES already bf16 and both grids fit 8x4, so no dtype/grid edits were needed |

## 4. Open items / blockers (none block the model shapes)

| # | Item | Symptom / location | Owner |
|---|---|---|---|
| 1 | Repo tests cannot run on Quasar as written | `op_utils.to_tt` (`tests/prototype_ops/op_utils.py`, `from_torch(... device=...)`) and `graph_case.build_tensor` (`tests/graph_ops/graph_case.py`) → `kernel.hpp:450` fatal in the legacy tilize. Fix belongs in the harness (host tilize when arch is Quasar), not the op. | llama32_1b_quasar test-suite owner |
| 2 | Simulator gap: bf16→fp32 pack conversion | `UnimplementedFunctionality: qsr_convert_pack_value: qsr pack in_format=5 out_format=0` (craq-sim aborts the process). Blocks `convert_df` bf16→fp32 only; fp32→bf16 passes. Model does not use `convert_df`. | craq-sim team |
| 3 | Factory sets `disable_dfb_implicit_sync_for_all=true` on both DM kernels (`interleaved_to_sharded_program_factory.cpp`, reader and writer `hw_config`) | `quasar_porting.md` §7 says the implicit-sync opt-out is a retired bring-up workaround and should not be used; no symptom fired here (explicit `reserve_back`/`push_back` + `wait_front`/`pop_front` are consistent), so per the reactive-fix rule it was left alone. Worth removing in a follow-up and re-running rows 1–8. | op owner (with runtime team if a double-count appears) |
| 4 | ROW_MAJOR unaligned path uses a same-core NoC loopback read (`reader_unary_stick_layout_...cpp`, `noc.async_read<TXN_ID>(self_ep, cb_in0, ..., {.noc_x=my_noc_x, .noc_y=my_noc_y, .addr=scratch_l1_base+...})`) | Recipe §6 warns the *emulator* (ZEBU) can spin on `can_post` or drop such reads. On craq-sim it is bit-exact (row 8), so no change; needs confirmation on ZEBU before the RM path is declared GREEN there. Off the model path. | op owner |
| 5 | `eltwise_copy` compute `KernelSpec` has no explicit `opt_level` → Metal 2.0 default O2 (legacy was O3) | perf-only, no correctness symptom; noted from the mainline audit | op owner |

## 5. Repro commands

Every command: `cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh i2s` first.
`S` = the scratch dir in §1. Recreate the temp copies from the archived files before the pytest lines.

```bash
# 1–3 prototype (host-tilize temp copy)
cp $S/archived_test_tmp_i2s_proto_qsr.py models/experimental/llama32_1b_quasar/tests/prototype_ops/test_tmp_i2s_proto_qsr.py
qsr_test timeout 2400 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/prototype_ops/test_tmp_i2s_proto_qsr.py -v -s

# 4–5 graph_ops (quasar _OP + host-tilize temp copy)
cp $S/archived_test_tmp_i2s_qsr.py models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_i2s_qsr.py
qsr_test timeout 2400 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_i2s_qsr.py -k 00_1x64 -v -s
qsr_test timeout 2400 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_i2s_qsr.py -k 01_32x2048 -v -s

# 6–8 ROW_MAJOR height-sharded (aligned + unaligned/scratch paths)
qsr_test timeout 1200 ./python_env/bin/python $S/rm_height_sharded.py rm_h32_w64_1core
qsr_test timeout 1200 ./python_env/bin/python $S/rm_height_sharded.py rm_h64_w64_2core
qsr_test timeout 1200 ./python_env/bin/python $S/rm_height_sharded.py rm_h64_w40_2core

# 9–10 convert_df probes
qsr_test timeout 1200 ./python_env/bin/python $S/convert_df.py                 # bf16->fp32: sim UnimplementedFunctionality
qsr_test timeout 1200 ./python_env/bin/python $S/convert_df_f32_to_bf16.py     # fp32->bf16: PASS

# debug-env variant (prefix any of the above)
TT_METAL_WATCHER=5 TT_METAL_WATCHER_DUMP_ALL=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 \
TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1 qsr_test timeout 2400 ...

# as-written repo test (fails in the harness's on-device tilize, before the op)
qsr_test timeout 2400 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/prototype_ops/test_interleaved_to_sharded.py -k rope-cos-sin-32x64 -v -s
```

---

## craq-sim re-run 2026-09-11

Same tree, HEAD `452fd8990cd` (merge of origin/vsureshTT/quasar_uplift_round_2), host libs `build_Release/lib/_ttnn.so` == `ttnn/ttnn/_ttnn.so` (2026-09-11 19:55, newer than every file under the op). Env `source /localdev/vsuresh/qsr-sim/env.sh i2s_rtl` (private `TT_METAL_CACHE=/localdev/vsuresh/tt-metal-cache-qsr-i2s_rtl`, JIT forced), one device session per pytest/script invocation, run via `qsr_test`. Golden = identity vs. the torch input. **No source edit; the op directory is untouched.** Temp harnesses (recreated from the description in §3, deleted after the run; archived in the scratch dir `.../scratchpad/qsr_i2s_rtl/`): `prototype_ops/test_tmp_i2s_proto_qsr.py`, `graph_ops/test_tmp_i2s_qsr.py`, `rm_height_sharded.py`.

| # | Case | Path | Cores | Result | PCC / exactness | Wall (process, incl. ~5 s device open + JIT) |
|---|---|---|---|---|---|---|
| 1 | prototype `rope-cos-sin-32x64` HEIGHT | TILE reader + borrowed-output writer | 1 | **PASS** | 1.0, bit-exact | 3 cases in one session: 19 s total (op+readback 1.0 s) |
| 2 | prototype `lm-head-input-32x2048` WIDTH | same | 1 | **PASS** | 1.0, bit-exact | (op+readback 0.1 s) |
| 3 | prototype `kv-width-32x512` WIDTH | same | 1 | **PASS** | 1.0, bit-exact | (op+readback 0.0 s) |
| 4 | graph `00_1x64_bf16_int-dram` | partial shard, 1 logical row | 1 | **PASS** | 1.0, bit-exact (floor 0.999 + `run_case` checks) | 15 s |
| 5 | graph `01_32x2048_bf16_int-dram` | 32-core WIDTH, grid `[[0,0,7,3]]` | 32 | **PASS** | 1.0, bit-exact | 40 s |
| 6 | RM `rm_h32_w40_1core` (80 B rows) | stick reader **unaligned**: Scratchpad TRID staging + same-core NoC loopback | 1 | **PASS** | bit-exact, PCC 1.0 | 12 s |
| 7 | RM `rm_h64_w40_2core` (80 B rows) | same, 2 cores (yesterday's row 8) | 2 | **PASS** | bit-exact, PCC 1.0 | 12 s |
| 8 | RM `rm_h32_w64_1core` (128 B rows) | stick reader aligned | 1 | **PASS** | bit-exact, PCC 1.0 | 11 s |
| 9 | RM `rm_h64_w64_2core` (128 B rows) | stick reader aligned | 2 | **PASS** | bit-exact, PCC 1.0 | 11 s |

Row 6 is new (1-core version of the unaligned path, so the ZEBU 1x3 run below has an identical craq-sim reference). `convert_df` bf16→fp32 was not re-run (craq-sim gap, §4 item 2, unchanged).

## RTL emulator run 2026-09-11

**First ZEBU run of this op.** Target `emu-quasar-1x3` (`/localdev/vsuresh/tt-umd-simulators/build/emu-quasar-1x3/soc_descriptor.yaml`: `functional_workers: [0-1]` → **1 worker core**, compute grid reported as 1x1, `dram: [[0-0]]` → 1 DRAM bank, `arch_name: QUASAR`). Env `source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3` (`TT_METAL_CACHE=/localdev/vsuresh/tt-metal-cache-emu-1x3`, `TT_METAL_SLOW_DISPATCH_MODE=1`), one job = one device session run as `qsr_emu timeout 2700 ./python_env/bin/python ...` (exclusive emulator lock shared with other agents; queue waits of 0–158 s are excluded from the wall times). Same temp harnesses as the craq-sim re-run above; golden = identity vs. the torch input. **No source edit; no retry was needed; no job hit the 2700 s timeout.** Launcher logs (`emu_2026-09-11_20-{34,36,39,42,43}_.log`) moved to the scratch dir as `launcher_<tag>_*.log`; none contains `WRP0625E` (slot contention) or `ZTDB0349F`. Each launcher log carries one `ERROR: tt_tensix_neo toolchain files still missing after submodule update` line from the remote checkout step — benign, the run boots and passes regardless.

| # | Case | Input → output | Factory path exercised | Result | PCC / exactness | Wall (job incl. ZEBU boot ≈ 36 s device open) |
|---|---|---|---|---|---|---|
| 1 | prototype `rope-cos-sin-32x64` | `[1,1,32,64]` bf16 TILE DRAM → HEIGHT shard `[32,64]` L1, 1 core | TILE reader + `writer_unary_sharded` (borrowed output DFB), no compute | **PASS** | 1.0, bit-exact | 50 s (pytest 39.3 s, op+readback 2.2 s) |
| 2 | prototype `lm-head-input-32x2048` | `[1,1,32,2048]` → WIDTH shard `[32,2048]`, 1 core (64 tiles) | same | **PASS** | 1.0, bit-exact | 54 s (pytest 43.1 s, op+readback 4.7 s) |
| 3 | graph `00_1x64_bf16_int-dram` | `[1,1,1,64]` TILE → HEIGHT shard grid `[[0,0,0,0]]` `[32,64]` (partial shard, 1 logical row) | same + `graph_case.run_case` shape/dtype/layout/memcfg/finiteness checks | **PASS** | 1.0, bit-exact (floor 0.999) | 60 s (pytest 48.9 s, run_case 1.5 s) |
| 4 | RM unaligned `rm_h32_w40_1core` | `[1,1,32,40]` bf16 ROW_MAJOR (**80 B rows**: 16 B-aligned, not 64 B DRAM-aligned) DRAM → HEIGHT shard `[32,40]` L1, 1 core | stick-layout reader **`aligned=false`**: `Scratchpad` TRID staging + same-core NoC loopback `async_read` (`padded_offset_bytes=80`, `80 % 64 != 0`) — the branch recipe §6 warned may spin / drop on ZEBU | **PASS — no spin, no drop** | bit-exact, PCC 1.0 | 49 s (op+readback 4.6 s) |
| 5 | RM aligned `rm_h32_w64_1core` | `[1,1,32,64]` ROW_MAJOR (128 B rows) → HEIGHT shard `[32,64]`, 1 core | stick-layout reader `aligned=true` | **PASS** | bit-exact, PCC 1.0 | 49 s (op+readback 2.6 s) |

Not runnable on 1x3: graph `01_32x2048` (needs the 8x4 grid — `graph_case._shard_grid_fits` would skip it), the 2-core RM cases; prototype `kv-width-32x512` was skipped for budget (same path as rows 1–2, strictly between them in size).

### Blockers / open items after the RTL run

| # | Item | Status | Owner |
|---|---|---|---|
| 1 | §4 item 4 (RM unaligned same-core NoC loopback on ZEBU) | **Retired** for this op: row 4 above is bit-exact on RTL with the plain env (no watcher). | — |
| 2 | §4 item 1 (repo tests use on-device tilize → `kernel.hpp:450` fatal on Quasar) | Unchanged; applies to the emulator too (same `DataMovementKernel` assert). Both repo tests still need the host-tilize harness change to run on either Quasar target. | llama32_1b_quasar test-suite owner |
| 3 | §4 items 2, 3, 5 | Unchanged (craq-sim bf16→fp32 pack gap; `disable_dfb_implicit_sync_for_all=true` on both DM kernels; compute `opt_level` default O2). Nothing fired on RTL. | as listed in §4 |

**Verdict:** GREEN on both Quasar targets for every 1-core llama32 shape and both ROW_MAJOR reader branches; craq-sim additionally GREEN for the 32-core graph shape. No fix needed in `ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded`.

### Repro (RTL)

```bash
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3
# recreate the temp copies from the scratch archive first (see the craq-sim re-run section)
qsr_emu timeout 2700 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/prototype_ops/test_tmp_i2s_proto_qsr.py -k rope-cos-sin-32x64 -v -s
qsr_emu timeout 2700 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/prototype_ops/test_tmp_i2s_proto_qsr.py -k lm-head-input-32x2048 -v -s
qsr_emu timeout 2700 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_i2s_qsr.py -k 00_1x64 -v -s
qsr_emu timeout 2700 ./python_env/bin/python $S/rm_height_sharded.py rm_h32_w40_1core   # unaligned / loopback
qsr_emu timeout 2700 ./python_env/bin/python $S/rm_height_sharded.py rm_h32_w64_1core   # aligned
```
