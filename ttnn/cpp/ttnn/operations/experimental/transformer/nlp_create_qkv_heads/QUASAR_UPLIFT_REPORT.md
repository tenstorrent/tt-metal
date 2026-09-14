# Quasar Uplift Report: `nlp_create_qkv_heads` (Metal 2.0 port, both factories)

**Date:** 2026-09-14 (supersedes the 2026-09-10 RED audit written against the legacy op)
**Branch:** `vsureshTT/quasar_uplift_round_2`, HEAD `56676348b9a` = the Metal 2.0 port of this op (draft PR #56534,
3 commits: two pre-port kernel guard fixes + the port of both factories, 151/151 on WH) cherry-picked on top.
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/` (+ the port's shared-kernel fork
`ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp`)
**Recipe:** `quasar_porting.md` + `ai/audit/quasar_audit.md`, `ai/audit/cb_dfb_quasar_audit_helper.md`,
`ai/post_port/style/sync_free_dfbs.md`, `ai/post_port/semantic/{dm_self_loop_dfbs,gen2_hardware_configs}.md`.
**Python entry:** `ttnn.experimental.nlp_create_qkv_heads`.

---

## Status

| factory | static uplift | craq-sim | ZEBU RTL (1x3 / 2x3) | status |
|---|---|---|---|---|
| `Interleaved` (reader + writer, + `transpose_wh_metal2` compute when `transpose_k_heads`) | 1 fix: arch-selected `ComputeGen2Config` beside the untouched Gen1 config | 4/4 cases + captured graph_ops case, PCC 1.0, also with LLK asserts and Watcher | 4/4 cases PCC 1.0 | **GREEN** |
| `Sharded` (one DM kernel source, 2-4 `KernelSpec`s) | 1 fix: the three borrowed output DFBs → tensor bindings + `LocalTensorAccessor` (removes two Gen2-rejected DM self-loops and one sync-free borrowed DFB) | 13/13 cases PCC 1.0 (1/2/8/32-core grids, fused / separate KV, tied KV, zero-Q-head instance), also with LLK asserts and Watcher | 4/4 cases PCC 1.0 (incl. the 2-core `q_only_cores` shape) | **GREEN — one item for the op owner to ratify (see §3.2)** |

Both factories were confirmed Metal 2.0 before the uplift: `create_program_artifacts` → `ProgramArtifacts`; `dfb::` /
`args::` / `tensor::` bindings; kernels on the device-2.0 API (`Noc`, `DataflowBuffer`, `TensorAccessor(tensor::…)`,
`get_arg(args::…)`, `get_vararg`, `get_tile_size()`); no `cb_*`, `get_arg_val`, `TensorAccessorArgs`,
`get_local_cb_interface`, `fifo_*`, `evil_*`, `CBIndex`. No RED-stop condition of `quasar_porting.md` §1 fires (§4).

No hang, no watcher fault, no LLK assert and no sim-only signature (`qsr_tile_counter_check_error`, pack-untilize
landing artifacts) was hit anywhere; every case that was tried on both targets agrees (PCC 1.0 on both).

---

## 1. Files changed (all under the op directory; `transpose_wh_metal2.cpp` needed no change)

| file | change | why / guard |
|---|---|---|
| `device/nlp_create_qkv_heads_program_factory.cpp` — `Interleaved::create_program_artifacts` | After the untouched `ComputeGen1Config compute_hw{…}` + `unpack_modes.emplace(K_IN, UnpackToSrc)` block: `ComputeHardwareConfig compute_hw_config = compute_hw; if (arch == tt::ARCH::QUASAR) compute_hw_config = ComputeGen2Config{.enable_32_bit_dest = …, .unpack_modes = …};` and `.hw_config = compute_hw_config` in `make_compute`. `TODO(#52269)` marker placed. | `gen2_hardware_configs.md` shape 4 (hand-written Gen1 config). Without it `ValidateProgramSpec` (`program_spec.cpp:~905`) fails on Quasar for every `transpose_k_heads=True` call: *"targets Gen2 (Quasar) but its ComputeHardwareConfig holds a ComputeGen1Config"*. **Arch-checked**: WH/BH take the byte-identical Gen1 config; `arch` was already a local in this factory. `bfp_pack_precision_mode` dropped (Gen1-only), `enable_2x_src_register` never set. Same idiom as the `experimental/paged_cache` factories on this branch. |
| same file — `Sharded::create_program_artifacts` + `build_sharded_core_args` | The three `DataflowBufferSpec`s `q_out` / `k_out` / `v_out` (`borrowed_from = Q/K/V`) and their `DFBBinding`s removed; each `KernelSpec` instance gets `TensorBinding{Q, "q_out"}` and, on the kv specs, `TensorBinding{K or V, "kv_out"}` (K for the reader-config instance, V for the writer-config instance — the same split the DFBs had). The `num_kv_tiles` runtime arg (it only sized the former `k_out`/`v_out` `reserve_back`/`push_back`) and the `k_num_tiles` builder field are removed; the now-unused `data_format` / `single_tile_size` / `*_num_tiles` locals go with them. `ProgramSpec.dataflow_buffers` is gone (the factory has no DFB left). | `sync_free_dfbs.md` (borrowed, sync-free → `LocalTensorAccessor`) for `q_out`; `dm_self_loop_dfbs.md` translation for `k_out` / `v_out` (see §3.2 for why this one needs owner ratification). Without it `ValidateProgramSpec` (`program_spec.cpp:1495`) fails at the first sharded dispatch on Quasar: *"DataflowBuffer 'k_out' is self-looped by data-movement kernel 'reader_kv' (bound as both PRODUCER and CONSUMER). Self-loop DFBs are not supported for data-movement kernels on Gen2"*. **Unguarded, behaviour-identical** on WH/BH (§6). |
| `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp` | `DataflowBuffer dfb_q_out(dfb::q_out)` + `get_write_ptr()` → `LocalTensorAccessor<uint8_t> q_out(tensor::q_out)` + `get_bank_base_address()`; in the `READ_KV_HEADS` block `DataflowBuffer dfb_kv_out(dfb::kv_out)` + `reserve_back(num_kv_tiles)` / `get_write_ptr()` / trailing `push_back(num_kv_tiles)` → `LocalTensorAccessor<uint8_t> kv_out(tensor::kv_out)` + `get_bank_base_address()`; `num_kv_tiles` arg read dropped; include `api/dataflow/dataflow_buffer.h` → `api/tensor/local_tensor_accessor.h`. Every NoC read, address walk, coordinate-table lookup and barrier is untouched. | Kernel side of the row above. `T = uint8_t` because the kernel never dereferences the region, only hands the address to the NoC (recipe table; same as `slice` / `concat` on this branch). |
| `QUASAR_UPLIFT_REPORT.md` | this file (rewritten for the ported state) | recipe deliverable — **uncommitted; delete before merge** |

Not changed: `transpose_wh_metal2.cpp` (audited, §2), the three `METAL2_PORT_*.md` / `METAL2_PREPORT_AUDIT.md`,
the device-operation / nanobind files, anything outside the op. Nothing copied into `experimental/quasar/`, no
namespace or directory change, no runtime / LLK / CMake / test edit (temp test copies were deleted, §7).

---

## 2. Quasar-uplift audit findings (ported state)

### 2.1 `quasar_audit.md` check 1 — device-side CB/DFB classification (Metal 2.0 DFBs as ported)

| DFB (factory) | binders | class / shape | Gen2 verdict | action |
|---|---|---|---|---|
| `qv` (Interleaved) | reader PRODUCER → writer CONSUMER, 4-tile explicit FIFO | 1, canonical DM→DM | portable | none (kernels peek `get_write_ptr()`/`get_read_ptr()` into a `CoreLocalMem` NoC operand; `Noc::get_src/dst_ptr` routes `LOCAL_L1` through `l1_cached_view()`, so the Quasar uncached alias is mapped back — same shape as the already-uplifted `paged_cache` / `untilize` DM kernels, verified on sim + RTL here) |
| `k_in` (Interleaved, `transpose_k_heads`) | reader PRODUCER → compute CONSUMER | 1 | portable | none |
| `k_out` (Interleaved, `transpose_k_heads`) | compute PRODUCER → writer CONSUMER | 1 | portable | none |
| `q_out` (Sharded, `borrowed_from = Q`) | reader-config instance PRODUCER, writer-config instance CONSUMER; **zero** FIFO calls anywhere, both write `get_write_ptr() + q_offset` | 6, sync-free borrowed (the "1P+1C" labels were cosmetic — on Gen2 a CONSUMER cannot write) | **site** (`sync_free_dfbs.md`, borrowed → `LocalTensorAccessor`) | **converted** (§1) |
| `k_out` / `v_out` (Sharded, `borrowed_from = K` / `V`) | one DM kernel each, PRODUCER **and** CONSUMER (`reserve_back(N)` … `get_write_ptr()` … `push_back(N)`, capacity exactly N, filled once, no consumer) | DM self-loop on borrowed memory (the legacy single-ended producer, mechanically ported) | **Gen2 TT_FATAL** (`program_spec.cpp:1495`) | **converted** (§1) — owner-ratification item, §3.2 |

GATE scans over the op's kernels + the fork: `get_local_cb_interface`, `fifo_*`, `evil_*`, `read_tile_value`,
`get_pointer_to_cb_data`, `pages_reservable_at_back`, `pages_available_at_front`, `async_write_zeros`, multicast:
**0 hits**. The `dfb::` handles are never passed to a helper / RAII guard / template (no opaque FIFO calls).
`dfb_run_overrides`: none.

### 2.2 `quasar_audit.md` check 2 — non-zero-init semaphores
No `SemaphoreSpec`, no `Semaphore<>` in the op or the fork. Not a site (nothing to convert to CTAD either).

### 2.3 `gen2_hardware_configs.md` survey
`grep -rn "hw_config\|to_compute_hardware_config\|Gen1Config\|std::get<\|std::get_if<\|holds_alternative"`:
- DM specs (both factories): `ttnn::create_reader/writer_datamovement_config(arch)` — shape 1, nothing to do.
- Interleaved compute: hand-written `ComputeGen1Config` — shape 4 → **fixed** (§1). Float32 path: the port already
  emits `unpack_modes = {{k_in, UnpackToSrc}}` when `enable_32_bit_dest`; copied verbatim into the Gen2 config
  (the `unpack_modes` accessor form was not needed — no `std::get<ComputeGen1Config>` exists).
- No `std::get<ComputeGen1Config>` / `get_if` anywhere.

### 2.4 `transpose_wh_metal2.cpp` (the only compute kernel), `quasar_porting.md` §7
- `compute_kernel_hw_startup(dfb::in, dfb::out)` exactly once at `kernel_main()` start → it runs `llk_pack_hw_configure`
  + `llk_pack_init` + `llk_pack_dest_init` on `dfb::out`; `transpose_init(dfb::in)` once. **One fixed (in, out) pair
  for the whole kernel: no DFB-id switch, so no re-`*_init` and no `pack_init` retarget is needed** (the Quasar
  pack-BFD-baked-at-init rule has no site here; verified by PCC 1.0 on the transpose-K cases on sim and RTL).
- TEN-4746 bare pairs: `wait_front(1)`, `reserve_back(1)` → `transpose_tile` (real UNPACR) → `pack_tile` (real PACR)
  → `push_back(1)`, `pop_front(1)`; the loop body is the only control path (no `if`/`continue`/`break`). Not a site.
- Quasar `transpose_init` / `transpose_tile` have real (non-stub) branches; the only LLK limit is
  `LLK_ASSERT(!enable_unpack_to_dest, "32-bit (unpack-to-dest) transpose not supported on Quasar") // tt-llk#1559`
  → **FLOAT32 × `transpose_k_heads=True` will assert on Quasar** (LLK-team item, not an op edit; bf16 is the model dtype).
- `opt_level` O3 explicit on both compute specs (base-port item, correct; untouched).

### 2.5 Other §7 / §11 items
- Bfp8: `is_supported_quasar()` has no `Bfp8_b`; the op forwards the dtype with no format branch → nothing to guard,
  model-level decision (the llama32 capture and both prototype tests are bf16, so nothing on the Quasar test path is bf8).
- Quasar Int32-only / uint16-uint32: op accepts FLOAT32 / BFLOAT16 / BFLOAT8_B only — N/A.
- Multicast: none in the op (§11 N/A). `in0_mcast_noc_x/y` in the sharded kernel are unicast coordinate tables.
- Implicit sync: not disabled anywhere.
- Borrow-with-offset (`partials_cb_uses_output`-style): none; the former `q_out` byte offset lives in the kernel
  (`q_offset` RTA), now added to `get_bank_base_address()` — expressible, no missing-offset issue.
- 64-bit pointer casts: none left (the port's vararg API removed the legacy `get_arg_addr` cast).
- `LocalTensorAccessor` requires an L1 tensor (`static_assert` on the binding token): the sharded outputs are L1
  (legacy already required L1 for the dynamic CBs, and the Metal 2.0 borrowed DFB is L1-only), so no behaviour change;
  a DRAM-sharded output would now fail at JIT-compile time instead of at CB/DFB creation. Noted, not a regression.

---

## 3. Deferred / follow-up items

### 3.1 Test-harness blockers (outside the op; worked around in temp copies, §7)
`ttnn.from_torch(..., layout=TILE_LAYOUT, device=mesh)` fails on Quasar for bf16 (`tilize` /
`tilize_with_val_padding` factories are not Gen2-ready), so `op_utils.to_tt` and `graph_case.build_tensor` cannot
build the inputs. Temp copies tilize on host and `ttnn.to_device(...)`. Owner: `tilize` op family /
llama32_1b_quasar test utilities. Same finding as the paged_cache / concat / untilize uplifts on this branch.

### 3.2 Owner decision to ratify: Sharded `k_out` / `v_out` fake-FIFO → `LocalTensorAccessor`
- **What the recipe says.** `dm_self_loop_dfbs.md` survey step 5: *"If `borrowed_from` is set, stop and report … it
  would have to be a `LocalTensorAccessor` over the borrowed tensor, and fake-FIFO bookkeeping over borrowed memory is
  a combination nothing in this suite has examined."* The concat uplift on this branch stopped on exactly this shape
  (`concat/QUASAR_UPLIFT_REPORT.md` §3.2). The 2026-09-10 audit of this op called the same buffers (as legacy CBs)
  "NEEDS-DESIGN-DECISION … recommended v1 for the owner to confirm: drop the pair and use `LocalTensorAccessor` over
  `tensor::k`".
- **What was done here, and why.** The invoker asked for the op to be taken through the sim and the RTL emulator with
  blockers fixed in place; without this conversion the Sharded factory cannot be dispatched on Quasar at all. The
  conversion is the recipe's own translation with every stop condition other than step 5 checked and clear: every use
  of the two handles was on the covered list (`reserve_back`, `get_write_ptr`, NoC destination via `CoreLocalMem`,
  `push_back`); the write pointer is captured **before** the only `push_back` and never re-read, so both indices are
  dead (no stride, no wrap, `entry_size` never needed); no `get_entry_size`, no `pages_*`, no `async_write_zeros`, no
  multicast, no helper; no `dfb_run_overrides`; `data_format_metadata` consulted by nothing. The credits synchronised
  nothing on any arch (sole single-threaded toucher, capacity == fill, no consumer), so removing them changes no read,
  write, size, order or barrier — the same bytes land at the same L1 address (a borrowed DFB's base is the shard base).
  Verified: PCC 1.0 on every sharded configuration on craq-sim (13 cases) and RTL (4 cases).
- **What the owner must do.** Either ratify this shape (and, ideally, feed "borrowed DM fake-FIFO with a dead index →
  `LocalTensorAccessor`" back into `dm_self_loop_dfbs.md` so concat's §3.2 can follow), or reject it — the revert is
  the Sharded half of this diff and puts the factory back to the Gen2-rejected self-loop state. The WH suite
  (`tests/tt_eager/.../test_nlp_create_qkv_heads.py`, sharded + program-cache cases) is the parity check; it has not
  been run in this session (user runs it).

### 3.3 LLK
FLOAT32 × `transpose_k_heads=True` asserts on Quasar (`transpose.h`: 32-bit unpack-to-dest transpose unsupported,
tt-llk#1559). Not reachable by the llama32 model (bf16). Flag only.

### 3.4 Model-level
BFLOAT8_B inputs are rejected by the Quasar format layer; nothing to change in the op.

### 3.5 Carried forward from the port report (unchanged, not Quasar-specific)
Sharded Q-loop coordinate refresh sits inside the row-wrap branch; CRTA candidates for the per-node-identical RTAs;
`nlp_create_qkv_heads_boltz` carries a pre-fix copy of the sharded kernel; two Metal 2.0 forks of the transpose body.

---

## 4. RED-stop conditions (`quasar_porting.md` §1) — none fire
- Not Metal 2.0 on Gen1: **no** (both factories `create_program_artifacts`; kernels device-2.0).
- Missing sanctioned Quasar capability: **no** (`ComputeGen2Config`, `TensorBinding` + `LocalTensorAccessor` are
  first-class API; no hand-rolled device interface).
- Construct needing an owner decision: **§3.2 was applied rather than stopped on, at the invoker's direction, and is
  flagged for ratification** — the Sharded factory is GREEN on device but not "GREEN by the letter of the recipe" until
  the owner signs it off.
- Only fix changes WH/BH un-guarded: **no** — the one unguarded change is behaviour-identical by construction (§6).
- Stub/unported LLK on the path: **no** for bf16 (Quasar `transpose_init`/`transpose_tile` are real); 32-bit transpose
  is an asserting LLK gap, flagged (§3.3).

---

## 5. craq-sim run (2026-09-14)

**Environment.** `cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh qkv`
(`TT_METAL_SIMULATOR=/localdev/vsuresh/qsr-sim/libttsim.so` craq-sim; **8x4 = 32 worker cores, 2 DRAM banks**
(`soc_descriptor.yaml`); slow dispatch; `TT_METAL_FORCE_JIT_COMPILE=1`; private kernel cache
`/localdev/vsuresh/tt-metal-cache-qsr-qkv`). Host libs rebuilt once from this tree after the edits (`qsr_rebuild` →
`REBUILD_OK`; only the experimental/transformer unity object recompiled). Every test through
`qsr_test timeout <s> ./python_env/bin/python -m pytest … -x -v -s`, one process at a time. Logs:
`scratchpad/qsr_nlp_create_qkv_heads/s*.log` (plain), `a*.log` (`TT_METAL_LLK_ASSERTS=1
TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1`), `w*.log` (+ `TT_METAL_WATCHER=5 TT_METAL_WATCHER_DUMP_ALL=1
TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1`; `w2_watcher.log` = `generated/watcher/watcher.log`, no error/assert lines).
Wall = whole pytest process incl. device open, JIT of 6-8 kernels and teardown (the op's device time is ~1 s).

**Re-gridding:** none needed. Interleaved cases split work over the device grid (`split_work_to_cores`); the sharded
temp test derives its shard grid from `compute_with_storage_grid_size()` (`num_cores_to_corerangeset(num_kv_heads,
grid, row_wise)`), so the 32-Q-head case used all 32 sim cores. The captured graph_ops case is DRAM-interleaved
(no shard spec to re-grid).

**Test shapes.** Interleaved (prototype temp copy): input `[1,1,seq,3072]` bf16 TILE DRAM, `num_heads=32,
num_kv_heads=8`, head_dim 64 (llama32-1B), `transpose_k_heads` False/True. Sharded (temp test derived from the WH
runner): input `[1,1,32,(q+2·kv)·64]` bf16 TILE, WIDTH_SHARDED L1 over `num_kv_heads` cores (separate KV: Q tensor
`[1,1,32,q·64]` + KV tensor `[1,1,32,2·kv·64]`), outputs HEIGHT_SHARDED L1 (Q over `num_q_heads` cores, K/V over
`num_kv_heads` cores), `transpose_k_heads=False`.

| # | factory / case | result | PCC (Q / K / V) | wall |
|---|---|---|---|---|
| s1 | Interleaved seq32, no transpose | PASS | 1.0 / 1.0 / 1.0 | 16 s |
| s2 | Interleaved seq32, transpose-K (compute kernel) | PASS | 1.0 / 1.0 / 1.0 | 19 s |
| s3 | Interleaved seq128, no transpose (4 cores) | PASS | 1.0 / 1.0 / 1.0 | 16 s |
| s4 | Interleaved seq128, transpose-K | PASS | 1.0 / 1.0 / 1.0 | 18 s |
| s5 | Sharded q2/kv1 fused (1 core: reader inst. 1 Q head + K, writer inst. 1 Q head + V) | PASS | 1.0 / 1.0 / 1.0 | 18 s |
| s6 | Sharded q2/kv1 separate KV tensor | PASS | 1.0 / 1.0 / 1.0 | 17 s |
| s7 | Sharded q1/kv1 fused (writer instance reads **zero** Q heads — the guarded path) | PASS | 1.0 / 1.0 / 1.0 | 17 s |
| s8 | Sharded q1/kv1 separate KV | PASS | 1.0 / 1.0 / 1.0 | 16 s |
| s9 | Sharded q4/kv2 fused (2 cores) | PASS | 1.0 / 1.0 / 1.0 | 16 s |
| s10 | Sharded q4/kv2 separate KV | PASS | 1.0 / 1.0 / 1.0 | 17 s |
| s11 | Sharded q16/kv8 fused (8 KV cores, 16 Q cores → `q_only_cores` WorkUnit active) | PASS | 1.0 / 1.0 / 1.0 | 18 s |
| s12 | Sharded q16/kv8 separate KV | PASS | 1.0 / 1.0 / 1.0 | 17 s |
| s13 | Sharded q32/kv8 fused (32 Q cores = full sim grid) | PASS | 1.0 / 1.0 / 1.0 | 17 s |
| s14 | Sharded q32/kv8 separate KV | PASS | 1.0 / 1.0 / 1.0 | 16 s |
| s15 | Sharded kv_tied fused q2/kv1 | FAIL (test shape, not the op) | op validation `TT_FATAL nlp_create_qkv_heads.cpp:44: Ambiguous kv_tied fused input shape: width 192 is divisible by both 3 (tied) and 4 (untied) sections` — same on every arch | 15 s |
| s15b | Sharded kv_tied fused q4/kv1 (width 320, unambiguous) | PASS | 1.0 / 1.0 / 1.0 | 17 s |
| s16 | Sharded kv_tied separate KV q2/kv1 | PASS | 1.0 / 1.0 / 1.0 | 16 s |
| s17 | `graph_ops/test_nlp_create_qkv_heads.py` case `00_1024x3072_bf16_int-dram` (temp host-tilize copy; 32 blocks over 32 cores) | PASS | 1.0 / 1.0 / 1.0 (out shapes `[1,32,1024,64]`, `[1,8,1024,64]`×2) | 17 s |
| a1-a5 | LLK + lightweight asserts ON: s2, s3, s5, s12, s17 | PASS ×5 | all 1.0 | 15-20 s each |
| w1, w2 | Watcher ON (NoC sanitizer off): s4, s13 | PASS ×2 | all 1.0 | 26 s, 17 s |

Nothing hung or faulted; no sim-only false-failure signature appeared, so no case had to be "taken to RTL" for
disambiguation — the RTL runs below are the planned confirmation set.

---

## 6. RTL emulator run (ZEBU, 2026-09-14)

**Environment.** `cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3|2x3`
(`emu-quasar-1x3` = 1 worker core, `emu-quasar-2x3` = 2 worker cores; slow dispatch; `TT_METAL_FORCE_JIT_COMPILE=1`
added). Each job `qsr_emu timeout 2700 ./python_env/bin/python -m pytest <file> -k <one case> -x -v -s`, one device
session per job, exclusive emulator lock; no job was interrupted, no slot stall (`WRP0625E`) or `ZTDB0349F` occurred.
Launcher logs `emu_2026-09-14_20-{26..36}_.log` moved to `scratchpad/qsr_nlp_create_qkv_heads/`. Wall = whole job
incl. emulator boot ("Waiting for ack msg from remote…") and JIT.

Interleaved shapes here are the minimal ones (temp file `test_tmp_nlp_create_qkv_heads_emu_qsr.py`): input
`[1,1,seq,(q+2·kv)·64]` bf16 DRAM interleaved, head_dim 64. Sharded shapes as in §5.

| # | config | factory / case | result | PCC (Q / K / V) | wall |
|---|---|---|---|---|---|
| e1 | 1x3 | Interleaved q1/kv1 seq32, no transpose | PASS | 1.0 / 1.0 / 1.0 | 60 s |
| e2 | 1x3 | Interleaved q1/kv1 seq32, transpose-K (compute kernel) | PASS | 1.0 / 1.0 / 1.0 | 58 s |
| e3 | 1x3 | Interleaved q2/kv1 seq64 (2 blocks on 1 core), no transpose | PASS | 1.0 / 1.0 / 1.0 | 69 s |
| e4 | 1x3 | Interleaved q2/kv1 seq64, transpose-K | PASS | 1.0 / 1.0 / 1.0 | 69 s |
| e5 | 1x3 | Sharded q2/kv1 fused | SKIP (test guard: needs 2 Q output cores, device has 1) | — | 61 s |
| e6 | 1x3 | Sharded q1/kv1 separate KV (1 core; writer instance zero Q heads) | PASS | 1.0 / 1.0 / 1.0 | 66 s |
| e7 | 1x3 | Sharded q1/kv1 fused | PASS | 1.0 / 1.0 / 1.0 | 61 s |
| e8 | 2x3 | Sharded q2/kv1 fused (2 Q cores, 1 KV core → both kernel instances write Q, `q_only_cores` WorkUnit + `READ_KV_HEADS` split exercised, K- and V-instance `LocalTensorAccessor`s) | PASS | 1.0 / 1.0 / 1.0 | 72 s |
| e9 | 2x3 | Sharded q2/kv1 separate KV tensor | PASS | 1.0 / 1.0 / 1.0 | 71 s |

---

## 7. Open blockers / flags

| item | evidence | owner |
|---|---|---|
| **Ratify the Sharded `k_out`/`v_out` fake-FIFO → `LocalTensorAccessor` conversion** (recipe `dm_self_loop_dfbs.md` step 5 says stop on `borrowed_from`; applied here so the factory can run on Gen2; PCC 1.0 on 17 sharded runs across sim + RTL) | §3.2; `git diff` of the Sharded half of `nlp_create_qkv_heads_program_factory.cpp` + the sharded kernel | op owner (nlp_create_qkv_heads) + recipe owner (`dm_self_loop_dfbs.md`: sanction or forbid the shape; concat §3.2 is the same shape) |
| `from_torch(device=, layout=TILE)` cannot run on Quasar (tilize factories not Gen2-ready) → the repo's Quasar tests for this op fail before the op | §3.1; every `s*`/`e*` run used a host-tilize temp copy | tilize op family / llama32_1b_quasar test utils |
| FLOAT32 × `transpose_k_heads=True` asserts on Quasar (`transpose.h`, tt-llk#1559) | §2.4 | LLK team |
| BFLOAT8_B not a Quasar format | §3.4 | model dtype decision |
| WH/BH parity of the unguarded Sharded change not run in this session | §8 commands | user |

No runtime-team item: no implicit-sync/credit stall, no tile-counter signature, no alias/lowering problem surfaced.

---

## 8. Parity claim (WH / BH) and confirmation commands

- **Interleaved:** the only change is inside `if (arch == tt::ARCH::QUASAR)`; the Gen1 `ComputeGen1Config` initializer
  and its `unpack_modes.emplace` are textually untouched and reach `.hw_config` as the same value. Zero Gen1 codegen
  change in every kernel (kernels untouched).
- **Sharded:** unguarded but behaviour-identical by construction — the kernel issues the same NoC reads of the same
  byte counts from the same remote addresses to the same L1 addresses (a borrowed DFB's write pointer *is* the shard
  base; `q_offset` is still added), with the same barriers; only the credit bookkeeping that had no counterpart
  (`reserve_back`/`push_back` on a buffer nobody drains, and the roles on `q_out` that were never used) is gone.
  Host side, the program no longer declares any DFB in this factory (borrowed DFBs consumed no extra L1), the
  `num_kv_tiles` RTA is dropped, and the tensor bindings carry the output bases exactly as the borrowed DFBs did
  (same re-pointing on cache hit through `override_runtime_arguments`, which was already a pure `tensor_args` echo).

User-run confirmation (from the repo root, venv active; kernels changed → `export TT_METAL_FORCE_JIT_COMPILE=1`):
```bash
# WH (and BH): the port's confirmed set, expected 147/147 + 4/4 as before the uplift
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py -v
pytest tests/ttnn/unit_tests/operations/experimental/transformer/test_nlp_create_qkv_heads_program_cache.py -v
```

Quasar repro (this session's exact protocol; the temp copies are archived under
`scratchpad/qsr_nlp_create_qkv_heads/temp_tests/` and were deleted from the repo — copy them back next to the
originals under their `test_tmp_*_qsr.py` names to re-run):
```bash
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh qkv
P=models/experimental/llama32_1b_quasar/tests/prototype_ops
qsr_test timeout 1800 ./python_env/bin/python -m pytest $P/test_tmp_nlp_create_qkv_heads_qsr.py -x -v -s          # 4 interleaved cases
qsr_test timeout 1800 ./python_env/bin/python -m pytest $P/test_tmp_nlp_create_qkv_heads_sharded_qsr.py -x -v -s  # 12 sharded cases
qsr_test timeout 1800 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_nlp_create_qkv_heads_qsr.py -x -v -s
# repeat with: export TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1
# as captured (fails in from_torch/tilize before the op until §3.1 is fixed):
qsr_test timeout 1800 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_create_qkv_heads.py -v

cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3 && export TT_METAL_FORCE_JIT_COMPILE=1
qsr_emu timeout 2700 ./python_env/bin/python -m pytest $P/test_tmp_nlp_create_qkv_heads_emu_qsr.py -k "q1kv1seq32 and transposeK" -x -v -s
source /localdev/vsuresh/qsr-sim/env_emu.sh 2x3 && export TT_METAL_FORCE_JIT_COMPILE=1
qsr_emu timeout 2700 ./python_env/bin/python -m pytest $P/test_tmp_nlp_create_qkv_heads_sharded_qsr.py -k "test_sharded_nlp_create_qkv_heads and q2kv1 and fused_qkv" -x -v -s
```

---

## 9. Definition-of-done checklist (`quasar_porting.md` §10)

- [x] In place, existing directory + namespace; nothing copied into `experimental/quasar/`, no `::qsr`
- [x] `create_program_artifacts` / `ProgramArtifacts`; `dfb::` / `args::` / `tensor::`; no `CBIndex`, `get_arg_val`, `TensorAccessorArgs`
- [x] `opt_level` matches resolved legacy (compute O3 explicit, DM default) — base-port item, unchanged
- [x] every remaining DFB has valid `data_format_metadata`; kernels read sizes via `get_tile_size()` (JIT descriptor)
- [x] sync-free / DM self-loop DFBs converted (`q_out` → LTA; `k_out`/`v_out` → LTA, **owner to ratify**, §3.2)
- [x] no `disable_dfb_implicit_sync_*`
- [x] no borrow-with-offset ported as-is (none)
- [x] no multicast (none)
- [x] every op `*_init`ed once per fixed DFB pair; no pack-destination switch
- [x] no bare compute `wait/pop` or `reserve/push` pair (TEN-4746 audited: BARE / CONFIG-ONLY / GUARDED-PATH)
- [x] no non-zero-init semaphore (no semaphores)
- [ ] BH and WH pass unchanged — **user runs** (§8); argued structurally here
- [x] Quasar builds and runs — craq-sim 17 op cases + graph_ops, RTL 8 cases, all PCC 1.0; Quasar-only change arch-checked
- [x] no DIAG/debug leftovers (temp tests deleted; watcher/assert env only in the shell)
- [x] missing core deps flagged (tt-llk#1559 32-bit transpose; tilize factories on Quasar; Bfp8 format)
- [x] `QUASAR_UPLIFT_REPORT.md` written; RED-stop conditions checked (§4)
