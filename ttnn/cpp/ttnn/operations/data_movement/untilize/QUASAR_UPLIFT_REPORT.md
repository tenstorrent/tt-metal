# Quasar uplift report — `untilize` / `UntilizeMultiCoreBlockProgramFactory`

**Status: GREEN for `UntilizeMultiCoreBlockProgramFactory`** — Metal 2.0 on Gen1, uplifted in place with one host-side,
arch-guarded change (Gen2 compute hw_config; the fp32 input stays `UnpackToSrc` on Quasar), bf16 bit-exact on craq-sim
(simple splits) and on the ZEBU RTL emulator (1024-wide DEST-chunked and 4-block multi-row splits, incl. the captured
model width); fp32 PASS at PCC 1.0 / max rel 9.7e-4 (not bit-exact, LLK flag). Two craq-sim-only failure signatures
(A, B) are recorded as simulator artifacts — both pass bit-exact on RTL.
**The model's call path is still RED for reasons outside this factory** (see Open items): `ttnn.untilize` auto-routes to
the legacy codegen tier, the captured input is Bfp8_b, and sub-`Wt=64` shapes select `UntilizeMultiCoreProgramFactory`.

Scope: exactly one program factory of `ttnn::prim::untilize` — `UntilizeMultiCoreBlockProgramFactory`
(`device/factories/untilize_multi_core_block_program_factory.{cpp,hpp}`) and the three kernels it binds:

| Kernel | Path |
|---|---|
| compute | `untilize/device/kernels/compute/untilize_wh_metal2.cpp` (→ `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.{hpp,inl}`, `pack_untilize.h`) |
| reader | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore_metal2.cpp` |
| writer | `untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore_metal2.cpp` |

Base: branch `vsureshTT/quasar_uplift_round_2`, with PR #56280 (2 commits: `6785ef0019f` codegen Native-tier
prerequisite, `8ed20cdeeab` the Metal 2.0 port) cherry-picked on top. Recipe: `quasar_porting.md` +
`metal_2.0/ai/audit/quasar_audit.md` + `metal_2.0/ai/post_port/semantic/gen2_hardware_configs.md`.

Out of scope (recorded RED where a test path selects them, not touched): the `codegen/` tier
(`create_descriptor`/`ProgramDescriptor`, 11 sites, 0 `create_program_artifacts`), the other five untilize
factories (still legacy), the legacy originals of the three kernels above. The `experimental/quasar/untilize` copy
was not opened.

---

## 1. Metal 2.0 confirmation (recipe §1 step 1)

| Check | Result |
|---|---|
| factory entry | `create_program_artifacts` → `ProgramArtifacts{spec, run_params}` |
| buffers | 4 `DataflowBufferSpec` (`in_full`, `out_full`, `in_cliffrow`, `out_cliffrow`), all with `data_format_metadata` = input/output format |
| tensors | `TensorParameter` `input`/`output` + `TensorBinding` `src`/`dst`; addresses travel through the bindings, no address RTAs |
| args | named CTAs / `runtime_arg_schema` + `AddRuntimeArgsForNode`; kernels use `get_arg(args::…)` |
| kernel APIs | reader/writer: `api/dataflow/{dataflow_api,noc,dataflow_buffer}.h`, `Noc`, `DataflowBuffer`, `TensorAccessor(tensor::…)`, `CoreLocalMem`; compute: `compute_kernel_hw_startup` + `compute_kernel_lib::untilize<…>` (device-2.0 `pack_untilize` API). No `cb_*`, no `noc_async_*` free functions, no `get_local_cb_interface`, no `CBIndex::c_*`, no `get_arg_val`, no `TensorAccessorArgs`. |
| opt_level | compute explicit `O3` (legacy resolved value); DM absent → O2 (legacy default) — base-port concern, unchanged |

Verdict: the factory is Metal 2.0 on Gen1; the uplift starts at the Quasar audit.

## 2. Quasar-uplift audit findings

### 2.1 CB/DFB classification (`cb_dfb_quasar_audit_helper.md`)

| DFB | producer → consumer | class | Quasar verdict |
|---|---|---|---|
| `in_full` / `in_cliffrow` | reader (DM, `reserve_back` → `noc.async_read(s, dfb, …)` → `push_back`) → compute (`wait_front`/`pop_front`) | 1, linear FIFO | portable; explicit sync (implicit sync is opt-in via `NocOptions::TXN_ID`, not used → no double-count) |
| `out_full` / `out_cliffrow` | compute (`reserve_back` → `pack_untilize_block` → `push_back`) → writer (`wait_front` → `get_read_ptr()` → `CoreLocalMem` → `noc.async_write` → `pop_front`) | 1, linear FIFO, pointer-only read | portable; same `get_read_ptr → CoreLocalMem → noc.async_write` shape as the already-uplifted `paged_cache/writer_fill_cache_interleaved.cpp` (passes craq-sim/RTL on this branch). Note: `get_read_ptr()` returns the UNCACHED alias on Quasar DM — see open items. |

No self-loop DFB (DM or compute), no borrowed DFB, no `evil_*`, no semaphores (so no non-zero-init semaphore), no multicast,
no `fifo_page_size` / `get_local_cb_interface` read (reader sizes via `dfb.get_tile_size()`, helper's
`get_dfb_num_pages` reads the HW tile counter on Quasar), no `Semaphore<>`.

### 2.2 Hazard checklist (task list + recipe §4/§7)

| Hazard | Finding | Action |
|---|---|---|
| hardcoded `ComputeGen1Config` | **yes** — `add_compute_region` builds `ComputeGen1Config{.enable_32_bit_dest = fp32_dest_acc_en}` (+ `unpack_modes` insert) by hand: recipe shape 4 | **FIXED** (§3) — arch-selected `ComputeGen2Config` beside it |
| DM `hw_config` | `ttnn::create_reader/writer_datamovement_config(arch)` → `DataMovementGen2Config{}` on Quasar (shape 1) | none |
| `pack_untilize` pack destination baked at init | one output DFB per compute instance, one `pack_untilize_init` (`InitAndUninit`), never switches output → no re-`pack_init` needed. Each work unit binds its own `src`/`out` pair via a separate kernel instance, so no DFB-id switch inside a kernel | none |
| DEST-wrap chunking | helper's block-based path calls `pack_untilize_block<sub_block_width, block_width_tiles>(…, b)` per DEST-sized sub-block with a fresh `wait_for_dest_available`/`dest_section_done` each; the Quasar LLK (`llk_pack_untilize_api.h`) accepts `full_ct_dim > block_ct_dim` + `block_c_index` (folded into one L1 dst offset). No manual DEST wrap / re-init. Block widths here are L1-bound (`cb_block_size_limit`), so this path is the normal one for the Block factory | **fails on craq-sim (signature A, every sub-block lands at column 0)** — not changed in the op; RTL decides (see runs) |
| TEN-4746 bare wait→pop / reserve→push | compute: `wait_front` → `pack_untilize_block` (real UNPACR+PACR) → `pop_front`; `reserve_back` → sub-block loop with ≥1 real pack → `push_back`. No BARE / CONFIG-ONLY / GUARDED-PATH shape (loop bodies unconditional, `num_sub_blocks ≥ 1`) | none |
| `*_init` before every execute on new DFB ids | single DFB pair per instance | none |
| `fp32_dest_acc` + Float32 DFB `unpack_modes` | present: `unpack_modes[in_dfb] = UnpackToDest` when `fp32_dest_acc_en` | **hangs on Quasar when carried verbatim** (§2.4) — Gen2 config uses `UnpackToSrc` for the 32-bit input |
| `Semaphore<>` explicit template args | none used | none |
| DM self-loop DFB → Scratchpad | none | none |
| `tt_memmove` / `copy_via_memmove` on DFB memory | not used | none |
| Bfp8_b input | the model capture (`graph_ops/test_untilize.py`) is bf8 `[1,1,32,128256]`; Quasar's `ValidateProgramSpec` rejects Bfp8_b DFBs | **model-level flag** — temp copy uses bf16; owner: llama32_1b_quasar |
| UInt32 input | factory sets `DST_ACCUM_MODE` for `UInt32`; Quasar has Int32 only — the op merely forwards the dtype, limitation lives at the format layer | flag only |
| `compute_kernel_hw_startup` once | yes, at `kernel_main` start | none |
| mcast rectangle | no multicast | none |
| non-zero-init semaphore | none | none |
| `disable_dfb_implicit_sync_*` | not set | none |

### 2.3 Routing (why a small case does not reach this factory)

`ttnn.untilize` → `supported_by_codegen` accepts bf16/bf8 interleaved TILE input with `use_multicore=True`, so the
model's call routes to the **codegen tier (legacy descriptor)** on every arch; the native prim is reached only
when codegen declines (`2·Wt·2048 > 800 KB` for multi-tile-row inputs, or the codegen live-L1 plan does not fit) or
via `ttnn._ttnn.operations.data_movement.untilize_force_native`. Inside the native prim the Block factory is chosen
only when `!enough_space_height` — `2·Wt·tile_bytes > free L1 per core` (Quasar: 4 MB L1). Hence on Quasar:

* `[1,1,32,128256]` (captured): native → **Block** (16 MB > 4 MB). Auto entry → codegen (RED, legacy).
* the threshold is much lower on Quasar than on WH: `get_max_l1_space()` reports only 128–256 KB free per core (see the
  routing note under the craq-sim run), so `Wt ≥ 64` (`[1,1,32,2048]`) already reaches the Block factory while
  `Wt ≤ 32` goes to `UntilizeMultiCoreProgramFactory` (Metal 2.0 on this branch with a hardcoded Gen1 config → RED,
  other factory). A pre-allocated L1 filler tensor does not change that number.

### 2.4 Found on device (reactive, §7/§8 style)

**Float32 input hangs on Quasar with a verbatim Gen1 `unpack_modes` copy.** The Gen1 config marks the 32-bit input
DFB `UnpackToDest` (legacy `UnpackToDestFp32`). On Quasar `genfiles.cpp` derives the kernel-wide
`constexpr bool UnpackToDestEn = any_unpack_to_dest(unpack_modes)`, and the Quasar `llk_math_wait_for_dest_available()`
then blocks on the `UNPACK_MATH` semaphore (`llk_math_common_api.h:126`). But the Quasar `pack_untilize` compute
path unpacks to **SrcA** unconditionally (`pack_untilize.h` `#else` branch: `llk_unpack_A<…, false /*unpack_to_dest*/>`),
so nothing ever posts that semaphore. Watcher (`watcher_fp32_prefix.log`, dump #44 at 234 s, all 32 cores identical):
TRISC0/unpack `D`one, **TRISC1/math `K` (stuck in kernel)**, writer `WFW` (waiting on `wait_front`), reader done,
`tiles_to_consume:3`. bf16 (no `unpack_modes` entry, `UnpackToDestEn=false`) is unaffected.

Fix (host, Quasar-only): the Gen2 config keeps `enable_32_bit_dest` and gives the 32-bit input an explicit
`UnpackToSrc` entry (the validator still wants an entry for a Float32 DFB) instead of `UnpackToDest`. WH/BH keep
`UnpackToDest`. Numerics: the SrcA route narrows fp32 (see the fp32 row of the craq-sim table for the measured
effect) — flagged to the LLK team as a missing Quasar `pack_untilize` unpack-to-dest route.

## 3. Files changed

| File | Change | Guard |
|---|---|---|
| `device/factories/untilize_multi_core_block_program_factory.cpp` | `add_compute_region`: keep the Gen1 `ComputeGen1Config` initializer textually unchanged; add `ComputeHardwareConfig compute_hw_config = compute_cfg;` and, when `device->arch() == tt::ARCH::QUASAR`, replace it with `ComputeGen2Config{.enable_32_bit_dest}` + (`fp32_dest_acc_en` only) `unpack_modes[in_dfb] = UnpackToSrc` (see §2.4 for why not `UnpackToDest`; `bfp_pack_precision_mode` dropped — Gen1-only; `enable_2x_src_register` left default; `TODO(#52269)` marker). `KernelSpec::hw_config` now takes the variant. | host arch check (`tt::ARCH::QUASAR`); WH/BH path identical |

No kernel edits were required. No file moved or renamed; namespace unchanged.

## 4. Parity claim

The only diff is an `if (arch == QUASAR)` branch in the host factory that replaces the hw_config *variant
alternative*; the Gen1 initializer, DFB specs, bindings, CTAs/RTAs and kernels are byte-identical, so WH/BH build
and run the original program. Confirmation command for the human (not run here):
`pytest tests/ttnn/unit_tests/operations/data_movement/test_untilize.py`.

## craq-sim run

Environment: `source /localdev/vsuresh/qsr-sim/env.sh untilize` (craq-sim `libttsim.so`, 8x4 worker grid, slow dispatch,
`TT_METAL_FORCE_JIT_COMPILE=1`), host lib rebuilt via `qsr_rebuild` after each factory edit. Driver:
`scratchpad/qsr_untilize/qsr_untilize_probe.py` (host tilize → `ttnn.to_device`, one op per case, PCC vs the input,
per-tile-segment source localization on failure, factory identified from the kernels JIT'd into `TT_METAL_CACHE`).
Runs with `TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` unless noted; passing cases were re-run with
asserts off (identical results). Logs: `scratchpad/qsr_untilize/sim_run*.log`, `watcher_fp32_prefix.log`, `*_fail.pt`.

Split per case (from `compute_ncores_wh`/`closest_square_larger_than_b`, grid 32, free L1 ∈ (128 KB, 256 KB] on
Quasar — see routing note below): "W×H tiles/core, blocks/core" and which compute-helper path that selects
(single-pass `pack_untilize_block<W,W>` when W ≤ 8, else DEST-chunked `pack_untilize_block<sub,W>(…, block_c_index)`).

| case | shape / dtype / mem | entry | factory reached | split (per core) | result | detail |
|---|---|---|---|---|---|---|
| route_auto_2048 | [1,1,32,2048] bf16 DRAM→DRAM | `ttnn.untilize` | **codegen tier (legacy)** | — | **RED (other tier)** | `TT_FATAL … DataMovementKernel is not supported on Quasar` (kernel.hpp:450) |
| E0 / E1 / E2 / E3 | [1,1,32,32] [1,1,64,64] [1,1,32,256] [1,1,32,1024] bf16 | force_native | **UntilizeMultiCoreProgramFactory** (Metal 2.0 on this branch, hardcoded `ComputeGen1Config`) | — | **RED (other factory)** | `TT_FATAL KernelSpec 'compute_full' targets Gen2 (Quasar) but its ComputeHardwareConfig holds a ComputeGen1Config` (program_spec.cpp:907) |
| P1_block_2048 | [1,1,32,2048] bf16 DRAM→DRAM | force_native | **Block** (3 `_metal2` kernels) | 2×1, 1 block, single-pass | **PASS** | PCC 1.000000, bit-exact; asserts on + off |
| E6_block_64x2048 | [1,1,64,2048] bf16 | force_native | Block | 3×2 (+1-wide cliff row), 2 blocks, single-pass | **PASS** | PCC 1.0, exact; asserts on + off |
| E8_block_32x4096 | [1,1,32,4096] bf16 | force_native | Block | 4×1, 1 block, single-pass | **PASS** | PCC 1.0, exact; asserts on + off |
| P3_fp32_2048 (before §2.4 fix) | [1,1,32,2048] fp32 | force_native | Block | 2×1, single-pass, `DST_ACCUM_MODE` | **HANG** | 20M sim cycles, no completion; watcher: TRISC1/math `K` on all cores, TRISC0 `D`, writer `WFW` (`watcher_fp32_prefix.log`) |
| P3_fp32_2048 (after §2.4 fix) | [1,1,32,2048] fp32 | force_native | Block | same | **PASS** (not bit-exact) | PCC 1.000000, max abs 3.9e-3, max rel 9.7e-4 (≈2^-10: fp32 narrowed through SrcA on Quasar) |
| E7_block_96x2048 | [1,1,96,2048] bf16 | force_native | Block | 3×3, 3 blocks, single-pass | **FAIL (sig. B)** | PCC 0.818; block 2 (rows 64–95) all **zero** |
| E4_block_128x2048 | [1,1,128,2048] bf16 | force_native | Block | 4×4, 4 blocks, single-pass | **FAIL (sig. B)** | PCC 0.751; block 2 holds **block 3's** data (`src{r64=[96:0 96:1 …]}`), block 3 correct |
| E5_block_2x64x2048 | [1,2,64,2048] bf16 | force_native | Block | 3×2 ×third_dim 2 = 4 blocks | **FAIL (sig. B)** | PCC 0.751; block 2 (= dim 1, tile-row 0) holds block 3's data |
| E11_block_256x2048 | [1,1,256,2048] bf16 | force_native | Block | 5×5 full (5 blocks) + 3-row cliff col (3 blocks) | **FAIL (sig. B)** | bad blocks 2, 4 (last → zero) on full cores; block 2 (last → zero) on cliff-col cores |
| E12_block_4x64x2048 | [1,4,64,2048] bf16 | force_native | Block | 3×2 ×4 = 8 blocks | **FAIL (sig. B)** | bad blocks 2, 4, 6, each holding block k+1's data |
| E9_block_32x8192 | [1,1,32,8192] bf16 | force_native | Block | 9×1 (+4-wide cliff), DEST-chunked 3 sub-blocks of 3 | **FAIL (sig. A)** | PCC 0.028; `r0=[0:6 0:7 0:8 <garbage> …]` — last sub-block landed at column 0, middle columns garbage |
| E10_block_32x16384 | [1,1,32,16384] bf16 | force_native | Block | 17×1, 17 sub-blocks of 1 | **FAIL (sig. A)** | PCC 0.015, 92% zeros; `r0=[0:16 Z Z …]` |
| P2_block_32768 | [1,1,32,32768] bf16 | force_native | Block | 34×1, 17 sub-blocks of 2 | **FAIL (sig. A)** | PCC 0.015, 94% zeros; `r0=[0:32 0:33 Z …]` |
| G01_128256_dram (captured, bf8→bf16) | [1,1,32,128256] bf16 DRAM | force_native | Block | 130×1 (+cliff), 26 sub-blocks of 5 | **FAIL (sig. A)** | PCC 0.0006, 96% zeros; `r0=[0:125 … 0:129 Z …]` |
| 00_32x128256_bf16_int-l1 / 01_…_int-dram (temp pytest copy of `graph_ops/test_untilize.py`) | [1,1,32,128256] bf16, L1→DRAM / DRAM→DRAM, `use_multicore=True` | force_native | Block | as G01 | **FAIL (sig. A)** | `PCC below 0.999: 2.49e-05`, both cases; mesh fixture + host-tilize harness work on the sim |

**Routing note (why small shapes reach the Block factory only on Quasar).** `enough_space_height` compares
`2·Wt·tile_bytes` against `get_max_l1_space()` = `lowest_occupied_compute_l1_address − base_allocator_addr`. On Quasar
that evaluates to 128–256 KB (E3 `Wt=32`/128 KB → MultiCore, P1 `Wt=64`/256 KB → Block; also on the ZEBU 1x3 grid), not the
~1.3 MB of WH, so many shapes the WH suite never sends to the Block factory land here on Quasar, including splits with
several blocks per core and DEST-chunked block widths (the block width is L1-bound, `cb_block_size_limit`, not
DEST-bound). Pre-allocating an L1 filler tensor did **not** move the number (sim_run6), so it is not a test lever.

**Signature A — DEST-chunked path, `block_c_index` not honoured.** Whenever the split gives a block width > 8
(`DEST_AUTO_LIMIT`), the helper issues `pack_untilize_block<sub, W>(icb, 1, ocb, b)` per sub-block. The Quasar API
folds `b` and the ring slot into one L1 offset (`llk_pack_untilize_api.h`: `l1_tile_idx = base_l1 + block_rt·y_stride +
block_c_index·block_ct_dim` → `_llk_pack_untilize_set_dst_offset_`). On craq-sim every sub-block lands at column 0 of
the block row (the last one wins, the rest of the row stays zero/garbage) — the same "pack_untilize mis-lands its L1
slot" family as the known craq-sim caveat. **Not changed in the op**; taken to RTL (below).

**Signature B — single-pass path, even blocks ≥ 2 corrupted.** With ≤ 8-wide blocks and ≥ 3 blocks per core, blocks
2, 4, 6 … come back holding block k+1's data, or zeros when k is the core's last block; blocks 0, 1 and all odd blocks are
right. That is a PACK-landing / DM-read-pointer (or reader / unpacker) ring desync after the first wrap of the 1-block
output ring — again the craq-sim pack_untilize ring-landing family (the caveat's "odd blocks all-zero" in a 2-slot ring
is its double-buffered form). **Not changed in the op**; taken to RTL.

**fp32 (§2.4).** Real Quasar issue, fixed in the factory (host, arch-guarded); pass is PCC 1.0 but not bit-exact
(max rel 9.7e-4) because the Quasar `pack_untilize` path unpacks fp32 through SrcA.

## RTL emulator run

Environment: `source /localdev/vsuresh/qsr-sim/env_emu.sh <cfg>`; jobs via `qsr_emu timeout 2700 …` (exclusive emulator
lock), one device session per job, same probe script. Logs: `scratchpad/qsr_untilize/emu_1x3_run1.log`,
`emu_2x3_run1.log` (the 1x3 launcher `emu_*.log` was swept from the repo root by a sibling job's cleanup before mine ran;
the probe log carries the full stdout).

### emu-quasar-1x3 (compute grid 1x1)

| case | shape | factory | result | detail |
|---|---|---|---|---|
| E0_block_32x32 | [1,1,32,32] bf16 | UntilizeMultiCoreProgramFactory (other factory) | **RED (other factory)** | same Gen1-config `TT_FATAL` as on the sim |
| P1_block_2048 | [1,1,32,2048] bf16 | Block | **FAIL (host, pre-existing)** | `TT_FATAL: WorkUnitSpec 'wu_full' targets node (0,1), which is out of bounds. The compute worker grid on this device is 1x1` (program_spec.cpp:727) |

The Block factory **cannot be placed on a single-core grid**: `closest_square_larger_than_b` only accepts a block size
whose block count is `< grid_area`, which is unsatisfiable for `grid_area == 1`, so it returns `{1,1}` and the split
asks for `Wt·Ht` cores. Arch-independent (the WH single-core path is `UntilizeSingleCoreProgramFactory` /
`use_multicore=False`, a different factory) — so the requested `[1,1,64,64]`-on-one-core RTL case is not reachable
through this factory. RTL evidence therefore comes from `emu-quasar-2x3` (2 worker cores), below.

### emu-quasar-2x3 (compute grid 2x1)

Routing differs again: on this config `get_max_l1_space()` is ≈ the whole 4 MB, so `Wt ≤ 512` (`P1`, `E6`, `E4`, `E8`,
`E9`, `E10`) all route to `UntilizeMultiCoreProgramFactory` (RED, other factory, same Gen1-config FATAL) and only
`Wt ≥ 1024` reaches the Block factory. Consequently the ≤ 8-wide-block single-pass configurations that carry signature
B on the sim cannot be produced here except on a cliff core of a ≥ 1024-wide shape (see E13).

| case | shape | factory | split (2 cores) | result | detail |
|---|---|---|---|---|---|
| P1 / E6 / E4 / E8 / E9 / E10 | Wt = 64 … 512 | UntilizeMultiCoreProgramFactory | — | **RED (other factory)** | Gen1-config `TT_FATAL` (program_spec.cpp:907) |
| **P2_block_32768** | [1,1,32,32768] bf16 | **Block** (3 `_metal2` kernels JIT'd) | 1024-wide block(s), DEST-chunked `pack_untilize_block<8,1024>`/`<7,1022>` with `block_c_index` 0…127 | **PASS** | PCC 1.000000, **bit-exact**, 65.6 s device time — **signature A does not reproduce on RTL** |
| **E13_block_128x32768** | [1,1,128,32768] bf16 | Block | 4 tile-rows → 4 blocks per core (block 2 is the signature-B-sensitive one), DEST-chunked | **PASS** | PCC 1.000000, **bit-exact**, 238.8 s — **signature B does not reproduce on RTL** |
| **G01_128256_dram** (captured model width, bf8→bf16) | [1,1,32,128256] bf16 DRAM→DRAM | Block | ≈2997 + 1011 wide split, DEST-chunked | **PASS** | PCC 1.000000, **bit-exact**, 169.7 s |

## Open items / blockers

| # | Item | Kind | Owner | Symptom / repro |
|---|---|---|---|---|
| 1 | `ttnn.untilize` (model entry, `use_multicore=True`) routes every bf16/bf8 interleaved TILE input to the **codegen tier**, which is still `create_descriptor`/`ProgramDescriptor` | legacy tier on the model path — RED, out of scope here | untilize codegen owner (PR #56280 base) | `TT_FATAL: DataMovementKernel is not supported on Quasar` (kernel.hpp:450); probe case `route_auto_2048` |
| 2 | `UntilizeMultiCoreProgramFactory` (and `…ParallelizeColumn…`) are Metal 2.0 on this branch but build a hand-written `ComputeGen1Config` — selected for every `Wt < 64` (sim/1x3) resp. `Wt < 1024` (2x3) | other factory needs its own Gen2 uplift — RED, out of scope | untilize Quasar uplift (next factory) | `TT_FATAL KernelSpec 'compute_full' targets Gen2 … holds a ComputeGen1Config` (program_spec.cpp:907) |
| 3 | Captured model input is **Bfp8_b** (`[1,1,32,128256]`); Quasar rejects Bfp8_b DFBs (no BFP formats on Gen2) | model-level dtype decision | llama32_1b_quasar | temp copies used bf16; decide bf16 logits on Quasar |
| 4 | fp32 untilize on Quasar is not bit-exact (max rel 9.7e-4): the Quasar `pack_untilize` path has no unpack-to-dest route, and requesting one (`UnpackToDest`) hangs MATH on the `UNPACK_MATH` semaphore | missing LLK route (Quasar) — op works around with `UnpackToSrc` | LLK team | `P3_fp32_2048` before/after §2.4; `watcher_fp32_prefix.log`; `llk_math_common_api.h:126`, `pack_untilize.h` `#else` |
| 5 | craq-sim **signature A**: DEST-chunked `pack_untilize_block<sub, W>(…, block_c_index)` lands every sub-block at column 0 (E9/E10/P2/G01 on sim); **passes bit-exact on RTL** (P2, E13, G01) | simulator bug (pack_untilize L1 dst offset) | craq-sim (same family as craq-sim#355 / the "ring slot 1" caveat) | `sim_run5_bracket.log`, `*_fail.pt` |
| 6 | craq-sim **signature B**: even blocks ≥ 2 of a 1-block output ring carry block k+1's data / zeros (E4/E5/E7/E11/E12 on sim); the 4-block RTL case (E13) **passes bit-exact** | simulator bug (ring landing / credit after first wrap) | craq-sim | `sim_run10_B_pattern.log`, `E4_block_128x2048_fail.pt` |
| 7 | The Block factory cannot be placed on a **single-core grid** (`closest_square_larger_than_b` needs `numX·numY < grid_area`) → `WorkUnitSpec 'wu_full' targets node (0,1) … grid is 1x1` on emu-quasar-1x3 | pre-existing, arch-independent split limitation | untilize owner (informational) | `emu_1x3_run1.log` |
| 8 | `DataflowBuffer::get_read_ptr()` returns the UNCACHED alias on Quasar DM and the writer feeds it to `noc.async_write` via `CoreLocalMem` (header says NOC APIs don't take uncached addresses). Works on sim and RTL here and matches the uplifted `paged_cache` writers — no action, noted for the runtime team's cache-strategy follow-up | informational | runtime | — |
| 9 | Quasar `get_max_l1_space()` differs per target (128–256 KB on the 8x4 sim / 1x3, ≈4 MB on 2x3), which moves the `enough_space_height` factory choice; WH never sends `Wt < ~330` to this factory | routing observation | untilize owner | routing note in the craq-sim section |

No WH/BH-affecting change; no kernel, LLK, runtime or CMake edits; nothing copied from or into `experimental/quasar/`.

## Repro commands

```bash
# host rebuild after the factory edit (exclusive build lock)
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh untilize && qsr_rebuild   # expect REBUILD_OK

# craq-sim (8x4), asserts on; drop the two env vars for the asserts-off repeat
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh untilize && \
TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1 qsr_test timeout 1500 ./python_env/bin/python \
  /tmp/claude-1211407858/-localdev-vsuresh-tt-metal/47c32300-1ca5-43ee-a917-81ae38742e1d/scratchpad/qsr_untilize/qsr_untilize_probe.py \
  route_auto_2048 E0_block_32x32 P1_block_2048 E6_block_64x2048 E8_block_32x4096 P3_fp32_2048 \
  E7_block_96x2048 E4_block_128x2048 E9_block_32x8192 P2_block_32768 G01_128256_dram

# fp32 hang evidence (pre-fix library), watcher on
TT_METAL_WATCHER=5 TT_METAL_WATCHER_DUMP_ALL=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 TT_METAL_LLK_ASSERTS=1 \
TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1 qsr_test timeout 240 ./python_env/bin/python <probe> P3_fp32_2048 ; cp generated/watcher/watcher.log <scratch>

# captured graph cases (temp copy of graph_ops/test_untilize.py: bf16, host tilize, untilize_force_native; deleted after the run)
qsr_test timeout 1500 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_untilize_qsr.py -v -s

# ZEBU RTL
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3 && qsr_emu timeout 2700 ./python_env/bin/python <probe> E0_block_32x32 P1_block_2048
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env_emu.sh 2x3 && qsr_emu timeout 2700 ./python_env/bin/python <probe> P2_block_32768
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env_emu.sh 2x3 && qsr_emu timeout 2700 ./python_env/bin/python <probe> E13_block_128x32768 G01_128256_dram

# WH/BH parity (human runs; not run here)
pytest tests/ttnn/unit_tests/operations/data_movement/test_untilize.py
```

Artifacts: `scratchpad/qsr_untilize/` — `qsr_untilize_probe.py`, `sim_run{1..10}*.log`, `watcher_fp32_prefix.log`,
`*_fail.pt` (input + output tensors of every sim failure), `emu_1x3_run1.log`, `emu_2x3_run{1..4}.log`.
