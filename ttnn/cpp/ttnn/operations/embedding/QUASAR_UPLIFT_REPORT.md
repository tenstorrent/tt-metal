# Quasar uplift report — `embedding` / `EmbeddingsFusedProgramFactory`

**Scope:** the tilized-output factory only (`device/embeddings_fused_program_factory.{cpp,hpp}` and the kernels it
binds: `device/kernels/dataflow/embeddings_tilize.cpp`, `device/kernels/dataflow/embeddings_common_metal2.hpp`,
`device/kernels/compute/tilize_chunked.cpp`, plus the shared-pool `ttnn/kernel/compute/tilize_metal2.cpp` and the
borrowed `eltwise/unary/.../writer_unary_interleaved_start_id_metal2.cpp`, both read-only here). The RM and
TilizedIndices factories are Metal 2.0 (#53425) but out of this uplift's scope; they still declare the weight
cache / index page as reader self-loop DFBs and would be RED on Quasar for that reason if a test path selected them
(none of the llama32_1b_quasar cases do: all are ROW_MAJOR uint32 indices with `layout=TILE`, which selects Fused).

**Base:** PR #56279 (`fc00ecad865` prerequisite + `924026b85b8` port), cherry-picked onto
`vsureshTT/quasar_uplift_round_2`. Recipe: `quasar_porting.md` + `metal_2.0/ai/audit/quasar_audit.md`,
`post_port/semantic/{dm_self_loop_dfbs,gen2_hardware_configs}.md`, `post_port/style/sync_free_dfbs.md`.

**Status: GREEN.** Two Gen2 validator blockers fixed in place (DM self-loop DFBs → `Scratchpad`; arch-selected `ComputeGen2Config`); every Fused-factory case run on craq-sim and on the ZEBU RTL emulator passes bit-exact (PCC 1.0), including the PADDED local-cache path and the chunked compute path. One craq-sim-only artifact (§4.1) and two other-op/harness blockers (§8) recorded.

---

## 1. Pre-check: is the factory Metal 2.0 on Gen1?

Yes. `create_program_artifacts` → `ProgramArtifacts`; DFBs `weights_staging` / `index_scratch` / `output`
(borrowed_from the output TensorParameter when sharded) / `weight_cache`; TensorBindings `input` / `weights` /
`dst`; all CTAs/RTAs named. Kernels are device-2.0: `api/dataflow/*` `Noc` + `DataflowBuffer` + `TensorAccessor`
in the reader, `api/compute/tilize.h` + `compute_kernel_lib::tilize` in compute. Verified green on WH B0 by the port
commit (273 passed / 2 skipped, matches baseline).

## 2. Audit findings (quasar_audit.md + quasar_porting.md §7–§12)

| # | check | finding | action |
|---|---|---|---|
| 1 | DM self-loop DFB (Gen2 validator: `program_spec.cpp:1495` *"Self-loop DFBs are not supported for data-movement kernels on Gen2"*) | **2 sites**, both on the reader: `index_scratch` (accessor `in1`: `reserve_back(1)` / `get_write_ptr()` / raw `volatile` pointer reads / NOC dst / `push_back(1)` at exit) and `weight_cache` (accessor `local_cache`, PADDED/BINARY only: `reserve_back(1|2)` / `get_write_ptr()` / NOC dst, never pushed). Survey per `dm_self_loop_dfbs.md`: every use on the covered list; no `get_entry_size`/`pages_*`/`write_zeros`/mcast; no `borrowed_from`; no `dfb_run_overrides`; each write pointer captured once and never re-read after the only push → **no stride, no wrap** (degenerate no-index case). | **FIXED** → `ScratchpadSpec` + `ScratchpadBinding` (§3). |
| 2 | Quasar device formats (`is_supported_quasar`: no `UInt32`, no `UInt16`, no `Bfp8_b`) | `index_scratch` carried `data_format_metadata = UInt32` (uint32 indices) → would fail *"DFB has data format UInt32 which is not supported on architecture QUASAR"*. Nothing consulted the metadata (DM-only, never bound by compute). | **Resolved by #1**: the DFB no longer exists. `weights_staging`/`output` are bf16 (the op validates `weights.dtype()==BFLOAT16`, so no Bfp8 exposure). |
| 3 | Hardcoded `ComputeGen1Config` (validator `program_spec.cpp:905`) | `ComputeHardwareConfig compute_hw = ComputeGen1Config{};` — `gen2_hardware_configs.md` **shape 4 (compute)**. Gen1 sets **no** field. | **FIXED**: `if (device->arch()==QUASAR) compute_hw = ComputeGen2Config{};` (default-constructed = same HiFi4 / Precise / 16-bit dest / double-buffered dest; `enable_2x_src_register` untouched) + the mandatory `TODO(#52269)` marker. Gen1 line textually unchanged. |
| 4 | DM `hw_config` | reader/writer use `ttnn::create_{reader,writer}_datamovement_config(arch)` (shape 1). | none |
| 5 | Tilize PACK state on Quasar (`tilize_init` programs no PACK; paged_cache needed `#ifdef ARCH_QUASAR pack_init(dfb::out)`) | Both compute kernels do `compute_kernel_hw_startup(dfb::in, dfb::out)` once (Quasar branch: `llk_pack_hw_configure(ocb)` + `llk_pack_init(ocb)` + `llk_pack_dest_init()`), then only tilize `dfb::in → dfb::out`. **No DFB-id switch anywhere**, so the packer already targets `dfb::out`. paged_cache's hazard came from `pack_untilize_init` retargeting the ring first; nothing here does. | **not applied** (recipe: reactive only). Re-check on device: an all-zero output would be the symptom. |
| 6 | Re-`*_init` on DFB-id change (§7) | single operand pair for the kernel's lifetime. | none |
| 7 | Bare compute `wait_front→pop_front` / `reserve_back→push_back` (TEN-4746) | inside `compute_kernel_lib::tilize`: `wait_front` → `reserve_back` → `tilize_block` (real UNPACR+PACR) → `push_back` → `pop_front`. Ordered on every path (`if constexpr` on `wait_mode` only). | none |
| 8 | Semaphores (non-zero init; `Semaphore<>` explicit args) | none in the factory or kernels. | none |
| 9 | Multicast / NOC direction | none. | none |
| 10 | `tt_memmove`/`copy_via_memmove` on DFB memory (#56283) | not used. | none |
| 11 | Borrow-with-offset | `output` borrowed_from the output param at offset 0 = the whole shard; no offset semantics. | none |
| 12 | 64-bit pointer casts | `reinterpret_cast<volatile tt_l1_ptr input_token_t*>(uint32)` in the reader — **removed by #1** (`Scratchpad::operator[]`). Remaining `CoreLocalMem<uint32_t>(addr)` constructions take a `uint32_t` by design. | none |
| 13 | `fifo_page_size` / `get_local_cb_interface` | not used. | none |
| 14 | `disable_dfb_implicit_sync_*` | not set. | none |
| 15 | NoC loopback self-read (§6: *"can spin on can_post or silently drop the read on the emulator"*) | `read_token_async` under **PADDED/BINARY** replays cached rows with a `UnicastEndpoint{my_x,my_y,addr}` L1→L1 NOC read. Not on any llama32 test path (all GENERIC). | **flagged**, reactive candidate; same finding as paged_cache's update-cache writers. |
| 16 | Sim grid / DRAM | `split_work_to_cores(compute_with_storage_grid_size, num_blocks)` adapts to the 8x4 sim grid; interleaved DRAM. | none |
| 17 | Chunked path (`tilize_chunked.cpp`, `num_tiles_per_block > 256`) | same single-operand tilize; its `if constexpr (num_chunks > 1)` re-init between partial chunk sizes is a template-count change on the **same** DFB ids (legal). Not on the test paths (DIM 2048 → 64 tiles → shared `tilize_metal2.cpp`). | none; untested on Quasar |

## 3. Files changed (all under `ttnn/cpp/ttnn/operations/embedding/`; nothing outside; no `_metal2` fork needed)

| file | change | guard |
|---|---|---|
| `device/embeddings_fused_program_factory.cpp` | `INDEX_SCRATCH`/`WEIGHT_CACHE` retyped `DFBSpecName`→`ScratchpadSpecName`; their `DataflowBufferSpec`s → `ScratchpadSpec`s on `spec.scratchpads` (`size_per_node` = former `entry_size * num_entries`: `TILE_HEIGHT*input_element_size_bytes*1`, `cache_page_size*(PADDED?1:2)`); the four PRODUCER/CONSUMER `DFBBinding`s → two `ScratchpadBinding`s (`indices`, `local_cache`) in `.scratchpad_bindings` (declaration order: after `dfb_bindings`, before `tensor_bindings`); `weight_cache` registration and binding keep the same `use_local_cache` guard; now-unused `input_data_format` local removed (`-Werror=unused`); arch-selected `ComputeGen2Config{}` beside the untouched `ComputeGen1Config{}`. | Scratchpad conversion: **unguarded but behaviour-preserving by construction** (canonical post-port pass; same reads/writes, same order, same remote addresses; only the L1 allocation order shifts). Gen2 config: `device->arch() == tt::ARCH::QUASAR`. |
| `device/kernels/dataflow/embeddings_tilize.cpp` | `#include "api/scratchpad.h"`; `DataflowBuffer dfb_in1` + `reserve_back(1)`/`get_write_ptr()`/`reinterpret_cast`/trailing `push_back(1)` → `Scratchpad<volatile input_token_t> indices(scratch::indices)`; the index NOC read takes `indices` as destination with `{.offset_bytes = 0}`; token reads `indices[k]` (bounds-checked under asserts); `prepare_local_cache(noc, scratch::local_cache, ...)`. NOC barriers untouched. `T` keeps the old `volatile`. | unguarded, behaviour-preserving (see above) |
| `device/kernels/dataflow/embeddings_common_metal2.hpp` | `#include "api/scratchpad.h"`; **added** a `prepare_local_cache` overload taking `const ScratchpadBindingToken&` (PADDED: `pad_local_addr = base`, one NOC read at offset 0; BINARY: `zero_local_addr = base`, `one_local_addr = base + weight_stick_size`, reads at offsets 0 / `weight_stick_size` — exactly the DFB overload's placement). The `DFBBindingToken` overload is **unchanged** and still serves `embeddings.cpp` / `embedding_ind_tilized.cpp` (RM / TilizedIndices factories). Header comment updated. | additive overload; the two other readers resolve to the old overload |
| `QUASAR_UPLIFT_REPORT.md` | this file (uncommitted; delete before merge) | — |

Not changed: `embeddings_fused_program_factory.hpp`, `tilize_chunked.cpp` (nothing to fix, see §2 #5/#17),
the shared `tilize_metal2.cpp` / borrowed writer (out of the op directory; no Quasar defect found by reading).

## 4. craq-sim run

Environment: `source /localdev/vsuresh/qsr-sim/env.sh embedding` (libttsim craq-sim, 8x4 grid, 2x1 GB DRAM, slow dispatch,
`TT_METAL_FORCE_JIT_COMPILE=1`), host lib rebuilt via `qsr_rebuild` (REBUILD_OK) after the uplift. JIT cache confirms the
Fused path kernels were built and run: `embeddings_tilize`, `tilize_metal2`, `writer_unary_interleaved_start_id_metal2`
(and `tilize_chunked` for the chunked cases). All runs in the foreground, one process at a time.

| # | case | source | shape (ids / table) | result | PCC / error |
|---|---|---|---|---|---|
| 1 | `test_embedding_token[seq128]` | prototype_ops/test_embedding.py (unmodified) | 128 uint32 RM / 128256x2048 bf16 RM → TILE, DRAM | **PASS** | ≥0.99 (harness threshold; 14 s) |
| 2 | same, `TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` | " | " | **PASS** | no assert fired |
| 3 | `test_embedding_token[seq512]` | " | 512 ids, 16 blocks over the grid | **PASS** | ≥0.99 |
| 4 | `test_embedding_token[seq1024]` | " | 1024 ids, 32 blocks | **PASS** | ≥0.99 |
| 5 | `test_embedding_rope_lookup[batch1]` | " (unmodified) | table [1,1,128,64] bf16 **TILE** | **FAIL (harness, not this op)** | `TT_FATAL kernel.hpp:450 DataMovementKernel is not supported on Quasar` raised at test line 81, `U.to_tt(..., layout=TILE)` = `from_torch(layout=TILE, device=)` legacy tilize. Never reaches embedding. |
| 6 | graph case `01_1x32_u32_int-dram` (vocab 128256→4096) | temp copy `test_tmp_embedding_qsr.py` (deleted) | [1,1,1,32] uint32 / [1,1,4096,2048] bf16 RM → [1,32,2048] TILE | **PASS** | ≥ graph_case golden PCC (bf16: 0.99) |
| 7 | graph case `01` with LLK + lightweight asserts | " | " | **PASS** | no assert fired |
| 8 | graph case `02_1x1024_u32_int-dram` (vocab→4096) | " | [1,1,1,1024] / [1,1,4096,2048] → [1,1024,2048] | **PASS** | ≥0.99 |
| 9 | graph case `00_1x32_u32_int-dram` (weights host-tilized + `to_device`) | " | [1,32] uint32 / [1,1,8192,64] bf16 **TILE** | **FAIL (other op)** | inside `ttnn.embedding`'s composite: `ttnn::untilize` → `ttnn::prim::untilize_codegen` builds a legacy `DataMovementKernel` (`kernel.hpp:450`). The Fused factory is never reached. Owner: untilize. |
| 10 | standalone GENERIC 32 ids / 512x64 (single core, 2 tiles) | `qsr_embed_case.py 32 512 64` | RM → TILE DRAM | **PASS** | **PCC 1.0, bit-exact** |
| 11 | standalone **PADDED** 32 ids / 512x64, `padding_idx=7` (11 of 32 tokens are pad) | `qsr_embed_case.py 32 512 64 0 7` | exercises the new `Scratchpad` overload of `prepare_local_cache` + the NoC-loopback replay (§2 #15) | **PASS** | **PCC 1.0, bit-exact** (0/11 pad rows, 0/21 other rows mismatching); also PASS with LLK asserts |
| 12 | standalone 32 ids / 64x**8192** (256 tiles/block = largest non-chunked) | `qsr_embed_case.py 32 64 8192` | `tilize_metal2.cpp` | **PASS** | PCC 1.0, exact |
| 13 | standalone 32 ids / 64x**12288** (384 tiles = 6 equal chunks of 64) | `qsr_embed_case.py 32 64 12288` | `tilize_chunked.cpp`, single-call path (`last_chunk_tiles == tiles_per_chunk`) | **PASS** | PCC 1.0, exact |
| 14 | standalone 32 ids / 64x**8224** (257 tiles = 4x64 + **partial chunk of 1**) | `qsr_embed_case.py 32 64 8224` | `tilize_chunked.cpp`, mixed-width path (mirrors WH regression `test_embedding_chunked_partial_last_chunk[8224]`) | **FAIL** | PCC 0.9815. Wrong: tile-cols 251–255 only (last 5 tiles of chunk 3), rows 1–31 (row 0 correct), real-looking values (not zeros). Chunk 4 (tile 256) itself is correct. See §4.1. |

Standalone = `ttnn.open_device(0)`, `ttnn.embedding(ids, w, layout=TILE, memory_config=DRAM)`, golden `w[ids]`, PCC via
`comp_pcc`, plus a per-tile-column mismatch diagnostic. Script: `scratchpad/qsr_embedding/qsr_embed_case.py`; logs `sim_*.log` beside it.

### 4.1 Chunked partial-last-chunk mismatch (row 14) — LLK-level, flagged

`tilize_chunked.cpp` tilizes chunks 0–3 with `tilize<64,...>(num_chunks-1)` then the partial chunk with `tilize<1,...>(1)`;
each call is `InitAndUninit`, so between them the UNPACK thread runs `tilize_uninit` + `tilize_init(icb, block=1)`.
On Quasar `_llk_unpack_tilize_init_` (`tt_llk_quasar/llk_lib/llk_unpack_tilize.h:68`) writes
`THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0 = FULL_CT_DIM * C_DIM_FACES` (the row stride, 64 tiles → 1 tile) with
`cfg_rmw` and re-programs the MOP **with no stall for the unpacker to drain the previous MOP's queued `UNPACR_TILIZE`s**
(no `STALLWAIT` anywhere in that file). The observed damage matches exactly: the last ~5 tiles of chunk 3 were unpacked with
the 1-tile stride — row 0 (offset 0) is right, rows 1–31 read from `base + k*64 B` instead of `base + k*4096 B`. The
equal-chunk (row 13) and non-chunked (row 12) controls are exact, and the same geometry passes on WH (suite regression
test, `test_embedding.py:630`). Not one of the two documented craq-sim caveat signatures. **RTL result (§5 row 2): the identical case is bit-exact on
the ZEBU emulator → craq-sim artifact** (the simulator applies the unpacker's stride config to already-queued
`UNPACR_TILIZE`s; hardware orders the `cfg_rmw` behind them). No op-level change made or warranted. **Owner: craq-sim**
(new caveat signature for the recipe: *chunked tilize with a narrower trailing chunk → the last few tiles of the
preceding chunk unpack with the trailing chunk's row stride; rows 1..31 wrong, row 0 right; passes on RTL*). The
llama32_1b paths (dim 2048 = 64 tiles) never chunk.

## 5. RTL emulator run

Environment: `source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3` (ZEBU `emu-quasar-1x3`, 1 worker core, slow dispatch),
jobs via `qsr_emu timeout 2700 ./python_env/bin/python <script>` in the foreground, one device session per job, same
host lib as §4 (kernels JIT-built into `/localdev/vsuresh/tt-metal-cache-emu-1x3`). Boot ~1–5 min each ("Waiting for ack
msg from remote..."), no launch failures. Launcher logs moved to `scratchpad/qsr_embedding/job*_emu_*.log` (jobs 1 and 3; job 2's `emu_2026-09-11_20-00/20-02_.log`
disappeared from the repo root before I could move it — another agent's sweep — its run log `emu_case_32_64_8224.log` is
complete); run logs `emu_case_*.log` beside them.

| # | case | command | result | PCC / error |
|---|---|---|---|---|
| 1 | GENERIC, 32 uint32 RM ids / 512x64 bf16 RM table → [1,32,64] TILE DRAM (1 block, 2 tiles, single core) | `qsr_embed_case.py 32 512 64` | **PASS** | **PCC 1.0, bit-exact** (op 6.1 s incl. JIT) |
| 2 | chunked partial-last-chunk, 32 ids / 64x8224 (257 tiles = 4x64 + 1; `tilize_chunked.cpp` mixed-width path; the craq-sim FAIL of §4 row 14) | `qsr_embed_case.py 32 64 8224` | **PASS** | **PCC 1.0, bit-exact** (27 s) → the §4.1 mismatch is a **craq-sim artifact** |
| 3 | PADDED (`padding_idx=7`, 11/32 pad tokens), 32 ids / 512x64 — new `Scratchpad` overload of `prepare_local_cache` + NoC-loopback replay | `qsr_embed_case.py 32 512 64 0 7` | **PASS** | **PCC 1.0, bit-exact**; 0/11 pad rows, 0/21 other rows mismatching |

Not run on RTL (covered on craq-sim only): multi-core / multi-block shapes (§4 rows 3, 4, 8) — the 1x3 emulator has one
worker core; the factory's `split_work_to_cores` puts everything on it, which rows 1–3 above exercise.

## 6. Parity claim (WH/BH)

- Gen2 compute config is behind `device->arch() == tt::ARCH::QUASAR`; the `ComputeGen1Config{}` line is textually unchanged.
- The Scratchpad conversion is the canonical `dm_self_loop_dfbs.md` translation in its degenerate form (no index is
  read after an advance, so no stride/wrap): identical NOC transfers, sizes, source pages/offsets, barriers and
  token decoding; only the reader's private L1 region moves (scratchpads allocate alongside DFBs). Two DFB ids are
  freed. WH/BH suite to confirm (user runs): `pytest tests/ttnn/unit_tests/operations/data_movement/test_embedding.py`
  (expect 273 passed / 2 skipped, the port's baseline) and
  `pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_embedding.py -k tilized`.

## 7. RED-stop conditions checked

- Not Metal 2.0 on Gen1: **no**. Missing sanctioned Quasar capability: **no** (`Scratchpad`, `ComputeGen2Config`
  are first-class). Owner decision needed: **no** (no non-zero-init semaphore, no CB redesign, no open HW bug on the
  path). Only-fix-changes-WH/BH-unguarded: **no**. Stub LLK: **no** (`tilize_init`/`tilize_block`/`tilize_uninit`
  have Quasar branches; `fast_tilize_*` forwards to them on Quasar).

## 8. Deferred / follow-up

1. **RM / TilizedIndices factories** (`embeddings_rm_program_factory.cpp`, `embeddings_tilized_indices_program_factory.cpp`)
   keep reader self-loop DFBs (`in1`, `local_cache`) and a `UInt32`-format index DFB → RED on Quasar when selected
   (RM-output embedding, TILE-layout indices). Same two-line conversion as here; owner: embedding op owners.
2. **PADDED/BINARY local-cache replay via NoC loopback** (§2 #15) — verify on Quasar when a padded llama case appears.
3. **TILE-layout weight tables** (graph_ops case `00`, prototype `test_embedding_rope_lookup`) go through
   `ttnn::to_layout(TILE→ROW_MAJOR)` = `ttnn::untilize` → `ttnn::prim::untilize_codegen` *before* embedding, which builds a
   legacy `DataMovementKernel` (`kernel.hpp:450`) on Quasar (§4 row 9). Owner: untilize / to_layout. Additionally the
   prototype RoPE case dies earlier in the harness's own `from_torch(layout=TILE, device=)` (§4 row 5). Owner: llama32_1b
   Quasar test harness (host-tilize + `to_device`, as the temp copy did).
4. **craq-sim artifact** (§4.1): chunked partial-last-chunk tilize mis-unpacks the tail of the preceding chunk on
   craq-sim only; bit-exact on RTL. Owner: craq-sim; add to the recipe's sim-caveat list.
5. **Latent Gen1/Gen2 ring-straddle in the chunked path (observation, untested, not changed):** `weights_staging` has
   `buffering * tiles_per_chunk` entries and the reader pushes `64,64,64,64,last` per block, so with `num_blocks > 1`
   and a partial last chunk the second block's first push starts at ring offset `last` and its 64-tile contiguous
   write crosses the ring end (the DFB requires one trip's pushes to sum exactly to the capacity). The WH regression
   test uses 64 tokens on a many-core grid (1 block/core) and never hits it; the sim/RTL cases here are 1 block. Worth a
   multi-block chunked case on WH first. Owner: embedding op owners.

## 9. Repro commands

```
# craq-sim (foreground, one at a time)
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh embedding
qsr_test timeout 1800 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/prototype_ops/test_embedding.py -k "token and seq128" -x -q
qsr_test timeout 1800 ./python_env/bin/python <scratch>/qsr_embed_case.py 32 512 64          # GENERIC, exact
qsr_test timeout 1800 ./python_env/bin/python <scratch>/qsr_embed_case.py 32 512 64 0 7      # PADDED, exact
qsr_test timeout 1800 ./python_env/bin/python <scratch>/qsr_embed_case.py 32 64 8224         # chunked: sim artifact (PCC 0.98)
# ZEBU RTL
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3
qsr_emu timeout 2700 ./python_env/bin/python <scratch>/qsr_embed_case.py 32 512 64           # PASS exact
qsr_emu timeout 2700 ./python_env/bin/python <scratch>/qsr_embed_case.py 32 64 8224          # PASS exact
qsr_emu timeout 2700 ./python_env/bin/python <scratch>/qsr_embed_case.py 32 512 64 0 7       # PASS exact
```
`<scratch>` = `/tmp/claude-1211407858/-localdev-vsuresh-tt-metal/47c32300-1ca5-43ee-a917-81ae38742e1d/scratchpad/qsr_embedding`
(script also reproduced verbatim in `qsr_embed_case.py` there; it is ~40 lines: RM bf16 table + RM uint32 ids on
`ttnn.open_device(0)`, `ttnn.embedding(ids, w, [padding_idx=], layout=TILE, memory_config=DRAM)`, golden `w[ids]`).
