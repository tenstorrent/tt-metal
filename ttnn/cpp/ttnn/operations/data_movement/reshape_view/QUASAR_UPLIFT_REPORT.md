# Quasar Uplift Report — `ttnn/cpp/ttnn/operations/data_movement/reshape_view` (`ttnn.reshape`)

**Date:** 2026-09-10 (audit) / 2026-09-10 (apply session) · **Branch:** `vsureshTT/quasar_uplift_round_2` (PR 55835's commit `b7dbd862164` already cherry-picked) · **Recipe docs:** reference copy under `scratchpad/recipe_ref/` (`quasar_porting.md` + `metal_2.0/ai/**`; not tracked in this checkout, so no `git log` provenance line is available).
**Quasar test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_reshape.py` (5 captured cases).
**No build, no device run in either session** (per the recipe the user runs all builds/tests; commands in §8). Changes are **uncommitted** for review.

## Status: GREEN — captured path needs nothing; the two in-op deferred items from the audit are now APPLIED

The audit session (first half of this report, §1–§2) found that **none of the five captured `ttnn.reshape` calls launches a device program** (views / identity returns), so the Quasar model test is unaffected by any factory. It deferred three items for the untested factories. This apply session did the two that are inside the op and have a sanctioned recipe fix:

| # | Item (from the audit's §5) | Recipe | Outcome |
|---|---|---|---|
| 1 | Tiled factory: `MAPPING` DFB `data_format_metadata = UInt32`, rejected by `ValidateProgramSpec` on Quasar | `quasar_porting.md` §4/§7 ("Quasar has Int32, no uint32"; guard the format branch), Quasar-only | **APPLIED** — `RawUInt32` on `ARCH::QUASAR`, dtype-derived format elsewhere (§3) |
| 2 | RM factory: four DM self-loop DFBs `src0..src3` (`TT_FATAL` at `program_spec.cpp:1494` on Quasar) | `ai/post_port/semantic/dm_self_loop_dfbs.md` | **APPLIED** — 4 sites → 4 `ScratchpadSpec`s (§3, §4a) |
| 3 | Out-of-op: `data_movement/common/kernels/common.hpp:129` uncached-offset double-apply | not an op edit | **still DEFERRED / flagged** (§5 item 1) |

The scoping question ("`reshape_view` or `reshape_on_device`?") resolves to **`reshape_view`**: the Python `ttnn.reshape` binding (`reshape_view/reshape_nanobind.cpp`) calls `ttnn::reshape` in `reshape_view/reshape.cpp`, which never calls `ttnn::reshape_on_device` (a separate `ttnn.reshape_on_device` entry point, still `create_descriptor`/`ProgramDescriptor` — not Metal 2.0, out of scope, untouched).

## 1. Per-case trace through `ttnn::reshape` (`reshape_view/reshape.cpp`) — from the audit

| Case | Input → requested | Path taken | Device work | `-m emulator`? |
|---|---|---|---|---|
| `00_32x3072_bf16_int-l1` | `[1,1,32,3072]` bf16 TILE L1-interleaved → logical `(1,1,1,3072)`, padded `(1,1,32,3072)` (overload 2) | `this_is_view` false (`reshape.cpp:613-619`); volumes differ → `tile_tensor_view_reshape_possible` (`:628-634`) true → `ttnn::experimental::view` | **None.** `tt::tt_metal::view` → `view_device` (`ttnn/core/tensor/tensor_ops.cpp:321-403`) aliases the `MeshBuffer` with a new `TensorSpec`. No `ProgramSpec`, no kernel. | **yes** — the only case the emulator marker selects |
| `01_32x2048_bf16_ws-l1` | `[1,1,32,2048]` WIDTH_SHARDED 8×8 → `(1,1,32,2048)` | logical+padded equal → `reshape.cpp:578-581` returns the input | None (identity) | no; `graph_case._shard_grid_fits` SKIPs it on a 2-node emulator |
| `02_1024x2048_bf16_int-dram` | `[1,1,1024,2048]` → `[1,1,1024,-1]` | `infer_dims_for_reshape` → same shape → `:578-581` | None (identity) | no |
| `03_32x1024x64_bf16_int-dram` | `[1,32,1024,64]` → `[1,32,-1,64]` | identity | None | no |
| `04_1024x2048_bf8_int-dram` | `[1,1,1024,2048]` bf8 → same | identity (the bf8→bf16 typecast at `:500` is not executed) | None | no |

`pytest … -m emulator` runs exactly case 00; a full run adds four identity returns. The harness golden (`graph_case._ref_view`) compares the leading 3072 elements, which a view satisfies by construction.

## 2. PR 55835 (`gchoudhary/55526/dm-self-loop-dfb-data_movement`) — from the audit

Not needed for `test_reshape.py` on Quasar (no captured case reaches the tiled factory), but **required for any real tiled `reshape_view` on Quasar**: its hunk converted the tiled writer's `WORKING` self-loop DFB to a `Scratchpad` (`reshape_tiled_program_factory.cpp` `ScratchpadSpec{WORKING}` + `writer_reshape_tiled.cpp` `Scratchpad<uint8_t> working(scratch::working)`), which is exactly the `dm_self_loop_dfbs.md` fix; the runtime otherwise fires `program_spec.cpp:1494-1500` `"DataflowBuffer 'working' is self-looped by data-movement kernel 'writer' … not supported for data-movement kernels on Gen2"`. Checked against the recipe (single entry, pointer captured before the only push → index frozen at 0, `borrowed_from` unset, no `dfb_run_overrides`, `data_format_metadata` never consulted, `T = uint8_t`): correct. `MAPPING`/`IN_TILES` (reader PRODUCER → writer CONSUMER) are genuine cross-kernel FIFOs and were rightly left alone. Keep the cherry-pick.

## 3. Files changed (this apply session) — all under the op directory

| File | Reason |
|---|---|
| `device/reshape_tiled_program_factory.cpp` | Item 1. `mapping_dataformat` is now `device->arch() == tt::ARCH::QUASAR ? tt::DataFormat::RawUInt32 : datatype_to_dataformat_converter(mapping_tensor.dtype())` (+ a comment saying why). Host-side arch check; WH/BH evaluate the identical expression as before. |
| `device/reshape_rm_program_factory.cpp` | Item 2, host side. `DFBSpecName SRC0..SRC3` → `ScratchpadSpecName READER_SOURCE_STAGE / READER_DEST_STAGE / WRITER_SOURCE_STAGE / WRITER_DEST_STAGE`; `make_scratch_dfb` (a `DataflowBufferSpec` with `data_format_metadata`) → `make_stage_scratchpad` (a `ScratchpadSpec{unique_id, size_per_node}`); the four `spec.dataflow_buffers.push_back` → `spec.scratchpads.push_back` with `size_per_node = entry_size * num_entries` (`source_stage_size_bytes * 2u`, `dest_stage_size_bytes` × 1), the `WRITER` pair still inside `if (can_use_dual_kernel)`; `make_rm_kernel`'s four `DFBBinding`s (PRODUCER+CONSUMER × 2) → two `ScratchpadBinding`s (`accessor_name` `source_stage` / `dest_stage`), placed at `.scratchpad_bindings` between `.source` and `.tensor_bindings` per `kernel_spec.hpp` declaration order. Repairs my change made necessary: deleted the now-unused `dfb_data_format` local; renamed `dfb_size0/1` → `source_stage_size_bytes` / `dest_stage_size_bytes` and the budget helper's `dfb_size0` / `source_dfb_bytes` / `min_dest_dfb_bytes` params/locals (Cleanup item 3: leftover `dfb_` names describing nothing); reworded the four comments that described the DFB/self-loop shape. |
| `device/device/rm_reshape_interleaved.cpp` | Item 2, kernel side. `#include "api/dataflow/dataflow_buffer.h"` → `#include "api/scratchpad.h"`; the fake FIFO (`DataflowBuffer dfb_in0/dfb_in1`, `reserve_back(1)` ×2, `get_write_ptr()` ×2, `push_back(1)` ×2) → `Scratchpad<uint8_t> source_stage(scratch::source_stage)` / `dest_stage(scratch::dest_stage)` and `source_buffer = source_stage.get_base_address()`, `dest_buffer = dest_stage.get_base_address()`. Everything downstream (the NOC reads/writes, `tt_memmove`, barriers, slot arithmetic) is textually unchanged. Repairs: the header "Resource bindings" comment and the loop comment "single fixed DFB slot" now say scratchpad/staging region. |

Nothing tempted a move/rename: the op stays in `data_movement/reshape_view`, namespaces `ttnn::prim` / `ttnn::operations::data_movement`. No file outside the op directory was touched (no `_metal2` fork was needed — the RM kernel is op-private).

## 4. Recipe passes and gotchas — applied vs considered

### 4a. `dm_self_loop_dfbs.md` semantic pass on `ReshapeViewRMProgramFactory` — APPLIED (4 sites)

- **Sites:** `reshape_rm_program_factory.cpp` (pre-edit lines) `:148-151` specs `SRC0..SRC3`, `:187-190` bindings (each spec bound PRODUCER+CONSUMER by exactly one DM `KernelSpec`: `SRC0/SRC1` by `reader`, `SRC2/SRC3` by `writer`), `:215-221` registration; `rm_reshape_interleaved.cpp:80-87` the fake FIFO.
- **Survey (Step 2):** every binder takes both roles ✓; all binders are DM ✓; every use of the handles is on the covered list — `reserve_back`/`push_back` (translated), `get_write_ptr()` (become the index) — and the handle/id is passed to no callee (only the raw `uint32_t` address `source_buffer`/`dest_buffer` is handed to `enhanced_noc_async_read/write` and `tt_memmove`) ✓; no `get_entry_size`, `pages_*`, `async_write_zeros`, multicast, `evil_set_*` ✓; `borrowed_from` unset ✓; no `dfb_run_overrides` in `ProgramRunArgs` ✓; `data_format_metadata` (`dfb_data_format`) consulted by nothing (no `get_dataformat`, sizes come from CTAs/RTAs) → dropped ✓. `sync_free_dfbs.md` (the sibling style pass) has zero sites: all four DFBs called the FIFO machinery.
- **Translation (Step 3):** each pointer is captured once (`wr == 0`) *before* the only `push_back`, and no index is read after it advances → no stride, no wrap; the `push_back` updates are dead and were deleted (Cleanup item 1). `T = uint8_t` per the "hands the address to a NOC call, never accesses elements" row. `size_per_node` is the DFB's whole allocation written as the product of the spec's two fields. The kernel's own slot arithmetic (`dest_buffer + dest_slot * dest_slot_size_bytes`, offsets `< 16`) stays inside the former single entry exactly as before.
- **Verification:** none run here (no build/test in this session, per the brief). Sentinels for the user in §8 — the RM cases of `tests/ttnn/unit_tests/base_functionality/test_reshape.py`.
- **Repaired because the change falsified it:** comments at factory `:26-27`, `:50-53`, `:141-145`, `:155-156`, `:176-177`; kernel header `:16` and loop comment `:100`; unused `dfb_data_format`; `dfb_*` local/param names.
- **Noticed, not done:** the factory comment "Quasar's factory is an intentional mirror of this file" (`:36-37`) refers to the `experimental/quasar/` copy, which the recipe forbids consulting; left as is. After this conversion the RM kernel's `tt_memmove` source is a Scratchpad base (a plain cached L1 address), so it **no longer** feeds an uncached DFB alias into `copy_via_memmove` — the §5 item 1 double-apply exposure now remains only in the tiled writer.

### 4b. Item 1 — Quasar-only `RawUInt32` for the tiled `MAPPING` DFB — APPLIED

- **Symptom fixed:** `tt::is_supported_quasar` (`tt_metal/common/tt_backend_api_types.cpp:97-123`) lists `Int32` and `RawUInt32` but not `UInt32`, so `ValidateProgramSpec` (`program_spec.cpp:1737-1747`) fired `"DFB 'mapping' has data format 'UInt32' which is not supported on architecture QUASAR"` for every tiled `reshape_view`.
- **Why the metadata is inert here:** the DFB is DM-only (reader PRODUCER → writer CONSUMER); neither kernel calls `get_dataformat`; `entry_size` is `mapping_page_size_bytes`, not `tile_size(format)`; the only other runtime consumers of `data_format_metadata` are compute-kernel `unpack_modes` checks (`program_spec.cpp:1080-1135`, no compute kernel here) and the `DataflowBufferConfig.data_format` passthrough (`:2712`). `RawUInt32` and `UInt32` have identical `datum_size` (4) and `tile_size` (4096) in `tt_backend_api_types.hpp`.
- **Guard:** host-side `device->arch() == tt::ARCH::QUASAR`; `device` (`MeshDevice*`) is already in scope and already used for `create_*_datamovement_config(device->arch())`.

### 4c. Considered, not needed (unchanged from the audit unless noted)

- **Metal 2.0 prerequisite:** both factories are `create_program_artifacts` → `ProgramArtifacts`; kernels use `api/dataflow/*`, `Noc`, `DataflowBuffer`/`Scratchpad`, `TensorAccessor(tensor::…)`, `get_arg(args::…)`. No legacy device API.
- **hw_config / Gen2 configs (`gen2_hardware_configs.md`):** all four `KernelSpec`s use the arch-agnostic `create_reader/writer_datamovement_config(device->arch())` → shape 1, zero sites. No compute kernel → no `unpack_modes`, no `#52269` marker anywhere (correct: nothing for it to mark). `disable_dfb_implicit_sync_*` not set.
- **`opt_level`:** DM kernels leave it absent → `O2` on both eras; nothing to flag.
- **Non-zero-init semaphores:** none.
- **Compute-only hazards (`hw_startup`, BFD re-init, TEN-4746 bare pairs, tilize pack config, DEST wrap, `matmul_partials` borrow):** N/A, no compute kernel. The DM-side `wait_front`/`pop_front` pairs in the tiled kernels are NOC-overlay credits, out of scope per §7.
- **Multicast / NoC direction (§11):** no multicast; unicast only.
- **`fifo_page_size`:** not used; sizes come from CTAs.
- **`-Werror=int-to-pointer-cast` (§8.1):** the kernels' `reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(uint32_t)` is **not** the failing shape — clang's `-Wint-to-pointer-cast` fires only on C-style casts (`checkIntToPointerCast` is gated on `CStyle`), which is why `common.hpp:140` keeps the same `reinterpret_cast` idiom right beside the `(void*)(uintptr_t)` fix and compiles for Quasar DM. No edit; no C-style int→pointer casts exist in this op.
- **Uncached DFB pointers on Quasar DM (§8.3 `a00dd45`/#52769):** reader hands `get_write_ptr()` to `enhanced_noc_async_read`, writer hands `get_read_ptr()` to `tt_memmove`'s NOC path; `Noc` applies `l1_cached_view` before the NOC sees an address, and the volatile CPU reads of the mapping page go through the uncached alias (coherent). The one exception is §5 item 1.
- **`common.hpp` pulls `ckernel.h` into a DM TU (§8.1):** already `#if !defined(ARCH_QUASAR)`-guarded (`common.hpp:17-24`).
- **`IN_TILES` format = input dtype:** forwarded dtype; when the input is `UINT32` Quasar rejects it the same way, but per §7 a forwarded `DataType` is a format-layer limitation to flag, not an op edit (§5 item 3).
- **Quasar has no Bfp8:** bf8 inputs are typecast to bf16 before `prim::reshape_view` (`reshape.cpp:407,500`); the bf8 question lives in `typecast`.
- **RM shard-width 16-byte alignment (§7):** `ReshapeViewRMProgramFactory` only ever sees interleaved tensors — `perform_reshape_on_2D_RM` (`reshape.cpp:229-232`) runs `sharded_to_interleaved` before `prim::reshape_view` and re-shards afterwards; nothing to validate here.
- **`experimental/quasar/` copies:** not consulted, not cited.

## 5. Deferred / follow-up items (exact symptoms; none reached by the Quasar model test)

1. **Out-of-op latent Quasar bug in the shared DM helper — flag to the runtime / `data_movement/common` owners; NOT edited (outside the op directory).** `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:129` (`copy_via_memmove`, from `a454d849c14` #55498) unconditionally does `src_read_addr = src_l1_addr + MEM_L1_UNCACHED_BASE` under `ARCH_QUASAR && COMPILE_FOR_DM`. Since `a00dd45324b` (#52769) `DataflowBuffer::get_read_ptr()/get_write_ptr()` already return the uncached alias on Quasar DM, so a caller that feeds a DFB pointer into `tt_memmove`'s memmove fallback double-offsets the source to ≥ 8 MiB (`MEM_L1_UNCACHED_BASE = MEM_L1_BASE + 4 MiB`), outside both L1 windows. **The tiled writer hits exactly this:** `writer_reshape_tiled.cpp:46,54` takes `input_base_addr = dfb_in_tiles.get_read_ptr()` and `:61` calls `tt_memmove<false, true, false, …>(noc, output_addr, input_addr, szbytes)`; the memmove fallback is taken whenever a segment's input and output byte offsets differ modulo 16 (`common.hpp:211-222`), which is routine for tiled reshapes. Expected symptom: wrong output / NoC-sanitizer or L1 out-of-range fault inside `copy_via_memmove`, Quasar only, on any real tiled `reshape_view` (e.g. the forced `[1,1,32,64]→[1,1,64,32]` command in §8). Suggested fix location is `common.hpp` (normalise/mask the alias before adding it, or accept either view), not the op. By inspection only — unverified on device. (The RM kernel was exposed through `dfb_in0.get_write_ptr()` before this session; after the Scratchpad conversion its `tt_memmove` source is a plain cached L1 base, so it is no longer exposed.)
2. **Related, for the same owners (observation, lower confidence):** on Quasar the RM kernel's dest staging slots are written both by the NOC (`tt_memmove`'s self-copy path lands in TL1) and by the CPU (`copy_via_memmove`, through L1 D$/L2 then `flush_l2_cache_range`). A CPU write-allocate that merges into a line the NOC updated after the line was last cached could publish stale bytes on the flush. This is a property of `common.hpp`'s coherence design for mixed NOC/CPU writers, pre-exists this session, and does not manifest on the flat-memory emulator; noted so the DM-common owners can decide whether `copy_via_memmove` should invalidate the destination range first.
3. **`IN_TILES` with a UINT32 input** (`reshape_tiled_program_factory.cpp` `input_dfb_data_format`): the same `ValidateProgramSpec` rejection as item 1 fired, but for a forwarded input dtype — a format-layer limitation (`tt::is_supported_quasar` has no `UInt32`), not an op edit. Symptom: `"DFB 'in_tiles' has data format 'UInt32' which is not supported on architecture QUASAR"` for `ttnn.reshape` of a UINT32 TILE tensor that fails the view predicates. Model-level dtype decision.
4. **`reshape_on_device`** (`ttnn.reshape_on_device`): still `create_descriptor` / `ProgramDescriptor`, binds the cross-op `eltwise/unary/…/writer_unary_interleaved_start_id.cpp`. Not Metal 2.0, not reached by `ttnn.reshape`, not reached by the test — RED-stop condition 1 applies if it is ever in scope. Untouched.
5. **Harness-level:** case 01 SKIPs on the emulator (captured 8×8 L1 shard grid); cases 02–04 are not `emulator`-marked; only case 00 runs under `-m emulator`.
6. **Not verified in this session (needs the user's runs):** Quasar `ValidateProgramSpec` now passes item 1 by construction, and the `program_spec.cpp:1494` self-loop `TT_FATAL` can no longer fire for the RM factory (no DFBs remain in it), but neither factory has been executed on Quasar; item 1 above is the next expected Quasar blocker for the tiled path.

## 6. Parity claim (WH/BH)

- **Tiled factory:** the only change is host-side and arch-gated; on WH/BH `mapping_dataformat` evaluates to exactly the pre-edit expression, so the `ProgramSpec` is byte-identical → no behaviour change by construction.
- **RM factory + kernel:** unguarded but **behaviour-preserving by the recipe's equivalence argument** (`dm_self_loop_dfbs.md`: "the converted kernel performs the same reads and writes, in the same order, to and from the same remote addresses; its own L1 region may land elsewhere"). Concretely: the kernel captured each DFB's base once, before its only `push_back`, and never re-read a pointer — so every address it computed was `base + k`, which is what `get_base_address() + k` yields now. The per-node L1 footprint is unchanged (`source_stage_size_bytes * 2` and `dest_stage_size_bytes` per instance, the DFBs' `entry_size * num_entries`); only the allocation order within the DFB/scratchpad region can shift. The two dropped `reserve_back`/`push_back` pairs synchronized nothing (single-threaded sole toucher) and never did barrier duty — all `noc.async_*_barrier()`/`async_writes_flushed()` calls are untouched. This is the same transformation PR 55835 applied to the tiled `WORKING` buffer and reported green (334 passed) on n150.
- Confirm with the BH → WH runs in §8; kernels are JIT-compiled from the working tree, so use `TT_METAL_FORCE_JIT_COMPILE=1` (or purge `~/.cache/tt-metal-cache`) so the edited RM kernel is actually rebuilt.

## 7. RED-stop conditions checked

Not Metal 2.0 (no — `reshape_view` is M2; `reshape_on_device` is not but is out of scope) · missing sanctioned Quasar API (none needed; `Scratchpad`, `ScratchpadSpec`, `RawUInt32` all exist) · construct needing an owner decision (none in this op's factories after the two fixes; §5 items 1–3 are owner items outside the op or at the format layer) · fix would change WH/BH un-guarded (item 1 is guarded; item 2 is the recipe's behaviour-preserving conversion, same as the accepted PR 55835 hunk) · stub LLK (no compute) → **none fire**.

## 8. Commands for the user (run from the repo root, venv active)

Host-side changes need a rebuild of `ttnn`; kernel changes are JIT-compiled at run time — force JIT so the RM kernel edit is picked up:

```bash
./build_metal.sh --build-tests            # or: ninja -C build ttnn && the install target
export TT_METAL_FORCE_JIT_COMPILE=1
```

**Blackhole, then Wormhole (parity — must be unchanged vs. `main`):**
```bash
# whole reshape suite (RM + TILE + sharded + program-cache)
pytest tests/ttnn/unit_tests/base_functionality/test_reshape.py -v
# RM factory sentinels for the self-loop -> Scratchpad conversion (single- and dual-kernel, clean and staged paths)
pytest tests/ttnn/unit_tests/base_functionality/test_reshape.py -v -k "rm or reshape_int or reshape_zero_element or reshape_oob"
# tiled factory sentinels (PR 55835 hunk + the arch-gated format line, which is a no-op here)
pytest tests/ttnn/unit_tests/base_functionality/test_reshape.py -v -k "tile or shard or subgrid or mamba or fp32 or bf8"
# the Quasar test file itself on Gen1 (all five cases run here; 01 needs an 8x8 grid, i.e. n150-class)
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_reshape.py -v
```

**Quasar emulator:**
```bash
# the one case the emulator marker selects (case 00 = pure view; expected PASS with zero device work)
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_reshape.py -m emulator -v
# all five (01 SKIPs on the shard grid; 02-04 are identity returns)
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_reshape.py -v
# force a REAL row-major reshape_view program on Quasar (RM factory). 100 B source pages / 200 B dest pages are
# not 16 B-aligned, so this takes the single-kernel STAGED path and exercises BOTH scratchpads (source + dest ring).
# Expected: no program_spec.cpp:1494 self-loop TT_FATAL any more; output must match torch.
python -c "import torch,ttnn; d=ttnn.open_device(device_id=0); t=torch.randn(1,1,64,50,dtype=torch.bfloat16); x=ttnn.from_torch(t,dtype=ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=d); y=ttnn.reshape(x,(1,1,32,100)); print(torch.equal(ttnn.to_torch(y),t.reshape(1,1,32,100))); ttnn.close_device(d)"
# same, on the clean (16 B-aligned) path, which also enables the dual-kernel split -> all four scratchpads bound
python -c "import torch,ttnn; d=ttnn.open_device(device_id=0); t=torch.randn(1,1,64,64,dtype=torch.bfloat16); x=ttnn.from_torch(t,dtype=ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=d); y=ttnn.reshape(x,(1,1,128,32)); print(torch.equal(ttnn.to_torch(y),t.reshape(1,1,128,32))); ttnn.close_device(d)"
# force a REAL tiled reshape_view program on Quasar (tiled factory).
# Expected: the former 'DFB mapping ... UInt32 ... not supported' TT_FATAL is gone; the next expected symptom is
# section 5 item 1 (common.hpp double uncached offset in copy_via_memmove) -> wrong output / L1 range fault.
python -c "import torch,ttnn; d=ttnn.open_device(device_id=0); t=torch.randn(1,1,32,64,dtype=torch.bfloat16); x=ttnn.from_torch(t,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=d); y=ttnn.reshape(x,(1,1,64,32)); print(torch.equal(ttnn.to_torch(y),t.reshape(1,1,64,32))); ttnn.close_device(d)"
```
Run Quasar both with `TT_METAL_LLK_ASSERTS` set and unset (§9); on the emulator use DPRINT / WATCHER / host gdb, not tt-triage.

## craq-sim run 2026-09-10

**Environment.** Quasar functional simulator craq-sim (`TT_METAL_SIMULATOR=/localdev/vsuresh/qsr-sim/libttsim.so`, `soc_descriptor.yaml` grid 11×8 with 32 functional workers → `compute_with_storage_grid_size = 8×4`, reported `Arch.QUASAR`), `TT_METAL_SLOW_DISPATCH_MODE=1`, `TT_METAL_FORCE_JIT_COMPILE=1`, private kernel cache `TT_METAL_CACHE=/localdev/vsuresh/tt-metal-cache-qsr-reshape`. Branch `vsureshTT/quasar_uplift_round_2` with the uncommitted §3 edits; host libs built from this tree (`build_Release/lib/_ttnn.so` 18:26, installed to `ttnn/ttnn/_ttnn.so`). Every run went through the shared-lock wrapper (`source /localdev/vsuresh/qsr-sim/env.sh reshape; qsr_test timeout …`); sim speed 1–3 kHz. No host rebuild was needed (no factory/host edit in this session). Debug env for the second pass: `TT_METAL_WATCHER=5 TT_METAL_WATCHER_DUMP_ALL=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1 TT_METAL_LOGGER_LEVEL=DEBUG`. Logs: `scratchpad/qsr_reshape/t*.log` (`t4_dbg_*` = debug-env pass, `*_watcher.log` = copies of `generated/watcher/watcher.log`).

### Per-test results

| # | Test / case | Result | Exactness | Notes |
|---|---|---|---|---|
| 1 | `tests/ops/test_reshape.py` **as-is** — all 4 cases (`mlp_prefill_fold`, `attn_qkv_unfold`, `mlp_collapse`, `attn_wo_fold`) | **FAIL (helper op, not reshape)** | — | `ttnn.from_torch(..., layout=TILE, device=mesh)` (`U.to_tt`) runs the legacy on-device `ttnn::tilize` (`TilizeDeviceOperation` → `CreateDataMovementKernel`) → `TT_FATAL kernel.hpp:450: DataMovementKernel is not supported on Quasar. Use QuasarDataMovementKernel instead.` Raised inside `ttnn.Tensor(...)` at `ttnn/ttnn/operations/core.py:376`, before `ttnn.reshape` is reached. |
| 1' | same 4 cases via temp copy `test_tmp_reshape_qsr.py` (host tilize + `ttnn.to_device`, otherwise identical) | **PASS 4/4** | PCC ≥ 0.999 **and** `torch.equal` bit-exact (extra assert in the copy) | All four are `PerformView` → `ttnn::experimental::view`: `out.buffer_address() == x.buffer_address()` (e.g. 24088704) — zero device work, as predicted in §1. 8.3 s wall. |
| 2 | `tests/graph_ops/test_reshape.py` **as-is** | 00 FAIL · 01 SKIP · 02 FAIL · 03 FAIL · 04 **PASS** | 04: harness golden (`_ref_view`) | 00/02/03: same helper-op `DataMovementKernel` TT_FATAL from `graph_case.build_tensor`'s device-side tilize. 04 passes as-is because bf8 `from_torch` tilizes on host. 01: `graph_case.py:203` "captured L1 shard grid needs cores up to (7,7); device compute grid is 8x4" (expected grid skip). |
| 2' | same via temp copy `graph_ops/test_tmp_reshape_qsr.py` (`G.build_tensor` replaced by host-tilize + `to_device`, grid-fit skip and partial-shard relayout kept) | **PASS 4, SKIP 1 (01)** | harness golden | 00 → view `(1,1,1,3072)` TILE L1; 02/03/04 identity returns. Confirms §1: none of the captured calls launches a program. |
| 3a | forced RM **staged** path `[1,1,64,50]→[1,1,32,100]` bf16 RM (100 B/200 B pages, single kernel, both scratchpads) | **PASS** (plain **and** debug env) | `torch.equal` True, 0 mismatches | Real program: `rm_reshape_interleaved` JIT-built for `dm2`; no `program_spec.cpp:1494` self-loop TT_FATAL → §3 item 2 (Scratchpad conversion) validated on Quasar. |
| 3b | forced RM **dual-kernel** path `[1,1,64,64]→[1,1,128,32]` bf16 RM (16 B-aligned, reader+writer, all four scratchpads) | **PASS** (plain and debug env) | `torch.equal` True, 0 mismatches | |
| 3c | forced **tiled** path `[1,1,32,64]→[1,1,64,32]` bf16 TILE | **PASS** (plain and debug env) | `torch.equal` True, 0 mismatches | `reader_reshape_tiled` + `writer_reshape_tiled` JIT-built; `ValidateProgramSpec` accepted `MAPPING` (→ §3 item 1 `RawUInt32` validated). Every segment of this shape has input/output byte offsets congruent mod 16, so `tt_memmove` stays on the NoC self-write path and never enters `copy_via_memmove` — it does **not** exercise §5 item 1. |
| 3d | forced tiled path with **misaligned segments** `[1,1,32,36]→[1,1,36,32]` bf16 TILE (added this session; output row 1 col 4 → 40 B vs input row 1 col 0 → 32 B ⇒ `copy_via_memmove` fallback) | **FAIL — simulator abort** (reproduced twice, plain and debug env, identical address) | — | Sim process dies: `ERROR: UnsupportedFunctionality: tile_mmio_rd8: addr=0x84dde0 size=1` right after the two DFB-config writes to cores 2-2/2-3 (2 output tiles → 2 cores). No Python traceback; watcher's only dump (#1, 8.0 s) is pre-launch (`GW`/`W1`, `k_ids 0`) because the sim exits within the 5 s interval, so the sim ERROR line is the authoritative symptom. **This is §5 item 1, confirmed on device** — see blocker B2. |

### Fixes applied in this session

**None inside the op** — no symptom fired against `reshape_view`'s own code. The two edits from the apply session (§3: Quasar-gated `RawUInt32` for `MAPPING`; RM self-loop DFBs → `ScratchpadSpec`/`Scratchpad`) were exercised for the first time on Quasar and behave as argued: the `UInt32` `ValidateProgramSpec` rejection and the `program_spec.cpp:1494` self-loop TT_FATAL no longer fire, and all three forced programs are bit-exact. Nothing was changed outside the op; no rebuild. The two work-around test copies (`tests/ops/test_tmp_reshape_qsr.py`, `tests/graph_ops/test_tmp_reshape_qsr.py`) were deleted after the run.

### Open blockers (symptom → owner)

- **B1 — test harness helper, not reshape:** `ttnn.from_torch(…, layout=TILE_LAYOUT, device=…)` for bf16 dispatches the legacy `ttnn::tilize` (`TilizeDeviceOperation`, `CreateDataMovementKernel`) → `kernel.hpp:450 "DataMovementKernel is not supported on Quasar"`. Trigger: `from_torch` with a `mesh_mapper` + bf16 + `TILE_LAYOUT` + `device=` (`py_to_tt_tensor.cpp:271-289`, on-device construction then `to_layout`); hits every bf16 TILE input in both repo test files (`op_utils.to_tt`, `graph_case.build_tensor`). bf8 inputs and mapper-less `ttnn.open_device` scripts are unaffected (host tilize). **Owner:** `tilize` op uplift / the `llama32_1b_quasar` test-harness owners (host-tilize + `ttnn.to_device` is the working idiom, already used by sibling `test_tmp_*_qsr.py` copies). Not an op edit; the repo test files were not modified.
- **B2 — CONFIRMED out-of-op Quasar bug (was §5 item 1, "by inspection only"):** `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:129` (`copy_via_memmove`, `#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)`) does `src_read_addr = src_l1_addr + MEM_L1_UNCACHED_BASE`, but on Quasar DM `DataflowBuffer::get_read_ptr()` already returns the uncached alias (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:326-327` + `:366-367`, `L1_UNCACHED_OFFSET = MEM_L1_UNCACHED_BASE`, from `a00dd45` #52769). `writer_reshape_tiled.cpp:46,54` feeds `dfb_in_tiles.get_read_ptr()` into `tt_memmove<false,true,false,…>` (`:61`), whose misaligned fallback is `copy_via_memmove`. Observed address arithmetic (`quasar/dev_mem_map.h`: `MEM_L1_SIZE = 4 MiB`, `MEM_L1_UNCACHED_BASE = 0x400000`): faulting `0x84dde0 = 0x400000 (DFB alias) + 0x400000 (common.hpp) + 0x4dde0 (cached in_tiles slot)`, i.e. ≥ 8 MiB — outside both the cached and the uncached L1 window, so the sim classifies it as tile MMIO and aborts on the memmove's 1-byte tail read. **Trigger:** any tiled `reshape_view` whose segment map has input/output byte offsets differing mod 16 (column offsets not multiples of 8 bf16 elements — e.g. widths like 36; the N150 capture has none, `[32,64]→[64,32]` has none). **Owner:** `data_movement/common` + runtime DFB owners; **suggested fix (not applied — outside the op; an in-op `- MEM_L1_UNCACHED_BASE` would hand-roll a private DFB interface, recipe §7):** in `common.hpp` normalise/mask the source before adding the alias (accept either view), or expose a public cached-pointer getter on `DataflowBuffer` and use it in `tt_memmove` callers. Once fixed, re-run 3d (expected bit-exact). The RM kernel is not exposed (its `tt_memmove` source is a Scratchpad base, applied once → valid alias; 3a passed on exactly that path).
- **B3 — expected grid skip:** graph case `01_32x2048_bf16_ws-l1` needs the captured 8×8 L1 shard grid; the sim exposes 8×4. Harness-level, no action.
- **Not reached / unchanged:** §5 item 3 (`UInt32` `IN_TILES`), item 4 (`reshape_on_device`), item 2 (coherence observation — the sim has no cache hierarchy, so it cannot manifest here).

### Reproduce

```bash
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh reshape
# 1 / 2 as-is (B1 fails them; 04 + skip 01 in graph_ops)
qsr_test timeout 2400 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/ops/test_reshape.py -v -s
qsr_test timeout 2400 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_reshape.py -v
# 1' / 2': temp copies = the same files with host tilize + ttnn.to_device (ops: replace U.to_tt; graph_ops: replace G.build_tensor)
# 3a / 3b / 3c (PASS) and 3d (B2 sim abort) — one-liners; add the debug env above for the second pass
python -c "import torch,ttnn; d=ttnn.open_device(device_id=0); t=torch.randn(1,1,64,50,dtype=torch.bfloat16); x=ttnn.from_torch(t,dtype=ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=d); y=ttnn.reshape(x,(1,1,32,100)); print(torch.equal(ttnn.to_torch(y),t.reshape(1,1,32,100))); ttnn.close_device(d)"
python -c "import torch,ttnn; d=ttnn.open_device(device_id=0); t=torch.randn(1,1,64,64,dtype=torch.bfloat16); x=ttnn.from_torch(t,dtype=ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=d); y=ttnn.reshape(x,(1,1,128,32)); print(torch.equal(ttnn.to_torch(y),t.reshape(1,1,128,32))); ttnn.close_device(d)"
python -c "import torch,ttnn; d=ttnn.open_device(device_id=0); t=torch.randn(1,1,32,64,dtype=torch.bfloat16); x=ttnn.from_torch(t,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=d); y=ttnn.reshape(x,(1,1,64,32)); print(torch.equal(ttnn.to_torch(y),t.reshape(1,1,64,32))); ttnn.close_device(d)"
python -c "import torch,ttnn; d=ttnn.open_device(device_id=0); t=torch.randn(1,1,32,36,dtype=torch.bfloat16); x=ttnn.from_torch(t,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=d); y=ttnn.reshape(x,(1,1,36,32)); print(torch.equal(ttnn.to_torch(y),t.reshape(1,1,36,32))); ttnn.close_device(d)"   # -> sim: UnsupportedFunctionality: tile_mmio_rd8: addr=0x84dde0
```
(run each `python -c` as `qsr_test timeout 1800 ./python_env/bin/python -c "..."`. Why the TILE one-liners are not stopped by B1: `ttnn/core/tensor/py_to_tt_tensor.cpp:254-320` — with a `mesh_mapper` (the tests) and `can_construct_on_device` true, `create_distributed_tensor` places the ROW_MAJOR source on device and `to_layout` runs the device `ttnn::tilize`; with a plain `ttnn.open_device` and no mapper, a TILE target sets `is_data_transformation_required` and the tensor is tilized on host via `Tensor::from_span` (#40850), so `from_torch` never launches the legacy op. The scripts are kept as `scratchpad/qsr_reshape/force_*.py`.)
