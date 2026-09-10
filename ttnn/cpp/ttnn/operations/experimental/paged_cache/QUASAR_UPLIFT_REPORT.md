# Quasar Uplift Report — `ttnn/cpp/ttnn/operations/experimental/paged_cache`

**Date:** 2026-09-10
**Branch:** `vsureshTT/quasar_uplift_round_2` (working tree = PR #54747 `edwinlee/Port_Paged_Cache`, 7 commits,
cherry-picked onto main `5c482a80f26`; the op's `main..HEAD` delta was diffed against `gh pr diff 54747` —
same 24 files, same +6296/−2186 lines, so this is exactly the PR's Metal 2.0 state).
**Recipe:** `quasar_porting.md` (field notes) + `ai/audit/quasar_audit.md` / `cb_dfb_quasar_audit_helper.md`,
with the in-place fixes taken from `ai/post_port/semantic/dm_self_loop_dfbs.md` and
`ai/post_port/semantic/gen2_hardware_configs.md`.
**No build and no test was run in this session** (the recipe leaves both to the user). Every claim below is
argued from source; §6 has the exact commands.

## Status: GREEN — uplift applied in place (3 fixes), no device run yet

The op is fully Metal 2.0 on Gen1 (all eight factories are `create_program_artifacts` /
`create_mesh_workload_artifacts`; every kernel is on the device-2.0 API — `DataflowBuffer`, `Noc`,
`TensorAccessor`, `get_arg(args::…)`, `compute_kernel_lib::tilize/untilize`; no `cb_*`, `get_arg_val`,
`TensorAccessorArgs`, `get_local_cb_interface`, `evil_*`, `fifo_page_size`, `read_tile_value`). So the
"not Metal 2.0 yet" RED condition does not fire for any factory the Quasar tests reach, and none of the
other RED-stop conditions apply (see §4).

Three Gen1-legal constructs would have stopped the op on Quasar; all three have a sanctioned in-place
fix and were applied (§2). **One item is blocked outside the op** and will make the *generated* Quasar
tests fail as they stand regardless of this uplift: both captured cases use a **BFLOAT8_B** cache, and
`tt::is_supported_quasar()` (`tt_metal/common/tt_backend_api_types.cpp`) has no `Bfp8_b`, so
`ValidateProgramSpec` rejects the `cache` DFB (`"DFB 'cache' has data format 'Bfp8_b' which is not
supported on architecture QUASAR"`). The op merely forwards the tensor dtype (recipe §7: nothing to
guard; flag it). This is a model-level dtype decision — §5 and §6 give a bf16 variant to run meanwhile.

### Which factories the Quasar tests reach

| test (`models/experimental/llama32_1b_quasar/tests/graph_ops/`) | exists | factory selected | kernels (defines) |
|---|---|---|---|
| `test_paged_update_cache.py` (cache `[128,8,32,64]` bf8 DRAM; input `[1,1,8,64]` bf16 L1 height-sharded 1 core; `update_idxs_tensor` int32; `page_table [1,128]` int32; `mesh_coords` unset) | yes | `PagedUpdateCacheProgramFactory` (`select_program_factory`: `mesh_coords` nullopt → single-device) | `reader_update_cache_interleaved_start_id.cpp`, `writer_update_cache_interleaved_start_id.cpp` (`USE_INDEX_TENSOR`, `IS_PAGED_CACHE`), compute `update_cache.cpp` |
| `test_paged_fill_cache.py` (cache `[128,8,32,64]` bf8; input `[1,8,512,64]` bf8; `page_table [1,16]` int32; `batch_idx=0` literal) | yes | `PagedFillCacheProgramFactory` (single-device) | `reader_fill_cache_interleaved.cpp`, `writer_fill_cache_interleaved.cpp` (no `USE_BATCH_IDX_TENSOR`, no `USE_VALID_SEQ_LEN`) — DM-only, no compute kernel |
| `test_paged_fused_update_cache.py` | **does not exist** | — (`PagedTiledFused…` / `PagedRowMajorFused…` are not exercised on Quasar) | — |

Both are `(1,1)`-mesh and tagged `emulator` by `graph_ops/conftest.py` (`_PRIMARY_ARG` exempts the tall
cache from the row cap). Both are checked by the `_check_cache_written` postcondition (sentinel 1024.0
must land exactly where the generated page table / positions say), not by a torch golden.

## 1. Files changed (all under the op directory; nothing outside it, no `_metal2` fork needed)

| file | change | reason |
|---|---|---|
| `device/fill_cache/paged_fill_cache_program_factory.cpp` | the three writer self-loop `DataflowBufferSpec`s (`page_table`, and the conditional `batch_idx`, `valid_seq_len`) → `ScratchpadSpec`s on `spec.scratchpads`; their PRODUCER+CONSUMER `DFBBinding` pairs → one `ScratchpadBinding` each; name constants retyped `DFBSpecName`→`ScratchpadSpecName` (`FC_*_SCRATCH`); the now-unused `page_table_data_format` / `batch_idx_data_format` locals removed (repair: they would be `-Werror` unused). Comments describing the self-loops rewritten (they would be false). | `program_spec.cpp:1495`: *"Self-loop DFBs are not supported for data-movement kernels on Gen2"* — `TT_FATAL` at first dispatch of `paged_fill_cache` on Quasar. `dm_self_loop_dfbs.md`, degenerate no-index case. |
| `device/kernels/dataflow/writer_fill_cache_interleaved.cpp` | `DataflowBuffer dfb_page_table/dfb_batch_idx/dfb_valid_seq_len` → `Scratchpad<volatile uint32_t>`; the `reserve_back(1)` / `get_write_ptr()` / `reinterpret_cast` triplets deleted (index stays 0: each pointer was captured once and never advanced — no push/pop, so no stride, no wrap); NOC reads take the scratchpad as destination with `{.offset_bytes = 0}`; element reads go through `operator[]`; `virtual_seq_tile_id_to_physical_tile_id` takes `const Scratchpad<volatile uint32_t>&` instead of a raw `volatile` pointer. `#include "api/scratchpad.h"` added. | same site, kernel side. `T` carries the old code's `volatile` (recipe: part of `T`, not decoration). NOC barriers untouched. |
| `device/update_cache/paged_update_cache_program_factory.cpp` | after the untouched `ComputeGen1Config compute_hw{…}` + `unpack_modes.emplace(...)` block: `ComputeHardwareConfig compute_hw_config = compute_hw; if (device->arch() == tt::ARCH::QUASAR) compute_hw_config = ComputeGen2Config{.enable_32_bit_dest = …, .unpack_modes = …};` and `.hw_config = compute_hw_config`. `TODO(#52269)` marker placed. | `program_spec.cpp:905`: *"targets Gen2 (Quasar) but its ComputeHardwareConfig holds a ComputeGen1Config"* — `TT_FATAL` at first dispatch of `paged_update_cache` on Quasar. `gen2_hardware_configs.md` shape 4 (hand-written Gen1 config; `bfp_pack_precision_mode` dropped, `enable_2x_src_register` never set). |
| `device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp` | same Gen2 block | same shape-4 site (not on the Quasar test path, but the pass is per-op and the branch is unreachable on WH/BH). |
| `device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp` | same Gen2 block | same. |
| `device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp` | `Semaphore<> receiver_sem(sem::receiver)` → `Semaphore receiver_sem(sem::receiver)` (CTAD) + comment | Quasar JIT compile error otherwise — see §2.3. |
| `device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp` | `Semaphore<>(sem::receiver).up(...)` → `Semaphore(sem::receiver).up(...)` | same. |
| `device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp`, `reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp`, `writer_paged_fused_update_cache_interleaved_start_id.cpp`, `writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp` | same one-token CTAD change | same construct; identical Gen1 behaviour; keeps the op uniformly compilable on Gen2. |
| `QUASAR_UPLIFT_REPORT.md` | this file (uncommitted; delete before merge) | recipe deliverable |

The op's directory, namespace (`ttnn::experimental::prim` / `ttnn::prim`), kernel paths and binding
vocabulary are unchanged. Nothing was copied into `experimental/quasar/`; nothing there was read or cited.
`METAL2_*.md` from the PR were read for context and not edited.

## 2. Why each fix, in detail

### 2.1 `paged_fill_cache`: DM self-loop DFBs → `Scratchpad` (test path: **yes**)

Survey per `dm_self_loop_dfbs.md` step 2: the writer bound `page_table` (always), `batch_idx`
(`USE_BATCH_IDX_TENSOR`) and `valid_seq_len` (`USE_VALID_SEQ_LEN`) as both PRODUCER and CONSUMER; no
other kernel binds them; the writer is DM. Every use of each handle was on the closed list —
`reserve_back(1)`, `get_write_ptr()` (captured once into a raw `volatile uint32_t*` / `CoreLocalMem`
NOC destination), no `push_back`/`pop_front`/`wait_front`, no `get_entry_size()`, no
`pages_*`, no `async_write_zeros`, no multicast, no helper taking the handle. `borrowed_from` unset;
no `dfb_run_overrides` anywhere in the op. `data_format_metadata` (Int32 / tensor dtype / `UInt32`) was
consulted by nothing (no compute kernel exists in this factory, and the kernel only takes an address) — it
drops. Note the `valid_seq_len` DFB carried `tt::DataFormat::UInt32`, which `is_supported_quasar()` also
rejects; the conversion removes that latent Quasar failure too.

Translation (recipe's degenerate case): no index is ever read after an advance, so `wr`/`rd` are 0 and
never written; `reserve_back` disappears; `get_write_ptr()` becomes the scratchpad itself (NOC operand,
`offset_bytes = 0`) or `operator[]` (element reads, now bounds-checked under `ASSERT`). `size_per_node`
is the former `entry_size * num_entries` written as that product. Behaviour-preserving on WH/BH by
construction: same NOC reads, same sizes, same order, same barriers; only the L1 allocation order shifts
(scratchpads are allocated alongside DFBs from the same region), which nothing depends on. It also frees
1–3 DFB ids per core.

### 2.2 Compute factories: Gen2 `hw_config` alternative (test path: **yes**, `paged_update_cache`)

All three compute factories build `ComputeGen1Config compute_hw{.enable_32_bit_dest = fp32_dest_acc_en}`
by hand (deliberately not via `to_compute_hardware_config`, to keep the legacy op's defaults — the PR's
comment explains why; that choice is preserved) and conditionally `emplace` `UnpackToSrc` entries. Shape 4
of `gen2_hardware_configs.md`. The Gen1 initializer and its emplace lines are textually untouched; the
only changed line inside `KernelSpec compute` is `.hw_config = compute_hw_config`. Gen2 disposition:
`enable_32_bit_dest`, `unpack_modes` copied verbatim (only fields the Gen1 config sets);
`bfp_pack_precision_mode` dropped (Gen1-only, Gen2 has MXFP instead); `enable_2x_src_register` left at
default. `device->arch()` is read once per factory, so no hoisted local (recipe rule). The `TODO(#52269)`
marker sits on each Gen2 config because Quasar's `unpack_modes` are Gen1-derived.

The DM kernels already go through `ttnn::create_reader/writer_datamovement_config(device->arch())`,
which returns `DataMovementGen2Config{}` on `ARCH::QUASAR` (shape 1: nothing to do).

### 2.3 `Semaphore<>` → CTAD `Semaphore` (test path: **yes**, `paged_update_cache` reader + writer)

`noc_semaphore.h`: constructing from a `sem::` token `static_assert`s `TOK_SCOPE == SCOPE`, and
`Semaphore<>` means `LOCAL_NONATOMIC`. The host census (`semaphore_scope.hpp::ResolveSemaphoreScope`)
returns `LOCAL_NONATOMIC` unconditionally on Gen1, but on Gen2 for `in0_sequential_mode` — bound by two
DM kernels (reader + writer) on every worker core — it returns `DM_LOCAL_CACHED` (single-core grid,
non-Emule) or `EXTERNAL` (multi-core, or Emule). So on Quasar the reader/writer JIT fails with
*"semaphore binding token's mechanism does not match this Semaphore's. Construct as `Semaphore
s(sem::name);`"* — the assert message itself prescribes the fix, the header provides the deduction guide,
and `sort`'s kernels already use the form. On WH/BH the deduced type is exactly `Semaphore<TENSIX,
LOCAL_NONATOMIC>` = `Semaphore<>`, so codegen is identical. The `up(noc, x, y, 1)` / `wait` / `set`
overloads exist for every scope. This is §8.1 "build skew"; it was applied ahead of a device run because
it is provable from source and one token wide. (Applied to the four fused-op kernels too — same
construct, same census outcome.)

## 3. Recipe gotchas — applied / considered and rejected

Applied: DM self-loop → Scratchpad (§6, `dm_self_loop_dfbs.md`); Gen2 hw_config (§4 / `gen2_hardware_configs.md`);
`Semaphore<>` build skew (§8.1 class). Non-zero-init semaphore check (`quasar_audit.md` §2): all three
`SemaphoreSpec`s use the default (zero) initial value — passes.

Considered, **not applied** (no symptom yet; recipe says reactive only) — in the order I would suspect
them if the bf16 Quasar run misbehaves:

1. **NoC loopback self-read in `writer_update_cache_interleaved_start_id.cpp`** (also both fused writers):
   `noc.async_read(local_src, CoreLocalMem(cache_l1_write_addr), Wbytes, {.noc_x = my_noc_x, .noc_y =
   my_noc_y, .addr = input_l1_read_addr}, {})` copies L1→L1 on the same core through the NoC. Recipe §6:
   *"Local self-read/self-copy (src==dst L1) on the emulator can spin on `can_post` or silently drop the
   read — use a direct L1→L1 RISC copy."* Symptom would be a hang at the writer (waypoint in
   `async_read_barrier`) or the updated row missing from the cache (postcondition "N of the M cells hold
   the written value"). Fix if it fires: `#ifdef ARCH_QUASAR` RISC copy of `Wbytes` from
   `input_l1_read_addr` to `cache_l1_write_addr` (both already uncached-alias addresses on Quasar DM);
   `#else` keep the NoC loopback. Not applied pre-emptively.
2. **Uncached DFB pointers on Quasar DM** (§8.3 / §12, `a00dd45`): every DM kernel still does
   `get_write_ptr()/get_read_ptr()` + `CoreLocalMem` as the NOC operand instead of passing the DFB.
   Checked: `Noc::get_src_ptr/get_dst_ptr` run `l1_cached_view()` on every `LOCAL_L1` operand and
   `UnicastEndpoint` does the same on `.addr`, so the uncached alias is mapped back before it reaches the
   NOC; the kernels' own `volatile` reads (index / page table) go through the uncached alias, which is what
   the getters intend. No manual `invalidate/flush_l2_cache_range` anywhere (good). Nothing to do.
3. **Bare `wait_front→pop_front` / `reserve_back→push_back` in compute (TEN-4746, §7 mandatory audit)**:
   the three compute kernels only call `compute_kernel_lib::untilize/tilize`. In `untilize_helpers.inl`
   every `wait_front` is followed by `pack_untilize_block` (real UNPACR + PACR) before `pop_front`, every
   `reserve_back` by the same before `push_back`, including the block-split path; in
   `tilize_helpers.inl` `wait_front → reserve_back → tilize_block → push_back → pop_front`. No
   `if`/`continue`/`return` bypasses the ordering work (GUARDED-PATH shape checked); the BH fast paths are
   compiled out on Quasar. Clean. DM kernels' bare pairs (fill_cache writer skip path, reader `input`
   publish) are out of scope per recipe.
4. **Re-`*_init` on every DFB-id change (§7)**: every helper call is `InitAndUninit` (or `InitOnly` on
   the first fused untilize) so `pack_untilize_init` / `tilize_init` re-run for `cache→untilized_cache`
   and `untilized_cache2→out` on every iteration; `compute_kernel_hw_startup` is called exactly once at
   `main()` start in all three kernels. Clean.
5. **`fifo_page_size` / `get_local_cb_interface`** (§5): none in op kernels. The kernels read
   `dfb.get_tile_size()` (JIT descriptor getter, `chlkc_descriptors.h` is emitted per program for every
   RISC on every arch) — fine. `compute_kernel_lib`'s `get_dfb_num_pages` has a Quasar branch
   (`g_dfb_interface`), so its debug ASSERTs compile.
6. **`dfb::` token → `uint32_t` in compute** (`compute_kernel_hw_startup(dfb::in, …)`, helper template
   args): the conversion is documented "intended for Gen1" but is unguarded, `DataflowBuffer(uint16_t)`
   exists on Quasar TRISC, and `pack_untilize_*` / `tilize_*` have real (non-stub) Quasar branches
   (`pack_untilize_uninit` is a documented no-op on Quasar). No LLK stub on the path → not RED.
7. **Multicast / NOC direction (§11)**: no multicast anywhere in the op. N/A.
8. **`partials_cb_uses_output` / borrow-with-offset (§7)**: the only borrow is `input` (whole shard,
   offset 0). N/A. `untilized_cache` / `untilized_cache2` `alias_with` group: validated by
   `program_spec.cpp` alias rules (same size, same nodes, consistent `borrowed_from`) and lowered on Gen2
   via `alias_primary_id`; if it misbehaves it is a runtime item, not an op edit.
9. **Int32 vs UInt32/UInt16 (§7)**: `index` / `page_table` DFBs take the tensor's format (Int32 in the
   tests). The fused readers have a `uint16` page-table branch and `paged_update_cache` accepts
   `UINT32` index tensors; a caller passing those on Quasar hits the validator's format check. Op merely
   forwards dtype → flag, don't guard.
10. **`opt_level`**: compute specs carry explicit `O3` (legacy resolved value); DM absent → O2. Base-port
    concern, correct as is; not touched.
11. **Implicit sync**: not disabled anywhere; the DM kernels use explicit `reserve/push/wait/pop` with
    `CoreLocalMem` operands, so implicit sync is not engaged. Nothing to do.
12. **Tilize pack config / `0x19` / `PACR0_TILE_INC` (§7, §8.2, §8.4)**: LLK-level; helpers run
    `pack_reconfig_data_format` + `tilize_init` before `tilize_block`. Reactive only.

## 4. RED-stop conditions checked

- Not Metal 2.0 on Gen1: **no** (all reached factories are M2; kernels are device-2.0).
- Missing sanctioned Quasar capability: **no** (`Scratchpad`, `ComputeGen2Config`, CTAD `Semaphore`
  are all first-class API; no hand-rolled device interface).
- Construct needing an owner decision: **no op-level one**. The bf8 dtype of the captured tests is a
  *model/test-level* decision (§5), not an op construct.
- Only fix changes WH/BH un-guarded: **no** (see §7 parity).
- Stub/unported LLK on the path: **no** (checked `pack_untilize_*`, `tilize_*` Quasar branches).

## 5. Deferred / follow-up items

1. **Blocked outside the op — test dtype.** `test_paged_update_cache.py` and `test_paged_fill_cache.py`
   are *generated* (do not hand-edit) and use `BFLOAT8_B` caches; `is_supported_quasar()` rejects
   `Bfp8_b`, so on Quasar they fail in `ValidateProgramSpec` before any kernel runs. Owner: llama32_1b
   Quasar model team (cache dtype for the Quasar variant; regenerate the capture or add a dtype override
   in `graph_case.py`). Until then use the bf16 copy in §6.
2. **Candidate reactive fix — NoC loopback self-read** in the three update-cache writers (§3 item 1).
   Apply only if the emulator run hangs/drops the in-place row write.
3. **Runtime team (only if it fires):** any implicit-sync/credit stall or `alias_with` lowering problem
   on Gen2 — report, do not work around.
4. **Base-port notes carried forward, not addressed here:** the PR's own open items (empty
   `mesh_coords` now raises; row-major fused factory has no test; `noop` attribute dead surface).
5. **Not on the Quasar test path but changed for op uniformity:** the two fused factories' Gen2 config and
   the four fused kernels' CTAD change. Gen1 codegen is identical; the Gen2 branch there is untested until
   a fused test exists on Quasar.

## 6. Test commands (user runs; order BH → WH → Quasar)

Kernels changed, so force JIT on every run: `export TT_METAL_FORCE_JIT_COMPILE=1`. Run from the repo root
with `export PYTHONPATH=$(pwd)` and the venv active. WH/BH expectations: identical pass/fail set and
numerics to the pre-uplift tree (the PR's baseline). The Quasar model conftest skips on Blackhole, so BH
parity is the op's own suite only.

```bash
# --- Blackhole and Wormhole (same commands on each; the PR's confirmed test set) ---
pytest tests/ttnn/unit_tests/operations/transformers/test_paged_cache_flexible_geometry.py -x -v
pytest tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py -x -v
# cache-hit path in isolation (override_runtime_arguments)
pytest tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py -v \
  -k "program_cache or program_caching or attr_idxs"
# fill_cache scratchpad conversion incl. batch_idx / valid_seq_len (bounded) paths
pytest tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py -v -k "fill_cache"
# fused factories (Gen2 branch is inert here; confirms the CTAD change is a no-op)
pytest tests/ttnn/unit_tests/operations/transformers/test_paged_fused_update_cache.py -x -v
pytest tests/ttnn/unit_tests/operations/transformers/test_paged_cache_mask.py -x -v
```

```bash
# --- Quasar emulator (your usual emulator env; 1x1 mesh) ---
# As generated (EXPECTED TO FAIL until the bf8 dtype decision: TT_FATAL "... 'Bfp8_b' ... not supported on architecture QUASAR")
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_update_cache.py -v
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_fill_cache.py -v

# bf16 variant that exercises the uplifted op (a temporary copy in the same directory so the
# module-scoped ttnn_mesh_device fixture and the emulator marker apply; delete both files afterwards):
sed 's/"BFLOAT8_B"/"BFLOAT16"/g' models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_update_cache.py \
  > models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_paged_update_cache_bf16.py
sed 's/"BFLOAT8_B"/"BFLOAT16"/g' models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_fill_cache.py \
  > models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_paged_fill_cache_bf16.py
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_paged_fill_cache_bf16.py -v      # DM-only: exercises 2.1
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_paged_update_cache_bf16.py -v    # exercises 2.2 + 2.3 (+ compute)
rm models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_paged_*_bf16.py
```

Run the Quasar tests both with `TT_METAL_LLK_ASSERTS=1` and unset (recipe §9); DPRINT needs the asserts
unset. If `paged_update_cache` hangs in the writer or the postcondition reports missing sentinel cells,
apply §3 item 1 (guarded RISC copy) and re-run.

## 7. WH/BH parity claim

Zero behavioural change on Gen1, argued structurally (no device run in this session):

- **Gen2 hw_config**: the `if (device->arch() == tt::ARCH::QUASAR)` branch is unreachable on WH/BH; the
  Gen1 `ComputeGen1Config` initializer and its `unpack_modes` emplaces are byte-identical to the PR, and
  `.hw_config` receives a `ComputeHardwareConfig` holding that same Gen1 value.
- **Semaphore CTAD**: on Gen1 the census bakes `LOCAL_NONATOMIC` into every `sem::` token, so
  `Semaphore receiver_sem(sem::receiver)` deduces `Semaphore<TENSIX, LOCAL_NONATOMIC>` — the exact type
  `Semaphore<>` spelled. Same code, same instructions.
- **fill_cache scratchpads**: not arch-guarded — it is the recipe's behaviour-preserving transformation
  (same NOC reads of the same byte counts to the same page ids, same barriers, same element reads; only
  the FIFO bookkeeping that synchronised nothing is gone, and the region moves from the DFB allocator to
  the scratchpad allocator in the same L1 range). WH/BH sentinel: the `fill_cache` tests in §6.

Every other line of the op is untouched. Confirm with the §6 BH/WH commands before merging; delete this
report before merging.

## Verification (independent review, 2026-09-10; no build, no test)

Checked against the declarations in the tree, not the recipe text. **No defect found; no code changed.**

1. **Scratchpad conversion** — `ScratchpadSpec{unique_id, size_per_node}` (`scratchpad_spec.hpp:53`) and
   `ScratchpadBinding{scratchpad_spec_name, accessor_name}` (`kernel_spec.hpp:166`, exported as a namespace alias at
   `:243`) match the initializers' field names, types and order. `Scratchpad<T>(const ScratchpadBindingToken&)`,
   `operator[]` (const, returns `T&`) and `noc_traits_t<Scratchpad<T>>::dst_args_type{offset_bytes}` are all in
   `tt_metal/hw/inc/api/scratchpad.h`; `Scratchpad<volatile T>` has precedent (scatter's `Scratchpad<volatile float>`
   passed by const-ref exactly as `virtual_seq_tile_id_to_physical_tile_id` now does). `scratch::` tokens are emitted
   by `genfiles.cpp:429` (JIT) and `emulated_program_runner.cpp:947` (emulator); the three token names equal the
   three `accessor_name`s. Byte sizes are the old `entry_size * num_entries` products. Removed locals
   (`page_table_data_format`, `batch_idx_data_format`, `batch_idx_arr` raw pointer, the three `*_wr_ptr`s,
   `page_table_ptr`) have no remaining use; every surviving local is used. Validator (`program_spec.cpp:454-513`)
   only requires `size_per_node != 0` and at least one binder — satisfied. `allocate_scratchpads`
   (`program.cpp:1748`) aligns every scratchpad base to the DRAM alignment, so the DRAM→L1 NOC reads keep the
   alignment the DFB allocator gave them.
2. **Gen2 compute config** — `ComputeGen2Config` declares `fpu_math_fidelity, sfpu_precision_mode,
   enable_32_bit_dest, double_buffer_dest, unpack_modes` (`compute_hardware_config.hpp:129`); the initializer names
   `enable_32_bit_dest` then `unpack_modes`, in declaration order, with a trailing comma. `device->arch()` is the
   accessor the same factories already use. `git diff -U0` touches no `ComputeGen1Config`/`emplace` line. On Gen2
   `UnpackToSrc` entries are always accepted (`program_spec.cpp:1067`).
3. **Semaphore CTAD** — `noc_semaphore.h:39` defaults to `<TENSIX, LOCAL_NONATOMIC>`; the deduction guide at `:461`
   yields `Semaphore<TENSIX, TOK_SCOPE>`; `ResolveSemaphoreScope` (`semaphore_scope.hpp:93`) returns
   `LOCAL_NONATOMIC` unconditionally on Gen1 and the token is emitted as `SemaphoreBindingToken<id, SemScope::X>`
   (`jit_build_settings.hpp:53`), so the Gen1 type is exactly what `Semaphore<>` spelled. No `Semaphore<` remains in
   the op's kernels. Kernels build as `-std=c++17` (`build.cpp:168`), which permits CTAD in the functional-cast
   temporary `Semaphore(sem::receiver).up(...)`.
4. **Precedent** — Scratchpad: `data_movement/copy/.../copy_default_row_major_program_factory.cpp` (same spec/binding
   idiom) and `repeat_higher_dim_tile.cpp` (Scratchpad as `async_read` destination). CTAD: `data_movement/sort`
   kernels. Gen2 compute: no other op on main uses the recipe's shape 4 (all use `to_compute_hardware_config` +
   `std::visit`, shape 3); the code follows the recipe's shape-4 template, except the variant local is named
   `compute_hw_config` because `compute_hw` was already the Gen1 local's name. Not a defect.
5. **Scope** — at the start of verification `git status --short` (tracked files) listed only the 11 paged_cache
   paths. By the end, tracked edits under `data_movement/{concat,reshape_view,slice}` had appeared from other
   agents' concurrent uplifts; none of them are part of this op's change and none were touched here.

Not verifiable here: no `compile_commands.json` entry exists for any ttnn TU (`build/` and `build_Release/` list only
CPM/third-party units), so a direct host compile-check was skipped.

## craq-sim run 2026-09-10

**Environment.** Quasar functional simulator (`TT_METAL_SIMULATOR=/localdev/vsuresh/qsr-sim/libttsim.so`, craq-sim,
32 worker cores in an 8x4 grid, ~1.1-1.3 kHz), branch `vsureshTT/quasar_uplift_round_2`, slow dispatch
(`TT_METAL_SLOW_DISPATCH_MODE=1`), `TT_METAL_FORCE_JIT_COMPILE=1`, private kernel cache
(`TT_METAL_CACHE=/localdev/vsuresh/tt-metal-cache-qsr-paged_cache`), host libs rebuilt from this tree once
(`qsr_rebuild`, after the factory change in fix 3 below). Each case was run twice: plain and with
`TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1`. All logs are under
`scratchpad/qsr_paged_cache/` (`t1*` fill, `t2*`/`t3*` update, `t4*`/`t6a` fused, `t5*` graph_ops).

**Test-harness blocker, worked around in temp copies (not an op issue).** Every test builds its tensors with
`ttnn.from_torch(..., device=mesh, layout=TILE_LAYOUT)`, which tilizes *on device* through the legacy
`ttnn.tilize` / `ttnn.tilize_with_val_padding` (`to_layout`) and dies on Quasar before `paged_*` runs:
`TT_FATAL kernel.hpp:450 "DataMovementKernel is not supported on Quasar. Use QuasarDataMovementKernel instead."`
(bf16 `tilize`) or `TT_FATAL program_spec.cpp:907 "KernelSpec 'compute' targets Gen2 (Quasar) but its
ComputeHardwareConfig holds a ComputeGen1Config"` (`tilize_with_val_padding`, the padded `[1,1,8,64]` input).
Temp copies (`test_tmp_paged_*_qsr.py`, deleted afterwards) replaced `U.to_tt` / `G.build_tensor` with
host tilize (`from_torch` without `device`) + `ttnn.to_device(host, mesh, memory_config=<same>)`; everything else
(shapes, memory configs, references, `_check_cache_written` postcondition) is the repo test's. Owner: the
llama32_1b_quasar test utilities / the `tilize` op family, not paged_cache.

### Results

| # | test | case | result | PCC / detail |
|---|---|---|---|---|
| 1 | `tests/ops/test_paged_fill_cache.py` (as-is) | seq128 | FAIL (harness) | `from_torch`→`ttnn.tilize`: "DataMovementKernel is not supported on Quasar" — before the op |
| 1 | same, host-tilize temp copy | seq128 / seq512 / seq1024 | **PASS** x3 | PCC 1.0 each (65536 / 262144 / 524288 non-zero cells, all expected); seq128 also PASS with LLK asserts |
| 2 | `tests/ops/test_paged_update_cache.py`, host-tilize copy, **before fixes** | batch1 | FAIL (silent) | PCC 0.0, cache all zero, no assert/hang (see fixes 2 and 3) |
| 2 | same, after fix 2 only | batch1 | FAIL | PCC 1.0 vs zero cache, but random-cache variant PCC 0.938: odd heads' tile row zeroed (see fix 3) |
| 2 | same, after fixes 2+3 | batch1 | **PASS** | PCC 1.0; random-cache variant PCC 1.0 (65536/65536); PASS again with LLK asserts |
| 2 | same | batch32 (32 cores = full sim grid, sequential-mode semaphore chain) | **PASS** | PCC 1.0; PASS with LLK asserts |
| 3 | `tests/ops/test_paged_fused_update_cache.py`, host-tilize copy (tiled fused factory, first Quasar run) | batch1 (K on core (0,0), V on (1,0)) | **PASS** | keys PCC 1.0, values PCC 1.0; PASS with LLK asserts |
| 3 | same | batch32 | SKIP (grid) | "height-sharded op needs 64 cores (32 shards at offset 32); device has 32" |
| 4 | `graph_ops/test_paged_fill_cache.py` as captured (bf8) | `00_128x8x32x64_bf8_int-dram` | FAIL (expected, dtype) | reaches the op: `TT_FATAL program_spec.cpp:1746 "DFB 'input' has data format 'Bfp8_b' which is not supported on architecture quasar"` |
| 4 | `graph_ops/test_paged_update_cache.py` as captured (bf8) | `00_128x8x32x64_bf8_int-dram` | FAIL (harness, before op) | `from_torch`→`tilize_with_val_padding`: "KernelSpec 'compute' targets Gen2 (Quasar) but ... ComputeGen1Config" (the bf16 `[1,1,8,64]` input); the op's own bf8 `cache` check is never reached |
| 4 | `test_tmp_paged_fill_cache_bf16_qsr.py` (sed bf8→bf16 only) | same | FAIL (harness) | `from_torch`→`ttnn.tilize` DataMovementKernel — before the op |
| 4 | `test_tmp_paged_update_cache_bf16_qsr.py` (sed only) | same | FAIL (harness) | same |
| 4 | both bf16 copies + host-tilize `build_tensor` | fill `[128,8,32,64]`+`[1,8,512,64]`, page_table `[1,16]` | **PASS** | `_check_cache_written` sentinel postcondition; PASS with LLK asserts |
| 4 | | update cache `[128,8,32,64]`, input `[1,1,8,64]` L1 height-sharded 1 core, `update_idxs_tensor` + `page_table [1,128]` (paged + index-tensor path) | **PASS** | sentinel postcondition; PASS with LLK asserts |

No hang, no watcher fault and no LLK assert fired in any run; the only device-side defect was silent (all-zero /
half-zero output), as the recipe warns for Quasar compute.

### Fixes applied in this run (all inside the op directory, all Quasar-gated)

1. *(from the uplift session, now verified on device)* fill_cache DM self-loop DFBs → `Scratchpad`s, Gen2
   `ComputeGen2Config` in the three compute factories, CTAD `Semaphore` in six DM kernels (§1 above). Without
   the Gen2 config the compute factories fail exactly as `tilize_with_val_padding` still does (row 4).
2. **`pack_init(dfb::out)` before the tilize** — `device/kernels/compute/update_cache.cpp`,
   `paged_fused_update_cache.cpp`, `paged_row_major_fused_update_cache.cpp`, `#ifdef ARCH_QUASAR`. Symptom: cache
   block written back all-zero (PCC 0.0 on a zero cache; with a random cache exactly the updated 32-row tile row
   of every head was zero). Cause: on Quasar the packer's destination ring is baked in by the last PACK init;
   `pack_untilize_init` (inside `compute_kernel_lib::untilize`) points it at `untilized_cache`, `tilize_init`
   has no PACK step on Quasar and the tilize helper's `pack_reconfig_data_format` only reprograms the format
   gasket, so the re-tilized tiles never reach `dfb::out` (fresh L1 zeros were stored). Same idiom as
   `experimental/quasar/binary_ng` compute kernels (`eltwise_utils_dfb.hpp`). WH/BH unchanged (their
   `tilize_init`/`pack_untilize_uninit` reprogram the packer).
3. **Single-buffer `untilized_cache` / `untilized_cache2` / `untilized_input` on Quasar** — the three compute
   factories (`num_interm_tiles = Wt` under `device->arch() == tt::ARCH::QUASAR`, else `2 * Wt`). Symptom after
   fix 2: odd heads' updated tile row came back zero except the freshly written row (random-cache variant PCC
   0.938 / 0.867). DPRINT evidence (`t2f_update_batch1_dprint.log`): the reader's `cache` ring alternates slots
   `0x44cbc0`/`0x44dbc0` with correct data on every head; the writer sees `untilized_cache` slot 0 (`0x44ebc0`)
   correctly untilized on even heads but slot 1 (`0x44fbc0`) all-zero on every odd head, and `out` faithfully
   carries those zeros. I.e. a `pack_untilize` into ring slot >= 1 of the aliased `untilized_cache/2` pair never
   lands there. The pipeline is strictly serial per head (compute cannot start `untilize(h+1)` before
   `tilize(h)`, which waits on the writer's republish of `h`), so the second slot is never in flight and
   single-buffering is behaviour-preserving; WH/BH keep the double buffer. Host change → rebuilt via
   `qsr_rebuild` (REBUILD_OK, `rebuild1.log`).

The suspected NoC L1→L1 loopback self-read in the update writers (§3 item 1) did **not** fire: the DPRINT run
shows the patched row landing in the untilized block and the batch-1/batch-32/fused rows all verify at PCC 1.0.
Left as is.

### Open blockers / flags (outside the op)

| symptom | evidence | owner |
|---|---|---|
| `pack_untilize` output into ring slot >= 1 of a double-buffered (here `alias_with`-paired) DFB does not land in that slot on Quasar craq-sim; slot 0 is fine | `t2e`/`t2f` logs above; `llk_pack_untilize` computes `base_l1 = tc_slots[tc_idx].wr_entry_idx * l1_index_per_entry` and hands it to `_llk_pack_untilize_set_dst_offset_` (`tt_metal/hw/ckernels/quasar/metal/llk_api/llk_pack_untilize_api.h:96-109`); either that offset, the alias lowering of `wr_entry_idx`, or the simulator's `PACR_UNTILIZE` dst offset is wrong. Note `experimental/quasar/halo` sidesteps the same shape by alternating two separate output DFBs. Op-side mitigation = fix 3; remove it once the root cause is fixed | LLK team (pack_untilize) / Metal 2.0 runtime (alias_with + tile-counter slot) — reproduce on emu/silicon before blaming the sim |
| Quasar `tilize_init` programs no PACK state; any kernel that switches the pack destination from a `pack_untilize` output to a tilize output needs an explicit `pack_init` (fix 2) | `tt_metal/hw/inc/api/compute/tilize.h:51-69` (Quasar branch: UNPACK+MATH only) | LLK / compute-API owners: consider a PACK re-init inside Quasar `tilize_init` so ops do not need the guard |
| `ttnn.from_torch(device=..., layout=TILE)` cannot run on Quasar for bf16 (`ttnn.tilize` = legacy `DataMovementKernel`; `tilize_with_val_padding` = `ComputeGen1Config`) — blocks every generated/graph test's tensor setup, not just paged_cache | rows 1 and 4 | `tilize` op family / llama32_1b_quasar test utilities (`op_utils.to_tt`, `graph_case.build_tensor`): tilize on host + `to_device` for Quasar |
| captured `paged_fill_cache` / `paged_update_cache` cases use a `BFLOAT8_B` cache and `paged_fill_cache` also a bf8 input; `is_supported_quasar()` rejects Bfp8_b (`program_spec.cpp:1746`) | row 4 | llama32_1b Quasar model team (cache dtype); confirmed bf16 variant passes end to end |
| batch-32 fused case needs 64 cores (K and V grids must not overlap); the sim has 32 | row 3 skip | test-only, expected grid skip |

### Commands to reproduce

```bash
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh paged_cache
# 1) hand-authored op tests: make a temp copy that tilizes on host (the repo test's U.to_tt fails on Quasar, see above)
D=models/experimental/llama32_1b_quasar/tests/ops
for n in paged_fill_cache paged_update_cache paged_fused_update_cache; do
  python3 - $D/test_$n.py $D/test_tmp_${n}_qsr.py <<'PY'
import sys,re; s=open(sys.argv[1]).read()
helper='''
def _to_tt(t, mesh, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper="replicate", shard_dim=None):
    host = ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh))
    return ttnn.to_device(host, mesh, memory_config=memory_config)
'''
s=s.replace('U.to_tt(','_to_tt(')
s=re.sub(r'(from models\.experimental\.llama32_1b_quasar\.tests\.ops import op_utils as U\n)', r'\1'+helper, s, 1)
open(sys.argv[2],'w').write(s)
PY
done
qsr_test timeout 2400 ./python_env/bin/python -m pytest $D/test_tmp_paged_fill_cache_qsr.py -v -s            # 3 PASS
qsr_test timeout 2400 ./python_env/bin/python -m pytest $D/test_tmp_paged_update_cache_qsr.py -v -s          # 2 PASS
qsr_test timeout 2400 ./python_env/bin/python -m pytest $D/test_tmp_paged_fused_update_cache_qsr.py -v -s    # 1 PASS, batch32 SKIP
# 2) graph_ops as captured (expected: bf8 format TT_FATAL for fill; tilize_with_val_padding Gen1 TT_FATAL for update)
G=models/experimental/llama32_1b_quasar/tests/graph_ops
qsr_test timeout 2400 ./python_env/bin/python -m pytest $G/test_paged_fill_cache.py $G/test_paged_update_cache.py -v -s
# 3) bf16 copies next to them, with host-tilize build_tensor (see the "TEMP Quasar craq-sim workaround" block in the run logs)
for n in fill_cache update_cache; do sed 's/"BFLOAT8_B"/"BFLOAT16"/g' $G/test_paged_$n.py > $G/test_tmp_paged_${n}_bf16_qsr.py; done
# append to each copy: a G.build_tensor replacement that does ttnn.from_torch(data, dtype, layout, mesh_mapper=replicate) on host
# followed by ttnn.to_device(host, mesh, memory_config=G.build_memory_config(spec["mem"], mesh) or DRAM), then:
qsr_test timeout 2400 ./python_env/bin/python -m pytest $G/test_tmp_paged_fill_cache_bf16_qsr.py $G/test_tmp_paged_update_cache_bf16_qsr.py -v -s
rm $D/test_tmp_paged_*_qsr.py $G/test_tmp_paged_*_bf16_qsr.py
# repeat any case with: TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1
```
