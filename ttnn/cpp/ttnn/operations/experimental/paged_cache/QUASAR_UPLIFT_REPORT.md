# QUASAR_UPLIFT_REPORT — experimental/paged_cache

**Op family:** `ttnn/cpp/ttnn/operations/experimental/paged_cache/`
**Recipe:** `docs/source/ttnn/ttnn/ai/quasar_porting.md` + canonical `.../metal_2.0/ai/{audit/quasar_audit,post_port/style/sync_free_dfbs,post_port/semantic/{dm_self_loop_dfbs,gen2_hardware_configs}}.md`
**Context:** Written after PR #54747 ("Port paged_cache single-device update_cache + fill_cache factories") was merged into `vsureshTT/llama_quasar_uplift`. The port is **single-device only** — the mesh-workload factories and the fused_update_cache op remain legacy. This report overwrites the pre-port audit.
**No build / no test run** (per task): parity is argued **structurally** below; exact BH/WH/Quasar commands are in the last section.

---

## Status: GREEN (in place) with one required, deferred Quasar prerequisite

- **1 file changed** — `update_cache` compute gets its Gen2 hardware config (arch-branched; WH/BH byte-identical).
- **1 required Quasar prerequisite deferred** — the three `fill_cache` writer **DM self-loop DFBs** must be converted to `Scratchpad` (`dm_self_loop_dfbs.md`). This is an *unconditional* semantic rewrite of shared WH/BH kernel + factory source, so it is **not** `ARCH_QUASAR`-guardable and cannot be validated in a no-build/no-test pass — deferred to a dedicated, test-validated change (fully scoped below). Until it lands, `fill_cache` will not run on Quasar (Gen2 legalizer rejects a DM self-loop DFB).
- No non-zero-init semaphores; no `fifo_page_size` / `get_local_cb_interface` / `evil_set_*` / `disable_*_implicit_sync` / `MEM_ZEROS_BASE` / L2-cache-flush / uint16/UInt32 **kernel** format branches anywhere in the metal2 kernels.

---

## Per-factory gate + verdict

| Factory | Concept | Gate | Quasar verdict |
|---|---|---|---|
| `PagedUpdateCacheProgramFactory` (single-device) | `create_program_artifacts` → `ProgramArtifacts`, `dfb::`/`args::`/`tensor::`, device-2.0 kernels | **M2 GREEN** | **GREEN** — Gen2 compute config applied; no self-loop/sync-free/semaphore/format debt |
| `PagedFillCacheProgramFactory` (single-device) | `create_program_artifacts` → `ProgramArtifacts`, device-2.0 kernels | **M2 GREEN** | **RED (deferred)** — 3 writer DM self-loop DFBs need `Scratchpad` conversion first |
| `PagedUpdateCacheMeshWorkloadFactory` (multi-device) | `create_descriptor` → `ProgramDescriptor`, `CBIndex::c_*`, legacy kernels | **legacy** | **RED** — not ported (single-device-only port); framework-blocked (per-mesh-coord program ≠ one ProgramSpec). Out of scope for the uplift. |
| `PagedFillCacheMeshWorkloadFactory` (multi-device) | `create_descriptor` → `ProgramDescriptor`, `CBIndex::c_*`, legacy kernels | **legacy** | **RED** — same as above |
| `PagedTiledFusedUpdateCacheProgramFactory` (fused_update_cache) | `create_descriptor` → `ProgramDescriptor`, `CBIndex::c_*` | **legacy** | **RED** — fused op was **not** ported by #54747. Run the Metal 2.0 port first (`ai/port/metal2_port.md`) before any Quasar uplift. |
| `PagedRowMajorFusedUpdateCacheProgramFactory` (fused_update_cache) | `create_descriptor` → `ProgramDescriptor`, `CBIndex::c_*` | **legacy** | **RED** — same as above |

The two `*MeshWorkloadFactory` bodies and both fused factories are intentionally untouched: they are still `create_descriptor`/`ProgramDescriptor` with `tt::CBIndex::c_*` and legacy kernel sources — a RED-stop "not Metal 2.0 on Gen1 yet." The single-device M2 factories are what this uplift audits.

---

## Files changed

| File | Change | Reason |
|---|---|---|
| `device/update_cache/paged_update_cache_program_factory.cpp` | Renamed the hand-written `ComputeGen1Config compute_hw` → `compute_gen1` (values/order byte-identical) and added `ComputeHardwareConfig compute_hw = compute_gen1; if (device->arch() == tt::ARCH::QUASAR) compute_hw = ComputeGen2Config{ .enable_32_bit_dest = …, .unpack_modes = … };` with the `TODO(#52269)` marker. `.hw_config = compute_hw` now carries the arch-appropriate variant. | `gen2_hardware_configs.md` **shape 4, compute**. The port authored only a Gen1 config; a compute kernel whose `hw_config` holds only a Gen1 config **cannot run on Quasar**. This op deliberately avoids `to_compute_hardware_config` (the helper's high-performance defaults would silently flip knobs the legacy op left at descriptor defaults), so the fix is the hand-copied Gen2 branch, not the helper. Copies exactly the two fields the Gen1 config sets (`enable_32_bit_dest`, `unpack_modes`); leaves `fpu_math_fidelity`/`sfpu_precision_mode`/`double_buffer_dest` at their identical defaults; drops `bfp_pack_precision_mode` (no Gen2 equivalent); never sets `enable_2x_src_register`. |

Op namespace + directory unchanged (`ttnn::experimental::prim`, in place). Nothing tempted a move/rename.

### Parity claim (structural, WH/BH unchanged)
The only functional change is an `if (device->arch() == tt::ARCH::QUASAR)` branch that is **never taken on WH/BH**. The Gen1 initializer and its three conditional `unpack_modes.emplace` calls are textually preserved (only the local's name changed), and `ComputeHardwareConfig compute_hw = compute_gen1;` reproduces the prior value on the non-Quasar path. So WH/BH take the original path unchanged; the Gen2 struct is compiled (type-checked) but never constructed off-Quasar. Confirm with the BH/WH commands below.

---

## Audit findings per §7–§12 / quasar_audit.md

**quasar_audit.md check 1 (device-side CB/DFB redesign / self-loop):**
- `update_cache` — no DM self-loops. `UC_INPUT_DFB` is a **borrowed** (`borrowed_from = UC_INPUT_TENSOR`) *cross-kernel* FIFO (reader PRODUCER → compute CONSUMER), a real FIFO, left as-is. `UC_UNTILIZED_CACHE`/`UC_UNTILIZED_CACHE2` are an aliased pair driving a genuine compute↔writer in-place handshake (real FIFOs). `UC_OUTPUT`, `UC_UNTILIZED_INPUT`, `UC_CACHE`, `UC_INDEX`, `UC_PAGE_TABLE` are all cross-kernel FIFOs. **No sites.**
- `fill_cache` — `FC_INPUT_DFB` is a normal reader→writer FIFO. **`FC_PAGE_TABLE_DFB`, `FC_BATCH_IDX_DFB`, `FC_VALID_SEQ_LEN_DFB` are DM self-loops** (the writer, a DM kernel, binds each as both PRODUCER and CONSUMER; the factory comments say so explicitly). Each calls `reserve_back(1)` (one of the four FIFO calls) → **`dm_self_loop_dfbs.md` sites** → convert to `Scratchpad`. **Deferred** (see below).

**quasar_audit.md check 2 (non-zero-init semaphores):** `update_cache`'s `UC_SEQUENTIAL_MODE_SEM` (`SemaphoreSpec`, no initial value → default 0; legacy descriptor set `initial_value = 0`). Zero-init → ports fine. `fill_cache` has no semaphores. **GREEN.**

**§4 / data_format_metadata validity:** every M2 DFB carries a concrete format (no `Invalid`). Note `FC_VALID_SEQ_LEN_DFB` = `UInt32` and `FC_BATCH_IDX_DFB` = `UInt32` (default) / tensor dtype — a format Quasar does not have (Int32-only). These are **DM-only raw-access** buffers (no LLK/compute consumes the format), so the format is inert at the DM layer; and the deferred self-loop→`Scratchpad` conversion **drops `data_format_metadata` entirely**, disposing of the concern. `update_cache` index/page-table DFBs are already `Int32`; cache/input/interm are bf16/fp32. No kernel-side uint16/uint32 format branch exists to guard.

**§5 / `get_entry_size` vs `fifo_page_size`:** kernels read sizes via `dfb.get_tile_size()` on real FIFOs; **no** `get_local_cb_interface().fifo_page_size` anywhere. GREEN.

**§7 gen2 hw_configs:** `update_cache` compute was the sole Gen1-only hardcoded config → **fixed** (above). All reader/writer configs use the arch-agnostic `create_reader/writer_datamovement_config(arch)` helper (shape 1 — no work).

**§7 implicit sync / §8 double-count:** no `disable_dfb_implicit_sync_for_all` / `disable_implicit_sync_for` set — Gen2 implicit-sync default is relied on. GREEN.

**§8.3 uncached-DFB-pointer-to-NoC (typecast fix pattern):** no `flush_l2_cache_range` / `invalidate_l2_cache_range` / manual-flush code in these kernels; NoC transfers go through `TensorAccessor` / `CoreLocalMem`. Nothing to fix.

**§7 `evil_set_*` / ring-rewind:** not used. **§11 NoC/multicast:** the `update_cache` share-cache path uses a point-to-point `noc_semaphore` inc to a single neighbor core (not multicast, no NOC0/NOC1 directional trick). No degenerate-grid mcast corner. GREEN.

**§7 `compute_kernel_hw_startup` once / tilize pack-config / DEST-wrap (update_cache compute):** these are **reactive** (§7–§8) items — apply only when a symptom fires on a device run, which this pass does not perform. The compute kernel uses `pack_untilize` + `tilize`; not statically fixable without a Quasar run. Left for reactive bring-up.

---

## Deferred / follow-up items

1. **[REQUIRED for Quasar] Convert the three `fill_cache` writer DM self-loop DFBs to `Scratchpad`** — `dm_self_loop_dfbs.md`. Sites: `FC_PAGE_TABLE_DFB`, `FC_BATCH_IDX_DFB`, `FC_VALID_SEQ_LEN_DFB`, all bound PRODUCER+CONSUMER on the single DM `writer` kernel (`writer_fill_cache_interleaved_metal2.cpp`).
   - **Why deferred here:** the conversion is an *unconditional* rewrite of shared WH/BH kernel + factory source (delete the `DataflowBufferSpec` + both `DFBBinding`s, add a `ScratchpadSpec`/`ScratchpadBinding`, replace the kernel's `DataflowBuffer` + `reserve_back`/`get_write_ptr` with a `Scratchpad`). It is behavior-preserving but **not** `ARCH_QUASAR`-guardable, so it falls outside this pass's "each fix guarded/arch-branched so WH/BH unchanged" bound, and it is a semantic pass that must be **test-validated** (which a no-build/no-test pass cannot do).
   - **Scoping (turnkey — every stop-check passed):** not borrowed (`borrowed_from` unset on all three); no `dfb_run_overrides`; uses confined to the covered list (`reserve_back` + `get_write_ptr()` + the address as a NOC-read destination via `CoreLocalMem<uint32_t>(get_write_ptr())`); **no** off-list method (`get_entry_size`, `pages_reservable_at_back`, `async_write_zeros`, multicast) on these handles; single-entry in effect (the write pointer never advances — no `push_back` — so `wr` stays 0: no stride, no wrap). `size_per_node` = `entry_size * num_entries` (`page_table_stick_size * 1`, `batch_idx_stick_size * batch_idx_num_elements`, `valid_seq_len_stick_size * 1`). `data_format_metadata` is consulted nowhere (raw pointer / NOC only) → drops cleanly, which also removes the `UInt32` formats (§ above). `batch_idx`/`valid_seq_len` are conditionally bound (`USE_BATCH_IDX_TENSOR` / `USE_VALID_SEQ_LEN`) → the `ScratchpadSpec` registration + `ScratchpadBinding` carry the same guard. Element reads (`batch_idx_arr[...]`, `page_table_ptr[...]`, `*valid_dfb_wr_ptr`) are `volatile` → keep `volatile` in `Scratchpad<T>`'s `T`.

2. **[NOT this uplift] Port the mesh-workload path and fused_update_cache to Metal 2.0.** The two `*MeshWorkloadFactory` bodies (still `ProgramDescriptor`; framework-blocked — per-mesh-coordinate program cannot be one `ProgramSpec`) and both `fused_update_cache` factories (`create_descriptor`/`CBIndex`, unported) are RED. They require the base Metal 2.0 port (`ai/port/metal2_port.md`) before any Quasar uplift.

3. **[reactive, needs a Quasar run] `update_cache` compute bring-up** — `compute_kernel_hw_startup`-once, tilize pack-config (`PACR0_TILE_INC` / `0x19`), DEST-wrap. Not statically determinable; chase on-device per §8.

4. **[perf, tracked] `TODO(#52269)`** left on the Gen2 compute config — `unpack_modes` copied verbatim from Gen1, not yet Quasar-optimized. Not a correctness item.

---

## Test commands (user runs; order BH → WH → Quasar)

**Parity (must be unchanged by this diff — the Gen1 path is byte-identical):**
```
# Blackhole / Wormhole
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_update_cache.py
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_fill_cache.py
# plus the op's mainline suites for regression:
pytest tests/ttnn/unit_tests/operations/test_paged_update_cache.py
pytest tests/ttnn/unit_tests/operations/test_paged_fill_cache.py
```

**Quasar (emulator):**
```
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_update_cache.py
# update_cache should now build its compute kernel on Quasar (Gen2 hw_config present).
# fill_cache stays RED on Quasar until deferred item #1 lands (DM self-loop DFB rejection).
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_fill_cache.py
```
Start with `TT_METAL_LLK_ASSERTS` **on**; re-run with it unset. DPRINT needs `unset TT_METAL_LLK_ASSERTS` + the `DPRINT("fmt {}", args)` form. Purge `~/.cache/tt-metal-cache` between baseline and post-port runs.
