# QUASAR_UPLIFT_REPORT — `nlp_create_qkv_heads_decode`

- **Op:** `ttnn.experimental.nlp_create_qkv_heads_decode`
  (`ttnn::experimental::NLPCreateQKVHeadsDecodeDeviceOperation`)
- **Directory:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/`
- **Driving test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_create_qkv_heads_decode.py`
  — one signature, 308 captured calls: input `[1,1,1,3072]` bf16 TILE, INTERLEAVED L1;
  `num_heads=32, num_kv_heads=8, overlap_qk_coregrid=True`; HEIGHT_SHARDED L1 outputs. With an
  interleaved input this routes (via `select_program_factory`,
  `device/nlp_create_qkv_heads_decode_device_operation.cpp:13-24`) to
  **`NLPCreateQKVHeadsDecodeInterleavedProgramFactory`**.
- **Date:** 2026-09-02
- **Audited state:** the **post-#54633-merge** tree (branch merge of
  `vsureshTT/Metal2_port_nlp_create_qkv_heads_decode_v2` into `vsuresh/quasar-porting-recipe`).
  This report **supersedes and overwrites** the 2026-09-01 RED report, which audited the
  pre-merge (`create_descriptor`) state and is stale.
- **Mode:** static audit + statically-determinable in-place fixes. No builds and no device runs in
  this session (recipe §9: user runs all builds/tests); §7–§8 runtime-symptom fixes were applied
  only where the defect is statically provable from the source, everything runtime-conditional is
  recorded as considered/deferred.

## Status: **GREEN** (driving-test path uplifted; 2 statically-required fixes applied)

Gate (`quasar_porting.md` §1 step 1) passes on the whole op: all three program factories are
`create_program_artifacts` → `ProgramArtifacts` with `dfb::`/`args::`/`tensor::` bindings, and all
three kernels are on the device-2.0 API (`api/dataflow/*` includes, `Noc`, `DataflowBuffer`,
`TensorAccessor`, `get_arg(args::…)`, `get_vararg`; no `dataflow_api.h`-legacy `cb_*`/`noc_async_*`
free functions, no positional `get_arg_val`, no address-RTA `TensorAccessorArgs`).

Two statically-determinable Gen2 defects were found **on the driving test's factory** and fixed in
place; the two sharded factories (off the driving-test path) carry analogous findings recorded as
deferred items below.

---

## Files changed (all inside the op directory)

| File | Change | Reason |
|---|---|---|
| `device/nlp_create_qkv_heads_decode_interleaved_program_factory.cpp` | `READER_SCRATCH`/`WRITER_SCRATCH` self-looped DFBs → `ScratchpadSpec`s (+ conditional `spec.scratchpads` registration, `ScratchpadBinding` per instance); `Q_OUT`/`K_OUT`/`V_OUT` borrowed DFBs → `TensorBinding`s on the existing `Q/K/V_OUT_TENSOR` parameters; dead `q/k/v_num_tiles` locals removed | Fix 1 + Fix 2 below |
| `device/kernels/reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` | `DataflowBuffer(dfb::aligned_scratch).get_write_ptr()` → `Scratchpad<uint8_t>(scratch::aligned_scratch).get_base_address()`; `DataflowBuffer(dfb::{q,k,v}_out).get_write_ptr()` → `LocalTensorAccessor<uint8_t>(tensor::{q,k,v}_out).get_bank_base_address()` (6 sites); includes updated (`api/scratchpad.h`, `api/tensor/local_tensor_accessor.h` added; `api/dataflow/dataflow_buffer.h` dropped — no `DataflowBuffer` remains) | Fix 1 + Fix 2 below |

No file outside the op directory was touched. The op's namespace and directory are unchanged;
nothing tempted a move or rename.

### Fix 1 — DM self-loop scratch DFBs → `Scratchpad` (Gen2-rejected shape)

The pre-merge audit predicted this exactly. The aligned path (`use_aligned_path`: DRAM input with
`sub_tile_line_bytes < dram_alignment`) staged reads through two per-RISC scratch DFBs, each bound
**PRODUCER + CONSUMER by one DM kernel instance** — the DM self-loop Quasar's legalizer rejects
(`dm_self_loop_dfbs.md`). Survey per `sync_free_dfbs.md`/`dm_self_loop_dfbs.md`: the kernel made
**zero FIFO calls** on the handle (only one `get_write_ptr()` grab at the base, before any advance),
`borrowed_from` unset, no `dfb_run_overrides`, no `get_entry_size()`, placeholder
`data_format_metadata` (`Float16_b`, comment said so; nothing consults it) — the pure sync-free
shape, so this is the *style*-pass conversion (the safe sibling of the dm-self-loop pass; no index
or wrap survives because the pointer was never re-read after an advance).
`size_per_node = dram_alignment * (head_tiles + 1)` carries the DFB's `entry_size * num_entries`
verbatim. Registration and bindings keep the `use_aligned_path` guard (an unbound `ScratchpadSpec`
is a `TT_FATAL` at program creation). Same conversion as this branch's reference
(`rotary_embedding_llama` `ZERO_DFB` → `ZERO_SCRATCH`).

Gen1 behavior: identical by construction — the FIFO machinery was never invoked; only the L1
allocation order shifts (scratchpads allocate from the same region DFBs do), which nothing
functional depends on.

### Fix 2 — borrowed sync-free q/k/v out DFBs → `LocalTensorAccessor` (uncached-pointer-to-NoC)

The q/k/v output DFBs (`borrowed_from = Q/K/V_OUT_TENSOR`, endpoints self-described in the factory
comment as "cosmetic") were only ever used as `get_write_ptr()` address sources for
`noc.async_read` **destinations** (`CoreLocalMem<uint32_t>(get_write_ptr() + offset)`), on the
direct-read path the driving test takes and on the aligned path alike. This is the
**uncached-DFB-pointer-to-NoC** pattern: on Quasar DM, `DataflowBuffer::get_write_ptr()/get_read_ptr()`
return the **uncached** L1 alias (`api/dataflow/dataflow_buffer.h:333-338`,
`+ MEM_L1_UNCACHED_BASE` since #52769), and "NOC APIs do not accept uncached addresses"
(ibid.:381 — the cached getters are private, exposed only to DFB `noc_traits`). Statically provable
breakage on the driving test's exact path, so it was fixed rather than deferred.

The fix is the canonical `sync_free_dfbs.md` disposition for a sync-free DFB with `borrowed_from`
set: bind the borrowed tensors directly (`TensorBinding`s reusing the existing
`Q/K/V_OUT_TENSOR` parameters, accessor names `q_out`/`k_out`/`v_out` preserved) and read the shard
base via `LocalTensorAccessor<uint8_t>::get_bank_base_address()` — a plain (cached) L1 address on
both generations. `T = uint8_t` per the recipe's table (the kernel never indexes elements; it only
hands byte addresses to NOC transfers / `tt_memmove`). Sync-free criterion verified against the
whole binding set: both binding kernel instances are the one shared source, and a grep +
read-through of every `dfb_q_out|dfb_k_out|dfb_v_out|dfb::q_out|dfb::k_out|dfb::v_out` hit shows
only `get_write_ptr()` — no FIFO/credit calls, no helper/guard/template the handle is passed into,
no `get_entry_size()`, no `data_format_metadata` consumer.

Gen1 behavior: identical — on Gen1 `L1_UNCACHED_OFFSET == 0`, so `get_write_ptr()` on a
never-pushed borrowed DFB *is* the tensor's node-local shard base, i.e. the exact address
`get_bank_base_address()` now returns; same transfers, same order, same addresses. Side benefit:
three DFB ids freed (the id budget is tight on Gen2).

Note: `LocalTensorAccessor` `static_assert`s the bound tensor is L1-resident. The op validates the
outputs HEIGHT_SHARDED (`validate_on_program_cache_miss`); a hypothetical DRAM-height-sharded
output memconfig would now fail at kernel JIT instead of misbehaving — the borrowed-DFB overlay
never supported that either.

---

## Audit checklist (quasar_audit.md + quasar_porting.md §7–§12)

Per-factory verdicts: **interleaved GREEN (fixed)**; **sharded / sharded_subcoregrid GREEN-with-
deferrals** — Metal 2.0 and no hard RED-stop on their default paths, but they carry the same
uncached-pointer pattern and (batch_offset path only) a construct needing an owner decision; see
Deferred.

| Check | Result |
|---|---|
| DM self-loop DFBs | **Interleaved: 2 found, converted (Fix 1).** Sharded/subcoregrid: `BATCH_OFFSET_DFB` is a DM self-loop **with real FIFO calls** (`reserve_back`/`push_back`) and `allow_instance_multi_binding` — two co-resident kernel instances drive one per-node instance. Neither pass covers it (FIFO calls ⇒ not sync-free; two kernels per node ⇒ `Scratchpad`'s 1:1 node rule fails; `dm_self_loop_dfbs.md` step 4 also stops on the multi-kernel shape). Deferred — see below. Only exists when `batch_offset` is passed (paged-attention path); the driving test never hits it. |
| Borrowed / sync-free DFBs | Interleaved: q/k/v converted (Fix 2). Sharded/subcoregrid: same borrowed cosmetic-endpoint q/k/v shape, same `get_write_ptr()`→NOC-dst pattern — deferred (off the driving-test path). Capacity checks moot where converted; unconverted borrows unchanged from the reviewed port. |
| Gen1-only hardcoded `hw_config` | **None.** All six kernel specs (2+4) take `create_reader/writer_datamovement_config(arch)` — gen2_hardware_configs **shape 1** ("no work; do not touch"). No compute kernels in this op, so no `unpack_modes` / `TODO(#52269)` marker applies anywhere (per the doc, that absence is meaningful, not an omission). No `std::get<…Gen1Config>` / `std::get_if` / `holds_alternative` sites. |
| Non-zero-init semaphores | None — the op declares no semaphores at all. |
| `fifo_page_size` / `get_local_cb_interface` | None in the op. |
| Uncached-DFB-pointer-to-NoC | Interleaved: fixed (Fix 2). Sharded/subcoregrid kernels: present (q/k/v `get_write_ptr()` → `CoreLocalMem` NOC dst; also `BATCH_OFFSET` `get_write_ptr()` → NOC dst) — deferred with the factories. |
| uint16/uint32 format branches | None. Input is validated FLOAT32/BFLOAT16; kernels branch on `ELEMENT_SIZE` bytes only. Nothing to guard. |
| `data_format_metadata` validity | All remaining DFBs (sharded factories) carry valid formats. The interleaved factory now declares **no DFBs**; the dropped placeholder `Float16_b` and q/k/v formats had no consumer (checked before dropping, per both passes). |
| `evil_set_*` | None. |
| Implicit-sync disables (`disable_dfb_implicit_sync_*`) | None — nothing set, nothing propagated. |
| `compute_kernel_hw_startup` / tilize / DEST rules (§7) | N/A — DM-only op, no compute kernels. |
| NoC / multicast (§11) | No multicast anywhere. Interleaved kernel: `TensorAccessor` page reads + local L1 writes only. Sharded kernels: unicast `UnicastEndpoint` reads from input-shard cores via vararg NoC coordinate tables — no NOC0/NOC1 directional trickery, no reverse rectangles. Degenerate-grid concerns don't apply (no `+1` corner math). |
| Semaphore raw-addr reads, `MEM_ZEROS_BASE`, `async_write_zeros` | None used. |
| Vararg limits (§ vararg API) | Sharded kernels use read-only `get_vararg` with a fixed per-KernelSpec count (`num_x + num_y`) — compliant. |
| `opt_level` | All kernel specs leave `compiler_options.opt_level` at the M2 default (O2), which equals the resolved legacy DM default — correct for DM kernels; not an uplift edit either way. |

### §7–§8 gotchas considered but not applied (and why)

- **§8.1 `common.hpp`-in-DM-build breaker:** the interleaved kernel includes
  `ttnn/operations/data_movement/common/kernels/common.hpp` (for `tt_memmove`). Already
  Quasar-guarded on main (`#if !defined(ARCH_QUASAR)` around `ckernel.h`, Quasar drain fallback) —
  no action. Its `TODO(ARCH_QUASAR)` notes the CPU-`memmove` fallback is not fully cache-coherent
  on Quasar; the fallback only triggers when 16-B parities mismatch, and the aligned path is
  DRAM-input-only (not the driving test's config) — runtime-reactive if ever hit.
- **§8.2 hangs / credit stalls, §8.3 wrong-output rows:** reactive by definition; no device run
  available this session. None of their static signatures (beyond the two fixed) appear in the op.
- **Scoped locks (`scoped_read/write_lock`)** instead of raw addresses: not adopted — Fix 2 removes
  the DFB pointer getters from this op's hot path entirely, which is the stronger end state.

---

## Deferred / follow-up items

1. **Sharded + subcoregrid factories: `BATCH_OFFSET_DFB` needs an owner decision** (paged-attention
   path only, `batch_offset.has_value()`). It is a DM self-loop with real FIFO calls whose one
   per-node instance is driven by **two co-resident kernel instances**
   (`allow_instance_multi_binding`) — Gen2 rejects DM self-loops, `Scratchpad` cannot take a
   two-kernels-per-node buffer (1:1 rule; `sync_free_dfbs.md` "When you can't convert a shared
   scratch DFB"), and `dm_self_loop_dfbs.md` stops on FIFO-called multi-kernel shapes. What the
   kernels do with it: each instance NOC-reads one page of the `batch_offset` tensor into it and
   reads back one scalar (`reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:43-57`). The clean
   redesign is probably per-instance staging (two scratchpads, one per instance — each instance
   already re-reads the scalar itself), but that changes the declared sharing structure, so it is
   the op-owner's call, not this pass's. Per the RED-stop taxonomy this would be RED **if** the
   uplift targeted the sharded batch_offset path; the driving test never reaches it.
2. **Sharded + subcoregrid kernels: same uncached-`get_write_ptr()`-to-NoC pattern on q/k/v (and
   `BATCH_OFFSET`)** as Fix 2 — `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:100/159/219`
   and `..._on_subcoregrids.cpp:96/150/205`. The same borrowed-sync-free → `LocalTensorAccessor`
   conversion applies to q/k/v there (the factories already declare the output `TensorParameter`s),
   but those factories are off this uplift's driving-test path and their kernels are separate
   sources, so the conversion was not applied blind; do it when a Quasar test drives the
   sharded-input path (together with item 1).
3. **`tt_memmove` CPU-fallback coherency on Quasar** (shared header, outside the op — see
   considered-not-applied above): flag rides with the existing `TODO(ARCH_QUASAR)` in
   `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp`; nothing to do in this op.

---

## Parity claim (structural — no device runs this session)

WH/BH take a behaviorally identical path. No change is arch-branched because none needs to be:
both fixes are the canonical behavior-preserving conversions
(`sync_free_dfbs.md`: "Results, numerics and observable behaviour are identical"), argued site by
site above — same NOC transfers, same order, same addresses on Gen1 (borrowed-DFB base ≡ tensor
shard base; scratch base re-derived identically, kernel still rounds it up). The only observable
Gen1 difference is L1 allocation order (scratchpads/DFB pool) and three freed DFB ids, neither of
which anything functional depends on. The sharded and subcoregrid factories and their kernels are
**textually untouched**.

**Confirm with (user runs; kernels changed → force JIT, and purge `~/.cache/tt-metal-cache`
between pre/post runs):**

```bash
# BH / WH parity — existing op suite (covers interleaved + sharded + batch_offset variants)
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_decode.py

# BH / WH parity — the driving graph-trace case
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_create_qkv_heads_decode.py

# Quasar (craq-sim, per the simulator runbook env)
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_create_qkv_heads_decode.py
```

Run Quasar both with `TT_METAL_LLK_ASSERTS` on and off (recipe §9) — moot for compute asserts in a
DM-only op, but the DFB/watcher assert paths still differ.

---

## Definition-of-done checklist (recipe §10)

- [x] Uplifted in place — existing directory + namespace; nothing copied to `experimental/quasar/`, no `::qsr`.
- [x] Factories on `create_program_artifacts`/`ProgramArtifacts`; kernels on `dfb::`/`args::`/`tensor::`/`scratch::`.
- [x] `opt_level` matches resolved legacy values (DM default O2).
- [x] Every remaining DFB has valid `data_format_metadata`; no `fifo_page_size` reads.
- [x] Sync-free / DM self-loop DFBs on the driving path converted to `Scratchpad`/`LocalTensorAccessor`; remaining (sharded batch_offset) recorded as an owner-decision deferral.
- [x] No implicit-sync disables; no non-zero-init semaphores.
- [ ] BH and WH pass unchanged — **user-run** (commands above).
- [ ] Quasar builds and runs — **user-run** (commands above).
- [x] No DIAG/debug leftovers.
- [x] Genuine blockers flagged (deferred items 1–2 feed back into `quasar_audit.md`).
- [x] This report; RED-stop conditions checked — none applies to the driving-test path.
