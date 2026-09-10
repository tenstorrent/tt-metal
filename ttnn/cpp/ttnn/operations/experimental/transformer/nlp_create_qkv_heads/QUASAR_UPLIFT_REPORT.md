# Quasar Uplift Report: `nlp_create_qkv_heads`

**Date:** 2026-09-10
**Branch:** `vsureshTT/quasar_uplift_round_2` (HEAD `0c5cb1f6d82`)
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/`
**Recipe:** `quasar_porting.md` (+ `ai/audit/quasar_audit.md`, `ai/audit/cb_dfb_quasar_audit_helper.md`,
`ai/post_port/{style/sync_free_dfbs,semantic/dm_self_loop_dfbs,semantic/gen2_hardware_configs}.md`)
**Quasar model test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_create_qkv_heads.py`
(one case: `[1,1,1024,3072]` bf16 TILE DRAM-interleaved, `num_heads=32, num_kv_heads=8,
transpose_k_heads=False` -> **Interleaved** factory, no compute kernel, no KV tensor, `kv_tied=False`)

---

## Status: RED — STOP (not Metal 2.0 on Gen1 yet)

This is RED-stop condition #1 of `quasar_porting.md` §1: *"Not Metal 2.0 on Gen1 yet — factory still
`create_descriptor`/`ProgramDescriptor`. Do the Metal 2.0 port first."* The Quasar uplift starts from an
already-Metal-2.0 op; this op has not had that port. **No source file was edited.** The only file written
is this report.

### Evidence (current working tree == `origin/main`, `git diff --stat origin/main -- <op>` is empty)

| Layer | What is there today | What the uplift requires |
|---|---|---|
| Factory API | `Interleaved::create_descriptor` / `Sharded::create_descriptor` -> `tt::tt_metal::ProgramDescriptor` (`device/nlp_create_qkv_heads_program_factory.cpp:84, 503`; header `:42, :56`) | `create_program_artifacts` -> `ProgramArtifacts` |
| Buffers | `CBDescriptor` with numeric `buffer_index` 0 / 1 / 16 (Interleaved) and `CBIndex::c_16/c_17/c_18` + `.buffer = <output shard buffer>` (Sharded) | named `DataflowBufferSpec` / `dfb::` bindings; `borrowed_from = tensor::<q|k|v>` for the sharded outputs |
| Tensor access | `TensorAccessorArgs(buf).append_to(cta)` + address RTAs (`factory:164-179`, kernels `TensorAccessorArgs<2>()`, `TensorAccessorArgs<5>()`) | `TensorBinding` + `TensorAccessor(tensor::name)` |
| Runtime args | positional `get_arg_val<uint32_t>(i)`, `get_arg_addr(19)` pointer into the RTA block (sharded reader `:29-30`) | `get_arg(args::name)` / vararg API |
| Kernel buffer API | `#include "api/dataflow/circular_buffer.h"`, `CircularBuffer cb_qv(cb_id)`, `get_tile_size(cb_id)` (all three DM kernels; compute donor `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` uses `tt::CBIndex::c_0/c_16`) | `DataflowBuffer(dfb::name)`, `get_entry_size()` |
| Cache-hit path | `override_runtime_arguments` + `UpdateDynamicCircularBufferAddress` (`factory:606-669`) | `ProgramRunArgs` / bindings |
| M2 markers | `grep -rE 'create_program_artifacts|ProgramArtifacts|KernelSpec|DataflowBufferSpec|dfb::|args::|tensor::|scratch::|ARCH_QUASAR'` over the op -> **0 hits** | — |

The kernels *are* on the Device 2.0 NoC API (`Noc noc; noc.async_read(...)`, PR #45843), which is why they
look half-ported at a glance. Per `quasar_porting.md` §1 step 1 that is not sufficient: they still use
`CircularBuffer` / `get_arg_val` / `get_tile_size(cb)` and the factory is a `ProgramDescriptor`.

### Why the Quasar model test cannot pass in this state

`ProgramDescriptor` kernels are materialised through the Gen1 path: `ReaderConfigDescriptor` /
`WriterConfigDescriptor` -> `ReaderDataMovementConfig` / `WriterDataMovementConfig` -> `CreateKernel(...)`
(`tt_metal/impl/program/program.cpp:437-489`). Quasar kernels are only created via the Metal 2.0
`KernelSpec` path (or the bring-up-only `CreateQuasarDataMovementKernel`, `tt_metal/impl/host_api/temp_quasar_api.cpp`).
So on a Quasar device the op is expected to fail at program construction/launch before any kernel runs; the
exact message was **not** verified (no build/run was performed in this session, per the rules). The base
Metal 2.0 port is the fix, not an `ARCH_QUASAR` guard.

### Prior art (do not skip the recipe because of it)

PR **#47878** "[Metal 2.0] Port nlp_create_qkv_heads to create_program_artifacts"
(branch `vsureshTT/metal2-nlp-create-qkv-heads`, split from umbrella #46909) was **auto-closed as stale on
2026-08-11**. Its description claims BH 111 passed and craq-sim + emu PCC 1.0 including transpose-K. It
predates the `kv_tied` feature (#53813, merged 2026-08-21) and the current post-port passes, and it was
never reviewed/merged, so it is a *reference*, not a base: the port must be redone (or rebased and
re-audited) against the current recipe (`metal2_audit.md` -> `metal2_port.md` -> post-port passes) and
the current `main` source of the op.

---

## Files changed

| File | Reason |
|---|---|
| `QUASAR_UPLIFT_REPORT.md` (this file) | Recipe deliverable. **Uncommitted; delete before merge.** |

No kernel, factory, header, CMake, or out-of-op file was touched. Nothing tempted a move/rename; the op's
directory and namespace (`ttnn::operations::experimental::transformer::NlpCreateHeadsDeviceOperation`,
`ttnn::prim::nlp_create_qkv_heads`) stay as they are.

---

## Quasar-uplift audit (`quasar_audit.md`) — run anyway, **advisory for the base porter**

Even though the uplift is stopped, the audit was run on the legacy CBs so the Metal 2.0 porter knows the
Gen2 end-state of each buffer up front (the helper doc classifies CBs and DFBs alike).

### Check 1 — device-side CB/DFB classification (`cb_dfb_quasar_audit_helper.md`)

**Scope (from factory `kernel_source` literals):**
- Interleaved: `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp`,
  `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp`, and (only when
  `transpose_k_heads`) the **cross-op donor** `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`.
- Sharded: `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp`, bound **twice**
  (as reader with CTAs `{c_16, c_17}` and as writer with CTAs `{c_16, c_18}`) on the Q shard grid.
- Unreferenced kernels in the op tree: none.

GATE scans (`get_local_cb_interface`, `fifo_*`, `get_cb_tiles_*_ptr`, `read_tile_value`, `get_tile_address`,
`get_pointer_to_cb_data`, `evil_*`): **0 hits** in all in-scope kernels.

| Buffer | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|---|---|---|---|---|---|---|
| cb 1 (`cb_id_qv`; also K when `!TRANSPOSE_K_HEADS`) | 1 | interleaved reader (producer), writer (consumer) | Portable | 4-tile explicit FIFO; port to `DataflowBuffer`; pass the DFB **as the NoC operand** instead of `CoreLocalMem<uint32_t>(get_write_ptr())` (whitelist NEEDS-FIX -> `noc.h`) | Portable | canonical DM->DM FIFO; implicit-sync candidate on Quasar DM (one NoC txn per entry) |
| cb 0 (`cb_id_k`, `TRANSPOSE_K_HEADS` only) | 1 | interleaved reader -> `transpose_wh.cpp` compute | Portable | reader->compute FIFO | Portable | — |
| cb 16 (`TRANSPOSE_K_HEADS` only) | 1 | `transpose_wh.cpp` compute -> interleaved writer | Portable | compute->DM FIFO | Portable | — |
| `c_16` q_out (Sharded; `.buffer` = Q output shard) | 6 | sharded reader **and** sharded writer, same core | Portable (prereq: LTA) | **sync-free, borrowed**: both kernels write `get_write_ptr() + q_offset` with **zero** `reserve/push/wait/pop`; reader fills the first `per_risc0_out_q_heads*head_size` bytes, writer the rest | Portable (prereq: LTA) | end-state **read-write `LocalTensorAccessor`** over `tensor::q` on both kernels (borrowed => LTA, not scratchpad; two writers on one node is fine for a tensor view) |
| `c_17` k_out (Sharded; `.buffer` = K output shard) | 6 (single-ended) | sharded reader only | **NEEDS-DESIGN-DECISION** | DM **single-ended producer**: `reserve_back(num_kv_tiles)` ... NoC reads via `get_write_ptr()` ... `push_back(num_kv_tiles)`, **no consumer anywhere**. Helper rule: "DM · single-ended producer (real reserve/push, no consumer) -> **STOP**, surface to API owner; prefer writing the tensor directly." | NEEDS-DESIGN-DECISION | recommended v1 for the owner to confirm: the CB is sized exactly `k_num_tiles` and filled once, so the credits synchronise nothing -> drop the pair and use **`LocalTensorAccessor` over `tensor::k`**. Not decided here. |
| `c_18` v_out (Sharded; `.buffer` = V output shard) | 6 (single-ended) | sharded writer only | **NEEDS-DESIGN-DECISION** | same shape as `c_17` (same kernel source, CTA 1 = `c_18`) | NEEDS-DESIGN-DECISION | same as `c_17`, over `tensor::v` |

Rollup for the **Interleaved** factory (the one the llama32 Quasar test exercises): **GREEN** — three Class 1
FIFOs. Rollup for the **Sharded** factory: **RED** until the op owner picks the end-state for the two
single-ended producer CBs (the helper forbids self-looping or fabricating a consumer for a real producer).
These are RED-stop condition #3 ("a construct needs an owner decision") for the eventual uplift of the
Sharded factory — but the immediate blocker for both factories remains the missing Metal 2.0 port.

Other sharded-path observations for the base port (host/spec audit territory, `metal2_audit.md`):
- The sharded kernels read **remote-core L1 by raw address** (`UnicastEndpoint` + `.addr = q_src_addr`) with
  base/start addresses smuggled as RTAs (slots 6/7 and 15/16, re-patched in `override_runtime_arguments`).
  The Metal 2.0 port must express these as tensor bindings (input shard base via `TensorAccessor` / shard
  addressing), not as address RTAs.
- `(tt_l1_ptr uint32_t*)(get_arg_addr(19))` casts a `uint32_t` to a pointer — on Quasar (64-bit pointers)
  this is the `-Werror=int-to-pointer-cast` row of §8.1. The M2 vararg API removes it anyway.
- Kernel-side `mcast` naming (`in0_mcast_noc_x/y`) is misleading: every transfer is a **unicast** read. No
  multicast exists in the op (§11 does not apply).

### Check 2 — non-zero-initialised semaphores

`grep -rE 'Semaphore|CreateSemaphore'` over the op -> **0 hits**. No semaphores at all. Not a blocker.

### DM self-loops (`dm_self_loop_dfbs.md`)

None. No CB is both produced and consumed by the same DM kernel (the sharded output CBs are single-ended,
which is a different shape). Zero-site pass.

---

## §7–§8 gotchas: applied vs considered

**Applied: none** (no uplift was performed). Everything below is what the audit *found* so the base porter
and the later uplift do not rediscover it.

| Gotcha (`quasar_porting.md`) | Verdict for this op |
|---|---|
| §7 bare `wait_front->pop_front` / `reserve_back->push_back` in **compute** (TEN-4746) | `transpose_wh.cpp`: `wait_front(1)`, `reserve_back(1)` -> `transpose_tile` (real UNPACR) -> `pack_tile` (real PACR) -> `push_back(1)`, `pop_front(1)`. Both pairs are ordered by real TDMA work on the only control path. **Not a site.** DM kernels are out of scope for this check by definition. |
| §7 `compute_kernel_hw_startup` exactly once | `transpose_wh.cpp` calls it once at `kernel_main()` start. OK. |
| §7 re-`*_init` on every DFB-id change | `transpose_wh.cpp` uses one fixed (in, out) pair and one `transpose_init`. **Not a site.** |
| §7 `partials_cb_uses_output` / borrow-with-offset | No matmul partials. **N/A.** The sharded `c_16` borrow *does* rely on a byte offset **inside the kernel** (`q_offset` RTA added to `get_write_ptr()`), not on a CB address offset, so it is expressible as an LTA index and does not hit the missing-offset limitation. |
| §7 Quasar has Int32, no Bfp8/uint16/uint32 | `validate_on_program_cache_miss` accepts FLOAT32 / BFLOAT16 / BFLOAT8_B. The op merely forwards the dtype (no format-specific code path), so per §7 there is **nothing to guard in the op**; a bf8 input will be rejected by the Quasar spec validator at the format layer. The llama32 case is bf16 and unaffected. |
| §7 implicit sync (do **not** disable) | Nothing in the op disables it; the Interleaved FIFOs are canonical Class 1 and should just rely on the Gen2 default once ported. |
| §7 non-zero-init semaphores | none. |
| §7 "don't invent Quasar-only device interfaces" | nothing needed. |
| §11 multicast rectangle normalisation / degenerate-grid clamp | **no multicast** in the op. N/A. |
| §5 `get_entry_size()` not `fifo_page_size` | kernels use `get_tile_size(cb_id)` today -> becomes `dfb.get_entry_size()` in the base port (base-port item, not an uplift edit). |
| §4 `opt_level` | legacy leaves it absent -> resolves to **O2 DM / O3 compute** (`program.cpp:441,448,481`). Flag **to the base port**: the M2 `KernelSpec` compute default is O2, so the port must set O3 explicitly on the `transpose_wh` compute kernel or perf silently shifts. |
| §4 / `gen2_hardware_configs.md` hw_config | legacy compute config is `ComputeConfigDescriptor{.fp32_dest_acc_en = (dtype == FLOAT32)}` (`factory:187,195,205`). After the port this is shape 2 (`to_compute_hardware_config`) for bf16/bf8, but for FLOAT32 with `enable_32_bit_dest = true` §4 requires explicit `unpack_modes` for the Float32 DFB the compute consumes -> shape 3: use the `m2::unpack_modes(cfg)` accessor (no arch branch) and add the `// TODO(#52269)` marker. DM configs: use `ttnn::create_reader/writer_datamovement_config(arch)` (shape 1, nothing to do). |
| §8.1 `-Werror=int-to-pointer-cast` | sharded reader `:29-30` (see above). Disappears with the vararg API. |
| §8.2/§8.3 runtime rows (`0x19`, `fifo_page_size` inflation, cache staleness, ...) | not reachable — no device run in this session; to be applied **reactively** after the base port, never pre-emptively. |
| `metal2_port.md` shared-kernel carve-out | `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` is bound by **four** legacy factories (this op, `nlp_create_qkv_heads_vit`, `nlp_create_qkv_heads_boltz`, `split_query_key_value_and_split_heads`), none of which is Metal 2.0. Converting it in place would break the other three, so the base port needs a `transpose_wh_metal2.cpp` fork **beside the original** plus a pointer comment (the only sanctioned out-of-op write). Note: `data_movement/transpose/device/kernels/compute/transpose_wh_metal2.cpp` already exists but is a fork of *transpose's own* kernel, not of this donor — inspect before assuming it can be reused. |

---

## Deferred / follow-up items

1. **Base Metal 2.0 port of `nlp_create_qkv_heads` (both factories)** — prerequisite for any Quasar uplift.
   Run `ai/audit/metal2_audit.md` -> `ai/port/metal2_port.md` -> post-port passes on this op. #47878 can be
   mined for the kernel/binding shapes but must be re-derived against current `main` (kv_tied landed after it)
   and reviewed. The Interleaved factory alone unblocks the llama32 Quasar test case.
2. **Owner decision (Sharded factory)**: end-state for the two single-ended DM producer CBs `c_17`/`c_18`
   (recommended: drop the inert `reserve_back`/`push_back` pair and write through a `LocalTensorAccessor`
   over the K/V output tensors). Per the audit helper this is a STOP for the porter, not a porter decision.
3. **Shared donor kernel** `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` -> `_metal2` fork beside it (see above).
4. **Format-layer limitation**: BFLOAT8_B inputs will not run on Quasar (no Bfp8). Model-level dtype decision;
   nothing to change in the op.
5. **Test-scope note**: the single llama32 graph_ops case is 1024 rows tall, above the conftest's
   `_EMU_MAX_ROWS = 128`, so it is **not** tagged `-m emulator` — it targets the full-size simulator, not the
   2-node ZEBU emulator. Expect `pytest ... -m emulator` to deselect it.

---

## Parity claim (WH / BH)

**Zero-diff.** No file other than this report was created or modified (`git status --short <op>` shows only
`QUASAR_UPLIFT_REPORT.md`), so WH/BH behaviour is unchanged **by construction**. There is no
`ARCH_QUASAR`-guarded code because there is no Quasar change. The commands below are the confirmation the
human should run anyway (they also serve as the pre-port baseline for the base Metal 2.0 port, which must be
recorded *before* the first kernel edit because kernels are JIT-compiled from the working tree).

---

## Test commands (user runs these; nothing was built or run in this session)

Run from the repository root with the venv active. `TT_METAL_FORCE_JIT_COMPILE=1` is only needed once
kernels change.

**Blackhole (parity baseline / control):**
```bash
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py -v
pytest tests/ttnn/unit_tests/operations/experimental/transformer/test_nlp_create_qkv_heads_program_cache.py -v
# note: test_nlp_cqkv_sharded_addr_change_on_hit is @skip_for_blackhole (#12349) — expected SKIPPED on BH
```

**Wormhole (parity baseline / control):**
```bash
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py -v
pytest tests/ttnn/unit_tests/operations/experimental/transformer/test_nlp_create_qkv_heads_program_cache.py -v
```

**Quasar (simulator / emulator) — expected to FAIL until the Metal 2.0 port lands:**
```bash
# full-size case (1024 rows; not in the -m emulator subset)
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_create_qkv_heads.py -v
# after the base port, run both with and without LLK asserts (quasar_porting.md §9):
TT_METAL_LLK_ASSERTS=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_create_qkv_heads.py -v
```

---

## Definition-of-done checklist (`quasar_porting.md` §10) — status

- [ ] In place, no `experimental/quasar/` copy, no `::qsr` — **nothing done; nothing copied** (audit only)
- [ ] Factory is `create_program_artifacts`/`ProgramArtifacts` — **NO (RED-stop #1)**
- [ ] `opt_level` matches resolved legacy — base-port item (flagged above)
- [ ] every DFB has valid `data_format_metadata`; `get_entry_size()` — base-port item
- [ ] sync-free / DM self-loop converted — no self-loops; sharded q/k/v borrows -> LTA (owner decision on k/v)
- [x] no `disable_dfb_implicit_sync_*` — none present
- [x] no borrow-with-offset ported as-is — none
- [x] every mcast rectangle normalised — no multicast
- [x] every op re-init on new DFB ids — single fixed pair in `transpose_wh.cpp`
- [x] no bare compute wait/pop or reserve/push — `transpose_wh.cpp` pairs are TDMA-ordered
- [x] no non-zero-init semaphore — no semaphores
- [ ] BH and WH pass unchanged — not run (commands above); zero-diff so structurally unchanged
- [ ] Quasar builds and runs — **NO**; blocked on the Metal 2.0 port
- [x] no DIAG leftovers — none added
- [x] missing core deps flagged — none found beyond the Bfp8 format-layer limitation
- [x] `QUASAR_UPLIFT_REPORT.md` written with RED status and RED-stop conditions checked
