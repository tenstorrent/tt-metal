# Operation Design: nlp_create_qkv_heads_decode

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (a gather of tile rows; no compute kernel, no compute config) |
| Goal | Cut one decode step's fused QKV row per user into the heads of q, k and v, and place each user's block of heads as an L1 height shard on that user's core, in one `ttnn.generic_op` dispatch of one `ttnn.experimental.ProgramSpec`. |
| Math | `q[0,u,h,d] = x[0,0,u,h*hd+d]`; `k[0,u,h,d] = x[0,0,u,(nh+h)*hd+d]`; `v[0,u,h,d] = x[0,0,u,(nh+nkv+h)*hd+d]`, `hd = W/(nh+2*nkv)`. Every output element is one input element, bit for bit (R1). |
| References | Specification `eval/specs/nlp_create_qkv_heads_decode/spec.py` (governs); target `.claude/references/targets/quasar.md`; backend `.claude/references/backends/ttsim.md`; host-program examples `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/`); `tt_metal/hw/inc/api/dataflow/noc.h`; `tt_metal/hw/inc/api/tensor/local_tensor_accessor.h`; `tt_metal/hw/inc/internal/local_mem/noc_traits.h`; catalog `ttnn/ttnn/operations/examples/master.md` (`split_reader`, `trid_double_issue`, `tile_reorder`, `read_coalescing`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 cores (queried: `device.compute_with_storage_grid_size()`) | target page |
| L1 per core | 4,194,304 B | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`; value on this target not established on paper. No predicate of this op needs it: the op allocates 0 B of L1 of its own (see `l1_ledger.md`). | `unknown` (value); query named |
| Host program | `ttnn.ProgramSpec` (`ttnn.experimental.ProgramSpec`, `KernelSpec`, `TensorParameter`, `TensorBinding`, `WorkUnitSpec`, `ProgramRunArgs`) dispatched by `ttnn.generic_op(io_tensors, spec, run_args, tensor_args)` — reference `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}); 6 user DM harts (R + W ≤ 6); R | C and W | C on STRIDED rings — this op has no ring and no compute kernel, so only R + W ≤ 6 binds | target page, *Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` (Metal 1, refused at kernel creation); legacy C++ device op with program factory (same refusal); `bfloat8_b` / `bfloat4_b` (no path — not in TARGET anyway). A NoC transfer without a transaction id: not used; every read of this op is tagged `NocOptions::TXN_ID` with `{.trid = 0}`, barriered on trid 0, and the kernel's last statement is `noc_async_read_set_trid(0, noc.get_noc_id())`. uint16/uint32 buffer formats: not applicable (no buffers). | target page *Not available* |

All worked numbers below are for this target (Quasar, 32 cores) and for the two traced calls: **Llama** `[1,1,1,3072]`, nh=32, nkv=8, hd=64; **Qwen** `[1,1,1,6144]`, nh=32, nkv=8, hd=128.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 4, `[1, 1, users, W]`, TILE layout, interleaved (L1 in Phase 0; DRAM in TARGET) | — | tensor binding `in` |
| `num_heads` (nh) | int, keyword-only | yes | ≥ 1 (X1); `tag_head_rows` ∈ SUPPORTED (32 in Phase 0, i.e. nh ≤ 32) | — | CT `num_q_heads` |
| `num_kv_heads` (nkv) | int or None, keyword-only | no | 1 ≤ nkv ≤ nh (X2, X3); None → nh (R4) | None | CT `num_kv_heads` |
| `overlap_qk_coregrid` | bool, keyword-only | no | any; never read, never gated on (R7) | True | — |
| `memory_config` | `ttnn.MemoryConfig` or None, keyword-only | no | read with the spec's `canonical_memory` into axis `output_memory`; a non-config raises `TypeError` (X7); value must be in SUPPORTED (`"L1/height"` in Phase 0) | None | host only |

Derived host quantities (each defined once, every dependent derived from it):

| Symbol | Definition | Passed as |
|--------|------------|-----------|
| `sections` | `nh + 2*nkv` | — |
| `head_dim` | `W // sections`; X4 if `W % sections != 0`; X5 if `head_dim % 32 != 0` | — |
| `head_dim_tiles` | `head_dim // 32` | CT |
| `input_width_tiles` | `ceil(W / 32)` (= `sections * head_dim_tiles` once X4/X5 pass) | derived in the kernel as `sections * head_dim_tiles` (one source; host asserts the equality) |
| `face_row_bytes` | `input_tensor.buffer_page_size() // 64` (16 elements of a 32×32 tile of 16×16 faces: 32 B bf16, 64 B fp32) | CT |
| `in_page_bytes` | `input_tensor.buffer_page_size()` | CT |
| `out_page_bytes` | `q.buffer_aligned_page_size()` (the shard's page stride in L1; same for q, k, v) | CT |
| `q_rows`, `kv_rows` | `ceil(nh/32)*32`, `ceil(nkv/32)*32` (spec `shard_shapes`) | host (shard specs) |
| `GATHER_THREADS` | module-level host constant, the thread plan R (Phase 0 = 2) | `num_threads` + CT `gather_threads` |
| `user_tile_row`, `user_row` | per core u: `u // 32`, `u % 32` | RT, per core |

Validation order in `validate()` / the entry point: (1) rank and leading dims (X6) and layout (X8, refuse ROW_MAJOR by name); (2) head counts X1–X3, then X4, X5; (3) `memory_config` through `canonical_memory` (X7 `TypeError`); (4) axes `memory_config`/`sharding` from `eval.memory.memory_axes(input_tensor)` (R9), `dtype`, `layout`, `output_memory`, and the taggers `INPUT_TAGGERS = AXES` (imported from the spec, never redefined) on `(shape, num_heads, num_kv_heads)`; (5) every axis value in SUPPORTED else `NotImplementedError` (R11). There is no index axis to canonicalize. The input's tile must be the default 32×32 tile with 16×16 faces (`input_tensor.tile`); any other tile is refused (`NotImplementedError`) — it is a mechanism cap (below).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[1, 1, users, W]`, `W = (nh + 2*nkv) * head_dim`, `head_dim % 32 == 0` |
| Dtype | bfloat16 (Phase 0); float32 (TARGET) |
| Layout | TILE (32×32, 16×16 faces) |
| Memory | interleaved; L1 (Phase 0), DRAM (TARGET) |

### Output

| Property | Value |
|----------|-------|
| Shape | q `[1, users, nh, head_dim]`; k, v `[1, users, nkv, head_dim]` |
| Dtype | input's |
| Layout | TILE |
| Memory | each: `ttnn.MemoryConfig(HEIGHT_SHARDED, L1, ttnn.ShardSpec(cores, [rows, head_dim], ROW_MAJOR))`, `cores = ttnn.num_cores_to_corerangeset(users, device.compute_with_storage_grid_size(), row_wise=True)`, `rows = q_rows` for q, `kv_rows` for k and v (R5). Allocated with `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(shape, dtype, TILE_LAYOUT, memory_config), device)` before the dispatch. The op returns the Python list `[q, k, v]` of these three tensors (R2), not `generic_op`'s return value. |

Head rows of a shard beyond nh (q) / nkv (k, v) are tile padding: they are not written and hold whatever L1 held (R3 forbids reading padding *input* rows; nothing constrains output padding, and `to_torch` drops it).

## Blocking Model

The first design section. Everything below is downstream of it.

The op's index space, per user u, in tile units: the input row u is `input_width_tiles` tiles wide, and within each tile the row is two **face-row runs** (left face, right face) of `face_row_bytes` each. Each run is one indivisible transfer unit: a run is contiguous in the source tile (one 16-element face row) and contiguous in the destination tile (one 16-element face row), and nothing larger is contiguous on both sides (adjacent runs of a source tile are 16 face rows apart; adjacent heads' runs land in adjacent destination face rows but come from different source tiles). So the transfer count per user is fixed at `2 * input_width_tiles` by the data layout — Llama 192, Qwen 384 — and no blocking choice changes it (`read_coalescing` in the catalog needs physical contiguity on the source side, which this layout does not have).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `users` (u, input row / output shard index) | independent — each user's heads come only from that user's row | `users_per_core` | 1 | fixed by R5 (one shard per user, user u on core u); a host constant asserted against the shard spec | **pinned by the output placement**: user u → u-th core of `num_cores_to_corerangeset(users, grid, row_wise=True)`; the grid is queried, the core count is `users` (a parameter, not inlined). Phase 0 TARGET has users = 1 → one core holds the whole op. | knob-turn (users > 1: per-core RT args `user_tile_row`, `user_row` already exist; host gating only) — outside TARGET today |
| `heads` (s ∈ [0, sections): q heads, then k, then v) | independent — head s's output row depends on head s's input columns only | `heads_per_block` | `sections` (all heads of the user: the whole per-core assignment) | host: `heads_per_block = sections`, passed as CT; the kernel's block walk uses it | not split across cores (rejected regime, see Regimes: it doubles the NoC crossings); within the core, the runs of all heads are dealt over the `GATHER_THREADS` harts | knob-turn (smaller block = more barriers; no reason to) |
| `head_dim_tiles` (dt, tile column within a head) | independent — pure copy per tile column | `head_dim_tiles_per_block` | `head_dim_tiles` (whole) | derived from `head_dim` once | not split | none needed |
| `face_col` (c ∈ {0,1}, left/right face of a tile row) | independent — two disjoint runs | `face_cols_per_block` | 2 (whole) | 32×32 tile / 16-wide face geometry (mechanism cap) | not split across cores; the flat run index's lowest bit, so with R = 2 hart 0 moves left-face runs and hart 1 right-face runs | none |
| `out_tile_row` (h // 32, the shard's tile row for head h) | independent — derived from `heads`, one tile row per 32 heads | `out_tile_rows` | `q_rows/32` = 1 for q, `kv_rows/32` = 1 for k, v (Phase 0, head_rows = 32) | derived from nh, nkv | not split (part of the user's shard) | knob-turn: head_rows = 64 needs only the SUPPORTED change — the destination offset already uses `h // 32` |

No dependent axis exists (nothing spans an axis; no reduction, no combine). No operand is reuse-shared: the one input varies along `users` and `heads`, the split axes, so the operand-reuse check finds nothing to broadcast. Intermediate data: none — the scheme has one stage (gather) and no internal result to re-block.

**The block** (per core): the whole user — rectangle `heads_per_block × head_dim_tiles × 2 face columns` of runs = all `2 * input_width_tiles` runs, issued without a barrier between them, then one barrier. One block per core, `num_blocks_this_core = 1`.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| — (none) | in-flight reads per hart (hardware-managed: `Noc::async_read<TXN_ID>` busy-waits until half the trid's transaction slots are free, `noc.h:258-262`) | as many as the NoC allows; one barrier per block | overlap of read latency across runs. There is no staging buffer, so there is no depth to size: reads land in their final place in the output shard. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Tile geometry (32×32 tile, 16×16 faces; `face_row_bytes = page/64`) | `face_col` = 2, rows per face = 16 | validate refuses any non-default tile (`NotImplementedError`) | offsets land in the wrong face → silently wrong values |
| NoC source/destination alignment (L1: 16 B, `ttnn.get_l1_alignment()`) | run offsets | every source offset (`face_idx*16*face_row_bytes + (row%16)*face_row_bytes`) and destination offset is a multiple of `face_row_bytes` ≥ 32 B → aligned for an L1 source; asserted on host: `face_row_bytes % ttnn.get_l1_alignment() == 0` | misaligned NoC read → wrong bytes |
| DRAM read alignment (64 B, `ttnn.get_dram_alignment()`) | run size for a DRAM source | not reachable in Phase 0 (input in L1 only); the DRAM regime (deferred) bounces through a scratchpad when `face_row_bytes % dram_alignment != 0` (bf16: 32 B) | a sub-alignment DRAM read returned wrong data on Blackhole (spec GUIDANCE, tt-metal #43270) |
| Output shard height (`head_rows` ∈ {32, 64} in TARGET) | `out_tile_rows` | `tag_head_rows` gated by SUPPORTED (32 in Phase 0); > 64 refused by name (not in TARGET) | destination beyond the shard → corrupts neighbouring L1 |
| Transaction id | tags | one user id, `{.trid = 0}`, on every read; no ring tags exist | — |
| `R + W ≤ 6` user DM harts | `GATHER_THREADS` | host asserts `1 ≤ GATHER_THREADS ≤ 6` (W = 0) | program refused / hart oversubscription |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **A. Direct gather, L1 source**: user u's core reads each face-row run of its input row straight from the interleaved input page into its final place in its own q/k/v shard (destination = `LocalTensorAccessor` of the shard, zero-copy) | **built** (Phase 0) | `memory_config == "L1"` (input interleaved in L1) | whole user: `heads_per_block = sections` × `head_dim_tiles` × 2 runs | **minimum**. Boundary: the NoC between the input's L1 banks and the user's core. Each of the user's `W * es` row bytes crosses it once (pages that sit in core u's own bank are loopback reads); outputs never leave the core (written in place); no DRAM crossing. Transfer count `2 * input_width_tiles` is the layout minimum (see above). | One block = one barrier per hart per op, one kernel launch, no ring handshakes at all. A smaller block would only add barriers (each a full drain of in-flight reads, `trid_double_issue`: batch barriers first). |
| **A-DRAM. Direct gather, DRAM source** | deferred — the input in DRAM is in TARGET (`memory_config: "DRAM"`, R8) but outside PHASE0; positive reason: PHASE0's rectangle is L1 only, and the regime layers onto A without restructuring | `memory_config == "DRAM"` | same as A | minimum: each row byte crosses DRAM once. Where `face_row_bytes % ttnn.get_dram_alignment() != 0` (bf16), each run is read as the aligned 64 B window containing it into a per-hart `Scratchpad` slot, then moved into the shard by one self-aimed local read (`local_copy_helpers_dataflow.hpp`): adds one core-local copy per run, no extra DRAM bytes beyond the 64 B window (2× the run for bf16). fp32 runs (64 B) read directly. | same as A; reachable because A's run walk, offsets and tags are unchanged — only the source accessor's alignment branch is added |
| **B. Whole-tile staging**: read each whole input tile (`in_page_bytes`) into a staging DFB, then extract row u locally | rejected — superseded by A: for users = 1 it moves 32× the bytes over the NoC (31 of 32 rows are padding, R3) and adds a ring plus a second, local copy; fewer, larger transfers do not pay for 32× the bytes (`tile_reorder` favours whole pages only when the whole page is wanted) | — | whole tile per run pair | 32× minimum NoC bytes at users = 1 | — |
| **C. Two-hop pipeline**: reader → DFB → writer → shard | rejected — superseded by A: the shard is already in the gathering core's L1, so a NoC read can land in it directly (`noc_traits_t<LocalTensorAccessor>` dst, `internal/local_mem/noc_traits.h:106`); the ring adds a local hop, credits and L1 for nothing | — | — | +1 core-local crossing of every byte | — |
| **D. Heads split across cores**: other cores gather some heads and write them into core u's shard | rejected — superseded by A with a thread plan: a remote core must first read a run into its own L1 then write it to core u (2 NoC crossings per byte instead of 1), and the extra issue parallelism it buys is available on core u itself from its idle DM harts (`GATHER_THREADS`) | — | heads/cores | 2× minimum NoC crossings + a completion handshake | — |
| **E. users > 1 (one core per user)** | deferred — positive reason: `users` is `[1]` in TARGET (spec DECISIONS: tt-metal's multi-user core range set is not stated); A is already per-user (RT args `user_tile_row`, `user_row`), so it is a host-gating change | `users > 1` | same as A, per core | minimum | same as A |

Only one regime is built in Phase 0, so no selection function runs yet. When A-DRAM is built the selection is `regime = "A-DRAM" if memory_axes(input)["memory_config"] == "DRAM" else "A"`, and both regimes get pinned tests (one cell each, by node id).

### Traffic ranking

| Rank | Split | NoC crossings per input row byte | Output crossings | DRAM crossings (Phase 0 L1 input) | Notes |
|------|-------|----------------------------------|------------------|-----------------------------------|-------|
| 1 | users → cores (R5's placement), all heads on the user's core, read into the shard directly (A) | 1 (0 for pages in core u's own bank) | 0 | 0 | chosen; also the placement the spec mandates |
| 2 | same, through a DFB and a writer (C) | 1 + 1 local | 0 | 0 | rejected |
| 3 | heads → cores, remote writes into core u (D) | 2 | — | 0 | rejected |
| 4 | whole-tile staging (B) | 32 (users = 1) | 0 | 0 | rejected |
| — | dependent-axis split + combine | n/a — the op has no dependent axis | | | |

Worked (bf16): Llama 6,144 B in 192 reads of 32 B; Qwen 12,288 B in 384 reads of 32 B. The cost is per-transaction (32 B against a 256 B/cycle NoC link), not bandwidth — which is what sets the thread plan below.

### Block schedule

```cpp
// per core (user u); num_blocks_this_core == 1
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    gather_head_rows(q_shard, /*first_section=*/0,          /*count=*/num_q_heads);   // this hart's share of the runs
    gather_head_rows(k_shard, /*first_section=*/num_q_heads, /*count=*/num_kv_heads);
    gather_head_rows(v_shard, /*first_section=*/num_q_heads + num_kv_heads, /*count=*/num_kv_heads);
    drain_block();   // one read barrier on trid 0
}
release_tag();       // noc_async_read_set_trid(0, noc.get_noc_id()) — last statement
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `gather_head_rows(dst_shard, first_section, count)` | `count` heads × `head_dim_tiles` × 2 runs; this hart takes flat run indices `t, t+R, …` of the user's run list (flat index `((s*head_dim_tiles)+dt)*2 + c` over all sections) | the three output shards (tensor-owned L1) | per run: one tagged `async_read` (source page `user_tile_row*input_width_tiles + s*head_dim_tiles + dt`, source offset `(2*(user_row/16)+c)*16*face_row_bytes + (user_row%16)*face_row_bytes`; destination offset in `dst_shard` `((h/32)*head_dim_tiles + dt)*out_page_bytes + (2*((h%32)/16)+c)*16*face_row_bytes + ((h%32)%16)*face_row_bytes`, `h = s - first_section`). No barrier, no per-run handshake. Accessors constructed once per kernel. |
| `drain_block()` | the whole block | — | once per block per hart: `noc.async_read_barrier<TXN_ID>({.trid = 0})` |
| `release_tag()` | — | — | once per kernel per hart |

Stall-shadow check: the only wait is `drain_block()`, and no work remains that is independent of it (all of the op's reads are already in flight when it starts). Nothing to reorder; no floating-point order exists (values are only moved).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Thread plan R = `GATHER_THREADS` = 2 | The op is bound by issuing 192–384 tiny reads; a second hart halves each hart's issue count (`split_reader`: pays only when one hart is issue-bound). Whether the NoC interface, not the hart, is the limit on Quasar is unmeasured. | R = 1 and R = 4 (one constant), on `ttsim-perf` model cycles; structural count on this backend: reads issued per hart (96/192 at R = 2 for Llama/Qwen). |
| Grid synchronization | Phase 0 runs on one core because R5 pins every shard of the single user to core (0,0); 31 cores idle. Regime D (spread heads) is rejected on traffic (2× crossings) — this lamp records that the occupancy side was considered. | none to build; measure only if a timed backend shows core (0,0)'s gather above the launch floor |
| Overlap | One block per core with a single drain: there is nothing downstream to overlap with (no compute, no writer), so no lamp applies. | — |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input row u in the interleaved input | TILE pages (`in_page_bytes`), face-major; row u is 2 runs per tile in faces `2*(u%32/16)` and `+1` | `TensorAccessor(tensor::in)` with `{.page_id, .offset_bytes}` source args (`api/tensor/noc_traits.h:137-138`) | L1 banks across the grid (Phase 0); DRAM in A-DRAM |
| NoC read | one face-row run, `face_row_bytes` | `noc.async_read<NocOptions::TXN_ID>(in_acc, dst_shard, face_row_bytes, {.page_id = p, .offset_bytes = so}, {.offset_bytes = do}, {.trid = 0})` (`noc.h:246-277`) | explicit tagging (no ring); no reserve/push |
| Output shards q, k, v on core u | TILE pages, height shard `[rows, head_dim]`, page stride `out_page_bytes` | `LocalTensorAccessor<uint32_t>(tensor::q / k / v)` as read destination (`local_tensor_accessor.h:59`, `internal/local_mem/noc_traits.h:106-125`) | zero-copy: the gathered run lands in its final place; never re-read over the NoC |

Placement as a TARGET axis: `output_memory` ∈ {None, "L1/height"} both select R5's placement (height-sharded L1, one shard per user) — identical device program; supporting None is a SUPPORTED change (knob-turn). The input's `sharding` is `interleaved` only in TARGET; no physical input shard is in scope. Regime E (users > 1) is the same per-core program on more cores and needs no inter-core contract. Regime D, were it ever revisited, would need remote writes into core u's shard address (`TensorAccessor` of the sharded output) and a program-end completion only (no semaphore: program completion orders it).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block = one user's whole row → its three shards |
| Grid | `ttnn.num_cores_to_corerangeset(users, device.compute_with_storage_grid_size(), row_wise=True)` — the output shard grid; it is the `WorkUnitSpec` `target_nodes`. Phase 0 (users = 1): core (0,0). |
| Per-core work | 1 block: `2 * input_width_tiles` runs (Llama 192, Qwen 384), dealt over `GATHER_THREADS` harts |
| Remainder | none across cores (one user per core by R5). Within the core the flat run list is dealt round-robin to harts (`t, t+R, …`), which may stop anywhere: the unit is a single transfer, not a ring entry, so no padding. Tile geometry: `input_width_tiles = ceil(W/32)`, input tile row of user u is `u // 32` (per image: the input is one image `[1,1,users,W]`), shard rows `ceil(n/32)*32`. |
| Bound | **data movement** (per-transaction issue): zero compute per unit; each unit is a 32 B (bf16) / 64 B (fp32) transfer, so the time is the count of transfers issued, not bytes and not compute |
| Thread plan | (R, C, W) = **(2, 0, 0)**: one DM kernel `gather` with `num_threads = GATHER_THREADS = 2`, no compute kernel, no writer kernel (the reads land in the output, so nothing is written). R = 2 because the bound is issue rate and a second hart halves each hart's issue count (catalog `split_reader`; tt-metal's own kernel splits the two face runs over two processors, spec GUIDANCE). C = 0 because there is no compute to do; the target page's (1,1,1) for a DM-bound op presumes a compute stage this op does not have. Host constant `GATHER_THREADS` in the program-artifacts module; the kernel takes it as CT `gather_threads` and deals by `get_my_thread_id()`. |
| Unit dealt to compute threads | none (no compute threads). Unit dealt to DM harts: one run (one tagged read); round-robin over the flat run index, so with R = 2 hart 0 moves every left-face run and hart 1 every right-face run. No rounds, no padding. |
| Tests the plan needs | (1) **every hart ran**: a test (outside the generated acceptance test, e.g. `tests/ttnn/unit_tests/operations/nlp_create_qkv_heads_decode/test_thread_plan.py`) runs the Llama and Qwen shapes with the host constant patched to 1, 2 and 3 (monkeypatch `GATHER_THREADS`) and requires an exact result each time — a hart that does not run leaves its runs (e.g. every right-face half) unwritten — and asserts the built `ProgramSpec`'s `gather` kernel has `num_threads == GATHER_THREADS`. (2) **one device session**: the op run right after another program (e.g. `ttnn.add` on L1 tiles, or a tt-metal Quasar op that uses implicit sync) and right before a tt-metal program on its output (`ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)`; if sharded untilize is unavailable here, `ttnn.to_memory_config(q, ttnn.L1_MEMORY_CONFIG)` then `to_layout`), all in one test function, with the results checked. |

## Circular Buffers

**None.** The op allocates no dataflow buffer, no scratchpad and no semaphore. The only L1 it touches on the gathering core is the three output shards, which are tensor-owned and allocated by `allocate_tensor_on_device` before the dispatch; they are bound as tensors, not as rings:

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|----------|
| `q_shard` (tensor binding `q`, `LocalTensorAccessor`) | — (tensor) | `out_page_bytes` | `(q_rows/32) * head_dim_tiles` | spans `out_tile_row`, `head_dim_tiles`; streams nothing — it is the output | input dtype | `gather` (NoC reads land in it) | host / next op | none: no ring; every read tagged `{.trid=0}`, drained by barrier on trid 0 | whole program and after |
| `k_shard` (binding `k`) | — | `out_page_bytes` | `(kv_rows/32) * head_dim_tiles` | as `q_shard` | input dtype | `gather` | host / next op | as above | as above |
| `v_shard` (binding `v`) | — | `out_page_bytes` | `(kv_rows/32) * head_dim_tiles` | as `q_shard` | input dtype | `gather` | host / next op | as above | as above |

Ownership: each shard has one writer kernel (`gather`, whose harts write disjoint runs) and no in-program reader. Inventory justification: Rule 3 pattern 1 ("pack into the destination") taken to its end — the gather's destination *is* the output, so no staging buffer exists to alias, size or solve.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input (semantic name, pages, state) | Output (semantic name, pages) | State after |
|---|-----------------|-------------|---------|-------------------------------------|-------------------------------|-------------|
| 1 | `gather_head_rows(q_shard, 0, nh)` | nh × `head_dim_tiles` × 2 runs (this hart's share) | no (raw NoC; see API Mapping) | input pages `user_tile_row*input_width_tiles + [0, nh*head_dim_tiles)`, read-only | `q_shard`, rows 0..nh-1 of every tile | reads in flight |
| 2 | `gather_head_rows(k_shard, nh, nkv)` | nkv × `head_dim_tiles` × 2 | no | pages `… + [nh*hdt, (nh+nkv)*hdt)` | `k_shard`, rows 0..nkv-1 | reads in flight |
| 3 | `gather_head_rows(v_shard, nh+nkv, nkv)` | nkv × `head_dim_tiles` × 2 | no | pages `… + [(nh+nkv)*hdt, sections*hdt)` | `v_shard`, rows 0..nkv-1 | reads in flight |
| 4 | `drain_block()` | whole block | no | — | — | all of this hart's runs landed |
| 5 | `release_tag()` | — | no | — | — | hart's tag register = 0 |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input | Output | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|-------|--------|------------------------------|
| construct source accessor | raw_api | `TensorAccessor(tensor::in)` | `tt_metal/hw/inc/api/tensor/tensor_binding_token.h:20-25` (usage), `api/tensor/noc_traits.h:137-142` (src args `page_id`, `offset_bytes`) | binding `in` | input | — | — |
| construct shard destinations | raw_api | `LocalTensorAccessor<uint32_t>(tensor::q/k/v)` | `tt_metal/hw/inc/api/tensor/local_tensor_accessor.h:58-78`; NoC endpoint traits `tt_metal/hw/inc/internal/local_mem/noc_traits.h:106-125` (dst args `offset_bytes`) | bindings `q`, `k`, `v` (L1 tensors; static_assert refuses DRAM) | — | shards | — |
| `gather_head_rows` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(src, dst, size, src_args, dst_args, {.trid = 0})` | `tt_metal/hw/inc/api/dataflow/noc.h:246-277` | `size = face_row_bytes`; `src_args = {.page_id, .offset_bytes}`; `dst_args = {.offset_bytes}` | input | shard | `heads_per_block` (count per call; Phase 0 whole), `gather_threads` (stride) |
| `drain_block` | raw_api | `Noc::async_read_barrier<NocOptions::TXN_ID>({.trid = 0})` | `tt_metal/hw/inc/api/dataflow/noc.h:738-746` | trid 0 | — | — | once per block |
| `release_tag` | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558` | — | — | — | — |
| host: program | raw_api (host) | `ttnn.experimental.ProgramSpec`, `KernelSpec(num_threads=GATHER_THREADS, hw_config=E.create_reader_datamovement_config(), tensor_bindings=[in, q, k, v], compile_time_args=…, runtime_arg_schema=["user_tile_row", "user_row"])`, `TensorParameter`, `WorkUnitSpec(target_nodes=shard grid)`, `ProgramRunArgs`; `ttnn.generic_op([input, q, k, v], spec, run_args, {"in":0,"q":1,"k":2,"v":3})` | `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py:96-158` (pattern) | no `dataflow_buffers`, no compute `KernelSpec`, no `disable_dfb_implicit_sync_for` (no rings) | | | `GATHER_THREADS` |

Compute phases: none, so no compute helper applies and `compute_kernel_hw_startup()` is not called (no compute kernel exists).

**Helpers considered and rejected** (for the raw dataflow calls):

| Candidate | File:Line | Why it cannot express this shape |
|-----------|-----------|----------------------------------|
| `dataflow_kernel_lib::local_addr` / `set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57,75` | self-aimed (source on this core's own L1) copies; the gather's sources are pages of an interleaved tensor on other banks, addressed by page id + offset through a `TensorAccessor`. (It is the right tool for A-DRAM's local bounce.) |
| `tile_column_dataflow.hpp` helpers | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:9-25,36` | writes a tile *column* with CPU stores into a DFB-bound tile (`TileColumnLayout<dfb_id>`); the op moves tile *rows* over the NoC into a tensor shard, with no DFB |
| `mcast/kernel/mcast_args.hpp` (`McastArgs`) | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` | no operand is reuse-shared; nothing to multicast |
| `tilize` / `untilize` / eltwise / reduce / matmul helpers | `ttnn/cpp/ttnn/kernel_lib/*_helpers*.hpp` | compute-side; the op has no compute (R1: values only moved) |

The block operation `gather_head_rows` is built by the implementer as a small kernel function over the raw calls above.

## Structural impossibilities

None to add: INVALID = [] is right for TARGET's rectangle.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Untagged NoC transfer / sticky tag | The kernel has no ring, so the implicit-sync API is not used | every read `NocOptions::TXN_ID` + `{.trid = 0}`; barrier on trid 0; last statement `noc_async_read_set_trid(0, noc.get_noc_id())`; the one-session test (Work Distribution) |
| Face layout of a tile row | Row r of a tile is two non-adjacent 16-element runs (faces `2*(r/16)` and `2*(r/16)+1`), not one contiguous row | offsets in the Block schedule; derive `face_row_bytes` from the page size so bf16 and fp32 share one formula; refuse non-default tiles |
| Users from the logical shape | The input `[1,1,1,W]` is padded to 32 rows; reading the tile's 32 rows (or using padded shape) would be wrong (R3) | only row `user_row = u % 32` of tile row `u // 32` is read; `users = input_tensor.shape[2]` (logical) |
| Output memory config compared with `==` | Callers and the spec's check compare the whole config: shard shape `[rows, head_dim]`, `ROW_MAJOR`, grid from `num_cores_to_corerangeset(users, grid, row_wise=True)` | build the configs exactly as spec `output_memory_configs` does; k and v of 8 heads still have shard `[32, head_dim]` with logical shape `[1,1,8,head_dim]` |
| Page stride inside the shard | The destination offset uses the shard's page stride; using the input's page size would break if alignment ever pads the output page | `out_page_bytes = q.buffer_aligned_page_size()` passed as CT, separate from `in_page_bytes` |
| Multi-thread barrier semantics | Both harts tag trid 0; if harts share a NoC's trid counters, a hart's barrier also waits for the other's reads | harmless (waits longer, never shorter); each hart barriers before returning, and program completion covers both |
| One dispatch | The three outputs must come from one `generic_op` | one `ProgramSpec` with three output tensor parameters; `[q, k, v]` returned from the host references |
| `overlap_qk_coregrid` | R7: never read and never gated | the entry point accepts it and ignores it |
