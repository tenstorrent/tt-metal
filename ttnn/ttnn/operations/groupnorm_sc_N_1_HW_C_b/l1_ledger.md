# L1 Ledger: groupnorm_sc_N_1_HW_C_b

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes, from `op_design.md` → Blocking Model:

| Axis | Meaning | Block extent |
|------|---------|--------------|
| `n` | image | 1 |
| `ht` | tile rows | `rows_core`, streamed in chunks of `h` rows; `K` chunks resident |
| `wt` | channel tiles | `W = Wt_core` |
| `g` | group lanes | `G_core ≤ 32`, one tile row |
| `i` | gather sender slots | `cpi`, `R = ceil(cpi/32)` tile rows |

`T = h·W`. `fp32_dest_acc_en = True`, `math_fidelity = HiFi3`. 2 KB = one bf16 tile page and 4 KB = one Float32 tile page.

## CB ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_rm` (0) | `D_RM·T` (RM only; `D_RM = 2`) | `2T`: one chunk being tilized plus one landing | `{n: streams → 1, ht: spans → h (streams over B chunks), wt: spans → W, g: streams → none, i: streams → none}` | Float16_b (input bytes; no DEST involved) | reader | compute | pass 1 + pass 2 (RM) | **Not shared.** It is live in both passes, concurrently with the input ring it feeds. Capacity exceeds one chunk because of double buffering (DRAM read ∥ tilize). |
| `cb_input_tiles` (2) | input ring, `K_alloc·T + STREAM·(TILE ? D_IN : 1)·T` (alias group A; `D_IN = 2`) | pass 1: up to `K·T` resident + `D_IN·T` streaming; pass 2: the `K·T` resident tail + the re-read chunk in flight | `{n: streams → 1, ht: spans → K·h resident + h streaming (streams over B chunks), wt: spans → W, g: streams → none, i: streams → none}` | Float16_b (TILE: the input bytes; RM: a lossless tilize copy of bf16) | TILE: reader; RM: compute (tilize) | compute | pass 1 view: every chunk is popped here after its statistics; pass 2: re-read chunks are popped here after normalize | **Shares** the allocation with `cb_input_keep` (group A). Implementation deviation: the design's separate `cb_input_stream` (1) and `cb_input_resident` (2) are one FIFO ring. Both views are pushed in lockstep per chunk, so the ring's order is exactly the pass-2 order (resident tail, then re-read head) and one producer site / one consumer site per view replaces the stream/resident split (half the helper instantiations - required to fit the kernel-config ring under `--dev`). The byte total equals the design's `STREAM·(TILE·D_IN + RM·1)·T + K_alloc·T`. |
| `cb_input_keep` (3) | same allocation as (2) (alias group A) | `K·T` resident tail held from pass 1 into pass 2, plus the re-read chunk | same as (2) | Float16_b | TILE: reader; RM: compute | compute | pass 1: the streamed head chunks `0..B−K−1` are skipped (popped unread); the tail `B−K..B−1` stays fronted → pass 2 normalize pops it, then each re-read | **Shares** with `cb_input_tiles` (group A): the second read cursor that keeps the tail resident without offsets. The reader (TILE) / tilize (RM) reserves both views before writing a chunk, so a resident chunk's bytes are never overwritten before pass 2 pops them. |
| ~~`cb_x_squared` (4)~~ | **deleted (Perf 1)** | — | — | — | — | — | — | Pass 1 accumulates `x − k` and `x·(x − k)` elementwise in fp32 DEST over the chunk's tile rows (`moments_rows`, raw LLK) and collapses each column once per window; there is no intermediate. Group B (x² / output) is gone with it. |
| `cb_output_tiles` (5) | DRAM: the **writer's share** of the pass-2 output, `out_writer_chunks·T` pages of 2 KB (2 when L1 has a chunk to spare after the resident ring, else 1; Perf 2). Sharded TILE: **the output shard** (in place: the input shard). Sharded RM: not used (pass 2 pack-untilizes into `cb_output_rm`) | DRAM: 1–2 chunks (normalize ∥ the writer's DRAM write) | `{n: streams → 1, ht: spans → h (streams over B chunks), wt: spans → W, g: streams → none, i: streams → none}` | Float16_b (output ABI: output dtype = input dtype) | compute | writer | pass 2 | **Not shared** (Perf 1: group B dissolved with `cb_x_squared`). Perf 2 (`dram_dynamic_balance`): the packer fills whichever of (5) / `cb_output_b` (27) has a free chunk slot, so the old 2-chunk ring is now split 1 + 1 (or 2 + 1) across the two DM RISCs. |
| `cb_neg_one` (6) | 1 (constant) | 1 | `{n: streams → none, ht: streams → none, wt: streams → none, g: streams → none, i: streams → none}` | Float16_b (exact −1) | compute (`FillScalar(−1)` once per kernel, unit roots only) | compute (root) | whole kernel | **Not shared.** Perf 2 (`combine_tail`): in0 of the root finalize, `−k = (−1) × [k row; zeros]` and the sender sum `(−1) × gathered`, replacing the ones tiles (17). Pass 2 no longer reads it (Perf 2 `normalize_bcast_srcb` seeds DEST with `0 − b'`). |
| `cb_sum_acc` (7) | `W` (alias group E, with `cb_chan_scale`) | `W`: every channel's running Σ(x−k) is live across the windows | `{n: streams → 1, ht: streams → (accumulates over B chunks), wt: spans → W, g: streams → none, i: streams → none}` | Float32 + `UnpackToDestFp32` (an exact reload; a TF32 reload biases the sum by about `B·2⁻¹²`) | compute | compute | pass 1 spill between DEST windows (multi-window blocks only), then the publish | **Shares** with `cb_chan_scale` (group E). The publish reads accumulator tile i and packs chan tile i over the same bytes (in-place lockstep). Perf 1: a single-window block (sharded `K == B`) never spills: the column collapse packs straight to the chan CBs. The `−n·k` seed (old CB 23) is gone: the pilot row holds `−k`, so DEST accumulates `x + (−k)` directly. |
| `cb_sumsq_acc` (8) | `W` (alias group C, with `cb_chan_work`) | `W` | `{n: streams → 1, ht: streams → (accumulates over B chunks), wt: spans → W, g: streams → none, i: streams → none}` | Float32 + `UnpackToDestFp32` | compute | compute | pass 1 (Σ x·(x−k)) spill, then the publish (→ `cb_chan_work`) | **Shares** with `cb_chan_work` (group C): the same in-place lockstep publish as group E. |
| `cb_chan_scale` (9) | `W` (alias group E) | `W` | `{n: streams → 1, ht: streams → none, wt: spans → W, g: streams → none, i: streams → none}` | Float32 | compute | compute | Σ(x−k): publish → collapse. Then a = rstd × gmap (expand matmul) → end of pass 2. | **Shares** with `cb_sum_acc` (group E) and with itself across the two sequential lifetimes. `a` is live through all of pass 2 together with `b'`, the input views and the output. |
| `cb_chan_work` (10) | `W` (alias group C) | `W` | `{n: streams → 1, ht: streams → none, wt: spans → W, g: streams → none, i: streams → none}` | Float32 | compute | compute | Σx(x−k): publish → collapse. Then b' = [mean·rstd, e] × [gmap; bmap] (expand matmul) → end of pass 2 | **Shares** with `cb_sumsq_acc` (group C). Perf 1: the expand writes b' in one matmul, so the `m = mean·a` hand-off view (old CB 11) is gone. |
| ~~`cb_chan_shift` (11)~~ | **deleted (Perf 1)** | — | — | — | — | — | — | The expand matmul emits b' directly; no `m = mean·a` intermediate. |
| `cb_group_map` (12) | `W` | `W` (all map tiles are read by every collapse and expand) | `{n: streams → none, ht: streams → none, wt: spans → W, g: spans → G_core (lanes in a tile), i: streams → none}` | Float16_b (exact 0/1 values; dataflow-generated) | reader | compute | whole kernel (block-invariant) | **Not shared.** It is a constant used by every block's collapse and expand (matmul in1 with NoWaitNoPop). |
| `cb_gamma_rows` (13) | `(HAS_GAMMA + HAS_BETA)·W` (iff γ; β-only is refused) | all: the reader builds both maps in the combine's stall shadow, ahead of the expand | `{n: streams → none, ht: streams → none, wt: spans → W, g: spans → G_core (gamma map rows), i: streams → none}` | Float16_b (affine input bytes) | reader | compute | load (stall shadow) → expand | **Not shared.** Perf 1: the expand matmul's in1, `[group lane × channel]` per tile: W gamma-map tiles (`gamma[c]` at row `g(c)`, column `c`'s lane), then W beta-map tiles (row 0 = `−beta[c]`). Same bytes as the old γ rows + β rows (14). |
| ~~`cb_beta_rows` (14)~~ | **deleted (Perf 1)** | — | — | — | — | — | — | The beta map rides behind the gamma map in (13). |
| `cb_local_partials` (15) | 2 | 2 | `{n: streams → 1, ht: streams → none, wt: streams → none (collapsed), g: spans → G_core, i: streams → 1 slot}` | Float32 | compute | writer | collapse → gather | **Not shared** (Perf 2: its group-D partner `cb_mr_e` (18) is gone). |
| `cb_gathered_partials` (16) | `2R` (a constant slot count: `R = ceil(cpi/32)`) | `2R` | `{n: streams → 1, ht: streams → none, wt: streams → none, g: spans → G_core, i: spans → cpi (R tile rows)}` | Float32 (an fp32 payload from DEST, moved by the NoC) | writer | compute | combine (root); remote rows can land any time after a sender's collapse | **Not shared.** It is a remote-written landing zone: senders write it at times set by *their* progress, which overlaps this core's pass 1. The remote-address invariant also requires capacity = quantum = 2R, and an alias group would force a common `total_size`. |
| ~~`cb_ones_tiles` (17)~~ | **deleted (Perf 2)** | — | — | — | — | — | — | The sender sum uses the −1 tile (6) as in0 (`combine_tail` fraw); the finalize gets `−inv_n`. |
| ~~`cb_mr_e` (18)~~ | **deleted (Perf 2)** | — | — | — | — | — | — | The finalize emits `mean·rstd` itself and the stats multicast carries `[rstd, mean·rstd, e]`, so the expand reads its in0 straight from `cb_group_stats` (20). Group D is now `cb_local_partials` alone. |
| `cb_stats_out` (19) | 2 | 2 | `{n: streams → 1, ht: streams → none, wt: streams → none, g: spans → G_core, i: streams → none}` | Float32 | compute | writer | finalize → mcast (root) | **Not shared.** It is concurrent with `cb_group_stats`, because the root's mcast/local copy reads it while that CB is being written. It must also not join group D, because the finalize chain reads `cb_group_sums` while packing this CB. |
| `cb_group_stats` (20) | 3 | 3 | `{n: streams → 1, ht: streams → none, wt: streams → none, g: spans → G_core, i: streams → none}` | Float32 | writer | compute | mcast landing → expand | **Not shared.** It is a remote-written landing zone (the root sets the write time), and the remote-address invariant requires capacity = quantum. Perf 2: `[rstd, mean·rstd, e]`; the multicast carries pages 0–1, page 2 (`e = 1.0`) is written once by the writer and re-credited per block. |
| `cb_affine_staging` (21) | 1 page of `align_up((HAS_GAMMA + HAS_BETA)·W·64, 64)` B (iff any affine) | 1 | `{n: streams → none, ht: streams → none, wt: spans → W (64 B per tile), g: streams → none, i: streams → none}` | raw bytes (dataflow-only payload) | reader | reader | load (affine) | **Not shared.** It is the copy source for (13)/(14) and is concurrent with them. It is ≤ 5 KB, with a non-tile page size. |
| `cb_output_rm` (1) | sharded RM only: **the output shard** (zero-copy, `cb_descriptor_from_sharded_tensor`; 2 KB views over `sh·sw·2` B). In place (R2): **the input shard** — the same bytes as `cb_input_rm` | the whole shard | `{n: streams → 1, ht: spans → rows_core (B chunks of h), wt: spans → the shard width, g: streams → none, i: streams → none}` | Float16_b (output ABI) | compute (untilize) | writer (credit drain only, no NoC) | pass 2 | **Not static CB space** — borrowed from the output tensor (in place: from the input tensor, so it **shares** the shard with `cb_input_rm`: pass 2 untilizes a chunk only over rows already tilized — in pass 1 for resident chunks, just before for re-tilized ones). Pass 2 untilizes each normalized chunk straight into the shard rows it came from. |
| `cb_pilot_rows` (22) | `W` | `W` (row 0 of each tile: `−k` of the lane's group; rows 1..31 zero) | `{n: streams → 1 per block, ht: streams → none, wt: spans → W, g: spans → G_core (one k per group), i: streams → none}` | Float16_b (`k` is a data sample, exactly bf16; the sign flip is exact) | reader | compute | pass 1 (`add_tiles_bcast<ROW>` srcB) | **Not shared.** Live through every pass-1 chunk. Perf 1: one copy (the `m`-fold replication went with the chain), and it holds `−k`. |
| ~~`cb_sum_seed` (23)~~ | **deleted (Perf 1)** | — | — | — | — | — | — | DEST accumulates `x + (−k)` from zero; no `−n·k` seed. |
| `cb_group_shift` (24) | 1 page of 4 KB | 1 (lane g = `k_g` in row 0; rows 1..31 zero) | `{n: streams → 1, ht: streams → none, wt: streams → none, g: spans → G_core, i: streams → none}` | Float32 (Perf 2: no longer `UnpackToDestFp32`; it is matmul in1) | reader | compute (root finalize) | block start → finalize (root); allocated on every core (uniform CB layout) | **Not shared.** Perf 2: zeroed once from MEM_ZEROS at the first pilot step, row 0 rewritten and pushed at each pilot step; the finalize forms `−k` on every row as `(−1) × [k row; zeros]`, so nothing is replicated. |
| `cb_output_b` (27) | DRAM only: `T` pages of 2 KB (`OUT_READER_CHUNKS = 1`) | 1 chunk | same as (5) | Float16_b (output ABI) | compute | reader | pass 2 | **Not shared.** Perf 2 (`dram_dynamic_balance`): the reader's share of the pass-2 output, written to DRAM on the reader's NoC. |
| `cb_out_seq_a` (28) / `cb_out_seq_b` (29) | DRAM only: 8 pages of 16 B each (uint32) | ≤ 2 + the writer's depth | `{n: streams → 1, ht: streams → 1 per chunk, …}` | UInt32 | compute (packer) | writer (28) / reader (29) | pass 2 | **Not shared.** The pass-2 index of each chunk packed into (5) / (27); `0xFFFFFFFF` closes the block. 128 B each. |

### Deviations from the planner's inventory (implementation)

| Change | Why | L1 effect |
|--------|-----|-----------|
| `cb_input_stream` (1) + `cb_input_resident` (2) / `_keep` (3) → one input ring with views (2) / (3) | One producer and one consumer site per view; pass-2 order falls out of FIFO order; halves the input-side helper instantiations (`--dev` kernel-config ring) | none (same bytes: `K_alloc·T + STREAM·(TILE·D_IN + RM)·T`) |
| Last-chunk reduce variants → accumulate every chunk + one exact publish copy | Halves the reduce instantiations (`--dev` kernel-config ring) | none (accumulators were already W pages; lifetime extends over the last chunk) |
| `UnpackToDestFp32` on `cb_group_sums` (18) | finalize reloads Σx / Σx² with `CopyTile`; a TF32 reload would round them | none |
| Writer split by role (`IS_ROOT` CT arg, two kernel binaries) | the member binary carries no zero-fill / sender code | none |
| R1: pilot shift `k` per group (22), exact `−n·k` seed (23), root group shift (24) | shifted moments Σ(x−k), Σx·(x−k): the variance cancellation scales with `|mean|·std`, not `mean²` | `+2 KB·W + 4 KB·W + 4 KB` |
| R1: accumulators alias the chan CBs (groups E, C) | the publish copy is an in-place lockstep hand-off (tile i read, tile i packed) | `−8 KB·W` (pays for the R1 additions) |
| R1: compute split by role (`IS_ROOT` CT arg) | the root-only combine/finalize is not in the member binary; keeps the RM root binary inside the kernel-config ring under `--dev` | none |
| R1: two-op per-chunk chains blocked over the DEST window (`CHAIN_BLOCK = DEST_AUTO_LIMIT`, `PerBlockSize` pack) | one FPU init per element per window instead of per tile | none (`cb_x_squared` / `cb_output_tiles` capacity `T ≥` the window) |
| R1: pilot rows replicated `m = pilot_repeat(W, h)` times (pass 1); `W = 1` pass 2 walks the chunk as `tiles(rows)` with `Scalar` a / b′ | the chain's DEST window only spans one walk row, so at `W < 4` it never filled (flux1 1M×64 at W = 1: 912 → 869 µs) | `+2 KB·(m−1)·W` (≤ 6 KB, only at `W < 4`) |
| R2: in place — the output CBs (`cb_output_tiles` TILE / `cb_output_rm` RM) are borrowed from the **input** shard | the result overwrites the input; no second shard exists | `+sh·sw·2 B` of budget (the output shard is no longer subtracted) |
| R2: sharded RM residency solved as `K_full` (stream-free footprint) first, `K_part` (with the stream slot) only when `K_full < B` | R1 solved every `K` against the streaming footprint, so a shard whose whole tilized copy fit was planned one chunk short and re-tilized it in pass 2 (same ring bytes) | none (same ring bytes; 16384×640 in place 91.5 → 86.9 µs) |
| Perf 1: x·(x−k) intermediate (4), sum seed (23), reduce scaler, group sums (18), chan shift (11) deleted; −1 tile (6) | moments accumulate in DEST; finalize inside the sender-sum DEST window; expand as two matmuls; pass 2 as two accumulating multiplies | `−4 KB·T − 4 KB·W − 4 KB` (+ `−2 KB·(m−1)·W`) |
| Perf 1: β rows (14) folded into (13) as the beta map; `cb_mr_e` (18) a view on the local partials (group D) | the expand's in1 is one `[gmap; bmap]` block, in0 `[mean·rstd, e]` | none |
| Perf 1: sharded paths allocate no output ring (TILE: output shard; RM: pack-untilize into the shard) | no tile copy of the output | `−2 KB·D_OUT·T` on sharded |
| Perf 2: ones tiles (17) and `cb_mr_e` (18) deleted; `cb_group_stats` (20) 3 pages; −1 tile (6) moves to the root finalize | the finalize uses (−1) as in0 and emits `mean·rstd`; the expand reads `[rstd, mean·rstd, e]` from (20) | `−2 KB·R + 4 KB` |
| Perf 2: DRAM output split into the writer's share (5) and the reader's (27), with sequence CBs (28, 29) | each DM RISC writes its pass-2 share on its own NoC | `+256 B` (+ `2 KB·T` when the writer's share takes a spare chunk) |

### Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `W = Wt_core` | `1 ≤ W`, and `F_fixed(W, T, R) ≤ L1_budget` | `_choose_gx` admits only the `gx` that satisfy it; otherwise `NotImplementedError` (regime `cross_column_group_combine`). |
| `G_core` | `≤ 32` | `_choose_gx` (one tile row of lanes) |
| `h = chunk_ht` | `1 ≤ h ≤ rows_core_max` | `h = clamp(floor(CHUNK_TILES_TARGET / W), 1, rows_core_max)` |
| `T = h·W` | `T ≤ max(CHUNK_TILES_TARGET, W)` | from `h` |
| `CHUNK_TILES_TARGET` | 32 (host constant; `TUNING_OVERRIDES["chunk_tiles_target"]`) | — |
| `B = ceil(rows_core / h)` | unbounded op extent; appears only through `K_alloc = min(B_max, K_max)` | residency predicate |
| `K_max` | `floor((L1_budget − F_fixed)/(2 KB·T))` | closed form below |
| `K_alloc` | `min(B_max, K_max)`, or `min(override, K_max)` | residency predicate (`dram_resident` iff `B_max ≤ K_max`) |
| `cpi` | `1 ≤ cpi ≤ P` | Work Distribution |
| `R = ceil(cpi/32)` | `≤ ceil(P/32)` (≤ 5 for P ≤ 160) | from `cpi` |
| `D_IN`, `D_RM`, `D_OUT` | 2, 2, 2 (host constants) | — |
| `HAS_GAMMA`, `HAS_BETA`, `RM`, `TILE` | 0/1 | the call |
| `STREAM` | `[B_max > K_alloc]` | residency predicate |
| `L1_budget` | `l1_cb_budget(device)` = `ttnn.get_memory_view(device, L1).total_bytes_per_bank` (L1 end − allocator base; 1 460 288 B on BH), capped at the lowest live L1 buffer offset in the view's block table. **Verifier fix:** the planner's `ttnn.get_max_worker_l1_unreserved_size()` is measured from the KERNEL_CONFIG base and includes the 70 688 B kernel-config ring that sits below the allocator base, so every shape solved to within 70 KB of it overflowed L1 (6 DRAM perf cases OOMed by 9–54 KB). The block-table cap excludes caller-owned L1 tensors (an R1 shard) automatically; CBs borrowed from a shard are not static CB space. | device query |

### Per-core footprint

```
F = 2 KB · [ RM·¬SHARDED·D_RM·T
             + ¬(SHARDED ∧ TILE)·ring(T, K_alloc, STREAM)
             + 1
             + W·(2 + HAS_GAMMA + HAS_BETA) ]
  + ¬SHARDED · (2 KB·(D_OUT + [out_writer_chunks = 2])·T + 2·128 B)
  + 4 KB · [ 2·W + 2·R + 8 ]
  + align_up((HAS_GAMMA + HAS_BETA)·W·64, 64) B
```
(Perf 1: the `4 KB·T` x·(x−k) intermediate (group B), the `+W` sum seed (23), the `m`-fold pilot replication and the
reduce scaler are gone; the `1` is now the −1 tile (6); the output ring exists only on DRAM (sharded TILE packs into
the output shard, sharded RM pack-untilizes into it). Perf 2: the `R` ones tiles (17) and `cb_mr_e` (18) are gone,
`cb_group_stats` has 3 pages (`+1` in the 4 KB term); on DRAM the output is split into the writer's share (5) and the
reader's (27) plus two 128 B sequence CBs, and the writer's share takes a second chunk only when one is spare after
the resident ring (`plan_or_raise`). `footprint_bytes()` in the descriptor is the executable form.
Refinement 1 history: `W·(1 + …)` → `W·(1 + m + …)` added the pilot rows; the accumulators alias the chan CBs.)

What each 4 KB term covers:

| Term | Buffers |
|------|---------|
| `2·W` | group E (`cb_sum_acc` / `cb_chan_scale`), group C (`cb_sumsq_acc` / `cb_chan_work`) |
| `2·R` | `cb_gathered_partials` |
| `8` | `cb_local_partials` (2) + `cb_stats_out` (2) + `cb_group_stats` (3) + `cb_group_shift` (1) |

### Block-sharded footprint (Refinement 1)

The input and output shards are caller-owned L1 buffers, not static CB space: `l1_cb_budget(device)` is capped
below the lowest live L1 buffer, and `_plan_sharded` subtracts the output shard (allocated after the plan).
Zero-copy CBs over the shards (`ttnn.cb_descriptor_from_sharded_tensor`):

| Layout | Input shard backs | Output shard backs | Static CBs dropped from `F` |
|--------|-------------------|--------------------|-----------------------------|
| TILE | the input ring, both views (2)/(3): `K = B`, the whole shard is resident | `cb_output_tiles` (5) | the ring and group B's output half (`x·(x−k)` keeps its own `4 KB·T`) |
| RM | `cb_input_rm` (0), stick pages viewed as 2 KB tile pages (`k = lcm(sw,32)/sw` sticks per virtual tile row) | `cb_output_rm` (1) — pass 2 untilizes into it | `cb_input_rm`'s `D_RM·T`; the tilized ring stays (`K_alloc·T`, `K_max` re-derived against the budget left beside both shards) |

### In place (Refinement 2)

The output tensor **is** the input tensor: `cb_output_tiles` (TILE) / `cb_output_rm` (RM) are
`cb_descriptor_from_sharded_tensor(input)`, the same bytes as the input ring (TILE) / `cb_input_rm` (RM). No CB is
added; the plan's budget simply stops subtracting an output shard:

`L1_budget(in place) = l1_cb_budget(device)` (already capped below the input shard) vs
`l1_cb_budget(device) − align_up(sh·sw·2, 64)` out of place.

Write ordering (why no extra buffer is needed): TILE pass 2 packs tile i over the tile i it just unpacked (unpack →
math → pack per tile; the unpacker only runs ahead onto tiles not yet packed). RM pass 2 untilizes a chunk only over
rows already tilized: resident chunks in pass 1, re-read chunks by the `tilize_chunk` right before. The column
members read the pilot from the root's shard at step 0, which precedes their gather and so the root's pass 2.

**Residency (RM).** `K_full = floor((L1_budget − F(K = 0, STREAM = 0)) / (2 KB·T))`; if `K_full ≥ B` the shard is
fully resident (`K = B`, no stream slot). Otherwise `K = K_part = floor((L1_budget − F(K = 0, STREAM = 1)) / (2 KB·T))`
(the ring also holds the re-tilized chunk in flight). R1 used `K_part` for both, one chunk short whenever the whole
copy fit with less than one chunk to spare.

Worked fits (BH, 8×8, γ+β, RM, `L1_budget` on an otherwise empty L1 = 1 460 288 B minus the input shard):

| Case | Shard | `k`, `W`, `rows_core` | `L1_budget` | `F(h)` with `K = B` | Chosen | Footprint / margin |
|------|-------|-----------------------|-------------|---------------------|--------|--------------------|
| `unet_rm_ip_16384x640` | 2048×80 = 320 KB | 2, 5, 32 | 1 132 608 B | `471 552 + 20 480·h` | `h = 32` (`T = 160`), `B = 1`, `K = 1` | 1 126 912 B / 5 696 B |
| `unet_rm_ip_16384x960` | 2048×120 = 480 KB | 4, 15, 16 | 968 768 B | `840 320 + 61 440·h` | `h = 2` (`T = 30`), `B = 8`, `K = 8` (`h = 4` needs 1 086 080 B) | 963 200 B / 5 568 B |

(Pre-Perf 1 fits; Perf 1 deleted the `4 KB·T` term, so both shapes now fit with more margin.) The `h` term was the Float32 x·(x−k) intermediate (group B, `4 KB·T`). At `h = 32` it is 640 KB, twice the shard —
the largest single buffer of the in-place RM footprint, and the lever if a later refinement wants L1 back (a smaller
`h` trades it for more chunks). Out of place neither shape is a TARGET cell: 16384×640 plans `h = 16, K = B = 2`
(804 928 B budget); 16384×960 does not fit beside two 480 KB shards.

Chunk = the per-core block: `h` is the coarsest divisor of the shard's tile rows whose fully-resident footprint
fits (the whole shard whenever it fits — every model geometry in `feature_spec`); `chunk_tiles_target` caps it.
RM falls back to partial residency (`K < B`, head chunks re-tilized from the input shard in pass 2) only when no
fully-resident chunk fits.

`F_fixed` is `F` with `K_alloc = 0` and `STREAM = 1`. That gives `K_max = floor((L1_budget − F_fixed)/(2 KB·T))`, so the residency is the *only* term solved against the budget.

What each term scales with:

| Terms | Scale with |
|-------|------------|
| `K_alloc·T` | residency knob × chunk knob |
| `D·T` terms and the `4 KB·T` group B | chunk knob (`CHUNK_TILES_TARGET`) and buffer depths |
| `W` terms | `gx` choice (channel slice) |
| `R` terms | `cpi` |
| everything else | constant |

Worked examples (BH, `L1_budget` = 1 460 288 B, P = 110; recomputed at Refinement 1 from the descriptor's
`_choose_gx` / `footprint_bytes()`, γ+β, TILE):

| Case | Parameters | `K` / `B_max` | `ρ` (non-resident fraction) | Footprint | Regime |
|------|------------|---------------|-----------------------------|-----------|--------|
| `vae_dram_1048576x128` | gx = 2, W = 2, h = 16, T = 32, cpi = 55, R = 2, m = 2 | 16 / 38 | 0.58 | 1 407 360 B | `dram_partial_resident` |
| `vae_dram_262144x256` | gx = 2, W = 4, h = 8, T = 32, cpi = 55, m = 1 | 16 / 19 | 0.16 | 1 444 480 B | partial |
| `vae_dram_262144x512` | gx = 2, W = 8, h = 4, T = 32, cpi = 55, m = 1 | 14 / 38 | 0.63 | 1 395 840 B | partial |
| `vae_dram_1048576x256` | gx = 2, W = 4, h = 8, T = 32, cpi = 55, m = 1 | 16 / 75 | 0.79 | 1 444 480 B | partial |
| `flux1_dram_1048576x64` | gx = 2, W = 1, h = 32, T = 32, cpi = 55, m = 4 | 17 / 19 | 0.11 | 1 454 336 B | partial |
| `mochi_dram_8x407040x128` | gx = 1, W = 4, h = 8, T = 32, cpi = 13 (104 cores), m = 1 | 16 / 123 | 0.87 | 1 434 240 B | partial |
| `vae_dram_65536x512` | gx = 2, W = 8, h = 4, T = 32, cpi = 55, m = 1 | 10 / 10 | 0 | 1 002 624 B | `dram_resident` |
| `refiner_dram_16384x1152` | gx = 4, W = 9, h = 3, T = 27, cpi = 27, m = 1 | 7 / 7 | 0 | 724 224 B | `dram_resident` |
| `flux1_dram_16384x256` | gx = 4, W = 2, h = 10, T = 20, cpi = 27, m = 2 | 2 / 2 | 0 | 250 240 B | `dram_resident` |

`F_fixed` for `vae_dram_1048576x128` is 358 784 B. Of that, 256 KB is the two `T`-scaled terms: the TILE stream depth
`D_IN·T` (128 KB) and group B `4 KB·T` (128 KB). Those two terms are the residency lever for the streaming rows: one
resident chunk is `2 KB·T` = 64 KB.

## Data-movement budget

`X` is the input bytes. `ρ = Σ_cores (B−K)·T / Σ_cores B·T` is the non-resident fraction (0 in `dram_resident`). `P_used = U·cpi`, or `P` when `U ≥ P`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | `1 + ρ` (block-sharded: **0** — the shard is consumed in place) | The tail residency keeps `K = min(B, K_max)` chunks per core in L1 across the combine. Only the `B−K` head chunks are re-read in pass 2. R1's pilot adds `2·W·64 B` per core (TILE: rows 0–1 of faces 0/1 of the slice's first tile row; RM: one stick segment) — read in chunk 0's barrier. | block-sharded: the pilot comes from the unit root's shard over the NoC (`2·W·64 B` TILE / `P·2 B` RM per core) |
| output | 1 (block-sharded: **0** — packed / untilized straight into the output shard; in place that shard is the input's, so no second L1 buffer either) | Written once, in pass 2 | 0 |
| gamma | `P_used·W·64 B` total (each core reads its own slice once per block) | Reuse-shared across a unit's `cpi` cores, read per core (`affine_mcast` deferred) | 0 |
| beta | same as gamma | same | 0 |
| per-channel / per-group partials | 0 (never leave L1) | Collapsed on-core to 2 × 32 lanes before the combine | gather: `P_used × 256 B` (4 × 64 B rows) + `P_used` semaphore incs |
| group stats | 0 | — | mcast: `U × 8 KB` sent, delivered to `cpi` cores per unit |

**Totals per tier:**

| Tier | Bytes |
|------|-------|
| DRAM | `X·(2 + ρ) + 2·P_used·W·64 B` |
| Cross-core NoC | `P_used·256 B + U·8 KB` (multicast) |
| Core-local re-reads | pass 2 reads the resident `K·T` tiles from L1 (≈ `X·(1−ρ)`) |

> Cheapest-traffic split considered: `n × xs × ht` row split with every chunk resident (input crosses DRAM once) — delta vs. the implemented split: 0 whenever `B_max ≤ K_max` (`dram_resident`); otherwise `ρ·X` DRAM bytes, which no split can remove because every core already holds a slice and `X` exceeds the aggregate free L1 `P·K_max·T·2 KB`. Implemented: the same `n × xs × ht` split with tail residency `K = min(B, K_max)`. Not deferred — the implemented split **is** the cheapest; the partial-resident tail is the maximum residency the L1 allows, and the variable-`cpi` lamp is the only remaining lever (occupancy, not traffic).
