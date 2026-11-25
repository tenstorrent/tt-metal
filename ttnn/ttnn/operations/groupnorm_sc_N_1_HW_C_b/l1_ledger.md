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
| `cb_x_squared` (4) | `T` pages of 4 KB (alias group B; `total_size = 4 KB·T`) | `T`: a sequential helper intermediate (square → reduce) needs one full reduce block | `{n: streams → 1, ht: spans → h, wt: spans → W, g: streams → none, i: streams → none}` | Float32 (matches the fp32 DEST: x² is not rounded to 16 bits before the reduce) | compute | compute | pass 1 | **Shares** with `cb_output_tiles` (group B): the lifetimes are disjoint (pass 1 vs pass 2). Before the first x² push of block `b > 0`, compute drains the output view with `cb_reserve_back(cb_output_tiles, 2T)`. |
| `cb_output_tiles` (5) | `2T` pages of 2 KB (group B) | `2T`: double buffer (normalize ∥ DRAM write) | `{n: streams → 1, ht: spans → h (streams over B chunks), wt: spans → W, g: streams → none, i: streams → none}` | Float16_b (output ABI: output dtype = input dtype) | compute | writer | pass 2 | **Shares** with `cb_x_squared` (group B). |
| `cb_reduce_scaler` (6) | 1 (constant) | 1 | `{n: streams → none, ht: streams → none, wt: streams → none, g: streams → none, i: streams → none}` | Float16_b (the bf16 packed format the reduce scaler requires) | reader | compute | whole kernel | **Not shared.** It is a constant read in pass 1 of every block. Its tile layout is the helper's private pool-type-aware fill (reduce_helpers_dataflow.hpp:70), so it cannot stand in for the all-ones tile. |
| `cb_sum_acc` (7) | `W` | `W`: every channel's running Σx is live across the chunks | `{n: streams → 1, ht: streams → (accumulates over B chunks), wt: spans → W, g: streams → none, i: streams → none}` | Float32 + `UnpackToDestFp32` (an exact reload; a TF32 reload biases the sum by about `B·2⁻¹²`) | compute | compute | pass 1, chunks `0..B−1`, then the publish copy | **Not shared.** It is concurrent with every pass-1 buffer. Implementation deviation: every chunk (including the last) accumulates here, and one exact `CopyTile` (UnpackToDestFp32 source) publishes the W tiles to `cb_chan_scale` after pass 1. The design's last-chunk reduce variant that packs straight into `cb_chan_scale` doubled the reduce instantiations (kernel text over the kernel-config ring under `--dev`); the copy costs W tile moves per block. Aliasing with `cb_chan_scale` stays illegal: the publish copy reads this CB while packing that one. |
| `cb_sumsq_acc` (8) | `W` | `W` | `{n: streams → 1, ht: streams → (accumulates over B chunks), wt: spans → W, g: streams → none, i: streams → none}` | Float32 + `UnpackToDestFp32` | compute | compute | pass 1, chunks `0..B−1`, then the publish copy (→ `cb_chan_work`) | **Not shared**, for the same reason as `cb_sum_acc` (the publish copy reads it while packing `cb_chan_work`). |
| `cb_chan_scale` (9) | `W` | `W` | `{n: streams → 1, ht: streams → none, wt: spans → W, g: streams → none, i: streams → none}` | Float32 | compute | compute | Σx: publish copy (after pass 1) → collapse. Then a = rstd·γ: expand → end of pass 2. | **Shares with itself** across the two sequential lifetimes (Σx, then a). It is not shared with anything else: `a` is live through all of pass 2 together with `b'`, the input views and the output. |
| `cb_chan_work` (10) | `W` (alias group C) | `W` | `{n: streams → 1, ht: streams → none, wt: spans → W, g: streams → none, i: streams → none}` | Float32 | compute | compute | Σx² (publish copy) → rstd_ch → mean_ch → b' (with β) through pass 2 | **Shares** with `cb_chan_shift` (group C): in-place lockstep hand-offs at 5d/5e. Every quantum is W in capacity W, so both views are at base at each hand-off. |
| `cb_chan_shift` (11) | `W` (alias group C) | `W` | same as (10) | Float32 | compute | compute | m = mean·a (5d → 5e); without β, b' through pass 2 | **Shares** with `cb_chan_work` (group C). |
| `cb_group_map` (12) | `W` | `W` (all map tiles are read by every collapse and expand) | `{n: streams → none, ht: streams → none, wt: spans → W, g: spans → G_core (lanes in a tile), i: streams → none}` | Float16_b (exact 0/1 values; dataflow-generated) | reader | compute | whole kernel (block-invariant) | **Not shared.** It is a constant used by every block's collapse and expand (matmul in1 with NoWaitNoPop). |
| `cb_gamma_rows` (13) | `W` (iff γ) | `W`: the reader produces all of them in the stall shadow, ahead of 5b | `{n: streams → none, ht: streams → none, wt: spans → W, g: streams → none, i: streams → none}` | Float16_b (affine input bytes) | reader | compute | load (stall shadow) → 5b | **Not shared.** Concurrent with the accumulators and the pass-1 input buffers, because it is filled while compute finishes pass 1 (the stall-shadow decision). Not aliased with `cb_beta_rows` either: both are produced at load time, and β is consumed only at 5e, after γ. |
| `cb_beta_rows` (14) | `W` (iff β) | `W` | same as (13) | Float16_b | reader | compute | load → 5e | **Not shared**, as for (13). Moving its production after 5b would serialize the reader behind Phase D. |
| `cb_local_partials` (15) | 2 (alias group D) | 2 | `{n: streams → 1, ht: streams → none, wt: streams → none (collapsed), g: spans → G_core, i: streams → 1 slot}` | Float32 | compute | writer | collapse → gather | **Shares** with `cb_group_sums` (group D, 2 pages of 4 KB each). The lifetimes are disjoint and ordered by local CB credit on the root: the writer pops this CB before it waits on the gather semaphore; the gathered push follows; compute packs `cb_group_sums` only after waiting on the gathered CB. |
| `cb_gathered_partials` (16) | `2R` (a constant slot count: `R = ceil(cpi/32)`) | `2R` | `{n: streams → 1, ht: streams → none, wt: streams → none, g: spans → G_core, i: spans → cpi (R tile rows)}` | Float32 (an fp32 payload from DEST, moved by the NoC) | writer | compute | combine (root); remote rows can land any time after a sender's collapse | **Not shared.** It is a remote-written landing zone: senders write it at times set by *their* progress, which overlaps this core's pass 1. The remote-address invariant also requires capacity = quantum = 2R, and an alias group would force a common `total_size`. |
| `cb_ones_tiles` (17) | `R` | `R` | `{n: streams → none, ht: streams → none, wt: streams → none, g: streams → none, i: spans → R}` | Float16_b (exact 1.0) | reader | compute | combine (root); the bytes persist across blocks | **Not shared.** Allocated on every used core (CBs are allocated uniformly so remote landing addresses match); filled and consumed only on roots. It is a constant whose bytes are re-pushed without rewriting on every block, so they must never be overwritten. |
| `cb_group_sums` (18) | 2 (alias group D) | 2 | `{n: streams → 1, ht: streams → none, wt: streams → none, g: spans → G_core, i: streams → none (summed)}` | Float32 + `UnpackToDestFp32` (the finalize chain's CopyTile reload of Σ stays exact) | compute | compute | combine (root): sum senders → finalize | **Shares** with `cb_local_partials` (group D). |
| `cb_stats_out` (19) | 2 | 2 | `{n: streams → 1, ht: streams → none, wt: streams → none, g: spans → G_core, i: streams → none}` | Float32 | compute | writer | finalize → mcast (root) | **Not shared.** It is concurrent with `cb_group_stats`, because the root's mcast/local copy reads it while that CB is being written. It must also not join group D, because the finalize chain reads `cb_group_sums` while packing this CB. |
| `cb_group_stats` (20) | 2 | 2 | `{n: streams → 1, ht: streams → none, wt: streams → none, g: spans → G_core, i: streams → none}` | Float32 | writer | compute | mcast landing → expand | **Not shared.** It is a remote-written landing zone (the root sets the write time), and the remote-address invariant requires capacity = quantum = 2. |
| `cb_affine_staging` (21) | 1 page of `align_up((HAS_GAMMA + HAS_BETA)·W·64, 64)` B (iff any affine) | 1 | `{n: streams → none, ht: streams → none, wt: spans → W (64 B per tile), g: streams → none, i: streams → none}` | raw bytes (dataflow-only payload) | reader | reader | load (affine) | **Not shared.** It is the copy source for (13)/(14) and is concurrent with them. It is ≤ 5 KB, with a non-tile page size. |

### Deviations from the planner's inventory (implementation)

| Change | Why | L1 effect |
|--------|-----|-----------|
| `cb_input_stream` (1) + `cb_input_resident` (2) / `_keep` (3) → one input ring with views (2) / (3) | One producer and one consumer site per view; pass-2 order falls out of FIFO order; halves the input-side helper instantiations (`--dev` kernel-config ring) | none (same bytes: `K_alloc·T + STREAM·(TILE·D_IN + RM)·T`) |
| Last-chunk reduce variants → accumulate every chunk + one exact publish copy | Halves the reduce instantiations (`--dev` kernel-config ring) | none (accumulators were already W pages; lifetime extends over the last chunk) |
| `UnpackToDestFp32` on `cb_group_sums` (18) | finalize reloads Σx / Σx² with `CopyTile`; a TF32 reload would round them | none |
| Writer split by role (`IS_ROOT` CT arg, two kernel binaries) | the member binary carries no zero-fill / sender code | none |

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
| `L1_budget` | `ttnn.get_max_worker_l1_unreserved_size()`. R1 subtracts the borrowed shard bytes. | device query |

### Per-core footprint

```
F = 2 KB · [ RM·D_RM·T
             + STREAM·(TILE·D_IN + RM·1)·T
             + K_alloc·T
             + 1 + R
             + W·(1 + HAS_GAMMA + HAS_BETA) ]
  + 4 KB · [ T + 4·W + 2·R + 6 ]
  + align_up((HAS_GAMMA + HAS_BETA)·W·64, 64) B
```

What each 4 KB term covers:

| Term | Buffers |
|------|---------|
| `T` | group B (x² / output) |
| `4·W` | `cb_sum_acc`, `cb_sumsq_acc`, `cb_chan_scale`, group C |
| `2·R` | `cb_gathered_partials` |
| `6` | group D (2) + `cb_stats_out` (2) + `cb_group_stats` (2) |

`F_fixed` is `F` with `K_alloc = 0` and `STREAM = 1`. That gives `K_max = floor((L1_budget − F_fixed)/(2 KB·T))`, so the residency is the *only* term solved against the budget.

What each term scales with:

| Terms | Scale with |
|-------|------------|
| `K_alloc·T` | residency knob × chunk knob |
| `D·T` terms and the `4 KB·T` group B | chunk knob (`CHUNK_TILES_TARGET`) and buffer depths |
| `W` terms | `gx` choice (channel slice) |
| `R` terms | `cpi` |
| everything else | constant |

Worked examples (BH, `L1_budget` ≈ 1.4 MB):

| Case | Parameters | Fixed part | `K_max` | Result |
|------|------------|------------|---------|--------|
| `vae_dram_1048576x128` | W = 2, T = 32, R = 2, TILE, γ+β | `F_fixed` ≈ 2 KB·(64 + 1 + 2 + 6) + 4 KB·(32 + 8 + 4 + 6) + 0.25 KB ≈ 346 KB | ≈ 16 chunks | B ≈ 38 |
| `refiner_dram_16384x1152` | W = 9, T = 27 (h = 3) | 171 tiles per core fit | — | `dram_resident` |

## Data-movement budget

`X` is the input bytes. `ρ = Σ_cores (B−K)·T / Σ_cores B·T` is the non-resident fraction (0 in `dram_resident`). `P_used = U·cpi`, or `P` when `U ≥ P`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | `1 + ρ` | The tail residency keeps `K = min(B, K_max)` chunks per core in L1 across the combine. Only the `B−K` head chunks are re-read in pass 2. | 0 |
| output | 1 | Written once, in pass 2 | 0 |
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
