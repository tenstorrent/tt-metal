# Operation Design: rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (reduction + broadcast normalization); Regime B adds cross-core CCL leg |
| Goal | Root-mean-square normalization over the last dim: `out = x / sqrt(mean(x², dim=-1) + eps) * gamma`. Performance-first: read input from DRAM exactly once, saturate the grid, bound per-core L1 by a constant. |
| Math | `out[..., h, w] = x[..., h, w] * rsqrt( (1/W) * Σ_{j<W} x[..., h, j]²  + eps ) * gamma[w]` |
| Mode | Hybrid (Regime A = embarrassingly-parallel row-resident; Regime B = W-split resident shard + cross-core all-gather plain-sum) |
| References | `references/cross_core_reduction_design.md`; `ttnn/cpp/ttnn/kernel_lib/{mcast_pipe,reduce_helpers_compute,reduce_helpers_dataflow,eltwise_convenience,eltwise_chain,dest_helpers}.hpp`; `/interleaved-parallel`, `/memory-budget-metal`, `/partial-scaler-reduce` skills; existing `fused_distributed_rmsnorm/.../rmsnorm_post_allgather.cpp` |

### The two performance invariants this design pins

- **P1 — read input from DRAM once.** Each core reads only the shard it owns into `cb_input_resident`, holds it resident in L1, and runs *both* passes (sum-of-squares, then normalize) over the resident data. DRAM traffic per element = 1 read + 1 write.
- **P2 — use the whole grid.** The host heuristic chooses Regime A vs B from `(grid, num_tile_rows, Wt, dtype, fp32_dest_acc_en)` and, in Regime B, picks the W-split factor K that *maximizes* active cores.
- **P3 — per-core L1 bounded by a constant.** Every scratch CB (squared, normalized, scaler, stats, gathered partials) is sized to a constant block, never to W. Only `cb_input_resident`/`cb_gamma` scale with the *shard* width `Wt_s`, and the heuristic guarantees `Wt_s` fits the resident budget.

## Parameters

| Name | Type | Required | Valid Range (Phase 0) | Default | CT/RT |
|------|------|----------|-----------------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | bf16, TILE_LAYOUT, tile-aligned (H,W % 32 == 0), interleaved DRAM, rank ≥ 2 | — | — |
| `gamma` | `Optional[ttnn.Tensor]` | no | bf16, TILE_LAYOUT, shape `(1,1,1,W)`, `W` matches input's last dim | `None` | — |
| `epsilon` | `float` | no | > 0 | `1e-6` | RT (compute) — passed as bf16/uint bits to `AddUnary` |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no (keyword-only) | bf16 input + `fp32_dest_acc_en ∈ {True, False}` | `HiFi4 / fp32_dest_acc_en=True / math_approx_mode=False` when `None` | CT (compute config) |
| `inv_W` (derived) | `float` | — | `1.0 / W` | — | RT (compute) — passed as uint bits to `MulUnary` |

`math_fidelity` and `math_approx_mode` are NOT gated — accept any value and forward to the compute `ComputeConfigDescriptor`. Only `fp32` input + `fp32_dest_acc_en=False` is an EXCLUSION, and `fp32` is out of Phase 0 scope, so within Phase 0 both `fp32_dest_acc_en` values are accepted.

## Tensors

### Input
| Property | Requirement |
|----------|-------------|
| Shape | `(..., H, W)`, rank ≥ 2; `H % 32 == 0`, `W % 32 == 0` (Phase 0) |
| Dtype | `bfloat16` |
| Layout | `TILE_LAYOUT` |
| Memory | interleaved DRAM |

### Gamma (optional)
| Property | Requirement |
|----------|-------------|
| Shape | `(1, 1, 1, W)` |
| Dtype | `bfloat16` |
| Layout | `TILE_LAYOUT` |
| Memory | interleaved DRAM |

### Output
| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | `bfloat16` |
| Layout | `TILE_LAYOUT` (matches input) |
| Memory | interleaved DRAM |

### Derived tile dimensions
- `Wt = W / 32` (tiles across the reduced dim)
- `Ht_total = (prod(leading dims) * H) / 32` (total tile-rows; each tile-row spans `Wt` tiles)
- `BLOCK_HEIGHT` (named param, Phase 0 = **1**) — tile-rows processed per work-unit. **Never hardcode `1`** — Phase 1 raises it.
- `Wt_s` — W-tiles each core holds resident. Regime A: `Wt_s = Wt`. Regime B: `Wt_s = Wt / K`.
- `REDUCE_BLOCK` — constant W-chunk for the squared/reduce loop, `= min(Wt_s, DEST_AUTO_LIMIT)` (`DEST_AUTO_LIMIT` = 4 when `fp32_dest_acc_en`, else 8; `dest_helpers.hpp:89-103`).

## Dataflow Strategy

### High-level path (both regimes)
```
DRAM input shard ──read once──▶ cb_input_resident (L1, held)         [reader / NCRISC]
                                       │
            ┌──────────────────────────┴─ PASS 1 (sum of squares) ────────────────┐
            │ for each REDUCE_BLOCK chunk of the resident shard (no pop):          │
            │   square: cb_input_resident[chunk] ─▶ cb_squared                     │
            │   reduce<SUM,REDUCE_ROW,Accumulate>: cb_squared ─▶ cb_partial_sumsq  │  [compute]
            └─────────────────────────────────────────────────────────────────────┘
                                       │  cb_partial_sumsq = local Σx² (bh tiles, col-vector per row)
        ┌── Regime B only: cross-core all-gather + plain sum (see Tensix-to-Tensix) ──┐
        │   reader mcasts cb_partial_sumsq to its W-group, gathers K partials into     │
        │   cb_partials_gathered; compute elementwise-adds K → cb_partial_sumsq (total)│
        └──────────────────────────────────────────────────────────────────────────────┘
                                       │  cb_partial_sumsq = global Σx² over full W
   FINALIZE: eltwise_chain( MulUnary(inv_W) → AddUnary(eps) → Rsqrt ): ─▶ cb_recip_rms  [compute]
                                       │
            ┌─ PASS 2 (normalize) — consumes (pops) cb_input_resident ────────────────┐
            │ for each W-tile block of the resident shard:                            │
            │   mul<cb_input_resident, cb_recip_rms, ·, BroadcastDim::Col> ─▶ tmp      │  [compute]
            │   [gamma] mul<tmp, cb_gamma, cb_output, BroadcastDim::Row>               │
            │   else    tmp == cb_output                                               │
            └──────────────────────────────────────────────────────────────────────────┘
                                       │
cb_output ──write once──▶ DRAM output shard                          [writer / BRISC]
```
Format is **tiles** end-to-end (TILE_LAYOUT in/out): no tilize/untilize. Reader→compute→writer communicate via the CBs above. `epsilon` and `inv_W` reach compute as scalar runtime args (uint bits), consumed by `MulUnary`/`AddUnary` — no scalar CB.

### Unified scaler choice (both regimes use the *same* compute kernel)
The local reduce always computes a plain **SUM** of squares (scaler = `1.0`, prepared via the pool-type-aware overload). The `× inv_W` (= mean) is applied in the finalize SFPU chain. This makes Regime A simply the `K = 1` case of Regime B — one compute kernel, one code path. After the all-gather sum, every core holds the identical global `Σx²` over the full `W`, multiplies by `inv_W = 1/W`, adds `eps`, and `rsqrt`s.

### Tensix-to-Tensix contract (Regime B only — cross-core all-gather, plain sum)
RMSNorm computes **only** `mean(x²)` — no mean subtraction, no variance, no catastrophic cancellation — so the cross-core combine is a **plain elementwise SUM**, fully associative/commutative. **Do NOT use Welford / `welford_combine.h` / the δ² term.** Topology = **Pattern B (all-gather)**: every core in a W-group ships its `bh`-tile partial-sum to all peers and sums redundantly. Payload is one column-tile per tile-row — tiny — so there is no master bottleneck and the redundant K-element add is negligible.

| Contract item | Decision |
|---------------|----------|
| Group | One **rectangular** band of K cores per row-group (validated at host; clean fallback to Regime A if no rectangular partition exists). |
| Transport | `mcast_pipe` `SenderPipe` / `ReceiverPipe` / `McastRect` (`mcast_pipe.hpp:160-290`). **Do NOT hand-roll `noc_semaphore_set_multicast`, corner selection, or data+flag linkage.** Pass canonical low→high coords to `McastRect`; `start_end_for_noc` (`mcast_pipe.hpp:179-181`) auto-swaps the corner per NoC. |
| Pattern | K-round all-gather: in round j, core j is the `SenderPipe` and multicasts its partial block to the K-core rectangle; all cores are `ReceiverPipe`s and land round j's payload in `cb_partials_gathered` slot j. After K rounds every core has all K partials. |
| Combine | compute elementwise-adds the K partial blocks (`add` chain, plain sum) → `cb_partial_sumsq` (global). |
| Semaphores | 2 IDs (1 data-ready flag, 1 consumed counter), created on the **union** of the group's cores. Disjoint groups reuse the same IDs (different L1 cells), so the 16-ID budget is never the constraint. |
| NoC / coords | Feed the pipe **virtual** coords. `consumed` counter host-initialized to 0. `num_dests` = receiver count (+1 only on loopback). Barrier-before-signal handled by the pipe's linked data+flag. Never multicast to only yourself (guard degenerate K=1 — which is Regime A and skips this leg entirely). |

§9 silent-hang checklist (virtual vs physical coords, barrier-before-signal on the gather leg, correct `num_dests`, semaphores on the union, never-mcast-to-self) is mandatory reading for the implementer.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one **block of `BLOCK_HEIGHT` tile-rows** (Phase 0: 1 tile-row = `Wt` tiles wide) |
| Grid | `device.compute_with_storage_grid_size()` (e.g. 8×8 = 64 on Wormhole) |
| Per-core work (A) | a disjoint set of tile-rows, full width `Wt` resident; zero cross-core comms |
| Per-core work (B) | one `Wt/K` W-shard of a row-group; local partial Σx² + 1 all-gather round + normalize own shard |
| Remainder | `split_work_to_cores` (`work_split.hpp:46-47`) hands `ceil` to `core_group_1`, `floor` to `core_group_2` (Regime A). Regime B requires `Wt % K == 0` (Phase 0 tile-aligned LOOSE_CASES all divide cleanly); reject/fall back otherwise. |

### Grid-aware host heuristic (chooses A vs B, picks K)
```
total_cores = grid.x * grid.y
bh          = BLOCK_HEIGHT                      # Phase 0 = 1
gamma_t(wt) = wt if has_gamma else 0
# RESIDENT_BUDGET_TILES derived from 1.2 MB usable L1 minus constant CB overhead
# (output 2·bh, squared REDUCE_BLOCK, normalized REDUCE_BLOCK, scaler 1, stats 2·bh,
#  partials K·bh). Phase 0 value ≈ 560 tiles (bf16 → ~1.12 MB for input+gamma resident).
fits_one_core(wt_s) := bh*wt_s + gamma_t(wt_s) <= RESIDENT_BUDGET_TILES

# ── Regime A: row-parallel, only when it already saturates the grid AND a full row fits ──
if Ht_total >= total_cores and fits_one_core(Wt):
    return RegimeA(num_cores = total_cores)        # split_work_to_cores over Ht_total

# ── otherwise Regime B (few rows underutilize the grid, OR a full row does not fit L1) ──
#   num_row_groups = Ht_total / bh   (Phase 0: = Ht_total)
#   Choose K to MAXIMIZE active cores = num_row_groups * K, subject to:
#     (1) Wt_s = Wt / K fits the budget:        K >= ceil(Wt / RESIDENT_BUDGET_TILES)
#     (2) Wt % K == 0                            (tile-aligned shard split, Phase 0)
#     (3) num_row_groups * K <= total_cores
#     (4) each group is a RECTANGULAR band of the grid
#   K never selected smaller than Regime A would have used — splitting W only ADDS cores.
K = largest divisor of Wt s.t. (1)-(4) hold and num_row_groups*K is maximized
if no valid rectangular K exists:
    fall back to RegimeA(num_cores = min(Ht_total, total_cores))   # documented clean fallback
return RegimeB(K, num_row_groups, rectangular_band_layout)
```
Worked examples (LOOSE_CASES, Phase 0 bf16+gamma): `(1,1,32,16384)` → Ht=1, Wt=512, full row 512+512=1024 > 560 ⇒ B; 1 row-group, K=64 (512/64=8 tiles/core), one 8×8 rectangle, all 64 cores busy. `(1,1,32,32768)` → Wt=1024, B, K=64, 16 tiles/core. `(1,1,64,12288)` → Ht=2, Wt=384, 2 row-groups, K=32 each (384/32=12 tiles/core), two 4×8 bands, all 64 cores busy. Default INPUTS top at W=8192 (Wt=256; 256+256=512 ≤ 560) ⇒ Regime A on one core per row.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Rationale / Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------------------|
| `cb_input_resident` | c_0 | `tile_size(bf16)` = 2048 B | `bh * Wt_s` | bf16 | reader | compute (both passes) | **Resident** — read once (P1), no-pop in pass 1, popped in pass 2. Only CB that scales with shard width. |
| `cb_gamma` | c_1 | 2048 B | `Wt_s` (gamma present) / 0 | bf16 | reader | compute (pass 2) | Resident, read once, reused across all tile-rows this core owns; broadcast-Row multiply in normalize. |
| `cb_scaler` | c_8 | 2048 B | 1 | bf16 (packed) | reader (`prepare_reduce_scaler`) | compute (`reduce`) | SUM scaler = 1.0, col-0 fill (matmul path). Phase 0 tile-aligned ⇒ 1 tile (no partial scaler). |
| `cb_partials_gathered` | c_9 | 2048 B | `K * bh` (Regime B) / 0 | bf16 | reader (mcast all-gather) | compute (sum-combine) | Regime B landing zone for K peers' partials. Constant (K·bh), not W-scaled. |
| `cb_output` | c_16 | 2048 B | `2 * bh` (double-buffer) | bf16 | compute (pass 2) | writer | Streaming output; drained to DRAM as produced — NOT resident. |
| `cb_squared` | c_24 | 2048 B | `REDUCE_BLOCK` | bf16 | compute (square) | compute (reduce) | Compute→compute, sequential ⇒ sized to one block (`REDUCE_BLOCK`), constant. |
| `cb_partial_sumsq` | c_25 | 2048 B | `bh` | bf16 | compute (reduce / combine) | compute (finalize / mcast source) | Per-row local→global Σx² (column-vector tile). Source read by reader's `SenderPipe` in Regime B. |
| `cb_recip_rms` | c_26 | 2048 B | `bh` | bf16 | compute (finalize chain) | compute (pass 2 broadcast) | `rsqrt(mean+eps)` column-vector per tile-row. |
| `cb_normalized` | c_27 | 2048 B | `REDUCE_BLOCK` (gamma present) / 0 | bf16 | compute (mul Col) | compute (mul Row) | Intermediate between the two normalize multiplies; sequential ⇒ one block. Absent when no gamma (single fused write to `cb_output`). |

**CB sync verification** (push == wait):
- `cb_input_resident`: reader pushes `bh*Wt_s`; compute waits `bh*Wt_s` once (held), reads by index both passes, single pop at end of pass 2. ✔
- `cb_scaler`: reader pushes 1; compute `reduce` waits 1, pops 1 at end. ✔
- `cb_squared`: per chunk compute pushes/pops `REDUCE_BLOCK` (last chunk ≤ block). ✔
- `cb_partial_sumsq`: reduce-accumulate produces `bh`; finalize/mcast consume `bh`. ✔
- `cb_partials_gathered`: K all-gather rounds push `K*bh`; combine waits `K*bh`. ✔
- `cb_output`: compute pushes one block per normalize iteration; writer waits the same. ✔

## API Mapping

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Manages own CB ops |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------------|
| boot | raw | `compute_kernel_hw_startup(in, scaler, out)` | reduce_helpers_compute.hpp:29-33 | once at kernel start | — | — | n/a |
| scaler setup | helper | `calculate_and_prepare_reduce_scaler<cb_scaler, SUM, REDUCE_ROW, W>()` (or `prepare_reduce_scaler<cb_scaler, SUM, REDUCE_ROW>(1.0f)`) | reduce_helpers_dataflow.hpp:65-67 / 94-101 | pool-type-aware overload; SUM ⇒ scaler 1.0, col-0 (matmul) fill | — | `cb_scaler` | yes (writes 1 tile) |
| pass 1 square | helper | `square<cb_input_resident, cb_squared, …>(EltwiseShape)` | eltwise_convenience.hpp:108-121 | input lifecycle = no-pop (read resident by index; see Risks); output Streaming | `cb_input_resident` | `cb_squared` | yes |
| pass 1 reduce | helper | `reduce<SUM, REDUCE_ROW, BulkWaitBulkPop>(cb_squared, cb_scaler, cb_partial_sumsq, ReduceInputBlockShape::of(bh, REDUCE_BLOCK), …, Accumulate(cfg, chunk))` | reduce_helpers_compute.hpp:405-420; shape 141-150; Accumulate 194-241 | SUM+REDUCE_ROW (matmul path, col-0 scaler); accumulate across W-chunks | `cb_squared`, `cb_scaler` | `cb_partial_sumsq` | yes |
| combine (B) | helper | `add<cb_partials_gathered+k, cb_partial_sumsq, cb_partial_sumsq, None>(bh)` looped over K | eltwise_convenience.hpp:44-61 | plain elementwise sum of K partials (associative) | `cb_partials_gathered` | `cb_partial_sumsq` | yes |
| finalize | helper | `transform_in_place<cb_partial_sumsq>(EltwiseShape::tiles(bh), MulUnary<>{inv_W_bits}, AddUnary<>{eps_bits}, Rsqrt<>{})` → pack to `cb_recip_rms` | eltwise_convenience.hpp:227-241; Rsqrt eltwise_math.hpp:35-37; AddUnary/MulUnary eltwise_scalar.hpp | DEST-only SFPU chain: `rsqrt(sum*inv_W + eps)`. (Pack to `cb_recip_rms`: use `eltwise_chain` with `CopyTile`→Mul/Add/Rsqrt→`PackTile<cb_recip_rms>` if distinct in/out CB needed.) | `cb_partial_sumsq` | `cb_recip_rms` | yes |
| pass 2 normalize | helper | `mul<cb_input_resident, cb_recip_rms, dst, BroadcastDim::Col>(EltwiseShape)` | eltwise_convenience.hpp:82-99; BroadcastDim eltwise_chain.hpp:425-430 | **Col** = broadcast column-vector (per-row recip) across W tiles | `cb_input_resident`, `cb_recip_rms` | `cb_normalized` or `cb_output` | yes |
| pass 2 gamma | helper | `mul<cb_normalized, cb_gamma, cb_output, BroadcastDim::Row>(EltwiseShape)` | eltwise_convenience.hpp:82-99 | **Row** = broadcast row-vector gamma down the 32 rows | `cb_normalized`, `cb_gamma` | `cb_output` | yes |
| cross-core (B) | helper | `SenderPipe<…>::send` / `ReceiverPipe<…>::receive`; `McastRect` | mcast_pipe.hpp:203-251 (send 221), 262-290 (receive 275), 160-182 (start_end_for_noc 179-181) | K-round all-gather; canonical low→high McastRect coords; 2 semaphores on union | `cb_partial_sumsq` (source) | `cb_partials_gathered` | manages mcast+handshake; caller wires CB reserve/push around landing |
| work split | helper | `split_work_to_cores(grid, Ht_total, row_wise)` | work_split.hpp:46-47 | Regime A row distribution | — | — | n/a |

### Helpers considered and rejected (raw-API fallbacks)
- **`compute_kernel_hw_startup` (raw, boot)** — there is no helper that performs the one-time HW configure; every helper (`reduce`, `eltwise_*`) explicitly requires the caller to call it exactly once first (reduce_helpers_compute.hpp:29-33). This is the documented prerequisite, not a bypass.
- **No raw-API compute hot-path bypass is taken.** Every compute phase (square, reduce, combine-sum, rsqrt finalize, both broadcast multiplies) maps to a kernel-lib helper above. The cross-core transport uses `mcast_pipe` rather than raw `noc_*` calls, per the prompt's explicit directive and the §9 hang-safety rationale. If the implementer finds a measured hot-path win in fusing the two normalize multiplies into a single `eltwise_chain` (Col-broadcast `BinaryFpu` → Row-broadcast `BinaryFpu` → `PackTile`, eliminating the `cb_normalized` round-trip), that is a *sanctioned* optimization — it must be documented at the bypass site and in `changelog.md`, naming the eliminated CB copy as the concrete cost.

## Compute Phases

| # | Operation | Helper? | Input CB (name, tiles, state) | Output CB (name, tiles) | CB State After |
|---|-----------|---------|-------------------------------|-------------------------|----------------|
| 0 | `compute_kernel_hw_startup` once | raw (prereq) | — | — | HW configured |
| 1 | Σx² over shard: per `REDUCE_BLOCK` chunk → `square` then `reduce<SUM,REDUCE_ROW,Accumulate>` | yes | `cb_input_resident` (`bh*Wt_s`, resident no-pop) + `cb_scaler` (1) | `cb_squared` (`REDUCE_BLOCK`) → `cb_partial_sumsq` (`bh`) | input still resident; `cb_partial_sumsq` = local Σx² |
| 2 | *(Regime B)* all-gather K partials + plain-sum combine | yes (`mcast_pipe` + `add`) | `cb_partial_sumsq` (`bh`) → `cb_partials_gathered` (`K*bh`) | `cb_partial_sumsq` (`bh`, overwritten) | `cb_partial_sumsq` = global Σx² over full W |
| 3 | Finalize: `rsqrt(Σx² * inv_W + eps)` | yes (`transform`/`eltwise_chain`) | `cb_partial_sumsq` (`bh`) | `cb_recip_rms` (`bh`) | per-row reciprocal RMS ready |
| 4 | Normalize: `x * recip_rms` (Col) [`* gamma` (Row)] | yes (`mul` ×1 or ×2) | `cb_input_resident` (`bh*Wt_s`, now popped) + `cb_recip_rms` (`bh`) [+ `cb_gamma` (`Wt_s`)] | `cb_normalized` (block) → `cb_output` (`2*bh`) | input drained; output streamed to writer |

## Broadcast Verification

| Phase | Op | CB_A valid region | CB_B valid region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| normalize (recip) | `mul` | `cb_input_resident` — full 32×32 tile | `cb_recip_rms` — column vector (col 0 filled, per-row scalar) | **`BroadcastDim::Col`** — `out[r][c] = x[r][c] * recip[r]`, same recip for all W positions in a row (eltwise_chain.hpp:421-430; mirrors `mul_tiles_bcast_cols` in rmsnorm_post_allgather.cpp:115) |
| normalize (gamma) | `mul` | `cb_normalized` — full 32×32 tile | `cb_gamma` — row vector (row 0 filled, per-column scalar) | **`BroadcastDim::Row`** — `out[r][c] = norm[r][c] * gamma[c]`, same gamma down the 32 rows (mirrors `mul_tiles_bcast_rows` in rmsnorm_post_allgather.cpp:137) |
| combine (Regime B) | `add` | `cb_partials_gathered[k]` — column vector | `cb_partial_sumsq` — column vector | `BroadcastDim::None` — elementwise same-shape sum |

## Reduce Direction Verification

| Logical Dim | Tile ReduceDim | Output valid region | BroadcastDim back | ReduceInputBlockShape | Notes |
|-------------|----------------|---------------------|-------------------|-----------------------|-------|
| last dim (W) | `REDUCE_ROW` | column vector (col 0), `bh` tiles | `Col` | `ReduceInputBlockShape::of(bh, REDUCE_BLOCK)` per chunk, accumulated over W-chunks | SUM+REDUCE_ROW uses the matmul path → `cb_scaler` must be **col-0 fill**, value 1.0, produced by the pool-type-aware `*_reduce_scaler` overload. Single reduce dim only — no ambiguity. |

## Key Risks and Gotchas

1. **`cb_input_resident` is resident across passes (P1).** Pass 1 must read it **without popping** (read by `cb_wait_front` once + indexed access, or a no-pop input lifecycle); only pass 2 issues the final `cb_pop_front(bh*Wt_s)`. If the `square` helper's default `Streaming` lifecycle pops the input, pass 2 has nothing to normalize → wrong output / hang. Use a wait-upfront/no-pop input lifecycle in pass 1.
2. **Scaler must be col-0 fill, value 1.0, bf16.** SUM+REDUCE_ROW takes the matmul path; use the pool-type-aware overload `prepare_reduce_scaler<cb_scaler, SUM, REDUCE_ROW>` / `calculate_and_prepare_reduce_scaler<cb_scaler, SUM, REDUCE_ROW, …>` — NOT the legacy index-only overload. `cb_scaler` is bf16-packed.
3. **`inv_W` applied post-reduce, not in the scaler.** The local reduce is a plain SUM so Regime A and B share one kernel. `mean = Σx² * inv_W` happens in the finalize SFPU chain (`MulUnary(inv_W)`), where `inv_W = 1/W` (full W, identical on every core after the all-gather). Folding `1/W` into the scaler would break the unified path and double-apply it in Regime B.
4. **No Welford for RMSNorm.** The cross-core combine is a plain associative SUM. Do not pull in `welford_combine.h` or any δ² correction — dead weight, and the count-weighting math is wrong for a pure Σx².
5. **Rectangular groups are a hard mcast precondition.** The host must lay every W-group out as a rectangular grid band and validate it; an L-shaped group cannot be served by one multicast and will hang. Clean fallback to Regime A (fewer cores) if no rectangular partition exists.
6. **§9 silent-hang checklist (Regime B):** virtual (not physical) coords into the pipe; `consumed` counter host-init 0; `num_dests` = receivers (+1 only on loopback); semaphores created on the **union** of group cores; never multicast to only yourself (K=1 is Regime A and skips the leg).
7. **Sequential compute→compute intermediates sized to one block.** `cb_squared`, `cb_normalized` are produced and consumed by the same compute thread sequentially — size to `REDUCE_BLOCK` (one block), never to `Wt` and never "×2 for pipelining" (would deadlock or OOM).
8. **DEST capacity depends on `fp32_dest_acc_en`.** `REDUCE_BLOCK ≤ DEST_AUTO_LIMIT` (4 when fp32-acc True, 8 when False; `dest_helpers.hpp:89-103`). The reduce/normalize block loops and the Phase 1 block-height heuristic must size against the active limit.
9. **`BLOCK_HEIGHT` is a named seam, not the literal 1.** Phase 0 sets it to 1, but the kernel must thread it as a parameter so Phase 1 row-blocking lands as a heuristic change, not a rewrite.

## Phase 1 design hook (DO NOT implement in Phase 0)

**Row-blocking (`BLOCK_HEIGHT > 1`)** is a pure-performance refinement (does NOT change SUPPORTED → not a registry refinement; capture in `changelog.md`):
- Regime B: the all-gather carries partials for several rows in ONE larger mcast → amortizes fixed handshake/mcast overhead.
- Both: one compute init/reconfig amortized over a taller block.
- **Hard constraint:** row-blocking must NOT reduce active core count — `BLOCK_HEIGHT` may grow only after every core already has work. The block-selection heuristic then trades block height against core occupancy under the same L1 budget AND against DEST capacity (halved when `fp32_dest_acc_en`). All CB sizings already reference `bh`, so this is a host-heuristic change plus larger `cb_input_resident`/`cb_partial_sumsq`/`cb_recip_rms`/`cb_partials_gathered` allocations — no kernel rewrite.

## Structural impossibilities (feature_spec pipeline mode)

`eval/golden_tests/rms_norm/feature_spec.py` already exists and is authoritative (pipeline mode — not edited here). Its `INVALID` set already encodes the relevant structural impossibilities for this op: `{dtype: bfloat8_b, layout: ROW_MAJOR}`, the same on the gamma tensor, and the `no_gamma` canonicalization of redundant `gamma_dtype × gamma_layout` cells. No additional op-specific INVALID candidates are needed — the Phase 0 narrowness (bf16-only, TILE-only, tile-aligned-only, fp32_dest_acc_en True) lives in the op file's `SUPPORTED`/`EXCLUSIONS` (implementer's job), where out-of-scope cells correctly xfail rather than skip.
