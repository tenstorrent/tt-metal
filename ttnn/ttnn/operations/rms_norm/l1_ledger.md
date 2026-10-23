# L1 Ledger: rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` §Blocking Model.
Units: capacities in **pages** (a page is one 32×32 tile of the stated format unless noted), byte totals in bytes.
Precision corner: `fp32_dest_acc_en = True` (Phase 0) — every DEST-produced intermediate is a `Float32` page.

Block axes (every row accounts for all three):
`row` — block extent `block_rows` (`b`); `w` — block extent `core_w_tiles` (`n`), resident row width
`row_stride_tiles` (`s = chunks_per_row · c`), streaming quantum `chunk_w_tiles` (`c`); `member` — the W-group member
axis of the R2 combine, extent `w_group_size` (`gw`).

## Inventory decisions taken before any sizing (Rule 1 / Rule 3)

| Phase boundary | Could it be storage-free? | Decision |
|---|---|---|
| x² → row sum | yes: x² never lands in L1 — `x·x` accumulates across the slice in DEST (PerRow) and only the per-row partial tiles are packed (Rule 3 #4, fold into the accumulator) | no `x²` buffer; `cb_sumsq` holds 2 tiles per row (`P_full`, `P_last`) because the masked last tile must stay uncollapsed (mechanism cap M8) |
| row sum → ×1/W, +eps, rsqrt | yes: fused as the reduce `post_reduce_op` in DEST | no buffer |
| inverse-RMS → normalize | no: the inverse-RMS is a per-row broadcast operand reused by every tile of the row | `cb_inv_rms`, `b` pages |
| x·γ → ·inverse-RMS | DEST-reuse form exists (lamp L8) but catalog `compute_fusion` measures the L1 round trip faster for an FPU consumer, and the L1 window is what lets x·γ fill the combine stall (S1) | `cb_x_gamma`, a window — not a block — in R1 (1 chunk) |
| normalize → output | pack into the destination (Rule 3 #1) | normalize packs straight into `cb_out_tiles` (the writer's CB) |
| RM: sticks → tiles, tiles → sticks | tilize/untilize need distinct in/out CBs (`tilize_helpers.hpp:99`, `untilize_helpers.hpp:89`) | `cb_x_rm_sticks` / `cb_out_rm_sticks` are chunk windows, never block-sized; RM output's `cb_out_tiles` is single-buffered (sequential helpers) |
| combine: gather / root result / receive | three different producer/consumer pairs (remote writers → root compute; root compute → root writer; writer multicast → every compute) | three CBs; `cb_inv_rms_root` cannot be `cb_inv_rms` (two consumers: root compute and the multicast) |

## Ledger

`T_in`, `T_out`, `T_g` = tile bytes of input / output / gamma dtype (2048 B bf16, 4096 B fp32); `F32` = 4096 B;
`L` = `xg_lookahead_chunks`; `d_x` = `x_depth_blocks`; `o` = 2 for `TILE_LAYOUT` output, 1 for `ROW_MAJOR_LAYOUT`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_rm_sticks` (RM input only) | `2·c` (page = `T_in` = 32 sticks × 32 elems) | `c` being tilized + `c` landing | `{row: streams → 1 tile-row, w: spans → c (chunk), member: n/a → 1}` | input dtype (raw sticks; no accumulation) | reader | compute | one (row, chunk) slab each | concurrent with `cb_x_tiles` (it feeds it) and with `cb_out_rm_sticks` (the reader lands block b+1 slabs while compute untilizes block b — verifier note); 2× live = double buffering (reader one slab ahead) |
| `cb_x_tiles` | `d_x · b · s` | `b · s` (the resident block) | `{row: spans → b, w: spans → s (resident slice, nominal), member: n/a → 1}` | input dtype — loaded data / tilize output, no accumulation crosses this boundary (audit 2: not a DEST-accumulated intermediate) | reader (TILE) / compute tilize (RM) | compute | a block; popped per chunk `b·c` | concurrent with every compute phase of its block; capacity = live set at Phase 0 (`d_x = 1`) |
| `cb_gamma_rm_stick` (RM gamma only) | 1 page of `round_up(n·32·e_g, dram_align)` bytes | 1 | `{row: n/a → 1, w: spans → n·32 elements, member: n/a → 1}` | gamma dtype, raw bytes | writer | writer | kernel start only (before first output) | could alias `cb_out_rm_sticks` / `cb_out_tiles` (disjoint lifetime: gamma is built before the first output chunk) — not done: ≤ `n·128` B (≤ 1/16 of one gamma tile row-set) is below the cost of an alias and its lockstep rule |
| `cb_gamma_tiles` (gamma only) | `s` | `slice_w_tiles_this_core` (≤ `n`) | `{row: n/a (reuse-shared over rows) → 1, w: spans → s, member: n/a → 1}` | gamma dtype (bf16 / fp32; loaded data) | writer | compute | whole kernel | concurrent with every block; `s − n` padding pages = chunk rounding (nominal quanta, M6) |
| `cb_reduce_scaler` | 2 | 2 (constant) | `{row: n/a → 1, w: n/a → 1 (the partial tile encodes the last tile's valid columns), member: n/a → 1}` | bf16 (reduce LLK scaler format) | reader | compute | whole kernel | constant; nothing to share with (2 pages) |
| `cb_sumsq` | `2 · b` | `2 · b` | `{row: spans → b, w: collapsed → 2 (P_full, P_last), member: n/a → 1}` | fp32 (DEST accumulation) | compute | compute | sum-of-squares → rowsum of one block | disjoint from `cb_x_gamma` in time (rowsum completes before x·γ starts) — not aliased: `2·b` fp32 pages are smaller than any `cb_x_gamma` window and aliasing would force lockstep accounting across two helpers for ≤ 8 KiB at `b = 1` |
| `cb_rowsum_partial` (R2 only) | `b` | `b` | `{row: spans → b, w: collapsed → 1, member: n/a → 1}` | fp32 | compute | writer | rowsum → partial sent | concurrent with `cb_x_gamma` (written while x·γ runs, S1); different consumer than `cb_inv_rms` |
| `cb_rowsum_gather` (R2 only) | `gw · b` | `gw · b` on roots; 0 on non-roots | `{row: spans → b, w: collapsed → 1, member: spans → gw}` | fp32 | writer (root) — remote members' NoC writes land in it | compute (root) | per block | must be allocated at the same address on every core (M1), so non-roots carry it unused — that symmetric allocation *is* the addressing mechanism (`tensix_all_reduce` README §L1 pressure) |
| `cb_inv_rms_root` (R2 only) | `b` | `b` on roots | `{row: spans → b, w: collapsed → 1, member: n/a → 1}` | fp32 | compute (root) | writer (root) | per block | cannot be `cb_inv_rms`: that CB's consumer is compute, and the multicast source needs the writer as consumer (one consumer per CB) |
| `cb_inv_rms` | `b` | `b` | `{row: spans → b, w: collapsed → 1 (broadcast over w), member: n/a → 1}` | fp32 | compute (R1) / writer (R2, multicast receive) | compute | rowsum/combine → last normalize chunk | concurrent with `cb_x_gamma` and `cb_out_tiles` |
| `cb_x_gamma` (gamma only) | `L · b · c` | `L · b · c` | `{row: spans → b, w: streams → L·c (L = chunks_per_row in R2 ⇒ spans s), member: n/a → 1}` | fp32 (DEST product) | compute | compute | per chunk (R1) / per block (R2) | concurrent with `cb_x_tiles` (produced from it) and `cb_out_tiles` (consumed into it) |
| `cb_out_tiles` | `o · b · c` | `b · c` | `{row: spans → b, w: streams → c, member: n/a → 1}` | output dtype (final result; packed from fp32 DEST to the output format by design) | compute | writer (TILE) / compute untilize (RM) | per chunk | TILE: 2× = double buffering against the writer; RM: 1× (sequential helpers on one thread) |
| `cb_out_rm_sticks` (RM output only) | `2 · b · c` | `b · c` | `{row: spans → b, w: streams → c, member: n/a → 1}` | output dtype, raw sticks | compute | writer | per chunk | 2× = double buffering against the writer |

## Symbol table (every non-block parameter in a capacity expression)

| Symbol | Meaning | Bound | Predicate establishing the bound |
|---|---|---|---|
| `b` = `block_rows` | block extent, row axis | `1 ≤ b ≤ rows_hi = ceil(tensor_rows_tiles / num_row_groups)`; `b = 1` if `chunks_per_row > 1` | `_select_blocking`: largest `b` with `footprint ≤ l1_budget_bytes` (M5) |
| `n` = `core_w_tiles` | block extent, W axis | `n = ceil(tensor_w_tiles / gw)` | `_select_blocking` raises `gw` until `footprint(1, n, …) ≤ l1_budget_bytes` (M3 if impossible) |
| `c` = `chunk_w_tiles` | streaming quantum, W axis | `min(n, 8) ≤ c ≤ n`, balanced: `c = ceil(n / k)` | `_select_blocking` chunk loop |
| `s` = `row_stride_tiles` | resident row width incl. chunk rounding | `n ≤ s = k·c < n + k` | derived |
| `k` = `chunks_per_row` | chunks per resident row | `1 ≤ k ≤ ceil(n / min(n, 8))` | derived |
| `gw` = `w_group_size` | W-group size | `1 ≤ gw ≤ num_cores_available` | `_select_blocking` (M2 recompute, M3 cap) |
| `L` = `xg_lookahead_chunks` | x·γ lookahead | `L = k` (R2 when it fits) or `1` | `_select_blocking` lookahead loop |
| `d_x` = `x_depth_blocks` | x buffer depth | `1` (Phase 0) | constant in `BlockingPlan`; lamp L1 |
| `o` | output depth by layout | 2 (TILE) / 1 (RM) | layout |
| `T_in`, `T_out`, `T_g`, `F32`, `e_g`, `dram_align` | tile / element / alignment bytes | `T ∈ {2048, 4096}`, `e_g ∈ {2, 4}`, `dram_align = ttnn.get_dram_alignment()` | dtype, HAL query |
| `l1_budget_bytes` | CB-allocatable worker L1 | free run from the L1 allocator base (`GetAllocatorBaseAddress`, 111 744 B on BH p150) up to the lowest allocated L1 buffer, i.e. `GetMemoryView(device, L1)`: `total_bytes_per_bank` (1 461 120 B on BH) clipped by the lowest allocated block | runtime query (`_l1_budget_bytes`). Implementation note: `ttnn.get_max_worker_l1_unreserved_size()` (1 531 776 B) over-states it by ~70 KB on BH and produced a CB-region overflow on the fp32 `4096×8192` R2 case |

## Footprint (per core, bytes) — the closed form `_select_blocking` evaluates

Single source (verifier fix): every capacity below is produced by ONE host function, `_cb_specs(...)` in
`rms_norm_program_descriptor.py`; `_footprint()` sums it for the L1 solve and `_build_cbs()` allocates from it, so a
knob change (`b`, `c`, `k`, `L`, `gw`, `d_x`, the stream / output depths) moves the budget and the real CBs together.

```
footprint(b, n, c, k, L, gw) =
      [RM in]   2·c·T_in                                  # cb_x_rm_sticks        — scales with c
    +           d_x·b·(k·c)·T_in                          # cb_x_tiles            — scales with b·s   (the block)
    + [γ, RM γ] round_up(n·32·e_g, dram_align)            # cb_gamma_rm_stick     — scales with n (tiny)
    + [γ]       (k·c)·T_g                                 # cb_gamma_tiles        — scales with s
    +           2·2048                                    # cb_reduce_scaler      — constant
    +           2·b·F32                                   # cb_sumsq              — scales with b
    + [gw ≥ 2]  (b + gw·b + b)·F32                        # partial, gather, root — scales with b and gw
    +           b·F32                                     # cb_inv_rms            — scales with b
    + [γ]       L·b·c·F32                                 # cb_x_gamma            — scales with b·L·c
    +           o·b·c·T_out                               # cb_out_tiles          — scales with b·c
    + [RM out]  2·b·c·T_out                               # cb_out_rm_sticks      — scales with b·c
```

Dominant terms: the resident block `b·s·T_in`, the resident gamma `s·T_g`, and the streaming windows
`b·c·(L·F32 + o·T_out [+ 2·T_out])`. The selector's response order (block rows, then chunk, then W-group) shrinks
them in that order.

Worked check (bf16 x/out/gamma, TILE, RM gamma, R1, `b = 1`): `n = 128, c = 128` →
`256 Ki + 8 Ki + 256 Ki + 4 Ki + 8 Ki + 4 Ki + 512 Ki + 512 Ki ≈ 1.53 MiB` > L1 ⇒ `k = 2, c = 64` → ≈ 1.03 MiB fits
(this is regime-pinned test `(1,1,8192,4096)`).

Implemented check (BH p150, 120 cores, budget 1 461 120 B): fp32 `(1,1,4096,8192)` + fp32 RM gamma selects `gw = 2, n = 128, k = 5, c = 26, L = 1, b = 1` (footprint 1 433 600 B); the `k = 4, c = 32` candidate (1 490 944 B) does not fit.

## Data-movement budget (chosen split)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (input) | 1 | the whole block (`b` tile-rows × the core's full W slice) stays resident in `cb_x_tiles` across sum-of-squares, x·γ and normalize | none |
| output | 1 | written once per chunk from `cb_out_tiles` / `cb_out_rm_sticks` | none |
| gamma | `num_row_groups` (each row group re-reads its W slices once; never per block) | resident `cb_gamma_tiles` for the whole kernel; RM gamma reads `W·e_g` bytes per row group; TILE gamma reads only row 0 of faces 0 and 1 (`2·round_up(16·e_g, 64)` B per tile) | none (R5 would multicast it) |
| per-row partial sums (R2) | 0 | — | per block per group: `(gw − 1) · b · 4096` B unicast into the root |
| inverse-RMS (R2) | 0 | — | per block per group: one multicast of `b · 4096` B to `gw` cores |

Totals per tier: DRAM = `|x| + |out| + num_row_groups·|gamma_read|` (gamma term ≤ 6.25 % of `|x|`, see R5);
NoC cross-core (R2 only) = `num_row_groups · num_blocks · b · 4096 · (2·gw − 1)` B; core-local reads: x read 2× from
L1 (sum-of-squares, then x·γ or normalize), x·γ once, inverse-RMS once per chunk.

> Cheapest-traffic split considered: row split with the slice resident (R1) — DRAM at the minimum (x 1×, out 1×), no
> cross-core traffic. Implemented: R1 wherever it fills the grid and fits; R2 (W split + root combine, also x 1×, out
> 1×, plus `≤ (2·gw−1)·b` fp32 tiles of NoC per block) where R1 would under-fill the grid or cannot keep a slice
> resident. The implemented split **is** the cheapest wherever R1 is selected; where R2 is selected, R1 is either
> infeasible (slice does not fit — the only alternative would be R3 at x 2×) or at equal DRAM bytes with idle cores.
