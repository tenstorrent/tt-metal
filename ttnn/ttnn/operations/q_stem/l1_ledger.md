# L1 Ledger: q_stem

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` Blocking
Model): **S1·rows**, **S1·cols**, **S1·K**, **X·seg** (exchange segment), **S3·heads**, **S3·rows**, **S3·N**.
Phase 0: bf16 everywhere, `fp32_dest_acc_en = True`. "tb" = bytes per page of that CB's format.

## Per-CB ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_q_slice` (compute rect ∪ ports) | `tensor_row_tiles * core_cols_max` | = capacity at the end of stage 1 (every block stays until the ports have pulled it; no ack path) | {S1·rows: spans → `tensor_row_tiles` (all blocks, `s1_block_m_tiles` each), S1·cols: spans → `core_cols_max`, S1·K: streams (not stored), X·seg: streams (ports pull), S3·heads/rows/N: none} | Float16_b — **16-bit under fp32 DEST by decision**: q is the exchanged payload, moved bit-exact as stored; bf16 halves link + NoC bytes (roofline counts q at 2 B/elt) and the prompt allows a bf16 q within default tolerances | compute | writer (stage-1 core) | stage 1 → end of call | Cannot share: concurrent with `cb_landing` (local quarter delivery writes landing while q is still being pulled) and with `cb_wkv` (prefetched during stage 1). Allocated on port cores only for address identity (ports never touch their own copy) — the bounded-ring alternative is regime R3 |
| `cb_x_round` (stage-1 rect ∪ injector) | `x_round_depth * s1_block_m_tiles * s1_kb` (**3**·1·48 = 144 Kimi; implementer: depth 2 → 3, measured stage 1 −6 µs) | 3 rounds (one computing, two landing; on the injector the DRAM-read RISC runs up to 3 rounds ahead of the multicast RISC) | {S1·rows: streams → `s1_block_m_tiles` per round, S1·K: spans → `s1_kb`, S1·cols: none, X·seg/S3·*: none} | Float16_b (latent dtype; unpacked operand, not a DEST product) | reader (receiver) / injector DRAM half (RISCV_0) | compute (receiver) / injector multicast half (RISCV_1) | stage 1 | Could alias `cb_wkv`/`cb_prefix_out` (disjoint from stage 3 on a core) — **not shared**: wkv_b1 is prefetched during stage 1 (stall-shadow decision), so the lifetimes overlap; `cb_prefix_out`/`cb_rope_out` (72 KiB) could alias it — deferred to the footprint lamp (no shape is near the budget) |
| `cb_qb_slice` | `Q_t * core_cols_max` | = capacity (resident operand, reused by every row block) | {S1·K: spans → `Q_t`, S1·cols: spans → `core_cols_max`, S1·rows: streams (reused), X·seg/S3·*: none} | Float16_b (Bfp8_b when `q_b_proj_dtype = bfloat8_b`) — operand as stored | reader | compute | stage 1 | Concurrent with `cb_x_round` and `cb_q_slice` (all three are live in every stage-1 block); after stage 1 it is dead — could alias `cb_prefix_out` (deferred, same reason as above) |
| `cb_q_partial` (only if `Q_t/s1_kb > 1`) | `s1_block_m_tiles * core_cols_max` | = capacity (one block's partials) | {S1·rows: spans → `s1_block_m_tiles`, S1·cols: spans → `core_cols_max`, S1·K: streams (accumulated), others: none} | Float32 (fp32 DEST: partials must keep fp32 between K-blocks) | compute | compute | stage 1 | Cannot be `cb_q_slice` (different format: Float32 vs the bf16 payload; matmul helper requires distinct interm with TileRowMajor and allows sharing only for SubblockMajor spill — kept distinct for the format) |
| `cb_landing` (**backed by the landing scratch shard**) | `hpc_max * B * quarter_row_tiles * head_w_tiles` (implementer: `hpc_max` from the load-weighted head placement — Kimi 4 heads on the cores outside the stage-1 rect; uniform shard; layout: remote heads row-major across heads, then local heads head-major) | = capacity (remote writes land without credit; every row of the call stays until consumed) | {S3·heads: spans → `hpc`, S3·rows: spans → `B*quarter_row_tiles`, X·seg: spans → `head_w_tiles` (one segment per row), S3·N: none, S1·*: none} | Float16_b (the exchanged bf16 q) | reader (arrival credits) | compute | whole call (written from the first remote block) | Cannot share: written by remote chips at any time after the fence, so it is live for the whole call; its address must be fixed across chips — it is a cached L1 sharded scratch tensor (persistent across calls by the prompt's rule: "scratch persistent per program") |
| `cb_wkv` (**two descriptors**: stage-1 rect / the rest) | R1: `(n_remote + min(WKV_DEPTH, n_local)) * Dn_t * N_t`, max over chips, per core set (Kimi: 4 heads = 512 KiB outside the rect, 1–2 heads in it); R2: `WKV_DEPTH * Dn_t * s3_block_n_tiles` | R1: every remote head's weights at once (remote rows are consumed row-major across heads) + local heads streamed; R2: 2 N-blocks | {S3·N: spans → `N_t` (R1) / `s3_block_n_tiles` (R2), S3·heads: streams → depth `min(2,hpc)`, S3·rows: streams (reused across rows, R1), S1·*: none, X·seg: none} | Float16_b (Bfp8_b when `wkv_b1_dtype = bfloat8_b`) | reader | compute | from stage 1 (prefetch) to end | Not shared: prefetch overlaps stage 1 by design (stall shadow) |
| `cb_prefix_out` | `out_depth * s3_block_m_tiles * s3_block_n_tiles` (2·1·16 = 32) | 2 row blocks | {S3·N: spans → `s3_block_n_tiles`, S3·rows: streams → `s3_block_m_tiles` per slot, S3·heads: streams, S1·*: none, X·seg: none} | Float16_b — output ABI (bf16 output); the K=Dn accumulation finished in DEST (single K-block), so no fp32 partial crosses this CB | compute | writer | stage 3 | Could alias stage-1-only CBs (`cb_qb_slice`, `cb_x_round`) — deferred; not needed for fit (max total 1032 KiB) |
| `cb_rope_out` | `out_depth * s3_block_m_tiles * Dr_t` | 2 row blocks | {S3·rows: streams → `s3_block_m_tiles`, S3·N: none (rope columns `Dr_t`), S3·heads: streams, S1·*: none, X·seg: none} | Float16_b — output ABI; NoPE copy is exact | compute | writer | stage 3 | Separate from `cb_prefix_out`: different push quantum (`Dr_t` vs `s3_block_n_tiles`) — one CB would break the ring-wrap invariant in R2 |
| `cb_port_stage` (ports only) | `2 * port_batch_segments * head_w_tiles` (2·4·6 = 48) | 2 batches | {X·seg: spans → `port_batch_segments * head_w_tiles`, all others: none} | Float16_b (raw payload) | reader (port) | writer (port sender) | port lifetime | Port cores hold no compute CBs except the unused `cb_q_slice` copy; nothing to share with |

## Symbol table (every non-block parameter in a capacity expression)

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `tensor_row_tiles = B*ceil(S/32)` | `tensor_row_tiles * core_cols_max * 2 KiB` is a term of `fits_l1`; ≤ 64 tiles × 2 in INPUTS (S=2048) | `fits_l1` |
| `core_cols_max = ceil(Cq_t / C1)` | ≤ 2 for every INPUTS shape | implementer: `C1 = W1*R1` = smallest rect holding `ceil(Cq_t / S1_COLS_PER_CORE)` cores (lamp L7; Kimi 9×8 = 72) |
| `Q_t` | ≤ 128 in INPUTS; resident `Q_t*core_cols_max` | `fits_l1` |
| `s1_kb` | ≤ `X_ROUND_TILES_CAP / s1_block_m_tiles` = 64 | mechanism cap (x round) |
| `hpc_max` | ≤ 4 in INPUTS (Kimi: 96 heads, 27 cores outside a 72-core stage-1 rect take most) | load-weighted greedy `head_placement` (S3_COST_FACTOR, FREE_CORE_START) |
| `quarter_row_tiles = (S/G)/32` | integer | `alignment == tile_aligned` |
| `head_w_tiles = Dn_t + Dr_t` | ≤ 8 in INPUTS | model dims |
| `N_t` in R1 | `min(2,hpc)*Dn_t*N_t*tb ≤ 256 KiB` | `wkv_resident` (selects R1) |
| `s3_block_n_tiles` in R2 | ≤ 16 | `S3_BLOCK_N_CAP` |
| `Dr_t` | ≤ 4 in INPUTS | model dims |
| `port_batch_segments` | 4 (knob) | — |
| `pkt_tiles` | `floor(max_payload / tb)` ≤ 7 | active fabric router config |

## Total per compute core

```
L1(core) = tb·tensor_row_tiles·core_cols_max          // cb_q_slice      — scales with S1·rows (whole), S1·cols
         + tb·x_round_depth·s1_block_m_tiles·s1_kb     // cb_x_round      — scales with s1_block_m_tiles, s1_kb, depth
         + tb_qb·Q_t·core_cols_max                     // cb_qb_slice     — scales with S1·K (whole), S1·cols
         + [Q_t/s1_kb > 1]·4096·s1_block_m_tiles·core_cols_max   // cb_q_partial
         + tb·hpc·B·quarter_row_tiles·head_w_tiles     // cb_landing      — scales with S3·heads, S3·rows (whole)
         + tb_wkv·(R1 ? min(2,hpc)·Dn_t·N_t : 2·Dn_t·s3_block_n_tiles)  // cb_wkv
         + tb·out_depth·s3_block_m_tiles·(s3_block_n_tiles + Dr_t)      // cb_prefix_out + cb_rope_out
fits_l1 := L1(core) ≤ cb_l1_budget(mesh)  (lowest allocated L1 address; checked on host before building — check_fits_l1)
```

Evaluated by the implemented predicate (`q_stem_program_descriptor.l1_terms` / `check_fits_l1`, verifier pass,
11×10 Blackhole grid, live budget `cb_l1_budget` = lowest allocated L1 address = **1427 KiB** with the landing
scratch allocated), statically allocated CBs per core in KiB (stage-1 core / stage-3-only core / port core;
`cb_landing` is the scratch, listed separately):

| Shape | Regime | C1 | hpc_max | stage-1 | stage-3-only | port | landing scratch |
|-------|--------|----|---------|---------|--------------|------|-----------------|
| Kimi S=640 | R1 | 72 | 2 | 760 | 328 | 176 | 120 |
| Kimi S=512 / 256 | R1 | 72 | 2 | 744 / 712 | 328 | 160 / 128 | 96 / 48 |
| DeepSeek-V3.2 S=640 | R1 | 99 | 3 | 1016 | 200 | 176 | 180 |
| GLM-5.3 S=640 / 512 / 256 | R1 | 64 | 2 | 984 / 968 / 936 | 456 | 208 / 192 / 160 | 160 / 128 / 64 |
| long chunk S=2048 | R1 | 24 | 1 | 936 | 200 | 352 | 192 |
| deep Q=4096 (K-blocked) | R1 | 12 | 1 | **1120** (max) | 200 | 112 | 12 |
| wide N=4096 | R2 | 12 | 1 | 504 | 328 | 112 | 12 |
| smallest (one tile) | R1 | 2 | 1 | 46 | 10 | 48 | 4 |

Every INPUTS shape fits with ≥ 307 KiB headroom (deep Q). A shape that does not raises RuntimeError naming the
role and the largest term (regime R3 is not built). An L1 latent (input_buffer = L1, refinement) lowers the
budget by its per-bank share (Kimi: 1.97 MB over 110 banks ≈ 18 KiB) — the predicate reads it live.

## Data-movement budget (per chip, Kimi K3.1 S=640)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| latent x (1.97 MB bf16) | 1 (0 in the model config: L1) | one injector reads each round once | multicast to `C1` = 99 cores (fan-out 99) |
| q_b_proj (14.16 MB bf16 / 7.52 MB bfp8) | 1 | column split: each column owned by one core, slice resident across row blocks | none |
| q (5.90 MB bf16) | **0** | resident in `cb_q_slice`, pulled by ports / delivered locally from L1 | port pulls 4.42 MB (NoC0); local quarter 1.47 MB (NoC1); fabric 4.42 MB sent |
| wkv_b1 (12.58 MB bf16) | 1 (R1); `B*quarter_row_tiles/s3_block_m_tiles` in R2 (=1 for the INPUTS shape) | resident per head across its rows | none |
| output (17.69 MB bf16) | 1 (write) | each row block written once | none |
| cos/sin (RoPE) | 1 | read once per core that owns heads (rows of the quarter) | — |

Totals: DRAM **46.4 MB** (Phase 0, bf16, latent in DRAM) → 90.6 µs at 512 GB/s = the Phase-0 roofline's DRAM term;
**37.8 MB** in the model config → 73.8 µs. Cross-core NoC: ~196 MB of multicast deliveries (1.97 MB × 99; one
injection of 1.97 MB) + 5.9 MB of q moves. Fabric: 4.42 MB out per chip (busiest directed link with direct routing
between 2 and 3 quarter-crossings × 1.47 MB / 2 links).

> Cheapest-traffic split considered: S1 column split with x multicast (R1) — DRAM minimum (each tensor once, q
> never), 0 B delta. Implemented: R1 (R2 for wide N). The implemented split **is** the cheapest; the K-split (R6,
> +11.8 MB NoC partials, −12% stage-1 time in the op model) and 2-band (R5, +7.5/14.2 MB DRAM) are deferred
> regime rows with their reasons in `op_design.md`.

## Data-movement search (`/perf-calculator`, `--phase Planner`, recorded in `dm_estimates.json`)

| Group | Sweep (fixed / varied) | Rank 1 | `pass_ns` | Tie with best | L1 B/core | Phase 0 choice |
|-------|------------------------|--------|-----------|---------------|-----------|----------------|
| q_b slice read (14.16 MB, `page_map {core_step 2, run 2, stride 144}`) | cores {48,72,96}, batch {4,8,16,32}, noc {0,1} | 96 cores, batch 4, NoC0 | 32 795 | ranks 1–4 (72 cores batch 8: 33 101) | 8 192 | NoC0, batch 8 on the 99-core rect (tie; rank-1 batch 4 kept as lamp L10) |
| wkv_b1 per-head read (12.58 MB, contiguous 64 pages per head) | cores {48,72,96}, batch {8,16,32,64}, noc {0,1} | 96 cores, batch 64, NoC0 | 29 582 | ranks 1–4 (batch 8: 29 584) | 131 072 | NoC0, batch 16 (tie; the CB holds the head anyway) |
| output write (17.69 MB TILE) | cores {48,72,96}, batch {6,9,18,36}, noc {0,1} | 96 cores, batch 9, NoC1 | 41 668 | ranks 1–4 (batch 6: 41 685) | 18 432 | NoC1, barrier every 9 tiles |
| x multicast (1.97 MB to the 11×7 rect, sender row 0) | bytes {2 K,16 K,32 K,96 K}, batch {1,2,4}, noc {0,1} | 16 KiB transactions, batch 4, NoC0 | 89 171 | rank 2 (NoC1) | 65 536 | 16 KiB linked transactions, depth 2 rounds; ≥ 32 KiB refused ("one packet of 1..16384 bytes per multicast") |

The x-multicast estimate (89 µs, bound by path reservation at 604 cycles per multicast, 77 receivers) is unlinked
and exceeds the op model's linked-resident multicast design (stage 1 total 57 µs). Phase 0 builds the linked
single-injector multicast (op-model rank) and keeps lamp L3 (2 rotating injectors) to settle the disagreement on
device. Schemes the schedule tool refuses (L1→L1 port pulls, fabric sends, `all_to_all`) are ranked on bytes alone
(table above) and by the measured fabric examples.

## Implementer notes (Phase 0)

- WKV_RESIDENT_BYTES raised 256 → 512 KiB: R1 now holds every remote head of a core at once (row-major consumption);
  the R2 predicate is `max units per core set * head bytes > 512 KiB` (wide-N shape still R2).
- Verified fits: every acceptance + supported golden shape builds (no CB / L1-buffer clash) with the single grow-only
  landing scratch per mesh.

## Verifier notes (Phase 0 audit)

- `fits_l1` was described here but not implemented; the verifier added `l1_terms` + `check_fits_l1` (RuntimeError
  naming the term) and re-evaluated the totals above (the planner's 724 / 1032 KiB figures were stale: the
  implementer's L7 / L8 changes moved C1 and hpc).
- `cb_q_partial` is Float32 unconditionally. Correct under Phase 0's fp32 DEST; under `fp32_dest_acc_en = False`
  (Refinement 1) it becomes a wide page behind a 16-bit DEST — the format must follow the flag (folded into
  Refinement 1).
- `cb_x_round` / `cb_qb_slice` (stage-1 only) and `cb_prefix_out` / `cb_rope_out` (stage 3) have disjoint
  lifetimes on a stage-1 core and are not aliased: justified (wkv prefetch overlaps stage 1; no shape is within
  300 KiB of the budget).
- `cb_q_slice` on the port cores is an unused copy kept for address identity (176–352 KiB per port core): justified
  (ports read the stage-1 cores' copy at the same address); the bounded-ring alternative is R3.
