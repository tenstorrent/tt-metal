# L1 Ledger: chunk_gated_delta_rule_fwd

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions:
`op_design.md` → Blocking Model. Every core allocates every CB (one CB config for the whole program;
the three stages reuse CB roles on the same allocation).

Block axes (every row accounts for all of them): `bh` (item's batch·head), `chunk`, `C` (`Ct` tiles),
`K` (`Kt`), `V` (item extent `Vi`, scan extent `Vs`), `gather window`.

`F` = Float32 tile (4096 B); `I` = input-dtype tile (4096 B fp32 / 2048 B bf16).
`Da = ACCUM_DEPTH = 2`, `Ds = SCAN_STREAM_DEPTH = 2`, `De = EGRESS_DEPTH = 2`, `Dg = GATHER_DEPTH = 2`.
Push quanta: `Qf = Ct·max(Kt, Ct, Vi)`, `Qo = max(Ct², Ct, Kt·Vs, Ct·Vi)`, `Qv = max(Ct, Kt)·Vi`.
Page format audit: `fp32_dest_acc_en = True` by default ⇒ every compute-produced CB is `Float32`
(no *under* finding). With a caller-supplied `fp32_dest_acc_en = False` (bfloat16 only) the `Float32`
pages are an audit-2 *over* case, accepted deliberately: the prompt mandates Float32 internal CBs
regardless, and it is a non-default configuration.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_const` | `4·Ct² + Ct` | same (constants: EYE, LT, SL, SU, ONES_ROW) | `{bh: streams, chunk: streams, C: spans → Ct², K: —, V: —, window: —}` | F | reader | compute | whole program | none — read by index in every stage; NSL and ONES `[C,C]` were removed from the inventory (`I − N` sign handled by `sub`; `Σg` via a one-tile-row `ONES_ROW`) |
| `cb_gather_stage` | `Dg` pages of `gather_stage_tokens·row_span_stride + 64` B | `Dg` windows (double buffer) | `{bh: streams, chunk: streams, C: streams → window, K/V: streams (one d-tile per window), window: spans}` | raw | reader | reader | P | reader-local; cannot share with any compute CB (single-thread scratch). Capacity > one window: double buffering of NoC reads vs re-pack (the measured dominant term) |
| `cb_scalar_stage` | 1 page of 512 B | 32 staged scalars | `{C: streams (32 tokens), others: —}` | raw | writer | writer | P | writer-local; separate from `cb_gather_stage` because that one belongs to the reader thread |
| `cb_q_in` | `Ct·Kt` | `Ct·Kt` | `{bh, chunk: streams; C: spans; K: spans; V: —}` | I | reader | compute | P | not with `cb_k_in` (both live through `key_prep_block`); not with E/S reader CBs (differs in format from the f32 ones; same format as `cb_vblock_in` but concurrent with it in P) |
| `cb_k_in` | `Ct·Kt` | `Ct·Kt` | `{bh, chunk: streams; C: spans; K: spans; V: —}` | I | reader | compute | P | `k` is live until `Pᵀ` in `key_products_block` (needed by `kβ@kᵀ`, `q̃@kᵀ`, `k⊙w`) |
| `cb_vblock_in` | `Qv = max(Ct,Kt)·Vi` | P: `Ct·Vi`; S: `Kt·Vs` (once per unit); E: `Kt·Vi` | `{bh, chunk: streams; C: spans (P); K: spans (S, E); V: spans → Vi / Vs}` | I | reader | compute | P, S, E | **shares** three disjoint roles (P `v`, S `initial_state`, E `h_i`), one quantum `Qv`; capacity − live set in P/S is the uniform-quantum tail |
| `cb_gate_in` | `2·Ct` | `2·Ct` (full-width `g`, `β`) | `{bh, chunk: streams; C: spans; K, V: —}` | I | reader | compute | P | live for the whole item (β used by `kβ` and `vβ`, g by three phases) |
| `cb_vec` | `4·Ct` | `4·Ct` (`decay, γ, w, Γ_full`) | `{C: spans; others: —}` | F | compute | compute | P | concurrent with every P block; `decay` also leaves through `cb_out_egress` (packed twice from DEST, no extra CB) |
| `cb_qs` | `Ct·Kt` | `Ct·Kt` (`q̃`) | `{C, K: spans; V: —}` | F | compute | compute | P | not with `cb_kmat_in` (different producer: compute vs reader) — R6 would make it the E source |
| `cb_kb` | `Da·Ct·Kt` | `Ct·Kt` (`kβ` → `U` in place) | `{C, K: spans}` | F | compute | compute | P | capacity 2× live: in-place `X ← f(X)` needs the new block reserved behind the old (ACCUM_DEPTH, not tunable) |
| `cb_kw` | `Ct·Kt` | `Ct·Kt` (`k⊙w`, transposed out) | `{C, K: spans}` | F | compute | compute | P | could alias `cb_qs` only after `Q` and `intra` are packed; they are packed before `Pᵀ` — **candidate alias**, not taken because a CB ring cannot hand the same pages to a second role without a pop/reserve cycle that reorders `key_products_block`; recorded here, 32 KB at the largest shape |
| `cb_L` | `Ct²` | `Ct²` | `{C: spans ×2}` | F | compute | compute | P | live from `decay_mask_block` to `intra` (end of item) |
| `cb_cc_a` | `Ct²` | `Ct²` (`diag(g)` → `N`) | `{C: spans ×2}` | F | compute | compute | P | `diag(g)` dies at `X`; `N` reuses the same CB (sequential roles, one quantum) |
| `cb_cc_b` | `Ct²` | `Ct²` (`X`) | `{C: spans ×2}` | F | compute | compute | P | `X` and `diag(g)` are live simultaneously (matmul operands) |
| `cb_T` | `Da·Ct²` | `Ct²` | `{C: spans ×2}` | F | compute | compute | P | in-place accumulator (`T ← T + T@Pw`) — ACCUM_DEPTH |
| `cb_pow` | `Da·Ct²` | `Ct²` | `{C: spans ×2}` | F | compute | compute | P | in-place squaring (`Pw ← Pw@Pw`) — ACCUM_DEPTH; concurrent with `cb_T` |
| `cb_vmat` | `Ct·Vi` | `Ct·Vi` (`vβ`) | `{C: spans; V: spans → Vi; K: —}` | F | compute | compute | P | `v_corr` is packed straight into egress (no CB); cannot share with `cb_vnew_in` (reader-produced) |
| `cb_intra_in` | `Ct²` | `Ct²` | `{C: spans ×2}` | F | reader | compute | E | disjoint from S's reader CBs, but a different quantum from all of them (`Ct²` vs `Kt·Ct`, `Ct·Vs`, 1) — sharing would force the larger quantum on the scan stream |
| `cb_vnew_in` | `Ct·Vi` | `Ct·Vi` | `{C: spans; V: spans → Vi}` | F | reader | compute | E | sharing with `cb_scan_vcorr` would force quantum `Ct·Vi` at depth `Ds` (`2·Ct·Vi` > `Ct·Vi + 2·Ct·Vs` whenever `Vs < Vi/2`) |
| `cb_kmat_in` | `Ds·Ct·Kt` | S: `2·Ct·Kt` (two chunks of `nkcd` in flight); E: `Ct·Kt` (`Q`) | `{chunk: streams (depth 2 in S); C, K: spans; V: —}` | F | reader | compute | S, E | **shares** S `nkcd` stream and E `Q` (same quantum, same producer/consumer, disjoint lifetime) |
| `cb_scan_pt` | `Ds·Kt·Ct` | `2·Kt·Ct` | `{chunk: streams; C, K: spans; V: —}` | F | reader | compute | S | concurrent with `cb_kmat_in` in S; its E-lifetime twin would be `intra` (quantum mismatch, see `cb_intra_in`) |
| `cb_scan_vcorr` | `Ds·Ct·Vs` | `2·Ct·Vs` | `{chunk: streams; C: spans; V: spans → Vs}` | F | reader | compute | S | see `cb_vnew_in` |
| `cb_scan_gamma` | `Ds` | 2 tiles (`Γ_full`) | `{chunk: streams; others: —}` | F | reader | compute | S | a 1-tile quantum; no other CB has quantum 1 |
| `cb_state` | `Da·Kt·Vs` | `Kt·Vs` | `{chunk: streams (resident across all NC steps); K: spans; V: spans → Vs; C: —}` | F | compute | compute | S | in place across every chunk step — ACCUM_DEPTH. Cannot share with `cb_vblock_in` (producer and, for bf16, format differ) |
| `cb_scan_vnew` | `Ct·Vs` | `Ct·Vs` | `{C: spans; V: spans → Vs}` | F | compute | compute | S | `v_new` must be resident as `in1` of `Pᵀ@v_new` while also leaving via egress (two packs, one DEST tile) |
| `cb_scratch_egress` | `De·Qf` | `De` blocks in flight | `{C: spans; K or C or Vi: spans (max); chunk: streams}` | F | compute | writer | P, S | one quantum `Qf` for `nkcd, Pᵀ, Γ_full, Q, intra, v_corr` (P) and `v_new` (S); capacity − live = the uniform-quantum tail + double buffering |
| `cb_out_egress` | `De·Qo` | `De` blocks in flight | `{C: spans; K/V: spans (max of Kt·Vs, Ct·Vi); chunk: streams}` | I | compute | writer | P, S, E | one quantum `Qo` for `Tinv→A`, `decay→g_cumsum` (P), `h_i`, `final_state` (S), `o`, `v_new` (E). Separate from `cb_scratch_egress` because the page format differs (I vs F) for bf16 |

Indices 19, 24, 29 are intentionally unused (merged into `cb_vblock_in` and `cb_kmat_in`).

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|---|---|---|---|
| `Ct` | `chunk_size/32` | 1..2 in TARGET; any positive integer admitted | `validate()`: `chunk_size % 32 == 0`; the footprint solve bounds it in practice |
| `Kt` | `K/32` | ≤ 8 (INPUTS ≤ 4) | `K % 32 == 0` validated; footprint solve raises if even `Vi = 1` does not fit |
| `Vt` | `V/32` | ≤ 8 | `V % 32 == 0` validated |
| `Vi` | item V extent | divisor of `Vt`, largest that fits | `_solve_item_block_val_tiles` |
| `Vs` | scan V extent | divisor of `Vt` | `_solve_scan_split`: `NV = Vt/Vs` largest divisor with `BH·NV ≤ G` |
| `NS` | handoff segments | ≤ 8 | `min(NC, READY_SEGMENTS_MAX = 4)`; 2 semaphores each, 16 per core |
| `gather_stage_tokens` | tokens per staging window | ≤ 32 | host constant 16 |
| `row_span_stride` | bytes per staged row | `round_up(272·esz + 64, 64)` (1152 fp32, 640 bf16) | NoC read alignment rule |
| `NC`, `NI`, `BH`, `T` | op dimensions | unbounded — **appear in no capacity expression** | scratch (DRAM) scales with them, L1 does not |

## Total per-core footprint (closed form)

```
footprint(Vi, Vs) =
    F · [ (4Ct² + Ct)                         # cb_const
        + 4Ct                                 # cb_vec
        + Ct·Kt + Da·Ct·Kt + Ct·Kt            # cb_qs, cb_kb, cb_kw
        + 3Ct² + 2·Da·Ct²                     # cb_L, cb_cc_a, cb_cc_b, cb_T, cb_pow
        + Ct·Vi + Ct² + Ct·Vi                 # cb_vmat, cb_intra_in, cb_vnew_in
        + Ds·Ct·Kt + Ds·Kt·Ct + Ds·Ct·Vs + Ds # cb_kmat_in, cb_scan_pt, cb_scan_vcorr, cb_scan_gamma
        + Da·Kt·Vs + Ct·Vs                    # cb_state, cb_scan_vnew
        + De·Ct·max(Kt, Ct, Vi) ]             # cb_scratch_egress
  + I · [ 2·Ct·Kt + max(Ct,Kt)·Vi + 2Ct       # cb_q_in, cb_k_in, cb_vblock_in, cb_gate_in
        + De·max(Ct², Ct, Kt·Vs, Ct·Vi) ]     # cb_out_egress
  + Dg·(gather_stage_tokens·row_span_stride + 64) + 512
```

Terms scaling with `Vi`: `cb_vmat`, `cb_vnew_in`, `cb_vblock_in`, and the two egress quanta — the solve
lever. Terms scaling with `Vs`: the scan stream and `cb_state` — small by construction (occupancy
drives `Vs` down). Everything else scales with `Ct` and `Kt` only.

Budget: `ttnn.get_max_worker_l1_unreserved_size() − L1_KERNEL_CONFIG_RESERVE (80 KB)` (≈ 1416 KB on
the part the backward measured). `Vi` = largest divisor of `Vt` with `footprint ≤ budget`; none ⇒
`ValueError`.

### Footprint at every INPUTS shape (`Vi = Vt`; `G = 110` and `G = 64` give the same `Vs`)

| Shape `(B,T,H,K,V)`, C | `NI` | `NV` | `Vs` | `NS` | fp32 KB | bf16 KB | `Vi` |
|---|---|---|---|---|---|---|---|
| (1,32,1,32,32), 32 | 1 | 1 | 1 | 1 | 209 | 179 | 1 |
| (1,64,1,64,64), 64 | 1 | 2 | 1 | 1 | 605 | 541 | 2 |
| (1,128,2,64,64), 32 | 8 | 2 | 1 | 4 | 293 | 249 | 2 |
| (2,64,4,64,64), 32 | 16 | 2 | 1 | 2 | 293 | 249 | 2 |
| (4,128,16,64,64), 32 | 256 | 1 | 2 | 4 | 337 | 285 | 2 |
| (1,128,2,64,128), 64 | 4 | 4 | 1 | 2 | 717 | 629 | 4 |
| (1,128,2,32,64), 32 | 8 | 2 | 1 | 4 | 237 | 201 | 2 |
| **(1,256,4,128,256), 64** | 16 | 8 | 1 | 4 | **1181** | 997 | 8 |
| (1,100,2,64,64), 64 | 4 | 2 | 1 | 2 | 605 | 541 | 2 |
| (1,72,2,64,64), 32 | 6 | 2 | 1 | 3 | 293 | 249 | 2 |
| (1,200,2,32,64), 32 | 14 | 2 | 1 | 4 | 237 | 201 | 2 |
| (1,160,2,64,128), 64 | 6 | 4 | 1 | 3 | 717 | 629 | 4 |
| (1,48,3,64,64), 64 | 3 | 2 | 1 | 1 | 605 | 541 | 2 |
| (1,1000,4,128,128), 64 | 64 | 4 | 1 | 4 | 925 | 805 | 4 |
| (1,512,8,128,128), 64 | 64 | 4 | 1 | 4 | 925 | 805 | 4 |
| (2,1024,4,128,128), 64 | 128 | 4 | 1 | 4 | 925 | 805 | 4 |
| (1,2048,2,128,128), 64 | 64 | 4 | 1 | 4 | 925 | 805 | 4 |
| (1,256,32,128,128), 64 | 128 | 2 | 2 | 4 | 981 | 861 | 4 |
| LOOSE (1,4096,16,128,128), 64 | 1024 | 4 | 1 | 4 | 925 | 805 | 4 |

`Vi = Vt` everywhere: the coarsest item block fits at every shape, so no split is taken. The solve is
a guard for `K, V` beyond INPUTS (e.g. `K = V = 256, C = 64` fp32 → 1453 KB at `Vi = 8`, solve takes
`Vi = 4`).

---

## Data-movement budget

Per-tensor DRAM crossings for R1 (the chosen split). "Face-row" = 272-element span read or two
16-element run writes per `(token, d_tile, head)`; "page" = one full-tile transfer.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `q`, `k` | 1 (face-row; 272 read per 32 used) | gathered once in P; `q̃, kβ, U, Q, intra, Pᵀ, nkcd` all derived while resident | none |
| `v` | 1 (face-row) | gathered once in P per V block; `v_corr` derived while resident | none |
| `g`, `beta` | 1 per head (page read, one column used) — `H` page reads per page | whole-page read + local extract | none |
| `initial_state` | 1 (pages) | read once per scan unit, its V block only | none |
| scratch `nkcd`, `Pᵀ`, `Γ_full` | write 1, read `NV` | **reuse-shared** across the scan's V split — every V unit re-reads | none (R4 would replace `NV−1` reads with NoC mcast) |
| scratch `v_corr` | write 1, read 1 | V-sliced to its scan unit | none |
| scratch `Q`, `intra` | write 1, read 1 | P core → same core's E (R6 would make it 0) | none |
| scratch `v_new` | write 1, read 1 | scan unit → E | none |
| `h` (output) | write 1, read 1 | E reads `h_i` back as its `o` operand | none |
| `o`, `v_new` (outputs) | write 1 (face-row) | E scatter, rows `t < T` | none |
| `A` (output) | write 1 (face-row) | P scatter | none |
| `g_cumsum` (output) | write 1 (scalar) | P scatter | none |
| `final_state` | write 1 (pages) | S | none |
| handoffs | — | — | `NV` semaphore incs per item (P→S); ≤ `#E cores` incs per (unit, segment) (S→E); no payload |

Totals at the LOOSE shape `(1,4096,16,128,128)` fp32, C=64 (`NI = 1024`, `NV = 4`):

| Tier | Bytes | Transactions |
|---|---|---|
| DRAM reads, `q,k,v` face-row gather | ≈ 855 MB moved (≈ 96 MB useful) | ≈ 786 k |
| DRAM writes, `o, v_new, A` face-row scatter + `g_cumsum` | ≈ 80 MB (all useful) | ≈ 1.38 M |
| DRAM scratch writes | ≈ 189 MB (45 f32 tiles per item) | ≈ 46 k pages |
| DRAM scratch reads (incl. `NV×` re-read of `nkcd/Pᵀ/Γ`) | ≈ 403 MB (of which 272 MB is the `NV×` term) | ≈ 98 k pages |
| DRAM `h` write + read | ≈ 134 MB | ≈ 33 k pages |
| cross-core | semaphores only | ≈ 4 k incs |

The governing term is the face-row transaction count (≈ 2.2 M), not bytes. The largest *byte* term
that is not intrinsic is the `NV×` re-read of the scan's shared operands.

> Cheapest-traffic split considered: **R3 page-harvest (all `H` heads per page read / write)** —
> gather transactions ≈ 786 k → ≈ 49 k, scatter ≈ 1.38 M → ≈ 90 k full-page writes, at the cost of one
> compact write + read of `q,k,v` (+192 MB) and of `o,v_new,A` (+80 MB) in DRAM, and a per-`(b,chunk)`
> fan-out/fan-in rendezvous. Implemented: **R1** (per-`(bh,chunk)` face-row gather/scatter; V-split
> scan). Deferred because R3 is layered on R1 — it replaces R1's gather and scatter with compact
> full-page reads and writes and leaves stages P, S and E unchanged — and R1 is correct on every
> shape. The structure keeps R3 reachable: every P/E per-item block already starts from a
> `[Ct, Dt]` tile block in L1, whatever produced it.
>
> The second cheaper option, **R4 (mcast of `nkcd/Pᵀ/Γ` to the scan's V units)**, saves 204 MB of DRAM
> reads at the LOOSE shape in exchange for an equal NoC payload. It is deferred as the successor
> R1's stepping stone (all V units reading the shared operand from DRAM) is built for: the scan
> reader is the only kernel code it changes.
