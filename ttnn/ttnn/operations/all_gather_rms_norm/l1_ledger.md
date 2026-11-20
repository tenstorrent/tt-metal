# L1 Ledger: all_gather_rms_norm (regime R1 mirror_resident_gather)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`):
**row** (`core_row_tiles`), **member W** (`member_w_tiles`), **source slice** (G, mapped onto members),
**wave** (transport order of row-groups), **link / direction** (port identity). Group is per chip and never
appears in a buffer. `rows = core_row_tiles_max`, `mw = member_w_tiles_max`, `n_stat` = 1 (rms) / 2 (ln).

## Compute cores (planned inventory)

> Verifier note: this table is the PLANNED inventory. The authoritative as-built rows (sizes, formats, CB indices)
> are in "As built" at the end, which the code's `cb_plan()` matches; rows below that the as-built table changes
> (`cb_out` sizing, `cb_pivot` split, `cb_tail_mask` format, `cb_out_dense`) are superseded there.


| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x` | `rows · mw` | `rows · mw` (whole block: landing slot, resident across all passes, read by ports for relays) | {row: spans → rows, member W: spans → mw, source slice: spans the member's slice(s), wave: n/a (one block per core), link/dir: n/a} | input dtype (raw slices; never through DEST) | reader | compute | call | none: it is the landing target and the relay source for the whole call — no other buffer is live-disjoint from it |
| `cb_gamma` | `mw` | `mw` | {row: absent (gamma is one row), member W: spans → mw, slice: spans, wave: n/a, link/dir: n/a} | gamma dtype, compact row-0 page | reader | compute | call | `cb_beta`: both live through normalize (concurrent) |
| `cb_beta` (ln ∧ beta) | `mw` | `mw` | as `cb_gamma` | gamma dtype | reader | compute | call | `cb_gamma`: concurrent |
| `cb_ones` | 1 | 1 (constant) | {row: n/a, member W: n/a, slice: n/a, wave: n/a, link/dir: n/a} | Float16_b (scaler format of the reduce API) | reader | compute | call | constant read by every collapse; not aliasable with a stream that is live at the same time |
| `cb_tail_mask` (ln ∧ W_local % 32 ≠ 0) | 1 | 1 (constant) | all axes n/a | T_acc (multiplies DEST-width centered values) | reader | compute | call | constant |
| `cb_moment_a` | `rows` | `rows` | {row: spans → rows, member W: reduced away, slice: reduced, wave: n/a, link/dir: n/a} | T_acc | compute | compute | pass → collapse | `cb_moment_b`: alias candidate (pass A done before pass B packs) — not aliased because the ln pass-A collapse output (mean_m → pivot) is produced while `cb_moment_a` is popped by the same reduce; kept separate for clarity, 1 block of column-0 tiles (≤ 4 KB/row) |
| `cb_moment_b` (ln) | `2 · rows` | `2 · rows` | {row: spans, member W: reduced, others n/a} | T_acc | compute | compute | pass B → member finalize | see `cb_moment_a` |
| `cb_pivot` (ln) | `rows` | `rows` | {row: spans, others n/a} | Float16_b (the rounding to bf16 **is** the pivot; must be bf16-exact for the FPU subtraction) | compute | compute | pass B (K_m), then normalize (K_g) | holds K_m then K_g — disjoint lifetimes already folded into one CB |
| `cb_partial` | `n_stat · rows` | `n_stat · rows` | {row: spans, member W: reduced, slice: reduced, others n/a} | T_acc | compute | writer | moments → exchange | k = 1: the combine is skipped and `combine_finalize` reads it directly; not aliased with `cb_gathered` (both live during the exchange) |
| `cb_gathered` (k > 1) | `n_stat · k · rows` (owner combine k > 8: the base op's owner sizing `n_stat · ceil(rows/k)·k` row-gather tiles) | same | {row: spans, member: spans → k, member W: reduced, others n/a} | T_acc | writer | compute | exchange → combine | peers write into it without credit (lockstep, single block): no sharing |
| `cb_rowstat` | `n_rowstat · rows` (rms 1, ln 2: inv_std, shift) | same | {row: spans, others n/a} | T_acc | compute | compute | finalize → end of normalize | — concurrent with cb_x / gamma through normalize |
| `cb_out` aligned | `out_depth · out_unit_tiles` = `2 · min(mw, 16)` | `out_unit_tiles` + one unit in flight | {row: streams, member W: streams (unit window), others n/a} | output dtype | compute | writer | per unit | capacity > live set by the double buffer (overlap) |
| `cb_out` dense (`w_non_aligned`) | `rows · mw` | `rows · mw` (assembly reads source tiles i−1, i of every row) | {row: spans, member W: spans} | output dtype | compute | compute | normalize → assembly | could alias `cb_x` (cb_x is popped after normalize) — **not** aliased: a relay read of cb_x by a port may still be in flight when normalize ends (relays of late waves), and the ports read cb_x by raw address |
| `cb_shift_mats` (dense) | `n_shift_mats` ≤ 4 | same (constant) | all n/a | Float16_b (0/1, exact) | reader | compute | call | constant |
| `cb_foreign` (dense) | `rows · max_foreign_in` | same | {row: spans, others n/a} | output dtype | writer (peer member, lockstep slot) | compute | assembly | peers write without credit: no sharing |
| `cb_contrib` (dense) | `rows · max_foreign_out` | same | {row: spans} | output dtype | compute | writer | assembly | concurrent with `cb_foreign` |
| `cb_out_dense` (dense) | `2 · min(home_out_tiles, 16)` | one unit + one in flight | {row: streams, output column: streams} | output dtype | compute | writer | per unit | replaces the aligned `cb_out` stream role; double buffer |

Port cores additionally hold `cb_x` (allocated, unused — the address-identity requirement) and:

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_port_stage` | `2 · port_batch_chunks · chunk_pages` | `port_batch_chunks · chunk_pages` + one batch in flight | {row: streams, member W: streams, slice: streams, wave: streams, link/dir: one port = one (d, l)} | input dtype (raw bytes; never through DEST) | port_reader | port_sender | per batch | double buffer of the example (reader fills b+1 while the sender drains b) |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `rows` = `core_row_tiles_max` | `ceil(trt / num_row_groups)` | `select_split()`; R1 predicate `resident_fits(k)` |
| `mw` = `member_w_tiles_max` | `ceil(sw / k_s)` or `d · sw` | `select_split()` slice-aligned candidates |
| `k` | ≤ `K_CAP` (base op) and ≤ N | `select_split()` |
| `n_stat` | ≤ 2 | norm_type |
| `max_foreign_in`, `max_foreign_out` | ≤ `ceil(32 / W_local) + 1` | `plan_assembly()` (dense only) |
| `n_shift_mats` | ≤ 4 per member | distinct (shift, valid width) pairs of the member's slices |
| `chunk_pages` | `max_payload // T_in` (bf16 7, fp32 3, bf8b 13 at 14,336 B) | fabric config |
| `port_batch_chunks` | 4 | constant (example-measured) |
| `out_unit_tiles` | ≤ 16 (`OUT_UNIT_CAP`, base op) | constant cap |
| `cb_l1_budget(device)` | free L1 between the allocator base and the lowest L1 buffer | base op's function |

## Footprint (per compute core)

```
footprint(k, rows) =  rows·mw·T_in                                   # cb_x            (row × member W)
                    + (1 + has_beta)·mw·G_page                      # gamma / beta    (member W)
                    + 2048 + [ln∧tail]·T_acc + n_shift_mats·2048    # constants
                    + rows·T_acc·(1 + 2·ln) + rows·2048·ln          # moments, pivot  (row)
                    + n_stat·rows·T_acc·(1 + [k>1]·k)               # partial, gathered (row × k)
                    + n_rowstat·rows·T_acc                          # row stats       (row)
                    + (aligned ? 2·min(mw,16)·T_out : rows·mw·T_out + rows·(max_in+max_out)·T_out + 2·16·T_out)
```

Terms scaling with the knobs: `rows · mw` (cb_x, and cb_out in the dense regime) dominates; `rows · k` (gathered)
grows with the W split; everything else is ≤ a few tiles per tile-row. `resident_fits(k) := footprint(k, rows(k)) ≤
cb_l1_budget(device)`; for port cores the footprint is `rows·mw·T_in + 2·port_batch_chunks·chunk_pages·T_in`.

Focus case (640×384 bf16, k = 4, rows = 1, mw = 12, fp32 DEST, gamma RM bf16), as built (`cb_plan()`, verifier):
cb_x 24,576 + cb_gamma 24,576 (12 FULL-tile pages, only row 0 used) + cb_ones 2,048 + cb_moment_a 4,096 + cb_partial
4,096 + cb_gathered 16,384 + cb_rowstat 4,096 + cb_out 2·4·2,048 = 16,384 + cb_affine_stage 1,792 = **98,048 B** per
compute core. Port: cb_x 24,576 + cb_port_stage 2·4·7·2,048 = 114,688 → 139,264 B.
Indexer LN case (640×32 bf16, k = 1, pitch 4, gamma + beta RM bf16, LN FPU path), per compute core after the
verifier fixes below: 89,216 B (was 105,600 B: cb_out capped at the 4-tile block, cb_moment_a dropped,
cb_moment_b = rows).

## Data-movement budget (focus case: G = 4 ring, 2 links, S = 640·384·2 B = 491,520 B per slice)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (own slice) | 1 (read S by the own members) | resident in cb_x for every pass; the ports read it from L1, not DRAM | ports read it from L1 twice (fwd and bwd port): 2·S local NoC |
| x (remote slices) | 0 | land in compute L1 from the fabric; the far slice is relayed out of the middle chip's L1 | relay reads: 0.5·S per relay port (far half), notifications |
| fabric | busiest link direction 1.5·S over 2 links (= roofline) | balanced ring schedule (example) | — |
| output | 1 (write G·S) | each member writes its block once | — |
| gamma (RM sticks) | per member: its `mw` sticks (64 B each) | reuse-shared over rows, no broadcast (deferred) | — |
| partials | 0 | intra-chip only | k²·rows·(faces 0/2: 2 KB) per row-group = 20·16·2 KB = 640 KB per chip |

Totals per chip: DRAM S + G·S = 5·S ≈ 2.46 MB (the roofline's DRAM term); fabric 1.5·S per busiest link direction; local
NoC ≈ 3·S + 0.64 MB.

> Cheapest-traffic split considered: R1 itself (raw slices into compute L1, mirror layout, relay) — it is at the minimum
> on the DRAM and fabric tiers. Implemented: R1. The alternatives (normalize-then-gather, gather-to-DRAM) move the same
> or more bytes and add a fabric round or 2·G·S of DRAM; they are `rejected` rows in the regime table.

## As built (Phase 0 implementation) — deltas from the planned inventory

Single source of truth: `cb_plan()` in `all_gather_rms_norm_program_descriptor.py` (both the L1-fit solve in
`select_split()` and the CB declarations read it). CB indices: `kernels/all_gather_rms_norm_common.hpp`.

| CB (index) | Capacity (pages) | Page format | Producer | Consumer | Delta vs plan / Shares with / why not |
|---|---|---|---|---|---|
| `cb_x` (0) | `rows · pitch` | input dtype | reader (DRAM read) / fabric landing | compute (+ port readers by raw address) | as planned; FIRST CB over compute_used ∪ ports_union on every chip. Never transformed in place: the ports read it for relays |
| `cb_port_stage` (1) | `2 · PORT_BATCH_CHUNKS · chunk_pages` | input dtype | port_reader | port_sender | as planned |
| `cb_gamma` / `cb_beta` (2/3) | `pitch` | gamma dtype, FULL tile page (only row 0 of faces 0/1 written) | writer | compute | delta: full-tile page instead of the base op's 192 B compact stride (≤ 2·pitch·T_g bytes; simpler addressing for the element-shift loader of non-aligned slices). Refinement candidate if L1-bound |
| `cb_ones` (4) | 1 | bf16 | reader | compute | as planned |
| `cb_mask` (5, ln ∧ W_local % 32 ≠ 0) | 1 | bf16 row-0 0/1 mask (`prepare_reduce_mask`) | reader | compute | delta: bf16 row-0 mask (unary_bcast ROW) instead of a T_acc full mask |
| `cb_moment_a` (6) | `rows` | T_acc | compute | compute | rms Σx² tiles; ln (SFPU path) Σx tiles |
| `cb_moment_b` (7, ln) | `2 · rows` | T_acc | compute | compute | SFPU path (D1, D2 interleaved); FPU path: D2 tiles (rows) |
| `cb_km` (8, ln) | `rows` | bf16 | compute | compute | the planned `cb_pivot` split in two (K_m here, K_g in `cb_kg`): both live in the combine window of a multi-row block |
| `cb_partial` (9) | `n_stat · rows` | T_acc | compute | writer (k > 1) / compute (k = 1) | as planned |
| `cb_gathered` (10, k > 1) | `n_stat · k · rows` | T_acc | writer (peers) | compute | as planned (symmetric only: k ≤ 8, owner combine deferred) |
| `cb_rowstat` (11) | `rows` | T_acc | compute | compute | rms inv_rms; ln inv_std (shift moved to `cb_shift`) |
| `cb_out` (12) | `OUT_DEPTH · out_unit`, `out_unit` = largest of {min(pitch, 4), /2, …, 1} that fits | output dtype | compute | writer | delta: unit cap 4 (stores overlap the normalize of the row) and shrinks under L1 pressure |
| `cb_mstat` (13, ln) | `2 · rows` | T_acc (UnpackToDestFp32) | compute | compute | column-0 D1 / D2 per row |
| `cb_shift` (14, ln) | `rows` | T_acc | compute | compute | ln shift = (K_g − μ)·inv_std |
| `cb_kg` (15, ln) | `rows` | bf16 | compute | compute | ln K_g = bf16(μ) (the pack is the rounding) |
| `cb_mu` (16, ln) | 1 | T_acc (UnpackToDestFp32) | compute | compute | μ between the two combine windows of a row |
| `cb_affine_stage` (17, gamma) | `Σ(n + 1) + 1` slots × 128 B | bytes | writer | writer | staging of dense gamma/beta sticks (non-aligned element shift only) |
| `cb_norm` (18, dense) | `rows · pitch` | output dtype | compute | compute | the planned dense-mode `cb_out` block |
| `cb_shift_mats` (19, dense) | `max n_mats` | output dtype (0/1) | reader | compute | generic (e0, e1, delta) shift matrices incl. identity; format = output dtype (matmul operands symmetric) |
| `cb_foreign` (20, dense) | `rows · max_fin` | output dtype | writer (peers) | compute | as planned |
| `cb_contrib` (21, dense) | `rows · max_fout` | output dtype | compute | writer | as planned |
| `cb_cent` (22, ln FPU path) | `rows · pitch` | T_acc | compute | compute | NEW: centered block c' = x − K_m (aligned bf16/bf8b LN). Cannot alias `cb_x` (relay source) — the in-place transform is ruled out by the ports' raw-address relay reads |

`cb_out_dense` of the plan is the plain `cb_out` stream in dense mode (the assembly packs straight into it).
Data-movement budget: unchanged from the plan (DRAM: x read once by the own members, output written once; fabric:
raw slices, balanced ring schedule; cross-core: port L1 reads of outgoing runs, k-member partial exchange, dense
foreign contributions).

## Verifier deltas (Phase 0 verification)

| CB | Change | Why |
|---|---|---|
| `cb_moment_a` (6) | not allocated on the LN FPU path (`AGRMS_LN_FPU`) | dead on that path: the pivot reduce goes `cb_x` → `cb_km` directly (rms and the LN SFPU path keep it) |
| `cb_moment_b` (7) | `rows` pages on the LN FPU path (was `2·rows`) | the FPU path pushes only D2 tiles (D1 goes `cb_cent` → `cb_mstat`); the SFPU path keeps `(D1, D2)` pairs |
| `cb_out` (12, aligned) | `min(OUT_DEPTH·out_unit, rows·pitch)` | capacity never exceeds the member block (indexer: 4 pages instead of 8); compute reserves per tile, the writer ring clamps at the CB end. The writer's ring size CT arg now reads `cb_plan()[CB_OUT]` (it restated `OUT_DEPTH·unit`: DRY) |

Open (folded into op_requirements.md Refinement 1, the L1-fit refinement): `cb_gamma` / `cb_beta` FULL-tile pages
for row-0 data (24 KB on the focus case vs ~2.3 KB with the base op's compact 192 B stride); `cb_moment_a` /
`cb_moment_b` on the LN SFPU path are live-disjoint (pass A is popped before pass B packs) and could alias — the
"kept separate for clarity" justification above is not a mechanism reason.

## Refinement 1 deltas (owner combine, compact gamma, port batch, aliasing)

| CB (index) | Capacity | Format | Producer | Consumer | Delta / Shares with / why not |
|---|---|---|---|---|---|
| `cb_gamma` / `cb_beta` (2/3) | `pitch + g_tail` pages of `gamma_page_layout()` stride (192 B bf16 / fp32) | gamma dtype | writer | compute | compact row-0 pages (was a full tile per page: 24 KB → 2.5 KB on the focus case) |
| `cb_port_stage` (1) | `2 · port_batch · chunk_pages`, `port_batch` ∈ {4, 2, 1} largest that fits | input dtype | port_reader | port_sender | the batch halves under L1 pressure (cap = the measured 4) |
| `cb_gathered` (10, owner k > 8) | `ceil(rows/k) · n_stat · ceil(k/32)` | T_acc | writer (peers, row form) | compute (owner) | row-form gather: 32 members per page instead of one page per member |
| `cb_own_ret` (23, owner) | `ceil(rows/k) · n_stat` | T_acc | compute (owner) | writer | rms inv_rms; ln (inv_std, shift) per owned row |
| `cb_own_kg` (24, owner ∧ ln) | `ceil(rows/k)` | bf16 | compute (owner) | writer | K_g per owned row |
| `cb_dev` (25, owner ∧ ln) | `2 · ceil(k/32)` | T_acc | compute | compute | **aliased** with the moments scratch (below) |
| `cb_wcol` (26, owner ∧ ln) | `ceil(k/32)` | T_acc | reader | compute | n_m weights, written at kernel start → cannot alias |
| `cb_km` (8, owner ∧ ln) | `max(rows, 2)` | bf16 | compute | compute | also carries the owner's pivot K and K_g after the member pivots are popped |
| `cb_mu` (16, owner ∧ ln) | 2 | T_acc | compute | compute | (mu, inv_std) between the owner's windows; **aliased** with `cb_mstat` |
| `cb_rowstat` / `cb_shift` / `cb_kg` (owner) | `rows` | — | **writer** (owners' returns) | compute | the writer is their only producer on the owner path (lockstep slot r written by the row's owner) |

Alias regions (`cb_alias_groups`, one CBDescriptor with several buffer indices; all compute → compute, never NoC-written):
`{cb_moment_a, cb_moment_b, cb_dev}` (LN SFPU path) or `{cb_cent, cb_dev}` (LN FPU path) — the moments scratch is popped by
the end of member_moments, `cb_dev` lives only in the owner combine; `{cb_mstat, cb_mu}` — `cb_mstat` is popped by the member
finalize. Region size is rounded to whole `cb_dev` / `cb_mu` blocks (they cycle per owned row). `cb_gathered`,
`cb_rowstat`, `cb_shift`, `cb_kg`, `cb_foreign` are written by peers at any time and are never aliased. The LN FPU path's
`cb_cent` is dropped (SFPU moments path) when no split fits with it and DEST is fp32.

Data-movement budget: unchanged on DRAM and fabric. Cross-core: the owner path replaces the k²·rows faces-0/2 partial
exchange with k·rows·n_stat row-form writes (2 × 16 datums) plus an n_ret·rows·k faces-0/2 return.

## Refinement 2 deltas (layer_norm, 16-bit DEST: `Sizes.m16`)

| CB | Change on the m16 path | Why |
|---|---|---|
| `cb_moment_a`, `cb_moment_b`, `cb_mstat`, `cb_cent`, `cb_km`, `cb_dev` | **not allocated** | the member moments accumulate in fp32 SFPU registers (LREG4-6) across the member's tiles and the Chan combine runs in LREGs too: no DEST-accumulated scratch, no pivot tile, no owner `cb_dev` round |
| `cb_mu` (16) | `3` pages (T_acc), not aliased | the row-form statistics (mean_m / M2_m: 2 pages; K_g / inv_std / shift: 3 pages) are packed here and transposed back to column form on unpack — the generic Welford reduction's round trip. Shares with / why not: its old alias partner `cb_mstat` is gone on this path; nothing else is live with a matching lifetime |
| `cb_mask`, `cb_wcol` | unchanged (reader still pushes them) | unused by the m16 compute; one tile / `ceil(k/32)` tiles, left in place so the reader is untouched |

Net: the m16 path frees `rows · (1 + 2 + 2) · T_acc + rows · pitch · T_acc (cb_cent) + rows · T_bf16` minus 3 · T_acc
(cb_mu) per compute core. DRAM / fabric crossings: unchanged.
