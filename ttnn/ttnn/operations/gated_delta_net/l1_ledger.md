# L1 ledger: gated_delta_net (Phase 0, regime R-STEP)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (op_design.md,
Blocking Model): `pair` (b, h; one pair per block), `t` (token rows; block = 1 tile row, step = 1
token), `K` (Kt, whole), `V` (Vt, whole), `c` (channel group: Kt or Vt columns), `j` (4 taps).
Page = one fp32 tile (4096 B) unless noted. `fp32_dest_acc_en = True` in every cell, so every
compute page is Float32 (audit 2: no 16-bit page carries a DEST result). Token buffers (Refinement 1:
cb_conv_w, cb_window, cb_gate_in, cb_gate_w, cb_z, cb_norm_w, cb_out_rows, cb_scratch; `TOKEN_CBS`
in the descriptor) take the token-tensor dtype: 2048 B bf16 pages, untagged (bf16 reaches DEST
losslessly through srcA); fp32 pages keep their U2D tag. cb_out_rows in bf16 is the one 16-bit page
fed by DEST: it is the output ABI (output dtype = input dtype).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| cb_state_in (+ alias cb_state_in_mm) | `Kt*Vt*STATE_IN_DEPTH` | Kt*Vt (S0 of one pair) | {pair: streams (1 pair per fill), t: streams (consumed at token 0), K: spans Kt, V: spans Vt, c: streams (none), j: streams (none)} | Float32 (U2D view) / Float32 (srcB view) | reader | compute | pair, until token 0 | alias pair shares one allocation (two unpack modes of the same bytes); not shared with cb_state: the reader prefetches the next pair's S0 while compute still updates the current state (pipelining decision) |
| cb_state (+ alias cb_state_mm) | `Kt*Vt` | Kt*Vt | {pair: streams, t: streams (in place per token), K: spans Kt, V: spans Vt, c: streams, j: streams} | Float32 / Float32 | compute | compute | pair, tokens 1..T-2 | alias pair shares one allocation; not with cb_state_out: cb_state_out must outlive the pair until the writer drains it while the next pair already writes cb_state |
| cb_state_out | `Kt*Vt*STATE_OUT_DEPTH` | Kt*Vt | {pair: streams, t: streams (filled by the last token), K: spans Kt, V: spans Vt, c: streams, j: streams} | Float32 | compute | writer | last token to drained | not with cb_state_in: for T = 1 both are live in the same token step (read S0, write S1) and the next pair's S0 prefetch overlaps the drain |
| cb_conv_w | `(2*Kt + Vt)` (FPU conv, bf16 tokens) / `4*(2*Kt + Vt)` (SFPU conv, fp32 tokens) | raw conv_weight tiles of the pair columns (tap j at row j, broadcast by row index on the FPU) / one row-0 tap tile per tap | {pair: streams (refilled per pair), t: streams (constant along t; resident to avoid a per-tile-row DRAM re-read), K: spans 2Kt columns, V: spans Vt columns, c: spans all groups, j: spans 4 (rows of one tile on the FPU path)} | token dtype | reader | compute | pair | nothing with disjoint lifetime. Refinement 2: the FPU conv reads raw tiles (the perf lamp), 1/4 of the SFPU path's expansion and one whole-tile read per column instead of 8 staged row reads |
| cb_window | `4*gcd(Kt, Vt)*WINDOW_DEPTH` | 4*gcd(Kt,Vt) per window group; capacity x2 for double buffering | {pair: streams, t: streams (1 tile row), K: streams (q, k groups pass through), V: streams (v group), c: spans one window group, j: spans 4} | token dtype | reader | compute | per window group | double buffered (depth 2: reader assembles group g+1 during conv of g); uniform group size so no reservation wraps the buffer end; rows outside the token rows are zeroed once by the writer at kernel start and re-zeroed by the reader only when a slot's dirty rows are not covered by the next group's rows; not shared: concurrent with every per-tile-row buffer |
| cb_y_qk | `2*Kt` | 2Kt | {pair: streams, t: spans 1 tile row (32 rows), K: spans Kt (q and k groups), V: streams (n/a), c: spans q,k groups, j: streams (collapsed)} | Float32 (U2D) | compute | compute | conv -> l2norm of one tile row | not with cb_kq_rows: different unpack mode (the L2 sums need U2D fp32, the matmul in0 needs srcA/srcB); not with cb_v: cb_v outlives it (kept through the token loop) |
| cb_v | `Vt` | Vt | {pair: streams, t: spans 1 tile row, K: streams, V: spans Vt, c: v group, j: streams} | Float32 (U2D) | compute | compute | conv -> end of recurrence_block | concurrent with the token loop: no sharing |
| cb_sq | `Kt` | Kt (q, k, then qn*kn squares/products) | {pair: streams, t: spans 1 tile row, K: spans Kt, V: streams, c: streams, j: streams} | Float32 (U2D) | compute | compute | inside l2norm | shared by the q, k and d reductions (disjoint, same Kt block); not with the RMS squares: pushed in Kt blocks and read by index, sharing with Vt-block pushes left a block wrapped past the CB end (measured: wrong RMS factor) |
| cb_sq_o | `Vt` | Vt | {pair: streams, t: spans 1 tile row, K: streams, V: spans Vt, c: streams, j: streams} | Float32 (U2D) | compute | compute | inside rmsnorm | split from cb_sq (wrap, above) |
| cb_rinv | 2 | rsqrt q, k (column-0 tiles) | {pair: streams, t: spans 1 tile row; others streams (reduced)} | Float32 (U2D) | compute | compute | inside l2norm | not with cb_rinv_o, same wrap reason |
| cb_rinv_o | 1 | 1 | {pair: streams, t: spans 1 tile row; others streams (reduced)} | Float32 (U2D) | compute | compute | inside rmsnorm | split from cb_rinv (wrap) |
| cb_conv_acc | `CONV_ACC_DEPTH` = 1 (SFPU conv only; not allocated on the FPU conv path) | 1 | {pair: streams, t: spans 1 tile row, c: one column tile; others streams} | Float32 (U2D) | compute | compute | per column tile | SFPU path: partial tap sum (taps 0,1 packed, taps 2,3 added by packer L1 accumulation), two fp32 half-sync DEST windows per column. FPU path (Refinement 2): the four taps accumulate in one DEST tile by ELWMUL, so no partial sum leaves DEST |
| cb_kq_rows | `2*Kt` | 2Kt | {pair: streams, t: spans 1 tile row, K: spans Kt, V: streams, c: q,k groups, j: streams} | Float32 (std) | compute | compute | tile row | concurrent with the token loop (read matmul in0 every token) |
| cb_knT | `Kt` | Kt | {pair: streams, t: spans 1 tile row (tokens on columns), K: spans Kt, V: streams, c: k group, j: streams} | Float32 (std) | compute | compute | tile row | concurrent with the token loop (increment in0 every token) |
| cb_d | 1 | 1 (column-0 tile) | {pair: streams, t: spans 1 tile row, K: streams (reduced), V: streams, c: streams, j: streams} | Float32 (U2D) | compute | compute | tile row | concurrent with the token loop; 1 page constant in every axis but t |
| cb_gate_in | `2*ROW_DEPTH` | 2 (b, a tiles) | {pair: streams, t: spans 1 tile row, K: streams, V: streams, c: streams (gate column h//32), j: streams} | Float32 (U2D; input dtype) | reader | compute | tile row | double buffered across tile rows |
| cb_gate_w | 2 | 2 (A_log, dt_bias rows) | {pair: streams (refilled per pair), t: streams (constant), K, V, c, j: streams} | Float32 (U2D) | reader | compute | pair | constant per pair; 2 pages genuinely constant |
| cb_gates | 2 | 2 (alpha, beta) | {pair: streams, t: spans 1 tile row, K, V, c, j: streams} | Float32 (std; dataflow reads elements) | compute | writer | tile row | consumed by the writer across the whole token loop: concurrent |
| cb_tok_coef | `3*TOK_COEF_DEPTH` | 3 (one token) | {pair: streams, t: streams (one token per fill), K, V, c, j: streams} | Float32 (U2D) | writer | compute | token | depth 2 so the writer runs one token ahead; not shared: concurrent with every token-loop buffer |
| cb_reads | `2*Vt` | 2Vt | {pair: streams, t: streams (one token), K: streams (contracted), V: spans Vt, c, j: streams} | Float32 (U2D) | compute | compute | token | not with cb_u: cb_reads is consumed while cb_u is produced in the same op (concurrent); different unpack mode |
| cb_u (+ alias cb_u_f32) | `Vt` | Vt | {pair: streams, t: streams (one token), K: streams, V: spans Vt, c, j: streams} | Float32 (std) | compute | compute | token (vector op -> update) | alias cb_u_f32 is a second format descriptor over the same allocation: u feeds the increment matmul (srcB view) and the fp32 o term (U2D view); lockstep push/pop |
| cb_o_acc | `Vt` | Vt | {pair: streams, t: spans 1 tile row (one row per token, packer L1 accumulation), K: streams, V: spans Vt, c, j: streams} | Float32 (U2D) | compute | compute | tile row | live through the token loop: concurrent with cb_u, cb_reads |
| cb_z | `Vt*ROW_DEPTH` | Vt | {pair: streams, t: spans 1 tile row, K: streams, V: spans Vt, c, j: streams} | Float32 (U2D; input dtype) | reader | compute | tile row | double buffered across tile rows |
| cb_norm_w | `Vt` | Vt | {pair: streams (refilled per pair), t: streams (constant), K: streams, V: spans Vt, c, j: streams} | Float32 (U2D) | reader | compute | pair | constant per pair |
| cb_out_rows | `Vt*ROW_DEPTH` | Vt | {pair: streams, t: spans 1 tile row, K: streams, V: spans Vt, c, j: streams} | Float32 (output dtype) | compute | writer | tile row | pack-into-destination (no intermediate between the norm chain and the writer); double buffered |
| cb_reduce_scaler | 1 (2048 B page) | 1 | {pair, t, K, V, c, j: streams (constant)} | Float16_b (bf16-packed SUM scaler 1.0) | reader | compute | kernel | constant; literal 1 page |
| cb_scratch | 1 | 3 rows | reader-private | token dtype | reader | reader | conv_state_out copy | face-row staging for the bit-exact conv_state_out copy; never seen by compute |
| cb_window_zeroed | 1 page of 32 B | 1 token | {all axes: streams} | n/a (handshake) | writer | reader | kernel start | writer-to-reader token: cb_window has been zeroed (the writer is idle until the first gates, so the zero fill leaves the reader's critical path) |
| cb_row_stage | 1 page of `(ROW_STAGE_SLOTS + 1) * dram_alignment` B (2112 B on Blackhole); allocated only when a face row is narrower than the DRAM read alignment (bf16) | up to ROW_STAGE_SLOTS aligned chunks | reader-private | raw bytes (token dtype descriptor) | reader | reader | between a row read and its RowReader::finish() | a bf16 face row is 32 B, a Blackhole DRAM read must be 64 B aligned: each half-row is read as its aligned 64 B chunk here, then copied L1-to-L1 into the destination tile. Not shared with cb_scratch: the conv_state_out copy reads rows into cb_scratch through this staging |

Inventory decisions (Rule 3): conv packs straight into cb_y_qk / cb_v (no tap-sum
intermediate); L2 scale packs straight into cb_kq_rows; the RMS norm chain packs into the writer's
cb_out_rows; O accumulates in place by packer L1 accumulation (no read-modify-write buffer);
cb_sq and cb_rinv were planned as shared by the L2 and RMS phases and split in the build (wrap, see their rows); the state buffers are distinct only where
pipelining across pairs or T = 1 makes their lifetimes overlap. No L1 budget solve, safety fraction
or blocking search exists: the footprint is a closed form checked once on host.

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| Kt = K/32 | 1..4 in INPUTS (K <= 128) | `footprint <= L1_BUDGET` (host check in validate step 5); K multiple of 32 (alignment tagger: `w_non_aligned` refused) |
| Vt = V/32 | 1..4 in INPUTS (V <= 128) | same |
| STATE_IN_DEPTH, STATE_OUT_DEPTH, CONV_ACC_DEPTH | 1 | host constants |
| WINDOW_DEPTH, ROW_DEPTH, TOK_COEF_DEPTH | 2 | host constants |
| B, T, N, Hv, Hk, C | unbounded | appear in no capacity expression (work and DRAM traffic only) |
| L1_BUDGET | `device L1 size - L1 reserved base` queried on host (1460288 B usable per risks.md R11) | device query |

## Total per-core footprint

```
footprint = 4096 * ( Kt*Vt*(STATE_IN_DEPTH + 1 + STATE_OUT_DEPTH)
                   + 14*Kt + 11*Vt + 2*ROW_DEPTH*Vt
                   + 4*gcd(Kt, Vt)*WINDOW_DEPTH
                   + CONV_ACC_DEPTH + 2*ROW_DEPTH + 3*TOK_COEF_DEPTH + 9 )
          + 2048 + 32
```

(fp32 token tensors, SFPU conv.) For bf16 the token-buffer pages above are 2048 B, cb_row_stage adds
2112 B, cb_conv_w is `2*Kt + Vt` pages and cb_conv_acc is not allocated (FPU conv).
This is `l1_footprint_bytes(Kt, Vt, dtype)` in `gated_delta_net_program_descriptor.py` (summed from
`_cb_plan`, the one place CB page counts are written).

Scaling: the state term scales with Kt*Vt and the two state depths; cb_conv_w, the per-tile-row
buffers and the token buffers scale linearly with Kt and Vt; cb_window with gcd(Kt, Vt) and
WINDOW_DEPTH; nothing scales with B, T, N, Hv or C. At Kt = Vt = 4 with Phase 0 depths (fp32):
216 pages + 2080 B = 886816 B (0.61 of 1460288 B); bf16 at Kt = Vt = 4 is smaller (FPU conv drops
36 conv_w pages and cb_conv_acc). At Kt = 2, Vt = 1 (the K > V entry, fp32): 77 pages + 2080 B.

## Data-movement budget

Per call, fp32 (Phase 0), chip geometry Hk=4, Hv=12, K=V=128, C=2560, r=3. "Row reads" means the
reader fetches 64 B face-row segments of only the rows a pair owns when the tile row is shared
(decode), whole tiles when the pair owns the tile row (prefill).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| recurrent_state | 1 (786432 B at B=1) | resident per pair for the whole call (R-STEP) | none |
| recurrent_state_out | 1 | written once from cb_state_out | none |
| mixed_qkv | 1.8 (q/k columns 3x, v columns 1x) | each of the r=3 pair cores of a key head reads that key head's q/k columns; row-granular in decode, so no B-fold tile re-read | none |
| conv_state | 1.8 (rows needed by the window) + 1 more read of the <= 3 rows copied to conv_state_out | window rows as in mixed_qkv; conv_state_out source rows re-read as row segments by the reader (3 rows per pair column) | none |
| conv_weight | 1.8 per pair-block (rows 0..3 only; whole tiles on the bf16 FPU conv path) | read once per pair, resident across tile rows | none |
| z | 1 | v-head columns are owned by one pair | none |
| b, a | Hv x one 64 B segment (decode) / one tile per pair per tile row (prefill) | the gate column tile is shared by the Hv heads | none |
| A_log, dt_bias, norm_weight | once per pair (row-0 segments) | constants re-read per pair; bytes negligible | none |
| output | 1 | each (row, head) written by exactly one pair | none |
| conv_state_out | 1 | each (row, column) written by exactly one pair (v columns by every h, q/k columns by h % r == 0) | none |

Totals (fp32): decode B=1: state 786432 B in + 786432 B out; token-side reads about 0.16 MB
(row reads; minimum about 0.09 MB); output and conv_state_out writes about 18 KB. Prefill B=1,
T=4096: state as decode; mixed_qkv 41.9 MB logical -> 75.5 MB read; z 25.2 MB; output 25.2 MB;
cross-core 0 B in both.

> Cheapest-traffic split considered: (b, h) pairs with kn/qn of each key head computed once and
> multicast to its r pair cores (R-BCAST) -- saves 64 KiB DRAM per decode call at B=1 and 33.5 MB
> per prefill call at T=4096 (mixed_qkv, conv_state, conv_weight from 1.8 to 1.0 crossings),
> adds 2Kt fp32 tiles of multicast per tile row per key head. Implemented: (b, h) pairs, each
> reading its own columns (R-STEP). Deferred because the saved bytes are 4 percent of a decode
> call's DRAM traffic (the state dominates) and a prefill call is bound by 4096 ordered token steps
> per pair, not by DRAM; the multicast layers onto the built structure (the injector is the pair
> core with `h % r == 0`, receivers skip the q/k conv and L2 norm), so R-STEP keeps it reachable.

bf16 token tensors (Refinement 1): crossings unchanged; token bytes halve. Row reads become 64 B
aligned chunks holding the wanted 32 B row (2x the useful bytes on those rows only); row writes
stay 32 B (Tensix-to-DRAM writes need only L1 alignment; repeated calls with neighbouring rows
written by different cores are bit-identical, `test_gated_delta_net_bf16.py`).

## Notes

cb_tok_coef pages are (alpha in every row, beta at row p, alpha at row p): `o = alpha_sel * ro + d * u`
needs no separate select multiply (the compute reads alpha itself as a scalar, element (0, 0), for the
u term and the state decay). cb_window hands off `gcd(Kt, Vt)` column tiles (4 tap pages each) at a
time, one DRAM-read flush and one L1-shift flush per group, with capacity
`4 * gcd(Kt, Vt) * WINDOW_DEPTH`, so q/k/v groups never wrap a block.
