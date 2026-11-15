# L1 ledger: gated_delta_net (regimes R-STEP, R-VSPLIT, R-CHUNK, R-BCAST, R-PREP and R-QKCORE)

Refinement 3 (R-VSPLIT): a core runs one unit = one piece of a pair, owning `Vb = Vt / pieces` state
columns (`v_block_tiles`; R-STEP is `Vb = Vt`). Every `Vt` below is the unit's `Vb`; capacities are
written once in `_cb_plan(Kt, Vb, conv_on_fpu, pieces, chunked)`.

Refinement 4 (R-CHUNK, calls with T >= CHUNK_MIN_TOKENS on Blackhole): the token loop of a tile row is
one chunk step. cb_tok_coef carries 2 chunk pages per tile row instead of 3 per token; cb_gates page 0
carries g instead of alpha; cb_d / cb_sq are not used by the chunk step (cb_d stays allocated, one
page); five buffers are added (rows marked R-CHUNK, allocated only when chunked).

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
| cb_window | `4*G*WINDOW_DEPTH`, G = `_window_group_cols` (R-STEP: largest divisor of 2Kt+Vb <= WINDOW_GROUP_MAX_COLS = 4; R-VSPLIT: 2Kt+Vb, one hand-off per tile row) | 4*G per window group; capacity x2 for double buffering | {pair: streams, t: streams (1 tile row), K: streams (q, k groups pass through), V: streams (v group), c: spans one window group, j: spans 4} | token dtype | reader | compute | per window group | double buffered (depth 2: reader assembles group g+1 during conv of g); uniform group size so no reservation wraps the buffer end; rows outside the token rows are zeroed once by the writer at kernel start and re-zeroed by the reader only when a slot's dirty rows are not covered by the next group's rows; not shared: concurrent with every per-tile-row buffer |
| cb_y_qk | `2*Kt` | 2Kt | {pair: streams, t: spans 1 tile row (32 rows), K: spans Kt (q and k groups), V: streams (n/a), c: spans q,k groups, j: streams (collapsed)} | Float32 (U2D) | compute | compute | conv -> l2norm of one tile row | not with cb_kq_rows: different unpack mode (the L2 sums need U2D fp32, the matmul in0 needs srcA/srcB); not with cb_v: cb_v outlives it (kept through the token loop) |
| cb_v | `Vt` | Vt | {pair: streams, t: spans 1 tile row, K: streams, V: spans Vt, c: v group, j: streams} | Float32 (U2D) | compute | compute | conv -> end of recurrence_block | concurrent with the token loop: no sharing |
| cb_sq | `Kt` | Kt (q, k, then qn*kn squares/products) | {pair: streams, t: spans 1 tile row, K: spans Kt, V: streams, c: streams, j: streams} | Float32 (U2D) | compute | compute | inside l2norm | shared by the q, k and d reductions (disjoint, same Kt block); not with the RMS squares: pushed in Kt blocks and read by index, sharing with Vt-block pushes left a block wrapped past the CB end (measured: wrong RMS factor) |
| cb_sq_o | `Vt` | Vt | {pair: streams, t: spans 1 tile row, K: streams, V: spans Vt, c: streams, j: streams} | Float32 (U2D) | compute | compute | inside rmsnorm | split from cb_sq (wrap, above) |
| cb_rinv | 2 | rsqrt q, k (column-0 tiles) | {pair: streams, t: spans 1 tile row; others streams (reduced)} | Float32 (U2D) | compute | compute | inside l2norm | not with cb_rinv_o, same wrap reason |
| cb_rinv_o | 1 | 1 | {pair: streams, t: spans 1 tile row; others streams (reduced)} | Float32 (U2D) | compute | compute | inside rmsnorm | split from cb_rinv (wrap) |
| cb_conv_acc | `CONV_ACC_DEPTH` = 1 (SFPU conv only; not allocated on the FPU conv path) | 1 | {pair: streams, t: spans 1 tile row, c: one column tile; others streams} | Float32 (U2D) | compute | compute | per column tile | SFPU path: partial tap sum (taps 0,1 packed, taps 2,3 added by packer L1 accumulation), two fp32 half-sync DEST windows per column. FPU path (Refinement 2): the four taps accumulate in one DEST tile by ELWMUL, so no partial sum leaves DEST |
| cb_kq_rows (+ alias cb_kq_rows_f32) | `2*Kt` | 2Kt | {pair: streams, t: spans 1 tile row, K: spans Kt, V: streams, c: q,k groups, j: streams} | Float32 (std) / Float32 (U2D view) | compute; R-BCAST: writer (multicast landing) | compute | tile row | concurrent with the token loop (read matmul in0 every token). Refinement 5 (R-BCAST): kn at [0, Kt) and qn at [Kt, 2Kt) land here by the key head's two multicasts, at the same address on every core; the U2D alias (second format descriptor over the same allocation, lockstep push/pop) feeds the fp32 qn . kn |
| cb_qk_stage | `Kt` (R-BCAST only) | Kt (the injector's qn or kn) | {pair: streams, t: spans 1 tile row, K: spans Kt; others streams} | Float32 | compute (injector) | writer (multicast source) | tile row | not packed into cb_kq_rows: on an injector the writer is that CB's single producer (it lands the other part), so the injector's own part goes through this source buffer; the multicast loops it back into the injector's cb_kq_rows |
| cb_knT | `Kt` | Kt | {pair: streams, t: spans 1 tile row (tokens on columns), K: spans Kt, V: streams, c: k group, j: streams} | Float32 (std) | compute | compute | tile row | concurrent with the token loop (increment in0 every token) |
| cb_d | `D_PAGES` = 2 | 2 ([kn; qn] x kn^T: d = qn . kn of token row p at (p, p) of tile 1, read as a scalar) | {pair: streams, t: spans 1 tile row, K: streams (reduced by the matmul), V: streams, c: streams, j: streams} | Float32 | compute | compute | tile row | concurrent with the token loop; one matmul output block (Refinement 8: was 1 page, the SFPU row sum of qn * kn) |
| cb_gate_in | `2*ROW_DEPTH` | 2 (b, a tiles) | {pair: streams, t: spans 1 tile row, K: streams, V: streams, c: streams (gate column h//32), j: streams} | Float32 (U2D; input dtype) | reader | compute | tile row | double buffered across tile rows |
| cb_gate_w | 2 | 2 (A_log, dt_bias rows) | {pair: streams (refilled per pair), t: streams (constant), K, V, c, j: streams} | Float32 (U2D) | reader | compute | pair | constant per pair; 2 pages genuinely constant |
| cb_gates | 2 | 2 (alpha, beta) | {pair: streams, t: spans 1 tile row, K, V, c, j: streams} | Float32 (std; dataflow reads elements) | compute | writer | tile row | consumed by the writer across the whole token loop: concurrent |
| cb_tok_coef | `3*TOK_COEF_DEPTH` (R-CHUNK: `2*TOK_COEF_DEPTH`) | 3 (one token: alpha, beta and 1 at row p, column 0) / 2 (R-CHUNK: g, beta of one tile row in column 0, 0 on rows that are no token) | {pair: streams, t: streams (one token per fill; R-CHUNK one tile row), K, V, c, j: streams} | Float32 (U2D) | writer | compute | token / tile row | depth 2 so the writer runs one token (tile row) ahead; not shared: concurrent with every token-loop buffer. R-CHUNK: the writer masks rows that are no token to exactly 0 (gate tiles hold arbitrary values there) |
| cb_gram (R-CHUNK) | 2 | [Kn Kn^T; Qn Kn^T] | {pair: streams, t: spans 1 tile row twice (32 x 32 tokens), K: contracted, V: streams} | Float32 (U2D) | compute (matmul_block) | compute | tile row, Gram -> L/QD window | own CB: matmul_block's in0/in1/out must be distinct, and the SFPU products need the U2D view; dead before cb_chunk_mat is read, but both are 1-2 pages |
| cb_chunk_vec (R-CHUNK) | 5 | D, c, exp(c), exp(c_last - c), beta exp(c) | {pair: streams, t: spans 1 tile row; others streams} | Float32 (U2D) | compute | compute | tile row (chunk_vectors -> end of chunk step) | read by index throughout the chunk step (D in the L/QD window, exp(c) in O, W in Uw, beta exp(c) in the right-hand side) and by read_tile_value (c_last, exp(c_last)); live together: no sharing |
| cb_chunk_mat (R-CHUNK) | 2 | L, (Qn Kn^T) * D | {pair: streams, t: spans 1 tile row twice; others streams} | Float32 (std) | compute | compute (triangle_solve reads L from L1; QD is matmul in0) | tile row | not U2D: QD feeds srcB; the solve reads L straight from L1 |
| cb_uw (R-CHUNK) | `Vt` | Vt | {pair: streams, t: spans 1 tile row, V: spans Vt} | Float32 (std) | compute | compute | tile row (Uw -> update) | the update operand exp(c_last - c) (*)col U; not packed over cb_u: U is still the in1 of the O matmul when Uw is produced |
| cb_tri_mask (R-CHUNK) | 1 | lower-triangular 0/1 (column <= row) | {all axes: streams (constant)} | Float32 (U2D) | reader (once: NoC zero + the 528 ones) | compute (never popped) | kernel | constant; masks the decay matrix D |
| cb_reads | `2*Vt` | 2Vt | {pair: streams, t: streams (one token), K: streams (contracted), V: spans Vt, c, j: streams} | Float32 (U2D) | compute | compute | token | not with cb_u: cb_reads is consumed while cb_u is produced in the same op (concurrent); different unpack mode |
| cb_u (+ alias cb_u_f32) | `Vt` | Vt | {pair: streams, t: streams (one token), K: streams, V: spans Vt, c, j: streams} | Float32 (std) | compute | compute | token (vector op -> update) | alias cb_u_f32 is a second format descriptor over the same allocation: u feeds the increment matmul (srcB view) and the fp32 o term (U2D view); lockstep push/pop |
| cb_o_acc | `Vt` | Vt | {pair: streams, t: spans 1 tile row (one row per token, packer L1 accumulation), K: streams, V: spans Vt, c, j: streams} | Float32 (U2D) | compute | compute | tile row | live through the token loop: concurrent with cb_u, cb_reads |
| cb_z | `Vt*ROW_DEPTH` | Vt | {pair: streams, t: spans 1 tile row, K: streams, V: spans Vt, c, j: streams} | Float32 (U2D; input dtype) | reader | compute | tile row | double buffered across tile rows |
| cb_norm_w | `Vt` | Vt | {pair: streams (refilled per pair), t: streams (constant), K: streams, V: spans Vt, c, j: streams} | Float32 (U2D) | reader | compute | pair | constant per pair |
| cb_out_rows | `Vt*ROW_DEPTH` | Vt | {pair: streams, t: spans 1 tile row, K: streams, V: spans Vt, c, j: streams} | Float32 (output dtype) | compute | writer | tile row | pack-into-destination (no intermediate between the norm chain and the writer); double buffered |
| cb_reduce_scaler | 1 (2048 B page) | 1 | {pair, t, K, V, c, j: streams (constant)} | Float16_b (bf16-packed SUM scaler 1.0) | reader | compute | kernel | constant; literal 1 page |
| cb_scratch | 1 | 3 rows | reader-private | token dtype | reader | reader | conv_state_out copy | face-row staging for the bit-exact conv_state_out copy; never seen by compute |
| cb_window_zeroed | `WINDOW_DEPTH` pages of 32 B | 1 token per cb_window slot | {all axes: streams} | n/a (handshake) | writer | reader | kernel start | writer-to-reader tokens: slot k of cb_window has been zeroed (the writer is idle until the first gates; slot by slot so the reader starts on slot 0 without waiting for the rest). Refinement 5: only the slots the core's reader will use, each the size of the core's window group (host RT args), instead of the whole buffer: on perf_decode_b1 147 KiB of self-addressed NoC reads per core became 8 KiB (v-only core) / 40 KiB (injector) |
| cb_hist_stage | 2*G (window group columns) | two tiles per column of a window group: history rows, then the single token row | reader-private | token dtype | reader | reader | between the window group's DRAM flush and its row copies | Refinement 3: the 3 conv_state history rows of a column are one aligned span read in place, then copied out row by row. Refinement 5: one request per 16-row band covers both face halves, and a single token row (decode) is read the same way into the second tile per column, then copied into the tap tile (whose other rows must stay zero). Not shared with cb_row_stage: both are live in the same flush |
| cb_osum_part | 1 (R-VSPLIT only) | 1 column-0 tile | {pair: streams, t: spans 1 tile row, V: reduced over the piece's Vb} | Float32 | compute | writer | tile row | the piece's partial sum o^2; not packed into cb_osum_gather directly: the gather buffer is filled by NoC writes from every piece (the writer also sends this tile to the peers) |
| cb_osum_gather | `2*pieces` (R-VSPLIT only) | `pieces` partials of one tile row | {pair: streams, t: spans 1 tile row, V: one partial per piece} | Float32 (U2D) | writer (+ peer writers' NoC writes into slot `piece`) | compute | tile row | two tile rows of slots: a peer can be at most one tile row ahead (its next combine needs this core's next partial, produced only after this core consumed the current slots); same L1 address on every core (identical CB layout) so peers write by address |
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
| Vb = Vt / pieces | 1..Vt | `_value_pieces`: largest divisor of Vt (<= MAX_PIECES = 4, the combine's DEST window) with B*Hv*pieces <= grid cores (one unit per core), else 1; the descriptor re-checks `l1_footprint_bytes(Kt, Vb, dtype, pieces) <= L1_BUDGET` |
| unit_depth = min(UNIT_INPUT_DEPTH, units per core) | 1..2 | host: 2 on cores that run several units (decode B=16), lowered until the plan fits L1 |
| STATE_IN_DEPTH, STATE_OUT_DEPTH | 2, capped at unit_depth | host constants |
| CONV_ACC_DEPTH | 1 | host constant |
| WINDOW_DEPTH, ROW_DEPTH, TOK_COEF_DEPTH | 2 | host constants |
| B, T, N, Hv, Hk, C | unbounded | appear in no capacity expression (work and DRAM traffic only) |
| L1_BUDGET | `device L1 size - L1 reserved base` queried on host (1460288 B usable per risks.md R11) | device query |

## Total per-core footprint

```
footprint = 4096 * ( Kt*Vt*(STATE_IN_DEPTH + 1 + STATE_OUT_DEPTH)
                   + 14*Kt + 11*Vt + 2*ROW_DEPTH*Vt
                   + 4*G*WINDOW_DEPTH + 2*G (cb_hist_stage)
                   + CONV_ACC_DEPTH + 2*ROW_DEPTH + 3*TOK_COEF_DEPTH + 10 )  (cb_d: 2 pages since Refinement 8)
          + 2048 + 32*WINDOW_DEPTH
          [+ 4096*(1 + 2*pieces) for R-VSPLIT]
          [+ 4096*Kt for R-BCAST (cb_qk_stage)]
          [+ 4096*(10 + Vt - TOK_COEF_DEPTH) for R-CHUNK]
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

R-CHUNK (Refinement 4): DRAM crossings unchanged (same reads and writes per tile row; the chunk
step only changes compute). The reader's conv window for a full tile row now copies each shifted
tap as row runs inside a face (6 L1 NoC copies per tap per column instead of one per row and half)
and reads the previous tile row's last 3 rows as one aligned span per column (cb_hist_stage),
replacing 6 staged row reads per column.

R-VSPLIT (Refinement 3), decode B=1 bf16, pieces = 4: each of 48 cores reads the key head's q/k
columns (8 window columns, 8 raw conv_weight row spans) plus its 1 v column, its Kt x 1 state slice
and writes its 1 output column and state slice. Token-side reads: q/k columns 4x per pair (12x per
key head instead of 3x: mixed_qkv/conv_state/conv_weight q/k bytes x4 vs R-STEP); state, z, output,
recurrent_state_out stay at 1 crossing. Weights and gate tiles are read as aligned row spans (conv
taps rows 0..3, rank-1 weights row 0, gate token rows), about 1/8 of the whole-tile bytes. Cross-core:
per tile row each piece writes one 4 KiB partial tile to every piece of its pair (48 x 4 x 4 KiB =
768 KiB per decode call) plus 3 semaphore increments. R-BCAST (deferred) would cut the q/k reads back.

R-BCAST (Refinement 5), decode B=1 bf16, 48 cores in 4 key-head groups of 12: per key head one core reads
the q columns (4 window columns, 4 conv_weight row spans) and one the k columns; all 48 read their own v
column, state slice, z and gates. mixed_qkv / conv_state / conv_weight q/k bytes: 1 crossing per key head
(were 12). Requests: every row span is one DRAM request per 16-row band (both face halves; were one per
half), a decode token row one request per column (was two staged half-row reads). Cross-core added per
call: per key head two multicasts of Kt fp32 tiles (16 KiB each, to the group's 12 cores, flag signal,
no handshake) = 128 KiB of multicast payload. Self-addressed NoC traffic: window zeroing 147 KiB -> 8 /
40 KiB per core; the single-row staging adds 2 x 32 B local copies per window column.

## R-PREP (Refinement 6)

Two roles with their own buffers (`_prep_cb_plan` returns shared, state and prep plans). Buffers written by other
cores (`REMOTE_CBS`) are allocated first and on every core, so each has one L1 address everywhere (the writers
take it from their own allocation); every other buffer exists only on the role that uses it (the merged
all-on-every-core plan first used carried about 700 KB of prep buffers on every state core and capped the inbox
at 6 slots). D = inbox slots (`INBOX_DEPTH` = 12 at most, the deepest that fits; 8 at model_prefill_4096 bf16).
Vb = the piece's v columns (2 at the model geometry under `PREP_MAX_PIECES` = 2); r = value heads per key head.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| cb_in_kq (shared) | `D*2*Kt` | 2Kt per slot: beta exp(c) (*) kn, exp(c) (*) qn | {t: D tile rows in flight, K: spans Kt; others streams} | Float32 (std, matmul in0) | prep writer (multicast) + state reader (push on landing) | state compute | grant -> pop | ring of D slots at fixed addresses (row i in slot i % D); not shared: all inbox buffers of D rows are live together |
| cb_in_knt (shared) | `D*Kt` | Kt: kn^T (*) exp(c_last - c)^T | as cb_in_kq | Float32 (std, update in0) | as cb_in_kq | state compute | grant -> pop | as cb_in_kq |
| cb_in_mat (shared) | `D*2` | M = (I + L)^-1, QD | {t: D tile rows; others streams} | Float32 (std, matmul in0) | as cb_in_kq | state compute | grant -> pop | as cb_in_kq |
| cb_in_vec (shared) | `D*1` | exp(c) (read_tile_value: exp(c_last) of the last token row) | {t: D tile rows} | Float32 | as cb_in_kq | state compute | grant -> pop | as cb_in_kq |
| cb_in_v (shared) | `D*Vb` | beta (*) v of the piece's columns | {t: D tile rows, V: spans Vb} | Float32 (U2D) | prep writer (unicast) + state reader | state compute | grant -> pop (popped last: a free slot here frees the row) | as cb_in_kq |
| cb_o_in (shared) | `O_IN_DEPTH*r*Vt` | o of one tile row of the r heads (head-major) | {t: 2 of the prep core's rows, V: spans r Vt} | Float32 (U2D) | state writers (NoC writes into slot k % 2) + prep writer (push after the landed count) | prep compute (RMS norm) | landing -> RMS norm one row later | two of the prep core's rows: the next row's o may land while the RMS norm of the previous one runs |
| cb_o_acc (state) | `O_ACC_DEPTH*Vb` | o of the piece | {t: 2 tile rows, V: Vb} | Float32 | state compute | state writer | tile row | double buffered so the next chunk step overlaps the o send |
| cb_uw (state) | `Vb` | rhs = beta V - beta exp(c) Rk, then (local) Uw | {t: 1 tile row, V: Vb} | Float32 (std, matmul in1) | state compute | state compute | rhs -> M x rhs | reused as the rhs scratch: dead before U is produced |
| cb_prep_send (prep) | `r*(3*Kt + 3 + Vt)` (`PREP_SEND_HEADS` head blocks, default one tile row's) | head blocks of one tile row | {t: 1 tile row, heads: r} | Float32 | prep compute | prep writer | head block packed -> sent | one tile row of head blocks: compute packs the next row's head j once the writer has sent this row's head j (the writer sends block by block); two rows (456 KB) did not fit with D = 6 |
| cb_v_prep (prep) | `r*Vt` | conv'd v of the r heads | {t: 1 tile row, V: r Vt} | Float32 (U2D) | prep compute (conv) | prep compute (beta scaling) | conv -> head blocks | not packed into cb_prep_send directly: each head's v is scaled by its beta after the gates of that head |
| cb_chunk_vec (prep) | `CHUNK_VEC_PAGES + 1` | D, c, exp(c), exp(c_last - c), beta exp(c), exp(c_last - c)^T | {t: 1 tile row, one value head} | Float32 (U2D) | prep compute | prep compute | one head's block | the transposed decay vector stays fp32 (U2D) for the knT scaling |
| cb_chunk_mat (prep) | 1 | L | {t: 1 tile row, one head} | Float32 (std; the solve reads L1) | prep compute | prep compute (solve) | L -> M | QD goes straight into the head block |
| cb_tri_mask | 2 (chunked) | lower-triangular mask, identity | constant | Float32 (U2D) | reader (once) | compute | kernel | the identity is the right-hand side of the M solve |
| prep: cb_z, cb_norm_w (Vt), cb_out_rows, cb_sq_o, cb_rinv_o (r) | `r*Vt*ROW_DEPTH`, `Vt`, `r*Vt*ROW_DEPTH`, `r*Vt`, `r` | the RMS norm of the r heads of a row | {t: tile row, V: r Vt} | as the local rows | reader / compute | compute / writer | tile row | the RMS norm moved to the prep core with the output store; the state core keeps none of them (no o^2 all-gather: the prep core sees every piece) |

Footprint at model_prefill_4096 (bf16, Kt = Vt = 4, r = 3, 2 pieces, D = 8): state core 809 KB, prep core
1421 KB (budget 1460 KB). fp32 tokens at this geometry do not fit any D (prep core 1.65 MB at D = 2: SFPU conv
weights 320 KB), so they keep the local regime.

R-PREP data movement (model_prefill_4096, bf16): DRAM as R-VSPLIT for the state, z and output; mixed_qkv,
conv_state, conv_weight read once per (key head, tile row) by its prep core (q/k columns 1 crossing per key
head, v 1). NoC per tile row and key head: r head blocks of 15 fp32 tiles (60 KB) multicast to the head's 2
pieces (one row run, two runs when a head straddles a virtual-column gap or a row), r Vt unicast v tiles (48 KB),
r Vt o tiles back (48 KB), 2 r pieces grant / landed semaphore increments.

## R-QKCORE (Refinement 7)

One plan on pair and q / k cores (the multicast lands at the same cb_kq_rows address everywhere: identical CB layout,
as R-BCAST). Changes against R-STEP decode, u = unit_depth (2 when a core runs several units):

| CB | Capacity (pages) | Change | Why / shares with |
|----|------------------|--------|-------------------|
| cb_state_in (+ mm) | `Kt*Vb*min(STATE_IN_DEPTH, u)` | 1 -> u | the next unit's S0 is read while this unit computes; STATE_IN_DEPTH = 1 measured 95.5 vs 93 us at B=16 |
| cb_state_out | `Kt*Vb*min(STATE_OUT_DEPTH, u)` | 1 -> u | the writer issues a non-last unit's state only after the next unit's coefficients and q/k rows (issuing blocks the writer under NoC back-pressure), so compute's next update needs the second slot |
| cb_conv_w, cb_gate_w, cb_norm_w | x u | 1 -> u | per-unit weights in flight; a same-head next unit copies them from the previous slot (local NoC copy) instead of re-reading page 0 of each small tensor, which all sits in one DRAM bank |
| cb_window | `4*max(Kt, Vb)*WINDOW_DEPTH` | G = Vb on pair cores (v only, one group per unit), 1 column per group on a q / k core | the q / k core hands each column over as soon as its rows are gathered, so conv overlaps the next column's copies |
| cb_hist_stage | `max(2*Vb, Kt*(1 + cs), 1 + 2*cs)`, cs = ceil(3B/32) | q / k core: its Kt current-row tiles and Kt*cs conv_state tiles (read whole), then for conv_state_out one x tile, cs input and cs output tiles | the window gather and the conv_state_out pass are sequential (the writer's slot tokens close the gather), so they share the staging |
| cb_qk_stage | `Kt` | as R-BCAST | the q / k core's qn or kn, multicast source |
| cb_scratch | 1 | q / k core: reader -> writer token "staging landed" | the writer copies the history rows from history_split(B) on out of the reader's staging |
| cb_window_zeroed | `WINDOW_DEPTH` | q / k core: one token per column, pushed after the writer zeroed the slot and copied its rows | |

Footprint (bf16, Kt = Vt = 4, B = 16, u = 2): 800896 B per core (0.55 of the budget; state 128 KB in, 64 KB working, 128 KB out).

R-QKCORE data movement (decode B=8 / 16, bf16): recurrent_state 1 crossing each way (6.3 / 12.6 MB per direction);
mixed_qkv and conv_state q/k columns 1 whole-tile crossing per key head (R-STEP read them as B r row spans), v
columns one row span per pair; conv_weight q/k once per key head, v once per pair core; b, a, A_log, dt_bias,
norm_weight once per pair core (later units copy them locally); conv_state_out whole tiles once (Ct column tiles
over the 2 Hk q / k cores; R-STEP wrote it as 3 B r row segments per column). Cross-core: 2 Hk multicasts of Kt fp32
tiles (16 KiB) to the key head's pair cores.

## Notes

cb_tok_coef pages are (alpha in every row, beta at row p, alpha at row p): `o = alpha_sel * ro + d * u`
needs no separate select multiply (the compute reads alpha itself as a scalar, element (0, 0), for the
u term and the state decay). cb_window hands off G column tiles (4 tap pages each) at a time, one
DRAM-read flush and one L1-shift flush per group, with capacity `4 * G * WINDOW_DEPTH`; G divides
2Kt + Vb, so no reservation wraps the buffer end (compute consumes per column, a group may straddle
q / k / v). The FPU conv batches a divisor of G columns per DEST window for the same reason.
