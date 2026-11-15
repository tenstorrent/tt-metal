# Operation Design: gated_delta_net

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (compute + row-granular data movement + carried state) |
| Goal | One device program per call that advances B sequences by T tokens: causal 4-tap depthwise conv + SiLU over a carried 3-row history, gates, L2-normalized q/k, the gated delta-rule recurrence on a carried float32 K x V state per (sequence, value head), gated RMS norm. Returns output rows and both states. |
| Math | M1..M14 of the spec. Recurrence evaluated in the F3 form per token: `alpha = exp(g)`, `[rk; ro] = [kn; qn]^T S_{t-1}`, `u = beta (v - alpha rk)`, `o = alpha ro + (qn . kn) u`, `S_t = alpha S_{t-1} + kn u^T` (algebraically equal to M10..M12, `docs/explorations/gated_delta_net/math.md` F3). |
| References | spec (task prompt); `eval/golden_tests/gated_delta_net/{feature_spec.py,helpers.py}`; `docs/explorations/gated_delta_net/{math.md,tensix.md,risks.md}`; `.claude/references/{blocking-model.md,l1-footprint-discipline.md,precision_convention.md}`; `ttnn/cpp/ttnn/kernel_lib/*` |

Symbols used throughout: `B` sequences, `T` tokens per sequence in this call, `N = B*T` token rows,
`Hk`, `Hv` key/value heads, `r = Hv/Hk`, `K`, `V` head dims, `C = 2*Hk*K + Hv*V`,
`Kt = K/32`, `Vt = V/32`, `Ct = C/32`, `Gt = ceil(Hv/32)` (gate tile columns),
`TRn = ceil(N/32)` (token tile rows), `pair = b*Hv + h` (the work unit), `hk = h // r`.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| mixed_qkv | ttnn.Tensor (1, N, C) | yes | dtype = op dtype, TILE, DRAM interleaved | - | tensor |
| z | ttnn.Tensor (1, N, Hv*V) | yes | same dtype | - | tensor |
| b | ttnn.Tensor (1, N, Hv) | yes | same dtype | - | tensor |
| a | ttnn.Tensor (1, N, Hv) | yes | same dtype | - | tensor |
| conv_state | ttnn.Tensor (1, 3B, C) | yes | same dtype | - | tensor |
| recurrent_state | ttnn.Tensor (B, Hv, K, V) | yes | float32 always | - | tensor |
| conv_weight | ttnn.Tensor (4, C) | yes | same dtype | - | tensor |
| A_log, dt_bias | ttnn.Tensor (Hv) | yes | same dtype | - | tensor |
| norm_weight | ttnn.Tensor (V) | yes | same dtype | - | tensor |
| epsilon | float | no (kw-only) | > 0 | 1e-6 | RT (float bits) |
| compute_kernel_config | ttnn.ComputeConfigDescriptor | no (kw-only) | fp32_dest_acc_en must be True; fidelity / approx not gated | `default_compute_kernel_config()` | passed as-is as `config=` |
| T, B, pair range, tile-row geometry | derived ints | - | - | - | RT |
| Kt, Vt, Hv, Hk, r, Gt | derived ints | - | - | - | CT (geometry; one compile per head geometry) |

`default_compute_kernel_config()` is exported from the op module (HiFi4, `fp32_dest_acc_en=True`,
`math_approx_mode=False`), returns a fresh descriptor per call, and is the only place the default
lives (`eval/golden_tests/gated_delta_net/axes.py` imports it). `None` resolves through it; a
caller config is passed as the compute kernel's `config=compute_kernel_config` after `validate()`
accepted it. The only field the program descriptor writes on that object is
`unpack_to_dest_mode` (the per-CB `UnpackToDestFp32` list of Circular Buffers, required by the
precision contract); `fp32_dest_acc_en`, `math_fidelity`, `math_approx_mode` and every other field
are left exactly as the caller set them.

### Registry contract (op file)

| Item | Value |
|------|-------|
| INPUT_TAGGERS | `alignment` = `tag_alignment(inputs, axes)` and `sequence_mode` = `tag_sequence_mode(inputs, axes)`, exactly as the spec defines them; `inputs = (mixed_qkv_shape, recurrent_state_shape)` |
| SUPPORTED (Phase 0) | `dtype: [ttnn.float32]`, `layout: [ttnn.TILE_LAYOUT]`, `fp32_dest_acc_en: [True]`, `alignment: ["tile_aligned", "h_non_aligned"]`, `sequence_mode: ["decode", "prefill"]` |
| EXCLUSIONS | `[]` |
| validate() order | (1) shape/dtype/epsilon checks -> `ValueError`; (2) build axes (dtype and layout from mixed_qkv, fp32_dest_acc_en from the resolved config, taggers); (3) SUPPORTED per axis -> `UnsupportedAxisValue`; (4) EXCLUSIONS -> `ExcludedCell`; (5) the L1 predicate of Work Distribution -> `RuntimeError` (unreachable for every INPUTS shape) |
| ValueError list | rank/shape not as the signature; N % B != 0; z/b/a row count != N or width != Hv*V / Hv; conv_state rows != 3B or width != C; conv_weight not (4, C); A_log/dt_bias not (Hv); norm_weight not (V); `C - Hv*V` not a positive multiple of `2K`; `Hv % Hk != 0`; recurrent_state not float32; the other nine tensors not one dtype; `epsilon <= 0` |
| Entry point | first line `validate(...)`; allocates the three outputs (`ttnn.allocate_tensor_on_device`, positional args), builds one ProgramDescriptor, one `ttnn.generic_op` call. No other ttnn op touches an input or output. |

Structural impossibilities: none beyond feature_spec.py (INVALID is empty and stays empty).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | as Parameters; B, Hv, K, V from recurrent_state; T = N / B; Hk = (C - Hv V) / (2K) |
| Dtype | Phase 0: float32 for all ten tensors (recurrent_state float32 in every cell) |
| Layout | TILE (rank-1 tensors A_log, dt_bias, norm_weight: one tile row, values in row 0) |
| Memory | DRAM interleaved (Phase 0; TARGET has no placement axis) |

Padding (rows beyond N / 3B, conv_weight rows 4..31, gate columns beyond Hv, rows 1..31 of the
rank-1 tensors) is arbitrary, finite or not, and never enters a sum, a state or a valid output row
(see Key Risks, "poison").

### Output

| Property | Value |
|----------|-------|
| Shape | output (1, N, Hv*V); conv_state_out (1, 3B, C); recurrent_state_out (B, Hv, K, V) |
| Dtype | output: dtype of mixed_qkv; conv_state_out: dtype of conv_state; recurrent_state_out: float32 |
| Layout | TILE |
| Memory | memory_config of mixed_qkv, conv_state, recurrent_state respectively (Phase 0: DRAM interleaved); always new tensors |

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| b (sequence) | independent: no result spans b (each sequence uses its own rows and state) | `pairs_per_block` (with h) | 1 pair | host const `PAIRS_PER_BLOCK = 1` in `gated_delta_net_program_descriptor.py` | split across cores: pairs `b*Hv+h` handed out by `ttnn.split_work_to_cores(grid, B*Hv, row_wise=True)`; each core walks its contiguous pair range | knob-turn |
| h (value head) | independent for M9..M14; reuse-shared operand: kn, qn of key head `hk` feed the r value heads `hk*r..hk*r+r-1` | `pairs_per_block` (shared with b) | 1 pair | same constant | same split (pair index folds b and h) | knob-turn (assignment); the kn/qn broadcast is a scheme-change (regime R-BCAST) |
| t (token, recurrence M10..M12) | dependent: S carries from each token to the next, ordered, non-commutative | `token_rows_per_block` (prep/norm block along t) ; step = 1 token (mechanism cap) | 1 tile row = up to 32 token rows | host const `TOKEN_TILE_ROWS_PER_BLOCK = 1` | not split across cores: one core owns every token of its pair in order | scheme-change (F4 segments, R-SEG) |
| t (token, conv window M1..M3) | dependent as a window: row t reads rows t-3..t of the same sequence (halo 3 rows) | same block (tile row) | 1 tile row | same | follows the recurrence owner (same core) | none needed |
| K (key dim) | dependent: L2 sums (M7, M8) and both reads (M10, M12) contract K | `k_block_tiles` | Kt (whole) | derived `Kt = K/32` (CT) | not split | scheme-change (partial-read combine), not planned |
| V (value dim) | independent for M9..M12 (state columns never mix); dependent for M13 (mean over V) | `v_block_tiles` (state column block) | Vt (whole) | derived `Vt = V/32` (CT); `V_BLOCK_TILES = None` means Vt | not split in Phase 0 | scheme-change (R-VSPLIT: needs the M13 partial combine) |
| c (channel, conv) | independent: depthwise conv, no result spans c | `conv_group_tiles` (columns per conv block) | one channel group: Kt (q), Kt (k), Vt (v) columns | derived from Kt, Vt | follows the pair (the pair's 2Kt+Vt columns) | knob-turn |
| j (tap) | dependent: 4 terms per output element (M2) | taps per block | 4 (whole) | literal of the math (`CONV_TAPS = 4`, spec constant, not a knob) | not split | none |
| Gate columns (Hv inside the gate tile) | independent; one element per (row, h) | gate tile column `h // 32` | 1 tile | derived | follows the pair | none |

Why `pairs_per_block = 1`: a pair's recurrence is one sequential chain over 16 (Kt*Vt) state
tiles; a second pair in the same block shares no state and, in Phase 0, no operand (kn/qn reuse
across the r heads of a key head is regime R-BCAST). The fixed per-pair cost (S0 load, conv weight
tap expansion) is paid once per pair, not per token, so a coarser pair block amortizes nothing on
the critical path. When a core owns several pairs they run back to back, with the next pair's S0
and conv weights prefetched behind the current pair (buffer depth, below).

Why the recurrence step is 1 token: the token-step regime (R-STEP) is ordered by construction; the
coarser step is the chunked regime (R-CHUNK), recorded as deferred. Everything that is not ordered
(conv, gates, L2 norm, transposes, RMS norm) runs on the whole tile-row block.

Work-assignment check per stage (every stage computes on the pair's own core; no stage runs on one
core while others wait):

| Stage | Data and its axes | Assignment |
|-------|-------------------|------------|
| gates, conv, L2 norm (prep) | 32 token rows x (2Kt+Vt) columns + gate tile of the pair | pair core; b/h independent so every core preps its own pairs concurrently |
| recurrence | S of the pair (Kt x Vt) x tokens | pair core; ordered along t |
| RMS norm + gate | 32 rows x Vt | pair core |
| conv_state_out copy | rows 3b..3b+2 x columns | pair core: v columns of h always, q and k columns of hk when `h % r == 0`; every column of sequence b written exactly once |
| recurrent_state_out | Kt x Vt tiles of the pair | pair core |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| cb_window | `WINDOW_DEPTH` | 2 conv groups | reader assembles the next channel group's tap tiles while compute convolves the current one |
| cb_tok_coef | `TOK_COEF_DEPTH` | 2 tokens | writer expands token p+1's alpha/beta while compute runs token p |
| cb_gate_in, cb_z | `ROW_DEPTH` | 2 tile rows | next tile row's b/a/z land while the current one is processed |
| cb_out_rows | `ROW_DEPTH` | 2 tile rows | compute normalizes tile row tr+1 while writer drains tr |
| cb_state_in | `STATE_IN_DEPTH` | 1 state (Kt*Vt pages) | reader prefetches the next pair's S0 once compute has consumed the current S0 (after token 0) |
| cb_state_out | `STATE_OUT_DEPTH` | 1 state | writer drains a finished state while compute starts the next pair |
| everything else | none (resident per tile row or per pair) | - | - |

All depths are host constants in `gated_delta_net_program_descriptor.py`; CB page counts are
computed from them, never restated.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Matrix engine operands (srcA/srcB) are 19-bit (`tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:314-318`) | S, alpha may never be an FPU operand | S decay and carry: unpack-to-DEST fp32 + SFPU only; increment added by matmul accumulation onto DEST; alpha only via unpack-to-DEST CBs | state loses bits every token (state_carry cases fail at 0.24 relative RMS) |
| A CB tagged `UnpackToDestFp32` cannot feed srcA/srcB (`tt_metal/api/tt-metalium/base_types.hpp:37-38`) | S is both an SFPU operand and a matmul srcB operand | two CB indices over one allocation (cb_state_in/cb_state_in_mm, cb_state/cb_state_mm), lockstep credit | silent truncation or garbage reads |
| fp32 DEST holds 4 tiles in half-sync (`ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:103`) | tiles live at once in the custom update and token-vector ops | update batches of `min(DEST_AUTO_LIMIT - 1, Kt*Vt)` state tiles + 1 alpha tile; read matmul subblocking left to the helper | DEST overrun, wrong values |
| SFPU broadcast reads column 0 / row 0 only (`tt_metal/hw/inc/api/compute/sfpu_binary_bcast.h:48-61, 309-325`) | per-token scalars, conv tap rows, norm weight rows | writer writes alpha/beta/select into column 0 of per-token tiles; reader writes tap row j of conv_weight into row 0 of tap tile j | wrong tap / wrong head used |
| Recurrence order (M10..M12) | token step within a pair | one token per step, valid rows only (`lo <= row < hi`) | wrong state; padded rows taken as tokens |
| Row-granular DRAM transfers: a tile row of fp32 is two 64 B face-row segments | row-segment reads/writes of the token tensors and conv_state_out | Phase 0 dtype fp32 only (64 B segments); bf16 (32 B segments) is gated behind the dtype refinement (Key Risks) | sub-alignment write corrupts a neighbouring sequence's row |
| L1 per core | Kt, Vt (state and weight residency) | host predicate `footprint(Kt, Vt) <= L1_BUDGET` (l1_ledger.md); true for K, V <= 128 (all INPUTS) | launch failure; above the predicate the deferred R-VSPLIT serves the shape |
| `ttnn.split_work_to_cores` | cores used | `min(B*Hv, grid cores)` | - |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (DRAM boundary: each input once, each output once) | What a bigger block buys |
|--------|--------|-----------|-------|------------------------------------------------------------------------------|--------------------------|
| R-STEP: state of one pair resident, token steps (F3) | built | every call (all INPUTS) | pair state Kt x Vt resident; prep/norm block = 1 tile row x pair columns; step = 1 token | recurrent_state 1 in / 1 out (minimum); output, z, conv_state_out 1 (minimum); v columns of mixed_qkv/conv_state/conv_weight 1; q,k columns r times (one read per value head): `(2HkK*r + HvV)/C` = 1.8 at the chip geometry; gate/A_log/dt_bias segments Hv times (tiny); token tensors read row-granular, so decode reads only the pair's own rows (no B-fold tile re-read) | tile-row block: one set of conv/gate/norm inits, one window assembly and one W expansion amortized over up to 32 tokens; per-token fixed cost is init churn of the update batches (intended: once per DEST batch) |
| R-BCAST: kn/qn of a key head computed once, multicast to the r * pieces cores of that key head | built (Refinement 5) on decode R-VSPLIT: of each (b, hk) group of r * pieces consecutive units (consecutive cores), the first core (q injector) reads, convolves and L2-normalizes only the q columns (+ its v), the second (k injector) only k (+ its v), the rest only their v columns; each injector multicasts its Kt fp32 tiles (qn or kn) into cb_kq_rows of the whole group (`ttnn.Mcast` + `McastArgs` on the writer, one channel per injector, flag signal, no handshake: one hand-off per call into untouched CB slots at the same address on every core); every core then derives d = qn . kn and kn^T from the landed rows (bit-identical to the unshared path). The injector runs conv + L2 of its part before its v column (the part is what the group waits for) | `_share_qk`: `SHARE_QK_PREP` and pieces > 1 and T = 1 (one unit per core, one tile row per unit); `SHARE_QK_PREP = False` gives the unshared path | same as R-VSPLIT | mixed_qkv/conv_state/conv_weight q/k: 1 crossing per key head (R-VSPLIT read them r * pieces times); adds 2 multicasts of Kt fp32 tiles per key head | q/k conv + L2 once per key head, split over 2 cores |
| R-VSPLIT: state columns split across `Vt / v_block_tiles` cores per pair, partial sum of o^2 combined across the group for M13 | built (Refinement 3), without R-BCAST: each piece reads and preps the pair's q/k columns itself (q/k reads x pieces); the combine is one column-0 fp32 tile per tile row per piece, all-gathered by unicast NoC writes into the same L1 slot on every piece core plus a cumulative semaphore count; weights / gate tiles read as aligned row spans to offset the replicated reads | `_value_pieces`: pieces = largest divisor of Vt (<= MAX_PIECES = 4) with `B*Hv*pieces <= grid cores` (one unit per core, lockstep tile rows); `V_BLOCK_TILES` forces it | pair state Kt x v_block_tiles | adds the combine; q/k token columns x pieces | per-core state update work / pieces |
| R-CHUNK: F1 chunk of 32 tokens per tile row (closed form inside the chunk, state carried across chunks) | built (Refinement 4), on R-STEP and R-VSPLIT alike: per tile row c = cumsum(g) (`cumsum_tile`, fp32), D = mask * exp(-relu(c[s'] - c[s])) (c^T by `transpose_dest`, fp32), [Kn Kn^T; Qn Kn^T] by matmul_block, L = beta (*)col (D * Kn Kn^T), U by `triangle_solve_tile` (fp32 SFPU forward substitution, one per column tile), O = exp(c) (*)col Rq + (Qn Kn^T * D) x U, S = exp(c_last) S (fp32 SFPU) + Kn^T x (exp(c_last - c) (*)col U). Rows that are no token get g = beta = 0 from the writer (no decay, no write; kn = v = 0 there already). Whole-tile SFPU scope in chunked builds | `_chunked`: T >= CHUNK_MIN_TOKENS (2) on Blackhole; decode keeps the token step | 32 tokens x pair state | state touched 4x per chunk instead of 2x per token in L1; DRAM unchanged | per-chunk matrix products replace 32 x (2 Kt Vt matmuls + Kt Vt updates) |
| R-SEG: T split across cores by F4 segment pairs (P, H), combined in order | deferred: the only way to put more than B*Hv cores on a prefill call; costs a K x K factor per pair (2x state work per token) and a second pass over mixed_qkv unless segment rows stay resident; the per-segment output pass is `recurrence_block` unchanged | prefill with `B*Hv` much smaller than the grid (future) | segment of tile rows x pair | mixed_qkv up to 2 crossings; pair combine 128 KiB per pair per boundary over NoC | - |
| R-GROUP: the r value heads of a key head on one core (`pairs_per_block = r`, ordered by key head) | rejected: superseded by R-BCAST, which reaches the same 1.0 crossing of mixed_qkv while keeping B*Hv cores busy instead of B*Hk | - | r pair states | mixed_qkv 1.0 | - |
| R-BHEAD: all B sequences of a head on one core | rejected: 16 states resident at B=16 (1 MiB, 0.86 of L1) on 12 cores; superseded by R-STEP, which already avoids the B-fold tile re-read by row-granular reads | - | - | - | - |
| R-PREP: the per-tile-row preparation on the cores R-VSPLIT leaves idle, the state cores run only the chunk step's state part | built (Refinement 6), chunked R-VSPLIT only. Per (sequence, key head) group, up to `MAX_PREP_CORES_PER_GROUP` prep cores (the groups take idle cores in turn, nearest first) take its tile rows round robin (stride = prep cores of the group). A prep core reads the gate / window / z inputs of its rows and runs gates, conv (2Kt + r Vt columns), L2 norm, [Kn Kn^T; Qn Kn^T] once per key head, then per value head the chunk vectors, L, M = (I + L)^-1 (fp32 SFPU solve on the identity) and the row scalings folded into the operands: one head block [beta exp(c) (*) kn, exp(c) (*) qn \| kn^T (*) exp(c_last - c)^T \| M, QD \| exp(c) \| beta (*) v] per head. Its writer multicasts the head block into slot i % D of the head's pieces' inboxes (cb_in_kq / knt / mat / vec, row runs of consecutive cores, unicast for a one-core run) and unicasts each piece its beta v columns (cb_in_v), after every state core of the group has granted the slot; a landed counter per slot on the state core. A state core's reader grants each slot as soon as compute frees it (up to D rows ahead) and hands landed rows to compute: R = [kq] x S, U = M (beta V - R0), O = R1 + QD x U, S = exp(c_last) S + kn^T' x U (fp32 SFPU decay, no broadcasts). The state core sends o back to the row's prep core (cb_o_in), which runs the RMS norm and writes the output one row later. Two pieces per pair under R-PREP (`PREP_MAX_PIECES`) | `_prep_layout`: `PREP_OFFLOAD`, chunked, pieces > 1, >= `PREP_MIN_TILE_ROWS` (4) tile rows per sequence, an idle core per group, the key head's value heads in one gate tile column, both roles' buffers inside L1 (the inbox depth D shrinks from `INBOX_DEPTH` until they fit) | state core: Kt x Vb state, D inbox rows; prep core: one tile row of r value heads | DRAM as R-VSPLIT for the state; mixed_qkv / conv_state / conv_weight q/k columns 1 crossing per key head; adds per tile row and key head r head blocks (15 fp32 tiles, multicast to the head's pieces), r Vt fp32 v tiles and r Vt fp32 o tiles of NoC traffic | the ordered work per tile row is the state part alone (model_prefill_4096: 5.3 us per tile row of a 2-piece state core, against 50 us of the local chunk step) |
| R-PREP-SPLIT: conv/gates/norms on separate cores, y handed to the pair cores | superseded by R-PREP (Refinement 6), which moves the whole state-independent part of a tile row (including the chunk vectors and the inverse of I + L) off the ordered path, not only the conv | - | - | - | - |
| R-QKCORE: kn/qn of a key head computed once for every decode sequence of the tile row on dedicated cores, multicast to the key head's pair cores | built (Refinement 7) on decode R-STEP. Per key head a q core and a k core (2 Hk idle cores) read the key head's Kt q (k) column tiles of the current row and the conv_state tiles whole, gather the window rows (tap 3 row p = x[p], tap j row p = conv_state[3p + j]; the history rows split between the reader and the writer, stateful 32 B local NoC copies), convolve and L2-normalize rows 0..B-1 at once (multi-band SFPU scope) and multicast qn (kn) into cb_kq_rows of the key head's pair cores (`ttnn.Mcast` with an explicit sender outside the group, flag signal, one round per call). Pair cores take consecutive sequences of one value head (unit stride Hv), read and convolve only their v columns, receive once and re-push the landed rows for later units, copy the head's weights and the decode gate tile from the previous unit's slots, and run d + kn^T, the token step and the RMS norm per unit. After their send the q / k cores write conv_state_out for Ct / 2Hk column tiles each (whole tiles, rows shifted in L1) | `_qk_core_units`: `SHARE_QK_CORES`, T = 1, pieces = 1, B <= 32, 2 Hk idle cores, a v-only FPU conv batch that divides the pair's columns; units per pair core = the smallest that fits | pair core: Kt x Vt state, up to 2 units in flight (`UNIT_INPUT_DEPTH`, `STATE_OUT_DEPTH`) | mixed_qkv / conv_state q/k columns: 1 whole-tile crossing per key head (R-STEP: B r row reads); conv_state_out whole tiles once; adds 2 Hk multicasts of Kt fp32 tiles | q/k conv + L2 once per key head for all sequences (R-STEP: per pair) |
| R-STREAM-STATE: state not resident, read/written per token | rejected: violates "state crosses DRAM once in and once out" (T crossings each way) | - | - | - | - |

Regime selection (Refinement 7): R-QKCORE on top of decode R-STEP when `_qk_core_units` returns a plan (decode
B=8: 96 pair cores of one sequence each + 8 q / k cores; B=16: 96 pair cores of two sequences each). R-VSPLIT decode
keeps R-BCAST; multi-token calls and B > 32 keep R-STEP. Regime-pinned tests: `test_gated_delta_net_qkcore.py`
(predicate; q / k cores vs per-pair prep bit-identical on chip B = 5 fp32, 8, 16, 27 bf16 and small B = 32 fp32,
clean and inf padding).

Regime selection (Refinement 6): R-PREP on top of chunked R-VSPLIT when `_prep_layout` returns a plan
(model_prefill_4096: 24 state cores = 12 pairs x 2 pieces, 84 prep cores = 21 per key head, inbox depth 8);
pieces are capped at `PREP_MAX_PIECES` = 2 when that plan exists (4 pieces measured slower: more state cores, fewer
prep cores, twice the fan-out per prepped row). Calls with fewer than `PREP_MIN_TILE_ROWS` tile rows per sequence,
fp32 tokens at the chip geometry (the prep core's buffers do not fit L1) and decode keep the local regimes.
Regime-pinned tests: `test_gated_delta_net_prep.py` (predicate; forced R-PREP vs local on one-tile-row, partial,
batched and long calls, clean and inf padding; state carried over two R-PREP calls and a decode call).

Regime selection (Refinement 5): R-BCAST on top of R-VSPLIT when `_share_qk(pieces, T)` (decode B=1/2 on the
chip geometry, small-geometry decode with few pairs); prefill and R-STEP keep every core's own q/k prep.
Regime-pinned tests: `test_gated_delta_net_bcast.py` (predicate; shared vs unshared bit-identical on chip
B=1/2 bf16, 48-head fp32, Hv = Hk fp32, clean and inf padding).

Regime selection (Refinement 4): R-CHUNK replaces the token steps of every tile row when
`_chunked(T, device)` (multi-token calls on Blackhole); it composes with the R-STEP / R-VSPLIT work split
below. Regime-pinned tests: `test_gated_delta_net_chunk.py` (CHUNK_MIN_TOKENS = 2 vs token steps on
T = 2, 5, 32, 33, 100, K > V, batched boundaries inside a tile row, Hv = 33, chip T = 70; clean and inf
padding; a state carried over three chunked calls and a decode call).

Regime selection (Refinement 3): R-VSPLIT when `_value_pieces(B*Hv, Vt, grid cores) > 1` (decode B=1/2,
prefill B=1, small-geometry shapes with few pairs), else R-STEP (pairs already fill the grid, e.g. decode
B=8/16). Regime-pinned tests: `test_gated_delta_net_vsplit.py` (host predicate; forced V_BLOCK_TILES
1 / 2 / auto on decode, multi-tile-row prefill and batched prefill, clean and inf padding).

### Traffic ranking

Counted per call, fp32 Phase 0, chip geometry (Hk=4, Hv=12, K=V=128, C=2560).

| Rank | Split | DRAM crossings | Cross-core traffic | Notes |
|------|-------|----------------|--------------------|-------|
| 1 | (b, hk) groups (R-GROUP) or (b, h) + kn/qn broadcast (R-BCAST) | every tensor once; state 1 in / 1 out | R-BCAST: 2Kt fp32 tiles per tile row per key head, fan-out r-1 | cheapest; R-BCAST keeps B*Hv cores |
| 2 | (b, h) pairs, each reads its columns (R-STEP, built) | q/k columns of mixed_qkv, conv_state, conv_weight r times (1.8 crossings overall); everything else once; state minimum | none | +64 KiB at decode B=1, +33.5 MB at prefill T=4096 vs rank 1 |
| 3 | (b, h, V-slice) without broadcast (R-VSPLIT alone) | q/k columns r*Vt/v_block times (6.5 MB vs 2.4 MB token bytes at decode B=1) | o^2 partial all-gather: 1 tile per tile row per piece | more bytes; buys compute occupancy only |
| 4 | T segments (R-SEG) | mixed_qkv up to 2 | 128 KiB per pair per boundary, ordered | more bytes and an ordered chain |
| 5 | state round trip per token | state T in / T out | none | forbidden |

Chosen: rank 2 as the built stepping stone, with rank 1 (R-BCAST) its successor on the
same structure (built for decode R-VSPLIT in Refinement 5). Rank 1 in its R-GROUP form is rejected because it divides cores by r for the same
bytes R-BCAST saves.

Stall-shadow reorders (scheme properties, recorded here):

| Wait | What runs in its shadow | Permission |
|------|-------------------------|------------|
| compute waits for the writer's per-token coefficient tiles (alpha/beta expansion of the gates) | `gates_block` runs first in each tile-row prep; conv, L2 norm, transposes run while the writer expands token rows | data independence |
| compute waits for S0 (64 KiB per pair at K=V=128, the largest fetch of a decode call) | the reader issues the first tile row's prep inputs (rows, taps, gates) before S0; the whole prep runs while S0 is in flight | S is first needed by the first token step |
| the output read of token t would wait for the state write of token t (definition form) | F3: `o = alpha*ro + (qn.kn) u` is read from S_{t-1} in the same matmul as rk, so the output accumulation does not wait for `update_state_block` | algebra: alpha commutes with the contraction over K, and `qn^T S_t = alpha qn^T S_{t-1} + (qn.kn) u` (math.md F3); reorders floating-point sums, measured 4.0e-15 in float64; gated on the fp32 tolerance row |
| next pair's S0 and weights | prefetched into cb_state_in / cb_conv_w behind the current pair (depth 1 each) | pairs are independent |

Nothing is independent of the token-to-token wait inside `recurrence_block` (token t+1 reads S_t);
R-CHUNK (built, Refinement 4) removes it: one state read and one state write per tile row.

### Block schedule

Logical schedule of one core (reader, compute and writer realize their parts asynchronously):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // one pair per block
    load_pair_block(block_idx);              // S0 (Kt x Vt), conv tap tiles, A_log/dt_bias/norm_weight
    for (uint32_t row_block = first_row_block; row_block < end_row_block; ++row_block) {
        gates_block(row_block);              // alpha, beta for 32 rows of the pair's gate tile column
        conv_block(row_block);               // 3 channel groups: y = silu(sum_j tap_j * W_j)
        l2norm_block(row_block);             // kn, qn, kn^T, d = qn.kn
        recurrence_block(row_block);         // token steps lo..hi-1: reads, u/o, state update
        rmsnorm_gate_block(row_block);       // output rows = w * o * rsqrt(mean o^2 + eps) * silu(z)
        store_rows_block(row_block);         // output rows of the pair (writer)
    }
    copy_conv_state_block(block_idx);        // conv_state_out rows 3b..3b+2 of the pair's columns (reader)
    store_state_block(block_idx);            // recurrent_state_out (writer)
}
```

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------------|------------------------|--------------------|------------------------------------|
| load_pair_block | S0: Kt x Vt fp32 tiles; taps: 4 x (2Kt+Vt) tiles; A_log, dt_bias: 1 tile each; norm_weight: Vt tiles | - | one barrier per group of reads; once per pair |
| gates_block | 1 b tile + 1 a tile (tile row x gate column `h//32`) -> alpha, beta tiles | expA, dt rows (per pair) | SFPU inits once per tile row |
| conv_block | per channel group: 4 tap tiles x group columns (Kt or Vt) -> y group | conv taps (per pair) | inits once per tile row; window hand-off once per channel group |
| l2norm_block | q group (Kt), k group (Kt) of y -> kn (Kt), qn (Kt), kn^T (Kt), d (1) | - | one square/reduce/scale chain per group per tile row |
| recurrence_block | pair state Kt x Vt; per token: kn/qn rows (2 x Kt), v (Vt), kn^T (Kt) | S (whole call for the pair); kn, qn, kn^T, v, d (the tile row) | per token: one read matmul over the whole state (helper init once per token), one token-vector chain, update in DEST batches (inits once per batch); never per tile of a phase |
| rmsnorm_gate_block | O (Vt) + z (Vt) + norm_weight (Vt) -> output (Vt) | norm_weight (per pair) | once per tile row |
| store_rows_block | Vt output tiles of the tile row | - | one write barrier per tile row |
| copy_conv_state_block | 3 rows x (Vt + [2Kt if h % r == 0]) column tiles | - | once per pair |
| store_state_block | Kt x Vt fp32 tiles | - | one write barrier per pair |

Token rows of a pair inside a tile row: `lo = max(b*T, 32*row_block)`, `hi = min(b*T + T, 32*row_block + 32)`;
row blocks of a pair: `first_row_block = (b*T) // 32` to `(b*T + T - 1) // 32` inclusive. No step
runs for a row outside `[lo, hi)` (padded rows, other sequences' rows).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| update DEST batch (overlap) | Phase 0 updates Kt*Vt state tiles in batches of 3 in half-sync fp32 DEST, re-initializing copy/SFPU/matmul per batch | full-sync DEST (7 state tiles + alpha per batch), or column-major update so the next token's read of column vt starts before the update finishes |
| conv on SFPU with expanded tap tiles | 4 SFPU bcast multiplies + 3 adds per column tile and 3/4 of the weight L1 spent on expansion | FPU `mul_tiles_bcast_rows(..., bcast_row_idx=j)` on the raw conv_weight tile with DEST accumulation (`tt_metal/hw/inc/api/compute/bcast.h:192`); operands narrowed to tf32 (y feeds only output and once-entering increment) |
| prefill window assembly | reader shifts rows inside L1 for taps 0..2 (about 24 runs per tile) | read the halo as 64 B row segments directly from DRAM for every row (more transactions, no L1 shifts) |
| token_rows_per_block = 1 tile row | prep inits and reconfigs amortized over 32 tokens only | 2 tile rows per prep block (doubles prep L1) |
| tok_coef depth 2 | writer expansion latency at the first token of a tile row | depth 4 |
| core placement | `row_wise=True` spreads pair cores along the DRAM-facing axis (catalog `noc_placement`) | default column order |
| grid synchronization (decode B=16) | 192 pairs on 110 cores leaves 82 cores with 2 pairs, 28 with 1 | cap cores at 96 (2 pairs each): same makespan, fewer DRAM requesters |

Catalog entries that informed the knobs: `noc_placement` (row-wise core order), `double_buffer`
(depth 2 on streaming CBs, batched barriers), `matmul_output_subblock` (block size 2 x Vt handed
whole to the helper; fp32 DEST caps the subblock at 4), `eltwise_l1_vs_dest_accumulate` (O
accumulated by packer L1 accumulation instead of an L1 read-modify-write), `sfpu_tile_scope`
(rsqrt of a column-0 norm denominator), `width_split` (a decode call is one tile row: work is
split by pairs/columns, never by tile rows), `shared_input_reuse` (R-BCAST).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| S0 -> cb_state_in (+ alias cb_state_in_mm) | fp32 tiles | reader, TensorAccessor on recurrent_state, whole tiles at tile rows `pair*Kt .. +Kt` of the (B*Hv*K, V) view | issued after the first tile row's prep inputs (stall-shadow) |
| conv_weight -> cb_conv_w | fp32 tiles | reader reads the pair's 2Kt+Vt weight tiles (rows 0..3 as 64 B segments) and writes row j into row 0 of tap tile j (4 tap tiles per column) | rows 4..31 never read |
| A_log, dt_bias -> cb_gate_w; norm_weight -> cb_norm_w | fp32 tiles | reader: row 0 segments of tile column `h//32` (A_log, dt_bias) and of the Vt norm_weight tiles | row 0 only |
| mixed_qkv, conv_state -> cb_window | fp32 tiles | reader assembles, per channel group and tile row, tap tiles X_0..X_3 with `X_j[p] = xx[row(p) + j - 3]` for p in `[lo, hi)`: source is mixed_qkv row `R + j - 3` when `t + j - 3 >= 0`, else conv_state row `3b + t + j` (`R = 32*row_block + p`, `t = R - b*T`); rows outside `[lo, hi)` written as 0 | full tile row owned: one tile read + L1 row shifts; partial ownership (decode): 64 B row-segment reads; zero rows sanitize padding and other sequences' rows |
| b, a -> cb_gate_in; z -> cb_z | fp32 tiles | reader: tile column `h//32` of b, a and the Vt z columns of head h, tile row `row_block`; whole tile when the tile row is owned, row segments otherwise | unread elements are never consumed (writer reads (p, h%32) only; output rows outside `[lo,hi)` never written) |
| reduce scaler -> cb_reduce_scaler | bf16 | reader, `dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f)` once per kernel | constant |
| gates -> writer | fp32 tiles (alpha, beta) in cb_gates | compute packs; writer reads elements `(p, h % 32)` | face offset: `face = (p/16)*2 + (c/16)`, element `face*256 + (p%16)*16 + c%16` |
| writer -> compute per token | fp32 tiles in cb_tok_coef | writer writes column 0 (32 values) of 3 pages: `alpha_p` in every row; `beta_p` at row p, 0 elsewhere; 1 at row p, 0 elsewhere | columns 1..31 ignored by `sfpu_*_bcast_col` |
| compute internal | fp32 | CBs below; O accumulated with packer L1 accumulation | - |
| output rows -> DRAM | fp32 | writer: Vt tiles of head h at tile row `row_block`; whole tiles when the tile row is owned (no other sequence's row in it), else 64 B row segments of rows `[lo, hi)` | decode B > 1 always uses row segments |
| conv_state_out | fp32 | reader: rows `3b+n`, n < 3, source `xx[b, T+n]` (conv_state row `3b+T+n` if `T+n < 3`, else mixed_qkv row `b*T+T+n-3`), 64 B segment reads and writes, for the pair's v columns, and q/k columns when `h % r == 0` | bit-exact copy, never through compute |
| recurrent_state_out | fp32 tiles | writer drains cb_state_out, whole tiles | - |

Deferred inter-Tensix contracts (kept reachable by this structure):
- R-BCAST (built, Refinement 5, decode R-VSPLIT): the q injector is unit 0 of each (b, hk) group of
  r * pieces units, the k injector unit 1; each packs its normalized part (Kt fp32 tiles) into
  cb_qk_stage, and its writer multicasts it into cb_kq_rows ([0, Kt) kn, [Kt, 2Kt) qn) of every core of
  the group, its own included (two `ttnn.Mcast` channels on the writer NoC, flag signal, no handshake);
  every writer lands the parts it did not prep and pushes cb_kq_rows (+ its U2D alias), and compute
  derives d and kn^T (`qk_finish_block`). Receivers read and convolve only their v columns. Units map to
  cores in row-major order, so a group is consecutive cores (two rectangles on the 11-wide grid).
- R-VSPLIT (built, Refinement 3): per pair, pieces own state columns `[s*v_block, (s+1)*v_block)`;
  after the token loop of a tile row each piece reduces `sum_m o^2` over its columns into cb_osum_part,
  its writer NoC-writes that tile into slot `s` of cb_osum_gather on every piece core (same address
  everywhere), increments the peers' semaphore and waits for a cumulative count of
  `rows * (pieces - 1)`; compute adds the `pieces` slots in piece order (identical rsqrt on every
  piece) before the rsqrt of `rmsnorm_gate_block`.

Physical sharding: TARGET has no memory_layout axis, so no sharded scheme is in scope. If one is
added, the (b, h) work split is a logical height-shard of recurrent_state (independent: a placement
knob-turn); a V-shard of the state is R-VSPLIT (scheme-change); shards are consumed in place via
`ttnn.cb_descriptor_from_sharded_tensor`, not re-read over the NoC.

TARGET classification: `sequence_mode: batched_prefill` is a knob-turn here (pairs, `[lo, hi)`
and row-segment I/O already handle a sequence boundary inside a tile row; the refinement is
SUPPORTED + tests). `dtype: bfloat16` is a knob-turn on CB page formats for the token tensors
(compute stays fp32 in DEST, state stays fp32) plus the row-segment alignment issue in Key Risks.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one piece of a pair, `unit = pair * pieces + s`, `pair = b*Hv + h` (R-STEP: pieces = 1, block = PAIRS_PER_BLOCK pairs; R-VSPLIT: one unit per core) |
| Grid | `device.compute_with_storage_grid_size()`, optionally capped by host const `MAX_CORES` (default None = full grid); never a constant |
| Split | `ttnn.split_work_to_cores(grid, B*Hv, row_wise=True)` -> `(num_cores, all_cores, core_group_1, core_group_2, pairs_per_core_1, pairs_per_core_2)`; each core gets RT args `first_pair`, `num_pairs_this_core` (contiguous range) |
| Per-core work | `num_pairs_this_core` pairs; per pair `ceil`-based tile rows `(b*T)//32 .. (b*T+T-1)//32` and `hi - lo` token steps per tile row |
| Remainder | handled by the two core groups of `split_work_to_cores`; ragged token rows by `[lo, hi)`; tile counts always `ceil`: `TRn = ceil(N/32)`, conv_state tile rows `ceil(3B/32)`, `Gt = ceil(Hv/32)` |
| Cores used (INPUTS) | decode B=1 chip: 48 (R-VSPLIT, 4 pieces); decode B=8: 96, B=16: 110 (R-STEP); prefill chip B=1: 48; small geometry decode B=40: 110 (120 pairs, R-STEP) |
| L1 predicate | `footprint(Kt, Vt, depths) <= L1_BUDGET` from l1_ledger.md, checked on host; RuntimeError otherwise (no INPUTS shape reaches it) |

Unused cores get no kernels (the program's core ranges are `all_cores`).

## Circular Buffers

Page size: fp32 tile = 4096 B unless noted. `n_state = Kt*Vt`. Unpack mode "U2D" =
`UnpackToDestFp32` in the compute config's `unpack_to_dest_mode`; "std" = default (srcA/srcB).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| cb_state_in | 0 | 4096 | `n_state * STATE_IN_DEPTH` | spans K, V of one pair | Float32, U2D | reader | compute | pair: until token 0 consumed |
| cb_state_in_mm | 1 | 4096 | alias of cb_state_in (same allocation, second format descriptor) | same memory, srcB view for the read matmul | Float32, std | reader | compute | same, lockstep with cb_state_in |
| cb_state | 2 | 4096 | `n_state` | spans K, V; in-place per token | Float32, U2D | compute | compute | pair: tokens 1..T-2 |
| cb_state_mm | 3 | 4096 | alias of cb_state | srcB view | Float32, std | compute | compute | same, lockstep |
| cb_state_out | 4 | 4096 | `n_state * STATE_OUT_DEPTH` | spans K, V | Float32, std | compute | writer | from last token to drained |
| cb_conv_w | 5 | 4096 | `4 * (2*Kt + Vt)` | spans the pair's columns x taps; constant along t | Float32, U2D | reader | compute | pair |
| cb_window | 6 | 4096 | `4 * max(Kt, Vt) * WINDOW_DEPTH` | spans one channel group x taps; streams over groups and tile rows | Float32, U2D | reader | compute | per channel group |
| cb_y_qk | 7 | 4096 | `2 * Kt` | spans q, k groups of one tile row | Float32, U2D | compute | compute | conv -> l2norm |
| cb_v | 8 | 4096 | `Vt` | spans v group of one tile row | Float32, U2D | compute | compute | conv -> end of recurrence_block |
| cb_sq | 9 | 4096 | `max(Kt, Vt)` | squares of one group (q, k, then o) | Float32, U2D | compute | compute | inside l2norm / rmsnorm |
| cb_rinv | 10 | 4096 | 2 | column-0 rsqrt of q and k (then of o) | Float32, U2D | compute | compute | inside l2norm / rmsnorm |
| cb_kq_rows | 11 | 4096 | `2 * Kt` | kn row (Kt) + qn row (Kt): in0 of the read matmul for the tile row | Float32, std | compute | compute | tile row |
| cb_knT | 12 | 4096 | `Kt` | kn^T (K on rows, tokens on columns): in0 of the increment matmul | Float32, std | compute | compute | tile row |
| cb_d | 13 | 4096 | 1 | column 0 = qn . kn per token row | Float32, U2D | compute | compute | tile row |
| cb_gate_in | 14 | 4096 | `2 * ROW_DEPTH` | b, a tiles of the gate column | Float32, U2D | reader | compute | tile row |
| cb_gate_w | 15 | 4096 | 2 | A_log, dt_bias tiles (row 0) of the gate column; exp(A) computed per tile row in gates_block | Float32, U2D | reader | compute | pair |
| cb_gates | 16 | 4096 | 2 | alpha, beta tiles of one tile row | Float32, std | compute | writer | tile row |
| cb_tok_coef | 17 | 4096 | `3 * TOK_COEF_DEPTH` | per token: alpha column, beta-select column, select column | Float32, U2D | writer | compute | token |
| cb_reads | 18 | 4096 | `2 * Vt` | rk (Vt) + ro (Vt) of one token | Float32, U2D | compute | compute | token |
| cb_u | 19 | 4096 | `Vt` | u (row p only nonzero): in1 of the increment matmul | Float32, std | compute | compute | token |
| cb_o_acc | 20 | 4096 | `Vt` | O of the tile row, one row per token by packer L1 accumulation | Float32, U2D | compute | compute | tile row |
| cb_z | 21 | 4096 | `Vt * ROW_DEPTH` | z columns of head h | Float32, U2D | reader | compute | tile row |
| cb_norm_w | 22 | 4096 | `Vt` | norm_weight row 0 | Float32, U2D | reader | compute | pair |
| cb_out_rows | 23 | 4096 | `Vt * ROW_DEPTH` | output tiles of the tile row | Float32 (output dtype), std | compute | writer | tile row |
| cb_reduce_scaler | 24 | 2048 | 1 | SUM scaler 1.0 | Float16_b, std | reader | compute | kernel |

Alias credit protocol (both alias pairs): the producer reserves and pushes both indices together
with equal counts; the consumer waits on both and pops both together. For cb_state/cb_state_mm per
token: wait both (`n_state`) -> read matmul via cb_state_mm -> token-vector -> pop both ->
reserve both -> update packs into cb_state slots -> push both. The next token's read matmul is
ordered after the update's packs by the cb_state_mm push (real credit; Rule 3.2 of
l1-footprint-discipline.md). Token 0 reads cb_state_in/_mm and packs into cb_state (or
cb_state_out when it is also the last token); the last token packs into cb_state_out.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | load_pair_block | S0 Kt x Vt; taps 4 x (2Kt+Vt); gate_w 2; norm_w Vt | dataflow (reader) | DRAM | cb_state_in + cb_state_in_mm (n_state), cb_conv_w, cb_gate_w, cb_norm_w | resident for the pair |
| 2 | gates_block | 1 tile (32 rows x gate column) | helper (eltwise_chain) + raw SFPU bcast row | cb_gate_in (b, a: 2, popped), cb_gate_w (2, kept) | cb_gates (2) | writer owns cb_gates |
| 3 | conv_block | per group: 4 taps x group columns | custom (SFPU bcast-row chain) | cb_window (4*group, popped per group), cb_conv_w (kept) | cb_y_qk (2Kt), cb_v (Vt) | cb_v kept to end of tile row |
| 4 | l2norm_block | Kt columns, twice | helper (eltwise square, reduce Accurate + PostReduceOp) + raw SFPU bcast col + raw transpose | cb_y_qk (2Kt, popped) | cb_kq_rows (2Kt), cb_knT (Kt), cb_d (1) | kept for the tile row |
| 5a | recurrence_block: read | in0 2 x Kt, in1 Kt x Vt, out 2 x Vt | helper (matmul_block) | cb_kq_rows (kept), cb_state_mm or cb_state_in_mm (kept per alias protocol) | cb_reads (2Vt) | - |
| 5b | recurrence_block: token vector | Vt tiles | custom (SFPU bcast-col chain) | cb_reads (popped), cb_v (kept), cb_tok_coef (3, popped), cb_d (kept) | cb_u (Vt), cb_o_acc (L1 acc, reserved for the tile row) | - |
| 5c | recurrence_block: state update | Kt x Vt tiles in DEST batches | custom (U2D copy + SFPU bcast-col + matmul_tiles accumulate) | cb_state / cb_state_in (alias protocol), cb_tok_coef alpha page, cb_knT (kept), cb_u (popped) | cb_state or cb_state_out | after last token: cb_o_acc pushed (Vt) |
| 6 | rmsnorm_gate_block | Vt tiles | helper (eltwise square, reduce Accurate + PostReduceOp, eltwise chain) + raw SFPU bcast col/row | cb_o_acc (Vt, popped), cb_z (Vt, popped), cb_norm_w (kept) | cb_out_rows (Vt) | - |
| 7 | store_rows_block | Vt tiles | dataflow (writer) | cb_out_rows (popped) | DRAM output | - |
| 8 | copy_conv_state_block | 3 rows x pair columns | dataflow (reader) | DRAM | DRAM conv_state_out | - |
| 9 | store_state_block | Kt x Vt | dataflow (writer) | cb_state_out (popped) | DRAM recurrent_state_out | - |
| - | per-token coefficient expansion (writer side of 5b) | 3 tiles per token | dataflow (writer) | cb_gates (kept for the tile row) | cb_tok_coef (3) | cb_gates popped after the tile row |

Math of each compute op (all fp32 in DEST unless stated):

| Op | Statement |
|----|-----------|
| gates_block | `beta = sigmoid(b)`; `alpha = exp(-exp(A_row) * softplus(a + dt_row))`, `+ dt_row` and `* (-exp(A_row))` as `sfpu_add_bcast_row` / `sfpu_mul_bcast_row` (row 0 of the A_log / dt_bias tile); softplus with beta 1, threshold 20 (identity strictly above 20) |
| conv_block | per column tile: `y = silu(sum_j X_j (*)row Wtap_j)`; silu = `x * sigmoid(x)` |
| l2norm_block | `ssk = rowsum(k^2)` (reduce REDUCE_ROW SUM, Accurate) with PostReduceOp `rsqrt(ssk + 1e-6)` -> cb_rinv; `kn = k (*)col rinv_k`; q likewise with PostReduceOp additionally `* K^-0.5`; `kn^T = transpose_wh(kn)` per tile; `d = rowsum(qn * kn)` (operands read std from cb_kq_rows: affects output only) |
| read (5a) | `[rk; ro] = [kn; qn] x S` (S via srcB: allowed narrow operand of the sums over K) |
| token vector (5b) | `u = beta_sel (*)col (v - alpha (*)col rk)`; `o = sel (*)col (alpha (*)col ro) + d (*)col u`; pack u -> cb_u; pack o -> cb_o_acc with packer L1 accumulation (accumulation off for the first token of the tile row) |
| update (5c) | per state tile `(kt, vt)`: DEST <- S (U2D copy); `DEST (*)col= alpha`; `DEST += matmul_tiles(knT[kt], u[vt])` (matmul accumulates onto DEST, `tt_metal/hw/inc/api/compute/matmul.h:151`); pack |
| rmsnorm_gate_block | `ms = rowsum(o^2)` with PostReduceOp `rsqrt(ms * (1/V) + eps)`; `out = ((o (*)col rinv) (*)row w) * z * sigmoid(z)` packed into cb_out_rows |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all compute | raw_api | `compute_kernel_hw_startup` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:100-106` (SrcOrder::Reverse for matmul), `reduce_helpers_compute.hpp:33-34` | once at the top of the compute kernel, before any helper | - | - | - |
| gates_block | helper | `eltwise_chain` with `Sigmoid`, `Exp`, `Softplus`, `Negative`, `CopyTile`, `PackTile` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:619-620`; `eltwise/unary/activations.hpp:24,56`; `eltwise/unary/activations.inl:132-141`; `eltwise/unary/math.hpp:22`; `eltwise/unary/misc.hpp:22` | `IterationShape::one_tile()`; Softplus(beta=1.0f bits, beta_recip=1.0f bits, threshold=20.0f bits) | cb_gate_in, cb_gate_w | cb_gates | shape = 1 tile per gate column (fixed by Hv layout) |
| gates_block (row bcast) | raw_api | `sfpu_add_bcast_row`, `sfpu_mul_bcast_row`, `sfpu_bcast_row_init` | `tt_metal/hw/inc/api/compute/sfpu_binary_bcast.h:298, 337-372` | DEST slots | cb_gate_in, cb_gate_w | (in DEST) | - |
| conv_block | raw_api | `copy_tile` (U2D), `sfpu_mul_bcast_row`, `add_binary_tile`, `sigmoid_tile`, `mul_binary_tile` | `sfpu_binary_bcast.h:361`; `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h:68,103`; `compute_kernel_api.h:164` | group = Kt or Vt columns | cb_window, cb_conv_w | cb_y_qk, cb_v | `conv_group_tiles` |
| l2norm_block (square) | helper | `square<cb_y_qk, cb_sq>(n)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:80` | n = Kt | cb_y_qk | cb_sq | Kt (whole K) |
| l2norm_block / rmsnorm (reduce) | helper | `reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sq, cb_reduce_scaler, cb_rinv, ReduceInputPolicy::BulkWaitBulkPop, ..., ReduceFp32Mode::Accurate, ..., PostReduceOp>` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:654-681`, PostReduceOp `509-552`, Accurate `reduce_helpers_common.hpp:18` | `ReduceInputBlockShape::row(Kt)` (or `row(Vt)`); PostReduceOp: AddUnary/MulUnary + Rsqrt on dst | cb_sq, cb_reduce_scaler | cb_rinv | block width Kt / Vt |
| scaler (reader) | helper | `prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:64-71` | - | - | cb_reduce_scaler | - |
| l2norm_block (scale) | raw_api | `sfpu_mul_bcast_col`, `sfpu_bcast_col_init` | `sfpu_binary_bcast.h:36,107` | rinv in DEST column 0 | cb_y_qk, cb_rinv | cb_kq_rows | - |
| l2norm_block (kn^T) | raw_api | `transpose_wh_init`, `transpose_wh_tile` | `tt_metal/hw/inc/api/compute/transpose.h:37,120` | per kn tile | cb_kq_rows | cb_knT | Kt |
| l2norm_block (d) | helper | `mul<cb_kq_rows, cb_kq_rows, cb_sq>` (offset operands) then `reduce` as above | `convenience.hpp:70` | n = Kt | cb_kq_rows | cb_sq -> cb_d | Kt |
| read (5a) | helper | `matmul_block<false, false, LastBlockTarget::Out, OutputCBLayout::TileRowMajor, InitMode::Short, InputPolicy::WaitAndRetainOnLastBlock, InputPolicy::NoWaitNoPop>` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:339-371`, policies `:77`, shape `:160-169` | `MatmulBlockShape::of(...)` with `in0_block_k = Kt`, `num_k_blocks = 1`, output block 2 x Vt; subblock factors from the tuner (fp32 DEST: <= 4 tiles) | cb_kq_rows, cb_state_mm / cb_state_in_mm | cb_reads | output block size 2 x Vt (block knob = `v_block_tiles`) |
| token vector (5b) | raw_api | `copy_tile` (U2D), `sfpu_mul_bcast_col`, `sfpu_add_bcast_col`... `sub_binary_tile`, `add_binary_tile`, `pack_tile` + `pack_reconfig_l1_acc` | `sfpu_binary_bcast.h:94,107`; `eltwise_binary_sfpu.h:103,131` | Vt tiles | cb_reads, cb_v, cb_tok_coef, cb_d | cb_u, cb_o_acc | Vt |
| update (5c) | raw_api | `copy_tile` (U2D), `sfpu_mul_bcast_col`, `matmul_tiles`, `pack_tile` | `sfpu_binary_bcast.h:107`; `tt_metal/hw/inc/api/compute/matmul.h:107,151,166` | batch = `min(DEST_AUTO_LIMIT - 1, Kt*Vt)` | cb_state(_in), cb_tok_coef, cb_knT, cb_u | cb_state / cb_state_out | Kt x Vt (whole state) |
| rmsnorm_gate_block | helper + raw_api | `square`, `reduce` (as above), `sfpu_mul_bcast_col`, `sfpu_mul_bcast_row`, `eltwise_chain` with `Sigmoid` and SFPU `MulBinary` (no `BinaryFpu`) | `convenience.hpp:80`; `sfpu_binary_bcast.h:107,361`; `eltwise/binary/sfpu/basic.hpp:20-30` | Vt | cb_o_acc, cb_z, cb_norm_w, cb_rinv | cb_out_rows | Vt |
| reader / writer | raw_api | `TensorAccessor`, `noc_async_read`, `noc_async_write`, barriers | `tech_reports/tensor_accessor/tensor_accessor.md` | `ttnn.TensorAccessorArgs(t).get_compile_time_args()` per tensor | - | - | - |

Helpers considered and rejected:

| Raw entry | Candidate helper | Mismatch (file:line) | Reason |
|-----------|------------------|----------------------|--------|
| gates / conv / l2norm / rmsnorm row and column broadcast | `eltwise_chain` `BinaryFpu` with `BroadcastDim::Row/Col` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:330-335, 358` (broadcast applies to the srcB operand of `BinaryFpu` only) | FPU operands are 19-bit (`cunpack_common.h:314-318`): alpha's inputs (a + dt_bias decides the softplus switch), the L2 scale (R7 bias enters the state) must stay fp32; the chain has no SFPU-broadcast element |
| conv_block silu | `SiluActivation` | `ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp:44` | packer-thread activation for matmul/bias outputs only; no chain element named Silu exists, so silu is composed as `x * sigmoid(x)` |
| conv_block taps | `eltwise_chain` Row broadcast | `chain.hpp:259` re-reads tile row 0 (no row index) and is FPU | tap j is row j of conv_weight; the reader moves it to row 0 and the SFPU row broadcast keeps fp32 |
| token vector, update | `matmul_block` | `matmul_block_helpers.hpp:339-371`: the helper owns DEST acquire/pack; no entry seeds DEST with a decayed state before the product | the update must accumulate the increment onto an SFPU-decayed fp32 state held in DEST (precision contract); custom block op over the whole state, inits once per DEST batch |
| update | `bias_add_helpers` `add_bias_bcast_rows` | `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp` (consumes `LastBlockTarget::Interm` partials, FPU broadcast) | FPU add would pass S through srcA (19-bit) |
| kn^T | matmul `transpose` flag | `matmul_block_helpers.hpp:277` transposes in1 tiles only; `matmul.h:208,258` | the increment needs in0 = kn^T; transposed once per tile row and reused by 32 tokens |
| O accumulation | `reduce` Accumulate | `reduce_helpers_compute.hpp:427-470` | accumulates reductions, not row inserts; packer L1 accumulation of the select-masked o row is the catalog's `eltwise_l1_vs_dest_accumulate` pattern |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| gates | a + dt_bias | cb_gate_in: rows [lo,hi) x col h%32 used | cb_gate_w (dt_bias): Row0 | Row (SFPU) |
| gates | softplus * (-exp(A_log)) | All (computed) | cb_gate_w (A_log): Row0 | Row (SFPU) |
| conv | X_j * Wtap_j | cb_window: rows [lo,hi) (others 0) | cb_conv_w tap j: Row0 | Row (SFPU) |
| l2norm | k * rinv_k, q * rinv_q | cb_y_qk: All | cb_rinv: Col0 | Col (SFPU) |
| token vector | alpha * rk, alpha * ro | cb_reads: All (row p used) | cb_tok_coef alpha page: Col0 | Col (SFPU) |
| token vector | beta_sel * (v - alpha rk) | cb_v / DEST: All | cb_tok_coef beta-select page: Col0 | Col (SFPU) |
| token vector | sel * (alpha ro), d * u | DEST: All | cb_tok_coef select page / cb_d: Col0 | Col (SFPU) |
| update | S * alpha | cb_state(_in): All | cb_tok_coef alpha page: Col0 | Col (SFPU) |
| update | knT x u (matmul) | cb_knT: All (col p meaningful) | cb_u: row p nonzero, others exactly 0 | none (matmul) |
| rmsnorm | o * rinv | cb_o_acc: rows [lo,hi) | cb_rinv: Col0 | Col (SFPU) |
| rmsnorm | * norm_weight | All | cb_norm_w: Row0 | Row (SFPU) |
| rmsnorm | * silu(z) | All | cb_z: All | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| State narrowed (R1) | any FPU touch of S or alpha truncates to 19 bits every token; state_carry cases grade at 0.002 relative RMS | S only via U2D copy + SFPU decay; increment by matmul accumulation onto DEST; alpha only in U2D CBs; S and alpha never srcA/srcB except S as the read operand (allowed). Verify on device that matmul accumulation onto a pre-loaded fp32 DEST keeps fp32 (fallback: pack the increment to a U2D CB and `add_binary_tile`) |
| Alias race | cb_state_mm read by the next token's matmul without credit sees pre-update data | alias credit protocol (Circular Buffers): pop/reserve/push both indices every token |
| Poison padding (inf) | `0 * inf = nan`: the increment matmul multiplies kn^T columns of other rows by u rows that are 0 | reader writes 0 into every window row outside `[lo, hi)`, so y, kn, qn, v, rk of those rows are finite (0 or derived from 0); u rows != p are exactly 0 by beta-select; writer reads gate elements only at (p, h%32); rows outside `[lo,hi)` never written to a shared output tile |
| Padded rows as tokens | a whole-tile step would write the state with padded rows | steps run only for `lo <= row < hi` |
| R7 one-signed sums | FPU reduce of q^2, k^2, o^2 carries a -0.1 percent bias that scales kn (enters the state); `closed_form_scale` catches it | square on SFPU (U2D input), reduce in `ReduceFp32Mode::Accurate` |
| Softplus fp32 arm (R6) | the float32 arm of `softplus_tile` is compiled only with `INP_FLOAT32` (`ckernel_sfpu_softplus.h:181`) | compute kernel defines `INP_FLOAT32`; if the arm is unavailable compose `log1p(exp(min(u, 20)))` with `UnaryGt`/`Where` (identity strictly above 20) |
| exp re-init | `exp_tile` must be re-initialized after any other SFPU init (`exp.h:61-62`) | gates_block issues each element's init per chain (eltwise_chain owns per-element init); raw sequences re-init after switching families |
| Window across tile rows and conv_state seam (R3) | rows t < 3 read conv_state rows `3b + t + j`; rows 0..2 of a tile row read the previous tile row; decode rows read conv_state only | one gather rule (Dataflow Strategy) for every case; INPUTS T = 2, 3, 31, 32, 33 and decode B = 3, 16, 40 exercise it |
| conv_state_out for T < 3 | the last 3 rows of `[hist; x]` include history rows | source rule `T+n < 3 -> conv_state row 3b+T+n` (Dataflow Strategy), bit-exact DM copy |
| Shared output / conv_state_out tiles | decode pairs of different b write rows of the same tiles from different cores | row-segment writes (64 B fp32 face rows) of owned rows only; whole tiles only when the tile row is owned |
| bf16 refinement: 32 B row segments | bf16 face rows are 32 B; a sub-64 B DRAM write may read-modify-write a neighbour's row | the dtype refinement must make one core own each shared tile row for writes (gather rows over NoC to the owner) or pad transfers to an owner; Phase 0 is fp32 only |
| Orientation (R8) | K on state rows, V on columns; kn arrives as a row | read uses kn as in0 rows; increment uses kn^T as in0 with u (row p) as in1; INPUTS with K != V |
| DEST budget (R10) | fp32 half-sync holds 4 tiles; update needs alpha + state tiles | batches of `DEST_AUTO_LIMIT - 1` state tiles + alpha |
| Per-token writer dependency | compute stalls if the writer's coefficient expansion lags | gates_block first in the tile row, `TOK_COEF_DEPTH = 2` |
| Long prefill on 12 cores (R2) | 4096 ordered steps per pair; R-STEP only | resolved by R-CHUNK (Refinement 4, 128 chunk steps per pair) on R-VSPLIT's 48 cores; R-SEG / R-BCAST remain the deferred speedups |
| Hv > 32 | gate columns span `Gt` tiles (INPUTS Hv = 33, 48) | gate tile column `h // 32`, element `h % 32` |
