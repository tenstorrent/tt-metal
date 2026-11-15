# Operation Requirements: gated_delta_net

## Definition
- **Formula**: per sequence b and value head h, over tokens t: causal 4-tap depthwise conv + SiLU over
  `[conv_state; mixed_qkv]`; `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)`;
  `qn = l2norm(q) * K^-1/2`, `kn = l2norm(k)` (eps 1e-6); `S_t = exp(g) S_{t-1} + kn (beta (v - kn^T exp(g) S_{t-1}))^T`;
  `o_t = qn^T S_t`; `output = rmsnorm(o; norm_weight, epsilon) * silu(z)`; returns
  `(output, conv_state_out = last 3 rows of [hist; x], recurrent_state_out = S_T)`.
- **PyTorch Reference**: `reference()` in `tests/ttnn/unit_tests/operations/gated_delta_net/test_gated_delta_net.py`.
- **Import Path**: `from ttnn.operations.gated_delta_net import gated_delta_net`
- **Function Signature**: `gated_delta_net(mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight, *, epsilon=1e-6, compute_kernel_config=None) -> (output, conv_state_out, recurrent_state_out)`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32] (recurrent_state float32 in every cell)
- **SUPPORTED layout**: [TILE]
- **SUPPORTED fp32_dest_acc_en**: [True] (TARGET is True only)
- **SUPPORTED shape-derived axes**: alignment in {tile_aligned, h_non_aligned}; sequence_mode in {decode, prefill, batched_prefill} (batched_prefill added by the verifier on probe evidence)
- **Cores**: (b, h) pairs over the full grid, `split_work_to_cores(row_wise=True)`; regime R-STEP
- **Compute config**: caller's `compute_kernel_config` passed through (default HiFi4, fp32 DEST, no approx); per-CB `UnpackToDestFp32` set by the descriptor
- **Golden baseline**: 64 supported_pass, 35 xfail_expected (all bfloat16), 0 loud

### [x] Refinement 1 — bfloat16 token tensors (lands the perf_decode_b1 contract)

**Goal**: add `ttnn.bfloat16` to `SUPPORTED["dtype"]`: mixed_qkv, z, b, a, conv_state and the four
weights in bfloat16, output and conv_state_out in bfloat16, recurrent_state and every carried / DEST
value staying float32. Input CBs (window, gate_in, z, conv_w, gate_w, norm_w), the writer-facing
cb_out_rows and the reader scratch take the tensor dtype; intermediate compute CBs stay Float32 with
`UnpackToDestFp32` where they feed copy/SFPU. Moves all 35 bf16 xfail cells to passing, including
the LOOSE_CASES `perf_decode_b1`, `model_decode_b8`, `model_decode_b16`, `model_prefill_4096`,
`model_prefill_then_decode`, `state_carry_bf16` (state graded at 0.002 relative RMS).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- This is the only TARGET gap left and the only knob separating Phase 0 from the flagged perf case
  (bf16, TILE, DRAM interleaved, HiFi4, `fp32_dest_acc_en=True`), so it goes first; Refinements 2
  and 3 measure exactly that cell.
- Kernels hard-code fp32 geometry (`TILE_BYTES = 4096`, `SEG_BYTES = 64`, `TILE_WORDS`, the writer's
  `elem_index` word reads of cb_gates). Derive tile/segment bytes from the dtype on the host once and
  pass them as compile-time args; cb_gates stays Float32 (compute-packed), so the writer's word reads
  keep working.
- Row-granular I/O: a bf16 face row is 32 B. Decode B > 1 and batched_prefill write rows of the same
  output / conv_state_out tile from different cores; a sub-64 B DRAM write can clobber a neighbour's
  row (op_design.md Key Risks, "bf16 refinement: 32 B row segments"). The fix must be in-kernel
  (one owner per shared tile row, or alignment-safe transfers); it must not serialize the decode B=1
  path, which has no shared tile rows. Reads of 32 B segments from DRAM need the same alignment check.
- conv_state_out stays a bit-exact copy (bf16 rows in, bf16 rows out, never through compute).
- Keep the Phase 0 performance bar: pairs still spread over the grid, batched barriers per block.
  Do not add `ttnn.typecast` or any other ttnn op on inputs/outputs (prompt rule).

**Done when**: every `dtype=bfloat16` golden cell and bf16 LOOSE_CASE passes; float32 cells unchanged;
`verify_supported` shows 0 loud categories.

### [x] Refinement 2 — Cut the per-pair fixed cost of perf_decode_b1

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` (`perf_decode_b1`, flagged PERF FOCUS: Hk=4, Hv=12,
K=V=128, B=1, T=1, bf16 token tensors, fp32 state, DRAM interleaved, HiFi4,
`fp32_dest_acc_en=True`) has a goal of 5.3 us device kernel time (bound 4.8 us); Phase 0 runs the
fp32 equivalent in ~106 us on 12 cores. Comparing decode with the per-token cost of long prefill puts
roughly 80 us in per-pair prep (reader window assembly, gates, conv, L2 norm, RMS norm) and ~20 us
in the single token step. Profile the reader / compute / writer zones on this case first, then
attack the dominant region with the relevant patterns in
`ttnn/ttnn/operations/examples/master.md`. Candidate levers (the implementer picks):
- `sfpu_tile_scope`: in decode each tile holds one valid row (`[lo, hi)`), yet gates, conv, SiLU,
  squares, rsqrt and the norm chain run all 32 SFPU vectors per tile; scope them to the valid row
  group(s).
- Reader window hand-off: today each column tile is zero-filled, read and row-shifted behind its own
  barriers and handed off alone; express it per channel group (the planned conv block) with one
  barrier per group (`double_buffer`, `read_coalescing`). Same for the per-column read+write barrier
  pairs of the conv_state_out copy.
- Conv on the FPU (`mul_tiles_bcast_rows` with DEST accumulation on the raw conv_weight tile, design
  perf lamp), which also removes the 4x tap expansion of cb_conv_w.
- State update batches: full-sync DEST (`dst_full_sync_en`, 8 fp32 tiles per batch instead of 3) and
  fewer copy/SFPU/matmul re-inits per batch (`compute_block_size`, `compute_fusion`).
No SUPPORTED change.

**Measured on this op at fdff462 (operator, device kernel duration, median of 11 launches, bf16 config of perf_decode_b1)**:
101.5 us on 12 cores; the DRAM bank bound for the bytes of this call is 4.8 us, so DRAM is 5% used and is
not what limits the kernel. Per-core zones: the reader (NCRISC) spans 39 us and issues 9,324 NoC calls
per launch (705 to 849 per core); the trace counts about 2.46 MB read where the inputs hold 1.39 MB, so
about 1.07 MB is read more than once. The compute kernel waits 35 us for its first input
(cb_wait_front), then runs about 62 us with the vector engine busy 43% of the window and the matrix
engine 3%. So two regions, in this order of size: (1) the compute chain after the first input, about 62 us,
vector-engine work on full tiles for one valid row; (2) the reader's window assembly, about 35 to 39 us,
many small reads behind many barriers and bytes read twice. Captures: generated/chip_runs/gdn_v00_d1_*.
Decode B=8 and B=16 (126.6 us and 262.8 us): the fastest core takes 100 us as at B=1 and the slow cores
wait on input (42 to 44% of compute time in cb_wait_front), not on more work: the reader side scales
badly with the number of pairs.

**Done when**: measured device-ns on `perf_decode_b1` (exactly its bf16 config) improves and moves
toward the 5.3 us goal, its precision gates (output PCC 0.995 / rel RMS 0.04, state PCC 0.999 / rel
RMS 0.01) still hold, the golden suite is green, and there is no regression across the
config-spanning guard set (one representative per distinct kernel path: decode B=1, decode B>1 with
shared tile rows, prefill single tile row, prefill multi tile row, batched_prefill; float32 and
bfloat16).

**Outcome**: perf_decode_b1 (bf16 config) 101.6 us -> 42.1 us device kernel time (2.4x, 12 cores);
guard set all faster (decode B=8 127.8 -> 87.3, B=16 264.5 -> 172.0, fp32 decode B=1 105.0 -> 62.6,
bf16 prefill T=96 2150 -> 1714, batched B=2 T=40 1016 -> 833 us). Levers that landed: FPU conv for
bf16 tokens (4 row-broadcast ELWMULs accumulated in one DEST tile on raw conv_weight tiles, no tap
expansion); band-scoped SFPU (the 4-row band holding the token rows: silu, gates chain, squares,
rsqrt post-op, column/row broadcasts, binaries); scalar alpha (read_tile_value) for the state decay
and the u term, which also lets the update batch fill DEST (4 vs 3 tiles); one unpack init per format
run instead of per CB; reader window assembly per window group (gcd(Kt, Vt) columns) with one flush
per group, the window zeroed once by the idle writer and re-zeroed only when a slot's dirty rows are
not covered; conv_state_out copy batched 10 columns per flush. Parked as knobs: DST_FULL_SYNC (measured
54.5 vs 53.2 us, loses math/pack overlap) and CONV_ON_FPU ("auto" = bf16 only; fp32 operands would be
truncated to 19 bits). Measured null and reverted: RISC word copies of staged rows (49.0 vs 43.1 us,
NCRISC L1 loads cost more than a NoC copy + barrier). Where the time is now (per-core zones, B=1):
reader window assembly ~10 us before the conv can finish (32 staged 64 B row reads + L1 shifts per
group, round-trip bound), then a serial compute chain of L2 norm ~7 us, token step ~12 us (u/o windows
5.4, state update 4.7, read matmul 1.7), RMS norm ~5 us, writer state drain ~2.6 us. Next levers:
spread each pair over cores (Refinement 3, R-VSPLIT, 98 of 110 cores idle at B=1), coalesce the
history-row reads (3 contiguous rows per face in one aligned read), and fuse the u and o windows.
Not done here: R-VSPLIT is its own queued refinement, and the remaining per-window costs are now
fixed init/unpack overheads that a cross-core split divides directly.

### [x] Refinement 3 — R-VSPLIT: spread each pair's state columns across cores for perf_decode_b1

**Type**: perf

**Goal**: same flagged target (`perf_decode_b1`, goal 5.3 us). Decode B=1 has only B*Hv = 12 pairs,
so R-STEP leaves ~98 of 110 cores idle and each busy core moves a whole 64 KiB state in and out
plus every token-step tile. Build the deferred R-VSPLIT regime from op_design.md: per pair, split
the state columns into `Vt / V_BLOCK_TILES` pieces on separate cores (`V_BLOCK_TILES` is already the
host knob; today it asserts `== Vt`), each piece running the per-pair loop on its column slice, and
combine the M13 partial `sum o^2` (one column-0 fp32 tile per tile row per piece) across the pieces
by semaphore-gated unicast or the multicast helper (`ttnn.Mcast` + `McastArgs`) before the rsqrt.
Without kn/qn sharing every piece re-reads and re-preps the pair's q/k columns; decide by measurement
whether to layer R-BCAST (kn/qn of a key head computed once, multicast to its consumers;
`shared_input_reuse`) or to split the q/k prep itself, since prep is replicated per piece. Relevant
catalog entries: `width_split`, `shared_input_reuse`, `mcast_topology`, `tensix_all_reduce`. No
SUPPORTED change.

**Verifier notes**: scheme-change (dependent M13 axis combined across cores), so it stands alone and
follows Refinement 2: Refinement 2 shrinks the per-piece prep that this regime replicates. The
regime must be selected by a host predicate (R-STEP stays the path for shapes where pairs already
fill the grid, e.g. decode B=16) with regime-pinned tests for both. The unused cb_state at T=1
(L1 audit) can be dropped here when the state buffers are resized to the column slice.

**Done when**: measured device-ns on `perf_decode_b1` improves over the Refinement 2 result and
moves toward 5.3 us, its precision gates hold, the golden suite is green, and no regression across
the config-spanning guard set (as in Refinement 2, plus one decode B=16 representative on R-STEP).

**Outcome**: perf_decode_b1 (bf16) 42.1 -> 33.9 us device kernel time on 48 cores (R-VSPLIT, 4 pieces,
Vb = 1); guard set all faster: decode B=8 87.3 -> 74.9, B=16 172.0 -> 150.4 (both R-STEP, gaining from the
shared reader changes), fp32 decode B=1 62.6 -> 52.6, bf16 prefill T=96 1714 -> 589 (48 cores),
batched B=2 T=40 833 -> 367 us (96 cores). Built: host predicate `_value_pieces` (pieces = largest divisor
of Vt <= 4 with one unit per core), the o^2 partial all-gather (unicast into the same slot on every piece
core + cumulative semaphore, two tile rows of slots), per-piece state / z / output / norm_weight slices.
Levers added to make the split pay off: aligned row-span reads of conv_weight taps, A_log / dt_bias /
norm_weight row 0 and gate token rows; the weights, gates and first window group completed by one DRAM
flush; one window group per unit in R-VSPLIT (was gcd(Kt, Vb) = 1, i.e. 9 flushes); history rows read
as one aligned span per column; window zeroed slot by slot; FPU conv and kn transposes batched per DEST
window. Measured: pieces = 2 is slower (36.2 vs 34.5 us decode, 940 vs 633 us prefill T=96, with
zones); row-granular *staged* weight reads were slower than whole tiles (request count, not bytes,
binds). Where the time is now (per-core zones, B=1): the reader's single flush completes ~11 us in,
bound by DRAM request rate (48 cores x ~70 small requests; ~80 ns per issued read), dominated by the
q/k window columns every piece re-reads (8 of 9 columns, 12 cores per key head); then a serial compute
chain of conv 4.1, L2 norm 8.2 (inverse norms 4.0, scale 2.3, d 1.0, transposes 1.0), token step 2.9,
RMS norm + combine 3.9, writer drain 2.7 us. Next: share the q/k prep (R-BCAST, or split the q/k conv
and L2 partial sums across the pieces of a pair and all-gather kn/qn), which removes 8/9 of the reader
requests and the replicated 12 us of conv + L2 norm. Not done here: it is a second cross-core topology
(kn/qn tile rows, 3Kt fp32 tiles per tile row per pair) on top of this one.

### [x] Refinement 4 — Chunked recurrence for prefill (model_prefill_4096)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[3]` (`model_prefill_4096`: Hk=4, Hv=12, K=V=128, B=1, T=4096, bf16
token tensors, fp32 state, DRAM interleaved, HiFi4, `fp32_dest_acc_en=True`) is the shape in which the
model ingests every prompt. Measured on this op at fdff462 (operator, device kernel duration, median of
11 launches): 89.63 ms on 12 cores, 21.9 us per token step, 4096 ordered token steps per pair; compute
bound (vector engine 35% of the window, matrix engine 10%), reader not limiting. Build the deferred
R-CHUNK regime of op_design.md: the closed form over a chunk of up to 32 tokens (one tile row) with the
float32 state carried across chunks, so that a tile row costs a few matrix products on Kt x Vt state
tiles instead of 32 token steps (math.md of the exploration bundle: reformulations F1 and F5 and their
licences; the oracle holds one function per form). The carried state and exp(g) stay float32 at every
chunk boundary (spec rules); inside a chunk, narrow operands of the products are allowed as the spec
states. If the measurement after R-CHUNK shows cores idle, consider the deferred R-VSPLIT / R-SEG on
top; decide by measurement. Regime selection by a host predicate on T (decode T=1 keeps its path).
No SUPPORTED change.

**Verifier notes**: scheme-change on the dependent t axis; regime-pinned tests for the chunked and the
token-step paths, including T not a multiple of 32, T < 32, a call that starts mid-sequence from a
non-zero state, and the state-carry loose cases (`state_carry_f32`, `state_carry_bf16`,
`state_carry_one_call_f32` gate the state at 0.002 relative RMS: a chunk form that narrows the carried
state fails them). Must not slow `perf_decode_b1` or the decode B=8 / B=16 model cases.

**Done when**: measured device-ns on `model_prefill_4096` improves by a large factor over 89.6 ms, its
precision gates and every state-carry case hold, the golden suite is green, and there is no regression
on `perf_decode_b1` and across the config-spanning guard set of Refinement 2.

**Outcome**: model_prefill_4096 22.94 ms (HEAD, R-STEP token steps on R-VSPLIT's 48 cores; 89.6 ms on 12
cores in the note above) -> 6.41 ms device kernel time (R-CHUNK, then reader row-run tap copies); guard set
faster or at parity (prefill T=96 589 -> 170 us, batched B=2 T=40 367 -> 230 us, decode B=8 / B=16 slightly
faster, perf_decode_b1 33.8-34.0 vs 33.9 us). State-carry gates hold (state_carry_one_call_f32 state rel RMS
8.7e-4). Now compute bound at ~50 us per tile row per core: gates 4, conv 13, L2 norm 11, chunk step 16, RMS 6;
reader 14 us and writer 5 us are hidden. Next: share the q/k conv + L2 norm (+ Gram and chunk vectors) of a key
head across its 12 cores (R-BCAST / split prep across pieces), which removes most of the 28 us of replicated
prep; then R-SEG to use the 62 idle cores. Not done here: both are separate cross-core topologies; and the
chunked compute sits near the 70688 B kernel-config text limit, so any added compute code needs a size cut.

### [x] Refinement 5 — Decode B=1: remove the loopback copies and share the q/k preparation per key head

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` (`perf_decode_b1`, flagged PERF FOCUS; goal 5.3 us, DRAM bank bound
4.8 us). Measured on this op at bd772ee (operator, device kernel duration, median of 11 launches): 34.0 us
on 48 cores, 7.1x the bound; DRAM is 14% used, so no DRAM resource limits the kernel. Per-core zones
(captures generated/chip_runs/gdn_v01_d1_*): NCRISC spans 21.8 us; TRISC_0 spends 51% of its 32.4 us in
cb_wait_front (about 16.5 us waiting for its first input), then about 16 us of serial compute with the
engines busy 19% of the window (vector engine 14%, matrix engine 5%; FPU instruction waits at the head of
math 42%, unpack stalled 9%): fixed per-phase setup on very little work. NoC trace per launch: DRAM -> core
1,130,496 B in 3,840 packets (about 80 per core, about 290 B each); core -> DRAM 804,864 B in 768 packets;
core -> core 589,824 B in 144 flows; and 7,274,496 B in 13,872 packets that every core's BRISC reads from
ITSELF over NOC1 (288 stateful 512 B reads, 151,552 B per core): more bytes than the whole call moves over
DRAM, spent on a copy or a zero-fill inside one core's L1. Three regions, largest first:
(1) the input side, about 16 to 22 us: 48 cores each issue about 80 small DRAM reads and 8 of 9 window
columns (q and k of the key head) are read and prepared by all 12 cores of that key head. Build the deferred
R-BCAST of op_design.md: per key head one core reads the q/k columns once, runs conv + L2 norm once and
hands kn/qn (and what the pieces need of them) to the other cores of that key head (`shared_input_reuse`,
`mcast_topology`); the per-piece cores then read only their own v columns, state slice, z and gates. Fewer,
larger DRAM requests (aligned spans of 2 KiB and more, `read_coalescing`).
(2) the loopback traffic: replace the self-addressed NoC reads by what they stand for (a local fill or a
plain L1 copy sized to the rows that are live; no window zeroing that the live-row scope makes
unnecessary), or show by measurement that they are free.
(3) the serial compute chain, about 16 us: per-phase init / reconfig / unpack setup dominates a one-row
workload; fuse phases that share a format and a DEST state, hoist setup that does not change between
phases, and keep the work scoped to the live rows.
No SUPPORTED change.

**Verifier notes**: must not slow `model_prefill_4096` (6.41 ms at bd772ee), decode B=8 (74.0 us) or
B=16 (148.4 us); those shapes may take the same sharing where it helps (at B=16 every key head's q/k is
also read by its 3 value-head pairs). The compute kernel's kernel-config text is near its size limit
(a NoC-traced build of the prefill shape already fails with `state.offset <= max_size`): do not grow it;
prefer removing code paths that the new scheme supersedes.

**Done when**: measured device-ns on `perf_decode_b1` improves and moves toward 5.3 us, its precision
gates hold, the golden suite is green, no regression across the config-spanning guard set of Refinement 2
plus `model_prefill_4096`.

**Outcome**: perf_decode_b1 34.0 -> 25.7 us (R-BCAST with a q injector and a k injector per key head, flag-signal
multicast of qn / kn; window zeroing scoped to the used groups, 147 -> 8 / 40 KiB of loopback per core; one DRAM
request per row span and per decode token row); guard set all faster or flat (B=8 62.9, B=16 132.2, fp32 B=1 34.4
us, prefill 4096 6.41 ms). Bound now: the injectors' serial chain (read to window 6.8 us, conv 2.2 + L2 3.2 us,
hand-off ~1.3-2.9 us), then d + kn^T 2.3 us and token step + RMS/combine ~7 us on every core. Next: drop d by
o = qn^T S_new (one row of the read matmul after the update), split each q/k part over more cores (partial L2
sums), and fuse the step / RMS phases (region 3, not attempted here: the compute kernel's text budget and the
d-free form changes the token-step numerics, which deserves its own measured pass).

### [x] Refinement 6 — Prefill: take the per-row preparation off the ordered path

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[3]` (`model_prefill_4096`). Measured on this op at bd772ee (operator,
median of 11 launches): 6.41 ms on 48 cores, 1.57 us per token, 50.1 us per 32-token tile row, 128 tile
rows in order; `compute_util`: COMPUTE BOUND (engine), vector engine 53% of the window, matrix engine 4%;
62 of 110 cores idle. The op it replaces in the model runs this shape in 1.28 ms, so the model cannot
take this op for prefill before it is at least there: the aim is well under 1.28 ms. The changelog's own
split per tile row per core: gates 4, conv 13, L2 norm 11, chunk step 16, RMS norm 6 us (reader 14 and
writer 5 hidden behind compute). Only the chunk step carries the state from one tile row to the next:
gates, conv, L2 norm (34 of the 50 us) depend on the token rows alone, and the RMS norm on o alone. Take
them off the ordered path: (a) prepare tile rows in parallel on the idle cores (several prep cores per
key head / per pair feeding the core that owns the state in order, or the prep of tile row n+1 running
while the chunk step of row n runs), so that the ordered part per tile row is the chunk step alone;
(b) share the q/k prep per key head as in Refinement 5; (c) then shorten the chunk step itself (its
matrix products on Kt x Vt tiles; the segment-pair form R-SEG of op_design.md if cores are still idle).
Decide each by measurement. No SUPPORTED change.

**Verifier notes**: the carried state and exp(g) stay float32 at every chunk boundary; the state-carry
loose cases gate the state at 0.002 relative RMS. The compute kernel is near the kernel-config size
limit: split work across kernels / cores rather than adding paths to one kernel. Must not slow
`perf_decode_b1`, decode B=8 or B=16.

**Done when**: measured device-ns on `model_prefill_4096` improves by a large factor over 6.41 ms (aim:
under 1.28 ms), precision gates and every state-carry case hold, the golden suite is green, no
regression on the decode model cases.

**Outcome**: model_prefill_4096 6.41 ms -> 1.06-1.13 ms device time on 108 cores (R-PREP: 24 state cores = 12 pairs x 2 pieces, 84 prep cores; levers (a), (b) and (c) all built: prep cores per key head, q/k prep shared per key head, and the state step cut from 16 to 5.3 us per tile row by moving M = (I + L)^-1 and the row scalings onto the prep cores). Decode, prefill T=96 and batched B=2 T=40 stay at parity (short calls keep the local chunk step). What binds now: 128 ordered state steps of 5.3 us (0.68 ms), a ~120 us pipeline fill, and inbox waits on two of the four key-head groups, whose rows land late (prep writer send p90 40-50 us against ~42 us of inbox slack at depth 8). Next, by measurement: reduce the fan-out jitter (fewer, larger multicasts per row or staggered grants between groups), shorten the fill (split the first tile rows' prep across more cores), and win back state precision (R-PREP state rel RMS 1.1-1.4e-3 vs 0.6-0.85e-3 local on the state-carry generator, gate 2e-3) with a hi/lo split of the M x rhs operands. Not done here because each needs its own measured iteration and the aim is met.

### [x] Refinement 7 — Decode B=8 and B=16: the input side at a full grid

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[1]` and `[2]` (`model_decode_b8`, `model_decode_b16`). Measured on
this op at bd772ee (operator, median of 11 launches): B=8 74.0 us on 96 cores (DRAM bank bound for its
bytes 26.4 us, goal 29.3 us), B=16 148.4 us on 110 cores (bound 51.7 us, goal 57.5 us). Per-core spans
B=16: min 40, median 103, max 151 us; compute waits for input 53% (B=8: 56%); busy-time balance 1.01,
so the slow cores wait, they do not work more; `grid_check`: SHARED STARVED, grid row y3 at 138% of the
median and y11 at 39%. The fastest core takes 40 us, about what B=1 takes: everything above that is the
input side saturating as pairs are added. Captures: generated/chip_runs/gdn_v01_d8_*, gdn_v01_d16_*.
Levers, by measurement: fewer and larger DRAM requests per core (whole aligned tiles / spans of 2 KiB
and more, several in flight, one barrier per group); q/k of a key head read and prepared once for its 3
value-head pairs and for all pieces; requests spread evenly over the DRAM banks and over both NoCs so
that no bank or link is shared by many readers (`noc_placement`, bank stagger); the state (most of the
bytes: 786 KB per sequence each way) read and written in large aligned transfers. No SUPPORTED change.

**Verifier notes**: B=16 has 192 pairs on 110 cores (2 pairs on most cores): check the per-core order
of the two pairs does not serialize their reads behind each other's compute. Must not slow
`perf_decode_b1` or `model_prefill_4096`.

**Done when**: measured device-ns on both cases improves toward the goals, precision gates hold, the
golden suite is green, no regression on `perf_decode_b1` and `model_prefill_4096`.

**Outcome**: model_decode_b8 62.0 -> 49.9 us, model_decode_b16 132.2 -> 86.0 us (74.0 / 148.4 at bd772ee; goals 29.3 / 57.5) with R-QKCORE: 8 q / k cores prep qn / kn of every sequence once per key head and multicast them; pair cores (sequences of one value head, two per core at B=16) read only v, S0, z and gates; conv_state_out written as whole tiles by the q / k cores; per-unit inputs double-buffered, same-head weights copied locally, writer barriers deferred past the next unit's coefficients. perf_decode_b1 / fp32 B=1 / prefill guard set at parity (same-session A/B). What binds now: per pair a fixed compute chain of ~19 us (token step 10.8 us of fp32 HiFi4 matmuls for one valid row, RMS 5.4, d + kn^T 2.3) after a 14-17 us q / k chain, then the state writes of all cores at once (6.3 MB, 12.4 us at the DRAM rate) draining 15-30 us, slowest on the top grid rows (NOC 0 paths wrap the torus; a NOC 1 swap fixed their reads but slowed their writes, kept off). Next: overlap the state writes with compute (stream update batches to the writer instead of one push per pair), shorten the token step (both units of a B=16 core share kn / qn: one qk_finish, and the read matmul could carry both sequences' rows), and place the slow-row cores' writes on the NoC whose path does not wrap. Not done here: each needs its own measured pass, and the decode step's numerics (one-row matmuls) deserve a dedicated look.

### [x] Refinement 8 — Decode: shorten the q/k chain that every core waits for, and the one-row token step

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` (`perf_decode_b1`, flagged PERF FOCUS; goal 5.3 us, DRAM bank bound
4.8 us). Measured on this op at f756a3d (operator, device kernel duration, median of 11 launches): 26.2 us on
48 cores, 5.5x the bound; DRAM 18% used, NoC not bound (static bound / measured 0.19). Per-core zones
(captures generated/chip_runs/gdn_v02_d1_*): every core ends within 22.6 to 26.4 us; TRISC_0 sits in
cb_wait_front 70% of its span, median 15.8 us (11 to 18 us across cores): the 44 piece cores wait for the
q / k injector chain of 4 cores (their readers span 12 to 15 us with 235 NoC calls each; the other readers
6 us); compute after the wait is 7 to 9 us with the engines busy 12% of the window (FPU instruction waits
at the head of math 21%, other 67%). Two regions:
(1) the injector chain, about 15 us on the critical path of all cores: its read (6.8 us by the changelog:
make it one burst of large aligned requests issued at once, no request waiting for another's barrier),
conv + L2 norm (5.4 us: split each q / k part over more cores with partial L2 sums combined, or run q and k
of a key head on more cores in parallel; 62 cores are idle), the hand-off (1.3 to 2.9 us). And let the
piece cores do everything that does not need qn / kn while they wait: their state read, the decay
exp(g) S, the v / z / gate prep, so that only the two reads and the rank-1 update follow the hand-off.
(2) the token step for one live row (10.8 us of fp32 HiFi4 matrix products at B=8/16 by the Refinement 7
outcome, plus RMS 5.4 and d + kn^T 2.3): the products run on whole tiles for a single valid row; use the
cheapest correct form for one row (row-scoped products, fewer phases with fused setup, o from the updated
state so that d is not needed if the numerics hold: gate it with the state-carry cases).
No SUPPORTED change.

**Verifier notes**: the carried state and exp(g) stay float32; narrow operands only where the spec allows
(sums over K, the increment). Guard set as in Refinement 5, including `model_prefill_4096` (1.13 ms at
f756a3d), `model_decode_b8` (49.3 us), `model_decode_b16` (86.3 us): none may get slower. The compute kernel
is near the kernel-config size limit: remove superseded paths rather than add.

**Done when**: measured device-ns on `perf_decode_b1` improves and moves toward 5.3 us, precision gates
hold (state-carry cases at 0.002), the golden suite is green, no regression on the guard set.

**Outcome**: perf_decode_b1 26.32 -> 22.59 us (device kernel, median of launches 2..11, same session; 4.7x the 4.8 us
bound). Landed: the R-VSPLIT partial sum o^2 before the state update (gather overlaps the update, -1.1 us); decode
window rows issued first on their own NoC transaction id with the weights / gates behind them (injector window
6.0 -> 4.9 us); d = qn . kn as one [kn; qn] x kn^T matmul read as a scalar (receiver post-hand-off d + kn^T 2.5 -> 0.8
us); the token vector u / o in one DEST window with o^2 packed alongside (shared-q/k builds); the q/k normalize
batched per DEST window. Guard set improved or flat: B=8 49.98 -> 48.72, B=16 86.54 -> 85.36, fp32 B=1 33.69 -> 29.63,
prefill T=96 168.7, batched B=2 T=40 222.5, model_prefill_4096 1.10 ms. Measured null, kept as a knob at off:
`BCAST_RECT_GROUPS` (one-rectangle key-head groups: send 1.28 -> 1.22 us, reads later, 22.95 us). Where the time is
now (zones, B=1): injector chain to the hand-off ~12 us (reads 4.9: 1.4 us issuing 10 staged requests, 1.5 us of 40
local row copies; FPU conv of 4 q / k columns 1.8; L2 part 2.9; multicast of 16 KB per channel 1.2-1.5), then per core
token step ~4 us (state read 0.7, token vector + partial 2.0, update 1.1) and the RMS combine ~2.5 us. Next: split
each q / k part over 2-4 cores with partial L2 sums exchanged (halves the reads, conv and normalize on the chain),
reader / writer split of the row copies, and half-tile hand-off payloads (faces 0-1 only for rows < 16). Not done
here: each needs new roles or CB layouts, and the fp32 token-step builds sit ~80 B under the kernel-config text
budget, so anything added to their compute or reader must be compiled out of them first.

### [x] Refinement 9 — Decode B=8 / B=16: the state write-back drain and the merging read chain

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[2]` and `[1]` (`model_decode_b16`, `model_decode_b8`). Measured on this
op at f756a3d (operator, median of 11 launches): B=16 86.3 us on 104 cores, 1.67x the DRAM bank bound of
51.7 us (goal 57.5 us); B=8 49.3 us on 104 cores, 1.87x of 26.4 us (goal 29.3 us). B=16 per-core spans
min 15, median 74, max 88 us; compute waits for input 44%; busy-time balance 1.03. NoC trace
(generated/chip_runs/gdn_v02_d16_noc_c0): CONGESTED (merging chain), static bound / measured 0.68; DRAM
bank 0 at 68% of its rate; link NOC0 E(0,11) at 45% with up to 169 flows on one link; 27.98 MB over DRAM
against 26.36 MB that the tensors hold (1.6 MB crosses twice). By the Refinement 7 outcome the state
writes of all cores start at once (6.3 MB, 12.4 us at the DRAM rate) and drain 15 to 30 us, slowest on
the top grid rows, after a 14 to 17 us q / k chain and a 19 us per-pair compute chain. So the call is
three serial stretches where the bound assumes one overlapped stream. Levers, by measurement: (a) overlap:
the state read of a pair can start at launch and its write-back as soon as a state tile is final (tile by
tile during the update, not all cores at the end), so reads, compute and writes of different pairs
overlap; with 2 pairs per core, the second pair's read behind the first pair's compute; (b) spread: reads
on NOC0 and writes on NOC1 with requests staggered over the 8 DRAM banks so that no bank and no link
carries most flows (`noc_placement`, bank stagger; a core's two pairs on different banks); (c) no byte
twice: find the 1.6 MB that crosses DRAM twice; (d) the q / k chain as in Refinement 8.
No SUPPORTED change.

**Verifier notes**: must not slow `perf_decode_b1` or `model_prefill_4096`. Outputs stay new tensors;
no write into an input.

**Done when**: measured device-ns on both cases improves toward the goals, precision gates hold, the
golden suite is green, no regression on `perf_decode_b1` and `model_prefill_4096`.

**Outcome**: model_decode_b8 48.8 -> 40.2 us, model_decode_b16 86.0 -> 73.5 us (device kernel, launches 2..3 of two fresh runs, same session; goals 29.3 / 57.5). Landed: pair cores in the top 3 grid rows read on NOC 1 / write on NOC 0 (SWAP_NOC_ROWS = 3, now also at B=8: their weights / gate-row reads had landed at 25-31 us vs 4-5 us on lower rows; alone 48.8 / 86 -> 42.6 / 74.1); the final state streamed to the writer per update batch; final-state tiles updated, written and read starting at tile (pair * 3) mod 16 (STATE_STAGGER: cores starting together no longer all hit the bank of tile 0; 42.0 -> 40.0 at B=8); multi-unit cores compute kn^T and d once (SHARE_QK_FINISH) and normalize unit k after unit k+1's token step (UNIT_LAG): compute ends ~48 us instead of ~59 at B=16. Measured null or worse, kept as a knob at off: EARLY_S0 (S0 issued with the window rows: 44.7 / 75 vs 42.8 / 74.4 with the swap). Swapping rows 3-4 too, or by x, was slower (51-56 / 78-93 us). Guard set at parity (same-session A/B against HEAD): perf_decode_b1 22.4 vs 22.1, decode_b1_f32 30.2 vs 30.3, prefill_t96 168.5, batched_b2_t40 224, model_prefill_4096 1.06-1.17 ms. What binds now (B=16 zones): the state writes, 12.6 MB starting when the first units' states are final (~30 us) and issued by one RISC per core on one NoC (16 us to issue a 64 KB state under back-pressure), ending 48-72 us with an x gradient per NoC (NOC 1 writes slowest at high x, NOC 0 at low x); and the pair cores in grid rows y = 5-6, whose S0 lands at 28-38 us on either NoC, so their compute ends 60-68 us. Next: split each core's state write across both RISCs / NoCs (the reader is idle after ~35 us), pick each core's NoC pair by its path to the banks rather than by row, and shorten the per-unit token step (10.3 us: update 4.3, token vector 4.5 for one valid row). Not done here: the split writer needs a second consumer protocol for cb_state_out and the per-core NoC choice a path model; each is its own measured pass.

### [x] Refinement 10 — Caller-provided state outputs that may be the state inputs (update in place)

**Goal**: add two keyword-only parameters, `conv_state_out: ttnn.Tensor = None` and
`recurrent_state_out: ttnn.Tensor = None`. When given, the op writes that output into the caller's tensor
(same shape, dtype, layout and memory configuration as the corresponding input; validate() refuses
anything else) instead of allocating one, and returns it. The caller may pass the SAME tensor object as
the input (`recurrent_state_out=recurrent_state`, `conv_state_out=conv_state`): the state is then updated
in place and no second copy of it exists. Default `None` keeps today's behaviour (new tensors) and every
golden cell unchanged. Why (operator, measured in the model this op is for, at 5e03ed5): the model decodes
by trace replay, where the persistent state tensors keep their addresses, so the caller has to copy the
op's new state back into them after every call: one extra device program that moves the whole state over
DRAM a second time. Measured per call on one chip: 68.9 us at B=16 (the op itself: 74.1 us), 32.9 us at
B=8, 4.7 us at B=1, plus 1.3 to 2.2 us for the conv history. Passing the input tensors as outputs to the
op as it stands gives wrong results at B=16 (output PCC falls from 0.997 to 0.943 over 4 decode steps)
while B=1 and prefill pass: so the kernels today read some input region after another core has written
the same region of the output. In-place must be made correct by construction: every byte of
recurrent_state and conv_state that any core reads is read before any core writes that region (note the
rotated start tile of the streamed state write-back, the top grid rows that swap NoCs, cores that hold
two pairs, conv_state tile rows shared by several sequences, and the M3 rule that conv_state_out rows
come from conv_state when T < 3), with no extra pass of the state over DRAM and no slowdown of the
default path.

**Implementation skill**: /memory-layouts

**Verifier notes**: extend the op's unit tests (tests/ttnn/unit_tests/operations/gated_delta_net): for
decode B=1, B=8, B=16, B=40, prefill T=33, T=96, T=4096 and batched prefill, in-place calls over at
least 8 consecutive calls must give the same outputs and states as the default path to the bit (same
kernels, same arithmetic), on both dtypes; a provided output that is neither the input nor a distinct
tensor of the right form is refused. The spec sentence "the op MUST NOT write into an input tensor"
now holds for every tensor the caller did not pass as an output. Record the in-place timing of
`model_decode_b16` and `perf_decode_b1` against the default path: it must not be slower.

**Done when**: the new parameters exist and are documented in the op file; in-place results are
bit-identical to the default path on the shapes above; the golden suite is green with 0 loud categories;
no regression on the perf guard set.

**Outcome**: in place is bit-identical to the default chain on every listed shape, both dtypes (test_gated_delta_net_in_place.py). Device kernel time, default vs in place: perf_decode_b1 22.0 vs 22.0 us, model_decode_b16 median 73.7 vs 73.3 us. recurrent_state needed no change; conv_state reads now signal their writer core before it overwrites them (in-place builds only).

### [x] Refinement 11 — Short and medium prefill (T = 32 to 512): faster than a chain of ops

**Type**: perf

**Goal**: prefill calls of one sequence with T between 32 and 512 tokens (the model ingests a short
prompt in one such call; time to first token is 48 such calls). Measured in the model this op is for
(operator, one chip, device kernel duration, median of 11 calls, at 5e03ed5): T=128 (mixed_qkv
(1,128,2560) bf16, state (1,12,128,128)) takes 188.8 us on 40 cores, where the chain of existing ops it
replaces takes 158.1 us for the same work: the model's time to first token got 8.7% worse with this op
(51.6 -> 56.1 ms at 128 tokens). T=4096 is fine (1.09 ms on 108 cores, R-PREP). So the regime chosen
for short calls leaves most of the grid idle and runs the local chunk step: 4 tile rows cost about 47
us each. Find by measurement the regime boundary and the form for short calls: R-PREP style
preparation cores from the first tile row on (the fill of the pipeline is what a short call consists
of, so spread the preparation of all its tile rows over the idle cores at once: with T=128 there are
4 tile rows and 110 cores), the ordered state step alone on the state cores, the q/k preparation
shared per key head. Aim: T=128 under 80 us, T=32 and T=64 not slower than today, a smooth hand-over to
the long-call regime, no slowdown at T=4096. No SUPPORTED change.

**Verifier notes**: guard set: `perf_decode_b1` (22.0 us), `model_decode_b8` (39.3), `model_decode_b16`
(73.5), `model_prefill_4096` (1.09 ms), prefill T=96, batched B=2 T=40. State precision as before
(state-carry cases at 0.002). The compute kernel is at its kernel-config size limit (a NoC-traced
build of decode B=16 is 1,920 B over): if this needs code, first move roles into kernels of their own
(separate source files per role: preparation, state, injector, piece) so that each fits with room.

**Done when**: measured device-ns at T=128 improves by a large factor (aim under 80 us), T=32 / 64 /
256 / 512 are measured and none is slower than today, the golden suite is green, no regression on the
guard set.

**Outcome**: device kernel time (model geometry, bf16, one fresh run of 3 launches each, same session), before -> after: T=32 67.5 -> 63.6 us (local chunk step kept: one tile row), T=64 118 -> 72.7, T=96 168.5 -> 77.6, T=128 188 -> 82.0 (2.3x, 96 cores; aim 80 not reached), T=256 220-239 -> 134, T=512 282-325 -> 196, T=1024 ~390 -> 298, model_prefill_4096 1.09 -> 0.91 ms; batched B=2 T=40 224 -> 126; decode guards unchanged (B=1 22.0, B=8 40.0, B=16 73.3). Landed: R-PREP from 2 tile rows on; prep groups of `PREP_HEADS` value heads (1 for short calls: one tile row's prep spread over 3x the cores) and the piece count (4 up to 5 tile rows, else 2) picked by a fitted cost model; each group's state cores placed among its prep cores with a per-core send NoC (also what moved T=4096); head blocks unicast and streamed part by part as compute packs them; norm_weight (*) silu(z) made before o lands; the triangular mask built on the writer; state cores read S0 before their first row lands. Measured worse and parked off: one window group per prep tile row (87.8 vs 82.7 us). What binds at T=128 now: the prep core's serial compute after its inputs land (~7 us): conv 17.6 (12 fp32 silu tiles), L2 norm 10, head prep 7.4, head block 9.8 us, then 4 state steps of 3.1 us and a ~6 us RMS + store tail. Next: share the q / k part per key head across cores (a q core and a k core per tile row feeding the value heads' prep cores, as R-QKCORE does for decode): about 22 of the prep core's 45 us is q / k work redone by every value head's prep core; not done here because it needs a new cross-core hand-off between prep cores. The compute kernel was not split per role: every addition compiles only into chunked (R-PREP) builds, so the decode builds near the kernel-config limit are unchanged.
