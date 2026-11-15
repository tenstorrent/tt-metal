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

### [ ] Refinement 2 — Cut the per-pair fixed cost of perf_decode_b1

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

**Done when**: measured device-ns on `perf_decode_b1` (exactly its bf16 config) improves and moves
toward the 5.3 us goal, its precision gates (output PCC 0.995 / rel RMS 0.04, state PCC 0.999 / rel
RMS 0.01) still hold, the golden suite is green, and there is no regression across the
config-spanning guard set (one representative per distinct kernel path: decode B=1, decode B>1 with
shared tile rows, prefill single tile row, prefill multi tile row, batched_prefill; float32 and
bfloat16).

### [ ] Refinement 3 — R-VSPLIT: spread each pair's state columns across cores for perf_decode_b1

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
