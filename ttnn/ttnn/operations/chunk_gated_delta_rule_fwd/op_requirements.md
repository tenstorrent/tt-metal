# Operation Requirements: chunk_gated_delta_rule_fwd

## Definition
- **Formula** (per batch `b`, head `h`; recurrent state `S ∈ R^{K×V}`, token by token):
  `S_t = exp(g_t)·S_{t−1} + k_t·u_tᵀ`, `u_t = β_t·(v_t − (exp(g_t)·S_{t−1})ᵀ k_t)`,
  `o_t = S_tᵀ (scale·q_t)`, `S_{−1} = initial_state` (or 0). Computed chunk-parallel (chunk `C`,
  `NC = ceil(T/C)`, tail zero-padded): `decay = cumsum_chunk(g)`, `L[t,s] = exp(decay_t − decay_s)·1[s≤t]`,
  `A_strict = −((kβ kᵀ) ⊙ L)` strictly lower, `Tinv = (I − A_strict)^{-1}`, `v_corr = Tinv vβ`,
  `kcd = Tinv (kβ ⊙ exp(decay))`, `h_i = S`, `v_new_i = v_corr − kcd h_i`,
  `o_i = (q̃ ⊙ exp(decay)) h_i + ((q̃ kᵀ) ⊙ L) v_new_i`,
  `S ← exp(decay_{C−1}) h_i + (k ⊙ exp(decay_{C−1} − decay))ᵀ v_new_i`.
- **PyTorch Reference**: `eval/golden_tests/chunk_gated_delta_rule_fwd/helpers.py::pytorch_chunk_gated_delta_rule_fwd`
  (float64 oracle, returns the same 6-tuple); token-by-token definition `naive_recurrent_gated_delta_rule`.
- **Import Path**: `from ttnn.operations.chunk_gated_delta_rule_fwd import chunk_gated_delta_rule_fwd`
- **Function Signature**:
  ```python
  chunk_gated_delta_rule_fwd(
      q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, g: ttnn.Tensor, beta: ttnn.Tensor, *,
      initial_state: ttnn.Tensor = None, chunk_size: int = 64, scale: float = None,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None, memory_config: ttnn.MemoryConfig = None,
  ) -> tuple  # (o [B,T,H,V], final_state [B,H,K,V], h [B,NC,H,K,V], v_new [B,T,H,V], g_cumsum [B,T,H], A [B,T,H,C])
  ```

## Queue shape (read first)

`TARGET − SUPPORTED` is **empty on every axis** — Phase 0 already serves the whole TARGET rectangle
(dtype × layout × state_mode × chunk_size × seq_alignment × head_dims) through one code path, with
`EXCLUSIONS = []` and `INVALID = []`; the golden suite has 76/76 supported cells passing and an empty
`xfail_expected` bucket. There are therefore **no generality refinements**, and per the 2:1 cadence
rule ("once generality candidates are exhausted, the remaining phases are all perf") every entry below
is a measured perf refinement. No `LOOSE_CASE` carries an `attention:` perf flag, so the target is
free-selected: the LOOSE Qwen3.5 prefill profile `((1, 4096, 16, 128, 128), 64)` in **both** of its
LOOSE configs (bfloat16 / with_h0 and float32 / no_h0, default compute config HiFi4 +
`fp32_dest_acc_en=True`) — its full contract is already in `SUPPORTED`, so no unlock precedes it.

**Measuring stick** for every entry: `tests/ttnn/unit_tests/operations/chunk_gated_delta_rule_fwd/test_chunk_gated_delta_rule_fwd_perf.py`
under `scripts/run_safe_pytest.sh --profile --run-all` (device kernel duration of the second dispatch
per case). Its `PERF_CASES` = the two target configs + the **config-spanning guard set** (one
representative per distinct extent regime × dtype × state_mode: `NV>1, Vs>1` H=32; `NV=1, Vs>1` with
>1 item/core; wide_v `NV=8`; long 32-chunk scan on 72 cores; ragged long bf16; the 2-core minimal
shape). Baseline after the verifier fixes (Blackhole p300a, 110 cores):

| Case | µs |
|---|---|
| LOOSE bf16 with_h0 | 4425 |
| LOOSE fp32 no_h0 | 5341 |
| (1,256,32,128,128) c64 fp32 | 813 |
| (4,128,16,64,64) c32 fp32 with_h0 | 339 |
| (1,256,4,128,256) c64 bf16 with_h0 | 344 |
| (1,2048,2,128,128) c64 fp32 with_h0 | 472 |
| (1,1000,4,128,128) c64 bf16 | 331 |
| (1,32,1,32,32) c32 fp32 | 56 |

`ttnn/ttnn/operations/examples/master.md` (the perf-pattern catalog) **does not exist on this
branch**; the levers below are argued from this op's own measurements (`l1_ledger.md` → Verifier
measurements) and the planner's Regimes / Perf-lamps tables in `op_design.md`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED state_mode**: [no_h0, with_h0]
- **SUPPORTED shape-derived axes**: chunk_size ∈ {32, 64}, seq_alignment ∈ {chunk_aligned, chunk_ragged}, head_dims ∈ {square, wide_v}
- **EXCLUSIONS**: none. **INVALID** (feature_spec.py): none.
- **Mechanism caps (hard ValueError, not axes)**: `H ≤ 32`; interleaved I/O only (DRAM or L1); `float32` requires `fp32_dest_acc_en=True`; no `item_block_val_tiles` fitting L1 → ValueError.
- **Cores**: multi-core, one `generic_op` dispatch — regime R1 (P: `(bh, chunk)` items over `min(G, NI)` cores; S: `(bh, v_block)` scan units over `BH·NV` cores; E: items on the P cores), segmented semaphore handoffs.
- **Compute config**: `compute_kernel_config` exposed; default `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False`; internal CBs Float32 always.
- **Golden baseline**: 90 / 90 tests passing; verifier CLI: 76 `supported_pass`, 0 `supported_fail`, 0 `xpass_drift`, 0 `xfail_wrong_mode`, 0 `xfail_expected`, 14 `no_axes_found` (the non-registry `test_regression.py` numerics tests, all passing).

### [ ] Refinement 1 — Cut the face-row gather/scatter transaction count (page-harvest / multi-head items)

**Type**: perf

**Goal**: at the LOOSE target `((1,4096,16,128,128), 64)` (bf16/with_h0 and fp32/no_h0) the op is
bound by stage P's face-row gather: stage-marker zones put P at **1.22–3.66 ms per core** (9–10 items
each) of a 4.4 ms wall, and the scan cores — done with their own P at ~1.2 ms — sit waiting on the
segment releases of the slowest P cores. The gather is **transaction/issue-bound**: turning
`GATHER_DEPTH` 2→4 and `GATHER_STAGE_TOKENS` 16→32 moved nothing beyond ±2% noise, and the Phase 0
stub measurement attributed ~54% of the H=32 wall to the gather and ~10% to the scatter. The lever is
therefore the **count** of gather reads and scatter writes (≈ 786 k + 1.38 M at the target, one per
`(token, d_tile, head)`), which R1 pays once per head. Reduce it by harvesting several heads per NoC
transaction. Two realizations, implementer's choice (both are the planner's R3 in
`op_design.md` → Regimes, and both keep stages P/S/E and their per-item blocks unchanged):

- **Multi-head items** (turn the planner's `block_heads` knob, currently `BLOCK_HEADS = 1` with a
  `static_assert` in `cgdr_common.hpp`): an item becomes `(b, head-group of hb heads, chunk)`; the
  rows of `hb` consecutive heads inside one 16-head face half are contiguous, so one span read of
  `256 + 16·hb` elements feeds `hb` heads (÷hb reads), and `o` / `v_new` / `A` rows of the group are
  contiguous runs (÷hb writes). No new cross-core topology; costs per-item L1 ×hb on the per-head
  blocks (re-run the `Vi` solve; keep `NI/hb ≥ G` at the target) and a `(bh)`→`(b, head-group)`
  re-derivation of the handoff targets. `hb` should be host-solved (largest divisor of `min(H,16)`
  that fits L1 and keeps the grid full), single source → CT arg.
- **R3 proper**: a harvest stage reads each `[32 heads × 32]` page once and writes head-major compact
  tiles (÷H reads) with a per-`(b, chunk)` fan-out rendezvous in front of P and a fan-in full-page
  assembly behind E / P for `o`, `v_new`, `A` (÷H writes); adds one compact DRAM round trip of
  `q,k,v` and of the outputs (`l1_ledger.md` → Data-movement budget).

Also examine the 3× per-core P spread (same item count, 1.2 vs 3.7 ms; slow cores are the low-row
cores away from the scan cores) once the transaction count drops — if it persists, balancing the
item assignment or issuing part of the gather on the writer's NoC are cheap follow-on levers. No
SUPPORTED change.

**Verifier notes**: first because it is the measured critical path and the only lever that shortens
P; Refinements 2 and 3 are tuned on whatever P/S/E balance this one produces (the scan is P-starved
today, so a scan-side win cannot show until P shrinks). This resolves the L1-ledger finding that the
cheapest-traffic split (R3) was deferred without a positive reason. The `g_cumsum` scalar scatter
also becomes a contiguous run of `hb` elements per token under multi-head items. Keep
`test_no_h0_first_state_is_exactly_zero` / `test_h0_is_first_state` semantics (h[:,0] bit-exact).

**Done when**: measured device-ns improves on both LOOSE target configs (expect well over 1.3× if the
per-core P time falls with the transaction count), the golden suite is green (90/90, verifier loud
categories 0), and no guard-set case in `test_chunk_gated_delta_rule_fwd_perf.py` regresses beyond
noise (≥ 0.97× of its baseline above).

### [ ] Refinement 2 — Scan critical path: multicast the reuse-shared scan operands (R4) + scan placement

**Type**: perf

**Goal**: once Refinement 1 shortens stage P, the V-split scan (target: 64 units × 64 sequential chunk
steps) becomes the critical path. Each step re-reads its `nkcd_i`, `Pᵀ_i`, `Γ_i` operands from DRAM
once **per V unit** (`NV = 4` at the target — ≈ 272 MB of the op's ≈ 403 MB scratch reads, fp32),
although they do not vary with `v_block`: they are reuse-shared by construction of the V split. Build
the planner's **R4**: one scan unit of each `bh` group reads the shared operands once per chunk and
multicasts them to the other `NV − 1` units (the units of a group already sit on consecutive cores
`G−1−u`, so the receiver set is a contiguous run). There is no `mcast_pipe.hpp` in this tree —
use raw `noc_async_write_multicast` + a per-group ready/ack semaphore pair (each core has 16
semaphores; the handoffs already use `2·NS ≤ 8`). Bundle the planner's **scan-placement** lamp:
measure excluding the scan cores from P (`num_item_cores = G − NU`) against the current overlap, and
co-tune `NV` against coarser `Vs` (`NV/2`) — occupancy-first `NV` multiplies exactly the traffic R4
removes. No SUPPORTED change.

**Verifier notes**: stands alone — a new reuse/broadcast dataflow topology (one injector + receivers
+ handshake) is the work. Validate R4 by building it and measuring device-ns, not by inferring its
value from a remove-one-stage ablation (a broadcast's benefit is under-shown that way). If
post-Refinement-1 zones show the scan still waits on P or is compute-bound per step rather than
read-bound, take the placement / `NV` co-tune alone and record why R4 was not built. The scan's
`Γ·S` carry must stay on the SFPU in fp32 DEST (precision contract).

**Done when**: measured device-ns improves on both LOOSE target configs over the post-Refinement-1
baseline, the golden suite is green, and no guard-set case regresses beyond noise — in particular the
`NV = 1` case `(4,128,16,64,64)` and the `NV = 8` wide_v case `(1,256,4,128,256)`.

### [ ] Refinement 3 — Per-item fixed cost and overlap: block-depth / segment / chunk-block co-tune

**Type**: perf

**Goal**: knob-turns on the block surface the planner exposed (all host constants → CT args, single
source in the program descriptor), measured on the target and the guard set after Refinements 1–2:

- **Overlap (items)**: `BLOCK_DEPTH = 1` on the reader→compute input CBs (`cb_q_in`, `cb_k_in`,
  `cb_vblock_in`, `cb_gate_in`) means the reader cannot gather item `r+1` while compute runs item `r`
  (≈ 9–10 items/core at the target). Depth 2 on those four only (+48–96 KB at the largest shapes;
  re-run the `Vi` solve).
- **Handoff segments**: `READY_SEGMENTS_MAX` ∈ {1, 2, 4, 8} (≤ 8 by the 16-semaphore cap).
- **Chunk blocking**: `BLOCK_CHUNKS = 2` for small `K·V` (items are ≪ L1 at `K = V ≤ 64`; ~15 phase
  inits per chunk are paid today). The kernels `static_assert(BLOCK_CHUNKS == 1)`, so this needs the
  item loop generalized — whole tiles/chunks only, coarser amortizes.
- **Phase fixed cost in compute**: `mm()` re-issues `matmul_block_init` per subblock whenever a DEST
  epilogue is fused (`MM_PRE/EXP/NEG/MUL/DUAL`), plus `reconfig_data_format` on fp32→fp32 boundaries
  — elide the reconfigs that cannot touch an in-dtype CB and hoist inits where the epilogue does not
  reprogram unpack.
- **Small problems**: the minimal shape `(1,32,1,32,32)` still costs 56 µs across three stages, two
  DRAM round trips and two handoffs on 2 cores; measure the planner's R2 predicate (`NI == 1`: one
  core, no handoffs, no scratch) for the `NI ≤ small` shapes.

No SUPPORTED change.

**Verifier notes**: several cheap levers in one phase (T1/T2-class), deliberately after the two
structural refinements because their optima move with the P/S/E balance those produce. Depth-2 input
buffers need L1 headroom: fold in the L1-ledger sharing findings (`cb_state` ↔ `cb_T`/`cb_pow`/`cb_kb`,
`cb_vmat` ↔ `cb_scan_vnew` — disjoint lifetimes, different quanta; see `l1_ledger.md` → Verifier
audit) if the solve starts shrinking `Vi`. `ACCUM_DEPTH` is not a knob (1 hangs).

**Done when**: measured device-ns improves on both LOOSE target configs and on the minimal shape,
the golden suite is green, and no guard-set case regresses beyond noise. Record the chosen knob
values and the sweep in `l1_ledger.md` / the changelog.
