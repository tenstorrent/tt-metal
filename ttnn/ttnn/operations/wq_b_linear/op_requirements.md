# Operation Requirements: wq_b_linear

## Definition
- **Formula**: `y[0,0,m,n] = Σ_k x[0,0,m,k] · W[k,n]` (no bias, no activation)
- **PyTorch Reference**: `torch.matmul(x.double(), W_q.double()).bfloat16()`, where `W_q` is W as the device holds it after bfloat8_b quantization (`eval/golden_tests/wq_b_linear/helpers.py::pytorch_wq_b_linear`)
- **Import Path**: `from ttnn.operations.wq_b_linear import wq_b_linear`
- **Function Signature**: `wq_b_linear(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor`
  - x `[1,1,160,1536]` bf16 TILE, DRAM interleaved
  - W `[1536,4096]` bf8_b TILE, DRAM interleaved
  - returns y `[1,1,160,4096]` bf16 TILE, L1 interleaved

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

**Queue shape.** `TARGET − SUPPORTED` is empty: TARGET is a single cell (bf16 x, bf8_b W, TILE, `fp32_dest_acc_en=False`), and Phase 0 supports it. That leaves no generality refinements, so the 2:1 cadence collapses and every remaining phase is perf. The perf-flagged `LOOSE_CASES` entry is the whole input space, and its full contract is already in SUPPORTED: shape, dtypes, TILE, DRAM inputs, L1 output, HiFi2, `fp32_dest_acc_en=False`, soft PCC gate 0.9998.

**Measurement baseline (Phase 0, p150 / 8 banks):**
- Device span: **26.6 µs** (median of 6).
- SOL: **14.0 µs** on this board. `feature_spec`'s `sol_ns_blackhole` = 16,018 ns is for a 7-bank p100a.
- Read time from `generated/profiler/.logs/profile_log_device.csv` (first `*-KERNEL` start to last end, excluding cores with unsynced timestamps). The ops-CSV duration column was corrupted by one core on this board.
- Bench: `tests/ttnn/unit_tests/operations/wq_b_linear/_op_bench.py` under `scripts/run_safe_pytest.sh --profile`.

**Hard constraint for every perf phase:** `block_k_tiles ≤ 16`. 16-bit DEST accumulation drifts the output scale with the in-DEST K length, and PCC is 0.999786 at kb ≥ 24, which is below the 0.9998 gate. Smaller kb is more precise (see `verification_report.md`, Precision Baseline).

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16] (x); **weight_dtype**: [bfloat8_b]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED shape-derived axes**: none (`INPUT_TAGGERS = {}`; one shape)
- **SUPPORTED op-specific axes**: `fp32_dest_acc_en` ∈ {False}
- **Cores**: 64 workers (8×8 rectangle; N-split, `block_n_tiles = 2`) + 1 x-sender, one `generic_op` dispatch
- **Compute config**: `compute_kernel_config` exposed; default HiFi2, `fp32_dest_acc_en=False`, `math_approx_mode=False`
- **Blocking**: `block_m_tiles = 5` (whole M), `block_k_tiles = 8` (`num_k_blocks = 6`), x resident on workers via handshake-free Counter multicast, W slice resident (`w_depth_blocks = num_k_blocks`)
- **Golden baseline**: 2 / 2 registry cells passing (cartesian + perf loose case), regression test passing; 0 supported_fail / xpass_drift / xfail_wrong_mode

### [x] Refinement 1 — Speed up x delivery on the perf-flagged profile

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags `((1,1,160,1536), (1536,4096))` (bf16 × bf8_b, TILE, HiFi2, `fp32_dest_acc_en=False`) as the mandatory perf target. Its perf goal is `sol_ns` (14.0 µs on this p150; 16.0 µs p100a figure), and its soft gate is PCC ≥ 0.9998. Optimize this exact config.

Phase 0 is **x-delivery-bound**:
- The single sender core reads x (491,520 B) from DRAM in **14.6 µs**, about 34 GB/s. The model predicts 8.3 µs.
- The sender finishes multicasting at **22.4 µs**. The model predicts 10.8 µs.
- Workers' W reads finish by ~9 µs, and compute (ideal 11.4 µs) ends at ~24.7 µs, waiting on x rounds.

Shrink the x path. Relevant `ttnn/ttnn/operations/examples/master.md` entries:
- `double_buffer` / `trid_double_issue`: keep more x reads in flight. Issue round r+1.. before retiring round r, because the resident CB lets every round's reads be outstanding.
- `shared_input_reuse`: its note that "a single injector reads serially" suggests more than one injector. Split x rounds across two sender cores or both NoCs; each extra `ttnn.Mcast` channel goes at `next_semaphore_id()`.
- `mcast_topology`: multicast placement.

The other part of the cost is per-round: each Counter-mode round pays a write barrier plus a semaphore increment to 64 receivers (~1 µs measured; this is why the implementer moved kb from 4 to 8). Remove it or amortize it, either with linked/posted multicast rounds or by fewer barriers per round. Once round cost is cheap, co-tune `block_k_tiles` **downward**. The model's rank 1/2 use kb 2, and smaller kb also improves precision. No SUPPORTED change.

**Calculator**: `python3 .claude/tools/perf_calculator/op_model.py matmul 5 48 128 --designs --c-mem l1 --fidelity HiFi2 --a-dtype bf16 --b-dtype bfp8_b --c-dtype bf16 [--kb 8 --workers 64]`.

| Design | Predicted | Rank / note |
|---|---|---|
| Current (kb 8, 64 workers + sender) | 18.75 µs | measured 26.6 µs |
| Candidate: same grid, kb 2, linked x multicast | 16.60 µs | rank 2 |
| 73-core variant | 16.53 µs | rank 1, tie with rank 2 |

- The candidate is 11.5% better on paper, which is inside the ~15% band and so counts as a tie on rank. The filed reason is the measured 12 µs gap between the model's sender timeline and the device.
- The model does not price the per-round Counter barrier or multiple injectors. Record the measurement next to these numbers.

**Verifier notes**:
- This is a dataflow change on the sender and the worker writer. The topology stays: x is still read once from DRAM and multicast to resident CBs, with no handshake.
- Adding a second injector is a new Mcast channel, not a new regime. It keeps DRAM traffic at the minimum.
- Respect `block_k_tiles ≤ 16`.
- The x CB address must stay identical across all sender and worker cores. Any extra sender must sit inside the `cb_x_resident` core-range union.

**Done when**:
- Measured device span on the flagged shape improves from 26.6 µs, moving toward the 14.0 µs SOL. Target the model's ~16.6 µs region.
- PCC ≥ 0.9998 still holds, and the golden suite (2 cells + regression bit-identity) is green.
- The precision baseline passes.
- No regression on the guard set. The op has one kernel path × one layout × one placement, so the guard set is the flagged shape itself under both call patterns: default config and explicit `ComputeConfigDescriptor`.

**Outcome**:
- Device span 26.6 µs → **22.55 µs** (−15%, 1.61× SOL). Shipped: trid-windowed x read (2 rounds in flight), linked per-round multicast, round published by an ordered count write instead of barrier + atomic (~1.1 µs/round removed), `block_k_tiles` 8 → 4. PCC 0.999957.
- Built and measured, parked as live knobs:
  - 2 senders (`NUM_X_SENDERS`): one NoC1 sender wins.
  - Lagged atomic signal (`X_SIGNAL_LAG`).
  - W gated on x round 0 (`W_WAIT_X0`): slower.
- x is no longer binding after round 0. Stubbing the W reads leaves the span unchanged (21.9 µs).
- The critical path is: x round 0 at ~2.5 µs, then the worker matmul at ~15 µs for kb 4 (11.4 µs ideal), then the ~2 µs output-store tail. The span is set by the slowest worker, whose W bank share ends ~2.5 µs after the median.
- Next: Refinement 2's tail and W-balance knobs (store overlap / NoC split, `bank_stagger`, column map), then the matmul's own efficiency (subblock 5×1 at kb 4). Not done here because they are outside x delivery and queued as Refinement 2.

### [ ] Refinement 2 — Trim the tail and the W side on the perf-flagged profile

**Type**: perf

**Goal**: same flagged shape and contract as Refinement 1. Once x no longer binds, the op becomes W-stream/compute/tail-bound: W reads take ~9 µs per worker on 8 banks, the DRAM floor is 14.0 µs, and compute is 11.4 µs ideal.

Several cheap ⭐ T1/T2 knobs from `op_design.md`'s perf lamps and `master.md`:
- **Output-store tail**: ~2.3 µs after compute ends in Phase 0. Split output writes across NoC0/NoC1 (model `c_noc = split`) and overlap the store with the last K-step.
- **W read coalescing** (`read_coalescing`): a worker's 2 columns `n, n+B` sit in adjacent slots on one bank, so issue one 2,176 B read per k-row instead of two 1,088 B reads.
- **Column map**: re-measure `bank_spread` against `contiguous` with the device's real bank count (8 here, not the design's 7).
- **`w_depth_blocks`**: 3–4 instead of full residency. This frees ~70 KB L1 per worker; keep it only if neutral or faster.
- **`bank_stagger`**: rotate each worker's K-start.

Measure each knob and keep only the device-ns wins. No SUPPORTED change.

**Calculator**: same command as Refinement 1. The model's rank-1 design already uses `c_noc = split` and `bank_spread`. It rates `c_noc` split vs NoC1 a tie (17.39 vs 17.55 µs, 1%), so these knobs are expected to be small. File each as measured or dropped.

**Verifier notes**:
- Runs after Refinement 1, which is a true dependency for measurement: while x binds, none of these knobs is visible on the span.
- First check the headroom: if Refinement 1 lands within ~10% of SOL, measure, record the result, and close this phase without forcing changes.
- The `w_depth_blocks` change folds in the L1-ledger lamp. Update `l1_ledger.md` with any CB size change.

**Done when**:
- Measured device span on the flagged shape improves over Refinement 1's result, or each knob is recorded as measured-neutral with numbers in `changelog.md`.
- PCC ≥ 0.9998, the golden suite is green, and the precision baseline passes.
- No regression on the guard set (flagged shape, both call patterns).

## Not queued (documented in `verification_report.md`)

- **K-split with a deterministic combine** (`deferred` regime): model 18.15 µs vs 16.53 µs rank 1. It adds 2.6 MB NoC partials to cut compute that is not binding.
- **2-D M×N split with W multicast** (`deferred`): model 22.3 µs (+35%, lower bound).
- **Per-worker DRAM x reads / M bands re-reading W** (`rejected`): 64× x or bands× W DRAM traffic.
