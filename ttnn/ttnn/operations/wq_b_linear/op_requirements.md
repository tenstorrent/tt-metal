# Operation Requirements: wq_b_linear

## Definition
- **Formula**: `y[0,0,m,n] = Σ_k x[0,0,m,k] · W[k,n]`, x `[1,1,160,1536]` bf16, W `[1536,4096]` bf8b, y `[1,1,160,4096]` bf16 L1 interleaved
- **PyTorch Reference**: `torch.matmul(x.double(), W_device_quantized.double()).bfloat16()`
- **Import Path**: `from ttnn.operations.wq_b_linear import wq_b_linear`
- **Function Signature**: `wq_b_linear(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

**Queue shape**: TARGET − SUPPORTED = ∅ (one cell; SUPPORTED == TARGET at Phase 0), and `INVALID = []`. No generality refinement exists, so every entry is a `perf` refinement on the one perf-flagged `LOOSE_CASES` entry. Each targets the term that binds once the previous one lands.

**Perf target (all entries)**: `feature_spec.LOOSE_CASES[0]`: x `(1,1,160,1536)` bf16 TILE DRAM-interleaved, W `(1536,4096)` bf8b TILE DRAM-interleaved, y L1-interleaved, `fp32_dest_acc_en=False`, `math_fidelity=HiFi2`, seed 0. Soft precision gate `pcc_threshold = 0.9998`. `sol_ns_blackhole = 16,018` (DRAM-bound). Phase 0 measures 26.6–27.0 µs device kernel time (1.66–1.69× SOL).

**Guard set**: the op has exactly one kernel path × layout × placement, so the config-spanning guard set is the flagged cell itself, plus the acceptance suite, the precision baseline and `eval/golden_tests/wq_b_linear/test_regression.py` (bit-identical repeated calls, a prompt MUST).

**Measurement tool**: `tests/ttnn/unit_tests/operations/wq_b_linear/_op_bench.py` under `scripts/run_safe_pytest.sh --profile`. `WQB_VARIANT` ∈ {full, no_compute, no_w_read, no_compute_no_w_read, sync_only, w_only, compute_only}; the stubs are text substitutions on the shipped kernels, so update them if a refinement renames the stubbed lines. Phase 0 ablation table: `verification_report.md` § Performance.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16] (x)
- **SUPPORTED weight_dtype**: [bfloat8_b]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED fp32_dest_acc_en**: [False]
- **SUPPORTED shape-derived axes**: none (`INPUT_TAGGERS = {}`; one shape)
- **Cores**: full grid (110 on p100a); n split row-wise, 18 cores × 2 columns + 92 × 1. x read once and multicast (8 rotating injectors, Flag + handshake); W streamed per core, whole-K resident
- **Compute config**: caller's `ComputeConfigDescriptor`, default HiFi2 / 16-bit DEST; packer L1 accumulation
- **Golden baseline**: 2 / 2 registry cells passing (plus 1/1 regression), 0 loud categories; PCC 0.999967; 27.0 µs device

### [x] Refinement 1 — Take the x broadcast off the critical path

**Type**: perf

**Goal**: the flagged case is bound by the x broadcast. x broadcast + y store alone is 25.4 µs of the 26.6 µs op. The one-page sync-only floor is 14.7 µs: 8 serialized Flag+handshake rounds at ≈ 1.7 µs each before any bytes, plus ≈ 10.7 µs of bytes at ≈ 34 B/cycle. The W stream alone (20.8 µs) and compute are hidden under it. Shorten the round chain until the x path sits below the W stream. Knob-turn levers, in order of expected yield:
- **Decouple the multicast round from the compute K-block.** Today round size = `block_m_tiles · k_block_tiles`, so `K_BLOCK_TILES` sets both the matmul quantum and the number of handshakes. Add a single-source host knob (rounds of `x_round_k_blocks` compute K-blocks) so fewer, larger rounds amortize the handshake while compute keeps its K-block. The writer pushes per compute K-block inside a round so compute still starts early. Trade-off: compute cannot start until round 0 lands.
- **Protocol / sender topology the helper permits.** `mcast_impl.cpp:459` forces a handshake for rotating senders with `Flag`; a single fixed injector with `Flag` may run without it (still linked). Also try `McastConfig.handshake_cores` restricted to a subset, and the injector count / row placement. Every variant must keep write-once slots.
- **Injector x prefetch on NoC0 vs NoC1** (catalog `noc_placement`).

Re-tune `K_BLOCK_TILES` / `W_READS_IN_FLIGHT` after the decoupling, because their Phase 0 optima were taken with the round count tied to them. Patterns: `ttnn/ttnn/operations/examples/master.md` entries `shared_input_reuse`, `mcast_topology`, `noc_placement`, `compute_block_size`. No SUPPORTED change.

**Verifier notes**: must run first; nothing downstream can be measured while x binds (no_w_read ≈ full). The success shape: full ≈ the W-only ablation (≈ 21 µs) and `no_compute_no_w_read` well below `w_only`. If the knob-turns cannot bring the x path below the W stream, ship `[~]` with the measured numbers and file **Refinement 1b**: an x-topology scheme-change. Two candidates: the design's R5 (`mn_2d_w_bcast_along_m`: two 1-D `ttnn.Mcast` channels at disjoint semaphore ids, W multicast along m), or concurrent banded x injectors, each multicasting to a sub-rectangle and paying one extra x DRAM read per band (+491,520 B ≈ +1.1 µs DRAM SOL each). The `l1_ledger.md` R5 deferral premise ("cross-core tier does not bind") is already falsified by the Phase 0 ablation.

**Done when**: measured device-ns on the flagged case improves over the 26.6 µs Phase 0 baseline, with the x-only ablation (`no_compute_no_w_read`) recorded below `w_only` (or `[~]` + 1b as above). `pcc_threshold` 0.9998 holds, the golden suite and acceptance suite are green, and `test_regression.py` stays bit-identical.

**Outcome**: full op 26.53 → 23.85 µs (median of 9, `_op_bench.py --profile`; 1.66× → 1.49× SOL). x-only (`no_compute_no_w_read`) 25.4 → 16.0 µs, now below `w_only` (21.0 µs). The full op sits 0.9 µs above the W + compute ablation with a trivial x (`w_compute`, 22.95 µs), so x is off the critical path. Levers: (1) round size decoupled from the compute K-block (`X_ROUND_K_BLOCKS`). (2) One fixed-sender Flag channel per round (`X_MCAST_CHANNELS="per_round"`): each fires once per launch, so there is no ready handshake. The rotating handshake cost ≈1.5 µs/round because 109 ready atomics converge on the next injector. (3) Injectors on a diagonal over the last rows (`X_INJECTOR_PLACEMENT`): the x path drops 17.7 → 16.6 µs because last-row injectors share row links for their prefetch. (4) `K_BLOCK_TILES` re-tuned 6 → 3 (16 K-blocks, 4 rounds): 25.1 → 23.8 µs; PCC improved 0.999968 → 0.999972. `W_READS_IN_FLIGHT` re-checked; 1 stays best. Measured and parked: concurrent injectors (`X_ROUNDS_CONCURRENT=True`) are correct but run 39–88 µs and erratic, with both unlinked Counter and linked Flag. Not built: x prefetch on NoC0. BRISC and NCRISC share read cmd buffer 1 in dedicated-NoC mode, so it needs a reader-side prefetch with a handoff; placement already removed most of the prefetch contention. Bottleneck now: the W stream on the 18 two-column cores (NCRISC end 21.4 µs) plus their compute tail (TRISC2 end 22.7 µs) — Refinement 2's scope. Remaining in the x path: 8.4 µs `sync_only` floor (launch + y store + 4 flag rounds); prefetch ≈2.4 µs over `mcast_only` (13.6 µs); multicast at ≈50 B/cycle.

### [x] Refinement 2 — W DRAM stream co-tune toward the DRAM SOL

**Type**: perf

**Goal**: once x is off the critical path, the W stream binds. It runs 20.8 µs alone vs the 16.0 µs DRAM SOL (1.30×). Per-core NCRISC end times show the cause: max 20.86 µs vs p50 16.68 µs (max/p50 1.25), and the 20 slowest cores are the 18 two-column cores of core_group_1 (logical rows 0–1, 104 KB of W each vs 52 KB). After the one-column cores finish, the 18 heavy cores drain the tail with too few reads in flight to keep DRAM busy. Knob-turn levers on the block surface the planner exposed:
- **Per-group `w_reads_in_flight`** (deeper on core_group_1, so the heavy cores take a larger DRAM share early); it is already a CT arg per kernel group.
- **Even split**: core count as a knob, e.g. 64 cores × 2 columns. Same busiest-core bytes, no light/heavy tail, 46 fewer x receivers. This is the design's "grid synchronization" perf lamp.
- **Spread core_group_1** across the grid instead of the first two rows (design perf lamp; catalog `noc_placement`).
- **K-start rotation / bank stagger** (catalog `bank_stagger`, `trid_double_issue` depth 2–4). Rotating each core's K start is algebraically legal and stays deterministic per core. It needs x round `r` resident before K-block `r` is consumed, which Refinement 1 makes cheap.
- **Compute tail on two-column cores**: subblock (5,2) via `dst_full_sync_en` (catalog `matmul_output_subblock`) and the last-K-block size, if the tail after the last W byte becomes visible.

Patterns: `ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.

**Verifier notes**: depends on Refinement 1; until x is off the path these levers do not show in full-op time. Measure each lever on both the full op and the `w_only` ablation, plus the per-core NCRISC max/p50 (`generated/profiler/.logs/profile_log_device.csv`, `NCRISC-KERNEL` end per core). Record the post-refinement max/p50; Refinement 3 is gated on it. Several T2 knobs in one phase is the intended size.

**Done when**: measured device-ns on the flagged case improves over the post-Refinement-1 baseline, and the W-reader max/p50 is recorded. `pcc_threshold` 0.9998 holds, golden + acceptance are green, `test_regression.py` stays bit-identical.

**Outcome**: full op 23.85 → 22.39 µs (median of 9, 1.49× → 1.40× SOL); `w_only` 21.0 → 19.8 µs; W-reader max/p50 = 1.045 in the full op (1.028 `w_only`). The full op is 13% above `w_only` (not within the ≈5% Refinement 3 gate). Levers kept: even split `CORE_N_TILES=2` (65 cores) and bank-strided column pairs `W_BANK_COALESCE` (one 2176 B single-bank read per k row). Measured, no win: per-group depth `W_READS_IN_FLIGHT_G1`, `CORE_PLACEMENT=spread`, `CORE_N_TILES=3` (compute binds at 27.1 µs), `K_BLOCK_TILES` 2/4, depth 2. Parked at 0: NoC1 share of W (`W_NOC1_K_BLOCKS`): `w_only` 20.4 → 17.9 µs, but in the full op the x broadcast holds BRISC/NoC1 until about 13–15 µs, so it loses. Bottleneck now: the W stream at ≈343 GB/s, capped by NoC0 link load next to the DRAM endpoints, not by imbalance or DRAM banks; on NoC1 alone it takes 36 µs. After it come a 1.3 µs last-block compute tail and a 0.8 µs y store. Next: take x off NoC1 or shrink it, so the parked NoC1 W share can run; or overlap the y store with the last K-block (needs an Out-target + interm reload variant of `matmul_block`, since the Interm path self-drains the y CB). Neither was built here: the first changes the x topology (Refinement 1's domain), the second adds a CB and a different rounding path for ≤0.5 µs.

### [x] Refinement 3 — Balanced W split: n + K-remainder with deterministic combine (design R4)

**Type**: perf

**Goal**: remove the structural per-core W imbalance that a pure n-split cannot avoid (128 columns on 110 cores ⇒ 18 cores read 2× W). Implement the design's deferred regime **R4 `n_split_k_remainder_combine`** (`op_design.md` § Regimes, and its "R4 Tensix-to-Tensix contract"):
- every core owns one full column `(5, 1, 48)`;
- the 18 remainder columns are cut into K-slices spread over the other cores;
- each slice core packs its `(5,1)` partial into a dedicated partial CB and unicasts it into the column owner's landing slot `s` (slot = K-slice index), then increments the owner's counter semaphore;
- the owner sums its own partial and the slots in ascending `s` order.

Busiest-core W drops ≈ 104 KB → ≈ 61 KB, and busiest-core compute drops 480 → 280 tile matmuls. The x broadcast and the W reader are reused unchanged, parameterized by an extra `(k_start, k_count)` slice. Patterns: `ttnn/ttnn/operations/examples/master.md` (`tensix_all_reduce*` for fixed-order cross-core combines). No SUPPORTED change.

**Verifier notes**: scheme-change. The combine topology is the work, so it stands alone. It runs after Refinement 2 because a cheap knob (even 64×2 split, per-group depth) may already remove the tail. If Refinement 2's recorded W-reader max/p50 is ≤ 1.05 and the full op is within ≈ 5% of the `w_only` ablation, there is no imbalance left for R4 to remove: record that measurement and close this entry without building R4. Prompt MUST: the combine order must be fixed (bit-identical across calls; `test_regression.py`). Precision margin is 1.7e-4 PCC: keep the partials in a format that does not add a bf16 rounding pass the Phase 0 path does not have, and re-run `test_wq_b_linear_precision_baseline.py`. Update `l1_ledger.md` with the partial CB and landing slots.

**Done when**: measured device-ns on the flagged case improves over the post-Refinement-2 baseline (or the no-imbalance measurement above is recorded). `pcc_threshold` 0.9998 holds, golden + acceptance + precision baseline are green, and `test_regression.py` stays bit-identical.

**Outcome**: R4 built (correct, bit-identical, PCC 0.999973 / relative RMS 0.0076) and parked behind `K_REMAINDER_SPLIT = False`. Default path 22.40 µs; best R4 22.70 µs (3 A/B runs each, slice 9 / combine 6). R4 cuts busiest-core W 104 → 62 KB and compute 480 → 285 tile-matmuls, and shortens the W-end → compute-end tail ≈ 1.35 → 0.6 µs. But 110 cores of single-tile 1088 B W reads finish ≈ 0.8 µs later (NCRISC p50 20.2–20.6 µs) than Refinement 2's 65 cores of bank-coalesced 2176 B reads (19.3 µs), so the net is +0.3 µs. Bottleneck: W transfer efficiency on the NoC0 links next to DRAM, not imbalance; Refinement 2's max/p50 of 1.045 already left none. Next: a coalesced K-split (core pairs split K over a bank-strided column pair, keeping 2176 B reads while halving per-core compute), and overlapping the y store (≈ 0.9 µs after compute). Neither was built: both change the scheme beyond R4's contract.
