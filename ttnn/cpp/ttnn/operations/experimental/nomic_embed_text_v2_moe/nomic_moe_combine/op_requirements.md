# Operation Requirements: nomic_moe_combine

## Definition
- **Formula**: `out[t, :] = round_bf16( b + W[0,0,t,0] · Y[0,0,I[0,t],:] + W[0,1,t,0] · Y[0,0,I[1,t],:] )` for
  t in 0..T'-1, with one fp32 accumulation and one rounding. Padding tokens t ≥ T have I = 0 and W = 0, so their rows
  are exactly `b`.
- **PyTorch Reference**:
  ```python
  def reference(values, bias, combine_index, combine_weights, tokens):   # eval/golden_tests/nomic_moe_combine/helpers.py
      rows = values.reshape(-1, 768).double()
      out = bias.reshape(1, 768).double().repeat(tokens, 1)
      for k in range(2):
          out += combine_weights[k, :tokens].double().unsqueeze(1) * rows[combine_index[k, :tokens]]
      return out
  ```
- **Import Path**: `from ttnn.operations.nomic_moe_combine import nomic_moe_combine`
- **Function Signature**:
  ```python
  nomic_moe_combine(expert_rows: ttnn.Tensor,      # (1, 1, R, 768) bf16 ROW_MAJOR interleaved, 2240 <= R <= 8512
                    combine_index: ttnn.Tensor,    # (2, T') uint32 ROW_MAJOR interleaved DRAM
                    combine_weights: ttnn.Tensor,  # (1, 2, T', 1) float32 TILE interleaved DRAM
                    bias: PreparedCombineBias,     # prepare_combine_bias(bias, device)
                    *, num_tokens: int, batch: int = 1,
                    memory_config: ttnn.MemoryConfig | None = None,
                    compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None) -> ttnn.Tensor  # (B,1,T/B,768) bf16 TILE
  ```
  Also exported: `prepare_combine_bias`, `default_compute_kernel_config`, `circular_buffer_bytes`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED rows_buffer**: [l1]
- **SUPPORTED output_layout**: [norm_layout] (the norm's ND-sharded L1 layout, read off `nd_shard_spec`)
- **SUPPORTED shape-derived axes**: window ∈ {aligned, ragged}, batch ∈ {single, multi}
- **Regime**: `owner_computes`. Each core of the output's shard grid computes its own shard and packs into it
  (zero-copy output CB). Choice 0 rows are read on RISCV_0/NOC_0 and choice 1 rows on RISCV_1/NOC_1. Compute
  realization R2 (tilize to CB, then `mul_tiles_bcast_cols` accumulated on a bias row-broadcast seed in fp32 DEST).
- **Cores**: the norm shard grid: 64 (1k), 96 (2k / 4k) of 110
- **Compute config**: HiFi4 + fp32_dest_acc_en (enforced by validate: precision contract)
- **Golden baseline**: 28 / 98 cells passing, 70 xfail_expected, 0 loud (per verifier CLI)
- **Perf baseline** (device kernel, µs): 8x512 14.8, 4x512 8.6–8.8, 2x512 6.8–7.3. All accept bars met; no stretch
  bar met.

The perf-1 target contract (8x512, R = 8416, bf16, rows L1, norm_layout, HiFi4 + fp32 DEST) is fully inside Phase 0
SUPPORTED. No generality refinement is needed to unlock it, so the anchor rule is already satisfied. The only
generality work left is the placement gap below. It is one refinement, so the remaining phases are perf.

### Deferred (nomic-opt, 2026-11-04): Placement generality: expert_rows in DRAM, interleaved outputs (L1 and DRAM)

> Not queued. In this op's range (1000 to 4096 tokens) the model writes the combine's output in the norm's layout and keeps w2's rows in L1, and it keeps the incumbent routed_combine wherever it asks for anything else; the run's scope is the model's configs only (no generality work). The refinement runner skips this heading on purpose.

**Goal**: add `"dram"` to `SUPPORTED["rows_buffer"]` and `"l1_interleaved"`, `"dram_interleaved"` to
`SUPPORTED["output_layout"]`, natively in the kernels, for every window/batch cell. That is the whole remaining
`TARGET − SUPPORTED` gap: the 70 cells now `xfail_expected` (rows dram x 3 output layouts x 14 shapes, plus rows l1 x
2 interleaved layouts x 14 shapes).
- **rows_buffer = dram** (op_design regime `owner_computes_dram_rows`): only the reader's row-address function
  changes. Replace the 110-bank L1 arithmetic with the DRAM interleaved bank/offset mapping (bank count and page
  stride as CT args from the tensor, never hard-coded). Keep the stateful one-packet issue, both readers on both
  NoCs, and the batched rotating-trid ring. Segment offsets `g·W·64` are already 64 B aligned. CBs, compute and the
  split are unchanged.
- **output_layout = l1_interleaved / dram_interleaved** (regime `interleaved_output`): the output CB stops being the
  shard and becomes a bounded streaming output CB (a tile-row-scaled block, not a whole-op dimension). Tiles go out
  by page id through a `TensorAccessor`. Add a write role to a DM RISC after its row reads are issued, batched one
  write barrier per tile row or coarser, never per tile. With no shard owner, the work split is free: either take
  the norm's plan as a virtual shard spec (as op_design proposes) or spread token-row x hidden-slice blocks over all
  110 cores. Pick by measurement. `circular_buffer_bytes()` and validate()'s budget/predicate must cover the
  interleaved plans.

**Implementation skill**: /interleaved-parallel

**Verifier notes**:
- Bundled because rows-in-DRAM alone is a one-function change, far smaller than a refinement should be. Both
  values are pure data-movement placement changes to the same reader kernel and host plan, with compute untouched.
  There is no dedicated skill for `memory_layout`/placement; `/interleaved-parallel` covers the free work split
  over the grid for the interleaved-output plan. Do **not** use `/memory-layouts` (that is RM↔TILE).
- **No wrapper fallback exists here.** The prompt's MUST rule (one native `generic_op`, never `ttnn.to_memory_config`
  or any other ttnn op on the path) rules out the usual partial-tick escape hatch. A value that cannot be landed
  natively stays out of SUPPORTED and gets a `Refinement 1b`.
- Build it performantly, not as a correctness stub (performance-conformance bar): full grid, both NoCs reading,
  batched writes. These cells are not perf targets, but Refinements 2–3 must not have to rewrite this path.
- The norm_layout path is the perf target. Its zero-copy shard CB and read-only NoCs must stay exactly as they are;
  the new paths are additional regimes selected by the plan, not a rewrite of the built one.

**Done when**: all 70 currently-`xfail_expected` cells pass. `verify_supported` reports 0 supported_fail /
xpass_drift / xfail_wrong_mode. Acceptance, precision baseline and golden regression stay green. The perf cases
show no regression: 8x512 / 4x512 / 2x512 device kernel time within run-to-run noise (±0.5 µs) of 14.8 / 8.7 / 7.0 µs.

### [x] Refinement 2 — Speed up the perf-flagged 8x512 profile: share the index and weight reads across each row group

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags 8x512 (T = 4096, R = 8416, real_text, rows L1, norm_layout, 96 KB reserved)
as the mandatory perf target: incumbent 32.22 µs, accept 20.0 µs, **stretch 13.0 µs**, now 14.8 µs. The G owners of
a row group (G = 3 at 8x512, consecutive row-major cores, not always a rectangle) each read the same index slice and
the same weight faces from DRAM. That is 1.5 MB of weight faces where 0.5 MB is distinct, plus 192 readers' index
reads on 2 DRAM banks. Op_design deferred `group_shared_index_weights` as "off the critical path", but measurement
says otherwise: removing the weight reads takes 8x512 from 14.86 to 12.03 µs (4x512 8.77 → 7.88). The grid's DRAM
burst delays every core's first tile row of L1 row reads. Read each group's index and weight faces once (one
reader, or split 1/G per member) and deliver them to the peers' CBs at the same L1 offsets, with a semaphore in
place of the DRAM wait. This is a reuse-shared operand broadcast across the hidden split: a scheme-change, so it
stands alone. Relevant patterns in `ttnn/ttnn/operations/examples/master.md`: `shared_input_reuse` (⭐⭐⭐: device
win < read-count reduction, the injector serializes), `mcast_topology` (multicast helper, `ttnn.Mcast` +
`McastArgs`, for groups that form a rectangle; unicast otherwise), `bank_stagger` (spread the remaining side reads
over banks). No SUPPORTED change.

**Done when**: build the shared variant and measure device-ns; do not infer its value from the ablation. The 8x512
device kernel time improves on the 14.8 µs baseline, moving toward the 13.0 µs stretch, with the golden precision
bars unchanged (PCC ≥ 0.999995, relative RMS ≤ 0.0019, one-hot exact). 4x512 and 2x512 do not regress and stay
under their 12.5 / 8.5 µs ceilings. The golden suite stays green. No regression across the config-spanning guard set:
one representative per distinct kernel path x placement (norm_layout G=2/r=1 and G=3/r=4; the rows-in-DRAM and
interleaved-output placements were deferred with Refinement 1, so there are none to guard).

**Outcome**: built `group_shared_index_weights`. Per choice, the row group's member 0 reads the index slice and the
weight faces 0/2 from DRAM. It unicasts them into the peers' `cb_index_*` / `cb_weights_raw_*` at the same L1 offsets
and bumps one semaphore per choice once the writes are acked. Peers wait on that count instead of DRAM. DRAM reads at
8x512 fall from 1.7 MB to 0.65 MB. Device kernel time, one fresh-cache run per variant, repeated across 3 sessions:
8x512 14.89 → 12.82–13.57 µs (median ≈ 13.0; with sharing off, 14.79–15.03 in the same sessions); 4x512 8.80 →
8.28–8.55; 2x512 6.88 → 6.83–7.45 (neutral; G=2/r=1 has only 4 row batches per reader, so the delivery hop barely
overlaps). In isolation, shared weights alone give most of the 8x512 win (13.5–13.6) and shared index alone gives
14.4–14.7. A per-choice leader stagger (`STAGGER_LEADERS`) was neutral and is parked at False. The first build bumped
peers' semaphores only at batch boundaries, which made 2x512 jitter to 9.3 µs. The fix polls the forwarding inside
the read waits. Now binding: compute is co-critical (Refinement 3's ablation: ~9.8 µs with every read stubbed). The
peers' first tile row also waits one extra L1 hop for the leader's index. Next: Refinement 3 (compute block surface
and the read/compute hand-off). Splitting the DRAM reads 1/G per member was not tried: it needs one semaphore per
source, and the leader's 4-tile DRAM queue is not what binds now.

### [x] Refinement 3 — Speed up the perf-flagged 8x512 profile: per-tile compute cost and compute/read overlap

**Type**: perf

**Goal**: same flagged target (8x512, stretch 13.0 µs). Compute is co-critical with the reads. With every read
ablated, 8x512 still takes 9.8 µs: the HiFi4 multiply ~4.0 µs (~170 cycles an output tile for two products), tilize
~1.9 µs and the bias seed ~0.8 µs, over a ~3.1 µs fixed floor. Compute also idles until each tile row's 64 rows land.
Co-tune the compute block surface the planner exposed, using op_design's perf lamps:
- L3 `dst_full_sync_en` (8-tile DEST groups, half the per-group inits);
- L4 bias seed as part of the multiply pass instead of its own init per group;
- L7 bias tile format;
- tilize/multiply pass structure (one init for both choices' tilize, already in place);
- realization R1 vs R2 per shape;
- the reader/compute hand-off granularity, so the first tile row starts sooner.

A lower math fidelity (or operand order) is in scope only if it keeps the weights at TF32 and passes the unchanged
golden bars. validate() pins HiFi4 today as the precision contract; relax that only with the measurement in hand.
Relevant patterns in `ttnn/ttnn/operations/examples/master.md`: `compute_block_size` (amortize inits and reconfig
over the block), `eltwise_l1_vs_dest_accumulate` (accumulation already in DEST, keep it), `compute_fusion` (FPU
consumer: do not DEST-reuse to skip L1), `double_buffer` (overlap). These are mostly T1/T2 knob turns, so try
several in one phase. No SUPPORTED change.

**Verifier notes**: runs after Refinement 2, because R2 changes when the first tile row's operands land, which moves
the overlap point this phase tunes. If R2 + R3 together still miss 13.0 µs, the next perf phase should build and
measure `whole_row_remote_write` (W = 24 over 110 cores) or `owner_plus_helpers`. Their "NoC-aggregate-bound"
deferral premise is weakened now that compute is shown co-critical (see verification_report.md).

**Done when**: the 8x512 device kernel time improves on the post-Refinement-2 baseline, measured, with the golden
precision bars unchanged (PCC ≥ 0.999995, relative RMS ≤ 0.0019, one-hot exact, padding = bias, bias multiplicity
1 ± 0.02). 4x512 and 2x512 do not regress and stay under 12.5 / 8.5 µs. The golden suite stays green. No regression
across the config-spanning guard set (one representative per kernel path x placement, as in Refinement 2).

**Outcome**: default math fidelity HiFi4 → HiFi3 (knob `MATH_FIDELITY`). validate() now accepts
`PRECISION_FIDELITIES = (HiFi4, HiFi3)`. The multiply is rows (bf16, srcA) x weights (TF32, srcB), so HiFi3 drops only
the last phase, the low-bits x low-bits product (under 2^-10 of a product, zero for one-hot weights). The precision bars
are unchanged: PCC 0.9999986, rel RMS 0.00168 (same as HiFi4), 98.4% of values within 1 ULP (HiFi4 98.8%); one-hot
exact, padding = bias and bias multiplicity pass. HiFi2 fails (rel RMS 0.0036). Device kernel time, one fresh-cache run
per variant, 2-4 sessions: 8x512 13.0–13.5 → 11.8–12.2 µs; 4x512 8.4 → 7.7–8.0; 2x512 7.0–7.2 → 6.5–7.2 (noise band).
Other levers were measured at HiFi3 and parked as live knobs at their old defaults. L3 `DST_FULL_SYNC`: 14.8 µs (the pack
stops overlapping the math). L4 `BIAS_SEED_SRCB` (seed through srcB, not unpack-to-dest): neutral. DEST group 2:
neutral. R1 `FUSED_TILIZE`: 28.9 µs. Two probes were not kept: matmul diag(w)·R in place of the two ELWMULs (≈ equal),
and the bias as a third K tile (12.6 µs, worse). Now binding (reader-only zones): the compute chain after tile row 0
lands. Row 0 lands ~5 µs in (index forward + start-up read burst), then 4 tile rows x ~2.3k cycles of compute, the same
pace the reads deliver at. In situ a tile row costs ~2.3k cycles vs ~1.6k with the reads stubbed (L1/NoC contention).
Stubbing tilize, the bias seed or the multiply each saves 0.8–1.4 µs. Next: shrink the compute chain's start, for
example by splitting tile row 0 (and the last) into column halves so compute starts on half the bytes. That doubles the
read requests on those rows; with reads at the same pace as compute, it needs its own measurement. Or move to the
verifier's `whole_row_remote_write` / `owner_plus_helpers` schemes, which spread compute over the 14 idle cores. Not
tried here: both are reader/topology rewrites, outside this compute-surface refinement.

### [x] Refinement 4 — Host time per call: cache the descriptor, patch only the addresses

**Type**: perf (host)

**Goal**: cut the Python host time of a call. The model calls the op 6 times a forward, and an untraced forward is
host-bound. Measured 2026-11-04 (nomic-opt, 50 calls at 8x512 after warm-up, no profiler, time until the call
returns): **nomic_moe_combine 212 us a call**, nomic_moe_dispatch (its sibling op, plan and descriptor cached)
53 us, the C++ routed_combine 18.6 us. Target: at most 60 us a call at 8x512, 4x512 and 2x512.
- Build the whole ProgramDescriptor once per plan key: (T, B, R, the output memory config, the rows buffer, the
  compute config, the device). That covers CB descriptors except the output shard binding, kernel descriptors,
  CT args, core lists and semaphores.
- On a call, only rebind what changes: the five buffer addresses (runtime args) and the output tensor's
  shard-backed CB. Do not rebuild Python objects per core.
- Cache validate()'s derived values (the norm-layout plan, the shard-to-core map) by the same key. Keep validate()
  as the first line of the entry point: its checks on the tensors themselves (shapes, dtypes, layouts, buffer
  types) still run every call, cheaply.
- Correctness first. A cached descriptor must never carry a previous call's addresses. The golden suite's
  test_new_tensors_between_calls (three calls, new tensors at new addresses) and the acceptance tests must stay
  green. Measure with a 50-call loop as above and record before and after in the changelog.

**Done when**: host time a call is at most 60 us at 8x512 / 4x512 / 2x512 (or the measured floor of
ttnn.generic_op, recorded, if that is higher), device kernel time unchanged within noise, and the golden suite
green.

**Outcome**: Host time per call (median of 50 calls after warm-up, this box) went 449 / 446 / 335 us -> 32.3 / 32.0 /
27.8 us at 8x512 / 4x512 / 2x512. Device kernel time is unchanged: baseline medians 11.87 / 7.85 / 8.0 us vs
12.0 / 7.87 / ~7.0 (2x512 is bimodal at 6.3-8.2 in both). Caching the descriptor alone gave 86 us. ttnn.generic_op
then cost 53 us of that, because it re-sends every per-core runtime arg on each call (measured on empty kernels at
110 cores: 21 us with none, 49-57 us with 4-26 a core). Moving the per-core values into common-arg tables took
generic_op to 7.5 us. The 32 us now splits as generic_op 7.5, allocate_tensor_on_device 11.5, validate 8 (5 of it
the per-call tensor checks, mostly tuple(shape) at ~1.3 us a tensor), bind 1.6, plus call overhead. Next: cut
validate's shape conversions (compare ttnn.Shape objects to cached ones), or let the caller pass a preallocated
output. Not done because the 60 us target is met at half, and the per-call tensor checks are required.
