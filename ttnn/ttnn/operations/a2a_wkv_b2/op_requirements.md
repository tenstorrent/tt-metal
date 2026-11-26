# Operation Requirements: a2a_wkv_b2

## Definition
- **Formula**: for chip r of an n-chip ring along `cluster_axis` (n = 4), `out_r[0, h, T·j + t, :] = attn_j[0, (H/n)·r + h, t, :] @ W[0, (H/n)·r + h, :, :]`, h ∈ [0, 24), j ∈ [0, n), t ∈ [0, 160)
- **PyTorch Reference**:
  ```python
  def ref(ring_attn, w):  # ring_attn: list of n per-chip [1, H, T, L]; w: [1, H, L, V]
      n, hpc = len(ring_attn), ring_attn[0].shape[1] // len(ring_attn)
      return [torch.cat([a[:, r*hpc:(r+1)*hpc] for a in ring_attn], dim=2) @ w[:, r*hpc:(r+1)*hpc]
              for r in range(n)]
  ```
- **Import Path**: `from ttnn.operations.a2a_wkv_b2 import a2a_wkv_b2`
- **Function Signature**: `a2a_wkv_b2(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, cluster_axis: int = 1, num_links: int = 2, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Why the queue is all perf**: `TARGET` (feature_spec.py) has one value per axis and Phase 0 `SUPPORTED` already equals it, so `TARGET − SUPPORTED = ∅`. There are no generality refinements, and the 2:1 cadence collapses to perf-only phases (feature_spec: "There is nothing to refine toward; the op's job is to be correct and as fast as possible").
>
> **Every perf refinement targets the flagged case**: `feature_spec.LOOSE_CASES[0]` (`# attention: PERF FOCUS`). Its full contract:
>
> - attn: per chip [1, 96, 160, 512], bf16, ROW_MAJOR, DRAM interleaved;
> - W: [1, 96, 512, 128], bf8_b, TILE, DRAM interleaved;
> - out: [1, 24, 640, 128], bf8_b, L1 interleaved;
> - config: HiFi2, `fp32_dest_acc_en=False`, `cluster_axis=1`, `num_links=2`, on the 2×4 `FABRIC_2D_TORUS_X` LoudBox;
> - soft gate: PCC ≥ 0.9995;
> - speed of light: `sol_ns_blackhole` = 43,776 ns.
>
> All of it is already in SUPPORTED. It is the only cell, so it is also the whole guard set.
>
> **How to measure.** Measure per-call time with `tests/ttnn/unit_tests/operations/a2a_wkv_b2/test_a2a_wkv_b2_trace_timing.py` (trace replay, µs per call). Get per-role end times with the `TT_METAL_DEVICE_PROFILER=1` probe `tests/.../a2a_wkv_b2/probes/test_probe_profile.py`, parsed as in `verification_report.md` § Performance. Run the golden suite with `EVAL_CAPTURE_L1=0`; the default L1 capture disables the program cache and breaks the trace test — a harness issue, see the report.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16] (attn)
- **SUPPORTED layout**: [ROW_MAJOR_LAYOUT] (attn)
- **SUPPORTED op-specific axes**: weight_dtype ∈ {bfloat8_b}, fp32_dest_acc_en ∈ {False}
- **SUPPORTED shape-derived axes**: none (`INPUT_TAGGERS = {}`; one shape)
- **Cores**: 96 worker cores (one head each) + 4 fabric_tx cores (one per (direction, link) stream) per chip, on all 8 chips
- **Compute config**: `default_compute_kernel_config()` = HiFi2, fp32_dest_acc_en=False; any fidelity accepted
- **Golden baseline**: 7 / 7 passing (2 supported_pass + 5 regression), 0 loud categories
- **Perf baseline**: 131.6 µs per call (trace replay) = 3.0× SOL. Critical path: the fabric_tx neighbour streams end at about 120 µs, against about 65 µs for the workers.

### [x] Refinement 1 — Speed up the fabric_tx sender path (perf-flagged profile)

**Type**: perf

**Goal**: speed up the perf-flagged `LOOSE_CASES[0]` toward its `sol_ns_blackhole` = 43,776 ns, with PCC ≥ 0.9995 holding. No SUPPORTED change.

The measured critical path is the `send` stage (`pipeline_spec.json`): fabric_tx reader `pull_block` plus writer `send_block`.

- The neighbour streams (link 0) send 120 packets of 4,352 B at about 1 µs each (≈ 4.3 GB/s per sender against a 50 GB/s channel). They finish at about 104–120 µs.
- All 96 workers have published their last block by about 65 µs.

Shorten the per-packet serial chain and balance the per-sender load. Levers are knob-turns on the surface the planner exposed, plus sync batching (T1/T2 in `ttnn/ttnn/operations/examples/master.md`: double-buffer / trid double-issue / keep bytes in flight):

- **Stop re-polling.** The tx reader polls the worker's `sem_blocks_done` with a remote read and a barrier before *every* block. The counter is monotonic, so remember the last value seen per head and poll only when it is short. Alternatively, issue the poll and the pull of the next block before waiting on the current one.
- **Keep more bytes in flight.** Raise `TX_DEPTH`, and batch the pulls of several heads of one mt round behind one barrier instead of one barrier per packet. Recycle a slot on `flush` only when that slot is about to be reused, not after every packet.
- **Bigger packets.** `tiles_per_packet` is capped at 4 by the scatter-chunk limit. Measure bank-contiguous multi-tile packets (the `op_design.md` packet-size lamp) up to `get_tt_fabric_max_payload_size_bytes()`.
- **Per-sender balance.** Neighbour senders send 120 packets and antipode senders 60. Rebalance heads between the two links of a direction so the two senders' packet counts even out, keeping each link direction's *bytes* within its share (the prompt's ring rule).

**Calculator**:
- Command: `python3 .claude/tools/perf_calculator/opgraph.py ttnn/ttnn/operations/a2a_wkv_b2/perf/ring_problem.json ttnn/ttnn/operations/a2a_wkv_b2/perf/ring_builder.py --opt mt_block=1 --opt cores=96 --opt antipode_split=True`.
- Predictions:

  | Per-sender rate | Predicted per call |
  |---|---|
  | Rated 50 GB/s per channel | 51.7 µs (DRAM-bound; link 21% busy) |
  | Measured ≈ 4.4 GB/s (`line_rate_gbs` edited) | 148.7 µs (link-bound, 83% busy) |
  | Measured on device | 118–131 µs |

- Rank: the current design is the model's rank 1 at the rated rate. The gap is the sender throughput the model assumes, not the split.
- Roofline: ≈ 49.5 µs (`op_design.md` Bounds).

**Verifier notes**: the `send` stage is the only stage currently past the worker stream end, so it gates every other perf lever. Refinement 2 (reader) shows no per-call win until this lands.

Re-profile after each lever: the tx tail is the end of the last fabric_tx BRISC minus the end of the last worker compute. Keep every handshake invariant intact:

- sem resets right after each wait;
- `done` armed with flush after the stream's last data packet;
- no data packet before `sem_ready`.

`test_regression.py` (back-to-back bit-identity, trace replay) is the guard for those.

**Done when**: measured per-call time on the flagged shape (trace timing) improves, and the profiled tx tail past the worker stream shrinks. Also required:

- PCC ≥ 0.9995;
- golden suite green (with `EVAL_CAPTURE_L1=0`) and the acceptance tests green;
- back-to-back and trace replay still bit-identical.

The guard set is this single cell, so no regression on it is the no-regression requirement.

**Outcome**: 131.5 → **102.5 µs per call** (trace replay). The tx tail past the worker stream shrank from about 54.5 to about 25 µs (last tx BRISC 120.6 → 90.8 µs; workers end at about 65.5 µs). PCC 0.999907; golden 7/7; regression bit-identical.
- **What landed**: batched pull (one barrier per `TX_BATCH = 8` packets) with cached monotonic polls, and the writer twin (one flush + pop per batch). `TX_BATCH` 4–8 tie; 12/24 are slower, and a deeper `TX_BUFFERS` is flat.
- **Bottleneck now**: the fabric send itself, about 0.32–0.4 µs per 4,352 B packet (≈ 11–13 GB/s per link-0 sender), applied to the data the workers publish late. Writer SEND is 47.5 µs of the 120-packet stream; the pulls are 4.9 µs.
- **Measured and not kept**:
  - ready-order send: 109 µs, and its metadata path costs about 11 µs in strict order;
  - tx pull on NoC 1: wrong output, undiagnosed, little headroom.
- **Not available**: bigger packets (the fabric payload is fixed at 4,352 B by the fabric config).
- **Next**: an earlier worker publish (Refinement 2), then more than one sender per link via the mux (Refinement 3) to parallelize the per-packet send cost. Not done here because both are out of this refinement's scope.

### [ ] Refinement 2 — Speed up the worker DRAM read stream (perf-flagged profile)

**Type**: perf

**Goal**: same flagged shape and goal (`LOOSE_CASES[0]`, toward 43,776 ns, PCC ≥ 0.9995). No SUPPORTED change.

The `read_w` + `read_attn` stages (worker reader, NCRISC, NoC 0) finish at about 59.6 µs. That is 36% above the DRAM SOL (43.8 µs) and above the planner's NoC 0 delivery estimate (54.2 µs).

Levers from `op_design.md`'s perf lamps and the master.md data-movement entries (`noc_placement`, `trid_double_issue`, `double_buffer`):

- **Split-NoC reads.** Issue alternate `load_attn_block` / `load_weight_block` requests on NoC 1; the worker writer's NoC is idle for 72 of 96 heads. `estimate.py` ranks this first (attn 32.3 vs 38.0 µs, W 13.9 vs 16.2 µs) but carries a measured 17–26% caveat on 96 cores, so build it and measure. Do not adopt it on the estimate.
- **Worker placement.** Place workers in a DRAM-bank-spread order instead of row-major.
- **Read depth and batching.** Co-tune `ATTN_RM_DEPTH` and the attn batch (one barrier per tile row today).
- **Overlap the W load with attn.** Start the first attn block's reads while W is in flight; W currently has its own barrier before any attn read.

**Calculator**:
- Command: `python3 .claude/tools/perf_calculator/estimate.py sweep ttnn/ttnn/operations/a2a_wkv_b2/perf/sweep_attn.json` (and `sweep_w.json`), recorded in `dm_estimates.json` ids eb378294 / d1adf26d.
- Current (NoC 0, batch 32/64): attn 38.0 + W 16.2 = 54.2 µs.
- Candidate (split NoC, batch 16/64): 32.3 + 13.9 = 46.2 µs, rank 1.
- The two are within the tool's stated error band for this pattern, so treat this as a tie to be decided by measurement.
- Roofline: ≈ 49.5 µs.

**Verifier notes**: land after Refinement 1. While the `send` stage ends 55 µs after the workers, a faster reader moves the worker end but not the per-call time. Measure the reader end time in the profile as the primary signal, and the per-call time as the secondary one.

**Done when**: measured worker reader end time improves on the flagged shape and per-call time (trace timing) does not regress (it improves if `send` is no longer critical). Also required: PCC ≥ 0.9995, golden and acceptance green, regression tests bit-identical.

### [ ] Refinement 3 — Remaining tail: sender topology and last-round drain (perf-flagged profile)

**Type**: perf

**Goal**: same flagged shape and goal. No SUPPORTED change.

Close what remains between the per-call time and the worker DRAM stream after Refinements 1–2. This is the ⭐⭐⭐ T3 phase, because it changes stage topology:

- **More than one sender per link.** Several fabric_tx cores share one (direction, link) router channel through `MuxConn<N>` (`ccl_helpers_dataflow.hpp`; regime R6 in `op_design.md`, currently deferred). This parallelizes per-packet sender overhead if a single sender per link still cannot keep up with the worker stream. This is a scheme-change, the new stream→link mapping. It only maps several streams onto one link through the mux policy; the tx kernel stays the same apart from the `ConnT` template parameter.
- **Last-round tail.** Compute and publish end about 5 µs after the reader; the design estimates about 2.6 µs. Measure:
  - a read order that ends on a smaller block;
  - local heads' last round scheduled last;
  - a measured co-tune of `MT_BLOCK`. The opgraph ranks `mt_block = 5` behind (70.2 vs 51.7 µs), so it is only worth a measurement if compute shows on the critical path.

**Calculator**: the mux is not modelled by `opgraph.py`'s ring builder (one sender per link). The tail lever is in `opgraph`: the drain of the last mt round is ≈ 2.6 µs of the 51.7 µs estimate. Roofline ≈ 49.5 µs.

**Verifier notes**: depends on Refinements 1 and 2 (it measures what they leave). Headroom gate: if, at the start of this refinement, the profiled per-call time is already within about 3 µs of the worker reader end plus the design drain, there is no headroom left on this region. Record the measurement and mark `[x]` without building the mux.

**Done when**: either (a) measured per-call time on the flagged shape improves, with PCC ≥ 0.9995, golden and acceptance green and regression bit-identical; or (b) the headroom gate above is shown by measurement to be already met.
