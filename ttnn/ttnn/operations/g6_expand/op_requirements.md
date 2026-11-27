# Operation Requirements: g6_expand

## Definition
- **Formula**: `out[i0, i1, h, w] = x[i0, i1, 0, w]` — `x.expand(*sizes).contiguous()` with only dim 2
  expanded; production `x (1,1,1,W) → out (1,1,W,W)`, W ∈ {185, 196, 207, 256, 512}. Bit-exact.
- **PyTorch Reference**:
  ```python
  def pytorch_g6_expand(x, sizes):
      return x.expand(*[int(s) for s in sizes]).contiguous()
  ```
- **Import Path**: `from ttnn.operations.g6_expand import g6_expand`
- **Function Signature**: `g6_expand(input_tensor: ttnn.Tensor, sizes) -> ttnn.Tensor` — `sizes` a
  positional list/tuple of 4 ints (`-1` keeps the dim); out takes dtype / layout / placement of
  `input_tensor`. No `memory_config`, no `compute_kernel_config`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Queue shape (verifier)**: `TARGET − SUPPORTED = ∅` on every axis (dtype, layout, alignment), so
> there are no generality refinements; the 2:1 cadence degenerates to an all-perf queue. Every
> refinement targets the perf-flagged `feature_spec.LOOSE_CASES[0]` — FLUX case 46, x (1,1,1,512)
> bf16 TILE DRAM-interleaved → (1,1,512,512), bit-exact, no compute config. Its full contract is
> already in SUPPORTED (anchor rule met at Phase 0). Goal from the spec: device kernel time
> **≤ 1.35 × 1.14 µs = 1.54 µs** (incumbent 21.57 µs). Baseline: **≈ 2.38–2.51 µs** on 110 cores.
> Guard set (no regression, one per distinct path — all share one kernel path/layout/placement, so
> the guards are the other production widths): W = 256 (≈ 1.70 µs, 64 cores, 1 tile/core),
> 196 / 207 (≈ 1.52 µs, 49 cores, w_non_aligned), 185 (≈ 1.49 µs, 36 cores). Bench:
> `tests/ttnn/unit_tests/operations/g6_expand/test_g6_expand_perf_bench.py` under
> `scripts/run_safe_pytest.sh --profile --no-py-zones`, column `DEVICE KERNEL DURATION [ns]`.
> Bit-exactness is the correctness gate throughout (golden + acceptance incl. special values,
> poisoned padding, repeat determinism, program-cache sequence).

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned} (= TARGET)
- **SUPPORTED op-specific axes**: none (sizes: dim 2 expanded only; DRAM interleaved)
- **Cores**: full grid — column-major `split_work_to_cores(grid, total_tiles, row_wise=True)` (110 cores at W=512)
- **Compute config**: none (pure data movement; no compute kernel)
- **Golden baseline**: 20 / 20 tests passing; verifier 14 supported_pass, 0 loud (per `verifier_report.json`)

### [x] Refinement 1 — Co-tune the cheap knobs on the perf-flagged W=512 profile

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` (W=512, PERF FOCUS, goal ≤ 1.54 µs, gate bit_exact) runs at
≈ 2.4 µs. Profile (p50/max cycles over 110 cores): fetch 615/832 → replicate 432/720 → writer idle
in `G6E_WAIT_BCAST` 1089/1602; writer drain 394/1338 (tail ≈ 3.4× p50 → DRAM-bank write queueing);
op span ≈ 1000 cycles over the per-core p50 (launch skew across 110 cores, ~2.3 tiles/core). Turn the
T1 knobs the design already names as perf lamps, co-tuned, guided by
`ttnn/ttnn/operations/examples/master.md` (`noc_placement`, `bank_stagger`, `split_reader`):
- **grid size** (design lamp "grid synchronization"): expose the `grid` knob as one host constant
  (e.g. a cap on the cores the split uses, still a few rectangular ranges per BB-17) and measure
  e.g. 64 / 32 cores (≥ 4 / 8 tiles per core) — fewer cores also cut the duplicate row-0 fetches;
- **write split** `READER_WRITE_EIGHTHS` re-tuned against the new tiles/core;
- **bank stagger / core order** (`Wt = 16` ⇒ every tile of a column lands on one DRAM bank): rotate
  each core's segment start, or re-try `CORE_ORDER_STRIDE` once the core count changes;
- `REPLICATE_CPU_ROWS` re-checked last.
No SUPPORTED change.

**Verifier notes**: knob-turns only (no new topology) — keep each knob a single host constant as
today. Pick the combination by measurement on W=512 and confirm it on the guard set; a per-width
choice is allowed only if it is a function of the geometry computed in one place and is reported in
`last_plan` (BB-08). Do the critical-path restructure (Refinement 2) separately so the gains can be
attributed.

**Done when**: measured device kernel ns on W=512 improves over the 2.38–2.51 µs baseline, the
golden suite (20/20) and acceptance tests stay green bit-exactly, and no regression on the guard set
(W = 256, 196, 207, 185).
**Outcome**: new `MAX_CORES` / `MIN_TILES_PER_CORE` grid knob (one host helper `_split_core_count`, reported in `last_plan`); co-tuned by a 9-rep-median device sweep on W=512 over cores {16…110} × `READER_WRITE_EIGHTHS` {0,2,3,4,5,6} × `CORE_ORDER_STRIDE` {1,3,7} × `REPLICATE_CPU_ROWS` {2,4,8,16,32}. Winner: `MAX_CORES=80`, eighths=4, stride=1, cpu_rows=8 → W=512 **2.49 → 2.31 µs** median (≈7 %, reproduced in 3 runs; 2.27–2.37 vs 2.37–2.59 range). Guards unchanged byte-for-byte (cap binds only when tiles > 80): W=256 1.73, 196 1.57, 207 1.57, 185 1.52 µs. Bank stagger is a no-op here: Wt=16 is a multiple of the 8 DRAM banks, so every tile of a column (and so every core's segment) is on one bank and rotating a segment start cannot change bank order; aggregate bank load is already balanced. Fewer cores (16–48) never beat ~2.45 µs: the cost is not grid sync alone but per-core fetch→replicate latency before the first write plus write drain — i.e. Refinement 2's critical path. Next: Refinement 2 (shorter fetch→replicate path), then R2 multicast.

### [x] Refinement 2 — Shorten the fetch → replicate → write critical path on W=512

**Type**: perf

**Goal**: on `LOOSE_CASES[0]` (W=512) the writer waits ≈ fetch + replicate before its first write;
replicate (≈ 430 cycles ≈ 0.32 µs) is pure on-core overhead on that path. Restructure the per-column
mechanism so the first output byte leaves earlier, levers (T2) per `master.md`
(`read_coalescing`, `tile_reorder`, `trid_double_issue`) and the design's lamps:
- an output tile is `[F0 F1 F0 F1]` (faces 2/3 = faces 0/1), i.e. two identical 1 KiB halves: build
  only faces 0/1 (16 rows each) and write each output tile as two 1 KiB writes from the same 1 KiB
  source — drops the faces 0/1 → 2/3 local copy and its barrier (and halves the slot), at the cost
  of 2× write requests; measure against 2 KiB writes (`tile_reorder` says whole-page writes win —
  this is the trade to measure);
- design lamp "row-fetch request count": 1 request per column instead of 2 (fetch the first 1 KiB, or
  overlap the two 64 B fetches with replicate of the previous column);
- start writing a column as soon as its first half-tile/rows are ready (finer publish), so writes
  overlap the remainder of the replicate.
No SUPPORTED change; out stays written once per page.

**Verifier notes**: T2 mechanism change inside R1 — reader/writer only, the CB stays the single
in-place slot (update `l1_ledger.md` if its page size/capacity changes, e.g. 1 KiB half-tile pages).
Bit-exactness and padding-independence must hold (only row 0 of faces 0/1 is a source; the fetched
row-1 filler must still be overwritten). Land after Refinement 1 so the core count is fixed when this
is measured.

**Done when**: measured device kernel ns on W=512 improves over the post-Refinement-1 number, golden
+ acceptance green bit-exactly (special values, poisoned padding, repeat, sequence), and no
regression on the guard set (W = 256, 196, 207, 185).
**Outcome**: landed three live knobs in the descriptor: `HALF_TILE=1` (faces 0/1 only in a 1 KiB slot, each page written as 2 × 1 KiB, no faces→2/3 copy), `REPLICATE_SPLIT=1` (BRISC fetches row 0 of face 1 on NoC1 and builds face 1 with RISC-V stores while NCRISC does face 0, local semaphore handoff; gated to the pure-RISC-V build and one block per core), and `REPLICATE_CPU_ROWS=16` (pure RISC-V build, so each column is published as soon as it is built). `ROW_FETCH_REQUESTS` (1 vs 2) and `REPLICATE_NOC_FANOUT` were measured flat or worse and are parked at the old byte-identical defaults (2 / 0). Device kernel time (paired A/B, 5–9 rep medians): W=512 **≈2.33 → ≈2.25 µs** (−3.5 %, consistent over 5 paired runs; first write p50 moved 2450 → 2200 cyc); guards improved: W=256 1.73 → 1.52, W=207 1.58 → 1.42, W=196 1.57 → 1.41, W=185 1.53 → 1.37 µs. What binds now is the write drain. A whole-op ablation (fetch + replicate both stubbed, sync kept) gives a **write-only floor of ≈1.79 µs** on W=512: 512 KiB in ≈1900 cyc ≈ 270 B/cyc, with the tail cores stalled in write issue/barrier. The 1.54 µs goal is therefore below the floor for this write pattern. The remaining pre-write path is the row-0 fetch latency (≈500 cyc) plus the halved replicate (≈230 cyc). Measured costs: RISC-V stores ≈1.8 cyc/word; a local NoC copy ≈25–30 cyc per issue plus ≈150 cyc per barrier, so more NoC fan-out loses. Next: raise write throughput (where the writes enter DRAM and how they are routed on the NoC — every column sits on one bank since Wt=16) and cut fetch contention (Refinement 3 multicast). Neither belongs in this mechanism-only refinement.

### [ ] Refinement 3 — Column-group multicast of the broadcast tile (regime R2) on W=512

**Type**: perf

**Goal**: build the design's deferred regime R2 and keep it only on a measured win on
`LOOSE_CASES[0]` (W=512). Today every (core, column) pair fetches row 0 of x from DRAM — ≈ 126 small
fetch pairs on the few banks holding x's 16 pages, queuing behind out's writes (RB-010; the spec's
soft rule "read each shared tile of x once per call or per group of cores"). In R2 the first core of
each consecutive run of cores sharing a column (column-major order makes it consecutive) fetches +
replicates once and multicasts the 2 KiB broadcast tile into the same `cb_bcast_tiles` slot of its
group; the writer side is unchanged. Pattern: `master.md` `shared_input_reuse` (T3) and
`mcast_topology`; mechanism: `ttnn.Mcast` + `attach()` host-side, `McastArgs` `sender(noc).send()` /
`receiver(noc).receive()` (`mcast/kernel/mcast_args.hpp`). No SUPPORTED change.

**Verifier notes**: scheme-change (new cross-core topology: injector + broadcast + semaphore), so it
stands alone. Its value depends on the core count chosen by Refinement 1 (fewer cores ⇒ fewer
duplicate fetches ⇒ less to win), hence it comes last. Validate by building the variant and
measuring device-ns — not by inferring from a fetch-removal ablation. Groups must remain few
rectangular ranges (BB-17); every semaphore/multicast must be complete before the kernels return
(BB-03); reader text must still fit the kernel-config region in the profiler build (BB-04/BB-18).
If it does not win, keep R1 and record the measurement as the reason in the regime table of
`op_design.md` / `l1_ledger.md`.

**Done when**: R2 is built and measured on W=512; it is kept only if device kernel ns improves over
the post-Refinement-2 number with golden + acceptance green bit-exactly and no regression on the guard
set (W = 256, 196, 207, 185) — otherwise the measured numbers are recorded and R1 stays.
