# Operation Requirements: all_to_all_matmul

## Definition
- **Formula**: within each group of G devices along `cluster_axis`, `chunks[p] = chunk(x[p], G, dim=2)`, `a2a[q] = cat_p(chunks[p][q], dim=1)`, `out[q][b, h] = a2a[q][b, h] @ w[0, h]` (head `h = p*Hl + j`)
- **PyTorch Reference**:
  ```python
  def ref(x_stacked, w, cluster_axis):           # x_stacked (R, C, B, Hl, S, K), w (1, H, K, N)
      G = x_stacked.shape[cluster_axis]
      out = {}
      for grp in groups(x_stacked.shape[:2], cluster_axis):
          chunks = [torch.chunk(x_stacked[r, c], G, dim=2) for r, c in grp]
          for q, (r, c) in enumerate(grp):
              out[r, c] = torch.cat([chunks[p][q] for p in range(G)], dim=1) @ w
      return out                                  # per device (B, H, S/G, N)
  ```
  (canonical: `eval/golden_tests/all_to_all_matmul/helpers.py::pytorch_all_to_all_matmul`)
- **Import Path**: `from ttnn.operations.all_to_all_matmul import all_to_all_matmul`
- **Function Signature**:
  `all_to_all_matmul(input_tensor, weight, *, cluster_axis: int, topology=Topology.Ring, num_links: int = 2, memory_config: ttnn.MemoryConfig | None = None, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Perf measurement (fused CCL op)**: whole-op numbers come only from the eval harness — `CCL_PERF_EXPERIMENTS=5 CCL_PERF_OUT=<dir> scripts/run_safe_pytest.sh --no-precompile eval/golden_tests/all_to_all_matmul/perf_measure.py -s` (fastest chip per call is the target; keep/revert by `python -m eval.ccl_perf compare`: fastest lower beyond noise AND slowest not higher beyond noise). Per-stage breakdowns use the off-by-default `A2AMM_ABLATE` stubs (`nowrite,noweight,nocompute,noportread,zones`) via `ccl_perf.run` (see `tests/.../probes/probe_010.py`).

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]; **weight_dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]; **memory_layout**: [INTERLEAVED]; **output_buffer**: [DRAM]
- **SUPPORTED shape-derived axes**: alignment = tile_aligned only
- **SUPPORTED op-specific axes**: cluster_axis ∈ {1}, topology ∈ {Ring}, num_links ∈ {2}
- **Scheme**: regime R1 (heads split across the compute grid = grid minus every chip's 2·num_links port cores; remote rows land zero-copy in the consuming core's L1; weight read while rows are in flight), transport T1 (far quarter direct 2-hop on the routing table's route, per-port ready fence, fused arrival inc per entry)
- **Cores**: 101 compute cores + 4 port cores per chip on QuietBox 2 (Kimi: 96 busy; DeepSeek: 101 with 27 × 2 heads; GLM: 64)
- **Compute config**: `compute_kernel_config` exposed (fidelity / approx ungated); `fp32_dest_acc_en` gated to True
- **Golden baseline**: 16 supported_pass / 132 xfail_expected / 0 loud (per verifier CLI); acceptance + precision 14/14
- **Perf (Phase 0 loose case, Kimi, 7232 B)**: fastest chip 94.4 us, slowest 104.2 us (roofline 63.0, target 84.0); exchange + fence alone 49.3 us

### [x] Refinement 1 — Model configuration: 16-bit DEST, bfloat8_b weight, L1 output

**Goal**: add `False` to `SUPPORTED["fp32_dest_acc_en"]`, `ttnn.bfloat8_b` to `SUPPORTED["weight_dtype"]` and `ttnn.BufferType.L1` to `SUPPORTED["output_buffer"]`. This is the full perf-focus contract (`LOOSE_CASES[0]`: Kimi, HiFi2, `fp32_dest_acc_en=False`, output in L1, bf16 weight) plus the remaining numeric value (bfloat8_b weight). Everything is in-kernel:
- 16-bit DEST: the host subblock rule already picks the 8-tile limit.
- bfp8 weight: `cb_weight` page = bfp8 tile (1088 B). `WEIGHT_BLOCK_BYTES` is in bytes, so `block_n` grows on its own. The unpacker reconfigures between bf16 A and bfp8 weight.
- L1 output: the writer's `TensorAccessor` follows the L1-interleaved output buffer. Add the output's per-bank share to the L1 budget.

Cells that fail out of the box go to `EXCLUSIONS` only if they are structural. An L1 output too large for L1 is skipped by the harness (`eval.oom`), not excluded.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: This single refinement lands every axis value the first perf slot (Refinement 3) needs. After it, all four `LOOSE_CASES` must pass and be measurable by `perf_measure.py`. Build it at the Phase 0 performance bar (grid filled, dataflow batched): perf 3 optimizes exactly this path, not a stub.
- The L1 output lands on every core, port cores included. Check that it does not clash with the CB region or the persistent `landing_scratch` (shard per compute core) on the tightest case (DeepSeek: 400 KiB of CBs + 80 KiB of A per core).
- The verifier already capped `cb_weight` at the blocks a core actually holds (`weight_depth`), which freed 128-192 KiB per core for this.
- Re-check the compute writer NoC choice (NoC0 today, measured for DRAM writes) for L1 writes; keep whichever the harness measures faster.
- Re-run the precision baseline under 16-bit DEST and bf8b; defaults are (0.99, 0.13) for bf16 / 16-bit DEST.

**Done when**: every `fp32_dest_acc_en=False`, `weight_dtype=bfloat8_b` and `output_buffer=L1` tile-aligned golden cell passes (or is harness-skipped as infeasible), all 4 `LOOSE_CASES` pass, and `perf_measure.py` with `CCL_PERF_CASES=all` reports a number for each case with no DISPATCH-BOUND flag.

### [x] Refinement 2 — Non-tile-aligned rows per device (regime R1')

**Goal**: add `"m_non_aligned"` to `SUPPORTED["alignment"]` (`S % 32 != 0` or `(S/G) % 32 != 0`; INPUTS: `(1,24,160,128)` 40 rows per device, `(1,8,96,128)` 24 rows, `(1,8,48,64)` input 1.5 tiles / 12 rows per device). This is the design's regime R1'. A destination's quarter starts at row `q*Mq`, which is not tile-aligned. The **port reader** and the **local-quarter reader** therefore re-tile the quarter's rows before the entry is sent or landed: read the two source tiles a destination tile straddles, copy the shifted face rows into the CB page in L1, and zero the padding rows. Landing, arrival counters, the fence and compute are unchanged (`Mq_t = ceil(Mq/32)`), and the output's padded tile rows are written as computed. No host-side `ttnn.to_layout` / pad / slice wrappers.

**Implementation skill**: /memory-layouts

**Verifier notes**: Standalone because the alignment is structural: rows straddle tile boundaries and the entry's tile content changes. It is not edge-tile masking. It comes before the perf slot because it reshapes the port reader, the TRANSPORT feeder that Refinements 3-4 tune; building it later would rework their result.
- Exchanged rows must still be **bit-exact** (prompt rule): the re-tiling copies stored bytes, and only padding is zeroed.
- `test_row_provenance` style closed-form data is the quickest check that row `q*Mq` lands at row 0 of the destination's first tile.
- The re-tile is an L1→L1 copy per entry on the port core (budget: one extra packet-sized staging page or an in-place shift in `cb_port_chunks`). Keep the port reader batched (one DRAM barrier per batch).

**Done when**: every `alignment=m_non_aligned` golden cell over the SUPPORTED dtype / weight_dtype / DEST / output-buffer values passes, with all prior cells and the regression suite still green.

### [x] Refinement 3 — OVERLAP: hide the local work behind the exchange on the perf-flagged case

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` is the mandatory perf target: Kimi K3.1 `(1,24,640,128) x (1,96,128,512)`, output in L1, HiFi2, `fp32_dest_acc_en=False`, 7232 B packets; roofline 32.3 us, target 43.0 us. This is an **OVERLAP** refinement, ranked first by the measured per-chip breakdown on the same shape (Phase 0 config).
- Full op 94.5 us vs exchange + fence alone 49.3 us, so ~45 us of local work is exposed.
- The weight reads alone add ~20 us, even though they are issued before the arrival wait: the 96 cores' 12.6 MB weight burst contends with the port readers and the router landing writes on DRAM and the NoC, and stretches the exchange.
- The compute + write chain after the last landing adds ~15 us (tail).

Candidate levers (the implementer picks and measures):
- Order DRAM traffic so the port readers' entries (critical path) are served first. For example, stagger or throttle the weight reads, or start them after the ports' first batches.
- Landing granularity: entries of fewer row tiles with an arrival inc each, so the tail shrinks to one row block. This is the design's perf lamp; measure the fused-inc cost.
- Send order biased toward the cores with the most remaining work.
- Use the idle compute cores (Kimi: 5 idle + 24 early-finishing local-head cores) for the last-landing heads' columns: R4 restricted to the tail.

See `ttnn/ttnn/operations/examples/master.md`: `noc_placement` (NoC selection under contention), `double_buffer`, `fabric_all_gather` / `fabric_reduce_scatter` (arrival counters, chunked landing). No SUPPORTED change.

**Done when**: on `LOOSE_CASES[0]` the harness-measured fastest-chip mean improves beyond noise (`eval.ccl_perf compare` verdict win) and the slowest chip does not regress beyond noise. Record the exchange-alone ablation alongside, so the exposed local time per call is reported. The soft precision gate still holds, the golden suite is green, and there is no regression across the guard set: the other three `LOOSE_CASES` (DeepSeek 2-heads-per-core, GLM K=192, Kimi Phase 0 DRAM-out / fp32 DEST) plus one multi-n-block shape and one `m_non_aligned` shape.

**Outcome**: LOOSE_CASES[0] harness fastest / slowest chip 81.2 / 84.6 → **60.5 / 64.6 us** (`compare` win). Guard set also wins: Kimi Phase 0 94.5 / 103.5 → 87.7 / 91.7, DeepSeek 99.5 / 104.3 → 77.1 / 83.3, GLM 79.1 / 83.8 → 59.9 / 64.7 us; not dispatch-bound. Two levers: (1) weight-read throttle `WEIGHT_READ_CHUNK_BYTES` = 16 KiB (one barrier per chunk). The whole 128 KiB block in flight on 96 cores stretched the exchange from 49.3 to 68.9 us. It was NoC contention, not DRAM: the `noportread` ablation was still 77 us, and moving the weight reads to NoC0 gave 93.5 us. 81.2 → 64.1 us. (2) Tail split `TAIL_SPLIT_MAX_HEADS` = 8: the latest-landing remote heads (estimated from the sender order) are halved over N with an idle core or a core whose heads are all local; the primary forwards the landed rows over the NoC and increments the helper's arrival counter. 64.1 → 60.5 us. More than ~8 split heads regresses Phase 0 (HiFi4): the local-head helpers become the new tail. Exposed local time per call, now: full 60.5 vs exchange-alone ablation 50.5 us = **~10 us** (was 31.5 us). Of it, the compute tail is ~5 us (`nocompute` 55.4), the output write ~4 us (`nowrite` 56.6) and the weight reads ~1 us (`noweight` 59.6). Next, untried: a writer-side twin of the throttle (bound the L1-output writes in flight; the writer on NoC1 measured worse, 66.0 vs 64.1 us); a 4-way split of the last one or two landings; per-row-tile landing increments (≤ ~1.5 us by the entry's landing span). The exchange itself (50 us, 2.5x the links roofline) now dominates: that is Refinement 4 (TRANSPORT).

### [ ] Refinement 4 — TRANSPORT: shrink the exchange itself on the perf-flagged case

**Type**: perf

**Goal**: **TRANSPORT** on `LOOSE_CASES[0]` (Kimi, 7232 B, 2 links).
- Measured in-op exchange + fence alone: 49.3 us (Phase 0 config). The links roofline is 20.3 us, and the prompt's standalone 4-ring all-to-all of the same tensor took 53-56 us at 7232 B.
- At this payload the floor is the router's ~235 ns fixed cost per packet (busiest link ~47-51% of 48.5 GB/s). The headroom is therefore in the fixed costs, not in raw bandwidth.

Take the levers from the examples and cite their measurements rather than re-deriving them:
- **The ready fence (~5-6 us).** `fabric_ring_exchange`: `fence_direct` 6.35 vs `fence` 5.09 us; a 1-hop fence (relay-style readies) saves ~1.3 us.
- **Partial packets.** An entry of 20 tiles at 3 tiles per packet is 7 packets, about 5% more packets than full packing. Cut entries at packet multiples, e.g. land A k-major so an entry boundary falls on a packet boundary.
- **Header / flush cadence** on the port sender (`blackhole-ccl-kernels.md` §6.1-6.2).
- **T2 relay** of the far quarter (design regime row): measured equal at 7232 B, so only worth trying if the per-link balance on this board is not what the design assumes.

The fabric config and router packet size MUST NOT change (prompt rule). Measure transport in isolation with the `nowrite,noweight,nocompute` ablation, judged by bandwidth against the bytes sent, and confirm the whole-op win with the harness. No SUPPORTED change.

**Done when**: the exchange-alone ablation on `LOOSE_CASES[0]` drops beyond noise, the harness-measured whole-op fastest chip on `LOOSE_CASES[0]` improves beyond noise with the slowest chip not regressing (`compare` verdict win), the golden suite and regression suite are green, and there is no regression across the same guard set as Refinement 3.

### [ ] Refinement 5 — Local work: weight-read and matmul efficiency on the perf-flagged case

**Type**: perf

**Goal**: after Refinements 3-4, measure the local-only breakdown on `LOOSE_CASES[0]` (ablate the exchange's effect by comparing to the exchange-alone and to the stubbed-weight / stubbed-compute variants). Then speed up whichever local stage still sits on the critical path toward the DRAM roofline (32.3 us: 3.9 MB input + 12.6 MB bf16 weight at 512 GB/s; 447 GB/s achieved).
- **Weight-read efficiency.** Coarser DRAM read transactions (runs of `block_n` consecutive pages per k-row instead of one tile per `noc_async_read`), bank-spread issue order (already rotated per head), and NoC choice per `noc_placement`.
- **Matmul subblock at 16-bit DEST.** `block_m = 5` admits only `out_subblock_h ∈ {1, 5}`; measure 1x8 vs alternatives per `matmul_output_subblock`.
- **Block-size / buffer-depth co-tune** on local-head cores (design perf lamp "whole-head weight block": `block_n = Nt/2` so a local-head core starts multiplying before its whole weight lands). Per the `compute_block_size` floor, chunks are whole tiles, and coarser is better until overlap is lost.

No SUPPORTED change.

**Verifier notes**: It is filed after the cross-device refinements because the Phase 0 breakdown shows overlap, not raw local throughput, as the dominant exposed cost. Re-rank against Refinement 4 if Refinement 3's measurement shows the local stages dominating with headroom against the roofline.

**Done when**: the harness-measured fastest chip on `LOOSE_CASES[0]` improves beyond noise (`compare` verdict win, slowest not regressing), the golden suite is green, and there is no regression across the guard set of Refinement 3.
