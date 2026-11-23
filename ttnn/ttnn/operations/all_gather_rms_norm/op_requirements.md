# Operation Requirements: all_gather_rms_norm

## Definition
- **Formula**: for a `cluster_axis` group of G devices, each holding `x[p]` of shape `(..., M, W_local)`:
  `full = concat_p x[p]` (W = G·W_local);
  RMSNorm: `out = full / sqrt(mean_W(full²) + eps) · gamma`;
  LayerNorm (`norm_type="layer_norm"`): `out = (full − mean_W(full)) / sqrt(var_W(full) + eps) · gamma + beta`.
  Every member of the group returns the same `(..., M, W)` tensor.
- **PyTorch Reference**: `eval.golden_tests.all_gather_rms_norm.helpers.pytorch_all_gather_rms_norm(x_stacked, gamma,
  cluster_axis, eps, norm_type=..., beta=...)` — concatenates the group's slices in group-position order and applies
  `torch.nn.functional.rms_norm` / `layer_norm` over the full W in fp32.
- **Import Path**: `from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm`
- **Function Signature**:
  ```python
  all_gather_rms_norm(input_tensor, *, cluster_axis: int, gamma=None, beta=None, norm_type="rms_norm",
                      epsilon=1e-6, topology=ttnn.Topology.Linear, num_links=None,
                      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
                      memory_config=None, output_tensor=None) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Measuring (perf refinements)**: the eval harness only —
> `CCL_PERF_EXPERIMENTS=5 CCL_PERF_OUT=<dir> scripts/run_safe_pytest.sh --no-precompile eval/golden_tests/all_gather_rms_norm/perf_measure.py -s`
> (`CCL_PERF_CASES=all` for both perf cases); keep / revert with `python -m eval.ccl_perf compare <base.json> <candidate.json>`
> (fastest chip lower beyond noise AND slowest chip not higher beyond noise). Verifier baseline (QB2, 1×4 ring,
> FABRIC_2D_TORUS_XY, 14 KiB payload, 2 links, 5 experiments): **focus 640×384 RMS 35.56 µs fastest / 35.94 µs
> slowest; LN 640×32 + γ + β 29.60 / 29.88 µs**; not dispatch-bound. Roofline / target: 7.6 / 10.1 µs and 0.6 / 0.8 µs.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16, bfloat8_b]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4}
- **SUPPORTED weight axes**: gamma_mode {gamma, no_gamma}, gamma_dtype {float32, bfloat16, "none"}, gamma_layout {TILE, ROW_MAJOR, "none"}, beta_mode {beta, no_beta}
- **SUPPORTED op-specific axes**: norm_type {rms_norm, layer_norm}, memory_layout {INTERLEAVED}, cluster_axis {0, 1}, topology {Linear, Ring}, num_links {1, 2}
- **EXCLUSIONS**: `{float32, fp32_dest_acc_en=False}` (precision convention, permanent), `{bfloat8_b, w_non_aligned}` (measured: 19/19 gate cells severity=bug with it lifted — the R_U element-shift cannot address shared-exponent blocks). The base op's `{bfloat8_b, h_non_aligned}` was **lifted** by the verifier (19/19 gate cells pass).
- **Cores**: workers = grid − 2L port cores (106 on QB2 at L = 2), row-groups × k W members (`select_split`); 2L port cores per chip (one per direction × link), link-0 ports are the stats hubs.
- **Compute config**: `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True, exact SFPU; fidelity / approx are not gated.
- **Golden baseline**: gate tier — see `verification_report.md` (verifier CLI: 0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode).

### [~] Refinement 1 — Port the base op's remaining input surface: HEIGHT_SHARDED + WIDTH_SHARDED input, and fp32_dest_acc_en=False

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` and `ttnn.TensorMemoryLayout.WIDTH_SHARDED` to `SUPPORTED["memory_layout"]` and `False` to `SUPPORTED["fp32_dest_acc_en"]` — the from-base features Phase 0 did not land, ported from the base op's mechanisms (`ttnn/ttnn/operations/base_ops/rms_norm/changelog.md`):

- **HEIGHT_SHARDED** (base op Refinement 4, "consume the resident row shard in place"): k = 1, block = the resident shard; `cb_x` backed on the input shard with `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy — never re-read a core's own shard through a `TensorAccessor`); workers = the input shard cores in shard-grid orientation (multi-range grids + ragged last shard, `select_sharded_blocking()`); port cores chosen from cores **outside** the shard grid (op_design.md Placement axis; RuntimeError if fewer than 2L remain). The base op's gamma multicast for sharded k = 1 (`GAMMA_MCAST_SHARDED_K1`) is the reuse-shared-gamma lever if gamma loads show up.
- **WIDTH_SHARDED** (base op Refinement 2, shard-resident blocks): one group, k = number of input shards (clamped by `P_CAP / G` and `stats_k_cap`, else RuntimeError), block = the shard, zero-copy `cb_x`. The k members' partials already meet in the stats table (no new topology here — the op's stats round *is* the cross-core combine). The W tail is **masked** in compute (base op: `prepare_reduce_mask` + `ReducePartialScaler::only_partial()`), never zeroed in the caller's shard — Phase 0's reader zero-fill mutates `cb_x`, which is not allowed on a shard.
- **Sharded output** (both): the golden suite passes a shard spec of the same strategy built for the gathered shape. Own and remote pages land at final shard addresses: the port walk (`agrn_port_common.hpp for_each_packet`, host replica `Walk`) maps page → shard core instead of page → DRAM bank (runs are `Wlt` tiles contiguous in one shard core, so `scatter_chunks` 1–2 suffices). `validate()` today refuses any non-DRAM-interleaved input/output — relax it to the sharded specs.
- **fp32_dest_acc_en=False** (base op Refinement 1, "intermediate CB formats follow the DEST width"): `T_acc` (Float16_b) for `cb_sumsq`, `cb_inv`, `cb_rstd_b` (`cb_inv` is a Float32 page today regardless of DEST — over-wide under 16-bit DEST, ledger check 3); the UnpackToDestFp32 tag list must be emptied of what 16-bit DEST cannot hold (the RMS `cb_sumsq` / `cb_gathered` tags and `partial_sum_fp32_mode` already key on `fp32_dest`). LN needs the design's 16-bit-DEST normalize (op_design.md Block Operation Realization row 12': FPU centering) and an FPU / `Fast` combine — the records stay Float32 pages (wire contract), and the combine runs in the caller's 16-bit DEST (op_design.md Key Risks; the `deferred` hub-side fp32 combine row is the remedy only if a `numerical-precision` failure demands it). `{float32, False}` stays an EXCLUSION.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: put first (from-base port, a knob-turn on every axis: HEIGHT is the k = 1 case of the existing split, WIDTH is k = #shards on the existing stats table, fp32_dest False is a CB-format / tag-list change) — ahead of the cross-device perf refinements. The perf-flagged configs (bf16, fp32_dest_acc_en=True, INTERLEAVED) are already supported, so nothing here gates Refinement 2. `/numeric-formats-metal` covers the fp32_dest_acc_en=False half only; sharding placement has no skill — follow the base op's zero-copy `cb_descriptor_from_sharded_tensor` pattern. The harness builds sharded inputs with `eval.sharding.auto_shard_config`; in the first gate run, three sharded cells failed *inside the harness's own from_torch* (tilize work split, "No solution found for single_block_size") while the op's persisted L1 leak was eating L1 — after the verifier's fix they xfail normally, but watch for INFEASIBLE_L1 skips on the 1×1×32×4096 / 1×1×64×1000 shards. 16-bit DEST: the precision gate is `helpers.TOLERANCES[(bf16, False)] = (0.999, 0.015)`.

**Done when**: every gate-tier cell with `memory_layout ∈ {HEIGHT_SHARDED, WIDTH_SHARDED}` or `fp32_dest_acc_en=False` (except the `{float32, False}` and `{bfloat8_b, w_non_aligned}` EXCLUSIONS) moves from `xfail_expected` to `supported_pass`; no sharded path reads its local shard through a `TensorAccessor`; all prior cells stay green.

**Outcome**: landed `fp32_dest_acc_en=False` (gate: 188/188 interleaved cells pass, rest are the permanent
EXCLUSIONS), HEIGHT_SHARDED (71/72 runnable gate cells; 128×8192 bf8b LN cannot fit its 1.1 MB output shard) and
WIDTH_SHARDED where R × k is small (≈ 57/86). Remaining blocker: WIDTH at large R × k → Refinement 1b.

### [x] Refinement 1b — WIDTH_SHARDED at large R × k: on-chip pre-combine of the k shard members + row-chunked stats CBs

**Goal**: the WIDTH_SHARDED gate cells that still fail with `RuntimeError: WIDTH_SHARDED k = … exceeds the
stats-partition cap` or an L1 CB overflow (≈ 29 cells: 2048×384, 2048×1280, 640×{896,1536,1792}, 1×640×1792,
4×1×512×512, 2×4×128×512, 128×4096 LN, 128×8192, and LN at R = 16–20: 640×{32,192,256,384,512}, 4×128×47). Measured
cause (Refinement 1): WIDTH fixes k = #shards (12–86) and one row-group (crt = R), so (a) the stats table is
G·R·k·nstats·128 B (393 KB at 2048×384 RMS, > `STATS_TABLE_CAP` 256 KB, and it is a CB on every core), and (b) every
member gathers and combines all R rows × P = G·k partitions: `cb_gather_rows` + `cb_gathered` = 2·R·nstats·⌈P/32⌉
Float32 tiles plus ~7 per-row LN CBs (R = 20 LN ≈ 1.3 MB). Lever: a **row-owner pre-combine** on the chip (the base
op's WIDTH owner-combine topology, Refinement 2 of base_ops/rms_norm): member m owns rows ⌈R/k⌉·m…, sums the k members'
partials of its rows (RMS: add; LN: the same exact (n, m, T, K) combine the cross-chip step uses) into ONE record per
row before the fabric round (P = G, table = G·R·nstats·128 B), then after the round combines its rows and returns
final per-row stats (inv / mean, rstd) to all k members; plus row-chunked (sub-block) per-row CBs so crt = R never
sizes a Float32 CB. A scheme-change (dependent-axis cross-core combine), not a knob-turn.

**Done when**: those WIDTH_SHARDED gate cells pass (or skip INFEASIBLE_L1 in the harness's own from_torch); no prior
cell regresses.

**Outcome**: built as specified (row-owner pre-combine, kt = 1 stats table, final stats returned to the k members,
row-chunked per-row CBs) plus strided x addressing for ragged WIDTH shards at R > 1 (128×8192). WIDTH_SHARDED gate
slice: 85 passed / 2 xfailed (EXCLUSIONS) of 87 (was ≈ 57). No regression: unit dir 55/55, HEIGHT + interleaved
golden sample 454 pass (the one failure is the known 128×8192 bf8b HEIGHT LN L1 overflow), test_regression 32/32;
perf cases unchanged (focus 35.52 / 35.89 µs, LN 29.59 / 29.84 µs).

### [x] Refinement 2 — TRANSPORT: bulk packet efficiency on interleaved final pages (focus case)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags `(1, 1, 640, 384)` RMS (bf16, fp32_dest_acc_en=True, gamma bf16 ROW_MAJOR sticks, INTERLEAVED, Ring, cluster_axis 1, 2 links, 1×4) as the mandatory PERF FOCUS: roofline 7.6 µs, target 10.1 µs, measured **35.56 µs** fastest chip. The measured per-chip breakdown (implementer's zone timeline, Phase 0 changelog) puts **TRANSPORT** at ~22 µs of it: own entry ~15 µs (240 KB per port at ~16 GB/s), relay ~4 µs, landing tail ~3 µs. `fabric_all_gather` moves 256 tiles per chip in ~18 µs on 2 links (29.1 µs × 0.63) at up to 77 GB/s per chip. The cause is the destination layout: interleaved final pages give 1–2-page bank runs per row (`Wlt = 12` pages at row stride `Wt = 48` over 8 banks), so a 4-run scatter packet carries ~6 pages (~12 KiB) in 4 DRAM writes. `blackhole-fabric.md` rule 1 says that is packet-rate bound. Raise the bytes per packet / per DRAM write on this shape. Candidate levers, measured not assumed:
- (a) Land remote slices in **receiver L1** (the idle workers' L1, or a port-side landing CB) at full-payload packets, then write final pages from the receiving chip (NoC writes to DRAM have no fabric packet-rate cap). This is a scheme-change of the landing topology: a per-entry arrival notification to the writing cores becomes OVERLAP work. Compare against the `R_RAW` / staging rows of op_design.md Regimes; forcing the aligned slice through the DRAM-staging regime measured 40.3 µs, so DRAM staging is not the answer.
- (b) `fabric_mcast` (op_design.md `deferred` row): removes the relay's DRAM re-read (6S → 5S) and the per-hop turnaround. Its blocker is the wrap hop of the route setter (`blackhole-ccl-kernels.md` §4), so treat it as an experiment.
- (c) TRANSPORT knobs already swept (`scatter_runs` 1/2/4, `batch_chunks` 2/4/6/8, `inc_every` 4/8/16, Phase 0 changelog): do not re-sweep them unless (a)/(b) change the packet shape.

No SUPPORTED change. Cite `ttnn/ttnn/operations/examples/fabric_all_gather/` (README rate table) for the transport ceiling instead of re-measuring it.

**Done when**: the focus case's fastest chip improves beyond noise (`eval.ccl_perf compare`, slowest chip not worse) toward the 10.1 µs target; the LN perf case does not regress; the golden suite stays green; and there is no regression across the config-spanning guard set (one representative per distinct kernel path × layout × placement: R_A interleaved RMS, R_A interleaved LN, R_U unaligned, Linear topology, 1 link, cluster_axis 0, plus the sharded paths once Refinement 1 lands).

**Outcome**: focus 35.51 → **30.45 µs** fastest / 35.89 → 31.35 µs slowest; LN 29.59 → **27.73** / 29.84 → 27.94 µs
(`eval.ccl_perf compare`: win on both, noise 0.05). The stated cause was refuted first: landing the packets in peer L1
without scatter did not move the own entry, and `fabric_all_gather` moves this slice at the same rate with 7-page or
1–2-page runs — the bulk is bound by per-packet work on the two port RISCs (walk, header address re-encoding, a
page-at-a-time own pull, batch-granular handoff), ~1200 ticks per ~10 KiB packet, against a link floor ~2.5× faster
(minimal-CPU ablation). Landed: header fast path (precomputed router-side bank addresses, word stores), pipelined
reader (trid per packet, per-packet push), own entry read back from the own pages (`OWN_SRC_DST`), reader → sender
run descriptors (`cb_port_desc`). Lever (a) (receiver-L1 landing) was not built: its premise (packet / DRAM-write
efficiency) is what the ablation refuted, and (b) `fabric_mcast` was not tried (it removes the relay read, not
per-packet work). Remaining: own entry ≈ 11 µs for 24 packets (~450 ns per packet vs ~345 ns link floor for a full
packet), now about evenly bound by both port RISCs (~550–600 ticks each); what is left needs less total per-packet
work (e.g. fewer packets per byte — `SPLIT_RUNS` measured +1.2 µs because more scatter chunks cost more than fewer
packets save), and the larger share of the call is now the serial chain before the bulk (stats → normalize → own
entry starts at ~11 µs), which is Refinement 3 (OVERLAP).

### [x] Refinement 3 — OVERLAP: start the bulk before the slowest worker finishes, and shorten the stats → normalize → own-entry chain

**Type**: perf

**Goal**: on both perf-flagged cases (focus 640×384 RMS, 35.56 µs; LN 640×32 + γ + β, 29.60 µs, roofline 0.6 µs — a fixed-cost case), the bulk is one serial dependency chain (op_design.md Stall-shadow check 2):
`stats round (~6 µs) → gather + combine + normalize + own write (to ~11 µs) → own entry → peers' relays`.
The own entry waits on `sem_own_ready == n_active` (all 80 workers on the focus case, all 20 on LN) before its first packet. Hide more of the exchange behind local work:
- **Bulk start granularity** (op_design.md Perf lamps, first row): per-row-group own-ready counters, so the port walks its own entry in row-group order and waits only for the groups a batch touches.
- **x·γ in the stats shadow** (RMS, lamp): precompute `x·γ` during the fence, leaving one multiply after the stats arrive.
- **Launch skew / spread** (0.38 µs focus, 0.27 µs LN): measure what the stats round exposes per chip.

Measure the exposed transport per call in the op (zones: `AGRN_PROFILE_ZONES=1`, `tests/ttnn/unit_tests/operations/all_gather_rms_norm/zone_timeline.py`), never on a dummy exchange. No SUPPORTED change; the master.md catalog (`fabric_ring_exchange`, `double_buffer`) bounds the levers.

**Done when**: measured device time improves beyond noise on at least one perf-flagged case (fastest chip lower, slowest not higher, `eval.ccl_perf compare`) with no regression on the other; the golden suite is green; and there is no regression across the config-spanning guard set (as Refinement 2).

**Outcome**: focus 30.45 → **29.21 µs** fastest / 31.38 → 30.00 µs slowest; LN 27.73 → **26.32** / 27.94 → 26.71 µs
(`eval.ccl_perf compare`: win on both, noise 0.05). Measured first: after the stats round the 80 workers' gathers
converged on the 2 hub cores (0.55–2.4 µs, set the slowest worker), and the port reader handed its first packet over
only after `READ_DEPTH` were issued (+1.6 µs after own-ready). Landed: the hubs multicast each stats slot into every
core's table as it completes (last slot linked to the stats-ready flag; the gather is now a flat 0.25 µs from local
L1), and eager push in the port reader. Built, measured and parked as live knobs: per-band own-ready
(`OWN_BANDS`, +1.4 / +2.4 µs at 16 bands: the early own-entry reads hit DRAM during the workers' own-write burst), and
x·γ in the stats shadow (`XG_PRE`, normalize 1.03 → 0.65 µs but the call is unchanged). Launch spread is now 0.79 µs
focus / 0.39 µs LN (slowest − fastest). What binds now: the own entry plus relays (~14 µs, port-CPU bound per
Refinement 2) behind a ~10 µs serial chain, and inside that chain the own DRAM write burst (1.1–2.3 µs: 480 KiB from
80 workers at once) followed by the port's read-back. Next would be landing the own slice without the DRAM
round trip before the first packet (a cheaper L1 source for the own entry: the per-page pull costs +4.5 µs today), or
fewer port per-packet cycles. Neither fits this refinement's chain scope.

**Verifier notes**: after Refinement 2. If R2 changes the landing topology (lever a), per-entry notification and the R_U `announce_landed` granularity (lamp: per-slot assembly) belong here.

### [ ] Refinement 4 — LN critical-path compute chain (indexer-k case)

**Type**: perf

**Goal**: the LN perf case (640×32, gamma + beta, one tile per slice: R = 20 tile-rows on 20 of 106 workers, k = 1) spends ~7 µs of serial compute per tile-row between the stats round and the own write (implementer zones: combine + normalize). That is ~25% of the 29.6 µs call, and every part of it sits on the critical path. The base op has no LayerNorm, so its perf rounds never touched this chain and the headroom is real. Visible costs:
- the local mean is reduced **twice** over the block (once into `cb_partial`, once into `cb_ln_m`), an exact Accurate SFPU reduce each time;
- `cb_x32` copies the whole block to Float32;
- each tile-row runs ~6 helper calls (copy/sub/square/mul chains, 3 reduces, 2 `unary_bcast`), each with its own init / reconfig.

Fuse the combine's SFPU chains, drop the duplicate S1 reduce (an exact copy needs a path that does not truncate through srcA), and consider splitting the 20 tile-rows over more workers. Grid occupancy needs a row split below one tile-row, which is impossible, so the remaining lever is the per-row chain length. Levers come from master.md: `compute_fusion`, `sfpu_tile_scope`, `reduce_block`. No SUPPORTED change.

**Done when**: the LN perf case's fastest chip improves beyond noise (slowest not higher), its PCC / RMS gates still hold (`test_layer_norm_distribution[big_offset]` included), the focus case does not regress, and the golden suite plus the config-spanning guard set are green.

**Verifier notes**: a local-compute refinement, so it is filed after the cross-device ones. On the LN case the transport bulk (~11 µs for a 20-page slot) is still the larger term, and R2 / R3 attack it first.
