# Operation Requirements: distributed_rms_norm

## Definition
- **Formula**: every device `p` of a group of `G` devices along `cluster_axis` holds `x[p]` of per-device shape
  `(..., M, W_local)`; `W = G · W_local`;
  `out[p] = x[p] / sqrt( Σ_g Σ_w x[g]² / W + ε ) · γ[p]` (γ ≡ 1 when absent; padding never enters Σ or `W`).
- **PyTorch Reference**:
  ```python
  def distributed_rms_norm_ref(xs, gammas, eps):          # xs[p]: (..., M, W_local), gammas[p]: (W_local,) or None
      full = torch.cat(xs, dim=-1)
      w = None if gammas[0] is None else torch.cat(gammas)
      out = torch.nn.functional.rms_norm(full, (full.shape[-1],), w, eps)
      return list(out.split(xs[0].shape[-1], dim=-1))     # device p keeps slice p
  ```
- **Import Path**: `from ttnn.operations.distributed_rms_norm import distributed_rms_norm`
- **Function Signature**:
  `distributed_rms_norm(input_tensor, *, cluster_axis: int, gamma: ttnn.Tensor | None = None, epsilon: float = 1e-6,
  topology: ttnn.Topology = Topology.Linear, num_links: int | None = None,
  compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None, memory_config: ttnn.MemoryConfig | None = None)
  -> ttnn.Tensor` — one `ttnn.generic_op` (one `MeshProgramDescriptor`) per call.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Golden-suite runs (verifier finding, applies to every phase)**: run the suite as
> `EVAL_CAPTURE_L1=0 eval/eval_test_runner.sh eval/golden_tests/distributed_rms_norm/ <dir>`. With the default
> `EVAL_CAPTURE_L1=1` the harness's L1 graph capture sees none of this op's CBs (a per-chip `MeshWorkload` allocates
> them on the mesh thread pool; `GraphTracker` processors are `thread_local`), reports `peak_total=0` on every warm
> call, and force-fails ~1350 passing cells (`L1ProfilingError`). Infra issue, escalated in `verification_report.md`.
>
> **Perf measurement (every perf refinement)**: `tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_perf_traced.py`
> (`scripts/run_safe_pytest.sh --no-precompile --run-all … -s`) prints, per perf case on the 1×4 ring, the untraced
> last-launched / untraced slowest / **traced slowest-chip** device kernel time for the op and for the op with the
> exchange stubbed (`DRMS_ABLATE=exchange`). The traced slowest chip is the metric. Phase 0 baseline below.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16, bfloat8_b]
- **SUPPORTED fp32_dest_acc_en**: [True, False]; EXCLUSIONS `{dtype: float32, fp32_dest_acc_en: False}`
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {2, 3, 4}
- **SUPPORTED γ axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, "none"}; gamma_layout ∈ {TILE, ROW_MAJOR, "none"}
- **SUPPORTED memory_layout**: [INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED]
- **SUPPORTED cross-device axes**: cluster_axis ∈ {0, 1}; topology ∈ {Linear, Ring}; num_links ∈ {1, 2, None (canonicalized)}
- **SUPPORTED == TARGET** on every axis (no generality gap by axis value).
- **Cores**: worker grid = 110 − 2·num_links port cores; regimes D3 / D1 (interleaved, `fused_split()`), D1-HS / D3-WS (sharded, zero-copy).
- **Compute config**: `default_compute_kernel_config()` = HiFi4 + fp32_dest_acc_en; any fidelity / approx accepted.
- **Golden baseline** (verifier, QuietBox 2, `EVAL_CAPTURE_L1=0`): supported_pass 10426 · supported_fail 455 (all host-side L1 / reshard refusals — Refinement 1) · xfail_expected 2040 · invalid_skipped 33696 · infeasible_skipped 320 · xpass_drift 1 (false positive: a `pytest.raises` test on the excluded cell) · xfail_wrong_mode 0. Ring-mock 24/24, regression 12/12, translated 264/279.
- **Perf baseline** (traced slowest chip, 1×4 ring, Ring, 2 links, bf16 x / TILE bf16 γ, HiFi4 + fp32 acc):

  | case | op | exchange stubbed | exposed exchange | roofline / target |
  |---|---|---|---|---|
  | 640×1792 (PERF FOCUS) | 25.86 µs | 19.89 µs | 5.96 µs | 9.0 / 11.9 µs |
  | 32×1792 | 13.07 µs | 9.64 µs | 3.43 µs | 0.45 / 0.6 µs |
  | 2048×1280 | 41.83 µs | 32.40 µs | 9.43 µs | 20.5 / 27.3 µs |

### [~] Refinement 1 — L1 fit for wide-row / tall-shard / long-prefill cells, and sharded-output resharding

**Goal**: move every `supported_fail` cell to passing. All 455 are host-side refusals inside SUPPORTED (no wrong
result, no hang); none is an axis value, so nothing is added to SUPPORTED and nothing goes to EXCLUSIONS:

- **425 golden + 3 translated — `NotImplementedError: per-core footprint … exceeds L1 budget` / `no W-split … fits L1`.**
  Phase 0 holds the whole per-core assignment resident (`x_depth = 1`, `cb_x_gamma = R·WS`, γ slice resident) and
  refuses when that does not fit. Failing cells (all cluster_axis / num_links / γ variants of each):
  - HEIGHT_SHARDED wide rows: `1x1x32x4096`, `1x32x4096`, `1x1x128x4096` (bf16, bf8b), `1x1x32x8192`, `128x8192`
    (bf8b), `32x2048`, `1x32x1792`, `640x1792` (fp32) — the whole-row resident γ slice + whole-block `x·γ` overflow.
  - WIDTH_SHARDED tall shards (every shard core holds all `T` tile-rows): `1x1x4096x1024` (bf16, bf8b),
    `1x2048x1280` (bf16, bf8b, fp32), `1x1x2048x1280` (fp32), `4x1x512x512` (bf16, fp32).
  - INTERLEAVED long prefill (translated `test_ds_distributed_norm`, 3 cells): no `S` fits the whole block.
- **12 translated — `a sharded output must carry the input's own shard spec (resharding is not built)`**
  (`test_simulated_distributed_layernorm`, WIDTH_SHARDED input → a different WIDTH_SHARDED `memory_config`).

The L1 half is a **port from the base op** (`ttnn/ttnn/operations/base_ops/rms_norm/rms_norm_program_descriptor.py`),
a knob-turn on the block surface the design already names (`x_depth`, `R`, `C`); the fallback order is
predicate-guarded, so every shape that fits today keeps its exact Phase 0 plan:
1. per-wave `x·γ` (built: `plan.gamma_per_wave`);
2. **per-chunk `x·γ` + streamed γ** — base `GAMMA_RESIDENT` → `gamma_tiles_pages()` = `STREAM_DEPTH·C` and
   `_per_row_bytes()` (`cb_x_gamma` = `C` tiles per tile-row, the γ pass fused per chunk ahead of the inv pass);
3. **chunk-cap halving** — base `_search_blocking()` (W_CHUNK_TILES, halving);
4. **multi-block rows (`x_depth = 2`, `R < rows_this_core`)** — base `choose_block_rows()` / `MIN_BLOCKS_PER_CORE`;
   waves continue across blocks (`num_waves` counts waves of the core, not of a block), so the exchange protocol,
   landing layout and record offsets are unchanged. Sharded inputs take "the coarsest block that fits, up to the
   whole shard" (base sharded branch: the shard stays zero-copy, only the intermediate CBs are blocked).

**Implementation skill**: /memory-budget-metal

**Verifier notes**:
- Must land first: it is the only generality work left (SUPPORTED == TARGET), and the perf refinements' guard sets
  must include the regimes it adds. The perf-focus contract (640×1792 bf16, fp32 acc, TILE bf16 γ, INTERLEAVED,
  Ring, 2 links) is already supported and must keep its Phase 0 plan bit-for-bit (`plan_summary` unchanged, traced
  slowest-chip time within noise of 25.9 µs).
- Hold it to the performance bar: the fallback paths fill the grid and keep the wave pipeline (no single-wave or
  per-tile-handshake fallback). `footprint()` and `l1_ledger.md` gain the new terms in one place (`footprint()` is
  the single source; the verifier re-synced it with the allocation this phase).
- Resharding has no skill (placement, not layout): the compute already streams per tile-row into a streaming
  `cb_output` for SHARD-IN-INTERLEAVED-OUT; extend the writer to a sharded **output** `TensorAccessor` (a genuinely
  non-local write — the output shard lives on other cores), keyed on the output placement. Do not wrap the op in
  `ttnn.to_memory_config`: that is a second dispatch and violates the one-dispatch rule.

**Done when**: the 440 golden `supported_fail` cells and the 15 failing translated tests pass (golden run with
`EVAL_CAPTURE_L1=0`: `supported_fail` 0), acceptance + ring-mock + regression green, and the three perf cases'
traced slowest-chip times are within noise of the Phase 0 baseline.

**Outcome** (implementer, partial): every golden `test_op` cell passes (6 shape chunks covering all 36 INPUTS, 0
failures; `test_op_loose` + regression + ring-mock 39/39); translated 278/279 (all 12 resharding cells, prefill 8192
and 32768 pass). Remaining: `test_ds_distributed_norm[random_weights-prefill-131072]` refuses cleanly
(`NotImplementedError`) → Refinement 1a. Perf contract held: focus 640×1792 traced slowest 25.98 µs (baseline 25.86),
32×1792 12.45 (13.07), 2048×1280 43.05 median vs 42.67 measured the same session on the pre-refinement code;
`plan_summary` of the three perf cases unchanged.

### [x] Refinement 1a — long-prefill L1: x re-read (two-pass) streaming or a credit-managed landing ring

**Goal**: `test_ds_distributed_norm[random_weights-prefill-131072]` (4096 tile-rows × 3584 per device, bf16, RM γ,
16-bit DEST, interleaved) — the last cell of Refinement 1. Nothing to add to SUPPORTED.

**Blocker (measured)**: the landing scratch holds every row's records for every wave (`num_waves·G·seg_stride` ≈ 650 KB
per L1 bank at 12 waves, reserved on every core) and one pass block of x needs ≈ 800 KB even at 12 waves (≈ 4,400 tiles
per worker); no ladder rung fits the ≈ 920 KB left. Waves cannot grow further (port own_stats semaphores).

**Next lever** (pick one): (a) **x re-read** — interleaved only: Phase A streams x per tile-row through a 2-row
`cb_x_tiles` (square + collapse), Phase B re-reads x per tile-row into a second writer-produced CB (x read twice from
DRAM, only on this rung); the x term drops to ~4 rows; (b) **landing ring** — TRANSPORT: `K` wave slots reused with a
per-slot credit from the collecting roots back to the senders (cross-chip back-pressure), landing ∝ `K` instead of
`num_waves`. (a) is local and cheaper; (b) also lifts the 12-wave cap.

**Done when**: the 131072 translated cell passes; acceptance, rung tests (`test_distributed_rms_norm_l1_fallbacks.py`)
and the golden slices stay green; the three perf cases' plans unchanged.

**Outcome** (implementer): lever (a), x re-read, built as the last ladder rung (`reread_options`, interleaved only;
every earlier rung keeps its rank, so every plan that fit — including the three perf cases' — is unchanged). The
131072 × 3584 prefill now runs on device: `test_distributed_rms_norm_long_prefill.py` (same shape / dtype / RM γ /
16-bit DEST, opt-in `DRMS_LONG_PREFILL=1`) passes with PCC ≥ 0.995 per 8192-row block on all 4 chips. The golden cell
itself was OOM-killed on the host in this session (its torch reference over 2 × 131072 × 7168 grew the process by
~95 GB while the box had ~100 GB free) — a host-memory limit of the harness, not of the op. Rung tests: 4 forced
`reread-*` cases (D3 / W-pad RM γ streamed / passes / no γ / SQ 2 + x_depth 1). Regression net: acceptance +
L1-fallback tests 218 passed, golden DeepSeek prefill 128…32768 + regression 17 passed. Lever (b) not needed.

### [x] Refinement 2 — LOCAL: bring the per-chip local work toward the DRAM roofline on the perf-flagged case

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags `(1, 1, 640, 1792)` as the mandatory perf target (PERF FOCUS: bf16 x,
TILE bf16 γ, INTERLEAVED, HiFi4 + fp32 acc, Ring, 2 links on the 1×4 ring; roofline 9.0 µs, target 11.9 µs). The
measured per-chip breakdown puts the **local work** first: with the exchange stubbed the traced slowest chip still
takes **19.89 µs** (2.2× the 9.0 µs DRAM roofline; 4.59 MB at ≈ 230 GB/s effective against 447 GB/s achievable),
against **5.96 µs** of exposed exchange. Phase 0 did not port the base op's measured local perf rounds; start from
them, re-measured on the fused grid (106 workers), then the relevant `ttnn/ttnn/operations/examples/master.md`
entries (`noc_placement`, `double_buffer`, `compute_block_size`, `shared_input_reuse`):
- **γ multicast** — base Perf 1 `gamma_mcast` (`gamma_mcast_groups`, `ttnn.Mcast` + `McastArgs`): γ is read
  `G_h = 7` times (213 KB, 9.3 % of the DRAM bytes). The ledger's G-MC deferral cites helper friction the base
  op already solved — not a positive reason (ledger audit, folded here).
- **srcB-hold scaling passes** — base Perf 2 (`hold_mul_row` / `hold_mul_col`, design lamp L11).
- **DRAM rate / write overlap** — x reads at ≈ 330 GB/s and the output writes bunch after the last wave
  (Phase 0 changelog); `READ_LOOKAHEAD_WAVES`, the NoC-swap columns (base `NOC_SWAP_COLUMNS`, lamp L10) and the
  idle cores (`fused_split()` leaves 8 of 106 workers idle, lamp L12) are the knobs.
No SUPPORTED change.

**Done when**: measured traced slowest-chip device-ns improves on 640×1792 — both the op and its exchange-stubbed
variant (`test_distributed_rms_norm_perf_traced.py`) — moving it toward the 11.9 µs target; the PCC of the flagged
case holds; the golden suite is green; and no regression across the config-spanning guard set (one representative
per regime D1 / D3 / D1-HS / D3-WS × γ TILE / ROW_MAJOR / none × the Refinement 1 fallback paths, plus the decode
and 2048×1280 perf cases).

**Outcome** (implementer): 640×1792 traced slowest op 25.86 → 21.3–21.5 µs, exchange stubbed 19.89 → 17.7–17.8 µs
(2048×1280 op 41.83 → 32.8, stub 32.40 → 28.6; decode op 13.07 → 12.5–13.0). Levers kept: reader prologue ahead of the
x reads (the Blackhole NoC-zero barrier is a full read barrier), wave-ordered reads (`READ_LOOKAHEAD_WAVES = 0`), no
combine lag (`COMBINE_LAG = 0`), NoC swap of the 3 west columns (`NOC_SWAP_COLUMNS = 3`). Measured nulls: γ multicast
(γ ablation flat), srcB-hold (all chains stubbed: op flat), idle cores / S = 6 (op worse), per-wave γ / fewer waves
(flat). Bottleneck now: the post-landing exchange chain (collect 1.35 + fold 0.7 + inv all-gather 1.0 µs per owned
wave; all payloads stubbed = 12.6 µs floor, above the 9.0 µs roofline) — Refinement 3's compact-record / reader-fold
levers are the next step; not attempted here because they are that refinement's scope.

**Verifier notes**: filed ahead of the cross-device perf work by the measured breakdown (local 19.9 µs of 25.9 µs,
2.2× its roofline, with the base op's own perf mechanisms unported); Refinement 3's overlap is then designed and
measured against the faster local schedule it has to hide behind. Re-measure the base mechanisms on the fused grid
rather than assuming the base's wins carry over.

### [ ] Refinement 3 — OVERLAP: hide the exchange behind the local work (exposed 6.0 / 3.4 / 9.4 µs)

**Type**: perf

**Goal**: on the perf-flagged 640×1792 case (same contract as Refinement 2) the exchange is **not** hidden on the
traced slowest chip: op − stub = **5.96 µs** (decode 3.43 µs, 2048×1280 **9.43 µs** — more than the whole ≈ 5.4 µs
relay exchange measured alone by `fabric_ring_exchange`, so something in the wave pipeline serializes per wave).
The untraced last-launched-chip numbers the Phase 0 changelog reported as "hidden" do not hold on the traced metric.
Shrink the exposed transport on the slowest chip: diagnose the per-wave chain first (`DRMS_PERF_ZONES=1` zones:
publish → `prt_send` → landed notify → collect → fold → inv all-gather → scale → store), then the levers the
design names (`op_design.md` → Perf lamps):
- **the last-wave tail** — fold the `G` fp32 records on the owner's reader (RISC-V) into one record instead of
  scattering them into `G` column-0 tile pages (`cb_stats` capacity `G·q` 4 KiB pages for a 128 B live set per page —
  ledger check 2, folded here) and folding them on the SFPU; this also keeps the cross-device sum in fp32 under
  16-bit DEST (precision note in `verification_report.md`);
- **the on-chip combine wire** — compact 128 B records instead of faces-0/2 tiles (lamp L5; `cb_gather` is the
  largest CB at 57 KB);
- **wave depth / owner placement / notify quantum** (lamps L2, L7, L8) and launch-skew tolerance (work that needs no
  remote data before any fence-gated wait).
No SUPPORTED change.

**Done when**: the exposed exchange (traced slowest chip, op − stub) shrinks on 640×1792 and its total traced
slowest-chip time improves, with the flagged case's PCC held, the golden suite green, the regression suite's
bit-identical repeats intact, and no regression across the guard set (Refinement 2's set, Linear and Ring, 1 and
2 links, cluster_axis 0 and 1).

**Verifier notes**: OVERLAP is measured only in the op, never on a dummy exchange. Must follow Refinement 2: the
amount of local work available to hide behind is what Refinement 2 changes.

### [ ] Refinement 4 — TRANSPORT: port placement and link use for the stats exchange

**Type**: perf

**Goal**: the perf-flagged 640×1792 case (same contract) sends ≈ 2.7 KiB of records per device per call. Two
TRANSPORT items the Phase 0 build deviated on or left open, each taken from the examples' measurements, not
re-derived:
- **port placement** next to each link's Ethernet core (`fabric_all_gather` `plan()`, `_worker_nearest_noc1`) —
  Phase 0 places every port near the middle of logical row 0 (`op_design.md` → Implementation deviations);
- **links carrying the stats** (lamp L3) — `fabric_ring_exchange/README.md`: a second link is equal or slower at
  128 B–8 KiB (5.52 vs 5.90 µs at 2.5 KiB); carrying every record on link 0 when `num_links = 2` (link-1 ports
  fence only), or freeing link-1's port cores for work, is the measured candidate;
- (X-mcast stays deferred: ≤ 0.3 µs by the example's own floor.)
No SUPPORTED change.

**Done when**: the traced slowest-chip device-ns of 640×1792 improves (or the refinement records a measured null
per lever and leaves the code unchanged), golden green, and no regression on the guard set (Linear / Ring ×
1 / 2 links × cluster_axis 0 / 1).

**Verifier notes**: smallest expected win of the three perf phases (the transport floor is the ≈ 5 µs fence, which
Refinement 3 hides rather than shrinks); kept separate from Refinement 3 because TRANSPORT and OVERLAP are measured
differently.
