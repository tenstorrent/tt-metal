# Changelog: distributed_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-10-28
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier), from
  base (`rms_norm`, local half ported) + the fabric examples (relay transport). One `ttnn.generic_op` per call.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16, bfloat8_b], fp32_dest_acc_en=[True, False] (EXCLUSION
  {float32, False}), layout=[TILE], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4],
  gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, none], gamma_layout=[TILE, ROW_MAJOR, none],
  memory_layout=[INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED], cluster_axis=[0, 1], topology=[Linear, Ring],
  num_links=[1, 2, None] — equal to TARGET.
- **Accuracy achieved** (worst device, 4 shapes, 1×4 ring; `test_distributed_rms_norm_precision_baseline.py`):
  bf16 + fp32 acc PCC=0.999998, max_abs_err=0.048, rel_rms_err=0.00187, ULP max 1; bf16 + 16-bit acc PCC=0.999993,
  max_abs_err=0.150, rel_rms_err=0.0049; fp32 PCC=1.000000, max_abs_err=0.035, rel_rms_err=0.0012 (tf32 operand
  truncation in the FPU scaling, verification_report.md).
- **Golden suite at Phase 0** (full matrix, `EVAL_CAPTURE_L1=0`): supported_pass 10426, supported_fail 455 (all
  L1 / output-reshard refusals → Refinement 1), xfail_expected 2040, invalid_skipped 33696, infeasible_skipped 320,
  xpass_drift 1 (false positive), xfail_wrong_mode 0. Acceptance 192 passed / 1 skipped; ring-mock 24/24;
  regression 12/12; translated 264/279.
- **Perf at Phase 0** (traced slowest chip, 1×4 ring, Ring, 2 links): 640×1792 25.86 µs (exchange stubbed 19.89;
  exposed 5.96), 32×1792 13.07 µs (9.64; 3.43), 2048×1280 41.83 µs (32.40; 9.43) — rooflines 9.0 / 0.45 / 20.5 µs.
- **Issues encountered** (verifier):
  - drift fix: `None` added to `SUPPORTED["num_links"]` (246 xpass_drift + 15 xfail_wrong_mode were translated
    calls that omit `num_links`);
  - code review: deprecated `copy_tile_to_dst_init_short` → `copy_init`; `footprint()` re-synced with the CB
    allocation (single `STATS_FORMAT`, combine CBs only when `G_w > 1`); dead code in `local_plan()`;
    `l1_ledger.md` total re-derived (211,584 B at the focus case);
  - harness: the default L1 graph capture force-fails this op's passing cells (thread-local graph tracking vs a
    per-chip MeshWorkload's thread-pool CB allocation) — escalated; up-front precompile zeroes module-level inputs
    of the measurement tests (run them with `--no-precompile`);
  - the implementer's sampled golden counts (31 L1 refusals) and untraced "exchange hidden" claim were corrected
    by the full run and the traced measurement.
- **Tests added**: test_distributed_rms_norm.py (planner), test_distributed_rms_norm_precision_baseline.py and
  test_distributed_rms_norm_perf_traced.py (verifier); debug / perf drivers from the implementer.

### Implementer notes (Phase 0)

One `ttnn.generic_op` per call (one `MeshProgramDescriptor`, one program per chip). Regimes D1 / D3 (interleaved,
`fused_split()` on the worker grid left after the ports), D1-HS / D3-WS (sharded, zero-copy); row waves; compact fp32
records (128 B per tile-row) all-gathered by store-and-forward relay (fabric_ring_exchange port kernel); every chip
folds the G records in group-position order (SFPU fp32) and finalizes its own inv_rms. Deviations: `op_design.md` →
"Implementation deviations".

Acceptance: `tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm.py` 192 passed,
1 skipped (no axis without a wrap link on a QuietBox 2), `--dev` and production.

Golden (QuietBox 2, 2x2 system mesh + 1x4 / 4x1 snake rings): ring-mock 24/24, regression 12/12, perf cases 3/3;
`test_op` samples 852 passed / 0 wrong-result failures, 31 L1 refusals (below); translated 264 passed, 15 refused
(12 output-reshard, 3 large prefill).

#### Perf (device kernel time, untraced, 1x4 snake ring, Ring, 2 links, bf16 x / γ TILE, HiFi4 + fp32 acc)

Device 3 = the last-launched chip (no launch-skew wait at the ready fence). `DRMS_ABLATE=exchange` = ports idle,
landed waits skipped (measurement only). Median of 4 warm calls.

| case | roofline | first pass | final | exchange stubbed |
|------|----------|-----------|-------|------------------|
| 640×1792 (focus) | 9.0 µs | 24.3 µs | 19.9 µs | 19.9 µs |
| 32×1792 (decode) | 0.45 µs | 9.4 µs | 9.0 µs | 9.0 µs |
| 2048×1280 | 20.5 µs | 42.4 µs | 33.0 µs | 32.4 µs |

On the last-launched chip the exchange is hidden (OVERLAP). The slowest chip per untraced call (54 / 17 / 71 µs) is
dominated by launch skew at the ready fence; the traced slowest-chip number (the metric that matters) was not measured
here (the shared mesh cache opens without a trace region).

What moved it (each measured on a device-zone timeline, `DRMS_PERF_ZONES=1`):
- OVERLAP: reader Phase B non-blocking (collect / all-gather / announce / inv-push queues), compute folds owned waves
  first — Phase B was serialized ~4 µs per wave.
- OVERLAP: port per-wave `own_stats` semaphores (no counted gate) and per-item wave counters (own segments go as
  published, relays as they land) — 2048×1280: 42.4 → 33.5 µs.
- LOCAL: wave-ordered x reads with transaction ids (`READ_LOOKAHEAD_WAVES`), prologue behind the reads, fold inits
  hoisted, non-volatile column-0 scatter.

#### Known gaps (implementer's list; the verified queue is op_requirements.md)

1. LOCAL (largest): the local work itself is 2.2× (focus) to 20× (decode) the DRAM roofline. Decode timeline (14 cores,
   4 tiles each): a serial chain of fixed costs — x read 0.8 µs, square/collapse 0.6, contribute + gather 0.9, publish
   0.4, collect (port reads + column-0 scatter) ~1.4, fold + finalize ~0.7, inv all-gather ~1.0, scale + store ~0.6.
   Focus: x reads ~7 µs (≈ 330 GB/s, below the 447 GB/s achievable) and output writes bunch after the last wave.
2. OVERLAP: the last wave's tail (collect → fold → all-gather → scale → store) is fully exposed; candidates: a compact
   record wire into a row-0 layout (no CPU scatter), multicast inv all-gather, lamp L2 (fewer / more waves).
3. L1 fallback `x_depth = 2` (multi-block) and streamed γ: 128×8192 HEIGHT_SHARDED, 4096×1024 WIDTH_SHARDED, prefill
   ≥ 8192 rows × 3584 are refused today.
4. Output resharding (sharded input → a different sharded output) is refused.
5. TRANSPORT: nearest-Ethernet port placement (design), traced slowest-chip measurement.

Debug / measurement aids: `test_distributed_rms_norm_debug.py` (row-scaled inputs localize any row ↔ statistic
mix-up), `test_distributed_rms_norm_perf.py` (perf driver); both run with `--no-precompile`.

## Refinement 1 — L1 fit for wide-row / tall-shard / long-prefill cells, and sharded-output resharding
- Date: 2026-10-29
- What was done: an L1 fallback ladder on the block surface the design names, predicate-guarded first fit
  (`blocking_options` / `first_fit`; option 0 = the Phase 0 plan, so every shape that fit keeps it): per-wave x·γ →
  per-chunk x·γ (γ resident, then streamed by the writer one chunk ahead of its stores) → chunk-cap halving → **passes**
  (multi-block rows: CBs sized for `PW` waves, waves numbered across passes so the exchange is unchanged; a row-group
  `SEM_PASS` barrier guards reuse of gather regions / inv slots; interleaved x_depth 2 then 1) → lean (single-buffered
  `cb_partial_col`, one-row square→collapse rounds) → up to 12 smaller waves (port own_stats semaphore ids made
  port-only; the fabric connection's semaphores are now appended after ours instead of being overwritten) →
  scratch-backed CBs (last resort: all CBs carved out of one L1 scratch tensor in the largest free hole when the
  caller's tensors leave no static CB region). Budget = gap below the lowest allocated L1 block, re-measured after the
  exchange scratch exists. Exchange resources cached per (mesh, ports, G), grown / shrunk on demand and released
  explicitly (Phase 0 kept one landing + semaphore set per distinct plan alive forever: the L1 it pinned was why many
  listed cells failed only inside the full suite). Resharding: any output other than the input's own shard placement
  streams through `cb_output` to the writer's output `TensorAccessor` (validate() no longer refuses it; one dispatch).
  `cb_table()` is the single source of the CB descriptors and `footprint()`.
  Reused: the Phase 0 kernels (pass loop around the existing Phase A / Phase B, existing γ chain with an offset per
  chunk, existing output store loop for resharded outputs). Added: no new kernel file; knobs `PW`, `cap`,
  `gamma_mode`, `gamma_stream`, `x_depth`, `combine_depth`, `sq_rows` (host, single source), `SEM_PASS`.
- Accuracy achieved: within the golden suite's per-(dtype, fp32_dest_acc_en) PCC / rel-RMS tolerances on every
  supported cell (e.g. HEIGHT_SHARDED 32×4096 / 32×8192 / 128×8192, WIDTH_SHARDED 4096×1024 / 2048×1280 incl. fp32,
  long prefill 8192 / 32768); rung tests at the acceptance PCC (0.999 fp32 / 0.995 bf16 / 0.99 bf8b).
- Golden test progress: `test_op` all cells green across 6 chunks covering all 36 INPUTS (0 failures, 0 hangs);
  `test_op_loose` + `test_regression` + `test_ring_mock` 39/39; `test_translated` 278/279 (was 264/279) — the 12
  resharding cells and prefill 8192 / 32768 now pass; prefill 131072 refuses cleanly (Refinement 1a).
- Perf (traced slowest chip, 1×4 ring): 640×1792 25.98 µs (Phase 0 25.86), 32×1792 12.45 (13.07), 2048×1280 43.05
  median vs 42.67 for the pre-refinement code measured in the same session — within noise; perf-case plans unchanged.
- Issues encountered: (1) fabric `setup_fabric_connection` appends its semaphores to the descriptor; Phase 0 overwrote
  them (it only worked because worker ids covered those slots) — moving own_stats to id 0 hung the ports in
  `close()` until our semaphores were set first. (2) Inside the suite, L1 fragmentation by the harness's own tensor
  creation (an input shard at the very bottom of L1) left a 13–400 KB static CB region → evict-and-retry, lean rung,
  scratch-backed CBs. (3) Running this directory's rung tests and a golden fp32 WIDTH_SHARDED cell in ONE session with
  precompile makes the golden helper build an all-zero device input (not the op; reproducer gated behind
  `DRMS_R1_DEBUG=1` in `test_distributed_rms_norm_r1_debug.py`). (4) 131072-row prefill: landing (~650 KB per bank)
  + one pass block of x (~800 KB) — Refinement 1a.
- Tests added: `test_distributed_rms_norm_l1_fallbacks.py` (every ladder rung forced on small shapes, >8-wave pass
  regime, scratch-backed CBs, 4 resharding placements; 22 cases), `test_distributed_rms_norm_r1_debug.py` (opt-in).

## Refinement 1a — long-prefill L1: x re-read (two-pass) streaming or a credit-managed landing ring
- Date: 2026-10-29
- What was done: lever (a), x re-read, as the last rung of the L1 ladder (interleaved input only; `reread_options`
  is tried after every other rung at every wave count, so every plan that fit before is unchanged). Phase A: the
  reader streams x one tile-row at a time into `cb_x_tiles` (`x_depth·SQ·WS`, pad tiles / W-pad columns zeroed per
  row), compute squares + collapses each round and pops it. Phase B: the writer re-reads x one (wave, chunk,
  tile-row) item ahead of the matching store into the new `cb_x_reread` (`STREAM_DEPTH·C`); compute does x·γ per row
  (`cb_x_gamma` = `C`) then × inv_rms → `cb_output`. The exchange, waves, passes and landing layout are unchanged.
  Reused: the Phase A square/collapse helper path (`square_collapse_rows`), the per-chunk γ chain and `scale_chunk`,
  the writer's γ-stream item pattern and store loop, the ladder / `cb_table()` single source. Added: `plan.reread`
  knob, `CB_X_REREAD`, reader `stream_wave`, writer `push_x_item`, compute re-read branches (one CT arg each).
- Accuracy achieved: 131072 × 3584 per device (bf16, RM γ, HiFi4, 16-bit DEST, 2×2 mesh, cluster_axis 1): PCC ≥ 0.995
  per 8192-row block on every chip; forced `reread-*` rungs at the acceptance PCC (bf16 0.995).
- Golden test progress: `test_ds_distributed_norm` prefill 128 / 1024 / 8192 / 32768 + `test_regression`: 17 passed
  (5 skipped: real-weights checkpoint absent). The 131072 cell runs the op successfully but the test process was
  OOM-killed on the host in this session (torch reference ~95 GB, ~100 GB free on the shared box) — host limit.
- Issues encountered: deadlock hazard on the streamed rung — the reader pushes wave w+1 before servicing wave w, so
  `cb_partial_col` / `cb_chip_partial` must stay two waves deep (the rung pins `combine_depth = COMBINE_DEPTH`).
  Measured landing ≈ 540 KB per bank at 131072 rows, independent of the wave count (it holds every row's records).
- Tests added: 4 `reread-*` rungs in `test_distributed_rms_norm_l1_fallbacks.py`;
  `test_distributed_rms_norm_long_prefill.py` (opt-in host-memory-light 131072-row twin of the golden cell).

## Refinement 2 — LOCAL: bring the per-chip local work toward the DRAM roofline on the perf-flagged case
- Date: 2026-10-29
- What was done (perf only; SUPPORTED unchanged). Classified first (640×1792, traced slowest chip, exchange stubbed,
  payload ablations with every handshake kept): stub 19.9 µs; reads + writes + γ stubbed 11.6 µs (the on-chip chain
  alone); reads −2.5, writes −3.2, γ −0.25 µs. Reads landed 3–8 µs, stores ran from 10.4 µs to ~20 µs (op: 17→25.5 µs,
  ≈ 286 GB/s, east columns last). Levers:
  - reader prologue moved ahead of the x reads — on Blackhole `write_zeros_l1_barrier()` is a full read barrier, so
    the scaler fill behind the reads stalled the next wave's reads ~3 µs;
  - `READ_LOOKAHEAD_WAVES` 1 → 0 (wave-ordered DRAM reads; wave 0 lands and publishes first);
  - `COMBINE_LAG` (new knob) 1 → 0: a wave is contributed / gathered / published right after it landed instead of after
    the next wave's reads; compute's chip sum follows its square;
  - NoC swap of the 3 westmost logical worker columns (base Perf 2 `dram_noc_balance`, re-measured: 1–5 columns →
    stub 19.1 / 18.3 / 17.7 / 17.8 / 18.7 µs), interleaved inputs only.
  Measured nulls (not built, bounded by ablation): γ multicast (γ ablation flat, 17.75 vs 17.80 µs); srcB-hold (all
  chains stubbed: op flat 21.5 → 21.7 µs — the op waits on the exchange); idle cores / S = 6 split (stub 17.2 but op
  26.1 / 24.8 vs 23.2 / 23.9 µs); per-wave γ, 1 / 2 waves (flat).
  Reused: the existing wave loop, CT/RT layout and kernel sources (one CT arg added to reader and compute; the swapped
  columns reuse the same sources through a second descriptor pair). Added: `COMBINE_LAG`, `NOC_SWAP_COLUMNS`,
  `noc_swap_cores` / `_split_noc_swapped`, measurement-only switches (`DRMS_ABLATE=reads,writes,gamma,chains`,
  `DRMS_LOOKAHEAD`, `DRMS_COMBINE_LAG`, `DRMS_NOC_SWAP`, `DRMS_WAVES`, `DRMS_GAMMA_MODE`, `DRMS_SLICE`).
- Perf (traced slowest chip, 1×4 ring, Ring, 2 links, bf16 / TILE bf16 γ, HiFi4 + fp32 acc; Phase 0 baseline →
  now): 640×1792 op 25.86 → 21.3–21.5 µs, exchange stubbed 19.89 → 17.7–17.8 µs; 2048×1280 op 41.83 → 32.8 µs, stub
  32.40 → 28.6 µs; 32×1792 op 13.07 → 12.5–13.0 µs, stub 9.64 → 9.9 µs (same-session A/B: +0.12 µs stub from the swap;
  op no worse). Guard set (same-session A/B against the Phase 0 knobs, `test_distributed_rms_norm_perf_guard.py`, op
  traced slowest): D3 TILE γ 26.3 → 21.6, D3 RM γ 25.5 → 21.6, D3 no γ 25.1 → 20.4, 2048×1280 44.8 → 33.2, streamed-γ
  chunk rung 27.6 → 24.4, x re-read rung 29.6 → 25.7; D1, D1-HS, D3-WS (± γ), the pass rung and decode within noise
  across repeats.
- Accuracy achieved: unchanged arithmetic (only transfer order / NoC change); guard cases PCC ≥ 0.99999 (bf16, fp32
  acc) on 640×1792, 2048×1280, 32×1792, 256×96, HEIGHT 1024×512, WIDTH 64×2048, forced pass / chunk / re-read rungs.
- Golden test progress: `test_op_loose` + `test_regression` + `test_ring_mock` 39/39; `test_op` slices (bf16 RM-γ
  cluster_axis 0 Linear 1 link, all shapes × placements: 78 passed / 78 invalid-skipped; bf8b no-γ non-aligned: 24
  passed / 120 skipped). Unit: acceptance + L1-fallback rungs 218 passed, 1 skipped.
- Issues encountered: the Blackhole NoC-zero barrier being a full read barrier (above). The op metric (traced slowest
  chip) carries ±1–2 µs of launch-skew noise per run; decisions were taken on repeated / 16-call runs.
- What binds now / next: op 21.4 µs vs stub 17.8 vs all-payloads-stubbed 12.6 µs. The op is bound by the exchange
  chain after the records land (collect 1.35 µs per owned wave, fold 0.7, inv all-gather 1.0) — Refinement 3's compact
  records / reader-side fold; the local DRAM traffic now adds ~5 µs over the chain floor (was ~8).
- Tests added: `test_distributed_rms_norm_perf_guard.py` (config-spanning same-session A/B, correctness-checked),
  `_zone_report.py` (zone table off `profile_log_device.csv`); `test_distributed_rms_norm_perf_traced.py` gained
  `DRMS_PT_EXTRA_ABLATE`.
