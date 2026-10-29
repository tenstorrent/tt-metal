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

## Refinement 3 — OVERLAP: hide the exchange behind the local work (exposed 6.0 / 3.4 / 9.4 µs)
- Date: 2026-10-29
- What was done (perf, OVERLAP; SUPPORTED unchanged). Diagnosed first on the traced slowest chip with stage zones: launch
  skew (~4 µs between the earliest and the last-launched chip, read off the ports' ready-fence waits) is most of op −
  stub, and every chip ends at about the same absolute time as the last-launched one; the controllable part is the chain
  from a wave's last landing to its stores (every core of the slowest chip gets wave w's inv within < 1 µs, then the
  wave's ~800 KB stores leave as one DRAM burst). That chain was collect 1.45 µs (0.95 of it a RISC-V column-0 scatter),
  fold 0.68, inv all-gather 1.1 (28 face writes), a 14-atomic SEM_INV announce, push, scale, store; and owners folded
  every owned wave before scaling any. Levers kept:
  - records land in row 0 of their `cb_stats` pages by two 64 B NoC reads (no RISC-V scatter; collect 1.45 → 0.6 µs;
    `cb_record_scratch` (G+1)·q → q pages);
  - fold unchanged in arithmetic (fp32 SFPU adds, position order), finalize on row 0 (`VectorMode::R`);
  - compact inv all-gather: the 128 B row-0 record into face 1 of each member's `cb_inv_rms` page; the member's reader
    moves it into column 0 (4 KiB → 128 B per row per member);
  - arrival by sentinel: a zeroed record slot means "not arrived" (inv is never +0; a zero inv is written as −0); no
    SEM_INV atomics, no wave-ordered announce gate;
  - `FOLD_AHEAD` knob, path-gated: 1 for G_w > 1 (2048×1280 33.4 / 32.5 / 32.2 → 30.8 / 30.8 / 30.7 µs), every fold
    first for G_w == 1 (D1-HS: 1 costs +0.35 Ring / +0.6 Linear).
  Measured nulls (reverted): a transposing fold (`transpose_tile`, +0.33 µs per fold — superseded by the row-0
  finalize); one joined 128 B write per member (20.9 vs 20.7 µs); flushed-only all-gather writes (flat); `FOLD_AHEAD = 0`
  (last-launched chip +2.6 µs).
  Reused: the Phase B poll loop, the fold / finalize bodies, `combine_write`-style stateful writes, `cb_inv_rms` /
  `cb_stats` / `cb_rms_root` (no new CB). Added: `FOLD_AHEAD` / `FOLD_AHEAD_SOLO` / `fold_ahead(plan)` (single source,
  `DRMS_FOLD_AHEAD` measurement override, in the program hash), `finalize_row0_body`, reader `inv_ready` / `unpack_inv`.
- Perf (traced slowest chip, 8 traced calls, 1×4 ring, Ring, 2 links, bf16 / TILE bf16 γ, HiFi4 + fp32 acc; this
  session's Refinement 2 code → now): 640×1792 op 21.4 / 22.0 / 21.8 → 20.4 / 20.6 / 20.7 µs, stub 17.4–17.8 → 16.7–16.9,
  op − stub ≈ 4.0–4.4 → 3.7–3.9 (6.0 at the heading's Phase 0 baseline); 32×1792 op 12.2–12.8 → 11.7–11.8, stub 9.9 →
  8.9; 2048×1280 op 32.4–33.0 → 29.6–31.0, stub 28.6 → 28.5, op − stub ≈ 4.3 → 1.0–2.4. The collect / fold / all-gather
  chain also runs in the stubbed variant, so part of the gain shows in the stub. Note: the traced-slowest metric depends
  on the trace length (32 traced calls read ~28–31 µs at 640×1792 for every variant); all numbers here use the driver's
  default 8.
  Guard set (`test_distributed_rms_norm_perf_guard.py`, now spanning topology / links / axis; Refinement 2 code vs now,
  op traced slowest): Ring 2 links axis 1 — D3 focus 21.6 → 20.1, RM γ 21.9 → 21.1, no γ 21.5 → 20.9, decode 12.7 → 11.8,
  2048×1280 32.9 → 30.1, D3-WS 20.7 → 18.0, D3-WS no γ 21.1 → 18.7, D1 10.8 → 10.6, D1-HS 11.8 → 11.3–11.7 (with the
  G_w == 1 gate), streamed-γ rung 23.0 → 22.1, pass rung 42.3 → 37.3, re-read rung 25.7 → 24.5; Linear 1 link axis 0 — D3
  focus 24.0 → 22.7, decode 14.2 → 13.3, D3-WS 21.2 → 19.3, D1 11.9 → 11.4–11.7, D1-HS 14.7 → 13.4–14.5 (gate), pass rung
  45.9 → 42.6, re-read rung 28.2 → 27.4. Noise on this metric: ±0.6 µs (two identical columns of the Refinement 2 run).
- Accuracy achieved: unchanged arithmetic (same fold order, same finalize body; only the layout of the records and the
  inv transport changed). Precision baseline (2048×1280 et al.): bf16 + fp32 acc PCC 0.9999985, max_abs 0.046,
  rel_rms 0.0018, ULP max 1; bf16 + 16-bit acc PCC 0.999993, rel_rms 0.0047; fp32 PCC 0.9999998, rel_rms 0.0011 — equal
  to the Phase 0 record. Guard cases PCC ≥ 0.99999.
- Golden test progress: `test_regression` (bit-identical repeats) + `test_ring_mock` + `test_op_loose` 39/39; `test_op`
  random 400-cell sample (seed 3): 99 passed, 282 invalid-skipped, 19 xfailed, 0 failed. Unit: acceptance 192 passed /
  1 skipped, L1 fallbacks 26/26, precision baseline 12/12.
- Issues encountered: none blocking. Ablating one lever at a time is unreliable on this metric: launch skew adds
  ±1 µs per run, so every decision here used interleaved A/B repeats.
- What binds now / next (640×1792, slowest chip): launch skew (~4 µs, not the op's to hide) plus the chain after the last
  landing (~5 µs: collect 0.6, fold 0.7, all-gather ~1.0 — per-packet issue cost, not bytes — member move 0.35,
  scale 0.3, last-wave store burst ~1.1–1.5). Next candidates, not attempted: the late chip's last publish lags its last
  x landing by ~2.2 µs (on-chip combine faces-0/2 wire, lamp L5 on-chip half); a smaller last wave (its store burst);
  TRANSPORT (Refinement 4).
- Tests added: `test_distributed_rms_norm_r3.py` (an overflowed Σx² row must not hang the sentinel protocol; other rows
  at the acceptance PCC; `--no-precompile`); `test_distributed_rms_norm_perf_guard.py` gained `DRMS_GUARD_TOPO` /
  `DRMS_GUARD_LINKS` / `DRMS_GUARD_AXIS`.

## Refinement 4 — TRANSPORT: port placement and link use for the stats exchange
- Date: 2026-10-29
- What was done: both TRANSPORT levers built as live knobs and measured in the op on the traced slowest chip (1×4 ring,
  FABRIC_2D_TORUS_XY, 2 links). (1) Port placement: `PORT_PLACEMENT` = `row0` (Phase 0, default) / `eth` (per chip,
  free core with the fewest NoC1 hops to the port's link Ethernet core — the Ethernet channel is the first word of
  `setup_fabric_connection`, mapped through the SoC descriptor tables copied from fabric_all_gather; no probe dispatch)
  / `eth_uniform` (one core per port index for all chips, fewest summed NoC1 hops). The worker grid excludes the union
  of every chip's ports; the port kernel now carries the destination chip's port coordinates per packet header
  (`DEST_WORDS` 3 → 5), the ready fence targets the target chip's port. (2) Links (lamp L3): `STATS_LINKS_MAX` caps the
  links that carry records and reserve port cores. Measurement-only env: `DRMS_PORT_PLACEMENT`, `DRMS_STATS_LINKS`,
  `DRMS_RESERVE_LINKS`, `DRMS_PORT_EXCLUDE_ALL` (A/B on one worker grid).
- Measured (traced slowest chip, µs): baseline 640×1792 21.00 / 21.75 / 21.09, 2048×1280 30.6 / 30.1 / 30.8, decode
  11.7–12.1. Placement on one grid: row0 20.8 / 21.5, eth_uniform 21.3 / 21.2, eth 22.6 / 25.4 / 25.1 (2048×1280 31.6 /
  31.5 / 35.4) — NoC distance null; per-chip placement grows the launch skew (`prt_ready_wait` 11–13 µs on two chips vs
  4–7). One link: 640×1792 21.4 / 21.5, 2048×1280 35.4 / 34.0 (ports freed, 108 workers); 21.9 / 34.4 with link-1 ports
  reserved. Defaults kept (row0, both links): measured null for 640×1792, regression avoided on 2048×1280.
- Accuracy achieved: unchanged (perf-case PCC > 0.999 on every variant; unit suites bit-for-bit paths unchanged).
- Golden test progress: loose perf cases 3/3; slice of 16 cells (640×1792, 17×50, 2048×1280, 2×1×128×100 bf16 γ TILE,
  every feasible topology × links × cluster_axis) 16/16; full suite left to the harness.
- Issues encountered: the example's "a second link is equal or slower" (fabric_ring_exchange, transport alone) does not
  carry into the op — its ports also serialize the publishes and the per-entry notifies of many roots.
- Tests added: `test_distributed_rms_norm_r4_debug.py` (host-only probe: per chip / direction / link Ethernet channel,
  its NoC coordinate and the nearest-NoC1 port core). Unit files green on default, `eth` and `eth_uniform` placement.

## Perf 1 — perf tournament round 1 (2 experiments): the inv distribution hop
- Date: 2026-10-29. Perf only; SUPPORTED unchanged. Focus = the perf-flagged LOOSE_CASE 640×1792 (bf16 x, TILE bf16 γ,
  HiFi4, fp32_dest_acc_en=True, 1×4 ring, Ring, 2 links, cluster_axis 1); every knob of it is in SUPPORTED.
- Instrumentation: the kernels' permanent `DRMS_ZONE(...)` stage zones (`DRMS_PERF_ZONES=1`; free when off) were reused —
  `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` / `MaybeDeviceZoneScope` does not exist in this tree. Added
  `rd_ag_flush` (the all-gather's source flush, split from its issue loop).

### Measured breakdown (traced slowest chip per call, µs, one fresh-cache run per variant)
| variant (cumulative peel) | op | exchange stubbed |
|---|---|---|
| full | 21.39 | 16.81 |
| x reads stubbed | 18.48 | 14.77 |
| + output writes stubbed | 17.37 | 12.70 |
| + γ reads stubbed (every DRAM payload off) | 16.76 | 12.71 |
32×1792: 11.45 / 8.90; 2048×1280: 30.50 / 28.58. Roofline 9.0 / 0.45 / 20.5.
- With every DRAM payload AND the exchange stubbed at once the focus case still takes 12.7 µs (> the 9.0 µs DRAM
  roofline): the op is bound by its on-chip / exchange chain, not DRAM; DRAM adds ~4.1 µs above that chain.
- Per chip (zones, last traced call): chip 0 launched last and runs 17.2 µs; the others 21.7–24.7 µs, of which their
  ports' `prt_ready_wait` is 2–8 µs (launch skew: every chip ends at about the last-launched chip's absolute end).
- Critical path on the last-launched chip: x landings end 8.5 (roots) / 6.4 (members) → last publish 10.6 → exchange
  done ~11.0 → collect 11.9 → fold 13.7 → inv all-gather 15.0 → member row0→col0 move 15.6 → scale 15.9 → stores 17.0.
  Per owned wave: collect 0.6, fold 0.67, inv all-gather 1.0 (28 packet issues), member move 0.44, scale 0.2.
- Ranked (headroom on the slowest chip): (1) OVERLAP — post-landing chain (~6 µs exposed; its inv distribution hop
  is ~1.5 µs per wave); (2) OVERLAP — root cores' x landings 2.1 µs behind the members' (owner duties on the same
  NCRISC); (3) launch skew ~4–7 µs (not the op's to remove; every chip's local work already runs in it); (4) DRAM
  reads ~300 GB/s (ablation: −2.0 µs stub).

### Portfolio (cap 2) and verdicts
| idea | kind | verdict | focus op / stub (µs, base → cand) | domain |
|---|---|---|---|---|
| `inv_distribution`: shorten the inv distribution hop | OVERLAP (post-exchange chain) | **WIN (modest)**, graduated | 20.84 → 20.35 / 16.79 → 16.58 (6 interleaved rounds, subagent copy) | everywhere, no exception |
| `owner_offload`: stop owner duties delaying the root's x landings | OVERLAP (work split / data availability) | **NULL** at the op level | c1 −0.95…+0.2 / −0.86; c2 +0.3…+0.7 / −1.1 | — |

`inv_distribution` option menu (all bit-exact vs the real op, PCC 0.999998): NoC multicast per row segment (a row-group
is 14 consecutive workers on the 11-wide grid = 2–3 one-row rectangles) 21.32 / 17.09 — **regression** (~180 ns per
multicast issue + slow acks); multicast with one 128 B packet 21.97 / 17.04 — regression; joined 128 B + posted
21.02 / 16.87 — flat; batched member move alone 20.63 / 16.58; **joined + posted + batched move 20.35 / 16.58** (chosen);
2-level relay tree 21.20 / 16.89 — regression (relays busy with their own waves); owner self-slot by RISC-V stores
+180 ns; two command buffers flat; second NoC via the writer's command buffer **hung** (that NoC is the other RISC's)
→ BLOCKED; dynamic-NoC two-NIU fan-out correct but slower (other NoC busy with stores). Members fold their own rows
(option C) not built: same fan-out cost moves to notifies. Finding: the fan-out is bound by per-packet issue cost
(~32 ns / 64 B packet, ~50 ns / 128 B), not bytes. Per-wave hop medians from the owner's fold end: last member arrival
1548 → 1284 ns, move end 1987 → 1567, scale end 2643 → 2367.

`owner_offload` (NULL): premise corrected — with LOOKAHEAD 0 the root's next-wave reads are already issued before
`service(w)`; its late `rd_land` is the trid barrier + push reached only after the gather wait / publish, plus compute's
`chip_sum(w)` before `square(w+1)` on owners. Variants (all bit-exact, no hangs): c1 non-blocking Phase A poll (root
landings 8.2–8.45 → 6.3–6.6 µs = members, publish end 10.6 → 9.8, untraced last-launched 16.7 → 15.6–15.8) but the op
metric does not move (exposed transport grows by what the stub loses) and c1 regresses 2048×1280 +1.05 (7/7 rounds,
decoupled roots break wave-ordered DRAM) and the chunk-stream rung +0.8…+1.2; c2 (= c1 + owner chip-sum one wave
later) wins D3-WS −1.0…−1.3 op / −1.5 stub (3 sessions) but regresses 2048×1280 +0.55…+1.7, chunk-stream +1.5, focus
+0.3…+0.7, decode +0.45; c3–c7 (merged loops, owner-first, DRAM-order gates) flat or worse. Option B (owner duties on
the writer BRISC) not built — dominated by c1's measurement. Not graduated: no variant wins op-wide, and c2 would need a
regime-gated dual reader loop for a non-perf-flagged regime (D3-WS already gains −1.2 µs from the graduated wire).
Candidate for round 2, re-measured on top of this round's wire.

### Graduated: joined + posted inv record, batched member move (reader only)
- `allgather`: row 0 of face 1 of the finalized `cb_rms_root` page is copied into row 1 of face 0 (unused), so each row's
  record is one contiguous 2·INV_HALF packet; one **posted** packet per member (`noc_async_write_one_packet_*<true>`);
  the owner waits only for the source to be read (`noc_async_posted_writes_flushed`) before popping `cb_rms_root`
  (arrival is by the existing sentinel, so no ack is needed); kernel tail `noc_async_write_barrier()` +
  `noc_async_posted_writes_flushed()` before the atomic barrier.
- `unpack_inv`: all record loads first, then the scattered column-0 stores (~450 → ~290 ns per row); the slot is cleared
  only when a later pass reuses it (`NP > 1`) — the prologue zeroes every slot on every call, so a single-pass call
  never sees a stale record.
- Domain: the ONE path everywhere (every regime, dtype, DEST mode, topology, link count, pass count, L1 rung); the
  two-packet acked wire and the per-element move were deleted. No carve-out: none was earned (no measured regression).
- Whole op, same-session interleaved A/B, real op (3 rounds, traced slowest median, op / stub µs): 640×1792 20.62 → 21.10
  (flat: inside the ±1 µs launch-skew noise, ranges overlap) / 16.86 → 16.60 (−0.26, every round); 32×1792 11.87 → 11.42
  / 8.92 → 8.57; 2048×1280 30.63 → 29.90 / 28.66 → 28.23.
- Guard set (`test_distributed_rms_norm_perf_guard.py`, current defaults, old → new op slowest / stub, µs; pcc 1.00000
  everywhere): D3 focus 21.03 → 21.20 / 16.82 → 16.50; RM γ 21.65 → 20.17 / 16.92 → 16.57; no γ 20.69 → 20.74 /
  16.67 → 16.44; decode 11.61 → 11.21 / 8.88 → 8.62; 2048×1280 30.85 → 30.40 / 28.63 → 28.14; D1 10.39 → 10.48 / 6.25 →
  6.06; D1-HS 11.62 → 11.40 / 7.34 → 7.18; **D3-WS 18.32 → 17.15 / 14.28 → 12.75**; **D3-WS no γ 18.64 → 17.22 / 14.25 →
  12.68**; passes rung 37.96 → 38.17 / 25.25 → 24.78; chunk-stream rung 22.28 → 22.05 / 18.25 → 17.99; re-read rung
  24.34 → 24.11 / 22.55 → 22.17. The subagent's copy also covered bf16 DEST, fp32 input and Linear / 1 link / axis 0
  (focus, decode, D1-HS, D3-WS, passes): flat or better, D3-WS −1.5…−2.2 µs.
- Accuracy: unchanged arithmetic (same records, same fold / finalize, only the wire and the move order changed);
  bit-identical to the pre-change op on every subagent config; R3 overflow / sentinel test green.
- Tests: unit acceptance + L1 fallbacks + R3 + precision baseline 234 passed / 1 skipped; golden `test_op_loose` +
  `test_regression` + `test_ring_mock` 39/39; golden `test_op` HEIGHT_SHARDED×Linear 3420 passed / 620 xfailed,
  INTERLEAVED×Linear 3600 / 720 xfailed, WIDTH_SHARDED×Linear 3599 passed / 720 xfailed / 1 failed — the failure
  (4096×1024 fp32 WIDTH_SHARDED axis 0 Linear 1 link: "footprint 301184 B exceeds L1 budget 216576 B") is
  **pre-existing**: the pre-change reader fails the same cell in the same chunk, and the cell passes alone (suite-order L1
  fragmentation by the harness's tensors, the Refinement 1 failure class) → upstream L1-fit work, not perf. Ring cells
  are pruned at collection on this box's 2×2 system mesh (covered by `test_ring_mock` and the 1×4 ring drivers).
- Summary: 2 ideas measured; 1 graduated (modest: −0.26 µs stub on the focus case, op flat within launch-skew noise;
  −1.2…−1.4 µs op on D3-WS), 1 null; no regression.
- What binds next (focus, slowest chip): launch skew, then the post-landing chain (collect 0.6 + fold 0.67 per wave are
  now the largest hops) and the root landing lag (`owner_offload` c1 removes it but needs the exchange chain shortened
  before it shows on the op metric).

### Helper bypasses — none
The graduated path uses the dataflow API it used before (`noc_async_write_one_packet_set_state` / `_with_state`, now
with the posted template flag, and `noc_async_posted_writes_flushed`); no kernel_lib helper was bypassed. (The losing
multicast variant used raw `noc_async_write_multicast*` because no helper multicasts to a list of 1-row rectangles derived
from a member list — `mcast_pipe` / `ttnn.Mcast` take one rectangle; recorded here only as a gap note, it did not
graduate.)

Artifacts: `perf_experiments/inv_distribution/` (op copy `drms_copy/` with every variant behind `DRMS_INV_VARIANT`,
interleaved A/B driver, overflow test, `hop_report.py`), `perf_experiments/owner_offload/` (op copy `drms_oo/` behind
`DRMS_OO`, `owner_offload_vs_op.diff`, A/B drivers, zone tools, `logs_summary.txt`; zone CSVs dropped). Both drivers
need `--import-mode=prepend`.

## Perf 2 — perf tournament round 2 (2 experiments): the owner's fold hop and the owner work split
- Date: 2026-10-29. Perf only; SUPPORTED unchanged. Focus = the perf-flagged LOOSE_CASE 640×1792 (bf16 x, TILE bf16 γ,
  HiFi4, fp32_dest_acc_en=True, DRAM interleaved, 1×4 ring, Ring, 2 links, cluster_axis 1); every knob of it is in
  SUPPORTED. Roofline 9.0 µs, target 11.9 µs.
- Instrumentation: the kernels' permanent `DRMS_ZONE(...)` zones (`DRMS_PERF_ZONES=1`) were reused —
  `perf_instrumentation.hpp` / `MaybeDeviceZoneScope` still does not exist in this tree. The new fold path keeps the
  `cmp_wait_stats` (wait) / `cmp_fold` (work) split, with every init hoisted out of the wait zone.

### Measured breakdown (current HEAD = Perf 1, one fresh-cache run per variant, µs)
Cumulative peel, focus case (traced slowest chip per call / exchange stubbed / untraced last-launched chip):
| variant | op traced slowest | stub | untraced last-launched |
|---|---|---|---|
| full | 20.03 | 16.50 | 16.44 |
| x reads stubbed | 18.29 | 14.57 | 14.69 |
| + output writes | 17.13 | 12.08 | 12.29 |
| + γ reads (every DRAM payload off) | 16.38 | 12.10 | 12.33 |
| + eltwise chains (every payload off) | 15.50 | 10.38 | 10.62 |
| eltwise chains alone | 19.91 | 15.30 | 15.01 |
32×1792: 11.19 / 8.61; 2048×1280: 31.22 / 28.30.

- With every DRAM payload, every eltwise chain AND the exchange stubbed at once, the last-launched chip still takes
  10.4 µs (> the 9.0 µs roofline). The op is bound by its latency chain: per-core data is tiny (12 tiles per core:
  S = 4, G_w 14 × G_h 7, 3 one-row waves on 98 of 106 workers).
- Per chip (zones): chip 0 launched last (`prt_ready_wait` 0.1 µs); chips 1–3 waited 2–7 µs in their ready fence
  (launch skew). The owner's reader runs Phase A for every wave before Phase B, so on the last-launched chip x reads
  (0.9 → 8.4 µs) and stores (10.0 → 17.2 µs) do not overlap.
- Last-launched chip: non-owner cores land their last wave by ≤ 6.5 µs (≈ 410 GB/s, near the box's 447 GB/s eltwise
  DRAM rate). Owner cores land it at 7.6–8.4 µs: the wave-1 owner's reader gathers/publishes wave 1 before it reaches
  its wave-2 landing. The last publish is at 10.5 µs.
- Early (slowest) chip, critical owner, last wave: records landed 18.4 → collect 19.24 → **fold + finalize + pack ≈
  1.05 µs** (all-gather starts 20.34) → inv all-gather 21.09 → member column-0 move 21.38 → scale 21.87 → last-wave store
  burst 23.38 (688 KB at ≈ 480 GB/s: at the DRAM rate).
- Ranked (headroom on the slowest chip, roofline-gated): (1) OVERLAP / work split — owner duties delay the owners'
  x landings on the last-launched chip (~2 µs on the landings, feeding the last publish); (2) post-exchange chain —
  the owner's fold hop, 1.05 µs for 32 useful values per record (4 full-tile fp32 `copy_tile`s, 3 full-tile SFPU adds,
  3 inits, 2 reconfigs); (3) all-gather 0.75 / move 0.3 (Perf 1 graduated, measured since); (4) store tail and member
  reads — at the DRAM rate, not floated; (5) launch skew — not the op's to remove.

### Portfolio (cap 2) and verdicts
| idea | kind | verdict | focus (base → cand, µs) | domain |
|---|---|---|---|---|
| `fold_finalize`: fold + finalize the G records on row 0 in one fused SFPU pass, one unpack transaction | local compute on the post-exchange chain | **WIN**, graduated | fold hop 1.07 → 0.25; op traced 20.99 → 19.86, untraced last 16.40 → 15.44 (4 interleaved rounds) | everywhere, no exception |
| `dedicated_owners`: the owner role on the idle workers, x-reading cores pure members | OVERLAP (work split) | **NULL** on the op metric | best (d3): op 20.43 → 20.62 (flat), stub 16.56 → 15.01, untraced last 16.42 → 14.79 (6 rounds) | not graduated |

`fold_finalize` option menu (isolated bench, ns per fold from records landed to `cb_rms_root` visible to NCRISC,
G = 4, Q = 1 / 2 / 5; **every option bit-identical** to the current fold):
base 1030 / 1920 / 4548; hoisted inits 994 / 1881 / 4507; G DEST slots + one custom SFPU fold+finalize pass 638 / 1137 /
2649; row-0-only adds (`VectorMode::R`) 685 / 1246 / 2932; `unary_bcast<ROW>` unpack + custom pass 689 / 1243 / 3104;
raw ROW unpack-to-dest + custom pass 553 / 968 / 2246; + 128 B record pack 531 / 945 / 2212; one unpack transaction for
all G records 357 / 598 / 1297; **compact** ([row][position] 128 B records, one 32-datum UNPACR per record in one
transaction, SFPU on face 0, packs the joined 128 B record) **251 / 412 / 866** (chosen). Finding: each `copy_tile`
under UnpackToDestFp32 costs one math↔unpack handshake (~80 ns), which is most of the old fold.

`dedicated_owners` (NULL): d3 (a dedicated core per row-group does gather / chip-sum / publish for every wave; collect /
fold / all-gather stay on the rotating member owners, gated by a new `SEM_INV` handshake from the dedicated core) brings
the last-launched chip's last landing 8.61 → 6.71 and its last publish 10.75 → 9.45 µs. The early chip's end does not
move: it is set by the late chip's **wave-0** records (published ~6 µs local in both variants: land → gather →
chip-sum → publish ≈ 2.4 µs) followed by a write-bandwidth-bound burst of all three waves' stores. d1 (the full owner
on the dedicated core) is worse (op +0.3…+1.2; D3-WS +2.1 Ring / +1.8 Linear): one core then runs every wave's collect →
fold → all-gather back to back. Re-splitting to free cores where none are idle (d4, 2048×1280 G_h 13 → 11): stub +0.5,
last-launched +0.4. Not graduated: the op metric is flat, and it would add two kernels, a handshake and a regime gate.
Next lever this points at: a shorter path to the first wave's publish on the late chip (e.g. a smaller wave 0).

### Graduated: compact records + fused fold/finalize (compute + reader; host unchanged)
- reader `collect`: one 128 B NoC read per record into `cb_stats` page `r * G + pos` ([row][position]), the record
  contiguous at the page start (rows 0–1 of face 0) — was two 64 B reads into row 0 of faces 0 / 1 of page `pos * Q + r`.
- compute `fold_finalize`: every init (reconfig, `ff_init`, `rsqrt_tile_init`) before the records' wait; per row
  `ff_fold_chunks<G, Q>`: under fp32 DEST, one unpack-to-dest transaction per DEST-capacity chunk (4 fp32 slots;
  `ff_unpack_records_to_dest`) + one plain datacopy; under bf16 DEST, `copy_tile` per record (8 slots); then one SFPU
  pass `ff_fold_body` over vectors 0 / 1 of face 0 — G loads, G − 1 fp32 adds in position order (bf16 DEST: each partial
  sum round-trips DEST exactly like the per-position `add_binary_tile`), and on the last chunk the unchanged finalize
  (·1/W + ε, rsqrt, zero inv → −0). The record is packed as rows 0–1 of face 0 (face_r_dim 2, 1 face); the full-tile
  pack MOP is restored after the push, off the chain.
- reader `allgather`: the RISC-V join copy (row 0 of face 1 → row 1 of face 0) is deleted — the pack now writes the
  joined record; the member wire / move are unchanged.
- Deleted: `finalize_row0_body`, the copy_tile + add_binary_tile fold, the two-half collect, the join copy. No
  predicate, no dual path, no carve-out (none earned: faster everywhere measured, bit-identical everywhere).
- Whole op, same-session interleaved A/B, real op (3 rounds, medians; op traced slowest / untraced last-launched /
  stub, µs): **640×1792 20.71 → 19.38 / 16.31 → 15.40 / 16.61 → 15.65** (every new traced run below every old one);
  32×1792 11.19 → 10.70 / 7.90 → 6.93 / 8.58 → 6.67; 2048×1280 30.58 → 29.61 / 27.76 → 27.52 / 28.17 → 28.06.
- Guard set (`test_distributed_rms_norm_perf_guard.py`, current defaults, old kernels → new kernels, op traced slowest /
  stub slowest µs; pcc 1.00000 everywhere):
  - Ring, 2 links, axis 1: D3 focus 21.12 → 19.64 / 16.57 → 15.70; RM γ 20.41 → 20.24 / 16.59 → 15.69; no γ 21.07 →
    19.33 / 16.51 → 15.63; decode 11.42 → 10.00 / 8.58 → 6.66; 2048×1280 29.95 → 29.72 / 28.30 → 28.18; D1 10.44 → 9.78 /
    6.04 → 5.10; D1-HS 11.39 → 9.98 / 7.18 → 5.91; D3-WS 16.66 → 15.86 / 12.72 → 11.99; D3-WS no γ 17.18 → 16.53 /
    12.69 → 11.93; passes rung 38.64 → 35.05 / 24.75 → 21.90; chunk-stream rung 22.74 → 21.13 / 17.87 → 16.97; re-read
    rung 24.67 → 23.61 / 22.21 → 21.33.
  - Linear, 1 link, axis 0: D3 focus 22.18 → 21.59 / 16.40 → 15.39; RM γ 23.06 → 22.21 / 16.58 → 15.56; no γ 22.74 →
    21.21 / 16.57 → 15.54; decode 12.87 → 12.45 / 7.67 → 6.70; 2048×1280 36.77 → 35.28 / 29.66 → 28.83; D1 12.26 →
    10.23 / 6.89 → 5.05; D1-HS 14.15 → 12.62 / 7.83 → 6.17; D3-WS 17.82 → 16.41 / 12.71 → 11.96; D3-WS no γ 17.90 →
    16.67 / 12.69 → 11.90; passes rung 42.82 → 39.90 / 24.71 → 21.86; chunk-stream rung 23.33 → 23.66 (+0.33, inside the
    ±0.6–1 µs launch-skew noise of this metric; its stub −0.70) / 18.10 → 17.40; re-read rung 26.63 → 25.00 / 22.29 →
    21.37.
  - The subagent's op copy also covered bf16 DEST on the focus case (untraced last 15.50 → 14.89, traced 20.31 →
    19.80) and fp32 input: bit-exact vs the pre-change op.
- Accuracy: unchanged arithmetic — same fp32 SFPU add instructions in the same position order (identical sum on every
  chip), same finalize, −0 sentinel kept. Bit-identical to the pre-change op on all 11 driver cases plus bf16 DEST, fp32
  input and Linear; bench bit-identical for G = 2, 3, 4, 6, 8, fp32 / bf16 DEST, Q = 1…5 (G > 4 under fp32 DEST takes
  the DEST-capacity chunk path; G = 8 on hardware covered only by the bench).
- Tests: unit acceptance + L1 fallbacks 218 passed / 1 skipped (cluster has no ≥ 3-device axis without a wrap link),
  R3 overflow / sentinel + precision baseline 16 / 16 (`--no-precompile`); golden `test_op_loose` + `test_regression` +
  `test_ring_mock` 39/39; golden `test_op` INTERLEAVED 3603 passed / 720 xfailed, HEIGHT_SHARDED 3468 passed / 620
  xfailed, WIDTH_SHARDED 3599 passed / 720 xfailed / 1 failed — the same pre-existing cell as Perf 1 (4096×1024 fp32
  WIDTH_SHARDED axis 0 Linear 1 link: "per-core footprint 301184 B … exceeds L1 budget 216576 B", a host-side L1 check
  under suite-order fragmentation), which passes alone → upstream L1-fit work, not perf. Ring cells are pruned at
  collection on this box's 2×2 system mesh (covered by `test_ring_mock` and the 1×4 drivers).
- Pre-existing note (not caused by this change, reproduced bit-for-bit by the new path): under bf16 DEST a +inf record
  finalizes to +0, not −0, in the bench, which the arrival sentinel would read as "not arrived". R3 says the FPU
  saturates before inf, so it is not reachable in practice; flagged for the next correctness pass.
- Summary: 2 ideas measured; 1 graduated (focus op −1.3 µs traced slowest, −0.9 µs last-launched, −1.0 µs stub;
  faster on every guard cell but one inside noise), 1 null; no regression. Focus is now 19.4 µs vs the 11.9 µs target.
- What binds next (focus, slowest chip): launch skew, then the late chip's path to its first publish (land → gather →
  chip-sum → publish ≈ 2.4 µs per wave) and the post-landing chain (collect 0.6, all-gather 0.75, move 0.3, scale 0.4,
  the last-wave store burst at the DRAM rate). `dedicated_owners` d3 is a measured −1.5 µs local (stub) lever kept in
  its experiment dir, worth re-measuring once the early chip stops being bound by the wave-0 publish.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `copy_tile` (UnpackToDestFp32) | capability | no API unpacks N tiles to DEST in one unpack transaction, or only a sub-face row range: every `copy_tile` pays its own math↔unpack handshake (~80 ns) and moves the whole 4 KiB tile for a 128 B record. `ff_unpack_records_to_dest` is `_llk_unpack_A_` with the MOP replaced by one 32-datum UNPACR per record and a z hop | 638 (G slots via copy_tile + custom pass) | 357 (one transaction) | distributed_rms_norm_compute.cpp:135 |
| `add_binary_tile` | capability | fixed at the whole tile (VectorMode::RC, 32 vector ops per add) and one SFPU call per position; nothing folds G DEST slots into one accumulator in a single pass. `ff_fold_body` does G loads + G − 1 fp32 adds (same SFPU ops and order) + the finalize over vectors 0 / 1 of face 0 | 1030 (per-position adds) | 638 (one fused pass) | distributed_rms_norm_compute.cpp:107 |
| `pack_tile` | capability | always packs the output CB's full tile shape; no partial-face pack of the 128 B record. Packed with the face_r_dim 2 / 1-face MOP (`llk_pack_init_impl`), full-tile MOP restored after the push | 553 (full-tile pack) | 531 (record pack; it also enables the compact wire's dropped RISC-V join) | distributed_rms_norm_compute.cpp:220 |
The compact combination of all three measures 251 ns per G = 4 fold (vs 1030). The bypass justification is also in the
compute kernel's head comment.

Artifacts: `perf_experiments/fold_finalize/` (isolated bench `test_fold_bench.py` + `kernels/`, op copy `drms_ff/` with
every option behind `DRMS_FF_VARIANT`, `test_ff_op.py` A/B driver, `test_ff_zones.py`, `test_ff_r3_overflow.py`,
`graduate/` + `graduate_*.diff`), `perf_experiments/dedicated_owners/` (op copy `drms_do/` behind `DRMS_DO`,
`dedicated_owners_vs_op.diff`, `test_do_perf.py`, `results_raw.txt`, `zones_focus_summary.txt`, `tools/`). Drivers need
`--import-mode=prepend`.
