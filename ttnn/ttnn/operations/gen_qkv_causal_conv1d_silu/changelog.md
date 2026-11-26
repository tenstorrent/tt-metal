# Changelog: gen_qkv_causal_conv1d_silu

## Phase 0 — Core Implementation
- **Date**: 2026-11-26
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - One `ttnn.generic_op` program per call, with two regimes: ROW_MAJOR sticks read directly, or TILE untilized in-kernel.
  - Windowed tilize does the token shift as exact data movement. Four row-broadcast MACs accumulate in one DEST slot, SiLU runs on the packer thread, and there is one pack.
  - The writer routes each tile to q/k/v.
  - Work is split across channel groups × token ranges, using the full device grid.
  - Taps are multicast once per channel group through the kernel_lib mcast helper.
- **SUPPORTED at Phase 0**:
  - dtype=[bfloat16], layout=[ROW_MAJOR, TILE]
  - in/history/taps buffer=[dram, l1], output_memory=[default, dram_interleaved, l1_interleaved]
  - fp32_dest_acc_en=[False, True], history_layout=[token_paged], input_window=[exact]
  - width_split / channel_count / seq_tiles / chunk_blocks = every value; EXCLUSIONS=[]
- **Accuracy achieved** (HiFi4, `test_gen_qkv_causal_conv1d_silu_precision_baseline.py`, 4 shapes T 32…4096 + the fp32 cancelling-taps cell):
  - 16-bit DEST: PCC=0.999987, max_abs_err≤0.181, mean_abs_err≈0.0033, rel_rms_err≈0.0056, ULP p99 17–18
  - fp32 DEST: PCC=0.999998, max_abs_err=0.0078, rel_rms_err=0.0018, ULP p99 0 (max 1); FP32_DEST_BOUND holds
- **Golden suite at Phase 0**: 17 / 21 passing, 4 xfail_expected (012, 013, 018, 019: channel_paged / column_window), 0 supported_fail / xpass_drift / xfail_wrong_mode (`verifier_report.json`).
- **Perf (flagged targets, BH 11×10)**: 001 8.7 µs, 002 10.4 µs, 003 13.2 µs, 004 128–131 µs. All are under native (30.5 / 43.1 / 48.1 / 346.9) and under target (17.0 / 18.5 / 22.5 / 164); 004 stretch is 117.
- **Issues encountered** (implementer, recorded in `l1_ledger.md` and breadcrumbs):
  - The tap fetch was DRAM-request-bound, so tap multicast was built (004: 185.9 → 152.9 µs).
  - SiLU was moved to the packer thread (197.7 → 185.9 µs).
  - Default `block_rows` was capped at 1 for overlap (153 → 129 µs).
- **Verifier fixes**:
  - Fixed the acceptance-test harness bug in `test_repeat_is_bit_identical` (`case.values[0]` on a dict → `TypeError`).
  - CB capacities now have one source, `cb_layout()` (it previously had to be kept in sync by hand between `footprint()` and the descriptor).
  - Added `page_channels` to the program-cache key (latent BB-07 hazard).
  - `HISTORY_ROWS` is now imported from `HALO_ROWS`.
  - Added a pinned TILE multi-row-block test.
  - Added an as-built note to `op_design.md` and a single-source note to `l1_ledger.md`.
- **Tests added**: `test_gen_qkv_causal_conv1d_silu.py` (acceptance, planner), `test_gen_qkv_causal_conv1d_silu_debug.py` (BB-07 cache-hit correctness + pinned TILE multi-row block), `test_perf_knobs.py` (implementer knob sweep), `test_gen_qkv_causal_conv1d_silu_precision_baseline.py` (verifier).

## Refinement 1 — Reader addressing generality: input_window column_window + history_layout channel_paged
- **Date**: 2026-11-26
- **What was done** (reader-only knob-turns; no CB, block, schedule or L1 change):
  - **column_window**: the existing `input_col_offset_tiles` addressing already covered both regimes (RM byte offset `2·(offset + c0)` in the token-row page; TILE page `r·W_t + offset/32 + c`, including the t0 > 0 halo rows / extra untilized tile-row). Opened `SUPPORTED["input_window"]`. The TILE page stride `W_t` is now `ceil(W/32)` (the padded width), so a W that is not a multiple of 32 is **supported** natively in both layouts rather than refused (RM: the read offset stays 64 B aligned inside the 2W-byte page).
  - **channel_paged**: new reader helper `read_history_halo` (reader CT arg `history_page_channels`, 0 = token-paged; token-paged path unchanged via `if constexpr`). History `[1, C/P, 3P]`: row r of channel c is page `c / P`, element `r·P + c mod P`; a group whose channel slice straddles pages issues one read per contiguous run per row (every run 32-channel / 64 B aligned). Only the `t0 = 0` core of each group reads history.
  - `page_channels` is threaded op → `create_program_descriptor` (it was already in the program-cache key).
  - Reused: the reader's halo/body addressing, the plan, every CB, the cache key. Added: `read_history_halo`, one CT arg, the padded-width stride.
- **Accuracy achieved** (golden operands vs float64 reference, q / k / v):
  - conv1d-012 (RM, P 160, 16-bit DEST HiFi4): PCC=0.999987, rel RMS=0.0056
  - conv1d-013 (RM, offset 32, HiFi2): PCC=0.999983, rel RMS=0.0061
  - conv1d-018 (TILE, L1 in/history, P 32, offset 64, fp32 DEST HiFi2): PCC=0.999992, rel RMS=0.0058
  - conv1d-019 (TILE, L1 in, P 32, W 448 > C 384, fp32 DEST HiFi4): PCC=0.999998, rel RMS=0.0018, ULP p99 0; FP32_DEST_BOUND holds
- **Golden test progress**: 21 / 21 expected (012, 013, 018, 019 move from xfail_expected to passing; confirmed on a `-k` slice of those four + 001, 002, 005).
- **Perf (hot-path no-regression, BH 11×10, device kernel ns, same session, before → after)**: 001 8.39 → 8.48 µs, 002 9.67 → 9.62, 003 12.86 → 12.92, 004 128.97 → 127.73, 020 74.49 → 73.30, 005 6.88 → 6.89 — all within noise.
- **Issues encountered**: `--profile` runs end with "tracy-capture exited with code 3" on this box (no `ops_perf_results*.csv`); the device profiler's own `generated/profiler/.logs/cpp_device_perf_report.csv` still has one row per program and was used instead.
- **Tests added**: `test_reader_addressing.py` — the four golden cells; pinned plans covering every halo source under a column window + channel-paged history (RM later-block L1 halo, RM DRAM halo at t0 > 0 with a page-straddling group, TILE later-block, TILE extra-untilized-row at t0 > 0, TILE multi-row block); W = 2200 (not a multiple of 32) in both layouts; C = 3P (token- and channel-paged histories with the same shape, back to back).

## Refinement 2 — Speed up the latency-bound flagged profiles (conv1d-001, -002, -003)
- **Date**: 2026-11-26
- **Diagnosis** (define-gated ablations, `test_perf_r2.py`, BH 11×10):
  - On 001, the reader's end time spreads from 2.2 to 6.8 µs across cores, ordered by core position. That is a shared, queued resource.
  - No input reads: 8.39 → 4.34 µs. Half the stick reads with all 8 banks kept: 6.35 µs. Half the stick reads on only 4 banks: 8.39, no change. No taps: −0.8 µs. No history: −0.8 µs.
  - So decode is bound by DRAM request *rate*, at about 12.5 ns per request per bank endpoint. Bytes don't matter, and neither does RISC issue.
  - NoC0 and NoC1 reach each bank through different endpoints (`dram_views` worker endpoints). A quick experiment with the writer reading half the rows on NoC1 gave 001 8.35 → 6.44 µs.
- **What was done** (all levers are plan knobs: reported in `Plan`, pinnable through `plan_limits`, part of the cache key):
  - **Split reader (ROW_MAJOR)**, `Plan.split_rows`, default `SPLIT_READER_ROWS = 24`.
    - The writer RISC (BRISC, NoC1) reads body rows `[32 − split_rows, 32)` of every tile-row into the reader's `cb_rows` slot. The reader (NoC0) reads the rest plus taps, history and halos.
    - The handshake uses two 32 B CBs: `cb_split_requests` (reader → writer, the row's L1 address) and `cb_split_done` (writer → reader). The reader pushes the rows only after both halves have landed.
    - Before waiting on output tile-row m, the writer serves every row through the end of the *next* block. This covers multi-row blocks and ragged tails and is deadlock-free.
    - Sweep (split 0 / 16 / 24 / 32): 001 8.37 / 6.45 / 5.96 / 6.79 µs; 002 9.90 / 7.74 / 7.21; 003 13.06 / 11.39 / 11.00; 020 73.7 / 62.1 / 59.5; 004 127.5 / 129.0 / 124.3 (neutral, compute-bound).
    - 24 balances NoC0 (8 rows + taps + history) against NoC1 (24 rows), at about 240 requests per bank port on 001.
    - The TILE regime keeps `split_rows = 0`: it issues w tile reads per tile-row, not 32 stick reads.
  - **Output push quantum**, `Plan.out_group_tiles`.
    - The compute acquires, packs and pushes each tile-row in DEST groups of this size. The writer waits for the same quantum and routes each tile to q/k/v (per-tile routing replaces the per-row segments).
    - Default is 1 when a core owns one tile-row: the whole output is a tail, so the writer stores tile c while SiLU runs on c + 1. Otherwise the default is `min(w, DEST capacity)`.
    - Single-row cells: 002 7.32 → 6.90, 003 10.93 → 9.89, 014 9.82 → 8.66, 016 19.29 → 15.64 µs.
    - Multi-row cells keep the full DEST group, because a 1-tile quantum regressed 004 (123 → 164) and 020 (60 → 69): one 2 KB write in flight per core under DRAM write contention.
  - **bank_stagger**: measured null (001 / 002 / 003: 5.87 / 6.81 / 10.02 → 5.85 / 7.02 / 9.95). Every core queues its whole tile-row within ~1 µs, so per-port totals bind, not issue order. Parked as the CT switch `STAGGER_ROW_WALK = False` in the reader and writer.
  - **Reverted**: a lagged-pop writer (flush and pop group g−1 after waiting on g). It regressed 004 at its default, 127 → 143 µs, by holding output CB space.
  - **Not landed**: a request-aware `block_ch_tiles` term in `select_plan`.
    - w = 2 on 001 measures 5.66 vs 5.86 µs (median of 4 runs each, −3.5%, on 40 cores).
    - Offline, a calibrated model flips 5 golden plans, including 3 multi-row cells (004 / 008 / 020) where the per-request cost does not apply: their 1 KB reads are bandwidth-bound.
  - Reused: the `cb_rows` slot and its push protocol, the reader's addressing, the writer kernel, the DEST-group loop of `conv_silu_block`, and the plan / `plan_limits` / cache-key plumbing. Added: two handshake CBs, the writer's input `TensorAccessor`, and two `Plan` fields.
- **Perf** (device kernel µs, Refinement 1 → Refinement 2, same session):

  | Group | Cell | R1 → R2 |
  |---|---|---|
  | Targets | 001 | 8.26 → 6.03 (−27%) |
  | Targets | 002 | 9.75 → 6.83 (−30%) |
  | Targets | 003 | 12.99 → 9.86 (−24%) |
  | Guard set | 004 | 128.2 → 127.4 (medians of 4 / 7 runs; HEAD spread 122.8–132.2, noise) |
  | Guard set | 005 TILE | 6.91 → 6.58 |
  | Guard set | 006 L1 in | 6.25 → 5.60 |
  | Guard set | 010 L1 out | 10.41 → 7.25 |
  | Guard set | 011 fp32 | 14.27 → 11.33 |
  | Guard set | 012 channel-paged | 19.65 → 15.64 |
  | Guard set | 013 column-window | 8.10 → 6.19 |
  | Guard set | 018 | 3.54 → 3.57 |
  | Guard set | 019 | 18.59 → 18.73 |
  | Guard set | 020 | 73.7 → 59.5 |

  All three targets stay well under target_us (17.0 / 18.5 / 22.5).
- **Accuracy achieved**: bit-identical to Refinement 1. Both knobs only move data; `test_split_reader.py` asserts `torch.equal` against `split_rows = 0` and across `out_group_tiles`.

  | Cell | PCC | rel RMS | ULP p99 |
  |---|---|---|---|
  | 001 | 0.999987 | 0.00556 | 18 |
  | 003 | 0.999987 | 0.00560 | 17 |
  | 020 | 0.999987 | 0.00558 | 17 |
  | 004 | 0.999987 | 0.00561 | 18 |
  | 011 (fp32 DEST, cancelling) | 0.999998 | 0.00177 | 0 |

  All match the Phase 0 precision baseline.
- **Golden test progress**: 21 / 21 expected. A 15-cell slice passed: 001–006, 010–014, 018–021.
- **Remaining headroom (finding)**:
  - **001**: the reader now sits at ~240 requests per bank port on banks 0–2 (history + taps + 8 rows on NoC0; 24 rows on NoC1), about 4.0 µs. A 1.5 µs compute tail follows (precise SiLU ≈ 0.7 µs) and a 0.3 µs write.
    - Next lever: "read wide, compute narrow". Pairs of cores read w = 2 sticks, which halves input requests, and exchange halves over L1 → L1, so compute stays at w = 1.
    - Estimated ≤ 0.7 µs. Not done: it is a cross-core scheme change.
  - **002 / 003**: now compute-bound per core: windowed tilize + 4 MACs + precise SiLU ≈ 1 µs per tile, on 80 of 110 cores.
    - Next levers: the SiLU cost (Refinement 3) and non-uniform channel groups to use all 110 cores (design perf lamp "Grid synchronization").
- **Issues encountered**:
  - `--profile` still ends with "tracy-capture exited with code 3", so `generated/profiler/.logs/cpp_device_perf_report.csv` was used again.
  - pytest de-duplicates repeated node ids, so repeats need separate invocations.
  - The first `-k "a or b"` through `run_safe_pytest.sh` split the expression into separate arguments; node ids work.
- **Tests added**:
  - `test_split_reader.py` (19 cases):
    - split on/off bit-identity at split 8 / 24 / 32 (decode, DRAM halo at t0 > 0, L1 input multi-block, L1 output multi-block);
    - `block_rows = 3` with a ragged tail and `block_rows = 4` with a DRAM halo;
    - `out_group_tiles` bit-identity (ragged group at w = 3, multi-row, fp32-DEST clamp);
    - the default rules.
  - `test_perf_r2.py`: the measurement harness (skipped unless `GQKV_PERF_R2=1`), with plan pins, `block_ch_tiles` overrides, define injection and channel-paged cells.
