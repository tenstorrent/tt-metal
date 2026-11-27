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

## Refinement 3 — Speed up the DRAM-bound prefill profile (conv1d-004)
- **Date**: 2026-11-26
- **What was done**:
  - **(a) SiLU SFPU cost — new body `silu_pack::silu_bf16_rows`** (compute kernel, 16-bit DEST only; a 32-bit DEST keeps the library `calculate_silu`):
    - exp(−x) = 2^y: the clamped `y + 639` lies in the exponent-9 binade, so ONE `SFPSHFT` by 9 yields the Schraudolph pattern z = 2^floor(y)(1+frac) (the library spends exexp/exman/shft on it).
    - A quartic correction in m = setexp(z, 127), Remez on relative error (1.1e-4 vs exp_21f's 1.7e-3); `approx_recip` + one Newton step without the NaN mask (1 + z·g ≤ 2^127 stays finite); final bf16 RNE.
    - Two rows interleaved per step, each polynomial coefficient loaded once per pair. Blackhole scoreboards dependent SFPU ops, so a serial one-row chain stalls on every SFPMAD: the same arithmetic one row per step cost 66.4 µs/pass despite fewer instructions than the library's 26.
    - SFPU time of one pass over conv1d-004 (96 tiles/core; T(3 passes) − T(2 passes), define `GQKV_SILU_REPEAT`): **62.8 µs vs 79.2 µs** library (−21%).
    - Precision: correctly rounded for 99.45% of inputs on device vs 90.6% for the library body (`test_silu_body.py`, isolated SiLU), always within 1 bf16 ULP, clamp extremes exact. `math_approx_mode=True` stays refused.
    - Dispatch: the packer-thread loop mirrors `apply_activation_from_pack` (the helper has no slot for another body); raw-LLK deviation documented at the kernel head.
  - **Selection = plan knob `Plan.silu_body`** (`default_silu_body`, pinnable via `plan_limits(silu_body=...)`, in the cache key, reported by `get_last_plan()`): quartic when a core owns one tile-row, library otherwise (and always with fp32 DEST). Reason, measured below: on multi-row plans the faster body exposes a DRAM/NoC fairness tail.
  - **(b) buffer depths** `ROWS_DEPTH` / `OUT_DEPTH` (existing knobs): measured null on 004, left at 2.
  - Parked live knobs (defaults reproduce the previous sequence exactly):
    - `SILU_TILE_STALL = True`: the library SFPU start's per-tile `STALLWAIT(STALL_SFPU, MATH)`. It is redundant after the MATH_PACK semaphore wait (math posts only after `STALLWAIT(MATH | WAIT_SFPU)`), but removing it measured null (129.9 → 129.9 µs).
    - `READER_NOC` / `WRITER_NOC`: single source for the data-movement kernels' NoCs (defaults NOC_0 / NOC_1). The swap measured 184 µs on 004.
  - Measurement hooks (define-gated, compiled out by default): `GQKV_ABLATE_SILU`, `GQKV_SILU_REPEAT` (compute) and `GQKV_ZONES` (reader/writer DM-timeline zones R_RSV / R_RD / R_SPLW / W_SERVE / W_WOUT / W_WR).
  - Reused: `conv_silu_block`, every CB, the plan / `plan_limits` / cache-key plumbing, the library init helper for the library body. Added: one SFPU body plus its init, the packer-thread dispatch, one `Plan` field, 2 compute CT args, `_dm_config`.
- **Diagnosis of conv1d-004** (BH 11×10, device kernel µs; harness `test_perf_r3.py`):
  - Library body 129.6 (median of 3; noise band 122.7–131.8); SiLU removed entirely 120.3; cheaper body ~150.
  - Two-stage model fits every repeat count: T(k) = max(B, A + k·S), with A ≈ 47 and S = 79.2 (library) / 62.8 (quartic). The library sits at the knee (A + S ≈ 126 ≈ B). So 004 is **balanced between the packer's SiLU path and the dataflow path**.
  - Padding the cheaper body with 8 SFPNOPs per row brought 004 back to 130.3, i.e. the cost is not SFPU work. DM zones show why: with the cheaper body the median core finishes at 105.9 µs (library: 115.1) and most of the grid ends at 99–112. But cores at physical x = 13–15, y = 4–10 straggle to 150: their first two-row read takes 34 µs (typical 17) and their first 8-tile output write 42.6 µs.
  - Mechanism: once compute stops pacing it, the aggregate DRAM demand (~470 GB/s) exceeds capacity, and NoC arbitration starves the cores with the longest routes from the DRAM endpoints.
  - The starved corner moves with the split-reader ratio:

    | Split | Op µs | Starved region |
    |---|---|---|
    | 24 (NoC1 carries 24 rows + all writes) | ~150 | bottom-right |
    | 16 | 138.7 | — |
    | 8 | 125.0 | p50 103, p100 136 |
    | 4 | 130.5 | — |
    | 0 (all reads on NoC0) | 137.8 | top rows |

  - Nulls on 004:

    | Lever | Op µs |
    |---|---|
    | Bank stagger | 155 |
    | NoC swap | 184 |
    | Depths 3/3 (median of 3, alternating with the default) | 128.8 |
    | Depth `ROWS_DEPTH = 3` (median of 3) | 130.7 |
    | `OUT_DEPTH = 3` | 138.4 |
    | Per-tile stall off | null |

- **Perf** (device kernel µs, library → shipped default; single runs, small cells' noise ~1–2%):

  | Group | Cell | Before → after |
  |---|---|---|
  | Target | 004 | 129.1 → 129.6 (median of 3), unchanged, library body |
  | One-tile-row plans (quartic) | 001 | 5.89 → 5.80 |
  | One-tile-row plans (quartic) | 002 | 6.88 → 6.56 (−4.7%) |
  | One-tile-row plans (quartic) | 003 | 9.84 → 9.30 (−5.5%) |
  | One-tile-row plans (quartic) | 005 TILE | 6.54 → 6.41 |
  | One-tile-row plans (quartic) | 006 L1 in | 5.39 → 5.11 (−5.2%) |
  | One-tile-row plans (quartic) | 010 L1 out | 7.23 → 6.85 (−5.3%) |
  | One-tile-row plans (quartic) | 012 channel-paged | 15.48 → 14.36 (−7.2%) |
  | One-tile-row plans (quartic) | 013 column-window | 6.10 → 5.80 (−4.9%) |
  | One-tile-row plans (quartic) | 014 | 8.60 → 8.01 (−6.9%) |
  | One-tile-row plans (quartic) | 016 | 15.60 → 14.33 (−8.1%) |
  | Same kernel as before (library) | 020 | 59.4 → 59.6 |
  | Same kernel as before (library) | 011 fp32 | 11.32 → 11.22 |
  | Same kernel as before (library) | 018 | 3.59 → 3.65 |
  | Same kernel as before (library) | 019 | 19.07 → 18.89 |

  No regression.
- **Accuracy achieved** (`test_gen_qkv_causal_conv1d_silu_precision_baseline.py`, HiFi4):

  | Cell | Body | PCC | rel RMS (Phase 0) | ULP p99 |
  |---|---|---|---|---|
  | 001 | quartic | 0.999987 | 0.00555 (0.00556) | 18 |
  | 003 | quartic | 0.999987 | 0.00559 (0.00560) | 17 |
  | 020 | library | 0.999987 | 0.00558 (=) | 17 |
  | 004 | library | 0.999987 | 0.00561 (=) | 18 |
  | 011 fp32 DEST, cancelling | library | 0.999998 | 0.00177 (=) | 0 |

  Everything is at or better than the Phase 0 baseline. A cubic correction (6.6e-4) measured +0.00001 rel RMS: its small positive bias no longer cancelled the 16-bit-DEST positive bias the way the library body's negative bias does. So the quartic, which carries exact rounding's bias, shipped.
- **Golden test progress**: 21 / 21 passing (full file, one run).
- **Remaining headroom (finding)**:
  - 004 is bound by DRAM/NoC fairness, not compute: the median core can finish ~100–106 µs (stretch 117), the op is set by the starved-corner tail.
  - Next lever is the dataflow side: per-core NoC load balancing (a per-core split ratio or a dynamic split where the reader and writer claim a tile-row's sticks from opposite ends), or demand shaping (bounded outstanding requests per core).
  - Not done here: a static per-core split is topology-specific (the starved corner depends on DRAM-endpoint positions and harvesting), and the dynamic split needs a race-free cross-RISC claim protocol. Both are scheme changes for a dataflow phase. With that tail removed, the quartic body would lift multi-row plans too: flip `default_silu_body`.
- **Issues encountered**:
  - A cheaper SFPU body made 004 slower, which looked like an SFPU problem. Bisection (knock-outs, NOP padding, repeat counts) plus DM zones and per-core end-time maps located the DM tail instead.
  - One transient JIT build failure on 018 under `--profile` did not reproduce.
- **Tests added**:
  - `test_silu_body.py` (7): the default-body rule, including fp32 DEST, and an isolated-SiLU accuracy test (taps 0,0,0,1, so y = x exactly) for both bodies on a one-row (001) and a multi-row (020) plan, with clamp extremes.
  - `test_perf_r3.py`: the measurement harness (skipped unless `GQKV_PERF_R3=1`; `GQKV_ZONES=1` adds DM zones).

## Perf 1 — Perf tournament round 1: measured breakdown, 2 experiments, none graduated
- **Date**: 2026-11-26/27. BH 11×10 (110 cores), device kernel µs, one fresh-cache profiled run per variant unless noted. Raw tables: `perf_experiments/perf1_coordinator_measurements.md`.
- **Focus shape**: all four `target` cases carry a PERF FOCUS note.
  - **conv1d-004** (prefill 4096) is the primary target. It is the only one with an unmet numeric goal: stretch 117.13 µs against ~124–131 µs today. Its target, 164 µs, is met.
  - 001, 002 and 003 are already ~3× under target; they were measured as secondary targets and must not regress.
  - Every config was measured exactly as flagged: RM input, all DRAM, `compute_kernel_config=None` (HiFi4, 16-bit DEST), bf16.
- **What shipped**: instrumentation only. The op's math, dataflow and plan are unchanged.
  - **`ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`** was absent in this tree. It was added with the same content other branches carry (`MaybeDeviceZoneScope` = `DeviceZoneScopedN`), plus a `MAYBE_DEVICE_ZONE_SCOPE_OFF` kernel define that compiles the zones out of a profiled build.
  - **Permanent stage zones** in every kernel:
    - reader: `reader_taps_issue`, then per block `reader_rows_reserve` and `reader_halo_issue`, then per tile-row `reader_rows_issue`, `reader_rows_barrier` and `reader_split_wait`; TILE regime `reader_tile_issue`, `reader_untilized_wait`, `reader_slot_copy`;
    - writer: per served row `writer_split_req_wait`, `writer_split_issue`, `writer_split_barrier`; per output group `writer_out_wait`, `writer_out_issue`, `writer_out_flush`;
    - compute: per block `compute_rows_wait`, `compute_tilize`, `compute_windows_wait`, `compute_conv_silu`; `compute_untilize` (TILE regime); `compute_taps_wait` once.
    - Wait, issue and barrier are separate zones throughout. They replace the define-gated `GQKV_ZONES` R_*/W_* zones.
    - The marker budget holds: at most 108 user markers per RISC on 004, against a budget of 250.
  - **Measurement-only ablation switches**, compiled out by default: `GQKV_ABLATE_TILIZE` and `GQKV_ABLATE_MAC` (compute), `GQKV_ABLATE_READS` (reader + writer), `GQKV_ABLATE_WRITES` (writer). These join the existing `GQKV_ABLATE_SILU` / `GQKV_SILU_REPEAT`.
  - **Harness**: `test_perf_t1.py` (env `GQKV_PERF_T1=1`: ablation, `nozones*` and descriptor-constant variants) and `zone_stages.py` (per-RISC span max/p50, per-zone summed duration, marker count).
- **Zone cost under the profiler (measured; production builds are unaffected)**:
  - The zones compile to nothing without `PROFILE_KERNEL`, but every device-ns figure is taken with the profiler on.
  - Same kernels, zones on vs `MAYBE_DEVICE_ZONE_SCOPE_OFF`: 001 6.03 vs 5.71 (+5.4%), 002 6.90 vs 6.49, 003 9.70 vs 9.15 (+6%), 006 5.54 vs 5.10 (+8%), 010 7.27 vs 6.90, 013 6.12 vs 5.83, 005 6.53 vs 6.28, 020 60.8 vs 59.5 (+2%); 016 and 004 within noise.
  - The cost is ~20–40 ns per zone on the critical path, spread over all three kernels (001: compute 0.09, reader 0.11, writer 0.02 µs).
  - With the zones off, the op reproduces Refinement 3 (001 5.70 vs 5.80, 003 9.15 vs 9.30, 020 59.5 vs 59.6).
  - **Compare profiled numbers across this boundary only with the switch.**
- **Measured breakdown, conv1d-004** (cumulative peel; µs):

  | Variant | µs |
  |---|---|
  | full | 131.0 |
  | − SiLU | 122.9 |
  | − SiLU − MAC | 110.7 |
  | − all compute (DM only) | 111.0 |
  | DM only − reads (writes only) | 75.1 |
  | sync only | 8.4 |
  | compute only (no DM) | 108.6 |
  | compute only − SiLU | 40.4 |
  | compute only − SiLU − MAC (tilize) | 18.3 |
  | no reads / no writes | 114.6 / 123.0 |
  | **faster compute**: − tilize / − MAC / quartic body | **149.2 / 151.2 / 148.8** |
  | quartic, compute only | 91.9 |

  - **Compute (108.6) and DM (111.0) are balanced**, and the overlapped op is 131.
  - The pack thread is the busiest compute thread: `compute_conv_silu` 93 µs + `compute_tilize` 24 µs per core. That is SiLU's SFPU pass (68 µs, library body), 4 window packs per output tile, and the output pack.
  - **Every variant that makes compute faster makes the op slower** (~150). DM is not at its roofline, and its slack depends on compute pacing it.
- **Core balance and where DM time goes (004)**:
  - `*-KERNEL` max/p50: full 1.13, DM-only **1.42** (per-core end 29 → 111 µs), writes-only **1.60** (12.8 µs at the top-left and the bottom row, 75 µs at the bottom-right).
  - In the writes-only run, `writer_out_issue` spans 4.9 → 65 µs across cores: **the NoC1 DRAM write path is strongly position-unfair**.
  - The writer RISC (NoC1) also carries 24 of the 32 input rows per tile-row, and serves them only between its output groups. So the reader waits `reader_split_wait` 71 µs (p50) of its ~84 µs span.
  - The reader's own 8 NoC0 rows (`reader_rows_barrier`) are uniform at 7–12 µs everywhere.
  - In the full run, the unpack thread's input starvation (`compute_rows_wait`) grows from ~7 µs at the left to 55–58 µs at the bottom-right corner (physical x 14–15, y 8–10).
- **Other shapes**:

  | Cell | Full | DM only | Writes only | Compute only | Sync only | DM max/p50 |
  |---|---|---|---|---|---|---|
  | 020 | 60.8 | 51.8 | 33.3 | 52.5 | 6.2 | 1.34 (writes-only 1.57) |
  | 003 | 9.69 | 6.75 | 3.86 | 6.05 | 2.62 | — |
  | 001 | 6.05 | 4.73 | 2.34 | 3.27 | 2.13 | — |

  On the one-tile-row cells, read → compute → write serialize through the single tile-row, behind a ~2.1–2.6 µs sync/launch floor.
- **Roofline gate**: the `noc_estimate` CLI is not built in this tree (it is a `--build-tests` target), so the gate used the spec's DRAM floor.
  - 004 moves 42 MB, so the floor is 82 µs at 512 GB/s, or ~89–100 µs at the practical 82–92% of peak.
  - DM-only (111) and writes-only (75 µs, 280 GB/s against a ~41 µs write floor) are **not** at their roofline. Their p50 cores finish at 78 / 47 µs; it is the tail that sets the time.
- **Ranked bottleneck**:
  1. NoC1 DRAM writes: position-unfair, and they gate the split reads.
  2. The writer RISC's input share, serialized behind its writes.
  3. Pack-thread compute (SiLU SFPU 68 µs, window packs). This is blocked behind DM: faster compute measured slower.
  4. Decode read request rate (001: 2.4 µs over a 2.1 µs sync floor).
- **Portfolio floated** (cap: 2 experiments):
  - **A `split_writes`** (tested): the NCRISC (NoC0), idle ~71 µs per core on 004, writes a share of the q/k/v tiles, so DRAM writes use both NoCs' link sets.
  - **B `dynamic_split_reads`** (tested): the reader (NoC0) and writer (NoC1) claim each tile-row's sticks dynamically instead of the static 8/24 split.
  - **C** cross-core tile-row work stealing within a channel group: not tested. The only fetch-and-add is the internal `noc_atomic_read_and_increment`, with no dataflow-API wrapper.
  - **D** fused tilize + MAC (`unpack_tilizeA_B` into the row-bcast multiply; the deferred `tilize_fused_mac` regime): not tested. It is compute, which measured blocked behind DM.
  - **E** quartic body on multi-row plans: to re-measure after any DM graduation.
  - **F** read-wide / compute-narrow pairs for decode: not tested; small absolute gain.
- **A — `split_writes`** (`perf_experiments/split_writes/`)
  - **Mechanism**: tile (r, cg) is written by the NCRISC iff `((r + cg)·S) % 8 < S`, with S = 4. The writer stays the single popper of `cb_output_tiles`. The NCRISC shadow-reads the CB's `tiles_received` counter and keeps its own read pointer, flushes its writes, then hands over a token in a new CB 8. Every reader wait becomes a poll that serves ready groups.
  - Bit-identical on 14 cells; clean under `--dev`.
  - **Optimizer, on 004**:
    - library body 123.8 → 123.6 (flat);
    - quartic body 149.6 → ~130;
    - the share menu with quartic: S=2 145.7, S=3 143.3, S=5 141.7, S=6 162.1, S=8 175.3;
    - dynamic and overflow variants 134–170;
    - writes-only 74.4 → 68.2 (S=2: 61.2).
  - **Coordinator whole-op A/B in the real op** (two sessions, alternating order; `perf1_coordinator_measurements.md`):
    - wins: 003 −6.1%, 006 −5.0%, 011 −4.5%;
    - 004 with the quartic body −14% (129 vs 147–152), but still slower than the library body (~123);
    - flat: 004 default, 005, 013, 019, 020;
    - **measured regressions: 016 +6.7%, 012 +5.2%, 001 +3.3%, 010 +2.9%**; 002 +2.2% and 014 +2.3% are at the noise edge.
  - **Mechanism of the regressions**: on 016 the NCRISC's NoC0 writes are slow only on the top four core rows (3–5 µs vs 0.4 µs elsewhere); on 003 they are 0.2 µs on every core. This is position-dependent NoC0 congestion of the single-row output burst.
  - **Verdict**: mixed domain, **not graduated**. Wins and regressions are all single-row plans. By tiles per core: w=1 regresses, w=2 is mixed (006 wins, 010 regresses), w=4 wins, w=8 regresses. No predicate that tracks the mechanism separates them. The only principled carve-out (single-row plans) removes every measured win and leaves a change that is flat everywhere it would apply. A guard keyed to the measured cells would be an allow-list of shapes, and graduating unguarded regresses four cells. The graduated version was applied and then reverted; `graduate/split_writes_graduation.diff` holds it.
- **B — `dynamic_split_reads`** (`perf_experiments/dynamic_split_reads/`)
  - **Mechanism**: shared L1 claim words per core; the reader claims from the front, the writer from the back while it waits on compute; NoC1 issue/response counters replace the writer's barrier. Bit-identical 76/76.
  - **Best variant on 004** (reader window 32, writer 8, writer counts its writes): library 124.1 → 127.0 (NULL, noise), quartic 148.7 → 126.4.
  - **Regressions**: 001 +22%, 002 +15%, 006 +15%, 010 +13%, 013 +15%, 003 +4%; 011, 012 and 020 +2.5–3%. The decode-friendly window (8/8) loses 7–10% on 004.
  - **Mechanism**: `reader_split_wait` fell 70 → 0.3 µs (p50), but the far corner moved to the NoC1 output writes (packer stalls behind the writer), and the top rows became NoC0 read-starved.
  - **Verdict**: NULL on the focus shape, REGRESSION on single-row plans; **not graduated**.
- **Graduated**: none. Whole-op before/after is unchanged by construction: the kernels differ from Refinement 3 only by zones that compile out without the profiler, and zones-off profiled runs reproduce Refinement 3 (see above).
  - Current profiled numbers with zones on:

    | Cell | µs |
    |---|---|
    | 001 | 6.01–6.06 |
    | 002 | 6.87–6.90 |
    | 003 | 9.56–9.75 |
    | 004 | 122.8–131.0 (bimodal: ~123.5 / ~130.5) |
    | 005 | 6.53–6.68 |
    | 006 | 5.53–5.62 |
    | 010 | 7.27–7.33 |
    | 011 | 11.7 |
    | 012 | 14.7–14.8 |
    | 013 | 6.12–6.24 |
    | 014 | 8.4–8.5 |
    | 016 | 14.4–14.7 |
    | 019 | 19.5–19.6 |
    | 020 | 60.5–60.8 |
- **Guard set and golden**: golden 21/21 and the op's unit directory (112 passed) are green, with the instrumentation and with the zone kill switch in the header. The guard set covers one representative per path: RM DRAM multi-row (004, 020); RM single-row (001–003, 014, 016); TILE (005, 019); L1 input (006); L1 output (010); fp32 DEST (011); channel-paged history (012); column window (013).
- **Accuracy**: unchanged (no math or dataflow change shipped). Both experiments were bit-identical to the op.
- **Round-2 leads**:
  - **(1)** Combine A and B on multi-row plans with the quartic body. Each optimizer found the other's stage is what remains at the far corner: after split writes, the NoC1 split reads; after dynamic reads, the NoC1 output writes. Each alone takes 004-quartic from ~149 to ~126–130, against ~123 for the library body. Neither helps single-row plans uniformly, so the combination should be designed for the multi-row DM path.
  - **(2)** A topology-agnostic fairness lever for the spatial tail: cross-core tile-row stealing (idea C).
  - **(3)** Once DM stops being the tail, the pack-thread compute (idea D, quartic body on multi-row plans).
  - Note for any round-2 comparison: compare profiled numbers with zones on against zones on.

### Helper bypasses — none
No raw-LLK or helper-bypassing path graduated this round: the shipped change is instrumentation only.
For the record, the two non-graduated experiments went below the helpers in data movement, with no helper covering the protocol:
- A read `cb_output_tiles`' `tiles_received` counter and advanced a private copy of its read pointer. That is a read-only observer of a CB another RISC consumes; no helper offers one, and the single-consumer rule needs it.
- B read the NoC status registers (read responses received, write acks, issue counters) to track the other RISC's reads.
