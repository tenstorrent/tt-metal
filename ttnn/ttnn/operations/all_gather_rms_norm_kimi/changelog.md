# Changelog: all_gather_rms_norm_kimi

## Phase 0 — Core Implementation
- **Date**: 2026-11-17
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - Regime R1: a Fabric ring all-gather that lands remote slices directly in compute-core L1, with fp32 per-row partials sent ahead of the bulk.
  - `S = rstd ⊗ γ` is precomputed before the data lands, then each landed chunk is normalized.
  - Everything runs in one `ttnn.generic_op` (one program per chip).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], alignment=[tile_aligned], rank=[4], gamma_dtype=[bfloat16], gamma_layout=[ROW_MAJOR], memory_layout=[INTERLEAVED], cluster_axis=[1], topology=[Ring], num_links=[2]. This equals TARGET (single-cell suite).
- **Accuracy achieved**: worst device and shape over 4 shapes (`test_all_gather_rms_norm_kimi_precision_baseline.py`):
  - PCC = 0.999998
  - max_abs_err = 0.029
  - mean_abs_err = 0.0013
  - rel_rms_err = 0.0025
  - max ULP = 1.19 (bf16), mean ULP = 0.30
  - median got/true ratio = 0.9989 (tf32 truncation of S; not a scale bug)
- **Golden suite at Phase 0**: 7 / 7 (1 `supported_pass` + 6 regression checks; 0 `supported_fail` / `xpass_drift` / `xfail_wrong_mode`, per `verifier_report.json`)
- **Perf at Phase 0 (harness, PERF FOCUS cell)**: roofline 7.6 µs, target 10.1 µs.
  - Fastest chip: 29.92 µs as handed over, 29.82 µs after the verifier fixes.
  - Slowest chip: 32.71 µs as handed over, 32.29 µs after the verifier fixes.
  - Not dispatch-bound.
  - Per-chip ablation breakdown: skeleton 8.8 / local-only 24.0 / transport-only 26.3 µs (`verification_report.md`).
- **Issues encountered / verifier fixes**:
  - Built the three CB aliases the L1 ledger planned (`cb_gamma`/`cb_out`, `cb_xsq_sum`/`cb_rstd`, `cb_partial`/`cb_rstd_full`); compute-core L1 dropped from 164 to 132 KiB.
  - The writer now flushes each chunk instead of a full barrier per chunk, with one barrier at the end. Measured win: −0.10 µs fastest chip, −0.42 µs slowest.
  - DRY:
    - the compute kernel's CB indices now come from the host;
    - `FP32_TILE_BYTES` / `STATS_ROW_BYTES` are defined once (kernels and host);
    - the γ stick read length comes from the `gamma_page` CT arg.
  - Ledger implementation notes rewritten to match the code.
  - Known infra artifact: running two test modules that each open the fabric mesh in one pytest session can fail the second one at mesh open. Run each file on its own.
- **Tests added**: `test_all_gather_rms_norm_kimi.py` (planner; acceptance), `test_all_gather_rms_norm_kimi_precision_baseline.py` (verifier).
- **Perf tooling added**: `perf/ablation_hooks.patch`, `perf/perf_ablate_nocheck.py.txt`, `perf/ablate_*.json`, `perf/verifier_baseline.json`, `perf/verifier_final.json`.

## Refinement 1 — TRANSPORT: bring the bulk ring exchange to the example's measured transport
- Date: 2026-11-17
- What was done:
  - **Port placement (TRANSPORT)**, taken from `examples/fabric_all_gather` (`_worker_nearest_noc1`, report.md):
    - each port sits at the logical row-0 worker with the fewest NoC1 hops to its link's Ethernet core;
    - the Ethernet core is resolved on the host with no device dispatch. The first fabric connection RT arg is the router's Ethernet channel (a throwaway `ProgramDescriptor` absorbs its semaphores), mapped through the example's `_chip_maps` (cluster + SoC descriptor);
    - the receivers and the hub take the row-0 cores left;
    - the placement is now per chip (`Row0Placement`), so a sender addresses the peer chip's ready port, receiver and hub through the peer's placement;
    - the stats scratch and the semaphores cover all of logical row 0;
    - `PORT_PLACEMENT` knob: `"centred"` reproduces Phase 0 exactly, and it is the automatic fallback when the host lookup fails.
  - **`PORT_BATCH_CHUNKS` 4 → 2** after a co-tune sweep: batch 1/2/3/4 × `INC_EVERY` 2/4/8/16. `INC_EVERY` stays 8.
  - `perf/ablation_hooks.patch` regenerated against the new descriptor. Added `perf/perf_tall.py.txt` (ccl_perf driver for the tall regime shape) and `perf/r1_*.json` (final + every ablation).
- Perf (harness, PERF FOCUS cell, 5 experiments):
  - fastest chip 29.83 → **24.67 µs** (placement −4.05, batch −1.12);
  - slowest chip 32.32 → 27.77 µs;
  - not dispatch-bound.
- Tall `(1,1,2048,384)`: fastest chip 80.4 → 58.0 µs.
- Transport-only (`nocompute,nostore`): 26.32 → 19.57 µs.
  - The bulk now adds 11.4 µs over the skeleton (was 17.5).
  - That is 360 KiB per link direction at 32.4 GB/s, 67% of the 48.5 GB/s link (was 43%).
- Breakdown now (fastest chip, µs):

  | Variant | µs |
  |---|---|
  | skeleton | 8.19 |
  | +store | 11.79 |
  | +compute | 17.92 |
  | local-only (`nobulk`) | 23.71 |
  | transport-only | 19.57 |
  | full | 24.70 |

  **Local-only now binds.** Refinement 3 is the larger remaining term, ahead of Refinement 2's tail.
- Accuracy achieved: PCC = 0.999998, relative RMS ≤ 0.00246 on shapes [(1,1,32,32), (1,1,128,256), (1,1,640,384), (1,1,2048,384)] (unchanged).
- Golden test progress: 7/7 (test_golden 1 + test_regression 6); acceptance 8/8; precision baseline 4/4.
- Issues encountered:
  - The ablation patch no longer applied after the placement change; it is regenerated.
  - Not done, recorded as findings:
    - 7-tile packets: a 12-tile slice block still needs 2 packets, so this needs a `block_h`/`block_w` restructure (Refinement 3's knobs).
    - stats on link 1: Refinement 2's overlap item.
- Tests added: none (the acceptance + determinism tests cover the changed path). Probe `probes/probe_001.py` printed the per-chip Ethernet channels and the nearest workers.

## Refinement 2 — OVERLAP: shrink the exposed tail and the stats-critical chain
- Date: 2026-11-17
- What was done (each lever measured with the harness, 5 experiments, PERF FOCUS cell, fastest / slowest chip µs):
  - Baseline at the start: 24.72 / 27.67 (`perf/r2_baseline.json`).
  - Before choosing levers, a per-stage zone timeline (`perf/zone_probe.patch`, `zone_probe_driver.py.txt`, `zone_analyze.py.txt`; never committed into the kernels) showed the chain:
    - the γ prep (~4 µs) ran before the own x load, so the own partial waited on it;
    - the stats round queued behind the bulk;
    - the hub's 10 KiB multicast + barrier and the stats→tile copy (volatile loads) followed;
    - expand + scale_block (2.4 µs of HiFi4) ran after the stats;
    - then everything was stored at once: the store was DRAM-bound (`nostore` −4.4 µs; output SOL 3.8 µs at 512 GB/s).
  - **Own x before γ** (reader reorder): 24.72 → 21.55 / 23.95.
  - **Stats row ↔ col-0 copies batched** (non-volatile, unrolled, between compiler barriers; `kernels/..._common.hpp`): → 21.06 / 23.24.
  - **γ off the stats chain**:
    - remote γ reads gated on the hub's own-partial count: → 20.62 / 21.90;
    - then **regime R6 built** (`GAMMA_HUB_MCAST`): the hub reads the 48 sticks once per chip and multicasts them into the stats-scratch tail, then `gamma_go`. 20.59 / 22.01: neutral on the headline, but the remote cores reach the stats wait ~5 µs earlier, which the later levers needed.
  - **xg = x ⊙ γ ahead of the stats, out = xg ⊙ rstd per chunk** (replaces scale_block's S):
    - it removes expand + scale from the post-stats chain;
    - with a uniform remote look-ahead it measured 0 / 1 / 2 chunks = 19.66 / 21.03 / 20.84 (a remote chunk planned ahead that lands late holds the 2 µs combine back);
    - kept at 0 → 19.66 / 21.23.
  - **Per-slot stats publish**: the hub multicasts each slot as it arrives, then a multicast set of a cumulative slot mask (no barrier: same NoC, same static VC); readers copy each slot as its bit appears. → 19.34 / 21.06.
    - The first version hung: the mask words sat at a 4 B stride, and the watcher flagged the source / destination 16 B alignment mismatch. Fixed with one 16 B slot per word.
  - **`FUSE_LATE_CHUNKS`** (x ⊙ γ ⊙ rstd_full in one DEST-reuse chain), parked False:
    - its rounding differs from the `cb_xg` path, so it must cover every core of a slice (replication test);
    - fused everywhere measured 19.50 / 21.57 vs 19.32 / 21.12 unfused.
    - rstd is now expanded once to `rstd_full` (neutral).
  - **`STATS_BULK_WINDOW = 2`**: a port that still owes a stats slot keeps ≤ 2 bulk packets queued in its router channel.
    - Windows 2 / 3 / 4: fastest 19.35 / 19.30 / 19.33 (null vs 19.32), slowest 20.61 / 20.83 / 20.98 vs 21.12. Kept at 2 for the slowest chip.
  - **Look-ahead by stream position** (`XG_LATE_STREAM_CHUNKS = 8`): the host packs a per-core table into the compute common args; remote chunks not among the last 8 of their upstream stream compute xg before the combine.
    - Sweep as absolute thresholds 0 / 8 / 12 / 16 / 20 / 24 / 26 / 28 / 32 / all: 19.32 / 19.25 / 19.11 / 19.02 / 19.01 / 18.94 / 20.74 / 20.85 / 20.93 / 20.94.
    - The cliff is where the last ~6 chunks would put the combine on the landing tail, so the knob is expressed as a distance from the stream end.
    - → **18.95 / 20.26** (`perf/r2_final.json`).
  - Not done:
    - `INC_EVERY = 4`: 19.29 / 20.57 vs 19.35 / 20.61, within noise; stays 8.
    - Stats on link 1's forward port: it needs a third hub counter (two senders on `stats_fwd` would break the ordered-sender rule), and the link-1 channel carries the same bulk. The queueing it targets is handled by `STATS_BULK_WINDOW` instead.
    - `out_depth_chunks`: `cb_out` already holds a whole block (2 chunks), so the store of chunk k already overlaps the normalize of chunk k+1.
- Breakdown now (fastest chip, µs; R1 in brackets; `perf/r2_abl_*.json`, hooks regenerated in `perf/ablation_hooks.patch`):

  | Variant | µs (R1) |
  |---|---|
  | skeleton | 6.91 (8.19) |
  | `nobulk,nostore` | 11.01 (17.92) |
  | local-only (`nobulk`) | 16.96 (23.71) |
  | transport-only | 15.71 (19.57) |
  | `nostore` | 17.82 (20.40) |
  | `nocompute` | 18.43 (20.78) |
  | full | 18.96 (24.70) |

  **full − transport-only: 5.13 → 3.25 µs.**
- Tall regime `(1,1,2048,384)`: 57.99 / 60.63 → 55.89 / 58.84 (`perf/r2_tall_*.json`).
- Accuracy achieved: PCC = 0.999998, relative RMS ≤ 0.00241 (max ULP 1.09) on shapes [(1,1,32,32), (1,1,128,256), (1,1,640,384), (1,1,2048,384)].
- Golden test progress: 7/7 (test_golden 1 + test_regression 6); acceptance 8/8 (incl. tall and bit-identical repeat/replication); precision baseline 4/4.
- Issues encountered:
  - A NoC alignment hang (mask-word stride; found with `--dev` watcher, fixed).
  - A replication mismatch with `FUSE_LATE_CHUNKS` on remote cores only (fixed by applying the knob to every core of a slice; parked).
  - Compute-core L1: 132 → 159 KiB (γ / out alias split, scratch tail).
- Remaining headroom (finding):
  - The tail after the last landing is now mostly the DRAM store (`nocompute` within 0.5 µs of full).
  - The stats-critical chain's largest piece is the 1-tile combine: ~2.2 µs of SFPU (4 copies, 3 fp32 adds, rsqrt over the whole tile on the math thread). The `sfpu_tile_scope` item (column-0 faces only) of Refinement 3 is the next lever on it.
- Tests added: none (the acceptance + determinism + precision tests cover the changed paths). Perf tooling: `perf/zone_probe.patch`, `perf/zone_probe_driver.py.txt`, `perf/zone_analyze.py.txt`, `perf/r2_*.json`.

## Refinement 3 — Local compute chain: cut the per-core compute + store time
- Date: 2026-11-17
- What was done (harness, 5 experiments, PERF FOCUS cell, fastest / slowest chip µs; start 18.96 / 20.27, `perf/r3_baseline.json`):
  - Per-stage zones first (the R2 probe plus combine/expand zones). On the stats-critical chain, the combine + rstd expand took 2.2 µs (7.4 → 9.6 µs). A last-chunk probe (`perf/r3_tail_probe.patch`, `perf/r3_tail_analyze.py.txt`) showed the full op ends on a remote core that holds a late block: last landing → compute reaches the last chunk +0.8 µs (still busy with the previous chunk) → gamma pass 0.4 → normalize pass 0.6 → store 0.55.
  - **`COMBINE_COL0`** (`sfpu_tile_scope`, c_skip): one custom chain element (`CombineCol0Rsqrt`, raw sfpi, with the reason in the kernel) does the G-way fp32 sum in the fixed q order, × 1/W, + eps and rsqrt on the even-parity vectors of the two left faces only (8 vector ops instead of 6 × 32). The combine + expand take ~1.0 µs; `nobulk` 17.02 → 15.80. In the full op alone: 19.34 / 20.26. The own cores' stores now overlapped the bulk: `nostore` was neutral, and delaying the own stores by 3000 cycles (diagnostic only) gave 18.91.
  - **`OWN_STORE_GATE`**: the ports increment an own core's `landed` (unused on own cores) once they have issued the reads of its block (1) or of the link's whole own entry (2). The own writer waits for the host-derived count (`own_block_reads`). Gate 1 gave 19.25; gate 2 gave 18.99 / 20.06. **`COMPUTE_STORE_NOC`** knob: stores on NoC0 gave 24.5, so it stays 1.
  - **`NORMALIZE_MODE`** knob: "scale" brings back the R1 S path (S = rstd_full ⊙ γ into `cb_xg`'s allocation, then x ⊙ S as chunks land; one pass per chunk). It measured 18.78 / 20.55 twice. The slowest chip is worse because early-landed remote cores then store in the middle of the bulk (12 tiles taking up to 8 µs from the right-hand columns). A global wall-clock store delay (diagnostic) did not fix it. Parked at "xg".
  - **`LATE_BLOCK`** (`compute_block_size`): when a block has no chunk ahead, compute runs one gamma pass (x waited cumulatively, xg pushed per chunk) and one normalize pass (xg waited cumulatively, out pushed per chunk) over the whole block, instead of a gamma + normalize pass per chunk. The per-tile arithmetic is identical, so the replication test stays bit-exact. → **18.72 / 19.76** (`perf/r3_final.json`; `compare`: fastest −0.245, slowest −0.505, noise 0.05: win).
  - Re-sweeps with everything in: `XG_LATE_STREAM_CHUNKS` 4 / 8 / 12 = 19.59 / 18.72 / 18.75 (stays 8). Gate 0 = 19.04 / 19.98. Col-0 off = 18.85 / 19.98. `FUSE_LATE_CHUNKS` + col-0 = 19.46 (stays off).
  - Not done: the **block knobs** (`block_w` < `Wl_t` on the idle cores, or `block_h = 2`). The per-core compute is no longer on the critical path: `nobulk,nostore` is 9.69 against a 6.92 skeleton. A `block_w` split also needs a same-slice partial combine with a bit-exact fixed order, which is a scheme change for < 1 µs of headroom.
- Breakdown now (fastest chip, µs; R2 in brackets; `perf/r3_abl_*.json`, hooks regenerated in `perf/ablation_hooks.patch`, which now also carries the nocompute hunk):

  | Variant | µs (R2) |
  |---|---|
  | skeleton | 6.92 (6.91) |
  | `nobulk,nostore` | 9.69 (11.01) |
  | local-only (`nobulk`) | 15.74 (16.96) |
  | transport-only | 15.85 (15.71) |
  | `nostore` | 17.73 (17.82) |
  | `nocompute` | 17.76 (18.43) |
  | full | 18.72 (18.96) |

- Tall regime `(1,1,2048,384)`: 55.89 / 58.84 → 51.00 / 53.65 (`perf/r3_tall_final.json`).
- Accuracy achieved: PCC = 0.999998, relative RMS ≤ 0.00241 (max ULP 1.09) on shapes [(1,1,32,32), (1,1,128,256), (1,1,640,384), (1,1,2048,384)].
- Golden test progress: 7/7 (test_golden 1 + test_regression 6); unit dir 12/12 (acceptance 8 incl. tall and bit-identical repeat/replication; precision baseline 4).
- Issues encountered:
  - The col-0 combine alone regressed the full op by moving the own stores into the bulk; the store gate fixed that.
  - Scale mode trades the fastest chip against the slowest through the same store contention.
- Remaining headroom (finding): the op is transport (15.85) plus a 2.9 µs tail. The tail is ~1 µs of last-chunk compute plus DRAM stores slowed by NoC1 contention with the fabric traffic; some cores take 8 µs to store 12 tiles mid-bulk. In local-only, the end-of-op store burst is +6 µs against a ~3.8 µs DRAM SOL. Next: route the stores away from the fabric's NoC1 paths (per-core NoC choice or compute placement by column), and time the stores of the early-landed remote cores against the stream position.
- Tests added: none (acceptance + determinism + precision cover the changed paths). Perf tooling: `perf/r3_*.json`, `perf/r3_tail_probe.patch`, `perf/r3_tail_analyze.py.txt`, regenerated `perf/ablation_hooks.patch`.

## Perf 1 — tournament round 1 (2-experiment cap)
- Date: 2026-11-18
- Focus: the one perf case of the suite (per-chip input `(1,1,640,384)` bf16, γ bf16 ROW_MAJOR, `fp32_dest_acc_en=True`, HiFi4, G = 4 ring, 2 links). Every axis it declares is in `SUPPORTED`. Roofline 7.6 µs (links), target 10.1 µs.
- Start (harness, 5 experiments): fastest chip **18.72 µs**, slowest 19.77 µs, not dispatch-bound (`perf/p1_baseline.json`).

### Instrumentation (permanent)
- Every kernel now carries `MaybeDeviceZoneScope` zones at its stage boundaries. The macro comes from `kernels/all_gather_rms_norm_kimi_perf.hpp`, an alias of `DeviceZoneScopedN`; `kernel_lib/perf_instrumentation.hpp` does not exist in this tree.
- Waits are zoned apart from the work they gate, and no zone sits inside a per-chunk loop:
  - **sender:** `sender_open`, `sender_fence`, `sender_bulk`, `sender_tail_stats`
  - **port reader:** `preader_own_wait`
  - **receiver:** `recv_notify`, `recv_drain`
  - **hub:** `hub_gamma`, `hub_own_wait`, `hub_stats`
  - **reader:** `reader_own_load`, `reader_gamma_wait`, `reader_stats`, `reader_landed_wait`
  - **writer:** `writer_partial_wait`, `writer_export`, `writer_gate_wait`, `writer_store`
  - **compute:** `compute_partial`, `compute_gamma_wait`, `compute_xg_ahead`, `compute_stats_wait`, `compute_combine`, `compute_normalize`
- **The zones compile in only with `AGRN_PERF_ZONES=1`** (the host adds the define and includes the flag in the program hash). With the zones on, the ccl_perf harness (500 traced calls with the profiler on) dropped programs: it saw `programs per chip {3: 579, ...}, expected 600`, because the zone markers overflow the profiler buffer. So the harness build leaves them out.
- With the zones compiled out, the harness is unchanged: 18.713 / 19.793 µs, `compare` null.
- Tools:
  - `perf/p1_zone_driver.py.txt`: run as `AGRN_PERF_ZONES=1 TT_METAL_DEVICE_PROFILER=1 scripts/tt-probe.sh all_gather_rms_norm_kimi < it`.
  - `perf/p1_zone_analyze.py.txt` and `perf/p1_tail_analyze.py.txt`: the analyzers.
  - `perf/ablation_hooks.patch`: regenerated against the zoned kernels.

### Measured breakdown (fastest chip, µs)
Cumulative payload ablation (`perf/p1_abl_*.json`); it reproduces Refinement 3 to within 0.02 µs:

| Variant | µs |
|---|---|
| full | 18.72 |
| `nostore` | 17.73 |
| `nocompute` | 17.77 |
| transport only (`nocompute,nostore`) | 15.86 |
| local only (`nobulk`) | 15.74 |
| `nobulk,nostore` | 9.69 |
| skeleton (`nobulk,nocompute,nostore`) | 6.92 |

The overlap loss is full − slowest part alone = 18.72 − 15.86 = **2.86 µs**.

Zone timeline. All times are µs from the chip's first kernel start, medians over calls, for chips 0 / 1 / 2 / 3:

| Stage | Measured | Part SOL | Gap |
|---|---|---|---|
| Sender setup before `conn.open` | 1.26 (+0.28 open) | ≈ 0.6 (one route + open) | the fence ends at 2.07 / 3.06 / 3.43 / 2.99, while the own data is ready at ~2.2 |
| `sender_bulk`: 360 KiB per port | 11.8 – 12.8, i.e. 63% of the 48.5 GB/s link | 7.6 at the link rate; the `fabric_all_gather` example reaches 74% at 2 links, which would be 10.3 | **2.2 – 5** |
| Last bulk packet handed to the router → last landing | 1.2 – 2.2 | the router channel drains at the link rate | — |
| Last landing → last normalize | **1.7** (a gamma pass plus a normalize pass over the late block) | ~0.4 (one 6-tile pass) | 1.3 |
| Last normalize → last store | 0.65 | ~0.5 (DRAM write latency) | — |
| Stores competing with fabric traffic | nocompute − transport only = 1.9 µs, of which ~0.65 is the inherent last store | — | ~1.2 |

Ranked headroom:
1. **TRANSPORT bulk stream rate.** Largest gap, but no concrete lever was cheap enough under the cap.
2. **OVERLAP post-exchange chain** (1.3).
3. **Store contention** (~1.2).
4. **TRANSPORT pre-bulk handshake** (up to ~1 on 3 of 4 chips).

The stats chain (the hub has every slot at 7.3 – 8.2 µs) and the compute before the landing are not on the critical path.

### Portfolio and verdicts (cap: 2 experiments)
| Idea | Kind | Verdict | Focus: fastest / slowest µs, base → candidate | Domain |
|---|---|---|---|---|
| `fence_first_sender` (`perf_experiments/fence_first_sender/`) | TRANSPORT, handshake | **NULL** | 18.708 / 19.795 → 18.673 / 19.796 (r1); 18.720 / 19.800 → 18.680 / 19.785 (r2). Fastest −0.035 / −0.040 against 0.050 noise | It would apply everywhere (header setup only). Tall shape: fastest −0.20, slowest +0.18; it shifts time between chips |
| `late_chunk_one_pass` (`perf_experiments/late_chunk_one_pass/`) | OVERLAP, post-exchange chain | **REGRESSION** | 18.71 / 19.82 → impl1 (SFPU chain) 21.13 / 22.31; impl2 (FPU DEST reuse) 19.25 / 20.33; impl3 (raw SFPU) 19.78 / 20.80 | Measured regression everywhere. Tall shape: +3.5 to +4.9 µs |

**fence_first_sender**
- What it found: almost all of the 1.07 µs before `conn.open` is three 2D `fabric_set_unicast_route` calls at ~0.35 µs each. A software copy of a routed header costs as much as routing it.
- The best variant, `ff2`:
  - `open_start`, then route only the control header, then `open_finish`;
  - send the ready increment;
  - fill the bulk headers by NoC reads of the routed control header while waiting on the fence.
- Effect: the pre-fence window drops from 1.55 to 0.88 µs, and the fence ends 0.3 – 0.9 µs earlier. But the bulk still ends at 14.5 – 14.9 µs, because the bulk's rate (link or port-reader supply), not its start, sets the end.
- Correct: golden check passed and repeated calls were identical. Not graduated: no measured win.

**late_chunk_one_pass**
- Idea: one DEST window per late chunk (x·γ on the FPU, then · rstd in DEST). Every chip must keep producing bit-identical output, so the look-ahead path has to use the same arithmetic.
- That costs more than it saves:
  - reloading fp32 xg into DEST exactly adds ~1.0 µs per 12 tiles;
  - the SFPU fp32 multiply adds ~1.3 µs per 12 tiles (~145 cycles per tile);
  - last landing → last normalize grows from 1.7 to 3.0 µs.
- All variants were correct and bit-identical across chips.
- Precision: impl1 and impl3 are more precise (relative RMS 0.0017 vs 0.0020). impl2 is less precise (0.0039, max ULP 2.5) because the DEST → srcA move keeps only part of xg's mantissa.
- None is graduated. Recorded as precision options with their cost.

Not tested (cap): 14,336 B full-payload packets (7 tiles; this needs packets that cross block boundaries, i.e. scatter writes, or a re-block), and routing the compute cores' stores away from the fabric's NoC1 paths.

### Graduated
- Nothing. The op's arithmetic and dataflow are unchanged; the only kernel change this round is the zones, which compile out by default.
- Whole op before / after: 18.72 / 19.77 → 18.71 / 19.79 µs (null).
- Guard set: the op has one kernel path × layout × placement. The golden suite passes, 7 / 7.

### Helper bypasses — none
Nothing graduated, so the op gained no raw-LLK path.
- The rejected `late_chunk_one_pass` impl3 bypassed the eltwise_chain helpers (`BinaryFpu` + `CopyTile` + `MulBinary`). The chain gives every DEST lane its own copy of the shared rstd_full operand, so it fits 2 tiles per window and re-reads rstd_full for every tile. No helper keeps one operand in a single DEST slot shared across a window.
- That bypass did not win, so it is not a graduated row. It is noted here as feedback to the helper library.

### Next round
- The bulk rate is now the clearly binding term: `ff2` showed that starting the bulk earlier does not end it earlier.
- Diagnose the sender's stall split (waiting on a router slot vs waiting on CB data) before choosing a lever:
  - full-payload packets if the sender is link-bound;
  - port-reader supply (batch depth, relay readiness) if it is data-starved.
- `ff2` costs nothing and should be re-measured together with any bulk-rate win.
