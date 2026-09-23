# tilize_pack_throughput — cut the tilize compute stage at bit-exact precision (Perf 2)

WH B0 n150, AICLK 1000 MHz (cycles = ns). Metric: DEVICE KERNEL DURATION [ns], one fresh-cache run
per (case, variant), several sessions for the close calls. A/A noise (`ctl` = the head code
compiled from a variant dir) is ±3–5 %.
Every correct variant is golden-checked (`helpers.check_output`) AND bit-compared with
`torch.equal` against head's readback of the same seeded input.

## Verdict: WIN — drop the redundant data-format reconfig (helper `NoReconfigure`)

`compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles)` already programs srcA, srcB and the
packer for exactly these CBs. The helper's `UnpackAndPackReconfigure` then re-issues identical
config: `reconfig_data_format_srca/srcb` + `pack_reconfig_data_format`, each with STALLWAITs and
config-register writes. This sits on the critical path in two places:
- the unpack reconfig delays the start of the unpack -> math -> pack chain;
- the pack reconfig (~180 cycles) is exposed whenever the pack thread's wait for math is short.

The fix is two enum values in `kernels/tilize_compute.cpp`, with no raw LLK and no helper change:
`graduate/tilize_compute.cpp`, diff in `graduate/tilize_compute.diff`.

Final A/B, 3 sessions, median ns (head / ctl / norc):

| case | shape / placement | head | ctl | norc | norc vs head |
|---|---|---|---|---|---|
| 9 | [1,1,2048,512] HS L1 -> HS L1 (64 cores x 16 tiles) | 1929 | 1935 | **1874** | -2.9 % |
| 10 | [1,1,512,512] BS COL L1 -> same (16 cores x 4x4) | 1964 | 1968 | **1809** | -7.9 % (ref 1832) |
| b9 | case 9, bf16 -> bfloat8_b | 1559 | 1539 | 1523 | -2.3 % |
| b10 | case 10, bf16 -> bfloat8_b | 1503 | 1515 | 1438 | -4.3 % |
| x2 | [1,1,2048,64] HS resident (1 x 2 tiles) | 1101 | 1105 | **930** | -15.5 % |
| x6 | [1,1,2048,96] HS resident (1 x 3 tiles) | 1134 | 1118 | **1004** | -11.5 % |
| x3 | [1,1,8192,160] HS resident (4 x 5 tiles) | 2183 | 2185 | 2104 | -3.6 % |
| 7 | [1,1,2048,512] HS L1 -> DRAM | 16921 | 16975 | 16589 | flat |
| 0 | [1,1,16384,64] DRAM (focus) | 23736 | 23941 | 23754 | flat |
| 6 | [1,1,8192,32] fp32 -> fp32 DRAM (Lossless) | 14076 | 14864 | 14079 | flat |
| b0 | focus, bf16 -> bfloat8_b | 19303 | 18978 | 18939 | flat |
| 3 | [1,1,128,64] DRAM | 2307 | 2417 | 2393 | flat (5-run medians: 2321 vs 2280) |
| 1 | [1,1,16384,32] DRAM (block_width 1) | 13247 | 13376 | 13360 | flat |
| 8 | [1,1,2048,512] DRAM -> HS L1 | 12098 | 11966 | 12070 | flat |
| x5 | [1,1,8192,256] DRAM | 43289 | 45208 | 43718 | flat |

Other regimes (one session, head -> norc): resident fp32->fp32 3673 -> 3636, u8 1767 -> 1588,
u16 2064 -> 1932, fp32->bf16 3479 -> 3513, bf16->bfloat4_b 1507 -> 1494, [1,1,2048,1024]
(1 x 32 tiles) 2926 -> 2894, [1,1,4096,512] (2 x 16) 2913 -> 2839. With `dst_full_sync_en`
forced: case 9 2582 -> 2371, case 10 2532 -> 2332.

Zones (TT_METAL_KERNEL_PERF_ZONES=1, per-core `compute_tilize`, p50 cycles, head -> norc):
- case 9: TRISC_0 501 -> 464, TRISC_2 1418 -> 1376.
- x2: TRISC_0 304 -> 248, TRISC_1 178 -> 132, TRISC_2 584 -> 400.

Correctness: bit-identical to head (torch.equal) and golden-passing on every cell run:
- 231 dtype cells (7 placements x 11 (in, out) pairs: bf16 -> bf16/bf8/bf4/f32, f32 -> f32/bf16/bf8, u8, u16, u32, i32, i32 -> u32);
- `dst_full_sync_en` and `fp32_dest_acc_en` forced on (120 cells);
- tiny tiles, retile 32 -> 16, `low_l1`, padded, wide DRAM (33 cells).

The golden suite re-run against `kernels_graduate` gives the same result as head: 760 passed / 3 failed.
The 3 failures are input-dependent bfp8 rank-0 / bfp4 pad-cell tolerance flakes, which also flip
between head reruns.

## Breakdown that drove it (ablations, payload stubbed, sync kept; cases 9 / 10)

| variant | 9 | 10 |
|---|---|---|
| compute stubbed (p2_breakdown/kernels_C) | 612–626 | 676–685 |
| init + reconfig + uninit + DEST handoffs, no payload (`ab_f_none`) | 1007–1013 | 1084–1119 |
| same, no reconfig / uninit (`ab_f_none_lean`) | 699 | 788 |
| sync only, no init (`ab_f_none_noinit`) | 634 | 733 |
| unpack + math only (`ab_f_nopack`) | 1253 | 1279 |
| pack only (`ab_f_packonly`) | 1769–1801 | 1843 |
| std path: pack only / unpack + math only / none | 1712 / 1622 / 908 | 1727 / 1675 / 993 |
| full (raw s8 = head schedule) | 1889–1897 | 1893–1895 |

What this breakdown shows:
- **Pack is the long pole.** The fast-tilize pack costs ~48 ns per tile (~790 ns for 16 tiles); unpack + math cost ~240 ns.
- **The standard pack is no faster.** It packs 1 PACR per tile against fast tilize's 16 row-PACRs, yet also costs ~48 ns per tile, so the pack is bound by packer bandwidth, not by the instruction form. No pack instruction stream beats it at a fixed dtype.
- **The fixed cost is reconfig + uninit (~315 ns) in the zero-payload floor.** With payload, only the reconfig part is exposed; skipping the uninit is flat (see below).

## Menu (all at the user's config: fp32_dest_acc_en / math_fidelity / dst_full_sync_en / dtypes unchanged)

Case 9 / case 10, same-session vs head, in ns. "Exact" means torch.equal against head plus the golden check.

| option | 9 | 10 | precision | notes |
|---|---|---|---|---|
| head (UnpackAndPackReconfigure, InitAndUninit) | 1929 | 1964 | exact | baseline |
| **norc**: helper `NoReconfigure` | **1874** | **1809** | exact | RECOMMENDED; helper-expressible |
| nou: helper `InitOnly` (skip kernel-end uninit) | 1896 | 1920 | exact in-op | flat (vs ctl 1902/1936). **Unsafe across programs**: math fast-tilize init leaves `CFG_STATE_ID=1` and `ALU_ACC_CTRL_Fp32_enabled` cleared, and trisc FW never resets them (`reset_cfg_state_id` only zeroes the SW variable). Not graduatable. |
| norc_nou | 1902 / 1880 | 1823–1867 | exact in-op | = norc within noise; same hazard |
| raw fast tilize, DEST section 6 / 4 / 2 tiles (s6/s4/s2) | 1958 / 2036 / 2151 | 1989 / 1952 / 2116 | exact | REGRESSION: every extra math->pack handoff drains the packer + ZEROACC; costs more than the earlier pack start saves |
| s4 / s2 + no reconfig / uninit (s4lean / s2lean) | 1862 / 2004 | 1840 / 1910 | exact | no better than norc |
| raw pack section done without ZEROACC (s4nz / s8nz) | — | — | **INCORRECT** (PCC 0.88–0.97) | the DEST clear is load-bearing for fast tilize |
| raw DEST sections spanning tile-row blocks (x8, bw <= 4) | 1921 | 1971 | exact | NULL on case 10 |
| x8 + one CB wait/reserve/push/pop per section (x8g) | 1881 | 1874 | exact | NULL vs norc |
| x8g + lean (x8glean) | 1864 | 1855 | exact | = norc on resident; **measured-regression** on [1,1,16384,32] DRAM bw=1: 13188–13505 -> 19081–19630 (+45 %), because an open DEST section waits on the streaming reader |
| raw batched standard tilize, 8 / 4 tiles per section (std8 / std4), bf16 | 2106 / 1963 | 1978 / 2005 | exact | REGRESSION vs fast tilize (the standard unpack is ~44 ns per tile) |
| mode 5 (b2nd): norc + 2-tile DEST sections on the helper's *slow* path only (u8/u16/fp32-out/full-sync; 32-bit unpack-to-DEST left per-tile) | — | — | exact | mixed, vs norc: bf16->fp32 2800 -> 2613 (-7 %), u16 4x5 2226 -> 2038 (-8 %), u16 1x16 1934 -> 1872, but u8 1578 -> 1621 (+3 %). With 4 / 8-tile sections u8 is +7 % and unpack-to-DEST fp32 is +4 % (3635 -> 3794). Not a unified path. |

## Domain of the recommended option (norc)

Applies everywhere. It is correct on every regime run, since it is the same config the kernel already had,
applied once instead of twice. It wins on compute-bound resident shapes (bigger wins with fewer tiles
per core) and is flat on every DRAM-bound shape. Exceptions: none measured.

Untested: Blackhole, where the helper's fast-tilize reconfig touches only srcA; the same reasoning
applies but it was not measured.

## Helper notes (no bypass in the recommendation)

- norc uses the helper as-is.
- The raw-LLK variants (modes 1/2/3/5) bypassed `compute_kernel_lib::tilize` because its WH fast
  path fixes the DEST-section schedule at "fill the DEST half". It cannot express a smaller
  math->pack handoff, a section spanning tile-row blocks, or a batched slow path. That is a
  *capability* gap. None of these won, so no helper change is proposed.
- Ergonomics gap: `UnpackAndPackReconfigure` reads like the safe default, but right after
  `compute_kernel_hw_startup` on the same CBs it is pure overhead (-3 % to -16 % here).

## Files

- `tilize_compute_pt.cpp`: hand-authored variant compute kernel (all modes and knobs; the header comment documents them).
- `make_variants.py`: regenerates the git-ignored `kernels_<name>/` dirs, including `kernels_graduate`.
- `graduate/tilize_compute.cpp` + `graduate/tilize_compute.diff`: the change that would graduate.
- `label.py`: pairs profiler rows with the harness' `P2` lines, reading the report path from the log.
- Harness: `tests/ttnn/unit_tests/operations/tilize/test_tilize_perf2_tilize_pack_throughput.py`
  (`TPT_CASES`: LOOSE index, `bN` = bfp8 output, `xN` = extra scenario, `d<tag>:<in>:<out>` = dtype override;
  `TPT_VARIANTS`, `TPT_CHECK=0` for ablations, `TPT_FULL_SYNC=1` / `TPT_FP32_DEST=1` to force the user's knobs).
  `..._golden.py` re-runs the golden suite on a variant (`TPT_GOLDEN_VARIANT`).

## Repro

```bash
python3 ttnn/ttnn/operations/tilize/perf_experiments/tilize_pack_throughput/make_variants.py
P=tilize_pack_throughput
TILIZE_PERF_EXPERIMENTS=1 TPT_CASES=9,10,7,0,6,b9,b0,x2 \
  TPT_VARIANTS=head,$P/kernels_ctl,$P/kernels_norc \
  scripts/run_safe_pytest.sh --profile --run-all \
  tests/ttnn/unit_tests/operations/tilize/test_tilize_perf2_tilize_pack_throughput.py -s > log
python3 ttnn/ttnn/operations/tilize/perf_experiments/tilize_pack_throughput/label.py log
# golden on the candidate:
TILIZE_PERF_EXPERIMENTS=1 TPT_GOLDEN_VARIANT=$P/kernels_graduate scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/unit_tests/operations/tilize/test_tilize_perf2_tilize_pack_throughput_golden.py
```

Keep `low_l1` cases (x9) out of profiled sessions: the A/B runs the op twice, which shifts the labels.
