# tapdirect_unit_cost (Perf 18): the per-unit / per-K-block cost of the K-blocked tap-direct walk

Verdict: **NULL** (one variant is a **REGRESSION** on P1/P4). The "everything stubbed" floor exists only in the
stubbed configurations. In the real op it is off the critical path: removing the reader's per-unit planning, the
compute's per-unit table handshake, the per-K-block DEST-remap drain or the whole-band wait moves no FLUX.2 case beyond
noise. Making compute faster also slows the DRAM-coupled P1 and P4. No integration patch is recommended.
Blackhole p150b, 110 cores, bf16 / HiFi4 / fp32 DEST, all variants under the same compute config.

## Layout
- `op_base/`: the working-tree op at HEAD d4414e00bd0, plus uncommitted user changes and the Perf 18 ablation hooks
  (`ablation_r18/apply_hooks.py`).
- `op_copy/`: the same, plus experiment knobs. Descriptor `TU_DEFS` = comma list of tokens, each becomes the define
  `TU_<tok>` on every kernel. All knobs are off by default, and then `copy` is bit-identical to `base` at the same time.
  The changes are in `variants.patch`.
  - `NOPLAN` (diagnostic, wrong output): `TapDirectFifo::unit` writes the zero block for every tap, with no planning.
  - `GRP`: `fifo_tilize` waits for and pops the table once per group of 4 entries (the reader's group), reading the
    entries by pointer with one `invalidate_l1_cache` per group. Bit-identical.
  - `NOINIT` (diagnostic, valid only with the tilize stubbed): no per-K-block reconfig, `fast_tilize_init` or uninit.
  - `SKIPREMAP`: `fast_tilize_init_skip_remap` after the first K-block. `fast_tilize_init` re-runs
    `_llk_math_reconfig_remap_(true)` (`tensix_sync` plus a wait for every outstanding pack) on every K-block, and
    nothing in the kernel turns the remap off. Bit-identical.
  - `FINENEED`: the ring's whole-band wait at kw = 0 K-blocks is replaced by a wait per unit group, covering only the
    pixels that group reads (`RmBankRing::need_to`). Chunks are still issued up to the band end. Bit-identical.
  - `PREWAIT` + `Z` / `ZI` (diagnostic zones): `tu_tblwait` (wait for the whole K-block's entries), `tu_units`,
    `tu_init`, `tu_mminit`, `tu_mmwait`.
- `test_tu.py`, `run.sh <label> <cases> <variants>` (variant `c.TOK.TOK`, suffix `@fill+weights+...` = ablation).
  `TU_ZONES=1` writes `runs/<label>_zones.txt`. `TU_SEQ=zone,...` writes one core's per-execution timeline
  (`zone_seq.py`). `safe_run.sh` copies the device log while the run still holds the device lock.

## Measurements (device kernel ns, one call per variant; the medians of the case's signposted window)
Everything stubbed (`fill,weights,store,tilize,bias,matmul`), d1:
| case | base | NOPLAN | GRP | NOINIT | NOPLAN+GRP+NOINIT |
|---|---|---|---|---|---|
| P4 (36) | 107804 | 86043 | 107839 | 107141 | 66944 |
| P2 (42) | 127801 | 110956 | 127395 | 127738 | 77773 |

The reader paces first, at about 20 cycles per unit. With the reader removed, the compute handshake and inits pace.

P1 matmul-only (`fill,weights,store,tilize,bias`), c3: base 1401128, NOPLAN 1381447, GRP 1295810 (-7.5%),
NOINIT+GRP 1270953. Here the floor is the compute's per-unit handshake.

Real op (all bit-identical except NOPLAN):
| case | base | NOPLAN | GRP | SKIPREMAP | FINENEED |
|---|---|---|---|---|---|
| P1 (41) | 1830967 / 1824181 | 1813022 (wrong output) | 2062987 / 2088524 (+12.7% / +14.5%) | 1831930 | - |
| P2 (42) | 1374744 / 1372599 | 1374739 | 1375447 | 1373984 | 1372525 |
| P3 (35) | 2397273 | - | - | 2407693 | 2397134 |
| P4 (36) | 597871 / 595669 / 602627 | 596388 | 614956 / 608266 | 597786 | 594384 |
| P6 (46) | 5315784 | - | 5322219 | 5325929 | - |
| P7 (43) | 1307970 / 1310196 | 1309222 | 1310259 | 1310341 | 1308826 |
| P8 (37) | 447641 | - | - | 448807 | 448239 |
| P9 (33) | 795391 | - | - | 796066 | 799264 |
| P12 (45) | 2796945 | - | 2797601 | 2799067 | - |
| P13 (47) | 5389271 | - | 5396748 | 5397279 | - |

## Why the floor does not show in the real op
- Reader planning: the reader runs a whole K-block ahead. The P8 zone timeline (`runs/z7_seq.txt`, core (6,6))
  shows compute waiting for the table at the first K-block only (25.5 us, the initial band fill, which is
  `reader_band_wait` 23.6 us). After that the wait is 0.03 us per K-block. NOPLAN, the upper bound of any planning
  cut, is -1% on P1 and flat elsewhere.
- Compute handshake: in the real op it sits under the BH fast-tilize hardware time. Steady state on P8 is 1.86 us per
  K-block for 19 two-tile units, about 132 cycles per unit, and the BH unpack issues 32 UNPACRs per unit whatever the
  width. GRP is flat on compute-bound cases. On P1 (+13-15%, confirmed in reversed order) and P4 (+2-3%) it is slower,
  consistent with the brief's solo tilize stub (P1 +180 us): faster compute on a DRAM-coupled case moves the
  last-K-block store burst onto the ring fill.
- Per-K-block DEST-remap drain: SKIPREMAP is flat on 10 cases, including P13 with 1341 K-blocks per core. The math
  thread waits for a free DEST half anyway.
- The whole tilize payload is a small upper bound: the brief's solo tilize stub saves at most 3-4% on compute-bound
  cases (P2 -55 us, P7 -47, P6 -200, P8 -13, P10 -14). The large costs on these cases are the first-band startup fill
  (P8 25 us per core) and the last K-block's store stall (P8 core (6,6): 77.7 us and 37.9 us against a 17 us K-block).
