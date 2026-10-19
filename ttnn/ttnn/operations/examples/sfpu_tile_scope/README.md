# SFPU tile scope — apply rsqrt/reciprocal to only the vectors of a tile that matter

**Problem type:** compute-bound: wasted SFPU work  ·  **Concept:** SFPU work-scoping
**First profiled on:** `bh-50-special` · BH · 1350 MHz · 2026-07-22 · `59de133afc4`
**Measured on:** WH B0 n150 (regimes, 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
An SFPU (vector-unit) unary op — `rsqrt`, `reciprocal`, … — runs over a 32×32 tile as a sequence of
**32-lane vector ops**. Each vector op covers **4 rows × 8 columns taken with stride 2** (the even
columns `0,2,…,14` *or* the odd `1,3,…,15` — column parity is one address bit). A whole tile is
**32 vector ops** (4 faces × 4 row-groups × 2 parities). But after a reduction the value you need lives
on **one axis**: a per-row result in **column 0**, a per-column result in **row 0**, a scalar at **[0,0]**.
The rest of every vector is computing lanes you'll never read. Scoping to just the vectors that matter is
the same math on the data you keep, for a fraction of the SFPU cycles.

## What this isolates — and how
- **Concept:** *SFPU work-scoping* — running only the vector ops that cover the meaningful axis.
- **Isolation:** the SFPU is the **only thing on the clock**. One core, input sharded in L1. The tile is
  copied into DEST **once** and packed out **once** — both *outside* a `DeviceZoneScopedN` — and inside the
  zone the scoped op runs `reps` times on the **MATH thread only**. The test reads **TRISC_1 (math)**, so the
  number is pure SFPU cycles: no unpack, no pack, no CB handshake. Zone unpack/pack come back **≈0 ns** (proof
  of isolation); `none` (empty loop) reports the loop's own overhead (≈0). **The cost is ~flat per vector op**
  (~24 ns rsqrt, ~28 ns recip), so the whole story reduces to *how many vector ops the scope runs*.
- **Kernel-level:** the vectors run are chosen by a `VectorMode`, an `ITERATIONS` count, and (for the column
  trick) a hand-written DEST address stride — all in the kernel author's hands.

## The methods being compared
`func` = `rsqrt`|`recip` is a *parameter*; each scope is measured for both.

| Variant | how it selects vectors | vec ops | valid region | note |
|---|---|---:|---|---|
| `none` | empty loop, no SFPU | 0 | (identity copy) | math-loop overhead floor (≈0) |
| `rc` *(baseline)* | `VectorMode::RC` — 4 faces | **32** | whole tile | the default you get |
| `r` | `VectorMode::R` — top 2 faces | 16 | rows 0–15 | a top-**half** result |
| `c` | `VectorMode::C` — left 2 faces | 16 | cols 0–15 | a left-**half** result |
| **`r_iter2`** | `VectorMode::R` + **`ITERATIONS=2`** | **4** | **row 0** | a **row-0** result — *iterations only* |
| **`c_skip`** | `VectorMode::C` + **even-parity stride** (`dst_reg+=2`) | **8** | **col 0** | a **col-0** result — *raw sfpi* |
| `face` | `VectorMode::None` — face 0 | 8 | [0:16, 0:16] | a 16×16 / `[0,0]` result |
| `face_iter1` | `VectorMode::None` + `ITERATIONS=1` | 1 | [0, 0] | a `[0,0]` scalar |

### The two axis-optimal tricks — and why one is a knob and the other isn't
The SFPU walks a face as `[rg0-even, rg0-odd, rg1-even, rg1-odd, …]` — **parity is the inner axis,
row-group the outer**. That single fact decides the tool:

- **Row-0 result → `ITERATIONS` (a knob).** The waste is the *rows* (you need rg0, not rg0–3), which is the
  **outer** walk axis — `ITERATIONS` truncates it. `VectorMode::R` + `ITERATIONS=2` keeps the top row-group
  (rows 0–3) of *both* top faces = **4 vectors** (row 0 needs one row-group, both parities, both faces). One
  line, no raw code.
- **Col-0 result → an address stride (raw sfpi).** The waste is column **parity** (column 0 is even; the odd
  vectors touch only columns `1,3,…,15`), which is the **inner** walk axis — `ITERATIONS` can't isolate it
  because it truncates *contiguously*. You skip the odd vectors with a custom loop that strides the DEST
  address by 2 (`dst_reg += 2`), keeping the even vectors of the two left faces = **8 vectors** (col 0 needs
  every row-group but one parity).

That is why **row-0 lands at 4 vectors but col-0 at 8**, and why the row trick is a config knob while the
column trick drops into hand-addressed sfpi — the "sometimes easier, sometimes harder."

## CLI — measure your own params
```bash
python -m ttnn.operations.examples.sfpu_tile_scope [--variant …] [--func …] [--reps N] [--trials N] [--report PATH]
# e.g. the two axis tricks vs their half-tile modes:
python -m ttnn.operations.examples.sfpu_tile_scope --variant rc r r_iter2 c c_skip
```
| Flag | Default | Meaning |
|---|---|---|
| `--variant {none,rc,r,c,r_iter2,c_skip,face,face_iter1}` | `all` | which scope(s) |
| `--func {rsqrt,recip}` | `all` | which SFPU op(s) |
| `--reps N` | `2000` | in-kernel math-loop trip count (amortizes the one zone marker) |
| `--trials N` | `5` | measured launches; median ± std |

## Measured result
*Illustrative — see the **First profiled on** stamp; re-run the CLI for your box.*

```
sfpu_tile_scope  box=bh-50-special  arch=BH  clock=1350MHz  cores=1  sharded-L1  N=5 (median)  reps=2000
  MATH-thread (TRISC_1) ns per SFPU call — copy+pack OUTSIDE the timed zone; cost is ~flat per vector op
  func=rsqrt                                                 vec  ns/call   vs rc    ns/vec
    rc     VectorMode::RC                     whole tile      32   748.2    1.00×    23.4
    r  /c  VectorMode::R / ::C                half tile       16   378.5    1.98×    23.7
    r_iter2  VectorMode::R + ITERATIONS=2     ROW 0            4   103.1    7.26×    25.8   ← iterations only
    c_skip   VectorMode::C + even-parity stride  COL 0         8   194.9    3.84×    24.4   ← 1.94× vs c (same body)
    face   VectorMode::None                  face 0           8   188.9    3.96×    23.6
    face_iter1  None + ITERATIONS=1          [0,0]            1    28.3   26.46×    28.3
  func=recip:  rc 890 → r/c 450 (1.98×) → r_iter2 121 (7.37×) → face 224 (3.97×) → face_iter1 33 (27.2×)
```

**Reading of the result:** the `ns/vector` column is flat, so every speedup is just a vector-count ratio.
The two axis tricks are the payoff: **`r_iter2` cuts a row-0 result to 4 vectors — 7.3× vs the whole tile, and
3.7× vs the coarse `VectorMode::R`** — with nothing but `ITERATIONS=2`. **`c_skip` cuts a col-0 result to 8
vectors — 3.84× vs the whole tile, a clean 1.94× over `VectorMode::C`** (identical rsqrt body, so the win is
purely the skipped odd-parity vectors). `r_iter2` (4) beats `c_skip` (8) because a row collapses to one
row-group while a column still spans all 32 rows.

**Caveat on `recip` `c_skip`:** reciprocal's fast path uses `SFPLOADMACRO` hardware addressing that *can't* be
strided, so the column skip forces a hand-written Newton reciprocal body instead. That body is also cheaper
per vector (ns/vector ≈11 vs stock recip's ≈28), so its headline speedup (~10×) **conflates fewer vectors with
a different algorithm** — it is not a clean apples-to-apples skip. The honest pure-skip number is `rsqrt`'s
**1.94× over `c`** (same body, half the vectors). `r_iter2` is clean for both ops (it's the stock body via the
wrapper).

*Board scope (added 2026-10-19):* the recip numbers above are bh-50 on `59de133afc4`. On BH p150b with the current
kernel library stock recip costs **6.5 ns/vector** (rc 206.8 ns) and the Newton `c_skip` body 11.0 ns/vector, so recip
`c_skip` is 2.34× vs `rc` — the Newton body is now the *more* expensive one per vector. The conclusion stands (only
`rsqrt`'s `c_skip` is a clean skip number); the "cheaper per vector" direction does not. rsqrt reproduces within 7 %
(24.9 vs 23.4 ns/vector).

**What this is and isn't:** the SFPU cost **in isolation**. In a full op the surrounding copy/pack/DRAM dilute
the whole-kernel win, and on a data-movement-bound op the SFPU isn't the bottleneck at all. Reach for scoping
when the SFPU *is* the bottleneck and your data sits on one axis.

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/examples/test_sfpu_tile_scope.py
```

## Code
`program_descriptor_with_inline_kernels.py` — one compute kernel: seed DEST once, a `DeviceZoneScopedN` around a
math-only `reps`-loop of the scoped op, pack once. `rsqrt_scoped`/`recip_scoped` thread `VectorMode`+`ITERATIONS`
through the stock call (covers `rc`/`r`/`c`/`r_iter2`/`face`/`face_iter1`); `cskip_*_body` are the raw-sfpi
even-parity stride loops for `c_skip`. The test reads the zone's TRISC_1 (math) duration from the profiler CSV.

## When it helps / when it does not
**Rule:** the SFPU cost is flat per 32-lane vector op, so a scope pays exactly by how many vector ops it removes
(32 → 4 for a row-0 result with `r_iter2`), on both boards; swapping between scopes with the same vector count gains
nothing. In a whole kernel the win is diluted by the non-SFPU floor (~690–740 ns per launch on WH B0 n150,
~440–460 ns on BH p150b) and tends to 1.0× as that floor dominates.

| Regime (same definition on every board; 1 Tensix core, bf16 tile in sharded L1) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Helps:** row-0 result, SFPU time isolated — rsqrt, math zone, `rc` (32 vectors) → `r_iter2` (4 vectors) | 1 139.1 → 156.1 ns **7.30×** | 795.6 → 109.0 ns **7.30×** |
| **Helps:** same, recip, math zone | 563.1 → 84.2 ns **6.69×** | 206.8 → 37.9 ns **5.46×** |
| **Helps but diluted:** whole kernel, 1 tile, 1 call — rsqrt, whole op | 1 642 → 739 ns **2.22×** | 1 042 → 459 ns **2.27×** |
| **Helps but diluted:** same, recip, whole op | 1 106 → 688 ns **1.61×** | 606 → 438 ns **1.38×** |
| **Does not help:** change of scope with the same vector count — rsqrt, math zone, `c` (16) → `r` (16) | 576.1 → 576.1 ns **1.00×** | 402.2 → 402.2 ns **1.00×** |
| **Does not help:** same, recip, math zone | 288.1 → 288.1 ns **1.00×** | 109.0 → 109.0 ns **1.00×** |
| Cost per vector op (`rc` / 32) | rsqrt 35.6 ns, recip 17.6 ns | rsqrt 24.9 ns, recip 6.5 ns |
| **Boundary** | vector count only; whole-op speedup = (floor + SFPU_rc) / (floor + SFPU_r_iter2) | same |

**Why the boards differ:**
- rsqrt costs about the same number of cycles per vector op on both (35.6 ns at 1000 MHz ≈ 36 cycles vs 24.9 ns at
  1350 MHz ≈ 34 cycles), so its ratios match exactly (7.30×).
- recip is 2.7× cheaper per vector on BH p150b (6.5 vs 17.6 ns ≈ 9 vs 18 cycles) — more than the AICLK ratio, so
  the Blackhole recip path is cheaper per vector (inferred from the cycle count; the per-arch implementation was not
  inspected). With so little work per vector, the fixed per-call cost is a larger share of the 4-vector `r_iter2`
  call (9.5 ns/vector vs 6.5 for `rc`), so recip's gain drops to 5.46× in the math zone and 1.38× whole-op.

- **Helps:** cost is flat per 32-lane vector op on both boards (WH B0 n150: rsqrt 35.6 ns, recip 17.6 ns per vector; BH p150b: 24.9 / 6.5 ns), so a scope that runs 4 vectors instead of 32 is 5.5-7.3x cheaper in the SFPU; a result that lives on row 0 never needs the other 28.
- **Does not help:** only the vector count matters, not which half of the tile: `VectorMode::R` and `VectorMode::C` both run 16 vectors and cost the same to the nanosecond, so swapping one for the other (or using either when a row-0 / col-0 scope is possible) gains nothing; the gain comes from reducing vectors (`r_iter2` 4, `c_skip` 8, `face_iter1` 1).
- **Dilution (whole op):** the win shrinks as the non-SFPU part of the op grows: with a single SFPU call per tile it is 2.22x (rsqrt) / 1.61x (recip) on WH B0 n150 and 2.27x / 1.38x on BH p150b instead of the math-zone ratios, because the ~740-690 ns (WH B0 n150) / ~460-440 ns (BH p150b) of unpack/copy/pack/handshake and launch floor does not shrink. The example kernel handles only one tile per launch, so a regime where the win vanishes entirely (data-movement-bound op) cannot be expressed with the existing API; per the formula, whole-op speedup = (floor + SFPU_rc) / (floor + SFPU_r_iter2), so it approaches 1.0x as floor >> SFPU time.
- **WH B0 vs Blackhole:** rsqrt per-vector cost on WH B0 n150 is ~1.4-1.5x the Blackhole numbers (35.6 ns vs 24.9 ns on BH p150b, 23.4 ns on bh-50). For recip the older bh-50 run (28 ns/vector, commit `59de133afc4`) made WH B0 look cheaper (~0.6x); with the current library BH p150b is 6.5 ns/vector, so WH B0 n150 recip is ~2.7x the BH p150b cost. Vector-count ratios (r_iter2 7.3x for rsqrt, `r`/`c` 2.0x) reproduce on all three.

*How measured:* WH B0 n150 rows — `test_sfpu_tile_scope_regimes` (1 Tensix core, 11 interleaved trials per variant,
median; `math zone` = MATH-thread (TRISC_1) ns per SFPU call from the `DeviceZoneScopedN` loop, `reps=2000`;
`whole op` = `DEVICE KERNEL DURATION [ns]` of a launch with `reps=1`, i.e. one SFPU call on one tile with
unpack/copy/pack/CB handshake; speedup = baseline_ns / lever_ns; min-max in `report.md`), 2026-10-12. BH p150b rows —
the same regimes test on that board, `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19 (details in `report.md`).
Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_sfpu_tile_scope.py::test_sfpu_tile_scope_regimes`
(same regimes on any board; asserts only what held on every measured board, `measure` regimes only log).
