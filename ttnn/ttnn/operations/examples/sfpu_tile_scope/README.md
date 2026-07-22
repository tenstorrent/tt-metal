# SFPU tile scope — apply rsqrt/reciprocal to only the vectors of a tile that matter

**Difficulty:** ⭐⭐ T2 (the `r_iter2` knob) / ⭐⭐⭐ T3 (the `c_skip` sfpi stride)  ·  **Concept:** SFPU work-scoping
**First profiled on:** `bh-50-special` · BH · 1350 MHz · 2026-07-22 · `59de133afc4`

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

**What this is and isn't:** the SFPU cost **in isolation**. In a full op the surrounding copy/pack/DRAM dilute
the whole-kernel win, and on a data-movement-bound op the SFPU isn't the bottleneck at all. Reach for scoping
when the SFPU *is* the bottleneck and your data sits on one axis.

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh ttnn/ttnn/operations/examples/sfpu_tile_scope/test_sfpu_tile_scope.py
```

## Code
`program_descriptor_with_inline_kernels.py` — one compute kernel: seed DEST once, a `DeviceZoneScopedN` around a
math-only `reps`-loop of the scoped op, pack once. `rsqrt_scoped`/`recip_scoped` thread `VectorMode`+`ITERATIONS`
through the stock call (covers `rc`/`r`/`c`/`r_iter2`/`face`/`face_iter1`); `cskip_*_body` are the raw-sfpi
even-parity stride loops for `c_skip`. The test reads the zone's TRISC_1 (math) duration from the profiler CSV.
