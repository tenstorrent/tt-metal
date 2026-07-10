# compute_fusion — fusing an expression through DEST vs. round-tripping intermediates through L1

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** compute fusion (keeping intermediates in DEST) vs.
per-op L1 round-trips; secondary: DEST-lane block size.
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · WH B0 · 1000 MHz · 2026-07-10

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
You have a small elementwise expression — say `exp(sqrt(x) + y)`, or `sqrt(x) * b`, or `1 / rowsum(x)`.
The easy way to build it is to call one helper per operation: each helper does its op, packs the
result to a circular buffer in L1, and the next helper reads it back. It is correct and readable,
but every intermediate makes a pack→L1→unpack round-trip. You *can* instead fuse the whole thing
into one compute pass that keeps intermediates in the DEST registers and packs only the final
result. This example measures when that fusion actually pays — and shows one case where it does
**not**.

## What this isolates — and how
- **Concept:** the cost of the intermediate **L1 round-trip** (pack + CB handshake + unpack, plus a
  little extra per-op init/reconfig) that fusing an expression removes — and, as a secondary axis,
  the DEST-lane **block size**.
- **Isolation setup (pure compute):** everything lives in **sharded L1 on one Tensix core**. There
  is no DRAM movement — inputs are resident before the kernel runs, and the kernel loops
  `--kernel-iters` times over them for a steady-state number. So the measured delta is purely
  on-core compute: the intermediate traffic between L1 and the math engine, nothing else.
- **Why it's kernel-level:** whether to fuse a chain or split it into separate helper calls — and
  whether to feed the next op from DEST or from L1 — is a decision the kernel author makes. It is
  not a model/dtype choice.

Each scenario runs the *same math* two (or three) ways, so a "win" means identical work, fewer ns.

## The scenarios and methods
Every scenario's baseline is its `unfused` variant (separate helpers, L1 round-trips).

| Scenario | Expression | Variant | What it does |
|---|---|---|---|
| `sfpu_chain` | `exp(sqrt(x) + y)` | `unfused` *(baseline)* | sqrt→L1, (·+y)→L1, exp→out (two round-trips) |
| | | `fused` | one chain: sqrt, SFPU add, exp — all in DEST |
| `fpu_sfpu` | `sqrt(x) * b` | `unfused` *(baseline)* | sqrt→L1, then FPU multiply reads it back |
| | | `dstreuse` | one chain: sqrt in DEST, FPU multiply **reuses DEST** as an operand |
| | | `sfpu` | one chain: sqrt in DEST, copy `b` to a 2nd DEST slot, **SFPU** multiply |
| `reduce_recip` | `1 / rowsum(x)` | `unfused` *(baseline)* | SUM reduce→L1, then reciprocal reads it back |
| | | `fused` | SUM reduce with a **post-reduce reciprocal** in DEST |

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.compute_fusion [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--scenario` | `{all,sfpu_chain,fpu_sfpu,reduce_recip}` | `all` | which expression(s) to run |
| `--tiles` | int... | `4 16 64` | tile counts to sweep (for `reduce_recip`, the reduce width in tiles) |
| `--blocks` | int... | `1 4` | DEST-lane block sizes (eltwise scenarios only) |
| `--trials` | int | `5` | measured trials; report shows median ± std |
| `--kernel-iters` | int | `100` | in-kernel loop count — **1 = per-launch latency; large = steady-state** |
| `--report` | path | *(print only)* | also write the report table to a file |
| `--microbench` | flag | off | per-phase `DeviceZoneScopedN` breakdown (unpack/math/pack ns) instead of the A/B sweep; uses `--tiles[0]` as the size |

```bash
# just the reduce post-op comparison, on your widths
python -m ttnn.operations.examples.compute_fusion --scenario reduce_recip --tiles 8 32 128

# the FPU-vs-SFPU combine bake-off at one size
python -m ttnn.operations.examples.compute_fusion --scenario fpu_sfpu --tiles 64 --blocks 1 4
```

## Measured result
*Illustrative — see the **First profiled on** stamp; re-run the CLI for your box. Full sweep in [`report.md`](report.md).*

```
compute_fusion  box=bgd-lab-...-40918  arch=WH_B0  cores=1  placement=single-core sharded-L1  N=5 (median)  kernel-iters=100
  sfpu_chain    n=64  blk=4  unfused    127219 ns ±0.0%  ✓
  sfpu_chain    n=64  blk=4  fused      121621 ns ±0.0%  ✓  → 1.05×
  reduce_recip  n=4         unfused       2736 ns ±0.1%  ✓
  reduce_recip  n=4         fused         2555 ns ±0.1%  ✓  → 1.07×
  fpu_sfpu      n=64  blk=1  unfused      67395 ns ±0.0%  ✓            (baseline)
  fpu_sfpu      n=64  blk=1  dstreuse     72049 ns ±0.0%  ✓  → 0.94×   (fusion LOSES)
  fpu_sfpu      n=64  blk=1  sfpu        115617 ns ±0.0%  ✓  → 0.58×   (SFPU multiply: big loss)
```

**Reading of the result — one rule explains all three scenarios:** fusion helps only when the
**next op reads DEST natively**, i.e. when the consumer is an **SFPU** op.

- **SFPU consumer → fuse.** In `sfpu_chain` every intermediate feeds an SFPU op (sqrt→add→exp), and
  in `reduce_recip` the reduced tile feeds an SFPU reciprocal. SFPU reads and writes DEST directly,
  so keeping the intermediate in DEST removes a pure-overhead round-trip: **1.03–1.12×** and
  **1.01–1.07×**. The reduce win shrinks as the width grows because the single reciprocal is a fixed
  cost amortized over more reduce work.
- **FPU consumer → the round-trip can be faster.** In `fpu_sfpu` the intermediate `sqrt(x)` feeds an
  **FPU** multiply. The FPU takes its operands from source registers (filled by the unpacker), *not*
  from DEST — so "reusing DEST" forces a DEST→source transfer. That transfer costs **more** than the
  pack+unpack it was meant to save: `dstreuse` runs at **0.94–1.02×** of the baseline. Isolating just
  the combine step (replace sqrt with a copy, so the only difference is FPU-from-DEST vs
  FPU-from-L1) makes it stark: **0.82×** — the L1 round-trip is 1.22× *faster* than dest-reuse.
- **Don't use the SFPU for what the FPU does.** The `sfpu` variant computes the multiply on the SFPU
  (vector engine) instead of the FPU: **0.58×**. A plain multiply belongs on the FPU.
- **Block size (DEST-lane batching) is a small win.** `blk=4` vs `blk=1` on the fused eltwise chains
  is ~1–3% faster — the per-op init is already hoisted to once per chain call regardless of block, so
  the block only amortizes loop/packer overhead across a few tiles. Real, but second-order next to
  the fuse/round-trip decision.

The general point: **dest-reuse is not a free "skip L1" win.** Its payoff comes from replacing a
slower engine or from accumulating many ops into DEST — not from avoiding L1 per se. For a single
FPU combine, reading the operand back from L1 is the faster path.

## Micro-benchmark: per-phase zone breakdown
The whole-kernel A/B above tells you *which* variant wins. To see *why* — where each phase spends
its time and which engine bounds it — build with the `CF_MICROBENCH` define (the `--microbench`
flag), which wraps every phase in a `DeviceZoneScopedN` zone. A compute-kernel zone records on all
three TRISCs, so per phase you get `unpack` / `math` / `pack` ns; the phase **wall** is the slowest
of the three (they pipeline).

```bash
python -m ttnn.operations.examples.compute_fusion --microbench --scenario all --tiles 32
```

Illustrative (WH B0, 1 core, n=32, 16 in-kernel iters — full table in [`microbench_report.md`](microbench_report.md)):

```
sfpu_chain  unfused  CF_SQRT  unpack 26852  math 30625  pack 30442  wall 30625   (math-bound: SFPU sqrt)
sfpu_chain  unfused  CF_ADD   unpack 13313  math 10146  pack 10646  wall 13313   (UNPACK-bound: reads s1 back from L1)
sfpu_chain  unfused  CF_EXP   unpack 20725  math 22991  pack 22661  wall 22991   (math-bound: SFPU exp)
fpu_sfpu    dstreuse CF_FUSED unpack 34998  math 36160  pack 36174  wall 36174
fpu_sfpu    sfpu     CF_FUSED unpack 55387  math 57954  pack 57976  wall 57976   (SFPU mul: +22k math vs FPU)
fpu_sfpu    unfused  CF_MUL   unpack  6608  math  3196  pack  4054  wall  6608   (UNPACK-bound: cheap FPU mul, L1 reads dominate)
reduce_recip unfused CF_RECIP unpack   723  math  2781  pack   291  wall  2781   (a whole extra SFPU pass the post-op folds away)
```

What the zones reveal (the mechanism behind the headline numbers):
- **The L1 round-trip shows up as `unpack` cost.** `sfpu_chain`'s add phase and `fpu_sfpu`'s mul phase
  are both **unpack-bound** even though their math (SFPU add / FPU mul) is cheap — the time is spent
  unpacking the round-tripped intermediate back out of L1.
- **SFPU multiply is ~22k ns more `math` than FPU multiply** (`sfpu` 57954 vs `dstreuse` 36160) —
  why the SFPU-combine variant loses badly.
- **Dest-reuse moves the combine cost onto the `math` engine.** `dstreuse` math (36160) is higher than
  the unfused sqrt+mul math (30663 + 3196 = 33859): the DEST→source path the FPU needs is ~2.3k ns of
  extra math — it trades the unfused path's L1 unpack for math-engine work, which is why the two come
  out close.
- **The reduce post-op folds away an entire pass.** Unfused adds a standalone `CF_RECIP` phase
  (2781 ns, SFPU-recip, math-bound); the fused post-op runs the reciprocal inside the reduce's DEST
  pass at ~no extra math.

Caveat: `Σ wall` is a **lower bound** on a variant's serial cost — it omits the inter-phase
CB-handshake gaps and the tiny zone instrumentation overhead, so it does not reproduce the
whole-kernel ratios exactly. Use `report.md` for the authoritative A/B; the zones are for attribution.

### Two follow-up experiments (what the zones led us to check)
Running the microbench at more than one tile count, plus one format ablation, pins down *why* the
numbers move — and corrects a plausible-but-wrong hypothesis.

**1. The dest-reuse penalty is a per-tile datapath cost, not a one-time reconfig.** Sweeping the
`fpu_sfpu` microbench (n=8 vs n=32), the extra math the dest-reuse FPU pays over the unfused
sqrt+mul math grows with the tile count:

| n | `dstreuse` math ns | unfused sqrt+mul math ns | dest-reuse premium |
|---:|---:|---:|---:|
| 8 | 9208 | 7822 + 922 = 8744 | **464** |
| 32 | 36160 | 30663 + 3196 = 33859 | **2301** |

The premium scales ~5× for a 4× tile increase — it is paid **per tile**, so it is the DEST→source
transfer the FPU performs on every tile, not a fixed init/reconfig. That is the mechanism behind
dest-reuse ≈ (or losing to) the L1 round-trip for an FPU consumer.

**2. A narrower intermediate does NOT dodge the round-trip — it is unpack-engine-bound, not
bandwidth-bound.** The round-trip surfaces as unpack cost, so the natural idea is "pack the
intermediate as bfp8 (half the bytes) to make the read-back cheaper." Measured on `sfpu_chain`
unfused (n=64), packing the two intermediates as `bfloat8_b` instead of `bfloat16`:

```
bf16 intermediate   127104 ns/eval   PCC 0.99996
bfp8 intermediate   127629 ns/eval   PCC 0.99983   → 1.004×  (no improvement)
```

Halving the intermediate's byte width changes nothing: the unpacker pays ~one op **per tile**
regardless of the tile's format, so the round-trip is bounded by unpack-engine occupancy, not L1
bandwidth. The takeaway sharpens: you cannot shrink a round-trip away with a narrower format — the
only way to remove it is to not round-trip (fuse, when the consumer is SFPU).

## Run the predefined sweep (regenerates `report.md`)
```bash
CF_REPORT=ttnn/ttnn/operations/examples/compute_fusion/report.md \
scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/unit_tests/operations/examples/test_compute_fusion.py::test_compute_fusion_device_perf
```

## Code
All variants are inline kernels in `program_descriptor_with_inline_kernels.py` (one compute kernel
per scenario, method selected by a compile-time arg; phases wrapped in `CF_PHASE` zones under the
`CF_MICROBENCH` define). The sharded-L1 single-core harness, the correctness gate, and the
CSV-based per-zone reader live in
`../../../../../tests/ttnn/unit_tests/operations/examples/test_compute_fusion.py`. Committed reports:
`report.md` (whole-kernel A/B) and `microbench_report.md` (per-phase zone breakdown).
