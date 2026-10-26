# compute_fusion — fusing an expression through DEST vs. round-tripping intermediates through L1

**Problem type:** compute-bound: L1 round-trips  ·  **Concept(s):** compute fusion (keeping intermediates in DEST) vs.
per-op L1 round-trips; secondary: DEST-lane block size.
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · WH B0 · 1000 MHz · 2026-07-10
**Measured on:** WH B0 n150 (sweep 2026-07-10, regimes 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19; whole-kernel sweep and regimes only — the per-phase micro-benchmark is WH-only) · BH p150b (`bh-50-special-adamjanovic-for-reservation-143315`, 1350 MHz, 2026-10-26: the `fpu_chain` scenario and the work-per-call sweep — BH only, not yet re-measured on WH B0)

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
| `fpu_chain` | `(a + b) * c` | `unfused` *(baseline)* | two passes: FPU add → L1 scratch CB, then FPU multiply reads scratch + `c` |
| | | `dstreuse` | one chain: FPU add → DEST, FPU multiply moves DEST into SrcA (`DEST_TO_SRCA`), `c` → SrcB |
| `reduce_recip` | `1 / rowsum(x)` | `unfused` *(baseline)* | SUM reduce→L1, then reciprocal reads it back |
| | | `fused` | SUM reduce with a **post-reduce reciprocal** in DEST |

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.compute_fusion [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--scenario` | `{all,sfpu_chain,fpu_sfpu,fpu_chain,reduce_recip}` | `all` | which expression(s) to run |
| `--tiles` | int... | `4 16 64` | tile counts to sweep (for `reduce_recip`, the reduce width in tiles); with `--per-call`, tiles per call (default `1 2 4 6 8 16 32 64`) |
| `--blocks` | int... | `1 4` | DEST-lane block sizes (eltwise scenarios only) |
| `--trials` | int | `5` | measured trials; report shows median ± std |
| `--kernel-iters` | int | `100` | in-kernel loop count — **1 = per-launch latency; large = steady-state** |
| `--report` | path | *(print only)* | also write the report table to a file |
| `--microbench` | flag | off | per-phase `DeviceZoneScopedN` breakdown (unpack/math/pack ns) instead of the A/B sweep; uses `--tiles[0]` as the size |
| `--per-call` | flag | off | work-per-call sweep: `--total-tiles` per launch split into calls of `--tiles` tiles (scenarios `fpu_chain fpu_sfpu` unless `--scenario` names others) |
| `--total-tiles` | int | `768` | tiles per launch for `--per-call` (the same total work at every tile count) |

```bash
# just the reduce post-op comparison, on your widths
python -m ttnn.operations.examples.compute_fusion --scenario reduce_recip --tiles 8 32 128

# the FPU-vs-SFPU combine bake-off at one size
python -m ttnn.operations.examples.compute_fusion --scenario fpu_sfpu --tiles 64 --blocks 1 4

# where DEST reuse crosses over: same total work, different tiles per call
python -m ttnn.operations.examples.compute_fusion --per-call --scenario fpu_chain --tiles 1 2 3 4 6 --blocks 1 4
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

## Work per call: where DEST reuse crosses over (FPU → FPU)
`fpu_sfpu` loses at every size it was swept at (4–64 tiles), but its producer is an SFPU op. `fpu_chain` asks the
cleaner question: two **two-operand FPU ops** back to back, `(a + b) * c`, done either as one chain that keeps
`a + b` in DEST and moves it into SrcA for the multiply (`dstreuse`), or as two passes through an L1 scratch CB
(`unfused`: the add packs n tiles to the scratch CB, the multiply unpacks them with `c`).

**What "per call" means.** One call is one pass of the expression over `n` tiles — one `eltwise_chain` call for
`dstreuse`, the two chain calls that produce the same result for `unfused` — and every call re-runs the per-op init,
as a separate helper call does. Each launch does the same **768 tiles** of work, as `768 / n` calls of `n` tiles,
so the ns-per-tile figures compare directly across rows. Variants alternate within each case (A, B, A, B …), so a
clock change mid-run would move both sides together.

BH p150b, 1 Tensix core, sharded L1, AICLK 1350 MHz (profiler `CHIP_FREQ`), 5 interleaved trials, median (spread
≤ 0.3 %; a second full run matched every row to within 0.1 %). Block = DEST lanes per chain step; at `n < 4` the
block-4 column runs with block `n`.

| Tiles per call | `fpu_chain` blk 1: unfused → dstreuse (ns per call) | speedup | blk 4: unfused → dstreuse (ns per call) | speedup | `fpu_sfpu` dstreuse, blk 1 / blk 4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 345 → 312 | **1.11×** | (blk 1) | | 1.03× / — |
| 2 | 482 → 486 | 0.99× | 480 → 456 (blk 2) | **1.05×** | 0.98× / 1.01× (blk 2) |
| 4 | 698 → 792 | 0.88× | 729 → 758 | 0.96× | 0.95× / 0.99× |
| 6 | 912 → 1 100 | 0.83× | 896 → 1 007 | 0.89× | 0.94× / 0.97× |
| 8 | 1 129 → 1 406 | 0.80× | 1 087 → 1 268 | 0.86× | 0.94× / 0.97× |
| 16 | 1 982 → 2 621 | 0.76× | 1 777 → 2 256 | 0.79× | 0.93× / 0.96× |
| 32 | 3 708 → 5 068 | 0.73× | 3 173 → 4 246 | 0.75× | 0.93× / 0.95× |
| 64 | 7 140 → 9 959 | 0.72× | 5 937 → 8 217 | 0.72× | 0.93× / 0.95× |

**Crossover: 2–3 tiles per call** (≈ 1.8 at block 1, ≈ 3.3 at block 4 from a straight-line fit). Below it DEST reuse
wins by 1.05–1.11×; from 4 tiles per call up it loses, down to 0.72× at 64 tiles per call. The cost of each side is
close to linear in `n`, so the fit explains the crossover:

| BH p150b, ns | fixed per call | per tile |
|---|---:|---:|
| `unfused`, blk 1 / blk 4 | ~237 / ~382 | ~108 / ~87 |
| `dstreuse`, blk 1 / blk 4 | ~158 / ~260 | ~153 / ~124 |

- **Why DEST reuse wins small calls:** `unfused` makes two chain calls per call — two init sequences, and the
  multiply cannot start until the add has packed its last tile and pushed the scratch CB, so the
  unpack → math → pack pipeline fills and drains twice. That is ~80 ns (blk 1) to ~120 ns (blk 4) more fixed cost
  per call on BH p150b.
- **Why it loses big calls:** for every face of every tile (4 per 32×32 tile) the dest-reuse multiply first moves
  the DEST face into SrcA and clears that DEST face (the FPU multiply accumulates into DEST), then runs the
  multiply. That is ~37–45 ns more per tile than unpacking the scratch tile, and it is paid on every tile. The
  per-phase zones at n = 32, blk 1 agree: the fused chain spends 5 056 ns (158 ns per tile) where the add pass
  takes 1 369 ns and the multiply pass 2 295 ns (115 ns per tile together).
- **Portable rule:** crossover tiles per call ≈ (fixed cost the second pass adds per call) ÷ (DEST→SrcA cost per
  tile). Anything that makes the second pass dearer per call — a data-format reconfig between the passes, a
  scratch CB fed by another RISC-V, a dataflow kernel in the loop — moves the crossover up; this example has none
  of those, so ~2–3 tiles per call is its floor.
- **`fpu_sfpu` (SFPU producer) barely gains even at 1 tile per call** (1.03×, below the 5 % noise band): its sqrt
  pass is ~0.8 µs per tile, so the second pass's fixed cost is a small fraction of the call.

**Precision.** The compute config here has `fp32_dest_acc_en = False`, so DEST holds 16-bit data and `a + b` is
already at bf16 precision when it moves into SrcA. `unfused` rounds the same intermediate to bf16 when it packs the
scratch CB, so the two variants compute the same values: their outputs were **bit-identical** (max abs difference 0)
in every case of the sweep. Both are checked against a torch golden that rounds `a + b` to bf16 too, with
`rtol = atol = 1/64` (one bf16 ulp of the intermediate times `c` ≤ 1.5, plus one ulp of output rounding, with room
for a truncate-vs-round difference — not enough to hide a wrong operand). With `fp32_dest_acc_en = True` the move
would take a 32-bit DEST value into SrcA, which holds fewer mantissa bits than fp32, so the two paths would round
differently; that configuration is not measured here.

```bash
CF_PC_REPORT=/tmp/per_call.md scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/unit_tests/operations/examples/test_compute_fusion.py::test_compute_fusion_per_call
```

## Run the predefined sweep (regenerates `report.md`)
```bash
CF_REPORT=ttnn/ttnn/operations/examples/compute_fusion/report.md \
scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/unit_tests/operations/examples/test_compute_fusion.py::test_compute_fusion_device_perf
```

## Code
All variants are inline kernels in `program_descriptor_with_inline_kernels.py` (one compute kernel
per scenario, method selected by a compile-time arg; phases wrapped in `CF_PHASE` zones under the
`CF_MICROBENCH` define; the FPU → FPU chain is `_FPU_CHAIN_KERNEL`). The sharded-L1 single-core harness, the
correctness gate, the work-per-call sweep (`test_compute_fusion_per_call`) and the CSV-based per-zone reader live in
`../../../../../tests/ttnn/unit_tests/operations/examples/test_compute_fusion.py`. Committed reports:
`report.md` (whole-kernel A/B) and `microbench_report.md` (per-phase zone breakdown).

## When it helps / when it does not
**Rule:** fuse through DEST only when the consumer of the intermediate is an SFPU op; with an FPU consumer keep the
L1 round-trip — unless a call covers only 1–2 tiles, where an FPU → FPU chain through DEST reuse wins (BH p150b;
crossover 2–3 tiles per call). The verdict is the same on WH B0 n150 and BH p150b; the fused gain is largest at few
tiles per call (~1.1× at 4 tiles) and fades to ~1.04–1.05× by 64 tiles on both boards.

| Regime (same definition on every board; 1 Tensix core, sharded L1, kernel-iters 100) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Helps:** every intermediate feeds an SFPU op, small work per call — `sfpu_chain` n = 4 tiles, blk = 4, `unfused → fused` | 9 674 → 8 632 ns **1.12×** | 6 386 → 5 745 ns **1.11×** |
| Same chain, more tiles per call — `sfpu_chain` blk = 4, n = 16 / 64 tiles (device_perf) | 1.06× / 1.05× | 1.05× / 1.04× |
| SFPU reciprocal on a reduced tile — `reduce_recip` n = 4 / 16 / 64 tiles, `unfused → fused` (device_perf) | 1.07× / 1.04× / 1.01× (2026-07-10); n = 64: 1.04× (2026-10-12) | 1.16× / 1.09× / 1.03× |
| **Does not help (loses):** the consumer of the intermediate is an FPU op — `fpu_sfpu` n = 64 tiles, blk = 1, `unfused → dstreuse` | 77 734 → 84 105 ns **0.92×** | 54 023 → 58 328 ns **0.93×** |
| `fpu_sfpu` `dstreuse` across n = 4 / 16 / 64, blk = 1 / 4 (device_perf) | 0.94×–1.02× | 0.93×–0.99× |
| The multiply moved onto the SFPU — `fpu_sfpu` `sfpu` variant (device_perf) | 0.58×–0.65× | 0.62×–0.67× |
| **Helps:** FPU → FPU chain at 1 tile per call — `fpu_chain` n = 1, blk = 1, `unfused → dstreuse` (regimes) | not measured | 349 → 314 ns **1.11×** |
| FPU → FPU chain at 2 tiles per call — `fpu_chain` n = 2, blk = 2 (per_call) | not measured | 480 → 456 ns 1.05× |
| **Does not help (loses):** FPU → FPU chain at 4 tiles per call — `fpu_chain` n = 4, blk = 4 (regimes) | not measured | 731 → 759 ns **0.96×** |
| FPU → FPU chain at 64 tiles per call — `fpu_chain` n = 64, blk = 4 (regimes) | not measured | 5 908 → 8 226 ns **0.72×** |
| **Boundary** | consumer engine: SFPU → fuse, FPU → round-trip | same; for an FPU → FPU chain DEST reuse pays only below ~2–3 tiles per call |

**Why the boards differ:** they do not, in verdict. BH p150b runs every variant faster (e.g. `fpu_sfpu` n = 64
unfused 54 023 vs 77 734 ns) but the fused/unfused ratios stay within ~0.02× of WH B0 n150; the one larger gap,
`reduce_recip` at 4 tiles (1.16× vs 1.07×), is against the older 2026-07-10 WH B0 n150 sweep, which predates the current
kernel library (see `report.md`), so it is not a clean cross-board comparison.

- **Helps** because pack -> L1 -> unpack of an intermediate that an SFPU op consumes is pure overhead, and removing it
  saves a roughly fixed ~1 us per call at 4 tiles on WH B0 n150 (~0.64 µs on BH p150b). The same fusion gives only ~1.05x at 64 tiles (the saving per tile is small, see the table above).
- **Does not help** when the next op is an FPU op: the FPU reads source registers, not DEST, so reusing DEST adds a per-tile DEST-to-source transfer
  that costs more than the pack+unpack it replaces; keep the L1 round-trip and the FPU op reading from L1.
- **Exception — tiny calls:** with only 1–2 tiles per call the second pass's fixed cost (re-init plus a pipeline
  fill and drain) outweighs the per-tile DEST-to-source cost, so an FPU → FPU chain through DEST reuse wins
  (1.05–1.11× on BH p150b); from 4 tiles per call it loses again (see "Work per call" above).

*How measured:* WH B0 n150 rows — `test_compute_fusion_regimes` (1 Tensix core, sharded L1, kernel-iters 100,
DEVICE KERNEL DURATION per expression evaluation, 11 interleaved trials, median; spread 9671..9679 / 8631..8634 and
77708..77746 / 83856..84816 ns), 2026-10-12; sweep rows from the 2026-07-10 `test_compute_fusion_device_perf` table.
BH p150b rows — the same regimes test and device_perf sweep on that board, `bh-qb-15-special-adamjanovic-for-reservation-139410`,
2026-10-19 (details in `report.md`); `fpu_chain` rows — regimes (11 interleaved trials, kernel-iters 100) and the
per-call sweep on `bh-50-special-adamjanovic-for-reservation-143315`, 1350 MHz, 2026-10-26.
Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_compute_fusion.py::test_compute_fusion_regimes`
(same regimes on any board; asserts only what held on every measured board, `measure` regimes only log).
