<!--
  perf-lab example README.
  Rules: no references to production ops / other examples; kernel-level concept only; correctness is
  the only pass/fail; every number is arch/box stamped; cores/placement/method front and center.
-->

# matmul_output_subblock — SRC-register argument reuse via the output-subblock shape

**Problem type:** compute-bound: SRC operand reload  ·  **Concept(s):** matmul output-subblock shape → SRC-register operand reuse
**First profiled on:** `bh-qb-11-special-dnijemcevic-for-reservation-42432` · BH · Blackhole · 2026-07-13
**Measured on:** WH B0 n150 (regimes + Kt sweep, 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
A blocked tiled matmul `C = A @ B` is computed one **output subblock** at a time — each subblock is one
block-matmul on the Matrix Unit. How big (and what shape) you make that subblock is a compile-time knob,
and it decides how often the operand tiles are re-fetched into the SRC registers. Written the naive way
— a `1×1` subblock, one output tile per block-matmul — every output tile re-loads *both* its A operand
and its B operand into SRC. Is grouping tiles into a bigger subblock worth it, and does the shape matter?

## What this isolates — and how
- **Concept:** the output-subblock shape `sb_h × sb_w` fed to the `matmul_block` helper, and the
  **operand reuse** it buys inside each block-matmul call:
  - a **wide** subblock (`sb_w > sb_h`, e.g. `1×8`) loads one A row-tile into SRC **once** and multiplies
    it against all `sb_w` B column-tiles — A is reused `sb_w×`;
  - a **tall** subblock (`sb_h > sb_w`, e.g. `8×1`) loads one B column-tile once and reuses it across all
    `sb_h` A row-tiles;
  - `1×1` reuses nothing — both operands are re-loaded per output tile.
- **Isolation setup:** *pure compute* row of the rule. A, B and C are all sharded in L1 on a **single
  Tensix core** — no DRAM in the fast path — so the measured delta is the operand-reuse (and per-subblock
  init/handshake) that a bigger subblock amortizes. The contraction is a **single K-tile** (`Kt=1`), which
  keeps the win entirely in how each subblock is produced (not K accumulation) and frees the full 8-tile
  DEST for wide/tall subblocks. Output is packed row-major (`OutputCBLayout::TileRowMajor`) so every
  variant produces the identical, correct C.
- **Why it's kernel-level:** the subblock shape and the `matmul_block` call are the compute-kernel
  author's choice, not a model or input decision.

## The methods being compared
| Variant | subblock | reuses | Why it should differ |
|---|---|---|---|
| `sb_1x1` *(baseline)* | 1×1 | nothing | both operands re-loaded into SRC per output tile |
| `sb_1x8` | 1×8 | A across 8 B-tiles | one A-load feeds 8 outputs |
| `sb_8x1` | 8×1 | B across 8 A-rows | one B-load feeds 8 outputs |
| `sb_2x4` / `sb_4x2` | 2×4 / 4×2 | A / B | 8-tile subblock, balanced-ish shape |
| `sb_2x2` | 2×2 | A + B | 4-tile subblock (half the DEST) |

All variants call the same `matmul_block` helper (`TileRowMajor`, single K-block); only the
`MatmulBlockShape` subblock changes. `sb_h·sb_w ≤ 8` (the fp16 DEST budget).

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.matmul_output_subblock [--mt 16] [--nt 16] [--iters 100] [--trials 10]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--mt` / `--nt` | int | 16 | output size in tiles (M, N) |
| `--iters` | int | 100 | in-kernel repeat (steady-state) |
| `--trials` | int | 10 | profiled rounds; report shows median |

## Measured result
*Illustrative — see the **First profiled on** stamp; re-run the CLI for your box.* M=N=16 tiles, Kt=1.

```
matmul_output_subblock  box=bh-qb-11…  arch=Blackhole  cores=1  single-core sharded-L1  M=16t N=16t K=1t
  variant   sb(hxw)  reuses        ns/op     vs 1x1
  sb_1x1    1x1      none        1039955      1.00x
  sb_1x8    1x8      A            710027      1.46x
  sb_8x1    8x1      B            710255      1.46x
  sb_2x4    2x4      A            712876      1.46x
  sb_4x2    4x2      B            713212      1.46x
  sb_2x2    2x2      A+B          741198      1.40x
```

**Reading of the result:** grouping output tiles into a bigger subblock is worth **~1.46×** here, and the
win tracks the subblock **size**, not its shape — every 8-tile subblock (`1×8`, `8×1`, `2×4`, `4×2`) lands
at 1.46×, and wide (reuse A) vs tall (reuse B) are symmetric because A and B enter the matmul the same
way. The 4-tile `2×2` gets 1.40× — a little less, because fewer tiles per block means the one-time SRC
operand load is amortized over fewer outputs. The ceiling is the DEST budget: a subblock must fit DEST
(8 fp16 tiles here). Note the tradeoff for a *real* multi-K matmul: `Kt>1` needs a 32-bit DEST
(`fp32_dest_acc_en`) for correct accumulation, which halves the budget to 4 tiles — so you would cap the
subblock at 4 (e.g. `2×2`, `1×4`) and take the ~1.40× rather than the full 1.46×. This lever matters most
for short-K matmuls (small contraction depth, K≈2–4) where the contraction can't hide the per-tile
operand-load overhead.

*Board scope (added 2026-10-19):* the size-not-shape reading holds on Blackhole (bh-qb-11 above and BH p150b) but
not on WH B0 n150, where tall `8×1` is the weakest 8-tile subblock; and the "short-K" remark does not match either
board as measured since — on WH B0 n150 the gain stays ~1.7× from Kt = 1 to Kt = 12, on BH p150b it is largest at
Kt = 1 and falls to 1.10× by Kt = 4. See *When it helps* below.

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_matmul_output_subblock.py::test_matmul_output_subblock_device_perf
```
Correctness (every subblock shape computes the identical C):
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_matmul_output_subblock.py::test_matmul_output_subblock_correctness
```

## When it helps / when it does not
**Rule:** an 8-tile output subblock instead of `1×1` pays when the matmul has many output tiles (≥ 16 output
tiles on WH B0 n150, ≥ 64 on BH p150b) — and on BH p150b essentially only at a single K-tile: WH B0 n150 keeps
~1.7× from Kt = 1 to Kt = 12, BH p150b falls from 1.46× (Kt = 1) to 1.17× (Kt = 2) and 1.10× (Kt = 4).
Subblock orientation (tall `8×1` vs wide `1×8`) matters on WH B0 n150 only.

| Regime (same definition on every board; 1 Tensix core, sharded L1, bf16) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Helps:** many output tiles, one K-tile — `sb_1x1 → sb_2x4`, 16×16 output tiles, Kt = 1 | 24 312 → 13 905 ns **1.75×** | 10 429 → 7 161 ns **1.46×** |
| **Helps (shape):** same 8-tile subblock, tall `sb_8x1` → wide `sb_1x8`, 16×16, Kt = 1 | 19 167 → 13 662 ns **1.40×** | 7 129 → 7 128 ns **1.00× — flat on BH (tall == wide)** |
| **Helps (deeper K):** `sb_1x1 → sb_2x4`, 16×16, Kt = 4 | **1.69×** (Kt sweep; `sb_1x1` 7 018 784 ns per 100 iterations) | 29 238 → 26 654 ns **1.10× — small on BH** |
| K sweep, 16×16 output tiles, best 8-tile subblock vs `sb_1x1` | Kt 1 / 2 / 4 / 8 / 12: 1.77× / 1.66× / 1.69× / 1.68× / 1.69× | Kt 1 / 2 / 4: 1.46× / 1.17× / 1.10×; Kt ≥ 8 does not fit L1 at 16×16 |
| Output-size sweep, Kt = 1, `sb_1x1` → largest legal subblock | 4×4 (16 output tiles): 1.27× (`2×4`); 2×4: 1.06×; 1×8: 1.07× | 8×8 (64 output tiles): 1.34×; 4×4: 1.08× |
| **Helps on WH only:** few output tiles but deep K — 2×2 output tiles, Kt = 8, `sb_1x1 → sb_2x2` | **1.34×** | **0.98× — flat on BH** |
| **Does not help:** only 4 output tiles in the whole matmul — 2×2, Kt = 1, `sb_1x1 → sb_2x2` | 670 → 656 ns **1.02×** | 372 → 382 ns **0.97×** |
| **Boundary** | ≥ 16 output tiles (4×4); independent of Kt up to 12 | ≥ 64 output tiles (8×8) and Kt = 1; at Kt ≥ 2 expect 1.1–1.2× |

**Agent rule:** on WH B0 n150 use a wide `1×8` or `2×4`/`4×2` subblock, never tall `8×1`, at any K. On BH p150b
any 8-tile subblock is equivalent; take it, but expect ~1.1–1.2× (not 1.7×) once Kt ≥ 2.

**Why the boards differ:**
- BH p150b runs the `sb_1x1` baseline 2.3× faster than WH B0 n150 at 16×16, Kt = 1 (10 429 vs 24 312 ns), more
  than its 1.35× AICLK ratio (1350 vs 1000 MHz) — the per-output-tile SRC operand reload is relatively cheaper on
  BH. What a bigger subblock still saves there is mostly the per-subblock fixed cost (init + DEST handshake per
  `matmul_block` call), which K inner steps amortize anyway — so the gain falls with Kt on BH and not on WH.
  (Mechanism inferred from the K and output-size sweeps; not isolated with a per-zone profile.)
- Tall `8×1` and wide `1×8` measure within 0.1 % of each other on both Blackhole boards (bh-qb-11: 710 255 vs
  710 027 ns per 100 iterations; BH p150b: 7 129 vs 7 128 ns), so the WH B0 n150 orientation penalty is a WH
  effect, not a property of the reuse scheme.

- **Helps** because a bigger subblock holds one operand in the SRC registers across the subblock (A across `sb_w` B-tiles for wide) instead of reloading both per output tile.
- **Does not help** when there is almost nothing to share: with 4 output tiles per matmul the per-matmul cost is
  ~670 ns on WH B0 n150 (~375 ns on BH p150b) either way (the operand-load saving is smaller than the fixed
  start-up cost). On WH B0 n150 only, deep K rescues small outputs (2×2 at Kt = 8: 1.34×; 4×4 at Kt = 1 is already
  1.26–1.27×); on BH p150b neither does (0.98× and 1.08×).

WH B0 n150 only (`test_matmul_output_subblock_device_perf`, 16×16 output tiles) — neither point holds on BH p150b:
- **Wide and tall are not symmetric on Wormhole B0.** At Kt=1: `sb_1x8` 1.77x, `sb_2x4` 1.74x, `sb_4x2` 1.58x, `sb_2x2` 1.56x, `sb_8x1` only 1.26x vs `sb_1x1`. At Kt>=2 `sb_2x4` and `sb_4x2` tie at ~1.65-1.69x, `sb_1x8` is ~1.6x and `sb_8x1` is the weakest (1.22-1.42x). Prefer a wide or 2x4/4x2 subblock over a tall 8x1. On BH p150b every 8-tile shape is 1.46× at Kt = 1 (`test_matmul_output_subblock_device_perf`, 2026-10-19).
- **A large Kt does not cancel the win on Wormhole B0.** Kt=1/2/4/8/12: best subblock 1.77x/1.66x/1.69x/1.68x/1.69x. Kt=16 does not fit in L1 at 16x16 output tiles (allocator failure). On BH p150b it does cancel most of it: Kt 1/2/4 = 1.46× / 1.17× / 1.10×, and Kt ≥ 8 already does not fit L1 at 16×16 output tiles on this build.

*How measured:* WH B0 n150 rows — `test_matmul_output_subblock_regimes` (1 Tensix core, sharded L1, bf16,
kernel-iters 20, DEVICE KERNEL DURATION per matmul, 11 interleaved trials, median; spread 24273..24325 / 13886..13915,
19151..19179 / 13647..13676, 666..675 / 654..659 ns), 2026-10-12; K and output-size sweeps from
`test_matmul_output_subblock_device_perf` on the same board. BH p150b rows — the same regimes test plus K /
output-size sweep runs on that board, `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19 (details in `report.md`).
Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_matmul_output_subblock.py::test_matmul_output_subblock_regimes`
(same regimes on any board; asserts only what held on every measured board, `measure` regimes only log).
