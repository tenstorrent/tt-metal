<!--
  perf-lab example README.
  Rules: no references to production ops / other examples; kernel-level concept only; correctness is
  the only pass/fail; every number is arch/box stamped; cores/placement/method front and center.
-->

# matmul_output_subblock — SRC-register argument reuse via the output-subblock shape

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** matmul output-subblock shape → SRC-register operand reuse
**First profiled on:** `bh-qb-11-special-dnijemcevic-for-reservation-42432` · BH · Blackhole · 2026-07-13

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

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_matmul_output_subblock.py::test_matmul_output_subblock_device_perf
```
Correctness (every subblock shape computes the identical C):
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_matmul_output_subblock.py::test_matmul_output_subblock_correctness
```
