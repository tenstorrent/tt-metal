# retile_face_regroup — change a TILE tensor's tile height by moving face-row runs, not by a round trip through row-major

**Problem type:** compute-bound: format conversion the math does not need  ·  **Concept(s):** (1) a tile-geometry change is a byte permutation of 32-byte face rows — gather the output tiles straight from the input tiles in the reader, no compute kernel; (2) issue one NoC read per *run* of consecutive face rows, and loop per run, not per face row
**Measured on:** BH p150b, 13×10 = 130 Tensix cores (`bh-50-special-adamjanovic-for-reservation-177924`, 2026-11-24) ([`report_bh_p150b.md`](report_bh_p150b.md))
**Wormhole:** not measured.

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
A `Layout::TILE` tensor with `[th_in, 32]` tiles has to become the same data with `[th_out, 32]` tiles
(th ∈ {1, 2, 4, 8, 16, 32}). A `[th, 32]` bf16 tile is stored as faces of min(th, 16) rows × 16 columns — two side
by side for th ≤ 16, four for th = 32 — so one **face row** (16 elements, 32 bytes) is contiguous in every tile
shape, and every face row of an output tile is exactly one face row of an input tile. The obvious implementation —
untilize to sticks, tilize again — converts twice and moves every face row as its own 32-byte transfer.

## What this isolates — and how
- **Setup:** interleaved DRAM bf16 in and out, same logical [H, W]. Work unit = max(th_in, th_out) rows × CW ≤ 8
  tile-columns, units split contiguously over min(units, grid) Tensix cores. Every variant reads the unit's input
  tiles from DRAM into the same L1 staging buffer and writes the output tiles with the same writer.
- **Varied:** only the middle step.

| Variant | What it does |
|---|---|
| `via_row_major` *(baseline)* | un-face the staged tiles into row-major sticks (one 32 B L1-local NoC read per face row), then a compute kernel tilizes into `[th_out, 32]` tiles |
| `face_rows` | gather the output tiles straight from the staged input tiles, one 32 B L1-local NoC read per face row; no compute kernel |
| `runs` | the same gather, one read and one loop iteration per run of min(input face height, output face height) face rows — consecutive in both faces (16 ↔ 32: a whole 512 B face) |

## Measured result (BH p150b, median of 3, per-launch latency)

| [H, W] | th_in → th_out | Tensix cores | via_row_major ns | face_rows ns | runs ns | runs vs baseline |
|---|---|---:|---:|---:|---:|---:|
| [32, 64] | 16 → 32 | 1 | 4 133 | 4 104 | 1 134 | **3.64×** |
| [32, 64] | 32 → 16 | 1 | 4 344 | 3 585 | 1 118 | **3.89×** |
| [32, 256] | 16 → 32 | 1 | 12 159 | 14 187 | 2 227 | **5.46×** |
| [32, 256] | 32 → 16 | 1 | 12 883 | 11 928 | 2 105 | **6.12×** |
| [128, 512] | 16 → 32 | 8 | 12 386 | 14 456 | 2 462 | **5.03×** |
| [32, 64] | 8 → 4 | 4 | 1 955 | 1 571 | 1 136 | 1.72× |
| [2048, 2048] | 16 → 32 | 130 | 65 088 | 73 815 | 45 489 | 1.43× |
| [2048, 2048] | 32 → 16 | 130 | 60 856 | 60 010 | 43 399 | 1.40× |
| [32, 64] | 1 → 32 | 1 | 5 817 | 5 426 | 5 437 | 1.07× |
| [128, 512] | 1 → 32 | 8 | 17 993 | 18 838 | 18 862 | **0.95×** |
| [2048, 2048] | 1 → 32 | 130 | 274 553 | 275 567 | 277 339 | 0.99× |

**Reading of the result:**
- **The cost is the reader's loop, not the bytes.** `face_rows` drops the compute kernel but keeps one 32 B read
  and one address computation per face row, and is no faster than the baseline (0.86–1.24×). `runs` issues the same
  bytes as 4 reads per 32×32 tile (16 ↔ 32) and is 3.6–6.1× faster on a few Tensix cores: ~512 RISC-V loop
  iterations per 8 tiles become 32.
- **It shrinks once DRAM sets the time:** full grid, [2048, 2048]: 1.40–1.43×.
- **Runs of one face row (th = 1 or 2 on either side) do not get faster this way:** 1 → 32 is 32-byte runs
  whatever the loop — 0.95–1.07×. There the per-read issue cost is the stage to cut (fewer instructions per read,
  e.g. a NoC read state set once and reused), or the 32-byte gather has to move to where the runs are long.

## When it helps / when it does not
- **Helps:** both tile heights ≥ 8 rows per face (16 ↔ 32: 3.6–6.1× on 1–8 Tensix cores; 8 → 4: 1.72×).
- **Does not help:** a side with 1- or 2-row tiles (runs of 32 B): 0.95–1.07×.
- **Smaller on a DRAM-bound full grid:** 1.40–1.43×.

## CLI
```bash
python -m ttnn.operations.examples.retile_face_regroup [--cells "H:W:th_in:th_out,..."] [--trials N] [--report PATH]
```
Correctness (every variant, bit-exact against the input):
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_retile_face_regroup.py -k correctness
```

## Code
`program_descriptor_with_inline_kernels.py`: `_READER` mode 0 (un-face into sticks), mode 1 (one read per face row),
mode 2 (`run` = min of the two face heights, one read per run), `face_row_offset()` (byte offset of a tile-local row
and column half); `_COMPUTE` (baseline tilize into `[th_out, 32]` tiles, `TileDescriptor` on its CBs); `_WRITER`.
