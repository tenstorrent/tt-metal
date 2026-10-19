# double_buffer — keeping bytes in flight on the NoC (reads/barrier · double buffering · transfer size)

**Problem type:** data movement below the ceiling: latency  ·  **Concept(s):** outstanding NoC reads per barrier (latency vs bandwidth) · double-buffered CBs (stage overlap) · transfer size (bytes per tile)
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · Wormhole B0 · 1000 MHz · 2026-07-09 · `bf5ebe7aacb`
**Measured on:** WH B0 n150 (`bgd-lab-t3003-special-mstaletovic-for-reservation-40918`, 2026-07-09; regimes 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
You write a tiny op as a three-stage pipeline: a **reader** pulls tiles from DRAM into a circular
buffer, a **compute** kernel applies an eltwise op and writes a second CB, and a **writer** drains
that CB back to DRAM. The most obvious way to write the reader is *read one tile, wait for it, push
it, repeat* — and to size each CB to the minimum that works, one tile. It's correct. It's also slow:
each tile pays a full DRAM round trip **before the next read is even issued**, so the NoC sits idle
most of the time and you never see DRAM bandwidth.

The single idea behind this example: **how many bytes you keep in flight on the NoC** is what decides
whether you're latency-bound (NoC idle) or bandwidth-bound (NoC saturated). Three kernel-level levers
raise bytes-in-flight.

## What this isolates — and how
- **Lever 1 — reads/writes per barrier (`block`):** issue `block` async reads back-to-back, then
  **one** barrier for all of them (the writer is symmetric). `block=1` is the trap; `block>1` keeps up
  to `block` transactions in flight so they pipeline → toward DRAM **bandwidth**.
- **Lever 2 — double buffering (`variant`):** the CBs are `depth * block` tiles deep. `single_buffered`
  (depth 1) makes the reader wait for a whole block to drain before refilling; `double_buffered`
  (depth 2) lets it prefetch the next block, so the reader (NoC0) and writer (NoC1) streams overlap.
- **Lever 3 — transfer size (dtype):** bytes per tile, i.e. per NoC transaction — bfloat8_b 1088 B,
  bfloat16 2048 B, float32 4096 B. Bigger transactions amortize fixed per-transaction latency.
- **Isolation setup:** the eltwise compute is held light (one `relu`), the tensor is interleaved in
  DRAM, reader on NoC0 / writer on NoC1. The reader / compute / writer kernels are **byte-identical**
  across every `block`, both variants, and all three dtypes — only compile-time args and each CB's
  `total_size` change (the CB page size is queried from the tensor, never hard-coded). So the measured
  delta is purely these three data-movement levers.
- **Why it's kernel-level:** `block` is a loop bound the kernel author picks, CB depth is a `total_size`
  in the program descriptor, and the transfer size is which dtype tile you move — all kernel-author
  decisions, not model choices.

## The methods being compared
| Lever | Values | Effect |
|---|---|---|
| `block` (reads/writes per barrier) | `1` (trap) … `32` | `1` = latency-bound (a DRAM round trip per tile); larger = more transactions in flight |
| `variant` (CB depth) | `single_buffered` / `double_buffered` | double lets the reader prefetch → read+write streams overlap |
| transfer size (dtype) | `bfloat8_b` / `bfloat16` / `float32` | bigger tile = more bytes per NoC transaction |

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.double_buffer [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--shape` | `H,W` | `512,512` | tile-aligned tensor (256 tiles) |
| `--cores` | int | `1` | cores running the pipeline (each independent) |
| `--blocks` | int list | `1,2,4,8,16,32` | reads/writes-per-barrier sweep |
| `--dtype` | `bfloat8_b\|bfloat16\|float32` | `bfloat16` | tile format = transfer size (~1088 / 2048 / 4096 B) |
| `--passes` | int | `1` | relu repeats = compute weight (kept light) |
| `--iters` | int | `1` | in-kernel repeat of the tile range (1 = latency, large = steady-state) |
| `--trials` | int | `20` | profiled launches per case (averaged) |

```bash
# one core: the latency->bandwidth curve, single vs double buffered
python -m ttnn.operations.examples.double_buffer --cores 1 --blocks 1,2,4,8,16,32
# transfer size: bigger tiles reach higher bandwidth
python -m ttnn.operations.examples.double_buffer --cores 1 --dtype float32 --blocks 1,4,8
# many cores: on WH B0 n150 already DRAM-bandwidth-bound, so the levers stop mattering (BH p150b: still ~1.08x)
python -m ttnn.operations.examples.double_buffer --cores 64 --shape 2048,2048 --blocks 1,8
```

## Measured result
*Illustrative — see the **First profiled on** stamp and `report.md`; re-run the CLI for your box.
GB/s = read+write DRAM traffic (2 × tensor bytes) ÷ device kernel time.*

```
double_buffer   box=bgd-lab-t3003...  arch=WH-B0   N=20 (avg)   passes=1   kernel-iters=1

  # block x depth, 1 core, bf16 (256 tiles):
  block= 1  single_buffered  162384 ns    6.5 GB/s   (trap)
  block= 4  single_buffered   93567 ns   11.2 GB/s   1.74×      # batching alone: ~2x, caps ~13 GB/s
  block= 4  double_buffered   58513 ns   17.9 GB/s   2.78×  ←   # + double buffering: compounds
  block=32  double_buffered   62311 ns   16.8 GB/s   2.61×      # past the sweet spot: slightly worse

  # transfer size, 1 core, best (double-buffered) cell per dtype:
  bfloat8_b (1088 B/tile)   block=4   56867 ns    9.8 GB/s      # least data -> fastest wall time
  bfloat16  (2048 B/tile)   block=4   58513 ns   17.9 GB/s
  float32   (4096 B/tile)   block=8   66232 ns   31.7 GB/s      # biggest transaction -> highest GB/s

  # 64 cores, bf16 (4096 tiles): already bandwidth-bound
  block= 1  single_buffered   87916 ns  190.8 GB/s   (trap)     # ~DRAM peak with no tuning at all
```

**Reading of the result:**
- **`block=1` is the trap** — on one core it caps you at 6.5 GB/s. Trap → `block=4, double_buffered`
  is **2.78×** (17.9 GB/s), and the levers **compound**: batching alone buys ~2× but *saturates at ~13
  GB/s* (still can't overlap read and write); double buffering then lifts the ceiling to the
  single-core NoC limit.
- **Batching plateaus by ~`block=4`; more does not help.** Verified flat from `block=4` to `block=64`
  (bf16 double: 18.3 → 17.7 GB/s; `block=256` OOMs L1). Past ~4 outstanding reads you are
  *transaction-rate-bound*, not latency-bound — the single core retires only ~8–9 M tile-transactions/s
  (~115 ns each) regardless of how many are in flight. Sweet spot ~4–8; bigger just wastes L1.
- **The single-core ceiling is set by TRANSFER SIZE, not block.** Since GB/s ≈ (transaction rate) ×
  (bytes per transaction) and the rate is fixed, the ceiling scales with tile bytes: **bfp8 ~10 → bf16
  ~18 → fp32 ~33 GB/s** (same kernel; only the dtype differs). So bf16 will *not* reach 32 GB/s on one
  core by cranking `block` — that needs bigger NoC transactions (coalescing multiple bank-contiguous
  tiles per read); fp32 gets there for free because its tile is already 4 KB. A smaller dtype moves
  less data (wins wall time); a bigger dtype moves more per transaction (wins GB/s).
- **Once DRAM-bandwidth-bound, none of the levers matter.** On WH B0 n150, at 64 Tensix cores even the
  `block=1` single-buffered trap hits **190.8 GB/s** (≈ DRAM peak); every richer setting is within noise or
  slightly worse. When the memory system is saturated the only lever left is moving less data. On BH p150b
  the DRAM ceiling is ~2.2× higher and the trap does not reach it even on the full 120-Tensix-core grid, so
  the lever keeps a small ~1.07–1.09× gain there (see below).
- Scope: this measures data movement, so compute is held light. If **compute** is your bottleneck,
  the compute engine's independent unpack/math/pack threads already hide the DRAM traffic and none of
  these levers help.

## When it helps / when it does not
**Rule:** the lever pays while `active Tensix cores × trap rate per Tensix core` is well below the board's DRAM
ceiling — it is flat from **≈32 Tensix cores on WH B0 n150** (~6.5 GB/s per Tensix core vs ~190 GB/s), while on
**BH p150b** (~8.8 GB/s per Tensix core vs ~415–430 GB/s) the gain shrinks to ~1.07–1.09× from 64 Tensix cores up
and **never reaches a clean flat**, even on the full grid. bf16 (2048 B per tile = per NoC read), interleaved DRAM
(`TensorMemoryLayout::INTERLEAVED`); baseline = the trap (`block=1`, `single_buffered`), lever = `block=4`,
`double_buffered` (4 reads per barrier, CB depth 2 × 4 tiles). GB/s = read+write DRAM traffic ÷ device time.

| Regime (same definition on every board) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Helps:** 1 Tensix core, latency-bound, 256 tiles `[512, 512]` | 155 933 → 59 156 ns **2.64×** (6.7 → 17.7 GB/s) | 118 883 → 40 361 ns **2.95×** (8.8 → 26.0 GB/s) |
| **Helps:** 8 Tensix cores, 512 tiles per Tensix core `[2048, 2048]` | 322 468 → 118 236 ns **2.73×** (52 → 142 GB/s) | 238 575 → 86 108 ns **2.77×** (70 → 195 GB/s) |
| Sweep: 16 Tensix cores, 256 tiles per Tensix core `[2048, 2048]` | 181 924 → 92 193 ns **1.97×** | 121 075 → 49 568 ns **2.44×** |
| Sweep: 32 Tensix cores, 128 tiles per Tensix core `[2048, 2048]` | 101 087 → 90 876 ns **1.11×** | 66 587 → 43 079 ns **1.55×** (252 → 390 GB/s) |
| **Does not help:** 64 Tensix cores, 64 tiles per Tensix core `[2048, 2048]` | 87 884 → 89 425 ns **0.98×** (trap 190.9 GB/s) | 44 143 → 40 994 ns **1.08× — small gain on BH** (380 → 409 GB/s) |
| **Small:** full grid, `[2048, 2048]` | — (64 Tensix cores is the full grid, row above) | 120 Tensix cores, ~34 tiles each: 43 554 → 40 519 ns **1.07×** (385 → 414 GB/s) |
| Full grid, 4× the tiles `[4096, 4096]` | not measured | 120 Tensix cores: 169 144 → 155 676 ns **1.09×** (397 → 431 GB/s) |
| **Boundary** | flat from ≈32 Tensix cores | shrinks to ~1.07–1.09× from 64 Tensix cores; no flat regime on BH p150b |

**Why the boards differ:**
- **DRAM ceiling.** The achievable read+write DRAM bandwidth is ~190 GB/s on WH B0 n150 (12 DRAM banks) and
  ~415–430 GB/s on BH p150b (8 DRAM banks, faster GDDR6). The trap saturates WH B0 at ~30 Tensix cores; on BH
  p150b even 120 Tensix cores of trap reach only ~385 GB/s, so there is still idle NoC time for the lever to
  recover.
- **Per-Tensix-core rate.** One Tensix core moves 26.0 GB/s with the lever on BH p150b vs 17.7 GB/s on WH B0
  (consistent with the 64 B vs 32 B NoC flit and 1350 vs 1000 MHz AICLK; not isolated here); the single-Tensix-core
  gain is a little larger (2.95× vs 2.64×).

- **Helps:** while each Tensix core's NoC is idle between one-tile round trips, batching 4 reads per barrier and
  letting the reader prefetch the next block (CB depth 2) overlaps reads with writes; the gain is per Tensix
  core, so it holds unchanged at 8 Tensix cores on both boards.
- **Does not help (WH B0 n150):** with enough Tensix cores the trap's aggregate reaches the DRAM ceiling and there
  is no idle NoC time left to recover; the lever is flat to ~3% slower. On BH p150b the same condition is never
  reached on the 120-Tensix-core grid, so expect ~1.07–1.09× there — keep the lever, it costs only L1.

*How measured:* `DEVICE KERNEL DURATION [ns]`, baseline → lever.
- **WH B0 n150:** 1, 8 and 64 Tensix cores from `test_double_buffer_regimes` (median of 12 trials
  per variant, a trial = mean of 5 launches, variants interleaved; speedups stable over 4 runs), branch
  `adamjanovic/perf-example-improvements`, 2026-10-12; 16 and 32 Tensix cores from that day's
  `test_double_buffer_device_perf` core-count sweep (mean of 20 launches; [`report.md`](report.md)).
- **BH p150b:** 1, 8 and 64 Tensix cores from `test_double_buffer_regimes`; 120 Tensix cores from a BH-only
  regime of the same A/B that the arch-agnostic test no longer runs (`small_120cores_34tiles_per_core`, same
  method); 16, 32 Tensix cores and `[4096, 4096]` from probes of the same interleaved A/B.
  `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19, commit `f74ee7aad64`.

Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_double_buffer.py::test_double_buffer_regimes`
(same regimes on any board; asserts only what held on every measured board, `measure` regimes only log).

## Run the predefined sweep (regenerates the report numbers)
```bash
# correctness: every dtype x variant x block == relu(input) (covers block-remainder + idempotency)
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_double_buffer.py::test_double_buffer_correctness
# device kernel duration + GB/s, block x variant (set DB_DTYPE / DB_CORES / DB_SHAPE to pick a config)
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_double_buffer.py::test_double_buffer_device_perf
```

## Code
`double_buffer.py`: `block` is the reader/writer loop's reads-per-barrier; depth is
`total_size = depth * block * page_bytes` on the two `CBDescriptor`s; `page_bytes` (transfer size) is
queried from the tensor per dtype. The three `kernels/db_*.cpp` are identical across every setting.
