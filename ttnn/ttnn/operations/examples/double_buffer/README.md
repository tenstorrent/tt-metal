# double_buffer — keeping bytes in flight on the NoC (reads/barrier · double buffering · transfer size)

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** outstanding NoC reads per barrier (latency vs bandwidth) · double-buffered CBs (stage overlap) · transfer size (bytes per tile)
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · Wormhole B0 · 1000 MHz · 2026-07-09 · `bf5ebe7aacb`

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
# many cores: already DRAM-bandwidth-bound, so the levers stop mattering
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
- **Once DRAM-bandwidth-bound, none of the levers matter.** At 64 cores even the `block=1`
  single-buffered trap hits **190.8 GB/s** (≈ DRAM peak); every richer setting is within noise or
  slightly worse. When the memory system is saturated the only lever left is moving less data.
- Scope: this measures data movement, so compute is held light. If **compute** is your bottleneck,
  the compute engine's independent unpack/math/pack threads already hide the DRAM traffic and none of
  these levers help.

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
