# transfer_alignment — device performance report

Metric: `DEVICE KERNEL DURATION [ns]` from the in-process device profiler, averaged
over `trials` launches (warmup discarded, flush-bracketed). Correctness is asserted
separately in `test_transfer_alignment_correctness` (both variants extract a bit-exact
sub-page span) and `test_transfer_alignment_control_byte_identical` (constant source ->
both variants byte-identical). Perf is evidence, never a pass/fail. Numbers are
illustrative of the stamped box/arch — re-run the CLI for yours.

## wormhole_b0 — 2026-07-22

- **box:** `bgd-lab-t3002-special-dnijemcevic-for-reservation-33181`
- **arch:** wormhole_b0 (WH), ~1000 MHz
- **git:** `b54bcdedf74`
- **config:** cores=1, single-core placement, iters=1 (per-launch latency), trials=10
- **alignment (queried):** DRAM window = 32 B, L1 = 16 B; misalignment residue used = 16 B
- **span element:** bfloat16 (2 B); misaligned over-read = span_bytes + 16 B

### Per-launch latency (iters=1), span-width x span-count

| N (spans) | width (B) | misaligned (ns) | aligned (ns) | over-read (B) | speedup |
|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 12024.5 | 7096.9 | 80 | 1.69x |
| 16 | 256 | 20571.7 | 7385.9 | 272 | 2.79x |
| 16 | 1024 | 55025.7 | 8137.8 | 1040 | 6.76x |
| 16 | 4096 | 192829.0 | 10529.9 | 4112 | 18.31x |
| 64 | 64 | 46681.5 | 26851.7 | 80 | 1.74x |
| 64 | 256 | 81206.7 | 28176.2 | 272 | 2.88x |
| 64 | 1024 | 218739.9 | 30739.7 | 1040 | 7.12x |
| 64 | 4096 | 769486.8 | 40734.2 | 4112 | 18.89x |
| 256 | 64 | 185884.5 | 105600.8 | 80 | 1.76x |
| 256 | 256 | 323364.2 | 111468.5 | 272 | 2.90x |
| 256 | 1024 | 873000.5 | 121845.7 | 1040 | 7.16x |
| 256 | 4096 | 3075847.6 | 161684.7 | 4112 | 19.02x |

## Findings

1. **The congruent (`aligned`) read is always faster — 1.7x at 64 B spans up to ~19x at
   4 KB spans.** Both variants deliver the same bytes (bit-exact; the constant-source
   control confirms byte-identity), so this is a real "same work, faster" result. The dodge
   is arranging the span start so its byte residue matches the destination residue: one
   direct `noc_async_read` moves exactly `span_bytes`.

2. **The tax is dominated by the per-span L1 realign, and it GROWS with width — the opposite
   of the naive "the over-read amortizes" intuition.** The `aligned` cost is nearly flat in
   width (7.1 us -> 10.5 us as the span goes 64 B -> 4096 B, i.e. ~1.5x for 64x the bytes):
   a single NoC transfer in the single-core latency regime, where per-span issue/barrier
   overhead dominates and bytes are cheap at NoC bandwidth. The `misaligned` cost grows
   nearly linearly with width (12 us -> 193 us, ~16x): the extra term over `aligned` is the
   CPU `memmove` that realigns the useful span out of the aligned scratch. A scalar RISC
   byte-move gets no NoC bandwidth and scales with the span size, so the relative penalty
   widens as spans grow.

3. **The over-read is negligible — the memmove is the whole tax.** The misaligned path reads
   only `residue` (16 B) extra, even at width=4096 (4112 vs 4096 useful). A 16-byte over-read
   cannot explain an ~19x gap; the entire delta is the forced L1 realign. The lesson is not
   "misaligned wastes bandwidth" — it wastes almost none — it is "a non-congruent sub-page
   read cannot be one NoC transfer, so it forces a CPU-side copy off the fast datamover."

4. **The speedup RATIO is flat across N; the ABSOLUTE gap grows with N.** Both variants do
   independent per-span work on one core, so total time scales linearly with N and the ratio
   is N-independent (18.3x -> 18.9x -> 19.0x at width=4096). The absolute ns saved grows with
   N because the per-span memmove is a fixed cost paid N times.

5. **Where it applies:** any reader extracting sub-page spans at arbitrary byte offsets
   (row-spans, unpadding, sub-tile gathers). Dodge it by arranging the span offset — or the
   CB write-pointer — so `(src_off % align) == (dst_off % align)`; then the read stays a single
   direct transfer. Matters most for wide spans (where the realign dominates); still a ~1.7x
   win even for tiny 64 B spans because the misaligned path adds a second (CPU) pass over data
   the aligned path never touches twice.
