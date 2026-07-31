# dual_noc_read — device report

Two independent DRAM operand streams, one Tensix core, one vs two data-movement RISC-Vs fetching
them. Metric is `DEVICE KERNEL DURATION [ns]` from the in-process device profiler. Correctness
(`C == A*B`) is gated separately and passes for every variant × block; perf below is measured
evidence, never a bound.

Reentrant: re-measuring on another box/arch **appends** a new block below rather than replacing
these numbers.

---

## Blackhole P150 — 2026-07-31

```
box=bh-qb-13-special-dnijemcevic-for-reservation-52900   arch=blackhole   git=4a1d6a97ca9
cores=1   placement=single core (0,0)
shape=(1024, 128)   tiles/operand=128   dtype=bfloat16   tile_bytes=2048
N=5 windows x 10 launches (median +- pstdev)
DRAM READ traffic = 0.524 MB/launch (both operands; output stays in L1 -> no write traffic)
read GB/s = 524288 B / kernel_ns
max spread across all cells = 0.5%  -> every delta below is far outside noise
```

### [1] Full op (`C = A*B`)

| block | variant | riscs | ns/op | ±% | read GB/s | vs base |
|---|---|---|---|---|---|---|
| 1 | `one_riscv` | 1 | 60047.4 | 0.0 | 8.7 | (base) |
| 1 | `two_riscv` | 2 | 52713.8 | 0.0 | 9.9 | **1.14×** |
| 1 | `two_riscv_sem` | 2 | 61981.3 | 0.0 | 8.5 | 0.97× |
| 2 | `one_riscv` | 1 | 33411.2 | 0.1 | 15.7 | (base) |
| 2 | `two_riscv` | 2 | 29349.9 | 0.5 | 17.9 | **1.14×** |
| 2 | `two_riscv_sem` | 2 | 32867.1 | 0.1 | 16.0 | 1.02× |
| 4 | `one_riscv` | 1 | 20543.6 | 0.1 | 25.5 | (base) |
| 4 | `two_riscv` | 2 | 16650.4 | 0.3 | 31.5 | **1.23×** |
| 4 | `two_riscv_sem` | 2 | 18402.8 | 0.2 | 28.5 | 1.12× |
| 8 | `one_riscv` | 1 | 14257.8 | 0.1 | 36.8 | (base) |
| 8 | `two_riscv` | 2 | 10668.8 | 0.2 | 49.1 | **1.34×** ← best full-op |
| 8 | `two_riscv_sem` | 2 | 11617.9 | 0.2 | 45.1 | 1.23× |
| 16 | `one_riscv` | 1 | 11477.3 | 0.1 | 45.7 | (base) |
| 16 | `two_riscv` | 2 | 9061.6 | 0.1 | 57.9 | 1.27× |
| 16 | `two_riscv_sem` | 2 | 9094.9 | 0.2 | 57.6 | 1.26× |
| 32 | `one_riscv` | 1 | 10815.9 | 0.1 | 48.5 | (base) |
| 32 | `two_riscv` | 2 | 9461.5 | 0.1 | 55.4 | 1.14× |
| 32 | `two_riscv_sem` | 2 | 9502.2 | 0.1 | 55.2 | 1.14× |

### [2] Payload-ablated — the pure read ceiling

`mul_tiles` removed; **every** CB wait/reserve/push/pop and the whole `tile_regs` + `pack` cycle kept.
Output is garbage (never correctness-checked); the read pipeline and all its synchronization are
unchanged.

| block | variant | ablated ns | ±% | read GB/s | vs base | math cost (full − ablated) |
|---|---|---|---|---|---|---|
| 1 | `one_riscv` | 60204.9 | 0.0 | 8.7 | (base) | ≈0 |
| 1 | `two_riscv` | 52859.5 | 0.0 | 9.9 | **1.14×** | ≈0 |
| 1 | `two_riscv_sem` | 61967.7 | 0.0 | 8.5 | 0.97× | ≈0 |
| 2 | `one_riscv` | 33410.4 | 0.0 | 15.7 | (base) | ≈0 |
| 2 | `two_riscv` | 29385.7 | 0.3 | 17.8 | **1.14×** | ≈0 |
| 2 | `two_riscv_sem` | 32731.0 | 0.1 | 16.0 | 1.02× | 136 |
| 4 | `one_riscv` | 20267.0 | 0.1 | 25.9 | (base) | 277 |
| 4 | `two_riscv` | 16384.4 | 0.1 | 32.0 | **1.24×** | 266 |
| 4 | `two_riscv_sem` | 18177.6 | 0.3 | 28.8 | 1.11× | 225 |
| 8 | `one_riscv` | 13717.1 | 0.3 | 38.2 | (base) | 541 |
| 8 | `two_riscv` | 10029.0 | 0.2 | 52.3 | **1.37×** | 640 |
| 8 | `two_riscv_sem` | 11006.8 | 0.1 | 47.6 | 1.25× | 611 |
| 16 | `one_riscv` | 10260.3 | 0.1 | 51.1 | (base) | 1217 |
| 16 | `two_riscv` | 6879.7 | 0.2 | 76.2 | **1.49×** | 2182 |
| 16 | `two_riscv_sem` | 7405.7 | 0.2 | 70.8 | 1.39× | 1689 |
| 32 | `one_riscv` | 8561.6 | 0.2 | 61.2 | (base) | 2254 |
| 32 | `two_riscv` | 5388.1 | 0.1 | **97.3** | **1.59×** ← best read | 4073 |
| 32 | `two_riscv_sem` | 5663.6 | 0.4 | 92.6 | 1.51× | 3839 |

The `math cost` column is the bottleneck handoff: ~0 while reads dominate, rising to ~4.1 µs at
`block=32`, and *larger* for `two_riscv` than the baseline at `block≥16`. Once reads are fast enough
the FPU sets the pace, which is why the full-op win peaks at `block=8` while the read win keeps
climbing.

### [3] Mechanism probe — commands vs bytes

Payload-ablated, `block=8`, **total bytes held constant** while the transaction size shrinks, which
issues proportionally more NoC commands for the same traffic.

| txn bytes | commands | variant | ns/op | ±% | read GB/s | vs base |
|---|---|---|---|---|---|---|
| 2048 | 256 | `one_riscv` | 13695.8 | 0.1 | 38.3 | (base) |
| 2048 | 256 | `two_riscv` | 10034.6 | 0.1 | 52.2 | 1.36× |
| 2048 | 256 | `two_riscv_sem` | 11000.1 | 0.2 | 47.7 | 1.25× |
| 1024 | 512 | `one_riscv` | 22876.1 | 0.0 | 22.9 | (base) |
| 1024 | 512 | `two_riscv` | 14433.7 | 0.5 | 36.3 | 1.58× |
| 1024 | 512 | `two_riscv_sem` | 15069.8 | 0.1 | 34.8 | 1.52× |
| 512 | 1024 | `one_riscv` | 38660.8 | 0.1 | 13.6 | (base) |
| 512 | 1024 | `two_riscv` | 22034.7 | 0.1 | 23.8 | 1.75× |
| 512 | 1024 | `two_riscv_sem` | 23187.7 | 0.0 | 22.6 | 1.67× |
| 256 | 2048 | `one_riscv` | 66814.4 | 0.0 | 7.8 | (base) |
| 256 | 2048 | `two_riscv` | 35956.1 | 0.0 | 14.6 | **1.86×** |
| 256 | 2048 | `two_riscv_sem` | 36900.3 | 0.0 | 14.2 | 1.81× |

Same bytes throughout. 8× the commands costs **4.9×** the time and collapses achieved bandwidth
38.3 → 7.8 GB/s, while the two-engine win climbs toward the ideal 2×. Cost tracks **commands**, not
bytes.

---

## Cross-check: a NoC-only model (tt-npe)

The same transfer set fed to `tenstorrent/tt-npe` (built from source at this date), which models NoC
links/NIUs and has **no** notion of RISC-V command-issue cost. All transfers released at cycle 0 —
i.e. the NoC-only lower bound, with no issue-rate or barrier serialization. Blackhole, 8 DRAM bank
NIUs → one worker core.

| txn bytes | commands | 1 NoC (cyc / GB/s@1GHz) | 2 NoC (cyc / GB/s) | model's 2-NoC speedup | avg link util |
|---|---|---|---|---|---|
| 2048 | 256 | 8624 / **60.8** | 4312 / 121.6 | 2.00× | ~11% |
| 1024 | 512 | 8640 / **60.7** | 4320 / 121.4 | 2.00× | ~11% |
| 512 | 1024 | 12295 / **42.6** | 6151 / 85.2 | 2.00× | ~11% |
| 256 | 2048 | 16324 / **32.1** | 8132 / 64.5 | 2.01× | ~11% |

Measured vs the model's one-engine bound:

| txn bytes | model bound | measured (1 RISC) | fraction of bound |
|---|---|---|---|
| 2048 | 60.8 GB/s | 38.3 | 63% |
| 1024 | 60.7 | 22.9 | 38% |
| 512 | 42.6 | 13.6 | 32% |
| 256 | 32.1 | 7.8 | **24%** |

**Reading:** average link utilisation is only ~11%, so nothing on the wire is saturated, and the
kernel falls further below the NoC's own bound the more commands it issues (63% → 24%). That
shortfall is per-command cost *upstream* of the NoC. Note honestly that the model *also* predicts a
clean ~2.0× from spreading transfers across two NoCs even with issue cost removed, so the port's
injection path is a real serializer too. Since RISC and NoC cannot be varied independently (below),
the two cannot be fully separated — but the measurements track **commands**, so issue rate dominates.

Two reproduction notes for anyone repeating this:
- This tt-npe build's JSON loader does an **unguarded** lookup of `golden_result` and aborts
  (`boost out_of_range`) if the key is absent; a dummy `{"golden_result": {"cycles": 0}}` is required
  even when only the prediction is wanted. Its shipped `programmatic_workload_generation.py` does not
  match the built pybind API (`Transfer`'s `dst` expects an `NocDestination`, not a `Coord`), and
  `tt_npe.py` reads a `Stats.wallclock_runtime_us` attribute that `Stats` does not expose.
- `dram_bw_util` came back `inf` (a divide-by-zero artifact) and was not used.

---

## Why the two contributions cannot be separated by A/B

One RISC is one NoC. Each data-movement processor is bound to a single port (NCRISC → NoC 0,
BRISC → NoC 1), and firmware initializes only that port's per-RISC state
(`noc_local_state_init(noc_index)`) — command buffers and outstanding-request counters included.

So the factorial that would name the mechanism outright — one RISC driving both ports, two RISCs
sharing one port — has no cells to fill: "add a RISC" and "add a port" are the same knob. Passing a
non-default NoC index to `noc_async_read` / `noc_async_read_barrier` is not a way around it; the
barrier would be waiting on state that was never set up for that RISC.

Hence the mechanism is established by measurement instead — table [3] (bytes fixed, command count
scaled) plus the NoC-model cross-check above. Both point at issue rate.

---

## Multi-core scaling — and why the split can turn into a LOSS

Per-core work held constant (32 tiles/operand/core, so total bytes grow with the grid), payload
ablated, `block=8`. 8-wide rectangles up to 8×8 = 64 cores.

`one_riscv_brisc` is a **diagnostic**, not part of the ladder: BRISC reads *both* operands, so all
reads ride NoC 1 with the SAME single-RISC issue load as the baseline. Since a RISC and its port move
together, this is the only way to see the port/route contribution on its own — any gap vs `one_riscv`
is the NoC, not issue rate.

| cores | grid | variant | ns/op | GB/s | GB/s/core | vs base |
|---|---|---|---|---|---|---|
| 1 | 1×1 | `one_riscv` | 3525 | 37.2 | 37.2 | (base) |
| 1 | 1×1 | `two_riscv` | 2686 | 48.8 | 48.8 | **1.31×** |
| 1 | 1×1 | `one_riscv_brisc` | 3607 | 36.3 | 36.3 | 0.98× |
| 8 | 8×1 | `one_riscv` | 4151 | 252.6 | 31.6 | (base) |
| 8 | 8×1 | `two_riscv` | 6129 | 171.1 | 21.4 | **0.68×** ← loss |
| 8 | 8×1 | `one_riscv_brisc` | 12152 | 86.3 | 10.8 | **0.34×** |
| 16 | 8×2 | `one_riscv` | 6368 | 329.3 | 20.6 | (base) |
| 16 | 8×2 | `two_riscv` | 8239 | 254.5 | 15.9 | 0.77× |
| 16 | 8×2 | `one_riscv_brisc` | 15795 | 132.8 | 8.3 | 0.40× |
| 32 | 8×4 | `one_riscv` | 12229 | 343.0 | 10.7 | (base) |
| 32 | 8×4 | `two_riscv` | 12349 | 339.7 | 10.6 | 0.99× |
| 32 | 8×4 | `one_riscv_brisc` | 22815 | 183.8 | 5.7 | 0.54× |
| 64 | 8×8 | `one_riscv` | 23426 | 358.1 | 5.6 | (base) |
| 64 | 8×8 | `two_riscv` | 20447 | 410.3 | 6.4 | 1.15× |
| 64 | 8×8 | `one_riscv_brisc` | 38082 | 220.3 | 3.4 | 0.62× |

**Two things are happening, and they fight each other.**

**(i) DRAM saturates.** Aggregate read bandwidth plateaus at ~340–410 GB/s from 16 cores up, while
per-core bandwidth collapses 37.2 → 5.6 GB/s. At 64 cores the read is DRAM-bound, not issue-bound, so
relieving issue rate has almost nothing left to give.

**(ii) NoC 1 is a much worse route for DRAM reads at spread placements.** The diagnostic isolates
this: same issue load, reads moved to the other port, and it costs **0.34–0.62×** from 8 cores up —
while being neutral (0.98×) on a single core, where routing is irrelevant. Reads want NoC 0: its
dimension-ordered east→south routing *disperses* column-localized DRAM traffic, whereas NoC 1's
north→west *concentrates* it onto the DRAM columns. Writes are the mirror image, which is exactly why
the reader/writer default pairing is NoC 0 / NoC 1. This reproduces the effect `noc_placement`
measures directly (reads on NoC 0 / writes on NoC 1 being 2.5–4.8× faster than the reverse for spread
placements) — here it arrives as a side effect of the operand split, since a RISC and its port move
together.

Splitting the operands puts **half the read traffic on the worse route**. At a bf16 tile page that
costs more than the second engine gains — hence the 0.68× on a single row of 8 cores. As the grid
fills out the penalty eases and DRAM saturation flattens both effects, so it returns to ~1.15×.

## Transaction size flips the sign

Same grids, `block=8`, payload ablated, sweeping bytes per NoC command.

| cores | txn B | `two_riscv` vs base | `one_riscv_brisc` (NoC-1 penalty) |
|---|---|---|---|
| 1 | 2048 | 1.32× | 0.97× |
| 1 | 1024 | 1.58× | 1.00× |
| 1 | 512 | 1.74× | 1.00× |
| 1 | 256 | **1.85×** | 1.00× |
| 8 | 2048 | **0.68×** | 0.34× |
| 8 | 1024 | 0.94× | 0.48× |
| 8 | 512 | **1.34×** | 0.76× |
| 8 | 256 | **1.74×** | 1.05× |
| 64 | 2048 | 1.14× | 0.62× |
| 64 | 1024 | 1.18× | 0.63× |
| 64 | 512 | **1.46×** | 0.78× |
| 64 | 256 | **1.66×** | 1.05× |

As transactions shrink, per-command issue cost grows (favouring the split) **and** the NoC-1 route
penalty fades (0.34 → 1.05 at 8 cores) — both push the same way. So the loss is specific to large,
tile-sized reads on a spread placement: by ~512 B the split is already a **1.34×** win at 8 cores and
**1.46×** at 64.

**Practical rule.** The split is worth it when reads are *command*-limited, i.e. many small
transactions — which is the normal case for block-float weight pages (a bfp4 tile page is 576 B, bfp8
1088 B). It is a regression when reads are large, tile-sized, and spread over a moderate core count,
because then you are paying the NoC-1 read-route penalty for nothing.

---

## Summary

1. **Splitting two independent operand streams across the two data-movement RISC-Vs wins at every
   block size measured** — 1.14–1.34× on the full op, 1.14–1.59× on the read alone — for zero extra
   L1 and no change to the compute kernel.
2. **What is relieved is predominantly RISC-V command-issue rate, not link saturation.** Links run at
   ~11% utilisation; cost tracks command count (8× commands → 4.9× time at fixed bytes); and the win
   grows toward 2× as commands multiply. The NoC port's injection path contributes a smaller share
   that cannot be isolated, since a RISC and its port move together.
3. **Corollary:** the smaller the transactions, the more this is worth (1.36× at 2048 B → 1.86× at
   256 B). bf16 whole-tile reads are the *least* favourable case for it.
4. **The full-op win peaks at `block=8` and decays** — not because the trick stops working, but
   because the FPU takes over as the critical path. Ablate before drawing conclusions.
5. **Skip the semaphores unless you need them.** `two_riscv_sem` is a loss at `block=1` (0.97×) and
   only converges to the handshake-free form by `block≥16`. Pay the handshake only when the reader
   must keep CB ownership to post-process the block.
