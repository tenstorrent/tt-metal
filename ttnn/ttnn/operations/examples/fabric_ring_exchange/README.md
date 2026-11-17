# fabric_ring_exchange — the fixed cost of a fabric exchange: a tiny all-gather from L1 to L1

**What it is:** an all-gather around a ring (after one call every chip holds every other chip's slot), cut
down to what a fused CCL op does: each chip's slot is already in L1, it is sent from L1 and lands in L1 on
the peers, and it is small (128 B–32 KiB, usually one packet). No DRAM, no circular buffer, no copy cores.

**Goal:** show the fixed cost of a fabric exchange — what a correct exchange costs when there is next to no
data to move. That cost is almost all synchronization (the ready fence between calls), about 5 µs on a
4-chip ring, and the bytes add little on top.

**Concepts:** (1) the fixed cost of a correct ring exchange of a small payload, with no other work;
(2) direct multi-hop sends vs one-hop store-and-forward on a 4-chip ring.

**Situation:** a fused CCL op (here distributed_rms_norm) all-gathers a small per-row partial across a 4-chip
ring inside one program. Its perf rounds stubbed the op's payloads and called the time left over (10–14 µs) a
floor. This example measures what the exchange really costs, with nothing else running: a transport
measurement, not a bound (the op's roofline is a pure resource bound and carries no fixed exchange cost).

## How it differs from `fabric_all_gather`

Both are ring all-gathers built the same way: a ready fence before the first send, packets that carry a fused
increment of the receiver's arrival counter, relays that forward a slot once its counter says it landed, and
counters re-armed to 0 so a trace can replay the call. What differs is where the bytes live and how many there
are:

| | `fabric_all_gather` | `fabric_ring_exchange` |
|---|---|---|
| Own data | a shard in DRAM, read into a circular buffer | a slot already in an L1 landing buffer |
| Sending | reader → circular buffer → sender, chunks of up to 7 tiles, batched | the port core sends the slot from L1 |
| Landing | the peer's output pages in DRAM; relays read them back from DRAM | the peer's L1 landing buffer (same address on every chip) |
| Own copy | copy cores write the own shard into the output | none: the own slot is already in place |
| Size | 4 tiles to 16 MiB per chip | 128 B to 32 KiB per chip |
| Answers | how fast a large exchange can go (bandwidth, link balance) | what an exchange costs when there is nothing to move (fixed cost) |

A small `fabric_all_gather` does not stand in for this example. At 8 KiB per chip on the same 1×4 ring, both
untraced and on the slowest chip, it takes 10.6 µs against the relay exchange's 7.96 µs here (1 link; see
`report.md`); the difference is the DRAM reads and writes, the circular-buffer handoff and the copy cores,
which a fused op whose data is already in L1 does not pay.

## Variants

All variants are one program per chip, one `ttnn.generic_op` for the 1×4 mesh, `FABRIC_2D_TORUS_XY`, router
payload 14 KiB + 64 B. Each chip owns one slot per link; after one call every chip holds every peer's slot.

| Variant | What runs |
|---|---|
| `empty` | an idle kernel on `idle` compute cores and on the port cores: a program of the op's size, no fabric |
| `fence` | the ports open their fabric connections, run the ready fence and send header-only items, relay schedule |
| `fence_direct` | the same on the direct schedule (a ready from every destination, including the 2-hop one) |
| `relay` | forward port: own slot to p+1, then relays p−1's on to p+2; backward port: own slot to p−1 |
| `direct` | forward port: own slot to p+1 and to p+2 (the second routed by the fabric through p+1's router); backward port: own slot to p−1 |

The port kernel is the distributed_rms_norm op's (`malimpic/2026_10_22_1342_run1_distributed_rms_norm`),
unchanged; the bench just gives it no roots to wait on or notify. Every relay / direct call checks every
chip's landing bit-exactly.

**Metric:** `DEVICE KERNEL DURATION [ns]` per chip and call. *Traced, slowest* = calls back to back from one
trace, slowest chip per call (what a traced model sees). *Untraced, last* = one call at a time, the
last-launched chip (it does not wait for its peers at the fence).

## Result (qb2-120-p02t03, Blackhole QuietBox 2, 4× p300c; see `report.md`)

Traced, slowest chip, median of 8 calls, 100 idle cores (the op's compute-core count), µs:

| Slot per link | `relay`, 1 link | `direct`, 1 link | `relay`, 2 links | `direct`, 2 links |
|---|---|---|---|---|
| 128 B | 5.38 | 6.48 | 5.39 | 6.78 |
| 2.5 KiB | 5.52 | 6.74 | 5.90 | 6.86 |
| 8 KiB | 6.06 | 7.34 | 6.52 | 7.39 |
| 32 KiB | 7.60 | 8.71 | 8.54 | 9.85 |

No-payload floor, same metric: `empty` 0.19, `fence` 5.04 (1 link) / 4.98 (2 links).

**1. The floor is the fence, about 5 µs, not the program.** An empty program of the op's size costs 0.19 µs
of device kernel time: the kernel-duration metric does not include launch, so a program's fixed start-up cost
is not where the time goes. The minimal correct exchange (connections opened, ready fence, header-only items)
costs about 5 µs, nearly all of it synchronization rather than data movement. Moving the data adds little on top: +0.3 µs
at 128 B, +1 µs at 8 KiB, +2.5 µs at 32 KiB (1 link, relay).

**2. Direct sends are slower than relay here, by 1.1–1.6 µs, and the fence is why.** On the same schedule
with header-only items, `fence_direct` costs 6.35 µs against `fence`'s 5.09 (+1.26). The data on top costs the
same either way (+0.27 relay, +0.23 direct). The direct schedule's forward port waits for a ready from its
2-hop destination before its first send; the relay's waits for one from its neighbour. The store-and-forward
turnaround the direct sends remove is cheaper than the extra fence hop they add.

**3. A second link does not help this payload.** At 128 B to 8 KiB, 2 links are equal or slower than 1
(each link carries half the bytes but pays its own fence); at 32 KiB, 2 links are slower by about 1 µs.

**Untraced, one call at a time:** the last-launched chip sees 4.3–4.7 µs (`fence` / `relay`, 128 B) while the
slowest chip sees 7–18 µs: the chips launch microseconds apart, and every chip but the last waits at the fence
for its peers. Traced calls launch together, which is why the traced slowest-chip number is the one to use.

## Reproduce

```
python -m ttnn.operations.examples.fabric_ring_exchange --bytes 128,2560 --links 1 --idle 0,100
```

Needs a 4-chip ring. The test is `tests/ttnn/unit_tests/operations/examples/test_fabric_ring_exchange.py`
(env overrides: `RX_VARIANTS`, `RX_BYTES`, `RX_LINKS`, `RX_IDLE`, `RX_UNTRACED`, `RX_TRACED`).

## Gotcha found while building it

`generic_op`'s program cache keys a kernel on its source, core ranges and compile-time args and only on the
outer size of its runtime-argument grid, not on how many arguments each core gets. Two programs that differ
only in their per-core argument layout (here relay and direct) share one cached program, and the second call
reads the first one's layout and hangs. `program_descriptor.py` sets `custom_program_hash` per variant and
size.
