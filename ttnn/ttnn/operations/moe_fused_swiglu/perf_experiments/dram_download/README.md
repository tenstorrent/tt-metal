# dram_download — the DRAM floor of moe_fused_swiglu

The mirror of `scatter_matmul`. That bench deletes all DRAM by construction and measures compute +
collective; this one deletes all compute and all collectives and measures **the download**. Between
them the op's two rooflines are bracketed by measurement instead of assumption.

Issues exactly the op's reads for one M-block, per core, at the op's own slices, through the same
`TensorAccessor`, with the op's own NoC split (reader/NOC_0 takes W_gate + x, writer/NOC_1 takes
W_up + W_down). Then it stops: no tilize, no multicast, no reduce, no matmul, no output write, no
consumer. Every read goes in flight and ONE barrier per RISC closes them — the most favourable
schedule that exists, so the number is a **ceiling on achievable rate / floor on time**.

## Results — count 256, emb 7168, 88 cores (11x8), 5 reps, spread < 1 %

| streams | placement | bytes | us | GB/s | % of 512 |
|---|---|---|---|---|---|
| weights (W_gate+W_up+W_down) | **ND shard** | 24.77 MB | **72.87** | **340** | **66.4 %** |
| weights | interleaved | 24.77 MB | 104.50 | 237 | 46.3 % |
| x only, bf16_rm | — | 0.46 MB | 1.80 | 254 | 49.5 % |
| x only, bfp8_tile | — | 0.24 MB | 1.33 | 184 | 35.9 % |
| **all (weights + x)** | **ND shard** | 25.2 MB | **72.9** | **345** | **67.4 %** |
| all | interleaved | 25.2 MB | 105.3 | 239 | 46.6 % |

Two results worth separating:

**1. 340 GB/s / 66 % of peak is what this access pattern can actually do.** That retires the "make the
requests bigger" family for good: at 1152-3456 B requests, with no compute and no collective in the
way, the op's own read shape reaches two thirds of peak. It also confirms the 60 % assumption used in
the roofline arithmetic is sound, and independently corroborates the 330 GB/s the tt-npe trace
measured during phase 1.

**2. ND sharding is worth 31.6 us on the isolated download and only ~4.7 us in the op.**
104.50 -> 72.87 us is -30 %; the same placement change measured on the real op at count 256 is
135.37 -> 130.67 us, i.e. **the op captures ~15 % of the available win.** That is not a defect in the
shard — it is the clearest possible statement that the weight read is NOT the op's critical path: 51
of the 73 us is already hidden, so making the download faster mostly buys nothing. The op has enough
DRAM-idle slack to absorb an extra 31 us of interleaved read almost for free, which is the same fact
the NoC trace reports as "50 % of the op has DRAM < 10 GB/s".

## What this does to the count-256 budget

| term | us | source |
|---|---|---|
| DRAM reads, ND-sharded | 72.9 | **this bench** |
| + output write (1.95 MB @ 340 GB/s) | ~5.7 | derived |
| **DRAM floor** | **~78.6** | |
| compute floor (4272 tile-MACs/core @ the measured 17.6 cyc) | ~55.7 | `scatter_matmul` |
| **perfectly overlapped bound = max(...)** | **~78.6** | |
| graded target | 108.0 | feature_spec |
| **measured op (bfp8_tile, sharded)** | **122.05** | |

So the op sits **43.4 us above its own DRAM floor**, and the target allows **29.4 us** of exposed
non-DRAM time. **Reaching 108 needs 14 us of that 43.4 removed — not the whole thing.** Against the
NoC trace's two DRAM-idle windows (45 us across the gate/up tail + reduce + h publish, 15 us across
the down tail + output write) that is a far smaller ask than the "phase-boundary restructure" framing
suggested, and it does not require the round count to change (which is measured to cost +12.8 %).

## Run

```bash
scripts/run_safe_pytest.sh --run-all --no-precompile \
  tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_dram_download.py
```

`mode` selects which streams participate (`weights` / `x` / `all`) so the halves can be attributed
separately; `wplace` selects interleaved vs the PERF-12 ND shard, exactly as a caller would.

## Deliberately out of scope

The x row-multicast and the h all-gather. Those are L1->L1 collective traffic, not DRAM download;
`comm_skeleton` already prices the mcast primitive, and folding them in would stop this from being a
clean DRAM number. If a combined "download + broadcast" floor is wanted it belongs here as a separate
mode, not mixed into these rows.

## Two traps this bench hit (both are CB-sizing, both silent-then-fatal)

* The x landing CB must be sized from the **slice the kernel writes**, not the accessor's page size. A
  bf16 ROW_MAJOR page is a whole `emb` stick (14 336 B) while only this core's kr-tile slice (1 792 B)
  is read; sizing from the page asked for 3.67 MB and threw on L1.
* `cb_reserve_back` count is format-specific and must match the host's page count: bf16 lands 32
  stick-slices per tile-row, bfp8 lands `kr` whole tiles. Reserving 32 in the bfp8 case asks for more
  pages than the CB holds and blocks forever — a device hang, not an error.

---

# Part 2 — the NoC split, and the collective floor

## The two NoCs are NOT equal for DRAM reads

`dl_split.py` puts one symmetric streamer on both data-movement RISCs and moves bytes between them.
ND-sharded weights, 24.77 MB, 88 cores, 5 reps:

| f (fraction of bytes on NOC_0) | us | GB/s | % of 512 |
|---|---|---|---|
| 1.00 — all NOC_0 | 61.77 | **401.0** | 78.3 % |
| 0.70 | 56.58 | 437.9 | 85.5 % |
| 0.60 | 54.11 | 457.8 | 89.4 % |
| **0.55 — optimum** | **52.76** | **469.5** | **91.7 %** |
| 0.50 | 54.91 | 451.1 | 88.1 % |
| 0.40 | 66.24 | 374.0 | 73.0 % |
| 0.00 — all NOC_1 | 109.55 | **226.1** | 44.2 % |

* **NOC_0 is 1.77x faster than NOC_1 for DRAM reads** (401.0 vs 226.1 GB/s on identical work).
* **The optimum is f = 0.55 -> 469.5 GB/s = 91.7 % of peak.** The analytic
  `f* = r0/(r0+r1) = 0.639` OVERSHOOTS: the NoCs share the DRAM controllers, so the best point sits
  between equal-bytes (0.50) and rate-proportional (0.639) and reaches 75 % of the 627 GB/s additive
  bound. The curve is asymmetric — f = 0.45 already costs 11 % — so bias toward NOC_0, not toward
  balance.
* Matrix-INTERLEAVING the requests (round-robin over the three buffers) is a null at every f
  (469.6 vs 469.5 at the optimum), so there is no bank-phase effect to exploit. Consistent with the
  tt-npe trace's 0.08 % congestion.
* Interleaved PLACEMENT caps at 259 GB/s no matter how it is split, vs 469.5 sharded.

**The op's own split is f = 0.318** (W_gate = 96 768 B on the reader; W_up + W_down = 207 360 B on the
writer) — 68 % of the bytes on the SLOW NoC. Interpolating the sweep there gives ~76 us against 52.76
available, so **the op leaves ~23 us on the table purely in NoC assignment.** Rebalancing means moving
roughly 70 % of W_up to the reader and leaving W_down on the writer. Caveat: this bench issues
everything before one barrier while the op has per-chunk barriers, so the RELATIVE result transfers but
the absolute will not fully.

## Collective (L1->L1) floor — no compute, no rendezvous

`dl_collective.py`. The collectives move **3.1x the DRAM bytes**:

| phase | L1 bytes | us | GB/s delivered |
|---|---|---|---|
| x row-multicast | 19.50 MB | 46.46 | 420 |
| column reduce-scatter | 10.03 MB | **10.29** | 975 |
| **h all-gather** | 48.46 MB | **61.33** | 790 |
| all three | 77.99 MB | **83.02** | 939 |

GB/s counts bytes DELIVERED (a multicast delivers N copies per injection), so it is not comparable to
the DRAM figures. Spreads are wide (x 43-56, h 49-67) because with no rendezvous the senders race.

**The h all-gather is the single largest term in the whole op**: 61.33 us of a 122.05 us wall, moving
48.46 MB because every core needs ALL of h to contract over hidden. The reduce-scatter, by contrast,
is nearly free (10.29 us) — which retires it as a target and confirms `scatter_matmul`'s finding that
the reduce's cost is mostly the ADDs, not the transport.

## Consolidated floors, count 256, 88 cores

| term | us | source |
|---|---|---|
| DRAM download at the optimal split | 52.8 | `dl_split` |
| collectives, no rendezvous | 83.0 | `dl_collective` |
| compute (4272 tile-MACs/core @ measured 17.6 cyc) | 55.7 | `scatter_matmul` |
| **max(...) = perfectly overlapped floor** | **83.0** | |
| all DM + collectives + the op's rendezvous, no math | 104.4 | `ABLATE=skip_compute` |
| graded target | 108.0 | feature_spec |
| **measured op** | **122.05** | |

**The binding constraint is the COLLECTIVES, not DRAM and not compute.** DRAM (52.8) and compute (55.7)
both have ~30 us of slack against the 83.0 us collective floor. And the op's own DM-with-rendezvous
number (104.4) sits 21 us above that floor, which is the rendezvous.

So the ranked levers for count 256 are now measured, not argued:
1. **Move fewer h bytes.** 61.33 us of the op is one all-gather of 48.46 MB. Halving it (bfp8 -> bfp4 h)
   is worth ~30 us — by far the largest single lever, and it costs precision (the op's documented
   PCC 0.9931), so it is a product decision.
2. **Rebalance the weight streams across the NoCs** — ~23 us on the isolated download, some fraction of
   which is exposed.
3. The reduce-scatter transport (10.29 us) and the DRAM request shape are both retired.

## CORRECTION — the first collective numbers double-counted every phase

The first version of `dl_collective` ran EVERY phase on BOTH data-movement RISCs, so each column root
multicast h twice and every figure was ~2x the real traffic. The op's actual assignment is: h and the x
row-multicast on the READER (NCRISC / NOC_0, `HSEND="reader"`), the column reduce-scatter SPLIT across
both (`SCATTER_NOC="split"`: gate on the writer, up on the reader). Re-measured with that assignment
(`OP_ASSIGN`), 5 reps:

| config | L1 bytes | us | GB/s delivered |
|---|---|---|---|
| **h only, NOC_0 (shipped)** | 48.46 MB | **38.44** | 1261 |
| h only, NOC_1 | 48.46 MB | 52.49 | 923 |
| h only, BOTH (the old bug) | 96.93 MB | 67.92 | 1427 |
| x only, NOC_0 | 19.50 MB | 34.22 | 570 |
| reduce, split | 20.05 MB | **10.21** | 1965 |
| **all three, op assignment** | 88.01 MB | **71.95** | 1223 |

So the h all-gather is **38.44 us**, not the 61.33 us reported above — still the largest single
collective, but a third smaller. (The `reduce` row is itself 2x over-counted: this kernel sends both
gate and up on each RISC that has the phase enabled, while the op sends gate on one NoC and up on the
other. It is 10.21 us either way and not a bottleneck, so it was left alone.)

**NOC_0 is 1.37x better than NOC_1 for the h multicast** (38.44 vs 52.49 us), which independently
confirms `HSEND="reader"` is the right default and matches the direction of the op's own +5-7 % null
for `HSEND=writer`.

### This also corrects the changelog's rendezvous model

Perf 15 / addendum 2 fitted `round period = 3.12 us FIXED + 0.147 us per m-tile` and attributed the
3.12 us fixed term to RENDEZVOUS, concluding "34.3 us per M-block of pure rendezvous, independent of M".
That attribution is wrong. This bench runs the same 11 multicasts with NO rendezvous of any kind and
measures 38.44 us = **~3.5 us per round** — i.e. essentially the whole "fixed" term is the MULTICAST
TRANSPORT itself, not a handshake. Against the op's measured 4.3 us/round cadence that leaves only
~0.8 us/round (~9 us per M-block) of actual rendezvous, not 34.3.

Consequence: the h all-gather is **mostly irreducible transport for the bytes it moves**, so the lever
is bytes or topology — not a better handshake. That is the same conclusion addendum 2 reached ("move
fewer h bytes") but for the opposite reason, and it explains why every rendezvous-shortening experiment
in this changelog (per-slot flags, ack-ahead, DEPTH_H, off-loop sender) measured small or null: there
was only ~9 us there to win, not 34.

### Revised consolidated floors, count 256, 88 cores

| term | us |
|---|---|
| DRAM download at the optimal NoC split | 52.8 |
| **collectives, op assignment, no rendezvous** | **72.0** |
| compute (4272 tile-MACs/core @ 17.6 cyc) | 55.7 |
| **max(...) = perfectly overlapped floor** | **72.0** |
| op DM + collectives + rendezvous, no math | 104.4 |
| graded target | 108.0 |
| measured op | 122.05 |

Collectives remain the binding term, and the h all-gather is 38.4 of that 72.0. Noise caveat: the
multicast phases spread widely without a rendezvous (h 35.6-65.5, x 23.7-84.1) because senders race;
the reduce phase is tight (10.15-10.34).

## POSTED multicast — worth ~2x on ONE round, and it does not survive contention

`ncrisc_noc_fast_write_any_len` takes a `posted` flag that the public `noc_async_write_multicast`
wrapper does not plumb through; it IS forwarded to every packet of a multi-packet transfer, so it
applies to the 52 224 B h block. With it the round closes on `noc_async_writes_flushed()` (SENT)
instead of `noc_async_write_barrier()` (LANDED), skipping the (NUM_CORES-1)-way ack incast.

The all-roots figure is useless for this question — 11 senders racing spreads it 27-75 us, which swamps
the effect. Restricting to ONE root (a single 52 KB whole-grid multicast) makes it decisive, 15 reps:

| case | non-posted | posted | delta |
|---|---|---|---|
| **h, 1 root** (spread 2.05-2.08 / 1.04-1.09) | **2.06 us** | **1.07 us** | **-47.8 %** |
| h, 11 roots (median) | 50.17 | 40.36 | -19.6 % |
| h, 11 roots (min) | 31.69 | 30.85 | -2.6 % |
| all phases (median) | 68.75 | 75.61 | **+10.0 %** |
| all phases (min) | 53.00 | 64.28 | **+21.3 %** |

**The ack incast is ~1.0 us per round — about half the cost of an uncontended round.** That is a real,
tight, reproducible result.

**But the win reverses under load.** With all three collectives running, posted is 10-21 % WORSE. The
reading: posted removes the natural back-pressure, so senders overrun and the bottleneck moves from ack
LATENCY to NoC injection/contention. Posted helps a latency-bound round and hurts a bandwidth-bound one.

### What that means for the op

The op's h rounds ARE serialized (round r+1's sender waits on round r), so each one pays the full ack
latency — the regime where posted wins. Against the op's measured 4.3 us/round cadence:

| term | us/round |
|---|---|
| uncontended transport + ack (measured, 1 root) | 2.06 |
| of which the ack incast | ~1.0 |
| op's measured cadence | 4.30 |
| residual (semaphore waits, CB reserve, loop) | ~2.2 |

So posting the h multicast is worth **~1 us x 11 rounds = ~11 us per M-block** if it can be made
correct — close to the 14 us count 256 needs. This also PARTIALLY WALKS BACK the correction above: the
non-transport share of a round is ~2.2 us, not the ~0.8 us that the 11-sender figure implied and not
the 3.12 us the changelog fitted. The truth is in between.

**The blocker is correctness, and it is the same one that has bitten this op before.** Posted writes
give no landing guarantee and `flushed` means "left this core", so the receiver cannot know h arrived —
exactly the data-before-signal hazard that made `HSIG=Counter` hang. Capturing the ~11 us needs an
ordering mechanism that posted writes cannot themselves provide (e.g. a trailing non-posted write to
the same destinations on the same VC, IF the NoC guarantees that ordering — unverified).

## Realistic DM recreation — and a CLOSED accounting of the op's 122 us

`dl_realistic.py` keeps the same traffic and slices but restores the op's dependency ORDER: per-GU-chunk
weight barriers, x stage -> mcast ordering, the reduce's two phases, a barrier per h round, W_down
prefetched one K-block per round, and the output write. STAGE 0 collapses it back to one trailing
barrier so the ordering's cost is isolated. No compute in either.

| stage | med us | min | max |
|---|---|---|---|
| 0 — one trailing barrier | **94.78** | 89.05 | 116.67 |
| 1 — barriers where actually needed | **103.37** | 94.94 | 113.65 |

**Stage 1 lands 1.0 us (0.9 %) from the op's own `ABLATE=skip_compute` = 104.35 us**, which is the same
DM plus the real semaphore rendezvous. That agreement is the validation: this recreation reproduces the
op's data movement almost exactly.

### The full ladder, count 256, 88 cores — every term measured

| term | us | increment |
|---|---|---|
| DRAM download alone, optimal NoC split | 52.8 | |
| collectives alone, op assignment | 72.0 | |
| all DM, one barrier | 94.78 | |
| + the op's dependency ordering | 103.37 | **+8.6** |
| + the semaphore rendezvous (`skip_compute`) | 104.35 | **+1.0** |
| + compute | 122.05 | **+17.7** |
| graded target | 108.0 | |

**Two conclusions, both of which overturn earlier claims in this changelog.**

1. **The semaphore rendezvous costs ~1 us, not 34.3 us.** Perf 15 / addendum 2 fitted a 3.12 us "fixed"
   per-round term and called it rendezvous, i.e. 34.3 us per M-block. Building the same DM with the
   rendezvous ABSENT (stage 1, 103.37) and comparing against the op WITH it (104.35) prices it at ~1 us
   total. The 3.12 us fixed term is the multicast transport plus the dependency barriers, not a
   handshake. This finally explains why per-slot flags, ack-ahead, DEPTH_H and the off-loop sender all
   measured null or tiny: there was ~1 us there, not 34.
2. **The op is DM-BOUND at count 256, and the DM is already 104.35 us against a 108 us target.** That
   leaves only 3.6 us of headroom for EXPOSED compute, and compute currently adds 17.7 us. So the target
   needs compute almost perfectly hidden — or the DM itself made cheaper.

### Ranked DM levers, now that the accounting is closed

| lever | size | note |
|---|---|---|
| fewer h bytes (bfp8 -> bfp4 h) | ~19 us | halves the 48.46 MB all-gather; costs the PCC 0.9931 baseline |
| rebalance the weight NoC split (f 0.318 -> 0.55) | ~23 us on the isolated download | much of it already hidden; unknown exposed share |
| relax the dependency barriers | up to 8.6 us | stage 0 vs stage 1; each barrier would need its own correctness argument |
| posted h multicast | ~11 us | ~1 us/round x 11; blocked on a landing guarantee |
| the semaphore rendezvous | ~1 us | RETIRED — measured, not worth attacking |

Noise caveat: the multicast-heavy phases spread widely (stage 0 89-117, stage 1 95-114), so the +8.6 us
ordering cost is directional rather than precise; the mins (89.05 -> 94.94) agree at +6.6 %.

### What stage 2 would add

The real per-round rendezvous (round r+1's sender waiting on every receiver clearing round r), which is
the only structural piece still missing. Given the ~1 us measured above it should be small, but it is
the piece that would make this recreation a drop-in predictor for schedule changes.
