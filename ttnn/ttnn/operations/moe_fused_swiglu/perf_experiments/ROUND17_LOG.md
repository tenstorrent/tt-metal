# Perf round 17 — implementing every knob in `NEXT_ROUND_PLAN.md`

One entry per knob: what was built, what was measured, and *why* it worked or did not.
A knob is only dropped after a measurement that explains the drop.

## Measurement contract (fixed for the whole round)

* Box/arch: `blackhole_p150`, grid `MOE_SWIGLU_GRID=11x8` (88 cores).
* Weights: **ND-sharded** (`MOE_SWEEP_WPLACE=nd_shard`) — the placement the op is designed for.
* emb 7168, capacity 5120, counts 128 / 256 / 512, `MOE_SWEEP_WARMUP=2`, `KNOB_REPS=5`, median of reps.
* Harness: `perf_experiments/knob_search.sh` (seqlen-sweep test, `--profile --no-precompile`).
* Correctness gate: `eval/golden_tests/moe_fused_swiglu/test_golden.py` **run with the candidate env**,
  plus the unit PCC cells the sweep harness itself asserts.
* Everything is quoted in **us, median of 5 reps**. This op's per-cell run-to-run band is ~1 %.

## Baselines (this session, shipped defaults)

| build | 128 | 256 | 512 | sum |
|---|---|---|---|---|
| bf16_rm  | 91.06 | 128.75 | 225.82 | 445.63 |
| bfp8_tile | 88.46 | 122.64 | 215.78 | 426.87 |

Matches `NEXT_ROUND_PLAN.md` §1 (91.00 / 128.43 / 226.00 and 87.66 / 122.05 / 214.91) to within the
noise band, so the tree is in the state the plan describes.

**Session drift is real and it is ~1 %.** Five separate `wd0` baseline sessions measured
91.06 / 91.26 / 90.81 / 90.91 at count 128 and 128.75 / 128.83 / 128.07 / 128.75 at count 256. Every
A/B below is therefore quoted against a `wd0`-equivalent baseline measured **in the same session**,
never across sessions.

---

## Knob #1 + #3 — `MOE_SWIGLU_WD_SPLIT`: the W_down NoC/RISC assignment — **SHIPPED at 3**

The plan lists these as two optimisations; they are two *effects of one lever* (its own §2/#3 review
correction says so), so they were built as one knob and swept as one curve.

### What was built

`MOE_SWIGLU_WD_SPLIT = 0..8` — eighths of every phase-2 W_down K-block's **hidden rows** read by the
**writer on NOC_1** instead of the reader on NOC_0. The writer takes the TAIL rows, so both sides
read a contiguous run and the bank-run coalescing is unchanged on either side. 0 is the shipped
all-reader stream byte for byte; 8 is the wholesale move.

Three implementation facts that made it cheap, all of which turn on things the op already had:

1. **The one-producer rule is preserved without any new CB.** Both RISC-Vs *read into* `cb_w_down`,
   but only the reader ever calls `cb_reserve_back` / `cb_push_back`. The writer needs no CB state
   because the destination is derivable: `WD_RESIDENT=1` forces `depth_wd == HGROUPS`, so K-block `r`
   permanently occupies slot `r` at `base + r*HN_PAD*EC_MAX*BFP4_TILE` off the base captured before
   any push — the identical derivation `HSEND=writer` already uses for `cb_h_base`.
2. **No flow control is needed at all.** Under `WD_RESIDENT` every W_down DRAM read happens at
   `b == 0`, where all HGROUPS slots are free from kernel start. The host forces the knob off if
   `WD_RESIDENT=0` rather than shipping a silently-wrong multi-M-block path.
3. **One cross-RISC completion signal IS mandatory**, because `noc_async_read_barrier()` is
   per-RISC-V. `SEM_WDSPLIT` is an intra-core L1 word (no NoC traffic), monotone in K-blocks
   completed, same shape as `SEM_XSTAGED`.

Everything the writer needed (`CB_W_DOWN_ID`, `WD_RESIDENT_D`, `SEM_WDSPLIT`, `WD_SPLIT`) went in as
**defines**, per §4 trap 6; the W_down accessor and its base address were **appended last** to the
writer's CT/RT lists so no existing index or `TA_BASE` moved.

### Measured — the curve is a real interior optimum

bf16_rm, 88 cores, ND-sharded, 5 reps, us, all four columns from ONE session:

| count | s=0 | s=2 | s=4 | s=6 | s=8 |
|---|---|---|---|---|---|
| 128 | 91.26 | 88.25 | 89.23 | 93.84 | **103.20** |
| 256 | 128.83 | 126.80 | 124.41 | 124.02 | 125.58 |
| 512 | 225.76 | 224.34 | 221.97 | 221.59 | 222.54 |

and the shipped point, both formats, each against its own in-session baseline:

| | 128 | 256 | 512 |
|---|---|---|---|
| bf16_rm  s=0 | 90.91 | 128.75 | 225.95 |
| bf16_rm  **s=3** | **86.66** (-4.7 %) | **125.38** (-2.6 %) | **223.42** (-1.1 %) |
| bfp8_tile s=0 | 88.46 | 122.64 | 215.78 |
| bfp8_tile **s=3** | **87.54** (-1.0 %) | **117.99** (-3.8 %) | **212.05** (-1.7 %) |

Golden **45/45** at s=4 and at s=6. Correctness was gated *before* any timing was believed (§4 trap 2).

### Why it works, and why the optimum is interior

`f` = fraction of whole-op weight bytes on NOC_0 = `(96 768 + (1 - s/8)·110 592) / 304 128`:

| s | 0 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|
| f | 0.682 | 0.621 | **0.561** | 0.500 | 0.318 |

The isolated concurrent-stream sweep's optimum is **f = 0.55** (469.5 GB/s, 91.7 % of peak).
**s = 3 is f = 0.561 — the closest point on the grid, and it is what the op picked.** That is the
plan's §3 prediction confirmed end-to-end, and it also confirms the plan's own correction that the
direction was to move *a little* off NOC_0, not to swap.

The two ends of the count range pull in opposite directions, which is the whole shape of the curve:

* **count 128 is weight-DOWNLOAD bound** (its fixed floor of ~64 us is weights only), so it wants the
  global `f` balanced and punishes overshoot hard — s=8 is **+13 %**, reproducing `moe_matmul`'s own
  8/8 role-swap regression (0.81x) on its isolated bench.
* **counts 256/512 have a long phase 2**, where the marginal value is de-loading NOC_0's *request
  path* during the h all-gather, so they keep improving out to s=6.

s=3 wins count 128 outright without giving up phase 2, and keeps 128 well inside its 91 800 ns target.

### Two sub-knobs measured on the way — one null, one a clear negative

**`MOE_SWIGLU_WD_WPUB=trid|batch` (publish granularity) — a NULL.** Hypothesis: the count-128
regression at high `s` is the reader blocking in `wd_split_gate` on a whole-batch barrier. Built one
transaction id per K-block (`noc_async_read_barrier_with_trid`, drained in block order, publishing as
it goes) so the reader is released block by block instead of after the whole 111 KB stream.
Measured at s=6: `trid` 93.49 / 125.83 / 223.41 vs `batch` 93.89 / 123.90 / 222.73 — inside noise, and
the s=8 count-128 penalty is untouched (104.02 vs 103.20). **So the reader was never waiting on that
gate**, which is what redirected the diagnosis to the `f` story above. `trid` is kept as the default
anyway: it is the strictly tighter gate, it costs nothing measurable, and it is what makes the
`WD_RESIDENT=0` path safe.

**`MOE_SWIGLU_WD_WPLACE=wup|scatter` (where the writer issues) — `scatter` is a clear negative.**
Issuing past the column all-to-all (the writer twin of `WD_LATE`) measured, at s=6/s=8:
99.46 / 141.22 / 239.36 and 110.60 / 152.23 / 251.46 — **+10 to +13 % at count 256** against the `wup`
placement. The reads then land immediately before phase 2 needs them instead of inside the 45 us
DRAM-idle window, so their latency is fully exposed. `wup` (straight after the W_up stream) ships.

---

## Knob #2 — `MOE_SWIGLU_HSPLIT`: dual-NoC h multicast — **BUILT, CORRECT, DROPPED (explained)**

### What was built

`MOE_SWIGLU_HSPLIT = 0..7` — eighths of every h all-gather round's **tiles** broadcast by the writer
on NOC_1, reader keeps the head tiles on NOC_0. `cb_h` keeps exactly one producer (the reader); the
writer does raw NoC writes into the region the reader already reserved and derives its landing slot
from `cb_h`'s base, the same way `HSEND=writer` does. Each RISC-V owns its own completion signal:

* reader — head tiles + the existing per-slot VALID flag, still **linked**, still no barrier;
* writer — tail tiles + an **acked** `noc_async_write_barrier()` + a monotone per-slot arrival
  counter (`SEM_H2_RDY_BASE`).

Golden **45/45** at HSPLIT=3. It is correct; it is just not faster.

### Measured — flat at the predicted optimum, monotonically worse beyond it

bf16_rm, us, each block from one session:

| count | s=0 | s=2 | s=3 | s=4 | | s=0 | s=6 |
|---|---|---|---|---|---|---|---|
| 128 | 87.50 | 88.60 | 87.88 | 89.14 | | 87.17 | 88.63 |
| 256 | 125.46 | 126.95 | 125.35 | 124.19 | | 125.24 | **131.98** |
| 512 | 223.01 | 223.49 | 226.34 | 226.81 | | 223.18 | **237.50** |

### Why — the second NoC obligates a signal the first one gets for free

The transport saving is real but bounded, and the signalling cost that buys it is not:

1. **The saving is small.** Perf 15 addendum 2 established that phase 2's rounds are a *pipeline whose
   period is set by bytes and FLOPs*, ~2.85-3.46 us/round of which h ingest is ~1.2 us. Moving 3/8 of
   the bytes to a path measured **1.37x slower** in isolation (52.49 vs 38.44 us) yields well under
   0.45 us/round even before any overhead.
2. **The free signal is unavailable to an off-loop sender.** mcast_pipe's Flag gets its ordering for
   nothing — data and VALID ride one `NOC_CMD_VC_LINKED` chain. That guarantee is **per-NoC**, so the
   writer's half needs its own. The linked-flag version of it is not merely risky, it is *already
   measured to hang*: Perf 15 addendum built exactly "off-loop sender + linked flag" and got a
   dispatch timeout on the documented cross-RISC reset race, and recorded that **per-slot flags do
   not immunise it** (cell `s` is shared by rounds `s`, `s+DEPTH_H`, ...). So the writer's half is
   forced onto the Counter shape: an acked `noc_async_write_barrier()` — a NUM_CORES-way ack incast —
   plus a non-posted atomic multicast, **per round**. That is precisely the pair the changelog
   already priced at **+11 % end to end** when it carried the whole payload.
3. **Which is why the curve degrades monotonically.** As `s` grows, the slower NOC_1 path becomes the
   round's critical half while the fixed acked-barrier cost stays: +5.4 % at 256 and +6.4 % at 512 by
   s=6.

**Default 0.** The knob is kept (off) because it is the measurement, and because #4 below reuses its
"is the h signal path the real cost?" framing — with the opposite answer.

---

## Knob #4 — `MOE_SWIGLU_HPOSTED`: POSTED h multicast — **SHIPPED at 1**

### The correctness gate first, because the plan's open item was exactly that

The plan's blocker was that posted `flushed` gives a receiver no point at which consumption is safe,
and that a post-hoc checksum is not a sufficient gate. The resolution is that **only the data needs
to be posted**: the VALID flag stays NON-POSTED and stays LINKED, so the payload is issued with
`NOC_CMD_VC_LINKED` on `NOC_MULTICAST_WRITE_VC` and the flag write terminates that link on the same
VC. The flag therefore cannot overtake the payload on the wire.

That is **not a new assumption**. It is the invariant the shipped Flag path already depends on
(`mcast_pipe.inl::send_data_`: *"LINKED ONLY FOR THE Flag SIGNAL. The link is terminated by the
following signal mcast"*). Posting the data changes only whether **acks return**, never wire order.
So the receiver's existing `wait(VALID)` remains a true arrival proof — and the golden suite, in
which the `down` matmul actually **consumes** every h byte, is a genuine consumption test.
**Golden 45/45 on the knob.** (Compile flags checked for `-DHPOSTED=1` so the path was really taken.)

Written raw in the reader rather than through mcast_pipe, because `noc.h` blocks posted multicast at
the library level (`static_assert(... "Mcasts with posted transactions are not supported")`, marked
TODO). The raw sequence is step-for-step `SenderPipe::send()` on the Flag path with one bit changed.

### Measured — a win in both formats, and it grows with count

| | 128 | 256 | 512 |
|---|---|---|---|
| bf16_rm  off | 87.47 | 125.13 | 223.76 |
| bf16_rm  **on** | **86.45** (-1.2 %) | **123.87** (-1.0 %) | **219.97** (-1.7 %) |
| bfp8_tile off | 86.12 | 118.60 | 211.97 |
| bfp8_tile **on** | **85.90** (-0.3 %) | **117.19** (-1.2 %) | **207.77** (-2.0 %) |

### Why the isolated bench mispredicted this

The plan's strongest prior was the isolated bench's **+10 % "under load"**, which predicted a LOSS.
It does not transfer, and the reason is a property of that bench: it closed with
`noc_async_writes_flushed()` and **nothing consumed the data**, so its senders genuinely could
overrun once back-pressure was removed. In the real op the linked non-posted flag still throttles the
sender to wire order, and `cb_h`'s reserve/ack window (`HACK_AHEAD`) still bounds how far ahead any
sender may run. **The back-pressure that mattered was never the write acks.** Consistent with that,
the gain scales with rounds x destinations — largest at count 512, the weak cell.

### Two follow-ups measured on the back of it

* **`HACK_AHEAD=3` (was 2) — a null.** Removing the acks changes the round pipeline, so the ack-window
  knob was re-swept: 87.13 / 124.38 / 219.61 against 86.96 / 123.33 / 220.03. Inside noise, and worse
  at 256. Stays at 2.
* **`DEPTH_H=4` — L1-dead, and fixing a self-inflicted bug to prove it.** The `SEM_H2_RDY_BASE` block
  added for HSPLIT (DEPTH_H cells) put the default build at exactly the device's 16 semaphores, so
  `DEPTH_H=4` failed with *"Semaphore id 16 exceeds max value 15"* — i.e. this round had silently
  taken away a pre-existing sweepable knob. Those cells are now allocated **only when HSPLIT is on**.
  With the knob reachable again, `DEPTH_H=4` fails for the real reason: *"circular buffers grow to
  1 614 528 B, beyond max L1 size of 1 572 864 B"*. L1, not semaphores.

---

## Knob #6 — `MOE_SWIGLU_REDUCE_MECH`: the slice-fold accumulate — **`dest_acc` SHIPPED**

§7 proposed `pack_l1_pair` and priced it at ~3 us; the changelog's Perf 7 had retracted the whole
family at "at most 712 ns". **Both were wrong, in opposite directions**, and the knob settles it.

### What was built

The reduce-scatter's per-worker fold is `copy` + (KGROUPS-1) in-place `add`s. Three mechanisms:

| value | what changes | packs per fold | inits per fold |
|---|---|---|---|
| `addchain` | the shipped fold | `nc` | `nc-1` |
| `hoist` | identical math and CB traffic; first add owns the setup, rest run `SetupOwner::Caller` | `nc` | 1 |
| **`dest_acc`** | running sum stays in a **sticky DEST** tile via `add_tiles_init(IN, IN, acc_to_dest=true)`, so one `add_tiles` folds TWO contributors (`DEST += A + B`) with no repack between | **1** | 2 |

`dest_acc` costs **no extra L1** — DEST is the accumulator. That is what separates it from
`pack_l1_pair`, which needed a bf16 accumulator CB (+102 KB against 10 560 B free at 11x8) and was
therefore L1-dead; it also cannot hit the packer-L1-accumulate-on-a-block-float correctness bug,
because it never L1-accumulates.

### Measured

| | 128 | 256 | 512 |
|---|---|---|---|
| bf16_rm `addchain` | 87.15 | 123.48 | 219.40 |
| bf16_rm `hoist` | 87.06 | 123.87 | 219.52 |
| bf16_rm **`dest_acc`** | **84.44** (-3.1 %) | **120.23** (-2.6 %) | **211.68** (-3.5 %) |
| bfp8_tile `addchain` | 86.33 | 116.78 | 207.90 |
| bfp8_tile **`dest_acc`** | **84.34** (-2.3 %) | **112.06** (-4.0 %) | **200.14** (-3.7 %) |

Golden **45/45** on `hoist` and on `dest_acc`.

**A second pass on the same lever, worth another ~0.5-0.7 us per cell.** The first working build kept
the LAST contributor on the ordinary `add<>` helper, on the theory that ending on a real chain was
what handed the reconfig state back. It is not — once `fold_dest` sets its own formats *before* its
own inits, the following helper chains reconfigure themselves, so that trailing call was a pure extra
pack + unpack pass. Folding all `nc` contributors in DEST instead: 85.20/120.72/212.35 ->
84.44/120.23/211.68 (bf16). Golden 45/45 again.

### Why — and `hoist` being flat is the informative half

`hoist` removes the redundant per-call init/reconfig **and nothing else**, and it measures **null**.
So the fold's cost is *not* the setup. `dest_acc` removes those same setups **plus `nc-1` packs and
`nc-1` accumulator re-reads**, and it is worth 2.7-4.0 %. **The accumulator's L1 traffic is the whole
term** — which is exactly the ranking `examples/eltwise_l1_vs_dest_accumulate` isolates.

This also explains the two bad priors. Perf 7 bounded the family at 712 ns using `skip_eltwise`,
which elides the eltwise **math** but leaves every **pack** and CB round-trip intact — so it never
priced this mechanism at all. And §7's `pack_l1_pair` estimate was in the right range but attached to
the one implementation that could not fit in L1.

### The bug that cost the most time, and its general lesson

The first three `dest_acc` builds produced `inf` at **pcc = 1.000000** — the right pattern at the
wrong scale. Bisecting showed it survived with the accumulate *entirely removed* (a bare raw
`copy_tile` + `pack_tile` in place of the helper's `copy<>` still produced `inf`), so it was never the
accumulation.

**Cause: reconfig must come before init, and both sides need it.** `copy_tile_to_dst_init_short` /
`add_tiles_init` set the math MOP; they do **not** set the unpacker's data-format registers, which
still held the gate/up matmul's operands. The fix is two lines placed *before* the inits:

```cpp
reconfig_data_format(IN, IN);      // srcA/srcB  <- the landing CB
pack_reconfig_data_format(ACC);    // packer     -> the accumulator CB
```

plus a matching `reconfig_data_format(ACC, IN)` on exit, because the following `eltwise_chain`
**compile-time-elides its own reconfig** against a static CB sequence that a raw block is invisible
to — so the hardware state has to be right whether or not the chain re-emits.

**This is the same failure class as §5's bfp4-h**, and it is now much better characterised: the
minimal reproducer is not a dtype change at all, it is *any* raw compute block dropped into this
kernel between helper chains. Anyone finishing #5 should look for a missing `reconfig_data_format`
before an init, not for a missing dtype.

---

## Knob #5 — `MOE_SWIGLU_H_DTYPE=bfp4` — **still broken; two corrections to the plan's diagnosis**

Re-run with the round's changes in place. **Still wrong**, and the signature has moved:
`pcc=1.000000 rms=inf max_abs=inf` on every cell — no longer the plan's recorded `pcc=nan`. That is
now recognisable: it is *character-for-character* the signature the `dest_acc` bug produced above,
i.e. **the right values read through the wrong data format**, not a numerical blow-up.

Two corrections to §5's recorded diagnosis, both from reading the helper source:

1. **The stated reason for its fix #2 is wrong.** §5 says the `down` matmul needed a manual
   `reconfig_data_format(cb_w_down, cb_h)` because "`InitMode::Short` deliberately SKIPS format
   reconfiguration". It does not. `matmul_block_helpers.inl:165` states it outright — *"Reconfig and
   init are independent compile-time gates"* — and the reconfig is controlled by the separate
   `DataFormatReconfig` parameter, which the op leaves at its default. So `InitMode::Short` never
   suppressed that reconfig and fix #2 was, at best, redundant. Whatever the remaining site is, it is
   not that one.
2. **The right lens is `reconfig_data_format` before `*_init`, on BOTH the unpack and pack side** —
   the rule the `dest_acc` bug above established the hard way. The two sites §5 already fixed are
   both single-sided.

**Not pursued further, and the reason is unchanged and independent of the bug:** bfp4 h stacks a
second block-float rounding on a contraction whose weights are *already* bfp4, so the knob cannot
ship on precision whatever the plumbing does; and `no_h_xfer` already gives an EXACT upper bound on
the prize (15.16 / 16.29 / 27.77 us) where bfp4 could only ever be a point estimate below it. The
branch stays **default-off and documented as known-broken**, which is the "isolate" half of §5's own
"finish it or isolate it" requirement.

---

## Round scoreboard

All numbers us, 88 cores, ND-sharded, 5 reps, median. Start-of-round vs shipped defaults now.

| | 128 | 256 | 512 |
|---|---|---|---|
| bf16_rm, round start | 91.06 | 128.75 | 225.82 |
| bf16_rm, **shipped now** | **84.44** | **120.23** | **211.68** |
| | **-7.3 %** | **-6.6 %** | **-6.3 %** |
| bfp8_tile, round start | 88.46 | 122.64 | 215.78 |
| bfp8_tile, **shipped now** | **84.34** | **112.06** | **200.14** |
| | **-4.7 %** | **-8.6 %** | **-7.2 %** |

Golden **45/45** on the default path and on every shipped knob individually.

Three defaults changed, each independently measured and golden-gated:
`MOE_SWIGLU_WD_SPLIT=3`, `MOE_SWIGLU_HPOSTED=1`, `MOE_SWIGLU_REDUCE_MECH=dest_acc`.

Against the graded targets (`PERF_TARGET_NS`): **count 128 MET with margin** (84.44 / 84.34 vs
91 800). **Count 256 is now 4.06 us short in bfp8** — 112.06 vs 108.00, against 14.05 short at the
start of the round. Count 512 remains the weak cell: 200.14 vs 161.816, i.e. 38.3 us short against
53.1 at the start.

### The three wins do not share a mechanism, which is why they added

* **WD_SPLIT** — moves weight BYTES off NOC_0's request path (a DM lever, phase 1 + phase 2).
* **HPOSTED** — removes ACK traffic from the h multicast (a DM lever, phase 2 only).
* **dest_acc** — removes accumulator L1 traffic in the epilogue (a COMPUTE lever, phase 1 only).

Measured end-to-end at every step against an in-session baseline, so the sum is a measured wall
reduction and not an addition of isolated estimates — the failure mode §1 of the plan warns about.

---

## §8 — count 512: the knobs are exhausted, and cross-block DM pipelining has a NEW blocker

### Re-tune after the round's three wins — all flat or worse

The optima could have moved once WD_SPLIT/HPOSTED/dest_acc landed. They did not. bfp8, count 512
only, 5 reps, one session:

| | base | `WD_SPLIT=4` | `WD_SPLIT=6` | `WD_AHEAD=2` | `GU_CHUNKS=2` |
|---|---|---|---|---|---|
| 512 | **199.54** | 200.05 | 199.07 | 200.33 | 207.03 |

`WD_SPLIT=6` is -0.2 % (inside the band), everything else flat or worse; `GU_CHUNKS=2` is +3.8 %,
re-confirming 3. **There is no knob-turn left at 512** — the remaining 38 us is structural, exactly as
§8 says.

### The structural idea, and the blocker that has to be cleared first

§8's named next step is to software-pipeline the M-block: at 512 `m_blocks = 2`, so block b+1's data
movement should be able to run during block b's phase-2 stalls. The most tractable slice of that is
**prefetching block b+1's `x` DRAM read** (exposed 14.29 us per block, and on the `bfp8_tile`
front-end it needs NO compute participation — tiles land straight in the resident slot, unlike the
bf16 path which needs the compute tilize). `DEPTH_X = 2` already provides the second landing slot,
and `cb_push_back` — not `cb_reserve_back` — is what advances the write pointer, so block b+1's slot
address is available during block b at no cost.

**It cannot work as-is, and the reason is a trap this codebase has already hit twice.** Phase 2 is
full of blanket `noc_async_read_barrier()` calls (the round's W_down drain, the sender's self-copy
drain). A read barrier is ALL-OR-NOTHING and — per the reader's own PERF 8 note — **drains
trid-tagged reads too**. So any x prefetch issued before those barriers has its full DRAM latency
paid inside phase 2 instead of hidden by it, and one issued after them has nothing left to hide
under. This is the identical defect class as the `XSTAGE_FIRST` and PERF-8 findings.

**So the prerequisite is not the prefetch, it is scoping phase 2's barriers**: the per-round W_down
drain must become `noc_async_read_barrier_with_trid` on a W_down-only trid before any other stream
can be in flight across it. That is a precondition worth recording, because it is invisible from the
phase timeline §8 asks for — the timeline shows the *stall*, not the fact that the stall is
un-fillable while the barriers stay blanket.

---

---

## §14.2 — `MOE_SWIGLU_SCATTER_ROT`: rotate the reduce-scatter peer-loop start — **NULL, on BOTH metrics**

### What was built

Core in row `r` starts its column peer walk at index `r` and wraps, instead of every core starting at
peer 0 simultaneously. Applied to all four walks: the writer's gate-gather writes, its `SEM_DATA`
signals, the reader's column INVITE fan-out, and the reader's `SCATTER_NOC_SPLIT` up-half.
Order-only — same destinations, same per-peer source offsets, same transaction counts — and legal
because every far-side wait is a MONOTONE counter that cannot observe arrival order.
Golden **45/45**.

### Measured — e2e AND tt-npe, which is the point

bfp8_tile, 88 cores, ND-sharded, 5 reps, one session:

| | 128 | 256 | 512 |
|---|---|---|---|
| rot=0 | 83.97 | **113.21** | **199.90** |
| rot=1 | 84.58 | 113.22 | 199.93 |

Dead flat at 256 and 512 (0.01 / 0.03 us apart); count 128's +0.7 % is inside the band.

tt-npe on matched traces (count 256, bf16_rm, `--cong fast`, `--device blackhole`):

| | rot=0 | rot=1 |
|---|---|---|
| max link demand | 319.3 % | **321.6 %** |
| avg link demand | 26.6 % | 26.6 % |
| max link util | 33.6 % | 32.6 % |
| congestion impact | 0.0 % | 0.0 % |
| golden cycles | 178 966 | 178 700 |
| DRAM BW util | 59.7 % | 59.7 % |

### Why — the hypothesis was wrong about what the hotspot IS

The premise was that the 302-320 % max-link-demand figure is caused by all contributors hitting peer
0 first. **It is not, and the rotation proves it: the hotspot does not move** (319.3 -> 321.6 %, i.e.
unchanged). The reason is that link demand is an AGGREGATE over the trace window, and rotating the
order within each core does not change which physical links carry which bytes — the column
all-to-all is a fixed set of core-pairs inside one grid column, so the same vertical links carry the
same total traffic whichever order the eight senders enumerate them in. All eight still issue all
eight writes inside the same short window.

And the second half of the result matters as much: **congestion impact is 0.0 % in both**, so the
hotspot was never costing anything to begin with. A >100 % max-link-demand reading with 0 %
congestion impact means the demand is spread across the window and never actually stalls a transfer
— it is a demand metric, not a realised cost. That closes the "302.6 % hotspot" line of attack
generally, not just this knob.

**Default 0.** Knob retained (off) as the measurement.

Incidental, worth recording: the same trace shows the op at **178 966 golden cycles vs the 194 111**
the pre-round trace recorded, and DRAM BW util up from 56.1 % to **59.7 %** — an independent
confirmation of this round's wins from a completely different instrument.

---

## §14.1 — `MOE_SWIGLU_SILU_FUSE`: gate SiLU inside the DEST accumulation — **NULL (the trade cancels)**

### What was built

`fold_dest_silu`: all KGROUPS gate contributors folded in DEST (the shipped `dest_acc` mechanism,
extended one step), SiLU applied **in DEST** with `silu_tile()`, then ONE `pack_tile` straight to
`cb_gate_silu`. `cb_slice_gate` is not touched at all on this path — its pack, its unpack and the
separate `add_bias_bcast_rows` pass all disappear. Golden **45/45**.

Deliberately NOT fused further: keeping the SiLU result in DEST for the SwiGLU multiply needs a
DEST x L1 product (`binary_dest_reuse_tiles`), which is the slow path and which the kernel already
carries a measurement against.

### Measured — a wash in both formats, with the sign flipping between sessions

| | 128 | 256 | 512 | sum |
|---|---|---|---|---|
| bfp8 off | 83.56 | 113.71 | 199.67 | 396.94 |
| bfp8 **on** | 84.19 | 113.01 | 199.25 | 396.45 |
| bf16 off | 84.70 | 119.43 | 211.68 | 415.81 |
| bf16 **on** | 83.97 | 119.90 | 211.26 | 415.13 |

Count 128 is -0.9 % in one format and +0.8 % in the other; count 256 is -0.6 % then +0.4 %. Only
count 512 is consistent, at -0.2 % in both — which is the edge of the band. Sums differ by 0.1-0.2 %.

### Why — and this is the useful part

The trade named up front is real and it **cancels**. Removing one full L1 round-trip of the
accumulator buys about as much as is lost by moving the SFPU SiLU off the PACK thread (where
`add_bias_bcast_rows` issues it via `silu_tile_init_pack()`, overlapping MATH work) onto the MATH
thread (where `silu_tile()` serialises with the adds).

**That measurement puts a price on the pack-thread issue slot: it is worth roughly one accumulator
L1 round-trip.** Which retroactively explains why `dest_acc` won so cleanly by comparison — it
removed `nc-1` round-trips and gave up NO thread overlap at all. The lever is not "fewer L1 passes"
in general; it is "fewer L1 passes that you don't pay for in lost MATH/PACK concurrency".

**Default 0.** Knob retained (off) as the measurement.

---

## §14.3 — `MOE_SWIGLU_SCATTER_NOBAR`: drop the gather's trailing atomic barrier — **NULL**

Dropped `noc_async_atomic_barrier()` after the semaphore increments at all four sites (writer's
gather signals, writer's h-slice signal, reader's column invites, reader's split up-half). The
data-before-signal `noc_async_write_barrier()` was left in place — that one is load-bearing.
Golden **45/45**, so the barrier really was guarding nothing local.

bfp8, 5 reps: `84.47 / 112.33 / 199.36` -> `84.50 / 112.60 / 199.28`. Flat.

**Why.** The premise was "two full ack round-trips per M-block". It is not two: the atomics are
issued IMMEDIATELY AFTER a write barrier that has just drained writes to the SAME eight destinations,
so the path is warm and the atomic acks — 4 bytes each — return almost immediately. The second
"traversal" costs essentially nothing to begin with. **Default 0** (keep the barrier: it is free and
strictly safer).

---

## §14.4 — `MOE_SWIGLU_DOWN_OUT`: last `down` K-block packs straight to the output — **BLOCKED, structural**

### What was built

`LastBlockTarget::Out` instead of `Interm`: non-last K-blocks still L1-accumulate into the bf16
interm, the LAST one reloads that partial into DEST, adds its own result and packs once to
`cb_out_tiles` — deleting the separate `compute_out_pack` bf16->bfp8 copy (24 tiles per M-block).
The helper `static_assert`s that `caller_owns_pack_target` "requires ... last_block_target ==
Interm", so the flag had to come off; that is exactly what brings its software-reload path to life.

**Result: `pcc = 1.000000` with `inf`** — the same wrong-format signature as before, but this time
it is NOT a missing reconfig. The helper's reload is format-clean on its own terms
(`copy_tile_to_dst_init_short_with_dt(in1, interm)` on the way in and
`mm_block_init_short_with_dt(in0, in1, interm, ...)` on the way back out).

### The real cause — a runtime-`m_eff` invariant that `caller_owns_pack_target` exists to hold

With `caller_owns_pack_target = false` the helper reserves/pushes/pops the interm CB itself, once per
K-block (`interm_buf.wait_front(...)` / `pop_front(...)` at `matmul_block_helpers.inl:524-538`). Packer
L1-accumulation then only lands on the previous K-block's bytes if the FIFO write pointer returns to
the CB base every block, i.e. if

    cb_out_interm capacity  ==  out_block_num_tiles  ==  m_eff * EC_MAX

The CB is sized ONCE, host-side, at the maximum: `_cb(CB_OUT_INTERM, ..., n_out_block, ...)` with
`n_out_block = M_BLOCK * EC_MAX` (24). At runtime `m_eff` SHRINKS with the token count — 4, 2, 1 —
so the per-block push becomes 12, 6, 3 tiles against a 24-tile CB and the pointer lands on a
different half (or quarter) each K-block. The accumulation then adds onto the wrong tiles. That is
exactly why the failing cell is `n32` (the smallest `m_eff`) and why the op passes
`caller_owns_pack_target = true` in the first place: ONE reserve over the whole block pins the write
pointer at the base, so every K-block accumulates at the same address **at every runtime m_eff**.

**Not fixable by a reconfig, and not worth the alternative.** Making it work would mean sizing the
interm CB to the runtime `m_eff` (impossible — CBs are allocated once) or teaching the helper a
caller-owned Out path (a kernel_lib change, for a ~1 % ceiling: the copy costs unpack + pack and the
reload costs unpack, so the saving is one 24-tile pack per M-block). **Default 0**, knob retained
off, and this is recorded as CLOSED on the invariant rather than open on effort.

---

## Determinism stress + race audit (post-round verification)

Full write-up: **`RACE_AUDIT.md`**. Harness:
`tests/.../test_moe_fused_swiglu_determinism_stress.py` — many shapes interleaved pseudo-randomly,
each compared BITWISE against its own first run, device-side.

**100 028 dispatches, 0 divergences**, over 14 shapes spanning m_eff 1/2/4/8, m_blocks 1/2/4, both
activation formats, emb 6144/7168, capacity 1024/5120, count 0, and the ragged 255/257/513 seams —
on both the default 110-core grid and the `11x8` grid the perf numbers are quoted at, across 4
interleave seeds.

Interleaving is the point: the op's synchronisation is entirely MONOTONE counters that are never
reset within a dispatch, so a single-shape loop settles into one steady-state rhythm and would
reproduce a timing-dependent bug bit-for-bit forever. Varying `m_eff`, `m_blocks` and format between
consecutive dispatches means no two dispatches start from the same state.

**Three real races were injected into the shipped op as negative controls and all three failed the
run** (then the tree was restored and re-verified: golden 45/45 + 5 028 dispatches). Notable finding:
in this op a desynchronised read does not drift, it produces `inf` — even the subtle "7 of 8
contributors synchronised" variant — because `h` is bfp8 and `W_down` bfp4, so a torn read corrupts
a SHARED EXPONENT rather than one mantissa.
