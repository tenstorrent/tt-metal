# moe_fused_swiglu — why each decision is what it is

Every tuning constant in `moe_fused_swiglu_geometry.py` and every non-obvious structure in the
kernels got there through a measurement. The code states the claim in a line or two and points
here; this file carries the evidence, including the parts that argue against the current choice.

Chronological detail lives in `changelog.md` and `ROUND17_LOG.md`; the race argument lives in
`RACE_AUDIT.md`. This file is organised by *decision*, so a reader who wants to change one thing
can find the one section that says whether it has already been tried.

Numbers are microseconds, 88 cores (11x8), emb 7168, capacity 5120, ND-sharded weights, median of
5 reps, quoted at counts 128 / 256 / 512. **The per-cell run-to-run band is ~1 % and session drift
is the same size**, so every A/B below was measured against a baseline taken in the same session.

---

## 1. The blocking model

The op is three matmuls with a SwiGLU between them, and the structurally hard part is that
`W_down` contracts over the 2048 axis, which is the *output* axis of gate/up. Any cross-core split
of that axis leaves every core holding a partial `[count, emb]` output.

Instead of splitting it, the design **rotates the axis between the two phases** and pays with an
all-gather of `h`:

| | payload moved across cores | at count 256, M_BLOCK 16 (the figure this table was computed at; the op ships M_BLOCK 8) |
|---|---|---|
| all-gather `h` (chosen) | `M_t x HID_T` tiles, once, bfp8, multicast | 8x64 = 512 tiles = 0.56 MB |
| reduce `[M_t, EMB_T]` partials KGROUPS-deep | `M_t x EMB_T` tiles per level | 8x224 = 1792 tiles = 1.95 MB **per level** |

`h` is narrow and the output is wide, so moving `h` is >= 3.5x cheaper per hop and >= 14x cheaper
in total — and it lands as a broadcast rather than a reduction.

`M` is deliberately NOT a core-assignment axis: it is a *runtime* value, so a compile-time grid
keyed on it would idle most of the machine at the graded counts (M_t = 4 at count 128).

---

## 2. The reduce: scatter, not tree

`REDUCE=scatter` replaced a binary Hillis-Steele reduce tree. **12-cell sum 8 400 268 -> 5 518 176
ns, i.e. -34.3 % / 1.52x.** About 85 % of the win is the epilogue: the tree ran SiLU on the column
root alone, while the scatter parallelises it KGROUPS ways and collapses the root's `m_eff`-call
bias walk to one call.

Two properties of the scatter are load-bearing and easy to break:

* **Landing CBs are reserved and pushed WHOLE.** That returns every core's write pointer to the CB
  base every M-block, which is what lets a contributor use its OWN write pointer as the
  destination address on a peer. Push a partial block and the address proxy silently breaks.
* **`P % B == 0` for every CB cycled in blocks of B pages.** A CB wraps only at its END, so a block
  starting mid-CB and running past the end overruns into the next CB. A plan violating this
  measured PCC 0.709-0.886 where every legal plan scored >= 0.9955. `scatter_plan()` enumerates
  every reachable `m_eff` and checks this.

The tree was deleted in the rewrite. `Blocking._choose_hn_pad` now guarantees the scatter plan is
expressible by padding `hn_pad` up, so the fallback it used to provide is unreachable.

**Slice accumulators are bf16, not bfp8, and that is correctness, not perf.** The scatter chains
KGROUPS contributors through one accumulator where the tree chained `ceil(log2(KGROUPS))`, and
bfp8's pack rounding is a biased half-LSB (every partial here is positive), so the error is linear
in chain length: max relative error 0.0204 on the tree, 0.0580 with bfp8 slice accumulators,
against that test's 0.05 gate. bf16 removes the per-step quantisation and measured back at 0.0204
— bit-identical to the tree's. Costs ~17 KB/core against the path's ~100 KB saving.

---

## 3. The h all-gather

`DataReadySignal` has two options and each gives up something the op needs:

* **Flag** — data and signal ride one `NOC_CMD_VC_LINKED` chain, so ordering is free and no acked
  write barrier is needed. But there is ONE cell, so round r+1's sender cannot set VALID until
  every core has cleared round r's — a per-round grid-wide serialisation.
* **Counter** — monotone per slot, so rounds overlap. But the signal is an atomic on a *different
  command buffer* and therefore cannot terminate the link, so the data must go unlinked and the
  sender pays an acked write barrier every round: a second NUM_CORES-way incast. **Measured +7 to
  +13 %.**

**Per-slot flags take the union** and are what ships: the link survives (no barrier) and rounds r
and r+1 touch different cells (no reset chain). Rounds r and r+DEPTH_H share a cell and are
ordered by the ack itself. Costs no extra semaphore and no L1.

`HACK_AHEAD = 2` — how many rounds' senders a receiver acks in one reserve. Legal only because of
the per-slot cells. Shipped with them: **-1.9 / -4.0 / -3.8 %.**

**Posted payload (`HPOSTED`), shipped: -0.3 to -2.0 %.** The data multicast is posted so the 87
destinations generate no write-acks; the VALID flag stays NON-posted and LINKED on the same VC, so
it cannot overtake the payload. This adds no new assumption — the shipped Flag path already takes
no barrier and already depends on exactly that wire ordering. Posting removes acks, not order.
Written raw because `noc.h` blocks posted multicast at the library level.

**Dual-NoC h (`HSPLIT`) — built, correct, DROPPED.** Flat, then -5..-6 % by 6/8 eighths. The NOC_1
half cannot ride the linked flag, so it needs an acked barrier of its own, and that costs more
than the extra lane wins.

**`HSEND=writer` — correct, +5 to +7 %.** Moving the whole broadcast off the reader removes the
HGROUPS-long serial chain in theory; in practice the reader's send is already off the critical
path once per-slot flags exist.

---

## 4. Weight streaming

**Cross-M-block residency (`W_RESIDENT`, `WD_RESIDENT`).** Every weight read is a pure function of
this core's `kstart`/`hstart`/`jstart` with no M-block index, so block b > 0 re-reads bytes still
sitting in the CB slot. Skipping the DRAM loops while keeping the reserve/push handshake exactly
as-is: **gate/up -9.36 %, W_down a further -2.04 %.** gate/up residency also collapses `DEPTH_W`
2 -> 1, freeing 155 KB — which is what funds `DEPTH_X`'s resident-x slot.

`WD_RESIDENT` has a precondition that is silent-wrong-answer class if broken: the reader pushes
exactly HGROUPS W_down K-blocks per M-block, so slot r holds K-block r on every M-block only while
the CB capacity DIVIDES that push count. Break it and blocks b > 0 matmul against the wrong weight
block — no hang, no compile error, just wrong numbers on the multi-M-block path alone.

**N-chunked gate/up stream (`GU_CHUNKS = 3`).** The weight block is issued and consumed in chunks
so the matmul on chunk c overlaps the DRAM read of c+1. Chunking on N rather than K is deliberate:
a K-chunk would cost `m_eff` extra L1-accumulating packs per extra K-block, roughly what the
overlap wins.

*Where* the tail chunks are issued is the whole result. Issuing all of them up front measured
+17 / +6 / +5 %, because the x staging prologue carries a blanket `noc_async_read_barrier()` which
is all-or-nothing and drains them too — so the whole weight block got paid for before a single
stick was tilized. Chunk 0 is issued before the barrier, chunks 1..N-1 after.

**Transaction-id ring on the gate/up stream (`WG_TRID`) — null.** Same idea one level up (tag each
chunk, issue all, drain per chunk). It does not help once the tail chunks are placed correctly.

**`WD_AHEAD = 1`.** Prefetch depth in phase-2 K-blocks: 1 -> 227.8, 4 -> 228.7, 11 -> 240.1.

**The W_down NoC split (`WD_SPLIT = 3`), shipped: -4.7 / -2.6 / -1.1 %.** Eighths of every phase-2
K-block's hidden rows are read by the WRITER on NOC_1 instead of the reader's NOC_0. A real
interior optimum — f = 0.561 measured against 0.55 predicted; 4 and 6 are worse.

Two things it needs:
* a **cross-RISC completion gate**, because `noc_async_read_barrier()` is per-RISC-V and the
  reader's barrier proves nothing about the writer's share of the same K-blocks. Publishing without
  it hands the `down` matmul a half-written weight tile.
* **publish per K-block by transaction id**, not one blanket barrier. Costs the writer nothing (the
  last `barrier_with_trid` returns when a whole-batch barrier would) but the reader stops waiting
  for the whole 111 KB stream. Worth +13 % at count 128.

**Activation-first DRAM priority (`XPRIO`).** The writer holds its W_up stream until this core's
reader has staged x, so 3.67 MB of activation is not queued behind 16.5 MB of weights.

**Read order (`XSTAGE_FIRST`), shipped: -1.8 / -1.7 %.** Stage x before issuing the W_gate
prefetch. Must be re-measured together with `XPRIO` if either changes — they are one lever.

**Stick-walk stagger (`XSTAGE_STAGGER`).** Start the stick walk at `(my_col + my_row) % 32` and
wrap. The only one of the two injector-map rotations that changes which BANK a core is on at a
given instant. Shipped: 91.29 -> 91.09 / 128.42 -> 127.92 / 226.96 -> 225.89.

**Diagonal injector map (`XSTAGE_DIAG`) — null, twice.** Rotating which COLUMN injects a tile-row
so a round's injectors form a diagonal rather than a vertical line. The upstream measurement it was
built from is real (789k ns column vs 204k diagonal, 3.9x, because NOC0 routes east->south), but
here it measured ~-1 % at 128/512 and flat at 256 both times it was tried.

---

## 5. Bank-run coalescing — deleted, and why it should stay deleted

Interleaved page -> bank is `page_id % num_banks` with in-bank slot `page_id / num_banks`, so a
stride-num_banks run of columns at a fixed row is physically contiguous inside one bank and could
read as one transaction. The op had this: `remap()` re-indexed the N axis so consecutive linear
indices walked one bank's slots, and `WRUN` capped the run length.

**Measured a NET NEGATIVE over two full guard-set samples, and shipped OFF (`WRUN = 1`).** A long
run trades NoC command count against DRAM-side locality and the trade does not pay here.

It is also mutually exclusive with the ND shard: the remap re-indexes N for the interleaved bank
layout, which would scatter a shard's own contiguous run.

What survives is `WeightRuns<SHARD_W>` — the shard run length, which needs no remap, no bank
arithmetic and no power-of-two bank count.

---

## 6. Weight placement

The op reads whatever placement it is handed. `nd_shard_n_tiles()` is the ONE place it learns the
layout; everything downstream is a run length. An unrecognised placement is SILENTLY CORRECT and
just slower, which is why every harness asserts the READER's own predicate rather than the memory
config it asked for — a detection bug otherwise shows up as a number attributed to the wrong path.

**Re-measured after round 17** (11x8, bf16_rm, emb 7168, capacity 5120, median of 3, us):

| placement | 128 | 256 | 512 | sum | vs preferred |
|---|---|---|---|---|---|
| preferred shard (N slice, 1 tile-row) | 84.87 | 122.18 | 213.71 | 420.76 | — |
| same N slice, 4 tile-rows tall | 84.77 | 120.07 | 212.48 | 417.32 | −0.8 % (in band) |
| interleaved (one transaction per tile) | 97.50 | 128.62 | 218.58 | 444.69 | **+5.7 %** |

Two things fall out, one of which contradicts what this file used to say.

**The RUN LENGTH is what matters.** Interleaved costs +14.9 % at count 128, +5.3 % at 256, +2.3 %
at 512 — largest where the weight stream is the biggest share of the op, which is the expected
shape. The standing "up to 11 %" figure predated round 17 and is superseded by these.

**The one-tile-row HEIGHT does not.** This file previously justified height 1 as a measured
constraint — a core pinned to one DRAM bank saturating near 30 GB/s against ~370 GB/s with the
bank rotating across K. At four tile-rows the op is within the noise band of height 1. So the
height claim does not reproduce at this op's request pattern, and height 1 is now a DEFAULT rather
than a requirement. Any height is correct and, on this evidence, about as fast.

That is also the check the generalized detection needed: a non-preferred but coalescible shard has
to EARN the fast path, not merely be tolerated. `[6, 6, 3]` at height 4 confirms it does.

**A harness trap worth recording.** The first run of this A/B produced three identical numbers,
because `KNOB_WPLACE` was set in the child environment while the script expanded it in its own
shell — all three arms silently ran the default. The self-verifying print (`reader shard widths`)
is what caught it. A placement A/B with no such print cannot distinguish "no difference" from
"the same configuration measured three times".

## 7. Compute

**Blocked eltwise (`ELTWISE_BLK`), worth 1.05-1.07x.** `input()`/`output()` default to per-TILE
wait/pop/reserve/push, and `eltwise_chain` only honours a block size when every CB reader uses a
compatible policy — so the default silently clamps the DEST window to one tile and pays a full DEST
sync round trip per tile. `OperandKind::Block` is required, not cosmetic.

**The slice fold in DEST (`dest_acc`), shipped: -2.6 to -4.0 %.** The reduce-scatter's per-worker
fold was `copy` + (KGROUPS-1) in-place adds, each a separate chain re-emitting its own setup and
packing to L1. `add_tiles_init(..., acc_to_dest=true)` makes `add_tiles` compute
`DEST[dst] += A + B`, so the running sum lives in DEST for the whole fold and is packed exactly
ONCE.

Two traps this hides:

1. **Reconfig before init, on BOTH sides.** Any raw compute block must emit
   `reconfig_data_format(srcA, srcB)` AND `pack_reconfig_data_format(out)` *before* its `*_init`
   calls. The inits set the math MOP, not the unpacker's format registers. Omitting it yields
   `pcc = 1.000000` with `inf` in the output — right pattern, wrong scale — and it took four
   attempts to find.
2. **`skip_eltwise` cannot price a pack-count change.** `SKIP_COMPUTE` elides only the inner matmul
   LLK; every `eltwise_chain` in the TU keeps running. An earlier round used it to retract this
   whole family at "at most 712 ns". It was wrong by 4x.

`hoist` (the same fold with `SetupOwner::Caller` so only the first step emits setup) was **flat** —
which is the useful finding: the fold's cost is the PACKS, not the setup.

**`pack_l1_pair` — same win, rejected on L1.** +102 KB for a bf16 accumulator CB against 10 560 B
free at 11x8. `dest_acc` gets the same halving with DEST standing in for the accumulator.

**Fold the gate SiLU into the DEST accumulation (`SILU_FUSE`) — null.** The real trade is not "free
packer SiLU vs SFPU SiLU": SiLU is SFPU work either way. What changes is WHICH THREAD issues it —
`add_bias_bcast_rows` uses `silu_tile_init_pack()` (PACK thread, overlappable) against `silu_tile()`
on MATH (serialised with the adds). One L1 round-trip against some MATH/PACK overlap, and the trade
cancels; the sign flipped between sessions.

**Pack the last `down` K-block straight to the output (`DOWN_OUT`) — originally blocked.** The
runtime-`m_eff` CB-wrap problem described here was real; the later fixed-address scratch protocol
solves it without advancing the CB. See §12 for the required PACK-to-UNPACK synchronization.

**bfp4 h (`H_DTYPE`) — precision failure, not a knob.** Phase 1's reconfigs are hoisted out of the
loop, so the SwiGLU multiply inherits the `cb_gate_acc` pack format. That is correct only while
`cb_h_slice` is bfp8. At bfp4 the packer emits bfp8 into a bfp4 CB and the unpacker reads garbage
exponents: **pcc = nan, max_abs = inf on 8 of 9 precision cells.**

**FPU multiply through L1, not DEST reuse.** The SwiGLU multiply keeps its operands in L1
deliberately; the L1 round-trip measured faster than `binary_dest_reuse_tiles` for an FPU consumer.

---

## 8. What is closed — do not re-walk

Measured directly and found to be at its limit, or found not to be the bottleneck:

* **the matmul** — at its real bfp8 x bfp4 LoFi limit, ~10.5 cycles/tile-MAC
* **the h round count** — HGROUPS rounds is the whole hidden extent; there is no round to remove
* **request sizing** — see §5
* **grid shape** — HGROUPS 8 and 10 overflow L1 and break the scatter plan
* **the peer-loop order (`SCATTER_ROT`)** — null on end-to-end AND on tt-npe. The hypothesis (that
  synchronised incasts explain the 302.6 % max link demand against a 24.5 % average) was wrong
  about what the hotspot is: rotating the walk does not move it.
* **the gather's trailing atomic barrier (`SCATTER_NOBAR`)** — null; the atomics ack on a warm path.

## 8b. N scaling, and the L1 ceiling

The op has no N=2048 dependency: `hidden` comes from `w_gate.shape[-1]` and `Blocking` derives
everything from it. What N actually costs, measured at 11x8, bf16_rm, median of 3, us:

| | 128 | 256 | 512 |
|---|---|---|---|
| emb 7168, N 1024 | 60.54 | 92.93 | 167.56 |
| emb 7168, N 2048 | 84.67 | 121.42 | 212.09 |
| emb 6144, N 1024 | 55.71 | 84.60 | 153.24 |
| emb 6144, N 2048 | 76.92 | 109.42 | 193.40 |

**Sub-linear, and that is expected.** Doubling N doubles the gate/up output, the h all-gather and
the `down` contraction, but the op is weight-DRAM-dominated and only 2 of the 3 weight matrices
scale with N in a way that is not already amortised — plus the fixed per-round rendezvous does not
move. At emb 7168 the count-256 cell goes 92.93 -> 121.42 for a 2x N, i.e. **1.31x**.

**`hn_pad` is a SEARCHED padding, not `ceil(hid_t/hgroups)`.** It has to (a) cover the extent,
(b) leave every column group a real column, (c) decompose into chunks inside the DEST budget, and
(d) satisfy the scatter plan. At N=2048/11 columns the search returns 6, unchanged. At N=3584 it
returns 11 with 11 chunks — legal, but an unusually fine chunk stream that has not been measured.

**The real ceiling is L1, and it is emb-dependent.** `cb_w_down` is `depth_wd * hn_pad * ec_max`
tiles and `hn_pad` is `hidden / columns`, so the whole CB set grows with N. Measured at 11x8 with
bfp4 weights: emb 7168 fits N <= 2048, emb 6144 fits N <= 2048 with residency. Beyond that the op
raises with the computed numbers rather than letting the allocator throw.

W_down residency is therefore a BUDGET DECISION, not a constant: when the resident CB does not
fit, the depth falls back to the smallest legal one. That also disables `WD_SPLIT`, which needs
`depth_wd == hgroups` for its address derivation — so the fallback costs both wins at once, and
the honest summary is that this op wants residency and is L1-bound before it is anything else.

**The L1 accounting trap, recorded because it cost time twice.** The allocator's "circular buffers
grow to N B" is an ADDRESS, not a size: the CB region starts above the kernel binaries, runtime
args and semaphores, and that base is PROGRAM-SPECIFIC. Measured: a descriptor whose CBs sum to
1 515 264 B threw at 1 626 752 B — a base of 111 488 B — while
`get_max_worker_l1_unreserved_size()` reported only 40 832 B reserved. `L1_CB_RESERVE` is that
difference. It is one measurement of a program-specific quantity, so the allocator stays the final
authority; the margin only makes the common case fail with a useful message.

## 8c. Worker grid

The grid is a `core_grid=` argument, defaulting to the device's full
`compute_with_storage_grid_size()`. Everything about the op's shape derives from (HGROUPS,
KGROUPS): the hidden split across COLUMNS, the emb contraction across ROWS, the reduce column
height, the all-gather round count, and `cb_w_down`'s slot cycle. A grid change moves all of them
at once, and the failure mode is a HANG, not a wrong number — the collectives only agree while
every core computes the same plan.

Verified on 11x8 (the graded grid), 11x10 (the device's full grid, i.e. what a caller gets by
DEFAULT — so testing only 11x8 would leave the default untested) and 8x8, which is the interesting
one: HGROUPS 8 gives hn_pad 4, 2 chunks, an 8-tall reduce column and 8 all-gather rounds. All three
clear the bfp4 format floor.

**Two constraints, both now derived rather than assumed:**

* `depth_wd` must DIVIDE hgroups when residency is on or `wd_ahead > 1`. The old code justified
  `depth_wd == hgroups` with "hgroups is 11 — prime", which is true and useless on any other grid;
  the predicate is the divisibility, and on a prime hgroups it happens to force the maximum.
* `WD_SPLIT` needs residency AND `depth_wd == hgroups`. The rewrite briefly checked only the
  second, which on some grids leaves the writer writing W_down slots that are live on b > 0 — a
  race, not a slowdown. Both conditions are required.

Also guarded: the trid ring tags K-block r with id r+1, so `hgroups > NOC_MAX_TRANSACTION_ID`
would alias two blocks onto one id and publish one whose bytes are still in flight. Above that the
split is dropped, not the correctness.

**Small grids are refused, not attempted.** Fewer cores means a larger `kr_pad` and `ec_max` per
core, so the per-core working set GROWS as the grid shrinks: at emb 7168 / N 2048 a 4x2 grid needs
~7.5 MB of CBs against ~1.4 MB available. The op reports the computed numbers and says what
actually helps — more grid COLUMNS, not fewer, which is the opposite of the intuition.

Non-rectangular and non-origin grids are not expressible: `core_grid` is an (x, y) extent and every
collective's multicast rectangle derives from a single `CoreRange` at (0, 0).

## 8d. Weight dtype

All three weights share one dtype; the stride and the CB format are one number taken from
`w_gate.dtype`, so a mixed set is rejected rather than half-supported.

**Accuracy, at emb 7168 / N 1024 / count 256, against an fp32 reference:**

| dtype | tile | op PCC | that format's own floor |
|---|---|---|---|
| bfp4_b | 576 B | 0.979354 | 0.979839 |
| bfp8_b | 1088 B | 0.999439 | 0.999973 |
| bf16 | 2048 B | 0.999473 | 1.000058 |

Each is gated against its OWN quantized floor, not a fixed number, and the floors are asserted to
be ORDERED. That ordering is the check that catches a weight CB left on the wrong format: a bf16
run that quantized to bfp4 somewhere still clears a fixed 0.975 gate, but it cannot land on the
bf16 floor.

**Perf, 11x8, bf16_rm, emb 7168 / N 1024, median of 3, us:**

| dtype | 128 | 256 | 512 | vs bfp4 |
|---|---|---|---|---|
| bfp4_b | 60.79 | 93.34 | 167.91 | — |
| bfp8_b | 75.94 | 102.33 | 178.73 | 1.11x |
| bf16 | 123.08 | 147.46 | 222.69 | 1.53x |

Sub-proportional to the bytes (bfp8 is 1.89x bfp4's bytes for 1.11x the time, bf16 3.56x for
1.53x), because the activation stream and the fixed per-round rendezvous do not scale with the
weight format. The gap widens at LOW count, where the weight stream is the larger share.

**L1 is what bounds this, not correctness.** Weight CBs are resident, so a wider dtype costs
proportionally, and W_down residency is given up first. At 11x8:

| | bfp4 | bfp8 | bf16 |
|---|---|---|---|
| emb 7168, N 1024 | fits, resident | fits, resident | fits, NOT resident |
| emb 7168, N 2048 | fits, resident | does not fit | does not fit |
| emb 6144, N 2048 | fits, resident | fits, NOT resident | does not fit |

So bf16 weights at the graded shape are not available on this device, and the op says so with the
computed numbers.

## 8e. External review (codex), and what it found

The rewritten op was reviewed by an independent agent with no history in this codebase, pointed at
the code and told to prioritise races in the shared-header extraction, N/grid generality holes,
and readability. It found **no race introduced by the extraction** — it traced `scatter_leg`'s
source/destination offsets, the write-pointer proxy, the barrier ordering and the mailbox fence
and confirmed each is preserved. It found five other defects, all real, all now fixed:

1. **W_down's non-resident fallback never reached the kernels** (high). The geometry can turn
   residency off and shrink `depth_wd`, but the descriptor emitted the module-level `WD_RESIDENT`
   constant instead of the resolved `blk.wd_resident`. The kernels therefore still skipped every
   W_down read after M-block 0 while the shrunk CB no longer held block r in slot r. Reproduced:
   at emb 6144 / N 2048 / bfp8 weights / count 1024, **M-block 1 came back at PCC −0.0002** —
   completely wrong, silently. There is now a test at exactly that shape, and it fails without the
   fix. This is the same class as the `wd_split` bug in §8c and the same lesson: the shipped
   configuration never exercises the fallback, so no amount of stress at 11x8 finds it.
2. **The bf16 output path wrote bfp8-sized pages** (high). `dtype=bfloat16` is accepted and the CB
   was sized correctly, but the writer's accessor and its write stride were both hardcoded to
   `BFP8_TILE`. `OUT_TILE_BYTES` is now a compile-time arg.
3. **`min()` of two gate/up shard widths can invent boundaries** (high). If the two tensors are
   sharded differently and the larger width is not a multiple of the smaller, the synthetic
   boundary does not subdivide the real one and a run crosses a real shard edge as though it were
   contiguous. The widths must now AGREE, or the op takes the uncoalesced stream.
4. **The `down` DEST budget was only checked above height 1** (medium). `ec_max` is the sub-block
   WIDTH and grows as the grid narrows, so a narrow grid busts the budget at height 1 — which the
   loop never tested. Now an explicit refusal.
5. **Weights were not required to be TILE_LAYOUT** (medium). The kernels address them as tile
   pages at a `W_TILE` stride, so a row-major weight would have been read as tiles.

It also caught two documentation errors, one of which matters: this file and two code comments
claimed "every semaphore is monotone". That is false — the h all-gather's per-slot VALID cells are
set and cleared every round. They are a Flag signal whose safety comes from the linked-VC ordering
and one cell per slot (§3), not from monotonicity. Corrected in all four places.

The review is the strongest argument in this document for reading the geometry off the shipped
grid: three of the five defects are invisible at 11x8 / N 2048 / bfp4 and were found by asking
what the code produces elsewhere.

## 9. Cross-block x prefetch: scope the phase-2 read barriers

After the round-17 wins there was no useful knob-turn left at count 512 (`WD_SPLIT` 4 -> 200.05,
6 -> 199.07, `WD_AHEAD` 2 -> 200.33, `GU_CHUNKS` 2 -> 207.03, against a 199.54 base). The structural
blocker was that phase 2's blanket `noc_async_read_barrier()` calls drain reads on every transaction
id, so a cross-block x prefetch paid its full DRAM latency inside phase 2.

The reader now tags current-block W_down and phase-2 local reads with transaction id 14, tags the
next block's local activation-row read with id 15, and uses scoped barriers for the former. The
existing depth-2 resident-x CB provides the destination; no L1 was added. bf16 input uses the
existing one-row pre-tilize slot, while bfp8 input reads directly into the next resident slot.

Do not use transaction id 0 with `noc_async_read_barrier_with_trid` here. Id 0 is the legacy
untagged stream on this architecture; the first prototype hung at the initial W_down wait. Phase 2
must be assigned its nonzero id before the first W_down request is issued.

At count 128/256 there is only one M block and this path is inactive. At 512/1024/5120 it changed
204.71/380.43/1757.32 us to 199.42/356.68/1612.46 us (-2.6/-6.2/-8.2 %). The marginal slope from
1024 to 5120 fell 336.2 -> 306.6 ns/token. The saving is 7.57 us per additional 256-token block;
the block's bf16 input is 3.67 MB, or 7.17 us at 512 GB/s, so the result is consistent with hiding
essentially the exposed activation DRAM read.

The adjacent gate/up subblock-height sweep supplied no second winner: height 2 was 5-7 % worse at
large M, while height 4 was flat at 512/1024 and 0.9 % worse at 5120. Height 1 remains.

## 10. Start gate/up before the x multicast finishes

For a full-size `M_BLOCK=8` slot, the reader reserves the whole resident `cb_x_tiles` region but
publishes one `KR_PAD`-tile M row after each completed multicast round. Compute runs W_up first,
because the independent writer already streams W_up on NoC1 while the reader is still in the x
chain, and uses cumulative waits of 1, 2, ... M rows. W_gate keeps its tuned reader schedule and
runs second against the same resident x. A full-slot wait immediately before the explicit
whole-slot pop covers ragged slots whose arithmetic deliberately omits the final padding rows.

Smaller power-of-two slots keep the old single push and gate-first order. Their multicast is too
short to repay the extra CB bookkeeping: applying the progressive schedule at count 128 regressed
the median from about 84.3 to 87.3 us. Keeping the choice runtime avoids specializing on the
device-resident expert count. The helper has one matmul template instantiation and a runtime shape
bit; duplicating the instantiation exceeded the compute kernel config budget (71,568 > 70,656 B).

Blackhole p150, 11x8, emb 7168, N 2048, bf16 RM input, bfp4 ND-sharded weights:

| count | before (us) | progressive M (us) | change |
|---:|---:|---:|---:|
| 128 | 84.25 | 84.38 | within run-to-run noise |
| 256 | 120.05 | 117.49 | -2.1 % |
| 512 | 212.38 | 206.30 | -2.9 % |
| 1024 | 396.92 | 378.80 | -4.6 % |
| 5120 | 1845.66 | 1755.93 | -4.9 % |

A temporary TRISC marker after the first W_up M row proved this is real overlap, not merely an
earlier blocking wait: 71/88 cores completed that row before the reader's x-multicast zone ended,
with a median lead of 5,546 cycles (4.108 us). The marker is not present in the shipped source.

## 11. Pack bf16 activation rows directly into resident x

The old row-major path had two L1 copies after the DRAM read: compute tilized 32 bf16 sticks into a
`KR_PAD`-tile bfp8 staging CB, then the reader issued a local NoC copy from that CB to the final
resident-x row. The second copy was not a format requirement. It existed to preserve the circular
buffer ownership rule: compute could push the staging CB while the reader remained the sole pusher
of `cb_x_tiles`.

The reader already reserves the whole resident slot before publishing the raw-stick input. Compute
can therefore pack to that reserved address without changing `cb_x_tiles`' write pointer or page
counters. `tilize_config::OutputPolicy::CallerOwned` expresses exactly that contract; the old
`CB_X_STAGE` allocation is now only a 64-byte ready channel, pushed after packing and popped before
the reader begins the multicast chain.

The destination is selected with `(b % DEPTH_X) * M_BLOCK * KR_PAD + t * KR_PAD`, not merely
`t * KR_PAD`. CB-interface pointer advancement is local to the RISC which performs the push: the
compute RISC never pushes resident x and therefore continues to observe its initial write pointer.
The shorter expression overwrote slot 0 on block 1 and failed M=512 at PCC 0.491; the explicit
physical slot passed the full 20-block wrap and the cross-revision bitwise gate.

At emb 7168 this frees 30,400 bytes/core. Same-session 21-repetition medians for copy/direct were
198.661/197.919 us at M=512, 356.890/356.515 us at M=1024, and 1613.117/1605.656 us at M=5120.
Thus the direct path is retained: it is bitwise equivalent, saves substantial L1, and gives a small
0.46 % large-M win. The modest time delta also answers why the obviously redundant copy survived
for so long—its latency was largely hidden behind other work.

## 12. Direct W_down final output without the bf16-to-bfp8 copy

The runtime-M obstruction was CB pointer movement, not arithmetic. If the ordinary intermediate
FIFO is pushed and popped by `m_eff * EC_MAX` on every K-block, a short runtime block walks through
the physical `M_BLOCK * EC_MAX` allocation instead of returning to the same accumulation address.
The solution is an unpushed, caller-owned bf16 scratch: every non-final K-block packer-accumulates
at the same absolute offsets, and only the final K-block reloads the partial, adds its contribution,
and packs bfp8 directly to the separately reserved output CB.

An unpushed scratch has no CB credit edge. The first implementation incorrectly assumed
`tile_regs_release` also ordered the later L1 read. It only orders reuse of DEST; under watcher
timing, PACK was still finishing the scratch spill while UNPACK began the final reload. M=32 and
M=64 differed between identical dispatches in about 1,000 elements, in both activation formats.
Returning to the old final copy made all 16 assertion-enabled M-tiles cases pass, isolating the
fault. The retained direct path posts Tensix `PACK_DONE` after the complete penultimate spill and
waits once before the final reload. That is the same primitive used by other held-CB pack-to-unpack
pipelines, and it preserves the fixed address.

The reload is a single `h*w` block copy when `w == row_stride`. Runtime blocking is 2x3 on the 48
cores owning three output columns and 4x2 on the 40 cores owning two columns. Against a synchronized
uniform-2x3 direct path, the combination changed M=256/512/1024/5120 by +0.179/-0.948/-1.028/+0.086
us: neutral at the endpoints and about 1 us better in the middle.

The safe end-to-end medians versus the old final-copy path were 118.533/119.035 us at M=256,
197.932/198.627 at M=512, 355.224/355.963 at M=1024, and 1600.939/1605.269 at M=5120. The one
handshake per M-block costs roughly 10 us over 20 blocks at M=5120, but it is mandatory; the larger
unsynchronized apparent win was a race, not performance. Final validation: the full watcher/LLK
M-tiles suite passed and 500 interleaved dispatches over 24 shapes remained bitwise stable.

## 13. Retaining two W_down K chunks in DEST: rejected block-matmul experiment

The intended saving was to halve the bf16 scratch traffic: wait for two consecutive `h` and
W_down chunks, accumulate both into one acquired output subblock, then spill once. The single-chunk
control passed under watcher, but every two-chunk `matmul_block` variant hung deterministically at
the first full `m_eff=8` block. Triage was uniform across the worker grid: TRISC0 waited in the
unpack source handshake, TRISC1 remained inside the block matmul MOP, TRISC2 waited for the DEST
commit, and the reader eventually filled `cb_h` and blocked reserving the next round.

Neither a short block-matmul re-init at the chunk boundary, a 1xN output subblock, nor programming
the pair as one logical `kt_dim=12` contraction changed the failure. `matmul_tiles` was explicitly
excluded. The experiment was therefore removed rather than leaving a disabled or unsafe knob.
With the present block-major `h` stream (`[chunk][M][K6]`), the viable block-matmul version would
need the producer to form a true row-major `[M][K12]` pair (and W_down `[K12][N]`) before compute;
that is a dataflow/layout change, not an easy compute-loop tweak.

## 14. Alias phase-disjoint bfp8 circular buffers

Three bfp8 CBs have disjoint payload lifetimes on each core: gate/up's gathered input, one gathered
hidden slice, and W_down's final output. Their logical capacities at `M_BLOCK=8` are:

| logical CB | pages | bytes |
|---|---:|---:|
| `CB_GATHER_GATE` | 48 | 52,224 |
| `CB_H_SLICE` | 6 | 6,528 |
| `CB_OUT_TILES` | 48 | 52,224 |

They now share one 52,224-byte physical allocation with three logical CB views. The allocation is
the least common multiple of the logical capacities, rather than simply their maximum; this keeps
every view's whole-capacity wrap valid if the geometry changes. Separate storage consumed 110,976
bytes, so the alias saves exactly 58,752 bytes/core. Total CB storage falls from 1,415,104 to
1,356,352 bytes/core. Including the measured 111,488-byte program region, free L1 rises from
46,272 to 105,024 bytes/core.

The same-block order was already causal: gathered gate/up data is consumed before the hidden slice
is packed, and completion of all hidden rounds precedes W_down output. The cross-block boundary was
not causal. A target reader could invite peers to write block `b+1` while its own writer still had
block `b` output DMA in flight. The writer now waits for that output DMA, pops the output view, and
publishes `SEM_PHASE_FREE=b`; the reader waits for it before sending the next gather invites.
Waiting before the invite is sufficient because reserving the logical gather CB does not itself
write the aliased storage. Payload ablations disable the alias because they can remove the normal
same-block causal edges.

Blackhole p150, 11x8, emb 7168, N 2048, bf16 RM input, bfp4 ND-sharded weights, bfp8 output,
five-repetition medians:

| count | aliased (us) | fresh pre-alias baseline (us) | change |
|---:|---:|---:|---:|
| 128 | 84.871 | 84.739 | +0.16 % |
| 256 | 120.019 | 118.734 | +1.08 % |
| 512 | 199.247 | 198.007 | +0.63 % |
| 5120 | 1594.546 | 1601.650 | -0.44 % |

The changes are within the measured session/run spread, so the alias is retained for its L1 gain.
It passed focused watcher runs at one and twenty full M blocks and the complete 16-case watcher/LLK
runtime-M suite, including bf16 RM and bfp8 tiled inputs, ragged blocks, zero count, and both 11x8
and full-device grids. bf16 output uses a different tile size and remains separately allocated.

## 15. Eight full-size-M-slot W_down rounds on the 11x8 grid

Changing the requested grid to 8x11 did not provide a useful eight-round schedule: the device
clamped it to 8x10 (80 workers), and M=5120 regressed to 1891.179 us. The retained experiment keeps
all 88 workers and changes only full-size runtime M slots (`m_eff == M_BLOCK`; the real prefix may
still be ragged). After reduce-scatter, the eleven hidden-column
fragments for token-tile row `r` gather horizontally onto diagonal core `(r,r)`. Those eight
aggregators then broadcast one complete 64-tile hidden row apiece, giving eight phase-2 rounds rather
than eleven 6-tile hidden-slice rounds.

Each round computes a `1 x 64 x ec` matmul against the complete resident W_down shard. Consequently
there is one K block and no intermediate bf16 accumulation spill/reload. Ragged and short blocks
retain the established eleven-round hidden-slice path. The larger H FIFOs cost about 69 KiB/core;
with phase-CB aliasing the focus shape uses 1,425,984 of 1,532,032 available bytes, leaving about
106 KiB. If that allocation does not fit, geometry disables the M-row schedule before giving up
W_down residency.

Controlled same-tree measurements on Blackhole p150, 11x8, emb 7168, N 2048, bf16 RM input and bfp4
weights:

| count | 11 hidden rounds (us) | 8 M-row rounds (us) | change |
|---:|---:|---:|---:|
| 128 | 85.145 | 85.444 | +0.35 % |
| 160 | 104.199 | 103.707 | -0.47 % |
| 192 | 108.916 | 105.061 | -3.54 % |
| 224 | 111.116 | 109.865 | -1.13 % |
| 256 | 117.836 | 111.873 | -5.06 % |
| 512 | 195.501 | 192.962 | -1.30 % |
| 1024 | 355.503 | 341.486 | -3.94 % |
| 5120 | 1588.984 | 1542.266 | -2.94 % |

Repeated medians were 112.046 us at M=256 and 1540.708 us at M=5120. The implied following-full-
block slope improves from `(1588.984 - 117.836) / 19 = 77.429 us` to
`(1540.708 - 112.046) / 19 = 75.193 us`, a 2.236 us/block or 2.9 % improvement. Reader phase 2
falls from about 32.5 to 29.2 us and W_down compute from about 30.5 to 28.7 us. The gain is real but
smaller than the round-count ratio: eight wider multicasts and matmuls still carry the same total
payload and arithmetic, while the reduction is only in per-round scheduling and accumulation cost.

The complete runtime-M suite passed for bf16 RM and bfp8 tiled inputs, including a full block followed
by a one-row tail. Six consecutive M=5120 dispatches were bitwise identical. The schedule is enabled
only when `KGROUPS == M_BLOCK`, W_down is resident, and runtime `m_eff == M_BLOCK`.
