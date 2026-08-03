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

| | payload moved across cores | at count 256, M_BLOCK 16 |
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
layout; everything downstream is a run length.

**The preferred shard is one tile-row tall, and that is measured.** A core pinned to a single DRAM
bank saturates near 30 GB/s; the same bytes with the bank rotating across K reach ~370 GB/s. Height
1 rotates banks every K-row.

**Interleaved vs ND-sharded, measured at 88 cores, 7 reps, bf16_rm:** interleaved
102.51 / 135.37 / 241.49 against ND-sharded 91.34 / 130.67 / 229.58 — up to 11 %. Note this
predates the round-17 wins and is due a re-measurement.

A placement that fails to apply is **silently correct and slower**, which is why the perf harness
asserts against the reader's own predicate (`nd_shard_n_tiles`) rather than against the memory
config it asked for.

---

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

**Pack the last `down` K-block straight to the output (`DOWN_OUT`) — structurally blocked.** It
needs `caller_owns_pack_target=false`, but that flag is precisely what holds a runtime-`m_eff`
CB-wrap invariant.

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

## 9. Still open

**Count 512.** After the round-17 wins there is no knob-turn left at 512 (`WD_SPLIT` 4 -> 200.05,
6 -> 199.07, `WD_AHEAD` 2 -> 200.33, `GU_CHUNKS` 2 -> 207.03, against a 199.54 base).

The recorded structural blocker: phase 2's blanket `noc_async_read_barrier()` calls drain
trid-tagged reads too, so any cross-block `x` prefetch pays full DRAM latency inside phase 2.
**Scoping the barriers** (`noc_async_read_barrier_with_trid` on a W_down-only trid) is the
prerequisite, before the prefetch.
