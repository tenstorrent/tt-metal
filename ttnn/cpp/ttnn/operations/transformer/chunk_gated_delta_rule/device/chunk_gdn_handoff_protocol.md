# `chunk_gated_delta_rule`, fused program: the producer → receiver hand-off protocol

The fused program of `ChunkGdnDeviceOperation` (`chunk_gdn_fused_program_factory.cpp`) runs the prep ("producer") cores
and the scan ("receiver") cores of one GDN layer in a single program and hands each chunk's seven prep intermediates
from producers to receivers over the NoC, with no DRAM intermediates. This document specifies that hand-off: the
memory it uses, the per-chunk protocol, the invariants it rests on, and what is checked at compile time.

The compute kernels take no part in the protocol. They see ordinary circular buffers (CBs) and are byte-identical to
the two-phase program's. The protocol lives in two dataflow kernels:

| side | kernel | RISC / NoC |
|---|---|---|
| producer | `kernels/dataflow/writer_chunk_gdn_fused.cpp` | BRISC, NOC_1 |
| receiver | `kernels/dataflow/reader_chunk_gdn_scan.cpp` compiled with `GDN_FUSED_RECEIVER` | NCRISC, NOC_0 |

and the constants both sides must agree on are in `kernels/dataflow/chunk_gdn_handoff.hpp`, compiled into the host
factories and every GDN kernel.

## 1. Participants

| role | cores | duties |
|---|---|---|
| producer (h, j), j < NP | `prod_set` | computes chunks c = j, j + NP, … of head h; hands each chunk's intermediates to the head's NV receivers |
| receiver (h, v), v < NV | `rcv_set` | carries state columns [v·Vtl, (v + 1)·Vtl) of head h; consumes every chunk of the head in order c = 0 … NC − 1 |

The two core sets are disjoint; `union_set` is their union. The owner of chunk c is producer `c mod NP`. Each side
addresses the other by virtual worker coordinates passed as runtime args: the writer gets its NV receivers, the
receiver gets its NP producers.

## 2. Memory

### 2.1 Hand-off CBs (the data channel)

Seven fp32 CBs, declared on `union_set` so that every core of the program has them at the same L1 address. Prep's
output index equals scan's input index, so one physical CB is both:

| tensor | CB | tiles per chunk, producer | tiles per chunk, receiver |
|---|---|---|---|
| t_inv | `kCbTinv` (13) | cc = Ct² | cc |
| v_beta | `kCbVbeta` (14) | cv = Ct·Vt | cvl = Ct·Vtl |
| nkd | `kCbNkd` (18) | ck = Ct·Kt | ck |
| q_decay | `kCbQdecay` (19) | ck | ck |
| intra | `kCbIntra` (20) | cc | cc |
| dl·I | `kCbDl` (22) | 1 | 1 |
| k_dec_t | `kCbKdecT` (24) | kc = Kt·Ct | kc |

Each CB holds `NBUF` producer-sized slots. The **producer** computes the receiver's slot address from the global chunk
index; nothing is exchanged at runtime:

- six shared tensors: `slot(c) = base + (c mod NBUF) · n · tile_bytes`, valid because the receiver reserves and pushes
  exactly n tiles per chunk, so its write pointer walks the same ring;
- v_beta: the CB is producer-sized (cv tiles per slot) but a receiver reserves only cvl per chunk, so its ring has
  NV·NBUF slots of cvl tiles: `slot(c) = base + ((c · cvl) mod (cv · NBUF)) · tile_bytes`; row r of receiver v's slice
  is source tiles [r·Vt + v·Vtl, +Vtl) of the producer's front slot.

Both formulas assume one tile size across the seven CBs; the kernels `static_assert` that all seven are fp32 with equal
`get_tile_size`.

### 2.2 Credit words (receiver → producer)

`BH × NBUF` plain 32-bit L1 words `credit[h][slot]` on every producer, at `CB_CREDIT base + CREDIT_OFF + 4·(h·NBUF +
slot)`, where `CB_CREDIT` is the union-declared u/mask CB (`kCbU`, 17) and `CREDIT_OFF = kMaskTiles · tile_bytes` puts
them in the tile behind the prep's mask tiles. They are plain words, **not** `Semaphore` objects: dispatch
re-initialises only semaphore objects per launch, so the kernel zeroes them itself (§4). Each producer reads only its
own words; receivers write them remotely by NoC atomic increment. The host caps `BH · NBUF` at the words of one fp32
tile.

### 2.3 Semaphores (producer → receiver, and init)

Declared on `union_set` with initial value `INVALID` (0); ids reach both kernels as compile-time args:

| id | name | direction | meaning |
|---|---|---|---|
| `kFusedSemReady` (0) | ready | unused in the fused variant | kept so the shared scan reader's trailing-arg layout is uniform across its variants |
| `kFusedSemInit` (1) | init | producer → receiver | "my credit words are zeroed"; the receiver waits for N_INIT = NP increments |
| `kFusedSemValid` + s, s < NBUF | valid[s] | producer → receiver | "chunk c, slot c mod NBUF, is in your CBs" (VALID / INVALID) |

The program has `kMaxSemaphores` (16) semaphores, hence `NBUF ≤ 14` here and, with the host's own cap, `NBUF ≤ 8`.
Semaphore ids are 16-byte-strided L1 words (`sem_l1_base + id · L1_ALIGNMENT`); the kernels never rely on adjacency.

## 3. Parameters

| symbol | source | default | constraint |
|---|---|---|---|
| NBUF | `ChunkGdnFusedProgramConfig::handoff_depth` | 2 | 1 ≤ NBUF ≤ 8; BH · NBUF ≤ 1024 |
| D | NBUF − 1 (1 if NBUF = 1) | 1 | credits issued ahead of the chunk being waited for |
| NP | producers per head | cost model | 1 ≤ NP ≤ NC |
| NV | receivers per head | cost model | NV divides Vt; the receivers form a dense rectangle (multicast transport) |
| UNICAST | `unicast` | true | per-receiver unicast writes; false = the linked multicast chain |
| POSTED | `posted` | false | posted data writes; requires UNICAST |

The writer derives `Vtl = Vt / NV` itself (asserting `Vt % NV == 0`) rather than taking it as an argument, so the two
cannot be swapped.

## 4. Initialisation (once per launch)

Producer:
1. sets its local copies of `valid[s]` to VALID for all s (the remote set sources its 4-byte payload from the local
   word of the same id, read asynchronously by the NIU, §7);
2. zeroes all BH·NBUF credit words in its own L1;
3. `init.up(+1)` on each of its NV receivers (non-posted atomic).

Receiver: `init.wait(N_INIT)`, N_INIT = NP. No credit may be sent before this returns. Then it pre-issues chunks
0 … min(D, NC) − 1 (§5.1).

## 5. Steady state, one chunk

### 5.1 Receiver

```
issue(c):
    reserve_back(D · n) on each of the 7 CBs        # succeeds iff compute popped chunk c − NBUF
    valid[c mod NBUF].set(INVALID)                   # local store, BEFORE the credit
    atomic_inc( producer (c mod NP) : credit[h][c mod NBUF], +1 )

for c = 0 … NC−1:
    valid[c mod NBUF].wait(VALID)
    push_back(n) on each of the 7 CBs                # compute may start chunk c
    if c + D < NC: issue(c + D)
```

`reserve_back` does not remember earlier unpushed reservations, so asking for D chunks' worth is the exact condition
that slot `c mod NBUF` is free (compute has popped chunk c − NBUF); the v_beta ring (NV·NBUF chunks deep for one slice)
is free a fortiori.

### 5.2 Producer

```
for c = j, j + NP, … < NC:
    slot = c mod NBUF
    wait_front on the 7 CBs                          # compute pushed item c
    wait credit[h][slot] == NV                       # all NV receivers reserved the slot
    credit[h][slot] = 0
    for v in 0..NV−1, r in 0..Ct−1:                  # v_beta slices
        write Vtl tiles → receiver v, v_beta slot(c) row r
    for each of the 6 shared tensors:                # nkd, q_decay, intra, k_dec_t, dl, t_inv
        write n tiles → every receiver, slot(c)      # NV unicasts, or one linked multicast
    async_write_barrier()                            # all writes ACKed (POSTED: flush only, §7)
    set valid[slot] = VALID on every receiver        # unicast set_remote, or multicast
    pop_front on the 7 CBs                           # compute may reuse the slots
```

The credit wait requires **exactly** NV: an over-credit is a protocol bug and manifests as a hang, never as corrupt
output.

## 6. Invariants

- **I1. One hand-off per (head, slot) in flight.** `credit[h][slot]` is incremented for chunk c only after the receiver
  reserved slot `c mod NBUF`, i.e. after compute popped chunk c − NBUF, which the producer's VALID for c − NBUF
  preceded, which its zeroing of the same word preceded. Hence the word counts one chunk at a time for any NP, and
  zeroing it after the wait never races a later increment.
- **I2. In-order delivery without sequence numbers.** The static owner map (`c mod NP`) plus I1 means receiver (h, v)
  sees VALID for chunk c only on slot `c mod NBUF`, and the main loop consumes the slots in order. VALIDs of different
  chunks cannot interleave on one slot.
- **I3. No overwrite of unconsumed data.** A producer writes slot(c) only at credit == NV, which every receiver grants
  only after reserving the slot, which succeeds only after compute popped the slot's previous chunk.
- **I4. Data before flag.** With UNICAST the barrier before VALID waits for the acknowledgement of every data write; a
  flush would prove only that the NIU read the source. With the multicast transport the six shared tensors and the flag
  chain on one static VC with `linked = true`, and the unlinked v_beta slices still need the barrier. With POSTED, same
  NIU, command buffer, VC and destination give in-order delivery of the flag behind the data.
- **I5. Slots are returned to compute only after the NIU has read them.** The pops follow the barrier (POSTED: the
  flush), so compute's next item cannot overwrite a slot the NoC is still reading.
- **I6. No lost wakeup.** The receiver stores INVALID into `valid[slot]` before crediting; a producer that sets VALID
  right after the credit lands cannot be overwritten by a late reset.
- **I7. Payload-source rule.** A local write to a `valid[s]` word on the producer must be preceded by a write barrier,
  because an in-flight remote set reads that word asynchronously: hence the preset at init and the barrier before the
  restore at teardown.

## 7. Transport variants

| variant | data writes per chunk | VALID ordered behind the data by |
|---|---|---|
| UNICAST (default) | NV·Ct slice writes + 6·NV shared writes, non-posted | `async_write_barrier` (ACKs), then `noc_semaphore_set_remote` per receiver |
| multicast | NV·Ct unlinked 1×1 multicasts + 6 linked multicasts to the head's rectangle | the barrier, then an unlinked `set_multicast` that ends the chain; the rectangle arrives pre-ordered for NOC_1 (bottom-right → top-left) |
| POSTED (needs UNICAST) | as UNICAST with `NocOptions::POSTED` | in-order delivery on one NIU / command buffer / VC; `async_writes_flushed<POSTED>` before the pops; the VALID write is non-posted and drained at teardown |

## 8. Teardown

Producer: `async_write_barrier` (then `async_writes_flushed<POSTED>` if POSTED), `async_atomic_barrier` (the init
atomics return their ACKs on this NoC while the credits that prove they landed arrive on the other), then the local
`valid[s]` words back to INVALID. Receiver: `valid[s]` back to INVALID, `async_atomic_barrier` to drain the credit
atomics. No NoC transaction may be outstanding at kernel exit. Relaunch correctness rests on dispatch rewriting every
semaphore's initial value; `init` is not restored in-kernel.

## 9. Pipelining

With depth NBUF the receiver keeps D = NBUF − 1 credits ahead of the chunk it is waiting for, so the credit → write →
ACK → VALID round trip is hidden behind D receiver steps instead of sitting on the chain. Each slot costs its seven CB
slots of L1 on every core of the union.

## 10. Compile-time guards

Everything the protocol fixes at build time is checked where the kernels are compiled:

| premise | check |
|---|---|
| the seven CB indices agree across prep compute, scan compute, prep writer, fused writer, scan reader and both factories | each kernel and factory `static_assert`s its local indices against `chunk_gdn_handoff.hpp` |
| all seven hand-off CBs are fp32 with one tile size | `static_assert` on `get_dataformat` / `get_tile_size` in writer and receiver |
| `Vt % NV == 0`, Vtl derived | writer `static_assert`; no separate Vtl argument |
| `Vt_full % Vt == 0` on the scan reader | `static_assert` |
| valid ids fit the semaphore cap and do not collide with init / ready | `static_assert` in both kernels, `TT_FATAL` in the factory |
| credit words sit behind the mask tiles | `static_assert(CREDIT_OFF == kMaskTiles · get_tile_size(CB_CREDIT))` |
| the compile-time arg layout of each kernel matches the factory | `kHandoffTag` appended last by the factory; both kernels `static_assert` on its position and value |
| the credit tile holds BH · NBUF words; NV divides Vt; receivers form a rectangle | host `TT_FATAL`s |

## 11. Failure modes

| symptom | cause |
|---|---|
| producer hangs at `tx_wait_credit` | a receiver never credited: wrong owner map, N_INIT mismatch, an over-credit, or a receiver stuck in compute |
| receiver hangs at `rx_wait_valid` | VALID lost (reset after the set), a flag carrying INVALID (payload-source rule violated), or a producer stalled on its CBs |
| wrong output | not from this protocol by design: every protocol error is a hang; a data race needs a pop before the NIU read the slot (I5) or a slot-address mismatch, which the union declaration and the compile-time guards prevent |

Device-profiler zones: `tx_wait_cb`, `tx_wait_credit`, `tx_issue`, `tx_barrier`, `tx_valid` on the producer;
`rx_reserve`, `rx_wait_valid` on the receiver.

## 12. Extension points

The protocol does not depend on the owner function being `c mod NP`: any static per-core item map works as long as
the receiver can name the owner of chunk c when it credits, and N_INIT counts the distinct producers that serve the
head. A dynamic owner (producers claiming items at runtime) needs the receiver to learn the owner before it credits;
I1 and I2 then rest on the claim order instead of the static map.

## 13. Runtime checks

The invariants above that only hold at run time are checked in the kernels where the watcher's `ASSERT` is compiled
(`WATCHER_ENABLED`); release kernels are unchanged. The CI legs that run the fused tests with the watcher exercise them
on every chunk.

| id | invariant | where | check |
|---|---|---|---|
| C1 | exact-NV credits (I1) | producer, credit poll | `ASSERT(credit <= NV)` on every poll: an over-credit reports instead of hanging |
| C2 | every credit consumed | producer, teardown | all BH × NBUF credit words are zero after the barriers |
| C3' | the producer's rings advance by exactly n per item | producer, before the sends of its k-th item | `get_read_ptr(cb) == base + (k mod NBUF) · n · tile_bytes` for the seven CBs |
| C3 | slot-address agreement (§2.1) | receiver, at the push of chunk c | `get_write_ptr(cb) == base + (c mod NBUF) · n · tile_bytes` for the six shared CBs and `base + ((c · cvl) mod (cv · NBUF)) · tile_bytes` for v_beta: both rings advance exactly as the producer's formulas assume |
| C4 | flag state before the reset (I2, I6) | receiver, in `issue(c)` | `valid[s] == VALID` for c ≥ NBUF (the consumed chunk c − NBUF left it so), `INVALID` for the first round |
| C5 | end state | receiver, teardown | every chunk issued and consumed; each used slot's flag still VALID |
| C7 | the flag belongs to this chunk (I2, I4, I6, I7 end to end) | both, `handoff_checks` | the producer sets `valid[s]` to **c + 1** after its barrier and the receiver waits for exactly `c + 1`; a flag of the wrong chunk, a stale flag or a reordering across slots hangs the equality wait instead of passing; C4 and C5 take their sequence-value forms |
| C8 | data before flag (I4), slot written only after consumption (I5) | both, `handoff_checks` | one canary word per slot in the tile behind the credit tile: the producer writes the chunk index there as the LAST data write before its barrier, the receiver asserts it after VALID and poisons it |
| C9 | liveness | both, `handoff_checks` | the credit wait and the VALID wait are bounded (`kHandoffSpinLimit` polls, ~0.2 s): on expiry the core pushes a timeout trace word and asserts, so a deadlock names the wait, the chunk, the slot and the value seen |
| C10 | event trace | both, `handoff_checks` | one packed word per protocol step (`handoff_trace_word`: stage, chunk, slot, value) in the watcher's ring buffer: credit seen, barrier done, flag sent on the producer; issued, flag seen on the receiver; Blackhole keeps the last 32 per core, printed by the watcher when an assert trips and in its log dumps |
| C6 | stage of each core in a hang dump | both | `WAYPOINT`s `TXCB` (CB wait), `TXCR` (credit wait), `TXBR` (barrier), `TXVL` (flag) on the producer; `RXRS` (reserve), `RXVL` (VALID wait) on the receiver; `DONE` at exit on both |

C1 to C6 need only the watcher. C7 and C8 change protocol-visible state (the flag value, one extra 4-byte write per
receiver per chunk), so they are compiled in only with `ChunkGdnFusedProgramConfig(handoff_checks=True)`, a hashed
field that is a define on the two hand-off kernels; they report through the watcher's `ASSERT`, so run them with the
watcher enabled.

**Fault injection.** `ChunkGdnFusedProgramConfig(handoff_checks=True, handoff_fault=n)` compiles one deliberate breach
into the kernels (`gdn_handoff::HandoffFault`, the `GDN_HANDOFF_FAULT` define): 1 the receiver credits chunk 0 twice
(C1 or C2 on the owner; or, when one receiver's doubled credit alone satisfies the owner, the early send reaches the
other receiver before its reset: C4 there, or C9 after the reset erased the flag — the lost wakeup of I1), 2 the producer writes c + 1 as the canary (C8), 3 the receiver pushes chunk 1's nkd
block one tile short (C3 at the slot's next push), 4 the receiver credits chunk 1 to producer 0 (C9 on both sides), 5
the receiver never credits chunk 3 (C9). `test_chunk_gdn_handoff_faults.py` runs each fault in a child process under
the watcher, asserts that the named check tripped on the expected RISC and waypoint, and resets the board between
faults, since a tripped assert leaves the device stopped; it is opt-in (`GDN_HANDOFF_FAULT_TESTS=1`). A skipped
INVALID reset is not a fault: with the sequence-valued flags of C7 the reset is redundant (the receiver waits for
exactly c + 1), which is what makes the flag-only protocol of the extension points viable.

**Host side.** The factory's build-time checks (NV divides Vt, dense receiver rectangles, the credit tile holds BH ×
NBUF words, NBUF within the semaphore cap) are the only host checks. An owner-map audit is void while both sides derive
the owner of chunk c as c mod NP (it becomes necessary when an item map replaces the formula), and a post-run read-back
of the credit words and flags would need L1 addresses the test harness does not expose; C2 and C5 cover that end state
in every watcher run.
