# `combine_fabric2d`: a visual guide

`combine_fabric2d` is the **return half** of routed MoE prefill.  An expert has
finished processing a token, but that token needs to return to the chip where it
originated and land in the output slot for its `(token_idx, topk_idx)`.

The defining implementation choice is that Fabric only transports a packet **one
physical hop**.  This operation, rather than Fabric, orchestrates a multi-hop
route using a private DRAM forwarding buffer on each intermediate chip.

```text
      expert output                         output contribution
  source chip DRAM                         origin chip DRAM
        │                                         ▲
        ▼                                         │
  worker's L1 ring ── Fabric hop ── forwarding DRAM ── Fabric hop ── ...
                                     intermediate chip
```

The public binding and its precise input descriptions are in
[`combine_fabric2d_nanobind.cpp`](combine_fabric2d_nanobind.cpp).  The program
is assembled in
[`device/combine_fabric2d_program_factory.cpp`](device/combine_fabric2d_program_factory.cpp).

---

## 1. What result does it create?

Per device it allocates a BF16, row-major tensor:

```text
(1, 1, seq_len_per_chip, num_experts_per_tok, embedding_dim)
```

The token result produced by expert `e` for original token `t`, top-k selection
`k`, is written to:

```text
output[t][k][:]
```

This is a **scatter**, not a reduction: a later operation can weight/sum the
top-k contributions.  The output is not zero-initialized, so slots not
represented by this dispatch group must not be considered valid.

---

## 2. The five inputs form a compact routing database

```text
dispatched_buffer        BF16 expert outputs, ordered by expert
dispatched_metadata      INT32 (linearized_coord, token_idx, topk_idx), one per buffer slot
expert_token_counts      number of valid tokens per expert
expert_region_offsets    start of each expert's region
expert_offsets           start of every origin chip's run within every expert region
```

For expert `e`, its region is divided into contiguous runs, one run per origin
chip in the dispatch ring:

```text
expert e region
┌────────────┬─────────────┬────────────┬─────────────┐
│ origin 0   │ origin 1    │ origin 2   │ origin 3    │
└────────────┴─────────────┴────────────┴─────────────┘
  offsets[0,e] offsets[1,e] offsets[2,e] offsets[3,e]
```

The exact interval for origin dispatch-group index `d` is:

```cpp
begin = expert_offsets[d][e];
end   = d + 1 < ring_extent
      ? expert_offsets[d + 1][e]
      : expert_region_offsets[e] + expert_token_counts[e];
```

That formula is important: all chips independently calculate exactly the same
chunk boundaries.  No chip needs to send a run size to another chip.

Although the metadata has `linearized_coord`, this implementation derives the
origin from the run and only consumes `token_idx` and `topk_idx` when it computes
the final output page.

---

## 3. A 2D mesh is treated as several 1D rings

`cluster_axis` selects the axis along which a dispatch group communicates.
Every fixed coordinate on the other axis represents another expert/dispatch
group.  The chosen axis is a **wrapped, even-sized ring**.

Example: a four-chip ring, two links per neighbor:

```text
                            clockwise →
                    ┌────────────────────────────┐
                    ▼                            │
              [chip 0] ── [chip 1] ── [chip 2] ── [chip 3]
                    │                            ▲
                    └────────────────────────────┘
                         ← counter-clockwise

Each chip owns these streams:
  stream 0 = link 0, clockwise       stream 1 = link 0, counter-clockwise
  stream 2 = link 1, clockwise       stream 3 = link 1, counter-clockwise
```

Each stream is placed on a worker core close to the Ethernet core for its
physical link.  Its two data-movement RISCs cooperate:

```text
reader, RISC 1 / NOC 0                      sender, RISC 0 / NOC 1
DRAM or untilizer ──► L1 token ring ──► Fabric connection ──► next chip
```

Keeping reader and sender on different RISCs/NoCs avoids their local NoC traffic
contending.

---

## 4. Which stream sends each run?

Every remote origin-to-destination run must be covered once, with no overlaps.

```text
destination nearer clockwise         → clockwise streams, split across links
destination nearer counter-clockwise → counter-clockwise streams, split across links
diametrically opposite destination   → split across every direction and link
```

For a run `[begin, end)` assigned to share `i` of `n`, a stream takes:

```text
[ begin + floor((end - begin) *  i      / n),
   begin + floor((end - begin) * (i + 1) / n) )
```

Integer slicing makes adjacent shares meet exactly even when the token count is
not divisible by `n`.

The host constructs and validates this static schedule in
[`device/combine_fabric2d_assignments.cpp`](device/combine_fabric2d_assignments.cpp).
For each local expert, a stream executes:

```text
1. Own remote assignments, furthest destination first
2. Relay chunks received from its upstream neighbor
3. Its share of same-chip work, using local NoC instead of Fabric
```

The forwarding chunks are dense; their sizes remain dynamic because each expert
may receive a different number of tokens.  Both writer and reader derive their
length from the control tables above.

---

## 5. One token's journey

Assume a clockwise stream on chip 0 is returning a result to chip 2 in the
four-chip example.

```text
chip 0 reader                    chip 0 sender       chip 1 reader          chip 1 sender
─────────────                    ─────────────       ─────────────          ─────────────
read token and metadata     →    send one hop   →    read from forwarding → send one hop
calculate final output page      to chip 1 DRAM       DRAM into L1 ring       to final output
create forwarding metadata

                                              chip 2 output DRAM
                                              ──────────────────
                                              output[token_idx][topk_idx]
```

The metadata appended to each L1 slot is:

```text
final_addr   final destination's output page address
dst_chip     final destination chip ID
cmd          FINAL_WRITE | FORWARD | FORWARD_END | END
this_addr    address this hop should write to
```

An adjacent destination uses `FINAL_WRITE`: the sender transfers only the token
payload directly to `final_addr`.  A non-adjacent destination uses `FORWARD`:
the sender transfers the token plus `final_addr` and `dst_chip` to the next
chip's forwarding-buffer page.  The next reader can re-read those three values
in one DRAM transaction and make the same next-hop decision.

`FORWARD_END` marks the final token in a chunk.  It causes an immediate
arrival-counter update, so the downstream reader is not stuck waiting for a
partial counter batch.

---

## 6. The two rings and their flow control

There are two separate rings in the design.

```text
L1 producer/consumer ring (depth 8)

reader claims slot → fills token + routing tail → increments `filled`
sender observes `filled` → sends packet → flushes L1 read → increments `freed`

DRAM forwarding region (one region per stream)

upstream sender writes dense pages → increments downstream `fwd_arrived`
downstream reader waits for pages → consumes chunks sequentially
```

The reader and sender use monotonically increasing counters during a launch;
neither needs a shared read-modify-write.  The sender resets its `filled` and
`freed` counters after consuming `END`; the reader resets the forwarding arrival
counter only after it has consumed its entire region.

The sender sends a final credit-drain sequence before completion.  It ensures
Fabric credits indicate remote receipt of payload packets, rather than merely
that the local worker emitted them.

---

## 7. TILE input adds an untilization pipeline

ROW_MAJOR input goes directly from `dispatched_buffer` to the reader's L1 ring.
TILE input needs rows before a token can be sent, so the operation additionally
launches untilizer cores.

```text
TILED DRAM
   │
   ▼
untilizer dataflow RISC: reads one tile-row (32 tokens) in small tile blocks
   │
   ▼
untilize compute RISC: produces BF16 row-major tokens into its L1 batch ring
   │                         ▲
   └──────── reader fetches only rows required by its stream's schedule ────┘
```

There are two untilizer groups, one for each ring direction.  A group generates
the contiguous tile-row walk needed by its direction and distributes batches
round-robin among its cores.  Readers release batches they skip as well as the
batches they use; otherwise an untilizer could wait forever on an old batch.

Relevant files:

- [`device/kernels/dataflow/combine_fabric2d_group_walk.hpp`](device/kernels/dataflow/combine_fabric2d_group_walk.hpp)
- [`device/kernels/dataflow/untilizer_combine_fabric2d.cpp`](device/kernels/dataflow/untilizer_combine_fabric2d.cpp)
- [`device/kernels/compute/untilize_combine_fabric2d.cpp`](device/kernels/compute/untilize_combine_fabric2d.cpp)

`CMBF2D_UNTILIZERS_PER_GROUP` controls the number of untilizer cores per
direction; its default is 5.

---

## 8. Useful code-reading order

1. [`combine_fabric2d_nanobind.cpp`](combine_fabric2d_nanobind.cpp): semantic contract.
2. [`device/combine_fabric2d_device_operation.cpp`](device/combine_fabric2d_device_operation.cpp): all legal shape/layout/fabric constraints.
3. [`device/combine_fabric2d_assignments.cpp`](device/combine_fabric2d_assignments.cpp): the static routing schedule.
4. [`device/combine_fabric2d_program_factory.cpp`](device/combine_fabric2d_program_factory.cpp): placement, semaphores, forwarding allocation, and kernel construction.
5. [`device/kernels/dataflow/reader_combine_fabric2d.cpp`](device/kernels/dataflow/reader_combine_fabric2d.cpp): routing decisions and DRAM movement.
6. [`device/kernels/dataflow/sender_combine_fabric2d.cpp`](device/kernels/dataflow/sender_combine_fabric2d.cpp): one-hop Fabric transport.
7. The untilizer files above, only when following the TILE path.

## 9. Constraints and present caveats

- The selected mesh axis must have at least four chips and an even extent.
- Inputs must be device, interleaved-DRAM tensors; token data is BF16.
- Metadata is exactly three INT32 values per token.
- `expert_offsets` has one replicated row for every origin chip in the ring.
- The token payload and token-plus-routing-tail must satisfy NoC/DRAM/Fabric
  alignment and maximum-payload checks.
- `topology` is accepted and is part of the operation attributes, but the
  current factory and kernels do not read it.  The actual implementation is the
  wrapped ring described here along `cluster_axis`.

## One-sentence model

**Partition each expert's output by origin chip, attach each token's final output
address, and march it one Fabric hop at a time through per-stream DRAM staging
until it can be written to that address.**
