# routed_experts_combine

`ttnn.experimental.moe_block.routed_experts_combine`: the MoE block's routed experts and its combine, fused into one
op that runs both at the same time.

## What it does

After the dispatch, every chip holds the token rows routed to its experts. This op:

1. runs those experts (the flat routed expert, `flat_routed_expert/`), and
2. sends every expert output row back to its token's home chip and adds it into that token's result there, weighted
   by the row's gate weight.

The two overlap: an output row leaves for its home chip as soon as the expert has produced it, straight from L1 (no
round trip through DRAM), while the experts keep computing the next rows.

The usual "post-combine reduce" is folded in as well. A plain combine would bring each token's top-k expert outputs
home as k separate rows, and a second op would compute `sum_k gate_k * y_k`. Here that sum is accumulated as the rows
arrive, so the output is already one row per token:

    out[token] = sum over the token's routed experts e of  gate_e * y_e[token]      (fp32 sums, bf16 out)

## Data flow

```
 expert chip                                                   home chip of the token
 +------------------------------------------+                  +--------------------------------------------+
 | flat expert y writer --(L1)--> port -----+--- 2D fabric --->| landing slot --> accumulator (token % A)   |
 |        |                                 |                  |                  reader -> compute -> out  |
 |        +--(L1, token homed here)--> accumulator             |                  fp32 sum += gate * row    |
 +------------------------------------------+                  +--------------------------------------------+
```

Each row carries a 32-byte trailer with its token, home chip, gate weight and arrival flags
(`device/kernels/dataflow/combine_formats.hpp`).

## Determinism (`combine_snap`)

Rows from different chips arrive in an order that depends on fabric timing, which changes from run to run. fp32
addition is not associative, so summing the same rows in a different order can give a different result in the last
bits: without care, the op would not be bit-reproducible.

`combine_snap` (a power of two q; on by default, q = 2^-14) rounds every weighted row to a multiple of q before it is
added. Sums of multiples of q are exact in fp32 (while they stay below 2^24 q), so the result is the same in any
arrival order.

## Tensix core types (per chip)

* **Flat expert cores**: run the experts (`flat_routed_expert/README.md`). Their y writers hand each output row to a
  port, or to an accumulator when the token lives on this chip.
* **Ports**: one per fabric direction and link, next to that link's Ethernet core. A port sends its rows over the
  fabric to their home chips and returns flow-control messages for the rows its own chip consumed.
* **Accumulators**: A per chip; token t is summed on accumulator t % A. Each has a reader (collects the landed rows),
  a compute kernel (the weighted fp32 sum, `kernels/compute/accumulator_compute.cpp`) and a writer (the output rows).

The planner (`combine_auto`) picks the arrangement that fits the L1 the experts leave free:

* accumulators on cores of their own, or on the ports' cores;
* the landing slots in the accumulators' L1, or in the free L1 of the flat expert cores (`CombineLanding`);
* flow control by **rings** (each accumulator gives every source chip a ring of slots and returns credits as it
  consumes them) or by a **shared landing** (each accumulator grants single slots to sources on demand).

## Program and per-call state

The op builds one program for the whole mesh (each chip's program, merged by `common/chip_program.hpp`; one program
per chip if they differ). Its L1 working area is allocated per call, and the program zeroes the combine's state at its
start and drains every in-flight message before it ends, so nothing it sends lands after the program has finished.
On a program-cache hit only buffer addresses are rewritten.

## Formats

* `device/kernels/dataflow/combine_formats.hpp`: a routed row and its trailer, the header word, the ring layout and
  grant words, the MISC words of a combine core's state.
* `common/kernels/combine_plan_words.hpp`: the device plan (rows per port, accumulator and source chip) the MoE head
  writes and the combine reads.

## Resources (`MoEBlockResources`)

`routed_experts_combine/state` (the plan, computed once), `routed_experts_combine/bar` (barrier and zeroing words on
every worker core), `routed_experts_combine/hcache` (the ports' routed fabric headers, DRAM),
`routed_experts_combine/lut` (per-chip core roles), `routed_experts_combine/out` (the output, persistent),
`routed_experts_combine/y` (the flat expert's DRAM y, unused with the L1 hand-off),
`routed_experts_combine/dispatch_sems` (the dispatch's semaphore words); the flat expert's own through
`ensure_flat_routed_expert_resources`.

## Tests

* `test_routed_experts_combine_host.py`: the credit policy and `combine_auto`'s invariants; no device.
* `test_experts_combine_checks.py`: device (Galaxy), the op on the real MoE head's and dispatch's outputs, snap
  2^-14, uniform and zipf routing (GLM-5.3 / Kimi-K3 / DeepSeek V4.1-Flash): accuracy vs torch, determinism, a re-routing
  (uniform -> zipf -> uniform, the second uniform bit-identical to the first) and traced device time.
