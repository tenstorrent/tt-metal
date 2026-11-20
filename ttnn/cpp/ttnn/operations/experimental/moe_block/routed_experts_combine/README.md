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

    out[token] = sum over the token's routed experts e of  gate_e * y_e[token]      (int32 sums of q, bf16 out)

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

`combine_snap` (a power of two q; on by default, q = 2^-14) turns every weighted row into whole multiples of q, and the
accumulators sum those as int32 counts of q: the SFPU rounds each term v to round(v / q) (ties to even) as a
sign-magnitude int32, and the packer adds it into the token's sum in L1 with its INT32 L1 accumulation
([tt-isa-documentation, packer accumulation](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/Packers/Accumulation.md)).
Integer adds are exact, so the result is the same in any arrival order, up to (2^31 - 1) q (2^17 at q = 2^-14; the
fp32 sums of multiples of q this replaced were exact only below 2^24 q). Past that the L1 add saturates at
+-(2^31 - 1) and stays there (Blackhole treats the largest magnitude as an infinity), and a single term is clamped to
+-(2^31 - 128) q (inf / NaN too), so an overflowing token reads out as about +-2^31 q, never a wrapped value. The output is the count cast to fp32, times q, rounded to bf16.

With `combine_snap` 0 the sums are fp32 (FP32 L1 accumulation), order-dependent in the last bits.

## Tensix core types (per chip)

* **Flat expert cores**: run the experts (`flat_routed_expert/README.md`). Their y writers hand each output row to a
  port, or to an accumulator when the token lives on this chip.
* **Ports**: one per fabric direction and link, next to that link's Ethernet core. A port sends its rows over the
  fabric to their home chips and returns flow-control messages for the rows its own chip consumed.
* **Accumulators**: A per chip; token t is summed on accumulator t % A. Each has a reader (collects the landed rows),
  a compute kernel (the weighted sum, `kernels/compute/accumulator_compute.cpp`) and a writer (the output rows).

The planner (`combine_auto`) picks the arrangement that fits the L1 the experts leave free:

* accumulators on cores of their own, or on the ports' cores;
* the landing slots in the accumulators' L1, or in the free L1 of the flat expert cores (`CombineLanding`);
* flow control by **rings** (each accumulator gives every source chip a ring of slots and returns credits as it
  consumes them) or by a **shared landing** (each accumulator grants single slots to sources on demand).

## Compile-time configuration

All named: the combine's options (`kernels/dataflow/combine_options.hpp`) and each kernel's own values, with the names
in one table the host and the kernels share (`kernels/combine_names.hpp`); no preprocessor defines. The only positional
compile-time args are the `TensorAccessorArgs` chains (offset-based), each named by the index of its first arg. A core
that is a port on some chips and an accumulator on others runs both roles in one binary
(`combine_port_or_accumulator_reader.cpp` / `_writer.cpp`): the roles' named args merge, equal names with equal values.

## Per-call state

The op needs no setup beyond its resources: it zeroes its own state at the start of every call and waits for all
in-flight messages before it ends, so nothing it sends lands after the call has finished.

## Formats

* `device/kernels/dataflow/combine_formats.hpp`: a routed row and its trailer, the header word, the ring layout and
  grant words, the MISC words of a combine core's state.
* `common/kernels/combine_plan_words.hpp`: the device plan (rows per port, accumulator and source chip) the MoE head
  writes and the combine reads.

## Resources (`MoEBlockResources`)

The op's persistent buffers (its plan, output, synchronization words and the flat expert's own) are owned by
`MoEBlockResources`, so they are created once and reused by every call.

## Tests

* `test_routed_experts_combine_host.py`: the credit policy and `combine_auto`'s invariants; no device.
* `test_experts_combine_checks.py`: device (Galaxy), the op on the real MoE head's and dispatch's outputs, snap
  2^-14, uniform and zipf routing (GLM-5.3 / Kimi-K3 / DeepSeek V4.1-Flash): accuracy vs torch, determinism, a re-routing
  (uniform -> zipf -> uniform, the second uniform bit-identical to the first) and traced device time.
