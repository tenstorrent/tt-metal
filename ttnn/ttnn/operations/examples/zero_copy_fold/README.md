# zero_copy_fold

**Trick (op-agnostic):** don't assume "fewer kernels = less overhead." A Tensix core has independent
RISC-Vs — a dataflow **reader** on NCRISC and **writer** on BRISC run *concurrently* with **compute**
on TRISC. A reader that arms a circular buffer and a writer that drains one therefore overlap with the
compute between them. **Folding** that arm/drain into the single compute kernel (`compute_only`)
*serializes* it onto the compute thread and loses the overlap — so the three-kernel
`reader_compute_writer` structure is **faster**, not slower, despite "more kernels."

This is a property of program structure, not of any one op. The gap is a fixed per-launch cost
(~75–130 ns here) that no NoC work is there to hide, so it dominates on **small** work-per-core and
amortizes as tiles/core grows.

**Payload is incidental.** To isolate pure program structure, the example uses a same-spec zero-copy
sharded tilize: input and output shard specs are identical, so both CBs alias directly onto the
resident L1 shard buffers (`cb_descriptor_from_sharded_tensor`) — **no DRAM, no NoC** — leaving only
the reader/compute/writer structure around a small `tilize_block`. Any op with a reader/compute/writer
split would show the same effect; the tilize is just the cleanest thing to measure it on.

**Consider this pattern** when deciding whether to merge dataflow kernels into compute: keep them
separate unless you have measured that the reader/writer RISCs are idle *and* the handshake cost
actually dominates. It is **not** a "make the kernel faster" trick — it only matters for how
arm/drain overlaps compute.

## Variants measured

- `reader_compute_writer` (baseline) — 3 kernels: dataflow reader (NCRISC) arms the input CB, compute
  (TRISC) tilizes, dataflow writer (BRISC) drains the output CB. Arm/drain overlap the tilize.
- `compute_only` (candidate) — 1 compute kernel that self-arms the input CB and self-drains the output
  CB. Arm/drain are serialized onto the compute thread.

Same tilize, same aliased CBs — the **only** difference is program structure.

## Result

Folding is **slower** across the board (~0.74× at 2 tiles/core, closing to ~0.95× at 64 tiles/core on
Wormhole B0). See [`report.md`](report.md) for the measured table (numbers are stamped with the box +
arch they were profiled on). Correctness is the only pass/fail; the durations are measured and
explained, never asserted.

## Run it

```bash
python -m ttnn.operations.examples.zero_copy_fold            # default sweep
python -m ttnn.operations.examples.zero_copy_fold --shape 128 128 2 --trials 5
```
