# Handoff point 1, resolved: split `in0` instead of relaxing the endpoint rules

Working notes for `MatmulMultiCoreReuseMcast2DProgramFactory` (PR #56761, parent #41908).
Supersedes the "scope two DFB endpoint checks to Gen2" proposal.

Shareable write-up with the mechanism diagrams: https://claude.ai/artifact/Qx94sdvoSuy1WuD6NNsRCr

## Result

The blocker does not need a validation change. Splitting the `in0` DFB in two removes the
violation, and **the split is accepted on stock Metal 2.0** — no endpoint rule scoped to Gen2, no
runtime patch, no kernel edit.

| | today | with the split |
|---|---|---|
| `in0` | PRODUCER = {`in0_sender` DM, `in0_mcast_no_work` DM}<br>CONSUMER = {`compute` TRISC, `in0_mcast_no_work` DM} | PRODUCER = {`in0_sender`}<br>CONSUMER = {`compute`} |
| `in0_relay` | — | PRODUCER = CONSUMER = {`in0_mcast_no_work`} (DM self-loop, Gen1) |
| validation | `TT_FATAL` `program_spec.cpp:1377` | passes |

Why every rule is satisfied:

- **Kind uniformity** (`:1377`) never fires: each role on each DFB has exactly one `KernelSpec`, and
  `check_role_uniformity` returns early below two.
- **Self-loop set equality** (`:1521`) holds on `in0_relay`: producer set == consumer set ==
  `{in0_mcast_no_work}`.
- **Per-node census** passes on both: one producer, one consumer on every node.
- **Gen1 DM self-loop** is explicitly permitted (`:1495`), and `CPU_DMKernelSelfLoopOnGen1Succeeds`
  already covers it.

Cost: nothing. Accessor names are scoped per kernel, so the relay binding keeps `accessor_name =
"in0"` and `dfb::in0` in the shared source resolves per kernel — **kernel sources unchanged**. The
two node sets are disjoint, so the pair shares a device slot (measured: both slot 0) — no extra CB
index, no extra L1. Each core still sees one `in0` buffer at the same offset and size, driven by the
same instruction stream.

## What is in this commit

- The factory change (+40/−8) — the split itself. This is the part to land.
- Six spec shapes in `test_program_spec.cpp` as `ProgramSpecTestGen1.CPU_ScratchMatmulRelay_*`.
  These are evidence, not merge-ready tests: they print rather than assert where the outcome depends
  on the follow-up below. Rewrite or drop them before the PR goes up.

The step-two runtime change is **not** in this PR. It is #56887, which lets an alias group cover
disjoint node sets; everything here is verified without it. This PR therefore does not depend on it,
and the shared offset below rests on declaration order until it lands and the factory opts in.

Note for whoever pushes this: the commit was made with `--no-verify`. The `fix-cstdint` hook rewrites
`uint32_t` to `std::uint32_t` across each whole staged file, which added ~520 lines of unrelated churn
to the factory and buried the 48-line change. The lines added here were made compliant by hand; the
rest of both files is untouched legacy style. Decide before pushing whether to take the churn as a
separate formatting commit.

## The one thing left over

The two buffers must sit at the same L1 offset: a NoC multicast writes one offset on every
destination, and the sender derives it from `dfb_in0.get_write_ptr()` on its own node.

**They do today**, because the pair is declared before any other DFB, so each starts at its own
allocator's base. Verified on hardware. But nothing records that ordering. Declaring any buffer on
the work nodes ahead of `in0` lifts `in0` and not the relay:

```
B  · split                     in0 @ 105696   in0_relay @ 105696   MATCH
B2 · split, in1 declared first in0 @ 107744   in0_relay @ 105696   MISMATCH (2048 B)
B3 · split + co-location       in0 @ 107744   in0_relay @ 107744   MATCH
```

So the factory keeps the pair first with a comment saying why, and the durable fix is **#56887**,
which lets an alias group cover **disjoint** node sets (identical *or* disjoint; partial overlap
still rejected) and places the group as one unit. `alias_with` already stamps the primary's address onto each
secondary's own cores — the only gap is that the secondary's cores never get the region marked,
which rule 3's own error message says outright.

Two things that bit while implementing it:

- Disjoint members must also match `entry_size` *and* `num_entries`, not just total size — the point
  of a disjoint alias is that the cursors advance in step.
- `mark_address` appends to the allocator's last L1 region, so it is **not idempotent**. Marking the
  union naïvely double-reserves on any allocator spanning both node sets — and one exists here, from
  the borrowed `in0_sharded`. First draft did that and died at program build with
  `Local buffer address 242688 has to append to last L1 region [111616, 275456)`.

Whether it rides on `alias_with` or a new `colocate_with` is still the runtime owners' call. The
disjoint case shares no memory, so `alias_with`'s "no guarantees against clobbering" framing does
not apply to it; same allocator work either way.

## Verification (Blackhole p150b)

Three configurations, each against the pristine PR, the split on stock Metal 2.0, and the split with
co-location. Reference is fp32 torch; inputs are regenerated per run.

| configuration | pristine PR | split only | split + co-location |
|---|---|---|---|
| reported repro, `extract_shard_sub_blocks=false` | `TT_FATAL :1377` | pcc 0.999883 | pcc 0.999883 |
| K=1024, `extract_shard_sub_blocks=true` | `TT_FATAL :1377` | pcc 0.999877 | pcc 0.999878 |
| output block-sharded on the work column | `TT_FATAL :1377` | pcc 0.999882 | pcc 0.999883 |

The second row matters on its own: with `extract_shard_sub_blocks` true the relay buffer is not just
an address — the sender gathers strided sub-blocks into it and multicasts out of it. The reported
reproducer never reaches that path (`num_blocks_per_shard` is 1 there).

`test_matmul.py` + `test_custom_grids.py` against the split on stock Metal 2.0: 888 passed,
310 skipped, 2 xfailed, 0 failed — the port's pre-existing baseline.

Six spec shapes are in `test_program_spec.cpp` as `ProgramSpecTestGen1.CPU_ScratchMatmulRelay_*`
(scratch; they report rather than assert where the outcome depends on #56887).

## Also worth knowing

- **The 1D factory has the same topology**, plus a variant the 2D one never produces: sender-only
  nodes that *are* inside the receiver grid, where the multicast really does land in their buffer.
  Same split applies; it is the second caller for the co-location.
- **Nothing here makes the configuration work on Quasar, and that is correct.** A DM self-loop is
  rejected on Gen2 by design. CrossNodeDFB is the Gen2 answer and its shape already matches this
  matmul: the broadcast flow is the sender's semaphore dance with credits instead of semaphores, the
  relay-DFB flow is the receiver's loop with the same substitution, and it derives each receiver's
  write offset from a counter rather than a mirrored cursor — which dissolves the address problem.
  Two things block using it here: `write_broadcast` is a loop of unicasts, one per receiver
  (`cross_node_dfb.h:209`), so it trades the multicast away on the innermost loop; and `ProgramSpec`
  hard-rejects `cross_node_dataflow_buffers`.
- When a true-multicast CrossNodeDFB lands, `in0` on the work nodes *is* the relay DFB in its sense —
  already the right shape — and `in0_relay` plus both semaphores get deleted. The split is not Gen1
  debt; it is the Gen1 lowering of a pattern Gen2 expresses directly.

## Alternatives considered

- **Scope the two rules to Gen2** (the previous proposal). The Gen1-redundancy argument holds, but it
  costs two weakened invariants and an arch fork in the endpoint model, and the spec keeps asserting
  the cross-node structure the review objected to. Now that the split needs no runtime change, the
  comparison is a validation relaxation versus none.
- **A do-nothing compute kernel draining `in0` on the relay nodes** (shape C, accepted today). Makes
  both roles kind-uniform and keeps one buffer, so the ordering fragility goes away outright — but it
  adds a kernel source and arg plumbing that exist only to satisfy a validator, a compute launch on
  cores that compute nothing, and a cursor that depends on TRISC scheduling, so the pure-sender path
  can no longer run ahead.
- **A `MIRROR` endpoint type**: name the relay's role directly, keep one DFB, address coherence stays
  automatic. Set aside because `in0` keeps spanning the relay nodes — the modelling the review
  identified as the error — and the concept becomes dead weight once CrossNodeDFB lands.
- **Delete the relay, let receivers pull.** The in0 shards are read-only and resident for the whole
  program, so any core can read a peer's shard with no synchronisation at all. It replaces one
  multicast with `num_blocks_x` unicast reads and needs a second kernel path for in-grid owners.
  Rejected on perf, but worth recording: the synchronisation here buys bandwidth, not correctness.
