# Plan: fabric CCL perf examples and guide

Goal: a set of perf-lab examples plus one guide that together show how to write fast CCL kernels on
Tenstorrent Fabric. The existing catalog (`master.md`) has three all-reduce examples
(`tensix_all_reduce`, `tensix_all_reduce_compute`, `tensix_all_reduce_ring_transport`), but they are
all Tensix-to-Tensix on one chip. These would be the first multi-chip, Fabric examples.

## Perf examples

Each example follows the perf-lab contract: it isolates one lever, measures the variants on device
(realtime profiler), treats correctness as the only pass/fail, and stamps every number with the box,
architecture, fabric config and router config.

| # | Example | Tier | What it shows |
|---|---|---|---|
| 0 | `fabric_link_ceiling` | T1 | The simplest possible fabric kernel: one core on one chip streams a buffer from L1 to one neighbour over one link, in full packets, and a single `ttnn.generic_op` launches it. It reports the throughput that simplest version reaches (one direction, then both), next to the fabric team's microbenchmark number for the same config, so readers see both the baseline and the gap to the ceiling. |
| 0b | `fabric_configs` | T1 | The same one-hop stream as `fabric_link_ceiling`, run under each `FabricConfig` the box supports (`FABRIC_1D_NEIGHBOR_EXCHANGE`, `FABRIC_1D`, `FABRIC_1D_RING`, `FABRIC_2D`, and torus where it applies), plus a two-hop send to show what each config allows. It explains what each config builds (routing mode, forwarding between routers, sender channels) and what that costs in throughput. The golden numbers already show NeighborExchange at 38.4 GB/s vs Linear at 28–29 GB/s for 4096 B packets. |
| 1 | `fabric_packet_size` | T1 | The same bytes sent as 1-tile, 2-tile, 4352 B (default max) and 14 KiB (router config) packets. It shows that the router's cost is per packet (~237 cycles measured on BH FABRIC_2D), not per byte. |
| 2 | `fabric_send_issue` | T1 | A flush-blocking send for every packet vs a ring of pre-built headers with `send_current_slot_non_blocking` and one flush per ring wrap. It isolates how much a worker's issue loop costs on top of the link. |
| 3 | `fabric_signal_cost` | T2 | A separate atomic-inc packet per chunk vs one fused write+atomic-inc packet, and a credit per chunk vs batched credits. It shows that control packets take whole router slots and how much fusing and batching recover. |
| 4 | `fabric_fan_in` | T3 | N producer cores feeding one link in three ways: one core that reads and sends everything, a fabric mux with W = 1/2/4/8 clients, and the port-core pattern (producers stage chunks, one core sends). It measures when a mux helps and what the extra copy costs. |
| 5 | `fabric_links_directions` | T2 | Scaling from one link in one direction, to both directions, to 2 lanes, to all links. It shows how much bandwidth each extra pipe adds and where DRAM or core limits start. |
| 6 | `fabric_sender_noc` | T2 | Worker-to-router writes on NoC0 vs NoC1, with the port core near vs far from its ERISC, with and without other traffic on the Ethernet row. It tests whether the "don't route along the Ethernet row" rule matters for the sender side. |
| 7 | `fabric_dram_feed` | T2 | Senders reading 64 KiB interleaved chunks (all banks per chunk) vs bank-owned reads, where each sender reads consecutive pages from its own bank. It shows how the DRAM read pattern limits how fast packets can be filled. |
| 8 | `fabric_relay` | T3 | Multi-hop relay by store-and-forward through the neighbour's DRAM (the `high_bw_all_gather` approach) vs an L1 landing ring with credits (the `high_bw_all_reduce` approach). It measures the cost of the DRAM round trip against the cost of credit traffic. |
| 9 | `fabric_chain_chunk` | T2 | A chunk-size sweep on a 4-device chain. It shows the trade-off between pipeline fill (about 2(G−1) chunk-times) and per-chunk overhead, and where the optimum sits. |

### Reference ceiling (existing, not ours)

The fabric team's microbenchmark already measures the raw link rate. Every example cites it for its
own config instead of treating our own number as the ceiling.

- Binary: `./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config
  tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_ubench_at_least_2x2_mesh.yaml`
- Goldens: `tests/tt_metal/tt_fabric/test_infra/golden/golden_bandwidth_summary_blackhole_p150_x4.csv`
  (Wormhole T3K and Galaxy files are next to it). Re-run it on the box before citing it.

Golden numbers (Blackhole p150 ×4, unicast write):

| Setup | Packet size | GB/s | Packets/s |
|---|---|---|---|
| NeighborExchange | 2048 / 4096 | 19.2 / 38.4 | 9.4 M |
| Linear | 4096 | 28–29 | 6.8–7.1 M |
| Linear, custom router max payload | 3K / 5K / 8K / 15K | 23.7 / 38.2 / 41.0 / 41.8 | 7.2 M → 2.7 M |
| Fused write + atomic-inc on every packet (NeighborExchange) | 2048 / 4096 | 5.9 / 10.9 in the golden; 17.8 / 35.5 re-measured 2026-09-29 | 2.9 M (golden) |

What these numbers already tell us:

- The ceiling depends on the fabric topology config, so every result has to name its config.
- Throughput is limited by packet rate up to about 5 KiB, then flattens at about 41 GB/s.
- The golden shows a fused atomic-inc on every packet as about 3.5× slower than plain writes, but a re-run on
  2026-09-29 (firmware 19.12.0) measured it at the same speed as plain writes. Trust fresh runs over the golden.
- The GB/s figure is the same for 1–4 links, which reads as a per-link number (not yet confirmed).

### Shared infrastructure (built once)

- A small `fabric_bench` harness: `MeshProgramDescriptor` wiring, fabric connection setup,
  parameterized sender/receiver kernels, and global semaphores.
- Measurement through the realtime profiler, as `test_high_bw_all_reduce_bw_compare.py` already does.
- A result stamp: box, architecture, fabric config (`FABRIC_1D`, `FABRIC_2D`, ...), router config
  (max payload) and number of links.

### Order

1. `fabric_link_ceiling`, `fabric_configs`, `fabric_packet_size`, `fabric_send_issue`. These are
   cheap (`fabric_configs` reuses the `fabric_link_ceiling` kernel), they set the baseline, and they
   cover the largest levers.
2. `fabric_fan_in`.
3. The rest, in any order.

## Guide: "Writing fast CCL kernels"

One document organised as rules. Each rule states the mechanism, links to the example that proves
it, and gives the numbers.

1. **One producer per (link, direction).** A router sender channel accepts exactly one producer. A
   mux provides fan-in; it does not reduce the router's load. (`fabric_fan_in`)
2. **Pay per packet, not per byte.** Send full packets, build headers once, and don't flush per
   packet. (`fabric_packet_size`, `fabric_send_issue`)
3. **Control traffic takes whole packet slots.** Fuse signals into data packets and batch credits.
   (`fabric_signal_cost`)
4. **Every wait loop keeps forwarding credits.** This is the deadlock-freedom rule. It goes in the
   guide as a correctness rule and needs no benchmark.
5. **Use both directions of every link, with one owner per (link, direction).**
   (`fabric_links_directions`)
6. **Receiving is free.** The router writes each packet to its final address itself, so there is no
   demux; send packets straight to where the data is needed. (`fabric_relay`)
7. **Feed from DRAM bank by bank.** (`fabric_dram_feed`)
8. **NoC choice and port placement.** The router writes locally on NoC1; the sender's NoC and the
   port core's position are yours to choose. (`fabric_sender_noc`)
9. **Pipeline with the right chunk size.** (`fabric_chain_chunk`)
10. **Measurement recipe.** Take the ceiling from the fabric microbenchmark for your config and the
    simplest-kernel baseline from `fabric_link_ceiling`, split a port's time into
    slot waits vs sending, and profile a mesh with the realtime profiler.

Case studies: `high_bw_all_reduce` (port cores, credits, L1 relay) and `high_bw_all_gather` (mux,
bank-owned workers, DRAM relay), each with its core-layout table.

The guide is written alongside the examples; each rule gets filled in once its example has numbers.

## Open decisions

1. **Audience and location.** Is the guide for humans (a `tech_reports/` page and/or an update of the
   "CCLs on Tenstorrent Fabric" artifact) or for agents (a `references/*.md` file the eval pipeline
   loads)? Or both?
2. **Hardware.** Only the 2×2 Blackhole QuietBox, or also Wormhole (T3K or Galaxy) for
   cross-architecture numbers?
3. **Router config.** The 14 KiB payload changes the whole fabric instance. Should it be a variant in
   every example, or only in `fabric_packet_size`?

## Enabling op code gen to write CCLs

### Where we are

Three fabric examples are built, measured and in `master.md`: `fabric_link_ceiling` (one core, one link, 48.5
GB/s per link direction on a Blackhole QuietBox), `fabric_gather_pair` (2-chip DRAM → DRAM gather), and
`fabric_all_gather` (a general line / ring / snake / dual-Hamiltonian-cycle all-gather on every fabric config).
On the QuietBox, `fabric_all_gather` beats `high_bw_all_gather` by 1.17–1.28× on every FABRIC_2D variant (72 MiB
bf16, 2 links) and 1.32× on a FABRIC_1D ring. It supports TILE and ROW_MAJOR, any dtype, preallocated output with
a fence between calls, sub-devices and caller-owned semaphores. Its test reports link utilization against the
busiest-hop bound. The Blaze Galaxy CI job that ran `high_bw_all_gather` now runs it (`fabric_all_gather_perf`).

The only CCL that code gen has written so far is `high_bw_all_reduce`. Its self-reflection
(`ttnn/ttnn/operations/high_bw_all_reduce/self_reflection.md`) shows three gaps:

1. **No fabric knowledge.** The design sent a fused write + atomic-inc on every packet, which cost the first
   1.8×. No reference or skill states a fabric rule, and `/perf-ceiling-dm` has no fabric ceiling, so the
   perf phase had nothing to aim at.
2. **Everything hand-rolled.** The kernel library has no fabric helpers, so connection setup, routing, header
   handling and signalling were all written from scratch.
3. **Blind coverage.** A 2×2 box only forms groups of G = 2, so the middle-of-the-line path never ran even
   though `SUPPORTED` claims the axis. The fabric is also dead after a passing `--dev` run.

### Plan

**1. Agent-facing fabric CCL reference** (small effort, biggest immediate win). A `references/*.md` file in
`tt_ops_code_gen`, loaded by the planner and implementer, plus a fabric section in `/perf-ceiling-dm`. Each rule
carries its measured number and points at the example that proves it:

- One port core per (link, direction), placed directly below its Ethernet core (probe the connection on
  device; map translated coordinates through the harvesting masks). Adjacent placement drops two links from
  48.5 to 40.6 / 30.5 GB/s.
- Send full packets (14336 B payload on 1D and TORUS_XY; 8704 B helps plain 2D by ~3% and hurts TORUS_XY by
  18%). Never flush per packet.
- Put the arrival increment on every 8th chunk, not every packet: an increment on every packet halves the
  rate. A relay waits for `chunk / 8 + 1` increments.
- Store-and-forward through the neighbour's output with arrival counters; the relay costs nothing in steady
  state (a 4-chip line runs at the 2-chip rate).
- Chunks are runs of pages that sit consecutively in one DRAM bank, visited round-robin; one copy core per
  link does the local copy.
- Balance even rings: split the shard opposite each chip between the two directions (+24–27% at G = 4; only
  ~3% at G = 32).
- On a 2D torus, a whole-mesh gather should use two edge-disjoint Hamiltonian cycles, so every chip uses all
  four neighbours (the busiest hop halves).
- Fence between calls: each sender tells its peer "you may write into my output" and waits for the same
  before sending; cost not measurable. Count host-side what can be counted host-side (on-device counting
  walks cost ~10 µs per call).
- Which config does what: 1D is fastest for one-hop traffic; TORUS_XY is the best 2D variant (~12% above the
  others); plain FABRIC_2D can still close a ring when the box is physically a cycle.
- Link peak per machine: QuietBox 48.5 GB/s per link direction; Galaxy ≥ 26.9 (from `high_bw_all_gather`'s own
  8-rank gate). Bound: time ≥ busiest-hop bytes / (links × link rate).
- Measured headroom to expect: a 1-link line reaches 98–99% of a link at 16–64 MiB; with 2 links each link
  drops to 80–90%; 2-link rings on 2D configs reach 70–79%.

**2. Building blocks extracted from `fabric_all_gather`** (medium effort), so agents compose instead of
rediscover:

- Kernel side: a fabric port sender (header ring, every-8th-chunk increment, ready fence, 1D/2D one-hop
  routing), a relay reader that waits on the arrival counter, and the bank-run chunk walk.
- Host side: Ethernet-core probing and harvesting-aware port placement, the ring / line / snake /
  dual-cycle planner with balanced schedules, semaphore setup (internal or caller-owned), and `link_load` for
  the bound.

**3. Harness fixes** (medium effort):

- Correctness for G ≥ 3 and torus / Galaxy topologies under tt-emule (shown bit-exact on an emulated 32-chip
  Galaxy; recipe in the `fabric_all_gather` README).
- Record group size and mesh shape as an observed facet, so `SUPPORTED` claims say where they were verified.
- Reset the fabric after `--dev` runs that opened one.
- The perf step reports link utilization against the bound, as `test_fabric_all_gather.py` does, so the
  perf tournament knows how much headroom is left.

**4. A new CCL eval to prove it worked.** Code gen writes an all-gather from scratch with the reference loaded
and the building blocks available. It is scored against `high_bw_all_gather` and `fabric_all_gather` on the
QuietBox matrix, then on the Galaxy job. Reduce-scatter comes next: the same building blocks plus a compute
stage.

### Decisions

- **Where the rules live.** Proposed: agent-facing reference in `tt_ops_code_gen/references/` first; the
  human guide (the "Writing fast CCL kernels" section above) is generated from it.
- **How far the helpers go.** Proposed: an examples-local kernel header and Python module first, upstreamed to
  the kernel library once the API has settled.
- **Which op for the eval.** Proposed: all-gather first, because it has two strong baselines to score against.
