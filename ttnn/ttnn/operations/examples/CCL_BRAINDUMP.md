# CCL braindump (unsorted)

Everything learned while building, debugging, reviewing and measuring `ttnn.experimental.fabric_all_gather` (branch
`mstaletovic/fabric_all_gather_op`, head `1619497e5d2`), to be sorted later into the fabric reference
(`tt_ops_code_gen/references/blackhole-fabric.md`), a CCL reference, op-template rules, test axes and the plan
(`FABRIC_CCL_PLAN.md`).

Each item has a **layer** tag and an **evidence** tag:

| tag | meaning |
|---|---|
| **[FABRIC]** | moving bytes from a worker on one chip into memory on another: routers, packets, links, Ethernet cores |
| **[CCL]** | the collective algorithm on top: rings, who sends what, forwarding, signalling, work split, DRAM traffic |
| **[OP]** | TTNN integration: device operation, program cache, trace, validation, kernels' host/device contract |
| **[TEST]** | what to test and how |
| **[PERF]** | how to measure and what the numbers were |
| **[PROCESS]** | how the work was done, and what went wrong in doing it |
| *measured* | a number from a run (run id or box given) |
| *code* | read from the source (file given) |
| *derived* | worked out from the above, not measured |
| *hypothesis* | believed, not shown |

Terms: see the glossary at the top of `fabric_all_gather_chunk_walk.hpp` (outer slice, valid prefix, page lane,
fabric chunk, CB batch, outgoing shards, shards-arrived / downstream-started counters, local copy core, conversion
block). "Bank" only ever means a physical DRAM bank.

---

## 1. [FABRIC] Fabric facts

1. **The receiving router writes each packet to its final address itself** (DRAM or L1 on its chip); there is no
   receiver kernel. A receiver only needs a semaphore to learn data has landed. *code*
2. **A fused write + atomic increment means "landed".** With `flush = true` the router waits until every
   outstanding write on that receive channel is acknowledged (`NIU_MST_REQS_OUTSTANDING_ID == 0`) before issuing
   the increment. So a counter value guarantees the data is in DRAM. *code:
   `tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp:214`, `flush_write_to_noc_pipeline:96`*
3. **The rate cost of that wait depends on how often you do it.** The old reference says an increment on every
   packet halves the rate (24.5 vs 46.4 GB/s per chip). One per outgoing shard (hundreds of packets) is not
   measurable. *measured (earlier QuietBox runs) + derived*
4. **`send_current_slot_non_blocking` writes payload then header from the worker's L1 over the NoC** and bumps
   the slot pointer. Neither the header nor the source may change until `noc_async_writes_flushed()`. *code:
   `edm_fabric_worker_adapters.hpp:353`*
5. **Packet header pool on Blackhole: `NUM_PACKET_HEADERS = 6 × 2 × 2 = 24` per core, 12 per data-movement
   RISC.** A sender using one header per chunk in flight (≤ 8) fits; nothing asserts the coupling. *code:
   `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:167`*
6. **The max payload is process-wide** (router config at mesh open). An op cannot pick it: the model does. GLM
   uses 6144 B because it is tied to the MoE token migration (`GLM53Config.FABRIC_PAYLOAD_SIZE = EMB_SIZE`, "must
   stay in sync with migration code"). A CCL has to be good at the payload it is given. *code*
7. **A page larger than the payload must be split by the sender** into several packets (the router will
   overflow its slot otherwise). `high_bw_all_gather` does this; `fabric_all_gather` did not until the Codex
   review. *code + Codex finding*
8. **Placement API exists, no probe kernel needed:** `tt_fabric::get_forwarding_eth_core(src, dst, link)` →
   `experimental::Device::get_closest_worker_to_eth_core(device, eth, NOC_1)` →
   `Device::get_worker_noc_hop_distance(device, a, b, NOC_1)`. The old reference's probe-and-map-harvesting
   recipe is obsolete. *code: `fabric_all_gather_factory.cpp:440`*
9. **All Blackhole Ethernet cores are in physical row 1** (columns are harvested, rows are not). So the cores
   nearest the routers are the rows right below; a CCL confined to a sub-device should get **rows**, not a column
   strip. *measured: gather on the two rows below = full-grid speed; on a 3-column strip 1.7–2× slower (QuietBox)*
10. **NoC routing:** NoC0 goes +x then +y, NoC1 −y then −x, both wrapping. Placement measures NoC1 hops because the
    sender talks to its router over NoC1. *code + fact*
11. **Link peaks:** QuietBox 48.5 GB/s per link direction (bare one-hop stream, FABRIC_1D, 14336 B). Galaxy ~27
    GB/s (user-stated; `high_bw_all_gather`'s own Galaxy gate uses ≥ 26.9, which is a *derived* busiest-link
    number of an unbalanced ring, not a per-link measurement). Galaxy has 2 links per edge. *measured / stated*
12. **Same-address guarantee:** a mesh tensor is allocated at the same address on every chip, and a global
    semaphore at the same L1 address on every core of every chip. So a sender can compute the destination address
    with its own chip's accessor and put it in the header. *code (relied on throughout)*
13. **Routing in the header is set once per header:** 2D routes by destination `(mesh_id, chip_id)`, 1D by hop
    count. One header ring per destination, routed at setup. *code: `fabric_all_gather_sender.cpp:26`*
14. **Payload size on the Galaxy barely matters above 6 KiB for these sizes:** 6144 → 9216 / 13824 B gave only
    +3–4% on the GLM KV gather. *measured: CI run 36790983426*
15. **A fused increment's NoC address uses the destination's virtual coordinates;** data addresses were built with
    `noc = 0` in the accessor while the increment used `noc_index`. Works on Blackhole (virtual coordinates are the
    same on both NoCs); worth confirming before Wormhole. *code + hypothesis*

## 2. [CCL] Algorithm

### Rings and schedules
16. **Rank and ring order are separate.** Rank = row-major index in the gather group (fixes the output layout);
    the ring order (snake, Hamiltonian cycle) is how data travels. Every outgoing shard names a rank, so ring order
    never leaks into the output. *code: `build_gather_plan`*
17. **Ring kinds:** axis line or ring (`cluster_axis` 0/1; a 2-chip "ring" is a line); full mesh: a snake from
    `resolve_mesh_ring_plan` (closes iff the mesh is a torus or has an even side, on a fully wired mesh); a torus
    with both sides ≥ 3: **two edge-disjoint Hamiltonian cycles**, so every chip drives all four neighbours. *code*
18. **Hamiltonian search cost:** randomized Warnsdorff for cycle A, accepted when the leftover edges form one cycle;
    fixed seed so every process agrees. 3×3 / 4×4 / 4×8: ≤ 0.03 ms; 8×4: 0.5 ms; 6×6: 0.5 ms; 8×8: 300 ms (21
    attempts). Once per program-cache miss. *measured: `generated/hamiltonian_bench/` on the dev box*
19. **Outgoing shards per ring position** (`outgoing_shards_at_ring_position`): closed even ring G ≥ 4: G/2 each
    way, the last one (the opposite chip) split by **lane half**; closed odd ring: ⌊G/2⌋ forward, the rest
    backward; open line: everything to both ends. Invariant: **forwarded shard k = what upstream sent as k − 1**.
    *code*
20. **Balanced even rings:** busiest link carries G/2 − ½ shards instead of G/2 (G = 4: 1.5 vs 2, about 25%
    less). *derived; measured +24–27% at G = 4, ~3% at G = 32 (older plan doc)*
21. **Busiest-link load model** per algorithm (shards per link direction, divided by links): balanced ring G/2 − ½;
    unbalanced (high_bw) G/2; open line G − 1; two Hamiltonian cycles halve it. Used to report utilization. Can be
    off for algorithms you didn't write (high_bw full grid reported 84–86%). *code: `test_fabric_all_gather_device_time.py::_busiest_link_shards`*

### Forwarding and signalling
22. **Store-and-forward through the neighbour's DRAM output:** the router writes into the same pages of the
    downstream output; the downstream reader later reads them back out to forward. Each hop = 1 DRAM write + 1 DRAM
    read per byte. *code*
23. **Signal granularity is one per outgoing shard per worker.** A forwarded shard can't start until upstream's
    whole shard (in the worker's lanes) has landed: one counter, cheap, but a bubble per hop. A per-CB-batch signal
    would overlap hops better on long rings. *code + hypothesis (unmeasured)*
24. **Empty units still count:** if an outgoing shard has no chunks in a worker's lanes (short prefix, lane half),
    the sender sends a **bare increment**, so "forwarded shard k = counter ≥ k" still holds. *code*
25. **…and the receiver must not wait for an empty unit.** For a non-empty one, its wait necessarily happens before
    the sender's final reset (the sender needs those chunks first). For an empty one there is no such ordering: the
    sender can reset first and the receiver waits forever. Both a deadlock and a race were hit here. *code + bug
    history*
26. **Fence between calls:** each sender first tells the worker that writes into *its* output "I've started", then
    waits for its own downstream's signal. Send-then-wait means no cycle can deadlock. A device runs programs in
    order, so any kernel of the next call running downstream proves the previous call is finished there. Cost: one
    round trip per worker per call. *code*
27. **Reset counters with an atomic −k, never by writing 0:** the neighbour may already have sent the next call's
    signal. *code*
28. **Two destination cores on the same neighbour:** data and shards-arrived increments go to the downstream worker
    of the **same** direction (it forwards); the started signal goes to the worker of the **opposite** direction
    (it writes into us). *code*

### Work split
29. **Page lanes:** lane k of an outer slice = pages k, k + 8, k + 16, … (`page_in_slice % num_dram_banks`). In an
    interleaved tensor each lane is one run in one DRAM bank and different lanes are different banks, whatever the
    slice offset (the offset only rotates which bank). *code + derived*
30. **A fabric chunk = up to `payload / page` consecutive pages of one lane:** one DRAM read, one packet, one remote
    write, because the output keeps the order within an outer slice. Consecutive pages would be 8 banks and 8
    transactions. *code*
31. **Lane ownership:** `first_owned_lane = ring × links + link`, `owned_lane_stride = rings × links`. Galaxy torus,
    2 links: 8 workers, stride 4, 2 lanes each (pages ≡ k mod 4); QuietBox snake, 2 links: 4 workers, stride 2.
    Even load iff rings × links divides 8 (not enforced). *code*
32. **Forward and backward workers of the same (ring, link) own the same lanes:** they send the same own shard
    (on purpose: it has to go both ways), different forwarded shards, and complementary halves of the opposite
    shard. They share two DRAM banks, but every bank pair carries the same load. *code + derived*
33. **One worker per bank (each lane one-way all around the ring) gives no gain:** same per-bank load, same total
    byte-hops (31 per byte at G = 32), longer pipelines, doesn't work on open lines. *derived*
34. **The exact lane-to-worker assignment is arbitrary**; only disjointness, even load and spreading matter. A
    bank-aware assignment (worker reads nearest banks) is possible in principle but the bank of a lane differs
    between input, local output and remote output. *derived, unmeasured*
35. **Walk order rotates across a worker's lanes** (outer slice → row in lane → owned lanes), so consecutive DRAM
    reads hit different banks. *code*
36. **Short outer slices shrink chunks:** a chunk can't cross an outer slice (the next slice is another rank's
    region in the output), so a chunk is ≤ ceil(pages_per_outer_slice / 8) pages. Width-32 tile gathers (one tile
    per tile row) have only lane 0: one worker per direction does everything, 2 KiB packets. Misalignment itself
    costs nothing. *code + derived*

### DRAM
37. **DRAM is not the main limiter (QuietBox, 18 MiB tile, 14 KiB payload):** removing all DRAM reads: 368 → 336
    µs (−8.6%), busiest link 79% → 87%. Remote writes moved to one L1 endpoint: −3% (confounded: concentrates
    writes on one core). Small gathers: −2%. So ~13 points of link peak are lost elsewhere (routers, per-packet
    overhead, per-batch flush, fence). *measured: ablation, `generated/ablate_*.log`, today*
38. **Forwarding from L1 (receive into a CB, forward from there)** would save at most ~9% for all-gather. Not worth
    a protocol redesign for all-gather. **Required for reduce-scatter / all-reduce**, where data must be reduced
    before forwarding. *derived from 37*
39. **Total DRAM traffic ≈ 2 × link traffic** (every hop writes and reads). Rough Galaxy estimate 216 GB/s reads +
    216 GB/s writes against ~512 GB/s aggregate DRAM. *derived, unmeasured*

### Local copy and ND-sharded input
40. **Keep the local copy off the sending cores:** separate local copy cores (one per link) write the own shard into
    the own output. *measured earlier (2-chip gather): copy on the sending core 32.6 vs 48.2 GB/s at 1 link; 57 vs 95
    GB/s per chip at 2 links*
41. **Non-interleaved input (ND-sharded GLM KV cache: 32 rows per bank shard) has no bank-strided contiguity**, so
    the local copy cores **convert** the own shard into the interleaved output, and the link workers read it from
    there. *code*
42. **Conversion is block-cyclic** (copy core c takes blocks c, c + n, …), with a **per-block semaphore** on every
    link worker, so the own shard starts moving after the first block rather than after the whole conversion. A
    link worker waits only for the blocks the chunk it reads touches. *code*
43. **The reader passes run descriptors to the writer** (`flat_page`, `num_pages | last_run_of_block`) so the writer
    doesn't redo the ND address probing. The writer precomputes the 8 bank base addresses. *code*
44. **Flush mid-block, barrier at the block's end:** batches are freed after `noc_async_writes_flushed`; only the
    block's last run waits for the writes to land, then signals. *code*
45. **Tuning:** 2 copy cores per link best at 6 KiB (4 total on QuietBox); 6 and 8 worse at 6 KiB; 2 CB batches per
    block (1 → 2: 995 → 964 µs at 32k rows). *measured: QuietBox*
46. **Output must be interleaved DRAM** in both ops; ND-sharded output would need a final local conversion (or
    chunking by destination contiguity). *code*

## 3. [OP] TTNN integration

47. **Drop-in replacement = derive the device operation** from the original: parameters, validation, program hash,
    output spec shared; only `program_factory_t` / `select_program_factory` differ. The framework mixes
    `type_hash<DeviceOp>` into the cache key, so the two ops never share programs; the op name in profiler reports
    is the derived type. Removed ~750 duplicated lines. *code: `ttnn/api/ttnn/device_operation.hpp:73`*
48. **Trace-safe runtime values:** values patched into runtime args on the host are frozen in a captured trace
    (silent corruption in chunked prefill). Read them from a tensor on device instead, call
    `invalidate_l1_cache()` after the read (the host rewrites the tensor at the same address), and give each RISC
    its own landing CB (they read at the same time). *code + two earlier bugs*
49. **Hash presence, not values:** hashing the per-call values (slot, prefix, layer) forks programs; hashing the
    layer index would compile one program per layer, each with global semaphores, exhausting GLM's 1152 B
    L1_SMALL. *code: high_bw device op comment*
50. **Validate caller-owned semaphores on the first call**, not only on rebinding (coverage of the cores that use
    them). *Codex finding*
51. **Partial extents along a tiled dim must be tile-aligned or rejected** (a 16-row prefix of a 32-row tile was
    silently rounded). *Codex finding*
52. **Local NoC writes larger than 16 KiB (`NOC_MAX_BURST_SIZE`) can't use `noc_async_write_one_packet`** (ND-sharded
    32 KiB rows hung). *Codex finding, reproduced*
53. **CB batches never cross the end of the CB** (contiguous in L1); push at the end of every unit; never block
    while holding read data (push first). *code*
54. **Common runtime args are the only thing a cache hit patches** (addresses, slot, valid length); everything
    structural is compile-time. The sender's common args are zero-padded so the geometry block sits at the same
    index in every kernel. *code*
55. **Program semaphores reset at every launch**, global semaphores don't (hence the atomic −k resets). *code*
56. **Same kernel source, two variants** (link worker, local copy core) via `if constexpr` compile flags: one file,
    no dead code in either binary. *code*
57. **Counting on device is expensive:** walking a shard just to count chunks cost ~10 µs per call; a closed-form
    count must match the walk exactly (the last chunk of each shard carries the increment). *measured earlier + code*
58. **Naming:** one glossary at the top of the kernel header; logical (indices) vs physical (bank, address, row)
    words kept apart; "bank" only for DRAM banks. Mechanical renames done with a code-only script (skip comments
    and strings), then comments by hand. *process*

## 4. [TEST]

59. **Default CCL test axes** (each caught or would have caught a real bug): ring size 2 / odd / even ≥ 4; line vs
    ring; interleaved vs ND-sharded input; page < payload, page > payload, page > 16 KiB; a short prefix so some
    workers get empty shards; a trace **captured once, then replayed with changed metadata**; tile-misaligned
    rejection; external semaphores on a sub-device strip. *process*
60. **A trace test that captures a new trace per case and warms up first proves nothing** (the warm-up already
    writes the expected output). *Codex finding*
61. **tt-emule** runs an emulated 32-chip Galaxy bit-exact (no timing), but cannot create sub-device managers and
    needs fast dispatch for traces. *measured*
62. **Ablation recipe:** temporary kernel edits (kernels JIT-compile, no rebuild), timing-only test, 3 reps,
    restore with `git checkout`. No-DRAM-reads = `#define noc_async_read(...)` no-op after the includes. Watch for
    confounds (moving writes to one L1 endpoint concentrates traffic). *process: `generated/ablate_dram.py`*
63. **A test that fails on the old code:** before trusting a new regression test, run it against the old kernel
    (the 32 KiB ND test hung on the old copy writer). *process*

## 5. [PERF]

64. **Report per case:** GB/s received per chip, busiest-link GB/s and its % of the per-machine link peak (per the
    busiest-link model of the algorithm). *process*
65. **Same-commit A/B, median of 3.** Comparing against an older baseline mixes in main's changes: the "Kimi got
    faster" speedup was the fused RMSNorm (#56108), not this op. *process + Kimi investigation*
66. **Perf CI can be silently confounded by data:** the Kimi perf test falls back to synthetic tokens when the golden
    trace is missing (it moved from weka scratch to stable); that read as a 20–30% "regression" for a whole day.
    Gated perf configs should fail when their data is missing. *measured: CI runs 2026-09-29/30*
67. **The sparse-MLA perf test (`test_sparse_mla_perf.py`) measures but doesn't gate, and skips itself when
    `CI=true`.** On the QuietBox the overlap window is top-k bound, so the gather difference is hidden (long: 10.6
    ms either op). Galaxy run pending (36907969181, throwaway branch). *measured + code*
68. **Key numbers (Galaxy, CI 36790983426 / 36800328403):** 18 MiB tile 184.6 vs 94.0 GB/s/chip (1.96×), link 85%;
    GLM KV bf16 56k 417 vs 844 µs (2.02×), 517k 3374 vs 7621 µs; overlap window 56k 568 vs 933 µs, 517k 4455 vs 7996
    µs (top-k bound with ours, gather bound with high_bw). QuietBox: KV 1728 rows ND 65–67 µs vs high_bw 83 µs. *measured*

## 6. [PROCESS]

69. **An independent reviewer finds real bugs:** two Codex rounds raised 9 findings, 8 acted on: payload overflow,
    NoC-burst hang, tile alignment, semaphore coverage on first use, search-budget wrap, a trace test that proved
    nothing, a perf helper swallowing errors, a stale docstring (the 9th, the snake-closure model, was not a bug).
    Run one before calling a CCL done.
    Codex's own sandbox fails on this box; `codex exec --dangerously-bypass-approvals-and-sandbox` with read-only
    instructions works. *process*
70. **CI infra:** runner pool `multihost-ci-sc1-lc6lw` repeatedly failed chip resets (no test ran); high-power
    Galaxy pool queues can exceed 2.5 h; rerun only the failed job once the run completes. *measured*
71. **Pipeline YAML lint** keeps a baseline of jobs using `|| rc=1`; restructuring a job's command (a loop) trips it.
    *measured*
72. **Kernels in a worktree:** export `TT_METAL_HOME` and `PYTHONPATH` to the worktree, and install (`build_metal.sh`)
    after host changes; `build/lib` holds install copies. *process*

## 7. Generalization notes (for the plan)

73. **What transfers between collectives is the host-side schedule:** rings, and per (ring, direction, link) an
    ordered list of `(shard, lane subset, action)` plus placement. All-gather: send/forward. Reduce-scatter: same
    rings, receive → reduce → send. All-reduce: RS + AG. All-to-all / broadcast: different lists, same transport.
74. **Kernel blocks that transfer:** lane walk; batched reader; fabric port sender (header ring, fused increment on a
    unit's last packet, large-page split, flush discipline); the counter protocol (per-unit increment, bare increment
    for empty units + receiver skip, started fence, atomic −k reset); block-wise conversion of non-interleaved input.
75. **Two transports are needed:** DRAM store-and-forward (all-gather; output is the forwarding buffer) and L1 landing
    with credits (anything that computes on the way: RS, AR). Which to use is a CCL decision, not a fabric one.

## 8. Open questions

76. Where do the remaining ~13 points of link peak go (QuietBox, no-DRAM ablation still 87%)? Next ablation: sender
    skips the fabric sends (flush and pop only) to separate the core-side loop from router/link.
77. Galaxy sparse-MLA block perf, ours vs high_bw at 512k (run 36907969181 queued).
78. Per-batch forwarding signals vs per-shard on 32-chip rings: worth it?
79. Assert the header-pool coupling (`chunks_per_cb_batch ≤ headers per RISC`).
80. Reword the Kimi perf baseline comments (they credit this op; the gain is #56108).
81. Bank-aware lane assignment: measurable or not?
82. Do the old reference's rules 4 (every-8th increment) and 6 (probe placement) get replaced by items 2–3 and 8–9?
