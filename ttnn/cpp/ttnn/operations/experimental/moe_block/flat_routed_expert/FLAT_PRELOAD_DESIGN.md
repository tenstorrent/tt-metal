# Flat routed expert: preloading the first experts during dispatch

Status: design + measured upper bound (emulation). Nothing is implemented outside the flat expert's measurement
switches (`FLAT_PRE_W`, `FLAT_PRE_X`). Branch `mstaletovic/custom-mm-full-in0-wip`; numbers from one Blackhole p150b,
GLM-5.3 (H 6144, I 2048, E 256, K 8) at 16 experts a chip (the 4x4 stage), `test_flat_routed_expert_checks.py` perf
medians (5 runs) through `scripts/run_safe_pytest.sh`.

## 1. Summary

The flat routed expert loses its first ~45-100 us to a cold start: every gate/up core waits for its first expert's
weights (one DRAM burst from all 64 cores at once) and for the first x (the x pipeline - helper DRAM reads, tilize,
landing, primary relay, then down a 40-core snake - takes 35-50 us to reach the snake's tail). At low M the whole op
is DRAM-bound, so every byte read inside the op's window costs time.

The MoE block allocates one L1 arena before `moe_head` and frees it after `routed_experts_combine`; the flat expert's
rings live in it, and `sparse_dispatch_2d`, which runs in between, does not use it. So the rings sit reserved and idle
for the whole dispatch. Filling them during dispatch - the first two loads' gate/up and down weights, and (at
dispatch's end) the first sub-block of x - takes that work out of the flat expert's window.

Upper bound, measured by emulating a perfect preload in the flat expert alone (GLM-5.3 4x4, us):

| | M 80 | M 160 | M 320 | M 480 |
|---|---|---|---|---|
| today | 798.8 | 883.3 | 1053.4 | 1375.2 |
| weights preloaded (2 loads) | 728.4 (-8.8%) | 831.4 (-5.9%) | 1008.2 (-4.3%) | 1322.5 (-3.8%) |
| x preloaded (first sub-block) | 791.3 | 896.0 | 1057.9 | 1365.7 |
| weights + x preloaded | **710.2 (-11.1%)** | **831.8 (-5.8%)** | **994.7 (-5.6%)** | **1315.5 (-4.3%)** |

The weights are the bulk of it and need nothing from dispatch but time (their order follows from the counts, which
exist before dispatch starts). x helps only together with the weights (alone the first expert just waits for its
weights instead) and only overlaps dispatch's tail.

## 2. Background

### 2.1 The block's ops and the arena

A forward (`models/demos/deepseek_v3_d_p/tt/moe_block/moe_block.py`, `MoEBlock`):

1. allocate the arena (`moe_block_arena_spec`: every worker core, height-sharded L1, max(the combine's need, the
   head's arena bytes) a core)
2. `moe_head(..., arena=)` - router, routing planner (writes the per-chip **counts, regions, token index** into the
   resources' tensors), the head's own work in the arena
3. `sparse_dispatch_2d(xbuf, route, resources)` - every chip's rows to the chips holding their experts (fabric), into
   `xbuf` (DRAM). No arena; its CBs are static, below the arena (the head plans its arena above the flat expert's and
   dispatch's CBs, `head_cb_floor`)
4. `routed_experts_combine(..., arena=)` - the flat routed expert and the combine in one program, in the arena
5. free the arena; `moe_tail`

So between step 2 and step 4 the arena's contents on the flat expert's cores belong to nobody: the head is done with
them and the combine has not started.

### 2.2 The flat expert's L1 (per role, offsets inside the arena)

- gate/up cores (64): the gate/up weight ring at offset 0 - `gate_up_weight_ring_blocks` (2 loads: 2 x 24 K-blocks of
  8 K x 2 tiles, bfp4; 442 KB at GLM-5.3) - then the x ring (`x_ring_offset`; 28 four-row x blocks, bfp8, ~975 KB), the
  h_local and pack CBs, the schedule page and the dynamic counts.
- down cores (32): the down weight ring at offset 0 (two experts with pinning, else 1.5), the h buffers, the output.
- relays (4: 2 primaries, 2 helpers): row-major chunks, tilized super-blocks, landing rings.

### 2.3 The schedule (`kernels/expert_schedule.hpp`)

Every flat-expert kernel builds the same per-call schedule from the per-chip counts (`load_schedule`): one entry and
one load per active expert, **most sub-blocks first** (ties: local expert order); with pinning (`pin_chunk_sub_blocks`),
a dominant expert is cut into chunks with the cold experts interleaved, its load staying in one ring region. Loads
land in the gate/up ring in load order, one region each (`assign_ring_regions`: load l in the region freed by the
(l - R)-th load to retire; plain order: region l % R, R = 2). So the first two loads - which experts, which regions -
are a pure function of the counts. The counts are written by `moe_head`'s routing planner, i.e. they exist before
dispatch starts.

### 2.4 Where the start goes today (M 320, probes `FLAT_ZZ_PROBES=1`)

- expert 0 takes ~100 us a gate/up core, the next experts ~50-60 us (x wait 22.7 us mean / 46.2 max, weight wait 32.9
  / 46.4)
- expert 0's x wait grows along the snake: 9.5-21.5 us at the west snake's head column, 29-46 us at its tail column
  (the pipeline's fill)
- the first weight block reaches the compute at ~19 us, the first load's last at ~34 us even with the weights
  preloaded through the old grant path (the startup grant cap: 8 blocks until the first x is in)
- at M 80 the op moves 366 MB in ~800 us (~465 GB/s, pure-read ceiling ~510): DRAM-bound; the first two loads' weights
  are ~42 MB of it (gate/up 2 x 14.2 MB, down 2 x 7.1 MB), ~90 us of DRAM time

## 3. The emulation (how the upper bound was measured)

Measurement switches in the flat expert only (`kernels/zz_probes.hpp` `zz::pre_w`, `zz::pre_x`; host env
`FLAT_PRE_W`, `FLAT_PRE_X`). The data is garbage (accuracy fails by design); the timing is that of a perfect preload.

- `FLAT_PRE_W=n` (n <= 2, the ring): the first n loads' gate/up blocks are in the ring at the start - the receiver
  (`gate_up_core_reader.cpp`) pushes them into the weight CB and sets the landed count before its loop, grants only the
  free slots on top; the reader (`gate_up_weight_read.hpp`) starts at the first non-preloaded chunk. The down reader
  skips the first n loads' blocks' reads (they still go through the down ring's CB protocol).
- `FLAT_PRE_X=n`: the first n x sub-blocks (x_row_tiles rows each: n = 2 is the first 4-row sub-block) are in every
  gate/up core's x ring at the start. The relays skip them entirely (the reader neither reads nor pushes them, the
  tilizer's count shrinks, the helpers neither land nor count them, the primaries start their landing counts and ring
  slots past them); every gate/up core's receiver counts them arrived, its forwarder counts them issued and forwarded.

Both run hang-free under the watcher + NoC sanitizer.

Two pitfalls hit while measuring, kept here because they change numbers:

- A first `FLAT_PRE_W` published the preloaded weights through the normal grant -> land -> publish loop: with the
  startup grant cap that took ~36 us to get the first load to the compute with no DRAM reads at all (and ~60 us with
  x preloaded too). A real preload must hand the blocks to the compute at the start (the current emulation does).
- Combinations of switches were first run as `env $e` under zsh, which does not word-split: only the first variable
  was set. Every combined number here was re-run with `env ${=e}` (the worklog marks the corrected ones).

## 4. Results in detail

### 4.1 Weights

Preloading the ring's two loads removes ~42 MB from the op's DRAM window and the first expert's weight wait:
-70 / -52 / -45 / -53 us at M 80 / 160 / 320 / 480. At M 80 that is roughly the bytes' DRAM time at the op's rate;
at M 320+ it is mostly the startup weight wait.

### 4.2 x alone: no gain

With the first sub-block of x preloaded and the weights not:

- most gate/up cores no longer wait for x in expert 0 (1.6 us), but expert 0's weight wait rises 32.9 -> 43.7 us
  and its span does not move (99.5 -> 102.4 us): the start is now weight-bound.
- the cores whose weights come early finish the preloaded sub-block in ~19 us and wait for sub-block 1, which still
  takes the pipeline's 35-50 us to reach the snake's tail (their x wait keeps the tail-column gradient: 36-49 us).

A deeper x preload would need ~2-3 sub-blocks (all of expert 0 at M 320), but the x ring holds ~1.2 sub-blocks.

### 4.3 Both

With both, expert 0 waits for neither at the start and the x pipeline has a sub-block of compute to fill: -1 to -2.5%
on top of the weights (M 80 728 -> 710, M 320 1008 -> 995, M 480 1323 -> 1316).

## 5. Design

### 5.1 Where the preload runs

The window is `sparse_dispatch_2d`. `moe_head` also runs in the arena; whether its regions overlap the flat expert's
rings is not checked here (the block plan places the head's arena "above the flat and dispatch CBs", `head_cb_floor`)
- if they do not, the head's window could carry part of the preload too. Not the combine program's start: that is the
cold start itself.

Options:

1. **Prefetch kernels in the dispatch program** (preferred): `sparse_dispatch_2d` is one program for the mesh merged
   from every chip's descriptor; add, on each chip, one data-movement kernel on every flat gate/up core and every flat
   down core that are not dispatch port cores (to check: the port cores are the cells nearest their link's Ethernet
   core; the flat plan's cores are fixed by the plan - if they overlap, those cores skip the preload and the flat
   expert reads their blocks as today, see 5.4). The kernel reads the counts, builds the schedule (`load_schedule`,
   the same header), and reads its share of the first two loads into its ring, exactly as the flat expert's readers
   would (`gate_up_weight_read.hpp`'s addressing for gate/up; `down_weight_reader.cpp`'s block order for down). It
   ends with its reads' barrier, so the program's completion implies the blocks have landed.
2. A separate small program between dispatch and the combine: simpler to build, but it serializes with both (no
   overlap) - only useful as a stepping stone for correctness.

### 5.2 What the preload needs from the block

- the flat expert's plan (cores, weight slots, ring offsets, block sizes), the gate/up and down weight tensors'
  addresses, the counts' address - the dispatch op does not see any of these today; the block passes a preload spec
  (from `moe_block_plan` / the flat expert's plan and the weights' buffers) to `sparse_dispatch_2d`, which adds the
  kernels when it is present. The weights' and counts' addresses go through the same cache-hit runtime-arg patch as
  the dispatch's own addresses.
- the arena's base address (the block's `arena_address`), since the rings are arena offsets.

### 5.3 The handover to the flat expert

The flat expert gets a `preloaded_loads` option (0 / 1 / 2) and, per call, nothing else: it recomputes the same
schedule from the same counts, so it knows which loads and ring regions are already there. With it:

- gate/up receiver: pushes the preloaded loads' blocks into the weight CB and sets the landed count at its start,
  grants only the free slots after them (what `FLAT_PRE_W` does today).
- gate/up reader: starts at the first non-preloaded chunk.
- down reader: treats the preloaded loads' blocks as landed in its ring (its ring holds two experts with pinning; the
  preload fills it the way the reader would have, in the same order).
- the startup grant cap (`gate_up_w_start`: 8 slots until the first x is in) applies to the reads after the preload.

Fewer than two active experts: the preload and the flat expert both take min(2, loads). Pinned schedules: the
pinned load is load 0 and stays in its region; the preload fills regions in the schedule's region order, so it
follows. A hot expert split into chunks is still one load (gate/up); its down weights are re-read a chunk unless
`down_weights_per_load` - the down preload covers the first chunk's read.

### 5.4 Partial preloads

Per core, a core whose preload did not run (a port core, a fallback) must not be told its blocks are there: the
`preloaded_loads` count is per call and per chip, so either every flat core of the chip gets the preload or none. If
the port cores overlap the flat cores, the overlapping cores' share is read by a neighbour (same bank) into the
port-core's ring over the NoC, or the preload is dropped for that chip configuration.

### 5.5 x (second phase)

The first sub-block of expert 0 (its first 128 rows) as bfp8 tiles in every gate/up core's x ring (offset
`x_ring_offset`, the slots the first x sub-blocks would take). Its rows are in `xbuf` only once dispatch has
delivered them (rows from other chips arrive throughout), so it can only start at dispatch's end: e.g. dispatch's
final fence followed by a tilize on two idle cores and one multicast per gate/up rectangle (each core holds the same
x: ~835 KB of bfp8 a rectangle). It serializes after dispatch unless dispatch's arrival tracking can tell when expert
0's first 128 rows are complete. Worth +1 to -2.5% on top of the weights in the emulation; build it after the weights
and only if its own time at dispatch's end stays well below that.

### 5.6 Correctness hooks

- The flat expert asserts (watcher builds) that its schedule's first loads match a small descriptor the preload kernel
  writes next to the ring (expert ids, region) - a mismatch means the two schedules diverged.
- Determinism and accuracy tests run the block with and without the preload; the outputs must be bit-identical.

## 6. Risks and open questions

- **Dispatch's DRAM headroom.** Dispatch writes the rows into `xbuf` (DRAM) and is fabric-bound on a Galaxy. ~42 MB of
  weight reads during it must not slow it; on a p150 there is no fabric to measure this - it needs a Galaxy run.
- **Dispatch's duration vs ~90 us of reads.** If dispatch is shorter, the preload spills into the combine's start
  (still correct: the flat expert waits for the program to end - but it then serializes; a partial preload of one load
  is the fallback).
- **NoC contention with dispatch's port cores** (the weight reads' responses cross the grid on NoC0).
- **Core overlap** between dispatch's port cores and the flat plan's gate/up / down cores (5.4).
- **API coupling**: dispatch takes a spec from the flat expert; the block owns both, so the coupling stays inside
  `moe_block`.
- **Combine overlap**: the combine's landing pool uses the arena L1 the flat expert leaves; the preloaded rings are the
  flat expert's own regions, so no change - but `x_ring_fill` (off in the combine) must stay off for the x preload's
  slots to be where the flat expert expects them.

## 7. Plan

1. Flat expert: `preloaded_loads` (and the startup descriptor check) replacing the `FLAT_PRE_W` emulation; unit test
   that fills the rings from a host-side helper program and checks bit-identical outputs.
2. Dispatch: the preload kernels behind an option; block plumbing (the spec, the addresses, the cache-hit patch).
3. Galaxy: the block's end-to-end time with / without, dispatch's own time with / without.
4. Only then the x phase (5.5).

## 8. Reproducing the numbers

```
FLAT_PRE_W=2 scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/experimental/moe_block/flat_routed_expert/test_flat_routed_expert_checks.py -k "perf and glm53-4x4"
FLAT_PRE_X=2 FLAT_PRE_W=2 scripts/run_safe_pytest.sh --run-all ... -k "perf and glm53-4x4"
```

(the 2560 / 15360 chunks - M 80 / 480 - are extra cases added for this study; the suite has 5120 / 7680 / 10240.)
Startup timeline probes (`FLAT_ZZ_PROBES=1` + `TT_METAL_DEVICE_PROFILER=1`): `ZZ-T-W0` / `ZZ-T-WL` (the compute's
first / first load's last weight block), `ZZ-T-RPUSH0` / `ZZ-T-RPUSHL` (the receiver's pushes), `ZZ-GU-EX` / `EW` /
`ET` (per-expert x wait, weight wait, span).
