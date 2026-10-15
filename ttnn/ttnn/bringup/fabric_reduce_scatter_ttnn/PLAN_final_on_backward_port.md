# Plan: drop the final cores, the backward port sums the own block

Goal: the reduce-scatter uses only the port cores (under the Ethernet cores), no final cores on the copy-core row, at no
measurable cost on the even rings the models use (GLM: 2x4 snake ring, G = 8; Galaxy whole mesh, G = 32).

## Why it can fit

On a ring each chip sends G - 1 blocks, forward `a = G // 2`, backward `G - 1 - a` (`_schedule`). For even G that is one
block fewer backward, so the backward port is idle for the last block-time. That is exactly when the final work exists:

- the backward arrival of the own block finished a block-time earlier (the backward link's last delivery);
- the forward arrival of the own block streams in during the last block-time (the forward link's last delivery).

So the backward port can run a final phase after its sends: own partial + forward arrival + backward arrival -> output.

Not a fit (keep today's final cores, selected by an option): odd G (both ports send `(G - 1) / 2`, no idle block-time),
lines (middle chips' ports are busy to the end), G = 2.

## The cost to watch

The port reader is the bottleneck (commit 49a78e2b036: dropping either port read 1565 -> 1260 us). A relay reads 2 DRAM streams per chunk; the final phase reads 3, on the whole lane's banks (today two
final cores split them). Estimate (unmeasured): the final phase runs ~1.5x a block-time, a tail of ~0.5 block-time:
~+25% at G = 4 (2 block-times), ~+12% at G = 8 (4 block-times). G = 4 is the harsher test: fine there, fine at G = 8.

Stage 2 removes the third read by landing the forward arrival in the backward port's L1.

## Stage 0: baseline

Needs a rebuild of this tree first: the current build predates `ttnn.fabric.get_neighbor_eth_directions`
(commit 38481806d09), and `tt_metal/third_party/umd` is checked out at a different commit than HEAD records.

    RS_MESH=2x2 scripts/run_safe_pytest.sh --run-all \
      ttnn/ttnn/bringup/fabric_reduce_scatter_ttnn/tests/unit/test_fabric_reduce_scatter_topologies.py -s -rs \
      -k "mesh_snake_ring or mesh_snake_line"
    RS_MESH=4x2 ...  (same, on the 4x2 box: the G = 8 snake ring)

Record the `FABRIC_RS` lines (device time, busiest chip, if the real-time profiler is active; wall time otherwise),
RS_ROWS 4096 and the GLM shape. Also note links per neighbour pair on the 2x2 (RS_LINKS 1 and 2).

## Stage 1: backward port does the final phase, arrivals in DRAM

Option `final="cores" | "bwd_port"` on `fabric_reduce_scatter` (default `"cores"` until measured); `"bwd_port"` only
for an even ring (G >= 4), else falls back to `"cores"`. With `"bwd_port"`, `_with_final_cores` adds none.

Counters (on the backward port of chip r; its bank set is the lane's, `first = lane`, `stride = lanes`, the same walk
as the senders, so a reader's chunk index matches the sender's 1:1):

- backward own arrival (from r+1's backward port): the tail of the stream that already feeds this port's relays.
  Keep it on counter A, continuing the stream's chunk count. Upstream sender: the `fin` branch targets `relay_noc`
  with the running count (`relay_total = nblocks * full`), no per-half split.
- forward own arrival (from r-1's forward port): counter B on this port (unused on ports today). Upstream sender:
  `fin` targets the peer's backward port, `sem_b`, one counter (no `final_x1` half).
- re-arm at the end: A expects relays + own, B expects `_incs(full)`.

Kernels:

- reader (`_PORT_READER`): after the relay loop, a final loop over the own block: own partial (`in`, own block),
  forward arrival (`scr`, slot own, wait B), backward arrival (`scr`, slot G, wait A continuing), into c_0 / c_1 /
  c_2 with the existing batch and `o0 / o1` offset bookkeeping (add `o2`).
- compute (`_ADD_COMPUTE`): a third phase, `n_final` 3-input adds (the final cores' path), after `n_copy` and `n`.
- sender (`_PORT_SENDER`): after the sends and `conn.close()`, write c_16's final chunks to the output (the
  `_FINAL_WRITER` loop), then re-arm A and B.
- a backward port that is "direct" today (G = 4: its one block leads, the reader feeds c_16 itself) must go through
  the compute's copy phase instead: two kernels producing into c_16 keep separate write pointers.

Host: c_2 on the backward ports; L1 per port 4 CBs x 128 KiB = 512 KiB. Semaphores: the core set is ports only.

Check: `test_fabric_reduce_scatter_topologies.py` (all cases, so the fallbacks run too), `test_fabric_reduce_scatter.py`,
`test_fabric_rs_model_shapes.py`, on 2x2 and 4x2. Measure against Stage 0.

### Stage 1 ablation: what Stage 2 can buy

Same build, the forward-arrival read in the final loop deleted (wrong numbers, timing only; confirm the result goes
wrong, so the read really went). That is the best Stage 2 can reach. If Stage 1 is already within noise of Stage 0,
stop here.

## Stage 2: the forward own arrival lands in the backward port's L1

- Landing buffer: `nslots x slot_bytes` on each backward port core, an L1 sharded tensor over the union of every chip's
  backward port cores (one address on every chip, like the global semaphores), cached like scratch. Back c_1 with it
  (`ttnn.cb_descriptor_from_sharded_tensor`) so the compute reads it in place; fallback: the reader copies slot -> c_1
  with a local L1 read.
- Upstream sender (r-1's forward port), `fin` block only: chunk k goes to `slot(k) = (k / slot_chunks) % nslots` of the
  landing buffer, fused write + increment of B per batch; before starting a slot, wait for its credit
  (`credits >= slots_started - nslots + 1`, a new counter on the forward port, re-armed at the end).
- Receiver (backward port): the reader pushes c_1 pages once B covers them (no read); the final loop reads 2 DRAM
  streams. Credit: when a slot's adds are done, the sender kernel (BRISC, its fabric connection toward r-1 is idle by
  then, keep it open) sends `+1` to r-1's forward port credit counter. The compute pops c_1 before pushing c_16, so the
  writer seeing a c_16 batch means its c_1 batch is free.
- Ready fence: unchanged; r-1's forward port already waits for this port's ready before writing.
- Sizing: link >= ~22 GB/s per direction (4096 x 4096 ring, 388 us), 128 KiB ~ 6 us. Sweep slots 2 x 64, 2 x 128,
  4 x 128 KiB. L1: 512 + 256..512 KiB.

## Stage 3 (independent, cheap): final cores next to the ports

If Stage 1/2 don't pay: keep the final cores but place them in the ports' row (8 cores for 1 ring x 2 links), leaving
the rest of the grid contiguous. Measure DRAM-read contention near that row.

## Decision

- Stage 1 within ~2% of Stage 0 at G = 4 and G = 8: done, make `"bwd_port"` the even-ring default.
- Else Stage 1 ablation; Stage 2 if the ablation closes the gap.
- Else Stage 3.
