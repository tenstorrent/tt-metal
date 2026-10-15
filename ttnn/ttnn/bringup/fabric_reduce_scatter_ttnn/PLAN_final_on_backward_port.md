# Plan: drop the final cores, the backward port sums the own block

Goal: the reduce-scatter uses only the port cores (under the Ethernet cores), no final cores on the copy-core row, at no
measurable cost on the even rings the models use (GLM: 2x4 snake ring, G = 8; Galaxy whole mesh, G = 32).

## Why it can fit

On a ring each chip sends G - 1 blocks, forward `a = G // 2`, backward `G - 1 - a` (`_schedule`). For even G that is one
block fewer backward, so the backward port is idle for the last block-time. That is exactly when the final work exists:

- the backward arrival of the own block finished a block-time earlier (the backward link's last delivery);
- the forward arrival of the own block streams in during the last block-time (the forward link's last delivery).

So the backward port can run a final phase after its sends: own partial + forward arrival + backward arrival -> output.

Not a fit (keep today's final cores, selected by an option): odd G (both ports send `(G - 1) / 2`, no idle block-time).

Lines (and G = 2, always a line) fit too but are not built: a line end's outward port sends nothing all op, and a middle
chip p's ports finish their G - 1 - p and p relays as (or before) its own block's arrivals stream in, at block-times
G - p and p + 1, while the op runs G - 1 block-times on the end links.

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

The GLM shape ([1, 1, 5120, 4096] bf16 per chip, 2 links, 8 KiB payload), `test_fabric_rs_model_shapes.py`:

    RS_MESH=2x2 scripts/run_safe_pytest.sh --run-all \
      ttnn/ttnn/bringup/fabric_reduce_scatter_ttnn/tests/unit/test_fabric_rs_model_shapes.py -s

| QuietBox 2x2, device time (busiest chip, median of 20) | us |
|---|---|
| mesh ring, G = 4 (the model's scatter_rows) | 398.2 / 399.1 |
| axis 0 + axis 1 (2 calls, 296 + 158) | 453.4 / 454.1 |
| mesh line | 463.8 / 463.6 |
| axis 1 alone (mla o_proj), G = 2 | 308.3 / 305.3 |

(two runs; the ring moves 2 blocks x 5 MiB per link direction, ~26 GB/s per link.)

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

Result (QuietBox 2x2, GLM shape, device time, two runs each): `final="bwd_port"` 417.4 / 416.9 us vs final cores
396.6 / 396.6 us, +5% (the estimate was +25%). Correct on every 2x2 group (`RS_FINAL=bwd_port` topology test: the
4-chip ring on the backward port, the lines falling back to final cores). The 8-chip ring (4x2) is not run yet.

### Stage 1 ablation: what Stage 2 can buy

Result (2x2, GLM shape, the forward-arrival read deleted, output wrong as it must be: rel 0.98-0.99):

| | with the read | without | the read costs |
|---|---|---|---|
| final cores | 396.6 | 364.8 | 32 us |
| backward port | 417.2 | 354.5 | 63 us |

The read is on the critical path in both designs. Without it the backward port beats today's op by ~10% (no
credits counted yet; Stage 2 also drops the arrival's DRAM write). Go to Stage 2.


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

As built (`final="bwd_port_l1"`, `land_kib`, default 256): c_3 on each backward port is backed by an L1 tensor
height-sharded over every chip's backward ports (one address everywhere); the compute adds c_0 + c_3 + c_2 in place.
The forward sender writes the last block's chunk k into slot k mod cap_land, waiting for its counter B (the credits)
to cover k - cap_land + 1. The backward port's sender returns credits over its own fabric connection: seeing c_16
chunk j + 1 proves the compute popped landing chunk j (it unpacks j + 1 only after popping j), so no timing assumption.

Result (2x2, GLM shape, device time, two runs): 358.4 / 360.6 us vs 397.0 / 397.0 with final cores (-9.5%), within a
few us of the ablation (354.5); correct, every 2x2 group passes with `RS_FINAL=bwd_port_l1`.

| land_kib (chunks) | device us |
|---|---|
| 72 (9) | hang |
| 128 (16) | 605 |
| 256 (32) | 358-361 |
| 512 (64) | 363 |

The reader hands c_3 to the compute a batch (8 chunks) at a time, once B covers the whole batch, and credits lag by
up to a batch: under 2 batches deadlocks, 2 crawls. The host now requires >= 3 batches (192 KiB at 8 KiB chunks).

Not yet run: the 8-chip ring (4x2), and the model itself with `final="bwd_port_l1"`.

## Stage 3 (independent, cheap): final cores next to the ports

If Stage 1/2 don't pay: keep the final cores but place them in the ports' row (8 cores for 1 ring x 2 links), leaving
the rest of the grid contiguous. Measure DRAM-read contention near that row.

## Decision

- Stage 1 within ~2% of Stage 0 at G = 4 and G = 8: done, make `"bwd_port"` the even-ring default.
- Else Stage 1 ablation; Stage 2 if the ablation closes the gap.
- Else Stage 3.
