# neighbor_pad_conv3d (fused NeighborPad + Conv3d)

Single-dispatch 2D halo exchange fused with Conv3d. Replaces the two-dispatch
(`neighbor_pad_async` then `conv3d`) pattern with one device program that runs
the halo write, halo read, and conv3d compute together — overlapping the
cross-device halo exchange with conv compute instead of serializing them.

Used by the WAN and LTX VAE decoders, where a latent that is spatially sharded
across the mesh (height and/or width) must be padded with one row/column of
halo from each neighbor device before every 3×3×3 convolution.

## 1. Why fuse

In the two-dispatch path, `neighbor_pad_async` performs the cross-mesh halo
exchange and must fully complete before `conv3d` starts — the halo latency is
entirely exposed and adds to the conv time. But:

- the halo exchange is **fabric NOC traffic**, while the conv is **Tensix
  matmul** — largely disjoint resources, and
- most conv output pixels — the **interior** — have a receptive field that
  never leaves the local device tile, so they do not depend on halo at all.

Fusing lets the conv begin computing the halo-independent work while the halo
is still in flight, hiding NP latency underneath the (usually dominant) conv.
The win is shape-dependent and is selected per-shape by the Python router
(see §6); on shapes where it does not pay, the router falls back to standalone
NP + conv3d.

## 2. Architecture

### 2.1 Core partition

The compute grid is split into two disjoint sets in the program factory:

- **Fabric NP cores** — column 0, `num_links*2` rows for the H exchange (top +
  bottom send directions) plus `pad2_num_links*2` rows for the W exchange when
  2D. These run only the halo-exchange kernels.
- **Conv cores** — `conv3d_core_range = full_grid.subtract(np_fabric_cores)`,
  i.e. everything else (≈102 of 110 cores at `num_links=2`). These run the
  vol2col + matmul conv3d.

The two sets run concurrently from `t=0`. The op completes when both finish.

### 2.2 2D halo exchange (H then W, ordered)

```
  ┌──────────────────┐   fabric writes   ┌──────────────────┐
  │  H-writer cores  │──────────────────▶│  H-top / H-bot   │  (halo buffer
  │  (each device)   │                   │  halo sections   │   on DRAM)
  └──────────────────┘                   └──────────────────┘
           │ barrier_sem (once per dispatch, end-of-kernel)
           ▼
  ┌──────────────────┐   fabric writes   ┌──────────────────┐
  │  W-reader cores  │──────────────────▶│  W-left / W-right│
  │  + paired W-wr   │                   │  halo sections   │
  └──────────────────┘                   └──────────────────┘
           │ per-(region,link) progress sems
           ▼
  ┌──────────────────┐  reads input+halo  ┌──────────────────┐
  │  Conv3d reader   │───────────────────▶│  output tensor   │
  └──────────────────┘                    └──────────────────┘
```

- **H halo** (top/bottom neighbors) is written *during* the H-writer's main
  loop; a single end-of-kernel barrier releases the W phase.
- **W halo** (left/right neighbors) is written *during* the W-reader's main
  loop; the paired W-writer fabric-sends each stick to the neighbor's halo
  buffer. Only W-boundary sticks (`pad2_left + pad2_right` per row) cross the
  fabric — the corners-only optimization.
- **H must complete before W** because the W exchange reads the *corner* halo
  (diagonal neighbor), which is materialized only after the H exchange has
  populated the H-extended W-edge strips. The end-of-H barrier enforces this.
- **Conv3d reader** waits on the per-(region,link) progress sems for the halo
  its block needs, then runs the normal vol2col + matmul over its output blocks.

### 2.3 The fusion mechanism (progress semaphores)

Both fused schemes (§3) share one substrate: per-(region,link) GlobalSemaphores
(one per face region {H-top, H-bot, W-left, W-right} × link) pinned to the L1 of
every conv3d reader core. The NP writers/readers **signal** progress as each
region's halo lands (per-T-batch, at the base blocking's `T_out_block`
granularity); a conv block that touches a device edge **waits** per-T-block on the
(region, owning-link) sems it reads before gathering that block's input. Two
properties make this overlap work:

- A core whose output range touches no device edge needs no halo — `core_needs_halo`
  is false and it **never waits**. This halo-free interior work is what fills the
  NP latency.
- An edge core's wait is per-T-block, not a single barrier, so it gathers early
  `t_out` blocks while later NP batches are still in flight — the hard NP→conv gate
  becomes a pipeline.

The two schemes do **not** differ in this mechanism (both use these sems); they
differ only in how they lay out the conv work to exploit it (§3). See §8.1 for the
exact signalling contract.

### 2.4 vol2col gather

The conv reader gathers the im2col patches (`conv3d_reader_vol2col.cpp` +
`conv3d_vol2col_lib.hpp`). This path is **re-based on the upstream conv3d
gather** (trid-ring) so it tracks main's conv3d optimizations rather than
diverging. The fused conv is **gather-bound** on every shape (see §6, ceiling
4) — the gather is the largest single cost and the primary optimization target.

### 2.5 Config

`NpConv3dConfig` subclasses the upstream `ttnn::experimental::prim::Conv3dConfig`
and adds the fusion-only fields (`halo_last`, `force_spatial_parallel`, the
progress-semaphore addresses, etc.). Standalone conv3d takes the base
`Conv3dConfig` and never reads these fields, so the upstream op is unaffected.

## 3. Overlap schemes

The router (`models/tt_dit/utils/conv3d.py`, keyed per VAE shape) picks one of
two fused schemes via `conv_config`, or falls back to standalone:

Both fused schemes use the §2.3 progress sems; they differ in the conv work
layout, not the gate.

| scheme | `conv_config` flag | wins on | mechanism |
|---|---|---|---|
| **halo_last** | `halo_last=True` | conv-heavy shapes where conv ≫ NP and the boundary fraction is small (s2/s3 res+chg, s4_res 2x4, s1_up 4x8) | temporal blocking (`t_out_parallel>1`); each conv core runs **two passes** — interior blocks first (halo-free, overlap NP), then boundary blocks (per-T wait, usually already satisfied by the end of the interior pass) |
| **force_spatial** | `force_spatial_parallel=True` | high-res, large-boundary shapes (s4_out 2x4, s4_res 4x8) | `t_out_parallel=1` spatial grid-fill, **single pass**; halo-free interior (h,w) cores run immediately, edge cores ramp per-T — the boundary is distributed across the fine grid, never serialized into a per-core tail |
| **standalone** | both False | NP-light or NP-bound shapes (conv_in, low-channel res, upsamplers, conv_out, s4_out 4x8) | fusion's fixed overhead exceeds the NP it could hide |

## 4. Kernels

| kernel | core set | role |
|---|---|---|
| `np_h_reader.cpp` | H fabric | read local H-edge rows, fabric-write to top/bottom neighbor halo |
| `np_phase2_w_reader.cpp` | W fabric | read W-edge sticks (incl. corner), fabric-write to left/right neighbor halo |
| `np_writer.cpp` | fabric | paired writer / local-copy + progress signalling |
| `conv3d_reader_vol2col.cpp` (+ `conv3d_vol2col_lib.hpp`) | conv | vol2col gather of input + halo into the matmul CB |
| `conv3d_compute.cpp` (+ `conv3d_compute_lib.hpp`) | conv | tilize → matmul → reduce → bias → untilize per block |
| `conv3d_writer.cpp` | conv | cross-core reduction + scatter output to DRAM |

The three `conv3d_*` kernels are **forked** from upstream conv3d (the NP
progress-wait and halo-read deltas are interwoven). De-forking them is the
top maintainability item — see §7.1.

## 5. Key design decisions

| decision | why | rejected alternative |
|---|---|---|
| **Imperative program factory** (`Program{}` + `CreateKernel`/CB/Semaphore) | NP needs explicit fabric kernels, GlobalSemaphores, and per-core RT args that conv3d's declarative `ProgramDescriptor` paradigm cannot express; the two cannot trivially merge | reusing conv3d's `create_descriptor` path |
| **Reserve fabric cores for NP only** | keeps the halo-exchange protocol (fabric sends + barriers) off the conv cores, avoiding NOC/handshake interleave hazards | reclaiming the 8 fabric cores for interior conv after NP — deadlocked on the reducer/halo handshake; see §7.5 |
| **Batch H-halo fabric reads** into a dedicated send CB (2 rows) | the per-stick read+barrier path was latency-bound (~18k pairs) | per-stick fabric reads |
| **H-then-W ordering via an end-of-H barrier** | the W exchange reads corner halo that only exists after H has populated the H-extended W strips | unordered / interleaved H+W |
| **Per-(region, link) progress semaphores** | with `num_links>1` a single progress count let the W-reader read across a link boundary too early → a bell-curve W-seam | one global progress count |
| **Two schemes, selected per-shape** | no single scheme wins every shape (the four ceilings in §6) | one scheme everywhere |
| **`NpConv3dConfig` subclasses `Conv3dConfig`** | carries fusion-only fields without modifying the shared upstream struct/nanobind | adding the fields to upstream `Conv3dConfig` |

Two dead-ends worth not re-attempting naively:

- **Border-split** (fabric cores compute the halo-dependent boundary *frame*
  after NP, bulk cores compute the interior) — implemented and reverted as a
  wash: the serialized boundary on ~8 fabric cores did not beat the
  progress-sem overlap.
- **W-halo read batching** (mirroring the H batching) — needs a cross-`outer_dim`
  open batch, which produces a **cross-device deadlock**. Reverted. A
  deadlock-safe batching scheme is still open (§7.4).

## 6. Performance & limits

Measured on BH-LB (device-FW MIN, fused vs standalone). No single scheme wins
every shape — four independent ceilings:

1. **Fixed NP co-residence overhead (~262µs)** — NOC contention + per-stick
   control chatter on the resident fabric, independent of conv size. On
   NP-light shapes it exceeds the entire NP, so standalone wins.
2. **NP-bound shapes (conv < NP)** — you cannot hide a 4.7ms NP behind a 2.9ms
   conv (s4_out 4x8); sequential standalone is the floor.
3. **Boundary-pass serialization (halo_last only)** — halo_last runs each conv
   core's boundary blocks as a second pass after its interior blocks. On high-res
   shapes the boundary frame is large, so this per-core boundary tail dominates.
   force_spatial avoids the tail by pinning `t_out_parallel=1`: the finer spatial
   grid distributes the boundary across many cores and ramps it per-T in one pass.
   Both schemes gate on the same progress sems — the difference is the work
   layout, not gate-vs-gateless.
4. **Conv is gather-bound (~24–39% of FW)** — the vol2col gather must read the
   input; only the redundant/uncoalesced fraction is recoverable, and the
   production split-C_in/small-W blockings cannot engage the coalesced
   bank-major gather without OOMing L1.

**Rule:** fusion wins iff `min(NP, conv) > ~262µs` **and** the hideable work
(interior, or the spatial sweep) actually covers NP. Per-shape selection is the
measured optimum.

Note: the e2e VAE decoder is heavily host-dispatch-bound (device-FW ≈1.5s vs
wall ≈7.5s per decode), so the device-time win from fusion is **latent until
trace mode** collapses dispatch overhead (§7.6).

## 7. Next steps for optimization

Ordered by leverage/effort.

### 7.1 De-fork the conv kernels (maintainability)
The op forks conv3d's reader/compute/writer. The clean fix is to share
upstream conv3d's kernels via a `build_conv3d_program_artifacts(Program&, …,
fused_op_signaler)` seam plus a `Conv3dFusedOpSignaler` hook (compile-time gate
+ RT-arg) — the pattern `all_gather_matmul` / `matmul_reduce_scatter` use (no
`device/kernels/` fork). Requires a cross-op change to upstream conv3d.

### 7.2 Reach the coalesced bank-major gather (largest perf lever)
The conv is gather-bound (§6.4). Upstream conv3d's coalesced bank-major gather
recovers more of that cost, but it is gated off by the production split-C_in
(64-wide) / small-W blockings, and forcing it OOMs L1 (≈5.12MB > 1.57MB).
Needs a blocking redesign or an L1-budget rework so the coalesced path fits.

### 7.3 Cut the fixed NP co-residence overhead (~262µs)
This fixed cost sets the fusion break-even. It is ~112µs of NP-traffic NOC
contention + ~150µs of per-stick control chatter on the resident fabric.
Reducing the per-stick control traffic (coarser batched signalling, fewer
semaphores) would lower the break-even and bring more NP-light shapes into the
win column.

### 7.4 Deadlock-safe W-halo batching
W fabric reads are still per-stick (the naive batched version deadlocks across
devices — §5). A batching scheme that keeps the cross-device dependency
acyclic would cut W-exchange latency the way H batching cut H latency.

### 7.5 Reclaim fabric cores for interior conv
The ~8 fabric cores idle during the conv phase. Giving them **interior-only**
(halo-free) tiles with `c_in_parallel=1` (no cross-core reducer handshake)
sidesteps the two things that deadlocked prior attempts, adding conv throughput
on the otherwise-idle cores.

### 7.6 Deploy under trace mode
The decoder is dispatch-bound; the measured device win does not show in
wall-clock until the decode runs under trace. Wiring the fused path into the
traced decode is what turns the device-time win into an end-to-end win.

## 8. Operational invariants

### 8.1 Progress-sem contract

Progress signalling is always-on; the per-T-batch granularity is the base
blocking's `conv_config.T_out_block` (one signal per `T_out_block` input T
frames), derived in the program factory. Each face region {H-top, H-bot,
W-left, W-right} has its own per-link GlobalSemaphore (`region_progress_sem_addr`,
indexed `[region*num_links + link]`); a conv3d edge tile maps the batch it needs
to the owning link and polls only that sem — race-free across links without
cross-link ordering.

- Host resets the region sems via `reset_global_semaphore_value(..., 0)`
  before each dispatch. They span the full compute grid (`ccl_cores`), so
  conv3d reader cores *are* reset.
- **Receiver-side W-reader only** signals the W region sems per batch (the
  sender has no incoming-data wait; its signal would fire before the remote
  fabric write for that batch had landed). The receiver waits on
  `w_neighbor_sem ≥ (batch + 1) * sticks_per_batch` before each increment —
  fabric in-order delivery guarantees the batch's data is in DRAM by then. The
  H-writer signals the H region sems from `handle_incoming_writes` per batch.
- The conv3d reader does a per-T-block `noc_semaphore_wait_min` on only the
  (region, owning-link) sems its current block reads (H-edge → HT/HB,
  W-edge/corner → WL/WR); each sem has one producer per (region,link) so its
  count is monotonic. A side with no real neighbor passes addr 0 and is skipped.

Per-batch signalling only pays off when `T_out > t_out_parallel` (each conv
core handles multiple `t_out` values); small-T shapes route to standalone.

### 8.2 RTA refresh contract

**Every per-dispatch address must be refreshed in
`NpConv3dMeshWorkloadFactory::override_runtime_arguments`:**

| RTA | Kernel | Index | Source |
|---|---|---|---|
| `input_tensor_address` | NP H-reader (common) | 0 | current input buffer |
| `halo_buffer_addr` | NP H-reader (common) | 1 | ping-pong halo buffer |
| `input_tensor_address` | NP H-writer (common) | 0 | current input buffer |
| `halo_buffer_addr` | NP H-writer (common) | 1 | ping-pong halo buffer |
| `halo_buffer_addr` | NP W-reader (common) | 0 | ping-pong halo buffer |
| `input_buffer->address()` | NP W-reader **per-core** | **10** | current input buffer |
| `halo_buffer_addr` | NP W-writer (common) | 0, 1 | ping-pong halo buffer |
| `input_addr` | Conv3d reader (per-core) | 0 | current input buffer |
| `halo_buffer_addr` | Conv3d reader (per-core) | 12 | ping-pong halo buffer |
| `output_addr` | Conv3d writer (per-core) | 0 | current output buffer |
| `weight_addr` | Conv3d writer (per-core) | 1 | weight buffer |
| `bias_addr` | Conv3d writer (per-core) | 2 | bias buffer |

The W-reader per-core RTA[10] is easy to miss because the W-reader otherwise
uses only common runtime args. Missing it makes the W-reader pull local data
from a stale DRAM address on every dispatch after the first, fabric-writing
garbage into the neighbor's halo buffer and producing a bell-curve seam at the
D/D+1 W boundary in the receiver's output (see commit `d8a939bf1e`).

## 9. Validation

- **Op correctness** — `test_neighbor_pad_conv3d_fused.py` (WAN
  `WanCausalConv3d` harness) and `test_vae_ltx.py::test_ltx_causal_conv3d_fused_vs_standalone`
  (LTX) compare fused vs standalone NP+conv3d; ≥99.999% PCC on the 2x4
  production shapes for both halo_last and force_spatial.
- **Perf** — `test_neighbor_pad_conv3d_fused_perf.py` (per-shape fused-vs-standalone
  device time) and `prof_vae_ltx.py` (LTX e2e decoder device time).
- 4x8 shapes require a 32-device mesh (CI on a 4x8 box).
