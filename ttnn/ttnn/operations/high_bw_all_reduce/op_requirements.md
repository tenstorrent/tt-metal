# Operation Requirements: high_bw_all_reduce

## Definition
- **Formula**: for every device `d` of the mesh, `out[d] = Σ_{d' ∈ group(d)} in[d']` (elementwise SUM). The
  group is chosen by `cluster_axis`: `0` gives the mesh column, `1` gives the mesh row, and `None` gives the
  whole mesh. Each reduction step accumulates in fp32.
- **PyTorch Reference**:
  ```python
  def ref(stacked, cluster_axis):          # stacked: (R, C, *shape), stacked[r, c] = device (r, c)
      dims = (0, 1) if cluster_axis is None else (cluster_axis,)
      return stacked.float().sum(dim=dims, keepdim=True).expand_as(stacked.float()).to(stacked.dtype)
  ```
- **Import Path**: `from ttnn.operations.high_bw_all_reduce import high_bw_all_reduce`
- **Function Signature** (fixed by the spec, do not extend):
  ```python
  high_bw_all_reduce(input_tensor: ttnn.Tensor, *, cluster_axis: int | None,
                     topology: Topology = Topology.Linear, num_links: int | None = None,
                     memory_config: ttnn.MemoryConfig | None = None) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Test-infra note (all phases)**: a passing `--dev` run with FABRIC_2D can leave the ethernet routers dead. Run `touch /tmp/tt-device.dirty` before every `run_safe_pytest.sh` / `tt-probe.sh` invocation so the script resets the board under its device lock. Run one fabric test module per session, because a second module-scoped mesh in the same `--dev` session fails at open.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = tile_aligned only
- **SUPPORTED op-specific axes**: cluster_axis ∈ {0, 1}, topology ∈ {Linear}, num_links ∈ {1, 2} (None means every usable link)
- **Regime**: R1 `chain_line`, plus R1-solo for `G == 1`
- **Cores**: per device, `num_lanes × (W + 1)`, where `W = 4` reducers plus 1 port core per lane (10 cores at 2 links)
- **Compute config**: `fp32_dest_acc_en=True` (hard-wired by the spec's fp32-accumulation rule), FPU add
- **Golden baseline**: 48 supported_pass / 336 xfail_expected / 0 loud (plus 4 of 16 regression tests passing; the other 12 are Phase 0 refusals)
- **Perf baseline (BH 2×2)**: 8 MB takes 392 µs at 1 link and 203 µs at 2 links. 32 MB takes 1.53 ms and 0.77 ms. That is ≈ 21 GB/s per link direction against a ≈ 38.4 GB/s plain-write ceiling.

### [x] Refinement 1 — Ring topology + whole-mesh snake (`cluster_axis=None`)

**Goal**: add `Topology.Ring` to `SUPPORTED["topology"]` and `None` to `SUPPORTED["cluster_axis"]`. This
lands two regimes from `op_design.md`:
- **R3 `rotated_chain_ring`** (scheme-change). With `G` slices per lane, slice `j`'s chain starts at ring
  position `j` and ends at `j−1 (mod G)`. Every link, including the wrap/closing edge, carries partials
  one way and finals the other: `(G−1)/G · S` per link direction, using both ring directions.
- **R2 `chain_snake_line`** (knob-turn). `_group_path` emits a row- or column-snake Hamiltonian path
  whose every edge is a direct one-hop link.

For `cluster_axis=None` with Ring, the path is a snake Hamiltonian **cycle** over plain FABRIC_2D, which
needs an even mesh dimension. Along an axis, the ring closes over the wrap link (FABRIC_2D_TORUS_Y wraps
axis 0, TORUS_X wraps axis 1, TORUS_XY wraps both).

**Verifier notes**:
- **Ordered first**: this is the only scheme-change in the queue. It reshapes the port kernels (per-slice
  roles and chunk ranges, `num_slices = G`) and the host path builder. Refinement 2's dtype and alignment
  work then extends that structure once, instead of being reworked underneath.
- On a 2×2 every Ring cell has `cluster_axis=None`, so the snake line and the snake cycle land together
  here. Per-axis rings are only emitted on a torus Galaxy.
- **No implementation skill covers this** (cross-core Fabric topology). The topology is the work: per-slice
  role table (`head` / `middle` / `tail` per slice), per-slice chunk ranges within each lane, and the
  existing one `port_fwd` + one `port_bwd` per lane. The EDM cap is still one worker per (link,
  direction), so both directions' slices are multiplexed on those two connections.
- Prompt rules re-arm here:
  - MUST route `None` over direct one-hop links only, with no multi-hop forwarding.
  - Prefer both ring directions; the rotated chains satisfy this.
  - A Ring request the cluster cannot satisfy is `ValueError`, not a support refusal: Ring along an axis
    whose fabric config does not wrap it, or a `None` Ring with no even mesh dimension.
  - `num_links` must be ≤ the usable links on **every** edge of the chosen path, including snake corners
    and the wrap edge, with no clamping.
- `_global_semaphores` currently keys on a hard-coded `"Linear"`. Key on the real topology and on the path
  kind, or Linear and Ring configs will share cumulative counters across invocations.
- **This is the first time the `middle` role runs on the 2×2 box** (`G = 4` snake). Add `G ≥ 3` exactness
  tests (rank-identity and single-contributor for `cluster_axis=None`, Linear and Ring) and a
  back-to-back Linear↔Ring alternation test for counter hygiene.
- Build it performantly, not correct-only: Refinement 4 tunes this path. Every lane's reducers must be
  busy on every slice, both ring directions must stream concurrently, and the pipeline must keep
  chunk-granular credits and the header ring from Phase 0.

**Done when**:
- Every `cluster_axis=None` golden cell (Linear and Ring) and every per-axis Ring cell the cluster emits
  passes for bf16 tile_aligned.
- The new exactness tests pass.
- Measured device-ns shows None-Ring faster than None-Linear at 32 MB. This is the `(G−1)/G` vs `1`
  per-link-direction traffic claim, confirming both directions carry data.
- Phase 0 cells and perf do not regress.

**Outcome**:
- **SUPPORTED**: `cluster_axis` gains `None` and `topology` gains `Ring`.
- **Per-axis Ring**: `{Ring, 0}` and `{Ring, 1}` are in EXCLUSIONS. They need a torus cluster, and the 2×2 box
  emits no such cell, so none could be verified.
- **Golden**: all 48 None cells (bf16, tile_aligned) pass.
- **Device-ns at 1 / 2 links**:

  | Case | 32 MB | 8 MB |
  |---|---|---|
  | None-Ring | 1.52 / 0.93 ms | 412 / 270 µs |
  | None-Linear | 1.79 / 1.16 ms | 469 / 344 µs |
  | Phase 0 axis lines | 1.53 / 0.77 ms (unchanged) | 392 / 204 µs (unchanged) |

- **Bottleneck**: the ring moves 24 MB per link direction in 1.52 ms, about 16 GB/s, against about 21 GB/s on
  the axis line. Every ring port core relays partials and finals at once (the "port co-location" lamp). The
  snake middles pay the same cost, which is why None-Linear at 2 links is 1.5× slower than an axis line.
- **Next**: split `port_fwd` and `port_bwd` onto separate cores, and deepen `FINAL_DEPTH_PER_REDUCER`.
  Ring perf is sensitive to how fast final credits come back. Both belong to Refinements 3 and 4.

### [x] Refinement 2 — float32 end-to-end + non-aligned shapes

**Goal**:
- Add `ttnn.float32` to `SUPPORTED["dtype"]`, with fp32 moved and accumulated end-to-end on every path
  (R1 line, R1-solo, the R3 ring and the R2 snake).
- Add `"w_non_aligned"` and `"h_non_aligned"` to `SUPPORTED["alignment"]`.

**Implementation skill**: /numeric-formats-metal, /memory-layouts

**Verifier notes**:
- **fp32 compute path (design Mechanism caps).** The FPU binary add reads SrcA/SrcB as tf32, which
  truncates fp32. Switch to SFPU `binary_sfpu<AddBinary<>, …>` with `UnpackToDestFp32` on
  `cb_remote_partial` and `cb_local_input`, and keep a `copy` for the head. Keep the chunk-granular
  lifecycle (`Upfront` / `AtEnd` + `InputTileMapping::Block`) from the verifier pass.
- **fp32 pages and wire.** Every CB page and the wire are Float32. The prompt says MUST NOT downcast
  partials to bf16 on the wire. `tile_bytes = 4096` gives `packet_tiles = 1` at the default 4352 B
  payload and `chunk_tiles = 16` at `CHUNK_BYTES_TARGET = 64 KiB`, so the L1 footprint is unchanged
  (every capacity is `∝ chunk_bytes`).
- **Signature is fixed by the spec.** Do **not** add a `compute_kernel_config` kwarg (the skill would
  normally expose one). `fp32_dest_acc_en=True` stays hard-wired by the fp32-accumulation rule. Only the
  dtype and intermediate-CB precision parts of the skill apply.
- **Alignment is SUPPORTED + tests, with zero kernel change.** A verifier probe (`probes/probe_023.py`)
  ran four non-aligned shapes on axes 0/1 with `validate()` bypassed. All returned the exact logical
  shape within 1 bf16 ULP. The op sums physical tile pages, and the output reuses the input TensorSpec,
  so padding never reaches the logical view (prompt rule "MUST NOT leak padding"). Add non-aligned
  exactness tests (rank-identity on 4096×2050 and 4001×2048) to pin it.
- Ordered after R1 so fp32 and alignment extend the final multi-slice structure once. There is no hard
  dependency, but the fp32 packet geometry (`packet_tiles = 1`, twice the packets per byte) must be
  exercised on the ring and snake paths too. Cells that fail for a structural reason go in `EXCLUSIONS`,
  never in their own refinement.

**Done when**:
- Every `dtype=FLOAT32` and every `alignment ∈ {w_non_aligned, h_non_aligned}` golden cell passes, across
  all (cluster_axis, topology) cells R1 enabled.
- All 16 `test_regression.py` tests pass, including `test_magnitude` in fp32 and the 4096×2050 exactness
  tests.
- Zero loud categories in `verify_supported`.

**Outcome**:
- **SUPPORTED**: `dtype` gains `float32`; `alignment` gains `w_non_aligned` and `h_non_aligned`. No
  EXCLUSIONS were added.
- **fp32 path**: a CT flag selects the SFPU add (`binary_sfpu<AddBinary<>>`) on `UnpackToDestFp32`
  operands. The bf16 FPU path is unchanged.
- **Golden**: 304/304 on the `FLOAT32 or non_aligned or test_regression` slice, covering every fp32
  cell, every non-aligned cell and all 16 regression tests.
- **Non-aligned**: no kernel change was needed.

### [x] Refinement 3 — Drive the axis lines toward link rate (bf16, 8–64 MB)

**Type**: perf

**Goal**:
- **Target**: the op's defining regime. That is the bf16, tile-aligned, `topology=Linear`,
  `cluster_axis ∈ {0, 1}` cells at 8–64 MB per device, `num_links ∈ {1, 2}`. The primary shape is
  `1×1×4096×4096` (32 MB), at 1.53 ms (1 link) / 0.77 ms (2 links).
- **Headroom**: that is ≈ 21 GB/s per link direction. The BH fabric golden
  (`tests/tt_metal/tt_fabric/test_infra/golden/golden_bandwidth_summary_blackhole_p150_x4.csv`) puts
  plain `NOC_UNICAST_WRITE` at 4 KiB packets at ≈ 38.4 GB/s, so the op runs at ~56% of link rate.
- **Candidate levers**, from the design's Perf lamps and the relevant `ttnn/ttnn/operations/examples/master.md`
  patterns (`noc_placement`, `double_buffer`):
  - **Port co-location.** `port_fwd` and `port_bwd` share one core, whose NoC carries staging-in,
    EDM-out, final-landing-in, DRAM-out and EDM-out, about 5 streams at link rate. Split them onto
    separate cores (+1 core per lane).
  - **Port placement / NoC selection.** Put ports on the ethernet-adjacent row and try swapping the port
    kernels' NoCs.
  - **Depth and granularity co-tune.** `STAGING_DEPTH_PER_REDUCER` / `FINAL_DEPTH_PER_REDUCER` = 2,
    `MAX_DATA_HEADERS`, and `CHUNK_BYTES_TARGET` × `REDUCERS_PER_LANE`. The W=8 × 64 KiB L1 collision
    noted in `l1_ledger.md` must be resolved by re-placing the port scratch, not by shrinking blocks
    below whole-chunk granularity.
  - **The reducer→port staging hop.** It is the single above-minimum buffer.
- **Measure the ceiling first.** Before tuning, a send-only variant of the port loop answers "is the gap
  in the port, the EDM, or the reducers?" (`/perf-ceiling-dm` covers only the NoC side). The chunk floor
  is whole tiles, and coarser blocks amortize handshakes. No SUPPORTED change.

**Done when**:
- Measured device-ns improves on `1×1×4096×4096` and `1×1×2048×2048` at both `num_links=1` and `2`,
  moving GB/s per link direction toward the ≈ 38 GB/s ceiling.
- The golden suite is green and the precision baseline is unchanged.
- There is no regression across the config-spanning guard set: one representative each for Linear axis
  0 and axis 1, num_links 1 and 2, bf16 and fp32, None-Linear and None-Ring, a ragged tail shape, and the
  single-tile shape.

**Outcome**:
- **Ceiling, measured first**: a send-only port loop (`probes/perf_ceiling_fabric.py`) sends 32 MB
  in 1.44 ms. That is about 237 cycles per 4 KiB packet, or 23.3 GB/s per link direction, on BH
  FABRIC_2D. The 38 GB/s golden is 1-D NeighborExchange and is not reachable here.
- **What bound the op**: the credit packets. Adding one header-only credit per 16 data packets
  brings the send-only stream to +5.5%, exactly the op's 1.53 ms. Credits take a full EDM slot.
  The port co-location, reducer and staging hops were not on the critical path of the axis lines.
- **Lever: credit coalescing** (`CREDIT_BATCH_CHUNKS = 4`) on both credit streams. Ring depths
  derive from it (`effective + batch − 1`). They fit L1 because one shared per-call scratch
  replaces the separate reducer and port tensors plus program CBs (max, not sum). The batch is
  clamped to the largest free L1 block.
- **Axis lines, device ns, 1 / 2 links** (32 MB is 22.9 / 22.6 GB/s per link direction, 98% of
  the send-only ceiling):

  | Shape | Before | After | Change |
  |---|---|---|---|
  | 32 MB | 1.530 / 0.773 ms | 1.467 / 0.741 ms | −4.1% / −4.1% |
  | 8 MB | 394 / 204 µs | 377 / 198 µs | −4.1% / −3.2% |

- **Side wins at 1 link**:
  - None-Ring: 1.51 → 1.38 ms (−8.7%).
  - None-Linear: 1.78 → 1.74 ms (−2.2%).
  - fp32 16 MB: −4%.
  - Ragged 4001×2048: −4%.
- **Unchanged (path-gated)**:
  - 2-link None cells keep per-chunk credits: deeper rings cost them 2–7% even without batching.
  - The single tile is within noise.
- **Next**: the remaining 2% is the last 1-in-64 credit plus fused-inc packets. The 2-link G = 4
  chains are latency-bound, and deeper rings hurt them. That belongs to Refinement 4's
  port co-location work.

### [x] Refinement 4 — Whole-mesh and ring cells: fill and packet-rate tuning

**Type**: perf

**Goal**:
- **Target**: the `cluster_axis=None` cells R1 lands (snake line `G = 4` on a 2×2, snake ring) and the
  fp32 cells R2 lands, at 32 MB bf16 / 64 MB fp32.
- **Snake line fill.** It pays about `2(G−1)` chunk-times of pipeline fill (design Perf lamp
  "Overlap / fill"). Measure fill versus steady state and co-tune `CHUNK_BYTES_TARGET` for long paths.
  Smaller chunks shorten fill; larger chunks amortize handshakes. Keep it a per-regime knob from one
  source, not a duplicate literal.
- **Ring.** Check that both directions saturate evenly across slices, since slice-boundary imbalance
  leaves one direction idle.
- **fp32.** At `packet_tiles = 1` the port loop issues twice the packets per byte, so it is
  packet-rate-bound. Check whether the header ring and credit forwarding keep up.
- Use the relevant `ttnn/ttnn/operations/examples/master.md` patterns (`double_buffer` for in-flight
  depth; `tensix_all_reduce_ring_transport` for direction-sensitive NoC contention on the port cores).
  If measurement shows fill dominating on long snakes, the deferred **R5 line reduce-scatter + all-gather**
  regime (about `G` fill hops instead of `2G`) is the scheme-level lever. File it as a follow-up rather
  than folding it in here. No SUPPORTED change.

**Done when**:
- Measured device-ns improves on the None-Linear and None-Ring 32 MB bf16 cells and on one fp32 64 MB axis
  line.
- The golden suite is green.
- There is no regression across the same config-spanning guard set as Refinement 3.

**Outcome**:
- **What bound the None cells**:
  - **1 link**: the relay port. On a snake or ring middle, `port_bwd` forwarded every final over
    Fabric *and* wrote it to DRAM, on one RISC and one NoC. An ablation that removed only the DRAM
    write gave −14%.
  - **2 links**: both lanes shared core row 0. This also explains the R3 finding that "deeper rings
    hurt 2-link None": that was row contention.
  - **Fill**: not the bound. The fixed overhead is ~40–47 µs at both 8 and 32 MB, and chunk size
    moves it ≤2%. The R5 reduce-scatter + all-gather scheme is not justified by fill.
  - **fp32**: not packet-rate bound. bf16 and fp32 packets are both 4096 B (2 tiles vs 1 tile), so
    packets per byte are equal. The fp32 64 MB axis line runs at 2.92 / 1.47 ms, 99% of the 2.88 /
    1.44 ms send-only ceiling.
  - **Ring slices**: balanced by construction (W = G = 4, one slice per reducer).
- **Levers**:
  1. **Split ports** (`SPLIT_PORTS`, gated to G ≥ 3): one `port_fwd` core and one `port_bwd` core per
     lane. A new `port_drain` kernel on the `port_bwd` core's other RISC and NoC writes output DRAM.
  2. **Row per lane** (`LANE_ROW_STRIDE = 1`).
  3. **Credit batch 4 on every path**: the R3 gate is removed.
  4. **Bank-run DRAM transfers** (`BANK_RUN_LAYOUT`): won 5–6% before the split, but costs 2–6% at 2
     links after it, so it is parked at 0.
  - **Measured flat or worse**: depths, W = 6/8, 32/128 KiB chunks, headers and NoC swaps.
- **Device ns, BH 2×2, max over devices, before → after (1 link / 2 links)**:

  | Case | 1 link | 2 links |
  |---|---|---|
  | None-Linear 32 MB | 1740 → 1489 µs (−14%) | 1167 → 766 µs (−34%) |
  | None-Ring 32 MB | 1414 → 1151 µs (−19%) | 938 → 601 µs (−36%) |
  | None-Linear 8 MB | 460 → 400 µs | 347 → 222 µs |
  | None-Ring 8 MB | 379 → 320 µs | 275 → 186 µs |
  | fp32 None-Ring 64 MB | 2618 → 2264 µs | 1805 → 1198 µs |
  | Axis lines 32 MB (bf16) | 1467 µs (unchanged) | 741 µs (unchanged) |
  | fp32 axis line 64 MB | 2920 µs (unchanged) | 1472 µs (unchanged) |

- **Where it stands now**:
  - None-Linear at 1 link is at the axis-line / send-only ceiling (32 MB per link direction).
  - None-Ring at 1 link runs 24 MB per link direction at 21.9 GB/s, 94% of 23.3 GB/s.
  - At 2 links the rates are 94% (None-Linear, 766 µs vs 0.72 ms) and 90% (None-Ring, 601 µs vs 0.54 ms) of the send-only ceilings.
- **Next**: a per-stage zone breakdown of the 2-link port loops. The most likely remaining cost is the
  EDM slot waits shared by data and credit packets. Not pursued here: every exposed knob measured flat.
