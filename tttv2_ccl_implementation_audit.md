# TTTv2 Attention CCL Implementation Audit

## Scope

- Repository: `/home/gwang/tt-metal`
- Target mesh: logical `(8, 4)` WH Galaxy
- Target tensor: tiled `(1, 1, 32, 1280)`
- Collective: reduction across `cluster_axis=1`, with scatter/gather tensor `dim=3`
- Constraints: source audit only; no TT hardware runs and no shared implementation edits

## Checkpoint 1: Entry points, dimensions, and mesh ordering

### Confirmed semantics

- `cluster_axis` is a mesh-coordinate axis, independent of tensor `dim`.
- Mesh coordinates are row-major: linear device index is `row * num_cols + col`.
- Axis 0 is north/south (rows); axis 1 is west/east (columns).
- On `(8, 4)`, `cluster_axis=1` creates eight independent four-device collectives. The ring/line index is the device's column coordinate (`mesh_coord[1]`).
- Neighbor lookup changes only the selected coordinate. For axis 1 it keeps the row fixed and increments/decrements the column.
- With an explicit `cluster_axis`, ring size comes from the global mesh extent, not the tensor's local device-coordinate list. Here it is exactly 4.

### Shape legality

- Standard reduce-scatter normalizes tensor `dim=3` and, for tiled data, requires `(padded_width / TILE_WIDTH) % ring_size == 0`.
- `(1280 / 32) % 4 == 0`, so the input has 40 width tiles and is legal.
- The per-device reduce-scatter output is `(1, 1, 32, 320)` (10 width tiles).
- All-gather on `dim=3` multiplies the width by four and reconstructs `(1, 1, 32, 1280)`.
- Input must be device-resident, allocated, page-aligned, rank at least 2, and use an interleaved or supported sharded memory layout. A caller-provided output must exactly match layout, dtype, page config, memory config, and computed shape.

### Topology behavior

- Standard `ttnn.reduce_scatter` and `ttnn.all_reduce` resolve the fabric topology, then convert Mesh/Linear to Linear and Torus/Ring to Ring for a one-axis collective.
- For an explicit axis on a four-device extent, ring boundary mode wraps if the active fabric topology supports wrapping; otherwise the operation is linear.
- Standard all-gather records axis 1 as E/W, with axis device count 4; inactive axis 0 is represented as one device and zero links.

### Initial risk assessment

- The exact shape and axis are accepted by the visible validation rules. A stall after submission is more consistent with runtime scheduling/resource disagreement than an unsupported tensor extent.
- The tensor's mesh topology still matters. All-gather replaces a mesh placement sharded on tensor `dim=3` with replication. A tensor whose placement does not describe axis-1 width sharding can have numerically surprising topology metadata even though communication neighbors are still chosen from `cluster_axis=1`.

### Primary source locations

- `ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp`
- `ttnn/cpp/ttnn/operations/ccl/common/host/moe_utils.cpp`
- `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_validate_utils.cpp`
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.cpp`
- `ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp`

## Checkpoint 2: Dispatch path, fabric compatibility, and subdevices

### Which all-reduce implementation is actually selected

- Public `ttnn.all_reduce` delegates to the composite `experimental.all_reduce_async` overload with no caller-provided semaphores.
- For tiled `(1,1,32,1280)`, its scatter-dim finder converts the last two padded dimensions to tile counts, producing `(1,1,1,40)`. Scanning from the right finds dim 3 divisible by four.
- With no external semaphores, the composite overload invokes standard `ttnn.reduce_scatter`, then standard `ttnn.all_gather`. It does **not** invoke the dedicated persistent width-sharded `AllReduceAsyncDeviceOperation`.
- The dedicated device operation is a different overload requiring a width-sharded input, width-sharded persistent buffer, one global semaphore, an explicit mesh device, and width-sharded output memory config.

### Deprecated all-gather controls are ineffective

- Current standard `ttnn.all_gather` accepts `num_links`, topology, chunks, workers, and channel buffers only for compatibility. If any are supplied it logs a deprecation warning, but the native implementation derives topology and links from active fabric and ignores those values.
- Therefore changing those arguments in the Attention fallback cannot diagnose or tune its standard all-gather stage.
- For the reduced BF16 tensor `(1,1,32,320)`, there are only 10 tile pages. On WH ring fabric the new all-gather heuristic selects multicast below 64 pages (unless NeighborExchange forces unicast). The multicast implementation allocates one worker per inferred link.

### Leading deadlock cause: NeighborExchange fabric with line reduce-scatter

- The Attention hardware test sets `fabric_config=True`. In the common fixture this value is passed through to the enum; numeric `True` is `FABRIC_1D_NEIGHBOR_EXCHANGE`, not `FABRIC_1D` or `FABRIC_1D_RING`.
- `FABRIC_1D_NEIGHBOR_EXCHANGE` explicitly provisions no forwarding between non-adjacent devices.
- Attention nevertheless asks standard reduce-scatter for `Topology.Linear` over four columns.
- The line reduce-scatter data path relays through immediate neighbors, but its startup barrier builds multicast ranges reaching every target in each direction. It clamps those ranges to one hop only when the active fabric is a 2D fabric. It does not clamp them for 1D NeighborExchange.
- On a four-device line, edge and near-edge participants request targets at distances greater than one. Those barrier messages require forwarding that NeighborExchange does not provide. A worker can therefore wait forever for the startup barrier even though host validation succeeds.
- `get_usable_topology` validates only geometric wrap behavior; it does not reject a requested operation topology that is incompatible with active fabric capabilities.
- The exact-shape upstream passing recipes use `FABRIC_1D` plus `Topology.Linear`, which supplies the required forwarding.

This is the highest-confidence explanation for a worker stall in the current fallback. The first experiment should change only the fixture fabric from `True` to `ttnn.FabricConfig.FABRIC_1D`; retaining `Topology.Linear` makes fabric and operation topology consistent.

### Subdevice behavior and failure modes

- Standard reduce-scatter and both standard all-gather factories resolve an omitted subdevice to `mesh_device.get_sub_device_ids().at(0)`. Passing the explicit worker subdevice is preferable whenever more than one subdevice exists.
- Standard reduce-scatter allocates three operation semaphores and one barrier semaphore across the chosen subdevice cores, then synchronizes that subdevice before constructing per-coordinate programs.
- Standard all-gather similarly allocates barrier state over the chosen subdevice and synchronizes it.
- Core selection warns when fewer cores than requested are available but can return a short vector; several factories later index as if the preferred count was met. This is more likely to cause a host exception or invalid program construction than the observed silent worker stall, but it warrants an explicit full-worker or canonical 50-core subdevice in experiments.
- A wrong explicit subdevice ID, stale loaded manager, or semaphore grid outside that subdevice can strand waits. The current one-full-worker CCL-only manager removes most of this risk, but the exact passing QKV recipe uses the canonical 50-core worker set and explicit `SubDeviceId(0)`.

### Source evidence added

- `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp`
- `ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.cpp`
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_multicast_factory.cpp`
- `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp`
- `tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp`
- `tt_metal/fabric/fabric_context.cpp`

## Checkpoint 3: Worker/link/channel geometry and topology metadata

### Standard reduce-scatter geometry for the target

Assuming BF16 tiled interleaved input, Linear topology, one link, ring size four:

- Local input has 40 tile pages and occupies 81,920 bytes.
- The worker heuristic estimates `81920 * 3 / 4 = 61,440` bytes moved per link.
- That is above the single-packet threshold and below the Linear 0.5 MB threshold, so it chooses two workers per direction.
- The program uses two directions and one mux per direction, requiring `2 * (1 mux + 2 workers) = 6` Tensix cores for one link.
- Each output contains 10 tiles. Split across two workers, the target is roughly five tiles each.
- Tile granularity is eight tiles for BF16/BF8 in this case, so the default synchronization cadence collapses to one chunk per worker.
- Default `num_buffers_per_channel` is one.

With four inferred links, the same heuristic requests 24 cores. Attention explicitly supplies one link to standard reduce-scatter, so its requested six cores fit either the canonical 50-core set or a full-grid worker subdevice.

No visible arithmetic produces a zero-sized tile assignment for this shape. Forcing `chunks_per_sync=10`, two workers, and two channel buffers in the dormant persistent RS/AG branch is unnecessary and differs from exact passing references; defaults should be used in the first persistent-path experiment.

### Dedicated persistent all-reduce geometry

- The exact upstream QKV test pads the 1280-wide input across 24 width-sharded cores: each input shard is `(32,64)`, so physical width is 1536 while logical width remains 1280.
- Its output uses 10 width-sharded cores with shard `(32,128)`.
- Its persistent intermediate uses those output cores with shard `(32,512)`, exactly four output shards per core for axis-1 ring size four.
- The dedicated factory reserves output cores, then allocates one sender core per link from the remaining selected-subdevice cores.
- Tested matrices include both one and three links, Linear topology, BF16/BF8 input, the full `(8,4)` mesh, `FABRIC_1D`, COL dispatch, and an explicit canonical 50-core worker subdevice.
- Every output core waits for exactly four global-semaphore contributions. A missing route, missing sender, semaphore not allocated on an output core, or incomplete mesh participation leaves that wait permanent; reset occurs only after all four contributions arrive.

### Mapper and topology implications

- The exact persistent recipe maps a global tensor shaped `(8,4,32,1280)` with `ShardTensor2dMesh(dims=(0,1), mesh_shape=(8,4))`. Device `(row,col)` receives one local `(1,1,32,1280)` partial. This mapping is about assigning independent partial tensors to mesh coordinates; reduction still occurs only across columns.
- Attention's input/weight mapping is mathematically coherent: activations shard K over columns and QKV weights shard output features over rows and K over columns. Each `(row,col)` matmul output is a row-owned output-feature block containing a column partial that must be summed over axis 1.
- Standard reduce-scatter does not define `compute_output_topologies`, while experimental minimal reduce-scatter explicitly marks the selected mesh axis as sharded on tensor `dim`. Standard all-gather only turns placements that are already sharded on its gather `dim` into replication.
- Consequently, even if standard RS+AG communicates correct bytes, its resulting tensor-topology metadata may continue to describe the producer rather than an axis-1 replicated reduction. This is a correctness/integration risk for the fallback and another reason to prefer the exact dedicated all-reduce contract.

### Ranked deadlock causes

1. **High confidence:** `FABRIC_1D_NEIGHBOR_EXCHANGE` cannot satisfy the multi-hop startup barrier emitted by four-device Linear reduce-scatter.
2. **Medium confidence:** a standard RS+AG fallback carries stale/incomplete distributed topology metadata and uses a less-qualified path than the exact QKV primitive. This should cause composition errors more readily than a worker stall.
3. **Medium-low confidence:** subdevice/manager mismatch or semaphore coverage mismatch. The isolated single full-worker manager reduces this risk; the exact recipe with explicit canonical subdevice removes it further.
4. **Low confidence for this shape:** too many workers, links, or zero-sized channels. One-link defaults produce six cores and nonzero work for both workers.
5. **Low confidence:** tensor-dim or mesh-axis confusion. The source consistently interprets axis 1 as columns and dim 3 as width, and all divisibility constraints pass.

## Checkpoint 4: Minimal sequential experiment matrix

Run one case at a time and synchronize immediately after the collective. Reset hardware after any timeout before proceeding. Keep all tensor and semaphore owners alive through synchronization.

| ID | Purpose | Fabric / topology | Mesh and subdevice | Operation and exact tensor contract | Expected result / interpretation |
|---|---|---|---|---|---|
| A0 | Confirm current failure signature | `FABRIC_1D_NEIGHBOR_EXCHANGE` (current `True`) / Linear | Full `(8,4)`, explicit one-full-grid `SubDeviceId(0)` | Standard `reduce_scatter`, BF16 tiled interleaved DRAM `(1,1,32,1280)`, `dim=3`, `cluster_axis=1`, one link, DRAM output | Expected stall. A reproduced stall isolates failure before all-gather and supports the barrier diagnosis. Do not repeat after confirmation. |
| A1 | Test the single-variable fix | `FABRIC_1D` / Linear | Identical to A0 | Identical to A0 | Expected pass with local output `(1,1,32,320)`. A0 fail + A1 pass confirms fabric forwarding as root cause. |
| A2 | Isolate standard gather | `FABRIC_1D` / derived Linear | Identical explicit subdevice | Standard `all_gather` from four distinct axis-1 shards of BF16 tiled interleaved DRAM `(1,1,32,320)`, `dim=3`, `cluster_axis=1`; do not pass deprecated tuning args | Expected pass and local `(1,1,32,1280)`. Failure here points to the new native all-gather factory, not RS. |
| A3 | Qualify the fallback composition | `FABRIC_1D` / Linear | Identical explicit subdevice | Standard RS immediately followed by standard AG, same buffers as A1/A2 | Expected communication pass. Inspect tensor topology as well as values; stale axis-1 placement metadata disqualifies this path even if PCC passes. |
| B0 | Reproduce the exact upstream QKV contract | `FABRIC_1D` / Linear | Full `(8,4)`, canonical 50-core worker subdevice, explicit `SubDeviceId(0)` | Dedicated `experimental.all_reduce_async`; BF8 tiled WIDTH_SHARDED L1 input on 24 canonical ring cores with shard `(32,64)`; BF16 WIDTH_SHARDED L1 output on 10 QKV cores with shard `(32,128)`; persistent BF8 L1 buffer on output cores with shard `(32,512)`; one global semaphore over all 50 worker cores; one link | Expected pass. This is the highest-value production-aligned gate and matches an upstream exact-shape case. |
| B1 | Rule out link-plane dependence | `FABRIC_1D` / Linear | Same as B0 | Same as B0, three links | Expected pass. B0 pass/B1 fail indicates routing-plane or sender-core allocation trouble; use one link for Milestone A. |
| B2 | Integrate the real producer | `FABRIC_1D` / Linear | Same as B0 | Run QKV matmul, convert its local partial from interleaved DRAM to the exact B0 24-core BF8 L1 input config, then invoke B0; synchronize before head creation | Expected pass. Failure after B0 passes isolates producer lifetime/conversion or program-cache interaction. |
| B3 | Remove diagnostic synchronization | `FABRIC_1D` / Linear | Same as B0 | Same as B2, but rely on queue ordering and synchronize only after head creation | Expected pass. B2 pass/B3 fail identifies lifetime/queue ownership rather than CCL geometry. |

### Contingency splits

Only if A1 or B0 fails:

| ID | Change from failing case | Diagnostic meaning |
|---|---|---|
| C0 | Run on a `(1,4)` submesh with otherwise identical data and explicit subdevice | Pass implies a bug in launching eight concurrent row collectives or full-mesh coordinate handling; fail keeps the issue inside one four-device line. |
| C1 | Replace the canonical 50-core subdevice with one full-worker subdevice | Pass implicates core availability/offset assumptions in the canonical subdevice; fail rules that out. |
| C2 | Use no custom subdevice manager and omit `subdevice_id` | Pass implicates manager/stall-group ownership; fail points back to fabric/program geometry. |
| C3 | B0 only: switch BF8 input to BF16 while preserving shard shapes and buffer dtype consistency | Pass isolates a BF8 data-format/page-size path; fail rules dtype out. |

Do not combine contingency changes. Each run must alter one variable from the immediately preceding failing case.

## Final audit conclusion

- `(1,1,32,1280)`, tensor dim 3, mesh axis 1, four devices, and one link are all supported by current validation and by exact upstream WH Galaxy tests.
- The current Attention fallback is configured on NeighborExchange fabric through `fabric_config=True`, but its four-device Linear reduce-scatter startup barrier requires multi-hop forwarding. This is the most likely sole blocker.
- The minimum corrective experiment is A1: use explicit `ttnn.FabricConfig.FABRIC_1D` and keep `Topology.Linear`.
- For the durable Attention implementation, B0/B2 is preferable to standard RS+AG because it matches the qualified QKV contract and explicitly returns axis-1 replicated topology.
- No TT hardware was run and no shared implementation file was edited during this audit.
