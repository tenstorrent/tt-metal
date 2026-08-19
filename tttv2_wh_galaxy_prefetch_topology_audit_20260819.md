# WH Galaxy Prefetcher Topology Audit Work Log

## Goal

Audit the proven WH Galaxy reference prefetcher topology without hardware execution or shared-code edits. Establish exact active and dummy sender/receiver mappings, address repeat count, subdevice partitioning, and global-CB containment, then compare the reference with `models/common/tests/modules/_wh_galaxy_hardware.py`.

## Checkpoint 1: Authoritative topology located

- Reference topology: `models/demos/llama3_70b_galaxy/tt/model_config.py::get_core_ranges`.
- Reference lifecycle and address metadata: `models/demos/llama3_70b_galaxy/tt/prefetcher_common.py::TtLlamaPrefetcherSetup`.
- Current common-module hardware helper: `models/common/tests/modules/_wh_galaxy_hardware.py::_create_hardware_prefetcher` and `galaxy_prefetch_decode_mode_plan`.
- The reference starts with 12 active sender coordinates and 12 two-core active receiver groups, then mutates the returned sender and receiver lists by appending 8 dummy sender/receiver groups. Its global-CB mapping therefore has 20 entries.
- Address metadata is independent of the global-CB mapping length: tensor addresses are repeated across `len(dram_cores) == 12` and height-sharded only over the 12 active sender cores.
- In the initial snapshot read at this checkpoint, the helper created only the 12 active mapping entries and set `address_repeat_count` to 12. The repeat count matched the reference, but that snapshot omitted all 8 dummy entries.
- The C++ global-CB containment check requires every core used by a shadowed circular buffer to be contained in `GlobalCircularBuffer::all_cores()`. The omitted dummy receiver groups are therefore a direct candidate for failures from matmul programs using the full worker subdevice.

Status: source comparison complete; exact coordinate tables and set-coverage proof pending.

## Checkpoint 2: Exact mapping and containment proven

### Active sender to receiver mappings (entries 1-12)

| Entry | Active sender | Active receiver range |
|---:|---|---|
| 1 | `(0,9)` | `(1,9)-(2,9)` |
| 2 | `(0,0)` | `(1,0)-(2,0)` |
| 3 | `(0,4)` | `(1,4)-(2,4)` |
| 4 | `(0,5)` | `(1,5)-(2,5)` |
| 5 | `(4,0)` | `(5,0)-(6,0)` |
| 6 | `(4,9)` | `(5,9)-(6,9)` |
| 7 | `(4,1)` | `(5,1)-(6,1)` |
| 8 | `(4,7)` | `(5,7)-(6,7)` |
| 9 | `(4,6)` | `(5,6)-(6,6)` |
| 10 | `(4,2)` | `(5,2)-(6,2)` |
| 11 | `(4,4)` | `(5,4)-(6,4)` |
| 12 | `(4,5)` | `(5,5)-(6,5)` |

### Dummy sender to receiver mappings (entries 13-20)

| Entry | Dummy sender | Dummy receiver `CoreRangeSet` |
|---:|---|---|
| 13 | `(0,1)` | singleton `(3,0)` plus range `(1,1)-(3,1)` |
| 14 | `(0,2)` | range `(1,2)-(3,2)` |
| 15 | `(0,3)` | range `(1,3)-(3,3)` plus singleton `(3,4)` |
| 16 | `(0,6)` | singleton `(3,5)` plus range `(1,6)-(3,6)` |
| 17 | `(0,7)` | range `(1,7)-(3,7)` |
| 18 | `(0,8)` | range `(1,8)-(3,8)` plus singleton `(3,9)` |
| 19 | `(4,3)` | range `(5,3)-(6,3)` |
| 20 | `(4,8)` | range `(5,8)-(6,8)` |

### Address metadata and subdevices

- DRAM reader cores: `(0,0)` through `(11,0)` in DRAM-core coordinates, count 12.
- Address repeat count: exactly 12, derived from `len(dram_cores)`, not from the 20-entry global-CB map.
- Address metadata shard grid: the 12 active sender singleton cores only. For one layer, each shard has shape `[1, n_tensors]`.
- Decode prefetcher subdevice: the 12 active sender singleton cores only; dummy senders are not part of this subdevice.
- Decode worker subdevice: ranges `(1,0)-(3,9)` and `(5,0)-(6,9)`, totaling 50 cores.
- Decode subdevice IDs: prefetcher `SubDeviceId(0)`, worker `SubDeviceId(1)`.
- Reference decode launch sequence initially stalls both IDs, invokes `ttnn.dram_prefetcher`, then changes the stall group to worker ID 1 only.
- Prefill reference subdevice: one range `(0,0)-(6,9)` with `SubDeviceId(0)` used as the worker.

### Coverage proof

- Active receiver groups contain 24 distinct cores.
- Dummy receiver groups contain 26 distinct cores.
- Their sets are disjoint and their union is exactly the 50-core decode worker subdevice.
- The 20 sender cores are all points in columns `x=0` and `x=4`, rows `y=0..9`; active and dummy senders partition those 20 points.
- For the worker-sender global-CB constructor used by the reference, `all_cores()` is sender union receiver. The proven 20-entry map therefore covers exactly `(0,0)-(6,9)`, 70 cores.
- The initial helper snapshot's 12-entry map covered only 12 active sender cores plus 24 active receiver cores, 36 cores. It omitted the 8 dummy senders and 26 complementary worker receivers.
- The shadow-global-CB invariant in `tt_metal/impl/buffers/circular_buffer.cpp` checks that the circular buffer's core ranges are contained in `GlobalCircularBuffer::all_cores()`. A matmul circular buffer placed on any of the omitted 26 worker cores fails this invariant.

Status: exact topology and containment proof complete; final mismatch assessment pending.

## Checkpoint 3: Concurrent update reconciled and final assessment

A concurrent shared-code update appeared between the initial and final reads. This audit did not make that update. The current worktree now has the following state:

1. **Global-CB mapping now matches:** `_create_hardware_prefetcher` defines the exact 8 dummy sender coordinates and 8 dummy receiver sets, concatenates them after the 12 active entries, and creates the required 20-entry mapping.
2. **Containment coverage now matches:** the helper's current global CB covers all 50 receiver cores in the worker subdevice. Together with all 20 sender points, `GlobalCircularBuffer::all_cores()` covers exact range `(0,0)-(6,9)`.
3. **Configuration model now matches:** `Prefetcher2DConfig.__post_init__` now permits `address_repeat_count < len(sender_receiver_mapping)` and rejects only repeat counts greater than the mapping length. It can therefore represent repeat count 12 with mapping length 20.
4. **Address placement matches:** the helper's `address_repeat_count=12`, active-sender shard grid, and `[1, len(weights)]` per-core shard shape match the one-layer reference behavior.
5. **Decode subdevice geometry matches:** `galaxy_prefetch_decode_mode_plan` correctly uses only the 12 active senders for subdevice 0 and the exact 50-core worker ranges for subdevice 1. Dummy senders belong only in global-CB mapping, not in the active sender subdevice.
6. **Decode launch stall-group sequencing still differs:** the helper plan records only `(worker_id,)`. The reference has both sender and worker IDs stalled when `ttnn.dram_prefetcher` is launched, then switches to worker-only. `Prefetcher2D._configure_mode` does not install a decode stall group before `_start_prefetch`; it applies worker-only afterward.
7. **Prefill geometry still differs:** the reference uses exact range `(0,0)-(6,9)`. `galaxy_mode_plan` builds `(0,0)-(grid.x-1, grid.y-1)`, so it is not statically equivalent and may include additional compute-with-storage cores depending on the device grid reported at runtime.

No hardware commands or tests were run. No shared production or test files were edited.
