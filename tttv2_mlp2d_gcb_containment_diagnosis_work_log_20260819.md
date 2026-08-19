# MLP2D GlobalCircularBuffer Containment Diagnosis

## Goal

Diagnose the WH Galaxy MLP2D hardware failure `Specified cores are not contained in associated GlobalCircularBuffer` by reading the C++ GlobalCircularBuffer and matmul implementation and comparing it with the existing Galaxy reference topology. This lane does not run hardware or edit shared production/test files.

## Checkpoint 1: Failure Site and Initial Topology Discrepancy

- Created a dedicated diagnosis goal and changed the session title.
- Confirmed no matching WH Galaxy MLP2D pytest process remains active.
- Located the fatal at `tt_metal/impl/buffers/circular_buffer.cpp:182`, in `CircularBufferImpl::set_global_circular_buffer`.
- Located the reference topology in `models/demos/llama3_70b_galaxy/tt/model_config.py` and GCB construction in `models/demos/llama3_70b_galaxy/tt/prefetcher_common.py`.
- Initial discrepancy: the reference constructs `sender_receiver_mapping` from `all_sender_cores` and `all_receiver_cores`, while the Milestone A helper needs verification for whether it includes only the 12 active sender/receiver pairs.
- No hardware command was run and no shared production/test file was edited.

## Checkpoint 2: Exact Containment Proof

- `CircularBufferImpl::set_global_circular_buffer` requires `global_circular_buffer.all_cores().contains(circular_buffer.core_ranges())` (`tt_metal/impl/buffers/circular_buffer.cpp:179-186`).
- `ttnn.create_global_circular_buffer` creates the legacy worker-sender GCB. Its `all_cores` is the union of every mapped sender and receiver (`tt_metal/impl/buffers/global_circular_buffer.cpp:59-87`; `ttnn/core/global_circular_buffer.cpp:20-35`).
- The gather-in0 matmul builds `all_worker_cores` from input A's shard grid, merges `hop_cores`, intersects that union with the selected worker subdevice, and creates the GCB-backed `src1` circular buffer over the resulting `all_cores` (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:2115-2148, 2258-2269`).
- The focused test's input shard grid contains the 24 `PREFETCHER_NOC1_GRID` ring points. Every one of those points is among the helper's 24 active receiver points.
- The program config additionally specifies hop core `(3,6)`. That point is not a mapped sender or receiver in the helper's 12-pair GCB, so the C++ subset check fails exactly as reported.
- The reference topology includes `(3,6)` in dummy receiver group 4: singleton `(3,5)` plus range `(1,6)` through `(3,6)`, paired with dummy sender `(0,6)`.
- The immediate root cause is therefore the helper's omission of the reference dummy sender/receiver mappings, with `(3,6)` being the concrete core that triggers this matmul failure.
- No hardware command was run and no shared production/test file was edited.

## Checkpoint 3: Reference Coverage and Required Contract Change

- `get_core_ranges` aliases `sender_cores = all_sender_cores` and appends eight dummy senders; it similarly appends eight dummy receiver sets to `all_receiver_cores`. Consequently, the reference GCB mapping has 20 entries, not 12 (`model_config.py:334-343`; `prefetcher_common.py:43-52, 76-85`).
- The 12 active receiver pairs cover 24 workers. The eight dummy receiver sets cover the remaining 26 workers: 4 + 3 + 4 + 4 + 3 + 4 + 2 + 2. Together they exactly cover the 50-core worker subdevice `(x=1..3,y=0..9) U (x=5..6,y=0..9)`.
- The 12 active plus eight dummy senders exactly cover columns `x=0` and `x=4` for `y=0..9`. Thus the reference worker-sender GCB `all_cores` covers the full 70 logical cores in columns `x=0..6`, including every worker and the `(3,6)` hop.
- Dummy senders are intentionally excluded from the prefetcher subdevice: `sender_core_range_set` is built from `active_sender_cores` only. The GCB topology and active reader execution topology are distinct.
- The reference address tensor repeats by `len(dram_cores) == 12`, not by the 20-entry GCB mapping (`prefetcher_common.py:116-138`). TTNN derives `num_readers` from the prefetched weight's 12-core DRAM shard grid and only validates/launches that many leading GCB senders (`dram_prefetcher_device_operation.cpp:31-40`; `dram_prefetcher_program_factory.cpp:75-81, 113-122`).
- `Prefetcher2DConfig` currently requires `address_repeat_count == len(sender_receiver_mapping)`. That prevents the proven 12-reader/20-mapping topology and is a secondary implementation defect exposed by the root-cause fix.
- No hardware command was run and no shared production/test file was edited.

## Final Recommendations

1. In `_create_hardware_prefetcher`, construct the exact reference 20-entry mapping: keep the existing 12 active sender/receiver pairs first, then append the eight reference dummy sender/receiver pairs in their existing order.
2. Keep the decode sender subdevice and address metadata shard grid restricted to the 12 active sender coordinates. Keep `address_repeat_count=12`.
3. Change `Prefetcher2DConfig` validation from equality to a reader-prefix constraint: require `0 < address_repeat_count <= len(sender_receiver_mapping)`. The ordered leading mappings are the active DRAM readers; trailing mappings only complete GCB coverage.
4. Add host tests asserting: mapping length 20; first 12 mappings are the active topology; receiver union equals the 50-core worker subdevice; `(3,6)` is covered; address metadata still has 12 rows; a repeat count larger than mapping length is rejected.
5. After those host checks, rerun the serialized WH Galaxy MLP2D hardware case. The current fatal occurs during GCB-backed circular-buffer creation, before kernel execution, so removing the missing-core condition should advance bring-up to the next validation or execution stage.
6. Align `out_subblock_w` with the reference's largest divisor up to 8 as a separate fidelity/performance cleanup. Its current value of 1 is divisible and is not the source of this containment fatal.

## Ruled-Out Suspects

- `compute_with_storage_grid_size=(8,3)` does not define the failing core set on the gather-in0 path. That path derives workers from input A's shard grid and then adds hop cores.
- The 24 ring points themselves are not missing: all are present in the 24 active receiver points.
- `out_subblock_w=1` is not involved in the subset check.
- This GCB is the legacy worker-sender type used by `ttnn.dram_prefetcher`; the experimental DRAM-sender GCB path is explicitly rejected by that operation.
