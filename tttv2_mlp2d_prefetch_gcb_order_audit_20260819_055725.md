# MLP2D Prefetcher/GCB Ordering Audit

Date: 2026-08-19 UTC

## Scope

Static source audit of the Wormhole Galaxy `(8, 4)` MLP2D decode path. The audit compares common `Prefetcher2D` registration, packed addresses, GCB topology, and fused `llama_rs_matmul` consumption with the legacy Galaxy implementation. No TT hardware was run. No production or test file was edited.

## Checkpoint 1: Registration and packed addresses

**Result: exact ordering match; no W1/W3 address swap found.**

- Legacy MLP registration is `w1`, `w3`, `w2` in `models/demos/llama3_70b_galaxy/tt/llama_mlp.py:111-116`.
- The common MLP hardware setup registers `mlp.w1`, `mlp.w3`, `mlp.w2` in `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py:528-532`.
- `Prefetcher2D` stores registrations in an `OrderedDict`, derives both `weights` and `weight_addresses` from that same order, and packs the address row without sorting in `models/common/modules/prefetcher/prefetcher_2d.py:272-281`.
- Legacy `insert_tensor()` appends the tensor and its address to parallel lists, preserving the same order (`prefetcher_common.py:112-114`). Legacy `get_tensor_addrs()` repeats the resulting row over the DRAM readers (`prefetcher_common.py:116-138`).
- Both paths therefore construct the same logical metadata matrix for one layer: 12 identical rows of `[addr(W1), addr(W3), addr(W2)]`, converted to a row-major `uint32` L1 tensor replicated to every mesh device.
- Common `dram_prefetcher` input is `[W1, W3, W2, address_metadata]` (`prefetcher_2d.py:526-531`), matching legacy `get_input_tensors()` (`prefetcher_common.py:142-153`).

There is no independent name lookup after packing. Correctness depends on insertion order, and the audited setup satisfies that contract.

## Checkpoint 2: Active/dummy GCB topology

**Result: exact topology and ordering match.**

- Both paths use 12 active sender cores, 8 dummy sender cores, two receivers per active sender, and the same dummy receiver sets.
- Active sender order is identical:
  `(0,9), (0,0), (0,4), (0,5), (4,0), (4,9), (4,1), (4,7), (4,6), (4,2), (4,4), (4,5)`.
- Dummy sender order is identical:
  `(0,1), (0,2), (0,3), (0,6), (0,7), (0,8), (4,3), (4,8)`.
- Active receiver pairs are identical and paired with the same active senders. The common mapping in `models/common/tests/modules/_wh_galaxy_hardware.py:98-134` reproduces legacy `get_core_ranges()` in `models/demos/llama3_70b_galaxy/tt/model_config.py:180-332`.
- Both GCBs use size `728 * 1088` bytes and all 20 active/dummy sender mappings (`prefetcher_common.py:76-87`; `_wh_galaxy_hardware.py:151-163`).
- Both address tensors are height-sharded only over the 12 active sender cores with shard shape `[1, 3]`. Dummy senders do not receive address rows and are not selected as DRAM readers.

No active/dummy displacement or sender-to-receiver pairing mismatch was found.

## Checkpoint 3: Prefetch producer order

**Result: producer emits W1 then W3 then W2.**

- The DRAM prefetch operation treats the final input as the packed address tensor and the preceding inputs as data tensors in vector order (`ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp:40-76`).
- Its reader kernel loops `layer` outermost and tensor index `t` next, loading `tensor_addrs_l1[layer * num_tensors + t]` (`device/kernels/reader_dram.cpp:40-48`).
- Its writer kernel uses the same tensor-major loop and pushes all ring blocks for tensor `t` before advancing to `t + 1` (`device/kernels/writer_l1.cpp:40-86`).
- Consequently, the remote FIFO sequence for the audited one-layer path is all W1 blocks, then all W3 blocks, then all W2 blocks.

The packed address row and the host tensor vector agree, so the prefetcher cannot select W2 or W3 while labelling it W1 in this setup.

## Checkpoint 4: Fused llama_rs_matmul consumers

**Result: fused consumer order matches the first two prefetched tensors.**

- Common MLP calls `llama_rs_matmul(input, self.w1, ..., second_weight_tensor=self.w3)` (`models/common/modules/mlp/mlp_2d.py:337-369`). Legacy passes the same argument order (`llama_ccl.py:907-951`).
- The fused program factory passes `{weight_tensor, second_weight_tensor}` and `{output0, output1}` in matching order to the multi-tensor matmul helper (`rs_matmul_program_factory.cpp:43-69`).
- The helper sets `batch = b_tensors.size()`, so this call has `batch=2` (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:2108-2111`).
- The global-CB reader loops `b=0..1`, consuming one complete remote tensor per batch slot (`reader_bmm_tile_layout_in1_ring_all_gather.cpp:139-266`). The compute kernel independently loops the same batch index and selects output/partial CB arrays by `b` (`bmm_large_block_zm_fused_bias_activation_gathered.cpp:279-299`).
- Therefore fused output 0 consumes the first remote tensor (W1), and fused output 1 consumes the second remote tensor (W3). W2 remains next in the FIFO for the later FF2 `ttnn.linear`.
- Cached-program override also preserves input order as `{activation, W1, W3}` and output order `{output0, output1}` (`rs_matmul_program_factory.cpp:103-120`).

No fused-reader off-by-one, reversed batch order, or output association mismatch was found.

## Geometry comparison

The common Llama test reproduces the legacy weight and ring geometry relevant to GCB addressing:

- W1/W3 local logical shape: `K=8192/4=2048`, `N=28672/8=3584`.
- DRAM shard allocation pads local N to 3840 over 12 banks, yielding shard shape `[2048, 320]` in both paths.
- Ring output/program N is padded to 3840 over 24 receiver cores.
- W1/W3 use the same 2D mesh placement: mesh axis 0 shards N and mesh axis 1 shards K.
- Address metadata contains buffer base addresses; bank selection and block offsets are derived by the prefetch program from each tensor's DRAM shard geometry, not encoded separately in the metadata.

This rules out a differing explicit per-bank address table: neither implementation has one.

## Caveat: repeated activation lifecycle

The common owner stops the previous prefetch result before starting a new decode activation (`prefetcher_2d.py:365-388`), while the legacy unit test launches `dram_prefetcher` directly on each iteration (`test_llama_mlp.py:95-104`). This is a lifecycle difference, but it does not alter the first invocation's W1/W3 address order and is not a plausible explanation for already-corrupt first-pass raw products. It could matter only for repeat-run synchronization or ownership behavior.

## Conclusion

**No registration-order or packed-address mismatch was identified that explains numerically corrupt raw W1/W3 products.** The common path is structurally equivalent to legacy from `W1,W3,W2` registration through the producer FIFO and the fused two-weight consumer.

The corruption should be investigated outside this ordering hypothesis. Highest-value next checks are:

1. Compare fused raw output 0/1 against per-device partial products before reduce-scatter. This distinguishes matmul/GCB corruption from padded reduce-scatter interpretation.
2. Run the same fused operation with the GCB disabled and identical tensor/program geometry. Correct direct-DRAM products would isolate the remote-CB data path; corrupt products would point to activation/weight layout or matmul program geometry.
3. Validate actual per-device local weight logical/padded shapes and DRAM shard specs immediately before dispatch, especially that W1 and W3 expose `[2048, 320]` per-bank storage and retain the expected mesh-coordinate placement.
4. If GCB-only corruption is confirmed, inspect runtime remote-CB read/write pointer state and block ordering per receiver. Static host ordering is correct, so the remaining GCB hypothesis would be dynamic pointer/credit behavior rather than address packing.
