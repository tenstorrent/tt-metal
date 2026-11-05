# Quasar uplift report: `ttnn.sparse_matmul`

Branch `fplavec/quasar_sparse_matmul`, 2026-11-04 and 2026-11-05. Process: [quasar_porting.md](../../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md). This report is not for commit; delete it before merge.

## Status: GREEN on Quasar and Wormhole; Blackhole not run

The op gives correct results on the Quasar simulator (craq-sim) in all its modes: B sparse, A sparse, A and B sparse, and indexed (gather) mode with and without fused bias, each with and without `fp32_dest_acc_en`. The three modes that take a sparsity mask (B sparse, A sparse, A and B sparse) pass with `nnz` given and with `nnz=None`.

One change is not limited to Quasar. With `nnz=None`, the in0 kernels now tell the compute kernel which batches to skip through a dataflow buffer on every arch, instead of through the BRISC mailbox, which Quasar does not have (see [Batch flags through a dataflow buffer](#batch-flags-through-a-dataflow-buffer)). Wormhole gives the same test results as before. Blackhole is not run yet; [Follow-ups](#follow-ups) lists the CI jobs that cover it.

The op directory and namespace did not change. Nothing was copied from `experimental/quasar/`.

## Results

**Quasar (craq-sim).** bf16 weights and bf16 output; Quasar has no BFP formats (see [Follow-ups](#follow-ups)). Each case ran in its own process, once with `TT_METAL_LLK_ASSERTS=1` and once without. Kt is the number of tiles along K; with `in0_block_w=1` and Kt=4 the compute kernel spills partial sums between K blocks. Before the uplift, every case failed at program creation with `TT_FATAL: Quasar has one NoC; configure the multicast NoC as NOC_0`.

| Mode | Configuration | Result |
|---|---|---|
| B sparse | 2x2 grid, Kt=4, 32x32 tiles, two outer batches with different mask pages | pass |
| B sparse | 16x32 in0 and output tiles | pass |
| B sparse | Kt=1 | pass |
| B sparse | `out_block_w=2`, `out_subblock_w=2` | pass |
| B sparse | `packer_l1_acc=True` | pass |
| B sparse | 4x2 grid, op called twice (program cache hit) | pass |
| A sparse | 2x2 grid | pass |
| A and B sparse | 2x2 grid | pass |
| Indexed | 8 groups, ids not in order, op called once and twice | pass |
| Indexed with fused bias | `packer_l1_acc` off and on | pass |
| B sparse; indexed | 4x4 grid (16 cores) | pass |
| B sparse | `fp32_dest_acc_en=True`: Kt=4; Kt=1; with `packer_l1_acc`; `in0_block_w=2`; 4x2 grid, op called twice | pass |
| A sparse; A and B sparse | `fp32_dest_acc_en=True` | pass |
| Indexed | `fp32_dest_acc_en=True` | pass |
| Indexed with fused bias | `fp32_dest_acc_en=True`, `packer_l1_acc` off and on | pass |
| B sparse, `nnz=None` | 2x2 grid, with a mask that zeroes every third group, a mask that zeroes no group, and a mask that zeroes every group | pass |
| B sparse, `nnz=None` | 4x2 grid; 4x4 grid; 16x32 tiles; op called twice | pass |
| B sparse, `nnz=None` | `fp32_dest_acc_en=True`; `packer_l1_acc=True` | pass |
| A sparse; A and B sparse | `nnz=None` | pass |

Every pass meets PCC ≥ 0.999 per output slot; the B-sparse fp32 case with Kt=4 printed PCC 1.0000 for all five slots. The `nnz=None` cases also check that every output slot of a skipped batch is zero.

- Compared with the report of 2026-11-04:
  - `fp32_dest_acc_en=True` (wrong output, PCC 0.85) and fused bias with `fp32_dest_acc_en=True` (compile error) now pass, after the change described in [fp32 partials reload through SrcA on Quasar](#fp32-partials-reload-through-srca-on-quasar).
  - `nnz=None` (compile error on a Quasar `static_assert`) now passes, after the change described in [Batch flags through a dataflow buffer](#batch-flags-through-a-dataflow-buffer).
  - The 4x4 grid cases, which had run with LLK asserts off only, also pass with LLK asserts on.
- The cases with `nnz` given and the indexed cases were not all run again after the `nnz=None` change. On Quasar, that change reaches the code of these cases only through a `static_assert` that `get_batch_from_reader` is false: these cases do not define `BATCH_VALID_DFB`, and the `ckernel.h` includes that the change removed were already left out of Quasar builds. As a check, B sparse (2x2 and 4x4 grids), indexed with fused bias, and indexed on a 4x4 grid ran again after the change and pass.

The in-tree tests cannot run on Quasar as written, because all of them use `bfloat8_b` or `bfloat4_b` weights. The Quasar cases above come from an ad-hoc script that copies the setups of the in-tree tests with bf16 weights (see [How to rerun](#how-to-rerun)).

**Wormhole (n300).**

- [test_sparse_matmul.py](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py) and [test_sparse_matmul_indexed.py](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py): 93 passed, 2 skipped, before the uplift and after each of its changes, with the same outcome for every test. The 2 skipped tests are Quasar-only tests of `ttnn.experimental.quasar.sparse_matmul`. Five of the passing cases run with `nnz=None`: those of [test_sparse_matmul_without_nnz](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py#L125-L125), [test_batched_sparse_matmul_without_nnz](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py#L335-L335) and [test_sparse_matmul_inputA_without_nnz](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py#L536-L536). The kernel cache of the last run holds builds of the in0 sender, the in0 receiver and the compute kernel that bind `batch_valid`, so these five cases ran the new code.
- The ad-hoc script: the `nnz=None` cases of the Quasar table, plus B sparse with `nnz=None` and bfp8 weights, pass.
- Dense matmul, because the compute kernel serves every dense matmul factory, and the two in0 kernels serve the dense 1D and 2D factories: [test_matmul.py](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_matmul.py), [test_linear.py](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_linear.py), the three `test_matmul_dram_sharded_*.py` files, [test_matmul_batched_dram_sharded_fused_bias.py](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_matmul_batched_dram_sharded_fused_bias.py), [test_matmul_fp32_partials_reload.py](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_matmul_fp32_partials_reload.py) and [test_matmul_program_cache.py](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_matmul_program_cache.py): 1412 passed, 452 skipped, 2 xfailed, 1 xpassed, no failures. The 2 xfailed tests are a strict xfail for issue [#49836](https://github.com/tenstorrent/tt-metal/issues/49836) ([test_matmul.py:5070-5073](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_matmul.py#L5070-L5073)). The xpassed test has a non-strict xfail mark ([test_linear.py:650](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_linear.py#L650-L650)) and also xpassed on this branch before the `nnz=None` change.
- The other seven files of the same folder (`test_addmm.py`, `test_custom_grids.py`, `test_experimental.py`, `test_matmul_2d_config_bitexact.py`, `test_matmul_batch_mismatch.py`, `test_matmul_deepseek.py` and `test_ring_matmul.py`): 434 passed, 35 skipped, no failures. With the files above, this covers every file in `tests/ttnn/unit_tests/operations/matmul`.

**Blackhole.** Not run; this machine has no Blackhole device. See [Follow-ups](#follow-ups) for the CI jobs.

## Files changed

| File | Reason |
|---|---|
| [factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp](factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L499-L533) | The two sparsity-page DFBs become kernel-private scratchpads. With `nnz=None`, a `batch_valid` DFB carries the batch flags from the in0 kernels to the compute kernel. On Quasar, the in0 multicast uses NOC_0 and the fp32 partials reload goes through SrcA. |
| [../factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp](../factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp#L3837-L3866) | The same scratchpad conversion and the same `batch_valid` DFB in the sparse path of `create_program_mcast_in0_artifacts`. That path binds the same kernels and serves sparse matmul over `prefetcher_pipes`. |
| [../kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp](../kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp#L175-L195) | Reads the sparsity page through a `Scratchpad`, with a `scoped_lock` around each read. Writes the batch flags to `batch_valid` instead of the BRISC mailbox. |
| [../kernels/dataflow/reader_bmm_tile_layout_in0_receiver_metal2.cpp](../kernels/dataflow/reader_bmm_tile_layout_in0_receiver_metal2.cpp#L45-L71) | Writes the batch flags to `batch_valid` instead of the BRISC mailbox. |
| [../kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp](../kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp#L304-L317) | Reads the sparsity page through a `Scratchpad`, as the in0 sender does. |
| [../kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp](../kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp#L332-L349) | Reads the batch flags from `batch_valid` instead of the BRISC mailbox. |

The scratchpad code in the kernels compiles only when `SPARSITY` is defined, and the `batch_valid` code only when `BATCH_VALID_DFB` is defined. Only the two sparse paths define `SPARSITY`, and they define `BATCH_VALID_DFB` only with `nnz=None`, never in indexed mode. In all other builds, which include every dense matmul factory, the changed kernels differ from before in two ways only: the in0 sender and the in0 receiver no longer include `ckernel.h` and `ckernel_defs.h`, and the in0 sender, the in0 receiver and the compute kernel each have a `static_assert` that `get_batch_from_reader` is false.

## Changes and reasons

### Multicast NoC (§11)

The factory passed `preferred_noc_for_dram_write(arch)` to the multicast helper, and that function returns NOC_1 on every arch. On Quasar the helper rejects NOC_1 ([mcast_impl.cpp:505-507](../../../../kernel_lib/mcast/host/mcast_impl.cpp#L505-L507)), and Quasar data movement kernels are built with `NOC_INDEX=0` ([kernel.cpp:1581-1588](../../../../../../../tt_metal/impl/kernels/kernel.cpp#L1581-L1588)). The factory now passes NOC_0 when `is_quasar` is set ([factory:1123-1127](factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L1123-L1127)). WH/BH still pass `in0_noc`. For NOC_0 the helper orders the multicast rectangle from the low corner to the high corner, which Quasar requires. Quasar ignores the `config_1xx` NOC fields of the data movement kernels, so those fields did not change.

### Data movement self-loop DFBs replaced by scratchpads (§6, `dm_self_loop_dfbs.md`)

The in0 sender and the in1 sender/writer each bound their own sparsity-page DFB as both PRODUCER and CONSUMER. Quasar rejects such a binding at program creation ([dfb.cpp:596-610](../../../../../../../tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp#L596-L610)). As `dm_self_loop_dfbs.md` prescribes, each of the two DFBs became a `ScratchpadSpec` of the same size (one aligned page), bound to the same kernel ([factory:513-533](factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L513-L533)). In the kernels:

- `reserve_back`, `push_back`, `wait_front` and `pop_front` on the page are gone. Each kernel read the write pointer once, before the only `push_back`, so no index needs to survive.
- The NoC read writes into the scratchpad.
- Element reads go through the scratchpad.

On WH/BH the kernels issue the same NoC reads and the same element reads as before. Only the SRAM address of the page can differ.

The conversion also removes two more Quasar problems:

- The in1 sender/writer pushed one entry into its page DFB and never popped that entry. On Quasar, the `finish()` that the framework calls at kernel end would wait for that entry forever (§7).
- In indexed mode, the in1 page holds the UINT16 group ids, so its DFB had format `UInt16`, which Quasar does not support. A scratchpad has no format.

### Cache handling for the scratchpad reads (§8.3, §12)

On Quasar, `DataflowBuffer::get_write_ptr()` returns an uncached address on data movement cores, so the old code read the page around the cache. A scratchpad is read through the cache, and the NoC writes the page directly to SRAM. Each read now takes `scoped_lock(i, 1)` ([in0 kernel:180-186](../kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp#L180-L186)), which invalidates the cache range before the read and flushes it after the read. On WH/BH both cache operations are empty ([scoped_lock_cache_ops.h:14-30](../../../../../../../tt_metal/hw/inc/internal/scoped_lock_cache_ops.h#L14-L30)). The embedding kernels use the same pattern.

Tested on craq-sim, which models the data movement caches: without the lock, the B-sparse case hung until the timeout killed it. That case reads two different mask pages into the same scratchpad. My guess is that the in0 sender read outer batch 1's flags from the cached copy of page 0, multicast three batches instead of two, and then waited for receivers that had already finished their `nnz` batches. With the lock, the same case passes.

### fp32 partials reload through SrcA on Quasar

With `fp32_dest_acc_en=True` and the K reduction split across blocks, the compute kernel reloads the Float32 partial sums into DEST between K blocks. On WH/BH the sparse factory marks the buffer that the reload reads `UnpackToDest`, so the reload keeps full fp32 precision. On Quasar the factory no longer sets that mark ([factory:246-261](factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L246-L261)). The reload then goes through SrcA, which rounds the partial sums to TF32 (10 mantissa bits) at each K-block boundary.

The reason is that on Quasar, unpack-to-dest is a mode of the whole compute kernel, not of one buffer: one `UnpackToDest` buffer sets `UnpackToDestEn` for the kernel ([genfiles.cpp:1172-1179](../../../../../../../tt_metal/jit_build/genfiles.cpp#L1172-L1179)). In that mode, `llk_pack_dest_section_done` does not zero the DEST half after it packs a section ([llk_pack_common_api.h:145-169](../../../../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_pack_common_api.h#L145-L169)), and `matmul_block` adds into DEST. Before this change, each later use of a DEST half added onto the result left there by its previous use. With five batches, the output norm grew by √2, √3, √4 and √5, and PCC dropped to 0.71, 0.58, 0.49 and 0.45.

This change follows the LLK design and an existing fix in another op:

- [#49510](https://github.com/tenstorrent/tt-metal/issues/49510) (closed 2026-10-05): the LLK owners keep unpack-to-dest kernel-wide. A kernel either unpacks to DEST or it does not, and kernels that mix both are not a goal. The matmul kernel mixes both: `matmul_block` unpacks to SrcA/SrcB.
- [#62518](https://github.com/tenstorrent/tt-metal/pull/62518) (merged 2026-10-28, fixes [#62517](https://github.com/tenstorrent/tt-metal/issues/62517)): on Quasar, layer_norm stops requesting `UnpackToDest` for its accumulator and restores the accumulator through SrcA.
- [#62102](https://github.com/tenstorrent/tt-metal/issues/62102) (open): the experimental Quasar unified matmul has the same cause. There the symptom is a hang.

The same condition also turns off `bias_reload_alias` on Quasar. The `intermed0_reload_alias` buffer exists only to carry `UnpackToDest` beside the SrcA view of the partials that the bias add reads. Without the alias, `MM_PARTIALS_RELOAD_ALIAS` is not defined, so on Quasar the compute kernel no longer compiles the WH/BH-only `evil_set_read_ptr` ([compute kernel:123-128](../kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp#L123-L128)). Fused bias with fp32 now compiles and passes.

### Batch flags through a dataflow buffer

This change concerns `nnz=None` only. Without `nnz`, the host does not know how many batches are valid. The in0 sender reads the sparsity mask on the device, decides for each batch whether to skip it, and multicasts that decision to the in0 receivers. Each of these in0 kernels (the in0 sender and every in0 receiver) must pass the decision to the compute kernel on its own node, so that the compute kernel skips the same batches as the in0 kernels.

Until this change, the in0 kernels passed the decision with `ckernel::mailbox_write` from the BRISC data movement processor to the three compute processors (unpack, math and pack), and the compute kernel read it with `mailbox_read(ckernel::ThreadId::BriscThreadId)`. Quasar has no mailbox from data movement cores to the compute processors. So on Quasar the compute kernel had a `static_assert`, and `nnz=None` did not compile.

The decision now goes through a DFB named `batch_valid`, on every arch:

- **Buffer.** Two entries of 16 bytes each (`hal::get_l1_alignment()`, the same on Wormhole, Blackhole and Quasar), format `RawUInt16` ([sparse factory:499-511](factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L499-L511), [dense factory:3854-3866](../factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp#L3854-L3866)). The host creates and binds the buffer only with `nnz=None`. On each node, the in0 kernel is the producer and the compute kernel is the consumer ([sparse factory:618-626](factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L618-L626)). Both factories run the compute kernel on every node that runs an in0 kernel.
- **Write.** For each batch, the in0 kernel reserves one entry, stores the flag (1 for a valid batch, 0 for a skipped batch) and pushes the entry, before it skips the batch itself ([in0 sender:237-247](../kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp#L237-L247), [in0 receiver:61-71](../kernels/dataflow/reader_bmm_tile_layout_in0_receiver_metal2.cpp#L61-L71)). The store is inside a `scoped_write_lock`. On Quasar, the release of that lock writes the flag from the data movement cache to SRAM before `push_back` gives the entry to the compute kernel. On Quasar, the guide requires the scoped locks whenever a data movement kernel reads or writes DFB memory through a pointer ([quasar_porting.md:210](../../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L210-L210)). On WH/BH the lock does nothing.
- **Read.** For each batch, the compute kernel waits for one entry, reads the flag with `read_tile_value<uint16_t>(0, 0)`, pops the entry, and skips the batch if the flag is 0 ([compute kernel:332-349](../kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp#L332-L349)). `read_tile_value` reads the flag on the unpack processor and sends it to the math and pack processors through their mailboxes, so all three processors take the same branch.
- **Bare wait and pop on Quasar.** `read_tile_value` is a plain SRAM read, not an unpack instruction, so on Quasar the `wait_front` and `pop_front` around it count as a bare pair (TEN-4746, [#48552](https://github.com/tenstorrent/tt-metal/issues/48552); [quasar_porting.md:142](../../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L142-L142)). On Quasar only, the compute kernel calls `dummy_unpack` between the read and the pop, as it already does for its other bare pairs.
- **Format.** Quasar does not support the `UInt16` and `UInt32` DFB formats. `RawUInt16` is supported on all archs, and nothing unpacks the entry, so the format only labels the bytes.
- **Define.** A binding that the host does not declare produces no `dfb::batch_valid` name in the kernel. So the three kernels use the buffer only under the define `BATCH_VALID_DFB`, which the host sets exactly when it binds the buffer ([sparse factory:579-585](factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L579-L585)). The factories use other defines for the same purpose, for example `SPARSITY` and `IN0_TRANSPOSE_TILE`. In each of the three kernels, a `static_assert` stops the build if the define and the `get_batch_from_reader` argument disagree.
- **Cost.** 32 bytes of SRAM per node, only with `nnz=None`. The speed was not measured.

The change applies to all archs so that all archs have one mechanism, not a mailbox on WH/BH and a DFB on Quasar. [#34407](https://github.com/tenstorrent/tt-metal/pull/34407) made the same move for `ttnn.manual_seed`: it replaced a BRISC mailbox handoff with a circular buffer that the compute kernel reads with `read_tile_value`, on all archs. [#34228](https://github.com/tenstorrent/tt-metal/issues/34228) asked for the same change in sparse matmul. The guide says that a Quasar fix that would change WH/BH goes behind `ARCH_QUASAR` ([quasar_porting.md:32](../../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L32-L32)). A version behind `ARCH_QUASAR` was possible; the version for all archs was chosen on 2026-11-05, with Blackhole to be tested in CI.

Other edits that come with this change:

- The in0 sender and the in0 receiver no longer include `ckernel.h` and `ckernel_defs.h`. They needed these headers only for `mailbox_write`, and Quasar builds already left them out.
- In the dense factory, a comment said that the in0 kernels run on BRISC because of the mailbox. It now says that the placement matches the sparse factory ([dense factory:3877-3878](../factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp#L3877-L3878)). The placement did not change.

## Checked and not needed

- §4 `opt_level`: unchanged; the compute kernel already states O3.
- Gen2 hardware configs (`gen2_hardware_configs.md`): the data movement configs set only `config_1xx`. On Quasar an unset `config_2xx` means the defaults, with implicit sync on. The compute config sets `unpack_modes`, which in this API is a field common to both generations, so there is no `std::get` on a generation variant.
- §7 implicit sync: not disabled, and no credit stall or double count was seen. (The dense matmul factories still disable implicit sync on Quasar; this op does not copy that.)
- §7 `finish()` balance: apart from the in1 page above, every DFB balances on every path: in0, in1, `out` (including padded subblocks), `intermed0` (in every spill, `packer_l1_acc` and bias branch), the per-group bias, and `batch_valid` (one push and one pop per batch, whether the batch is valid or skipped). No run hung at kernel end.
- §7 `pack_init` after `pack_reconfig_data_format`, and `dummy_unpack` between a bare `wait_front` and `pop_front`: the shared compute kernel already has these Quasar guards. The bias and `packer_l1_acc` paths pass.
- §7 partials aliased onto the output at an offset: `out` and `intermed0` alias one region from offset 0 with the same size (one output block), so this op does not use the offset pattern. That aliasing passes.
- §7 semaphores with a non-zero initial value: none. `SemaphoreSpec` has no initial value, and the multicast helper's semaphores start at 0.
- §11 degenerate grids: the receiver rectangle comes from the real set of cores; 2x2, 4x2 and 4x4 grids pass.
- §8.1 build errors: none.
- §5 `fifo_page_size`: not used; the kernels use `get_tile_size` and `get_entry_size`.

## Open problems

None.

## Follow-ups

1. **Blackhole run (CI).** The "Sanity tests" workflow, with its default inputs, runs two jobs on Blackhole p100 and p150b that cover this branch:
   - "ttnn matmul group" ([ttnn_sanity_tests.yaml:160-173](../../../../../../../tests/pipeline_reorg/ttnn_sanity_tests.yaml#L160-L173)) runs `tests/ttnn/unit_tests/operations/matmul`, which includes the sparse matmul tests and the dense matmul tests.
   - "ttnn misc ops group" ([ttnn_sanity_tests.yaml:242-257](../../../../../../../tests/pipeline_reorg/ttnn_sanity_tests.yaml#L242-L257)) runs `tests/ttnn/unit_tests/operations/transformers`, which includes [test_prefetcher_BH_tensor_large.py](../../../../../../../tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_tensor_large.py#L1956-L1960). That file has `nnz_none` cases and several more tests that call the op without `nnz`, so it also runs the `batch_valid` buffer in the dense factory.
2. **Sparse matmul over `prefetcher_pipes`: Blackhole only.** The `sparse_matmul` docstring allows only Tensor-prefetcher pipes for `prefetcher_pipes` ([matmul_nanobind.cpp:1127](../../matmul_nanobind.cpp#L1127-L1127)). The Tensor prefetcher needs programmable DRAM cores, and tt-metal registers that core type only on Blackhole ([bh_hal.cpp:351-356](../../../../../../../tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp#L351-L356)); Wormhole and Quasar do not have it. This path got the scratchpad conversion and the `batch_valid` buffer in `create_program_mcast_in0_artifacts`, and only follow-up 1 tests it. That function still marks the fp32 partials `UnpackToDest` on Quasar (follow-up 3), which matters for this op only if the Tensor prefetcher comes to Quasar.
3. **Dense matmul with fp32 accumulation on Quasar.** The dense 1D factory marks the fp32 partials `UnpackToDest` in `create_program_mcast_in0_artifacts` ([dense factory:4235-4259](../factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp#L4235-L4259)) and in `create_program_mcast_in1_artifacts` ([dense factory:5808-5832](../factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp#L5808-L5832)), with the cause described in [fp32 partials reload through SrcA on Quasar](#fp32-partials-reload-through-srca-on-quasar). On craq-sim, with a 2x2 grid:
   - `ttnn.matmul` with mcast_in0 and fp32 gives output tiles 2 and 3 of each core at PCC near 0.70.
   - `ttnn.matmul` with mcast_in1 and fp32 gives PCC 0.52.
   - `ttnn.linear` with bias and fp32 fails to compile on `evil_set_read_ptr`.

   On 2026-11-05 the same Quasar-only `reload_partials_to_dest` condition was tried in both functions. All of these cases then passed on craq-sim with LLK asserts on and off, and Wormhole passed `test_matmul.py` and `test_linear.py` with `-k "1d or 1D or fp32 or systolic or bias"` (465 passed, no failures). The change was removed from this branch afterwards, because it belongs to dense matmul, not to this op. Not run: `create_program_gather_in0_artifacts` in the same file marks its partials the same way, and its compute kernel has no Quasar handling. The other mainline matmul factories (2D multicast, Reuse optimized, DRAM-sharded) also mark their fp32 partials the same way. Open issue [#61018](https://github.com/tenstorrent/tt-metal/issues/61018) proposes adding `UnpackToDest` to the experimental Quasar matmul factories, which would bring the same problem there.
4. **In-tree tests on Quasar.** Quasar supports no BFP format ([tt_backend_api_types.cpp:97-123](../../../../../../../tt_metal/common/tt_backend_api_types.cpp#L97-L123)), and every in-tree sparse matmul test uses `bfloat8_b` or `bfloat4_b` weights. A `bfloat8_b` output already fails in the output pre-zeroing `zeros_like` ([sparse_matmul_device_operation.cpp:658-667](sparse_matmul_device_operation.cpp#L658-L667)). Adding a `bfloat16` weight value to some existing parametrizations, with the BFP values skipped on Quasar, would let the in-tree suite cover Quasar. The op only forwards the dtype, so the op itself has nothing to guard (§7).
5. **Duplicate Quasar test.** `test_quasar_sparse_matmul_interm0_format_tile_size` ([test_sparse_matmul.py:1419-1481](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py#L1419-L1481)) tests `ttnn.experimental.quasar.sparse_matmul`. With bf16 weights, the mainline `test_sparse_matmul_interm0_format_tile_size` covers the same case for this op.
6. **Mailbox code in the legacy kernels.** The legacy (not Metal 2.0) copies of the three kernels, `reader_bmm_tile_layout_in0_sender_padding.cpp`, `reader_bmm_tile_layout_in0_receiver.cpp` and `bmm_large_block_zm_fused_bias_activation.cpp`, still hold the mailbox handoff. Only the CCL fusion factories bind these copies, and they always pass `get_batch_from_reader` as false (for example [ccl_fusion_mcast_1d.cpp:372-378](../../../experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_1d.cpp#L372-L378)), so the mailbox code never runs. The headers of the Metal 2.0 copies say that a change to one copy likely belongs in the other too. Removing the mailbox code from the legacy copies is left for whoever retires them.
7. **gpt-oss on Quasar.** The Quasar copy of gpt-oss decode calls `ttnn.sparse_matmul(..., nnz=None)` ([decode.py:73-91](../../../../../../../models/experimental/ops/quasar/gpt_oss/tt/experts/decode.py#L73-L91)), which no longer stops at the compile error. Not run: the model uses BFP weights, which Quasar does not support.

## WH/BH parity

- **Quasar only.** Two choices are behind `is_quasar` in the sparse factory ([factory:142](factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L142-L142)): the multicast NoC and the fp32 partials reload. WH/BH still pass `in0_noc` and still mark the reload `UnpackToDest`. One call is behind `ARCH_QUASAR` in the compute kernel: the `dummy_unpack` before the `batch_valid` pop.
- **All archs, same behavior.** The scratchpad conversion keeps the same NoC reads, the same element reads, the same page size and the same kernel for each page. The removed FIFO calls only moved credits on a buffer that one kernel used alone. The `scoped_lock` cache operations are empty except in Quasar data movement builds.
- **All archs, changed on purpose.** With `nnz=None`, the batch flags go through `batch_valid` instead of the BRISC mailbox, in the sparse factory and in the sparse path of the dense 1D factory (see [Batch flags through a dataflow buffer](#batch-flags-through-a-dataflow-buffer)). The kernels skip the same batches as before.
- Wormhole: see [Results](#results).
- Blackhole: not run. The CI jobs are in follow-up 1. Commands:
  - `pytest tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py`
  - `pytest tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_tensor_large.py -k sparse_matmul`
  - `pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py tests/ttnn/unit_tests/operations/matmul/test_linear.py`

## How to rerun

The ad-hoc scripts are not in the repository. They are in the session scratchpad `/tmp/claude-1211409539/-localdev-fplavec-git-2026-11-04-quasar-sparse-matmul-tt-metal/960470a7-da8c-490c-bbba-d06674f147f8/scratchpad`:

- `qsr_sparse_check.py`: the check script. It runs on any arch.
- `qsr_env.sh`: the craq-sim environment, including `TTSIM_DRAM_CHANNEL_SIZE=3G`, which matches the 3 GiB DRAM size in the simulator's `soc_descriptor.yaml`.
- `run_cases.sh <tag> <case>...`: runs each case in its own process on craq-sim and writes `<tag>_summary.txt`. `run_cases_wh.sh` does the same on the local Wormhole or Blackhole device.

Examples, from the repository root:

```bash
SP=/tmp/claude-1211409539/-localdev-fplavec-git-2026-11-04-quasar-sparse-matmul-tt-metal/960470a7-da8c-490c-bbba-d06674f147f8/scratchpad
source $SP/qsr_env.sh
timeout --signal=INT --kill-after=20s 20m python $SP/qsr_sparse_check.py --mode b_sparse
timeout --signal=INT --kill-after=20s 20m python $SP/qsr_sparse_check.py --mode b_sparse --no-nnz --zero-every -1
timeout --signal=INT --kill-after=20s 20m python $SP/qsr_sparse_check.py --mode gather --experts 8 --bias --ckc --fp32
timeout --signal=INT --kill-after=20s 20m python $SP/qsr_sparse_check.py --mode b_sparse --ckc --fp32 --report-all
timeout --signal=INT --kill-after=20s 20m python $SP/qsr_sparse_check.py --mode dense --ckc --fp32 --mcast-in1 --mt 8 --nt 4   # follow-up 3
bash $SP/run_cases.sh mytag "--mode b_sparse --no-nnz" "--mode a_sparse --no-nnz --grid 4 2"
```

Modes: `b_sparse`, `a_sparse`, `ab_sparse`, `gather` and `dense`. `--no-nnz` calls the op with `nnz=None`. `--zero-every N` sets the mask to zero for every Nth group; 0 zeroes no group and -1 zeroes every group. Without `--ckc` the op's default compute kernel config is used; `--fp32` and `--l1acc` take effect only together with `--ckc`. `--report-all` prints the PCC of every output slot and every output tile column. `--grid X Y` needs `--nt` to be a multiple of X times Y. In `dense` mode, `--bias` calls `ttnn.linear` with a bias and `--mcast-in1` uses a 1D config with `mcast_in0=False`; `--mt` must then be a multiple of the number of cores.
