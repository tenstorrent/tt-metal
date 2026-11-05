# Quasar uplift report: `ttnn.sparse_matmul`

Branch `fplavec/quasar_sparse_matmul`, 2026-11-04. Process: [quasar_porting.md](../../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md). This report is not for commit; delete it before merge.

## Status: RED (partial uplift)

The op now gives correct results on the Quasar simulator (craq-sim) in its main modes, all with `nnz` given: B sparse, A sparse, A and B sparse, and indexed (gather) mode with and without fused bias. Three configurations stay blocked by capabilities outside this op (details in [Open problems](#open-problems)):

1. `fp32_dest_acc_en=True` gives wrong results. The cause is in the Quasar LLK API, and dense matmul has the same problem.
2. `nnz=None` does not compile, because Quasar has no mailbox from data movement cores to compute.
3. Fused bias with `fp32_dest_acc_en=True` does not compile, because `evil_set_read_ptr` exists only on WH/BH.

The op directory and namespace did not change. Nothing was copied from `experimental/quasar/`.

## Results

**Quasar (craq-sim).** bf16 weights and bf16 output; Quasar has no BFP formats (see [Follow-ups](#follow-ups)). Each case ran in its own process, once with `TT_METAL_LLK_ASSERTS=1` and once without, unless the table says otherwise. Kt is the number of tiles along K; with `in0_block_w=1` and Kt=4 the compute kernel spills partial sums between K blocks. Before the change, every case failed at program creation with `TT_FATAL: Quasar has one NoC; configure the multicast NoC as NOC_0`.

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
| B sparse; indexed | 4x4 grid (16 cores) | pass (LLK asserts off only) |
| B sparse; indexed | `fp32_dest_acc_en=True` | wrong output, PCC 0.85 ([problem 1](#1-fp32_dest_acc_entrue-gives-wrong-results)) |
| B sparse | `nnz=None` | compile error ([problem 2](#2-nnznone-does-not-compile-on-quasar)) |
| Indexed with fused bias | `fp32_dest_acc_en=True` | compile error ([problem 3](#3-fused-bias-with-fp32_dest_acc_entrue-does-not-compile-on-quasar)) |

The in-tree tests cannot run on Quasar as written, because all of them use `bfloat8_b` or `bfloat4_b` weights. The Quasar cases above come from an ad-hoc script that copies the setups of the in-tree tests with bf16 weights (see [How to rerun](#how-to-rerun)).

**Wormhole (n300).** [test_sparse_matmul.py](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py) and [test_sparse_matmul_indexed.py](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py): 93 passed, 2 skipped, both before and after the change, with the same outcome for every test. The 2 skipped tests are Quasar-only tests of `ttnn.experimental.quasar.sparse_matmul`.

**Blackhole.** Not run; this machine has no Blackhole device.

## Files changed

| File | Reason |
|---|---|
| [factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp](factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L488-L508) | The two sparsity-page DFBs become kernel-private scratchpads, and the in0 multicast uses NOC_0 on Quasar. |
| [../factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp](../factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp#L3826-L3843) | The same scratchpad conversion in the sparse path of `create_program_mcast_in0_artifacts`. That path binds the same two kernels and serves sparse matmul over `prefetcher_pipes`. |
| [../kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp](../kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp#L180-L192) | Reads the sparsity page through a `Scratchpad`, with a `scoped_lock` around each read. The FIFO calls on the page are gone. |
| [../kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp](../kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp#L304-L317) | The same change as in the in0 sender. |

The kernel changes compile only when the `SPARSITY` define is set, and only the two sparse paths set `SPARSITY`.

## Changes and reasons

### Multicast NoC (§11)

The factory passed `preferred_noc_for_dram_write(arch)` to the multicast helper, and that function returns NOC_1 on every arch. On Quasar the helper rejects NOC_1 ([mcast_impl.cpp:505-507](../../../../kernel_lib/mcast/host/mcast_impl.cpp#L505-L507)), and Quasar data movement kernels are built with `NOC_INDEX=0` ([kernel.cpp:1581-1588](../../../../../../../tt_metal/impl/kernels/kernel.cpp#L1581-L1588)). The factory now passes NOC_0 when `device->arch() == tt::ARCH::QUASAR` ([factory:1067-1071](factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L1067-L1071)). WH/BH still pass `in0_noc`. For NOC_0 the helper orders the multicast rectangle from the low corner to the high corner, which Quasar requires. Quasar ignores the `config_1xx` NOC fields of the data movement kernels, so those fields did not change.

### Data movement self-loop DFBs replaced by scratchpads (§6, `dm_self_loop_dfbs.md`)

The in0 sender and the in1 sender/writer each bound their own sparsity-page DFB as both PRODUCER and CONSUMER. Quasar rejects such a binding at program creation ([dfb.cpp:596-610](../../../../../../../tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp#L596-L610)). As `dm_self_loop_dfbs.md` prescribes, each of the two DFBs became a `ScratchpadSpec` of the same size (one aligned page), bound to the same kernel ([factory:488-508](factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L488-L508)). In the kernels:

- `reserve_back`, `push_back`, `wait_front` and `pop_front` on the page are gone. Each kernel read the write pointer once, before the only `push_back`, so no index needs to survive.
- The NoC read writes into the scratchpad.
- Element reads go through the scratchpad.

On WH/BH the kernels issue the same NoC reads and the same element reads as before. Only the SRAM address of the page can differ.

The conversion also removes two more Quasar problems:

- The in1 sender/writer pushed one entry into its page DFB and never popped that entry. On Quasar, the `finish()` that the framework calls at kernel end would wait for that entry forever (§7).
- In indexed mode, the in1 page holds the UINT16 group ids, so its DFB had format `UInt16`, which Quasar does not support. A scratchpad has no format.

### Cache handling for the scratchpad reads (§8.3, §12)

On Quasar, `DataflowBuffer::get_write_ptr()` returns an uncached address on data movement cores, so the old code read the page around the cache. A scratchpad is read through the cache, and the NoC writes the page directly to SRAM. Each read now takes `scoped_lock(i, 1)` ([in0 kernel:185-191](../kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp#L185-L191)), which invalidates the cache range before the read and flushes it after the read. On WH/BH both cache operations are empty ([scoped_lock_cache_ops.h:14-30](../../../../../../../tt_metal/hw/inc/internal/scoped_lock_cache_ops.h#L14-L30)). The embedding kernels use the same pattern.

Tested on craq-sim, which models the data movement caches: without the lock, the B-sparse case hung until the timeout killed it. That case reads two different mask pages into the same scratchpad. My guess is that the in0 sender read outer batch 1's flags from the cached copy of page 0, multicast three batches instead of two, and then waited for receivers that had already finished their `nnz` batches. With the lock, the same case passes.

## Checked and not needed

- §4 `opt_level`: unchanged; the compute kernel already states O3.
- Gen2 hardware configs (`gen2_hardware_configs.md`): the data movement configs set only `config_1xx`. On Quasar an unset `config_2xx` means the defaults, with implicit sync on. The compute config sets `unpack_modes`, which in this API is a field common to both generations, so there is no `std::get` on a generation variant.
- §7 implicit sync: not disabled, and no credit stall or double count was seen. (The dense matmul factories still disable implicit sync on Quasar; this op does not copy that.)
- §7 `finish()` balance: apart from the in1 page above, every DFB balances on every path: in0, in1, `out` (including padded subblocks), `intermed0` (in every spill, `packer_l1_acc` and bias branch) and the per-group bias. No run hung at kernel end.
- §7 `pack_init` after `pack_reconfig_data_format`, and `dummy_unpack` between a bare `wait_front` and `pop_front`: the shared compute kernel already has these Quasar guards. The bias and `packer_l1_acc` paths pass.
- §7 partials aliased onto the output at an offset: `out` and `intermed0` alias one region from offset 0 with the same size (one output block), so this op does not use the offset pattern. That aliasing passes.
- §7 semaphores with a non-zero initial value: none. `SemaphoreSpec` has no initial value, and the multicast helper's semaphores start at 0.
- §11 degenerate grids: the receiver rectangle comes from the real set of cores; 2x2, 4x2 and 4x4 grids pass.
- §8.1 build errors: none.
- §5 `fifo_page_size`: not used; the kernels use `get_tile_size` and `get_entry_size`.

## Open problems

### 1. `fp32_dest_acc_en=True` gives wrong results

Owner: LLK team. Dense matmul has the same problem. Seen on craq-sim only.

- **Symptom.** The compute kernel uses the two halves of DEST in turn (`double_buffer_dest`). The first use of each half gives the correct result. Each later use of a half adds the result left in that half by its previous use. With five batches, the output norm grows by √2, √3, √4 and √5, and PCC drops to 0.71, 0.58, 0.49 and 0.45. Dense `ttnn.matmul` (1D mcast_in0, 2x2 grid, `per_core_N=4`, Kt=1) shows the same pattern: output tiles 2 and 3 of each core have PCC near 0.70 and norm ratio near 1.41, while tiles 0 and 1 are correct. A dense matmul with one output tile per DEST half passes only because it never reuses a half.
- **Cause.** For fp32 accumulation, the matmul factories mark the Float32 partials DFB `UnpackToDest`, for the reload between K blocks ([factory:810-837](factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp#L810-L837)). On Quasar, `UnpackToDestEn` is one flag for the whole compute kernel ([genfiles.cpp:1172-1179](../../../../../../../tt_metal/jit_build/genfiles.cpp#L1172-L1179)). When that flag is set, `llk_pack_dest_section_done` takes its unpack-to-dest branch, which does not zero the DEST half after packing ([llk_pack_common_api.h:145-169](../../../../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_pack_common_api.h#L145-L169)). `matmul_block` adds into DEST, so it needs that zero.
- **Tested workaround, not applied.** With the partials DFB marked `UnpackToSrc` on Quasar, every fp32 case passed with PCC 1.0 on every output: Kt=1, Kt=4, Kt=4 with `packer_l1_acc`, and A sparse. The cost is that the reloaded partial sums are rounded to TF32 at each K-block boundary. I did not apply the workaround, because every dense matmul factory has the same problem and the fix belongs in the Quasar LLK API: zero DEST for a section that the unpacker did not write.

### 2. `nnz=None` does not compile on Quasar

- **Symptom.** `static assertion failed: get_batch_from_reader (batch sparsity) is unsupported on Quasar until the DM->TRISC is_batch_valid mailbox handoff is implemented` ([compute kernel:325-336](../kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp#L325-L336)).
- **Cause.** Without `nnz`, the readers tell the compute kernel, batch by batch, whether to skip the batch. They do this with `ckernel::mailbox_write` from BRISC. Quasar has no mailbox from data movement cores to the TRISCs.
- **Impact.** The Quasar copy of gpt-oss decode calls `ttnn.sparse_matmul(..., nnz=None)` on purpose ([decode.py:73-91](../../../../../../../models/experimental/ops/quasar/gpt_oss/tt/experts/decode.py#L73-L91)).
- **Needs an owner decision.** One option is a runtime mailbox from data movement cores to compute. Another is to pass the per-batch flag in a small DFB that compute reads with `read_tile_value`; I have not tried that. §7 forbids an interface that only this op defines.

### 3. Fused bias with `fp32_dest_acc_en=True` does not compile on Quasar

- **Symptom.** `'class DataflowBuffer<>' has no member named 'evil_set_read_ptr'` ([compute kernel:123-128](../kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp#L123-L128)).
- **Cause.** This configuration adds the `intermed0_reload_alias` DFB, and the compute kernel aligns the read pointer of that DFB with `evil_set_read_ptr`, which exists only on WH/BH ([dataflow_buffer.h:436-441](../../../../../../../tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L436-L441)).
- **Needs.** A sanctioned Quasar API to align the read pointer of an alias DFB (runtime team), or a reload design without the alias. Problem 1 also applies to this configuration.

## Follow-ups

1. **Blackhole-only path, built but not run.** Sparse matmul over `prefetcher_pipes` got the same scratchpad conversion in `create_program_mcast_in0_artifacts`. Run on Blackhole: `pytest tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_tensor_large.py -k sparse_matmul`.
2. **In-tree tests on Quasar.** Quasar supports no BFP format ([tt_backend_api_types.cpp:97-123](../../../../../../../tt_metal/common/tt_backend_api_types.cpp#L97-L123)), and every in-tree sparse matmul test uses `bfloat8_b` or `bfloat4_b` weights. A `bfloat8_b` output already fails in the output pre-zeroing `zeros_like` ([sparse_matmul_device_operation.cpp:658-667](sparse_matmul_device_operation.cpp#L658-L667)). Adding a `bfloat16` weight value to some existing parametrizations, with the BFP values skipped on Quasar, would let the in-tree suite cover Quasar. The op only forwards the dtype, so the op itself has nothing to guard (§7).
3. **Duplicate Quasar test.** `test_quasar_sparse_matmul_interm0_format_tile_size` ([test_sparse_matmul.py:1419-1481](../../../../../../../tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py#L1419-L1481)) tests `ttnn.experimental.quasar.sparse_matmul`. After problem 1 is fixed, and with bf16 weights, the mainline `test_sparse_matmul_interm0_format_tile_size` covers the same case for this op.

## WH/BH parity

- The only arch-dependent change is the multicast NoC, behind `device->arch() == tt::ARCH::QUASAR`. WH/BH still pass `in0_noc`.
- The scratchpad conversion keeps the same NoC reads, the same element reads, the same page size and the same kernel for each page. The removed FIFO calls only moved credits on a buffer that one kernel used alone. The `scoped_lock` cache operations are empty except in Quasar data movement builds.
- Wormhole: same results before and after (see [Results](#results)).
- Blackhole: not run. Commands:
  - `pytest tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py`
  - `pytest tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_tensor_large.py -k sparse_matmul`

## How to rerun

The ad-hoc Quasar script and its environment file are not in the repository. They are in the session scratchpad:

- `/tmp/claude-1211409539/-localdev-fplavec-git-2026-11-04-quasar-sparse-matmul-tt-metal/960470a7-da8c-490c-bbba-d06674f147f8/scratchpad/qsr_sparse_check.py`
- `/tmp/claude-1211409539/-localdev-fplavec-git-2026-11-04-quasar-sparse-matmul-tt-metal/960470a7-da8c-490c-bbba-d06674f147f8/scratchpad/qsr_env.sh` (craq-sim environment, including `TTSIM_DRAM_CHANNEL_SIZE=3G`, which matches the 3 GiB DRAM size in the simulator's `soc_descriptor.yaml`)

Examples, from the repository root:

```bash
SP=/tmp/claude-1211409539/-localdev-fplavec-git-2026-11-04-quasar-sparse-matmul-tt-metal/960470a7-da8c-490c-bbba-d06674f147f8/scratchpad
source $SP/qsr_env.sh
timeout --signal=INT --kill-after=20s 20m python $SP/qsr_sparse_check.py --mode b_sparse
timeout --signal=INT --kill-after=20s 20m python $SP/qsr_sparse_check.py --mode gather --experts 8 --bias --ckc --l1acc
timeout --signal=INT --kill-after=20s 20m python $SP/qsr_sparse_check.py --mode b_sparse --ckc --fp32 --report-all   # problem 1
```

Modes: `b_sparse`, `a_sparse`, `ab_sparse`, `gather` and `dense` (dense `ttnn.matmul` for comparison). Without `--ckc` the op's default compute kernel config is used; `--fp32` and `--l1acc` take effect only together with `--ckc`.
