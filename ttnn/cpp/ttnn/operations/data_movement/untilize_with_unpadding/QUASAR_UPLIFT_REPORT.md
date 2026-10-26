# Quasar uplift report: `untilize_with_unpadding`

**Status: GREEN, with one limit outside this op.** The op passes its test suite on Wormhole and, for BFLOAT16, on the Quasar simulator (craq-sim); see [Test results](#test-results). FLOAT32 input does not work on Quasar because of a gap in the shared untilize compute API (deferred item 4); the sibling `untilize` op has the same failure. Blackhole was not tested (no Blackhole board on the test machine).

Procedure: [`quasar_porting.md`](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md), plus the canonical recipe on branch `akertesz/op-porting-recipe` (`ai/audit/quasar_audit.md`, `ai/audit/cb_dfb_quasar_audit_helper.md`, `ai/post_port/semantic/dm_self_loop_dfbs.md`, `ai/post_port/style/sync_free_dfbs.md`, `ai/post_port/semantic/gen2_hardware_configs.md`).

Delete this file before merge.

## Summary

- The op was already Metal 2.0 on WH/BH. All five factories build `ProgramArtifacts` through `create_program_artifacts`, and every kernel uses the device 2.0 APIs (`DataflowBuffer`, `Noc`, `TensorAccessor`, `get_arg`).
- The audit found one Quasar blocker. In the sharded factory, same-shard-type sharded output (HEIGHT to HEIGHT, WIDTH to WIDTH, BLOCK to BLOCK) used a DFB borrowed from the output tensor, and the writer bound it as both producer and consumer. That is a data-movement self-loop. The Gen2 host validator rejects it with a `TT_FATAL` ([dfb.cpp:589-602](../../../../../../tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp#L589-L602)), so this path could not run on Quasar at all.
- The fix replaces that DFB with a `LocalTensorAccessor` over the output tensor. This is the same conversion `tilize_with_val_padding` received for its borrowed input shard in #57616 ([reader_unary_pad_height_width_sharded.cpp:24-43](../tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_height_width_sharded.cpp#L24-L43)).
- The audit also found a `finish()` imbalance in the block-interleaved writer, reached when the output drops one or more whole tile rows of the padded input. It hangs on Quasar, and in some of those inputs on WH/BH too. At the reviewers' request the fix is part of this change; see [Block-interleaved writer fix](#block-interleaved-writer-fix-57354-item-11). It is not `ARCH_QUASAR`-guarded, because on WH/BH it only turns hangs into correct output.
- Nothing else needed a change. The op stays in its directory and namespace. Nothing was copied into or taken from `experimental/quasar/`.

## Files changed

| File | Reason |
|---|---|
| [untilize_with_unpadding_multi_core_sharded_program_factory.cpp:220-252](device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp#L220-L252) | Remove the `SH_SHARDED_OUT` DFB and its two self-loop bindings. Bind the output tensor (`SH_OUTPUT`, accessor `out`) to the writer as a `TensorBinding`. Remove the `num_unpadded_output_rows` runtime arg, which only fed the removed `reserve_back` ([lines 389-404](device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp#L389-L404)). |
| [writer_unary_unpad_batch_rows_sharded.cpp:24-48](device/kernels/dataflow/writer_unary_unpad_batch_rows_sharded.cpp#L24-L48) | Write each row into the output shard through `LocalTensorAccessor<uint32_t>(tensor::out)` at a byte offset, instead of through the borrowed DFB's write pointer. Remove the `reserve_back`. |
| [writer_unary_unpad_width_16_sharded.cpp:19-44](device/kernels/dataflow/writer_unary_unpad_width_16_sharded.cpp#L19-L44) | Same conversion. The tile size now comes from `dfb::untilize_out`, the buffer whose faces the kernel copies, because `dfb::out` no longer exists. |
| [untilize_with_unpadding_device_operation.cpp:167-175](device/untilize_with_unpadding_device_operation.cpp#L167-L175) (and lines [247-250](device/untilize_with_unpadding_device_operation.cpp#L247-L250), [302-305](device/untilize_with_unpadding_device_operation.cpp#L302-L305)) | Comments only. They described the old borrowed circular buffer (`sharded_output_cb_index`). The rule they justify (same-shard-type output must be `BufferType::L1`) still holds, because `LocalTensorAccessor` rejects a DRAM tensor at compile time. |
| [writer_unary_stick_layout_wh_multicore_metal2.cpp:35-99](device/kernels/dataflow/writer_unary_stick_layout_wh_multicore_metal2.cpp#L35-L99) | Pop every sub-block the compute kernel pushes, including those of a tile row that starts at or past the output end, and set that tile row's row count to 0 explicitly. See [Block-interleaved writer fix](#block-interleaved-writer-fix-57354-item-11). This writer is shared with the `untilize` block factory. |
| [test_untilize_with_unpadding.py:215-245](../../../../../../tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py#L215-L245) | New test for HEIGHT_SHARDED to HEIGHT_SHARDED output of width 16. Before this test, no test reached `writer_unary_unpad_width_16_sharded.cpp`. The two heights run the 8-tile batch loop and the tail loop of that writer. |
| [test_untilize_with_unpadding.py:1032-1069](../../../../../../tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py#L1032-L1069) | New test for block-interleaved untilize that crops away whole tile rows: a tile row starting exactly at the output end, the same in two batch slices, one starting past the end, both on one core, and a combined height and width crop. |

## WH/BH parity claim

The self-loop conversion has no `ARCH_QUASAR` guard because it is behaviorally identical on WH/BH:

- **Before:** the writer called `reserve_back(n)` once on an empty DFB of exactly `n` entries, so it returned at once and moved no pointer. It then took `get_write_ptr()`, which is the base of this core's output shard, and wrote row `k` at `base + k * aligned_page_size` with a NoC read from its own SRAM.
- **After:** `LocalTensorAccessor(tensor::out)` gives the same base address (the tensor binding's address of this core's shard). The NoC reads have the same sources, sizes, order and destination addresses. The barriers and the `untilize_out` DFB calls are unchanged.
- **Width-16 tile size:** `get_tile_size(dfb::untilize_out)` equals the old `get_tile_size(dfb::out)`. Both DFBs had the same data format and the default 32x32 tile, and the JIT fills the tile size from those two values.
- **`.addr = 0` in `set_async_read_state`:** WH, BH and Quasar program only the NoC coordinate (and on BH and Quasar the upper address bits) in read `set_state`. The low address bits are ignored, as the kernel's own comment says.
- **Program cache:** the output address now reaches the writer through the tensor binding, which the framework refreshes on every dispatch, the same way the borrowed DFB address was refreshed before.

The block-interleaved writer fix does change WH/BH behavior, but only for inputs that hang there today; see [What the fix changes on WH/BH](#what-the-fix-changes-on-whbh).

## Quasar audit (`quasar_audit.md`)

**Check 1: CB/DFB redesign (`cb_dfb_quasar_audit_helper.md`).** Results per buffer:

| Buffer | Factory | Kernels | Class | 2xx status after this change |
|---|---|---|---|---|
| `in` / `out` | single core, multi-core interleaved, block interleaved, ND sharded | readers, untilize compute, writers | 1 (linear FIFO) | Portable |
| `SH_IN` (borrowed from input) | sharded | `reader_unary_sharded_metal2.cpp`, compute | 1 (reader posts the whole shard with one `push_back`, compute consumes it) | Portable |
| `SH_OUT` (`untilize_out`) | sharded | compute, all sharded writers | 1 | Portable |
| `SH_SHARDED_OUT` (borrowed from output) | sharded | same-shard writers | was 6 (DM self-loop, borrowed) | Removed; now `LocalTensorAccessor` |

No `get_local_cb_interface`, `fifo_page_size`, `get_pointer_to_cb_data`, `read_tile_value`, `get_tile_address` or `evil_set_*` appears in any in-scope kernel. Rollup: GREEN.

**Check 2: non-zero-initialized semaphores.** The op creates no semaphores.

## Guide items considered

| Item | Result |
|---|---|
| §6, §10: DM self-loop DFB | **Applied** (see Summary). |
| §7: `finish()` balance | Audited every DFB in every factory. One imbalance found, in the block-interleaved writer, on a path no existing test reached. **Fixed** in this change; see [Block-interleaved writer fix](#block-interleaved-writer-fix-57354-item-11). |
| §7: implicit sync | Not disabled anywhere. The DM configs come from [`create_reader/writer_datamovement_config()`](../../core/data_movement_kernel/datamovement_kernel_config.hpp#L24-L38) with the default `false`. All DM transfers use plain NoC calls with explicit `reserve_back`/`push_back`/`wait_front`/`pop_front`. |
| §7: `compute_kernel_hw_startup` once | Yes, at the start of each compute kernel. |
| §7: re-init on DFB change, `pack_reconfig` re-`pack_init` | Not applicable. Each compute kernel uses one input/output pair and `NoReconfigure`. |
| §7: bare wait/pop or reserve/push pairs in compute (TEN-4746) | None. The untilize helper has `pack_untilize_block` between each pair ([untilize_helpers.inl:228-271](../../../kernel_lib/untilize_helpers.inl#L228-L271)), and the width-16 copy kernel has `copy_tile` and `pack_tile` ([eltwise_copy_metal2.cpp:32-48](../../../kernel/compute/eltwise_copy_metal2.cpp#L32-L48)). |
| §7: uint16/uint32 formats | No format-specific code path in the op; see deferred item 3. |
| §7: `-inf`/NaN arithmetic, multicast (§11), reduce face geometry (§8.4) | Not used by this op. |
| §4: hardware configs (`gen2_hardware_configs.md`) | Zero sites. The compute config is the single `ComputeHardwareConfig` struct with only common fields (`enable_32_bit_dest`, `unpack_modes`). The DM configs come from the ttnn helpers, which set `config_2xx`. |
| §4: `unpack_modes` for Float32 | Already set to `UnpackToDest` on the input DFB whenever `fp32_dest_acc_en` is set. |
| §4: `opt_level` | Not changed (base-port concern). Compute kernels state `O3`; DM kernels use the default. |
| §8.3: uncached `get_read_ptr()` on Quasar DM | No change needed. The writers wrap `get_read_ptr()` in a `CoreLocalMem` NoC operand, and the NoC maps the uncached alias back to the cached view ([noc.h:131-141](../../../../../../tt_metal/hw/inc/api/dataflow/noc.h#L131-L141)). |
| §6: NoC loopback copies within one core | The same-shard writers still read their own SRAM through a NoC loopback, as before. §6 says the emulator can drop such reads. Not needed: all same-shard cases pass on craq-sim with exact output. If rows come back missing on the physical emulator, replace the loopback with a RISC copy. |
| §8.1: build errors | None hit. The host build passed, and every kernel the tests use compiled for Quasar. The only compiler message is the expected deprecation warning in deferred item 1. |

## Block-interleaved writer fix (#57354 item 11)

**The bug.** In the block-interleaved factory the writer's `total_num_rows` is the output height, while the work split walks the padded input height. When the output is at least 32 rows shorter than the padded input, a tile row can start at or past the output end. For a tile row starting exactly at the end, the old writer computed 0 rows and skipped `wait_front`/`pop_front` for all of that row's sub-blocks, which the compute kernel still pushes. For a tile row starting past the end, the row-count subtraction wrapped, but so did the write loop's bound (`start_row_id + (total_num_rows - start_row_id)` is exactly `total_num_rows` modulo 2^32), so the loop ran zero times and the sub-block was still popped. #57354 item 11 says that second case loops about 4 billion times; it does not.

- **Quasar:** the unpopped entries stay posted, so the writer and the compute kernel hang in `finish()` at kernel end.
- **WH/BH:** the output DFB holds exactly one sub-block (`num_entries = set.block_tiles`), so the compute kernel can be at most one sub-block ahead of the writer. The writer re-reads `start_row_id` in every `third_dim` pass, so on a core it skipped K = (sub-blocks per tile row) x `third_dim` sub-blocks in total. With K = 1 the compute kernel still finished and the output was correct, with one sub-block left in the buffer. With K of 2 or more the compute kernel waited in `reserve_back` forever, so the op hung.
- No existing test reached this: the block-factory tests keep the output height within one tile row of the padded input height.

**The fix** ([writer_unary_stick_layout_wh_multicore_metal2.cpp:35-99](device/kernels/dataflow/writer_unary_stick_layout_wh_multicore_metal2.cpp#L35-L99)), as proposed in #57354 item 11: a tile row that starts at or past the output end gets a row count of 0 explicitly, and `write_block` is called for every sub-block and always waits for and pops it, writing rows only when there are any. Both the exact-end and the past-end case now take this one explicit path.

### What the fix changes on WH/BH

- Outputs that never drop a whole tile row (every existing test, and every `untilize` call, which shares this writer): unchanged, because every sub-block already had rows to write.
- Tile rows that start past the output end: unchanged; the sub-block is popped and no row is written.
- K = 1: same output; the writer also pops the one sub-block that used to stay in the buffer.
- K of 2 or more: the op no longer hangs and gives correct output.

### Scenarios tested

All inputs are BFLOAT16, TILE, DRAM interleaved, with a row of 64 or more tiles so that the block-interleaved factory is selected; the kernel cache confirmed that `writer_unary_stick_layout_wh_multicore_metal2` was compiled in each run.

| # | Scenario | Padded input -> output | K | Old writer, Wormhole | Old writer, Quasar sim | Fixed writer, Wormhole | Fixed writer, Quasar sim |
|---|---|---|---|---|---|---|---|
| 1 | A tile row starts exactly at the output end | `[1,1,64,2048]` -> `[1,1,32,2048]` | 1 | correct | hang | pass | pass |
| 2 | Same, in two batch slices (`third_dim` = 2) | `[2,1,64,2048]` -> `[2,1,32,2048]` | 2 | hang | not run (scenario 1 already hangs) | pass | pass |
| 3 | Same, with several sub-blocks per tile row | `[1,1,64,8192]` -> `[1,1,32,8192]`, SRAM filled to about 12.7 KB free per core | 2 on WH, 4 on Quasar | hang | not run (scenario 1 already hangs) | pass | pass |
| 4 | A tile row starts past the output end | `[1,1,64,2048]` -> `[1,1,30,2048]` | 0 | correct | correct | pass | pass |
| 5 | One tile row exactly at the end and one past it, on one core | `[1,1,96,2048]` -> `[1,1,32,2048]` | 1 | correct | hang | pass | pass |
| 6 | Height and width cropped together, two batch slices | `[2,1,96,2048]` -> `[2,1,32,1000]` | 2 | not run | not run | pass | pass |

- Scenarios 1, 2, 4, 5 and 6 are the cases of the new test `test_untilize_with_unpadding_block_crops_whole_tile_rows`. On the Quasar simulator it passed with LLK asserts on and off.
- Scenario 3 needs a sub-block split, which the factory only chooses when the per-core SRAM budget is smaller than the block width. A scratch script reaches it by first allocating an SRAM tensor that leaves about 12.7 KB free per core (a budget of 3 tiles). The kernel cache confirmed the split: a 4-tile block in 2 sub-blocks on Wormhole, and an 8-tile block in 2-tile sub-blocks on the simulator. It is not a committed test, because the free-space window it needs depends on the device.
- Scenario 4 with the old writer confirms the wrap behavior described above, on both targets.
- "Hang" means the run was killed after its timeout (2 minutes on Wormhole, 5 to 10 minutes on the simulator); the Wormhole board was reset afterwards.

## Deferred and follow-up items

1. **Id-only untilize helper in two shared compute kernels.** [untilize_metal2.cpp:21-28](../untilize/device/kernels/compute/untilize_metal2.cpp#L21-L28) and [untilize_variable_num_blocks_metal2.cpp:29-36](../untilize/device/kernels/compute/untilize_variable_num_blocks_metal2.cpp#L29-L36) call the id-only `untilize<...>(num_blocks)`. On Quasar that form is `[[deprecated]]` ([id_only_helpers.hpp:7-21](../../../kernel_lib/id_only_helpers.hpp#L7-L21)), because a helper-owned `DataflowBuffer` drains when it goes out of scope (#60848). Kernels build with `-Wno-error=deprecated-declarations`, so this is a warning, not a build failure. In both kernels the helper call is the last statement, so the drain happens at kernel end; my guess is that no hang follows. The fix is to pass the kernel's own buffers by reference, as [untilize_wh_metal2.cpp:21-32](../untilize/device/kernels/compute/untilize_wh_metal2.cpp#L21-L32) does. #62092 lists this as a follow-up. These kernels are shared with `untilize` and other ops, so the change is not made here.
2. **`get_tile_size()` in a shared DM reader.** [reader_unary_interleaved_wh_multicore_metal2.cpp:36-37](../../eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore_metal2.cpp#L36-L37) (block-interleaved factory) reads its NoC transfer size with `dfb.get_tile_size()`. #58133 changed similar DM reads to `get_entry_size()` and gave Quasar staleness as the reason ([reader_unary_nd_sharded_blocks_metal2.cpp:31-34](../sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks_metal2.cpp#L31-L34)). Reading the code, the value looks correct on Quasar: the descriptor slot equals the logical DFB id, and the host fills the tile size for every kernel. If a wrong read size shows up, switch to `get_entry_size()`. That is byte-identical on WH/BH, because both binding factories size the entry as one tile.
3. **Formats.** The op passes through UINT32, INT32, FLOAT32 and BFLOAT8_B inputs. Quasar has no UInt32 or Bfp8_b device format. The op has no code path of its own for these formats (the `float32_dtype` compile-time arg only selects a 4-byte element size), so there is nothing to guard here. The limit belongs to the format/LLK layer. On the simulator, BFLOAT8_B input is rejected cleanly at program creation (`DFB 'bi_in_full' has data format 'Bfp8_b' which is not supported on architecture quasar`). The test file has no INT32 or UINT32 case.
4. **FLOAT32 untilize fails on Quasar (shared compute API, not this op).** All 23 FLOAT32 tests in the file abort the simulator with `UNPACKER_0 ILLEGAL_FORMAT_CONVERSION: Float32(0) -> Float32(0) to SrcA`. The sibling `untilize` op's `test_untilize_fp32` aborts with the same message. The cause: on Quasar, `pack_untilize_init_impl` calls `llk_unpack_A_init` with `unpack_to_dest` hard-coded to `false` ([pack_untilize.h:63-68](../../../../../../tt_metal/hw/inc/api/compute/pack_untilize.h#L63-L68)), so a Float32 input always unpacks to SrcA, which Quasar does not allow. The op does request `UnpackMode::UnpackToDest` for its FLOAT32 input DFB. This is tracked in #62275 (open), which gives the fix shape and also lists INT32, from the code only (the issue says INT32 was not run); the tilize side of the same gap is #57780 / #61930, with fix PR #61421 (open). #62275 reports a hang on the Zebu emulator; on craq-sim the same gap aborts with the illegal-conversion error above. This op requests unpack-to-DEST for INT32, UINT32 and FLOAT32 alike, so all three are affected. INT32 aborts the simulator the same way (`Int32(8) -> Int32(8) to SrcA`): all 10 INT32 cases of `test_untilize.py`, which use the same compute kernel, stopped on that message. #61930 notes that an fp32 or int32 tile unpacked into a source register is a hardware programming error on Quasar, and that ttsim refuses both the same way. This op's test file has no INT32 case. The failure does not come from this uplift: on the branch base `fe999bc568b`, built and run with none of the uplift's changes, an INT32 `test_untilize` case, `test_untilize_fp32`, and a FLOAT32 `test_untilize_with_unpadding_fp32` case abort with the same messages. Until #62275 is fixed, the op owner could reject those inputs on Quasar with a `TT_FATAL`, as `tilize` does for its own FLOAT32 limit (#58970). That is not done here, because it is an owner decision.

## RED-stop check

None applies to the op as a whole. The op is Metal 2.0. No required capability is missing (`LocalTensorAccessor` covers the self-loop). No construct needs an owner decision (the conversion follows an existing precedent). The self-loop conversion leaves WH/BH unchanged; the block-interleaved writer fix changes WH/BH only for inputs that hang there today, and is included at the reviewers' request. One condition applies to FLOAT32 input only: the Quasar compute API has no fp32 unpack-to-DEST path for untilize (deferred item 4). That gap is shared with every untilize user, so it is reported for the compute API / LLK owners rather than stopping this uplift.

## Test results

All runs were built from the repository root with `./build_metal.sh -e --enable-fake-kernels-target`. Rows marked "before the writer fix" ran on this change without the block-interleaved writer fix; the fix does not touch the code those runs measure (noted per row).

| Target | Tests | Result |
|---|---|---|
| Wormhole n150 | `test_untilize_with_unpadding.py` (442 tests) | 428 passed, 8 skipped, 6 xfailed, 0 failed. The skips and xfails are marked in the test file: 6 wait on ND-sharded reshape support, 2 need 2 devices, 6 are expected out-of-memory cases. Before the writer fix the same file gave 423 passed with the same skips and xfails; the 5 extra passes are the new block-crop test. |
| Wormhole n150 | `test_untilize.py` (844 tests; shares the fixed writer) | 838 passed, 4 skipped, 2 xfailed, 0 failed. The skips and xfails are marked in the test file. |
| Wormhole n150 | `test_sharded.py -k untilize_with_unpadding` | 12 passed. |
| Quasar simulator, LLK asserts on and off | `test_untilize_with_unpadding_block_crops_whole_tile_rows` | 5 passed in each run. |
| Quasar simulator, LLK asserts on (before the writer fix; these tests use the sharded factory, not the fixed writer) | the new width-16 test, `test_untilize_with_unpadding_height_sharded`, `test_untilize_with_unpadding_width_sharded` (all cases that reach the changed sharded writers) | 18 passed. |
| Quasar simulator, LLK asserts off | the whole file except FLOAT32, BFLOAT8_B and the 2 two-device cases (411 tests) | 398 passed, 7 skipped, 6 failed. All 6 failures need more cores than the simulator's 8x4 grid: 4 cases of `..._narrow_width_regression` ask for 64 cores (`target_num_cores <= total_available_cores`), and 2 cases of `..._round_robin_input_to_grid_2d_output` allocate the input on a 6x6 grid (`allocator.has_bank`), before the op runs. The skips: 6 marked in the test file, 1 trace capture not supported in slow dispatch. |
| Quasar simulator (before the writer fix; both failure points are outside the writer) | the 29 FLOAT32 and BFLOAT8_B tests | 23 FLOAT32 tests abort the simulator (deferred item 4). 6 BFLOAT8_B tests are rejected at program creation (deferred item 3). |
| Quasar simulator, LLK asserts off | `test_untilize.py` except FLOAT32, BFLOAT8_B and `fp32` cases (821 tests) | 728 passed, 79 failed, 4 skipped, 10 aborted; none involves the fixed writer, which was compiled in this run. 72 failures place a sharded tensor on cores or DRAM banks the simulator does not have (`Invalid shard grid` / `allocator.has_bank`), before the op runs. 7 are tests that force the codegen path, which Quasar refuses by design (#60367). The 10 aborts are INT32 cases that hit `ILLEGAL_FORMAT_CONVERSION: Int32(8) -> Int32(8) to SrcA`, the INT32 side of #62275 (deferred item 4). |

The runs confirmed from kernel cache file times that each edited writer was compiled after its last edit.

## Tests to run

The factory changed, so rebuild first. Run all commands from the repository root.

```bash
./build_metal.sh -e --enable-fake-kernels-target
```

**WH/BH parity.** Expect identical results before and after the change, except for the two new tests: the old code has no test that reaches the width-16 writer, and it hangs on two cases of `test_untilize_with_unpadding_block_crops_whole_tile_rows` (K = 2), so leave that test out of a run on the old code. `test_untilize.py` is included because the `untilize` block factory shares the fixed writer.

```bash
source python_env/bin/activate && pytest tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py
source python_env/bin/activate && pytest tests/ttnn/unit_tests/operations/data_movement/test_untilize.py
source python_env/bin/activate && pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py -k untilize_with_unpadding
```

**Quasar (craq-sim)**. Start with the tests that reach the changed writers, then run the whole file. Run once with `TT_METAL_LLK_ASSERTS=1` and once without. Tests that ask for more than 32 cores (for example `test_untilize_with_unpadding_height_sharded_narrow_width_regression`, which uses 64) do not fit the simulator's 8x4 grid. The whole-file command leaves out FLOAT32, because a FLOAT32 case aborts the simulator process (deferred item 4) and ends the run.

```bash
export METAL=$(pwd) QSR_SIM=/localdev/fplavec/temp/qsr-sim
source "$METAL/python_env/bin/activate"
export TT_METAL_HOME="$METAL" TT_METAL_SIMULATOR="$QSR_SIM/libttsim.so" TT_METAL_SIMULATOR_HOME="$QSR_SIM" \
       ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
       LD_LIBRARY_PATH="$METAL/build/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$METAL/ttnn:$METAL"
T=tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py
timeout --signal=INT --kill-after=20s 60m pytest \
  "$T::test_untilize_with_unpadding_height_sharded_to_height_sharded_width_16" \
  "$T::test_untilize_with_unpadding_height_sharded" \
  "$T::test_untilize_with_unpadding_width_sharded" \
  "$T::test_untilize_with_unpadding_block_crops_whole_tile_rows"
timeout --signal=INT --kill-after=20s 240m pytest "$T" -k "not FLOAT32 and not float32"
timeout --signal=INT --kill-after=20s 480m pytest tests/ttnn/unit_tests/operations/data_movement/test_untilize.py \
  -k "not FLOAT32 and not float32 and not fp32"
```
