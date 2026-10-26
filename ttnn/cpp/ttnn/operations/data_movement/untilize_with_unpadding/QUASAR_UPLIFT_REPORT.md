# Quasar uplift report: `untilize_with_unpadding`

**Status: GREEN, with one limit outside this op.** The op passes its test suite on Wormhole and, for BFLOAT16, on the Quasar simulator (craq-sim); see [Test results](#test-results). FLOAT32 input does not work on Quasar because of a gap in the shared untilize compute API (deferred item 5); the sibling `untilize` op has the same failure. Blackhole was not tested (no Blackhole board on the test machine).

Procedure: [`quasar_porting.md`](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md), plus the canonical recipe on branch `akertesz/op-porting-recipe` (`ai/audit/quasar_audit.md`, `ai/audit/cb_dfb_quasar_audit_helper.md`, `ai/post_port/semantic/dm_self_loop_dfbs.md`, `ai/post_port/style/sync_free_dfbs.md`, `ai/post_port/semantic/gen2_hardware_configs.md`).

Delete this file before merge.

## Summary

- The op was already Metal 2.0 on WH/BH. All five factories build `ProgramArtifacts` through `create_program_artifacts`, and every kernel uses the device 2.0 APIs (`DataflowBuffer`, `Noc`, `TensorAccessor`, `get_arg`).
- The audit found one Quasar blocker. In the sharded factory, same-shard-type sharded output (HEIGHT to HEIGHT, WIDTH to WIDTH, BLOCK to BLOCK) used a DFB borrowed from the output tensor, and the writer bound it as both producer and consumer. That is a data-movement self-loop. The Gen2 host validator rejects it with a `TT_FATAL` ([dfb.cpp:589-602](../../../../../../tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp#L589-L602)), so this path could not run on Quasar at all.
- The fix replaces that DFB with a `LocalTensorAccessor` over the output tensor. This is the same conversion `tilize_with_val_padding` received for its borrowed input shard in #57616 ([reader_unary_pad_height_width_sharded.cpp:24-43](../tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_height_width_sharded.cpp#L24-L43)).
- Nothing else needed a change. The op stays in its directory and namespace. Nothing was copied into or taken from `experimental/quasar/`.

## Files changed

| File | Reason |
|---|---|
| [untilize_with_unpadding_multi_core_sharded_program_factory.cpp:220-252](device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp#L220-L252) | Remove the `SH_SHARDED_OUT` DFB and its two self-loop bindings. Bind the output tensor (`SH_OUTPUT`, accessor `out`) to the writer as a `TensorBinding`. Remove the `num_unpadded_output_rows` runtime arg, which only fed the removed `reserve_back` ([lines 389-404](device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp#L389-L404)). |
| [writer_unary_unpad_batch_rows_sharded.cpp:24-48](device/kernels/dataflow/writer_unary_unpad_batch_rows_sharded.cpp#L24-L48) | Write each row into the output shard through `LocalTensorAccessor<uint32_t>(tensor::out)` at a byte offset, instead of through the borrowed DFB's write pointer. Remove the `reserve_back`. |
| [writer_unary_unpad_width_16_sharded.cpp:19-44](device/kernels/dataflow/writer_unary_unpad_width_16_sharded.cpp#L19-L44) | Same conversion. The tile size now comes from `dfb::untilize_out`, the buffer whose faces the kernel copies, because `dfb::out` no longer exists. |
| [untilize_with_unpadding_device_operation.cpp:167-175](device/untilize_with_unpadding_device_operation.cpp#L167-L175) (and lines [247-250](device/untilize_with_unpadding_device_operation.cpp#L247-L250), [302-305](device/untilize_with_unpadding_device_operation.cpp#L302-L305)) | Comments only. They described the old borrowed circular buffer (`sharded_output_cb_index`). The rule they justify (same-shard-type output must be `BufferType::L1`) still holds, because `LocalTensorAccessor` rejects a DRAM tensor at compile time. |
| [test_untilize_with_unpadding.py:215-245](../../../../../../tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py#L215-L245) | New test for HEIGHT_SHARDED to HEIGHT_SHARDED output of width 16. Before this test, no test reached `writer_unary_unpad_width_16_sharded.cpp`. The two heights run the 8-tile batch loop and the tail loop of that writer. |

## WH/BH parity claim

The change has no `ARCH_QUASAR` guard because it is behaviorally identical on WH/BH:

- **Before:** the writer called `reserve_back(n)` once on an empty DFB of exactly `n` entries, so it returned at once and moved no pointer. It then took `get_write_ptr()`, which is the base of this core's output shard, and wrote row `k` at `base + k * aligned_page_size` with a NoC read from its own SRAM.
- **After:** `LocalTensorAccessor(tensor::out)` gives the same base address (the tensor binding's address of this core's shard). The NoC reads have the same sources, sizes, order and destination addresses. The barriers and the `untilize_out` DFB calls are unchanged.
- **Width-16 tile size:** `get_tile_size(dfb::untilize_out)` equals the old `get_tile_size(dfb::out)`. Both DFBs had the same data format and the default 32x32 tile, and the JIT fills the tile size from those two values.
- **`.addr = 0` in `set_async_read_state`:** WH, BH and Quasar program only the NoC coordinate (and on BH and Quasar the upper address bits) in read `set_state`. The low address bits are ignored, as the kernel's own comment says.
- **Program cache:** the output address now reaches the writer through the tensor binding, which the framework refreshes on every dispatch, the same way the borrowed DFB address was refreshed before.

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
| §7: `finish()` balance | Audited every DFB in every factory. Balanced on every path that the tests reach. One imbalance found on a path no test reaches; see deferred item 1. |
| §7: implicit sync | Not disabled anywhere. The DM configs come from [`create_reader/writer_datamovement_config()`](../../core/data_movement_kernel/datamovement_kernel_config.hpp#L24-L38) with the default `false`. All DM transfers use plain NoC calls with explicit `reserve_back`/`push_back`/`wait_front`/`pop_front`. |
| §7: `compute_kernel_hw_startup` once | Yes, at the start of each compute kernel. |
| §7: re-init on DFB change, `pack_reconfig` re-`pack_init` | Not applicable. Each compute kernel uses one input/output pair and `NoReconfigure`. |
| §7: bare wait/pop or reserve/push pairs in compute (TEN-4746) | None. The untilize helper has `pack_untilize_block` between each pair ([untilize_helpers.inl:228-271](../../../kernel_lib/untilize_helpers.inl#L228-L271)), and the width-16 copy kernel has `copy_tile` and `pack_tile` ([eltwise_copy_metal2.cpp:32-48](../../../kernel/compute/eltwise_copy_metal2.cpp#L32-L48)). |
| §7: uint16/uint32 formats | No format-specific code path in the op; see deferred item 4. |
| §7: `-inf`/NaN arithmetic, multicast (§11), reduce face geometry (§8.4) | Not used by this op. |
| §4: hardware configs (`gen2_hardware_configs.md`) | Zero sites. The compute config is the single `ComputeHardwareConfig` struct with only common fields (`enable_32_bit_dest`, `unpack_modes`). The DM configs come from the ttnn helpers, which set `config_2xx`. |
| §4: `unpack_modes` for Float32 | Already set to `UnpackToDest` on the input DFB whenever `fp32_dest_acc_en` is set. |
| §4: `opt_level` | Not changed (base-port concern). Compute kernels state `O3`; DM kernels use the default. |
| §8.3: uncached `get_read_ptr()` on Quasar DM | No change needed. The writers wrap `get_read_ptr()` in a `CoreLocalMem` NoC operand, and the NoC maps the uncached alias back to the cached view ([noc.h:131-141](../../../../../../tt_metal/hw/inc/api/dataflow/noc.h#L131-L141)). |
| §6: NoC loopback copies within one core | The same-shard writers still read their own SRAM through a NoC loopback, as before. §6 says the emulator can drop such reads. Not needed: all same-shard cases pass on craq-sim with exact output. If rows come back missing on the physical emulator, replace the loopback with a RISC copy. |
| §8.1: build errors | None hit. The host build passed, and every kernel the tests use compiled for Quasar. The only compiler message is the expected deprecation warning in deferred item 2. |

## Deferred and follow-up items

1. **`finish()` imbalance in the block-interleaved writer when the output drops a whole tile row.** In [writer_unary_stick_layout_wh_multicore_metal2.cpp:77-95](device/kernels/dataflow/writer_unary_stick_layout_wh_multicore_metal2.cpp#L77-L95), `total_num_rows` is the output height, while the work split walks the padded input height. If the output height is at least 32 rows smaller than the padded input height, a tile row can start exactly at `total_num_rows`. Then `this_block_num_rows` is 0, and the writer skips `wait_front`/`pop_front` for that row's sub-blocks, which the compute kernel still pushes.
   - Quasar: the entries are never popped, so the writer and the compute kernel hang in `finish()` at kernel end (waypoint `AAW`).
   - WH/BH: the output DFB holds exactly one sub-block (`num_entries = set.block_tiles`), so the compute kernel can be at most one sub-block ahead of the writer. The writer re-reads `start_row_id` in every `third_dim` pass, so on a core it skips K = (sub-blocks per tile row) x `third_dim` sub-blocks in total. If K is 1, the compute kernel finishes, the output is correct, and one sub-block stays in the buffer. If K is 2 or more, the compute kernel waits in `reserve_back` forever, so the op hangs on WH/BH too.
   - A tile row that starts *past* `total_num_rows` makes the unsigned subtraction wrap. `write_block` then waits and pops but writes no row, so only the exact-equality case leaves entries behind.
   - No current test reaches this. The block-factory tests keep the output height within one tile row of the padded input height.
   - Tracked as item 11 of #57354 (open, static audit, written against the legacy file `writer_unary_stick_layout_wh_multicore.cpp`), which proposes the same fix. That item says the zero-row case always hangs; the runs below show that WH/BH hang only when K is 2 or more. It does not mention Quasar. It also says that a tile row starting past the output end loops about 4 billion times. That does not happen: unsigned arithmetic wraps modulo 2^32, so the loop bound `start_row_id + (total_num_rows - start_row_id)` is exactly `total_num_rows`, which is below `start_row_id`, and the loop runs zero times while the sub-block is still popped.
   - **Confirmed by a run.** Input `[1, 1, 64, 2048]` (BFLOAT16, TILE), `output_tensor_end=[0, 0, 31, 2047]`, `use_multicore=True` selects the block-interleaved factory on both targets. Wormhole: correct output, no hang. Quasar simulator: hang until killed after 10 minutes. With a temporary edit that does `wait_front`/`pop_front` of `single_sub_block_size_row_arg` entries when `this_block_num_rows == 0`, the Quasar run finishes with correct output. On Wormhole, input `[2, 1, 64, 2048]` with `output_tensor_end=[1, 0, 31, 2047]` (`third_dim` = 2, so K = 2) hangs with the current writer until killed after 2 minutes. With the temporary edit it finishes with correct output, and the `[1, 1, 64, 2048]` case is still correct. The temporary edit was reverted; it is not part of this change.
   - Proposed fix: the form in #57354 item 11. Set `this_block_num_rows` to 0 whenever `start_row_id >= total_num_rows` (otherwise `min(tile_height, total_num_rows - start_row_id)`), and always `wait_front`/`pop_front` each sub-block, writing rows only when `this_block_num_rows > 0`. This covers the exact-end case (what the temporary edit fixed) and the past-end case through one explicit path, instead of relying on the unsigned wrap for the past-end case. A test with padded height 96 cropped to 32 rows covers both cases on one core. What the fix changes on WH/BH: inputs whose output never drops a whole tile row are unaffected (this includes every current test, and every `untilize` call). Tile rows that start past the output end behave as before: the sub-block is popped and no row is written. Inputs with K = 1 give the same output; the writer only pops the one sub-block that was left in the buffer. Inputs with K of 2 or more stop hanging and give correct output. So on WH/BH it only fixes hangs. Not applied here, because the guide asks that a Quasar uplift leave WH/BH behavior unchanged and that fixes be ARCH_QUASAR-guarded otherwise; a fix for a WH/BH hang also belongs in its own PR with its own test. It can go in this change instead, either unguarded or guarded by `ARCH_QUASAR`, if the reviewers prefer.
2. **Id-only untilize helper in two shared compute kernels.** [untilize_metal2.cpp:21-28](../untilize/device/kernels/compute/untilize_metal2.cpp#L21-L28) and [untilize_variable_num_blocks_metal2.cpp:29-36](../untilize/device/kernels/compute/untilize_variable_num_blocks_metal2.cpp#L29-L36) call the id-only `untilize<...>(num_blocks)`. On Quasar that form is `[[deprecated]]` ([id_only_helpers.hpp:7-21](../../../kernel_lib/id_only_helpers.hpp#L7-L21)), because a helper-owned `DataflowBuffer` drains when it goes out of scope (#60848). Kernels build with `-Wno-error=deprecated-declarations`, so this is a warning, not a build failure. In both kernels the helper call is the last statement, so the drain happens at kernel end; my guess is that no hang follows. The fix is to pass the kernel's own buffers by reference, as [untilize_wh_metal2.cpp:21-32](../untilize/device/kernels/compute/untilize_wh_metal2.cpp#L21-L32) does. #62092 lists this as a follow-up. These kernels are shared with `untilize` and other ops, so the change is not made here.
3. **`get_tile_size()` in a shared DM reader.** [reader_unary_interleaved_wh_multicore_metal2.cpp:36-37](../../eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore_metal2.cpp#L36-L37) (block-interleaved factory) reads its NoC transfer size with `dfb.get_tile_size()`. #58133 changed similar DM reads to `get_entry_size()` and gave Quasar staleness as the reason ([reader_unary_nd_sharded_blocks_metal2.cpp:31-34](../sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks_metal2.cpp#L31-L34)). Reading the code, the value looks correct on Quasar: the descriptor slot equals the logical DFB id, and the host fills the tile size for every kernel. If a wrong read size shows up, switch to `get_entry_size()`. That is byte-identical on WH/BH, because both binding factories size the entry as one tile.
4. **Formats.** The op passes through UINT32, INT32, FLOAT32 and BFLOAT8_B inputs. Quasar has no UInt32 or Bfp8_b device format. The op has no code path of its own for these formats (the `float32_dtype` compile-time arg only selects a 4-byte element size), so there is nothing to guard here. The limit belongs to the format/LLK layer. On the simulator, BFLOAT8_B input is rejected cleanly at program creation (`DFB 'bi_in_full' has data format 'Bfp8_b' which is not supported on architecture quasar`). The test file has no INT32 or UINT32 case.
5. **FLOAT32 untilize fails on Quasar (shared compute API, not this op).** All 23 FLOAT32 tests in the file abort the simulator with `UNPACKER_0 ILLEGAL_FORMAT_CONVERSION: Float32(0) -> Float32(0) to SrcA`. The sibling `untilize` op's `test_untilize_fp32` aborts with the same message. The cause: on Quasar, `pack_untilize_init_impl` calls `llk_unpack_A_init` with `unpack_to_dest` hard-coded to `false` ([pack_untilize.h:63-68](../../../../../../tt_metal/hw/inc/api/compute/pack_untilize.h#L63-L68)), so a Float32 input always unpacks to SrcA, which Quasar does not allow. The op does request `UnpackMode::UnpackToDest` for its FLOAT32 input DFB. This is tracked in #62275 (open), which also covers INT32 and gives the fix shape; the tilize side of the same gap is #57780 / #61930, with fix PR #61421 (open). #62275 reports a hang on the Zebu emulator; on craq-sim the same gap aborts with the illegal-conversion error above. This op requests unpack-to-DEST for INT32, UINT32 and FLOAT32 alike, so all three are affected. Until #62275 is fixed, the op owner could reject those inputs on Quasar with a `TT_FATAL`, as `tilize` does for its own FLOAT32 limit (#58970). That is not done here, because it is an owner decision.

## RED-stop check

None applies to the op as a whole. The op is Metal 2.0. No required capability is missing (`LocalTensorAccessor` covers the self-loop). No construct needs an owner decision (the conversion follows an existing precedent). No fix changes WH/BH. One condition applies to FLOAT32 input only: the Quasar compute API has no fp32 unpack-to-DEST path for untilize (deferred item 5). That gap is shared with every untilize user, so it is reported for the compute API / LLK owners rather than stopping this uplift.

## Test results

All runs used the code in this change, built from the repository root with `./build_metal.sh -e --enable-fake-kernels-target`.

| Target | Tests | Result |
|---|---|---|
| Wormhole n150 | `test_untilize_with_unpadding.py` (437 tests) | 423 passed, 8 skipped, 6 xfailed, 0 failed. The skips and xfails are marked in the test file: 6 wait on ND-sharded reshape support, 2 need 2 devices, 6 are expected out-of-memory cases. |
| Wormhole n150 | `test_sharded.py -k untilize_with_unpadding` | 12 passed. |
| Quasar simulator, LLK asserts on | the new width-16 test, `test_untilize_with_unpadding_height_sharded`, `test_untilize_with_unpadding_width_sharded` (all cases that reach the changed writers) | 18 passed. |
| Quasar simulator, LLK asserts off | the whole file except FLOAT32, BFLOAT8_B and the 2 two-device cases (406 tests) | 393 passed, 7 skipped, 6 failed. All 6 failures need more cores than the simulator's 8x4 grid: 4 cases of `..._narrow_width_regression` ask for 64 cores (`target_num_cores <= total_available_cores`), and 2 cases of `..._round_robin_input_to_grid_2d_output` allocate the input on a 6x6 grid (`allocator.has_bank`), before the op runs. The skips: 6 marked in the test file, 1 trace capture not supported in slow dispatch. |
| Quasar simulator | the 29 FLOAT32 and BFLOAT8_B tests | 23 FLOAT32 tests abort the simulator (deferred item 5). 6 BFLOAT8_B tests are rejected at program creation (deferred item 4). |

The run with LLK asserts on also confirmed that the edited writers were compiled for Quasar after their last edit (kernel cache file times).

## Tests to run

The factory changed, so rebuild first. Run all commands from the repository root.

```bash
./build_metal.sh -e --enable-fake-kernels-target
```

**WH/BH parity** (expect identical results before and after the change):

```bash
source python_env/bin/activate && pytest tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py
source python_env/bin/activate && pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py -k untilize_with_unpadding
```

**Quasar (craq-sim)**. Start with the tests that reach the changed writers, then run the whole file. Run once with `TT_METAL_LLK_ASSERTS=1` and once without. Tests that ask for more than 32 cores (for example `test_untilize_with_unpadding_height_sharded_narrow_width_regression`, which uses 64) do not fit the simulator's 8x4 grid. The whole-file command leaves out FLOAT32, because a FLOAT32 case aborts the simulator process (deferred item 5) and ends the run.

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
  "$T::test_untilize_with_unpadding_width_sharded"
timeout --signal=INT --kill-after=20s 240m pytest "$T" -k "not FLOAT32 and not float32"
```
