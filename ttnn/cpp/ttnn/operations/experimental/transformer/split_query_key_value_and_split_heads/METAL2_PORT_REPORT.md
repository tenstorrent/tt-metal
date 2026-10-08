# Port Report — split_query_key_value_and_split_heads

## Outcome

**PORTED.** Both factories of `SplitFusedQKVAndSplitHeadsDeviceOperation` are on `ProgramSpecFactoryConcept`: `SplitFusedQKVAndSplitHeadsProgramFactory` (interleaved) and `SplitFusedQKVAndSplitHeadsShardedProgramFactory` (block-sharded). No factory is left on the legacy API.

**Verification (Wormhole b0, 8×8, Release build, Metal 2.0 validation forced).** The `METAL2_CHECKS_FORCED` markers from `program_spec.cpp` and `program_run_args.cpp` appear in every post-port log. The pre-port baseline ran on the same build before the cherry-pick, with identical commands.

| Test | Pre-port | Post-port |
|---|---|---|
| `tests/tt_eager/python_api_testing/unit_testing/misc/test_split_query_key_value_and_split_heads.py` (interleaved, `num_heads` 12 / 16, batch 8) | 2 passed | 2 passed |
| Scratch: interleaved, 3 dispatches with fresh, re-placed tensors (cache hits) | pass, PCC 1.0 ×9, 1 cache entry | pass, PCC 1.0 ×9, 1 cache entry |
| Scratch: sharded at grid (8, 8) (the BERT sharded test with only the grid changed), BFLOAT8_B / BFLOAT16 | 2 passed | 2 passed, identical PCCs |
| Scratch: sharded, 3 dispatches, different seeds and re-placed tensors (cache hits) | pass, 4 cache entries (op + reshard helpers) | pass, identical PCCs, 4 cache entries |
| `models/experimental/bert_large_performant/unit_tests/test_bert_large_split_query_key_value_and_split_heads.py` (needs a 12×9 grid) | 25 skipped | 25 skipped |
| `models/.../test_bert_large_split_query_key_value_and_split_heads_sharded.py` (hard-coded 12×8 grid) | error in `interleaved_to_sharded` before the op runs (`buffer.cpp:272`, core (8,0) has no L1 bank) | same error |

All per-tensor PCCs (Q / K / V, every seed and iteration) are bit-identical before and after the port.

**Watcher.** The interleaved runs used `TT_METAL_WATCHER=10` with `TT_METAL_WATCHER_DISABLE_ASSERT=1`; the sharded runs used full Watcher. The legacy interleaved writer trips a Watcher assert before the port (see Friction / Gaps). Post-port it trips the same assert at the same line, which confirms the behavior was carried over unchanged. `test_bert_large_split_and_transform_qkv_heads.py` (12-column grid, skips here) and the sweep files were not run.

**Not verified here:** the BERT-large configurations at their native 12×9 / 12×8 grids. They need a larger card (Blackhole, or a full-grid Wormhole).

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` (Port_Recipe checkout, `akertesz/op-porting-recipe`)
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept` for both factories, as the audit chose. Neither factory had an `override_runtime_arguments`. Each factory was a method swap inside its existing struct (`create_descriptor` → `create_program_artifacts`, same parameters) in `device/split_query_key_value_and_split_heads_device_operation.hpp:22-36`. The op already had a `program_factory_t` variant, so exception 3 did not apply.

### Device-op-class edits
- Pybind entry points removed: none. `split_query_key_value_and_split_heads_nanobind.cpp` binds only the op function.
- Custom `compute_program_hash`: none (default reflection hash), so nothing to keep intact.
- The device-op header changed only in the two factory-struct declarations, their two doc comments (which described CB bindings), and the include swap (`program_descriptors.hpp` → `ttnn/metal_v2_artifacts.hpp`). `device_operation.cpp` is byte-identical.

### Open items
- Relaxation candidates: none noticed. The audit cell is `none`.
- **Readiness-sheet waiver (recorded per the brief).** The live sheet (fetched 2026-10-07) still lists both factories as `Concept` = `legacy device-op`, with `Factory definition path` cells pointing at per-factory headers that PD batch #57409 (`f5093e705ae`) deleted. The user waived the gate on 2026-10-07 ("Waive"). The sheet refresh is still owed to the readiness-sheet owner.

## Handoff points

- **Readiness-sheet owner:** update both `split_query_key_value_and_split_heads` rows: `Concept` should be `descriptor` (now, after this port, Metal 2.0), and the `Factory definition path` should be `device/split_query_key_value_and_split_heads_device_operation.hpp`.
- **Op owners (test coverage):** `models/experimental/bert_large_performant/unit_tests/test_bert_large_split_query_key_value_and_split_heads_sharded.py:40` hard-codes `grid_size = (12, 8)` with no skip guard. On an 8×8 card it errors inside `interleaved_to_sharded` instead of skipping, so this op's only committed sharded-path test fails on every 8×8 Wormhole and gives no signal there. Either add a grid skip like the interleaved BERT test (`:15-17`), or derive the grid from the device. At grid (8, 8) with 16 heads the op's math still holds (2 heads per core); that is the configuration this port was verified with.

No kernel-lib gaps, boundary-rule violations, framework gaps, or removed pybind surface.

## Successes

- **[Two-toucher DFB → assign 1P+1C](../shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split).** The sharded input shard (legacy c_0) has two role-free touchers (reader and writer, both `get_read_ptr` only). The census procedure gave reader PRODUCER + writer CONSUMER directly, matching the brief, with no multi-binding flag (`sharded_program_factory.cpp`, reader and writer `dfb_bindings`).
- **[Sync-free and single-ended CBs → self-loop](../shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb).** The Q / V output shards (DM reader) and the K output shard (compute packer) self-loop cleanly. The validator accepted all three under forced checks.
- **[Porting a shared kernel](../shared/port_patterns.md#caution-porting-a-shared-kernel), rung 1.** Both compute forks already existed, and their binding vocabulary (`dfb::in` / `dfb::out` + `NHtWt` / `num_tiles`) fit without change. No new fork, no pointer-comment edits.
- **Whitelist rule 7.** All four `get_tile_size(cb_id)` sites were `const`, not `constexpr`, so they moved to `dfb.get_tile_size()` with no constexpr carve-out needed.
- **Compiler options.** The `grep -n opt_level` self-audit check paired each of the two compute KernelSpecs with an explicit `O3`.

## Friction

### Gaps
- **Recipe §Hardware configuration is stale against this tree.** It shows `create_reader_datamovement_config(device->arch())`, `ComputeGen1Config`, and `std::get<ComputeGen1Config>(compute_hw).<field>`. Here the TTNN DM helpers take no arch (`ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp:24,32`; the only parameter is `bool disable_dfb_implicit_sync_for_all`). `ComputeHardwareConfig` is a single struct (`tt_metal/api/tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp:45`) with top-level `fpu_math_fidelity` / `sfpu_precision_mode` / `enable_32_bit_dest` / `double_buffer_dest` / `unpack_modes` and a `config_1xx.bfp_pack_precision_mode`. I resolved the shape from the header (as the recipe says to) and from the sibling ports. The defaults match legacy `ComputeConfigDescriptor{}` field by field, so `ComputeHardwareConfig{}` is exact for Style B.
- **"Run every test with Watcher on" does not fit an op whose legacy kernel already trips Watcher.** `writer_tm_tile_layout_create_qkv_heads.cpp` pops one page more than it waited for (see Open items). Under Watcher, `cb_pop_front`'s `fifo_rd_ptr <= fifo_limit` assert (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:270`) fires on the 32-tile K buffer before the port. The recipe has no guidance for a pre-existing Watcher trip. I used `TT_METAL_WATCHER_DISABLE_ASSERT=1` for the interleaved path only (NoC sanitization stays on), ran the sharded path under full Watcher, and checked post-port that the same assert still fires. Suggest a recipe line: take a Watcher baseline pre-port; if the legacy op trips, keep Watcher on with asserts disabled and record it.
- **Unity-build hygiene.** `port_patterns.md` §Unity-build hygiene recommends per-factory prefixes or a shared header for anonymous-namespace constants. This `ttnncpp` target already holds about 15 ported factories in `ttnn::experimental::prim`, each declaring `READER` / `WRITER` / `INPUT`. The sibling ports avoid collisions by declaring spec names as function-local `const`s, and I did the same. That is simpler than either option the pattern lists, and worth adding to it.

### Confusion
- **Runtime DFB selection.** The brief suggests `DataflowBuffer cb_cur = (out_tensor == 1) ? dfb_k : dfb_qv;`. That copies a `DataflowBuffer`, giving two objects per DFB, which `port_patterns.md` §Same-FIFO aliasing tells porters to avoid. The legacy kernel rebuilt `CircularBuffer cb_cur(cb_id)` from a runtime id each iteration (`reader_tm_tile_layout_create_qkv_heads.cpp:33-46` pre-port). I kept one object per DFB and selected through a `DataflowBuffer*` in the unchanged `if` / `else if`. Recommend the whitelist or patterns doc name this idiom (runtime selection between two bound DFBs) explicitly.
- **Name collision with legacy locals.** The sharded factory has a legacy `uint32_t K` (the K dimension). A `TensorParamName K{"k"}` would shadow it, and renaming the legacy variable is out of scope. I named the tensor parameters `OUTPUT_Q/K/V` (in both factories, for symmetry).
- **Dead CB-index declarations.** The sharded reader declared `cb_im0` / `cb_out1` and the writer `cb_out0..2`, none of them used (audit Misc anomalies). The CB→DFB transition has to drop them, since there is no token for an unbound DFB. I treated this as part of "magic CB indices → DFB binding". The recipe covers dead *CBs* but not dead *index declarations* for CBs that are live in another kernel.

## Open items for downstream

- **Shared kernel touches.**
  - `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`: **reused the existing fork** `ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp`, no new file. Remaining legacy consumer: `experimental/transformer/nlp_create_qkv_heads_boltz`. The brief also listed `nlp_create_qkv_heads_vit`, but that op was ported on this branch (2026-10-07) and already binds the fork.
  - `kernels/compute/transpose_wh_sharded.cpp` (this op's own, lent): **reused the existing fork** `kernels/compute/transpose_wh_sharded_metal2.cpp`, no new file. This op and `create_qkv_heads_from_separate_tensors` now bind the fork. Remaining legacy consumer: `experimental/transformer/create_qkv_heads` (`create_qkv_heads_program_factory.cpp:167-169`). Once it migrates, the legacy `transpose_wh_sharded.cpp` can be retired.
  - The four dataflow kernels were converted in place. Each is bound only by one factory of this op.
- **Legacy bugs carried over unchanged (owner review):**
  - `writer_tm_tile_layout_create_qkv_heads.cpp`: `out_num_tiles_read_out0/1` start at `block_size` and increment after each wait. The final `pop_front` pops 33 pages from the 32-entry `k_transposed` buffer and 65 from `qv`. This moves `fifo_rd_ptr` past `fifo_limit`, which is exactly what the Watcher assert catches. The fix would be to start both counters at 0 for the `block_size_is_one` path, or to pop `num_tiles` pushed. It is benign at kernel exit on Gen1 today, but the op cannot be tested under full Watcher until it is fixed.
  - `reader_tm_tile_layout_create_qkv_heads.cpp`: the `// V heads` comment on the `out_tensor == 1` branch labels the K-heads branch. Preserved as-is.
  - Hard-coded `out_w = 64` (`program_factory.cpp`, interleaved) and `num_heads_per_tensor = 2` (`sharded_program_factory.cpp`) ignore `num_heads` (audit Misc anomalies).
  - Interleaved grid-fit checks are `TT_ASSERT` (compiled out in Release); the sharded path uses `TT_FATAL` (audit Misc anomalies).
- **Self-audit results** (op directory, 15 `.cpp` / `.hpp` files):
  - CB-name sweep: 0 hits outside the legacy `transpose_wh_sharded.cpp`, which stays on the legacy API for `create_qkv_heads`.
  - `TensorAccessorArgs` / `get_arg_val` / `get_compile_time_arg_val` in the bound kernels: 0.
  - Buffer address in run-args: 0. The `Buffer*` locals remain only as `TT_ASSERT` operands.
  - `.md` citations in code: 0.
  - `TT_FATAL` / `TT_ASSERT` / `TT_THROW` census: unchanged per file (device_operation.cpp 9, interleaved factory 7, sharded factory 4).
  - `opt_level`: `O3` on both compute KernelSpecs.
  - hw_config: reader / writer defaults on all DM kernels; `ComputeHardwareConfig{}` on both compute kernels, matching the legacy all-default `ComputeConfigDescriptor{}`. No `unpack_modes`: the op accepts BF16 / BFP8_B only.
- **Test coverage note.** The committed suite has no sharded-path test that runs on an 8×8 card (see Handoff points). The (8, 8) scratch variant used here is a near-copy of the BERT sharded test and would be a cheap addition.
