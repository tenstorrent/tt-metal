# Metal 2.0 Port Report — `data_movement/untilize` (`UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory`)

## Outcome

**`PORTED`** — `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory`, the last legacy factory of `UntilizeDeviceOperation`. All 8 factories in its `program_factory_t` are now on `ProgramSpecFactoryConcept`. `codegen/` (`UntilizeCodegenDeviceOperation`) is a separate device-op, still on `create_descriptor`, and was out of scope.

Host-side only: two files changed (the factory `.hpp` / `.cpp`); no kernel, device-op, nanobind or `tt_metal/` edit.

## Provenance

- **Recipe docs (this port):** `ed3eb002115 2026-11-03 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`
- **Audit docs (inherited):** `ed3eb002115 2026-11-03 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`
- **Target concept source:** taken from code evidence and the 7 sibling rows, not from the readiness sheet's `Porting Target` cell. The sheet has no row for this factory; the user waived that gate on 2026-11-03. `TensorParameter relaxation = none` is code-derived on the same waiver. The row request to the sheet owner still stands.

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept` — `create_descriptor` replaced by `create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`; header now matches the 7 siblings.
- **Custom hash:** none (default reflection hash), untouched.
- **Pybind entry points removed:** none (no pybound `create_descriptor`).
- **Device-op-class edits:** none. The Quasar exclusion at `device/untilize_device_operation.cpp:338` is untouched, so the factory stays Gen1-only.
- **Open items:** none for the concept fit.

## Handoff points

none — no capitulation, no boundary-rule violation, no kernel-lib gap, no removed pybind surface.

## Successes

- **The brief was close to a ready spec.** It named the template (the ported 2D sibling), the three differences from it, the fork vocabularies and the compute `KernelSpec` to copy from the ND-input sibling. Each of those matched the code. The port was a near-mechanical rewrite of `create_descriptor`, and it built and passed first time.
- **Borrowed-DFB sizing watch-for paid off as a test target.** The brief asked for an uneven, multi-shard-per-core ND case. The existing tests already cover it: the reachability marker recorded group-1/group-2 splits such as 3+2, 2+1, 11+10, 62+61 and 220+219 shards per core, plus a grid core with no shard (9 cores, 8 with data). `ValidateBorrowedMemory` accepted all of them with `num_entries = num_tiles_per_shard * num_shards_per_core_in_group_1`, which confirms group 1 holds the per-bank maximum.
- **Hardware config / `opt_level` checklist.** Legacy compute was Style B (`ComputeConfigDescriptor` set directly) with only `fp32_dest_acc_en` and `unpack_to_dest_mode` non-default. Porting just those two, plus an explicit `O3` (legacy compute default), reproduces it; the Metal 2.0 `ComputeHardwareConfig` defaults (HiFi4, Precise, `double_buffer_dest = true`, bfp Approximate) match the legacy defaults. Both DM configs were the reader/writer defaults → TTNN helpers.

## Friction

- **Gap — the "Force" grep misses the spec-side choke point.** [Ensure the Metal 2.0 host-side legality checks are enabled] says to find sites with `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp`. On this tree that finds only the 5 sites in `program_run_args.cpp`. `BuildProgramFromSpec` and its 3 wrappers live in `tt_metal/impl/metal2_host_api/program_spec/program_spec.cpp`, one directory down. That is the function the recipe itself calls "the spec-side choke point", and it holds the `ValidateProgramSpec` gate. A porter who follows the grep literally leaves spec validation bypassable, and only the "two markers" check would catch it. Fix: `grep -rn` over the directory (9 sites found). I forced all 9; both markers fired.
- **Gap — reachability of a narrowly-selected factory isn't covered.** The recipe says to confirm the test set but gives no way to prove the tests actually *select* the factory being ported. Here that matters: selection checks the legacy `shard_spec` before `nd_shard_spec`, so an ND-sharded test can silently land on the 2D sibling. The factory's own `log_debug` was useless for this because `log_debug` is compiled out of Release builds (`SPDLOG_ACTIVE_LEVEL`), and the pre-port baseline came back with zero markers. I used a temporary `log_warning` marker in the ported factory, never committed, the same way as `METAL2_CHECKS_FORCED`. Suggest the recipe add this as a standard step for factories reached only through a narrow selection predicate.
- **Confusion — line numbers in the audit / brief are off.** The audit says it was taken on a tree identical to `origin/main`, but its factory citations are about 19 lines early (e.g. `cb.buffer` at `.cpp:81`/`:95` vs actual `:100`/`:114`; RTA loop `.cpp:138-157` vs `:157-176`). The constructs were easy to find, but the drift is worth checking in the audit tooling.
- **Input mismatch at start.** On first read the op directory held a RED (sheet-only) audit and no brief, while the request said GREEN with a brief. After asking, a GREEN-on-waiver audit and the brief appeared. Not a recipe issue, but the precondition check did its job.

## Verification

WH n150, Release build, Watcher on (`TT_METAL_WATCHER=10`), `skip_validation` forced off at all 9 sites, both `METAL2_CHECKS_FORCED` markers present (scaffolding reverted before commit).

| test file | pre-port | post-port | factory constructions (program-cache misses) |
|---|---|---|---|
| `tests/ttnn/unit_tests/operations/data_movement/test_untilize.py` | 838 passed, 4 skipped, 2 xfailed | 838 passed, 4 skipped, 2 xfailed | 57 |
| `tests/ttnn/unit_tests/base_functionality/test_to_layout.py` | not run (see note) | 672 passed, 45 skipped, 8 xfailed | 26 |

Test set confirmed by the invoker. `test_to_layout.py` was added after the pre-port build was already replaced; it ran fully green post-port, so there is no regression to compare. The 2 xfails in `test_untilize.py` are a pre-existing `from_torch` allocator `TT_FATAL` in `test_untilize_multi_core_nd_sharded_to_interleaved`, unrelated to this factory.

The ND-identical cases in `test_untilize.py` are `test_untilize_nd_shard_spec_to_same_shard_spec3D/4D` and `test_untilize_nd_shard_to_same_shard_spec_uneven_input_shard_spec`. All are bfloat16, so the `fp32_dest_acc_en` → `unpack_modes` path was not exercised (see Open items).

Self-audit (op directory: 50 `.cpp`/`.hpp` files scanned):

- buffer address / `Buffer*` / `emplace_runtime_args` in the factory: 0 hits
- CB leftovers in the factory `.cpp` / `.hpp`: 0 (was 30). The remaining op-wide hits are all pre-existing, in out-of-scope files: `codegen/`, legacy kernel originals still bound by other ops, `untilize_device_operation.cpp`, `untilize.cpp`.
- `TT_FATAL` census: per-file counts unchanged. `TT_FATAL(dst_buffer != nullptr, …)` is kept as `TT_FATAL(output.buffer() != nullptr, …)`, same condition and message.
- `opt_level`: the one compute `KernelSpec` → `O3`; DM kernels unset (O2 = legacy).
- positional CTAs: none; varargs: none; multi-binding: none; conditional bindings: none.
- `.md` citations from code: 0 (4 changed code files scanned).
- `tt_metal/` files and scaffolding strings in the diff: none.

## Open items for downstream

- **Shared kernel touches (all rung 1, reused existing `_metal2` forks; no fork created or edited):**
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp`. Legacy original still bound by `experimental/slice_write` (tiled + RM sharded-input factories) and `data_movement/sharded_partial/sharded_to_interleaved_partial`.
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp` (#52228). Legacy original still bound by `data_movement/sharded_partial/interleaved_to_sharded_partial`.
  - `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks_metal2.cpp`. The legacy original `untilize_variable_num_blocks.cpp` now has no consumer in this op directory; still bound by `experimental/deepseek_prefill/update_padded_kv_cache`.
- **Test coverage:** no test drives this factory with `fp32_dest_acc_en = true` (float32 ND-sharded, identical spec). The `unpack_modes` entry is a straight carry-over of the legacy `UnpackToDestFp32` on `c_0`, but it is unexercised. Worth one fp32 parametrization of `test_untilize_nd_shard_spec_to_same_shard_spec3D`.
- **Harmless leftovers kept per scope discipline** (factory `.cpp`): `total_shards` / `num_cores` exist only for a `log_debug`; `compute_defines` is always empty; `orientation` only orders the per-core walk.
- **Readiness sheet:** still needs a `data_movement/untilize` row for this factory (now `MetalV2`). The audit suspects a name collision with the `experimental/quasar/untilize` copy's row.
- **Next in this op:** `codegen/` (`UntilizeCodegenDeviceOperation`) is the only remaining `create_descriptor` user under `untilize/`; it needs its own audit.
