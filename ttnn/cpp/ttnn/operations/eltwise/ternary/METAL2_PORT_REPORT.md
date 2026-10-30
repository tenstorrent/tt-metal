# Port Report — `eltwise/ternary`

## Outcome

**PORTED** — `TernaryDeviceOperation::TernaryProgramFactory`, the op's only factory, now satisfies
`CustomProgramSpecFactoryConcept`. All 22 kernels it can select and the op's kernel-side header converted together.
Test results are under *Verification*.

## Provenance

- **Recipe docs (this port):** `b90203d162f 2026-10-30 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `b90203d162f 2026-10-30 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- The branch's `metal_2.0/` tree equals upstream `akertesz/op-porting-recipe` `9425f50ae93`, two commits behind
  `c41f7f4ef0a` (2026-10-29). The port followed the in-place recipe, as asked. The one upstream change that touches
  porting (`DFBAccessor` → `DFBBindingToken`) matches what this tree's headers already have.
- Base code: `origin/main` `3f57b50ff00` (contains #62084, the audit's pre-port fixes).

## TTNN ProgramFactory

### Concept realized

`CustomProgramSpecFactoryConcept`, as the audit chose.

- `create_program_artifacts` (`device/ternary_program_factory.cpp`) builds the `ProgramSpec` and returns the full
  `ProgramRunArgs`.
- `override_runtime_arguments` now returns `ProgramRunArgs` (no `Program&` parameter). It returns the same object the
  miss path returns: a `TensorArgument` for **every** `TensorParameter` (`predicate`; `value_true` for TTT / TTS;
  `value_false` for TTT / TST; `output`) and every named runtime arg of the three kernels on **every** core of the
  worker grid. No `TensorParameter` is skipped. That mirrors the legacy override, which rewrote all 27 / 11 / 4 slots of
  every core (addresses included) and re-pointed every tensor-backed CB (`ternary_program_factory.cpp:1543-1649` at
  `b90203d162f`).
- Both paths build their run args from the legacy `build_per_core_runtime_args`, kept as the single source of truth.
  Its rows keep the legacy slot layout; name tables map slots to named args, and the slots that held addresses (reader
  0-2, writer 0) or padding (writer 10) have no name.

### Device-op-class edits

- Pybind entry points removed: none (`ternary_nanobind.cpp` binds no factory entry point).
- Custom `compute_program_hash`: left intact (`device/ternary_device_operation.cpp:582-637`, `to_hash()` at `:261-271`).
  `ternary_device_operation.cpp` is byte-identical to the base.
- `device/ternary_device_operation.hpp`: only the factory struct changed (the two method shapes, and the includes they
  need: `ttnn/metal_v2_artifacts.hpp`, `metal2_host_api/program_run_args.hpp` in, the descriptor headers out).

### Open items

- **Relaxation, applied from the analysis doc** (not self-decided): `{.dynamic_tensor_shape = true,
  .relax_logical_rank = true}` on every `TensorParameter`, unconditionally, transcribed from
  `analyses/relaxations/eltwise_ternary.md` §2 (blob `c89ba329e90`), which the audit validated on this base.
- **Host cost on cache hits.** The legacy hit path copied fixed-width rows into the cached program. The Metal 2.0 hit
  path builds `Table`s of named args (linear-search inserts: per name, O(cores²) on a full grid), then
  `UpdateProgramRunArgs` applies them. Not measured. Worth a look on large grids if dispatch-bound models show it.

## Handoff points

none. Nothing reached outside the op directory, no API assumption was violated, and no kernel-lib helper needed a change.

## Successes

- **Shared-kernel census** (*Legacy inventory → Shared kernels*). The `grep -rl` sweep with
  `ternary_reader_nobcast_ttt.cpp` (lent to binary_backward) as the positive control confirmed that every bound kernel
  is private. The port converted in place and left the lent reader untouched.
- **Conditional / optional resource bindings** (`port_patterns.md`). The c_4 / c_5 / c_6 staging buffers are now created
  and bound only when the LLK row broadcast runs and that operand is broadcast. The kernels gate on the existing
  `BCAST_A/B/C` defines. `ternary_addc_ops_fpu_rowbcast.cpp` needed the gate added around its three `DataflowBuffer`s.
  The writer's `out` binding follows the existing `DST_SHARDED` define, and its `DataflowBuffer` moved inside
  `#if !DST_SHARDED`.
- **Self-loop for a single-ended DFB** (`port_patterns.md`, *Sync-free and single-ended CBs*). With a natively sharded
  output the packer writes into the output shard and nothing drains it. `out` is borrowed from `output` and self-looped
  on compute (`create_program_artifacts`, the compute bindings block).
- **`unpack_modes` on a producer-only DFB.** The legacy vector set `UnpackToDestFp32` on c_3 for a Float32 output. The
  validator's table (`hardware_config.cpp:54-83`) says such an entry is inert and tolerated on a producer-only binding,
  and governs it as a consumer when self-looped. The port carries it as `UnpackToDest` on `out` in both cases. The
  required-entry rule for a consumed Float32 DFB is then met in the self-loop case too.
- **`hw_config` / `opt_level` checks.** The op is Style B (it never reads `compute_kernel_config`; the device op passes
  `std::nullopt`). The Metal 2.0 compute config sets only `enable_32_bit_dest` and `unpack_modes`. The lowering
  (`kernel_lowering.cpp` `MakeGen1ComputeConfig`) maps every other default to the legacy `ComputeConfigDescriptor`
  default. The compute `KernelSpec` sets `O3` explicitly, the legacy compute default.

## Friction

### Gaps

- **Recipe API names trail this tree.** The recipe and catalog name `ComputeGen1Config`, `DataMovementGen1Config`,
  `std::get<ComputeGen1Config>(…)`, `create_reader_datamovement_config(device->arch())` and
  `to_compute_hardware_config(device->arch(), config)`. This tree has `ComputeHardwareConfig` with an optional
  `config_1xx`, `DataMovementHardwareConfig::config_1xx`, `create_reader_datamovement_config(bool)` and
  `to_compute_hardware_config(config)`. The values map one to one; only the spellings differ. The headers resolved it.
- **Rule 7 covers `constexpr` free functions, not `constexpr` array lookups.** `ternary_sfpu_col_scalar_bcast_ttt.cpp`
  declares `constexpr bool init_once = unpack_src_format[c_0] == …`, and `ternary_sfpu_copy_init.hpp` compares the
  `unpack_*` arrays of two buffers inside a `constexpr` function. A member getter cannot be `constexpr`, and the free
  `get_dataformat(token)` reads host metadata rather than the JIT array. The port indexes the same arrays with the token
  (`unpack_src_format[dfb::in_a]`), using the token's `constexpr` conversion. **Token-form sites** (Gen1-only, Quasar
  uplift debt): `ternary_sfpu_col_scalar_bcast_ttt.cpp` (`init_once`), `ternary_addc_ops_sfpu_bcast.cpp` (`init_once`),
  and `ternary_sfpu_copy_init.hpp` (`copy_init_shared_with<ref, other>`, whose template arguments are tokens).
- **`ternary_sfpu_copy_init.hpp` had no precedent.** `copy_init_shared_with_c0<cb>()` hardcoded `tt::CBIndex::c_0`. It
  became `copy_init_shared_with<ref, other>()`: the reference buffer is the new first template parameter, and the name
  lost the `_c0` that no longer means anything. Its `copy_init_unless(uint32_t cb, …)` parameter became `dfb_id`.
- **The override-guard rule and the declarative override.** The recipe says an override's guards always come across.
  The legacy override's two `TT_FATAL`s checked the cached `Program` itself: the runtime-arg vector width of each core,
  and that each globally allocated CB carries one tensor index. The Metal 2.0 override never sees the `Program`. See the
  `TT_FATAL` census under *Verification* for how each was placed.
- **The `cb` self-audit grep has a false positive here.** `[Cc][Bb]_` matches `srcB_num_tiles`
  (`tst_tts_reader_outer_bcast.cpp`, `tst_tts_reader_scalar_bcast.cpp`): "src" + "B" + "_". It is a legacy local, not
  a CB, and the rename rule allows only CB→DFB renames. It stays.
- **Validation forcing in `tt_metal/impl` not applied.** In an earlier port the auto-mode permission classifier denied
  the recipe's `skip_validation = false` edit in `tt_metal/impl/metal2_host_api/*.cpp`. This port used the TTNN route
  instead (see *Verification*). There are no `METAL2_CHECKS_FORCED` markers to grep.

### Confusion

- **`get_tile_size()` vs `get_entry_size()`.** The brief suggested `get_entry_size()` for the 8 fill-helper readers'
  `cb.get_tile_size()` and asked for a whitelist check. Whitelist §A maps `get_tile_size(id)` (what
  `CircularBuffer::get_tile_size()` calls, `circular_buffer.h:113`) to the member `get_tile_size()`. That member reads
  the same `unpack_tile_size` array by the DFB's id (`dataflow_buffer.h:274-280`). The JIT fills it at each DFB's slot
  for DM kernels too, when the DFB has `data_format_metadata` (`dataflow_buffer.cpp:3236-3253`). Every DFB here has it.
  `get_entry_size()` would read the runtime `fifo_page_size`. The values are equal here; the getter follows the whitelist.
- **Shared kernel sources vs. per-variant tensors.** TTS and TST share their reader and compute sources, and the second
  operand slot holds `value_true` (TTS) or `value_false` (TST). The recipe's examples name bindings by role. Here the
  kernel-side names are slot names (`in_a`/`in_b`/`in_c`, `src_a`/`src_b`/`src_c`), while the `TensorParameter`s keep role
  names (`predicate`, `value_true`, `value_false`, `output`). A line in *Plan the spec* about shared sources across
  variants would help.
- **Named-arg names across sources.** All 11 readers share one 27-slot row but name the same slot differently
  (`nD_stride_b`, `true_nD_stride`, `src_b_nD_stride`, `tensor_nD_stride`). Rule 4 says to name args after their
  variables. With one host schema per role, the port picked the dominant name per slot. Kernels whose local differs keep
  the local: `const uint32_t true_nD_stride = get_arg(args::nD_stride_b);`.

## Open items for downstream

- **Shared kernel touches:** none. No borrowed, lent-and-bound or intra-op shared source. `dataflow/ternary_reader_nobcast_ttt.cpp`
  stays legacy for its only binder, `eltwise/binary_backward` (`binary_backward_program_factory.cpp:122`).
- **Bugs and stale code noticed, left alone** (no behaviour change in the port):
  - `device/ternary_program_factory.cpp:673`: "compute_program_hash coarsens each input tensor to its padded VOLUME". The
    key pins logical H/W (the audit's *Misc anomalies*).
  - `device/ternary_device_operation.cpp:580`: "re-applied via get_dynamic_runtime_args on cache hit". The hook is gone,
    and with this port the scalars ride the override's `ProgramRunArgs`.
  - `device/ternary_device_operation.cpp:8` includes `ttnn/operations/cb_utils.hpp` and uses nothing from it.
  - Five compute kernels include binary_ng's `eltwise_utils_common.hpp` / `eltwise_utils_sfpu.hpp` and use nothing from
    them; `eltwise_utils_sfpu.hpp` pulls in `api/dataflow/circular_buffer.h`. Kept: the kernel-side whitelist does not
    cover removing them.
  - The host still emits the `SRC_ROW_BCAST_CB1` / `SRC_SCALAR_CB1` reader defines, and two TTS / TST readers define
    `SRC_BCAST_CB1`. The self-audit grep does not match them. Renaming a define both sides must agree on fails silently
    (`#if` of an undefined macro is 0). They stay.
  - Dead reads carried as named reads: `src_b_num_tiles` and `src_a_num_tiles` in the five TTS / TST broadcast readers.
    The TTS / TST slot-2 read (`src2_addr`, always 0) went away with the address slots.
  - The audit's latent items stand unchanged: the missing pack-format restore after the LLK row broadcast, the
    divide-by-zero in `ternary_addcmul_int_sfpu_bcast.cpp`, the hang in the unreachable sharded branch of
    `tst_tts_reader_outer_bcast.cpp`, and the ND-sharded empty-optional dereference.
- **Removed with the code it described:** comments that documented the legacy CTA layout ("CB IDs from compile-time
  args (indices 0, 1, 2)", "TensorAccessorArgs start at index 3", "2 CB ids, then 2 TensorAccessorArgs blocks") and the
  legacy override's notes on common runtime args and CB re-pointing. Comments that described a buffer's role moved to
  the `DataflowBuffer` construction or the `DFBBinding`. The factory's "Output CB depends on variant: c_2 for binary_ng
  compatibility" comment was already wrong (the output was always c_3); it is now "Output DFB".
- **Test coverage:** the local run was a 552-case subset (below). The op's full set (about 3,400 cases across 15 files,
  plus the sweeps) is left to CI.
- **Doc suggestions:** say in *Plan the spec* how to name bindings for sources shared across variants; extend rule 7 to
  `constexpr` array lookups; give the override-guard rule a carve-out for guards that inspect the `Program` itself.

## Verification

Wormhole n150, 2026-10-30. `./build_metal.sh --build-tests` at the base plus this port: exit 0, no failed step, no
compiler warning from ternary sources. The loaded `_ttnncpp.so` contains the new override's `TT_FATAL` text. Every run
used Watcher (`TT_METAL_WATCHER=10`) and a fresh `TT_METAL_CACHE`.

**Legality checks.** The recipe's `skip_validation = false` force in `tt_metal/impl/metal2_host_api/*.cpp` was not
applied (see *Friction*). There are no `METAL2_CHECKS_FORCED` markers. Instead:
- The cache-miss path always validates. The spec adapter calls `MakeMeshWorkloadFromSpecs` and `SetProgramRunArgs`
  with the default `skip_validation = false` (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:994-998`).
- The custom concept's cache-hit path passes `skip = !validate_program_args` to `UpdateProgramRunArgs` (`:1032-1039`).
  The post-port run set `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'` for every stage. In that environment
  `ttnn.CONFIG.validate_program_args` reads `True`, and a misspelled key is rejected: the override is parsed.

**Test set.** Confirmed by the invoker before the runs: a 552-case pytest subset (all of `test_ternary.py`, the
category1 / composite / snake_beta / unsupported-bcast / mixed-rank / nightly lerp / docs-example files, 1-3 spread
cases per test function from the large sharding / broadcast / where files with every program-cache test in full, the
where / addcmul empty-shard cases and the lerp comparison-mode case), the gtests
`EltwiseSmoke.Where*`, `TestScopedGraphCapture.TernaryOp*`, `TestLevelizedGraphCapture.TernaryOp*`, and the audit's
relaxation probes. The same node-id list ran before and after the port.

| run | gtests | pytest (552) | relaxation probes | ternary kernels JIT-compiled |
|---|---|---|---|---|
| legacy baseline (base tree) | 5 passed, 2 skipped | 551 passed, 1 skipped | 6 / 6 | 22 / 22 |
| port | 5 passed, 2 skipped | 551 passed, 1 skipped | 6 / 6 | 22 / 22 |

- The set of passing node ids is identical. The skip is `test_lerp_with_public_keywords_in_comparison_mode`
  (`requires_fast_runtime_mode_off`); the two gtest skips are `TestScopedGraphCapture.TernaryOp*`. All skip in both runs.
- Probes: P1 (`[4,1,32,32]` then `[1,4,32,32]`) and P2 (`[1,32,64]` then `[1,1,32,64]`) for TTT `where`, TTS `lerp` and
  TST `mac`. Each pair made one new cache entry, and the second call (a hit, validated against the relaxed
  `TensorParameter`s) returned correct results: `where` bit-exact, `lerp` / `mac` with the same max error as legacy.
- The kernel caches of both runs hold the same 22 ternary kernel sources, which is every source the factory can select.
- Error-pattern lines (`TT_FATAL`, Watcher, `TensorSpec` mismatch) in the pytest logs: 25 before, 25 after. All come
  from the negative tests that expect a rejection.

**Anti-pattern self-audit** (op directory: 38 `.cpp` / `.hpp` files; 23 bound kernel files incl. the header):

| check | result |
|---|---|
| buffer address / `Buffer*` / `emplace_runtime_args` in the factory | 0 hits |
| CB indices in CTAs; `TensorAccessorArgs` / `get_arg_val` / `get_compile_time_arg_val` in bound kernels | 0 / 23 files |
| `cb` names (`[Cc][Bb]_\|_[Cc][Bb]\b\|\b[Cc][Bb]\b\|\bCB[A-Z]`, code files, unbound legacy files excluded) | 3 hits: `srcB_num_tiles` ×2 (false positive, *Friction*), and `#include "ttnn/operations/cb_utils.hpp"` in the off-limits `ternary_device_operation.cpp` (*Open items*) |
| conditional DFBs | `bcast_a/b/c` bound only with `is_llk_bcast` ∧ operand broadcast, kernels gated by `BCAST_A/B/C`; writer's `out` bound only when `!DST_SHARDED`, kernel gated by `#if !DST_SHARDED` |
| `.id` / `get_id()` at LLK sites | 0 / 23 files |
| CTA→RTA demotion | none (no per-group CTAs in legacy) |
| `allow_instance_multi_binding` | 0 |
| positional CTAs | 0 (CTAs: `has_sharding` on reader / writer; `num_tiles_per_cycle`, `scalar_is_true` on compute) |
| varargs | 0 |
| `tt_metal/` files in the diff; `METAL2_CHECKS_FORCED` / `DO NOT COMMIT` outside `docs/` | 0; 0 (the 6 hits under `docs/` are the recipe text) |
| `.md` cited from changed or new code (25 files scanned) | 0 |
| `opt_level` | 1 line, on the one compute `KernelSpec` (`O3`); DM kernels keep the default `O2`, as legacy |
| install coverage | all 22 kernels + `ternary_sfpu_copy_init.hpp` listed in the regenerated `cmake_install.cmake` |

**`hw_config` against legacy (resolved values):**

| kernel | legacy | Metal 2.0 |
|---|---|---|
| reader | `ReaderConfigDescriptor{}` → RISCV_1, NOC_0, dedicated | `create_reader_datamovement_config()` → RISCV_1, NOC_0, dedicated |
| writer | `WriterConfigDescriptor{}` → RISCV_0, NOC_1, dedicated | `create_writer_datamovement_config()` → RISCV_0, NOC_1, dedicated |
| compute | HiFi4; `fp32_dest_acc_en` = any 32-bit operand; `dst_full_sync_en` false; `math_approx_mode` false; `bfp8_pack_precise` false; `UnpackToDestFp32` on each Float32 c_0..c_3 | HiFi4; `enable_32_bit_dest` = same expression; `double_buffer_dest` true; Precise; `config_1xx` unset → Approximate; `UnpackToDest` on each Float32 `in_a` / `in_b` / `in_c` / `out` |

**`TT_FATAL` census** (`git grep -c` base vs working tree): every op file unchanged except
`device/ternary_program_factory.cpp`, 8 → 6.
- Kept (5): "Output must have a shard spec"; the shard-shape zero check; both "Unsupported Where variant" checks (builder
  and factory); the ragged-rows check.
- Subject deleted (1), the canonical case: `TT_FATAL(bs.slot < n, "ternary buffer slot {} out of range …")` guarded the
  `Buffer*` re-wrap of the address slots, which the tensor bindings replaced.
- Legacy override, translated (1): `TT_FATAL(data.size() == n, "… runtime args in the cached program but the builder
  produced {}")` checked each core's arg vector in the cached `Program`. The Metal 2.0 override cannot reach the
  `Program`. Its intent (every core of the program gets a full row, because a partial update keeps stale values) is now
  `TT_FATAL(per_core.cores.size() == worker_grid.num_cores(), …)` in the override.
- Legacy override, subject deleted (1) — **reviewer attention**: `TT_FATAL(num_tensor_indices <= 1, "… globally-allocated
  CB {} carries {} of c_0..c_3 …")` guarded the CB re-point chain. The chain is gone: each borrowed DFB names exactly one
  `TensorParameter` in its spec and is refreshed from that tensor's argument. The recipe says override guards are never
  subject-deleted; this one inspected the `Program`'s CBs, which the declarative override has no handle on.
