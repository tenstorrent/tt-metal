# Metal 2.0 Port Report — `experimental/dropout`

## Outcome

**`PORTED`.** Both factories of `DropoutDeviceOperation` converted to Metal 2.0, with all three kernels. Nothing in the op builds a `ProgramDescriptor` any more.

- `DropoutProgramFactory` → **`CustomProgramSpecFactoryConcept`**, the brief's target.
- `DropoutMeshWorkloadFactory` → **`MeshWorkloadSpecFactoryConcept`**, the brief's target. That concept is outside the port recipe's stated coverage; the invoker authorised it, with `experimental/paged_cache` (#56022) as precedent.

**Behaviour is bit-identical to the pre-port op** on everything measured:
- **Confirmed test set:** identical pass/skip results before and after, with Watcher on.
- **Metal 2.0 legality checks:** forced on and proven live.
- **Program cache:** the same number of entries before and after.
- **Output equivalence:** 113 dropout outputs compared bit for bit against the pre-port build, 0 mismatches. They cover both factories, cache hits, in-place, and a real 1×2 mesh. Details under [Verification](#verification).

The port ran on an **invoker override of a RED audit**: the TTNN factory concept gate is RED only because the readiness sheet is stale (Handoff #1). See `METAL2_PORT_BRIEF.md`.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`. The worktree is a `main` checkout without the docs, so this line comes from the `akertesz/op-porting-recipe` tip, re-checked unchanged on 2026-10-14.
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Code:** `origin/main` @ `3c202f9944d`.

## TTNN ProgramFactory

### Concept realized

- `DropoutProgramFactory` → **`CustomProgramSpecFactoryConcept`**: `create_program_artifacts` plus an `override_runtime_arguments` returning `ProgramRunArgs`.
- `DropoutMeshWorkloadFactory` → **`MeshWorkloadSpecFactoryConcept`**: `create_mesh_workload_artifacts(…, tensor_coords)` emits one single-coordinate `PerCoordProgram` per device, built by `DropoutProgramFactory::create_program_artifacts` with `seed = per_device_seed(args, coord, input)`. This is the same delegation as before, and the same program set the descriptor adapter built. Its range-scoped override delegates to `DropoutProgramFactory::override_runtime_arguments` with `range.start_coord()`; every range is one coordinate, so that is exact.

**Cache-hit tensor-arg completeness.** On both concepts the override replaces the framework's tensor-binding refresh. So it returns a `TensorArgument` for **both** `TensorParameter`s (`input`, `output`) on every dispatch, each sourced from its own tensor, as the ported-from override sourced both addresses (`dropout_program_factory.cpp`, `override_runtime_arguments`). No parameter is skipped. The refresh set otherwise mirrors the ported-from override exactly:
- `num_tiles` / `start_id` per node for the reader and writer, which it rewrote "anyway";
- the compute `seed` per node, with the `use_per_device_seed` branch unchanged.

### Device-op-class edits

- **Pybind entry points removed: none.** `dropout_nanobind.cpp` binds only `ttnn.experimental.dropout`. **No user-visible API change.**
- **Custom `compute_program_hash`: left intact.** The backdoor `DropoutParams::attribute_values()` (`device/dropout_device_operation_types.hpp:28-32`, excludes `seed`) is untouched.
- `dropout_device_operation.{hpp,cpp}`, `dropout_device_operation_types.hpp`, `dropout.{hpp,cpp}` and `dropout_nanobind.*` are byte-identical to `main`. The diff touches only `device/dropout_program_factory.{hpp,cpp}` and the three kernels.

### Open items

- Relaxation candidates: none acted on. The audit's fallible candidate (the old volume-only hash suggests a shape relaxation) stands as a team-only note; the port keeps strict matching, matching `TensorParameter relaxation = none`.

## Handoff points

### 1. The readiness sheet is stale for this op — the port ran on an invoker override

**Owner: readiness-sheet owner (Diego).**
- Rows 187-188 (`experimental/dropout`) record `Runtime-args update (get_dynamic_runtime_args) = yes` and `Override runtime args method? = no`. #50341 (2026-08-19) made both false: there is no `get_dynamic_runtime_args`, and both factories have `override_runtime_arguments`.
- `Is able to port? = no` and `Op Classification = Broken Op` are derived from those cells.
- Once corrected, the `DropoutMeshWorkloadFactory` row's derived `Porting Target` should come out as the mesh-workload concept, **not** `CustomProgramSpecFactoryConcept`. On that concept the per-device seed is lost silently: one `ProgramArtifacts` stamped on every range (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:984-991`).
- The row's `Execution Model = SPMD` is also questionable for a per-coordinate factory. The four paged_cache mesh rows (201, 203, 205, 207) carry the same pair of values, so this is likely a sheet-wide derivation gap.

### 2. A second recipe-uncovered `MeshWorkloadSpecFactoryConcept` port

**Owner: port-recipe maintainer.** paged_cache (#56022) was the first port to land a mesh factory on this concept; dropout is the second, and was mechanical given that precedent. The four decisions paged_cache had to make unaided are recorded in `METAL2_PORT_PLAN.md` (TTNN ProgramFactory → Implementation notes):
- one program per coordinate;
- no omitted coordinates;
- ranges bounded by `tensor_coords.coords()`;
- `range.start_coord()` in the override.

With two ports now sharing the "per-coordinate `create_descriptor` → per-coordinate `PerCoordProgram`" translation, it is ready to be written into the recipe. The audit side needs the matching recognition rule (see `METAL2_PREPORT_AUDIT.md` Recipe notes).

### 3. No kernel-lib or LLK gaps

`compute_kernel_lib::eltwise_chain`'s `input(...)` / `output(...)` take the buffer id as a `constexpr uint32_t` inside a template argument. `dfb::in0` / `dfb::out` passed straight through with no donor-side change and no fork. `compute_kernel_hw_startup` likewise took the tokens directly. No `sem::` or `tensor::` handle crosses the op boundary.

## Successes

- **Anti-pattern: Demoting per-group CTA to RTA (port_patterns.md) — its "correct port" shape worked first build.** The legacy split built `compute_group_1` / `compute_group_2` with different `per_core_block_cnt`. The port keeps two compute `KernelSpec`s in two `WorkUnitSpec`s, with the reader and writer listed in both work units. The validator accepted it, and the per-group CTA stays compile-time.
- **The override-translation rule ("fidelity, not analysis") plus the in-place watch-for** produced an override that is bit-exact on cache hits, including the in-place case (input and output bound to one tensor) on both factories. The capture compares 2 in-place cache-hit outputs per shape and factory.
- **The `opt_level` absent-line check.** `grep -n opt_level` on the legacy factory printed nothing, which reads as "nothing to carry". The recipe's table turned that into an explicit `O3` on the shared compute helper, so both compute `KernelSpec`s carry it.
- **Style B compute config.** The legacy `ComputeConfigDescriptor` set all four knobs as literals. Building `ComputeHardwareConfig` field by field, with the `dst_full_sync_en` → `double_buffer_dest` inversion, reproduced it exactly.
- **The paged_cache precedent** (invoker-supplied) made the mesh factory a direct transcription: the whole port compiled on the first build, and every test passed on the first run.

## Friction

### Gaps

1. **§Hardware configuration names types and helpers that `main` has replaced.**
   - The recipe uses `ComputeGen1Config` / `DataMovementGen1Config` and `create_reader_datamovement_config(device->arch())`.
   - On `main` the types are `ComputeHardwareConfig` / `DataMovementHardwareConfig`: generation-agnostic fields, plus optional `config_1xx` / `config_2xx` for generation-specific knobs. `bfp_pack_precision_mode` moved under `config_1xx`.
   - The TTNN DM helpers take `bool disable_dfb_implicit_sync_for_all`, not an arch.
   - The field mapping table itself still holds; only the names and nesting moved. The headers answered it, as the recipe says they will.
2. **§Ensure the legality checks: the grep misses the spec-side gate.** `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp` no longer finds `BuildProgramFromSpec`. It moved to `tt_metal/impl/metal2_host_api/program_spec/program_spec.cpp`, which is the site the recipe calls the spec-side choke point. Used literally, the grep forces only the run-args file. The marker step would then catch the miss, but only as a missing second marker. Fix: `grep -rn`.
3. **No brief shape for an invoker override of a RED audit.** The audit issues a brief only on GREEN; the port requires a GREEN audit. Here the only RED gate was a stale-sheet conflict that the code had already resolved, and the invoker chose to proceed. The brief was written in the GREEN template with the failed gate shown as ✗ plus the override recorded. A sanctioned form for this would help: it is likely to recur while the sheet lags code changes.

### Confusion

4. **Whitelist rule 7 keys on the legacy *declaration*, but here the legacy *argument* was a constant expression.** The reader and writer passed `get_tile_size(cb_id)` inline (reader `:38`, writer `:40` pre-port), with `cb_id` a `constexpr uint32_t`, so the legacy call was effectively constant. By the rule's test (no `constexpr` declaration) it became the member getter `dfb_in0.get_tile_size()` / `dfb_out.get_tile_size()`. Whether the getter folds to the same immediate depends on the optimizer seeing through the `DataflowBuffer` constructor, which is exactly the gamble §A says the port should not take for declared `constexpr` values. Outputs are bit-identical; per-tile kernel time was not profiled (see Open items #2). The rule could say whether a constant-expression *argument* counts like a `constexpr` *declaration*.

## Open items for downstream

1. **Shared kernel touches: none.** All three kernels are owned by this op and converted in place. Both binders (the two factories) converted in this change, so no `_metal2` fork was created and there is no sunset list.
2. **Perf not measured.** The port changes no work split, kernel config or opt level, so a delta is not expected. The one candidate is Confusion #4 (tile-size getter in the per-tile DM loop). A Tracy device-kernel-duration A/B on `test_dropout.py` would settle it.
3. **Op-team findings preserved, not acted on** (they restate the audit's Misc anomalies; the port carries each unchanged):
   - Dead `#ifdef BACKWARDS` / `#ifdef OUT_SHARDED` branches in the reader and writer; the factory sets no defines. Converted mechanically.
   - Sharded **ROW_MAJOR** input passes `validate_on_program_cache_miss` but is processed as tiles. Unverified on hardware.
   - `per_device_seed = seed + device_id` correlates masks across devices and consecutive seeds.
4. **Device-op header keeps an unused include.** `dropout_device_operation.hpp` includes `<tt-metalium/program_descriptors.hpp>`, unused after the port. The device-op class is off-limits, so it stays.
5. **Test coverage.**
   - Per-device seed differences are checked only by `Mesh1x2UtilsTest.DropoutDifferentSeed` (2 devices). It passed before and after on the 2× p150b bench.
   - The bit-exact mesh capture adds a stronger check: identical per-device outputs before and after, with devices differing exactly when `use_per_device_seed`.
   - No in-repo test runs the mesh factory on more than 2 devices.

## Verification

### Legality checks were provably live

`skip_validation = false` was forced as the first statement of all **nine** functions found by `grep -rn 'bool skip_validation' tt_metal/impl/metal2_host_api/` (see Friction #2). A one-shot marker went into each of the two translation units. The post-port run logged:

```
51 METAL2_CHECKS_FORCED (program_spec.cpp:27)
51 METAL2_CHECKS_FORCED (program_run_args.cpp:675)
```

Both files were fresh and both gates ran on every program construction. The scaffolding has been reverted: the diff contains no `tt_metal/` file and no `METAL2_CHECKS_FORCED` / `DO NOT COMMIT` string.

### Tests

Same confirmed set before and after, `TT_METAL_WATCHER=10`, fresh `TT_METAL_CACHE` per run, on a Blackhole p150b (×2 for the mesh test):

| | Baseline (pre-port) | Post-port |
|---|---|---|
| `tests/ttnn/unit_tests/operations/data_movement/test_dropout.py` | 8 passed | 8 passed |
| `ttml_tests --gtest_filter='DropoutTest.*'` | 2 passed, 1 skipped | 2 passed, 1 skipped |
| `ttml_tests --gtest_filter='Mesh1x2UtilsTest.DropoutDifferentSeed'` (1×2 mesh) | 1 passed | 1 passed |

- The skip is `TestKeepRatioApproximatelyNormal`'s unconditional `GTEST_SKIP` (RNG-quality note); it is identical before and after.
- The Watcher log is 8,680 lines with no error, assert or `0xdeadc0de`.
- `test_dropout.py` exercises both cache-hit overrides on real hits: `test_dropout_seed_distinguishes_cache_entries` (×4, both factories, in-place and not) and `test_dropout_cache_hit_rederives_buffer_address`.

### Bit-exact equivalence (beyond the test set)

A scratch harness (not committed) ran `ttnn.experimental.dropout` on the pre-port and post-port builds and compared outputs with `torch.equal`:
- **Single device:** 4 shapes × 2 factories × 4 seeds, each shape/factory run as successive calls so later seeds hit the program cache, plus in-place calls. **49/49 identical.** Program-cache entry count 16 before and 16 after.
- **1×2 mesh:** the same shapes, factories and seeds, comparing each device's output. **64/64 identical.** In both builds, devices differ exactly when `use_per_device_seed=True`.

### Anti-pattern self-audit

Denominator for every code sweep: **12** `.cpp` / `.hpp` files under the op directory (the `.md` artifacts are excluded).

| Check | Result |
|---|---|
| Buffer address in run args (`->address()`, `emplace_runtime_args`, `RTArgList`, `Buffer*`) | **0** |
| Magic CB indices (`CBIndex`) / positional CTA vectors | **0** |
| `TensorAccessorArgs` surviving | **0** |
| `cb`-shaped names (`grep -rnE '[Cc][Bb]_\|_[Cc][Bb]\b\|\b[Cc][Bb]\b\|\bCB[A-Z]'`) | **0** |
| `CircularBuffer` / `CBDescriptor` / `circular_buffer.h` / `.cbs` | **0** |
| `.id` on a `dfb::` handle | **0** |
| `allow_instance_multi_binding` | **0** |
| Varargs | **0** |
| Legacy positional arg readers (`get_compile_time_arg_val`, `get_arg_val<`, `get_common_arg_val`) | **0** |
| Ephemeral `.md` cited from code (5 changed code files scanned) | **0** |
| Forced-legality scaffolding in the diff | **0** |
| Files outside the op directory in the diff | **0** |

- **TT_FATAL census:** `diff` of per-file `TT_FATAL|TT_ASSERT|TT_THROW` counts, pre-port vs. working tree, gives **no output**. The factory keeps all four guards: the core-range check in `for_each_dropout_core`, both "core group 2" guards (cache-miss build and override), and the mesh factory's `use_per_device_seed` assert.
- **`opt_level`:** one source line (`create_compute_kernel`), reached by both compute `KernelSpec` construction sites, so both carry an explicit `O3`. The DM kernels set none, matching the legacy O2.
- **`hw_config`:**
  - Reader and writer use `create_reader_datamovement_config()` / `create_writer_datamovement_config()`. Their Gen1 triples (RISCV_1/NOC_0 and RISCV_0/NOC_1, dedicated) equal the legacy `ReaderDataMovementConfig` / `WriterDataMovementConfig` on every arch (`tt_metal/impl/kernels/kernel_types.cpp:20-38`, `preferred_noc_for_dram_{read,write}`).
  - Compute: `HiFi4` / `Precise` / `enable_32_bit_dest=false` / `double_buffer_dest=true`, field by field from the legacy literal. No `unpack_modes` (no FP32 Dest, and the legacy entry is unset). `config_1xx` is unset, which matches the legacy `bfp8_pack_precise` default.
- **Conditional bindings:** none. The only conditional construct is the whole `compute_g2` `KernelSpec` plus its work unit, present exactly when `core_group_2` is non-empty, as in legacy.
