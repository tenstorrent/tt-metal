# Metal 2.0 Port Report — `experimental/bcast_to`

## Outcome

**PORTED.** The op's single factory (`BcastToOperation`, previously a direct `create_descriptor`) is now `BcastToOperation::BcastToProgramFactory` on `ProgramSpecFactoryConcept`. All 12 kernel sources it can select were converted together; one of them (the empty NONE compute kernel) needed no edit. Every baseline test passes, with results identical test-by-test before and after the port. The Metal 2.0 legality checks were forced on and confirmed live.

| test set (confirmed with invoker) | pre-port | post-port |
|---|---|---|
| `unit_tests_ttnn --gtest_filter='*Broadcast_to*'` (`tests/ttnn/unit_tests/gtests/test_broadcast_to.cpp`) | 22 passed | 22 passed |
| `pytest tests/ttnn/unit_tests/operations/eltwise/test_broadcast_to.py tests/ttnn/nightly/unit_tests/operations/experimental/test_bcast_to.py` | 108 passed, 5 skipped | 108 passed, 5 skipped (same per-test results) |

- All runs used `TT_METAL_WATCHER=10`. No watcher asserts.
- The post-port logs show both `METAL2_CHECKS_FORCED` markers: `program_spec.cpp` and `program_run_args.cpp`, 22× each in the gtest run and 99× each in the pytest run.
- The 5 skips are the pre-existing `test_broadcast_to_bf8_b` skip ("not stable").
- `test_bcast_to_program_cache` passed, which covers binding refresh across cache hits. The int32/uint32 row/col/scalar tests passed, which covers the `enable_32_bit_dest` + `unpack_modes` path.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`. This checkout (`Metal_Ports`) has no recipe tree, so the hash comes from the `/localdev/edwinlee/Port_Recipe` checkout. The `/localdev/edwinlee/metal2_port.md` copy I followed is byte-identical to `Port_Recipe/.../ai/port/metal2_port.md`.
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit gate override (carried from the brief):** the TTNN-factory-concept gate was mechanically RED because the readiness-sheet row is stale. It predates #57409 and still lists `legacy device-op` / `BcastToTileFactory`. The user (op author of #57409) directed proceeding on the code-side evidence. The port found nothing contradicting that evidence: the op was in the `descriptor` concept with no `override_runtime_arguments`, as the audit said. Reconciling the sheet row is still an open bookkeeping item.
- **Test baseline:** the invoker confirmed the three files above.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. There is no `override_runtime_arguments`, so the framework refreshes the `input` / `output` tensor bindings on cache hits. `test_bcast_to_program_cache` exercises this.

### Device-op-class edits
- **Direct-descriptor → conventional factory (ttnn_factory.md exception 3).** The op arrived in the direct-descriptor shape: `static ProgramDescriptor create_descriptor(...)` on `BcastToOperation` itself, with no `program_factory_t`. I replaced it with a nested `struct BcastToProgramFactory { static ProgramArtifacts create_program_artifacts(...); }` and `using program_factory_t = std::variant<BcastToProgramFactory>;` (`device/bcast_to_device_operation.hpp:38-47`). The existing comment about hashed RTAs moved with the method, verbatim. Includes swapped: `<tt-metalium/program_descriptors.hpp>` → `ttnn/metal_v2_artifacts.hpp` + `<variant>`. Nothing else in the class changed.
- Pybind entry points removed: none. `bcast_to_nanobind.cpp` never bound `create_descriptor`.
- Custom `compute_program_hash`: none. The default reflection hash is untouched.

### Open items
- Relaxation candidates: none noted. The kernels bake `N/C/Ht/Wt` in as RTAs that the default hash keys on, so strict matching is right.

## Handoff points

- **API surface: device-op shape change (exception 3).** `BcastToOperation` gained `BcastToProgramFactory` / `program_factory_t` and lost the static `create_descriptor`. Nothing outside the op directory referenced `BcastToOperation::create_descriptor` (grep over `ttnn/`, `tests/`, `models/`), and there is no Python surface change. Owner: TTNN infra, for awareness only.
- **Readiness sheet:** the `experimental/bcast_to` row still needs reconciling. It should read `Concept` = Metal 2.0 (ported), and the phantom `BcastToTileFactory` factory row should go. Owner: readiness-sheet owner (Diego).

No capitulation, boundary-rule violation, kernel-lib gap, or framework gap was hit.

## Successes

- **Brief + census agreed on CB endpoints.** I re-derived the census from the kernels: `c_0` is 1P+1C in every config, `c_1` is dead under NONE and 1P+1C otherwise. It matched the brief exactly. The brief's warning *not* to bind `c_0` to the empty NONE compute kernel kept the spec clean. The result is two DFBs, no multi-binding flag, and the `dst` DFB declared only when `subtile_broadcast_type != NONE` (`device/bcast_to_program_factory.cpp:191-205`). The validator accepted it on the first run.
- **[Pass DFB handles directly to LLKs and kernel-lib helpers](port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers).** The template-parameter note was exactly right. `dfb::src` / `dfb::dst` go straight into `ckl::input(...)` / `ckl::output(...)` in NTTP position, and into `compute_kernel_hw_startup` / `unary_bcast_init`, with no donor change (e.g. `compute_interleaved_row_bcast_to.cpp:30-31,45,50`).
- **Recipe §Compiler options / opt_level self-audit.** The legacy factory has no `opt_level` line at all. The recipe's "an absent field still means O3 on compute" rule made me set `KernelBuildOptLevel::O3` explicitly (`device/bcast_to_program_factory.cpp:297`). Nothing would have caught this otherwise.
- **Hardware config Style B.** The op builds `ComputeConfigDescriptor` directly, so I built `ComputeHardwareConfig` directly as well, setting only `enable_32_bit_dest` plus the `unpack_modes` entry. All other defaults coincide with legacy: HiFi4, Precise ↔ `math_approx_mode=false`, `double_buffer_dest=true` ↔ `dst_full_sync_en=false`, Approximate BFP pack. Routing through `to_compute_hardware_config` would have been the wrong tool.
- **Legality-check forcing + markers** made it unambiguous that the green post-port run was validated. The markers were absent in the baseline run, as expected for the legacy path, and present post-port.
- **"While a test run is in flight, kernel sources are frozen."** I staged kernel edits in a scratchpad script and applied them only after the baseline run exited. Host edits went in during the run, which the recipe says is safe.

## Friction

### Gaps
- **Recipe hw_config spellings are stale vs. the headers in this tree.** The recipe (§Hardware configuration) and its tables use `ComputeGen1Config`, `DataMovementGen1Config{…}`, `std::get<ComputeGen1Config>(compute_hw).unpack_modes`, `create_reader_datamovement_config(device->arch())` and `to_compute_hardware_config(device->arch(), config)`. The real headers differ:
  - `compute_hardware_config.hpp` / `data_movement_hardware_config.hpp` have a single `ComputeHardwareConfig` / `DataMovementHardwareConfig` struct, with optional `config_1xx` / `config_2xx` sub-configs.
  - `unpack_modes` is a direct member.
  - `ttnn::create_{reader,writer}_datamovement_config()` and `ttnn::to_compute_hardware_config(config)` take no arch.

  The values map 1:1, so the port was unaffected. But a porter copying the recipe snippets gets compile errors. "Headers first" (recipe §Read this first) resolved it in minutes.
- **`unpack_modes` for Int32/UInt32 with `UnpackToDestFp32`.** The recipe says the required-entry rule is Float32-only, and not to add Int32/UInt32 entries preemptively. It says nothing about a legacy vector that *explicitly* set `UnpackToDestFp32` on an Int32/UInt32 CB, as this op does for all 32-bit formats (`bcast_to_program_factory.cpp` legacy `:199-203`). I translated it faithfully to `UnpackMode::UnpackToDest` for all three formats (`device/bcast_to_program_factory.cpp:289-290`), since that is a mirror rather than a preemptive entry. The validator accepted it and the int tests pass. One sentence in the recipe distinguishing "mirror an explicit legacy entry" (always) from "add a newly-required entry" (Float32-only) would remove the doubt.
- **Conditional `unpack_modes` under a conditional *binding set*.** Under NONE the compute kernel binds no DFB, so the legacy `unpack_to_dest_mode[c_0]` has no key to land on. The validator rejects keys for unbound DFBs. I gate the entry on the same `uses_compute` condition as the bindings. This is behaviorally inert, because the NONE compute kernel is empty and unpacks nothing. The recipe's conditional-`unpack_modes` note covers this in spirit ("gated on the same condition as its binding"), but only for conditional *DFBs*, not a conditional *kernel role*.

### Confusion
- **Brief claim "NONE config has no active test coverage" is wrong.** The C++ gtest `tests/ttnn/unit_tests/gtests/test_broadcast_to.cpp` exercises NONE in several suites: `ChannelAndBatch` `{1,1,64,64}→{1,3,64,64}` / `{3,1,64,64}`; `LargeTensor` `→{1,32,64,64}` / `{32,1,64,64}`; `CombinedDimensions` `{1,1,32,32}→{7,17,32,32}`; and `NonAlignedDimensions` `{1,1,7,13}→{1,1,7,13}`, `{1,1,30,30}→{1,5,30,30}`, etc. The audit looked only at the pytest files. Because the gtest is in the confirmed baseline, the ad-hoc NONE run the brief asked for was unnecessary: those gtest cases passed before and after. Suggestion: the audit's test-coverage heads-up should sweep gtests too.
- **Reference port provenance.** No reference port was supplied. I glanced at `data_movement/clone` only to sanity-check the factory shape against the headers. It includes `api/tensor/noc_traits.h` in its kernels, which is outside the recipe's "adds exactly two headers" rule. I did not copy that. Recorded per the recipe's note that a contradicting ported op is friction.
- **Shared working tree.** A parallel session was porting `experimental/deepseek_moe_post_combine_tilize` in the same checkout. Its uncommitted changes appeared in my `git diff "$BASE"` scans (the self-audit commands are not scoped to the op directory). I scoped the sweeps to `$OP` and staged only `bcast_to` paths. The recipe's audit commands assume a single-port tree. Scoping the "no tt_metal/ in diff" and ".md citation" checks to `-- $OP` (plus any peer-fork dirs) would make them robust to this.

## Open items for downstream

- **Shared kernel touches:** none. All 12 sources are private to this op and were converted in place. No `_metal2` forks.
- **Dead RTAs carried forward** (behavior-preserving, see the audit's *Misc anomalies*). `n_stride` / `c_stride` are read but unused in every writer and compute kernel. Other slots are dead depending on the config. They are now *named* schema entries, so trimming them later is an obvious cleanup for the ops team.
- **RTA→CRTA candidates** (not converted; that would change dispatch semantics). `N`, `C`, `Ht`, `Wt`, `n_stride` and `c_stride` take the same value on every active core, but idle cores get 0. A future cleanup could make them CRTAs (idle cores already exit on `num_tiles == 0`).
- **Empty NONE compute kernel** is still built and dispatched with 12 RTAs. Dropping it (and its KernelSpec) under NONE is an ops-team optimization, not port work.
- **Legality-check scaffolding** (`skip_validation = false` + `METAL2_CHECKS_FORCED` in `tt_metal/impl/metal2_host_api/program_{spec,run_args}.cpp`) is left **uncommitted** in the working tree. The parallel port in this checkout may still rely on it. Revert it with `git checkout -- tt_metal/impl/metal2_host_api/` once no port in this tree needs it.
- **Test coverage note:** there is no Float32 case in the confirmed set. The `enable_32_bit_dest` + `UnpackToDest` path is covered only via Int32/UInt32 (nightly). A Float32 bcast_to case would close that gap.
