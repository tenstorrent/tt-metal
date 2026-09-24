# Metal 2.0 Port Report — `eltwise/unary`

## Outcome

**PORTED.** `UnaryDeviceOperation::ProgramFactory` is the op's only factory. It now satisfies `CustomProgramSpecFactoryConcept`, and so do all 11 of its bound kernel entry points. That is 9 runtime-selected compute sources, one of them a new `_metal2` fork, plus the reader and writer.

The port carries **one sanctioned exception** to `ttnn_factory.md`: an edit to the op's custom `compute_program_hash`. The user sanctioned it explicitly. It is recorded under [Device-op-class edits](#device-op-class-edits) and as the first [Handoff point](#handoff-points).

Verification ran the user-confirmed set with Watcher on (`TT_METAL_WATCHER=10`), once before the port and once after, and compared the two per test from JUnit XML:

| group | tests | pre-port | post-port |
|---|---|---|---|
| gtest (`UnaryProgramHashFixture.*`, `EltwiseSmoke.UnaryAbsNegExp`, `TTNNFixtureWithDevice.TestGenericOpUnaryRelu*`) | 5 | 5 pass | 5 pass |
| core (`test_unary.py`, `_sharding`, `_program_cache`, `_fp32`, `_ops_ttnn`, `_lgamma`, `test_activation.py`, mac_tss tests) | 1263 | 1245 pass / 18 skip | identical |
| `test_where.py -k "tss or TSS"` | 104 | 101 pass / 3 skip | identical |
| nightly `test_unary_shard.py`, `test_unary_subcoregrids.py` | 54 | 54 pass | identical |
| `test_unary_category{1..6}_bfloat16.py` (all) | 384 | 381 pass / 3 skip | identical |
| **total** | **1810** | **1786 pass / 24 skip / 0 fail** | **1786 pass / 24 skip / 0 fail — no test changed state** |

The set of `TT_FATAL`s logged, all of them tests' expected errors, is also identical before and after (31 each).

Validation was on for the post-port run, with one caveat: the recipe's forced-scaffolding proof was not possible. See [Friction](#friction), first entry.

One comment-only edit landed after the post-port run started: the `get_shard_specs` comment at `common/unary_utils.cpp:75`, "CB-aliasing" → "borrowed-DFB". It has no code effect.

## Provenance

- **Recipe docs (this port):** `edf75ffab9c 2026-09-23 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`.
  - The recipe is tracked on this branch and was read in place.
  - The auto-mode permission classifier denied the verbatim `git log -1 --format='%h %cs %s' -- docs/.../metal_2.0/` command. The line above comes from an earlier first-hand `git log` of the recipe directory in this session: `edf75ffab9c` is `HEAD` and is the latest commit touching `metal_2.0/`.
- **Audit docs (inherited):** `edf75ffab9c 2026-09-23 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`.
  - Tree-identical to `origin/akertesz/op-porting-recipe` `4bd4bf42bfe`.
  - Relaxation doc `analyses/relaxations/eltwise_unary.md`, blob `653172edfca`. It is committed with this port per decision 3.
- **User decisions carried in** (answers to the audit's *Questions for the user*):
  1. The `distribution_key` swap is sanctioned.
  2. Sheet provenance stands as supplied.
  3. Commit the relaxation doc on this branch.

## TTNN ProgramFactory

### Concept realized

`CustomProgramSpecFactoryConcept`, as the audit chose.

- `override_runtime_arguments` now returns `ProgramRunArgs` and drops the `Program&` parameter (`device/unary_device_operation.hpp`).
- The override returns a `TensorArgument` for **both** io `TensorParameter`s (`input`, `output`) on every hit. There are no op-owned tensors.
- The ported-from override refreshed every tensor-backed value: RTA0 addresses, accessor CRTAs, and sharded CB addresses. So nothing is skipped.

Translation fidelity: the legacy override wrote exactly the slot set `create_descriptor` wrote. That is every RTA on every `worker_grid` core, with noop cores zero-filled, RM chunk tails, and compute scalars. So miss and hit now share one builder, `make_run_args` (`device/unary_program_factory.cpp`), over the pre-existing shared `enumerate_core_rt_args`. The two statements in the legacy override that did not become RTAs were:

- the accessor-CRTA rebuild, now covered by the `dynamic_tensor_shape` binding re-emission;
- `apply_descriptor_runtime_args` for sharded CB addresses, now covered by `borrowed_from` plus `tensor_args`.

Both became tensor bindings. None became RTA values.

### Device-op-class edits

1. **⚠ SANCTIONED EXCEPTION — custom `compute_program_hash` edited** (`device/unary_device_operation.cpp`, `distribution_key`). This is a named, deliberate exception to `ttnn_factory.md` § "The cache key: leave the custom hash alone". The user sanctioned it (decision 1), and relaxation doc §2 requires it.
   - **What changed:** `distribution_key` now reads the sharded geometry from `spec.compute_buffer_sharding_args().buffer_distribution_spec()` for both slots. It no longer reads the `Buffer`'s `buffer_distribution_spec()`, whose fallback to the spec applied only when the output had no buffer yet.
   - The lambda lost its `const Tensor*` parameter.
   - The `TODO(port)` naming this swap was retired, and the explanatory comment was updated to give the new rationale.
   - The change swaps the source rather than adding one. Hashing both sources measured +16% (doc §2).
   - Nothing else in the hash changed. `tensor_layout` terms, the RM `padded_shape` term, the shard-volume optionals, and `to_hash()` are untouched.
   - **Why the recipe's rule couldn't hold here:** the rule assumes a hash fix can land upstream before the port. Here it can't. Legacy bakes the **buffer's** geometry into accessor CTAs via `TensorAccessorArgs(*src_buffer)`, so before the port the Buffer source is the correct key, and a spec-keyed hash would under-pin what legacy bakes. After the port, the `TensorParameter` bindings resolve geometry from the spec, and `tensorspecs_match_with_relaxation` compares `spec.compute_buffer_sharding_args()` (`tensor_spec_relaxations.cpp:105-109`). So the key's correct source changes *with* the port and must change in the same edit.
   - The two resolutions agree for every freshly allocated tensor (`tensor_impl.cpp` builds the buffer from `tensor_spec.compute_buffer_sharding_args()`). The only known decoupler is a TILE-sharded `view` / `reshape`, and no divergent pair was ever constructed (doc §2). So cache behavior on the test set is unchanged, as the identical program-cache test results show.
2. **Factory struct signatures and includes** (`device/unary_device_operation.hpp`). `create_descriptor` → `create_program_artifacts`, and the override's return type and parameters changed. These edits are forced by the concept. `program_descriptors.hpp` / `program_descriptor_patching.hpp` were replaced with `ttnn/metal_v2_artifacts.hpp`. No other device-op edits.
3. **Pybind entry points removed:** none. `unary_nanobind.cpp` never exposed `create_descriptor`.

### Open items

- **Relaxations declared, per the analysis doc:** `{.dynamic_tensor_shape = true, .relax_logical_rank = true}` on both `input` and `output`, unconditionally. `match_page_size` and `match_padded_shape_only` are not set.
  - At analysis time this was the first shipped non-experimental factory to declare any `TensorSpecRelaxations`, and the first anywhere to set `relax_logical_rank`. No validation throw was seen.
  - That includes `test_unary_sharded_input_on_interleaved_path_cache_reuse`, which is regime row 5 of the doc: a sharded buffer on the live-accessor path.
- All six of the relaxation doc's validity checks still hold on the post-port tree, with check 2 now reading the spec source. Check 6's "override re-applies the whole split" holds via `make_run_args`.
- The doc's §3 concern about the legacy override's `&&`-bounded accessor-CRTA refresh loops (silent truncation) is **moot after the port**. That loop is gone, and the binding re-emits the shape words itself.

## Handoff points

1. **Recipe maintainers — recipe deviation: a hash edit carried by a relaxation analysis.** *(Tag: recipe / ttnn_factory.md.)*
   - `ttnn_factory.md` forbids any port-time edit to a custom hash and routes a hash/declaration mismatch to "stop, upstream error". The recipe assumes an analysis only *declares*, and that any hash fix lands upstream before the port.
   - For unary, the analysis doc (§2) shows the key's correct *source* changes with the port itself, because the geometry source moves from Buffer to TensorSpec when `TensorAccessorArgs` becomes a binding. So no upstream-only sequencing exists.
   - This port made the edit under the user's explicit sanction.
   - **Suggested recipe change:** name "hash edit carried by a relaxation analysis doc, in the same edit as the declaration" as a fourth sanctioned device-op-class exception, with the conditions this case met:
     - the analysis doc prescribes the exact edit;
     - the edit is a source swap, not a widening;
     - the invoker sanctions it;
     - it is recorded here.
   - Alternatively, rule it out explicitly so the next such op stops at audit.
2. **Recipe maintainers — validation forcing done without the `tt_metal/impl` scaffolding.** *(Tag: recipe / environment.)* See Friction, first entry. If the markers are required evidence, a maintainer or the invoker needs to apply the force in a session where the edit is permitted, then re-run.
3. **Framework (Metal 2.0 host API): no gaps hit.** The first production use of `relax_logical_rank`, and relaxations on sharded and borrowed `TensorParameter`s, all validated without incident.
4. **Kernel-lib / LLK:** none. `dfb::name` passed straight into `compute_kernel_hw_startup`, `copy_init`, `copy_tile`, `pack_tile`, and `compute_kernel_lib::input` / `output` in NTTP position. It compiled first time.
5. **Removed pybind surface:** none.

## Successes

- **Brief watch-fors all fired correctly.**
  - Compute `opt_level = O3`, set explicitly (`unary_program_factory.cpp`, compute `KernelSpec`). Without it the spec compiles at O2.
  - The unread trailing compute CTA was carried as `input_data_format`, not dropped.
  - The `tmp0` `unpack_modes` entry is gated on the LOGIT-conditional DFB. Legacy set `unpack_to_dest_mode[c_1]` even when `c_1` didn't exist, and the validator rejects a key for an unbound DFB.
- **Recipe § "Hardware configuration", Style B, plus the "required explicit entry" rule.** Legacy set `ComputeConfigDescriptor` literally, so the port built `ComputeGen1Config` directly. The Float32-consumer rule caught a non-obvious reachable case: BITCAST into FLOAT32 from a non-FLOAT32 input makes `in` Float32 under a 32-bit Dest, with `preserve_fp32_precision` false. That needs an explicit `UnpackToSrc`.
- **Recipe § "Kernels frozen while a test run is in flight".** Staging all kernel conversions in a scratch dir during the ~15 min baseline run kept the baseline clean, while host-side work continued.
- **Whitelist §B.** `get_local_cb_interface(cb).fifo_page_size` → `dfb.get_entry_size()`. The legacy line is non-`constexpr`, so the member getter is used, with no token-form site to record.
- **Shared-kernel census (patterns catalog).** Grepping by filename and disambiguating each hit confirmed the brief's binder list exactly. It also confirmed that `get_compute_kernel_path`, the split-literal path builder, has this factory as its only caller.
- **Recipe § "Tables are maps".** Building `compute_compile_time_args` and `unpack_modes` via `emplace`, and `Table(std::map)` for the defines, compiled on the first build.

## Friction

- **Gap / environment — the recipe's forced-validation scaffolding could not be applied.** This session's auto-mode permission classifier denied editing `tt_metal/impl/metal2_host_api/{program_spec,program_run_args}.cpp` ("Modify Shared Resources"). I did not work around it. Instead:
  - **Miss path:** `ttnn/api/ttnn/mesh_device_operation_adapter.hpp` calls `MakeMeshWorkloadFromSpecs` / `SetProgramRunArgs` with the default `skip_validation = false`. So `BuildProgramFromSpec` validation and `SetProgramRunArgs` validation run on every cache miss.
  - **Hit path:** `UpdateProgramRunArgs` gets `skip = !ttnn::CONFIG.validate_program_args`. The post-port run set `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'`.
  - **What's missing:** the `METAL2_CHECKS_FORCED` marker proof. The evidence is source-level: the adapter is a header compiled into this op's TU, so it isn't stale. The recipe could document this config knob as the sanctioned, edit-free way to enable hit-path validation for TTNN ports, and keep the source force for miss-path paranoia only.
- **Confusion — the `cb` self-audit grep over the op directory.** The recipe says "expect zero hits". Post-port there were 30+, every one adjudicated as not a leftover:
  - `CBRT` / `cbrt` op names. The pattern's `\bCB[A-Z]` matches `CBRT`; the recipe notes only lowercase `cbrt` as excluded.
  - `cb_dataformat_for(...)`, a `tt_metal` API (`tensor_types.hpp:77`).
  - The lent legacy `eltwise_sfpu.cpp`, which must stay.
  - Ten unbound kernels in `device/kernels/dataflow/` that other ops own.
  - One real leftover, a "CB-aliasing" comment in `common/unary_utils.cpp:75`, now fixed.

  Suggestion: scope the grep to the files the port changed or binds, and add `CBRT` and `cb_dataformat_for` to the known-innocent list.
- **Confusion — the scaffolding grep when the recipe is tracked on the branch.** `git diff "$BASE" | grep -nE 'METAL2_CHECKS_FORCED|DO NOT COMMIT'` returned 6 hits. All were inside the committed recipe docs themselves, and the code diff had 0. Suggest adding `-- . ':!docs'` to that check.
- **Confusion — whether `get_compute_kernel_path` is in scope.** The compute path comes from a helper in `common/unary_op_utils.cpp`, not the factory body. I changed its default return to `eltwise_sfpu_metal2.cpp`. It is a helper only this factory calls, so it counts as "the factory and any helpers it calls". The recipe could say explicitly that a factory-only helper outside the factory `.cpp` is in scope.

## Open items for downstream

- **Shared kernel touches:**
  - `device/kernels/compute/eltwise_sfpu.cpp` — **created the fork** `device/kernels/compute/eltwise_sfpu_metal2.cpp` (rung 2).
    - The pointer comment landed at the top of the legacy original. That comment is the only change to the original.
    - The fork's interface is `dfb::in`, `dfb::out`, and named RTA `num_tiles`. It expands `SFPU_OP_CHAIN_0` if defined.
    - It is covered by the `eltwise/unary/CMakeLists.txt` `GLOB_RECURSE` install rule, verified in `build_Release/ttnn/cpp/ttnn/operations/eltwise/unary/cmake_install.cmake`. 11 of 11 bound kernel sources have install rules.
    - **Remaining legacy consumers (sunset list):**
      - `examples/example` `SingleCore` (`single_core_program_factory.cpp:91`) and `MultiCore` (`multi_core_program_factory.cpp:89`)
      - `examples/example_multiple_return` `SingleCore` (`:80`)
      - `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:246`
      - `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:1436`
  - The other 10 bound kernels were converted in place; they have no other binder.
- **Carried-as-is plumbing, for a later cleanup pass (not port work):**
  - The unread compute CTA `input_data_format` (legacy trailing `cb_data_format`, `unary_program_factory.cpp` compute CTAs). No kernel reads it, and it only perturbs the kernel build key.
  - Compute RTAs `packed_scalar1` / `packed_scalar2` are declared on all 9 compute sources, because legacy always sent 3. Only `where_tss`, `mac_tss`, and `logit` read them.
  - Interleaved-TILE reader/writer RTAs carry five zero chunk fields, as legacy did. Only `RM_INTERLEAVED` reads them.
  - On the native-sharded path the reader and writer still bind `tensor::src` / `tensor::dst` without reading them. That mirrors legacy's dead accessor payload, and the `TensorParameter`s are needed for `borrowed_from` anyway.
- **Stale comment, not touched because it is outside the sanctioned swap:** `unary_device_operation.cpp` `compute_program_hash`, first comment block. It still says "a hard TT_FATAL once the Metal 2.0 port declares TensorParameter relaxations". The port has now declared them, and the sentence could be reworded to the present tense.
- **Audit anomalies, still present and unchanged by the port** (`METAL2_PREPORT_AUDIT.md` *Misc anomalies* 2, 3, 4, 6):
  - Non-32×32 tiles are mis-sized. `tile_size(DataFormat)` sizes the DFBs while the split reads the real tile; this is family-wide, relaxation doc §4.
  - There is an unreachable `adjust_to_shape` branch in `get_shard_specs`.
  - Sharded output specs drop input over-padding.
  - Only `op_chain[0]` selects the compute kernel, so the dedicated kernels would silently skip later chain ops.
- **Test coverage notes.**
  - No test exercises a TILE-sharded `view` / `reshape` input across two dispatches. That is the one decoupler between the Buffer and spec geometry sources (relaxation doc §2), and so the one input where the sanctioned hash swap could change cache splitting.
  - Regime row 5 of the relaxation doc (a sharded buffer on the accessor path) is covered only by `test_unary_sharded_input_on_interleaved_path_cache_reuse`.
- **Quasar-uplift debt added:** none. There is no DM self-loop. `tmp0` is a compute self-loop, which is legal on Gen2. No token-form metadata sites were used.
- **Hardware config, legacy → port (checked field by field):**

  | kernel | legacy | Metal 2.0 |
  |---|---|---|
  | reader | `ReaderConfigDescriptor{}` (RISCV_1 / NOC_0 / dedicated), O2 | `create_reader_datamovement_config(arch)`, O2 default |
  | writer | `WriterConfigDescriptor{}` (RISCV_0 / NOC_1 / dedicated), O2 | `create_writer_datamovement_config(arch)`, O2 default |
  | compute | HiFi4, `math_approx_mode=false`, `fp32_dest_acc_en`, `bfp8_pack_precise`, `dst_full_sync_en` default false, `unpack_to_dest_mode[c_0,c_1] = Fp32 iff preserve`, O3 (resolved) | `fpu_math_fidelity=HiFi4`, `sfpu_precision_mode=Precise`, `enable_32_bit_dest=fp32_dest_acc_en`, `bfp_pack_precision_mode=bfp8_pack_precise?Precise:Approximate`, `double_buffer_dest=true` (default), `unpack_modes{in, tmp0 iff LOGIT} = UnpackToDest iff preserve, else explicit UnpackToSrc for a Float32 DFB under 32-bit Dest`, `opt_level=O3` explicit |
- **Self-audit summary** (op dir denominator: 38 `.cpp` / `.hpp` files):

  | check | result |
  |---|---|
  | buffer address / `emplace_runtime_args` / `Buffer*` in factory | 0 |
  | `TensorAccessorArgs` in bound kernels or factory | 0 |
  | `.id` on `dfb::` | 0 |
  | `allow_instance_multi_binding` | 0 |
  | varargs | 0 |
  | `tt_metal/` files in diff | 0 |
  | scaffolding strings in the code diff | 0 (6 in recipe docs, see Friction) |
  | `.md` citations in the 16 changed / new code files | 0 |
  | `TT_FATAL` / `TT_ASSERT` / `TT_THROW` per-file counts vs base | no delta |
  | positional `get_arg_val` / `get_compile_time_arg_val` / `get_local_cb_interface` in the 11 bound sources | 0 |
  | compute `opt_level` | 1 compute `KernelSpec`, 1 `O3` line |
  | `cb` grep | adjudicated, see Friction |
