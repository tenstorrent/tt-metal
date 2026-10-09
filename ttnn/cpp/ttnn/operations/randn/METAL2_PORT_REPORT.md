# Metal 2.0 Port Report — `randn`

## Outcome

**PORTED.** The single factory `RandnDeviceOperation::ProgramFactory` now targets `CustomProgramSpecFactoryConcept`. Both kernels were converted in place. The outputs are bit-exact against the legacy op on cache misses and on every kind of cache hit. All 82 `test_randn.py` tests pass under Watcher, with the Metal 2.0 legality checks forced on.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(from the `Port_Recipe` checkout. `/localdev/edwinlee/metal2_port.md` symlinks there; the Metal_Ports checkout carries no recipe docs.)*
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit gate waiver:** the TTNN-factory-concept gate was RED only because the readiness-sheet row is stale after PD batch #57409 (`Concept` = `legacy device-op`, `Override runtime args method?` = `n/a`). The user waived it on 2026-10-07 ("Waive"). The sheet refresh (`Concept` = `descriptor`, override = `yes`) is still owed to the readiness-sheet owner. The target concept came from the audit's code reading, and the port confirmed it: the factory carries an override, and the override's seed redraw is behavior the base concept cannot express.

## TTNN ProgramFactory

### Concept realized
`CustomProgramSpecFactoryConcept`, as the audit chose.

`override_runtime_arguments` now returns `tt::tt_metal::experimental::ProgramRunArgs`, with no `Program&` parameter (`device/randn_device_operation.hpp`). It mirrors the legacy refresh set exactly, and nothing else:

| Legacy override (`randn_program_factory.cpp:146-172` @ `a41252b877b`) | Metal 2.0 override |
|---|---|
| Compute RTA 0 (`seed`) per core: one draw per core in `randn_core(i, grid)` order, from `seeded_rng(seed)` or the persistent `default_rng` | `KernelRunArgs{COMPUTE}.runtime_arg_values["seed"][core]`, same draw order, same RNG selection (a partial update) |
| Writer RTA 0 = `output.buffer()->address()` | `tensor_args = {{OUTPUT, output.mesh_tensor()}}` |
| (`start_id`, `num_tiles` untouched) | (untouched; partial update leaves them as set at cache miss) |

Every io-tensor `TensorParameter` gets a `TensorArgument`. There is only one (`output`); the op has no inputs and no op-owned tensors. The legacy override held no guards, so no TT_FATAL had to move.

### Device-op-class edits
- **Pybind entry points removed:** none. `randn_nanobind.cpp` binds only the function.
- **Custom `compute_program_hash`:** left intact and untouched at `device/randn_device_operation.cpp:47-52`. It hashes everything except `seed`. The `TensorSpec` legality check passed on every cache hit.
- **Header edits:** only the two `ProgramFactory` method signatures changed, which is the concept's required shape. The include swapped `program_descriptors.hpp` → `metal_v2_artifacts.hpp` + `program_run_args.hpp`. The struct, `program_factory_t`, and everything else stay as they were.

### Open items
- **Relaxation candidates:** none.

## Handoff points

none

## Successes

- **[Translating `override_runtime_arguments`](metal2_port.md#translating-override_runtime_arguments-custom-concept-only)** mapped cleanly. Its three-bucket inventory (per-node RTA / CRTA / address → `TensorArgument`) covered both legacy writes. The "refresh exactly that set" rule kept `start_id` and `num_tiles` out of the override, and the `UpdateProgramRunArgs` partial-update contract in `program_run_args.hpp:62-64` made a seed-only update legal.
- **[The override owns the tensor bindings too](ttnn_factory.md#the-override-owns-the-tensor-bindings-too)** flagged the silent failure ahead of time. Without `params.tensor_args`, the output binding would stay frozen at its cache-miss address. The harness shifts allocations between calls, and all 18 cache-hit outputs are bit-exact, so the binding refresh is exercised.
- **The [Compute config](metal2_port.md#compute-kernels) "dropped/forced field" check** caught that legacy forces `fp32_dest_acc_en = true` regardless of the caller's config. The port uses the Style A helper and then re-forces `enable_32_bit_dest = true`. The helper's other three fields are exactly the subset legacy copied. `packer_l1_acc` has no counterpart.
- **[Compiler options](metal2_port.md#compiler-options):** explicit O3 on compute (`randn_program_factory.cpp`, `.opt_level`).
- **Taking the Watcher baseline before the port** (memory from earlier ports) was worth it. Legacy is Watcher-clean (82/82), so the post-port Watcher run is a real like-for-like comparison.

## Friction

### Gaps
- **The recipe's hw_config type names are stale** (same as on earlier ports). The recipe uses `ComputeGen1Config` / `std::get<ComputeGen1Config>(…)`, but the headers have a flat `ComputeHardwareConfig`. Also, `ttnn::to_compute_hardware_config` takes only the config, not `(arch, config)`. The fix to the returned config was a plain field assignment (`compute_hw_config.enable_32_bit_dest = true`).
- **The recipe doesn't say whether a partial `KernelRunArgs` is allowed in the override.** `ttnn_factory.md` says `UpdateProgramRunArgs` is "an arbitrary partial update" at the `ProgramRunArgs` level. Whether a *subset of one kernel's RTA names* (here `seed` but not `num_tiles`) is accepted was only answered by the header comment (`program_run_args.hpp:62-64`). A one-line example in the override section would remove the doubt.

### Confusion
- none

## Open items for downstream

- **Shared kernel touches:** none. Both kernels are op-owned and were converted in place.
- **Carried-over anomalies** (from the audit, unchanged and not port work):
  - The file-scope `distribution` / `default_rng` globals are not thread-safe.
  - `fp32_dest_acc_en` is forced on, but the full `compute_kernel_config` is still hashed, so configs that build the same program can get different cache entries.
  - The hash includes a raw `MeshDevice*`.
- **Later cleanup (not port work):** the override walks cores with `randn_num_cores` / `randn_core`, so it depends on the `TT_FATAL` that ties them to `split_work_to_cores`. That coupling is unchanged.
- **Test set:** not confirmed with the invoker. The sweep found:
  - `tests/ttnn/nightly/unit_tests/operations/rand/test_randn.py` (primary)
  - two incidental users that don't exercise the op's paths beyond the above: `tests/ttnn/docs_examples/test_tensor_creation_examples.py` and `tests/ttnn/unit_tests/base_functionality/test_comparison_mode_context.py`
  - `test_rand.py` is a different op

## Verification

- **Build:** `./build_metal.sh --build-tests` (Release), success.
- **Legality checks:** all `skip_validation` sites forced, using uncommitted scaffolding already in the main checkout. Both markers were present:
  - harness: `program_spec.cpp` ×6, `program_run_args.cpp` ×6, one per built program (6 cases → 6 cache misses);
  - pytest: ×91 each.
- **Watcher:** `TT_METAL_WATCHER=10` for the baseline and the post-port runs. No asserts.
- **Pre-port baseline (legacy):** `test_randn.py` 82 passed. The harness saved 24 outputs.
- **Post-port:**
  - `test_randn.py` **82 passed**. That includes `test_randn_callback` (cache-entry count stable across seeds), `test_randn_unseeded_calls_differ`, and `test_randn_same_seed_is_reproducible`.
  - Scratch harness: **24/24 bit-exact vs legacy.** It ran 6 cases (BF16/FP32; TILE/ROW_MAJOR; DRAM/L1; 1 to 256 tiles, so both one-group and two-group splits; an odd tile count). Each case made 4 calls: seed 1234 (miss), 1234 (hit), 99 (hit), 1234 (hit), with allocations shifted between calls. An unseeded repeat still differs.
  - The harness (`/home/edwinlee/.claude/jobs/daa75c58/tmp/randn_harness.py`) is job scratch and is not committed.
- **Self-audit** over the 9 `.cpp`/`.hpp` files:
  - zero `cb` names, `CircularBuffer`/`CBDescriptor`/`ProgramDescriptor`, `Buffer*`/address RTAs, `TensorAccessorArgs`, positional args, `get_cb_id`/`.id`, or `.md` citations;
  - TT_FATAL/TT_THROW counts unchanged per file;
  - O3 on the one compute spec;
  - no `tt_metal/` files in the diff.
- **hw_config vs legacy:**
  - writer = writer default (RISCV_0 / NOC_1 / dedicated), same as legacy `WriterConfigDescriptor{}`;
  - compute fidelity, approx, and `dst_full_sync_en` come from the config, with 32-bit dest forced as before;
  - `bfp_pack_precision_mode` is at its default, as before;
  - no `unpack_modes` needed (compute consumes no DFB).
