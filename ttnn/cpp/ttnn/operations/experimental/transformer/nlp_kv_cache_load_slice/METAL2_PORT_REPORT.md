# Metal 2.0 Port Report — `experimental/transformer/nlp_kv_cache_load_slice`

## Outcome

**PORTED.** The single factory (`create_descriptor` on the device op → `NlpKVCacheLoadSliceProgramFactory::create_program_artifacts`)
is on `ProgramSpecFactoryConcept`. `test_nlp_kv_cache_load_slice.py` passes 54/54 (bfloat8_b / bfloat16 / float32 ×
18 shapes, including the in-test program-cache checks), with Watcher on and the Metal 2.0 legality checks forced.
Both `METAL2_CHECKS_FORCED` markers (`program_spec.cpp`, `program_run_args.cpp`) were present in the run log.
The legacy baseline on the same card, before the port, was also 54/54.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
  (from the `/localdev/edwinlee/Port_Recipe` checkout; this workspace has no copy of the docs).
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **TTNN gate waiver (inherited from the audit):** the readiness-sheet row for this op is stale (pre-#57409:
  `Concept = legacy device-op`, phantom factory `NlpKVCacheLoadSliceProgramFactory`). The user waived the gate on
  2026-10-07 ("waive"). The sheet refresh is still owed by the sheet owner. Coincidentally, the port now *creates* a
  nested struct with that same name (exception 3 naming convention), so after this lands the row's factory name is
  right again, but its `Concept` and definition path (`…_program_factory.hpp`, deleted) are not.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. No `override_runtime_arguments` existed; the framework's tensor-binding
refresh covers the cache hit (input accessor address + output shard backing the borrowed DFB). The slice window is an
attribute, so per-core `start_id` RTAs are structural and correctly frozen at cache miss.

### Device-op-class edits
- **Exception 3 (direct-descriptor op):** the op arrived in the direct-descriptor shape (`create_descriptor` as a static
  on `NlpKVCacheLoadSliceDeviceOperation`, no `program_factory_t`). Added the nested `NlpKVCacheLoadSliceProgramFactory`
  with `create_program_artifacts` and `using program_factory_t = std::variant<NlpKVCacheLoadSliceProgramFactory>`
  (`device/nlp_kv_cache_load_slice_device_operation.hpp`). The cache-hit comment moved with the method; its "CB c_0"
  wording was updated to the borrowed output DFB. Includes: dropped `program_descriptors.hpp`, added `<variant>` and
  `ttnn/metal_v2_artifacts.hpp`. Nothing else in the class changed.
- Pybind entry points removed: none (the nanobind file binds only the user function).
- Custom `compute_program_hash`: none.

### Open items
- Relaxation candidates: none considered (strict matching kept, per recipe).

## Handoff points

- **Direct-descriptor → program factory (exception 3)** — recorded above; structural but mechanical.
- No capitulations, boundary-rule violations, kernel-lib gaps, or framework gaps.

## Successes

- **Whitelist §A, `constexpr` metadata rule.** The reader declared `constexpr uint32_t tile_size = get_tile_size(cb_id_in0)`
  and feeds it to the `get_barrier_read_threshold<tile_size, num_readers>()` template. The whitelist's explicit
  "the legacy declaration is the entire test" rule (and its example, which is this exact barrier helper) resolved the
  brief's "confirm which form compiles" question up front: `get_tile_size(dfb::in0)` (reader `:30`). It compiled first time.
  **Gen1-only token-form site, recorded per §A:** reader `get_tile_size(dfb::in0)` — Quasar-uplift debt.
- **Shared-kernel rung 1.** The brief named the existing fork `writer_unary_sharded_metal2.cpp` and its vocabulary
  (`dfb::out`, `args::num_units`); the plan just adopted it. No writes outside the op directory.
- **Self-audit "print the denominator" + `emplace_runtime_args` search.** The legacy address arrived as a `Buffer*`
  through `emplace_runtime_args`, never as `->address()`; the recipe's explicit second search is what would catch a miss here.
- **Borrow-only `TensorParameter`.** The migration guide's exemption (a parameter named only by `borrowed_from` needs no
  kernel binding) answered the one spec-validity question the output tensor raised.

## Friction

- **Gap — DM helper signature.** The recipe's Hardware configuration section shows
  `create_reader_datamovement_config(device->arch())`, but the helper in
  `ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp:24,32` takes no arch
  (its only parameter is `bool disable_dfb_implicit_sync_for_all = false`). Used `create_reader_datamovement_config()` /
  `create_writer_datamovement_config()`. The recipe snippet should drop the arch argument.
- **Confusion (minor) — DFB spec name vs. per-kernel accessor name.** The reader's vocabulary (`cb_id_in0`) and the
  borrowed writer fork's (`dfb::out`) name the same buffer differently. The resolution (one spec `out`, accessor `in0` on
  the reader and `out` on the writer) is legal because `accessor_name` is per binding, but neither the recipe nor the
  catalog says outright that a fork's vocabulary constrains only *its* accessor name, not the spec name. One sentence
  in the shared-kernel Caution would help.
- **Workflow (not a doc issue).** Another port session (segformer) had uncommitted edits in the shared main checkout
  during this port's build, so the build compiled their in-progress tree too. It built clean, but a broken sibling WIP
  would have blocked verification here.

## Open items for downstream

- **Shared kernel touches.**
  - (a) `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp`;
    (b) **reused the existing `_metal2` fork** `writer_unary_sharded_metal2.cpp` (no new file, original untouched);
    (c) remaining legacy binders of the original: `data_movement/sharded_partial/interleaved_to_sharded_partial`,
    `data_movement/untilize` (`untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical` factory),
    `experimental/padded_slice` (`padded_slice_rm`).
  - The reader is op-owned and was converted in place.
- **Pre-existing anomalies, preserved unchanged (from the audit, confirmed while porting):**
  - `memory_config` is accepted and ignored (`device/nlp_kv_cache_load_slice_device_operation.cpp:121`); output is always HEIGHT_SHARDED L1.
  - `create_output_tensors` returns `preallocated_output` unvalidated (`device_operation.cpp:107-108`); the factory then
    assumes a shard spec whose grid matches `fused_batch_heads`.
  - RTA core coords use `num_cores_x` from only the first range of the shard grid (`program_factory.cpp`, `core_range` /
    `num_cores_x` and the per-core loop). Correct for the grid the op builds itself, wrong for a preallocated output
    whose first range is narrower.
  - Stale comment `// This should allocate a DRAM buffer on the device` above the shard-spec lookup in the factory
    (nothing is allocated there). Left as-is.
- **Tidy-up for a later pass (not port work):** the writer's `num_units` RTA is the same value on every node, so it is
  really a CRTA; left as an RTA (RTA→CRTA changes dispatch).
- **Test coverage.** The only test is `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_kv_cache_load_slice.py`
  (no gtests, no sweeps, no model callers found by `grep -rn nlp_kv_cache_load_slice tests models`). I took it as the
  baseline without a separate confirmation round-trip. Its largest grid is 32 cores, which fits the 8×8 Wormhole here.
  The preallocated-output path (`output_tensor=`) has no coverage.
