# Port Report — create_qkv_heads_from_separate_tensors

## Outcome

**PORTED** — the op's single factory (`CreateQKVHeadsSeparateTensorsProgramFactory`, introduced by this port) is on `ProgramSpecFactoryConcept`, in both configurations (`transpose_k_heads` true / false). Verification: see [Verification](#verification).

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`. This line comes from the `Port_Recipe` checkout (branch `akertesz/op-porting-recipe`); the provenance command prints nothing in `Metal_Ports`.
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit waiver:** the TTNN factory concept gate was RED only because the readiness-sheet row is stale (PD batch #57409). The user waived it on 2026-10-06. Refreshing the sheet stays with the readiness-sheet owner.
- **Branch base:** `edwinlee/PD_Metal_Ports` at `b77c495b871`. No commit touched the op's `device/` or kernels after the audit (last op commit `35e83c6252f`, 2026-10-05; audit 2026-10-06).

## Verification

Hardware: one Wormhole b0, 8×8 grid. Every test grid (2×4 / 2×8 / 7×6 / 7×8) fits, so nothing skipped. All runs had `TT_METAL_WATCHER=10`. Metal 2.0 legality checks were forced on in the working tree (all 9 `skip_validation` sites) and confirmed live: `METAL2_CHECKS_FORCED` was logged from both `program_spec.cpp` and `program_run_args.cpp` on every program build. None of that scaffolding is in the commit.

- **Build:** `./build_metal.sh --build-tests` — success, no errors.
- **`test_create_qkv_heads.py::test_nlp_create_q_and_kv_heads_separate_test`** (the op's only on-device test): **16/16 passed**. That is 4 shapes × `transpose_k` {True, False} × {bf8b, bf16}, so both configs and both kernel sets ran.
- **`tests/ttnn/unit_tests/operations/transformers/test_head_count_zero.py`**: **22/22 passed** (validation paths, run in the same session as above — 38 passed total).
- **Cache-hit check** (throwaway test, not committed). It runs each of 4 configs ({bf16, bf8b} × transpose on/off) three times, with an extra allocation held between calls so the later tensors land at different addresses.
  - **4/4 passed.** Program-cache entry count is unchanged across the repeats, i.e. 8 cache hits through `UpdateTensorArgs` with validation forced.
  - PCC is identical on every repeat (1.0 bf16; ≈0.99997 bf8b), so the borrowed-DFB rebinding on a cache hit works.
- **Sibling sanity:** `test_create_qkv_heads.py::test_nlp_create_qkv_heads_test` (the `create_qkv_heads` op, which still binds the legacy `transpose_wh_sharded.cpp` that gained only a pointer comment): **42/42 passed**.
- **Test-set confirmation:** I picked the baseline myself because the invoker was not available live. It is the three files above, found by a full-tree search for the op name. Only the Stable Diffusion cross-attention demo also calls the op; it was not run.
- **Self-audit:** all checklist sweeps clean over 9 op files + the fork:
  - zero `cb`/`CircularBuffer`/`CBDescriptor` hits;
  - no address args, positional CTAs, varargs, `.id`, or multi-binding flags;
  - O3 on the single compute spec;
  - TT_FATAL counts unchanged;
  - no `.md` cited from code; no `tt_metal/` file in the commit.
- **`hw_config` diff vs legacy:**
  - reader: RISCV_1 / NOC_0 / dedicated (reader default) → `create_reader_datamovement_config()`, O2.
  - compute: HiFi4 / Precise / `enable_32_bit_dest = (kv dtype == FLOAT32)` / `double_buffer_dest = true` (legacy `dst_full_sync_en = false`) / bfp pack Approximate / no unpack-to-dest (an explicit `UnpackToSrc` entry for `k_pre_transpose` only when 32-bit Dest is on, as the validator requires), O3.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. No `override_runtime_arguments` existed or was added. All five io tensors are `TensorParameter`s with a `TensorArgument` each, so the framework refreshes them on a cache hit. They back borrowed DFBs, the replacement for the legacy CB `.buffer` re-pegging.

### Device-op-class edits
- **Direct-descriptor shape → nested factory (ttnn_factory exception 3).** The op arrived with `create_descriptor` as a static member of `CreateQKVHeadsSeparateTensorsDeviceOperation` and no `program_factory_t`. I re-checked this at port time. The port nests `CreateQKVHeadsSeparateTensorsProgramFactory::create_program_artifacts` and adds `using program_factory_t = std::variant<CreateQKVHeadsSeparateTensorsProgramFactory>;`. It removes `create_descriptor` and swaps the `program_descriptors.hpp` include for `ttnn/metal_v2_artifacts.hpp` + `<variant>` (`device/create_qkv_heads_from_separate_tensors_device_operation.hpp:13-34`). The device-op comment above the factory was retargeted from "CB `.buffer` bindings" to "tensor bindings"; its meaning is unchanged. Nothing else in the device-op class changed.
- Pybind entry points removed: none (`create_descriptor` was never pybound).
- Custom `compute_program_hash`: none.

### Open items
- Relaxation candidates: none (strict matching kept).
- No capability gaps on this concept.

## Handoff points

- **Device-op restructure (exception 3)** — see above. A structural but mechanical edit: the op now has the same shape as every other ported op. Owner: TTNN, for awareness only.
- No capitulations, no boundary-rule violations, no kernel-lib gaps.

## Successes

- **CB→DFB whitelist §A, `constexpr` metadata.** The brief said to "confirm the DFB getter is usable in a constant expression, else demote to `const`". The whitelist settles it the other way: keep `constexpr` and use the token form. `constexpr uint32_t single_tile_size_bytes = get_tile_size(dfb_inq);` (`reader_create_qkv_heads_sharded_separate.cpp:42`). Without that section I would have followed the brief and demoted four derived `constexpr`s too. **Quasar debt** (token form is Gen1-only): this is the one site.
- **Sync-free / single-ended → self-loop pattern.** It mapped every borrowed buffer directly:
  - `in_q`, `in_kv`: sync-free raw read-ptr peeks.
  - `out_q`, `out_v`: reader-produced, never drained.
  - `out_k`: reader-produced without transpose; packer-produced on compute under transpose.

  The "classify per instantiation" note covered `out_k`, whose sole toucher flips between kernels with config.
- **Shared-kernel caution, rung 2.** Census clean. The brief's warning not to reuse `data_movement/transpose/.../transpose_wh_sharded_metal2.cpp` (a different kernel with the same stem) prevented a wrong rung-1 reuse.
- **Compiler options.** The explicit O3 on the compute spec (`program_factory.cpp:199`) comes straight from the recipe's table. Legacy set no `opt_level`.

## Friction

### Gaps
- **Recipe hw-config spellings don't match the tree.** The recipe names `DataMovementGen1Config`, `ComputeGen1Config`, `std::get<ComputeGen1Config>(…)` and `create_reader_datamovement_config(device->arch())`. In this tree:
  - `ComputeHardwareConfig` is a flat struct (common fields + optional `config_1xx` / `config_2xx`), not a variant.
  - `DataMovementHardwareConfig` likewise has `config_1xx` / `config_2xx`.
  - The TTNN DM helper takes no arch argument (only `disable_dfb_implicit_sync_for_all`).

  Values mapped 1:1 (`enable_32_bit_dest` ← `fp32_dest_acc_en`; everything else at defaults, which match `ComputeConfigDescriptor{}`), but every porter will stop to re-derive this from the headers.
- **Background-job worktree isolation vs. building.** The harness requires edits in a worktree, but a cold tt-metal build in a fresh worktree on this 6-core box would take hours. The only built tree is the main checkout. Workflow used:
  - edit and commit in the worktree;
  - leave the worktree and cherry-pick onto the branch in the main checkout;
  - build and test there (with its pre-existing forced-validation scaffolding).

  It works, but the harness also blocks `git -C` into the main checkout from inside the worktree, so a fix found during testing means re-entering the worktree. Worth a note in workspace_setup for bulk-port jobs.

### Confusion
- **Brief vs. whitelist on `constexpr get_tile_size`.** The audit brief offered demoting to `const` as the fallback, which the whitelist explicitly forbids. The audit template should point at whitelist §A instead.
- **Precedent contradicting the recipe.** `data_movement/sharded/.../reader_unary_sharded_blocks_interleaved_start_id_metal2.cpp:48` avoids `get_tile_size(dfb::in)` and threads tile bytes through a new CTA, citing Quasar portability. That contradicts whitelist §A (keep the token form, record the Quasar debt). I followed the recipe. Flagging it so doc owners can settle one rule.

## Open items for downstream

- **Shared kernel touch — created fork (rung 2).**
  - (a) `experimental/transformer/split_query_key_value_and_split_heads/device/kernels/compute/transpose_wh_sharded.cpp`
  - (b) Created `transpose_wh_sharded_metal2.cpp` beside it. The pointer comment landed at the top of the original. Fork vocabulary: `dfb::in` (consumer, tiles to transpose), `dfb::out` (producer), named CTA `num_tiles`. No `#ifdef`s.
  - (c) Remaining unmigrated consumers: `experimental/transformer/split_query_key_value_and_split_heads` (sharded factory, `split_query_key_value_and_split_heads_sharded_program_factory.cpp:114-116`) and `experimental/transformer/create_qkv_heads` (`create_qkv_heads_program_factory.cpp:167-169`).
- **Preserved bugs / anomalies (from the audit, re-confirmed, untouched):**
  - Validation computes `q_shard_ht` with `num_w_cores` while the factory uses `num_h_cores` (`device_operation.cpp:127` vs `program_factory.cpp:64`).
  - The L1 check counts borrowed buffers as extra L1 (`device_operation.cpp:142-143`).
  - `optional_output_tensors` are not validated against `compute_output_specs` (`device_operation.cpp:222-225`).
  - Reader CTA comments are wrong (`reader :15-21`, e.g. `q_shard_ht` commented "number of Q heads in the group"). Preserved verbatim per comment rule 8. The host names are correct and now appear as the named-CTA keys.
  - Every buffer's page size is the **Q**-format tile size, including KV-format buffers (`program_factory.cpp`, `single_tile_size`). Carried verbatim as `entry_size`. Harmless when Q and KV share a dtype. Flagging for the owner.
- **Test coverage.** Only `test_nlp_create_q_and_kv_heads_separate_test` exercises this op on device: bf8b/bf16 only, no FLOAT32. So the new `unpack_modes = {k_pre_transpose: UnpackToSrc}` entry (required under FP32 + transpose) is not exercised by any test. `test_head_count_zero.py` covers validation only.
- **Quasar uplift notes:**
  - The reader self-loops four or five borrowed DFBs (DM self-loops are rejected on Gen2).
  - One `get_tile_size(dfb::…)` token-form site in the reader.
