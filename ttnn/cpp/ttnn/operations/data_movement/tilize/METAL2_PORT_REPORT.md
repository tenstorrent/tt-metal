# Metal 2.0 Port Report — `data_movement/tilize` (`TilizeMultiCoreBlockProgramFactory`)

## Outcome

`PORTED` — the one remaining tilize factory, `TilizeMultiCoreBlockProgramFactory`, converted to
`CustomProgramSpecFactoryConcept`. The five sibling factories were already ported (PR #54805); the op is
now fully on Metal 2.0.

**Verified (Wormhole n150, `wormhole_b0`), all with `METAL2_CHECKS_FORCED` live (both choke-point TUs) and
`TT_METAL_WATCHER=10`:**
- Build: clean (`build_metal.sh --build-tests`, 0 errors).
- C++ gtests: `unit_tests_ttnn --gtest_filter='*Tilize*:*tilize*'` → 5/5 passed (incl. `test_work_split_tilize`).
- Pytests (confirmed baseline, each its own invocation): `test_tilize.py` 457 passed / 93 skipped;
  `test_tilizer.py` 1 passed; `test_tilize_test.py` 22 passed; `test_tilize_pad_cb.py` 12 passed. **0 failed.**
- Block factory positively exercised with checks forced: `test_run_tilize_large_row_input[(32,15936)]` and
  `[(160,5210112)]`, the bfloat4_b `deep_seek…large_number_of_pages_per_row[(1,7168,2304)]`, and
  `test_tilize_block_two_pair_program_cache_addr_change[(1,1,32,7328)]` (the two-buffer-set case **and** the
  custom-concept cache-hit `override`) — all PASSED.
- Note: `test_tilize_pad_cb.py` drives `ttnn.tilize_with_zero_padding` → the `tilize_with_val_padding` op
  (legacy `descriptor`, **not** ported here), so it shows 0 `METAL2_CHECKS_FORCED` markers as expected. Its
  pass confirms the rung-2 forks + pointer comments left the legacy kernel originals — which
  `tilize_with_val_padding`'s block factory still binds — undisturbed.

## Provenance

- **Recipe docs (this port):** `bd9e9f36292 2026-09-23 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `bd9e9f36292 2026-09-23 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## TTNN ProgramFactory

- **Concept realized:** `CustomProgramSpecFactoryConcept` (`create_program_artifacts` returns
  `ProgramArtifacts`; `override_runtime_arguments` returns `ProgramRunArgs`). Matches the audit's decision
  and the five ported siblings.
- **Custom `compute_program_hash`:** none — nothing to preserve.
- **Pybind entry points removed:** none — nanobind binds `ttnn::tilize` only; there was no pybound
  `create_descriptor`.
- **Device-op-class edit forced:** removed the now-dead `patch_tilize_kernel_slot0`
  (`tilize_device_operation.hpp` decl + `tilize_device_operation.cpp` def). This factory was its only
  caller; the five ported siblings reference it in comments only.
- **Open items:** none for the concept fit.

## Handoff points

None. No capitulation, no out-of-op-directory change required (the three `_metal2` forks are the sanctioned
shared-kernel carve-out; see Open items), no `sem::`/`tensor::` boundary violation, and no pybind surface
removed (nanobind binds `ttnn::tilize` only — there was no `create_descriptor` pybind).

## Successes

- **The `BlockBufferSet` model + the untilize block precedent made this near-mechanical.** #51305's
  per-width buffer sets (`data_movement/common/common.cpp`) gave each named DFB a single uniform size, and
  the already-merged inverse-op block factory
  (`untilize/device/factories/untilize_multi_core_block_program_factory.cpp`, #56280) was a line-for-line
  structural template for the split, the per-set reader/writer, the disjoint-node work-split, and the
  runtime-arg loop. The only additions over that template were tilize-specific: the staging DFB + its
  self-loop, and the custom-concept `override`.
- **[Self-loop DFB binding] / [Sync-free and single-ended CBs → self-loop] fired exactly as written.** The
  reader's staging buffer (`c_1`/`c_3`) is a single-ended FIFO producer with no consumer and exactly one
  toucher → self-loop (reader bound PRODUCER + CONSUMER, shared accessor name `staging`). The one-toucher
  gate kept me from mis-reading it as anything needing the multi-binding flag.
- **[Two-toucher / preserved-multiplicity] guidance prevented over-flagging.** `IN_*`/`OUT_*` are each
  bound by two same-source compute KernelSpecs, but over **disjoint** sub-ranges (`core_range` vs
  `cliff_col`), so per node the census is 1P+1C. Listing the shared reader/writer in each work unit over
  disjoint `target_nodes` (the disjoint-node work-split) — not the multi-binding flag — is the correct shape,
  matching the untilize precedent.
- **Compiler-options rule caught the silent O2 default.** The legacy compute `ComputeConfigDescriptor` set
  no `opt_level` → resolves O3; Metal 2.0 defaults O2. Set `KernelBuildOptLevel::O3` on every compute
  `KernelSpec` (via the shared `add_compute_region` lambda).

## Friction

- **Confusion (minor) — Gen2 compute config, recipe vs. a sibling.** The recipe's *Hardware configuration*
  section says build only the Gen1 config and add no `if (arch == QUASAR)` branch. The most recent precedent
  (untilize block, #56280) follows that. But a *sibling* tilize factory
  (`tilize_multi_core_default_program_factory.cpp`) does add a manual `ComputeGen2Config` mirror branch
  (its comment cites TODO #52269). I followed the recipe + the untilize precedent (Gen1-only). If the
  intent is that ported factories should carry the manual Quasar mirror, the recipe and the default sibling
  disagree and the recipe should say so. No functional impact on the Gen1 (WH/BH) target.
- **Gap (small) — the `cb`-name self-audit flags stale data-format variable names.** The legacy factory's
  `input_cb_data_format` / `output_cb_data_format` are *data-format* variables, not CBs, but they trip the
  `cb`-sweep. Renamed to `input_data_format` / `output_data_format` (matching the untilize block factory and
  the prior tilize port). Worth a one-line note in the recipe's self-audit item that these `*_cb_data_format`
  locals are a common benign hit to rename.

## Open items for downstream

### Shared kernel touches (rung 2 — created the first `_metal2` fork of each)

All three kernels were already DFB-based; each fork is a named-binding conversion, not a CB→DFB rewrite.
Consumer set of each = {`data_movement/tilize` block (this port), `data_movement/tilize_with_val_padding`
block}. The **remaining unmigrated consumer after this port is `data_movement/tilize_with_val_padding`'s
block factory** (`tilize_with_val_padding_multi_core_block_interleaved_program_factory.cpp`, still on
`descriptor`). When it ports, it should reuse these forks (rung 1) and can then sunset the legacy copies.

| kernel (original) | fork created (beside original) | pointer comment added |
|---|---|---|
| `data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_multicore_both_dims.cpp` | `..._metal2.cpp` | yes |
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp` | `..._wh_metal2.cpp` | yes |
| `data_movement/tilize/device/kernels/compute/tilize_wh.cpp` | `tilize_wh_metal2.cpp` | yes |

Fork binding vocabulary (the interface the next consumer inherits):
- reader `_metal2`: `dfb::in` (input), `dfb::staging` (alignment scratch, self-loop), `tensor::src`;
  named CTAs {total_num_rows, third_dim, tile_height, element_size, unpadded_X_size, dram_alignment};
  named RTAs {pad_value, width_size, start_row_id, start_column_id, single_block_size_row_arg,
  single_block_size_col_arg, sub_block_width_size, single_sub_block_size_row_arg}.
- writer `_wh_metal2`: `dfb::out`, `tensor::dst`; named CTAs {num_tiles_per_2d, third_dim,
  total_tiles_per_row}; named RTAs {start_id, single_block_size_row_arg, single_block_size_col_arg}; keeps
  `#ifdef BACKWARDS`.
- compute `tilize_wh_metal2`: `dfb::in`, `dfb::out`; named CTAs {block_size_col, block_size_row, third_dim}.

### TT_FATAL census — subject-deleted guards (legitimate losses)

The block factory's guard count drops from 10 (BASE) to 5. The 5 that go away all validated the **legacy
manual cache-hit patch plumbing** — the `dm_kernel_metadata` common-args carrier and the
`patch_tilize_kernel_slot0` slot-0 re-point — which the `CustomProgramSpecFactoryConcept` `override`
replaces wholesale with the framework's tensor-binding refresh (`return ProgramRunArgs{.tensor_args=…}`).
Their subjects no longer exist, so this is the sanctioned subject-deleted loss, matching the translation the
five sibling factories made in PR #54805. The dropped guards were:
- `TT_FATAL(!desc.kernels.empty(), …)` — guarded the kernel list before the common-args metadata attach.
- `TT_FATAL(dm_kernel_handles.size() == 2 * plan.num_dm_pairs(), …)` — guarded the recorded DM-kernel handles.
- `TT_FATAL(num_pairs == 1 || num_pairs == 2, …)` (override) — guarded `dm_kernel_metadata[0]`.
- `TT_FATAL(dm_kernel_metadata.size() == 1 + 2*num_pairs, …)` (override) — guarded metadata consistency.
- `TT_FATAL(args.size() == expected_args, …)` (belt-and-braces width check before manual slot-0 patch).

The 5 retained guards are the real ones: output-buffer-allocated, `block_tiles > 0`, `single_block_size %
single_sub_block_size`, compute `block_size_row == set.block_tiles`, and per-core
`single_sub_block_size_row_arg == set.block_tiles`. Device-op files show no TT_FATAL delta (removing the dead
`patch_tilize_kernel_slot0`, which had none).

### Other

- **Device-op cleanup done in this change:** removed the now-dead `patch_tilize_kernel_slot0` decl + def from
  `tilize_device_operation.{hpp,cpp}` (this factory was its only caller; siblings reference it in comments
  only). No other device-op-class edits.
