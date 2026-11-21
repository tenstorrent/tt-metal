# prepared_program (ttnn.bringup.prepare_program / run_prepared; Python ttnn.bringup.prepared_program.generic_op)

- Source: ttnn/ttnn/bringup/prepared_program on mstaletovic/combine-lean-sender-llk @ 856d97d5864 (Milos Staletovic;
  added in d95eb046e13, changed in c6c1888efba), carried as is with Marko's MoE block (2026-10-27).
- New op, no TTNN source on this tree. Uses only the descriptor-era cache-hit hook every factory has
  (`override_runtime_arguments`, ttnn/api/ttnn/mesh_device_operation_adapter.hpp) and generic_op's
  `compute_program_descriptor_hash`; it does NOT need Marko's by-reference generic_op change (1bc1002c13d), which is
  not carried (owner decision D9, reports/moe_block_decisions_2026-10-27.md).

## Changes

imported, unchanged (C++ and Python as on 856d97d5864). The registration in `bringup_nanobind.cpp` sits after the
fork_op.py markers (a new op, not a fork), as on Marko's branch.

### Carried for DeepSeek-V4.1's MoE block A/B (2026-10-27)
- What: the folder, its two lines in `ttnn/ttnn/bringup/bringup_nanobind.cpp`, README.md, this file, the INDEX.md
  entry. No behaviour change: no existing caller.
- Why: Marko's token-split exchange (`models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py`) imports
  `ttnn.bringup.prepared_program.generic_op` at module load; the MoE block's fused head and fused tail call
  `prepare_program` / `run_prepared` directly.
- Needed by: deepseek_v41_flash_d_p, MOE_IMPL=block (models/demos/deepseek_v41_flash_d_p/tt/MOE_BLOCK.md)
- Files: all

### value_slots: per-run plain runtime args (2026-10-29)
- What: `prepare_program(desc, io, value_slots=[(kernel, arg)])` / `run_prepared(p, io, values=[...])`: the named
  runtime arg of a kernel, on every core that has it, is patched from the run's values on every call (create_descriptor
  and override_runtime_arguments); a value slot is never matched to an io address; the cache key hashes a marker for
  it instead of its value. `PreparedProgram.n_values`. Default (no value slots): the old behaviour and the old cache
  key (the hash walk is unchanged when no slot is named).
- Why: ttnn.bringup.ckv_bfp4's writer takes row0 (the table row it writes at), which changes every call; without a
  value slot each call needed a new descriptor (DS41-135: 1.33 ms host a call on 8x4).
- Needed by: deepseek_v41_flash_d_p P.7 (CKV_BFP4_PREPARED)
- Files: prepared_program.hpp, prepared_program.cpp, prepared_program_nanobind.cpp,
  device/prepared_program_device_operation.hpp, device/prepared_program_device_operation.cpp, README.md
- Tests: ckv_bfp4_ttnn/tests/unit/test_ckv_bfp4.py::test_ckv_bfp4_prepared (value slot, moved tensors, byte-equal to
  generic_op). The option-off regression (tests/test_prepared_program.py) has only 2x2 cases: skipped on the 8x4
  Galaxy (2 skipped, as before); the option-off users (fabric_reduce_scatter prepared, TailA2A prepared) run in the
  model's ladder.

### value_slots on common runtime args (deepseek_v41_flash_d_p MB.4, 2026-11-21)
- What: a value slot whose runtime-arg index has bit 31 set (`(kernel, 0x80000000 | i)`) names common runtime arg i of
  the kernel: written from the run's values once a kernel a mesh program (create_descriptor and
  override_runtime_arguments), never matched to an io address, its marker (3, slot) hashed instead of its value. New
  `CommonValPatch` / `Prepared::cval`; patch_count includes them. Default (no such slot): the old programs, patches and
  cache key (the hash walk adds nothing when no common value slot is named).
- Why: ttnn.bringup.ckv_bfp4_rows takes three per-run values on every core; as per-core value slots on 110 cores x 32
  chips one call cost 0.31 ms host in run_prepared alone (8x4, tests/perf/test_ckv_rows_host.py in
  deepseek_v41_flash_d_p), more than the slice + tilize + ckv_bfp4 it replaces (0.18 ms). Common args: one word a
  kernel a chip.
- Needed by: deepseek_v41_flash_d_p MB.4
- Files: prepared_program.hpp, prepared_program.cpp, prepared_program_nanobind.cpp,
  device/prepared_program_device_operation.cpp, README.md
- Tests: ckv_bfp4_ttnn/tests/unit/test_ckv_bfp4_rows.py (common value slots, moved values across 4 calls on one
  prepared program, byte-exact). The option-off regression (tests/test_prepared_program.py) has 2x2 cases only
  (skipped on the 8x4); the option-off users (ckv_bfp4 prepared, page copies, fabric_reduce_scatter, TailA2A) run in
  MB.4's ladder and no-rebuild check.
