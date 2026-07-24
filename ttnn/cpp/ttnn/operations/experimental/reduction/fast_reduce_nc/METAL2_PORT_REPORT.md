# Metal 2.0 Port Report — `fast_reduce_nc`

## Outcome

**PORTED** — the single `FastReduceNCProgramFactory` (reader + writer + compute, `descriptor` →
`MetalV2FactoryConcept`) converted. Build/test verification is the orchestrator's (this porter did
not build or run tests, per orchestration constraints).

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — matches the audit. `FastReduceNCProgramFactory::create_program_artifacts`
returns `ProgramArtifacts{.spec, .run_params}` (no op-owned tensors). Single program, single
factory variant; the device-op `program_factory_t` variant is unchanged.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op had no custom hash).
- Pybind entry points removed: none (`fast_reduce_nc_nanobind.cpp` binds the user-facing function
  `bind_function<"fast_reduce_nc">`, not a factory `create_descriptor`; no host-wrapper references
  the factory entry point).

### Open items
- **Relaxation candidates:** none. Both tensors bind with strict `TensorSpec` matching (no
  `ArgConfig::Runtime*` in the kernels).

## Handoff points

None. The port stayed entirely within the op directory. The one out-of-directory dependency —
`ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` (`dataflow_kernel_lib::prepare_zero_tile<uint32_t dfb_id>()`,
called by the reader) — took a `dfb::name` token directly through its `uint32_t` NTTP via the
constexpr cast (as the audit predicted). No donor edit was needed and none was made.

## Successes

- **Anti-pattern "Demoting per-group CTA to RTA" (`port_patterns.md`) fired correctly.** The legacy
  factory emits two compute `KernelDescriptor`s (core_group_1 / core_group_2) whose only difference
  is the per-group `num_cols_per_core_group` CTA. The catalog's explicit "two KernelSpecs, two
  WorkUnitSpecs, disjoint node sets, 1:1 DFB bindings (not the multi-binding flag)" recipe mapped
  the split 1:1 (`fast_reduce_nc_program_factory.cpp`, `make_compute` + `wu_g1`/`wu_g2`). Without
  that section the reflex would have been a single compute KernelSpec with `num_output_tiles`
  demoted to an RTA.
- **Dead-CB drop (brief + `CB endpoints`).** c_24 ("accumulated sum") was allocated but touched by
  no kernel (compute accumulates in DST). Dropped the `DataflowBufferSpec` cleanly; the brief's
  census made this a mechanical, zero-behavior removal.
- **`dfb::name` → LLK / kernel-lib pass-through** worked verbatim for `binary_op_init_common`,
  `add_tiles_init`, `reconfig_data_format`, `add_tiles`, `pack_reconfig_data_format`, `pack_tile`
  (compute) and `prepare_zero_tile<dfb::in1>()` (reader) — no `.id` extraction, no temp wrappers.

## Friction

### Gaps
- **DM-kernel tile-size getter is `get_entry_size()`, not the whitelist's `get_tile_size()`.**
  Kernel-side whitelist rule 7 / §A maps `get_tile_size(cb_id)` → `dfb.get_tile_size()`, but that
  member is documented as available only when `DFB_DESCRIPTORS_DEFINED` (compute-side
  `chlkc_descriptors.h`). The legacy reader/writer are **DM** kernels calling the free
  `get_tile_size(cb_id)`; on a DFB the DM-available equivalent is `get_entry_size()` (the value we
  set as `entry_size`). Followed the proven transpose-port precedent (its ported reader/writer use
  `dfb.get_entry_size()`) rather than the whitelist's compute-oriented mapping. Worth a whitelist
  note distinguishing the DM (`get_entry_size()`) from the compute (`get_tile_size()`) case.

### Confusion
- **Reference-port state on-branch.** The recipe steers away from the `experimental/quasar/*`
  pre-completion ports as templates. The freshest recipe-following example was the transpose port,
  which lives on a *descendant* commit (`68f020d`), not on the current `HEAD` (`de19c9df758`) — its
  working-tree files still show the legacy `create_descriptor`. Reading it required `git show
  68f020d:<path>`. Minor, but a porter who greps the working tree for `create_program_artifacts`
  finds only quasar hits and could be misled.

## Open items for downstream

- **Cross-op kernel touches:** none. All three kernels are owned in-directory; no shared/forked
  kernel files.
- **`test_fast_reduce_nc.py` lives under `tests/.../operations/reduce/`, not `.../reduction/`** —
  the family slug remaps `reduction` → `reduce` (a case the recipe's "Locate tests" section calls
  out). Recorded here so the next porter of a `reduction`-family op does not miss coverage.
- **Unused locals left in place (not port work):** `constexpr uint32_t onetile = 1;` in the reader
  and `constexpr uint32_t dst1 = 1;` in the compute kernel are dead in the legacy source and remain
  dead; left untouched per scope discipline.
- **`compute_kernel_config` resolution:** the port extracts `fp32_dest_acc_en` via
  `std::get<2>(get_compute_kernel_config_args(...))` (for the `FP32_DEST_ACC_EN` define and the
  `unpack_modes` guard) and builds the hardware config via `to_compute_hardware_config(...)` (Style
  A). Both read the same `ComputeKernelConfig`, so the resolved `fp32_dest_acc_en` is consistent
  across the two calls.

## Verification (orchestrator-run)

Not built or run by this porter. Exact commands for this op:

```bash
# Build (Metal + all TTNN test binaries)
./build_metal.sh --build-tests

# Correctness pytest (no C++ gtest exists for this op).
# Note the family-slug remap: reduction -> reduce.
pytest tests/ttnn/unit_tests/operations/reduce/test_fast_reduce_nc.py -x -v
```

No-regression baseline: `tests/ttnn/unit_tests/operations/reduce/test_fast_reduce_nc.py`
(the primary + only unit coverage; 2 test functions, sweeping shapes/dims/dtypes and
`compute_kernel_options` incl. fp32 dest-acc). Broader/optional coverage exists under
`tests/sweep_framework/sweeps/model_traced/fast_reduce_nc_model_traced.py` (sweep) and
`tests/nightly/t3000/ccl/test_deepseek_moe_reduce_scatter.py` (indirect, multi-device CCL).

## Anti-pattern self-audit

- [x] No `tensor.buffer()->address()` in the factory.
- [x] No magic-number CB indices in CTAs (all CB refs are `dfb::` handles / `DFBBinding`s).
- [x] No `TensorAccessorArgs<N>()` in any kernel.
- [x] No conditional DFB bindings (none needed).
- [x] No `.id` extraction at LLK call sites.
- [x] No CTA→RTA demotion in compute (per-group `num_output_tiles` stays a CTA; two KernelSpecs).
- [x] No `allow_instance_multi_binding` (two compute KernelSpecs over disjoint node sets → 1:1).
- [x] All CTAs named.
- [x] No nameable argument smuggled into varargs (no varargs used).
- [x] `hw_config` reproduces legacy resolved values: reader = reader default (RISCV_1/NOC_0),
      writer = writer default (RISCV_0/NOC_1) via the arch-agnostic helpers; compute via
      `to_compute_hardware_config` (math_fidelity / math_approx_mode / fp32_dest_acc_en /
      dst_full_sync_en) + explicit `unpack_modes` for the Float32 input DFB under
      `enable_32_bit_dest` (legacy Default → `UnpackToSrc`).
