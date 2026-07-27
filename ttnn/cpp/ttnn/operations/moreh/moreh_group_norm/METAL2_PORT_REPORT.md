# Metal 2.0 Port Report — moreh_group_norm

## Outcome

**PORTED** — the single `MorehGroupNormOperation` factory converted from the `descriptor`
(`create_descriptor` → `ProgramDescriptor`) concept to `MetalV2FactoryConcept`
(`ProgramFactory::create_program_artifacts` → `ProgramArtifacts`). Both the small and large
runtime-selected kernel paths converted together (atomic unit). Build + tests are the
orchestrator's to run (see Test commands).

## Provenance

- **Recipe docs (this port):** run `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` at the checkout root and paste verbatim. (Not captured here — orchestrator to fill; the working tree recipe was read at commit-time HEAD.)
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept`, as the audit specified. Realized as a nested `ProgramFactory` struct with
`create_program_artifacts`, plus `using program_factory_t = std::variant<ProgramFactory>;`. No custom
`select_program_factory` — the adapter auto-returns the single-type variant
(`mesh_device_operation_adapter.hpp:213`).

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op never had one).
- Pybind entry points removed: none — the nanobind binds the plain `&ttnn::moreh_group_norm` host
  function (`moreh_group_norm_nanobind.cpp:19-22`), not `create_descriptor`, so no pybind edit was
  forced.
- Header change (`device/moreh_group_norm_device_operation.hpp`): replaced the direct
  `static ProgramDescriptor create_descriptor(...)` with the `ProgramFactory` struct +
  `program_factory_t`; swapped the `program_descriptors.hpp` include for `metal_v2_artifacts.hpp`.
  This is the concept-change wiring the port forces, not an edit to the op's validate/attribute logic.

### Open items
- **Relaxation candidates:** none applied. `TensorParameter`s kept strict.
- **`mean_memory_config` / `rstd_memory_config` dead-for-placement** (audit "Misc anomalies"): still
  present in `operation_attributes_t`; `compute_output_specs` builds mean/rstd specs from
  `memory_config`, not these. Not touched (op-level host code, off-limits). Route to ops team.
- **Small reader double-reserve** (`reader_moreh_group_norm_small.cpp`): the legacy
  `dfb_input.reserve_back(num_inner_tiles)` at the top of the outer loop plus a second
  `reserve_back(num_inner_tiles)` inside the inner loop is preserved verbatim (faithful port; do not
  "fix" legacy kernel logic). Flagged for the ops team.

## Handoff points

- **Cross-op / forked kernels (compute):** the compute kernels
  `moreh_layer_norm/device/kernels/moreh_layer_norm_{small,large}_kernel.cpp` are shared with
  `moreh_layer_norm` (in-family, co-instantiated). Per orchestration constraints (moreh_layer_norm is
  being ported in parallel), they were **FORKED** — not modified in place — into this op's directory:
  - `device/kernels/compute/moreh_group_norm_small_kernel.cpp`
  - `device/kernels/compute/moreh_group_norm_large_kernel.cpp`
  Based on the committed (HEAD) legacy version. The legacy copies are untouched and remain for
  moreh_layer_norm. **Sunset:** when moreh_layer_norm's own Metal 2.0 port lands, consider whether a
  single shared Metal-2.0 compute kernel can replace both forks (the two ops pass `is_group_norm` /
  `is_lastdim_layernorm` CTAs to select behavior, so a shared source is feasible). Until then the fork
  is the coordination-free path. Reader/writer kernels are owned by this op and edited in place.
- No boundary-rule (`sem::` / `tensor::` out-of-op) violations. No kernel-lib gaps: `moreh_common.hpp`
  (dataflow + compute) and `reduce_helpers_compute.hpp` already take `DataflowBuffer` / CB-index
  template params and needed no change; `dfb::name` flows into them via the implicit `→ uint32_t`.

## Successes

- **Conditional / optional DFB bindings** ([pattern](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)): the recipe's define+`#ifdef` scaffold was exactly right for the 7 conditional DFBs (gamma/beta/mask_h/mask_w/mean/rstd/gamma_beta). The `(gamma_has_value || beta_has_value)` file-scope ternary for `cb_gamma_beta_or_out` is precisely the "file-scope ternary resolves both branches" case the pattern warns about — gated with `#if defined(GAMMA_HAS_VALUE) || defined(BETA_HAS_VALUE)`.
- **Self-loop DFB** for the 8 compute-internal intermediates (c_24–c_31): faithful, one accessor name each, PRODUCER+CONSUMER.
- **Preserved multiplicity**: kept `num_rows_per_core` as a per-group CTA across `COMPUTE_G1`/`COMPUTE_G2` in two WorkUnitSpecs (did not demote to RTA).
- **DM/compute hw_config helpers**: `create_reader/writer_datamovement_config(arch)` matched the legacy `ReaderConfigDescriptor{}`/`WriterConfigDescriptor{}` defaults; `to_compute_hardware_config(arch, compute_kernel_config)` carried the 4 knobs the legacy `ComputeConfigDescriptor` set (Style A).

## Friction

- **Gap — presence flags were CTAs, not the kernel's `#ifdef`s.** The borrowed compute kernels gate
  gamma/beta/mean/rstd via `constexpr bool ... = get_compile_time_arg_val(N)` and `if (...)`, and
  mask via a `constexpr bool do_mask_h` derived from the `origin_H` CTA. The conditional-binding
  pattern requires promoting these to preprocessor defines. This is the "Promote a CTA gate to a
  define" sub-case; it touched many `if (flag)` → `#ifdef FLAG` sites across 5 kernel files. Mechanical
  but the largest single source of edits. Mask was kept as a `constexpr bool` (still valid, derived
  from the retained `origin_H`/`origin_W` named CTAs) with only the `dfb::mask_*`-referencing blocks
  wrapped in `#ifdef DO_MASK_*`, to keep the masking-index logic diff minimal.
- **Confusion — `unpack_modes` under Float32 + fp32_dest_acc_en.** The legacy op never sets
  `unpack_to_dest_mode` (defaults to UnpackToSrc). The Metal 2.0 validator additionally *requires* an
  explicit entry for any compute-consumed Float32 DFB when `enable_32_bit_dest` is true. Implemented
  as: add `UnpackToSrc` for every present compute-consumed DFB **only** in that case, else omit. This
  is a no-op for the common bf16 path but keeps a Float32 + fp32-acc config valid. Flagging because
  the "which DFBs must carry an entry" set (compute-consumers only, excluding producer-only
  c_16/c_17/c_18) had to be re-derived from the kernel by hand.

## Open items for downstream

- **Cross-op kernel forks** (see Handoff points): `moreh_group_norm_{small,large}_kernel.cpp` forked
  from `moreh_layer_norm`. Remaining unmigrated consumer: `moreh_layer_norm` itself (and any other
  moreh op borrowing those compute kernels). Sunset the forks when moreh_layer_norm ports.
- **RTA → CRTA opportunity (later pass, not port work):** reader RTAs `scaler`, `eps`,
  `num_inner_tiles`, `num_channels`, `origin_h`, `origin_w`, `block_size` and writer RTAs
  `num_inner_tiles`, `num_groups`, `block_size` are the same on every node — genuine CRTA candidates.
  Left as per-node RTAs to match legacy dispatch semantics (RTA→CRTA is a separate cleanup).
- **Shared Metal-2.0 compute kernel** for moreh_layer_norm + moreh_group_norm once both are ported
  (they already differ only by CTA/define selection).

## Test commands (orchestrator to run)

Build:
```bash
./build_metal.sh --build-tests
```

Forward-op correctness (this op's no-regression baseline). The `backward` tests in the same file
exercise a *different* op (`moreh_group_norm_backward`) — excluded:
```bash
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_group_norm.py \
  -k "not backward" -v
```
This covers `test_moreh_group_norm` and `test_moreh_group_norm_callback` (the callback test exercises
program-cache reuse). Parametrization sweeps `affine` (gamma/beta present/absent) and
`compute_mean_rstd` (mean/rstd outputs present/absent), so all conditional-binding permutations are
exercised. There is no dedicated C++ gtest for this op.

**Build/test verification is the orchestrator's** (this porter did not build or run tests, per
orchestration constraints).
