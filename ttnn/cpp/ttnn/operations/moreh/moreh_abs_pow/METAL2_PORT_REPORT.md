# Metal 2.0 Port Report — moreh_abs_pow

## Outcome

**PORTED** — the single `MorehAbsPowOperation` factory converted from the TTNN `descriptor`
concept to `MetalV2FactoryConcept` (`create_program_artifacts`). Build/test verification is the
**orchestrator's** (this porter did not build or run tests, per orchestration constraints).

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Files created / modified

Created:
- `METAL2_PORT_PLAN.md` — the port plan (inventory + structural decisions).
- `METAL2_PORT_REPORT.md` — this report.

Modified (all under the op directory):
- `device/moreh_abs_pow_device_operation.hpp` — replaced the direct `create_descriptor` static
  with a nested `MorehAbsPowProgramFactory { create_program_artifacts(...) }` and
  `using program_factory_t = std::variant<MorehAbsPowProgramFactory>`; swapped the
  `<tt-metalium/program_descriptors.hpp>` include for `ttnn/metal_v2_artifacts.hpp`.
- `device/moreh_abs_pow_program_factory.cpp` — full rewrite of the factory body from
  `ProgramDescriptor` (`CBDescriptor`/`KernelDescriptor`/RTAs) to a Metal 2.0 `ProgramSpec` +
  `ProgramRunArgs` (`DataflowBufferSpec`, `KernelSpec` with DFB/tensor bindings, `TensorParameter`,
  `WorkUnitSpec`, `KernelRunArgs`). Compute config built as `ComputeGen1Config`; DM configs via the
  arch-agnostic reader/writer helpers.
- `device/kernels/reader_moreh_abs_pow.cpp` — named args (`get_arg(args::…)`), `TensorAccessor(tensor::input)`,
  `DataflowBuffer(dfb::…)`, `dfb_input.get_tile_size()`; dropped `input_addr` RTA + `TensorAccessorArgs<0>()`
  + the CB-id counter block; added `experimental/kernel_args.h`.
- `device/kernels/writer_moreh_abs_pow.cpp` — same transformation on the output side
  (`TensorAccessor(tensor::output)`, `dfb::out`, `dfb_output.get_tile_size()`).
- `device/kernels/moreh_abs_pow_kernel.cpp` (compute) — named args; `DataflowBuffer(dfb::…)` for all
  nine buffers; `dfb::x`/`dfb::mask_w` passed directly to `copy_tile`; `binary_op_init_common(dfb::x, dfb::x, dfb::y)`;
  dropped the CB-id counter block; added `experimental/kernel_args.h`. Kernel loop/logic unchanged.

Not modified: `moreh_abs_pow_device_operation.cpp` (validate / compute_output_specs /
create_output_tensors / the `ttnn::prim` launch), `moreh_abs_pow.cpp/.hpp`,
`moreh_abs_pow_nanobind.cpp/.hpp` — nothing there references a vanished factory entry point.

## Port summary

Single-program op, one factory, no semaphores, no op-owned tensors. The spec has 9 DFBs
(input `x`, `one`, `decimal`, `mask_w`, output `y`, and four compute-only intermediates
`xabs`/`xpow`/`logx`/`exp_lxmd`), 2 tensor parameters (input, output), 3 kernels (reader, writer,
compute), and one work unit over `all_cores`.

- **Tensor bindings**: both `input` and `output` are Case 1 — `TensorParameter` + `TensorBinding`,
  kernel builds `TensorAccessor(tensor::name)`. Buffer-address RTAs and `TensorAccessorArgs` plumbing
  dropped end-to-end.
- **DFB endpoints**: `input`/`one`/`decimal`/`mask_w` are 1P (reader) + 1C (compute); `output` is
  1P (compute) + 1C (writer); the four intermediates are self-looped on the compute kernel (bound
  both PRODUCER and CONSUMER, shared accessor name).
- **`mask_w`** is bound **unconditionally**. Its produce/consume is gated on `do_mask_w`
  (`origin_w % 32 != 0`), but `origin_w` is a **runtime** arg, so `do_mask_w` is a runtime value —
  the conditional-binding `#ifdef` pattern (which requires host-time knowledge) does not apply. A
  runtime-unused always-bound DFB is harmless (matches legacy, where the `c_3` CB was always allocated).
- **Compute-group collapse**: the two legacy per-core-group compute descriptors collapse to one
  `KernelSpec` — see Friction.
- **Hardware config**: reader/writer use the default reader/writer DM triples via
  `ttnn::create_reader_datamovement_config` / `create_writer_datamovement_config`. Compute is built
  as a `ComputeGen1Config` directly — see Friction for why the TTNN helper was not used.

## Test command(s) — verification is the orchestrator's

No test exercises `moreh_abs_pow` directly (no `test_moreh_abs_pow.py`, no C++ gtest — grep clean).
The op is a building block of the p-norm: `moreh_norm` calls `moreh_abs_pow(input, p, …)` then
`moreh_abs_pow(tmp, 1/p, …)` (`moreh_norm/moreh_norm.cpp:33-35,56-58`), so both integer and
fractional exponents (the `decimal` / fractional-power path) are covered by the moreh_norm tests.

No-regression baseline (please confirm completeness):

```bash
# Primary — exercises moreh_abs_pow via p-norm (integer + fractional p)
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_norm.py -x -v

# Secondary — moreh_clip_grad_norm builds on moreh_norm
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_clip_grad_norm.py -x -v
```

There is no C++ gtest layer for this op.

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept`, as the audit specified. The op previously used `HasDirectDescriptor`
(a `create_descriptor` static directly on the operation struct). Because the MetalV2 adapter has no
direct-`create_program_artifacts` shortcut (`mesh_device_operation_adapter.hpp:189-196`
special-cases only `create_descriptor`), the port introduces a nested factory struct plus a
single-alternative `program_factory_t` variant. `create_descriptor` was removed (its continued
presence would also satisfy `ProgramDescriptorFactoryConcept`, breaking `AllFactoriesValid`).

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op never had one).
- Pybind entry points removed: **none** (`moreh_abs_pow_nanobind.cpp` binds only the user-facing
  `ttnn::moreh_abs_pow` op entry point; no `create_descriptor` pybind existed).

### Open items
- **Relaxation candidates**: none applied. The op did not use `ArgConfig::Runtime*` and the audit
  flagged no relaxation, so `TensorParameter` matching stays strict.

## Handoff points

None — no capitulation, no out-of-directory edits, no boundary-rule (`sem::`/`tensor::`) violations,
no kernel-lib gaps. The shared `moreh_common.hpp` headers were not touched (their helpers already
take `DataflowBuffer`).

## Successes

- **Self-loop pattern** ([Sync-free and single-ended CBs → self-loop DFB]): the four compute-only
  intermediates (`xabs`/`xpow`/`logx`/`exp_lxmd`) are single-toucher, so binding the compute kernel
  as both PRODUCER and CONSUMER with a shared accessor name satisfied the validator with no kernel
  change (`moreh_abs_pow_program_factory.cpp` compute `dfb_bindings`).
- **`get_tile_size(cb_id)` → `dfb.get_tile_size()`** (kernel whitelist rule 7): applied cleanly in
  reader (`reader:...input_tile_bytes`) and writer; the DFB object exposes the arg-less getter.
- **Runtime-gated `mask_w`**: the recipe's rule that conditional *runtime* usage does not require the
  `#ifdef` conditional-binding scaffold (only host-time conditions do) resolved this cleanly — bind
  unconditionally, use at runtime.

## Friction

### Gaps
- **Dead per-group compute CTA blocks a literal "preserve multiplicity".** The legacy factory
  emits two compute `KernelDescriptor`s over `core_group_1` / `core_group_2` differing only by the
  CTA `{num_units_per_core_group_*}`. That CTA is **never read** by the compute kernel (no
  `get_compile_time_arg_val` in `moreh_abs_pow_kernel.cpp`; the loop count is the RTA
  `num_rows_per_core`). The recipe's default is "preserve multiplicity, reproduce the per-group
  CTA," but reproducing a **named** CTA the kernel never references is a build error
  (recipe Build §: "host added a named CTA/RTA without the kernel referencing it → reconcile").
  Resolution: collapse to a single compute `KernelSpec` over `all_cores`. This is **not** the
  "Demoting per-group CTA to RTA" anti-pattern — that anti-pattern's harm is losing compile-time
  loop unrolling on a dimension that *was* a CTA, whereas here the loop dimension was *already* an
  RTA in the legacy kernel, so nothing is demoted and no unrolling is lost. The docs could add a
  note for the "dead per-group CTA" case, where collapse is the correct move.

### Confusion
- **Which compute-config path (Style A helper vs. build directly).** The op resolves its config
  via `get_compute_kernel_config_args` (Style A → recipe prefers `to_compute_hardware_config`), but
  the legacy factory then builds a `ComputeConfigDescriptor` that sets only `math_fidelity`,
  `fp32_dest_acc_en`, `math_approx_mode`, **leaving `dst_full_sync_en` at the descriptor default
  (false)** — it never forwards the resolved `dst_full_sync_en`. Using `to_compute_hardware_config`
  would forward the resolved `dst_full_sync_en` and set `double_buffer_dest = !dst_full_sync_en`,
  diverging from the legacy *applied* config whenever a caller sets `dst_full_sync_en = true`. To
  reproduce legacy exactly I built a `ComputeGen1Config` directly with the three fields the legacy
  descriptor set, leaving `double_buffer_dest` at its default `true` (== legacy `dst_full_sync_en =
  false`). The recipe's "prefer the TTNN helper for a TTNN port" guidance is the right default, but
  this op is a case where the helper would *not* be faithful; a doc note on "legacy op drops a
  resolved knob when building its descriptor → build the Gen1 config directly" would help.
- **`unpack_modes` for the FP32 intermediates.** When `fp32_dest_acc_en` is true the four
  intermediate DFBs become `Float32` and are *consumed* by the compute kernel (self-loop), so the
  validator requires explicit `unpack_modes` entries. Legacy `unpack_to_dest_mode` was empty (all
  `Default`) → mapped to `UnpackMode::UnpackToSrc`. Added only under `fp32_dest_acc_en`; the io/scalar
  DFBs are never Float32 (BFLOAT16/INT32), so they need no entry.

## Open items for downstream

- **Latent: legacy compute config drops `dst_full_sync_en`.** The legacy factory resolves
  `dst_full_sync_en` from the compute-kernel config but never applies it to `ComputeConfigDescriptor`
  (`double_buffer_dest` is effectively fixed). Reproduced faithfully here (not fixed — scope
  discipline). The op owner should decide whether `dst_full_sync_en` was meant to be honored.
- **Dead RTAs `input_is_dram` / `output_is_dram`.** Both are read by their kernels and never used
  (addressing goes through the `TensorAccessor`). Carried as-is per the brief. Candidate cleanup for
  a separate PR (drop the RTA + the kernel read together).
- **Dead compute CTA `num_units_per_core_group_*`.** Dropped during the port (never read on device;
  see Friction). No behavior change.
- **RTA → CRTA candidates.** `input_is_dram`, `output_is_dram`, `decimal`, `origin_w`, `p`,
  `p_is_negative` take the same value on every node and are really CRTAs. Left as per-node RTAs
  (RTA→CRTA changes dispatch semantics; out of scope per the recipe). Candidate for a later cleanup.
- **Cross-op kernel touches**: none. The shared `ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp`
  headers were not modified (their helpers already take `DataflowBuffer`).
- **Test coverage**: `moreh_abs_pow` has no dedicated test; coverage is transitive via `moreh_norm`.
  A direct unit test would make future ports of this op safer.
