# Metal 2.0 Port Report — `experimental/transformer/nlp_concat_heads_decode`

## Outcome

**PORTED** — both factories (`NLPConcatHeadsDecodeProgramFactory` full-grid and
`NLPConcatHeadsDecodeSubcoregridsProgramFactory`) converted to `MetalV2FactoryConcept`
(`create_program_artifacts`). The op's full pytest baseline
(`tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads_decode.py`,
`test_concat_head` + `test_concat_head_subcoregrids`) passes: **15/15**, covering both factories,
`padded_heads > 32`, several head_dims/batches, and multiple sub-core-grid layouts. No C++ gtests
exist for this op (confirmed by grep).

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for both factories — as the audit chose. Each factory replaced its
`create_descriptor` (→ `ProgramDescriptor`) with `create_program_artifacts` (→ `ProgramArtifacts`).
`select_program_factory` and the rest of the device-op class are unchanged; the framework dispatches
per-factory on the concept. No op-owned tensors.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** — the op never had one.
- Pybind entry points removed: **none** — the nanobind binds the high-level `nlp_concat_heads_decode`
  function, not a factory `create_descriptor`; nothing to remove.

### Open items
- **Relaxation candidates:** none applied. `TensorParameter`s kept strict (default). The kernels use
  the input base only via `get_bank_base_address()` (raw NoC arithmetic), so a padding-only relaxation
  is not obviously safe; not pursued during the port.
- No capability gaps hit; the op fits the single-program concept cleanly.

## Handoff points

None. The port is entirely within the op directory; no capitulation, no out-of-directory kernel
changes, no `sem::`/`tensor::` boundary violations, no kernel-lib gaps.

## Successes

- **Unity-build hygiene pattern fired exactly as documented**
  ([Pattern: Unity-build hygiene for anonymous-namespace symbols](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)).
  First build failed with `redefinition of 'READER' / 'WRITER' / 'Q_OUT' / 'INPUT' / 'OUTPUT'` because
  both factory `.cpp`s declared the same anonymous-namespace `StrongType` constants and the transformer
  unity target merged them into one TU. Fix: move the five constants into each `create_program_artifacts`
  body (function-local), keeping the *string* values identical (they feed `dfb::q_out` / `tensor::input`
  / `tensor::output` that both kernels reference) while de-conflicting the C++ identifiers. Rebuild green.
  The catalog entry named this precise failure — a genuinely valuable warning for a two-factory op.
- **Two-toucher → 1P+1C** ([Pattern: Two-toucher DFB → assign 1P+1C](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)).
  Re-derived the census independently and agreed with the brief: `q_out` is one source instantiated
  Reader-config + Writer-config over the *same* `q_cores`, both raw-writing disjoint tile phases via
  `get_write_ptr()`, nothing draining it. Bound reader PRODUCER / writer CONSUMER, no multi-binding flag.
  The recipe's warning against mis-slotting this as multi-binding was on point.
- **Borrowed-memory DFB + Case-2 tensor binding** matched the proven reshard-generic factory shape
  1:1 (`.borrowed_from = OUTPUT`; input base via `TensorAccessor(tensor::input).get_bank_base_address()`).
  The reshard port was the most useful current-code reference (recipe steered me off the quasar/ ports
  and the stale accumulation reference, correctly — see Friction).

## Friction

### Gaps
- **Stale sanctioned reference (accumulation).** The recipe points to the accumulation port
  (`akertesz/porting-experiment-accumulation-jun10`) as the shape reference, but it is stale against the
  current headers in three ways that would have miscompiled if copied: (1) its factory method is named
  `create_program_spec`, but `MetalV2FactoryConcept` now requires `create_program_artifacts`; (2) it sets
  `hw_config = DataMovementHardwareConfig{.role = ...RoleHint::READER}` — the current
  `data_movement_hardware_config.hpp` has no `.role`, using `DataMovementGen1Config` +
  `Create{Reader,Writer}Gen1DataMovementConfig()` / the TTNN `create_{reader,writer}_datamovement_config`
  helper instead; (3) it builds `runtime_arg_values` node-first (`push_back({core, args})`), but the field
  is now name-first `Table<name, Table<node, value>>` filled via `AddRuntimeArgsForNode`. The recipe does
  warn that reference ports "do not represent current best practice"; still, a one-line note in the recipe
  that the accumulation reference specifically predates the `create_program_artifacts` rename, the DM
  `.role` config, and the name-first RTA table would save a porter from copying a miscompile. The current
  quasar reshard-generic factory was an accurate up-to-date reference.

### Confusion
- **Case-2 kernel with no prior `TensorAccessor`.** Kernel-side whitelist rule 5 says the port adds
  "exactly two headers" (`experimental/kernel_args.h`, `api/dataflow/dataflow_buffer.h`) and that
  `TensorAccessor` "comes from the same headers before and after." These Case-2 reader kernels used **no**
  `TensorAccessor` before the port (memory was a hand-rolled NoC gather off a `Buffer*` RTA), so there was
  no existing include to inherit. Introducing the `get_bank_base_address` bridge required a **third**
  include, `api/tensor/tensor_accessor.h`. Confirmed correct against the reshard-generic kernel, which
  includes it for the same reason. Suggest rule 5 note: a Case-2 kernel that did not *previously*
  construct a `TensorAccessor` needs `api/tensor/tensor_accessor.h` added — the "two headers" count
  assumes the accessor was already in use.

## Open items for downstream

- **Cross-op kernel touches:** none — both kernel files are owned by this op.
- **RTA-varargs recognition sub-shape (carried from the audit).** The two NoC-coordinate blocks were
  read in the legacy kernel as L1 `uint32_t*` arrays via `get_arg_addr(2)` / `get_arg_addr(2 + num_x)`,
  not the `get_arg_val(arg_index++)` loop the recipe's vararg recognition is framed around. It is the same
  variable-count indexed-collection case (CTA-bounded by `in_num_cores_x`/`in_num_cores_y` or
  `in_num_cores`); ported to `num_runtime_varargs` + per-node `runtime_varargs`, read kernel-side via
  `get_vararg(i)`. Worth adding the pointer-cast-into-L1-array shape as an explicit vararg sub-example so
  a future porter/auditor doesn't read the `get_arg_val` loop framing narrowly and miss it.
- **RTA → CRTA opportunity (not applied, per recipe).** The `noc_x`/`noc_y` vararg payload is identical
  on every output node (the input grid is fixed), so it is morally a *common* runtime vararg
  (`num_common_runtime_varargs`). Kept as a per-node RTA vararg to faithfully mirror the legacy per-core
  push and avoid a dispatch-semantics change during the port. A later cleanup could promote it to a CRTA
  vararg for dispatch efficiency.
- **Pre-existing dead local `q_write_addr`** in both kernels (declared then immediately shadowed by the
  inner-loop redeclaration) — flagged by the audit's Misc anomalies, left untouched (kernel behavior must
  not change). A trivial cleanup for the ops team, not port work.
