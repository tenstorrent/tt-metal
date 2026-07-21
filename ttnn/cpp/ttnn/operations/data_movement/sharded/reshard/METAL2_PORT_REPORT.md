# Port Report — reshard (`NdReshardCopyPagesFactory`)

Post-port report for the Metal 2.0 port of `NdReshardCopyPagesFactory`, one of the 8 factory
instantiations in `ttnn::prim::ReshardDeviceOperation`. The other 7 remain on the legacy `descriptor`
concept (enumerated in `METAL2_PORT_PLAN.md`); the op builds and runs half-ported via per-factory dispatch.

## Provenance
- **Recipe docs (this port):** `6a16e3bf8d8 2026-07-20 recipe: reframe migration-guide 'not yet available' as surface-maturation`
- **Audit docs (inherited):** `6a16e3bf8d8 2026-07-20 recipe: reframe migration-guide 'not yet available' as surface-maturation`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — `NdReshardCopyPagesFactory::create_program_artifacts` returns
`ttnn::device_operation::ProgramArtifacts`. No re-decision from the audit.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op never had one).
- Pybind entry points removed: none (`reshard_nanobind.cpp` binds no `create_descriptor`).

### Open items
- No relaxation candidates: strict `TensorSpec` matching is correct for this factory (audit recorded no
  relaxation; the kernels are page-index driven, not shape-agnostic in a way that needs `dynamic_tensor_shape`).
- Only 1 of 8 factories ported; the op stays on `descriptor` for the rest via per-factory dispatch. No
  friction with the `MetalV2FactoryConcept` fit for this factory.

## Handoff points
- **None.** No out-of-op-directory kernel edits (both kernels are in the op's own dir), no `sem::` / `tensor::`
  boundary violations, no pybind `create_descriptor` to remove (the op never bound one). The audit's
  team-only "Misc anomalies" (dead code in `is_valid_for_legacy_reshard`; dead `num_output_pages` RTA in the
  legacy readers) are pre-existing, outside this factory, and already routed to the ops team by the audit —
  not re-raised here.

## Successes
- **Case 1 vs Case 2 by kernel-side usage (audit brief + recipe [Dropped Plumbing]).** The brief classified
  the ND factories' input/output as **Case 1** even though the base arrives via a `Buffer*` common-RTA
  binding — because the kernel feeds it into `TensorAccessor(args, base)`. That matched exactly: the port
  dropped the `emplace_common_runtime_args({buffer})` + `TensorAccessorArgs` plumbing and collapsed the
  kernel to `TensorAccessor(tensor::name)` with no raw-pointer bridge. The brief's Recipe-note #1 (that the
  `Buffer*`-binding form is a *delivery* mechanism orthogonal to the Case 1/2 split) is confirmed correct by
  this port.
- **Migration guide Example 1 was a near-exact template.** This factory is the canonical
  reader → one-DFB → writer pipeline; the example's plain `DataflowBufferSpec` (PRODUCER on reader, CONSUMER
  on writer) + `TensorBinding` + `create_reader/writer_datamovement_config` mapped 1:1 with no improvisation.
- **`AddRuntimeArgsForNode` bridged the node-first loop cleanly** — the legacy per-core
  `emplace_runtime_args(core, {start_page, end_page})` loop kept its shape; the helper transposed it into the
  name-first table without my having to hand-invert the loop (recipe [Construct — KernelRunArgs]). Also
  frictionless because `NodeCoord` is a type alias for `CoreCoord`, so legacy `CoreCoord`s dropped straight
  into `target_nodes` / the helper with no conversion.

## Friction

### Gaps
- **Build command diverges from repo norm.** The recipe's Verification step prescribes
  `cmake --build build_Release --target ttnncpp unit_tests_ttnn -j 8`, but this repo builds via
  `./build_metal.sh` (and its guidance warns against hand-targeting a test binary in the unity build). I
  used `./build_metal.sh --build-tests`. Suggest the recipe note that repo-specific build wrappers take
  precedence where they exist.
- **Verification step omits Python-env activation.** `workspace_setup.md` / the recipe say to set
  `PYTHONPATH=$(pwd)` but don't mention activating the repo virtualenv. A bare
  `PYTHONPATH=$(pwd) python -m pytest …` under the system interpreter fails at conftest import with
  `ModuleNotFoundError: No module named 'ttnn.device'` (it resolves `ttnn` as a namespace package with
  `__file__ = None`). The working invocation is `source python_env/bin/activate && PYTHONPATH=$(pwd) …`.
  A one-line note in the verification section would save a porter a confusing dead-end.
- **No guidance for "no device available."** The verification step assumes a physical accelerator is
  present. On a node with none (this one), every test errors at fixture setup and the on-device gate simply
  cannot be met. The recipe has no off-ramp for this — it is neither a code capitulation nor a pass. Suggest
  the recipe state explicitly that absent hardware blocks verification (not the port) and that the porter
  should surface it and hand off the test run, rather than treat a clean build + self-audit as sufficient.

### Confusion
- **Latent namespace-shadow in the migration guide's DFB example.** `migration_guide.md:598` writes
  `experimental::DataflowBuffer dfb(dfb::my_dfb);` — a local variable named `dfb` whose initializer
  references the `dfb::` namespace. The variable name is in scope within its own initializer, so
  `dfb::my_dfb` would resolve `dfb` to the variable (not the namespace) and fail to compile. Every real
  ported kernel avoids this by naming the local something else (`act_cb`, `input_cb`, `cb_weight_obj`, …).
  I named mine `staging_dfb`. Suggest the guide example rename the variable to match the observed
  convention, so a literal copy of the example compiles.

## Verification status
- **Build:** SUCCESS. `./build_metal.sh --build-tests` (repo norm; see Friction) rebuilt `_ttnncpp.so` /
  `_ttnn.so` clean with the ported factory linked in. Host-side compilation of the new `ProgramSpec` /
  `ProgramRunArgs` construction is verified.
- **Anti-pattern self-audit:** PASS on all four ported files — no `->address()`, no `TensorAccessorArgs`,
  no magic CB index, no positional CTAs, no `.id` extraction, no varargs; `hw_config` reproduces the exact
  reader/writer DM defaults.
- **On-device tests: NOT RUN — no hardware.** The device kernels JIT-compile and execute only on a
  physical accelerator, so `test_nd_reshard.py` (incl. `test_DRAM_nd_reshard`'s DRAM→DRAM case, the one
  routing to this factory) is the real kernel-side gate — and it could not run. This node has **no
  Tenstorrent device**: `lspci` shows none, no `/dev/tenstorrent`, `tt-smi` reports "No Tenstorrent driver
  detected." All 117 collected cases errored at the `device` fixture (`TT_FATAL … num_chips > 0 / No chips
  detected`). **The kernel-side port is therefore unverified at runtime.** Re-run on a device-equipped node:
  `source python_env/bin/activate && PYTHONPATH=$(pwd) python -m pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nd_reshard.py::test_DRAM_nd_reshard -v`.

## Open items for downstream
- **Remaining reshard factories.** 7 of 8 instantiations still on the legacy concept — see
  `METAL2_PORT_PLAN.md` § Deferred / Flagged for the enumerated next-pass work (the 6 shared-pool kernels
  are an in-place rewrite that must land together with the Generic/SameWidth/SameHeight factories).
- **On-device verification owed** (see Verification status) — must pass before this port is trustworthy.
