# Port Report — reshard (`NdReshardCopyPagesFactory`, `ReshardGenericFactory`)

Post-port report for the Metal 2.0 port of **two** of the 8 factory instantiations in
`ttnn::prim::ReshardDeviceOperation`:
- **`NdReshardCopyPagesFactory`** — DRAM↔DRAM page copy (the clean case). Committed `cbc70d0efce`.
- **`ReshardGenericFactory`** — general L1-sharded reshard (the complex case: Case 2 raw pointer, borrowed
  multi-binding DFB, varargs, shared-pool kernel rewrite).

The other 6 instantiations remain on the legacy `descriptor` concept (enumerated in `METAL2_PORT_PLAN.md`);
the op builds and runs partially-ported via per-factory dispatch. **Both ported factories are
hardware-verified** (see Verification status).

## Provenance
- **Recipe docs (this port):** `6a16e3bf8d8 2026-07-20 recipe: reframe migration-guide 'not yet available' as surface-maturation`
- **Audit docs (inherited):** `6a16e3bf8d8 2026-07-20 recipe: reframe migration-guide 'not yet available' as surface-maturation`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for both — `NdReshardCopyPagesFactory::create_program_artifacts` and
`ReshardGenericFactory::create_program_artifacts` return `ttnn::device_operation::ProgramArtifacts`. No
re-decision from the audit.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op never had one).
- Pybind entry points removed: none (`reshard_nanobind.cpp` binds no `create_descriptor`).

### Open items
- No relaxation candidates: strict `TensorSpec` matching is correct for both factories.
- 2 of 8 factories ported; the op stays on `descriptor` for the rest via per-factory dispatch. No friction
  with the `MetalV2FactoryConcept` fit.

## Handoff points
- **Shared-pool kernel rewrite (Open-items, not a blocker).** `ReshardGenericFactory` rewrote two kernels
  that live *outside* the op's own directory — `reshard_reader.cpp` and `reshard_reader_diff_width.cpp` in
  the in-family pool `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/`. Path taken:
  **in-place modification** (not a `_metal2` fork), justified because a grep confirmed **no other op**
  instantiates those exact paths (`experimental/quasar/reshard` keeps its own copies), so there is no
  unmigrated co-consumer to break. See Open items for the coordination note.
- No `sem::` / `tensor::` out-of-op boundary violations; no pybind `create_descriptor` to remove. The
  audit's team-only "Misc anomalies" (dead code in `is_valid_for_legacy_reshard`; dead-but-read
  `num_output_pages`) are pre-existing and already routed to the ops team by the audit — not re-raised.

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
- **(Generic) The sync-free self-loop pattern held on silicon.** The borrowed output DFB (c_16) is an
  address source only (`get_write_ptr`, no FIFO ops), touched by two co-resident instances. The recipe's
  [sync-free self-loop pattern] + [multi-binding disposition] composed exactly as documented: each instance
  bound PRODUCER+CONSUMER (shared accessor name `shard_cb`) with `allow_instance_multi_binding = true`. The
  spec validator accepted it and the numerics passed — the audit's multi-binding disposition was correct,
  and the recipe's pattern for "sync-free CB touched by 2 co-resident instances" is real, not hypothetical.
- **(Generic) `borrowed_from` satisfies the TensorParameter-referenced rule without a dummy binding.** The
  output tensor is used *only* through the borrowed DFB — no kernel builds `TensorAccessor(tensor::output)`.
  The validator (`program_spec.cpp:540`) explicitly registers `borrowed_from` as a use, so `output` needed
  a `TensorParameter` + `TensorArgument` but **no** kernel `TensorBinding`. Reading the validator saved a
  pointless dummy binding.
- **(Generic) Case 2 raw-pointer bridge worked as the whitelist describes.** Dropping the patched
  `input_addr` RTA slot host-side (`erase()` at `grid.x+grid.y`, `input_addr=0` into the packing helper) and
  pulling the base from `TensorAccessor(tensor::input).get_bank_base_address()` kernel-side left the raw
  `noc.async_read({.noc_x,.noc_y,.addr=base+off})` stride walk untouched — exactly the rule-5 contract.

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
  present. The copy_pages port was first attempted on a node with none — every test errored at fixture
  setup (`TT_FATAL … num_chips > 0`). It's neither a code capitulation nor a pass; the recipe has no
  off-ramp. (Resolved by moving to a device-equipped node.) Suggest the recipe state that absent hardware
  blocks *verification*, not the port, and that the porter should surface it and hand off the test run.
- **Node without `python_env` needs `create_venv.sh`.** The device node had no repo virtualenv at all
  (`source python_env/bin/activate` → no such file), and the two ambient venvs (`/opt/venv`,
  `~/.tenstorrent-venv`) lacked tt-metal's Python deps (`ModuleNotFoundError: No module named 'tracy'`).
  `./create_venv.sh` at repo root creates `python_env` with everything and an editable `ttnn`. The recipe's
  verification/setup docs should point at `create_venv.sh` as the prerequisite for the pytest step.
- **`get_bank_base_address` takes no argument on today's API.** [Kernel-side whitelist rule 5] and its
  migration-guide example both write `input.get_bank_base_address(bank_id)`, but the actual kernel API
  (`tt_metal/hw/inc/api/tensor/tensor_accessor.h:167,473`) is arg-less: `get_bank_base_address()`. For a
  sharded input the shard base L1 address is bank-independent, so no argument is needed. The doc example
  should drop the `bank_id` parameter (or note it's arg-less today) to avoid a compile error on a literal copy.
- **(Generic) Borrowed-DFB size trips on padded shards — needs a clamp the recipe doesn't mention.** A
  sharded output shard shape can be padded larger than the tensor's packed size (e.g. a `[32,96]` tensor
  with a `(32,128)` shard), so the shard-derived `total_size` exceeds the backing tensor's real bytes and
  the Metal 2.0 borrowed-DFB size check rejects it. Fix: `num_entries = min(total_size,
  output.tensor_spec().compute_packed_buffer_size_bytes()) / output_buffer->page_size()`. The recipe's
  borrowed-DFB guidance ([DataflowBufferSpec]) should flag that a padded shard's declared size must be
  clamped to the backing tensor's packed size.

### Confusion
- **Latent namespace-shadow in the migration guide's DFB example.** `migration_guide.md:598` writes
  `experimental::DataflowBuffer dfb(dfb::my_dfb);` — a local variable named `dfb` whose initializer
  references the `dfb::` namespace. The variable name is in scope within its own initializer, so
  `dfb::my_dfb` would resolve `dfb` to the variable (not the namespace) and fail to compile. Every real
  ported kernel avoids this by naming the local something else (`act_cb`, `input_cb`, `cb_weight_obj`, …).
  I named mine `staging_dfb`. Suggest the guide example rename the variable to match the observed
  convention, so a literal copy of the example compiles.
- **(Generic) Two valid encodings of the borrowed multi-binding DFB; the recipe/audit point at one, an
  existing port at the other.** The audit dispositioned c_16 as **multi-binding** (2 same-kind writers/node),
  which — combined with the recipe's sync-free self-loop pattern — realizes as *both* instances bound
  PRODUCER+CONSUMER + `allow_instance_multi_binding = true` (2P+2C/node). But the validator also accepts a
  simpler **1 PRODUCER + 1 CONSUMER** encoding (reader=PRODUCER, writer=CONSUMER, no flag), which the actual
  Quasar port of this op uses — because for a sync-free borrowed DFB the endpoint role is cosmetic on Gen1
  and 1P+1C satisfies the per-node "exactly one of each" invariant without the unsafe flag. I followed the
  audit's disposition (self-loop + multi-binding) and it passed on hardware. Worth a recipe note that these
  two encodings are both legal, with the trade-off: the multi-binding flag is the honest Gen2-debt signal
  (hard-errors on Quasar, forcing a refactor) while 1P+1C is quieter but avoids the "unsafe" flag.

## Verification status
- **Build:** SUCCESS. `./build_metal.sh --build-tests` (repo norm; see Friction) rebuilt clean with both
  ported factories + the two rewritten shared-pool kernels linked in.
- **Anti-pattern self-audit:** PASS on all ported files (both factory `.cpp`/`.hpp` + 4 kernels) — no
  `->address()`, no `TensorAccessorArgs`, no magic CB index, no positional CTAs, no `.id` extraction;
  `hw_config` reproduces the exact reader/writer DM defaults; varargs used only where the Generic kernels
  loop-read a runtime-count tail.
- **On-device tests: PASS on a Wormhole/Blackhole device.**
  `python -m pytest test_reshard.py test_nd_reshard.py -v` → **246 passed, 12 skipped, 0 failed** (~4m15s).
  - `test_reshard.py`: 137 passed, 4 skipped — exercises `ReshardGenericFactory` (width-sharded L1 paths)
    plus regression on the untouched SameWidth/SameHeight factories.
  - `test_nd_reshard.py`: 109 passed, 8 skipped — exercises `NdReshardCopyPagesFactory` (DRAM↔DRAM, incl.
    `test_DRAM_nd_reshard`) plus the untouched CopyLocalShard paths.
  - Skips are pre-existing (illegal layout/dtype configs; `bfloat8_b` unsupported for ROW_MAJOR) — unrelated
    to the port. No hang, no segfault, clean device teardown.
  - Re-run: `source python_env/bin/activate && PYTHONPATH=$(pwd) python -m pytest <the two files> -v`.

## Open items for downstream
- **Cross-op kernel touch (coordination signal).** `reshard_reader.cpp` and `reshard_reader_diff_width.cpp`
  in `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/` were rewritten **in place**
  (not forked) for Metal 2.0. Current co-consumers of those exact paths: **none** (grep-confirmed;
  `experimental/quasar/reshard` uses its own copies). If a future op starts instantiating these shared paths
  before it is on Metal 2.0, it will break — the next porter touching this pool should re-check consumers.
- **Remaining reshard factories.** 6 of 8 instantiations still on the legacy concept — see
  `METAL2_PORT_PLAN.md` § Deferred / Flagged. The SameWidth/SameHeight factories reuse the exact patterns
  proven here (borrowed multi-binding DFB, Case 2 `get_bank_base_address`, varargs), so they should port
  faster; each carries its own shared-pool kernels (`reshard_same_width_*` / `reshard_same_height_*`).
- **(Generic) Perf observation, not acted on (out of scope).** The legacy factory recomputes the full
  output-core→page-range map *inside* the per-core loop (O(num_cores) redundant). The port preserves this
  (faithful, minimal-diff) and it's fine in Release. The Quasar port hoisted it out of the loop, noting the
  redundancy can exceed the pytest timeout for the biggest case in a **Debug** build. A worthwhile separate
  cleanup for the ops team; left untouched here per scope discipline.
