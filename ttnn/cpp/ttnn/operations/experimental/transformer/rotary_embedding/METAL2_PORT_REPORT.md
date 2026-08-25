# Port Report — rotary_embedding

Post-port report for `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding`, ported from the
`ProgramDescriptor` API to Metal 2.0 (`CustomProgramSpecFactoryConcept`).

## Outcome

**PORTED** — the op's single factory (`RotaryEmbeddingProgramFactory`, both internal variants: single-tile and
multi-tile, all config axes: decode/prefill x in-sharded/interleaved x out-sharded/interleaved) converted, all
7 kernel entry points converted (6 in place + 1 shared-kernel `_metal2` fork). Verification: full confirmed
baseline (`tests/tt_eager/python_api_testing/unit_testing/misc/test_rotary_embedding.py` +
`tests/ttnn/integration_tests/falcon7b/test_falcon_rotary_embeddings.py`) — **763 passed / 0 failed**, exact
pre-port baseline match, with the Metal 2.0 legality checks proven live (`METAL2_CHECKS_FORCED` markers in the
test log).

## Provenance

- **Recipe docs (this port):** `2cd0286fa17 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `2cd0286fa17 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory

### Concept realized

**`CustomProgramSpecFactoryConcept`**, as the audit chose — `create_program_artifacts` (method swap inside the
existing `program_factory_t` struct; no direct-descriptor conversion needed) plus the translated
`ProgramRunArgs`-returning `override_runtime_arguments`. The override returns a `TensorArgument` for **every**
io-tensor `TensorParameter` (`src`, `cos`, `sin`, `dst`) on every dispatch — the legacy override refreshed all
four addresses in every config (arg-slot rewrites `:958-969` and the `UpdateDynamicCircularBufferAddress` block
`:980-991` of the pre-port factory), so nothing is deliberately skipped. Decode additionally re-emits the two
token-derived, hash-excluded scalars (`cos_sin_start_id` on the reader, `cos_sin_offset` on the writer) per
core, over the same `compute_rotary_work_split` + `grid_to_cores` order as the miss path. Prefill re-emits no
kernel args, matching legacy.

### Device-op-class edits

- Pybind entry points removed: **none** (no pybound `create_descriptor` existed).
- Custom `compute_program_hash`: **left intact** at `device/rotary_embedding_device_operation.cpp:146-162`
  (confirmed untouched — TT_FATAL census and diff are clean outside the factory).
- No other device-op-class edits; `validate_on_program_cache_miss` / `compute_output_specs` /
  `create_output_tensors` / nanobind files are byte-identical.

### Open items

- No relaxation candidates observed (strict `TensorSpec` match; the custom hash is a tightening, per the audit).
- Concept fit was clean; the landed `kv_cache/update_cache` custom-concept port is a good precedent pointer for
  future ops with hash-excluded per-dispatch scalars.

## Handoff points

- **None requiring escalation.** No capitulations, no boundary-rule violations, no kernel-lib gaps, no pybind
  surface removed. (The multi-binding validator-rule doc gap is filed under Friction/Gaps; the framework itself
  had a legal shape for this op's construct, so it is a doc fix, not a framework gap.)

## Successes

- **Endpoint-assignment procedure & multi-binding guidance** (`port_patterns.md`, Two-toucher DFB / self-loop
  entries): the "re-derive, don't transcribe" instruction had me run the kernel-touch census myself, which is
  what made the verification-round validator failure quickly resolvable — the writer's role-free raw in-place
  write was already identified in the census as a distinct touch, so promoting it to the open PRODUCER side was
  a lookup, not a re-analysis. Applied at `device/rotary_embedding_program_factory.cpp` (writer decode bindings).
- **`opt_level` mechanical check** (recipe, Compiler options): the legacy factory sets no opt_level anywhere, so
  without the "absent line" warning the O3-on-compute requirement would have been invisible. Explicit O3 on all
  four possible compute KernelSpecs (`grep -n opt_level` pairs each with a construction site).
- **Style-A dropped-field check** (recipe, Compute kernels): fired exactly as described — the legacy factory
  resolves the TTNN config but copies only `math_fidelity`/`fp32_dest_acc_en` onto `ComputeConfigDescriptor`, so
  the port uses `to_compute_hardware_config` then resets `sfpu_precision_mode = Precise` /
  `double_buffer_dest = true` (the dropped fields' descriptor-default results). Verified both helpers are pure
  field reads (`compute_kernel_config.cpp:99-136`).
- **"Locate and confirm the op's tests" broad sweep**: primary coverage lives in
  `tests/tt_eager/.../misc/test_rotary_embedding.py`, not `tests/ttnn/unit_tests/` — the "don't assume a mirror
  path" warning was exactly right; there are no C++ gtests for this op.
- **Shared-kernel Caution rung 2**: the fork + pointer comment + kernel-role binding vocabulary flow worked
  cleanly, and the parallel `rotary_embedding_hf` port consumed the fork (its device runs built
  `rotary_embedding_single_tile_metal2` from this directory) without coordination beyond the convention.
- **kv_cache/update_cache as reference shape** (invoker-independent find via `create_program_artifacts` grep):
  a landed `CustomProgramSpecFactoryConcept` port with the same hash-excluded-decode-scalar structure, borrowed
  input DFB, and aliased interm pair — nearly every construct had a recipe-consistent precedent there.

## Friction

### Gaps

- **Self-loop set-equality rule is undocumented** — the cause of verification round 1 (653 decode failures).
  `ValidateProgramSpec` (`tt_metal/impl/metal2_host_api/program_spec.cpp:1441-1460`) enforces that once any
  kernel binds a DFB as both PRODUCER and CONSUMER, the producer and consumer *kernel sets* must be equal;
  `allow_instance_multi_binding` relaxes the per-node census to ">=1 per role" (`:1368-1377`) but does **not**
  lift set-equality. Neither the patterns catalog's multi-binding/self-loop entries nor the audit's CB-endpoints
  subject mentions this, and this op's audit brief prescribed the exact shape the validator rejects ("bind
  compute P+C and set the flag for the writer's extra consumer binding"). The legal shape follows from the
  endpoint-assignment procedure itself: a role-free co-toucher (here the writer's raw in-place write) takes the
  open PRODUCER label so both kernels appear on both sides. Suggested doc fix: add to the multi-binding entry —
  "when a self-loop participant shares a multi-bound DFB with another kernel, the other kernel must be bound on
  *both* sides; a role-free touch supplies the missing side. If the co-toucher genuinely has no touch of the
  missing kind, there is no legal shape — stop and report."
- **On-Gen1 inertness of multi-bound DFB config is only discoverable in impl comments**: that role labels /
  risc masks / `tensix_scope` are inert for a multi-bound Gen1 DFB (plain shared circular buffer) is stated at
  `program_spec.cpp:2876-2882` and `dataflow_buffer.cpp:1870-1889` (WH/BH early-return), not in any header or
  doc. A porter validating a binding-shape change needs this to know the cosmetic labels are safe.
- **Runtime-variable DFB ids**: kernels that select a buffer at runtime (`updated_cos_dfb` flipping between
  `dfb::cos` and `dfb::retilized_cos`; helper functions taking ids as `uint32_t` parameters) port cleanly via
  `DataflowBuffer(uint16_t logical_dfb_id)` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:113`), but no doc
  mentions this constructor's sanctioned use for path-selected buffers. Worth a whitelist/patterns line.

### Confusion

- **cb-name sweep hit-class**: the anti-pattern grep initially returned ~60 hits, nearly all legacy *local
  variable* suffixes (`in_cb`, `cos_cb`, `*_cb_data_format`) and "CB" in comments rather than spec-name strings.
  A one-line note in the self-audit that locals/comments are the dominant hit class (and a mechanical
  `_cb`→`_dfb` rename suffices) would set expectations.
- **Verification round 2's 4 residual failures were environment, not port**: a transient stale read at the
  first four decode JIT compiles of the multi-tile interleaved reader paired *legacy source text* with this
  port's generated args — a pairing that never existed on disk (all source-tree and installed copies verified
  converted; identical JIT build keys succeeded seconds later in the same run; the shared JIT cache still held
  17 baseline-era legacy build dirs, and the sibling port's device runs share that cache). **Resolved:** purging
  the JIT cache (`~/.cache/tt-metal-cache/<key>`) and rerunning with exclusive device access gave 763/763.
  Lesson for the docs/workflow: when two ports share a tree and device, serialize device runs *and* treat the
  persistent JIT kernel cache as shared mutable state — purge it between eras (baseline / port) or at least
  before adjudicating a compile failure that cites content not on disk.

## Open items for downstream

- **Shared kernel touch (coordination signal + sunset list):**
  - Kernel: `device/kernels/compute/rotary_embedding_single_tile.cpp` (lent).
  - Rung taken: **created the fork** — `device/kernels/compute/rotary_embedding_single_tile_metal2.cpp`, with
    the pointer comment landed in the legacy original (that original's only change). Fork binding vocabulary
    (kernel-role names): dfb accessors `in`, `cos`, `sin`, `trans_mat`, `rotated_in_interm`, `cos_interm`,
    `sin_interm`, `out`; decode surface behind `DECODE_MODE`: `untilized_cos`, `untilized_cos_sync`,
    `untilized_sin`, `untilized_sin_sync`, `retilized_cos`, `retilized_sin`; named CTA `num_rows`. No
    tensor/semaphore bindings.
  - Remaining unmigrated consumer of the legacy original: `rotary_embedding_hf` (`RotaryEmbeddingHfMultiCore`,
    legacy factory). Its Metal 2.0 port (in flight in parallel) reuses the fork (rung 1); once its legacy
    factory is gone, the legacy original can sunset (fork takes over the name).
- **Findings preserved, not fixed (op-owner review candidates):**
  - The promised single-tile dtype constraint is absent: the comment at
    `device/rotary_embedding_device_operation.cpp:44-49` says the `Wt==1` path should constrain input/cos/sin
    to bfloat16 (WH LLK corrupts bfp8-input @ bf16-trans_mat matmul packs), but no `TT_FATAL` follows, and the
    factory explicitly handles a Bfp8_b trans_mat. Either the comment is stale or the check was dropped
    (carried forward from the audit's Misc anomalies; preserved as-is).
  - g1/g2 compute-config asymmetry (multi-tile): group 1 runs the default-constructed config while group 2
    carries the caller's `math_fidelity`/`fp32_dest_acc_en` — deliberate legacy parity per the in-code comment,
    reproduced exactly (`ComputeGen1Config{}` vs translated config in the port). The two core groups of one
    dispatch run at different fidelity; latent behavioral inconsistency for the ops team.
  - Config-dead legacy args resolved structurally by the named schemas: `start_row_id` is not in the decode
    schemas, and the writer's `dst`/`start_id` are not in the OUT_SHARDED schema — the kernel-side reads are
    `#ifdef`-gated in step. Zero functional change (the values were unread), but reviewers should know the
    per-config schemas differ where legacy padded unread slots.
- **Test-coverage note:** the single-tile (`Wt==1`) path is exercised only by
  `test_rotary_embedding_decode_program_cache_reuse` (X=32 rows at
  `tests/tt_eager/python_api_testing/unit_testing/misc/test_rotary_embedding.py:480-483`); no single-tile
  prefill or interleaved-only pytest exists. Verification passed with what exists; a single-tile prefill test
  would close the gap.
- **RTA-really-CRTA cleanup candidates (not converted — RTA→CRTA changes dispatch semantics):** writer `Wt` and
  `Wbytes` (and decode `cos_sin_offset`/reader `cos_sin_start_id`) are set to the same value on every node;
  a later, separate pass could promote them to common runtime args.
