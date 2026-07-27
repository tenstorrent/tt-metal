# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/moreh/moreh_fold`

## Outcome

**PORTED** — the single `MorehFoldOperation` factory (`fold_program_factory_rm.cpp`) is converted to
`MetalV2FactoryConcept` (`MultiCore::create_program_artifacts`). Both kernels (`reader_fold_rm.cpp`,
`writer_fold_rm.cpp`) are converted. Build and test verification are the orchestrator's (not run here).

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept`, as the audit specified. The legacy op used the `HasDirectDescriptor` shape
(`create_descriptor` placed directly on `MorehFoldOperation`, no `program_factory_t`). The port introduces the
standard variant wiring: a nested factory struct `MorehFoldOperation::MultiCore` with
`create_program_artifacts`, `using program_factory_t = std::variant<MultiCore>`, and a `select_program_factory`
returning `MultiCore{}`. (`create_program_artifacts` is only detected by the framework as a variant alternative,
so the variant is required even for a single-factory op.)

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op had no custom hash).
- Pybind entry points removed: **none** — `fold_nanobind.cpp` binds only the `ttnn::moreh_fold` free function via
  `ttnn::bind_function`; there was no `create_descriptor`/device-op pybind surface to remove.

The only device-op-class edits were the sanctioned factory wiring (variant + `select_program_factory`) and dropping
the now-unused `<tt-metalium/program_descriptors.hpp>` include (added `ttnn/metal_v2_artifacts.hpp`, `<variant>`).
`validate_inputs` / `validate_on_program_cache_miss` / `compute_output_specs` / `create_output_tensors` are untouched.

### Open items
- **Relaxation candidates:** none applied (audit: `TensorParameter relaxation = none`). The dated 3rd-arg triage
  (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`) lists `moreh_fold` as Class 1 (`dynamic_tensor_shape`),
  but the audit reclassified it Class 2 (drop the arg, no relaxation) because the op has no custom hash and the full
  shape is hashed. The port followed the audit: dropped the 3rd arg, added **no** relaxation. If the triage/readiness
  owner later wants a width-relaxed hash for cache reuse, that is a separate custom-hash change, not this port.

## Handoff points

none — no capitulation, no out-of-directory changes, no `sem::`/`tensor::` boundary violations, no kernel-lib gaps.
Both kernels are op-owned and were edited in place (no fork needed; not shared with peer ops).

## Successes

- **CB-endpoint self-loop patterns** ([Sync-free and single-ended CBs → self-loop DFB], [Self-loop DFB binding]).
  The reader is the sole toucher of both `input_cb` (c_0, full FIFO on one kernel) and `scratch_cb` (c_1, raw
  `get_write_ptr` peek only). The recipe's one-toucher → self-loop disposition applied directly: each bound
  PRODUCER+CONSUMER on the reader with a shared accessor name, so the kernel construction (`DataflowBuffer
  input_dfb(dfb::input)`) is unchanged. `output_cb` (c_16) is a clean legal 1:1 (reader PRODUCER, writer CONSUMER).
- **Conditional / optional DFB binding** (`port_patterns.md` — Conditional / optional DFB bindings). `scratch_cb`
  exists only when `(src_is_dram && page % dram_alignment != 0) || is_blackhole`. Binding it conditionally on the
  host + emitting a matching `HAS_SCRATCH_CB` define + `#ifdef`-gating the kernel-side construction and the
  two-step-read block is exactly the documented shape. This avoided the "bind unconditionally to dodge name lookup"
  trap the recipe warns against (which would allocate scratch L1 the legacy op didn't).
- **`AddRuntimeArgsForNode`** (migration guide — ProgramRunArgs). Kept the legacy node-first per-core RTA loop
  verbatim and let the helper transpose into the name-first table, avoiding a manual loop inversion.

## Friction

### Gaps
- **Brief/audit says the page-size RTAs are dead; they are not.** Both `METAL2_PORT_BRIEF.md` and
  `METAL2_PREPORT_AUDIT.md` state: "the `input_cb_page_size` / `output_cb_page_size` RTAs that fed the dropped 3rd
  arg are also dead — remove them." Per the recipe's instruction to re-derive from the kernel census, this is
  **incorrect**:
  - `input_cb_page_size` (reader RTA[15]) is the NOC transfer size at `reader_fold_rm.cpp:96` (direct read into
    `input_dfb`) and `:113` (scratch→input copy), not only the 3rd constructor arg.
  - `output_cb_page_size` (writer RTA[1]) is the NOC write size at `writer_fold_rm.cpp:31`.
  Dropping them would have broken both kernels. I kept both as named RTAs and dropped **only** the redundant 3rd
  constructor argument and the buffer-address RTAs. The 3rd-arg *drop* itself (removing the third
  `TensorAccessor(...)` argument) was correct and mechanical, as the brief intended — the error is only in the
  claim that the underlying RTA value becomes dead. Suggest the audit/brief distinguish "drop the 3rd argument"
  from "the RTA feeding it is dead": here they are the same value used in two places, so only the argument slot drops.

### Confusion
- **`create_program_artifacts` is not detected directly on the op struct.** `MetalV2FactoryConcept` is only
  checked as a `program_factory_t` variant alternative (`operation_concepts.hpp`), and `DeviceOperationConcept`
  accepts either `HasDirectDescriptor` *or* a `program_factory_t` variant. A single-descriptor op (like this one,
  where `create_descriptor` sat directly on the op) therefore cannot "just rename the method" — it must grow a
  `program_factory_t` variant + factory struct. The recipe's atomic-unit discussion assumes a factory struct
  already; a one-line note that direct-descriptor ops must introduce the variant would have saved a detour.

## Open items for downstream

- **Cross-op kernel touches:** none — both kernels are op-owned; edited in place; no forks.
- **Dead code left as-is (route to ops team; not touched by the port, per scope discipline):**
  - `reader_fold_rm.cpp:15` `int i{0};` — unused local (already flagged in the audit's Misc anomalies).
  - `reader_fold_rm.cpp:91` `uint32_t l1_write_addr = input_dfb.get_write_ptr();` — unused local.
  - reader RTA `output_cb_page_size` (host `aligned_output_cb_page_size`) — declared/read by the reader but never
    used in the reader body; kept as a faithful 1:1 named RTA (writer *does* use its own `output_cb_page_size`).
  - `reader_fold_rm.cpp:78-82` `if (lh < 0 ...)` / `if (lw < 0 ...)` on `uint32_t` — the `< 0` half is always
    false (already flagged in the audit's Misc anomalies).
- **Obsolete comment removed:** the two-line comment above each `TensorAccessor` (reader ~47-48, writer ~22-23)
  described the now-removed 3rd (page-size) argument's override behavior. Since Metal 2.0 supplies the aligned page
  size implicitly and the argument is gone, the comment documented deleted behavior and would have been actively
  misleading if kept; removed with the argument (per kernel-side whitelist rule 8's "opposite trap" — the concept
  it documented does not survive).
- **RTA→CRTA candidate (separate cleanup, not this port):** every reader/writer RTA except `start_id` and
  `num_units_per_core` has the same value on every node (N, C, H, W, kernel/stride/pad/dilation, page sizes,
  `aligned`). These are morally CRTAs and would dispatch more efficiently as `common_runtime_arg_values`. Not
  converted here (RTA→CRTA changes dispatch semantics; the recipe routes it to a later name-first pass).

## Test command(s) — verification is the orchestrator's

No C++ gtests exist for `moreh_fold`. The no-regression baseline is the nightly pytest:

```bash
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_fold.py -x -v
```

Covers float32 and bfloat16 across 2D/3D inputs and padding/dilation/stride variants (11 shapes × 2 dtypes). All
should pass post-port (no functional change). (`tests/ttnn/unit_tests/operations/conv/data_movement/test_fold_op.py`
is the *unrelated* `ttnn.fold` op — not moreh_fold — and is out of scope.)

**Build (orchestrator):** `./build_metal.sh --build-tests`
