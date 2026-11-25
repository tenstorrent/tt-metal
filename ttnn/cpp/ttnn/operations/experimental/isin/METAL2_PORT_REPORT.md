# Port Report — experimental/isin

## Outcome

**`PORTED`**: `IsInProgramFactory`, the op's only factory, now satisfies `ProgramSpecFactoryConcept`. Both test files pass on Blackhole (p150b): 24/24 before and after the port, and again under Watcher with every Metal 2.0 legality check forced on. A 40-case capture, including 14 isin program-cache-hit dispatches, is byte-identical to the legacy build, and so is its program-cache entry trace.

## Provenance

- **Recipe docs (this port):** `` c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form ``
- **Audit docs (inherited):** `` c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form `` (audit 2026-11-03; readiness-sheet row 191 reconciled 2026-11-25)
- **Code base:** `233a155fba1` (`origin/main` `86f63a50136` merged into the audit branch)

## Verification

| Run | Build | Result |
|---|---|---|
| Baseline: `tests/ttnn/unit_tests/operations/experimental/test_isin.py` + `tests/ttnn/nightly/unit_tests/operations/experimental/test_isin.py` | legacy | 24 passed |
| Same two files | ported | 24 passed |
| Same two files, `TT_METAL_WATCHER=10`, legality checks forced (below) | ported + temporary scaffolding | 24 passed; Watcher log clean |
| Bit-exact capture (40 cases), plain and under Watcher + forced checks | legacy vs ported | 40/40 byte-identical (dtype, shape and bytes); program-cache entry count identical after every call (48 final) |

The test set comes from a `find tests -iname '*isin*'` sweep; every other hit was an `isinf` false positive. There are no C++ gtests for this op.

The capture covers int32 RM/TILE, uint16, uint8, bfloat16 and float32. It includes both `invert` values, multi-dimensional and 10-D shapes, one-subchunk and multi-subchunk sizes (80000 int32; 160000 uint16 / bf16; 210000 uint8), and multi-core work splits. It also makes six same-spec, different-data call groups of three calls each. In those groups, 14 dispatches hit the isin program cache. They come from separately allocated tensors, so the refreshed bindings are exercised.

**Legality checks.** In this tree the forcing step from the recipe doesn't apply as written (see Friction). `ValidateProgramSpec` is unconditional, and `set_program_run_args` defaults to `ValidationMode::FULL`. Only the cache-hit `update_tensor_args` is gated, by TTNN's `validate_program_args` (default off). For one pass I forced `validation = ValidationMode::FULL` in `set_program_run_args`, `update_tensor_args` and `update_program_run_args`, and added `log_warning` markers in those (except `update_program_run_args`) and after `ValidateProgramSpec`. The markers fired in that run: test log 47 × `ValidateProgramSpec`, 47 × `set_program_run_args`, 8 × `update_tensor_args`; capture log 45 / 45 / 26. The scaffolding was then reverted (`git checkout` of the two `tt_metal/impl` files), and the tree was rebuilt before committing. Nothing outside the op directory is in the port diff.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. `create_descriptor` was replaced by `create_program_artifacts` (`device/isin_program_factory.hpp:13-14`), which returns one `ProgramSpec` + `ProgramRunArgs`. There are no op-owned tensors and no override.

### Device-op-class edits
- Pybind entry points removed: none (`isin_nanobind.cpp` binds only the free function).
- Custom compute_program_hash: none. `isin_device_operation.{hpp,cpp}` are untouched.

### Open items
- No relaxation candidates. Every tensor is flattened to a 1-D row-major page, so strict matching is the only reasonable fit.

## Handoff points

none. No capitulation, no out-of-op call site needed a `sem::` / `tensor::` handle, no kernel-lib gap, and no pybind surface was removed.

## Successes

- **The brief's CB endpoint census held up on re-derivation** (`port_patterns.md`, *Two-toucher DFB → assign 1P+1C*, step 3). I recounted it from the kernel bodies, and it matched: `elements` and `test_elements` have one toucher each (self-loop on the reader, `isin_program_factory.cpp` reader `dfb_bindings`), and `output` is 1P+1C. The validator accepted it as written.
- **Token form for the template-argument `get_dataformat`** (`cb_dfb_api_whitelist.md` §A; audit Recipe note 4). `using elements_number_type = std_type_t<get_dataformat(dfb::elements)>;` (`isin_reader.cpp:88`) compiled first time. The DM-kernel token carries the format because the host sets `data_format_metadata` on every DFB (`make_dfb`).
- **Ground-truth headers first** (recipe, *Read this first*). `dataflow_buffer.h` / `dfb_binding_token.h` settled the helper signatures: a function parameter must spell `DFBBindingToken<P, C>`. The Gen1 `DataflowBuffer` constructor in `internal/tt-1xx/dataflow_buffer.inl` is a view over the shared `LocalCBInterface`, which confirms that the per-call wrapper in `load_to_dfb` / `write_to_dram` behaves exactly like legacy.
- **Shared CTA decoder.** Following the brief's heads-up, I reshaped `IsInCTAs` to hold only the six scalars (`isin_common.hpp:55-74`). Both kernels receive the same six named CTAs, just as legacy gave both the full positional list. Each kernel constructs only the `DataflowBuffer` / `TensorAccessor` it binds, so the shared header names no binding token.

## Friction

**Gaps**
- **Header paths moved** (recipe *Read this first* and *Go to the headers first*; migration guide *Header Files*). `tt_metal/api/tt-metalium/experimental/metal2_host_api/` now holds only `advanced_options.hpp`, `prefetcher_pipe_parameter.hpp` and `sram_arena_parameter.hpp`. The spec headers are in `tt-metalium/workload/specification/`, run-args in `tt-metalium/workload/run_args/`, and `Program` in `tt-metalium/workload/program.hpp`, all in namespace `tt::tt_metal` (not `::experimental`). The migration guide's `#include` list and its "symbols live in `tt::tt_metal::experimental`" line are stale. In practice the factory needed only `ttnn/metal_v2_artifacts.hpp`.
- **The validation-forcing step no longer matches the code** (recipe *Ensure the Metal 2.0 host-side legality checks are enabled*; self-audit item "No forced-legality scaffolding"). `grep 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp` returns nothing. The parameter is now `ValidationMode` (`workload/run_args/validation_mode.hpp`). `BuildProgramFromSpec` validates unconditionally (`program_spec/program_spec.cpp:32`). `set_program_run_args` / `update_program_run_args` default to `FULL`. The TTNN adapter validates the miss path always and gates only the hit path (`update_tensor_args`) on `ttnn::CONFIG` `validate_program_args` (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:55-57`, `:1325-1330`). So `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'` would turn on the remaining check without any scaffolding. A marker is still needed to *prove* it, because the validators don't log.
- **DM helper signature** (recipe *Hardware configuration*). The recipe writes `create_reader_datamovement_config(device->arch())`. The helper now takes no arch argument, only `bool disable_dfb_implicit_sync_for_all = false` (`ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp`). I called `ttnn::create_reader_datamovement_config()` / `create_writer_datamovement_config()`. Their Gen1 values (RISCV_1/NOC_0 and RISCV_0/NOC_1, dedicated) equal the legacy `ReaderDataMovementConfig` / `WriterDataMovementConfig` (`tt_metal/impl/kernels/kernel_types.cpp:11-40`).

**Confusion**
- **The scaffold self-audit grep false-fires on this branch layout.** `git diff $BASE | grep -nE 'METAL2_CHECKS_FORCED|DO NOT COMMIT'` hits the recipe's own text, because the porting branch carries `docs/.../metal_2.0/` and `$BASE` (the merge-base with `origin/main`) predates it. Scoping it as `git diff $BASE -- ':!docs'` would avoid that. The companion `git diff --name-only $BASE | grep '^tt_metal/'` check came back empty, as it should.
- **No step covers a GREEN reached by a sheet fix.** The audit doc describes RED → (fix) → re-audit, but not "the sheet owner fixed the row; the code is unchanged". I amended `METAL2_PREPORT_AUDIT.md` in place: the status rows, Result and Gate detail went GREEN with the 2026-11-25 row cited, and the 2026-11-03 cells were kept as superseded history. I then issued the brief from the audit's existing Port-work and Heads-ups sections.

## Open items for downstream

- **Shared kernel touches:** none. `isin_reader.cpp` / `isin_writer.cpp` / `isin_common.hpp` are op-owned and bound only by this factory. They were converted in place; no fork was needed.
- **Quasar-uplift debt (Gen1-only shapes in this port):**
  - DM self-loops on `elements` and `test_elements` (the reader is bound as PRODUCER + CONSUMER).
  - Token-form `get_dataformat(dfb::elements)` at `isin_reader.cpp:88`. The token's `uint32_t` / metadata path is Gen1-only.
  - `load_to_dfb` / `write_to_dram` construct a fresh `DataflowBuffer` per call while `kernel_main` holds another over the same DFB (`isin_common.hpp:76-124`, `isin_reader.cpp:96-98`). This is harmless on Gen1, but on Quasar DFBs carry per-object state.
- **Comments removed with their subject.** Two comments went away: "Compile-time args 0-2 are unused placeholders … TensorAccessorArgs at index 11" (`isin_common.hpp`, legacy `:54-55`) and "Slots 0-2 stay unused …" (`isin_program_factory.cpp`, legacy `:75-76`); both slots no longer exist. The "Third argument page_size … may be stale on program cache hits" comments (legacy `isin_reader.cpp:95-96`, `isin_writer.cpp:17-18`) went with the dropped 3rd argument. The audit had already flagged them as inaccurate.
- **Kept guards.** The three `TT_FATAL(*_buffer != nullptr)` guards (`isin_program_factory.cpp:51-53`) are kept verbatim, although their `Buffer*` locals now serve only those checks; the `TT_*` census per file is unchanged. Whether to restate them on the `MeshTensor` is an ops-team call.
- **Op findings carried from the audit, still present (not fixed, per scope):** the dead-but-hashed `assume_unique` attribute; the barely-validated preallocated `out` (only its dtype is checked, but the kernels assume a 1-D single-page interleaved RM output); and the legacy writer 3rd arg's wrong units. That 3rd arg is now gone with the port: the writer's accessor page size comes from the output `TensorSpec`, so it is correct by construction, and it is still unused because every access is `page_id 0`.
- **Test coverage:** no test passes `out=` (the audit's preallocated-output finding), and no test runs a float32 input. The capture script covered float32 on both builds, and the outputs were bit-identical.
