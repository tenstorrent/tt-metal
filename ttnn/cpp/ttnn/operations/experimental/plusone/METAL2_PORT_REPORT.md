# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/experimental/plusone`

## Outcome

**PORTED** — the single factory (`PlusOneProgramFactory`) converted from the `descriptor`
(`ProgramDescriptor`) concept to `MetalV2FactoryConcept` (`create_program_artifacts`).
Build and test verification are the orchestrator's (this porter did not build or run tests,
per orchestration constraints).

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — as chosen by the audit. `PlusOneProgramFactory::create_program_artifacts`
returns `ttnn::device_operation::ProgramArtifacts{ spec, run_params }` (no op-owned tensors).
Single-program, stamped identically across the op's node set.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op never had one).
- Pybind entry points removed: **none** (`plusone_nanobind.cpp` binds only the free function `plus_one`; no `create_descriptor` binding existed). The device-operation class (`plusone_device_operation.cpp/.hpp`, `plusone_device_operation_types.hpp`) was **not** touched — `validate`/`compute_output_specs`/`create_output_tensors` and `program_factory_t = std::variant<PlusOneProgramFactory>` are unchanged; the framework dispatches on the new concept automatically.

### Open items
- **Tensor-matching relaxation:** none applied; none needed. Kernel sources contain no `ArgConfig::Runtime*` usage.
- The op would not benefit from any concept capability not on `MetalV2FactoryConcept` today.

## Handoff points

None. The port is self-contained within the op directory; no out-of-op kernel edits, no
kernel-lib gaps, no framework gaps, no removed pybind surface.

## Successes

- **Sync-free single-toucher CB → self-loop DFB** ([port_patterns.md — Sync-free and single-ended CBs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)). `c_0` is used purely as an address source (`get_write_ptr()`, no FIFO ops); the pattern's one-toucher rule gave the exact disposition — bind the reader PRODUCER **and** CONSUMER of `IN0` under one accessor name `in0` (`device/plusone_program_factory.cpp:88-92`). No multi-binding flag needed; the DM self-loop is Gen1-legal.
- **Borrowed-memory DFB** ([migration_guide.md — DataflowBufferSpec](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#dataflowbufferspec)) and the `quasar/fold` shape reference. The sharded config's legacy `.buffer = src_buffer` maps cleanly to `borrowed_from = INPUT` (`device/plusone_program_factory.cpp:78`); `fold` confirmed that `borrowed_from` alone satisfies the "every TensorParameter needs a binding" validator rule (no `TensorBinding` required on the sharded path).
- **Conditional binding via `#define`** ([port_patterns.md — Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings), whitelist rule 6). The legacy `src_is_dram` CTA gated whether the `TensorAccessor` is used; since the accessor is now a *conditional binding* (present only on the DRAM path), the condition became a host-emitted `SRC0_IS_DRAM` define and the kernel `#ifdef`-gates the `TensorAccessor(tensor::input)` construction and both NoC transfer blocks (`device/kernels/reader_plusone_interleaved.cpp:25-27,39-42,56-59`). Following the rule avoided the trap of an unconditional `tensor::input` reference that would fail name-lookup on the non-DRAM builds where the binding is absent.

## Friction

### Gaps
- None blocking. The recipe + patterns catalog + `fold` covered every construct.

### Confusion
- **Three host-known configs, one kernel source — is this "runtime kernel-source selection"?** The op serves DRAM / sharded / L1-interleaved from one kernel file, differing only in host-side bindings/defines decided at factory time. The recipe's "runtime kernel-source selection" discussion (Legacy Inventory) is about a factory picking different *source files*; here it is one source with config-varying bindings, so it is a single `KernelSpec` — not a multi-source atomic unit. This resolved quickly but the distinction between "host-time config branch on one source" and "runtime source selection" could be called out more explicitly for single-kernel ops.

## Open items for downstream

- **Cross-op kernel touches:** none — the sole kernel is op-owned.
- **Test coverage gap (surfaced, not acted on):** `tests/ttnn/unit_tests/operations/transformers/test_plus_one.py` creates inputs via `ttnn.from_torch(..., device=device)`, i.e. **DRAM-interleaved** only. The **sharded (borrowed-DFB)** path and the **L1-interleaved anomaly** path (audit Misc anomalies — kernel increments uninitialized scratch, preserved verbatim) are **not** exercised by the unit tests. The port preserves all three behaviors, but only the DRAM path has a regression net. Flagging for the op/test owners.
- **Pre-existing anomaly (unchanged):** an L1-*interleaved* (non-DRAM, non-sharded) input is still "unhandled" — the kernel increments uninitialized scratch and never touches the input. Preserved exactly per the brief; this remains an ops-team concern, not port work.

## Test commands

The confirmed no-regression baseline is the op's unit-test file (family slug `transformers`, not `plusone`):

```bash
# Python pytest (primary coverage — dtype/layout/subdevice/neg-entries)
pytest tests/ttnn/unit_tests/operations/transformers/test_plus_one.py -x -v
```

No C++ gtest exercises this op (the `unit_tests_ttnn` filter `*PlusOne*` matches nothing).
A sweep exists at `tests/sweep_framework/sweeps/model_traced/plus_one_model_traced.py` (not part
of the unit baseline).

**Build (orchestrator):** `./build_metal.sh --build-tests`
**Test verification is the orchestrator's** — this porter did not build or run anything.

## Things I was unsure about (for the orchestrator/reviewer)

1. **`data_format_metadata` on a DM-only DFB.** I set it (`= input_cb_data_format`) to mirror the legacy CB's `data_format` and match the `quasar/fold` DM-borrowed-DFB shape. The header says it is only *required* for compute-bound DFBs; here it is harmless metadata. If the validator objects on any path, it can be dropped with no functional change.
2. **Reader `hw_config`.** Legacy used `ReaderConfigDescriptor{}` (the reader default: RISCV_1 / NOC_0 / DEDICATED), so I used the arch-agnostic `ttnn::create_reader_datamovement_config(arch)`. No custom triple.
3. **Unit tests cover only the DRAM path** (see Open items). The sharded borrowed path + L1-interleaved anomaly compile-and-bind but have no correctness net in the confirmed test set — worth a manual sharded check if the orchestrator has an easy way to construct an L1-sharded INT32/UINT32 row-major input.
