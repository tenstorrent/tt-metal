# QUASAR_UPLIFT_REPORT — `ttnn::prim::binary_ng` (BinaryNgDeviceOperation)

- **Op:** `ttnn::prim::binary_ng` — the device op behind `ttnn.add` / `ttnn.multiply`
  (entry points in `ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp`, all routed through
  `ttnn::detail::invoke_binary_ng` → `ttnn::prim::binary_ng`).
- **Op directory:** `ttnn/cpp/ttnn/operations/eltwise/binary_ng/`
- **Driving tests:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_add.py`,
  `models/experimental/llama32_1b_quasar/tests/graph_ops/test_multiply.py`
  (bf16/bf8_b TILE, interleaved-DRAM and width-sharded-L1 cases; `ttnn.multiply` cases also
  carry `input_tensor_a_activations=[SILU]` — same `binary_ng` path, activations are fused
  into the compute kernel, not a separate op).
- **Recipe:** `docs/source/ttnn/ttnn/ai/quasar_porting.md` (+ `ai/audit/quasar_audit.md`,
  `ai/audit/metal2_audit.md` under `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`).
- **Date:** 2026-09-01

## Status: RED — Not Metal 2.0 on Gen1 yet

This is the recipe's first RED-stop condition (`quasar_porting.md` §1 / "RED status"):
the factory is still `create_descriptor` → `ProgramDescriptor`, not
`create_program_artifacts` → `ProgramArtifacts`. Per the recipe, the uplift stops here;
the Metal 2.0 port (`ai/port/metal2_port.md`, gated by `ai/audit/metal2_audit.md`) must
happen first. **No Metal 2.0 port was performed in this session** (explicitly out of scope),
and **no source file was changed** — this report is the only artifact.

### Gate evidence (per `quasar_porting.md` §1)

The gate requires BOTH: (a) factory on `create_program_artifacts`/`ProgramArtifacts` with
`dfb::`/`args::`/`tensor::`/`scratch::` bindings, and (b) kernels on the device-2.0 kernel
APIs with those binding tokens. Findings:

1. **Factory is `ProgramDescriptor`-era (fails the gate).**
   - `device/binary_ng_device_operation.hpp:133–150` — the single `ProgramFactory`
     (the only `program_factory_t` variant, so every case the driving tests hit —
     interleaved and sharded — goes through it) declares
     `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` plus
     `override_runtime_arguments(...)`.
   - `device/binary_ng_program_factory.cpp:833` —
     `ProgramDescriptor BinaryNgDeviceOperation::ProgramFactory::create_descriptor(...)`,
     building `CBDescriptor`s (e.g. lines 1050–1141, keyed by `tt::CBIndex`) and
     `KernelDescriptor`s (lines 1192, 1334, 1355) with positional runtime-arg vectors and
     `TensorAccessorArgs` compile-time args.
   - `grep -rn "create_program_artifacts\|ProgramArtifacts" ttnn/cpp/ttnn/operations/eltwise/binary_ng ttnn/cpp/ttnn/operations/eltwise/binary` → **zero hits**.

2. **Kernels are Device 2.0 but not Metal 2.0 (also fails the gate).**
   The kernels under `device/kernels_ng/` and `device/kernels/` already include the
   device-2.0 headers (`api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`,
   `api/dataflow/dataflow_buffer.h`, `api/compute/eltwise_binary.h`, …) — i.e. the earlier
   **Device 2.0** kernel-API migration is done — but they still use the pre-Metal-2.0
   argument/CB model:
   - positional `get_arg_val<uint32_t>(i)` throughout (e.g.
     `device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp:13–23`);
   - `tt::CBIndex`-keyed buffers (e.g. `device/kernels/compute/eltwise_binary.cpp`,
     whose own comment notes `CircularBuffer` is still used for the shared
     `preprocess_*_impl` helper call sites);
   - **zero** occurrences of `experimental/kernel_args.h`, `dfb::`, `args::`,
     `tensor::`, or `get_arg(` in either kernel tree.

Per `metal2_audit.md`, the op is "on the `ProgramDescriptor` API" — the *prerequisite* for
a Metal 2.0 port — and Device 2.0 kernel migration is largely done, so it is a candidate
for the Metal 2.0 port track; but the port itself has not happened, so a Quasar uplift
cannot start.

## Files changed

**None.** No source, kernel, factory, or build file was modified. The working tree diff for
this op is exactly this report.

## §7–§8 gotchas: applied vs considered

- **Applied:** none — the RED gate stops the uplift before the audit's reactive fixes
  become applicable, and the recipe forbids manufacturing changes.
- **Considered but out of scope until the op is Metal 2.0** (noted for the eventual uplift,
  from a read of the current legacy code — these are observations, not applied fixes):
  - **Semaphores (quasar_audit.md check 2):** the op creates **no semaphores at all**
    (no `SemaphoreDescriptor`/`CreateSemaphore`/non-zero `initial_value` anywhere in
    `device/*.cpp`) — no non-zero-init-semaphore blocker.
  - **DM self-loop / sync-free CBs (quasar_audit.md check 1):** not audited to conclusion —
    the CB→DFB classification (`cb_dfb_quasar_audit_helper.md`) is defined over the
    *post-Metal-2.0* DFB bindings, which don't exist yet. Must be run after the M2 port.
    Note the op does allocate scalar/preprocess intermediate CBs whose class will need
    that audit.
  - **uint16/uint32 device formats (§7):** the driving test cases are BFLOAT16/BFLOAT8_B
    only; `binary_ng` does have int32/uint dtype paths (`test_binary_int32.py` etc.) that
    the eventual uplift must check against Quasar's Int32-only support.
  - **`fifo_page_size` staleness (§5/§8.3), implicit sync (§7), `hw_config`/`unpack_modes`
    (§4):** all Metal-2.0-form concerns; nothing to check in the legacy form.

## Deferred / follow-up items

1. **Metal 2.0 port of `binary_ng` (blocking).** Run the pre-port feasibility audit
   (`ai/audit/metal2_audit.md` → `METAL2_PREPORT_AUDIT.md` + `METAL2_PORT_BRIEF.md` in this
   directory), then the port (`ai/port/metal2_port.md`). Favorable signs from this gate
   check: kernels already Device 2.0, single `ProgramFactory` variant, no semaphores.
   Complexity signs: very large factory (~1500 lines) with a shared runtime-arg builder
   used by both `create_descriptor` and `override_runtime_arguments`, a custom
   `compute_program_hash`/`to_hash`, two kernel trees (`kernels/` legacy where-op era +
   `kernels_ng/`), sharded + broadcast + fused-activation (SILU) variants — all exercised
   by the llama32_1b graph traces.
2. **Re-run this Quasar-uplift audit after the M2 port lands** — including the
   `cb_dfb_quasar_audit_helper.md` per-buffer classification and the Int32-only format
   check for the integer dtype paths.
3. No out-of-op-directory edits were needed or made; nothing else to defer.

## WH/BH parity claim (structural)

**Zero-diff parity.** No file was changed (this report aside), so WH and BH take exactly
the code path they took before this session — there is nothing to regress. No device run
was performed (per the recipe, the user runs all builds/tests).

## Test commands (for the user to run)

Quasar graph-trace tests (the driving workload; Quasar emulator per the craqsim runbook):

```bash
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_add.py -v
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_multiply.py -v
```

BH/WH parity (op's existing unit suite — should be a no-op check since the diff is empty):

```bash
pytest tests/ttnn/unit_tests/operations/eltwise/test_add.py -v
pytest tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py -v
pytest tests/ttnn/unit_tests/operations/eltwise/test_binary_ng_program_cache.py -v
```

---
*This report is uncommitted by design; delete it before any merge (recipe "Deliverable" section).*
