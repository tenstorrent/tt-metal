# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/matmul`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Port scope: ONE factory — `MatmulMultiCoreProgramFactory`.** The op has eight factories across two
DeviceOperations; only this one was audited and only this one is cleared. Two of the others are
already `no` on the readiness sheet. Do not widen.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A — no site)

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the factory ports to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `create_descriptor` returning a `ProgramDescriptor`,
  `device/factory/matmul_multicore_program_factory.hpp:14`.
- **Op-owned tensors:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** (base). The factory has no
  `override_runtime_arguments`, so the framework refreshes tensor bindings on cache hit and you
  write one method. Do **not** add an override.
- **Custom `compute_program_hash`:** none framework-visible — the op uses the default reflection
  hash. You will find a `compute_descriptor_program_hash` helper at
  `device/matmul_device_operation.hpp:50` with a comment explaining it is *deliberately* not named
  `compute_program_hash`, plus a pybind that exposes it under that name. **Leave all of it alone** —
  it is not a custom hash and it is not yours to touch.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none`
  `TensorParameter relaxation` · `get_dynamic_runtime_args`. A pybound `create_descriptor` **is**
  present — it does not gate, and removing it is port work (below).

## Construct — to do

**Tensor bindings** (per binding) — three, all **Case 1** (base fed into a `TensorAccessor`):

- `in0` — **Case 1** → express as `TensorParameter` / `TensorBinding`; kernel uses
  `TensorAccessor(tensor::in0)`. Drops reader RTA slot 0 (`src0_addr`) and its
  `TensorAccessorArgs<2>()` line.
- `in1` — **Case 1** → same; drops reader RTA slot 1 (`src1_addr`) and the chained
  `TensorAccessorArgs<src0_args.next_compile_time_args_offset()>()`.
- `output` — **Case 1** → same; drops writer RTA slot 0 (`dst_addr`) and its
  `TensorAccessorArgs<0>()`.

All three currently arrive as tensor objects pushed into `emplace_runtime_args` (not
`->address()`), which is the framework's patch-on-cache-hit shape — correct today, but it becomes a
`TensorParameter` + `TensorBinding` like any other. Re-index the surviving RTAs after the address
args disappear.

**TensorParameter relaxation:** none — the only value that reaches a brief.

**TensorAccessor 3rd arg:** none — no accessor in this factory passes one. Nothing to drop.

**CB endpoints:** **all legal.** Three CBs, each a plain 1 producer + 1 consumer FIFO across
distinct kernels. Nothing to self-loop, nothing to assign, no multi-binding flag, no dead CB, no
conditional DFB.

| CB | Producer | Consumer |
|---|---|---|
| `c_0` (in0) | reader `reserve_back`/`push_back` | compute `wait_front`/`pop_front` |
| `c_1` (in1) | reader `reserve_back`/`push_back` | compute `wait_front`/`pop_front` |
| `c_16` (out) | compute `reserve_back`/`push_back` | writer `wait_front`/`pop_front` |

(The reader's `dfb_in0.get_write_ptr()` feeding `pad_last_ktile` is its own PRODUCER binding peeking
its own buffer — not a second toucher. Do not re-census it as one.)

**Two kernel-side metadata rewrites, and they go opposite ways.** Both are Device-2.0-legal today
and both are Metal 2.0 port work; the `constexpr` declaration is the whole test (whitelist §A):

- `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:19` —
  `get_local_cb_interface(dfb_id_out).fifo_page_size` → **`dfb_out.get_entry_size()`**. The value is
  `const uint32_t`, so the member getter fits.
- `device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp:70,76` —
  `get_dataformat(dfb_id_in0)` is declared **`constexpr`**, so it **keeps the free-function form
  with the binding token**: `get_dataformat(dfb::in0)`. Do not move it onto the object and do not
  demote it to `const` to make a getter fit.

**`opt_level` — set `O3` explicitly on both compute KernelSpecs.** `grep -n opt_level` on the
factory returns nothing. An absent `KernelDescriptor::opt_level` still resolves to the legacy
per-kernel-type default — **`O3` for a `ComputeConfigDescriptor`** — while Metal 2.0's
`CompilerOptions` defaults to `O2` for both kinds. Leaving it unset silently drops a level. There
are **two** compute KernelSpecs here (see multiplicity below); each needs its own
`compiler_options.opt_level = KernelBuildOptLevel::O3`. The two DM kernels need nothing.

**Hardware config — Style A, no dropped field.** The factory resolves a TTNN `ComputeKernelConfig`
via `get_compute_kernel_config_args` (factory line 55), so translate with
`to_compute_hardware_config(device->arch(), config)`. All four helper-covered knobs are already set
on both compute descriptors, so there is no resolved-but-unset field to reapply. `packer_l1_acc` is
resolved and discarded (`(void)packer_l1_acc;`, line 57) — it has no Metal 2.0 counterpart, no
action. `unpack_modes` needs no entry: no compute kernel here consumes a Float32 DFB. Both DM
kernels are plain `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` (lines 147, 160) → the
arch-agnostic `create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`.

**Device-op-class edits this port forces** — two sanctioned exceptions, both to be recorded
prominently under Handoff points:

1. **Delete the pybound factory entry point.** `matmul_nanobind.cpp:1260-1274` is a whole
   `nb::class_<ttnn::prim::MatmulMultiCoreProgramFactory>` block whose only member is
   `create_descriptor`. That method vanishes at port time, so the entire block goes. This is a
   user-visible API surface change. **Leave the separate `nb::class_<MatmulDeviceOperation>` block
   at lines 1222-1237 untouched** — it binds device-op methods that survive.
2. **Drop the pybind-hook-only parameter.** `create_descriptor`'s fourth argument,
   `const std::optional<CoreRangeSet>& core_range_set`, is **ignored by the factory body** — spelled
   `/*core_range_set*/` at `matmul_multicore_program_factory.cpp:31`. It exists only for the hook
   above. Drop it; there is no production default to inline, because nothing reads it.

Exception 3 does not apply — the op has a proper `program_factory_t` variant, so this is a method
swap inside the existing struct.

## Watch for

- **Preserved multiplicity — there are TWO compute KernelSpecs, not one.** The factory runs
  `split_work_to_cores` and emits two `KernelDescriptor`s of `bmm.cpp`: `core_group_1` with CTA
  `num_output_tiles_per_core_group_1`, and (conditionally, when non-empty) `core_group_2` with
  `num_output_tiles_per_core_group_2`. Port as **two KernelSpecs of the same source in two
  WorkUnitSpecs over disjoint node sets**, both binding the same three DFBs with the same roles.
  Each node sees exactly one instance, so these are ordinary single-role bindings — **not**
  `allow_instance_multi_binding`. Moving the per-group tile count to an RTA to collapse them into
  one KernelSpec is the documented anti-pattern; it costs compile-time unrolling and is unnecessary.

- **Cross-op / shared kernels: NONE. Convert all three kernels in place.** Create no `_metal2` fork,
  add no pointer comment, touch no peer directory. This is worth reading twice, because the naive
  census says the opposite: `grep -rl writer_unary_interleaved_start_id.cpp` returns **24
  factories**. Twenty-three of them bind a *different, same-named copy* —
  `eltwise/unary/device/kernels/dataflow/…` (22 of them) or
  `data_movement/slice/device/kernels/dataflow/…` (one). An exhaustive grep for the bound path
  `matmul/device/kernels/dataflow/writer_unary_interleaved_start_id` returns exactly one hit in the
  repo: this factory, at line 155 (note the literal is split across lines 154-155, which is why a
  path grep alone is lossy). Same story for `bmm.cpp`: its two extra hits are a build file and a
  `// Implemented based on bmm.cpp` comment in moreh.

  **Do not bind an existing `_metal2` fork.** Two exist for this filename —
  `eltwise/unary/.../writer_unary_interleaved_start_id_metal2.cpp` and
  `copy/typecast/.../writer_unary_interleaved_start_id_metal2.cpp` — but the rung-1 check is
  *locational*, and `ls` of `matmul/device/kernels/dataflow/` shows no `_metal2` sibling. Those
  forks belong to other copies and are not yours.

- **RTA varargs: none — name every argument.** All three kernels read their args as distinct fields
  at constant indices in a block at the top (reader slots 0-11, writer slots 0-2). No variable-count
  loop, no data-selected index, no sentinel. Nothing here justifies `get_vararg`.

- **Positional CTAs to name:** `bmm.cpp` reads four (`get_compile_time_arg_val(0..3)` → `batch`,
  `Mt`, `Kt`, `Nt`); the reader reads two (`in0_last_ktile_w`, `in0_last_ktile_h`). The CB indices
  already arrive as *named* CTAs (`get_named_compile_time_arg_val("cb_in0")` and friends) — per
  whitelist rule 2 those become **`DFBBinding`s**, not named args.

- **A kernel branch that is dead from this factory — do not "fix" it.**
  `matmul_multicore_program_factory.cpp:135` hardcodes `uint32_t last_ktile_h = 0;`, so the reader's
  `if constexpr (in0_last_ktile_h > 0) { … pad_last_transposed_ktile … }` block
  (`reader_bmm_8bank_output_tiles_partitioned.cpp:74-79`) is unreachable here. It is live for
  sibling factories' transposed paths. Carry it across unchanged; it is flagged for the op owner in
  the audit's Misc anomalies, not for you.

- **Locate the tests carefully — this factory is the fallback of last resort.**
  `tests/ttnn/unit_tests/gtests/test_matmul.cpp` has cases commented as explicitly pinning
  `MatmulMultiCoreProgramFactory` (line 227, and a not-tile-aligned `[1,1,60,60] x [60,60]` case at
  line 276). A factory reachable only when nothing else matches is easy to leave uncovered by a
  filter that looks reasonable, so confirm the set with your invoker before relying on it.
