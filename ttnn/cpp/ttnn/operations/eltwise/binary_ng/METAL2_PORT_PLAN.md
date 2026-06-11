# Port Plan — binary_ng (`BinaryNgDeviceOperation`)

Port plan for `eltwise/binary_ng`, ported from `ProgramDescriptorFactoryConcept` to Metal 2.0
(`ProgramSpecFactoryConcept`). Written during the inventory and planning steps; committed alongside
the port for review.

**Scope decision (invoker-confirmed):** the single `ProgramFactory` runtime-selects ~37 kernel
sources. The host `create_program_spec` is atomic (one function, all paths) and is ported in full.
Kernels are JIT-compiled per selected path, so the C++ host builds/validates independently and kernel
conversion proceeds path-by-path. **This pass: full host + the no-broadcast · tile · FPU · tensor-tensor
path fully converted and device-validated, then extend to further paths as the session allows.** The
remainder is enumerated below as mechanical follow-on (Deferred / Flagged).

**Dynamic-shape decision (invoker-confirmed, self-sanity-checked):** adopt
`TensorParameter::advanced_options.dynamic_tensor_shape = true` on a/b/c. Ground truth: all three
accessors are built with `ArgConfig::RuntimeTensorShape` via the two-arg
`append_to(cta, common_rta)` form (shape flows as a common runtime arg, read by the accessor at
runtime — not a compile-time bake). The migration guide names this the faithful mirror of
`RuntimeTensorShape`. Coupled with the **mandatory custom-hash deletion** (the strict default hash
keys per-shape; the relaxation is correctness-accurate for these shape-independent accessors and is
relied on by the future generalized factory concept that auto-computes a relaxation-capturing hash).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `ProgramFactory::create_descriptor(...)` returns
  `tt::tt_metal::ProgramDescriptor` (`device/binary_ng_program_factory.cpp:384`).
- Variants: **single** factory. (Runtime kernel-source selection within it — see below — is *not*
  multiple factory variants.)
- Custom `compute_program_hash`: **delete → default** (sanctioned exception). Was at
  `device/binary_ng_device_operation.cpp:487` (declaration `binary_ng_device_operation.hpp:79`).
  Keys on `attributes` + `input_tensor_a.dtype()/.memory_config()` +
  `input_tensor_b->dtype()/.memory_config()` + `shard_volumes`; **omits tensor shape**.
- Pybind factory hook: **none** — grep finds no pybind/nanobind exposure of `create_descriptor` for
  binary_ng. Device-op-class edit #2 (pybind removal) does **not** apply. Edit #3 (drop pybind-hook
  parameter) does not apply either — `create_descriptor` has the standard 3-arg signature.

### Kernels (3 KernelDescriptors per selected path; pushed reader, writer, compute)
| unique_id | source (runtime-selected) | core_ranges | CTAs (positional) | RTAs (per-core) | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|
| reader | `get_kernel_file_path(kernel_config.reader_kernel, is_sfpu, is_where_op)` (`:863`) — 12 possible sources | `all_device_cores` | `[TensorAccessorArgs(a, RuntimeTensorShape)…, TensorAccessorArgs(b‖a, RuntimeTensorShape)…, has_sharding]` (`:855-860`) | tile: 21 (`:1255`); rm: 26 (`:1227`). Lead with `a.address` (+`b.address`) | TA(a) + TA(b) runtime-shape args (`:855-859`) | `make_dataflow_defines(a_dtype,b_dtype)` + `SRC_SHARDED`/`SRC_SHARDED_B` + per-bcast `SRC_BCAST*`/`BCAST_LLK` (`:684-823`) | `ReaderConfigDescriptor` |
| writer | `get_kernel_file_path(writer_kernel, …)` (`:705`) — 3 possible sources | `all_device_cores` | `[TensorAccessorArgs(c, RuntimeTensorShape)…, has_sharding]` (`:700-702`) | tile-b: 11 (`:1132`); scalar: 12 (`:1198`); rm: 14 (`:1116`). Includes `c.address` (+packed scalar on scalar path) | TA(c) runtime-shape args (`:700`) | `make_dataflow_defines(b_dtype)` + `SRC_SHARDED`/`DST_SHARDED` (`:680-682`) | `WriterConfigDescriptor` |
| compute | `get_kernel_file_path(compute_kernel, is_sfpu, is_where_op)` (`:842`) — ~22 possible sources | `all_device_cores` | `[num_tiles_per_cycle]` (`:846`) — single value, **not** per-group | 4: `{compute_tiles, freq, counter, compute_scalar_value}` (`:1214`); 5 for ISCLOSE (`:1159`) | none | `op_config.as_defines` + activation defines + `BCAST_INPUT`/`BCAST_LLK`/`WHERE_*`/`FILL_*`/`PACK_RELU`/`ISCLOSE_*` (`:444-839`) | `ComputeConfigDescriptor{ fp32_dest_acc_en, unpack_to_dest_mode[NUM_CIRCULAR_BUFFERS] }` (`:847`) |

**No per-group CTA work-split multiplicity.** `split_work_to_cores` produces
`num_tiles_per_core_group_1/2`, but those feed **per-core RTAs** (`compute_tiles`), not per-group CTAs.
The compute CTA (`num_tiles_per_cycle`) is identical across all cores. → one `KernelSpec` per kernel,
no multi-spec multiplicity. (Contrast the [Demoting per-group CTA to RTA] anti-pattern — N/A here,
legacy already used a single CTA.)

### CBs (`CBDescriptor`s; all `format_descriptors` single-element — no aliasing)
| index | role | total_size | created when | borrowed (sharded) |
|---|---|---|---|---|
| c_0 | a input | `a_tile_size * a_num_pages` | always (`:562`) | `a_buffer` if `a_sharded` (`:570`) |
| c_1 | b input | `b_tile_size * b_num_pages` (1 page if scalar) | always (`:593`) | `b_buffer` if `b_sharded` (`:601`) |
| c_2 | c output | `c_tile_size * c_num_pages` | always (`:653`) | `c_buffer` if `c_sharded` (`:661`) |
| c_3 | a activation interim | `a_interim_size * num_tiles_per_cycle` | `PROCESS_LHS_ACTIVATIONS` non-empty (`:574`) | — |
| c_4 | b activation interim | `b_interim_size * num_tiles_per_cycle` | `PROCESS_RHS_ACTIVATIONS` non-empty (`:605`) | — |
| c_5 | a bcast scratch | `a_tile_size * 2` | bcast type ∈ {ROW_A, ROW_A_COL_B, COL_A, SCALAR_A} (`:621`) | — |
| c_6 | b bcast scratch | `b_tile_size * 2` | bcast type ∈ {ROW_B, ROW_B_COL_A, COL_B, SCALAR_B} (`:635`) | — |

`NUM_CIRCULAR_BUFFERS`-sized `unpack_to_dest_mode` array (`:728`) is SFPU-only; entries set for
c_0/c_1/c_3/c_4/c_5/c_6 (`:732-751`). Must become a per-DFB `Table` gated on each DFB's binding
condition (migration guide spec-validator rule + layernorm friction finding).

### Semaphores
none — this op has no semaphores of any kind.

### Tensor accessors
| host site | originating Tensor | RTA slot (host, legacy) | binding | dynamic_tensor_shape |
|---|---|---|---|---|
| `:855` | input a | reader RTA[0] (`a.buffer()->address()`) | `PARAM_A` → reader `ta::a` | true |
| `:858` | input b (or `*a_buffer` placeholder when scalar) | reader RTA (`b.buffer()->address()`, `0u` when scalar) | `PARAM_B` → reader `ta::b`, **conditional (b present only)** | true |
| `:700` | output c | writer RTA (`c.buffer()->address()`) | `PARAM_C` → writer `ta::c` | true |

All three are **Case 1** (clean, already `TensorAccessor` end-to-end). No Case-2 bridge. The
`b = *a_buffer` placeholder (`:858`) becomes a *conditional* binding (bind b only when present), per
the brief — not a placeholder.

### Work split
- Driver: `tt::tt_metal::split_work_to_cores(grid, rt_c_num_tiles, row_major)` (`:1019` / `:1024`),
  or sharded grid directly (`:994`). `rt_c_num_tiles` = output tiles (tile path) or row-blocks (rm).
- Outputs: `num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1/2`.
- Sharded path: `core_group_1 = grid`; per-core tile counts from `ShardShapeGenerator`.
- Unused cores get zero-filled dummy RTAs sized to the active variant (`:1042-1066`).

### Cross-op kernels
**none.** Every kernel source is binary_ng-owned (`device/kernels` + `device/kernels_ng`).
Out-of-dir `#include`s are tt_metal HAL/LLK only; in-op cross-subdir helpers are
`kernels/compute/eltwise_utils*.hpp` and `kernels/dataflow/fill_tile_utils.hpp` (binary_ng-owned).

### Flags
- `binary_ng_utils.cpp:125` — `ComputeScalar`/where branch returns `"eltwise_where_sfpu_scalar"`
  (missing `.cpp`). Latent bug, **op-owner's** (audit Misc anomalies). NOT touched by the port.
- `binary_ng_program_factory.cpp:548-550` — `num_tiles_per_cycle = 2` SFPU TODO. Pre-existing; not
  port scope.
- Device 2.0 migration of the 7 rm dataflow kernels + CB-index holdovers already landed on this
  branch (provenance note in audit). The kernels this factory selects are all Device-2.0-clean.

### Full runtime-selected kernel-source set (the atomic unit)
**Readers (12):** tile `kernels_ng/dataflow/`: `reader_interleaved_no_bcast`, `…_row_bcast`,
`…_col_bcast`, `…_row_col_mixed_bcast`, `…_scalar_bcast`; legacy tile `kernels/dataflow/reader_interleaved_no_bcast`
(scalar-b path, `BinaryNgKernelConfig` default); rm `kernels_ng/dataflow/`: `reader_interleaved_rm_no_bcast`,
`…_rm_row_bcast`, `…_rm_col_bcast`, `…_rm_row_col_mixed_bcast`, `…_rm_scalar_bcast`, `…_rm_scalar_op`.

**Writers (3):** legacy `kernels/dataflow/writer_interleaved_scalar` (scalar path);
`kernels_ng/dataflow/writer_interleaved_no_bcast` (tile tensor-b); `kernels_ng/dataflow/writer_interleaved_rm_no_bcast` (rm).

**Compute (~22):** `{NoBcast, Bcast, Scalar}` legacy `kernels/compute/` × `{fpu, sfpu, where}`
(`eltwise_binary[_sfpu]_no_bcast` / `eltwise_where_no_bcast`; `eltwise_binary[_sfpu]` /
`eltwise_where_sfpu`; `eltwise_binary[_sfpu]_scalar` / `eltwise_where_sfpu_scalar`); `_Ng`
`kernels_ng/compute/` row/col/scalar/row-col bcast × {fpu, sfpu, where where present}.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: **delete** (was `binary_ng_device_operation.cpp:487`,
  decl `binary_ng_device_operation.hpp:79`).
- **Caching strategy**: default `MaximizeCacheReuse` (no `caching_strategy` declaration; no op-owned
  resources, Decision 4/Advanced does not fire).
- **Implementation notes**: `dynamic_tensor_shape = true` on all three `TensorParameter`s (above).

## Planned Spec Shape
- **KernelSpecs**: 3 — `READER`, `WRITER`, `COMPUTE` (one per legacy `KernelDescriptor`; source is
  the runtime-selected path). No work-split multiplicity.
- **DataflowBufferSpecs**: up to 7 — `DFB_A` (c_0), `DFB_B` (c_1), `DFB_C` (c_2) always;
  `DFB_A_INTERIM` (c_3), `DFB_B_INTERIM` (c_4), `DFB_A_BCAST` (c_5), `DFB_B_BCAST` (c_6) conditionally
  (preserve legacy conditions). `borrowed_from = PARAM_A/B/C` on the sharded path (gated on
  `a/b/c_sharded`). No aliasing.
- **SemaphoreSpecs**: none.
- **TensorParameters**: 3 — `PARAM_A` (a), `PARAM_B` (b, conditional), `PARAM_C` (c). All
  `dynamic_tensor_shape = true`.
- **WorkUnitSpecs**: 1 — `{READER, WRITER, COMPUTE}` on `all_cores` (the active work grid). (Unused
  cores in the legacy zero-RTA loop are outside the split's `all_cores`; the WorkUnit targets the
  active cores only.)

## Preserved Multiplicity
none — no work-split multiplicity in legacy (per-core variation is RTA-only; the single compute CTA
is core-invariant).

## Dropped Plumbing
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA[0] (`:1228`/`:1256`) | `a.buffer()->address()` | `TensorBinding(PARAM_A, "a")` on reader |
| reader RTA (`:1235`/`:1271`) | `b.buffer()->address()` / `0u` placeholder | `TensorBinding(PARAM_B, "b")` on reader, **conditional** |
| writer RTA (`:1117`/`:1133`/`:1200`) | `c.buffer()->address()` | `TensorBinding(PARAM_C, "c")` on writer |
| reader CTA (`:855-859`) | `TensorAccessorArgs(a/b, RuntimeTensorShape).append_to(cta, crta)` | `TensorParameter` + `TensorBinding`, `dynamic_tensor_shape=true` |
| writer CTA (`:700`) | `TensorAccessorArgs(c, RuntimeTensorShape).append_to(cta, crta)` | `TensorParameter` + `TensorBinding` |
| kernel-side | `TensorAccessorArgs<N>()` + `get_arg_val<uint32_t>(addr)` + `TensorAccessor(args, addr[, page_size])` | `TensorAccessor(ta::a/b/c)` |
| reader CTA tail (`:860`) | positional `has_sharding` | named CTA `{"has_sharding", …}` |
| writer CTA tail (`:702`) | positional `has_sharding` | named CTA `{"has_sharding", …}` |
| compute CTA (`:846`) | positional `{num_tiles_per_cycle}` | named CTA `{"num_tiles_per_cycle", …}` |
| all reader/writer/compute positional RTAs | `get_arg_val<uint32_t>(N)` | named RTAs `get_arg(args::name)` |
| CB indices in kernels | `tt::CBIndex::c_*` / `CircularBuffer(idx)` | `DFBBinding` + `dfb::name` (verify per kernel at conversion) |
| compute `unpack_to_dest_mode[NUM_CIRCULAR_BUFFERS]` (`:728`) | CB-id-indexed array | per-DFB `Table`, gated on each DFB's binding condition |

## Applied Patterns
- [Conditional / optional DFB bindings](metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings):
  c_3/c_4 (activation interims), c_5/c_6 (bcast scratch), and the conditional `b` tensor binding —
  bound only when their legacy condition holds; matching `KernelSpec::defines` gate + kernel-side
  `#ifdef`. (The bcast-scratch CBs already carry `SRC_BCAST*` defines; reuse/extend those.)
- Borrowed-memory DFB (migration guide — DataflowBufferSpec): `DFB_A/B/C.borrowed_from = PARAM_A/B/C`
  on the sharded path, gated on `a/b/c_sharded`.
- Per-DFB `unpack_to_dest_mode` gated on binding condition (layernorm friction finding; spec-validator
  rule). SFPU-only here.
- `to_compute_hardware_config` helper for the compute `hw_config` is **not** directly usable — the op
  builds `fp32_dest_acc_en` from data formats (`:717`), not from a `DeviceComputeKernelConfig`. Build
  `ComputeHardwareConfig` fields directly (fp32_dest_acc_en + unpack_to_dest_mode), as the layernorm
  port did.

## Deferred / Flagged
- **Yellow audit items**: none gating. Dynamic-TensorAccessor relaxation resolved above (adopt).
- **First validated path (this pass)**: tile · no-broadcast · FPU · tensor-tensor —
  reader `kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`,
  writer `kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`,
  compute `kernels/compute/eltwise_binary_no_bcast.cpp`. Interleaved (not sharded) for the first
  validation; the borrowed-DFB sharded sub-case uses the same sources, host-supported but
  not-yet-validated.
- **Remaining kernel conversions (mechanical follow-on)**: the other ~34 sources enumerated above
  (rm path × 6 readers + 1 writer + computes; broadcast computes; sfpu/where variants; scalar path).
  Each follows the validated pattern. Tracked in the port report.
- **Do NOT touch**: the `eltwise_where_sfpu_scalar` missing-`.cpp` anomaly (`:125`) — op-owner's.
