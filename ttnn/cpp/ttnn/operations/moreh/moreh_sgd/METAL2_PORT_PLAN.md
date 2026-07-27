# Port Plan — moreh_sgd

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_sgd`, ported from the `descriptor`
(`create_descriptor` → `ProgramDescriptor`) API to Metal 2.0 (`MetalV2FactoryConcept` /
`create_program_artifacts`). Written during the inventory and planning steps; committed alongside
the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, realized as `HasDirectDescriptor` — `create_descriptor`
  lives directly on `MorehSgdOperation` (`device/moreh_sgd_device_operation.hpp:37`), no
  `program_factory_t` variant, no nested factory struct.
- Variants: single (`create_descriptor` in `device/moreh_sgd_program_factory.cpp:25`).
- Custom `compute_program_hash`: none — already the default reflection-based hash (audit confirmed).

*(Target concept chosen during audit: `MetalV2FactoryConcept`. Carried forward in the
[TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Kernels
All three kernels are op-owned (`device/kernels/`), already Device 2.0 (`Noc`, `DataflowBuffer`,
`TensorAccessor`). They `#include` the shared `ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp`
helper pools, which already take `DataflowBuffer` objects (function-call escapes only — no file-path
kernel borrows).

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_sgd.cpp` | `all_cores` | `TensorAccessorArgs(param_in)`, `TensorAccessorArgs(grad)`, `TensorAccessorArgs(momentum_in)` (last only if `momentum_buffer_in.has_value()`) | `param_in.buffer()`, `grad.buffer()`, `momentum_in_buf`, `num_tiles`, `tile_offset`, `lr`, `momentum`, `dampening`, `weight_decay`, `one` (per core) | `WEIGHT_DECAY`/`MOMENTUM`/`MOMENTUM_INITIALIZED`/`NESTEROV`/`FP32_DEST_ACC_EN` (conditional) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/writer_moreh_sgd.cpp` | `all_cores` | `TensorAccessorArgs(param_out)`, `TensorAccessorArgs(momentum_out)` (last only if `has_momentum_buffer_out`) | `param_out.buffer()`, `momentum_out_buf`, `num_tiles`, `tile_offset` (per core) | `MOMENTUM` (conditional) | `WriterConfigDescriptor{}` |
| compute (group 1) | `device/kernels/moreh_sgd.cpp` | `core_group_1` | `num_tiles_per_core_group_1` | none | `WEIGHT_DECAY`/`MOMENTUM`/`MOMENTUM_INITIALIZED`/`NESTEROV`/`FP32_DEST_ACC_EN` | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` |
| compute (group 2) | `device/kernels/moreh_sgd.cpp` | `core_group_2` (if non-empty) | `num_tiles_per_core_group_2` | none | same as group 1 | same as group 1 |

### CBs (`CBDescriptor`) → DFBs
Ten CBs, all plain (no GlobalCircularBuffer, no `address_offset`, no `.tile`):

| CB | role | entry_size | num_entries | data_format |
|---|---|---|---|---|
| `c_0` param_in | reader→compute | `data_tile_size` | 2 | `data_format` |
| `c_1` grad | reader→compute | `data_tile_size` | 2 | `data_format` |
| `c_2` momentum_in | reader→compute (**optional**) | `data_tile_size` | 2 | `data_format` |
| `c_16` param_out | compute→writer | `data_tile_size` | 2 | `data_format` |
| `c_17` momentum_out | compute→writer (**optional**) | `data_tile_size` | 2 | `data_format` |
| `c_24` scalar_args | reader→compute | `intermed_tile_size` | 5 | `intermed_cb_format` |
| `c_25` tmp1 | compute self-loop | `intermed_tile_size` | 1 | `intermed_cb_format` |
| `c_26` tmp2 | compute self-loop | `intermed_tile_size` | 1 | `intermed_cb_format` |
| `c_27` tmp3 | compute self-loop | `intermed_tile_size` | 1 | `intermed_cb_format` |
| `c_28` tmp4 | compute self-loop | `intermed_tile_size` | 1 | `intermed_cb_format` |

`data_format = datatype_to_dataformat_converter(param_in.dtype())` (BFLOAT16 per `validate_inputs`);
`intermed_cb_format = fp32_dest_acc_en ? Float32 : data_format`.

### Semaphores
None (op uses no semaphores).

### Tensor accessors (all Case 1, via `TensorAccessor`)
| tensor | kernel(s) | host RTA slot dropped |
|---|---|---|
| `param_in` (input) | reader | `param_in.buffer()` (reader RTA 0) |
| `grad` (input) | reader | `grad.buffer()` (reader RTA 1) |
| `momentum_buffer_in` (input, optional) | reader (gated `MOMENTUM && MOMENTUM_INITIALIZED`) | `momentum_in_buf` (reader RTA 2) |
| `param_out` (output) | writer | `param_out.buffer()` (writer RTA 0) |
| `momentum_buffer_out` (output, optional) | writer (gated `MOMENTUM`) | `momentum_out_buf` (writer RTA 1) |

### Work split
`split_work_to_cores(grid, num * Ht * Wt)` →
`(num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2)`.
reader/writer run on `all_cores`; compute splits into `core_group_1` / `core_group_2`.

### Cross-op kernels
None (no file-path kernel borrows). Shared helper pools `ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp`
are function-call escapes only, already `DataflowBuffer`-native — bind cleanly, left untouched.

### Flags
No unreferenced kernels. No descriptor type outside the audit's scan.

## TTNN ProgramFactory
- **Concept (inherited from audit):** `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`:** none — nothing to delete.
- **Implementation notes:** the legacy op is `HasDirectDescriptor` (create_descriptor directly on the
  op struct, no `program_factory_t`). `MetalV2FactoryConcept` is **not** detected directly on the op
  struct (framework `mesh_device_operation_adapter`); it is only reached as a `program_factory_t`
  variant alternative. So the port introduces a nested factory struct
  `MorehSgdOperation::MorehSgdProgramFactory` carrying `create_program_artifacts`, a single-alternative
  `using program_factory_t = std::variant<MorehSgdProgramFactory>`, and a `select_program_factory`
  returning `MorehSgdProgramFactory{}`. This mirrors the wiring the sibling moreh ports (moreh_fold,
  moreh_abs_pow, moreh_dot_backward) adopted. No pybind change (nanobind binds `&ttnn::moreh_sgd`,
  not `create_descriptor`).

## Planned Spec Shape
- **KernelSpecs (4):** `reader` (all_cores), `writer` (all_cores), `compute1` (core_group_1),
  `compute2` (core_group_2, built only when non-empty). **Preserved multiplicity** — see below.
- **DataflowBufferSpecs (10):** 1:1 with legacy CBs. `param_in`, `grad`, `param_out`, `scalar_args`,
  `tmp1`–`tmp4` unconditional; `momentum_in` (c_2) and `momentum_out` (c_17) conditional.
- **SemaphoreSpecs:** none.
- **TensorParameters:** `param_in`, `grad`, `param_out` unconditional; `momentum_in`, `momentum_out`
  conditional (declared only when the tensor exists — a `TensorArgument` is mandatory for every
  declared `TensorParameter`, and the optional tensors are `nullopt` otherwise).
- **WorkUnitSpecs (1–2):** `group_1` = {reader, writer, compute1} on core_group_1; `group_2` =
  {reader, writer, compute2} on core_group_2 (only when non-empty). reader/writer appear in both
  (legal — a kernel may be in multiple work units; their union is all_cores).
- **Op-owned tensors:** none (outputs are ordinary returned device tensors).

## Preserved Multiplicity
```
Legacy KernelDescriptors [compute_desc_1, compute_desc_2] of source device/kernels/moreh_sgd.cpp
  → KernelSpecs [compute1, compute2] of same source
  → in WorkUnitSpecs [group_1, group_2] (target_nodes core_group_1 / core_group_2)
  → each carries its own named CTA {"num_tiles": num_tiles_per_core_group_N}
  → sharing DFBs param_in/grad/scalar_args (CONSUMER role each), param_out/momentum_out (PRODUCER
    role each), momentum_in (CONSUMER), tmp1-4 (self-loop PRODUCER+CONSUMER).
```
The two compute KernelSpecs bind the shared DFBs over **disjoint** node sets (core_group_1 vs
core_group_2), so each is a legal single-role-per-node binding — **no** `allow_instance_multi_binding`
flag. This multiplicity is **non-negotiable**: the compute kernel reads its loop count from the CTA
(`get_compile_time_arg_val(0)` → `get_arg(args::num_tiles)`), so the per-group CTA is *live* and must
not be demoted to an RTA (contrast moreh_abs_pow, whose CTA was dead and could collapse to one spec).

## Dropped Plumbing
- **Buffer-address RTAs → `TensorBinding`:** reader RTA slots 0/1/2 (`param_in.buffer()`,
  `grad.buffer()`, `momentum_in_buf`); writer RTA slots 0/1 (`param_out.buffer()`, `momentum_out_buf`).
  (`moreh_sgd_program_factory.cpp:266-279`.)
- **`TensorAccessorArgs` plumbing → binding mechanism:** reader `TensorAccessorArgs<0/…>` chain
  (`reader_moreh_sgd.cpp:42-49`) and writer `TensorAccessorArgs<0/…>` chain
  (`writer_moreh_sgd.cpp:28-34`) and their host `TensorAccessorArgs(*buffer).append_to(...)` sites
  (`moreh_sgd_program_factory.cpp:178-197`).
- **Magic CB indices → `DFBBinding`:** every `tt::CBIndex::c_*` constant in all three kernels.
- **Positional compute CTA → named CTA:** compute `get_compile_time_arg_val(0)` → `get_arg(args::num_tiles)`.
- **Positional RTAs → named args:** reader/writer scalar RTAs become `get_arg(args::name)`.
- **Page-size 3rd arg:** none — no accessor passes a page size.
- **Semaphore-ID RTAs:** none.

## Applied Patterns
- **Self-loop DFB binding** (port_patterns "Sync-free and single-ended CBs → self-loop DFB"):
  `tmp1`–`tmp4` on each compute KernelSpec, bound both PRODUCER and CONSUMER (same accessor_name).
- **Conditional / optional DFB + tensor bindings** (port_patterns "Conditional / optional DFB
  bindings", recipe kernel rule 6): `momentum_in` (c_2) gated `MOMENTUM && MOMENTUM_INITIALIZED`;
  `momentum_out` (c_17) gated `MOMENTUM`. Host conditionally builds the DFB spec + bindings +
  TensorParameter + TensorArgument; `KernelSpec::compiler_options.defines` carries the flags; the
  kernels `#ifdef`-gate the `DataflowBuffer`/`TensorAccessor` construction (already gated in reader/
  writer; the compute kernel's two momentum `DataflowBuffer` constructions are newly `#ifdef`-gated to
  match).
- **Preserved work-split multiplicity** (port_patterns "Demoting per-group CTA to RTA" anti-pattern):
  two compute KernelSpecs, live per-group CTA.
- **Compute `unpack_modes` under FP32** (recipe Hardware configuration): when `fp32_dest_acc_en`, the
  `intermed_cb_format` DFBs consumed by compute (`scalar_args`, `tmp1`–`tmp4`) are Float32 with
  `enable_32_bit_dest = true`, so the validator requires an explicit `unpack_modes` entry. Legacy set
  no `unpack_to_dest_mode` (all `Default`) → `UnpackMode::UnpackToSrc` for each.

## Design decision — intermediate DFBs kept unconditional (deviation from brief)
The brief suggested classifying the four intermediates `c_25`–`c_28` per compile-define config and
self-looping "where live" (e.g. only `c_27` under the minimal config). I keep all four DFB specs and
their compute self-loop bindings **unconditional**, because:
1. The legacy factory allocates all four CBs **unconditionally** (`moreh_sgd_program_factory.cpp:116-139`),
   and the compute kernel constructs all four `DataflowBuffer` wrappers **unconditionally**
   (`moreh_sgd.cpp:23-30`). Unconditional specs/bindings reproduce legacy L1 usage byte-for-byte.
2. A self-loop provides both PRODUCER and CONSUMER endpoints from the single compute kernel, so an
   untouched-in-this-config intermediate still validates (no missing-endpoint failure — unlike the
   cross-kernel momentum DFBs, which *must* be gated).
3. Gating them would require compound `#ifdef` guards over the compute kernel's `cb_grad_tmp` /
   `cb_momentum_tmp` selection variables — added complexity and risk with zero behavioral benefit.
The genuinely conditional resources here are the cross-kernel **momentum** DFBs/tensors (gated as
above). See the report's Friction section.

## Deferred / Flagged
- None. No structural issue beyond what the audit anticipated; the intermediate-DFB conditionality
  nuance is resolved above as a documented judgment call, not a stop signal.
