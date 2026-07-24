# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/repeat`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## Scope reminder

`RepeatDeviceOperation` has two program factories, both on the `descriptor` concept:

- `RepeatProgramFactoryLastDim` (`device/repeat_program_factory_last_dim.cpp`) → kernels `repeat_last_dim_rm_sharded.cpp`, `repeat_last_dim_rm_interleaved.cpp`
- `RepeatProgramFactoryHigherDim` (`device/repeat_program_factory_higher_dim.cpp`) → kernels `repeat_higher_dim_tile.cpp`, `repeat_higher_dim_rm_sharded.cpp`, `repeat_higher_dim_rm_interleaved.cpp`

Each factory selects its kernel and CB set from the runtime sharding state, so port both factories and all five kernels. The `ttnn::repeat` host composite in `repeat.cpp` calls other ops (`view`, `to_layout`, sharded/interleaved conversions); leave those alone.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (see `ttnn_factory.md`); both factories port to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (both factories).
- **Op-owned tensors:** none (carried by nothing; a `descriptor` op cannot have them).
- **Target concept:** `MetalV2FactoryConcept` (both factories, no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no` on both factory rows of the readiness sheet, code-confirmed. `Smuggled pointer` is also `no`: the factories bind buffers through the `Buffer*` channel, which the framework patches on cache hits, not through a raw `->address()` runtime arg.

## Construct — to do

**Tensor bindings** (per binding — identical shape across both factories and all five kernels):

- **input** — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)` in place of `TensorAccessor(src_args, src_addr)`. The `src_addr = get_arg_val<uint32_t>(0)` runtime arg and the input half of the `TensorAccessorArgs(...).append_to(compile_time_args, common_runtime_args)` plumbing both disappear.
- **output** — **Case 1** (via `TensorAccessor`) → same treatment. The `dst_addr = get_arg_val<uint32_t>(1)` runtime arg and the output half of the `TensorAccessorArgs` plumbing disappear.

Both bindings are pure Case 1: every memory access already goes through the accessor (`s.get_noc_addr(...)`, `noc.async_read(s, ...)` / `noc.async_write(cb_mem, d, ...)`, and `noc_async_read_sharded(noc, cb_slot, s, ...)` / `noc_async_write_sharded(noc, cb_slot, d, ...)`, which take the accessor by value). No raw-pointer (Case 2) bridge is needed, and there are no borrowed-memory bindings.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none present. All ten accessor constructions are already the two-argument form; there is no page-size override to drop.

**CB endpoints:** **self-loop every circular buffer**, in every config. Each program has a single kernel that is the only toucher of its CBs, so bind that one kernel as both PRODUCER and CONSUMER of each CB. The CBs to self-loop, per `(CB, config)`:

| Factory | Config | CB buffer_index | Role |
|---|---|---|---|
| HigherDim | tile-native | `0` | self-loop |
| HigherDim | RM sharded | `0` | self-loop |
| HigherDim | RM interleaved | `0`, `1` | self-loop (each) |
| LastDim | RM sharded | `0` | self-loop |
| LastDim | RM interleaved | `0`, `1` | self-loop (each) |

CB `1` (the write-alignment scratchpad) exists only in the interleaved row-major configs, guarded by `needs_alignment_cb`. No dead CBs to drop, no 1P+1C assignments, and no multi-binding flags.

## Watch for

- **CB endpoints (multi-binding):** none. Every CB is single-toucher; there is no hidden second writer to hunt (the op has no semaphores) and no split-reader or dual-instance work-split shape (each program instantiates exactly one kernel).
- **Cross-op / shared kernels:** the kernels `#include "ttnn/operations/data_movement/common/kernels/common.hpp"` for `tt_memmove`, `noc_async_read_sharded`, `noc_async_write_sharded`, `align_address`, and the `MASK_*` / `OFFSET_*` constants. This is an in-family shared header (same `data_movement` family), already Device 2.0. Its signatures take the `Noc` object, a raw CB-slot address, and a `TensorAccessor`, so your named tokens pass through cleanly: hand the sharded helpers `TensorAccessor(tensor::name)` and the `DataflowBuffer` write pointer. If the shared header is rewritten for Metal 2.0, treat the `data_movement` family that uses it as one port-together unit. Use the `Noc`-first overloads (the address-only overloads in that header are deprecated).
- **RTA varargs:** none. Name each runtime arg from its kernel-side variable (`src_addr`, `dst_addr`, the per-dim start/end bounds, `repetitions` / `num_repeats`, `nop`). No vararg mechanism is needed.
- **Note (not port work):** the audit flagged an operator-precedence defect in the last-dim factory's `cb_size_bytes` expression (`repeat_program_factory_last_dim.cpp:53-57`). It is a pre-existing host-logic issue routed to the ops team; the port carries the expression verbatim into `DataflowBufferSpec::total_size` and makes no functional change. Do not "fix" it as part of the port. See the audit's Misc anomalies section.
