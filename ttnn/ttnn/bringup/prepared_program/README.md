# ttnn.bringup.prepare_program / run_prepared (prepared_program)

## What it does
New op (no TTNN source; Marko Staletovic's, `mstaletovic/combine-lean-sender-llk` @ `856d97d5864`): a `generic_op`
program prepared once and dispatched with only its tensor addresses patched. Host time only; the device program is
the descriptor's.
- `generic_op(io_tensors, mesh_program_descriptor)` re-applies every runtime arg of every core of every chip on each
  program-cache hit. `prepare_program(desc, io_tensors)` copies the descriptor once and records which runtime-arg slots
  hold an io tensor's address (exact matches) and which CBs an io tensor backs; `run_prepared(prepared, io_tensors)`
  then patches only those slots on a hit (the factory's `override_runtime_arguments`), and an io tensor's slots only
  when its first slot does not already hold its address.
- Cache key: generic_op's structural descriptor hash of every (range, program), salted, plus every unpatched runtime-arg
  value (a rebuilt descriptor with other frozen values does not hit the old program).
- Python `ttnn.bringup.prepared_program.generic_op(io_tensors, desc)`: drop-in for `ttnn.generic_op` at call sites that
  keep their descriptor object across calls (prepared on first sight of the object; at most 512 kept).

## Consider it instead of
- `ttnn.generic_op` at a call site that reuses one MeshProgramDescriptor object every call and changes only the
  addresses of the tensors it passes (eager host time; under trace it makes no difference).

## Options
- `prepare_program(mesh_program_descriptor, io_tensors)`: no options. Returns a `PreparedProgram` (`patch_count`,
  `n_io`, `hash`, and diagnostic `payload`, `kernel_stats`, `invariant_runs`).
- `prepare_program(..., value_slots=[(kernel index, runtime-arg index), ...])` (default `[]`: the behaviour before
  the option, same cache key): runtime arg `arg` of kernel `kernel` on every core of every mesh program that has it
  takes a per-run value (slot = position in the list); such a slot is never matched to an io address and is left out
  of the cache key. `PreparedProgram.n_values` is the slot count.
- `run_prepared(prepared, io_tensors, values=[])`: `values` one per value slot (written on every run, cache miss and
  hit); the output is the last io tensor, as `generic_op`.
- Contract (prepared_program.hpp): every per-call tensor a kernel addresses (a runtime arg holding it, or a CB on it) is
  in `io_tensors`, at the same positions at prepare and every run; every other runtime-arg value and address is frozen
  at prepare; a constant equal to an io tensor's address is taken for it (prepare refuses two io tensors at one
  address); a descriptor object is not mutated after its first call.

## Measured
- Marko's worklog (MOE_BLOCK_WORKLOG.md on his branch, "Host side of an untraced forward: prepared dispatch",
  2026-10-26, LoudBox 2x4, chunk 1280): K3 g224 MoE block untraced host enqueue 3839 -> 1494 us, synced wall 4310 ->
  2573 us; GLM-5.3-Flash 2757 -> 1281 / 3847 -> 2438 us; bit-exact block outputs; traced replays unchanged.
- Not measured on this branch.

## Limits
- Only addresses of io tensors (and, with `value_slots`, the named per-core args) are patched: anything else that changes between calls needs a new descriptor (a new
  prepare). Exact-match patching: an argument that holds an io tensor's address plus an offset is frozen.
- The bring-up framework's step profiler on this branch does not wrap `run_prepared` (Marko's branch does, in
  models/demos/common/bringup/testing/profiler.py); its launches are missing from per-step device times here.
- No unit tests of its own. Model cases (tests/, task O.1): each program a model runs through it, built as the model builds
  it, run by generic_op and prepared with tensors moved, bit-identical, and the determinism check (DeepSeek-V4.1: the
  fused MoE head, the exchange dispatch).

## How to call
```python
from ttnn.bringup.prepared_program import generic_op as prepared_generic_op

out = prepared_generic_op([in0, in1, out], desc)          # same desc object every call

p = ttnn.bringup.prepare_program(desc, [in0, in1, out])   # or explicitly: once
ttnn.bringup.run_prepared(p, [in0, in1, out])             # every call
```

## Used by
- models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py (Marko's token-split exchange: planner, dispatch prep,
  arena zeroing, sparse dispatch), models/demos/moe_block/tt/fused_head.py and tt/tail_engine.py (Marko's MoE block).
- deepseek_v41_flash_d_p under MOE_IMPL=block (through the exchange).
- ttnn.bringup.fabric_reduce_scatter(prepared=True) (deepseek_v41_flash_d_p tt/ccl.py, GENERIC_OP_PREPARED, P.1
  galaxy): one PreparedProgram per cached descriptor, called directly (prepare_program / run_prepared), not through
  this module's 512-entry cache.
- models/demos/moe_block/tt/tail_a2a.py TailA2A(prepared=True) (MoEOptions.tail_prepared; deepseek_v41_flash_d_p
  tt/moe_block_adapter.py, setting TAIL_A2A_PREPARED, P.2 galaxy): one PreparedProgram per cached descriptor (keyed by
  the y / se / out addresses, so a hit patches nothing), called directly. 8x4 warm chunk wall 594.3 -> 544.4 ms, ladder
  metrics identical.
- ttnn.bringup.ckv_bfp4(prepared=True) (deepseek_v41_flash_d_p tt/compressor.AttnTable.append_tiles, setting
  CKV_BFP4_PREPARED, P.7 galaxy): one PreparedProgram per (mesh, n, rounded, memory configs), row0 a value slot.
