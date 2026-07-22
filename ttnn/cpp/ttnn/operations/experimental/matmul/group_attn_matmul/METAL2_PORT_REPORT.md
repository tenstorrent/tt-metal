# Port Report — `experimental/matmul/group_attn_matmul`

Post-port report. Records what happened during the Metal 2.0 port: handoff points, doc
successes/friction, and open items for downstream. Companion to `METAL2_PORT_PLAN.md`.

**Outcome: complete.** Whole op ported (single factory + 3 kernels), all config paths converted.
- **Build:** clean — `./build_metal.sh --build-tests`, all 792 targets, no errors.
- **Tests (no-regression baseline):** `pytest .../test_attn_matmul.py -k group_attn_matmul -x -v` →
  **322 passed, 132 skipped (pre-existing documented skips), 0 failed** in 138s. Covers interleaved,
  IN0/IN1/OUT-sharded, fp32, program-cache, and exhaustive parametrizations.
- **Anti-pattern self-audit:** clean (no buffer-address RTAs, no `TensorAccessorArgs`, no positional
  args, no magic CB indices, no `.id` extraction, no multi-binding flag, no `CircularBuffer`/`CBDescriptor`
  in code).

## Provenance
- **Recipe docs (this port):** `b6dadc46ee0 2026-07-21 docs: fix metal_2.0 doc links (READ_ME_FIRST rename + two stragglers)`
- **Audit docs (inherited):** `b6dadc46ee0 2026-07-21 docs: fix metal_2.0 doc links (READ_ME_FIRST rename + two stragglers)`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — `GroupAttnMatmulProgramFactory::create_program_artifacts` returns
`ProgramArtifacts{spec, run_params}` (no `op_owned_tensors`). Matches the audit's decision; no
re-decision.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op had no override; uses default hash).
- Pybind entry points removed: **none** (`group_attn_matmul_nanobind.cpp` bound only the op
  function, never a `create_descriptor`).
- The device-operation class was **not** touched — the port is contained to the factory
  `.hpp`/`.cpp` and the three kernels. This is the success case for scope discipline.

### Open items
- **Concept fit was clean.** Single-program `MetalV2FactoryConcept` with no op-owned tensors, no
  op-owned `GlobalSemaphore`s, and no multi-program need — the common, well-supported case. No
  device-op-class edits were forced (no custom hash, no pybind `create_descriptor`, no pybind-hook-only
  factory parameter).
- **No tensor-spec relaxation candidate noticed.** The op's DFB sizes are shape-derived (`KV_HEADS`,
  `Mt`, `Kt`, `Nt`) and keyed into the default hash via `padded_shape`, so strict `TensorParameter`
  matching is correct here; no kernel would tolerate a relaxed match. (Pre-migration grep for
  `ArgConfig::Runtime*` in the kernels: no hits — this op does not use runtime tensor-shape args.)

## Handoff points

**None.** No boundary-rule assumption violations (no `sem::`/`tensor::` handle needed at an out-of-op
call site), no kernel-lib/LLK gaps, no framework-gap surprises, and no pybind surface removed (the op's
nanobind bound only the user-facing op function, never a `create_descriptor`). The port is contained to
the op's own directory.

## Successes
*(where the docs steered the port right — cite section + file:line)*

- **`INVALID`/`VALID` semaphore init.** Migration guide "SemaphoreSpec" says semaphores default-init
  to zero and non-zero init is deprecated. The legacy `SemaphoreDescriptor{.initial_value = INVALID}`
  looked like it might need the deprecated `SemaphoreAdvancedOptions.initial_value`, but `INVALID == 0`
  (`hostdevcommon/common_values.hpp:13`), so the default-zero-init is exactly faithful — no advanced
  option needed. The doc's framing pointed straight at the check.
- **`borrowed_from` counts as a TensorParameter use.** The three tensors are bound via `TensorBinding`
  only on their *interleaved* path; on the sharded path they are referenced solely via a DFB's
  `borrowed_from`. The migration guide's "every TensorParameter needs ≥1 TensorBinding" rule made this
  look like it might reject a borrowed-only tensor — but `program_spec.cpp:533-543` explicitly counts a
  `borrowed_from` reference as a use. So declaring all three `TensorParameter`s unconditionally (and
  supplying all three `TensorArgument`s) is correct in every config. Confirmed in code before relying on it.
- **Two-toucher vs self-loop census (brief agreement).** Re-derived every DFB's endpoint census from the
  kernel bodies per the [endpoint-assignment procedure]; it agreed with the brief exactly — all CBs are
  1P+1C except `c_2` (single raw `get_read_ptr` toucher → self-loop). No multi-binding, no disagreement to
  report. The [Two-toucher] pattern's "sync-free touch is role-free / cosmetic label" framing settled the
  `c_2` self-loop cleanly.
- **DFB is a drop-in NoC endpoint.** The kernels pass CB objects straight into `noc.async_read` /
  `async_write` / `async_write_multicast`. `noc_traits_t<DataflowBuffer>` (`dataflow_buffer.h:368`) mirrors
  `noc_traits_t<CircularBuffer>`, so the object-type swap left every NoC call site unchanged (only the
  object's type/construction changed). The whitelist's "1:1, names unchanged" promise held.
- **FP32 `unpack_modes` required-entry rule caught a latent validator failure.** The compute section's
  point about "a newly-required explicit `unpack_modes` entry when a compute kernel consumes a Float32
  DFB with `enable_32_bit_dest = true`" applies here: `test_group_attn_matmul_fp32` runs float32 inputs
  → `fp32_dest_acc_en = true` and the compute kernel consumes Float32 `IN0`/`IN1`/`INTERMED1`. Without
  the recipe's warning I'd have shipped a spec that fails validation only on the fp32 test. Added the
  entries conditionally (`...factory.cpp:388-398`), value `UnpackToSrc` (legacy `unpack_to_dest_mode`
  was empty → all `Default`). The audit's feature table marked compute features "all N/A" and did not
  surface this; the recipe's compute section did.

## Friction

### Gaps
- **Compute `hw_config` when the legacy op wires only a *subset* of the `ComputeKernelConfig` knobs.**
  The recipe's compute section (`Hardware configuration → Compute kernels`) says to translate the op's
  `ComputeKernelConfig` with `to_compute_hardware_config(arch, config)`, then hand-set the two knobs the
  helper can't (`bfp_pack_precision_mode`, `unpack_modes`). It assumes the legacy factory routed the
  config's four helper-covered knobs into its descriptor. This op does **not**: its
  `create_descriptor` destructures all five values from `get_compute_kernel_config_args` but wires only
  `math_fidelity` + `fp32_dest_acc_en` into `ComputeConfigDescriptor`, leaving `math_approx_mode` /
  `dst_full_sync_en` at their `false` descriptor defaults (`program_descriptors.hpp:101-106`) — i.e. it
  *drops* two knobs the caller can set. Calling `to_compute_hardware_config(arch, config)` unqualified
  would honor those dropped knobs and silently change `sfpu_precision_mode` / `double_buffer_dest` for a
  caller who sets `math_approx_mode`/`dst_full_sync_en`. To stay faithful (the section's own "port the
  *resolved* values, diff before/after" rule) I built `ComputeGen1Config` directly
  (`group_attn_matmul_program_factory.cpp:382-398`), setting only `fpu_math_fidelity` +
  `enable_32_bit_dest` and leaving the other fields at defaults, which reproduce the legacy descriptor
  defaults exactly. **Suggested doc improvement:** add a note to the compute-config section — when the
  legacy factory wired only a subset of the config knobs into its descriptor, mirror the *resolved
  descriptor*, not the raw `ComputeKernelConfig`; `to_compute_hardware_config` is faithful only when the
  legacy op passed all four knobs through.

### Confusion
- **"The only #include a porter adds is `kernel_args.h`" undersells the DFB include.** The kernel
  arg-retrieval section states the sole added kernel include is `experimental/kernel_args.h` (which
  supplies `get_arg`/`args::`/`dfb::`/`sem::`/`tensor::`). But the generated headers provide only the
  `dfb::` *tokens* — a kernel that constructs `DataflowBuffer` objects for FIFO ops (all three here) also
  needs `#include "api/dataflow/dataflow_buffer.h"` for the `DataflowBuffer` *class* (the legacy
  `CircularBuffer` came from `circular_buffer.h`). Verified against ported kernels (`welford_reduce_h.cpp`,
  `reshard_reader_diff_width.cpp`) which add it explicitly. A one-line note under whitelist rule 1
  ("swap `circular_buffer.h` → add `dataflow_buffer.h` when you construct DFB objects") would remove the
  ambiguity. (A kernel that only passes `dfb::` tokens to LLKs, never constructing a `DataflowBuffer`,
  genuinely needs only `kernel_args.h` — e.g. the reduce compute kernel — which is probably why the
  claim reads as it does.)
- **Test-helper subagent returned before the pytest finished.** The build/test-helper prompt template
  worked well for the build, but for the test run the helper launched pytest as a *background* process
  and returned immediately ("waiting for completion notification") rather than blocking to completion.
  The pytest ran fine and I recovered by monitoring the log file directly, but the helper template could
  say explicitly "run synchronously to completion; do not background the command."

## Open items for downstream

- **Misleading `compute_program_hash` comments (ops team).** `...factory.hpp:15-24` and
  `...factory.cpp:57-59,155-158` (legacy) described CB sizing "folded into `compute_program_hash()` via
  `padded_shape`", implying a custom hash that does not exist. The port rewrote the factory, so the
  header comment now states the Metal 2.0 reality (no override; default hash keys on tensor specs);
  the ops team should confirm the intended caching story. (Audit Misc anomalies flagged this too.)
- **RTA→CRTA opportunity (perf, follow-up pass).** The reader's `in1_mcast_sender_noc_x/_y` tail is
  identical on every node (a mcast sender-grid constant). Ported as per-node runtime *varargs*
  (`...factory.cpp:319`, reader `get_vararg`) to preserve dispatch semantics; it is really a *common*
  vararg (`num_common_runtime_varargs`, `get_common_vararg`). A later cleanup could demote it for
  dispatch efficiency. Not port work (would change dispatch semantics).
- **Writer dead metadata read (dropped).** Legacy `writer_transformer_group_attn_matmul.cpp:59`
  `const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);` was never used and the writer does not
  otherwise touch `c_1`. It could not be ported (would require binding `c_1` to the writer — a wrong
  3rd endpoint that would break the clean 1P+1C census) so the dead line (and its now-orphaned
  `cb_id_in1`) was dropped. Zero functional change; the live `get_tile_size` calls for `c_0`/`c_5`
  became DFB member getters (`writer...cpp:51,57`).
- **Legacy uses `detail::preferred_noc_for_dram_read`** (`kernel_types.hpp:134`, documented "only used
  in op_profiler, unstable") to pick the reader NOC. On Gen1 it returns `NOC_0` = the reader DM default,
  so the port uses the standard `create_reader/writer_datamovement_config` helpers and keeps the same
  value only for the mcast dest-coord ordering (`...factory.cpp:165`). Noting the odd dependency for the
  ops team; not acted on.
- **Two stale CB-terminology comments aligned** (in-directory, comment-only): compute
  `transformer_group_attn_matmul.cpp` ("CB::intermed1 / CBIndex::c_16" → "intermed1 / out"; the c_16 was
  already wrong in legacy) and reader `...cpp` ("cb_id_in1" → "cb_in1_obj", a name the port removed).
  Meaning preserved; done to keep the ported kernels free of dangling CB-index references.
