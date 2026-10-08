# Metal 2.0 Port Report — experimental/transformer/nlp_concat_heads_boltz

## Outcome

**PORTED; interleaved path verified, sharded path unverified.** The single factory
(`NLPConcatHeadsBoltzProgramFactory`, new) is on `ProgramSpecFactoryConcept`, and both paths are converted. It builds
cleanly (`./build_metal.sh --build-tests`) and passes the static self-audit.

**Interleaved path, verified on the 8×8 Wormhole.** `test_nlp_concat_heads_boltz.py` gives `2 passed`, both pre-port
(baseline) and post-port. Watcher was on (`TT_METAL_WATCHER=10`), and the legality checks were forced: both
`METAL2_CHECKS_FORCED` markers (`program_spec.cpp`, `program_run_args.cpp`) appeared once per test case, i.e. once per
cache miss. Each case's 3-iteration loop moves the input address and asserts exactly one new cache entry. That
exercises the cache-hit refresh of the new `input` / `output` tensor bindings, with bit-exact output.

**Sharded path, compile- and runtime-unverified.** No test covers it, and no device can construct the config. With
a sharded input, validation allows only a non-HEIGHT_SHARDED output. Its BLOCK_SHARDED spec has a shard height of S
over S·S rows, so it needs ≥ S ≥ 32 shard rows, and the card is 8×8. The sharded `KernelSpec`s have therefore never
been through the spec validator or the JIT. It mirrors the already-ported sibling `nlp_concat_heads` sharded path,
with the differences listed below. The invoker confirmed the test set on 2026-10-07.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
  (from the `Port_Recipe` checkout; `git log` over the docs path prints nothing in `Metal_Ports`)
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. There is no `override_runtime_arguments`.
- **Interleaved path:** the cache-hit refresh is the `input` / `output` tensor bindings (reader `tensor::in0_tensor`,
  writer fork `tensor::dst`). These replace the two legacy `Buffer*` RTAs.
- **Sharded path:** the refresh is the two borrowed DFBs (`in0` borrowed from `input`, `out0` from `output`). These
  replace the legacy CB `.buffer` bindings.

That covers the same per-dispatch state legacy had: buffer addresses only.

### Device-op-class edits
- **Exception 3 (direct-descriptor shape).** The op arrived with `create_descriptor` as a static member of
  `NLPConcatHeadsBoltzDeviceOperation`, and no `program_factory_t` (still true at port time). Changes in
  `device/nlp_concat_heads_boltz_device_operation.hpp`:
  - added nested `struct NLPConcatHeadsBoltzProgramFactory` with `create_program_artifacts`
  - added `using program_factory_t = std::variant<NLPConcatHeadsBoltzProgramFactory>;`
  - removed the device-op-level `create_descriptor`

  Its cache-refresh comment moved onto the new method, with "runtime-arg bindings / CB `.buffer` bindings" reworded
  to "tensor bindings / borrowed-DFB bindings". The includes swapped `program_descriptors.hpp` for `<variant>` and
  `ttnn/metal_v2_artifacts.hpp`. The body stays in `device/nlp_concat_heads_boltz_program_factory.cpp`.
- Pybind entry points removed: none. The only binding is the user function (`nlp_concat_heads_boltz_nanobind.cpp`).
- Custom `compute_program_hash`: none. The default hash is untouched.

### Open items
- Relaxation candidates: none considered (strict matching kept).
- No concept-fit friction.

## Handoff points

- **Sharded input + interleaved output: the failure mode changes (legacy bug, not fixed).** Validation accepts this
  config: with a sharded input it forbids only a HEIGHT_SHARDED output (`nlp_concat_heads_boltz_device_operation.cpp:47-50`).
  - **Legacy:** ran the sharded kernel against an unallocated CB 16. The kernel does `reserve_back` and raw writes
    through `get_write_ptr()` on it, so expect a hang or stray L1 writes.
  - **After the port:** the host binds `out0` only when the output is sharded, and the kernel references `dfb::out0`
    unconditionally. So this config now fails at kernel JIT (`'out0' is not a member of 'dfb'`), and the device is
    never touched.

  Per the brief, I added no `TT_FATAL` and no `#ifdef` gate. A gate would keep the bug running, and a guard is a
  validation change for the op owners. The sibling `nlp_concat_heads` already has the guard
  (`nlp_concat_heads_device_operation.cpp:52-56`). Owner: ops team (transformer TMs). Suggested fix: the same
  `TT_FATAL` requiring a sharded output when the input is sharded.
- **Direct-descriptor op converted to a conventional factory** (exception 3, above). Owner: TTNN. This is
  informational: the shape that PR #57409 introduced is undone for this op.
- **Readiness sheet is stale** (carried from the audit waiver). The sheet row still says `legacy device-op` /
  `NLPConcatHeadsBoltzProgramFactory` at a deleted `.hpp` / `yes (with PD step)`. After this port the factory name
  exists again, now nested in `nlp_concat_heads_boltz_device_operation.hpp`, on `ProgramSpecFactoryConcept`. Owner:
  readiness-sheet owner.
- **Sharded path unverified** (see Outcome). Owner: invoker / ops team. It needs a device whose grid has ≥ S rows,
  or a test that can construct a sharded output.

## Successes

- **Endpoint census → 1P+1C** (patterns catalog, "Two-toucher DFB → assign 1P+1C"). I re-derived it from the
  sharded kernel: each instance does one dead `reserve_back` on each DFB, then only raw peeks. That is two touchers
  per node per DFB. Read literally, the census rule's "`reserve_back` ⇒ locked producer" would push this into the
  multi-binding row, which the validator can't satisfy (zero consumers). The audit had already worked through this,
  and the census agrees with the brief: `in0` / `out0` are reader PRODUCER + writer CONSUMER, with no flag
  (`factory.cpp`, the sharded `dfb_bindings`). The kernel stays verbatim.
- **Shared-kernel rung 1.** Listing `eltwise/unary/device/kernels/dataflow/` found
  `writer_unary_interleaved_start_id_metal2.cpp` beside the original. I bound it with its own vocabulary
  (`dfb::out`, `tensor::dst`, `num_pages`, `start_id`, no defines), with no new fork and no edit outside the op
  directory. The sibling `nlp_concat_heads` binds it the same way.
- **Dead-CB rule.** The interleaved-path CB 16 (legacy allocates it whenever the output is sharded, but neither
  interleaved kernel touches index 16) gets no spec. The recipe's "bindingless DFB is rejected" note made it clear
  that the conditional has to become `in_sharded && out_sharded` rather than `out_sharded`.
- **Whitelist rule 7.** Both kernels' `get_tile_size(cb_id)` were non-`constexpr` locals, so both became
  `dfb_in0.get_tile_size()`.
- **Self-audit "Buffer\* not only `->address()`" note.** Legacy delivered both addresses as `Buffer*` objects inside
  `emplace_runtime_args`, with no `->address()` anywhere. Both sweeps run clean post-port.

## Friction

- **Gap — census rule vs dead `reserve_back`.** The endpoint-assignment procedure locks a role on any FIFO op. It
  has no wording for a FIFO op that is functionally dead (a whole-buffer reserve on an empty borrowed DFB, never
  pushed), where "two locked producers" has no legal disposition at all. The audit and this port both resolved it
  as 1P+1C by judgment. The sibling `nlp_concat_heads` port resolved it differently, by deleting the calls. A
  sentence in the pattern entry would stop the two ports of near-identical kernels diverging.
- **Gap — DM helper signature** (again). Recipe §Hardware configuration shows
  `create_reader_datamovement_config(device->arch())`. In this tree the helper takes only
  `bool disable_dfb_implicit_sync_for_all = false`
  (`ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp:24,32`). I used the no-arg
  form.
- **Gap — untestable path.** No device can construct a sharded output for this op: the output BLOCK_SHARDED spec
  needs ≥ S ≥ 32 shard rows. So the recipe's "JIT-compile the path" fallback isn't reachable through the op at all.
  A compile-only harness for a `ProgramSpec` (build the program and JIT the kernels without enqueueing) would close
  this class of gap.

## Open items for downstream

- **Shared kernel touches:**
  - (a) `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`.
  - (b) **Reused the existing fork** `writer_unary_interleaved_start_id_metal2.cpp`. No new file was created, and
    the original already carries its pointer comment.
  - (c) Remaining unmigrated binders of the legacy original, per the brief and #52228: `data_movement/reshape_on_device`,
    `data_movement/slice` (tile), `eltwise/unary_backward` (+ `gelu_bw`), `examples/example`,
    `experimental/matmul/attn_matmul`, `matmul` (multicore).
- **Deliberate differences from the sibling `nlp_concat_heads` port** (for whoever verifies the sharded path, or
  later unifies the two ops). All of them follow from preserving this op's legacy behaviour:
  - **Dead `reserve_back` calls kept** in the sharded kernel, together with the commented-out `push_back`. The sibling
    deleted them. The brief left this as an open question, and the default (keep verbatim) was followed.
  - **Sharded RTA node order kept** as legacy `corerange_to_cores(all_cores, std::nullopt, row_major)`. The sibling
    uses `row_wise=true`. Every node gets the same values, so the order is immaterial either way.
  - **`out0` DFB created iff `in_sharded && out_sharded`**, where legacy (and the sibling) allocated CB 16 iff
    `out_sharded`. This drops the dead interleaved-path CB 16.
  - **`dfb_in0.get_tile_size()`** (whitelist rule 7), where the sibling uses `get_entry_size()`. Both resolve to the
    tile size here.
  - **No `TT_FATAL`** for sharded input + interleaved output (see Handoff points). The sibling has one in its
    device op.
- **`reserve_back` narrowing** (sharded kernel). `CircularBuffer::reserve_back(int32_t)` became
  `DataflowBuffer::reserve_back(uint16_t)`, and the `block_size` CTA (`num_blocks_per_core * in0_HtWt`) can exceed
  65535 for large S. The calls are dead, so behaviour is unchanged. If a large-S sharded config ever becomes
  constructible, though, the constant conversion may also warn at JIT.
- **Carried audit anomalies** (not acted on, routed to the ops team):
  - The sharded path reads `in0_h_tiles = S·S/32` tile-rows per block, against a shard holding `S/32`, which looks
    like an over-read by a factor of S (`factory.cpp`, the `in0_h_tiles` computation; sharded kernel inner loop). It
    is masked because no sharded output can be constructed.
  - `single_tile_size_bytes` is unused in the sharded kernel.
  - Stale comments: "WRITER RUNTIME ARGS" in both readers, "interleaved accessor args" in the sharded kernel,
    "Grayskull Device Setup" and "Output shape is: [B, 1, s, 4544]" in the factory.
- **Test coverage:** `test_nlp_concat_heads_boltz.py` covers only the interleaved path, at two shapes and with
  bfloat16 only. FLOAT32 / BFLOAT8_B are accepted by validation but untested.
