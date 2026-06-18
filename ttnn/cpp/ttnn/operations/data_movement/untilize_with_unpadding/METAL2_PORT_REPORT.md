# Port Report — `untilize_with_unpadding`

Post-port report for the Metal 2.0 port of `untilize_with_unpadding`. Companion to
`METAL2_PREPORT_AUDIT.md`, `METAL2_PORT_BRIEF.md`, and `METAL2_PORT_PLAN.md`.

## Scope delivered this pass

The device-op holds six factories in `program_factory_t`. `MultiCoreInterleaved` was already on Metal
2.0 (the in-repo reference). This pass ported the two **clean, reachable** legacy factories:

- **SingleCore** → `create_program_artifacts` (`MetalV2FactoryConcept`). ✓
- **MultiCoreBlockInterleaved** → `create_program_artifacts`. ✓

The remaining three stay on the legacy `ProgramDescriptorFactoryConcept` (`create_descriptor`). The
variant supports mixed concepts and the framework dispatches per-factory, so the op continues to
build and run with a mix of ported and legacy factories.

- **ColInterleaved** — capitulated (see Handoff points).
- **MultiCoreSharded**, **MultiCoreNDSharded** — deferred (see Open items).

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for SingleCore and MultiCoreBlockInterleaved, matching the audit decision.
No deviation; no re-decision needed.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op already used the default reflection-based hash).
- Pybind entry points removed: **none** (nanobind binds only the user op function; no factory pybind).

### Open items
- Strict tensor-arg matching kept (no relaxation applied). No `ArgConfig::Runtime*` use in the ported
  kernels, so no `dynamic_tensor_shape` opt-in is required (this op is not in the eltwise family that
  the migration guide flags).

## Handoff points

- **ColInterleaved latent host/kernel RTA mismatch + dead code (op owner).**
  `device/kernels/dataflow/writer_unary_stick_layout_col_multicore.cpp:72-74` reads runtime-arg slots
  `3, 4, 5` (and `0, 1`), skipping slot `2` and reading out-of-bounds slot `5`, while
  `device/factories/untilize_with_unpadding_multi_core_col_interleaved_program_factory.cpp:166-174`
  supplies only slots `0..4`. By variable-name intent the kernel wants
  `size_per_row_per_block / number_blocks_per_core / width_size`, which sit at host slots `2 / 3 / 4`
  — i.e. the kernel reads are shifted by one. This is masked today because
  **`select_program_factory` (`device/untilize_with_unpadding_device_operation.cpp:18-63`) never
  returns `UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory`** — the factory is unreachable
  dead code, so the mismatch never executes. A Metal 2.0 positional→named conversion cannot reproduce
  an out-of-bounds read, and the recipe forbids fixing the legacy kernel during a port. **Action for
  owner:** decide whether to (a) delete the dead Col factory + kernel, or (b) fix the RTA pairing and
  wire it into `select_program_factory`, in a dedicated PR; then it can be ported.

## Build verification

**Could not compile/link in this environment — blocked by a broken toolchain, unrelated to the
port.** Two independent breakages:
- The configured compiler `/usr/bin/clang++-20` is absent (only `clang-14` and `g++` are installed);
  `ccache` aborts every compile with `execute_noreturn ... clang++-20: No such file or directory`.
- The configured MPI lib `/opt/openmpi-v5.0.7-ulfm/lib/libmpi.so.40` is gone (a copy exists at
  `/lib/x86_64-linux-gnu/libmpi.so.40`), so `libtt_metal.so` cannot link.
- `compile_commands.json` does not contain the data_movement op sources (unity build), so clangd has
  no per-file flags either — its 52 "file not found / undeclared identifier" diagnostics on the ported
  files all cascade from one unresolved `#include` and are an indexing artifact, **not** real errors
  (the already-working reference factory uses byte-identical include lines and the header exists).

Reconfiguring the project onto clang-14/g++ + the system MPI is a heavy, risky environment change
outside the scope of this port, so it was not attempted. **Recommended next step for someone with a
working toolchain:** `ninja -C build ttnncpp` (then `unit_tests_ttnn` if enabled). The two ported TUs
compile into `ttnn_op_data_movement` unity files `unity_19` (SingleCore) and `unity_20` (Block).

In lieu of a compile, a line-by-line static review was done against the known-good reference factory
and the original legacy sources (verified: spec field names/types, `Nodes`/`CoreRangeSet` usage,
`split_blocks_for_tilize_wh` structured-binding order, per-region compute CTAs, DFB sizing, and every
forked kernel's named-arg→legacy-slot mapping). No discrepancies found.

## Successes

- **Caution: Modifying a shared dataflow kernel** (patterns catalog) fired correctly. The grep for
  consumers of `writer_unary_stick_layout_wh_multicore.cpp` found a second consumer
  (`untilize/device/factories/untilize_multi_core_block_program_factory.cpp:196`, un-ported), so it
  was **forked** to `_metal2` rather than modified in place — the legacy copy stays valid for that
  co-borrower. Conversely `writer_unary_unpad_dims_split_rows.cpp` had a single consumer (SingleCore)
  and was converted **in place**.
- **Anti-pattern: Demoting per-group CTA to RTA** (patterns catalog) steered the Block port: the
  legacy per-sub-region compute `KernelDescriptor`s (full / cliff-row / cliff-col / cliff-col-row,
  each with different `{block_size_col, block_size_row}` CTAs) map to one compute `KernelSpec` per
  region, each in its own `WorkUnitSpec`, all binding the shared IN/OUT DFBs — no CTA→RTA demotion,
  loop unrolling preserved.

## Friction

### Gaps
- **DFB single-size vs. legacy per-region CB sizes (Block).** The legacy Block factory emitted a CB
  pair *per sub-region* with a per-region size (`single_sub_block_size` for full/cliff-col,
  `single_block_size_cliff_row` for the row-cliff regions), via four `CreateCircularBuffer` calls on
  disjoint core ranges sharing one `buffer_index`. A `DataflowBufferSpec` carries one
  `entry_size`/`num_entries` set at construction (recipe: "set once at construction"), and its
  placement is derived from bindings — there is no per-node size. Resolved by sizing the IN/OUT DFBs
  to the **max** present region tile count (a superset of every legacy per-region buffer; FIFO depth
  only needs to be ≥ the block consumed at once). This is a (small) L1-footprint increase on the
  smaller cliff regions vs. legacy. The recipe's "one DataflowBufferSpec per legacy CBDescriptor"
  guidance doesn't directly cover the "same `buffer_index` re-declared per core-range with different
  sizes" legacy shape — a doc note on consolidating per-region same-index CBs into one max-sized DFB
  would help the next porter.

### Confusion
- The reference port (`..._multi_core_interleaved_program_factory.cpp`) uses the
  `ProducerOf`/`ConsumerOf` convenience binding factories, which the port recipe explicitly
  discourages in favor of full `DFBBinding{}` designated-initializers. The brief says to use the
  reference port as the binding-style template. Followed the **reference** (ProducerOf/ConsumerOf) for
  in-op consistency and because it is known-good here. Flagging the doc/reference inconsistency: the
  recipe and the canonical in-repo reference disagree on this stylistic point.

## Open items for downstream

- **Fake-CB self-loop bindings:** none in the ported factories (SingleCore / Block use real
  producer/consumer DFB pairs). The audit's one fake-CB FYI (`c_17`, out-sharded path) is in the
  deferred Sharded factory.
- **Cross-op kernel touches (forks created this pass):**
  - `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore_metal2.cpp` — **fork**
    of `reader_unary_interleaved_wh_multicore.cpp`. Remaining legacy consumers: eltwise/unary family,
    `untilize/.../untilize_multi_core_block_program_factory.cpp`. Sunset the legacy copy when they
    port.
  - `data_movement/untilize/device/kernels/compute/untilize_wh_metal2.cpp` — **fork** of
    `untilize_wh.cpp`. Remaining legacy consumers: `untilize` block factory (+ any other co-borrowers
    of `untilize_wh.cpp`).
  - `data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore_metal2.cpp`
    — **fork** of `writer_unary_stick_layout_wh_multicore.cpp`. Remaining legacy consumer:
    `untilize/.../untilize_multi_core_block_program_factory.cpp`.
  - Reused existing forks (no new file): `untilize/.../dataflow/reader_unary_start_id.cpp` and
    `untilize/.../compute/untilize_compute_metal2.cpp` (created by the interleaved port).
  - In-place conversion (single consumer, no fork): `writer_unary_unpad_dims_split_rows.cpp`.
- **Deferred factories — remaining work for the next pass:**
  - **MultiCoreSharded:** borrowed-memory DFBs — input CB `c_0` (`...sharded...:97`,
    `.buffer = a.buffer()`) → `DataflowBufferSpec::borrowed_from = INPUT`; out-sharded CB `c_17`
    (`...sharded...:126`, `.buffer = output.buffer()`) → `borrowed_from = OUTPUT`, **producer-only**
    edge → apply the fake-CB self-loop workaround if the validator rejects it. Three runtime-selected
    writer variants (out-sharded `writer_unary_unpad_batch_rows_sharded.cpp`, W=16
    `writer_unary_unpad_width_16_sharded.cpp`, interleaved-out shared
    `kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp`) all convert together with the
    factory. Reader `reader_unary_sharded.cpp`, compute `untilize.cpp` / shared `eltwise_copy.cpp`.
  - **MultiCoreNDSharded:** Case-1 input bound on **both** reader and writer (writer uses
    `accessor_src.shard_pages`); reader `reader_unary_nd_sharded_blocks.cpp`, writer
    `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp`, compute
    `untilize_variable_num_blocks.cpp` (a `_metal2` fork already exists — reuse it). The stale
    `ccl/kernel_common/sharding_addrgen.hpp` include in both ND kernels is unused — leave it.
- **ColInterleaved:** see Handoff points — resolve the latent RTA mismatch + reachability in a
  dedicated PR before it can be ported.

## Successful failure (capitulation)

- **Op/factory:** `untilize_with_unpadding` / `UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory`.
- **File/lines:** `device/kernels/dataflow/writer_unary_stick_layout_col_multicore.cpp:72-74` (kernel
  reads) vs. `device/factories/untilize_with_unpadding_multi_core_col_interleaved_program_factory.cpp:166-174`
  (host supplies).
- **Why mechanical conversion failed:** the legacy host/kernel positional RTA contract is internally
  inconsistent (kernel reads OOB slot 5, skips slot 2). Named bindings pair by name and cannot
  reproduce the out-of-bounds read; "fixing" the pairing would be an unsanctioned behavior change to a
  legacy kernel during a port. Compounded by the factory being unreachable from
  `select_program_factory` (dead code), so there is no live behavior to preserve or test against.
- **Off-rules change that would unblock it:** correct the host RTA list (or the kernel indices) so the
  three trailing args line up, and wire the factory into `select_program_factory` — both outside the
  scope of a mechanical Metal 2.0 port.
