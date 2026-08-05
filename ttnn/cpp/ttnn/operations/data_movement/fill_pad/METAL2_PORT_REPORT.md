# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/data_movement/fill_pad`

## Outcome

**PORTED** — both program factories (`FillPadProgramFactory` DRAM, `FillPadL1ShardedProgramFactory` L1-sharded) converted to `ProgramSpecFactoryConcept` and their tests pass. Ported together in one change because they share the compute kernel.

## Provenance

- **Recipe docs (this port):** `56373090d3d 2026-08-05 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `56373090d3d 2026-08-05 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept` for both factories — each `create_program_artifacts` returns a `ttnn::device_operation::ProgramArtifacts{spec, run_params}` (no op-owned tensors). No deviation from the audit's decision.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op had no custom hash).
- Pybind entry points removed: **none** — `fill_pad_nanobind.cpp` binds only the `fill_implicit_tile_padding` free function; there was no `create_descriptor` pybind to remove.
- No factory-parameter-for-pybind-hook to drop.

The device-operation class (`fill_pad_device_operation.{hpp,cpp}`), the types header, and the nanobind file were **not touched** — the port is confined to the two factory bodies, the five kernel sources, and the factory header's return types.

### Open items
- **Tensor-arg relaxation candidates:** none applied (kept strict, per default). The op is in-place and its `TensorParameter` spec equals the io tensor spec exactly; no `ArgConfig::RuntimeTensorShape` use in the kernels, so no relaxation is warranted.

## Handoff points

None. The port stayed entirely within the op directory; no kernel-lib / LLK / framework change was needed, and no `sem::`/`tensor::` boundary-crossing assumption was violated.

## Successes

- **Conditional / optional DFB bindings** fired exactly as documented for the conditionally-allocated right/bottom mask buffers (`c_1`/`c_2`). Host binds `RIGHT_MASK`/`BOT_MASK` only when `has_right_pad`/`has_bottom_pad`, emits `FILL_PAD_HAS_RIGHT_PAD`/`FILL_PAD_HAS_BOTTOM_PAD` via `compiler_options.defines`, and the writer + compute kernels `#ifdef`-gate the `DataflowBuffer` construction and every mask reference. This is the *promote-a-CTA-gate-to-a-define* sub-case (legacy gated with `if constexpr (has_right_pad)`); the catalog's warning that `if constexpr` still name-looks-up the discarded branch was directly load-bearing here — a naïve `if constexpr` port would not have compiled once `dfb::right_mask` stopped existing on the unpadded path.
- **`dfb.get_entry_size()` as the DM transfer/tile size** (whitelist §B) is the clean replacement for the legacy free `get_tile_size(cb_id)`; confirmed against ported peers (`copy/typecast`, `eltwise/binary_ng`) before adopting.
- **Case 2 raw-base bridge** (`TensorAccessor(tensor::input).get_bank_base_address()`) let the sharded reader/writer keep their hand-rolled `UnicastEndpoint` self-read/write arithmetic byte-for-byte while dropping the `shard_l1_base` address RTA — exactly as the recipe's rule 5 describes.
- **`opt_level = O3` on both compute `KernelSpec`s** — the legacy `ComputeConfigDescriptor` resolves to O3 while Metal 2.0's `CompilerOptions` defaults to O2; set explicitly per the recipe (and consistent with a prior observation that a compute kernel can fail to compile at O2 after porting).

## Friction

### Gaps
- None blocking. The recipe/patterns/headers covered every construct this port needed.

### Confusion
- **Sharded factory endpoint alignment (resolved, but non-obvious).** In the legacy sharded factory the reader/writer are grouped by `has_right_pad` while the compute is grouped by the full `ComputeKey`, and the *bottom* mask is produced at **runtime** (`has_bottom_pad_core` RTA) yet consumed at **compile-time** (`key.has_bottom_pad`). Under Metal 2.0's derived DFB placement + per-node 1P+1C invariant, binding the writer's `BOT_MASK` producer over a whole `has_right_pad` group would place the producer on `has_bottom_pad=0` nodes that have no consumer → invalid. Nothing in the recipe/catalog names this "producer/consumer groupings must align for a conditionally-produced DFB" interaction directly; I derived the fix (regroup reader/writer by the full `ComputeKey` so each `WorkUnitSpec` carries a matched {reader, writer, compute} triple). It is a behavior-identical multiplicity change (reader/writer binaries depend only on `has_right_pad`; splitting per-key yields identical binaries over disjoint nodes) with **no kernel-logic change** — the sharded writer still reads `has_bottom_pad_core` as a runtime arg (it also drives the Mode A/B write-back geometry). **Suggested catalog note:** when a conditionally-bound DFB's producer and consumer are grouped on different axes, align the coarser-grained kernel's grouping to the finer one so the endpoint placement matches per node.

## Open items for downstream

- **Dead-but-preserved args (faithful port; candidates for a separate cleanup PR, per the audit's "Misc anomalies"):**
  - `elem_size` CTA — read into an unused `constexpr` in `fill_pad_reader.cpp`, `fill_pad_compute.cpp`, `fill_pad_sharded_reader.cpp`.
  - `W_tiles` / `H_tiles` CTAs — read-but-unused in `fill_pad_compute.cpp` (the loops are driven by the per-phase counts).
  - `num_work` RTA — inert in `fill_pad_sharded_reader.cpp` (the sharded *writer* uses it as an early-return guard).
  These were kept as named args (not removed) to preserve exact behavior; removing them is a mechanical follow-up.
- **Sharded reader/writer KernelSpec redundancy:** grouping reader/writer by the full `ComputeKey` (rather than by `has_right_pad`) can emit multiple identical reader/writer `KernelSpec`s over disjoint node sets for keys that share `has_right_pad`. This is legal and behavior-identical but slightly more `KernelSpec`s than strictly necessary; a future refinement could group reader/writer by `(has_right_pad, has_bottom_pad)` only, if the framework's per-node DFB validation is confirmed to accept a producer/consumer split across differently-scoped WorkUnitSpecs.

## Verification

- Host build: `./build_metal.sh --build-tests` — clean (no compiler errors; ninja + install completed).
- On-device (`ARCH_NAME=blackhole`, `scripts/run_safe_pytest.sh --run-all`):
  - `tests/ttnn/unit_tests/operations/data_movement/test_fill_pad.py` — **280 passed, 0 failed** in ~197s (both factories; all dtypes bf16/fp32/uint32/int32/uint16/block-float; DRAM interleaved + HEIGHT/WIDTH/BLOCK sharded; incl. the WIDTH_SHARDED `(97,97)` case that was the #50904 revert root cause). No dispatch-timeout hang.
  - `tests/ttnn/unit_tests/operations/data_movement/test_pad.py` — indirect-consumer integration net (ttnn.pad routes tile-padding through fill_pad): **506 passed, 63 skipped, 8 xfailed, 0 failed** in ~245s.
- Anti-pattern self-audit: all checklist items clean (no `buffer()->address()`, no magic CB indices, no `TensorAccessorArgs`/`get_arg_val`/`get_compile_time_arg_val` residue, no `.id` extraction, no multi-binding flag, all CTAs named, no varargs, no `.md` citation from code, hw_config values match legacy, O3 on both compute specs).
