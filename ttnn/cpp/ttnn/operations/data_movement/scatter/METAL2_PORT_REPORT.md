# Metal 2.0 Port Report — `data_movement/scatter`

*Written during the port; captures handoffs, successes, friction, and open items for downstream.*

## Outcome

`PORTED` — both factories (`ScatterProgramFactory` and `ScatterReduceBfloat16ProgramFactory`)
converted to `MetalV2FactoryConcept`. Host build green (`./build_metal.sh --build-tests`), and the
confirmed no-regression baseline passes: **95 passed, 2 skipped, 2 xfailed, 0 failed** across the
two unit-test files (99 cases, 11 test functions). The 2 skips (`dim=-2` invalid for a rank-1
tensor) and 2 xfails (an unsupported forge shape config) are test-defined outcomes, unchanged by
the port — the xfails stay `xfailed` (not `xpassed`), so behavior is preserved. Coverage exercised
both factories (bf16 + reduction → reduce factory; everything else → general), TILE and ROW_MAJOR
layouts, uint16 / int32 / uint32 index dtypes, sub-core grids, program-cache re-dispatch
(`*_with_callback` cases), and the `tosa_scatter` host wrapper. No device hang.

- Verified test set (confirmed with invoker): `tests/ttnn/nightly/unit_tests/operations/data_movement/test_scatter.py`, `tests/ttnn/unit_tests/operations/data_movement/test_tosa_scatter.py`.

## Provenance

- **Recipe docs (this port):** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for both factories — the default single-program path. No re-decision
against the audit.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op had no custom hash).
- Pybind entry points removed: none (`create_descriptor` was never pybound; the pybind files bind
  only the host `scatter` / `scatter_add` / `tosa_scatter` functions).

### Open items
- *(see Open items for downstream)*

## Handoff points

None. The port stayed entirely within the op directory: no out-of-op kernel edits, no
kernel-lib / LLK gaps, no `sem::`/`tensor::` boundary-rule violations, no capitulation, and no
pybind entry point removed (`create_descriptor` was never pybound).

## Successes

- **Self-loop DFB pattern ([port_patterns.md — Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)) fit exactly.**
  `INPUT`/`INDEX`/`SRC` (and `FP32_TEMP` in the reduce factory) are each touched only by the reader,
  which both fills and drains them; binding the reader PRODUCER + CONSUMER with a shared
  `accessor_name` gave one `dfb::name` handle and passed the ≥1-producer/≥1-consumer validator with
  no code change to the kernel FIFO calls. The re-derived census matched the brief with no
  multi-binding flag anywhere (`scatter_program_factory.cpp` reader `dfb_bindings`).
- **Kernels were already on Device 2.0**, so rule 1 (CircularBuffer → DataflowBuffer) was a no-op —
  the kernels already used `DataflowBuffer` objects, `Noc`, `TensorAccessor`, `CoreLocalMem`. The
  entire kernel-side change reduced to: named-arg retrieval, `tensor::`/`dfb::` construction,
  `get_dataformat(dfb::name)`, and the vararg switch. This matches the audit's "unusually far along"
  note and made the port predominantly a host-side spec rewrite.
- **Vararg caution ([port_patterns.md — Avoid varargs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)) steered the split correctly.**
  The nine leading reader scalars are distinct fields read once → named RTAs; only the two
  per-dimension shape blocks (count = rank-1, no stable per-element name) went to varargs. The
  caution's "distinct field → named, indexed-collection element → vararg" rule drew the line
  exactly where the kernel does (`common.hpp` `make_shape_array_from_runtime_args` reads
  `get_vararg(i)` in a loop).

## Friction

### Gaps

- **`get_dataformat` can't move onto the DFB object in a compile-time context (rule 7 vs. reality).**
  Kernel-side whitelist rule 7 says compile-time format metadata should move from the free
  `get_dataformat(cb_id)` to the object getter `dfb.get_dataformat()`. But the scatter kernels use
  it in a **non-type-template-argument** position — `std_type_t<get_dataformat(...)>` — which
  requires a *constant expression*. `DataflowBuffer`'s constructor is not `constexpr`
  (`dataflow_buffer.h:75`), so a `DataflowBuffer` object can't be built in a constant expression,
  and `dfb.get_dataformat()` therefore cannot appear there. The only compile-time-correct form is
  the free function fed the named handle: `get_dataformat(dfb::input)` — the constexpr
  `DFBAccessor → uint32_t` conversion (`dataflow_buffer.h:55`) plus the constexpr free
  `get_dataformat(operand)` (`dataflow_api.h:300`). This is really rule 2 (pass the DFB handle
  directly to a helper taking a `uint32_t` cb id), not a rule-7 violation, but rule 7 reads as
  unconditional ("these lines must be rewritten — query the object"). **Doc suggestion:** rule 7
  should carve out the constexpr/template-argument case, where the object getter is unusable until
  `DataflowBuffer` gains a `constexpr` constructor, and point at the free-function-via-`dfb::name`
  form instead. (Kernel sites: `reader_scatter.cpp` `using input_std_type = ...`,
  `writer_scatter.cpp` `using output_std_type = ...`, and the two reduce kernels.) Note: the runtime
  call `input_dfb.get_dataformat()` inside `scatter_along_chunk` already uses the object getter and
  needed no change — only the compile-time-constant sites hit this.

### Confusion

- **Two files named `scatter_common.hpp`.** A host-side `device/scatter_common.hpp` (`ScatterCB`
  enum + `ceil32` + `calculate_optimal_chunk_size`) and a kernel-side
  `device/kernels/scatter_common.hpp` (`ScatterCTAs` + `get_ctas()`). The audit's Device 2.0 bullet
  cites `scatter_common.hpp:18-21` meaning the *kernel-side* one; easy to mis-resolve. Not a doc
  problem, but a naming hazard worth a one-line flag for the next reader.

- **Environment (workspace-setup, not a Metal 2.0 recipe issue) — this checkout was cloned/copied
  from a sibling `baseline` checkout, and three baseline-pointing artifacts blocked verification
  until made self-consistent.** Each cost a build/test cycle to diagnose: (1) `build_Release/`
  carried a stale `CMakeCache.txt` whose `CMAKE_HOME_DIRECTORY` was the baseline path, so
  `build_metal.sh` refused to configure — fixed by `rm -rf build_Release` and rebuilding fresh;
  (2) the `python_env` venv had a PEP 660 editable `ttnn` install whose `MetaPathFinder`
  (`__editable___ttnn_*_finder.py`) pointed at the baseline checkout and overrode `PYTHONPATH`, so
  `import ttnn` loaded baseline's `_ttnn.so` (running baseline's *legacy* scatter factory against
  this checkout's Metal 2.0 kernel sources — a guaranteed JIT mismatch) — fixed by
  `./create_venv.sh --force`; (3) the JIT kernel build keys off `TT_METAL_HOME`, which must be set
  to this checkout for the generated `kernel_args_generated.h` / `kernel_bindings_generated.h` to be
  injected with this checkout's headers. **Doc suggestion for `workspace_setup.md`:** a porter
  handed a checkout copied from another one should verify, before building, that `build_*` caches,
  the venv's editable-install finder, and `TT_METAL_HOME` all point at the working checkout — not
  the source it was copied from. The recipe's `PYTHONPATH=$(pwd)` note is necessary but not
  sufficient when a MetaPathFinder-based editable install is present (it silently wins over
  `PYTHONPATH`).

## Open items for downstream

- **RTA → CRTA / common-vararg conversion (deliberately not done).** `input_and_output_chunk_size`,
  `index_chunk_size`, `source_chunk_size`, `scatter_reduction_type`, and both per-dimension shape
  vararg blocks hold the **same value on every node**. They are faithfully ported as per-node RTAs /
  per-node varargs (mirroring the legacy per-core emission). Converting them to common runtime args
  / common runtime varargs would cut dispatch traffic, but RTA→CRTA changes dispatch semantics and
  is explicitly out of scope for a Metal 2.0 port ([recipe: Construct](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#construct-paired-spec--run-args)). Good candidate for a follow-up cleanup.
- **Dead compile-time args removed.** Beyond the four dead buffer-address CTAs the brief directed
  removing, the audit's Misc-anomalies also flagged `output_stick_size` and
  `input`/`index`/`source_stick_size_bytes` as declared-but-never-read. These fell away naturally:
  the port replaces the positional `get_ctas()` struct (which read all 17 CTAs regardless of use)
  with per-kernel named-arg reads, so only kernel-referenced CTAs are re-emitted
  (`input`/`index`/`source_stick_size`, `input_rank`, `output_stick_size_bytes`). Zero
  functional change (the fields were never read).
- **Kernel-side per-factory common headers deleted (in-op, not cross-op).**
  `device/kernels/scatter_common.hpp` and `device/kernels/scatter_bf16_reduction_common.hpp` held
  only the `ScatterCTAs` struct + `get_ctas()` (the positional-CTA plumbing the port removes); with
  that gone they were empty indirection over `common.hpp`, so they were deleted and the four kernels
  now `#include "../common.hpp"` directly. These are the op's own files (not shared with sibling
  ops), so no cross-op coordination is needed.
- **`ScatterCB` enum removed** from `device/scatter_common.hpp` — it was the magic-CB-index
  vocabulary that named DFBs replace; unused after the port.
