# Metal 2.0 Port Report — `data_movement/indexed_fill`

## Outcome

**`PORTED`.** The single `IndexedFillProgramFactory` (all four internal geometry paths — generic, native, shard-local interleaved-B, shard-local same-sharded-B) is on `MetalV2FactoryConcept`. The full confirmed test set passes: **61 passed, 0 failed, 0 skipped** (`tests/ttnn/unit_tests/operations/data_movement/test_indexed_fill.py`, `tests/tt_eager/python_api_testing/unit_testing/misc/test_indexed_fill.py`, `tests/ttnn/docs_examples/test_data_movement_examples.py::test_indexed_fill`). No C++ gtests exist for this op.

## Provenance

- **Recipe docs (this port):** `ca0b78e9ad7 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `ca0b78e9ad7 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — `IndexedFillProgramFactory::create_program_artifacts` returns `ProgramArtifacts` (spec + run-args; no op-owned tensors). The four internal geometry paths branch inside the single factory method.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** — the op already used the default reflection-based hash.
- Pybind entry points removed: **none** — the nanobind exposes only the free function `ttnn::indexed_fill`; no `create_descriptor` was ever pybound.

### Open items
- **Tensor-arg matching kept strict** on all four `TensorParameter`s; no relaxation applied and none surfaced as a candidate (the kernels are geometry-specialized per program-cache key, so relaxed shape matching would not be a faithful mirror of any legacy behavior).
- **No device-op-class edits were needed.** No custom hash to delete, no pybind factory entry point to remove, no pybind-hook-only factory parameter to drop. The device-operation class is byte-for-byte unchanged; only the program factory body was rewritten.

## Handoff points

none. The port stayed entirely within the op's own directory (factory + its three kernels + the four artifacts). No capitulation, no `sem::`/`tensor::` boundary-rule assumption violations, no kernel-lib/LLK gaps, no framework gaps, and no pybind surface removed (the nanobind exposes only the free function `ttnn::indexed_fill`, which is untouched).

## Successes

- **Borrowed-memory DFB + wait/pop stub writer ported mechanically.** The legacy native/shard-local paths globally-allocated the data CB onto `output.buffer()` and used a wait/pop stub writer. `DataflowBufferSpec::borrowed_from = IF_OUTPUT` expressed this in one line; the validator ([`program_spec.cpp:540`](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L540)) counts the `borrowed_from` reference as a TensorParameter user, so the output parameter needs no kernel `TensorBinding` on those paths — exactly as the migration guide's borrowed-memory section and the transpose/s2i reference ports indicated. No friction.
- **Self-loop for the single-ended batch DFB was the right call, straight from the catalog.** The batch DFB is filled by the reader and read back through a raw `get_write_ptr` pointer (never a FIFO consume). [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb) named this exactly (one toucher → bind PRODUCER+CONSUMER, shared accessor), and the DM self-loop is Gen1-legal ([`program_spec.cpp:1439`](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L1439) rejects it only on Gen2). The census matched the brief.
- **Case 2 `get_bank_base_address` bridge dropped in with zero arithmetic change.** The shard-local `input_a` (and same-sharded `input_b`) raw-base reads became `s0.get_bank_base_address()` / `s1.get_bank_base_address()`; the existing offset arithmetic (`input_addr_a + a_offset`, etc.) is untouched, per [kernel-side whitelist rule 5](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#kernel-side-whitelist).

## Friction

- **Gap — the conditional-*named-argument* case is undocumented, though it has the same mechanics as conditional bindings.** The docs cover conditionally-bound resources (`dfb::`/`tensor::`/`sem::`) and their `#ifdef` gating thoroughly ([Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)), but not the parallel case for *named args*: a single kernel source whose `if constexpr (mode)` branches read different `args::` names. Because `args::<name>` tokens are emitted per-kernel from the schema, and C++ name lookup still runs on discarded `if constexpr` branches in the non-template `kernel_main`, every `args::` name any branch references must be declared for every compiled path. The brief mandated keeping the `if constexpr` structure, so the faithful resolution is a **union RTA schema** on the reader (14 names) with the host writing `0` for the names a given path omits — see [`indexed_fill_program_factory.cpp:332`](device/indexed_fill_program_factory.cpp#L332). A short catalog note ("named args obey the same discarded-branch name-lookup rule as `dfb::`/`tensor::`; either union the schema or `#ifdef`-gate the reads") would have saved a first-principles derivation. I chose the union schema over `#ifdef`-promotion specifically because the brief said keep `if constexpr`.
- **Confusion (minor) — Case 2 base-address computation in a multi-path shared kernel needs scoping care.** `get_bank_base_address()` replaces the legacy address RTA, but where the legacy read that RTA unconditionally at the top (it fed the always-constructed `TensorAccessor`), the Metal 2.0 base is needed *only* in the shard-local branch. Computing it at the top makes it an unused variable in the generic/native compiles. Resolved by moving the two `get_bank_base_address()` calls inside the `if constexpr (IS_SHARD_LOCAL)` block (and `[[maybe_unused]]` on `input_addr_b`, used only in same-sharded mode) — see [`indexed_fill_reader.cpp:67`](device/kernels/dataflow/indexed_fill_reader.cpp#L67). Not a doc gap so much as a wrinkle worth a one-line heads-up for Case 2 bindings inside runtime-selected paths.

## Open items for downstream

- **MODE_SHARD_LOCAL_SHARDED_B not covered by the confirmed test set.** The Case 2 `input_b` raw-base path (input_b same WIDTH_SHARDED grid as input_a → `s1.get_bank_base_address()`, [`indexed_fill_reader.cpp:104-113`](device/kernels/dataflow/indexed_fill_reader.cpp#L104-L113)) is preserved byte-for-byte but is not exercised by the confirmed no-regression tests (which cover generic, native, and shard-local interleaved-B via the block-sharded test). A WIDTH_SHARDED input_b test would close this gap; flagging for test-coverage follow-up.
- **Batch DFB carries `input_a`'s data format though it stores `uint32` indices** ([`indexed_fill_program_factory.cpp:302`](device/indexed_fill_program_factory.cpp#L302)). Preserved verbatim from the legacy CB; the field is inert (the buffer is filled/read by raw byte access at a byte page size, never as tiles), as the audit's "Misc anomalies" noted. The ops team may wish to set an integer/`UInt32` format for clarity — not porter work, so left as-is.
- **Cross-op kernel touches:** none. All three kernels are owned by this op and instantiated from its own directory; no shared/donor kernel sources were modified or forked.
