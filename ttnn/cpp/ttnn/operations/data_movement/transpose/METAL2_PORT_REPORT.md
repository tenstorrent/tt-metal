# Port Report — transpose (data_movement)

## Post-test fix (cross-op shared kernels — round 2)

First on-device run: 49 failed / 400 passed / 70 skipped. Root cause: **three transpose
kernels I Metal-2.0-ified in place are cross-op shared** — reused *by file path* from
other, still-legacy ops that live outside the transpose directory:

- `kernels/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` — also
  bound by the legacy **permute** op (`permute_tiled_program_factory.cpp`).
- `kernels/compute/transpose_wh.cpp` — also bound by legacy **permute**,
  **nlp_create_qkv_heads{,_boltz,_vit}**, **split_query_key_value_and_split_heads**.
- `kernels/compute/transpose_wh_sharded.cpp` — also bound by legacy **create_qkv_heads**,
  **create_qkv_heads_from_separate_tensors**,
  **split_query_key_value_and_split_heads_sharded**.

Those peers create the kernel via the legacy `CreateKernel` path (`is_metal2_kernel =
false`), so the JIT emits **no** `kernel_args_generated.h` / `kernel_bindings_generated.h`
for them (`genfiles.cpp` fences the Metal 2.0 headers behind `is_metal2_kernel()`). My
ported kernels reference `args::` / `dfb::` / `tensor::`, which then don't exist →
`'args'/'get_arg'/'tensor'/'dfb' has not been declared`. The two kernels the coordinator
saw (permute-routed high-rank / `test_transpose_16411` / `test_transpose_21803`, and the
tiled-padding-aware reader) are exactly this; `transpose_wh_sharded.cpp` was a *latent*
instance (the transpose suite exercises it only through my metal2 factory, but the qkv ops
would have broken).

**Fix (per `port_patterns.md` — "Modifying a shared dataflow kernel", fork path):** restored
the three legacy kernels verbatim (from the pre-port commit) so the peer ops keep compiling
them non-Metal-2.0, created Metal 2.0 forks in the op dir, and repointed only the three
transpose factories at the forks:

- `..._tiled_padding_aware_transpose_m2.cpp` ← HC-Tiled-Interleaved reader
- `transpose_wh_transpose_m2.cpp` ← WH tiled compute
- `transpose_wh_sharded_transpose_m2.cpp` ← WH-Sharded compute

No factory logic changed beyond the three `KernelSpec::source` paths. All other in-place
kernels were confirmed transpose-only (`grep` across `ttnn/cpp`, excluding the transpose
dir and the separate `experimental/quasar` tree). The legacy inventory should have flagged
these three as cross-op — they live *inside* the transpose dir but are consumed by peer ops
outside it, which the file-path-only inventory sweep missed. Noted as a recipe/audit gap
below.

## Outcome

`PORTED` — the 6 clean factories (`TransposeCNProgramFactory`,
`TransposeHCRMProgramFactory`, `TransposeHCTiledInterleavedProgramFactory`,
`TransposeHCTiledProgramFactory`, `TransposeWHProgramFactory` [tiled + row-major],
`TransposeWHShardedProgramFactory`) converted to `MetalV2FactoryConcept`. The 2 gated
factories (`TransposeHCShardedProgramFactory`, `TransposeWHShardedRMProgramFactory`) left
on the legacy `create_descriptor` path. **Build and on-device test verification are
performed by the orchestrator** (this port did not build or run tests, per the
orchestration constraints).

## Provenance
- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1
  porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters
  away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for all 6 clean factories, matching the audit. Each factory's
`create_descriptor(...) -> ProgramDescriptor` became
`create_program_artifacts(...) -> ttnn::device_operation::ProgramArtifacts`. The device
operation class was not modified (variant unchanged; the framework dispatches per-factory
by concept; the 2 gated factories keep `create_descriptor`).

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op already used the default hash).
- Pybind entry points removed: none (nanobind binds only the `transpose` free function; no
  `create_descriptor` was pybound).

### Open items
- **Relaxation candidate (`dynamic_tensor_shape`).** CN / HC-RM / HC-Tiled-Interleaved /
  WH interleaved legacy host code emitted `TensorAccessorArgs(buffer,
  ArgConfig::RuntimeTensorShape)`. Per the migration guide these tensors *could* take
  `TensorParameter::advanced_options.dynamic_tensor_shape = true`. The audit recorded
  relaxation = none and the recipe bias is strict, so the port uses **strict** tensor
  parameters. Left as a downstream tuning candidate — the strict form is correct (TTNN's
  program-cache key already folds the tensor spec, so a shape change forces a fresh
  program regardless), just with narrower cache equivalence than the legacy runtime-shape
  path nominally allowed.

## Handoff points

### Removed pybind surface
none.

### Boundary-rule assumption violations
none — no out-of-op call site required a `sem::` or `tensor::` handle.

### Kernel-lib / framework gaps
none encountered.

## Successes
- **Self-loop / conditional-DFB patterns** (`port_patterns.md`) applied cleanly to the
  HC-Tiled scratch `c_1`, HC-Tiled-Interleaved padding `c_1`, and WH-RM tilize `c_24`
  buffers — each `#ifdef`-gated on the host-emitted define, as the catalog prescribes.
- **Shared top-level entry point detection** (recipe §atomic-unit) caught
  `transpose_wh_rm.cpp` being bound by both the in-scope WH factory and the gated
  WH-Sharded-RM factory; the fork kept the gated factory building on its legacy path.
- **Dead-CB drop** — the brief's `c_25` call-out matched the code (`// TODO REMOVE`,
  `#ifdef SHARDED`-only use); dropped with zero functional change.

## Friction

### Gaps
- **Inventory missed *inbound* cross-op kernel sharing.** The recipe's cross-op guidance is
  framed around kernels the ported op *borrows* from elsewhere (out-of-dir sources). It does
  not call out the reverse: kernels that live *inside* the op's own directory but are
  reused by *other* (legacy) ops via file path. Metal-2.0-ifying such a kernel in place
  silently breaks the peer op's legacy JIT (no `kernel_args_generated.h` for a non-metal2
  build → `args`/`dfb`/`tensor` undeclared). This cost a full test round. A legacy-inventory
  step of "grep every kernel in this op's dir for references from outside the op dir
  (`grep -rl <kernel> ttnn/cpp/... | grep -v <op dir>`); fork any with a legacy consumer"
  would have caught all three up front. Strong candidate for the recipe's Legacy Inventory /
  cross-op section.
- **`DST_ACCUM_MODE` injection contract undocumented.** The recipe/migration guide don't
  state whether Metal 2.0's `ComputeGen1Config::enable_32_bit_dest` still injects the
  `DST_ACCUM_MODE` compile define into the compute JIT (as legacy `fp32_dest_acc_en` did).
  The WH-RM compute kernel depends on it. A one-line note in the Hardware-configuration
  section would remove the guess. (See Notes #2.)

### Confusion
- **Audit "relaxation = none" vs. `RuntimeTensorShape` in the host code.** The audit's
  Port-work summary says "TensorParameter relaxation: none", but CN/HC-RM/
  HC-Tiled-Interleaved/WH host factories clearly append
  `TensorAccessorArgs(..., ArgConfig::RuntimeTensorShape)`. Resolving this required
  reading the migration guide's TensorParameter pre-flight plus the ttnn_factory strict
  bias to conclude the audit's "none" was the intended call (strict is correct; the
  runtime-shape reuse is dead at the TTNN cache layer). A one-line audit note that
  `RuntimeTensorShape` was seen and deliberately not relaxed would have removed the
  ambiguity.

## Open items for downstream

### Cross-op kernel touches (forks)
The orchestration constraint (only files under the op directory) meant every out-of-op
donor was **forked into** `transpose/device/kernels/dataflow/` rather than modified
in place. Legacy copies are untouched; the forks are Metal-2.0-only:

- `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` →
  forked as `transpose/device/kernels/dataflow/writer_unary_interleaved_start_id_transpose_m2.cpp`
  (used by HC-Tiled + WH-tiled). Remaining unmigrated consumers: ~42 host files
  (all other eltwise/unary + siblings that reference the legacy copy).
- `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` →
  forked as `transpose/device/kernels/dataflow/reader_unary_sharded_transpose_m2.cpp`
  (WH-Sharded). Remaining unmigrated consumers: ~17 host files.
- `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` →
  forked as `transpose/device/kernels/dataflow/writer_unary_sharded_transpose_m2.cpp`
  (WH-Sharded). Remaining unmigrated consumers: ~15 host files.
- `transpose/device/kernels/compute/transpose_wh_rm.cpp` (in-directory, but a shared
  top-level entry point with the gated WH-Sharded-RM factory) → forked as
  `transpose/device/kernels/compute/transpose_wh_rm_transpose_m2.cpp` with the
  `#ifdef SHARDED` branch stripped. The legacy `transpose_wh_rm.cpp` stays for the gated
  factory; sunset the fork when that factory ports.

Round-2 forks (transpose-owned kernels that turned out to be **inbound** cross-op shared —
consumed by legacy peer ops outside the transpose dir; legacy originals restored, forks
added):

- `kernels/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware_transpose_m2.cpp`
  — HC-Tiled-Interleaved reader. Legacy consumer left on the original: **permute**.
- `kernels/compute/transpose_wh_transpose_m2.cpp` — WH tiled compute. Legacy consumers:
  **permute**, **nlp_create_qkv_heads{,_boltz,_vit}**,
  **split_query_key_value_and_split_heads**.
- `kernels/compute/transpose_wh_sharded_transpose_m2.cpp` — WH-Sharded compute. Legacy
  consumers: **create_qkv_heads**, **create_qkv_heads_from_separate_tensors**,
  **split_query_key_value_and_split_heads_sharded**.

These forks are the sunset checklist: when the sibling consumers migrate, the shared
donor rewrites can land once and the transpose forks can be retired.

### Test-coverage notes
See "Test commands" section below; the primary transpose coverage lives outside
`unit_tests/operations/` (recipe warns about this for transpose specifically).

## Test commands (for the orchestrator to run)

Build:
```
./build_metal.sh --build-tests
```

C++ gtests (transpose ops):
```
./build/test/ttnn/unit_tests_ttnn --gtest_filter='*Transpose*'
```

Python pytests (the confirmed transpose coverage — split across trees; transpose has no
`test_transpose.py` under `unit_tests/operations/`):
```
pytest tests/ttnn/unit_tests/operations/data_movement/test_transpose.py -q
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_transpose.py -q
pytest "tests/ttnn/unit_tests/operations/test_transpose.py" -q
```
(Exact confirmed set to be reconciled with the invoker; the transpose sweeps under
`tests/sweep_framework/sweeps/data_movement/transpose/` also exercise the op. Because the
2 gated factories are left on the legacy path, tests that select native height-sharded RM
transpose / WH-sharded-RM continue to run the descriptor path unchanged.)

## Notes / uncertainties for review

The port was written without a local build (orchestrator builds/tests). Points a reviewer
/ the build should scrutinize:

1. **Compute `hw_config` wrapping.** The WH / WH-Sharded factories build a
   `ComputeGen1Config` and assign `.hw_config = ComputeHardwareConfig{compute_cfg}`. If the
   double-variant wrap doesn't resolve, wrap or `using`-qualify as needed. `UnpackMode`,
   `ComputeGen1Config`, `ComputeHardwareConfig` are expected visible transitively via
   `ttnn/metal_v2_artifacts.hpp` + the `tt::tt_metal` / `tt::tt_metal::experimental`
   using-directives; if not, add `<tt-metalium/base_types.hpp>`.
2. **`DST_ACCUM_MODE` on the WH-RM compute fork.** `compute_num_blocks_per_col` reads
   `DST_ACCUM_MODE` unconditionally; the legacy factory only *explicitly* defined it for
   INT32/UINT32, relying on the framework to inject it from `fp32_dest_acc_en` for the
   Float32/bf16 cases. The port preserves that: it sets `enable_32_bit_dest` (expected to
   drive the same JIT injection) and keeps the explicit `DST_ACCUM_MODE=1` define for
   INT32/UINT32. If Metal 2.0 does *not* inject `DST_ACCUM_MODE` from `enable_32_bit_dest`,
   the fork will fail to compile for Float32/bf16 — flag for the compute-config owner.
3. **Strict tensor parameters despite `RuntimeTensorShape`** — see Open items; deliberate,
   matches the audit.
4. **Borrowed-DFB-only tensor params (WH-Sharded).** `WHS_INPUT` / `WHS_OUTPUT` are
   referenced only via `DataflowBufferSpec::borrowed_from` (no `TensorBinding` on any
   kernel). This mirrors the `interleaved_to_sharded` quasar reference (its borrowed
   `I2S_OUTPUT` path), where `borrowed_from` alone satisfies the
   "every TensorParameter needs ≥1 use" validator. If the validator rejects it, a
   TensorBinding would need adding — but the reference indicates it is accepted.
5. **Vestigial `aligned_page_size` CTA** dropped from WH-RM reader/writer (legacy emitted a
   CTA the kernel never read, past its `TensorAccessorArgs<N>` boundary). Confirmed by the
   slot arithmetic; noted in case a reviewer expects a 1:1 CTA count.
