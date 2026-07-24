# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/data_movement/split`

*Opened at the start of the port; friction captured as it occurred, polished at the end.*

## Outcome

**`PORTED`** — the single factory (`SplitProgramFactory`, the op's only factory) is fully converted to
`MetalV2FactoryConcept` (`create_program_artifacts`) and both of its kernels are on Metal 2.0 named args +
binding tokens. The op has no other factories; the port is complete.

- Unit of work: `SplitDeviceOperation` → `SplitProgramFactory` (single factory), plus its two op-owned
  kernels (`reader_tm_tile_layout_split_two_chunks.cpp`, `writer_split_n_chunks_tile.cpp`).
- **Build:** clean (0 errors) after resetting a cross-wired build directory (see Friction › Environment).
- **Tests (confirmed baseline, run against this checkout's ttnn — `import ttnn` verified to resolve here):**
  - `tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_split.py`: **83 passed, 0 failed**.
  - `tests/ttnn/nightly/unit_tests/operations/data_movement/test_split.py`: **282 passed, 224 skipped (dim>rank), 0 failed**.
  - The native TILE device-op paths are covered directly: N-way equal split (the multi-output binding), program-cache reuse/rebuild, `num_splits=1`, interleaved↔sharded, bf16/fp32/bfloat8_b, the two-chunk `test_split_last_dim_kernel`, and odd-padded-tile fallback boundaries.

## Provenance

- **Recipe docs (this port):** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — `SplitProgramFactory::create_program_artifacts` returns
`ttnn::device_operation::ProgramArtifacts`. Single-variant `program_factory_t`; no
`select_program_factory` needed (framework auto-dispatches the sole variant, concept detected from the
method).

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op was already on the default reflection hash).
- Pybind entry points removed: **none** (`split_nanobind.cpp` binds only the free function `ttnn::split`;
  the factory entry point was never pybound).

### Open items
- **Multi-output per-core-subset binding is unworked in the docs.** The op produces `N = num_splits`
  outputs, each written by a disjoint core band. The recipe/catalog work examples of variable-count
  *input* tensors and of disjoint-node work-splits that vary the *compute* kernel, but not of binding
  `N` distinct *output* tensors across disjoint bands. The correct Metal 2.0 shape is `N` same-source
  writer `KernelSpec`s (identical CTAs), each with one `TensorBinding` to its chunk's output and one
  CONSUMER binding of the shared `src0` DFB, placed on its band; the reader is a single shared
  PRODUCER across all `N` work units. This is grounded in the `dataflow_buffer_spec.hpp` endpoint
  invariant ("multiple bindings on one endpoint over non-overlapping node sets"), but a worked catalog
  entry for *multi-output* binding would remove the judgment call for the next multi-output
  data-movement port. Confirmed with the invoker before building (per the brief's request).
- **Relaxations:** none. All `N` output `TensorParameter`s carry identical specs (equal split), and the
  input is bound strictly; no `dynamic_tensor_shape` / `match_padded_shape_only` warranted.

## Handoff points

None — the port is fully contained in the op's own directory (factory `.cpp`/`.hpp` + two op-owned
kernels + the four `METAL2_*.md` artifacts). No custom-hash deletion, no pybind surface removed, no
cross-op kernel touches, no kernel-lib / LLK / framework changes, no capitulation.

## Successes

- **The `dataflow_buffer_spec.hpp` endpoint INVARIANT (lines 41-50) directly answered the port's one
  non-mechanical question.** Its explicit "you MAY bind more than one KernelSpec to a producer (or
  consumer) endpoint … non-overlapping node coverage, same kernel kind, identical binding-site
  parameters" clause is exactly the multi-output writer shape. Without it I might have reached for one
  of the unexpressible single-KernelSpec shapes. Applied at
  [`split_program_factory.cpp`](device/split_program_factory.cpp) (the N CONSUMER bindings of `SRC0`).
- **The [Anti-pattern: Demoting per-group CTA to RTA] disjoint-node constraint note confirmed the shape
  and kept the writer CTAs as CTAs.** It states the "two `KernelDescriptor`s per work split" idiom maps
  1:1 to "same-source `KernelSpec`s in separate `WorkUnitSpec`s, both binding the shared DFBs," which is
  precisely the reader/writer topology here — so no dimension was demoted to an RTA.
- **The kernel-side whitelist made both kernels a pure syntax swap.** Because the kernels were already
  on the Device 2.0 `DataflowBuffer` object, the only additions were `#include "experimental/kernel_args.h"`
  and the `dfb::`/`tensor::`/`args::` token substitutions; the `Noc` transfers, FIFO ops, and loop
  structure were untouched, exactly as the whitelist prescribes.

## Friction

### Gaps
- **Recommended reference port (`accumulation`) not present in this checkout.** The recipe points to
  branch `akertesz/porting-experiment-accumulation-jun10` as the first worked `create_program_artifacts`
  example (`git show akertesz/...:<path>`). That ref does not resolve in this workspace
  (`git rev-parse` → "Needed a single revision"), so it could not be consulted. Not blocking — the
  migration guide's Example 1 (single-core reader/writer with one DFB) plus the framework headers were
  sufficient. Worth noting for porters who inherit a checkout without the sibling branch fetched.

### Environment (not a port issue)
- **Cross-wired build directory.** `build_Release/CMakeCache.txt` in this checkout
  (`.../git_2026_07_23_ops2_0_split/tt-metal`) was generated against a *different* source tree
  (`CMAKE_HOME_DIRECTORY = .../git_2026_07_23_ops2_0_baseline/tt-metal`), so `./build_metal.sh` fails at
  the CMake configure step ("source ... does not match the source used to generate cache") before any
  compilation. The two checkouts have *separate* physical `build_Release` directories (different
  realpaths / mtimes), so resetting this one does not affect the baseline. Resolution: reset this
  checkout's `build_Release` and cold-rebuild. Not a port defect — the port's C++ was never compiled at
  the point this was hit. **After the reset the cold build succeeded with 0 errors — the host-side port
  compiles cleanly** (`_ttnn.so` / `_ttnncpp.so` rebuilt for this checkout).

- **Virtualenv pinned to the baseline checkout (blocks on-device test verification).** This checkout's
  `python_env` is a setuptools *editable* install whose finder `MAPPING` hardcodes
  `ttnn → .../git_2026_07_23_ops2_0_baseline/tt-metal/ttnn/ttnn` (plus a `ttnn-custom.pth` with baseline
  paths). Consequences, confirmed with an in-tree diagnostic test run under pytest:
  `ttnn.__file__` and `_ttnn.so` both resolve to the **baseline** checkout, even with
  `PYTHONPATH`/`TT_METAL_HOME` exported to this checkout (a MetaPathFinder wins over `PYTHONPATH`).
  Meanwhile `TT_METAL_HOME` (auto-detected from CWD, or exported) resolves to **this** checkout, so the
  relative kernel path in the factory lands on *this* checkout's ported Metal 2.0 kernels. Net effect:
  the tests run **baseline's** unported `ProgramDescriptor` split factory but feed it *this* checkout's
  `args::`/`dfb::`/`tensor::` kernels, so the legacy JIT flow compiles them without the generated
  binding/arg headers → `'args'/'dfb'/'tensor' has not been declared`. **The port's own library is never
  loaded**, so this is neither a validation of nor a regression against the port. Resolution requires
  pointing this checkout's `python_env` at *this* checkout (re-create via `create_venv.sh`, re-run the
  ttnn editable install from here, or re-point the editable `.pth`/finder mappings) so `import ttnn`
  loads this checkout's freshly-built `_ttnn.so`. Root cause is the same as the build-dir issue: this
  checkout was set up as a copy of the baseline checkout with baseline paths baked in.
  **Resolved:** recreated the venv with `./create_venv.sh` (old one moved aside to
  `python_env.baseline-stale`); an in-tree diagnostic then confirmed `import ttnn` resolves to *this*
  checkout (`VERDICT: SPLIT`), and both nightly files passed against it (see Outcome). The
  `*.baseline-stale` backups (build dir + venv) are left in place for the workspace owner to remove or
  restore; they are not part of the port and are gitignored / not staged.
- **Takeaway for porters inheriting a copied checkout:** verify *both* the build dir
  (`build_Release/CMakeCache.txt` `CMAKE_HOME_DIRECTORY`) and the venv (`import ttnn; ttnn.__file__`)
  point at the checkout you are porting *before* building/testing. A silently baseline-wired venv is a
  false-GREEN hazard — tests would pass against the unported op.

### Confusion
- **"Identical WorkUnitSpec membership" vs the per-node DFB invariant.** The migration guide's
  troubleshooting bullet ("A local DFB's producer and consumer kernels must share *identical*
  `WorkUnitSpec` membership") reads, taken literally, as though a producer bound across N work units and
  a consumer bound in only one would be rejected — which would forbid both the catalog's *Demoting
  per-group CTA to RTA* worked example and this port's shape. The authoritative statement is the
  `dataflow_buffer_spec.hpp` INVARIANT (lines 41-50): it is a **per-node** rule (exactly one producer +
  one consumer instance per node), and it explicitly permits multiple bindings on one endpoint over
  non-overlapping node sets. Aligning the troubleshooting bullet's wording with the header invariant
  would remove the apparent contradiction.

## Open items for downstream

- **Non-gating anomalies (routed from the audit, not acted on by the port):** dead reader RTA
  `split_last_dim` (the port drops it as part of the buffer-address/plumbing sweep, which is in scope);
  vestigial single-iteration multi-tensor scaffolding in the reader (`out_num_tensors = 1`, left as-is —
  kernel logic is not the port's to rewrite); stale kernel filename
  `reader_tm_tile_layout_split_two_chunks.cpp` (it is the generalized N-chunk reader; not renamed, out of
  scope).
- **Cross-op kernel touches:** none — both kernels are owned by the op directory.
