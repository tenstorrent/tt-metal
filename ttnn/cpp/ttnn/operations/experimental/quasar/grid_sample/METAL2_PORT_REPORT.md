# Metal 2.0 Port Report — experimental/quasar/grid_sample

## Outcome

`PORTED` — both factories (`GridSampleBilinearProgramFactory`, `GridSampleNearestProgramFactory`), in one change with `experimental/quasar/rotate` (shared kernels). Tests match the pre-port baseline exactly (see Verification).

## Provenance

- **Recipe docs (this port):** `9df4aa1e011 2026-11-27 docs(metal_2.0): fix-ups from an adversarial review of the trial-port fixes` (the port started under `c41f7f4ef0a`; the recipe was updated mid-port and the work was re-aligned to this revision)
- **Audit docs (inherited):** `9df4aa1e011 2026-11-27 docs(metal_2.0): fix-ups from an adversarial review of the trial-port fixes` (audit first run under `c41f7f4ef0a`, re-checked)

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept` on both factories, as the audit named (`device/grid_sample_device_operation.hpp`).
- **Custom hash:** none; default reflection hash over `GridSampleParams` untouched.
- **Pybind entry points removed:** none (the op never bound `create_descriptor`).
- **Device-op-class edits:** none beyond the two factory declarations.

## Verification

- **Build:** `./build_metal.sh --build-tests` — success.
- **Legality checks:** `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'`, confirmed `ttnn.CONFIG.validate_program_args == True`; Watcher on (`TT_METAL_WATCHER=10`), no Watcher errors.
- **Tests** (Wormhole n150; the confirmed set is both `pool` grid_sample test files pointed at `ttnn.experimental.quasar.grid_sample`, now committed as `tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_quasar_grid_sample{,_extended}.py`), run with `-rs`:

  | | passed | failed | skipped |
  |---|---|---|---|
  | before the port (unported quasar copy) | 471 | 2 | 127 |
  | after the port | 471 | 2 | 127 |

  - The 2 failures are the same two cases, with the same mismatch values (`test_grid_sample_bilinear_weight_format[channels=384-align_corners={False,True}-sharded_grid=False-grid_kind=precomputed-input_dtype=torch.bfloat16]`, max abs diff 1.30 and 3.16). Both **pass when run on their own**, before and after the port. The main-tree op (`pool/grid_sample`, untouched) fails the same two cases in a full run of its unit file with Watcher on (2 failed, 222 passed, 8 skipped) and passes them with Watcher off (224 passed, 8 skipped). So this is a pre-existing upstream issue exposed by Watcher's timing together with earlier tests in the file, not something the port or the quasar copy introduced; see Open items.
  - The 127 skips are the tests' own condition skips (e.g. "Precomputed grid only supports bfloat16", "Tensors too large for sharded output…"), identical before and after; none are hardware skips.

## Handoff points

none.

## Successes

- **Audit "look the other way" check (audit recipe, Out-of-directory coupling).** Added in the mid-port recipe update; it is what exposed that quasar rotate binds this op's writer and reader header (`experimental/quasar/rotate/device/rotate_bilinear_program_factory.cpp:327-329`). Without it the in-place writer conversion would have broken rotate's bilinear path. Resolved by porting both ops together.
- **Conditional / optional resource bindings (port_patterns.md).** Covered the compute copy's optional second-reader and tiled-output buffers (`device/kernels/compute/compute_pool_2d.cpp`, `SPLIT_READER` / `OUTPUT_TILED`) and the nearest writer's grid accessor (`GRID_INTERLEAVED`) without binding anything unconditionally.
- **Two-toucher DFB → 1P+1C.** The nearest writers' shared `fill` / `out` / sharded `grid_0`, and the sharded bilinear readers' `grid`, are plain 1P+1C assignments; no multi-binding flag was needed anywhere.

## Friction

- **Gap — `CBFormatDescriptor::face_geometry` has no `DataflowBufferSpec` field and no doc mentions it.** It is expressible: the legacy build turns a face geometry into the equivalent `Tile` and takes every unpack/pack value (face rows, face count, tile dims, tile size) from it (`tt_metal/jit_build/jit_build_options.cpp:29-37,100-131`); the Metal 2.0 DFB path feeds `tile_format_metadata` into the same build hook (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3317-3329`). So `tile_format_metadata = <that tile>` reproduces it exactly (`device/grid_sample_bilinear_program_factory.cpp`, `tile_for_face_geometry`). Worth a line in the migration guide's DataflowBufferSpec section.
- **Confusion — the recipe forbids anything from `experimental/quasar/` entering a port, and the shared-kernel rungs all assume a main-tree port.** This port *is* a quasar copy, and the user ruled out main-tree changes, so the borrowed `pool/generic` compute kernel got a fourth rung: a local copy inside this op, converted in place (bound by grid_sample and rotate). Did not read or reuse other quasar ports beyond checking that `experimental/quasar/pool_generic`'s converted compute kernel does not fit (no split reader, scratch-buffer packing, equal-width blocks).
- **Resolved by the mid-port recipe update** (noted only so the record is complete): validation forcing via `skip_validation` is gone (the env setting now documented is what was used), and the header locations / `hw_config` shapes now match the code.

## Open items for downstream

- **Shared kernel touches.**
  - `pool/generic/device/kernels/compute/compute_pool_2d.cpp` — rung: local copy (user decision; none of the recipe's three). Copied to `device/kernels/compute/compute_pool_2d.cpp` here and converted; bound by quasar grid_sample and quasar rotate. The original is untouched, no pointer comment. Remaining legacy binders of the original: `pool/generic/device/pool_multi_core_program_factory.cpp:939`, `pool/grid_sample/device/grid_sample_bilinear_program_factory.cpp`, `pool/rotate/device/rotate_bilinear_program_factory.cpp`.
  - This op's `device/kernels/dataflow/writer_grid_sample_interleaved.cpp` and `device/kernels/grid_sample_reader_common.hpp` — lent to quasar rotate; converted in place in the same change (bundled port of both ops).
- **Order- and timing-dependent test failures (pre-existing, upstream too) — tracked as #65096.** The two `test_grid_sample_bilinear_weight_format[channels=384-…-sharded_grid=False-grid_kind=precomputed-…bfloat16]` cases fail in a full-file run with Watcher on, on the main-tree op as well as the quasar copy, and pass in isolation or with Watcher off. Cause, per open issue https://github.com/tenstorrent/tt-metal/issues/65096 (root cause in the kernel library: #59368): `compute_pool_2d.cpp` packs the last, narrower channel block with the full-width untilize setup and writes 256 B past the output buffer at 384 channels, so the result depends on what sits next in L1. In the committed `test_quasar_grid_sample.py` those two cases are marked `xfail(strict=False)` with a link to #65096. Checked with three full runs of that file (Watcher on): the two cases report XFAIL; one run also had a single other failure that two further runs (one with a fresh kernel cache) did not reproduce. Its name was not captured; a stray write from #65096 landing on a different test is the likely explanation, but that is unconfirmed. The port keeps the behavior as it is (it is a functional bug, out of port scope).
- **Dead plumbing dropped (never read by a kernel):** nearest interleaved RTA `grid_sticks` (`device/grid_sample_nearest_program_factory.cpp`), compute placeholders CTA 17–37, the bilinear interleaved reader's unread CTA 5, the nearest sharded placeholder grid accessor. No argument a kernel reads was dropped.
- **Audit misc anomalies** (`METAL2_PREPORT_AUDIT.md`) stand: unused `global_circular_buffer.hpp` include, both nearest writers zeroing the shared fill page, and the quasar copy missing `verify_numerical_configuration`.
- **Temporary test copies became permanent** quasar tests (user decision): `test_quasar_grid_sample.py`, `test_quasar_grid_sample_extended.py`. The two cases above are marked as expected failures (non-strict) pointing at #65096.
- **Adversarial review (pre-PR):** two independent reviews (host factories; kernels) found no verified bugs. Follow-ups they raised, none blocking: (1) Quasar (Gen2) only — nearest writer 1 is bound as consumer of `grid` / `fill` / `out` but reads `get_write_ptr()`; fine on Gen1, unverified on Quasar's DFB implementation, for the Quasar uplift. (2) Optional hardening in the compute copy: a `static_assert` that `split_reader` / `is_output_tiled` agree with the `SPLIT_READER` / `OUTPUT_TILED` defines, so a future caller can't set one without the other. Left out of the port (not a port change).
