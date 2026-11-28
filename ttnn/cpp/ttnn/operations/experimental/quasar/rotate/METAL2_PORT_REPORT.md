# Metal 2.0 Port Report — experimental/quasar/rotate

## Outcome

`PORTED` — both factories (`NearestProgramFactory`, `BilinearProgramFactory`), in one change with `experimental/quasar/grid_sample` (shared kernels). Tests match the baseline exactly.

## Provenance

- **Recipe docs (this port):** `9df4aa1e011 2026-11-27 docs(metal_2.0): fix-ups from an adversarial review of the trial-port fixes`
- **Audit docs (inherited):** `9df4aa1e011 2026-11-27 docs(metal_2.0): fix-ups from an adversarial review of the trial-port fixes`

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept` on both factories (`device/rotate_device_operation.hpp`).
- **Custom hash:** none; default reflection hash untouched.
- **Pybind entry points removed:** none.
- **Device-op-class edits:** none beyond the two factory declarations.

## Verification

- **Build:** success. **Legality checks:** `validate_program_args` on (confirmed `True`); Watcher on, no errors.
- **Tests** (Wormhole n150, `-rs`). Baseline: the unmodified `pool/rotate` tests on `ttnn.rotate`, whose code is identical to the quasar copy before the port and whose kernels the port does not touch. After: the same tests pointed at `ttnn.experimental.quasar.rotate`, committed as `tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_quasar_rotate{,_extended}.py`.

  | | passed | failed | skipped |
  |---|---|---|---|
  | baseline (`pool/rotate`) | 392 | 0 | 0 |
  | after the port (quasar copy) | 392 | 0 | 0 |

## Handoff points

none.

## Successes

- **Dead CB guidance (audit CB endpoints / recipe Construct).** The nearest factory's sharded input buffer (`device/rotate_nearest_program_factory.cpp`, legacy `:169-182`) is borrowed from the input shard but no kernel references it; confirmed across interleaved, height-sharded and ND-sharded configs and dropped, with the reason written at the site.
- **Rule 4's new "name args that are read but unused".** Kept the nearest kernels' `input_batch` / `num_dfb_pages` as named args instead of dropping them.

## Friction

- Same `face_geometry` → `tile_format_metadata` gap as grid_sample (`device/rotate_bilinear_program_factory.cpp`, `tile_for_face_geometry`). Also hit an ambiguous `Tile` (the file has `using namespace tt;` and `using namespace tt::tt_metal;`); qualified it.
- The legacy `TT_FATAL` message "Not enough L1 for even a single CB page: output_cb_page_size=…" (`device/rotate_nearest_program_factory.cpp`) is kept verbatim, so it is the one hit the `cb` naming sweep reports; only the variables in its condition and arguments were renamed (`num_cb_pages` → `num_dfb_pages`).

## Open items for downstream

- **Shared kernel touches.** Bilinear now binds the grid_sample-local `compute_pool_2d.cpp` copy (`experimental/quasar/grid_sample/device/kernels/compute/compute_pool_2d.cpp`) instead of the main-tree original, and quasar grid_sample's converted `writer_grid_sample_interleaved.cpp`; its reader includes grid_sample's `grid_sample_reader_common.hpp`. Anyone changing those files must consider both ops.
- **Audit misc anomalies stand:** bilinear lets a sharded input pair with an interleaved output and still borrows the output buffer for the output DFB; worth an owner look.
- **Tests:** `test_quasar_rotate.py`, `test_quasar_rotate_extended.py` (user decision).
- **Adversarial review (pre-PR), unconfirmed edge case:** bilinear with an interleaved input and a sharded output whose shard grid has more cores than `ceil(total_sticks / shard_height)` (`device/rotate_bilinear_program_factory.cpp`, the `is_output_sharded` branch). The kernels run on the whole shard grid but runtime args are set only for the first `num_cores` cores. Legacy launched the extra cores with no runtime args (most likely a hang: compute waits for sticks the reader never sends); the port fails run-args validation with a clear error instead. Not exercised by any test; if it matters, the owner should decide what the extra cores should do.
- **Quasar (Gen2) follow-up, out of port scope:** the hand-built data-movement configs set only the Gen1 block, so on Quasar the DFBs use the default implicit sync. To be looked at by the Quasar uplift, not here.
