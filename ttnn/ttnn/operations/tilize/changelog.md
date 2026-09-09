# Changelog: tilize

## Phase 0 — Core Implementation

- **Date**: 2026-09-09
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). ROW_MAJOR → TILE re-lay as **one native TTNN
  dispatch**: a single `ttnn.generic_op` over one `ProgramDescriptor` with three
  kernels (reader on NoC0, compute, writer on NoC1), two circular buffers and zero
  semaphores. The work is a 2-D block grid over the output tile grid
  `R × C` — `num_row_groups × num_w_chunks`, linearized and handed to
  `split_work_to_cores(..., row_wise=True)` — so **both** grid axes are
  core-assignment indices and the split reaches the whole device on every geometry,
  including `short_wide` (`R = 1`) where a row-only split collapses onto one core.
  Compute is `compute_kernel_lib::tilize` (one call per block); the reader is
  `dataflow_kernel_lib::read_sticks_for_tilize` (one call per block, using
  `byte_offset_within_page` for wide-W column chunking); the writer is a raw
  block operation, justified in-file — the symmetric tiled-side helper does not exist
  and the untilize counterpart addresses by stick index, which is wrong for a
  TILE-layout destination.

- **SUPPORTED at Phase 0**:
  `dtype=[bfloat16]`, `output_dtype=[bfloat16]`, `low_l1=[False, True]`,
  `shard_api=["none"]`, `out_scheme=["interleaved"]`,
  `buffer=[dram_to_dram, dram_to_l1, l1_to_l1, l1_to_dram]`, `rank=[2,3,4,5,6]`,
  `orientation=["none"]`, `pad_mode=["none"]`, `pad_value=["none"]`,
  `alignment=["tile_aligned"]`, `tile_height=[32]`, `in_tile_height=["none"]`,
  `tile_grid=[single_tile, small, tall_narrow, short_wide, square_large]` (all five).
  `EXCLUSIONS` is empty.

- **Accuracy achieved**: PCC = **1.0**, max_abs_err = **0.0**, mean_abs_err = **0.0**,
  relative RMS err = **0.0**, got/true ratio = 1.0 with zero spread — on all 5 shapes
  in `test_tilize_precision_baseline.py` ((1,1,32,32), (1,1,64,128), (2,3,128,256),
  (1,1,1024,1024), (1,1,32,16384)). tilize does no arithmetic, so bit-identity is the
  contract and these are asserted as equalities, not tolerances.
  `comp_allclose`: `Max ATOL Delta 0.0, Max RTOL Delta 0.0` throughout.

- **Performance at Phase 0** (8×8 Wormhole, `--profile`, fresh cache, two dispatches
  each; core count reported alongside every duration as the prompt requires):

  | Shape | R × C | block | cores | device kernel ns | achieved |
  |---|---|---|---|---|---|
  | `[1,1,32,16384]` (perf focus, `attention:`) | 1 × 512 | 1 × 8 tiles | **64 / 64** | 13106 / 14133 | ≈160 GB/s |
  | `[1,1,16384,32]` (transposed pair, same tile count) | 512 × 1 | 8 × 1 tiles | **64 / 64** | 21220 / 20268 | ≈101 GB/s |

  Against an untuned 64-core DRAM→DRAM reference of ~190 GB/s (`double_buffer/report.md`),
  that leaves ≈1.19× of headroom on the flagged geometry and ≈1.9× on its transpose.
  Per-core L1 ranges from 20 KiB to 512 KiB against a 1.43 MiB budget, with **no tensor
  dimension** in the footprint expression at either `low_l1` setting.

- **Golden suite at Phase 0**: **249 / 4548 cells passing** (`supported_pass`), with
  `supported_fail = 0`, `xpass_drift = 0`, `xfail_wrong_mode = 0`,
  `supported_marked_xfail = 0`, `invalid_unexpected = 0` — all five categories the
  verifier reads as loud are clean. `xfail_expected = 1430`, `invalid_skipped = 2090`,
  `xfail_other = 304` (Blackhole-only arch gates: fp8 196, retile 108).
  Whole-run totals 266 / 4548 passed, 0 hangs. Per `verifier_report.json`.

- **Issues encountered** (all found and fixed during verification):
  1. **validate() check ordering** — the padding-contract checks
     (`_check_alignment_request`, the `output_padded_shape` validation) ran *ahead* of
     the SUPPORTED loop, so 83 cells outside SUPPORTED came back as `ValueError`
     instead of a registry support refusal and landed in `xfail_wrong_mode`. Split the
     malformed-request checks into unconditional (still first) and support-conditional
     (now after the gate). The immutable acceptance test stays green because
     `UnsupportedAxisValue` is a `NotImplementedError`, hence also a `RuntimeError`.
  2. **Rank-0/1 pad target rejected** — 30 of those 83 were
     `output_padded_shape rank 2 != input rank 0/1`. A rank-0/1 input has no tile dims
     of its own; the pad *synthesizes* them, so a rank-2 target is well-formed. The
     check now left-pads the input shape to 2 before comparing.
  3. **`_spec_of` mis-tagged every sharded call as `nd`** — measured on device, a live
     tensor's `memory_config()` populates *both* `shard_spec` and `nd_shard_spec`
     whichever API allocated it. Replaced with a discriminator that holds for fresh and
     live configs alike. Latent at Phase 0, load-bearing for Refinement 1.
  4. **The op was unreachable as `ttnn.tilize`** — the externally-authored graded case
     set dispatches through that name, so all 159 of its cases died at `AttributeError`
     before entering the op. Bound the alias in the op package's `__init__.py`; 10 of
     those cases now genuinely pass and the rest refuse honestly by axis.
  5. **Three SUPPORTED axes were under-claimed** — `rank` (`[4]` → `[2,3,4,5,6]`),
     `low_l1` (`[False]` → `[False, True]`), `buffer` (`[dram_to_dram]` → all four
     transitions). Each probed on device first, each bit-exact, each needing **zero**
     kernel or descriptor change; the `low_l1` A/B pair the suite grades is
     bit-identical on the `low_l1_forcing_width` geometry. Together with fix 1 these
     took `supported_pass` from 41 to 249 with nothing lost.
  6. **One `l1_ledger.md` accuracy defect** — the data-movement budget claimed "the
     fewest read transactions of any split that fills the grid", which the
     divisor-constrained `block_width_tiles` overshoots on rough `C` (2.08× on
     `[1,1,1,50304]`). Quantified in the ledger, and the fix (two disjoint core ranges
     in one `ProgramDescriptor`) folded into perf Refinement 6 rather than filed
     standalone.

  Open items carried to refinements rather than fixed here: `fp32 -> fp32` is not
  bit-identical (tf32 truncation — needs the `Fp32Mode::Lossless` triple, Refinement 5);
  `uint8 -> uint8` looks broken rather than unsupported (Refinement 5); the golden
  harness's `axes.py:_spec_of` carries the same nd-vs-legacy bug as fix 3 and is
  gate-side, so it was reported rather than edited; 4 golden-suite setup errors come
  from a `use_module_device` × `device_params` conflict that fails before the op is
  entered.

- **Tests added**: `test_tilize_precision_baseline.py` (5 shapes; PCC / max-abs /
  mean-abs / relative-RMS / got-true ratio spread). Pre-existing and still green:
  `test_tilize.py` (17, the immutable acceptance spec), `test_tilize_geometry.py` (13),
  and the three lever harnesses `test_tilize_lever_block_width.py`,
  `test_tilize_lever_input_depth.py`, `test_tilize_lever_write_batch.py`.
  **58 / 58 passing** across `tests/ttnn/unit_tests/operations/tilize/`.
