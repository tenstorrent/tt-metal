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

---

## Refinement 1 — Sharded and L1 placement: `shard_api`, `out_scheme`, `orientation`

- **Date**: 2026-09-09
- **What was done**: the design's deferred `grid2d_sharded` regime, as a knob-turn on the
  Phase 0 schedule rather than a second implementation.

  **The block grid is now READ off a shard spec instead of solved.** tilize has no
  dependent axis, so a shard IS a block: a HEIGHT shard is a `row_group`, a WIDTH shard a
  `w_chunk`, a BLOCK shard the 2-D block the op already builds — no cross-core combine, no
  mcast, no semaphore. `shard_partition()` re-expresses a shard spec (either API) as
  `num_shard_rows x num_shard_cols` blocks of `shard_rows_tiles x shard_cols_tiles`, and the
  shard's LINEAR index is the `block_id` the three kernels already derive `(row_group,
  w_chunk)` from — so `tilize_reader/compute/writer.cpp`'s block arithmetic is unchanged.
  The shard→core order is `corerange_to_cores(grid, n, row_wise=(orientation == ROW_MAJOR))`,
  which is the enumeration `buffer.cpp:generate_buffer_page_mapping` places shards with, so
  COL_MAJOR needs no separate path. One new runtime arg, `block_stride` (1 on the solved
  plan, the core count on the shard plan), covers ND's ROUND_ROBIN_1D placement where a core
  owns shards `{i, i+N, i+2N, ...}`.

  **The sharded side is consumed natively — a zero-copy CB placed on the shard buffer.**
  `_cb_on_shard()` wraps `ttnn.cb_descriptor_from_sharded_tensor`, re-stating the page size
  as the tile-sized unit the helpers count in (the shard's own paging is one ROW_MAJOR
  stick; `block_width_tiles` sticks-worth of contiguous bytes IS one tile-row page group,
  because `block_width_tiles == shard_cols_tiles` there). On that path the reader issues no
  NoC read at all — it marks the block's resident pages available — and on the native OUTPUT
  path the program carries **no writer kernel**: the packer has already written every tile
  into the output shard. A core's own L1 shard is never re-read through a `TensorAccessor`.
  The accessor keeps the interleaved leg and the genuinely non-local legs (cross-spec,
  DRAM-sharded), which is where a remote transfer is the correct mechanism.

  Which side drives: input partition if the input is an L1 shard (output then native too iff
  it is the SAME cut), else the output partition, else the solved plan. That ordering is
  what makes `dram_sharded_out` and `cross_spec_height_in_width_out` land on a native input
  with a real remote write, rather than a local shard re-read.

  **Two defects found and fixed while landing it**, both invisible on the interleaved path:
  1. `R`/`C` were derived from the INPUT's padded shape. A ROW_MAJOR tensor's padded shape
     rounds its last dim to its PAGE width, which a width-cutting shard sets — `[3,160,160]`
     sharded 64 wide reports `[3,160,192]`, giving a 6-column plan for a 5-column output.
     The tile grid IS the output; it now comes from `output_tensor.padded_shape`.
  2. `read_sticks_for_tilize` is stick-indexed by construction (`start_page + block_row +
     row`, stride 1), which is only a stick index when a page is a whole row. A shard that
     cuts the width pages by SHARD width, so consecutive sticks are `input_pages_per_row`
     apart. Added a strided reader branch — the helper's own shape (one reserve / read-burst
     / barrier / push per TILE-ROW) with the page stride written out. RECORDED GAP: a
     `page_stride_per_row` parameter alongside `byte_offset_within_page` would close it in
     the helper. The block width is constrained to `gcd(C, page_width_tiles)`'s divisors so a
     block's row segment always sits inside one source page.

  **L1-budget correction folded in** (the ledger finding this refinement was asked to carry):
  `ttnn.get_max_worker_l1_unreserved_size()` reports the worker arena, not what is free in
  it, and an L1-resident operand — every sharded one, by definition — sits in that same
  arena. `derive_plan` now subtracts both operands' per-core resident bytes.

  **`_output_tensor_spec()`**: an ND `MemoryConfig` carries an `NdShardSpec` and NO legacy
  2-D spec, so the entry point's sharded `TensorSpec` overload was a `bad optional access`
  on every ND output. Three constructors, one per placement family.

- **SUPPORTED after Refinement 1** (delta only): `shard_api=["none", "legacy_2d", "nd"]`,
  `out_scheme=["interleaved", HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED, "nd"]`,
  `orientation=["none", ROW_MAJOR, COL_MAJOR]`. `EXCLUSIONS` still empty.

- **Accuracy achieved**: PCC = **1.0**, rtol = **0**, atol = **0** — bit-identity
  (`torch.equal`) on all 14 placements in `test_tilize_sharded.py`: `[1,1,512,64]` HEIGHT,
  `[1,1,64,512]` WIDTH, `[1,1,128,128]` BLOCK at both orientations, `[1,1,128,64]`
  DRAM-sharded out / interleaved↔height / ND same-spec / ND-in-legacy-out / interleaved→ND,
  `[1,1,128,128]` cross-spec, `[1,1,32,4096]` short-wide width-sharded, `[4,64,64]` rank-3,
  `[1,1,256,64]` ND round-robin (2 shards/core). tilize does no arithmetic, so these are
  asserted as equalities, not tolerances.

- **Golden test progress**: `test_golden.py` **32/32 of the reachable bf16 cells pass, 0
  failed, 0 XPASS** — the 11 cells this refinement targets (`sharded_legacy_2d` 7,
  `sharded_nd` 3, `short_wide_width_sharded` 1) all land, on top of the 21 Phase 0 cells,
  with the three loud categories still at 0. `test_golden_main_tests.py` **47 passed** (from
  26 immediately after the SUPPORTED widening, and ~10 at Phase 0); all 112 remaining
  failures are honest `UnsupportedAxisValue` refusals on `pad_mode` / `dtype` / `rank`
  (Refinements 2 and 5) — **zero** non-refusal failures. `test_translated.py` 422 passed.

- **Issues encountered**:
  1. The two defects above (input-padded-shape tile grid; stick-strided reader) — both found
     by `test_golden_main_tests.py::test_tilize_nd_sharded`, which is why that file was run
     rather than trusted.
  2. `test_tilize_lever_block_width.py` monkeypatches `_largest_divisor_at_most`; a first cut
     added a second `_largest_common_divisor_at_most` and silently disarmed the lever. Fixed
     by narrowing what the divisor is taken OF (`gcd(C, page_width_tiles)`) instead of adding
     a search — one knob, one source, lever intact.
  3. `test_translated.py::test_tilize_program_cache_addr_change[sharded_width_l1]` asserts
     the program-cache delta is exactly 1. It passes in isolation and the op is verifiably
     correct (probe_012: 1 entry, bit-exact at four different buffer addresses, the sharded
     CB tracking each new address on the cache hit) — the delta is 0 when an earlier test in
     the same module already built the identical program on the shared module-scoped device.
     A harness ordering artifact, newly *reached* rather than newly broken, and the golden
     suite is not ours to edit.
  4. ND shards whose leading-dim extent exceeds 1 (`[2,64,64]` over `[4,128,128]`) are not
     2-D-expressible blocks, so they cannot drive the grid. They are served correctly through
     the accessor legs instead — all such cases pass.

- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_sharded.py` — 27
  cases. 14 identity cases covering every placement, and 12 that assert **nativeness off the
  `ProgramDescriptor`** rather than off the values: a local shard read back through a
  `TensorAccessor` returns exactly the right bytes, so a green identity test says nothing
  about whether the axis was implemented. Those 12 pin `cb.has_buffer()` and
  `cb.buffer_address() == tensor.buffer_address()` per side, and pin that the native-output
  path emits 2 kernels rather than 3. Plus one assertion that the block extents are read off
  the shard spec. **85/85 passing** across `tests/ttnn/unit_tests/operations/tilize/`.

---

## Refinement 2 — Padding: `pad_mode`, `pad_value`, `alignment`, ranks 0 and 1

- **Date**: 2026-09-09
- **What was done**: the design's deferred `grid2d_padded` regime, as an ADDITIVE step on
  the Phase 0 block rather than a second path.

  **The block schedule is untouched.** The grid, the core assignment, the two streaming CBs,
  `compute_kernel_lib::tilize`'s call and the writer are byte-identical; the fill lives
  entirely inside the reader's `load_block`, behind a compile-time `pad_active` branch, so
  the unpadded path compiles to exactly what it was.

  **Two pad regions, two mechanisms**, because they are genuinely different arithmetic:
  * the **W tail** — bytes `[valid_bytes, block_row_bytes)` of a row that HAS data. At most
    one tile's worth (`C = ceil(W/32)`), filled in place with
    `dataflow_kernel_lib::fill_l1_range<elem_size>`, the alignment-aware helper written for
    exactly "a row whose pad offset is not 4-byte aligned".
  * a **fully padded ROW** — an H tail row, a row of an all-pad tile COLUMN (an explicit
    target past the round), or a row of an all-pad leading slice. Sourced by ONE local
    L1→L1 NoC read out of **`cb_pad_row`**, a new scratch CB holding a single block ROW of
    pre-filled bytes. The DM engine moves it; the RISC never store-loops a row. `cb_pad_row`
    is seeded once per kernel — `TILE_WIDTH` elements by hand (`<= 128` B), then
    `log2(block_width_tiles)` doubling local reads — so no store loop is ever proportional
    to the block width.

  The two phases are ordered read → barrier → fill, so a RISC store into the tail of a row
  cannot race the NoC write into its head.

  **The reader's block read is SEGMENTED per image.** `read_sticks_for_tilize` spans one
  contiguous stick run (`start_page + block_row + row`), which is only a tile-row index when
  `H % tile_h == 0`; with an H tail the source rows restart at every image boundary. The
  padded branch splits the global tile-row index into `(image, row_in_image)` first, keeping
  the helper's own shape (one reserve / read-burst / barrier / push per tile-row).
  RECORDED GAP: a `rows_per_segment` + `pad_value` pair alongside `byte_offset_within_page`
  would close both the segmentation and the fill upstream in the helper.

  **`pad_active` is derived from the GEOMETRY, not the argument** — 1 iff the output's padded
  grid reaches past the input's LOGICAL extent on any axis. That is what keeps an
  already-aligned call on the Phase 0 branch whether or not `pad_value=` was passed, and it
  is asserted directly in the new test file rather than inferred from values.

  **The fill is encoded host-side into the INPUT dtype's bit pattern** (`pad_fill_word`), so
  the output cast happens at pack time exactly as it does for a real element. bfloat16 rounds
  to nearest EVEN (checked against torch's own conversion, since the oracle is
  `F.pad(x.bfloat16(), value=v)`); integers are a two's-complement bit_cast masked to
  `elem_size * 8` bits, written width-generically so Refinement 5's integer dtypes need no
  second implementation.

  **Ranks 0 and 1 needed no special case.** A TILE `TensorSpec`'s default alignment is already
  rank 2, so `[]` → padded `[32,32]` and `[64]` → `[32,64]` fall out; `derive_plan` left-pads
  the input's logical shape to 2 and H=1 / W=1 go through the same tail arithmetic as any
  other short dim.

  **One additive nanobind binding**, `ttnn.TensorSpec.with_padded_shape(logical, padded, ...)`
  (`ttnn/cpp/ttnn-nanobind/tensor.cpp`), for the one thing that is NOT derivable from
  (logical shape, tile): an explicit target beyond the tile round. A spec's default alignment
  caps the padded shape AT the round, so `[1,1,32,50] -> [1,1,32,128]` was inexpressible. It
  takes the `MemoryConfig` whole (covering every placement at once) and degenerates to the
  matching default constructor when the target IS the round — and it is used ONLY where the
  target exceeds the round, so every already-verified call keeps the constructor it was
  verified with.

- **SUPPORTED after Refinement 2** (delta only): `pad_mode=["none", "auto", "explicit"]`,
  `pad_value=["none", "zero", "positive", "negative"]`,
  `alignment=["tile_aligned", "w_non_aligned", "h_non_aligned", "hw_non_aligned"]`,
  `rank=[0, 1, 2, 3, 4, 5, 6]`. `EXCLUSIONS` still empty.

- **Accuracy achieved**: PCC = **1.0**, rtol = **0**, atol = **0** — `torch.equal` on BOTH
  views of every case: the logical readback (`to_torch`, which must still be the input at the
  input's shape) and the PADDED readback (`to_torch_with_padded_shape`, which must equal
  `F.pad(x, value=fill)`). 32 cases in `test_tilize_padded.py`: w/h/hw tails at all three
  pad_value signs, ranks 0/1/2/3/5, five explicit targets (three of them past the tile round),
  four buffer/low_l1 crossings, a height-sharded output, an ND-sharded input on a width-cut
  page (strided read + W tail), an ND-sharded output, and three grid-scale geometries
  (`[1,1,1,2048]`, `[1,1,32,4090]`, `[8,1,249,2048]`). tilize does no arithmetic, so these are
  equalities, not tolerances.

- **Golden test progress**: `test_golden.py` **53 passed, 0 failed, 0 XPASS** (from 32 after
  Refinement 1) — the three loud categories stay at 0. New: `padding_auto` 7,
  `padding_explicit` 4, `padding_crossed` 3, the four padded `work_geometry` members, rank 0
  and rank 1, and both padded LOOSE_CASES (`[1,1,1,50304]` at C=1572, `[8,1,49,2048]`).
  `test_golden_main_tests.py` **127 passed** (from 47), every one of the 32 remaining failures
  an honest `dtype` refusal (float32 15 / int32 7 / uint32 6 / uint16 4 — Refinement 5) plus
  the 2 pre-existing `use_module_device` × `device_params` collection errors.
  `test_translated.py` **707 passed** (from 422).

- **Issues encountered**:
  1. **A sharded `memory_config` may carry no ShardSpec.** `MemoryConfig(WIDTH_SHARDED, L1)`
     with only the family named is a contract case (`test_translated.py`'s own comment: "the
     op derives the output shard spec itself"). `_output_tensor_spec` handed that config's
     `None` spec to a constructor and `TypeError`'d — latent since Refinement 1, newly
     REACHABLE here because those cases all pass `output_padded_shape`. Fixed by applying
     `TensorSpec`'s own `height_sharded` / `width_sharded` / `block_sharded` cut over the
     compute grid, derived off the PADDED spec so the shard's height is the padded height.
     +5 cases.
  2. **A resident input shard has nowhere to put the fill**, so `pad_active` clears
     `input_native` and a padded call reads its input through the accessor. Not a regression
     of Refinement 1's nativeness claim — a shard holds only the caller's own bytes, and
     writing the pad into it would corrupt the input tensor. The OUTPUT side stays native.
  3. **A deliberate divergence from the translated reference suite, left as-is.** 50
     `test_translated.py` cases assert that a non-tile-aligned input with **no** padding
     argument is zero-padded by default. The immutable acceptance test
     `test_tilize.py::test_tilize_rejects_unaligned_input_without_padding` ("DO NOT MODIFY")
     asserts the opposite, and `feature_spec.TARGET`'s own `pad_mode` comment agrees
     ("none — no padding argument; an unaligned input is refused"). The acceptance test and
     the registry declaration win. Those 50 were failing before this refinement as well (on
     the `alignment` gate); they now fail with the intended `ValueError` instead of a support
     refusal, which is what the Phase 0 note said would happen once padding landed.
  4. My first hand-derived bfloat16 round-to-nearest-even expectation was wrong (an exact tie
     with an even mantissa stays put, it does not round up). Replaced the hand-computed
     constant with a comparison against torch's own conversion across seven values, since
     torch's rounding IS the oracle.

- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_padded.py` — 32 cases.
  Beyond values, four of them assert the *mechanism* rather than the result:
  `pad_active` tracks the pad REGION and not the argument (both directions), `cb_pad_row`
  exists iff the fill path does and costs exactly one block row, and `pad_fill_word`'s
  bfloat16 rounding matches torch while a negative integer fill is a non-truncating
  two's-complement bit_cast at the element width. Plus the two malformed-request refusals
  (target smaller than the input; target not a whole number of tiles).
  **117 / 117 passing** across `tests/ttnn/unit_tests/operations/tilize/`.

---

## Refinement 3 — Speed up the perf-flagged attention profile

- **Date**: 2026-09-09
- **What was done**: a **measurement-first** perf pass on the `attention:` LOOSE_CASE
  `[1,1,32,16384]` (bf16 → bf16, interleaved DRAM→DRAM, R=1 × C=512). Three levers
  landed, all kept, plus a calibration that reframes the target.

  **The ablation came first, whole-op.** 8×8 Wormhole, 64/64 cores, fresh cache:

  | variant | device kernel ns | implies |
  |---|---|---|
  | all payloads stubbed (dispatch + CB scaffolding only) | **660** | fixed floor |
  | + compute payload (`tilize`) | **1515** | compute = 855 ns |
  | + read payload only (writes stubbed) | **8241** | reads = 6726 ns (156 GB/s) |
  | + write payload only (reads stubbed) | **9459** | writes = 7944 ns (132 GB/s) |
  | full op | **13247** | 2 MiB @ 162 GB/s |

  Reads and writes are **balanced** (6726 vs 7944) and overlap by only 2938 of the
  14670 ns they would take in series — because a core owns exactly ONE tile-row-tall
  block, so it reads, computes and writes strictly in sequence with no second
  pipeline wave to overlap against.

  **The calibration is what reframes the goal.** A production `ttnn.clone` — a pure
  whole-tile-page DRAM→DRAM copy, the simplest possible move — on the same box and
  the same 64 cores takes **15268 ns for 2 MiB (137 GB/s)** and **87375 ns for
  16 MiB (192 GB/s)**. So 192 GB/s (the `double_buffer` figure the refinement's
  1.19×-headroom estimate was taken against) is this part's *asymptotic* ceiling and
  is only reachable at ≥16 MiB; at 2 MiB the ramp plus the 660 ns dispatch floor cap
  even a plain copy at ~137 GB/s. tilize's 162 GB/s at 2 MiB is therefore **1.15×
  faster than a plain tiled copy of the same size** — the flagged shape is
  data-movement-saturated, not under-tuned.

  **Lever 1 — `PIPELINE_WAVES_PER_CORE` × `MIN_BLOCK_ROW_BYTES` (the one that wins).**
  A WAVE is one tile-row of the block: the reader's push quantum, the compute
  helper's per-call unit, the writer's minimum wait. The tensor holds
  `R * num_w_chunks` tile-rows, so `waves_per_core = R * num_w_chunks / num_cores` —
  which means "fill the grid" and "fill each core's pipe" are the **same expression**
  on the column axis, differing only by a factor. That factor is the knob, and
  `w_chunks_for_occupancy` became `w_chunks_for_waves` rather than gaining a sibling.
  Each extra wave is bought by HALVING the block width, i.e. by halving the read
  transaction, so the value taken is the deepest pipe whose read still clears
  `MIN_BLOCK_ROW_BYTES`. Both terms measured, five geometries, read size → wall ns:

  | shape | w1 | w2 | w4 | w8 |
  |---|---|---|---|---|
  | `[1,1,32,16384]` | 512 B **13759** | 256 B 13620 | 128 B 15204 | 64 B 26947 |
  | `[1,1,32,32768]` | 1024 B 25267 | 512 B **24077** | 256 B 23757 | 128 B 28964 |
  | `[1,1,1024,1024]` | 1024 B **23322** | 512 B 23555 | 256 B 24436 | 128 B 23895 |
  | `[1,1,2048,2048]` | 4096 B 92722 | 2048 B 89365 | 1024 B **86085** | 512 B 85380 |
  | `[1,1,2048,64]` | 128 B **4797** | 64 B 6146 | — | — |

  512 B is the largest floor that is never the wrong call: every geometry that stays
  at or above it improves or lands inside the ±3% band, and every one that would
  have to go below it to buy a wave loses. Shipped as `PIPELINE_WAVES_PER_CORE = 4`
  (cap) × `MIN_BLOCK_ROW_BYTES = 512` (floor, in BYTES so it means the same at every
  element size).

  **Lever 2 — compute per-call overhead**, both of them the tilize helper's own
  documented parameters, neither replacing it with raw LLK.
  `ReconfigureRegisterDatatypeMode::NoReconfigure` is correct here because
  `compute_kernel_hw_startup(cb_in, cb_out)` programs srcA/srcB and the pack format
  once and nothing else runs on the TRISCs — including on the casting diagonal,
  where the cast is carried by the two CBs' formats, not by the per-call reconfig.
  `InitUninitMode::InitOnly / Neither / UninitOnly` pays the tilize LLK init+uninit
  once per core instead of once per block.

  **Lever 3 — one-packet NoC issue path on the writer.** `noc_async_write` defaults
  `max_page_size` to `NOC_MAX_BURST_SIZE + 1` and so takes the generic multi-packet
  `*_any_len` setup; a whole tile page never needs it. `noc_async_write<out_tile_bytes>`
  is the one-token fix.

- **Measured result** (medians of 3 fresh-cache runs, 64/64 cores on every row —
  a number taken on a fraction of the grid would describe the split, not the kernel):

  | shape | before | after | ratio | L1/core |
  |---|---|---|---|---|
  | `[1,1,32,16384]` **the flagged config** | 13247 | 13498 | **1.00** (tie; program unchanged) | 64 KB |
  | `[1,1,2048,2048]` square_large | 94481 | 87373 | **1.08×** | 512 → **128 KB** |
  | `[1,1,32,32768]` short_wide_wide | 25267 | 24472 | **1.03×** | 128 → **64 KB** |
  | `[1,1,1024,1024]` square_mid | 23322 | 22339 | **1.04×** | 128 → **64 KB** |
  | `[1,1,2048,64]` full_width | 4793 | 4743 | 1.01 | 24 KB |
  | `[1,1,32,2048]` width_chunked | 3790 | 3876 | 0.98 (noise) | 20 KB |
  | `[1,1,16384,32]` tall_narrow | 20615 | 20937 | 0.98 (noise) | 20 KB |
  | `[8,1,249,2048]` padded | — | 86901 | — | 516 → **129 KB** |
  | `[1,1,2048,2048]` height-sharded | — | 4791 | — | zero-copy |

  The flagged config is a **measured tie**: at the shipped floor its program is the
  Phase-0 one (bw=8, 64 chunks), and the alternative (bw=4, 2 waves, 256 B reads) was
  measured against it in isolated sessions — w1 13759/13252/13793 vs w2
  13620/13815/13252 — dead even. The wins are on the geometries whose read stays
  large while the pipe deepens, and they come with **4× less L1** on the widest ones.

- **Accuracy achieved**: PCC = **1.0**, rtol = **0**, atol = **0** — `torch.equal`
  bit-identity on every case. tilize does no arithmetic, so any deviation is a bug,
  not a budget; every lever variant in both new harnesses asserts equality, not
  tolerance. 161/161 passing across `tests/ttnn/unit_tests/operations/tilize/`.

- **Golden test progress**: `test_golden.py` **53 passed, 0 failed, 0 XPASS** —
  identical to Refinement 2, with the three loud categories still at 0.
  `test_golden_main_tests.py` **127 passed / 32 failed / 2 errors** — also identical
  (all 32 honest `dtype` refusals for Refinement 5, plus the 2 pre-existing
  `use_module_device` × `device_params` collection errors). No SUPPORTED change: this
  is a perf refinement.

- **Issues encountered**:
  1. **The Goal's 1.19× headroom estimate was against the wrong regime.** It compared
     a 2 MiB transfer to `double_buffer/report.md`'s 190.8 GB/s, which is a 16 MiB-scale
     number. Establishing the same-size `ttnn.clone` baseline (137 GB/s) is what turned
     "1.19× to go" into "already 1.15× past the reference", and it is the single most
     load-bearing measurement in this refinement.
  2. **The init/uninit amortization's DEAD instantiations cost 4% of the wall.**
     Emitting the InitOnly/Neither/UninitOnly variants on a kernel where every core
     owns one block grows the TRISC binary the dispatcher ships; `[1,1,32,2048]`
     (a ~3.8 µs kernel) went 3790 → ~3940 ns consistently across four runs. Fixed by
     gating the emission on `max blocks per core > 1`, which put it back to
     3790/3815/3835. The lever is unchanged where it can actually fire.
  3. **The first floor value (256 B) regressed `[1,1,1024,1024]` by ~5%.** Caught only
     because the guard set was widened to the two geometries whose 1-wave read is
     1024 B — the sizes *between* the flagged shape's 512 B and square_large's 4096 B.
     A floor calibrated on two shapes would have shipped that regression.
  4. **Tried and rejected, each measured**: staggering the per-core read row order to
     de-conflict DRAM banks (13708 vs 12587 control — 32 outstanding reads already
     spread across banks); the one-packet path on the reader (flat, and it would have
     cost the `read_sticks_for_tilize` helper call, so reverted); `WRITE_BATCH_MIN_TILES`
     ∈ {4,8,16} at every wave setting, i.e. the writer twin of the wave lever
     (13580–14001, flat — at bw ≥ 4 the writes are already past the in-flight knee);
     CB depths {2,4} × {2,4} (13711–14104, flat).

- **Recorded gap**: `read_sticks_for_tilize` has no `max_page_size` hook, so the
  reader's stick reads still take the generic `*_any_len` issue path. Measured flat on
  this op (it is bandwidth-bound, not issue-bound), so it was not worth bypassing the
  helper for — but a defaulted `max_page_size` template parameter would close it
  upstream for readers that ARE issue-bound.

- **What is left, and why I did not take it** (a finding, not a queued task): the only
  remaining way to enlarge the flagged shape's 512 B read at 64/64 cores is to have
  several cores share one wider DRAM read and redistribute it core-to-core — a
  multicast/peer-unicast topology change that adds ~0.75 MiB of cross-core NoC to chase
  at most the ~1.13× gap to a ceiling this op is already above. The genuinely
  off-ceiling geometry is the transposed pair `[1,1,16384,32]` at ~100 GB/s, whose 64 B
  read is forced by `C = 1`; that is Refinement 6's subject and it needs the two-disjoint-
  core-range restructure already recorded in `l1_ledger.md`, not another knob.

- **Tests added**:
  `tests/ttnn/unit_tests/operations/tilize/test_tilize_perf_attention.py` — 9 cases:
  the flagged config measured exactly (pinned at R=1 × C=512 and 64/64 cores, so a
  later plan change that drops occupancy fails here rather than hiding in the ns), plus
  one representative per distinct kernel path × placement (full-width, width-chunked,
  square_large, tall_narrow, square_mid, short_wide_wide, padded, height-sharded). Every
  case asserts `torch.equal`.
  `tests/ttnn/unit_tests/operations/tilize/test_tilize_lever_pipeline_waves.py` — 27
  cases: the raw wave sweep with the floor disabled (the measurement), the writer-twin
  grid `waves × WRITE_BATCH_MIN_TILES` (because reader and writer are one pipeline and a
  wave doubling halves the writes in flight unless the batch moves with it), and
  `test_production_wave_choice`, which pins the `(block_width_tiles, num_w_chunks)` the
  shipped rule picks on six geometries — that last one is what stops a later plan edit
  from silently reverting the tuning.
  **161 / 161 passing** across `tests/ttnn/unit_tests/operations/tilize/`.
