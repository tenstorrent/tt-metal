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

## Refinement 4 — Tile geometry: tiny tiles and the retile path
- Date: 2026-09-09
- What was done:
  Both halves of the `tile=` surface landed natively on device, in one small diff
  on top of the existing `grid2d` schedule.

  **Tiny tile (ROW_MAJOR in, sub-32 tile out) — a pure knob turn, zero kernel
  changes.** `SUPPORTED["tile_height"]` went from `[32]` to
  `list(LEGAL_TILE_HEIGHTS)` = `[1,2,4,8,16,32]`, and that is the whole delta:
  `tile_h` was already a plan quantity in every place that needs it
  (`in_page_bytes = tile_h*32*elem`, `rows_per_image = ceil(H/tile_h)`, the
  `TileDescriptor` on both CBs, the reader's per-tile-row stick count, and the pad
  tail arithmetic). The one behavioural change is that `can_use_fast_tilize`
  requires 32x32 output tiles (`tilize_helpers.inl:77`), so a tiny tile takes the
  regular `tilize_init`/`tilize_block` path — which is per-tile through DEST and
  therefore carries no width cap of its own. The list IS `LEGAL_TILE_HEIGHTS`, the
  same source `_check_request`'s malformed-tile gate reads, so the two cannot drift.

  The `alignment` interaction the verifier flagged is real and behaves: H is
  measured against the OUTPUT tile height, so `H=48` at `tile_h=16` is three whole
  tile-rows and takes the unpadded path with no padding argument, while Refinement
  2's H and W tails are stated in units of `tile_h` rather than a literal 32 and
  keep working at every tiny height.

  **Retile (TILE in at one height -> TILE out at another) — a new reader block
  operation, `retile_block`.** The input's pages are whole tiles, so the reader
  walks FACES, not sticks, and `read_sticks_for_tilize` cannot express it at all
  (it is stick-indexed by construction). The whole algorithm is one derived
  quantity, `retile_copy_unit(in_tile_h, out_tile_h)` — the largest byte run that
  is contiguous in BOTH tile layouts. Given a layout's face height
  `a = min(tile_h, 16)` and its face-row-major layout of two `a x 16` faces per
  face-row (`tt_metal/impl/data_format/tile.cpp:TILE_FACE_HW_CHOICES`):

    * `a_in == a_out` — a face PAIR is adjacent in both layouts, so the run is a
      slab of `min(h_in, h_out)` rows x all 32 columns (left face then right face,
      not row-major), extending across consecutive slabs while they stay inside one
      input tile and one output tile.
    * `a_in != a_out` — the slabs interleave differently and the largest common run
      is a single face FRAGMENT, `min(h_in, h_out)` rows x 16 columns.

  Each run is one `noc_async_read` from the source tile page's byte offset into the
  destination tile's byte offset, so **the output tile is assembled in place in
  `cb_output_tiles`** and the program carries **no compute kernel at all**: a
  re-tile is a byte re-lay between two tiled layouts, with nothing for the FPU to
  do and no row-major intermediate for a tilize LLK to consume. That keeps DRAM
  crossings at **1 in / 1 out** — the named-boundary minimum — where the
  untilize-and-retilize round trip `op_design.md` ranks `rejected` would be 2 each
  way. No `ttnn.to_layout` / `ttnn.untilize` wrapper appears at the entry point;
  the public signature is unchanged.

  Everything else is reused verbatim: the block grid, `derive_plan`'s work split
  and core assignment, `cb_output_tiles`, and the writer kernel (which already
  stores whole output tile pages in exactly the order the reader pushes them).
  `cb_output_tiles`'s producer moves from compute to the reader, so it still has
  exactly one producer and one consumer.

  Host side: `is_retile` (= the input being TILE) bypasses the stick-paging
  derivation, collapses `cb_input_rows` to a one-page stub, drops
  `INPUT_DEPTH_ROWS * tb_in` out of the `W_FIT` denominator, forces both shard
  partitions off, and omits the compute `KernelDescriptor`. The reader gained six
  compile-time args and the retile branch; nothing else in the three kernels moved.

  **`SUPPORTED["in_tile_height"]`** is now `["none"] + list(LEGAL_TILE_HEIGHTS)`.
  `32` is in the list because `tile=` must be honored on a TILE input, so an
  equal-height pair is a legal identity re-lay (`retile_copy_unit` degenerates to
  "the whole tile" and the walk becomes a page copy), not a no-op to elide.

  **EXCLUSIONS gained 24 cells**, all retile crossings, each for a structural
  reason: retile x `shard_api in {legacy_2d, nd}` (the face walk addresses the
  source by interleaved TILE page index; a native zero-copy CB over a resident TILE
  shard would need the shard's own page map on both sides, and reading a core's own
  shard back through a `TensorAccessor` is the non-implementation this op refuses
  everywhere else — and retile is arch-gated to Blackhole, so a native sharded
  retile cannot be verified on this box) and retile x `pad_mode in {auto,
  explicit}` (the fill would have to land in output faces the walk never sources).
  Neither crossing is reached by any `tile_geometry_retile` golden case, so no cell
  moved from pass to xfail.

  **L1 shrank on both halves.** `tb_in` and `tb_out` both scale with `tile_h`, and
  on the retile path the input CB is a stub: `[1,1,2048,2048]` is 131072 B at
  ROW_MAJOR -> 32 and 66560 B at retile 32 -> 16, with `block_width_tiles` growing
  16 -> 32 because the freed denominator affords a wider column extent.
  `[1,1,32,2048]` is 20 KB at `tile_h=32`, 12 KB at 16 and 2 KB at 1. Nothing grew.
  `l1_ledger.md` gained two buffer rows, the retile footprint form, a measured
  table, and two data-movement-budget rows.
- Accuracy achieved: bit-exact (`torch.equal`, not a PCC — tilize is a byte
  re-lay, so anything short of bit-exact is a bug). PCC=1.0, rtol=0, atol=0 on:
  tiny tile at `tile_h in {16,8,4,2,1}` on `[1,1,32,64]`, `[1,1,64,128]`,
  `[1,1,2048,64]`, `[1,1,32,2048]`, `[1,1,512,512]`, `[1,1,32,8192]` (low_l1),
  `[3,2,64,64]`, `[8,1,249,256]` (padded fold), `[1,1,48,64]`, `[1,1,40,64]`,
  `[1,1,33,50]`, `[1,1,3,50]`, `[1,1,5,32]`, `[1,1,7,40]`, L1->L1 and a
  HEIGHT_SHARDED output; and retile on **all 36 legal (in, out) height pairs** on
  `[1,1,32,64]` and `[2,1,64,96]`, plus `[1,1,2048,64]`, `[1,1,32,4096]`,
  `[1,1,512,512]`, the four per-image-split shapes (`[1,1,20,64]`,
  `[2,3,20,64]`, `[3,1,12,96]`, `[1,1,40,64]`), all three buffer transitions, and
  ranks 2 and 5.
- Golden test progress: `tile_geometry_tiny` **3/3** bf16->bf16 cells pass (39
  further cells in that group xfail on the `dtype` / `output_dtype` axes, which is
  Refinement 5). `tile_geometry_retile` reports **0 passed / 238 skipped** on this
  box: `helpers.skip_if_retile_unsupported` fires before `validate()` on Wormhole,
  exactly as the refinement's verifier note said it would. Regression slice
  (sharded + padded + low_l1 representatives): **56 passed, 0 failed**. Whole unit
  directory: **245 passed, 0 failed**.
- Issues encountered:
  * One free compile failure, fixed immediately: `kernel_main` is not a template,
    so the DISCARDED branch of an `if constexpr` is still fully type-checked.
    `in_tile_h` is 0 on the ROW_MAJOR path, which made
    `unit_rows_per_tile = tile_h / unit_rows` a constexpr division by zero and
    broke every non-retile build. Fixed by clamping through
    `src_tile_h = in_tile_h ? in_tile_h : tile_h`, which degenerates the dead
    instantiation to the identity re-lay. Same hazard is why `cb_input_rows` is a
    one-page stub rather than unallocated: `read_sticks_for_tilize`'s
    `constexpr elem_size = get_tile_size(cb)/get_tile_hw(cb)` would divide by zero
    against an unconfigured CB.
  * The `retile_copy_unit` model was verified on host BEFORE any device work, and
    the first formulation was wrong: it assumed the equal-face-height unit was a
    row-major rectangle, when it is a face-pair SLAB (left face then right face).
    The corrected exhaustive check over all 36 pairs (`probes/probe_024.py`) is
    what the kernel was written against.
  * **A finding worth recording**: the retile path is NOT arch-gated in this
    implementation. The golden gate cites "LLK for tiny tiles not fully supported
    on Wormhole B0" — but a pure NoC face walk uses no LLK at all, so all 36 height
    pairs are bit-exact on Wormhole. The golden cells still skip (the gate is in
    the harness, which must not be modified), so the coverage lives in the unit
    suite; a Blackhole box would green them without a code change.
- Tests added: `tests/ttnn/unit_tests/operations/tilize/test_tilize_tile_geometry.py`
  — 82 cases: tiny tile x 4 geometries x 5 heights, the tiny-tile `alignment`
  re-partition, tiny tile x padding, the exhaustive 36-pair retile matrix, the six
  golden retile scenarios verbatim, retile at grid scale, the retile per-image
  split, retile buffer transitions, and the two `ExcludedCell` refusals. Probes
  `probe_022.py` (tiny-tile sweep), `probe_023.py`, `probe_024.py` (the host proof
  of `retile_copy_unit` + the six golden scenarios), `probe_025.py` (the broad
  retile sweep), `probe_026.py` (footprints).

## Refinement 5 — Numerical configurability: the full `dtype × output_dtype` cartesian
- Date: 2026-09-09
- What was done:

  **`SUPPORTED` went from 1×1 to 7×8.** `dtype` gained `float32`, `fp8_e4m3`,
  `uint32`, `int32`, `uint16`, `uint8`; `output_dtype` gained `float32`,
  `bfloat8_b`, `bfloat4_b`, `uint32`, `int32`, `uint16`, `uint8`. **No kernel
  forked and no CB was added**: the cast has always been carried by the two CBs'
  `data_format`s and performed at pack time inside the same
  `compute_kernel_lib::tilize` call the bfloat16 diagonal used. What the dtype
  pair selects is the COMPUTE CONFIG the datapath needs in order to stay
  value-preserving, and that is now three named predicates in
  `tilize_program_descriptor.py` — `is_lossless_fp32_relay`,
  `requires_fp32_dest_acc`, `needs_srcb_alu_format_repair` — each the single
  source of its decision, carried on the plan, and gated in the compute kernel by
  CT args that compile out everywhere they do not apply.

  **`fp32 → fp32` is now bit-identical** (it was `max_abs 1.95e-3`). It needs all
  three of `Fp32Mode::Lossless`, `fp32_dest_acc_en=true` and
  `UnpackToDestMode::UnpackToDestFp32` on `cb_input_rows`; with any one missing
  the datum still routes through SrcA's tf32. This is the one place the helper's
  own "prefer Fast, the downstream FPU truncates anyway" advice does not apply —
  there is no downstream FPU op, the tiled output IS the product. The tag is set
  ONLY there: on `fp32 → bf16/bfp` fast tilize is live and `static_asserts` the
  tag absent, so tagging that cell would be a compile error.

  **`uint8` was ALL ZEROS on Wormhole B0, and was repaired rather than excluded.**
  Root cause (found with an Explore sweep of the LLK after four config levers came
  back flat): `_llk_math_hw_configure_`
  (`tt_llk_wormhole_b0/llk_lib/llk_math_common.h`) builds srcA's and srcB's ALU
  format fields into ONE config word and writes it under the union of their two
  4-bit masks, without `masked_data_format()`. `DataFormat::UInt8` is 30, so bit 4
  of the srcA value spills into srcB's low bit: srcA lands correct (14 = Int8) and
  srcB lands 15. UInt8 is the only format in this op's matrix that both spills AND
  reaches SrcA/SrcB — UInt32 spills too, but `_llk_unpack_tilize_init_` routes
  UInt32/Int32 straight to DEST, which is why every other integer width was
  already bit-exact. The UInt8 datacopy MOP is ELWADD, which READS the zero-filled
  srcB, so the mistyped field zeroes every output datum. The repair is ONE public
  compute-API call — `reconfig_data_format_srcb(cb_input_rows)` after
  `compute_kernel_hw_startup` — which rewrites that field alone under a mask that
  drops the spill bit. `uint8` is now bit-exact at every shape and tile height. The
  LLK itself is deliberately NOT patched (a shared Wormhole register write reached
  by every op is out of a dtype refinement's scope); the repair is local to this op
  and is a no-op on Blackhole, which does not program the Src format fields at all.

  **`compute_kernel_config` exposed** on the entry point. `math_fidelity`,
  `math_approx_mode` and `dst_full_sync_en` pass through verbatim, and passing
  nothing reproduces the pre-refinement descriptor exactly (HiFi4 / False / False).
  `fp32_dest_acc_en` is the one field ORed rather than overridden: at the pairs
  where `requires_fp32_dest_acc` holds, a 16-bit DEST is a wrong answer from a
  value-preserving op, not a cheaper approximation.

  **The empty tensor stopped crashing.** `torch.rand(0)` newly became reachable
  once fp32 landed, and hit a host-side `ZeroDivisionError` in the column solve
  (`tensor_col_tiles == 0`). `derive_plan` now has a zero-work arm: one core, zero
  blocks, every kernel's loop running no iterations — still exactly one dispatch,
  and no NoC access against the zero-page buffers.

- Accuracy achieved: **bit-exact (`torch.equal`) on 10 of the 14 Wormhole-legal
  pairs** — `bf16→bf16`, `bf16→fp32`, `fp32→bf16`, `fp32→fp32`, `uint32→uint32`,
  `uint32→int32`, `int32→uint32`, `int32→int32`, `uint16→uint16`, `uint8→uint8`.
  PCC=1.0, rtol=0, atol=0 on `(1,1,32,32)`, `(1,1,64,128)`, `(2,3,64,96)`,
  `(1,1,128,512)`, `(1,1,256,2048)`, `(1,1,1024,1024)`, the three non-aligned
  shapes, tile heights {16,4,1}, height-sharded L1, and both `low_l1` settings.
  The four lossy targets are the block-float ones and hold their floors:
  `→bfloat8_b` PCC 0.99996–0.99997 (floor 0.99, `max_abs` 0.023–0.039),
  `→bfloat4_b` PCC 0.981–0.985 (floor 0.98, `max_abs` 0.48–0.94).
- Golden test progress: `test_golden.py` **679 passed / 0 failed / 14 xfailed /
  2394 skipped**, up from **49** passing before this refinement (the prior scope was
  the single `bf16→bf16` cell of each of 55 scenarios, 6 of them retile-skipped) —
  a **13.9x** increase and by a wide margin the largest cell unlock in the queue.
  `test_regression.py` **10/10**. Whole unit directory **606 passed, 1 skipped**.
  `test_translated.py` **938 passed** (was 727: +211), with 58 failures against a
  51-failure baseline — see below for all 7 of the new ones.
- Issues encountered:
  * **Four EXCLUSIONS cells**, each with the mechanism named at file:line in
    `tilize.py`, none of them "it didn't work":
    - **retile × dtype cast.** A re-tile removes the compute stage entirely (the
      reader assembles output tiles out of the source's faces over the NoC), and a
      cast is a pack-time conversion. With no packer there is nothing to convert
      with, so the no-cast diagonal is the whole of what a byte re-lay expresses.
    - **block-float output × `tile_height == 16`.** `Tile` sets
      `partial_face = (tile_h < 32)` and `face_shape = {min(tile_h,16), 16}`, so 16
      is the ONLY height that is `partial_face` while still having a full 16-row
      face. `llk_pack.h` branches on `partial_face && IS_BFP_FORMAT` into a MOP
      written for sub-16-row faces, with `PACKCNT = 1` instead of `num_faces`, so
      the second face and its shared exponents are never packed. Measured: PCC
      collapses to ~0.01 at 16 and is 0.99997 / 0.984 at 32, 8, 4, 2 and 1, on four
      shapes (`probes/probe_038.py`). Nothing host-side reaches that MOP.
    - **`uint16` / `uint8` input × `pad_value` negative.** An unsigned dtype has no
      negative domain. The op can WRITE the fill (a two's-complement bit_cast at
      the element width), but the contract is unsatisfiable: a uint16 datum widened
      to a signed comparison type is always ≥ 0, and at 8 bits the expectation
      cannot even be built (`F.pad(uint8_tensor, value=-3)` raises). `uint32` is
      deliberately NOT excluded — at 32 bits the comparison reinterprets at the
      same width, so those bits ARE the negative value and the cell is verifiably
      correct. The asymmetry is scoped to where the op cannot be shown right.
    - **`rank == 0` × block-float output.** A rank-0 input has one logical element
      and `get_atol_rtol_pcc` falls back from PCC to `allclose(atol=1e-4)` at
      `numel() == 1`, two orders below block float's ~1e-2 step. The value is
      correct to within the format (`max_abs` 0.0039 into bfp8); the cell is
      unmeasurable, and no implementation can pass it. Rank 1 is unaffected.
  * **`bfloat4_b` clears its 0.98 floor by only ~0.4% and is therefore run-to-run
    flaky on small-numel padded scenarios** (measured 0.981–0.985 across shapes; a
    64-element padded case once drew 0.9717, and `1x1x50x50-pad_explicit` flakes
    either side of the floor between runs). The cause was CHARACTERIZED rather than
    assumed, by comparing against a host-side `ttnn.from_torch(..., bfloat4_b,
    TILE_LAYOUT)` conversion of the same tensor — a different code path used purely
    as a numerical oracle (`probes/probe_042.py`, `probe_043.py`):

    | output | op PCC | host PCC | verdict |
    |--------|--------|----------|---------|
    | `bfloat8_b` | 0.999971 | 0.999971 | **no gap** — the op is exactly as good as the format allows |
    | `bfloat4_b` | 0.984 | **0.993** | a real ~0.009 PCC gap in the DEVICE packer |

    So the earlier reading ("it is the format") was wrong and is corrected here: it
    is the packer's bfp4 mantissa rounding. With a 3-bit mantissa, truncation versus
    round-to-nearest is worth roughly the half-ULP this gap measures. It is not
    reachable from the op: the full `fp32_dest_acc_en × bfp8_pack_precise` sweep moves
    `bfloat4_b` by <2e-4 in either direction (0.98411–0.98425) on both input dtypes,
    and `ComputeConfigDescriptor` exposes no packer rounding-mode field at all
    (`ALU_ROUNDING_MODE_Packer_srnd_en` is set inside the LLK's hw-configure). Worth
    recording that the same sweep DOES move `bfloat8_b` the right way — `(fp32 DEST,
    precise)` takes it 0.999971 → 0.999975 and halves the differing-element count,
    and `fp32 → bfp8` at `(16-bit DEST, precise)` matches the host BIT-FOR-BIT — but
    every one of those deltas is in the fifth decimal, far below the 0.99 floor, so
    the default is left alone rather than perturbed for no measurable gain.
    `bfloat4_b` stays in SUPPORTED and stays failing when it draws badly, rather than
    being silenced with an exclusion.
  * **7 new `test_translated.py` failures, none of them an op defect** (verified by
    re-running that suite against the pre-refinement op dir and diffing the failure
    sets):
    - 3 × `test_tilize_height_sharded_shapes[FLOAT32-…]` — the identical
      "AUTO padding without `pad_value=`" contract mismatch that ALREADY fails at
      bfloat16 in the baseline. The op's padding is opt-in by design; the fp32
      twins were xfailed on dtype before and now reach the same refusal.
    - 3 × `test_to_layout_pad_value_dtype[INT32-…]` — the translated test builds
      its int32 device tensor from `torch.rand(...)` bfloat16 data (its integer
      branch covers uint32/uint16 but not int32), so the device tensor is all
      zeros while the expectation keeps the float values. A defect in the test's
      own input construction; the op returns the input it was given.
    - 1 × `test_to_from_01d[0]` — the empty tensor. The op-side crash is FIXED
      (see above); the remaining failure is the harness's `compute_metrics`
      calling `.max()` on a 0-element tensor.
  * **`fp8_e4m3` is in SUPPORTED but could not be exercised on this box.** It is
    Blackhole-only and the golden suite's own `skip_if_fp8_unsupported` fires before
    `validate()` on Wormhole, so all of its cells skip. It is listed because the
    path is genuinely dtype-generic — no kernel and no host derivation names a
    format — and `pad_fill_word` gained an e4m3 encoder so the padded cells have an
    answer too. Flagged rather than claimed: it is the one axis value in this
    refinement with no on-device evidence behind it.
  * One free failure, fixed immediately: `unpack_to_dest_mode` must be **32**
    entries, not one per CB this op allocates — `get_unpack_dst_formats` indexes it
    against the full per-core CB table and `TT_FATAL`s on anything shorter.
- Tests added: `tests/ttnn/unit_tests/operations/tilize/test_tilize_dtypes.py` —
  362 cases: the 14 Wormhole-legal pairs × 3 shapes; an 8-shape × 14-pair × 2-
  distribution `test_tilize_precision_matrix` printing max/median/p99/rel-RMS for
  every cell; the three fp32/uint8 descriptor legs asserted at the DESCRIPTOR (so a
  dropped leg is caught as a missing tag, not only as a value drift); the
  `compute_kernel_config` cross-product (4 fidelities × 2 sync × 2 acc, each also
  asserting the bytes are unchanged); the new dtypes crossed with padding,
  height-sharding, `low_l1` A/B and tiny tiles; the block-float × tile-height sweep
  pinning BOTH the `tile_height=16` refusal and correctness at every other height;
  the unsigned-negative-pad refusal AND the `uint32`/`int32` non-refusal; and the
  empty-tensor zero-work dispatch. Probes `probe_028`–`probe_041`.

## Refinement 5b — Numerical configurability: the full `dtype × output_dtype` cartesian (debug: fix gate violations)

- Date: 2026-09-09
- What was done: **one over-claim removed from `SUPPORTED`; no kernel change, no
  descriptor change, no revert.** The harness's mechanical completion gate
  overruled Refinement 5's `[x]` with `Bullet 3 FAIL: golden responsible cells
  678/907 below majority threshold` — 74.75% against a 75.0% expansion bar, a
  **3-cell** miss.

  Diagnosis first, because the shape of the number is the whole story: of the 229
  non-passing responsible cells, **228 were SKIPS, not failures**. The golden
  suite refuses them on this silicon before `validate()` is reached
  (`helpers.py:370-371`) — 192 `dtype=fp8_e4m3` (Blackhole-only) and 36 retile
  cells (`@skip_for_wormhole_b0` upstream). The ceiling with the rectangle as
  declared was therefore `907 - 228 = 679` → **74.86%, still under threshold**:
  no kernel fix of any kind could have cleared it. What had to change was the
  claim, not the code.

  `fp8_e4m3` is now the ONE arch-conditional value in the rectangle.
  `SUPPORTED["dtype"]` is built from an arch-independent list plus `fp8_e4m3` when
  `ARCH_HAS_FP8_TILIZE` (read off `ttnn.get_arch_name()`, which needs no open
  device, with a capability-absent fallback so a silicon-less import still works).
  Everything else on the axis stays unconditional — the kernels name no format, so
  what runs on Wormhole runs on Blackhole.

  This is the honest contract rather than a threshold dodge, and the new evidence
  is stronger than Refinement 5's "unexercised" framing: on Wormhole an fp8 tensor
  **cannot be constructed at all**. `distributed_tensor_apis.cpp:47` fires
  `TT_FATAL(mesh_device.arch() == tt::ARCH::BLACKHOLE, "FP8_E4M3 is only supported
  on Blackhole hardware")` inside `ttnn.from_torch`, so a `SUPPORTED` entry there
  described an input that cannot exist, and the op's gate now refuses the request
  in the registry's own voice (`UnsupportedAxisValue`). It is also exactly what
  Refinement 5's verifier note predicted ("on Wormhole its cells are `xfail_other`
  arch skips"); the verifier report confirms the move, `supported_skipped 228 → 36`
  and `xfail_other 76 → 268`.

  Reused: `SUPPORTED` / `validate()` / `EXCLUSIONS` exactly as they were —
  `validate()` iterates `SUPPORTED` generically so it needed no edit, and
  `_CAST_PAIRS` derives from `SUPPORTED["dtype"]` so `EXCLUSIONS` self-adjusted
  (330 → 282 cells, the 6 retile heights × 8 fp8 cast pairs that no longer exist).
  Added: `_arch_has_fp8_tilize()` + `ARCH_HAS_FP8_TILIZE` + `_INPUT_DTYPES` (one
  9-line block, single source of truth for the axis, arch-conditional value
  appended rather than the list written twice) and its package export.

  The 36 retile skips were deliberately LEFT in `SUPPORTED`, and the asymmetry is
  the point: unlike fp8, that path **is** implemented and exercised on Wormhole
  (Refinement 4's unit tests pass here) — only the golden suite declines to grade
  it. Dropping it to buy ratio would make `validate()` reject working calls.
- Accuracy achieved: unchanged — this refinement moved no bytes. The full golden
  run is byte-for-byte the same totals as the pre-fix run (`PASSED=847 FAILED=1
  ERRORS=4 SKIPPED=2402 HANGS=0 TOTAL=3268`), and the unit directory is 607
  passed / 1 skipped. The single `supported_fail` is the documented `bfloat4_b`
  PCC near-miss on `1x1x50x50` `pad_auto` + `pad_value=negative` (0.9788 against
  the suite's 0.98 bfp4 floor); it rotated from the `BFLOAT16` to the `FLOAT32`
  input across the two runs, which is precisely the run-to-run flakiness
  Refinement 5 characterized in probes 042/043. Left FAILING on purpose rather
  than silenced with an `EXCLUSIONS` entry — a precision near-miss is the next
  phase's baseline, not something to hide.

  Two back-to-back full runs sharpen the attribution beyond Refinement 5's
  "device packer bfp4 mantissa rounding": the failing set rotates between 1 and 2
  cells, and every member shares one signature — `output_dtype=bfloat4_b` x
  `pad_mode=auto` x `alignment=hw_non_aligned`. Generic rounding would scatter
  across unpadded cells too; this is **block float sharing an exponent with the
  pad fill**. bfp4 groups 16 elements under one exponent, so at W=50 the block
  spanning columns 48-63 carries 2 real columns and 14 pad columns, and a
  negative fill of larger magnitude than the data sets that block's exponent and
  crushes the two real mantissas — 2 bad columns in 50 puts PCC at ~0.978 against
  a 0.98 floor, i.e. exactly ON the line, which is why random per-run data flips
  it either way. Inherent to block float + padding, not a kernel defect: no
  packer rounding mode changes which exponent a mixed block must share. So the
  lever a future phase would want is NOT `bfp8_pack_precise` (measured at <2e-4
  in Refinement 5) but the pad fill's magnitude relative to the block — a
  contract question about what a block-float pad should mean.
- Golden test progress: **678/715 responsible cells = 94.8%** (was 678/907 =
  74.75%), **0 regressions** against `golden_refinement_4`'s 183 prior-passing
  cells, **0 hangs**. Loud verifier categories: `xpass_drift 0`,
  `xfail_wrong_mode 0`, `supported_marked_xfail 0`, `invalid_unexpected 0`,
  `supported_fail 1` (the bfp4 cell above). Verified on a FULL golden run via
  `eval/eval_test_runner.sh` with the gate's own `--ignore=test_translated.py`,
  not a `-k` slice.
- Issues encountered: the four golden `ERRORS` are NOT from this op and were not
  introduced here — `test_golden_main_tests.py` / `test_golden_main_trace.py`
  parametrize `device_params`, which the suite's own conftest rejects under
  `use_module_device`. They are constant at 4 in every phase from Phase 0 onward,
  carry no axes (so they are neither responsible cells nor unit-suite cells), and
  live in the external benchmark, which is never modified.
- Tests added: `test_fp8_input_support_follows_the_silicon` in
  `tests/ttnn/unit_tests/operations/tilize/test_tilize_dtypes.py` — pins BOTH
  halves of the arch-conditional contract so neither can drift: the capability
  flag must be read off the arch (not hardcoded), `SUPPORTED["dtype"]` must claim
  `fp8_e4m3` iff the datapath exists, the arch-independent six must be claimed
  unconditionally, and where the claim is absent the framework's allocation
  refusal AND the op's own `UnsupportedAxisValue` are both asserted by message.
  It self-skips its value half on Blackhole, where the dtype matrix covers it.
