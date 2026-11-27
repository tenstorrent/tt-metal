# Verification Report: g6_permute

Phase 0 regime `rows_tile_scatter` (bfloat16 TILE, DRAM interleaved, rank 4, dims (0, 2, 1, 3)).
Board: Blackhole p150, 11 × 10 worker grid (110 cores), 8 DRAM banks, 110 L1 banks.

## Code Review

Every item below was fixed in this pass.

| # | Finding | Fix |
|---|---------|-----|
| 1 | The plain build compiled each kernel's block loop twice: a zoned copy for the first 16 blocks and a tail copy. `DeviceZoneScopedN` compiles to nothing without `PROFILE_KERNEL`, so the two copies were identical code. That doubled kernel text against BB-18 (kernel text must be small) and BB-21 (zones must not change the plain build). | `ZONED_BLOCKS` is 16 only under `PROFILE_KERNEL` and 0 otherwise, so the zoned loop is dead code in the plain build. On `[1,64,100,256]` the reader `.text` went 1264 → 840 B and the writer 1952 → 1304 B (about −33 % each). In the zone build all nine zones still fire on all 110 cores. |
| 2 | `test_g6_permute_debug.py` targeted the superseded `rows_tile` regime (`describe_plan`, `_PLANS`, `BLOCK_WT`, none of which exists in the package). Its `ImportError` broke collection of the whole test directory. | Deleted. `test_g6_permute_scatter_debug.py` covers the current regime's knobs: multi-block, depth 1/2, ragged last block, no stagger, cache-hit address patch. |
| 3 | `SUPPORTED["alignment"]` claimed only `h_non_aligned`. `rows_tile_scatter` keeps W, so a half-row carries its padding columns into the output's padding columns, and a tile-aligned B only makes every tile fully live (`op_design.md` regime row 1 states that domain). SUPPORTED under-claimed. | Added `tile_aligned` and `w_non_aligned` after 6 shapes passed bit-exact with the input padding poisoned with `inf` (`test_g6_permute_alignment.py`): `[1,32,64,64]`, `[2,64,32,96]`, `[1,6,16,40]`, `[1,8,6,8]`, `[2,34,64,100]`, `[1,16,1,2]`. |
| 4 | The writer issues each row as `noc_async_write_one_packet_set_state` followed by two `_with_state` writes. That change sat uncommitted in the working tree and was unverified. | Checked against `dataflow_api.h`: both calls default to non-posted, and `_with_state` updates the write counters (`update_counter = true`), so `noc_async_writes_flushed` and the final `noc_async_write_barrier` cover them. `set_state` runs per row because consecutive rows target different pages and banks. The second half's `+ face_bytes` stays inside one page, so it never carries into the address mid-bits. This is the design's *scatter issue rate* alternative. Kept. |
| 5 | Dead code: `RowsLive::of` was never called, and the reader bound CT arg 3 (`tensor_at`) without using it. | Removed both. The shared CT-arg layout is unchanged, and a comment in the reader names the writer-only args. |
| 6 | The precision baseline computed max abs error by hand. | Max abs error now comes from `comp_allclose` (exact tolerance, asserted); PCC from `comp_pcc`. Mean abs and relative RMS stay computed, because `comp_allclose` does not report them. |

Nothing was deferred for architectural reasons.

### Prompt rules (`eval/prompts/g6_permute.txt` `## Rules`)

Every MUST rule whose condition applies to the current SUPPORTED scope holds:

- **bf16 bit-identity:** no unpacker, DEST or packer sits in the path. Special values and denormals pass (acceptance test, loose L12).
- **Padding:** `h_non_aligned` and the newly claimed `w_non_aligned` padding never reach a logical position. Poison tests use `inf`, `-inf`, `NaN` and `1000`.
- **One pass, one program:** `device_num_programs = 1` on every golden cell. There is no intermediate tensor and no host-side tensor transform.
- **Cores:** the core count comes from `split_work_to_cores` over the full grid, not from the shape.
- **Production cases:** none is excluded or gated.
- **Every accepted shape:** A > 32, B > 32, N > 1, `tensor_wt = 3` and the alignment cases all pass bit-exact.
- **BB-03:** writes are non-posted, flushed per block, with a write barrier at kernel end.
- **BB-04 / BB-18:** kernel text is 0.6–0.84 KB (reader) and 1.0–1.3 KB (writer) in the plain build after fix #1.
- **BB-05:** `block_cap` is derived from 1347776 B, and the host refuses with `RuntimeError` when no block fits.
- **BB-06 / BB-07:** the repeat-call tests pass; a cache hit patches only the two addresses.
- **BB-08:** `g6_permute_plan` reports the plan.
- **BB-17:** `all_cores` comes from `row_wise=True`, which gives 1–3 rectangles on the production shapes.
- **BB-21:** named zones on every block step and wait, compiled out of the plain build.

This prompt has no soft (prefer/consider) rules.

## Registry Conformance

- **Op file.** `INPUT_TAGGERS` (six taggers, all `(inputs, axes)`), `SUPPORTED`, `EXCLUSIONS = []` and `validate()` are present and wired. The entry point calls `validate()` first, and the op file declares no `INVALID`. The validate order is:
  1. malformed dims → `ValueError` (from `tag_dims`, before any support check);
  2. non-32×32 tile → `ValueError`;
  3. SUPPORTED per axis → `UnsupportedAxisValue`;
  4. EXCLUSIONS → `ExcludedCell`.
- **Auto-fix from evidence.** `SUPPORTED["alignment"]` was widened (Code Review #3). No XPASS was involved, because no golden cell pairs dims (0,2,1,3) with these alignments; the evidence is the new test file.
- **INVALID audit** (`feature_spec.py`): one entry, `{dtype: bfloat8_b, layout: ROW_MAJOR}`.
  - It is the canonical bf8b+RM entry, both axes describe `input_tensor`, and the universe really changes (a block format needs tiles).
  - There are no cross-tensor couplings and no "not yet supported" entries.
  - g6_permute is not norm-like, so no weight canonicalization applies.
  - The rank↔dims mismatches are structurally impossible and correctly carry no INVALID entry: `tag_dims` raises `ValueError`, and every axis is projected from a scenario.

## Design Conformance

| Dimension | Design | Implementation | Verdict |
|-----------|--------|----------------|---------|
| Algorithm | rows scattered as 32 B half-rows straight from the input slot to output pages | same; no output CB, no compute kernel | ✓ |
| Pipeline / RISC ownership | reader on NCRISC/NoC0 (`load_block`), writer on BRISC/NoC1 (`scatter_rows_block`) | `ReaderConfigDescriptor` / `WriterConfigDescriptor` | ✓ |
| Work distribution | `split_work_to_cores(grid, tensor_in_tiles, row_wise=True)` over the full grid | same; 2–32 cores on the production shapes (one tile each), all 110 cores on `[1,64,100,256]` | ✓ fills the grid the design allows |
| Inter-core communication | none | none | ✓ |
| Block knobs | `block_in_tiles`, `input_depth`, `stagger_rows` as parameters | set once in `build_plan` and passed as CT args 6 and 9. The CB capacity `input_depth · block_in_tiles · page` and every trip count derive from the same plan fields | ✓ single source, no duplicated literal |
| Expression | reader: one reserve, N reads, one barrier, one push per block. Writer: one wait, N·rows·2 writes, one flush, one pop per block; one final write barrier | same. The per-tile and per-row loops sit inside the block operations, with no per-unit handshake | ✓ |
| Ragged tail | nominal push/pop counts, narrowed transfers | `cb_push_back` / `cb_pop_front(block_in_tiles)` on every block; `extent` narrows only the reads and writes | ✓ |

**Half-turned split: none.** The per-core loop runs whole blocks (`block_in_tiles = min(tiles_g1, block_cap)`), not single tiles.

**Occupancy on the production shapes.** 2–32 of 110 cores run, one tile each. This is the design's traffic-ranked choice: splitting tile rows across cores needs mid-tile DRAM reads that break the mod-64 rule, and output-tile splits read 2× or more. Every production cell is ≤ 32 tiles, so `perf_rules.txt` puts none in scope for measurement.

## L1 Ledger Audit

- **Currency:** the code has one CB (`cb_input_tiles`, index 0, `plan.input_depth · plan.block_in_tiles · in_page_bytes`) and the ledger has one row with a matching size expression. ✓
- **Capacity vs live set:** capacity equals the live set. With ≥ 2 blocks per core, the reader fills block k+1 while the writer drains block k. Axis accounting: `in_tile` spans `block_in_tiles · depth`; `r` and `h` span the whole tile slot, because the prefix read lands in place and rows are read in place. Nothing is over- or under-sized. ✓
- **Page format vs DEST:** Float16_b is a dataflow-only payload, and with no compute kernel there is no DEST to match. ✓
- **Disjoint lifetimes:** there is one buffer and nothing to share; the ledger says why no output CB exists (pack-into-destination). ✓
- **Bounds:** `block_cap = floor((1347776 − out_l1) / (input_depth · in_page_bytes))` is closed form, and every symbol is bounded in the symbol table. ✓
- **Data-movement budget:** consistent with the code, and the per-case numbers were re-derived. x crosses DRAM once as the live prefix; out crosses once as the logical bytes in 32 B writes; cross-core traffic is 0. The cheapest-traffic split is the one implemented, and the alternatives are recorded with their deltas. ✓
- **Block-size defaults:** held. The split spreads over the full grid, then takes the coarsest block that fits (the whole assignment).
- **Per-core footprint:** `min(input_depth, max_blocks_per_core) · block_in_tiles · in_page_bytes + out_l1_bytes_per_core`. That is 2048 B on every production shape, matching the measured `device_l1_peak_bytes`. The first term scales with the block and depth knobs, the second only with output placement (0 in DRAM).
- **Findings needing a fix or a fold into a refinement:** none.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_permute/test_g6_permute_precision_baseline.py`, randn bf16, seed 1234.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP (bf16) | got/true ratio median [p5, p95] |
|-------|-----|-------------|--------------|------------------|----------------|----------------------------------|
| (1,1,16,64) | 1.0 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] |
| (1,6,16,64) | 1.0 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] |
| (2,34,47,96) | 1.0 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] |
| (1,64,100,256) | 1.0 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] |

**Assessment:** bit-exact. No value passes through an arithmetic unit, so no precision lever exists or is needed. The ratio spread is a single point at 1.0, so there is no scale bug.

**Recommended tolerances:** exact for bfloat16, as `helpers.TOLERANCES` already says. bfloat8_b keeps its one-step gate (`bf8_step`) once Refinement 2 lands it.

## Verifier CLI Summary

`verifier_results/verifier_report.json` (golden run `eval/eval_test_runner.sh`, 39 tests, after the fixes above):

- supported_pass: 20 (8 golden Qwen cells and 12 Phase-0 loose cases)
- xfail_expected: 12 (FLUX ×4, Panoptic RM ×3, Panoptic bf8b ×5). Refinements 1, 2 and 4 move them to passing.
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 7 (`test_regression.py`, not registry-driven): 4 passed, and 3 failed on the harness artifact below.

**Harness artifact.** `test_regression.py::test_malformed_dims_raise[short|repeated|out_of_range]` fails only under the graded runner.
- `pytest.raises` succeeds: the op raises `ValueError` from `validate()` before allocating anything.
- The runner's L1 graph capture (`EVAL_CAPTURE_L1=1`, `eval/l1_profiling.py:84-88`) then rejects `peak_total=0` and raises `L1ProfilingError`.
- A correct host-side refusal allocates no L1, so every op with this test pattern hits it. Under `run_safe_pytest.sh`, which has no L1 capture, all 7 regression tests pass.
- The op cannot fix this without allocating L1 on a refusal. The harness should exempt tests whose graded op never dispatches.

**Device time (golden run, `device_kernel_ns`):** 1.26–2.23 µs per Phase 0 cell, 1 program, 2–32 cores, 2048 B L1 peak per core. The B = 16 cases (32 live rows, 64 writes per tile) take ~2.1 µs, and the B = 1 cases ~1.27 µs.

## TARGET − SUPPORTED

| Axis | Missing value | Where it goes |
|------|---------------|---------------|
| layout | ROW_MAJOR | Refinement 1 (FLUX `blocks_rm_gather`), extended by Refinement 4 |
| rank | 6 | Refinement 1 |
| dims | (0, 1, 4, 2, 5, 3) | Refinement 1 |
| dims | (0, 3, 1, 2) | Refinement 2 (TILE bf8b), Refinement 4 (RM) |
| dtype | bfloat8_b | Refinement 2 |
| buffer | l1 | Refinement 2 (for every built regime) |
| alignment | — | closed in this pass |

**Documented omission: cartesian cells with no regime.** SUPPORTED is per-axis, so each new value admits cells no production case uses: TILE × (0,1,4,2,5,3), ROW_MAJOR × (0,2,1,3), bf16 TILE × (0,3,1,2), bf8b × (0,2,1,3) or (0,1,4,2,5,3), and non-aligned bf8b.
- The refinement that admits each cell puts it in `EXCLUSIONS` as a cell dict; `op_requirements.md` lists them per refinement.
- These cells get no refinement of their own, because no golden cell or production call exists for them and BB-22 limits tests to production cases.
- Their regimes are recorded in `op_design.md` (`rows_rm_page_remap`, `rows_tile_bfp8`, "other TARGET combinations").

## Recommendations

- **The FLUX perf goal in `feature_spec.LOOSE_CASES[0]` ("1.35 × roofline = 12.29 µs") cannot be met at this placement.**
  - The roofline counts the 2 MiB of values. The input, however, is 524288 ROW_MAJOR pages of 4 B, each in its own 64 B DRAM slot (`NOC_DRAM_READ_ALIGNMENT_BYTES = 64`), so 32 MiB are stored.
  - Each page costs at least one DRAM burst, so the input alone is ≥ 16 MiB of DRAM traffic (≈ 33 µs at the p150's 512 GB/s peak), before NoC request overhead (65536 page requests per bank).
  - BB-02 fixes the input placement, so the op cannot change it.
  - Refinements 3 and 5 therefore gate on a measured device-ns win over Refinement 1's baseline (and staying below stock 1056 µs), not on 12.29 µs. This is an estimate, not a measurement; worth raising with whoever owns the case.
- **`rows_tile_scatter` perf lamps that the golden suite cannot measure** (every cell of this regime is ≤ 32 tiles):
  - the overlap lamp: `block_in_tiles` 4–8 with depth 2 on many-tile shapes, where today the whole assignment is read before the first scatter write;
  - the per-row `get_noc_addr` bank decomposition in the writer, which could advance bank and offset incrementally;
  - prefix-vs-exact-row reads for `rows_live < 16`.

  None of these is a refinement: no cell of this regime is in perf scope, and Qwen runs at the 1–2 µs launch floor.
- **Cartesian EXCLUSIONS discipline.** Each generality refinement must put the cells it admits but does not build into `EXCLUSIONS` in the same change. Otherwise `validate()` accepts shapes the kernels get wrong, which violates the "every shape validate() accepts" MUST rule.
- **Environment.**
  - A direct `import ttnn` (`python3 -m eval.verify_supported` included) needs `source .tt_chip_scope`, whose `LD_PRELOAD` stubs the missing ULFM `MPIX_Comm_*` symbols. The pytest wrappers do this themselves.
  - Untracked probes `probes/probe_001..005.py` predate this pass. 001 and 005 call the removed `describe_plan` API. They are left out of the commit.
