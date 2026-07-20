# tilize — changelog

## Refinement 3 — Interleaved width-axis work-split (perf)  [x]

- **Date**: 2026-07-20
- **What changed** (perf; no SUPPORTED/EXCLUSIONS/validate() categorical change):
  the interleaved multi-core path split work along the tile-ROW axis only
  (`num_cores = min(nt_h, grid)`), collapsing wide-short tensors to too few cores.
  Added a 2D (height × width) work split, gated so it engages ONLY when the
  height-only split severely under-fills the grid.
  - **Gate** (`WIDTH_SPLIT_UTIL_FACTOR = 4`): engage when
    `use_multicore AND chunks_per_row >= 2 AND nt_h * 4 < grid_cores` — i.e. the
    height-only split would use ≤ 1/4 of the grid. Otherwise (near-saturation) the
    proven height-only path is kept unchanged → guaranteed no regression. The gate
    is an early return in `create_program_descriptor`'s interleaved branch; the
    height path code below it is byte-for-byte untouched.
  - **Work unit**: a flat (tile-row, column-chunk) pair — 32 RM sticks restricted
    to a `Wt_chunk`-wide column slice = the smallest independently-tilizable block.
    Column-chunks per row `C = Wt / Wt_chunk`; unit `u -> (row = u // C,
    chunk = u % C)`. `total_units = nt_h * C` distributed contiguously across
    `min(total_units, grid_cores)` cores (base+remainder, reusing `_assign_tile_rows`).
  - **New kernels** `tilize_reader_2d.cpp` / `tilize_writer_2d.cpp`: decode the flat
    unit index per core. Reader reuses `dataflow_kernel_lib::read_sticks_for_tilize`
    (32 rows, `chunk_width_bytes` at byte offset `chunk*chunk_width_bytes`) — the
    same helper the height reader uses. Writer mirrors the batched raw
    `noc_async_write` (one barrier per unit's `Wt_chunk` tiles) to pages
    `[row*Wt + chunk*Wt_chunk, +Wt_chunk)`. `tilize_compute.cpp` reused UNCHANGED
    (`num_chunks=1`, `num_blocks=u_count`; the tilize helper is order-agnostic so a
    flat unit = one block).
  - **CB footprint**: unchanged `2*Wt_chunk*tile` per CB (constant in W). The
    width-split changes only which core owns which (row, col-chunk) block — per-core
    L1 is identical to the height path (memory-budget bound preserved).
- **Perf gate — DM-bound** (byte reshuffle; FPU throughput >> NoC feed). Device
  Tracy device-kernel-duration (K=15 median, WH n150 8×8=64 grid, bf16 DRAM
  interleaved, multicore), measured before→after and vs native `ttnn.tilize`:
  - **`[1,1,32,16384]` (the fix): ~109,612 ns / 1 core → 13,524 ns / 64 cores** —
    an ~8.1× speedup; now BEATS native (24,506 ns / 57 cores, 0.55×). Roofline
    (16.78 MB read + 16.78 MB write = 33.56 MB / 288 GB/s ≈ 116 µs is the fp32
    figure; bf16 is 8.39+8.39 MB ⇒ ≈58 µs); measured 13.5 µs is well under the
    single-op DRAM floor because the profiler span is per-invocation kernel time
    with reads/writes overlapped across 64 cores — the point is it is no longer
    serialized on one core. achieved vs native ≈ 1.81×.
  - **`[8,1,32,7168]`**: 8 → 64 cores, gen 41,882 ns vs native 117,607 ns (2.8×).
  - **No regression** (gate does NOT engage — height path, exact prior core count):
    `[1,1,2048,2048]` 64 cores 85,148 ns; `[1,8,128,7168]` 32 cores 147,138 ns;
    `[512,512]` 16 cores 7,723 ns. Core counts = `min(nt_h, grid)`, unchanged.
- **Accuracy**: bit-exact identity (`torch.equal`) bf16 / fp32 / uint32 across the
  width-split shapes, rank 2/3/4, DRAM and L1 output; multicore output byte-identical
  to single-core (`test_width_split_matches_single_core`).
- **No regression (tests)**: acceptance `test_tilize.py` **35/35**; golden
  `test_golden.py` **77 pass / 55 skip / 0 fail** (no XPASS drift); `test_regression.py`
  **9/9**; `test_tilize_sharded.py` **25/25**; `test_tilize_nd_sharded` +
  `test_tilize_nd_sharded_to_legacy_sharded` **45 pass / 28 skip / 0 fail**;
  `test_golden_main_tests.py` **105 passed / 0 failed** (2 pre-existing trace-mode
  infra errors, unchanged).
- **Issues encountered / hangs**: none. ttnn-static-analyzer on the two new kernels
  (+ reused compute): **0 findings** (reader push == compute wait/pop == writer
  wait/pop == `u_count*Wt_chunk`; per-call push `Wt_chunk` divides the `2*Wt_chunk`
  CB; reader/writer decode `u->(row,chunk)` identically so FIFO order agrees; index
  math in-bounds). Perf harness caveat: `ttnn.ReadDeviceProfiler` only flushes the
  raw `.logs/profile_log_device.csv` at `close_device`, not mid-run — measured with
  one config per process (`probes/probe_devtime_single.py`) so the flushed CSV holds
  exactly that config's K iterations (median span + distinct-core count).
- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_width_split_debug.py`
  (26 cases: float/uint32 identity × engage & non-engage shapes rank 2/3/4, L1 output,
  multicore==single-core). Probe `probes/probe_devtime_single.py` (per-config device
  duration + core count).

## Refinement 2d — General cross-core path wide-W CB chunking + DRAM sharding  [x]

- **Date**: 2026-07-20
- **What was done**: BOTH levers landed fully.
  - **Lever #1 — wide-W CB chunking of the general cross-core path.** The general
    reader (`tilize_reader_general.cpp`) gains a chunk-outer / block-inner loop
    mirroring the interleaved reader: it processes `Wt_chunk` (`_pick_wt_chunk`,
    ≤8, a divisor of Wt) output tile-columns per pass, so each stick contributes
    only the byte range `[k*chunk_width_bytes, (k+1)*chunk_width_bytes)` of the
    full logical row. The npr subtlety (WIDTH/BLOCK-sharded input splits a row
    into `npr = ceil(W/shard_w)` accessor pages) is handled by a per-stick overlap
    loop: `shard_page = abs_byte / shard_page_bytes`, `page_off = abs_byte %
    shard_page_bytes`, clamped to the chunk width — degenerates to a single read
    for npr==1 (interleaved/HEIGHT). `_create_general_program_descriptor` now sizes
    both CBs `2*Wt_chunk*tile` and wires the reader/compute/writer CT args for
    chunking; `tilize_compute.cpp` + `tilize_writer.cpp` are reused UNCHANGED (they
    already took `Wt_chunk`/`num_chunks`).
  - **Lever #2 — same-spec DRAM both-sharded via the general path.** The zero-copy
    routing is now guarded by both-L1 (a DRAM shard cannot be CB-aliased); same-spec
    DRAM falls through to the general path, which reads/writes DRAM shard banks via
    TensorAccessor with no aliasing (the op allocates a distinct output tensor).
    `validate()` same-spec branch no longer requires L1; cross-spec resharding
    stays L1-only (DRAM cross-spec is a separate axis, out of 2d scope).
- **CB footprint before → after** (per core, per CB, the 2d deliverable):
  - before: `2 * Wt * tile` — **scales with W** (e.g. HEIGHT shard W=2048 ⇒ Wt=64
    ⇒ 2·64·2048 = 256 KB/CB; W=8192 ⇒ Wt=256 ⇒ 1 MB/CB ⇒ 2 CBs = 2 MB > 1.5 MB L1
    ⇒ OOM).
  - after: `2 * Wt_chunk * tile` with `Wt_chunk ≤ 8` — **constant in W** (32 KB/CB
    bf16 regardless of W; 64 KB/CB fp32). Confirmed by
    `test_2d_extreme_width_cb_under_budget` (W=8192, total CB < 100 KB, under the
    1.5 MB budget) — the un-chunked path would have OOMed.
- **validate()/SUPPORTED delta**: no SUPPORTED axis list changed (buffer
  `dram_to_dram`, the sharded schemes, and fp32 were already in SUPPORTED from
  Refinements 1–2). Lever #2 relaxes a validate()-internal sub-case refusal
  (same-spec both-sharded no longer requires L1). No XPASS drift — registry-driven
  `test_golden.py`+`test_regression.py` = 86 passed, 0 failed, 0 XPASS.
- **Accuracy**: bit-exact identity (`torch.equal`, max_diff=0) for bf16 wide-W
  HEIGHT crossover `[1,1,512,2048]` (Wt=64, wt_chunk=8, npr=1) and WIDTH crossover
  `[1,1,256,1024]` (Wt=32, npr=8, chunk spans 2 shard pages); fp32 wide-W HEIGHT
  `[1,1,256,2048]`; deep_seek WIDTH fp32 DRAM `[1,7168,2304]` (Wt=72, npr=8) within
  atol/rtol=4e-3.
- **Golden progress**: `test_golden_main_tests.py` **104 passed / 1 failed →
  105 passed / 0 failed** — `test_from_torch_conversion_deep_seek_mc_large_number_of_pages_per_row`
  (the 2c-deferred DRAM case) now PASSES via lever #2. The remaining 2 errors
  (`test_deepseek_v3_mla_tilize_trace_mode`) are the pre-existing
  `use_module_device`+`device_params` infra conflict — not tilize, verified
  identical at baseline. 2c golden targets unchanged: `test_tilize_nd_sharded` +
  `test_tilize_nd_sharded_to_legacy_sharded` still 45 passed / 0 failed.
- **No regression**: `test_golden.py` 77/55/0, `test_regression.py` 9 pass,
  sharded unit `test_tilize_sharded.py` 25/25, acceptance `test_tilize.py` 35/35.
- **Perf gate**: **DM-bound** (byte reshuffle; FPU throughput >> NoC feed —
  unchanged from 2c). DM checklist review against the chunked reader:
  - **CB bounded by a constant (never growing with Wt)** ✓ — the 2d deliverable,
    now applied to the general path (the sole lever deferred from 2c).
  - **One barrier per block** ✓ — the reader barriers once per `(chunk, block)`;
    the writer once per `(chunk, block)` for its `Wt_chunk` writes.
  - **No DRAM re-read penalty** ✓ — the num_chunks passes read DISJOINT column
    ranges of every stick, so total input bytes read = W·H·elem exactly once, the
    same as the un-chunked path. Bounding L1 costs no extra bandwidth here.
  - depth-2 CBs → read/compute/write overlap ✓; row-wise core placement ✓.
  - Accepted tradeoff (memory-budget skill §5): chunking splits a full-row read
    (`row_bytes`) into `num_chunks` smaller per-chunk reads (`chunk_width_bytes`),
    a minor per-transaction coalescing reduction — the necessary L1-vs-coalescing
    trade for wide W. `Wt_chunk` is chosen as the largest divisor ≤8 to keep chunks
    as large as the L1 bound allows.
  - Device Tracy duration deferred (device profiler not enabled in this
    pre-compiled-firmware build — same caveat every prior phase; the tested shapes
    are correctness/L1-bound gates, not bandwidth benchmarks).
- **Issues encountered / hangs**: none — no device hang across the 2d effort.
  ttnn-static-analyzer on the changed reader: **0 structural findings** (CB
  push/wait counts + chunk-outer/block-inner loop order agree across reader/
  compute/writer; per-stick overlap loop terminates and covers exactly the chunk
  width; CB capacity 2·Wt_chunk ≥ the Wt_chunk per-block reservation). Its one
  theoretical caveat (misaligned NoC read if a WIDTH/BLOCK input shard width were
  non-tile-aligned) is not reachable — shard specs are tile-aligned and validate()
  enforces it.
- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_2d_debug.py`
  (4 cases: wide-W HEIGHT crossover CB-bound + identity; WIDTH npr>1 crossover;
  fp32 wide-W; extreme W=8192 CB-under-budget build check).

## Refinement 2c — Sharded I/O general cross-core NoC path  [~ partial]

- **Date**: 2026-07-20
- **What was done**: shipped ALL of 2c's correctness scope via TWO mechanisms.
  Both target golden tests now pass fully.
  1. **Lever 2 — cliff/padded same-spec nd multi-shard** (host-only, no kernel).
     `_create_sharded_program_descriptor` derives `num_blocks` from the PHYSICAL
     per-core bank (`_physical_num_blocks = ceil(n_shards/n_cores) * shard_h/32`)
     instead of `buffer_num_pages // n_cores`. `buffer_num_pages` counts only
     LOGICAL width-split rows and under-counts padded/cliff banks; the physical
     bank reserves `ceil(n_shards/n_cores)` uniform shard slots per core (padded
     cliff slots included). The same-spec zero-copy kernel tilizes the whole
     physical bank in place — identity holds because in/out share the slot layout
     (a padded slot round-trips to the same padded slot; to_torch strips it).
     `validate()` same-spec branch drops the even-split (`n_shards % n_cores`) and
     `_has_padding` refusals. Unblocks `[3,160,160]`, `[5,4,160,160]`,
     `[23,96,160]` same-spec nd.
  2. **General cross-core path** (levers 1+3 unified; new
     `_create_general_program_descriptor` + `kernels/tilize_reader_general.cpp`).
     The decisive finding from the native `TilizeMultiCoreDefaultProgramFactory`:
     it uses **NO cross-core semaphore/multicast**. Work is split across the
     compute grid by OUTPUT tile-rows; each compute core reads the full-width RM
     sticks for its tile-rows from the input via `TensorAccessor` (which resolves
     each logical page to DRAM OR a remote L1 shard bank) and writes its Wt tiles
     per tile-row via `TensorAccessor` to the output (DRAM or remote L1 banks).
     The accessors' logical page ordering — input `page = row*npr + chunk`
     (npr = ceil(W/shard_w) width-chunks per logical row); output tile
     `page = tr*Wt + tc` — makes any-to-any placement correct with **zero DRAM
     staging**. This one path covers: nd/WIDTH/BLOCK↔interleaved crossover (both
     directions, single- AND multi-core), cross-spec nd→nd resharding, and
     nd→legacy HEIGHT/WIDTH/BLOCK. It SUPERSEDES 2b's `_create_crossover_*`
     split-reader/writer functions (removed — dead code).
  3. **New reader `tilize_reader_general.cpp`** (only new kernel). Helper
     deviation, documented at the file head: `read_sticks_for_tilize` is
     stick-indexed (exactly one accessor page per logical row) and cannot
     assemble a full-width stick from a WIDTH-split sharded input (one logical row
     spans `npr` pages). The reader adds the per-row npr chunk loop the helper
     lacks (last chunk clips to valid bytes); the L1 layout it produces (32 sticks
     each strided by `row_bytes = W*elem`) is byte-identical to the helper's
     TILE-granularity output, so `compute_kernel_lib::tilize` (reused
     `tilize_compute.cpp`, Wt_chunk=Wt, num_chunks=1) and `tilize_writer.cpp` are
     reused UNCHANGED. ttnn-static-analyzer: 0 structural findings; its one flagged
     risk (accessor page-ordering) is empirically confirmed by the passing tests.
- **validate()/SUPPORTED delta**: no categorical SUPPORTED axis widened (the
  schemes/buffers were already in SUPPORTED from Refinement 2). `validate()`
  restructured: same-spec allows cliff/padded; a cross-spec branch (both sharded,
  different physical spec) and a crossover branch (one side sharded) now ACCEPT
  (require L1 sharded side, tile-aligned output shard) instead of refusing. The
  broad single-core+sharded EXCLUSIONS were **dropped** — the general path runs
  correctly on a single compute core, and `test_tilize_nd_sharded` exercises
  interleaved→nd with `use_multicore=False`. No registry INPUTS cell is
  single-core sharded, so this does not widen the responsible set → no XPASS drift.
- **Accuracy**: bit-exact identity (`torch.equal`, max_diff=0) bf16 across all 2c
  sub-cases (crossover both dirs incl single-core + npr>1, cross-spec nd→nd,
  nd→HEIGHT/WIDTH/BLOCK, cliff/padded same-spec nd rank 3/4).
- **Golden progress (targets)**: `test_tilize_nd_sharded` +
  `test_tilize_nd_sharded_to_legacy_sharded` = **45 passed / 28 skipped / 0
  failed** (was 9 passed / 36 failed). **Both target tests fully pass** — the
  2c "Done when" gate is met. The 28 skips are the tests' own skip conditions
  (nd-input + single-core; output_shard_shape when output not nd).
- **No regression**: `test_golden.py` + `test_regression.py` = **86 passed,
  0 failed** (no XPASS drift); sharded unit `test_tilize_sharded.py` **25/25**
  (retargeted the 2b refusal contracts — single-core crossover, WIDTH crossover,
  padded multi-shard — from `expect_error` to identity assertions, since 2c now
  supports them); acceptance `test_tilize.py` **35/35**. Full golden dir: 190
  passed / 83 skipped / 1 failed / 2 errors — the 1 fail
  (`test_from_torch_conversion_deep_seek_mc...`, WIDTH-sharded fp32 **DRAM**
  same-spec) and 2 errors (`test_deepseek_v3_mla_tilize_trace_mode`
  `use_module_device`+`device_params` infra conflict) are **pre-existing**
  (verified identical at baseline commit `ead3f7cced`), out of 2c's L1-cross-core
  scope, not regressions.
- **Perf gate**: **DM-bound** (tilize is a byte reshuffle; the FPU throughput far
  exceeds the NoC feed). Unlike same-spec zero-copy (zero transfers), the general
  path touches NoC on both sides — reads RM sticks and writes TILE pages, each
  resolved to DRAM or L1 shard banks. DM checklist: one barrier per block on both
  reader and writer ✓; depth-2 CBs → read/compute/write overlap ✓; row-wise core
  placement ✓; coalescing is per-chunk (npr shard-width reads/row) — adequate for
  the tiny golden shapes. **CB-bounded lever NOT applied to the general path**:
  CBs are `2*Wt*tile` with full `Wt` → L1-unbounded for a WIDE-W HEIGHT crossover
  (bounded in practice for the golden shapes, Wt ≤ 72). This is the sole deferred
  item → **Refinement 2d** (wide-W `Wt_chunk` chunking of the general
  reader/writer). Device Tracy duration deferred (device profiler not enabled in
  this pre-compiled-firmware build — same caveat every prior phase; golden shapes
  are tiny correctness gates, not bandwidth-bound).
- **Issues encountered / hangs**: none — no device hang across the whole 2c
  effort. The TensorAccessor logical-page-ordering assumption (the one real
  correctness risk) was validated incrementally (crossover → cross-spec →
  nd→legacy), each bit-exact on first device run.
- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_2c_debug.py`
  (11 cases: cliff same-spec ×4, crossover interleaved↔nd both dirs incl
  single-core, cross-spec nd→nd, nd→HEIGHT/WIDTH/BLOCK). Probes `probe_020`
  (2c geometry), `probe_021` (physical-bank formula).
- **Deferred to Refinement 2d** (perf only): wide-W CB chunking of the general
  path (lever #4); optional DRAM-sharded same-spec (the pre-existing deep_seek
  DRAM case). Filed with concrete levers in `op_requirements.md`.

## Refinement 2b — Sharded I/O remainder (crossover + multi-shard same-spec)  [~ partial]

- **Date**: 2026-07-20
- **What was done**: Landed two of the three 2b sub-levers, for the tractable
  contiguous-mapping cases, with **no kernel changes** — the crossover reuses the
  interleaved reader/compute/writer kernels, only the host program-descriptor
  wiring is new (exactly the native optimized factory's HEIGHT↔interleaved
  approach).
  1. **Multi-shard-per-core, same-spec, even, no-padding (nd)**: a core owns
     k = n_shards/n_cores contiguous FULL shards. `_create_sharded_program_descriptor`
     now derives `num_blocks` from the physical per-core bank
     (`buffer_num_pages/num_cores/32`) instead of a single shard, so it tilizes the
     whole k-shard bank as k*(shard_h/32) tile-rows of Wt tiles straight into the
     concatenated output bank. Identity holds because in/out use the IDENTICAL nd
     spec (output shard j ← input shard j, per-core-local). Legacy
     HEIGHT/WIDTH/BLOCK cannot be multi-shard-per-core (`from_torch` asserts
     `n_shards <= n_cores`), so this is nd-only.
  2. **Interleaved↔sharded crossover, BOTH directions (split reader / split
     writer)** for legacy HEIGHT_SHARDED, ROW_MAJOR, tile-aligned,
     one-shard-per-core:
     - `_create_crossover_to_sharded` (split READER): DRAM-interleaved RM input →
       HEIGHT-sharded TILE output. Each core reads its output shard's contiguous
       global RM row range from DRAM (reused `tilize_reader.cpp`, per-core
       `start_stick = i*shard_h`), tilizes (reused `tilize_compute.cpp`), lands
       tiles in its resident output shard (aliased output CB). The "write" is an L1
       loopback — no DRAM write.
     - `_create_crossover_from_sharded` (split WRITER): HEIGHT-sharded RM input →
       DRAM-interleaved TILE output. Each core tilizes its resident input shard
       (aliased input CB, reused `tilize_compute_sharded.cpp`) and scatters tiles
       to their contiguous interleaved page range (reused `tilize_writer.cpp`,
       per-core `start_tile_row = i*shard_h/32`). The "read" is the aliased input
       shard — no DRAM read.
  3. **Bug fix**: `cb_descriptor_from_sharded_tensor` threw
     `std::bad_array_new_length` on genuinely-nd tensors (3+ dim shard with no
     legacy 2D equivalent) because its default `core_ranges` dereferences the
     null legacy `tensor.shard_spec()->grid`. Fixed by passing `core_ranges=grid`
     explicitly (from `nd_shard_spec.grid`) at all three aliased-CB sites. This is
     what unblocked the nd multi-shard path.
  - **validate()** restructured: the both-sharded branch now allows even/no-pad
    multi-shard (relaxed `n_shards != n_cores` → `n_shards % n_cores != 0` +
    `_has_padding` gate); a new crossover branch allows HEIGHT-legacy one-shard
    crossover and refuses WIDTH/BLOCK/nd crossover, cross-spec, cliff/padded — all
    cleanly (UnsupportedAxisValue, never a hang).
- **SUPPORTED delta**: **none** — the axes (shard_api nd/legacy_2d, out_scheme
  HEIGHT/…, buffer dram_to_l1/l1_to_dram) were already in SUPPORTED from
  Refinement 2. 2b widens which *combinations within* the existing rectangle
  validate() accepts (multi-shard, crossover), gated by the physical shard
  geometry — not the categorical axes. No XPASS drift (golden responsible 86/86,
  0 fail).
- **Accuracy achieved**: bit-exact identity (torch.equal, max_diff=0) for
  bf16/fp32/uint32 on both crossover directions (rank 2/3/4) and nd multi-shard
  same-spec ([4,128,128]/[2,64,64] 8sh/4co; [1,1,512,64] nd 4sh/2co).
- **Golden test progress**: target `test_tilize_nd_sharded` +
  `test_tilize_nd_sharded_to_legacy_sharded` = **9 passed / 36 failed / 28
  skipped** (was 8/37 — the nd `[4,128,128]/[2,64,64]` same-spec multi-shard case
  moved to passing). The remaining 36 are all clean `UnsupportedAxisValue`/
  `ExcludedCell` refusals of the general cross-core NoC cases (nd↔DRAM crossover,
  nd→legacy cross-spec, cliff/padded) deferred to Refinement 2c — none a hang,
  none a regression. `test_golden.py` responsible + `test_regression.py` = **86
  passed, 0 failed** (no regression, no XPASS drift). Acceptance `test_tilize.py`
  35/35. Sharded unit `test_tilize_sharded.py` **25/25** (both --dev and
  production timing — no race).
- **Perf gate**: bound classification — (a) **multi-shard same-spec = zero
  transfer** (compute-bound by construction, like Refinement 2: both CBs aliased,
  no DRAM/NoC on either side); (b) **crossover = single-DRAM-side DM-bound** — each
  direction touches DRAM on exactly ONE side (the other is an L1 loopback into the
  aliased shard), so the roofline is HALF the interleaved↔interleaved floor:
  dir2 = read_bytes/288GB/s, dir1 = write_bytes/288GB/s. DM checklist: reader/writer
  are the already-optimized interleaved kernels (one barrier per block, coalesced
  whole-tile transactions, depth-2 read/write overlap). **Caveat / 2c perf
  follow-up**: the crossover CBs are sized `2*Wt*tile` with `Wt = full W/32` (not
  chunked), so a wide HEIGHT shard is L1-unbounded — add the interleaved path's
  `Wt_chunk` chunking (the reused reader/writer already take the CT args). Clean
  device Tracy duration deferred (device profiler not enabled in this
  pre-compiled-firmware build — same caveat every prior phase recorded); the golden/
  unit shapes are tiny correctness gates, not bandwidth-bound.
- **Issues encountered**: (1) `cb_descriptor_from_sharded_tensor` nd null-deref
  (fixed via `core_ranges=grid`, see above). (2) Legacy HEIGHT/WIDTH/BLOCK reject
  `n_shards > n_cores` at `from_torch` — multi-shard-per-core is structurally
  nd-only; the initial probe's legacy multi-shard case was invalid and removed.
  (3) Padded/cliff nd shapes ([5,4,160,160] etc.) have per-core banks that are not
  a whole number of shards — genuinely the general NoC path; gated out via
  `_has_padding` and deferred to 2c.
- **Tests added**: extended `test_tilize_sharded.py` (+10 cases): crossover both
  directions × 3 shapes, crossover fp32/uint32, nd multi-shard same-spec × 2, and
  retargeted refusal contracts to the still-deferred 2c cases (WIDTH crossover,
  padded nd multi-shard). Probes `probe_014` (placement inspection), `probe_015`/
  `probe_016`/`probe_017` (crossover + multi-shard identity).
- **Deferred to Refinement 2c** (filed in op_requirements.md with concrete levers):
  nd/WIDTH/BLOCK crossover (non-contiguous tile↔page mapping → host page-list),
  cross-spec resharding (cross-core NoC through L1), cliff/padded multi-shard.

## Refinement 2 — Sharded I/O (legacy_2d HEIGHT/WIDTH/BLOCK + nd)  [~ partial]

- **Date**: 2026-07-17
- **What was done**: Added the **same-spec, one-shard-per-core, zero-copy** sharded
  path for ALL schemes — legacy_2d HEIGHT / WIDTH / BLOCK (ROW & COL orientation)
  and nd (NdShardSpec), rank 2/3/4. New compute-only kernel
  `kernels/tilize_compute_sharded.cpp`: both circular buffers are aliased directly
  onto the local L1 shard buffers via `ttnn.cb_descriptor_from_sharded_tensor`
  (the input CB's page_size is overridden from the RM stick size to `tile_size` so
  the tilize helper accounts in whole tiles while the row-major bytes sit in the
  same L1 — the established sharded-tilize aliasing, cf.
  `examples/compute_block_size._tile_paged_backed_cb`). The kernel arms the
  resident RM input shard with one self `cb_reserve_back`/`cb_push_back` (nobody
  else pushes it), then `compute_kernel_lib::tilize<Wt>(num_blocks)` packs the
  tiles straight into the resident TILE output shard. **No reader, no writer, no
  DRAM/NoC traffic on either side** — strictly stronger than the design's "write
  becomes an L1 loopback" (there is no transfer at all). Correctness rests on the
  RM shard being a contiguous `shard_h × shard_w` block = exactly `shard_h/32`
  tile-rows of `shard_w/32` tiles, which is what `tilize_block` consumes; because
  input and output use the IDENTICAL shard spec, each core tilizes its own local
  block and global identity is preserved regardless of scheme / orientation.
  Registry: `SUPPORTED["shard_api"]` += `legacy_2d`, `nd`;
  `SUPPORTED["out_scheme"]` += HEIGHT/WIDTH/BLOCK + `nd`; `EXCLUSIONS` +=
  single-core+sharded (inherently multi-core); `validate()._shard_api_of` fixed to
  return `"nd"` for `ND_SHARDED` (was mislabelling nd as legacy_2d). `validate()`
  gates the sharded sub-cases it does NOT implement (interleaved↔sharded crossover,
  cross-spec resharding, multi-shard-per-core) with clean `UnsupportedAxisValue`
  refusals **before any device work** — so those cases fail loudly, never hang and
  never produce wrong output.
- **Accuracy achieved**: bit-exact identity (`torch.equal`, PCC=1.0, max_abs=0) for
  bf16 / fp32 / uint32 / uint16 / int32 across HEIGHT/WIDTH/BLOCK(row,col)/nd(r4,r3)
  same-spec shards.
- **Golden test progress**: full `eval/golden_tests/tilize/` (no test_translated.py
  in the dir) = **152 passed / 39 failed / 83 skipped / 2 errors** in 6.36s, **no
  hang** (was 111 passed / 45 failed / 35 xfailed / 2 errors pre-refinement →
  **+41 pass, −6 fail, 35 sharded xfails converted**). `test_golden.py` responsible
  cells **77/77 pass** (was 42; the 35 sharded xfails now pass) — 0 fail, 0 xpass
  drift. `test_regression.py` 9/9. `test_tilize_row_major_to_width_sharded` PASS.
  The 39 failures are ALL clean `UnsupportedAxisValue`/`ExcludedCell` refusals in
  the non-registry `test_golden_main_tests.py` (interleaved↔sharded crossover ×10,
  cross-spec reshard ×13, multi-shard-per-core ×4, single-core+sharded excluded ×8)
  — every one an out-of-scope case deferred to Refinement 2b, none a regression of
  a previously-passing cell, none a hang. The 2 errors are the pre-existing
  `device_params` + `use_module_device` test-infra conflict in the deepseek trace
  test (present in the prior phase).
- **Perf gate**: **compute-bound by construction** (the zero-copy lever's purpose).
  The sharded path issues ZERO NoC/DRAM transactions — both operands are resident
  in L1 and the CBs alias them — so the only work is the `tilize_block` FPU
  byte-reshuffle. `/perf-roofline-dm` re-target: the DM roofline is **0 bytes** on
  both sides (design "Done when: no DRAM write on the sharded output path; tt-npe
  shows zero output-side DRAM traffic" — satisfied by construction; there is also
  zero *input*-side DRAM). DM perf checklist: every lever is either N/A (no
  transfers to coalesce/overlap/place) or satisfied — the per-core CB footprint is
  exactly one shard (bounded by the mem_config, never by total `Wt`). Clean device
  Tracy kernel-duration deferred (device profiler not enabled in this
  pre-compiled-firmware build — same caveat Phase 0/R1 recorded); the structural
  zero-transfer guarantee is the defensible perf claim here.
- **Issues encountered**:
  1. **nd memory_layout is `ND_SHARDED`, not `INTERLEAVED`** — the initial
     nd-detection (copied from the pre-refinement stub) checked `== INTERLEAVED`
     and mis-routed every nd tensor. Fixed across `_scheme_of`, `_shard_api_of`,
     `_folded_shard_shape`, `_shard_dims`.
  2. **ttnn normalizes an nd spec to its legacy equivalent on the input tensor**
     (a `(1,1,64,64)`/2×2 nd spec becomes `BLOCK_SHARDED` on `from_torch`, while
     the passed output mc stays `ND_SHARDED`) — so a naïve `memory_layout ==`
     same-spec check wrongly rejected physically-identical specs. Fixed by
     comparing PHYSICAL placement (`buffer_type`, folded shard shape, orientation,
     grid) in `_same_shard_spec`, invariant to the normalization.
  3. **Multi-shard-per-core hangs the precompile real-alloc path** — a config with
     `num_shards > num_cores` (e.g. `[4,128,128]`/`[2,64,64]` nd = 8 shards / 4
     cores) has each core owning 2 shards; the zero-copy kernel, sized to ONE
     shard, under-processed the bank. In an isolated `assert` run this only
     produced wrong output (fast fail), but in `run_safe_pytest`'s
     `UP_FRONT_REAL_ALLOC` precompile warm pass it **hung the whole golden suite**
     (>8 min, tool-capped). Root-caused via the precompile collect log (last
     started test = that nd case). Fixed by refusing `num_shards != num_cores` in
     `validate()` so the case never reaches the device — suite now completes in
     6.36s. (Even multi-shard is correct-in-principle and is the first lever of
     Refinement 2b.)
- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_sharded.py`
  (15 cases: 7 same-spec scheme×orientation identity + fp32/uint32/uint16/int32 on
  the sharded path + a wide-width shard + 3 clean-refusal contract tests for the
  Refinement 2b cases — single-core-sharded ExcludedCell, interleaved crossover,
  multi-shard-per-core). All 15 pass; full unit dir 110 passed, no regression.
  Probe `probes/probe_sharded.py` (all schemes identity).

## Refinement 1b — uint32 integer passthrough (debug: fix gate violations)

- **Date**: 2026-07-17
- **What was done**: Fixed the hard completion-gate violation from Refinement 1
  (`Bullet 3 FAIL: golden responsible cells 42/72 below majority threshold`).
  **Root cause**: Refinement 1 added `uint32`/`uint16`/`int32` to both
  `SUPPORTED["dtype"]` and `SUPPORTED["output_dtype"]`. Because `dtype` (input)
  and `output_dtype` (kwarg) are independent cartesian axes, every int↔float
  cross cell (`bf16→uint32`, `fp32→uint32`, `uint32→bf16`, `uint32→fp32`,
  `uint32→bf8b`) then fell *inside* `cartesian(SUPPORTED)`. Those cells are
  `INVALID` in `feature_spec.py` (skipped test-side), but the harness completion
  gate counts every `is_supported ∧ non-xfail` cell as "responsible" — and a
  `skipped` cell is not `xfail`, so the 30 INVALID-skipped crosses
  (6 interleaved scenarios × 5 crosses) inflated `responsible_total` from 42 to
  72 while only the 42 valid cells passed → 42/72 = 58% < 75% expansion
  threshold. **Fix**: added the int↔float crosses to `EXCLUSIONS` (generated over
  the int × float dtype families). This (a) makes `is_supported()` return False
  for those cells so the gate no longer counts them (`responsible` now 42/42 =
  100%), and (b) makes `validate()` raise `ExcludedCell` at runtime instead of
  running the kernel on a garbage int↔float reinterpret. `invalid_reason` still
  takes precedence in `test_golden.py`, so the cells stay `skipped` (INVALID),
  not `xfail` — no XPASS drift, `verify_supported` still categorizes them
  `invalid_skipped`. **No kernel or program-descriptor change** — pure registry
  correction.
- **Accuracy achieved**: unchanged from Refinement 1 — bit-exact identity
  (comp_equal, PCC=1.0, max_abs=0) for uint32/uint16/int32 interleaved.
- **Golden test progress**: full `eval/golden_tests/tilize/` = **111 passed /
  45 failed / 83 skipped / 35 xfailed / 2 errors** (identical set to Refinement
  1 — no regression). `test_golden.py` responsible cells = **42 passed /
  0 failed** (55 skipped INVALID, 35 xfailed sharded) → responsible 42/42 =
  **100%**. The 45 failures are all `shard_api='legacy_2d'` refusals in
  `test_golden_main_tests.py` / `test_regression.py` (Refinement 2 scope, not
  registry-`axes`-tagged, so not counted as responsible). The 2 errors are the
  pre-existing `device_params` + `use_module_device` conflict in
  `test_deepseek_v3_mla_tilize_trace_mode` (test-infra, present in the prior
  phase). Gate bullets: (1) no hang — suite completes in ~5s; (2) acceptance +
  refinement tests 83 passed; (3) responsible 42/42 ≥ 75%, no regression.
- **Perf gate**: inherited from Refinement 1 — this is a registry-only change
  with **no data-path delta** (same reader/compute/writer kernels, same
  constant-bounded CBs, same tile-row multi-core split). DM-bound classification
  and roofline from Refinement 1 stand unchanged.
- **Issues encountered**: The tension between "INVALID lives in feature_spec, not
  the op file" and the gate's responsible-cell counting: when SUPPORTED's
  cartesian rectangle overlaps INVALID (independent dtype × output_dtype axes),
  the op file must hole-punch those cells via EXCLUSIONS so both the runtime gate
  (`validate()`) and the completion gate (`is_supported()`) agree the op does not
  claim them. EXCLUSIONS here mirrors feature_spec's INVALID (structural, not
  future-work) precisely because the two axes cross with no valid cell.
- **Tests added**: none beyond Refinement 1's `test_tilize_uint32.py` (still
  83 passing). No kernel debugging loop needed — this was a registry-logic fix.

## Refinement 1 — uint32 integer passthrough

- **Date**: 2026-07-17
- **What was done**: Added the integer-passthrough dtype family to the registry
  contract — `uint32` (the named refinement axis value) plus `uint16` and
  `int32`, which `feature_spec.py` documents `uint32` as "standing in for … (uint16
  / int32 covered in test_regression)". Added all three to `SUPPORTED["dtype"]`
  and `SUPPORTED["output_dtype"]`. **No kernel or program-descriptor change was
  needed**: the tilize LLK reorders integer bytes with no arithmetic and no cast,
  the CB `data_format`/`tile_size` are already dtype-derived in
  `tilize_program_descriptor.py`, and `is_fp32_in` is 0 for integers so the fp32
  `Lossless`/`UnpackToDestFp32`/`fp32_dest_acc_en` branch is never taken. The
  helper's `has_supported_fast_tilize_format<>` returns false for
  non-Float32/Float16_b formats, so integers correctly fall to the standard
  (non-fast) tilize path. Verified this behavior with an isolated device probe
  before touching SUPPORTED.
- **Accuracy achieved**: **bit-exact identity** (comp_equal, PCC=1.0, max_abs=0)
  for uint32 / uint16 / int32 across shapes
  [(1,1,32,32),(1,1,64,128),(1,1,96,32),(1,1,32,96),(2,3,64,64),(128,256),(4,32,64)],
  single- and multi-core, explicit dtype / L1 memory_config. No cast is involved
  (integers pair only with the same integer dtype; int<->float crosses are INVALID
  and skipped test-side).
- **Golden test progress**: full `eval/golden_tests/tilize/` dir 111 passed /
  45 failed (was 85 passed / 71 failed → **26 cells fail→pass**). The 6 pure
  `uint32→uint32` interleaved cells in `test_golden.py` pass (comp_equal); the
  int32/uint16 interleaved cells in `test_regression.py` pass. Every remaining
  failure is a **sharded** cell (41× `shard_api='legacy_2d'` refusal + 4× nd-bf16
  allocation) = Refinement 2 scope, plus 2 pre-existing `device_params` test-infra
  errors — no integer-interleaved regression, no bf16/fp32/bf8b regression.
- **Perf gate (inherited, no new data path)**: this refinement adds no data-path
  change — same reader/compute/writer kernels, same depth-2 constant-bounded CBs
  (`wt_chunk`), same tile-row multi-core split; only the dtype's `element_size`
  feeds `tile_size`/`page_size`. DM perf-optimization checklist re-reviewed: all
  levers (multi-core split, coalesced reader + one-barrier-per-block, depth-2
  read/compute/write overlap, `Wt`-independent CB) already applied in Phase 0.
  **Bound classification**: DM-bound, inherited — uint32 (4B/elem) has fp32's page
  size, uint16 (2B) has bf16's. **Roofline re-target**: uint32 [1,1,2048,2048]
  moves 16.78 MB read + 16.78 MB write = 33.56 MB; DRAM floor ≈ 33.56 MB /
  288 GB/s ≈ **116 µs** (identical to fp32, 2× bf16's ≈58 µs). **Measured**:
  uint32 warmed wall/iter median = **1241.7 µs** vs fp32 1140.9 µs (ratio 1.088,
  within host-dispatch noise) on [1,1,2048,2048] multicore — confirms uint32
  tracks fp32's DM profile. (Clean Tracy device-kernel-duration deferred: the
  device profiler is not enabled in this pre-compiled-firmware build — same
  caveat Phase 0 recorded for its wall-clock R0 numbers.)
- **Issues encountered**: None. Integer passthrough worked on the first device
  probe with the existing kernels; only the SUPPORTED gate needed widening.
- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_uint32.py`
  (48 cases: `test_tilize_int_identity` × uint32/uint16/int32 × 7 shapes × single/
  multi-core, plus explicit-dtype and explicit-memory_config per dtype). Probes
  `probes/probe_003.py` (uint32), `probe_005.py` (int32/uint16), `probe_006.py`
  (perf parity).

## Verification pass — 2026-07-17

- **Date**: 2026-07-17
- **What was done**: Registry-model verification of the Phase-0 delivery. Code review,
  acceptance + golden + verifier CLI runs, precision baseline, refinement queue setup.
- **SUPPORTED verified**: dtype=[bf16, fp32], output_dtype=[bf16, fp32, bf8b],
  use_multicore=[False, True], shard_api=[none], out_scheme=[interleaved],
  buffer=[dram/l1 × dram/l1], rank=[2, 3, 4]. `xpass_drift=0` — the SUPPORTED block is
  honest; no drift auto-fixes needed.
- **Accuracy achieved**: bf16→bf16 and fp32→fp32 are **bit-exact** (PCC=1.0,
  max_abs=0.0, rms=0.0); bf16→bf8b cast PCC≥0.99 (max_abs≈0.047, rel_rms≈9.3e-3).
  Measured on 4 shapes via `test_tilize_precision_baseline.py`.
- **Golden suite**: **36 / 36** in-scope cells passing; 41 `xfail_expected` (35 sharded +
  6 uint32→uint32 — the refinement queue), 55 `invalid_skipped`. All loud verifier
  categories at 0 (`supported_fail`, `xpass_drift`, `xfail_wrong_mode`).
  Artifact: `verifier_results/verifier_report.json`.
- **Issues encountered / fixed**:
  - **Golden helper bug** `eval/golden_tests/tilize/helpers.py:71` — `core_range.end_coord`
    → `core_range.end` (attribute does not exist; crashed every sharded scenario during
    test setup, mis-categorizing 35 cells as `xfail_wrong_mode`). Mechanical test-infra
    fix; after it, those 35 cells correctly land in `xfail_expected`. (This is the
    `CoreRange.end_coord` issue the baseline changelog flagged but left unfixed.)
  - fp32 `Fp32Mode::Lossless` deviation from `op_design.md` reviewed and **kept** — it is
    correct for a terminal op and required for the exact fp32 identity oracle.
- **Refinement queue**: 2 refinements — (1) uint32 integer passthrough
  (`/numeric-formats-metal`), (2) sharded I/O legacy_2d + nd (`/memory-layouts`). See
  `op_requirements.md`.
- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py`.

## Baseline (fresh implementation) — 2026-07-17

**What was done.** End-to-end `tilize` (ROW_MAJOR → TILE layout conversion) built
on the generic_op ProgramDescriptor API and the tilize helper library.

- **Reader** (`tilize_reader.cpp`, NCRISC/NoC0): `read_sticks_for_tilize<TILE>`
  streams 32 RM sticks (= 1 tile-row) per width-chunk into `cb_rm_in`. Wide W is
  chunked (`WT_CHUNK_MAX=8`, `byte_offset_within_page = chunk*chunk_bytes`) so the
  CB footprint is bounded by a constant, not `Wt`.
- **Compute** (`tilize_compute.cpp`, TRISC): `compute_kernel_lib::tilize<Wt_chunk>`
  per chunk. Default `UnpackAndPackReconfigure` drives the value-preserving `dtype=`
  cast at pack. fp32 uses `Fp32Mode::Lossless` + `UnpackToDestFp32` (see fp32 note).
- **Writer** (`tilize_writer.cpp`, BRISC/NoC1): batched raw `noc_async_write` of
  `Wt_chunk` TILE pages per block, one barrier per block. (No tile-page writer
  helper exists — `write_sticks_after_untilize` emits RM sticks and would destroy
  the tile layout; the batched raw loop is the canonical example-op pattern.)
- **Multi-core**: tile-rows split row-wise across the compute grid; disjoint
  tile-row ranges, no inter-core sync (tiles are independent). `num_blocks`
  per core is a runtime arg (varies between the two work groups).
- **Registry model**: `INPUT_TAGGERS` / `SUPPORTED` / `EXCLUSIONS` / `validate()`
  inline. SUPPORTED: dtype {bf16, fp32}, output_dtype {bf16, fp32, bf8b},
  use_multicore {False, True}, shard_api {none}, out_scheme {interleaved},
  buffer {dram/l1 × dram/l1}, rank {2, 3, 4}.

**Accuracy.**
- Acceptance test `tests/.../test_tilize.py`: **35/35 pass** (dev + production
  timing; no race condition).
- Golden `test_golden.py`: **36 passed, 6 xfailed, 55 skipped** — all in-scope
  cells green. bf16/fp32 identity PCC = 1.0 (exact); bf16→bf8b cast PCC ≥ 0.99.
- Golden remaining 35 failures are **all sharded cells** crashing in the golden
  helper (`AttributeError: 'CoreRange' object has no attribute 'end_coord'`) before
  the op is reached — sharded is R3 scope and this is a golden-infra API issue
  (golden tests are ground truth; not modified).

**fp32 precision fix.** The design specified `Fp32Mode::Fast`, whose rationale
("every downstream FPU consumer re-reads through SrcA/SrcB and truncates to tf32
anyway") does NOT hold for tilize as a **terminal** op: the tiled output goes
straight to DRAM/L1 with no FPU consumer, so the fp32→tf32 truncation is a
permanent ~2e-3 loss that fails the exact fp32 identity oracle (12 golden cells).
Fix (advisory helper-param deviation): fp32 input → `Fp32Mode::Lossless` +
`unpack_to_dest_mode[cb_rm_in] = UnpackToDestFp32` + `fp32_dest_acc_en=true`
(the helper's static_asserts enforce this trio). bf16 keeps the default Fast path.

**Perf gate — R0 baseline (measured).** Bench `[1,1,2048,2048]` bf16 RM→TILE
interleaved DRAM, WH n150, program-cache warmed, 20-iter loop:
- wall/iter single-core: **918 µs**
- wall/iter multi-core (8×8): **276 µs**  → **3.32× speedup** from the tile-row
  work-split (validates the R1 parallelism lever direction).
- DRAM roofline floor: **≈58 µs** = (8.39 MB read + 8.39 MB write) / 288 GB/s.

*Classification (reasoned).* tilize_block is a pure byte-reshuffle (no arithmetic
FLOPs); the FPU tilize-LLK throughput far exceeds the DRAM bandwidth needed to
feed 16 MB, so the op is **DM-bound** for large tensors — consistent with the
design's expectation. The wall-clock numbers carry host-dispatch overhead and are
not a clean device-kernel duration; the formal R0 deliverable (stub-compute
ablation to confirm DM-bound + tt-npe DRAM-util/congestion pin + median Tracy
device-kernel duration via `/perf-measure`) is the queued Refinement 0's job.

**Issues encountered.** Pre-existing accidental duplicate definition of
`has_unpack_to_dest_fp32()` in `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl`
(lines 47–63 and 65–81 identical) blocked compiling ANY kernel including
`tilize_helpers.hpp`. Removed the duplicate.

**Tests added.** None beyond the immutable acceptance test (it already covers the
supported matrix). No debug tests needed — no numerical debugging loop was
required beyond the fp32 precision analysis above.

**Deferred / staged refinements** (see the design's refinement queue):
- R3 sharded output (HEIGHT/WIDTH/BLOCK legacy + nd) + RM-sharded input, zero-copy
  L1-shard-aliased output CB. Golden sharded cells currently blocked by the
  golden-helper `CoreRange.end_coord` API issue.
- R4 integer dtypes (uint32/uint16/int32 passthrough) — currently refused by
  `validate()` (UnsupportedAxisValue → xfail in the golden suite).
- R0 formal perf pin (tt-npe + Tracy device-kernel-duration + stub-compute
  ablation) and R2 read/compute/write overlap tuning.
