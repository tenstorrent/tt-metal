# Changelog: to_memory_config

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier),
  for `quasar` on ttsim. One reader + one writer kernel per engaged cluster, one staging ring `dfb_pages`
  with implicit sync on both ends, regimes R-in / R-out / R-bal as core tables over the same kernels,
  `block_pages` derived once in `ring_geometry` (the whole per-core assignment when it fits the ring
  budget and the 255-entries-per-txn-id cap, else `BUFFER_DEPTH = 2` blocks). Thread plan `(1, 0, 1)`:
  no compute kernel (the op computes nothing; R1 forbids unpacking bytes). The design's risk "DM→DM ring
  with implicit sync on both ends is unexercised" did not bite.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE, ROW_MAJOR], memory_config=[DRAM, L1],
  sharding=[interleaved, height], to=[DRAM, L1, L1/height], alignment=[tile_aligned, w_non_aligned,
  h_non_aligned], rank=[2, 3, 4] (= spec PHASE0)
- **Accuracy achieved**: PCC=1.000, max_abs_err=0.000, rms_err=0.000, 0 ULP, got/true ratio exactly 1
  (measured on 4 shapes via test_to_memory_config_precision_baseline.py) — bit-exact
- **Golden suite at Phase 0**: 558 / 558 supported cells passing (558 float32 cells xfail as expected,
  372 INVALID skipped, X1 / X2 rejections pass; 0 loud categories, per `verifier_report.json`)
- **Issues encountered**: no code fix was needed in verification. Findings queued: the height-shard
  owner side is not zero-copy (Refinement 1); single-txn-id rings serialize reads and writes on some
  R-bal cores (Refinement 3); R-bank is reachable after all (Refinement 4). L1 ledger given the
  verifier's disposition of its two deferrals.
- **Tests added**: test_to_memory_config.py (harness acceptance, 55), test_to_memory_config_regimes.py
  (47: R-bal / R-in / R-out pinned, reshard 2→1 and 1→2 both layouts, COL_MAJOR + partial shard,
  depth-2 multi-block, `(2, 0, 2)` plan, one device session with `ttnn.to_layout` after),
  test_to_memory_config_precision_baseline.py (4)
- **Structural counts at Phase 0** (no timing claimed on this backend): DRAM crossings 1 per tensor side
  in DRAM; mesh hops per page 1 (R-in / R-out), 2 when an L1-interleaved side is in R-bal; one
  core-local ring copy per page on the shard side; ring txn-id slices 1 on 6 R-bal configurations of
  INPUTS × layout.

## Refinement 0 — Width-sharded tensors and configs (ResNet-50, Llama, Qwen, YOLOv8)
- Date: 2026-11-21
- What was done: `"width"` added to SUPPORTED["sharding"] and `"L1/width"` to SUPPORTED["to"]. Probe first, as
  the operator asked (gate bypassed, no kernel change): TILE width shards were already served on both sides
  (15/15 exact, since a width shard's page is a tile and the accessor maps global tile ids to shard cores). ROW_MAJOR
  was wrong in every case (15/15). The accessor's page of a row-major width-sharded tensor is a `shard_w`-wide run of
  one row (page shape `(1, shard_w)`, `buffer_distribution_spec.cpp`), not a row, so the two sides' page spaces
  differ. Fix: the ring now carries *entries* (`op_design.md`, "Entry space"). In TILE an entry is a tile, as before.
  In ROW_MAJOR it is a run of `L = lcm(Pi, Po)` elements of one row: `L / Pi` input pages and `L / Po` output pages,
  entry `g` = input pages `g * in_pieces + i` = output pages `g * out_pieces + j`, so no page straddles an entry. A
  side with one piece per entry keeps implicit sync, unchanged. A side with several pieces assembles or splits the
  entry with sized, user-tagged transfers under explicit sync (ring listed in `disable_dfb_implicit_sync_for`). This
  relies on Quasar's NoC having no address alignment (`NOC_*_ALIGNMENT_BYTES = 1`): 2-byte pieces at 2-byte offsets
  are exact on ttsim. The owner regimes R-in / R-out take a width side when its shards own whole entries (always in
  TILE). A width shard's entries are a strided set of runs (`run_len`, `run_stride` runtime args; height shards and
  R-bal ranges are one run). A row-major width side cut finer than an entry falls to R-bal or to the other side's
  owner split. Row-major width shards that do not divide W are refused (`NotImplementedError`, new mechanism cap).
  Reused: the one kernel pair, the one ring, `ring_geometry` / `block_pages` (now counted in entries), the regime
  table and its selection rule. Added: `entry_space()`, `_shard_grid()` (height + width), the `pieces` branch in
  each kernel, run-strided entry ids.
- Accuracy achieved: bit-exact (relative RMS 0, PCC 1) on every width case run: 84/84 probe cases (6 shapes × 2
  layouts × 7 placement pairs), the 16 new unit tests and the golden slice.
- Golden test progress: every buildable bfloat16 cell with `sharding=width` or `to=L1/width` passes, 532/532
  (266 TILE + 266 ROW_MAJOR; another 76 in the slice are INVALID DRAM/height skips). 40 randomly drawn prior bfloat16
  cells pass, as do test_to_memory_config.py (55), the precision baseline (4) and the regimes file (63). float32
  width cells stay xfail (Refinement 2). The full suite was not run here; the harness runs it.
- Issues encountered: none blocking. Findings: (1) an explicit side moves one entry per barrier, with
  `in_pieces` / `out_pieces` transfers per barrier. It could batch a block per barrier, which with R > 1 needs the
  STRIDED per-thread entry addressing; recorded as a perf lamp, not done. (2) Row-major width ↔
  interleaved/height cells are R-bal: the width side's pages each take one mesh hop. The shard cores own pages,
  not entries, so no owner split applies. (3) The R-in owner side still reads its own shard through the accessor
  (loopback), as for height (Refinement 1).
- Tests added: test_to_memory_config_regimes.py, Width section (16 cases): interleaved (DRAM, L1) ↔ width and back,
  height ↔ width and back, width 2 → 4 and 4 → 2 cores (R6), widths 64 ↔ 96 of W = 192 (3 and 2 pieces, explicit on
  both sides), COL_MAJOR width grid with a partial tile shard, the (2, 0, 2) thread plan on the explicit sides, one
  device session with to_layout after the explicit-sync programs, each in both layouts where it applies. Plus the
  refusal of a row-major partial width shard. The two "width is refused" tests became block-sharded refusals
  (input and destination).
