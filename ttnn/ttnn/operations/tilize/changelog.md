# tilize — changelog

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
