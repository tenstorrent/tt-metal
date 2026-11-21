# Verification Report: nlp_concat_heads_decode

Target: quasar (8×4 = 32 clusters) on the ttsim simulator (no clock: no timing figure below is a performance
result). Specification: `eval/specs/nlp_concat_heads_decode/spec.py` (sha256 `afec79a8…`), lint PASSED
(0 errors, 5 UNRESOLVED open items, all spec-side: `tt_regen_spec.governs`, `special_values`, `freedoms`,
`PERFORMANCE.bar/measure`).

## Code Review

The implementation is small and matches the design closely: one Metal 2.0 `ProgramSpec` with one DM kernel
(`gather`) on each output core, run with one `ttnn.generic_op` dispatch. Each core pulls row `head_idx` of every user's
input tiles over the NoC straight into its own output shard: all reads are tagged trid 0 and issued back to back,
followed by one barrier and the trid hand-back.

Fixed in this pass:
- **Unused thread-plan CT arg → compile-time check of the host split.** The kernel received `reader_threads` but
  marked it `[[maybe_unused]]`. It now `static_assert`s that the host's split (`READER_THREADS` →
  `block_tiles`, `reader_threads`, one source in `nlp_concat_heads_decode_program_artifacts.py`) covers the head
  and leaves no thread empty. A future change to the clamp that breaks the partition fails at kernel compile
  instead of silently dropping or duplicating tiles. The stale comment ("or empty when the plan is wider") was
  corrected.
- **R5 had no test.** R5 is a `review` check: no valid call can reach it on a 32-core part, since
  `num_heads ≤ H ≤ 32`. Added the host-only `test_r5_too_few_cores_raises_target_number_of_cores` (a stub 2×1-grid
  device), which asserts the exact `Target number of cores 32 is greater than total number of available cores 2`
  message and that a 2-head config on that grid builds.
- **Precision baseline test added**: `test_nlp_concat_heads_decode_precision_baseline.py`.

Reviewed, no change needed:
- **Spec RULES.** R1: raw-byte NoC reads, no DEST/packer, float32 exact by probe. R2/R3: output shape
  `[1,1,32,n·hd]`, TILE, input dtype. R4: memory config built from `num_cores_to_corerangeset(...,
  row_wise=True)` unchanged, `head_idx` enumerated from `corerange_to_cores(cores, None, True)` on that same
  set. R5: an explicit `RuntimeError` with tt-metal's wording in `output_memory_config()`, before
  `allocate_tensor_on_device`. R6: `memory.memory_axes` gated through SUPPORTED, plus the `[32, head_dim]`
  shard-shape check. R7: `sub_core_grids is not None` → `ValueError`. R8: the kernel reads only row `head_idx` of
  each tile. R9: `UnsupportedAxisValue` (`NotImplementedError`). All REJECTIONS X1–X7 pass in the golden suite.
  No RULE has a `structure` part. GUIDANCE is followed: the output config is built and R5 is checked before
  allocation, and the padding rows are zeroed.
- **One native dispatch**: exactly one `ttnn.generic_op` per call. No `to_layout` / `to_memory_config` wrappers.
- **Kernel API**: `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor` for the remote input,
  `LocalTensorAccessor` for the local output shard (zero-copy; the output shard is the per-core block), no
  `InterleavedAddrGen`.
- **Helpers**: raw Noc API is justified per the design's "Helpers considered and rejected" table, because no helper
  moves a sub-tile row remote → local into a tensor shard. There is no mcast/semaphore pattern to replace.
- **Quasar tag discipline**: every transfer is `NocOptions::TXN_ID` `{.trid = 0}`, with the barrier on the same id
  and `noc_async_read_set_trid(0, …)` as the last statement. `test_session_order` runs the op after
  `to_layout` and before `to_memory_config`/`to_layout` of its result in one session, and passes.

Deferred: none (no architectural rework found).

Cosmetic deviation, left as is: the design's host checklist names the kernel `gather_head_rows.cpp`, while the
file is `kernels/nlp_concat_heads_decode_reader.cpp`.

## Design Conformance

- **Algorithm / topology**: `pull_rows`, the design's only built regime. There is no compute or writer kernel, as
  designed.
- **Work distribution**: one head per core on the R4 core set. All `num_heads` cores are engaged (1–32 clusters
  over INPUTS; the 32-head calls fill the 8×4 grid). The contract fixes occupancy, so no idle-silicon defect is
  possible.
- **Thread plan**: read off the host program as (R, C, W) = (`READER_THREADS` = 1, 0, 0). The design states the op
  is bound by data movement, with the reason ("every unit is a 32–64 B copy, no compute"), so one reader thread
  is acceptable. The plan is one named constant, clamped to `head_tiles`, passed as CT args and sized with
  `get_my_thread_id()`. The every-thread-ran test (`READER_THREADS ∈ {1,2,3,4,6}`, ragged at head_dim 160) and the
  session-order test are present and pass. This backend has no timing model, so the design's alternative
  (`READER_THREADS = 2`) cannot be measured here. `pipeline_logs/model_cycles.jsonl` does not exist for this
  run.
- **Blocking-model fidelity**:
  - `block_tiles` is derived once on the host from `head_tiles` and the plan constant, and the kernel derives its
    range from it. `num_users` and the geometry (`tile_bytes`, `face_bytes`, `run_bytes`, `face_h`,
    `runs_per_row`) are each derived once in `tile_geometry()`. There are no duplicate literals and no collapsed
    knobs.
  - Expression: each thread does one zero call and one zero barrier, then issues all `2·block_tiles·users` reads
    with one read barrier. The block is the synchronization unit, with no per-unit handshake.
  - Axis accounting matches the ledger.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from the spec. No op-local axis function.
- `SUPPORTED` covers every gated axis (`dtype`, `layout`, `memory_config`, `sharding`, `users`, `alignment`) and
  equals `PHASE0` (lint).
- `EXCLUSIONS = []`.
- `validate()` checks malformed calls (`ValueError`) first, then SUPPORTED per axis (`UnsupportedAxisValue`),
  then EXCLUSIONS (`ExcludedCell`). It is the entry point's first line.
- The op file declares no `INVALID`.
- No auto-fixes to SUPPORTED: no XPASS drift.
- **INVALID audit (spec)**: `INVALID = []`, which is correct. TARGET has a single placement (L1 / height) and a
  single layout (TILE), so there is no DRAM×sharded or bf8b×ROW_MAJOR pair to exclude. bfloat8_b is not in TARGET
  and is family-unbuildable on quasar anyway. This is not a norm-like op, so no weight canonicalization applies.
  No issues.

## L1 Ledger Audit

- **Ledger currency**: the op declares no DataflowBuffer, scratchpad or semaphore (the `ProgramSpec` has no
  `dataflow_buffers`). The ledger's two rows are the tensor-owned input/output shards, and they match the
  bindings in the kernel (`TensorBinding("input")`, `TensorBinding("output")`). Current.
- **Capacity vs live set**: the op allocates no capacity. The output shard's live set is the whole shard (the
  single block's destination) and spans `head_dim_tiles`, `user` and `face_half`, which the shard holds. No
  over/under finding.
- **Page format vs DEST**: n/a, since no value traverses DEST (raw bytes). No finding.
- **Disjoint lifetimes**: n/a (no op-owned buffers).
- **Bounds / closed form**: `head_tiles`, `tile_bytes`, `num_users`, `num_heads` and `reader_threads` each have a
  bound and an establishing predicate in the symbol table. The op-owned total is `0 B`.
- **Data-movement budget**: present and consistent with the code. DRAM 0 B; NoC `num_heads·users·head_dim·elem`
  in `2·head_tiles·users` reads per core. The core-local iDMA zero of `head_tiles·tile_bytes` per core is counted.
  The cheapest-traffic split (`pull_rows`) is the one implemented.
- **Block-size defaults**: sharded output → one block = the whole resident shard per core (per thread, its
  contiguous share). Held.
- **Per-core footprint**: op-owned 0 B. Tensor-owned output shard `head_tiles · tile_bytes` (≤ 32 KiB at head_dim
  256 float32). No term scales with a knob.

## Precision Baseline

`test_nlp_concat_heads_decode_precision_baseline.py`, bfloat16, standard-normal inputs, height-sharded via
`eval.sharding.auto_shard_config`. Compared rows 0..users−1 (R3):

| Shape (num_heads) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p99 | got/true ratio median [p5, p95] |
|-------------------|-----|-------------|--------------|------------------|---------|-------------------------------|
| (1,1,32,32) h1    | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1,1,32,128) h2   | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1,1,32,128) h32  | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1,1,32,256) h32  | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |

Verifier probe beyond SUPPORTED: validate gate relaxed in-process, kernel unchanged, not a golden result. float32
and H ∈ {8, 12, 17} (bf16 and fp32) were also bit-exact, with zeroed padding rows. See Refinement 1.

**Assessment**: the op is a bit-exact movement, as R1 / NUMERIC (`exact`) require. The ratio is pinned at 1.0 with
zero spread, so there is no scale or structural error.
**Recommended tolerances**: exact (both NUMERIC_RESOLVED entries, bfloat16 and float32, are `exact`).

## Verifier CLI Summary

Golden suite: `eval/eval_test_runner.sh` (8 sim workers, forked), 38 tests: 19 passed (12 cells + 7 rejections),
18 xfail, 1 skipped (`test_op_loose[NOTSET]`: no LOOSE_CASES). Re-run after this pass's kernel edit with the same
result.

- supported_pass: 12
- xfail_expected: 18 (12 float32 tile_aligned + 3 bfloat16 h_non_aligned + 3 float32 h_non_aligned; all queued
  in Refinement 1)
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 8 (the 7 rejection tests, all passed, plus the empty loose-case placeholder)
- Queue check (`--queue op_requirements.md`): 2 values in TARGET − SUPPORTED, 0 queue gaps.

Unit tests: `tests/ttnn/unit_tests/operations/nlp_concat_heads_decode/`, 40 passed.

## Recommendations

- **Queue shape.** `TARGET − SUPPORTED` is only `{dtype: float32, alignment: h_non_aligned}`. Both are knob-turns
  that a probe shows already work, so they are bundled into one generality refinement (Refinement 1).
  Generality is then exhausted, and the queue continues with a perf refinement (Refinement 2), gated on
  structural counts because ttsim has no clock.
- **Perf headroom is small and latency-shaped.** The op moves `head_dim·elem_bytes` ≤ 1 KiB of NoC payload per core (8 KiB total at the
  32-head Qwen call), so launch and fixed per-transfer and per-barrier costs dominate. Refinement 2 targets the two
  that are structural (per-read setup; the zero barrier ahead of the first read). Not filed:
  - `READER_THREADS = 2` (catalog `split_reader`): ≤ 16 reads per core, and the design expects it not to pay. On
    this backend the only count it changes is reads per thread, which says nothing about time. Measure it on
    `ttsim-perf` or hardware before queueing.
  - Input-core stagger (catalog `bank_stagger`): with users = 1 all cores read one source core, so rotating the
    tile order does not spread the load over endpoints. It is a likely no-op.
- **Spec observation (not a defect):** R5 cannot be reached by any valid call on quasar, because X2/X7 bound
  `num_heads ≤ H ≤ 32 = cores`. It is covered by the new host-side test and by review.
- **Simulator caveat:** the spec's GUIDANCE notes that on the simulator a fresh L1 buffer satisfies the callers'
  finiteness check, so `zero_padding_rows = 0` (tt-metal's behaviour) passing here would prove nothing about hardware. Keep the padding zeroed (GUIDANCE).
