# Verification Report: paged_update_cache

Target: Quasar (`quasar`) on the ttsim simulator. Specification: `eval/specs/paged_update_cache/spec.py` (sha256 757781d7…). Regime built: `direct_row_write` (op_design.md).

## Code Review

**Fixed**
- **DRY: tile geometry stated twice.** The kernel hard-coded `TILE_ROWS = 32` and `TILE_FACE_ROWS = 16` while the host already held `TILE_DIM` / `FACE_DIM`, which validate() uses for its tile-shape gate and `face_row_bytes` / `face_bytes` are built from. Both now reach the kernel as CT args (`tile_rows`, `face_rows`) from the host constants. The kernel adds `static_assert(tile_rows == 2 * face_rows)` for the two-face-halves loop. They are still `constexpr`, so `p / tile_rows`, `% face_rows` etc. still compile to shifts and masks (no `__udivsi3`: the Quasar XIP risk in op_design.md). Acceptance, plan and golden suites re-run green.
- **Ledger currency** (`l1_ledger.md`): `index_scratch` was said to be dead during `write_rows_block`. The kernel resolves each user just before that user's writes, under one `scoped_lock` held across the user loop, so the slot is live through the writes. The row is corrected. Footprint is unchanged: there is no other buffer that could reuse the bytes.

**Reviewed, no change needed**
- Kernel syntax `void kernel_main()`, includes under `api/…`, `TensorAccessor` for the DRAM tensors, `LocalTensorAccessor` for the resident shard. There are no CBs or DFB rings, so no push/wait accounting applies.
- Every NoC transfer is tagged `NocOptions::TXN_ID` with `{.trid = 0}`, barriers are on trid 0, and both tag registers are handed back as the last statements (target page, *Ring protocols*). The session-order test passes: a tt-metal tilize, then the op twice, then `ttnn.to_layout(…, ROW_MAJOR)` of the cache, all in one session.
- Helpers: none applies. There is no compute phase, and the data-movement helpers address L1→L1 reads or tile columns staged in L1, while every transfer here is a 32/64-B face-row write from L1 into a DRAM page at an offset. I agree with op_design.md's "Helpers considered and rejected". There is no multicast or semaphore, so the Mcast helper does not apply.
- Broadcast efficiency: not applicable (no binary/unary compute).

**Deferred to refinements**: none of the review findings (both were fixed in place).

## Specification rules

| Rule | Implementation | Status |
|------|----------------|--------|
| R1 exact move, no dtype round trip | bytes copied NoC L1→DRAM; no compute, no staging buffer | holds (precision baseline: 0 ULP) |
| R2 nothing else changes | only `face_row_bytes` runs at the row's offset in the touched tiles are written; the cache is never read back or rewritten | holds (whole-cache comparison exact) |
| R3 in place, return the caller's tensor | cache is the last `io_tensors` element of the one `generic_op`; entry point returns `cache_tensor`; nothing allocated | holds (HOOKS["check"] passes every cell) |
| R4 position −1 skips | int32 compared with −1 on device before any address is formed | holds (`skipped_user` loose case) |
| R5 only entry `p // block_size` read | the whole row is fetched (stall shadow), only word `k` is interpreted | holds (`unmapped_blocks`) |
| R6 rows ≥ heads ignored | `heads` from `cache.shape[1]` only; `h < heads` | holds (2 heads in 8 rows cell) |
| R7 placement axes via `memory_axes`, no core-count assumption | `validate()` derives `memory_config` / `sharding` with `eval.memory.memory_axes(input_tensor)`; target nodes come from the shard spec | holds (X9; off-origin-shard tests at (5,2), (7,3)) |
| R8 cache / index tensor form, input dtype == cache dtype | `_check_placement` raises `NotImplementedError` for each (review check) | holds by reading `validate()` |
| R9 refuse outside SUPPORTED with NotImplementedError | `UnsupportedAxisValue` / `ExcludedCell` (`SupportRefusal`) | holds (19 xfail_expected, 0 wrong-mode) |

GUIDANCE advisories: the positions and the page-table entry are read on the device, as advised. The untilize/patch/tilize scheme tt-metal uses is replaced by an in-place face-row write; GUIDANCE allows either ("a row can equally be patched in place"). Nothing is left open.

## Registry Conformance
- `INPUT_TAGGERS = AXES`, imported from the specification. The op defines no axis function of its own.
- `SUPPORTED` covers every axis the op gates on (`dtype`, `layout`, `memory_config`, `sharding`, `users`, `alignment`) and equals `PHASE0` (`python3 -m eval.validate_contract paged_update_cache --phase0`: PASSED, 0 errors, 5 UNRESOLVED open items that belong to the specification, not the op).
- `EXCLUSIONS = []`. `validate()` checks malformed calls (ValueError, X1–X8, X10, X11), then R8 placement, then SUPPORTED per axis, then EXCLUSIONS, then the mechanism caps (tile geometry, index-scratch budget). It is the first statement of the entry point.
- The op file declares no `INVALID`.
- No auto-fixes to SUPPORTED: there was no XPASS evidence, and the lint requires SUPPORTED to equal PHASE0 at this phase.
- **INVALID audit (specification)**: `INVALID = []`, which is correct. TARGET has one placement (L1 height-sharded), so there is no DRAM × sharded pair. It has no `bfloat8_b` and no ROW_MAJOR layout, so the canonical bf8b + ROW_MAJOR entry does not apply. There are no weight axes. No cross-tensor coupling is possible.

## Design Conformance
- **Algorithm**: matches `direct_row_write`. Each `(user, head, D-tile)` unit is two face-row segments written from the resident shard to the cache page at `(2·(r>>4)+s)·face_bytes + (r&15)·face_row_bytes`. The cache is never read.
- **Pipeline / RISC ownership**: one DM kernel `row_writer` does the index reads, resolve and writes, with no compute kernel, as designed.
- **Work distribution**: the users axis (the only axis the design splits) maps onto the input's shard cores. Phase 0 has users = 1, so one cluster is engaged. This is not a single-core contract violation: op_design.md keeps `heads × dtiles` on the shard core with a positive traffic reason. Spreading it (`grid_spread_pull`) moves every input byte across the NoC twice and repeats the index round trip on every engaged core. The several-users test shows 3 users on 3 shard cores are handled.
- **Thread plan (read off the host program)**: one `KernelSpec` `row_writer`, `num_threads = ROW_WRITER_THREADS` (one named constant, = 1), passed as CT `writer_threads`; no compute kernel; no rings, so there are no per-thread buffers or ring protocols. The design states the op as bound by data movement (no compute at all, R1), with its reason, so (0, 0, 1) is acceptable. Required tests are present and green: the session-order test (after another program, before tt-metal's own `to_layout`), the off-origin shard, the scratch-cap refusal, and W = 2, 3 (a thread that did not run leaves its round-robin share unwritten, so a correct result shows each thread ran). ttsim has no timing model, so the plan could not be measured against its alternatives and no `model_cycles.jsonl` exists for the phase. The writer-thread lamp is Refinement 2, gated on a structural count.
- **Blocking-model fidelity**: the block is the shard's whole assignment (`num_users_this_core` RT from the shard spec; `heads`, `dtiles` CT from the cache shape); `writes_per_barrier` comes from one host constant (`WRITE_TXN_CAP`), the scratch size from one function (`_index_slot_bytes`, used by both validate() and the program), and the thread count from `ROW_WRITER_THREADS`. Expression: one read barrier for all index reads and one write barrier per block (32,768-transfer cap, unreached). No per-unit acquire/sync/retire. No collapsed knobs.

## L1 Ledger Audit
- **Currency**: the one op allocation, `ScratchpadSpec index_scratch` = `ROW_WRITER_THREADS · (pos_page_bytes + users_per_shard · pt_row_bytes)`, matches `create_program_artifacts`. Lifetime row corrected (above). The input shard row is the caller's tensor, consumed in place.
- **Capacity vs live set**: capacity = live set (one read barrier fills it all). The whole page-table row instead of the one 4-byte entry is a stated trade (stall shadow; `dependent_entry_read` deferred with a positive reason). The `users` axis is spanned and its capacity scales with `users_per_shard`. Nothing is collapsed.
- **Page format vs DEST**: not applicable (no DEST, no format; DM-only int32 bytes).
- **Disjoint lifetimes**: a single buffer, so there are none.
- **Bounds**: `pt_row_bytes` and `users_per_shard` are bounded by the host predicate `index_scratch_bytes ≤ get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES` (refusal test present). The total is closed-form.
- **Block-size defaults**: sharded → one block = the whole resident shard. Held.
- **Data-movement budget**: present and consistent with the code (cache 0 B read and `heads·D·E` B written; input 0 DRAM reads; one positions page and one page-table row per user). The cheapest-traffic split is the one implemented. Its successor `grid_spread_pull` is a `deferred` regime with a positive reason.
- **Per-core footprint**: `W · (align_up(users·4, 64) + users_per_shard · align_up(blocks_per_seq·4, 64))`; 576 B at the Llama call, 128 B at Qwen. It scales with W (Refinement 2's knob) and with `blocks_per_seq`, and is constant in `heads`, `D`, dtype.

## Precision Baseline

`tests/ttnn/unit_tests/operations/paged_update_cache/test_paged_update_cache_precision_baseline.py`, bfloat16, measured on the whole cache and on the written region:

| Shape (cache) | Scope | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio (p5 / median / p95) |
|---|---|---|---|---|---|---|---|
| 128x8x32x64 (Llama) | whole / written | 1.0 / 1.0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 1 / 1 |
| 8x8x32x128 (Qwen) | whole / written | 1.0 / 1.0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 1 / 1 |
| 6x12x96x96 | whole / written | 1.0 / 1.0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 1 / 1 |
| 8x4x32x512 | whole / written | 1.0 / 1.0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 1 / 1 |

**Assessment**: bit-exact, as a pure byte move must be. The ratio spread is a single point at 1.0, so there is no scale bug. A float32 probe (validate bypassed) was also bit-exact at 5 shapes (`probes/probe_001.py`).
**Recommended tolerances**: exact (every cell's `NUMERIC_RESOLVED` entry is `exact`).

## Verifier CLI Summary
(`/tmp` results; copy at `ttnn/ttnn/operations/paged_update_cache/verifier_report.json`; golden run after the code-review fix)
- supported_pass: 18 (13 cells + 5 loose cases)
- xfail_expected: 19 (16 float32, 3 bfloat16 tile_aligned) → Refinement 1
- invalid_skipped: 0 (INVALID is empty)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- rejection tests (no axes sidecar): 11 / 11 pass
- queue coverage (`--queue op_requirements.md`): 0 gaps

## Recommendations
- **Refinement 1** (float32 + tile_aligned) is expected to change validate() only. A probe confirmed the kernel already writes both exactly. It must not add `compute_kernel_config`: the signature is fixed by the specification.
- **Refinement 2** (perf, writer threads) is the only perf lever with plausible headroom: the op is latency-bound and write issue is the only cost that scales with the shape. It is gated structurally (writes per hart). The gain in time stays unconfirmed until ttsim-perf or hardware measures it.
- **Not filed as perf, by design**: `grid_spread_pull` / `index_broadcast` (more bytes on both tiers; it cannot shorten the index round trip, which is the critical path) and `dependent_entry_read` (the largest page-table row in any input is 512 B; the trade only pays past ~925 K entries). Write coalescing is impossible: the two segments of a unit are 512/1024 B apart, and consecutive D-tiles are separate interleaved pages.
- L1 pressure: none. The footprint is under 1 KiB at every specified shape.
- Specification: its 5 UNRESOLVED items (`tt_regen_spec.governs`, `special_values`, `freedoms`, `PERFORMANCE.bar` / `measure`) are open items, not defects. With no `PERFORMANCE` cases and no perf-flagged loose case, the perf region in Refinement 2 was selected by the verifier.
