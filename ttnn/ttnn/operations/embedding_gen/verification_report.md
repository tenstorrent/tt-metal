# Verification Report: embedding_gen

Target: Quasar (`quasar`), backend: ttsim simulator (no clock). Specification: `eval/specs/embedding_gen/spec.py` (sha256 `8620ad2c7fe9…`). Design: `op_design.md` (gather-direct). Phase verified: Phase 0.

## Code Review

What was fixed (all in this pass; acceptance, op-carried and golden suites re-run green after each):

1. **DRY: block geometry defined twice.** The unit → (batch, token group, first token, `n_tok`, first column, `n_cols`) mapping was written out separately in the reader (`block_geometry()`) and the writer (inline). They had the same inputs but two copies of the formula, which would drift the moment a refinement changes the block geometry (Refinement 3 does exactly that). Moved into one header, `kernels/embedding_gen_common.hpp` (`embedding_gen::Geometry::block()`), which both kernels include. Both build the `Geometry` from the same common runtime args of `plan_blocks`.
2. **DRY: the face-halves count was a literal `2`.** The reader's `face_off` (`(r/16)·2·face_bytes`) and the half loop (`h < 2`) hard-coded the number of 16-element halves per tile row. It is now a CT arg, `faces_per_row = TILE_WIDTH // FACE_WIDTH`, derived on the host from the same tile constants as `face_bytes` / `tile_bytes`.
3. **Stall shadow: index prefetch moved ahead of `reserve_back`.** The next block's index-slice read was issued after `ring.reserve_back(1)`, so a full ring (writer behind) delayed it. It depends only on the block number, and the slot it fills held the previous block's indices, which that block's gather has already loaded. Issuing it before the reserve is safe, and it hides the read under the ring wait as well as the gather. The block's single read barrier still covers it. This deviates from the order of op_design.md's block-schedule pseudo-code, and the reader's header comment documents it.

Reviewed and left as is:
- **Unbound `kernels/embedding_gen_compute.cpp`** (an empty `kernel_main`). The design has no compute kernel (thread plan C = 0) and the program binds only reader + writer. The implementer added the file because the harness's output list names it (the friction is logged in `agent_logs/`). It is harmless, so it is kept rather than risk a harness file check.
- **Raw DM API, no helper.** Correct for this op: no compute phase exists, and the DM helpers do not cover a gather (tilize/untilize helpers are compute-side over whole tile-rows; mcast has no shared operand). No `InterleavedAddrGen`, no deprecated kernel syntax, and the includes use the `api/...` paths.
- **Ring sync.** One `reserve_back(1)`/`push_back(1)` per block on the reader and one `wait_front(1)`/`pop_front(1)` per block on the writer, ragged blocks included. Explicit sync with the ring in both kernels' `disable_dfb_implicit_sync_for`; every transfer carries trid 0, barriers are on trid 0, and the trid is reset at exit (target page *Ring protocols*). Push count = pop count.
- **R2 guard.** `v = v < V ? v : V-1` comes before any address is formed. The index-slice read stays inside page `b`, and TILE-table pages are bounded by `ceil(V/32)·Dt`. Op-carried test `test_index_beyond_table` passes.
- **Broadcast efficiency:** n/a (no compute, no broadcast operand).

Deferred to refinements (architectural, not fixable in this pass): the TILE-output grid fill on small-S calls (Refinement 3), single-block-per-core overlap and issue parallelism (Refinement 4), RM-row read coalescing (Refinement 5).

### Specification rules

| Rule | Applies at Phase 0? | Finding |
|---|---|---|
| R1 (move unchanged) | yes (bf16); float32 cells later | Byte copies only (DRAM → ring entry → DRAM), never through DEST or an unpacker; holds by construction. Precision baseline: bit-exact. |
| R2 (no index check, no out-of-tensor access) | yes | Device clamp before any address; op-carried test. Re-arms for `w_non_aligned` RM tables (Refinement 1 must clip the last row's read). |
| R3 (indices in DRAM or L1) | DRAM only | L1 refused by name (`memory_config` axis); probe shows it already runs (Refinement 2). |
| R4 (refuse with NotImplementedError) | yes | `UnsupportedAxisValue` / `ExcludedCell` subclass `NotImplementedError`; 800 xfail_expected, 0 wrong-mode. |
| R5 (bf8b + RM output refused) | guard present | Explicit check in validate(); bf8b is also outside SUPPORTED. |
| R6, R7 (padding_idx / embeddings_type result-neutral) | yes | Read with `type_word` / `tag_embeddings_type`; X3, X4 pass; padded and binary cells pass. |
| R8, R9 (output layout; default = table's) | yes | `canonical_layout`; the default-call test passes. |
| R10, R11 (output rank, table rank) | rank 2/4, table rank 4 | `output_shape` already handles `[S]` and `[V, D]` (Refinement 1). |
| R12, R13, R14 (placement) | yes | Output allocated INTERLEAVED in `memory_config` or the indices' buffer; `canonical_memory`, X5 passes; `eval.memory.memory_axes(input_tensor)` gates `memory_config` / `sharding`. HOOKS["check"] placement passes in every cell. |

No rule has a `structure` part. **GUIDANCE advisories**: all followed (no compute, token split, shapes reduced up front, rows read straight out of tiles, padding/BINARY not cached, indices through their accessor). The output is allocated with the `allocate_tensor_on_device(shape, dtype, layout, device, memcfg)` overload rather than the spec-object form GUIDANCE mentions; they are equivalent.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from `eval.specs.embedding_gen.spec`; no op-local axis function.
- `SUPPORTED` equals `PHASE0` on all 10 axes (the spec lint `--phase0` passes).
- `EXCLUSIONS = []`.
- `validate()` reads the cell, then checks SUPPORTED per axis, then EXCLUSIONS, then R5 and the non-axis kernel facts (uint32 RM indices, 32×32 tiles, table in DRAM interleaved). The REJECTIONS (X1–X5: malformed calls) are raised first, inside `cell_axes`, before the support check. The public entry calls `validate()` first.
- The op file declares no `INVALID`.
- No XPASS drift; SUPPORTED needed no auto-fix.
- **INVALID audit (specification)**: one entry, `{dtype: bfloat8_b, layout: ROW_MAJOR}`. It is a single tensor (the table), structurally impossible (block float has no RM form), and the canonical bf8b+RM entry for a tile-or-RM operand. No cross-tensor entry. Not norm-like. R5 (bf8b table × RM *output*) is correctly a rule and not INVALID, because `output_layout` is an argument. `python3 -m eval.validate_contract embedding_gen --phase0` reports 0 errors and 5 UNRESOLVED open items (spec-owned).

## Design Conformance

- **Algorithm / pipeline**: gather-direct as designed. Each piece is one NoC read from DRAM to its final L1 position in the output layout; the writer stores the entry. No compute kernel, no staging copy.
- **Work distribution**: `split_work_to_cores(grid, num_units, row_wise=True)` over the full `compute_with_storage_grid_size()`; token groups primary, column chunks fill the idle cores. Both dataflow halves are block-batched: all of a block's reads, then one barrier; all of its writes, then one barrier. This matches the design's worked plans exactly (e.g. Llama TILE→TILE on 2 cores, `(4,1,1,128)×(1,1,2048,512)` on 32).
- **Thread plan** (read off the host program): `THREAD_PLAN = (1, 0, 1)`, one constant. Reader and writer `num_threads` come from it, there is no compute kernel, and the CT args `reader_threads` / `writer_threads` stride the block loops from `get_my_thread_id()`. The ring has `RING_DEPTH·max(R, W)` entries and the scratchpad holds `R·INDEX_SLOTS·BLOCK_TOKENS_MAX·4` B. The design states the op is data-movement bound, with its reason: one thread of each kind is acceptable and matches the target page's *Choosing the plan*. Tests present and passing: the plan probe (`test_plan_is_dealt_to_every_unit`: thread counts, every unit dealt, no empty core), other plans (2,0,2), (2,0,1) and (1,0,2) exact, and the one-device-session test (after `to_layout`, and before tt-metal's `to_layout`). Backend is ttsim (no timing model), so the plan was not measured against the alternatives; that is the gate of Refinement 4 on a timed backend.
- **Blocking-model fidelity**: every knob (`block_tokens`, `block_cols`, `num_col_chunks`, `RING_DEPTH`, `INDEX_SLOTS`, `BLOCK_TOKENS_MAX`, `THREAD_PLAN`, grid, ring cap) is a parameter decided once in `plan_blocks()`. Ring entry size, loop bounds and runtime args derive from it. No CB scales with a whole-op dimension (entry = `block_tokens·block_cols·32·esize`, capped by `DFB_MAX_BYTES`). After fixes 1–2 the geometry and tile constants also have a single source. Expression: the reader and writer are both block-scoped (one reserve/push, one wait/pop, one barrier per block); the per-token loop sits inside the block with no per-unit sync.
- **Departure noted, accepted by the design**: TILE-output `block_tokens` is pinned at 32. The design calls it a mechanism cap, but the real constraint is "within one output tile-row". Small-S TILE-output calls therefore under-fill the grid (Llama TILE: 2 of 32 cores; `(1,1,1,32)×(1,1,64,128)` TILE: 4 of 32). The design's worked plans show this and list it as perf lamps, so it is not a contradiction of the design; it is filed as the first perf refinement (Refinement 3).

## L1 Ledger Audit

- **Currency**: two buffers, both present in the code, sized as the ledger says. Ring entry = `plan.entry_bytes`, entries = `plan.ring_entries = RING_DEPTH·max(R, W)` (the ledger records the `max(R, W)` form). Scratchpad = `plan.scratch_bytes = R·2·32·4 = 256 B`. This pass changed no buffer.
- **Capacity vs live set**: ring capacity 2 entries = live set (one being assembled, one draining). The entry spans token × column (both simultaneously live in an entry), streams over units and V. Index slots: 2 per reader thread (current + in flight). Neither over nor under.
- **Page format vs DEST**: no DEST in this op. Both buffers are DM-only byte payloads with `data_format_metadata` unset, which is legal for DM-only rings and scratchpads. No finding.
- **Disjoint lifetimes**: none; the two buffers are concurrent, and the ledger justifies it (index slot read while the entry fills).
- **Bounds / closed form**: every capacity symbol is bounded (`block_tokens ≤ 32`, `block_cols ≤ cols_cap`, `RING_DEPTH`, `INDEX_SLOTS`, `R ≤ 6 − W`). V, D, B, S enter no capacity. Total `≤ DFB_MAX_BYTES + 256 B`.
- **Per-core footprint**: `RING_DEPTH·max(R,W)·block_tokens·block_cols·32·esize + R·INDEX_SLOTS·BLOCK_TOKENS_MAX·4`. The first term scales with the token × column knobs and depth; the second is constant. Phase 0 worst case: 33,024 B.
- **Data-movement budget**: present and consistent with the code. Indices are read `num_col_chunks` times (counted). Table selected bytes are read once per token occurrence (counted; the resident-rows regime is a `deferred` row with a positive reason: R6/R7 make it result-neutral, and gather-direct serves every shape). Output written once. Cheapest-traffic split = the token split, implemented; the column split is engaged only to fill the grid and costs ≤ 4 B/token/chunk of index re-reads, with no extra table crossing.
- **Block-size defaults**: interleaved default = spread the units over the full grid, then take the coarsest block that fits. Met for RM output and for TILE output with ≥ 32 token groups × column chunks. **Not met** for small-S TILE output (see Design Conformance). Disposition: folded into Refinement 3 (no standalone ledger entry).

## Precision Baseline

`tests/ttnn/unit_tests/operations/embedding_gen/test_embedding_gen_precision_baseline.py`, bf16 table, standard-normal values, metrics against the float64 reference:

| Shape (indices × table, table→out) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP (bf16) | got/true median [p5, p95] |
|---|---|---|---|---|---|---|
| (1,1,1,32) × (1,1,64,128), RM→TILE | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1,32) × (1,1,8192,64), TILE→TILE | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (4,1,1,128) × (1,1,2048,512), TILE→RM | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (2,32) × (1,1,64,4096), RM→RM | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |

**Assessment**: bit-exact on all four layout pairs, as a byte-copy op must be. No scale signature (the ratio is exactly 1.0 everywhere).
**Recommended tolerances**: exact (every cell resolves to `exact` in `NUMERIC_RESOLVED`; `legacy_default (0.9999, 0.0)` holds).

## Verifier CLI Summary

Golden suite: `eval/eval_test_runner.sh eval/golden_tests/embedding_gen/` (8 sim workers), then `python3 -m eval.verify_supported … --queue op_requirements.md`. The report is copied to `verifier_report.json` in this directory.

- supported_pass: 96
- xfail_expected: 800
- invalid_skipped: 448 (bfloat8_b × both table layouts: INVALID for RM, unbuildable on quasar for TILE)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 6 (the 5 rejection tests X1–X5, all passed, and the empty `test_op_loose` placeholder)
- queue_gaps: 0

Every `xfail_expected` cell maps onto a refinement:
- float32 → Refinement 2
- L1 indices → Refinement 2
- `w_non_aligned` → Refinement 1
- rank 1 → Refinement 1
- table rank 2 → Refinement 1

bfloat8_b is recorded in the queue as an `Omitted:` line, unbuildable on quasar.

Op-carried + acceptance + baseline: 50 / 50 pass (25 acceptance, 21 plan/sub-mode/session, 4 precision).

## Recommendations

- **Queue shape**: two generality refinements, because the whole TARGET gap is five values, all knob-turns. Refinement 1 (shape surface, kernel clipping) is the only one with kernel work. A probe showed Refinement 2 (float32, L1 indices) needs zero kernel changes. Then three perf refinements, all gated on structural counts: grid fill for small-S TILE output, blocks per core / thread plan, RM-row read coalescing.
- **No timed measurement was possible here** (ttsim, no clock). Every perf gain in the queue is structural, and the time effect is unconfirmed until `ttsim-perf` or hardware. In particular, the thread-plan lever (Refinement 4) goes against the target page's default for data-movement-bound ops and should be confirmed on a timed backend before it is kept.
- **Grid-sync lamps not queued** (no structural headroom to gate on): RM output with one token per core (`(1,32)` RM: 32 cores × 1 token, `block_tokens` floor), and the column split at small widths (Qwen `(1,1)` RM: 4 cores × 64 B). Both trade per-core fixed cost against parallelism, which only a timed backend can settle.
- **Resident rows** (padding row / BINARY table in L1): remains deferred. R6/R7 make it result-neutral, GUIDANCE asks for nothing, and only duplicate-row reads would be saved.
- **Spec open items** (not defects): `special_values` and `freedoms` are UNRESOLVED; NaN/Inf move unchanged by construction (byte copy).
