# Verification Report: g6_sharded_to_interleaved

Board: Blackhole (11 × 10 worker grid), one chip. Verifier pass over the Phase 0 implementation (commit `5058b9ee480`).

## Code Review

What was reviewed: `g6_sharded_to_interleaved.py`, `g6_sharded_to_interleaved_program_descriptor.py`, `kernels/g6_sharded_to_interleaved_writer.cpp`, `l1_ledger.md`, the acceptance tests and `op_design.md`.

Fixes applied:
- **DRY**: the op file restated the tile edge as a literal `32` in the shard-shape check. It now imports `TILE` from the program descriptor, so the tile size has one source of truth.
- **Stale tests removed**: `test_..._impl_debug.py`, `test_..._perf.py` and `precision_matrix_results.md` were leftovers from an earlier, interrupted run. They imported APIs that don't exist in this implementation (`_dispatch`, `describe_plan`, `Knobs(num_write_lanes=…, bank_stagger=…)`), so collecting them broke the acceptance directory. Deleted. `test_..._precision_baseline.py` was rewritten (see below) and `test_..._perf.py` re-written as a lean harness for the current `Knobs` (`blocks_per_shard`, `writer_lanes`), with kernel-ns (`--profile`) and trace-replay ns/op tests.
- **L1 ledger**: added the note that both implemented knobs add 0 B of L1 at every setting (they change only RT args / kernel count).

Checked and found conformant (no change needed):
- **Prompt rules (MUST / MUST NOT)**, all in scope at Phase 0:
  - Every output page is written every call, from the same tile row/col. The `dirty_destination` regression passes.
  - The output is newly allocated and the input stays intact. The `input_intact` check and the acceptance test's address/placement checks pass.
  - The op uses its own `generic_op` program (one dispatch, SPMD on a mesh via `make_spmd_mesh_program_descriptor`). There is no host round-trip and no layout or reshard op.
  - Each tile crosses the NoC once. The shard is read through the zero-copy CB (`cb_descriptor_from_sharded_tensor`), never through a TensorAccessor, and then one `noc_async_write_page` goes to its page.
  - Grid, shard shape and orientation come from the tensor (`corerange_to_cores(grid, row_wise=…)`). The column-major / short-last-shard "other spec" cases pass.
  - There are no extra taggers and no EXCLUSIONS.
  - The number of cores is the blocking choice (owner-writes on the shard grid: 32 cores for p2, 103 for p1).
- **Model-wide rules**:
  - BB-03: one write barrier before return.
  - BB-07: kernels built once per key. `custom_program_hash` covers the orientation and range order, which only change RT args. The `cache_order` regression passes.
  - BB-08: `last_plan()`.
  - BB-16: the trace replay test passes.
  - BB-17: kernel core ranges = the shard grid's own rectangles.
  - BB-21: zones `S2I_STORE_BLOCK` / `S2I_DRAIN`, and `roles()`.
- **Design conformance**:
  - Algorithm and topology match the owner-writes design: no reader or compute kernel, one writer lane on BRISC/NoC1.
  - Work distribution fills the shard grid, which is the minimum-traffic split. No idle silicon would help, because a non-owner core would need a second crossing.
  - The writer is batched: every write of the block is issued, then one barrier.
- **Blocking-model fidelity**:
  - `blocks_per_shard` and `writer_lanes` live in the frozen `Knobs` dataclass, which is the single source. Every RT arg (`num_blocks`, `block_rows`, `block_cols`, `first_page`, `src_first_slot`) is derived from it on the host. Knobs are part of the cache key.
  - The slot stride (`shard_col_tiles`) and page stride (`tensor_col_tiles`) are CT args defined once.
  - No CB is sized to a whole-op dimension. The only CB aliases the input shard.
  - Expression is fine: the per-tile loop sits inside one block-scoped phase whose live set is the resident shard, with no per-tile sync. The one-page-per-write granularity is a mechanism cap (consecutive interleaved pages live in different banks).
- **Helpers**: none applies. Every `kernel_lib` helper is either a compute helper (unpack → DEST → pack, which must not touch the bf8b bytes) or an L1→L1 local-copy/mcast helper. This op is a one-to-one L1 → interleaved page write.
- **API usage**: `void kernel_main()`, `api/dataflow/dataflow_api.h`, `TensorAccessor` + `TensorAccessorArgs`, non-deprecated `noc_async_write_page<AddrGen>`.

Advisory (not fixed; no correctness impact):
- `bind_call` builds a fresh `ProgramDescriptor` + sharded CB descriptor on every eager call. Kernels and RT args are cached and only the address is patched, so this is a small Python object build, not a program rebuild, and it is absent under trace replay. It could be removed by caching the descriptor and rebinding the CB buffer if host-eager overhead ever matters.

## Registry Conformance
- `INPUT_TAGGERS` (`tag_dtype`, `tag_layout`, `tag_in_scheme`, `tag_buffer`, all `(inputs, axes)`), `SUPPORTED`, `EXCLUSIONS = []` and `validate()` are present and correctly wired. `validate()` runs structural ValueErrors first, then SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`). It is the first line of the entry point.
- The op file does **not** declare `INVALID`.
- No SUPPORTED auto-fix: no XPASS drift. The Phase 0 SUPPORTED (bf16 / TILE / width_sharded / dram) is exactly the prompt's Phase 0 contract.
- INVALID audit (`feature_spec.py`): `INVALID = []`. This is correct because TARGET is TILE-only, so the canonical bf8b + ROW_MAJOR entry does not arise. There are no cross-tensor entries, and there is no weight operand, so no canonicalization cells are needed.

## L1 Ledger Audit
- **Ledger currency**: one CB (`cb_input_shard`, index 0), which matches the code. Its size is set by `cb_descriptor_from_sharded_tensor` = the shard, globally allocated on the input buffer, so it adds 0 B of new L1.
- **Size vs live set**: capacity = live set (the whole resident shard is the block). Nothing is flagged.
- **Page format vs DEST**: n/a. This is a dataflow-only payload with no compute kernel; the input dtype format passes through verbatim.
- **Disjoint lifetimes**: n/a (single CB).
- **Bounds / closed form**: total op CB L1 = 0. The only L1 term is the output itself when `buffer == l1`: `ceil(1024 / num_l1_banks) × 1088 B` ≈ 10 KB per bank for p1, far below 1 347 776 B.
- **Block-size defaults**: the sharded default held (one block = the whole resident shard).
- **Data-movement budget**: present and consistent. 0 DRAM reads, one NoC crossing per tile. The cheapest-traffic split (owner-writes) is the one implemented.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_sharded_to_interleaved/test_g6_sharded_to_interleaved_precision_baseline.py`, run through the public entry point. The p1 rows ran with SUPPORTED widened by `monkeypatch` (pre-claim measurement for Refinement 1).

| Case | Shape / dtype / scheme → dest | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | Max ULP | Bit mismatches | got/true ratio (median, p5, p95) |
|------|-------------------------------|-----|-------------|--------------|-------------|---------|----------------|-----------------------------------|
| p2 production | (1,5120) bf16 width (32,160) row-major → DRAM | 1.000000 | 0 | 0 | 0 | 0 | 0 | 1.0, 1.0, 1.0 |
| p2 other spec | (1,5120) bf16 width (32,96) col-major, short last → DRAM | 1.000000 | 0 | 0 | 0 | 0 | 0 | 1.0, 1.0, 1.0 |
| p1 production (pre-claim) | (1,128,256,32) bf8b height (320,32), 2 ranges → L1 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 1.0, 1.0, 1.0 |
| p1 other spec (pre-claim) | (1,128,256,32) bf8b height (352,32) col-major, last of 1 → L1 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 1.0, 1.0, 1.0 |

**Assessment**: bit-exact on every case. The bf8b case is compared against the host's unpack of the device input, so the shared exponents are preserved. There is no scale or structural signature (ratio identically 1).
**Recommended tolerances**: exact (the `helpers.TOLERANCES` entries `("exact", None)` hold).

## Verifier CLI Summary
(`eval.verify_supported` over `eval/eval_test_runner.sh eval/golden_tests/g6_sharded_to_interleaved/`, 17 cells; artifact `verifier_report.json` in this directory)
- supported_pass: 13
- xfail_expected: 4 (p1 production INPUTS cell, p1 perf-focus loose case, p1 other-spec loose case, p2 → L1 loose case). All four are queued in Refinement 1.
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

Acceptance directory: 9 passed, 3 xfailed (the three acceptance cases outside SUPPORTED). Trace replay test: pass.

## Performance baseline (device, Blackhole)

`test_g6_sharded_to_interleaved_perf.py`. Kernel ns = DEVICE KERNEL DURATION, median of calls 3–12 of 12 (`--profile`). Replay = host clock / op over 50 replays of a 16-call trace.

| Case | Knobs | Cores | Kernel ns (median) | FW ns (median) | Replay ns/op |
|------|-------|-------|--------------------|----------------|--------------|
| p2 (1,5120) bf16 → DRAM | default (1 lane) | 32 | 1429.5 | 1975 | 2819–2821 |
| p2 | `writer_lanes=2` | 32 | 1319.0 | 1858.5 | 2946–2950 |
| p1 (1,128,256,32) bf8b → L1 (perf focus, pre-claim) | default (1 lane) | 103 | 1939.0 | 2873.5 | 5954–5960 |
| p1 | `writer_lanes=2` | 103 | 1564.5 | 2515 | 6161–6175 |

Reading:
- p1's kernel is already below the 2.05 µs stock incumbent at Phase 0.
- Kernel and replay pull in opposite directions for the second lane: −19% kernel on p1, but +3.5% replay, because the second kernel adds binary/launch dispatch cost.
- Replay per op (≈ 6 µs) is about 3× the p1 kernel time, so per-call dispatch/launch overhead, not NoC issue, dominates what a traced model pays.

## Recommendations
- **Queue**: Refinement 1 claims the three remaining TARGET values (bfloat8_b, height_sharded, l1). This is a SUPPORTED-list edit, measured bit-exact above. Refinements 2 and 3 are perf passes on the flagged p1 case:
  - Refinement 2: kernel-side issue levers.
  - Refinement 3: per-call dispatch/launch cost.

  Both are gated on kernel ns **and** trace-replay ns/op, since the model replays traces and the two measures disagree on the lane knob.
- **Mesh**: the four-chip p2 case cannot be run on this one-chip board. The SPMD path (`generic_op` → `make_spmd_mesh_program_descriptor`, output allocated on `input_tensor.device()`) is the mechanism that covers it. Confirming it on a mesh is a finding for a multi-chip board, not a refinement.
- **No L1 pressure**: the op allocates 0 B of CBs, and the only L1 cost is the interleaved output when L1 is named.
- **Precision**: no concern; the op is exact by construction (raw page copy).
