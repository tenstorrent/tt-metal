# Verification Report: padded_slice

Target: Quasar (`quasar`) on the ttsim simulator (untimed). Specification: `eval/specs/padded_slice/spec.py`
(sha256 `e8d02fdd539a…`). Phase 0 = PHASE0 of the specification.

## Code Review

Implementation reviewed: `padded_slice.py` (validation, entry point), `padded_slice_program_artifacts.py`
(Metal 2.0 `ProgramSpec`), `kernels/padded_slice_reader.cpp` (the single DM kernel `gather`).

**Fixed**
- **Element size derived from the page size** (`padded_slice_program_artifacts.py`): `elem_bytes` was
  `input_tensor.buffer_page_size() // C`, which is right only while a page is one row-major row. A TILE
  page is a tile, so the same line would silently compute a wrong `src_offset_bytes` / `row_bytes` the moment
  Refinement 1 adds `TILE_LAYOUT`. Now `input_tensor.element_size()` (from the dtype); the unused `in_c`
  went with it. Golden suite and all unit tests re-run green after the change.

**Checked, no change needed**
- Kernel syntax `void kernel_main()`, include paths `api/dataflow/...`, `TensorAccessor` for the input,
  `LocalTensorAccessor<uint8_t>` for the own shard: correct.
- Ring protocol (target page): the kernel uses no ring; every read is `NocOptions::TXN_ID` with
  `{.trid = 0}`, the barrier is on the same id, and the last statement hands the tag back
  (`noc_async_read_set_trid(0, noc.get_noc_id())`). The one-device-session test (tt-metal `to_layout`
  before the op, `to_layout` of the op's own output and another program after) passes.
- Sync counts: no DFB, so no push/pop pairing to check; one barrier per hart per block covers every read
  that hart issued in the block.
- Helper usage: no compute phase exists (spec: "moves values and computes nothing"), so no compute helper
  applies; no byte goes to more than one core, so the mcast helper does not apply. Raw `Noc` /
  `TensorAccessor` use is the dataflow layer itself — correct for this op.
- Broadcast efficiency: not applicable (no broadcast operand).
- `validate()` is the first statement of the public entry point; malformed calls (X1–X3) are checked
  before any axis is derived, `canonical_to` raises X5/X6, and the axis gate refuses X4 by name.

**Deferred**: nothing architectural. Two perf lamps of the design (NoC per hart, overlapped padded
zeroing) and regime D are queued as perf refinements (Refinements 3, 4).

### Specification rules

| Rule | Applies now | Status |
|------|-------------|--------|
| R1 move unchanged | yes | exact (byte copy, never through DEST); probe also exact for float32 |
| R2 rank 4 → `[1,1,n*h*w,width]` | yes | passes every `rows` cell; other rank refused by the `rank` axis (or X1 when the bounds do not match the rank) |
| R3 padded rows | no (`L1/height/padded` outside PHASE0) | re-arms in Refinement 2; probe shows the shard tail is stale today, so `zero_block` is required |
| R4 row-major result, logical elements only | output RM: yes; TILE input: no | re-arms in Refinement 1 |
| R5 result in the given config | yes | `HOOKS["check"]` passes on every supported cell; output allocated from a `TensorSpec` carrying the config object unchanged; shard order from `corerange_to_cores(grid, row_wise = orientation == ROW_MAJOR)` |
| R6 `canonical_to`, refuse non-height/non-L1/no-shard-spec | yes | X4, X5 pass; other words refused by the `to` axis |
| R7 step None / ones / other | ones yes; None not yet | "other" refused by name; None re-arms in Refinement 2 |
| R8 lists or tuples | yes | loose case `bounds_as_lists` passes |
| R9 input placement via `memory.memory_axes` | yes | gated on `memory_config` / `sharding` |
| R10 any number of shard cores | yes (review) | core set = the shard grid, never the device grid; tested at 1, 2, 27 and 32 cores |
| R11 `NotImplementedError` outside SUPPORTED | yes | 533/533 outside-SUPPORTED cells xfail through `validate()` |

GUIDANCE: followed throughout (no compute; each core fetches its own rows into its own shard; core order
from the shard spec). No advisory.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from the specification; no op-local axis function.
- `SUPPORTED` lists all nine axes and equals PHASE0 (`python3 -m eval.validate_contract padded_slice
  --phase0 --text`: PASSED, 0 errors; 6 UNRESOLVED items are the specification's own open items, not defects).
- `EXCLUSIONS = []`; `validate()` checks SUPPORTED per axis, then EXCLUSIONS, raising `UnsupportedAxisValue`
  / `ExcludedCell`.
- The op file declares no `INVALID`.
- No SUPPORTED auto-fix was applied (0 `xpass_drift`).
- **INVALID audit (specification)**: `INVALID = []`. Correct: TARGET has no block-float dtype (so no
  bf8b+ROW_MAJOR entry is due; Quasar has no bf8b path anyway), the op is not norm-like, and no TARGET cell
  is structurally impossible (a tiled input with a row-major result is the whole point of R4).

## Design Conformance

- **Algorithm / topology**: regime A as designed — each core of the caller's shard grid reads its own rows
  from DRAM straight into its own shard (`LocalTensorAccessor` destination), one tagged read per row; no
  DFB, no compute kernel, no writer.
- **Work distribution**: the core set is `memory_config.shard_spec.grid` (R10), not the device grid — the
  caller pins it, so "filling the machine" means filling the caller's cores; each core's harts split its
  rows in contiguous sub-ranges. The golden configs use up to 32 cores.
- **Thread plan**: read off the host program — one named constant `GATHER_THREADS = 2` sets the kernel's
  `num_threads` and its CT arg `reader_threads`; the kernel sizes its loops from that count and
  `get_my_thread_id()` (not `get_num_threads()`), so a missing hart shows as missing rows. Plan
  (R, C, W) = (2, 0, 0) matches the design. The design states the op as bound by data movement
  (transaction issue on a caller-pinned core set) with the reason for R = 2 over the page's (1, 1, 1)
  default (no compute or writer exists; harts are the lever for issue rate). Tests present and passing:
  every hart writes its rows (sentinel-filled shard, address reuse asserted; R ∈ {1, 2, 3, 6} at 1, 2 and
  32 cores), the built kernel's `num_threads == R`, and one device session with tt-metal programs before
  and after. **Plan vs. alternatives not measured**: this backend has no timing model and
  `pipeline_logs/model_cycles.jsonl` does not exist; the design's lamp 1 (R ∈ {1, 2, 4, 6}) is the
  structural lever of Refinement 3.
- **Blocking-model fidelity**: `MAX_BLOCK_ROWS` (0 = whole shard) is the single source of `block_rows`
  (common RT arg); `GATHER_THREADS` is the single source of R (num_threads + CT arg). No literal restates
  either. No collapsed knob: the per-core loop's block is the whole shard (the sharded default), and the
  test `test_block_knob` turns it (1, 5, 7, 64 rows with R ∈ {2, 3}).
- **Expression**: per hart per block one `(n, h, w)` decomposition, one tagged read per row (the interleaved
  page is the mechanism's transfer quantum), and **one** barrier per hart per block — no per-row
  completion boundary.
- **Axis accounting**: the shard region spans `out_row` (→ `shard_h`) and `channel` (→ width); harts stream
  over it. Consistent with the ledger.

## L1 Ledger Audit

- **Currency**: the program declares no DFB and no scratchpad; the ledger's single row is the
  caller-owned output region, sized `shard_h × out_row_stride` (`output.buffer_aligned_page_size()`),
  matching the code.
- **Capacity vs live set**: nothing op-owned to flag; the output region is the result (fully live).
- **Page format vs DEST**: dataflow-only bytes, never through DEST — audit does not apply.
- **Disjoint lifetimes**: no buffer pairs exist.
- **Bounds / closed form**: `op_owned = 0 B`; `output_region = shard_h × out_row_stride`, bounded by the
  allocator (`allocate_tensor_on_device` refuses a shard that does not fit). Symbol table complete.
- **Block-size defaults**: sharded → one block = the whole resident shard (`MAX_BLOCK_ROWS = 0`): held.
- **Data-movement budget**: present and consistent with the code — input block bytes cross DRAM once,
  0 cross-core bytes, 1 read per output row. The cheapest-traffic split (owner-computes into own shard) is
  the one implemented; the alternatives (staged ring E, helper cores F, channel split G, coalescing H) are
  rejected or deferred with positive reasons.
- **Footprint total**: op-owned 0 B, independent of R and `block_rows`; Refinement 4 (regime D) will add the
  first op-owned ring and must ledger it.

## Precision Baseline

`tests/ttnn/unit_tests/operations/padded_slice/test_padded_slice_precision_baseline.py` (bfloat16, RM,
DRAM → L1 height shards on the runner's core count; results also in
`precision_baseline_results/*.json` beside the test).

| Shape (block) | Rows | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio median [p5, p95] |
|---------------|------|-----|-------------|--------------|------------------|---------|-------------------------------|
| (1,4,4,16) → (1,2,4,16) | 8 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1,64,32,32) → (1,32,32,32) | 1,024 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1,17,50,64) → [3:12, 7:40] | 297 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1,115,115,32) → (1,59,115,32) | 6,785 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |

**Assessment**: bit-exact on every shape (`torch.equal`), as a movement op must be (R1; `NUMERIC_RESOLVED`
is `exact` for bfloat16 and float32). No scale signature: the ratio is exactly 1.0 everywhere.
**Recommended tolerances**: exact (bfloat16 and float32; the resolved numeric entry is `exact`).

Additional probe (not a gate; `tests/ttnn/unit_tests/operations/padded_slice/probes/`): regime A run past
`validate()` is already exact for float32 rows, an L1-interleaved input, `channels` blocks with a 2-byte
(bf16, `start[3]=1`) and a 32-byte (fp32) mid-page offset, and a wide `channels` block (c = 2048 of 4096).
A padded config (32 → 64 wide) leaves the shard's stale tail (32,768 non-zero tail elements over a
sentinel), confirming that regime B's `zero_block` is required.

## Verifier CLI Summary

Results: `golden_results/` (golden run of 551 tests, 8 sim workers; `verifier_report.json` from
`python3 -m eval.verify_supported`).

- supported_pass: 12 (11 PHASE0 cells + loose case `bounds_as_lists`)
- xfail_expected: 533
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- rejections (no axes sidecar): 6 / 6 passed (X1–X6)
- queue coverage (`--queue op_requirements.md`): 6 values in TARGET − SUPPORTED, 0 gaps

The 533 xfail cells are the six TARGET values outside PHASE0 — `dtype` float32, `layout` TILE,
`memory_config` L1, `step` None, `to` L1/height/padded, `sliced_region` channels — and their products; every
one is named in Refinement 1 or 2.

## Recommendations

- **Queue order**: Refinement 1 (TILE, native segment gather) → Refinement 2 (padded + channels + float32 +
  L1 input + step None; after it SUPPORTED = TARGET) → Refinement 3 (perf: issue parallelism on the traced
  1–2-core configs + overlapped padded zeroing) → Refinement 4 (perf: whole-tile reads, regime D). No
  scheme-change exists in this op (no dependent axis, no reuse-shared operand); regime D is the only
  restructure and stands alone.
- **No perf-flagged loose case** in the specification: Refinement 3's region (the traced few-core configs)
  is free-selected. On this untimed backend both perf refinements are gated on structural counts; their
  time gains stay unconfirmed until `ttsim-perf` or hardware measures them.
- **Thread-plan lamp**: whether 2 harts beat 1 on 32-core configs with ≤ 8 rows per core is unknown here;
  Refinement 3 may make R a per-call rule — it must remain one host-side statement.
- **Unspecified corners** (spec DECISIONS, not queued): configs whose shards hold more or fewer rows than
  the result, negative indices (refused here with `ValueError`), empty blocks (refused with
  `NotImplementedError`), whole-tensor blocks.
- **L1**: op-owned footprint is 0 B; no OOM risk until Refinement 4 adds a staging ring. The traced
  single-core stem shard (434 KB) is the caller's and fits the 4 MiB cluster L1.
