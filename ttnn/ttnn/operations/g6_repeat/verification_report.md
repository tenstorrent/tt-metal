# Verification Report: g6_repeat

Board: Blackhole, 110-core compute grid (W = 9600 spreads over all 110). Golden results and the
CLI report: `ttnn/ttnn/operations/g6_repeat/golden_results/`.

## Environment blocker (host, not the op)

`libtt_metal.so` has RUNPATH `/opt/openmpi-v5.0.7-ulfm/lib`, and that directory no longer exists on
this host (every slot's build is affected). The loader falls back to the system Open MPI 4.1.2,
which lacks the ULFM symbols `MPIX_Comm_revoke` / `MPIX_Comm_shrink`, so `import ttnn` fails. The
orchestrator's own precompile probe hit the same failure (`pipeline_logs/precompile_support.txt`).

Every device run in this pass used an LD_PRELOAD stub that defines those two symbols and aborts if
either is called: `generated/mpix_stub/libmpix_stub.so`, gitignored and not committed. A
single-device run never calls them. **Later phases need the same preload, or the ULFM install
restored**, or no device test will import.

## Code Review

Reviewed `g6_repeat.py`, `g6_repeat_program_descriptor.py` and `kernels/g6_repeat_dataflow.cpp`
against `op_design.md` (regime 1, `column_split_row_broadcast`).

### Fixes

| # | Finding | Fix |
|---|---------|-----|
| 1 | **DRY.** The kernel re-derived `face_row_of_faces_bytes = face_cols * face_bytes`, the same quantity the host already passes as CT arg `row0_faces_bytes`. That is one value with two definitions. | The face-row stride in the replicate now uses `row0_faces_bytes`, so there is one source. Re-verified: acceptance 40/40, golden 5/5 supported cells, and the zone build. |
| 2 | **BB-21.** The final `noc_async_write_barrier()` (the kernel's last blocking wait) sat outside every profiler zone. | It is wrapped in `DeviceZoneScopedN("write_barrier")`. This compiles out of the plain build. The zone build (`--profile`) compiles and passes on all 5 acceptance shapes. |
| 3 | A hard-coded L1 NoC alignment of 16 on the host restated a HAL constant. | Replaced by `ttnn.get_l1_alignment()`, which returns 16 on this board. |
| 4 | The acceptance NaN-padding case could never pass: it poisoned through `ttnn.fill_implicit_tile_padding(x, nan)`, which lands `0x7F80` (+inf), not NaN. That fails the test's own precondition before the op runs. | Padding is now poisoned through a `(1,1,32,W)` tensor whose rows 1..31 hold the poison, viewed as `(1,1,1,W)` over the same buffer (the test asserts the buffer is shared). Poisons: 1000.0, `0x7FC1` and `0xFFC5`, at (W, R) = (256, 16), (64, 17) and (96, 31). All 9 combinations pass bit-exact. |
| 5 | `l1_ledger.md` named `out.buffer_page_size()` as the page-size source, but the code uses `x.buffer_aligned_page_size()`. | Ledger corrected. Same value (2048 B). |

Fixes 2–5 were in the working tree, uncommitted, left by an interrupted earlier attempt of this
verifier phase. I re-reviewed each against the code, re-ran its test, and kept it. Fix 1 is new in
this pass.

### Reviewed and correct as written

- **Kernel API.**
  - `void kernel_main()` and `api/dataflow/dataflow_api.h`.
  - `TensorAccessor` + `TensorAccessorArgs`, with CT args appended last on the host.
  - `noc_async_write_page(id, accessor, l1)` takes its page size from the accessor (`dataflow_api.h:1418`).
  - There is no CB handshake to balance: `cb_repeat_tiles` is single-thread workspace (0 pushes, 0 pops).
- **Load.** `row0_faces_bytes` (1024 B) is read from page offset 0 into the slot base, so both ends
  are 0 mod 64 (Blackhole DRAM-read alignment). The block's `block_extent` reads are issued, then
  one barrier.
- **Replicate.** It starts strictly after the load barrier (risk R-A).
  - Per tile and face column it calls `dataflow_kernel_lib::set_read_state<face_row_bytes>` once,
    then `rows_to_fill - 1` `read_with_state` issues.
  - One `noc_async_read_barrier()` per block. It waits on the same `noc_index` the helper's bare
    `Noc` uses.
  - Rows 16..30 land in faces 2/3 (R = 17 and R = 31 pass).
- **Store.** Each out page is written once, whole. `noc_async_writes_flushed()` runs before a slot
  set is reused (exercised by `test_g6_repeat_multi_block`). `noc_async_write_barrier()` runs
  before return (BB-03).
- **Helper usage.** The replicate is the only block op with a matching `kernel_lib` helper, and it
  uses it. The design's rejected-helper table holds:
  - `noc_async_read_page` has no size argument, so it cannot do the partial read.
  - The tilize/untilize stick helpers target row-major sticks, not TILE pages.
  - No compute kernel exists, so there are no compute helpers to use.
- **Program cache (BB-07).**
  - The descriptor cache key covers shape, reps, dtype, layout, tile/face shape, memory layout,
    buffer type, device id, grid and all four knobs.
  - A hit only rewrites the two common-RT buffer addresses.
  - `generic_op`'s `compute_program_descriptor_hash` hashes RT-arg *counts*, not values
    (`generic_op_device_operation.cpp:78-140`), so the hit replays the cached program with the new
    addresses.
  - `repeat_determinism` (golden) and `test_g6_repeat_determinism` prove the refresh.

### Design conformance (binding dimensions)

| Dimension | Design | Implementation |
|---|---|---|
| Algorithm | x upper faces → out slot; in-place NoC-loopback replicate; whole-page store | Same |
| Pipeline / RISC ownership | One data-movement kernel on RISCV_1 / NOC_0, no compute kernel | Same (`-DCOMPILE_FOR_NCRISC -DNOC_INDEX=0` in the build line) |
| Work distribution | `num_cores_to_corerangeset(min(Wt, max_cores), grid, row_wise=True)` + `split_work_to_cores(row_wise=True)` | Same. p1: 8 cores, 1 tile each. W = 9600: 110 cores, 3 or 2 tiles each, `block_width_tiles = 3` |
| Inter-core | None | None |
| Fills the machine | Spread `out_tile_col` over `min(Wt, grid)` cores | Yes. Both dataflow halves are batched: all reads, then one barrier; all writes, then one barrier at kernel end |

### Blocking-model fidelity

- **Knobs are parameters with one source.**
  - `REPEAT_BUFFER_DEPTH`, `L1_BUDGET_BYTES`, `MAX_BLOCK_WIDTH_TILES` and `MAX_NUM_CORES` live in
    `g6_repeat_program_descriptor.py`.
  - `block_width_tiles`, the CB `total_size`, `num_cores` and the kernel trip counts all derive
    from them.
  - Tile and face byte sizes are CT args computed from `x.tile` and `x.element_size()`.
  - After fix 1, no derived quantity is defined twice.
- **No collapsed knob.** The CB scales with `block_width_tiles`, a block factor clamped by L1. No
  CB scales with W.
- **The split is turned on both count and size.** `out_tile_col` is spread across the grid, and
  each core's block is its whole assignment (clamped by L1). `test_g6_repeat_multi_block` turns two
  knobs (2 cores, block of 3, ragged last block) and passes.
- **Expression.** At the scheduling boundary of each block op (load, replicate, store), every unit
  issues before one completion point per block. Nothing is acquired and retired per tile.
- **Axis accounting.** See the L1 ledger audit below.

### Prompt rules (`eval/prompts/g6_repeat.txt`, `## Rules`)

| Rule | Status |
|---|---|
| New tensor; bit-identical logical elements; every out page written once and never read back; no delegated op; no DRAM intermediate; no value-changing path | **Holds.** One `generic_op` program (`device_num_programs = 1` in every golden row). Only NoC byte moves. One write per page. |
| TILE input: out independent of x's tile padding | **Holds.** Poisons 1000.0, `0x7FC1` and `0xFFC5` pass at p1, R = 17 and R = 31. The golden 1000.0 case passes. |
| Non-aligned value in SUPPORTED: handled natively | **Holds.** The only host-side tensor work is `allocate_tensor_on_device`. |
| More than one out tile: the core count is the blocking decision's | **Holds.** |
| OOM on a golden shape | n/a. Peak L1 is 2048 B per core at p1 (`device_l1_peak_bytes`). |
| BB-01 / 02 / 03 / 05 / 06 / 07 / 08 / 17 | **Hold.** One program; no conversions; final write barrier; L1 clamp at 1347776 B; determinism tested; descriptor cache; `plan_report()`; p1 is one row range (0,0)-(7,0). |
| BB-04 (every build flavour fits) | **Holds.** The zone build (`PROFILE_KERNEL=1`) compiles and runs on all 5 acceptance shapes. |
| BB-21 (every blocking wait zoned, role dump) | **Holds** after fix 2. `plan_report()` dumps the per-core role (`repeat`). The setup before the first block (arg reads, accessor construction) is unzoned. It holds no wait. |

Advisory (soft): out padding rows `R..31` carry x padding (rows `R..15` of faces 0/1) and stale L1
(faces 2/3 when R ≤ 16). The contract leaves out padding unspecified, so this is legal. It is not
deterministic across runs, though. A neighbour that reads whole tiles without masking would see
garbage.

Non-binding design drift: `op_design.md` names `out.buffer_page_size()` as the CB page size, while
the code uses `x.buffer_aligned_page_size()`. Same value, since out takes x's dtype, layout and tile.

## Registry Conformance

- **`INPUT_TAGGERS`** = `{"alignment": tag_alignment}`, with signature `(inputs, axes)`.
  - It tags the **output** shape, as the prompt requires.
  - `test_tag_alignment` covers all three values.
- **`SUPPORTED`** = `{dtype: [bfloat16], layout: [TILE], alignment: ["h_non_aligned"]}`, covering
  every axis the kernel gates on. **`EXCLUSIONS`** = `[]`.
- **`validate()`** is the first line of `g6_repeat()`. Its order:
  1. Contract checks, raising `ValueError`. Every message is prefixed `g6_repeat:`.
  2. SUPPORTED, raising `UnsupportedAxisValue`.
  3. EXCLUSIONS, raising `ExcludedCell`.

  Contract-before-SUPPORTED is deliberate. `reps[2] = 32` tags `tile_aligned`, and the spec
  demands a `ValueError` for it (risk R-J). `W % 32 != 0` lands as `UnsupportedAxisValue`, a
  `NotImplementedError` subclass (`_op_contract.py:23`), which the spec allows.
- The op file does **not** declare `INVALID`.
- No XPASS drift, so SUPPORTED needed no auto-fix.
- **INVALID audit** (`eval/golden_tests/g6_repeat/feature_spec.py`): `INVALID = []`. That is correct:
  - TARGET has no block-float dtype and no ROW_MAJOR layout, so the canonical bf8b + ROW_MAJOR
    entry has nothing to pair with.
  - There is no optional weight tensor and no cross-tensor coupling.

## L1 Ledger Audit

- **Currency.** The one declared CB, `cb_repeat_tiles` (index 0, `REPEAT_BUFFER_DEPTH * block_width_tiles` pages of the aligned page size), has its row.
  - Its size expression equals `CBDescriptor.total_size`.
  - The measured `device_l1_peak_bytes` of 2048 at p1 matches the ledger.
- **Capacity vs live set.**
  - Capacity equals the live set: one slot set is fully resident from its load until its writes flush.
  - Not over: the input CB and the replicate scratch were eliminated by landing x in the
    destination slot and replicating in place.
  - Not under: the CB spans `out_tile_col` with capacity `block_width_tiles`, and spans
    `row_in_tile` with a whole tile per slot.
  - It streams over `out_tile_row` and `out_batch`, both extent 1 by contract.
- **Page format vs DEST.** bfloat16 is a dataflow-only payload that never passes through DEST, so
  `fp32_dest_acc_en` does not apply.
- **Disjoint lifetimes.** There is a single CB, so there is nothing to share.
- **Bounds and closed form.**
  - `block_width_tiles ≤ floor(1347776 / (REPEAT_BUFFER_DEPTH * 2048)) = 658`, enforced on the host.
  - `tensor_out_wt_tiles` is unbounded, but it enters only trip counts.
  - Total: `L1_op = REPEAT_BUFFER_DEPTH * block_width_tiles * out_page_size ≤ 1347776 B`. It scales
    with `block_width_tiles` and `REPEAT_BUFFER_DEPTH` only.
- **Traffic budget.** Present and consistent with the code:
  - x: one 1024 B partial read per page.
  - out: one 2048 B write per page.
  - Cross-core traffic: 0.
  - Core-local loopback: `2*(R-1)*32` B per tile.
- **Cheapest-traffic split.** It is the one implemented (`out_tile_col`), and the ledger says so.
- **Block-size defaults.** Both held: spread across the full grid, then take the coarsest block.
  No departure needed justifying.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_repeat/test_g6_repeat_precision_baseline.py`, randn input,
seed 0.

- PCC: `comp_pcc`. Abs deltas: `comp_allclose`.
- Relative RMS: RMS(err) / std(ref).
- ULP: the max distance on the ordered bf16 bit line.
- Ratio: actual / expected over the non-zero reference elements.

| Shape (x → out) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio p5 / p50 / p95 |
|---|---|---|---|---|---|---|
| (1,1,1,256) → (1,1,16,256), p1 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,1,1,64) → (1,1,17,64) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,1,1,96) → (1,1,31,96) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,1,1,9600) → (1,1,5,9600) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |

**Assessment.** Bit-exact on every shape; the test also asserts int16 bit equality. Special values
(NaN payloads, ±inf, −0.0, denormals, ±max finite) are bit-exact in the golden `special_values`
case and in `test_g6_repeat_special_values`. There is no scale signature: the ratio is exactly 1
everywhere.

**Recommended tolerances.** Exact. `helpers.TOLERANCES[bfloat16] = ("exact", None)` holds.

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/g6_repeat/ ttnn/ttnn/operations/g6_repeat/golden_results`, then
`python3 -m eval.verify_supported ttnn/ttnn/operations/g6_repeat/golden_results ttnn.operations.g6_repeat`.

- supported_pass: 5
- xfail_expected: 0
- invalid_skipped: 0
- supported_fail: **1**. `test_op_loose[poisoned_padding_nan-…]`, category `other`. **A golden-harness defect, not the op.** Details below.
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 3. These are the `test_regression.py` numerics cases, outside the registry; all pass.

Acceptance + debug + precision directory: **40 / 40 pass**.

### The one `supported_fail` (escalated, not fixable in the op)

```
[poisoned_padding_nan] HARNESS PRECONDITION failed (not the op): the tile padding of x does not hold the poison nan
```

The failing step is `eval/golden_tests/g6_repeat/helpers.py:_poison_tile_padding`, which runs
**before** `g6_repeat` is dispatched. A locked probe
(`tests/ttnn/unit_tests/operations/g6_repeat/probes/probe_003.py`) shows two things:

- `ttnn.fill_implicit_tile_padding(x, nan)` lands exactly one bit pattern, `0x7F80` (+inf), in
  every padding element on this build.
- `g6_repeat` on that same tensor is bit-exact.

What follows from that:

- **The op passes the property this case tests.** NaN payloads `0x7FC1` / `0xFFC5` in x's padding,
  placed through a same-buffer view, give bit-exact output at p1, R = 17 and R = 31.
- **It is not in EXCLUSIONS or removed from SUPPORTED.** That would refuse the production cell,
  which works.
- **It is not filed as a refinement.** No change to the op can move it.
- **Owner: the golden suite** (read-only for agents). Either of these fixes it:
  - (a) `_poison_tile_padding` builds a `(…, 32·k, W)` tile tensor with poisoned rows and takes a
    `ttnn.reshape(t, logical, padded)` view, the mechanism the acceptance test now uses.
  - (b) `fill_pad` learns to write a NaN fill without passing it through DEST.

## Performance observations (not refinements)

Device kernel duration, measured on this board:

| Case | Tiles (largest tensor) | Cores | Plain build, golden rows (ns) | Zone build, `--profile` (ns) |
|---|---|---|---|---|
| p1 (W=256, R=16) | 8 | 8 | 1756–1768 | 1622 |
| W=32, R=1 | 1 | 1 | — | 805 |
| W=64, R=17 | 2 | 2 | — | 1562 |
| W=96, R=31 | 3 | 3 | — | 2043 |
| W=9600, R=5 | 300 | 110 | — | 5324 |

- **p1 against its goal** (`LOOSE_CASES[0]`: < 9.46 µs and ≤ 1.35 × 0.02 µs):
  - The "faster than the incumbent" half is met at ~1.76 µs, about 5.4× under the 9.46 µs stock figure.
  - The 27 ns half is a bytes-over-bandwidth bound that ignores latency, and it is not physically
    reachable. Each core must pay one DRAM read round trip and then one DRAM write ack, in series,
    because the store depends on the load. That floor alone is hundreds of ns.
- **No perf refinement can be filed.** `eval/prompts/perf_rules.txt`: "Measure only cells of more
  than 32 tiles; smaller cells only have to pass the golden suite, and this overrides any other
  instruction that would have you measure or target one."
  - p1 is 8 tiles, and every golden INPUT and LOOSE_CASE is that same shape.
  - So the mandatory perf target is out of bounds for perf work, and there is no other golden cell
    to free-select.
- **Large W (not production, not golden).** At 300 tiles the op runs 5.3 µs. The DRAM-byte bound is
  ~0.9 MB at ~0.5 TB/s ≈ 1.8 µs, so there is headroom. The design's lamps L4 (writes on NOC_1, a
  reader/writer split) and L6 (write only faces 0/1 when R ≤ 16) are the levers for that regime.
  File them only if a production case above 32 tiles enters `LOOSE_CASES`.

## Recommendations

1. **Infra owner:** restore `/opt/openmpi-v5.0.7-ulfm` on this host, or keep exporting
   `LD_PRELOAD=<repo>/generated/mpix_stub/libmpix_stub.so` for every device run. Without one of the
   two, `import ttnn` fails in every slot.
2. **Golden-suite owner:** fix the `poisoned_padding_nan` precondition with option (a) or (b) above.
   It is the only red cell, and the op passes the property it tests.
3. **The refinement queue is empty by construction** (see `op_requirements.md`):
   - TARGET == SUPPORTED on every axis.
   - The only perf-flagged case is under the perf-rules tile floor.

   Re-open it if `feature_spec.TARGET` grows, or if a production case above 32 tiles is added. For
   a placement axis, the design already classifies the cases: WIDTH_SHARDED out is a knob-turn, and
   an `out_tile_row` cut is a scheme-change.
4. **Out-padding determinism (advisory).** If a neighbour is ever found reading whole out tiles
   unmasked, replicate into rows `R..31` as well. That costs a few more loopback issues and no new
   pages.
