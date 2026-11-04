# Verification Report: nomic_qkv_rotary_heads

Verified 2026-11-03 on a Blackhole p300c (11 x 10 compute grid, AICLK 1350 MHz per the profiler log, so the
feature-spec perf references apply unscaled). Commit under review: `a42176ccb5e` (implementer, regime
`mcast2d_fused_rotary`).

## Code Review

### Fixes applied

1. **Blocking-model DRY: `head_tiles` restated in compute.** `nomic_qkv_rotary_heads_compute.cpp` declared
   `constexpr uint32_t head_tiles = 2;`, a second copy of the host's `HEAD_TILES`. That value sets the DEST slot of C
   and S and the q/k subblock width check. It is now the named CT arg `head_tiles`, the same one the writer already
   took.
2. **Blocking-model DRY: the table block size computed three times.** The host sized `cb_rotary_table` as
   `ROTARY_TABLE_DEPTH * 2 * block_m`, and reader and compute each recomputed `2 * block_m_tiles` for their
   reserve/push and wait/pop counts. There is now one host constant, `TABLE_TILES_PER_ROW = 2` (C and S per position
   tile row). The descriptor derives `table_block_tiles` from it once and passes that to reader and compute. The CB
   capacity and `_cb_bytes` use the same constant. The kernels keep a `static_assert(table_block_tiles == 2 *
   block_m_tiles)` because they index `2*row` / `2*row+1` (the table page order).
3. **`_placement()` exception handling.** There were two `except` arms (`SupportRefusal`, then `Exception`) with the
   same body. They are now one arm, and the unused `SupportRefusal` import is gone.
4. **Ledger currency** (`l1_ledger.md`): the `cb_rotary_table` expression now uses `TABLE_TILES_PER_ROW`. The
   ones-tile wording now matches the code (NoC `async_write_zeros` plus 32 bf16 stores, not `fill_l1_range`). A
   "Verifier audit" section is appended (see the L1 Ledger Audit below).

After the fixes: acceptance 35/35, golden unchanged (68 pass), precision baseline unchanged.

### Reviewed and found conformant (no change)

- **Prompt `## Rules` (MUST):**
  - One `ttnn.generic_op` per call, and no ttnn op sequenced on the path.
  - Kernels named `nomic_qkv_rotary_heads_{reader,compute,writer}.cpp`.
  - io tensors passed as `[x, weights, tables, q, k, v]`.
  - No intermediate in DRAM or in an L1 tensor: the QKV product lives only in `cb_partials`, DEST and
    `cb_heads_out`.
  - q/k/v are interleaved in `memory_config`'s buffer type and written into `output_tensors` when given (the golden
    harness checks the buffer address).
  - Padding rows are zeroed by **selection**: the writer zeroes rows `[valid_last_rows, 32)` of the CB page over the
    NoC before the page write. The poisoned-padding regression tests (inf, NaN, 3e37) pass.
  - q and k share `head_dim_order()` (identity), and v keeps the reference order.
- **Prompt soft guidance (prefer kernel-lib helpers):** followed.
  - Compute uses `compute_kernel_lib::matmul_block` (packer L1 accumulation, `LastBlockTarget::Out`, TileRowMajor).
  - Both multicasts use `McastArgs` / `ttnn.Mcast` + `attach()`.
  - The one raw-LLK piece is the fused 4-in/2-out SFPU `rotate_head_pair`. It is justified in the kernel header: no
    helper computes a 4-in/2-out map inside a matmul DEST window, and half-sync fp32 DEST has no fifth slot for the
    two-operand SFPU binaries.
  - The helper gap recorded in op_design.md was closed in kernel_lib. `is_custom_pack_activation` in
    `sfpu_activation_helpers.hpp` lets `matmul_block`'s Activation slot take a custom packer-thread functor.
- **Correctness hygiene:**
  - CB push counts equal wait counts on every CB, including ragged last blocks (nominal counts, garbage rows
    computed and never written).
  - `TensorAccessor` everywhere (no `InterleavedAddrGen`).
  - `void kernel_main()`.
  - `api/...` include paths.
- **Broadcast efficiency:**
  - The bias is K tile 24 (a ones column in x times the bias row of W), so there is no bias CB or broadcast phase.
  - One C/S pair per row serves all of the row's heads (copied into DEST per head, which the half-sync DEST
    mechanism forces).

### Retained on purpose: measurement scaffolding

`nomic_qkv_rotary_heads_program_descriptor.py::_env()` reads `QKV_<NAME>` environment overrides (`ABLATE_*` stubs,
`QK_SUBBLOCK_W`, `ROW_MAJOR`, `ROTARY_ON_PACK`). Every default is the production value, and the kernel stubs compile
out at 0. I kept them because the perf refinements below rely on exactly these ablations, and every region number in
`op_requirements.md` was measured with them. Recommendation: delete them, or gate them behind one explicit switch,
once the perf queue is done. They change results when set.

## Registry Conformance

- **`INPUT_TAGGERS`** = `{"seq_alignment": tag_seq_alignment}`, signature `(inputs, axes)`.
- **`SUPPORTED`** declares every gated axis: dtype, fp32_dest_acc_en, layout, placement, output_buffer, output_mode,
  seq_alignment.
- **`EXCLUSIONS`** = `[]`.
- **`validate()`**:
  - Shape gate (M range, needed before the norm layout lookup).
  - SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`).
  - Then the grid, L1-budget, prepared-object and output checks.
  - It is the first line of the entry point.
- The op file does **not** declare `INVALID`.
- No drift fixes were needed: `xpass_drift = 0` and `xfail_wrong_mode = 0`. The `model_layout` cells are refused by
  `validate()` (placement `"other"`) with `NotImplementedError`.
- **INVALID audit** (`eval/golden_tests/nomic_qkv_rotary_heads/feature_spec.py`): `INVALID = []`, which is correct.
  - TARGET has one dtype (bf16) and one layout (TILE), so the canonical bf8b+ROW_MAJOR entry does not apply.
  - The op is not norm-like (no optional weight axes), so no canonicalization cells are needed.
  - No cross-tensor coupling exists.

### TARGET − SUPPORTED accounting

| Axis | Missing value | Disposition |
|------|---------------|-------------|
| placement | `model_layout` | **Refinement 1** (56 `xfail_expected` cells) |
| placement | `op_layout` | Already in SUPPORTED. `input_memory_config()` returns the norm's layout, so the harness skips these 56 cells as duplicates of `ln_layout` (`supported_skipped`) |
| all other axes | — | TARGET == SUPPORTED |

No `INVALID_FOR_ARCH` values. Every `xfail_expected` cell maps to Refinement 1.

## Design Conformance

| Dimension | Status |
|-----------|--------|
| Algorithm | Matches. fp32 K accumulation with packer L1 accumulation across 5 K blocks. Bias as K tile 24, added in fp32 DEST. q scale folded into the weights. Rotary on the finished fp32 tile, one bf16 rounding. |
| Pipeline topology / RISC ownership | Matches. RISCV_0/NOC0 does the x gather + column multicast + table loads. RISCV_1/NOC1 does the W stream + row multicast + resident re-push + head scatter. Compute does one `matmul_block` per M block. |
| Work distribution | Matches. 81–99 cores (`num_columns` x 9), M over columns, 8-tile lanes over rows. Grid row 9 is idle, by design (`strip_row_10` was rejected by measurement in PIROTSKI_MATMUL.md §8.1). |
| Inter-core communication | Matches, through the multicast helper. in0: 11 fixed column senders. in1: 9 fixed row senders. |
| Rotary placement | Matches the design: packer-thread SFPU through the new custom-functor slot. The math-thread fallback is kept behind `ROTARY_ON_PACK=0`; it is measured 13 µs slower at 8x512. |
| Blocking-model fidelity | Knobs are host constants threaded as CT/RT args (`BLOCK_M_TILES_MAX`, `K_BLOCK_TILES`, `LANE_N_TILES`, `X_BLOCK_DEPTH`, `ROTARY_TABLE_DEPTH`, `OUT_DEPTH_BLOCKS`, `WEIGHT_READ_AHEAD_BLOCKS`), after the two DRY fixes above. No CB scales with an op dimension. `cb_weights = K_TILES * LANE_N_TILES` is the design's resident ring, pinned by mechanism. |
| Block expression | Compute: one block call per M block. Reader: one read barrier per (M block, K block), one table barrier per M block. W stream: one DRAM batch per K block with one block of read-ahead. Writer: one wait/flush/pop per **tile row** (8 tiles). That is the design's drain granularity, chosen so the drain overlaps compute's per-row pushes under TileRowMajor. It is not a per-tile handshake, and I did not flag it. |
| Mechanism deviations (not flagged, per scope boundary) | Ones tile and padding zeroing use NoC `async_write_zeros` instead of `fill_l1_range`. Output layout is `TileRowMajor` instead of `SubblockMajor`; the implementer measured it faster, and the tile order is the same because `out_subblock_h = 1`. |

The design is mapped faithfully. The op is **compute-bound on the q/k rows** at the focus shape and
**data-movement-bound at small M** (measurements below), which is where the perf queue points.

## L1 Ledger Audit

- **Ledger currency**: five rows, five CBs. Index, page size, format and capacity expression match the descriptor.
  Two stale wordings were fixed (above).
- **Capacity vs live set**:
  - `cb_heads_out` is **flagged**. Its capacity is one whole M block (24 tiles, 48 KiB at `block_m_tiles = 3`),
    justified as "compute never waits on the writer". Under TileRowMajor, compute pushes one tile row at a time. The
    writer is already parked on block m's first row before compute reaches block m's last K block, because it only
    needs block m's K block 0 popped to re-push the resident K blocks 0..3 of block m+1.
  - So the steady-state live set is about 1–2 rows. A row-window capacity frees about 16 KiB a core. On its own that
    does not admit `block_m_tiles = 4` (672 KiB against the 640 KiB budget).
  - **Folded into Refinement 4** (block-size co-tune), which must measure it. No other over- or under-sized CB:
    `cb_x_block` double-buffers one (M, K) block; `cb_weights` is the resident ring; `cb_partials` spans the whole
    accumulator block; `cb_rotary_table` spans the block's rows.
- **Page format vs DEST**: `fp32_dest_acc_en` is required.
  - `cb_partials` is fp32 (packer L1 accumulation).
  - `cb_rotary_table` is fp32: a dataflow payload copied into fp32 DEST, kept for C/S precision.
  - The rest are bf16 (I/O ABI).
  - No finding.
- **Disjoint lifetimes without justification**: none.
- **Bounds / closed form**: total `= 409,600 + 77,824 * block_m_tiles` B (Phase 0 depths): 552 KiB at
  `block_m_tiles = 2`, 628 KiB at 3. The constant 400 KiB term is the resident weights (`K_TILES * LANE_N_TILES`).
  Every other term scales with `block_m_tiles`.
- **Block-size defaults**: the split's work units (M rows x lanes) fill 81–99 of 110 cores, and `block_m_tiles` is
  the coarsest block that fits (capped at 3 by L1, as the ledger table shows). The sharded-input default does not
  apply: x is gathered, not consumed in place.
- **Data-movement budget**: present and consistent with the code.
  - W crosses DRAM once (M block 0 only; resident afterwards).
  - x is gathered once per column and multicast.
  - Tables are read once per q/k core per M block.
  - Every q/k/v tile is written once.
- **Cheapest-traffic split**: `holder_direct_x_mcast` and `table_broadcast` are `deferred` with a positive reason:
  off the critical path. This is now **measured**: stubbing the whole x gather moves 8x512 by +0.1 µs (noise).

## Precision Baseline

`tests/ttnn/unit_tests/operations/nomic_qkv_rotary_heads/test_nomic_qkv_rotary_heads_precision_baseline.py`:
- Inputs: ln_layout x, q/k/v preallocated in L1, float64 reference.
- ULP is the bf16 ordinal distance to the reference rounded to bf16, over elements with |ref| > 0.1σ (near-zero
  elements make ULP meaningless).
- Ratio = got/true over the same elements.

**Default config (HiFi2, fp32 DEST acc)**: identical within noise at every shape, so one row per output and shape:

| Shape (M) | Out | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP mean / max | got/true [p5, median, p95] |
|-----------|-----|-----|-------------|--------------|------------------|----------------|----------------------------|
| 2x512 (32) | q | 0.9999942 | 3.89e-3 | 4.61e-4 | 0.00476 | 1.10 / 26 | [0.9849, 0.99666, 1.0084] |
| 2x512 (32) | k | 0.9999942 | 3.22e-2 | 3.68e-3 | 0.00477 | 1.10 / 22 | [0.9849, 0.99666, 1.0085] |
| 2x512 (32) | v | 0.9999942 | 3.19e-2 | 3.53e-3 | 0.00456 | 1.07 / 24 | [0.9852, 0.99694, 1.0087] |
| 3x400 (39, padded) | q / k / v | 0.9999942 | 3.9e-3 / 3.4e-2 / 3.1e-2 | 4.6e-4 / 3.7e-3 / 3.5e-3 | 0.00477 / 0.00476 / 0.00457 | 1.07–1.10 / ≤ 26 | median 0.99665 / 0.99666 / 0.99694 |
| 4x512 (64) | q / k / v | 0.9999942 | 4.1e-3 / 3.2e-2 / 3.1e-2 | 4.6e-4 / 3.7e-3 / 3.5e-3 | 0.00477 / 0.00477 / 0.00457 | 1.07–1.10 / ≤ 27 | median 0.99665 / 0.99665 / 0.99693 |
| 8x512 (128) | q | 0.9999942 | 4.55e-3 | 4.61e-4 | 0.00477 | 1.11 / 26 | [0.9849, 0.99665, 1.0084] |
| 8x512 (128) | k | 0.9999942 | 3.25e-2 | 3.69e-3 | 0.00477 | 1.11 / 30 | [0.9849, 0.99665, 1.0084] |
| 8x512 (128) | v | 0.9999942 | 3.19e-2 | 3.53e-3 | 0.00457 | 1.08 / 27 | [0.9851, 0.99693, 1.0087] |

(q's absolute errors are 8x smaller because q carries the 1/8 scale.)

**Scale-vs-precision triage.** The got/true median sits at 0.9967 rather than 1.0, which is the detector's signature,
so I ruled out a scale bug before accepting the numbers:
- **v shows the same offset** (0.9969). v has no rotary, so the rotary path is not the cause.
- **The spread is broad**: p5..p95 is about 0.985..1.008. That is not a tight cluster at a constant.
- **HiFi4 removes it.** Same op, `math_fidelity=HiFi4` (diagnostic parametrization in the same test): median 0.99946
  for q/k and 0.99975 for v; relative RMS 0.00180 for q/k and 0.00173 for v; ULP mean 0.15; PCC 0.9999985.

The offset is therefore HiFi2's srcB mantissa truncation of x (toward zero), which shrinks every product's
magnitude. It is the model's chosen fidelity, and the incumbent chain has it too. **No scale/structural bug, nothing
to route.** The test gates the median at |1 − median| ≤ 5e-3 for HiFi2 and ≤ 1e-3 for HiFi4.

**Assessment**:
- The op meets the golden gate (PCC ≥ 0.99998, relative RMS ≤ 0.0050) and beats the incumbent chain (q/k 0.0049,
  v 0.0045 → 0.00477 and 0.00457).
- Headroom is thin: q/k relative RMS is 0.00477 against 0.0050 (4.6%).
- Error is flat across shapes and padding, so it is set by the HiFi2 matmul plus the single bf16 rounding.
- The rotary adds ≈ 0.0002 relative RMS over v (tf32 truncation of the reloaded partial and of C/S through srcA).

**Recommended tolerances**:
- Keep PCC ≥ 0.99998 and relative RMS ≤ 0.0050.
- For allclose-style checks: atol = 0.05 (k/v max abs 3.3e-2; q 4.6e-3) and rtol = 0.02.
- **Any perf change that adds a bf16 intermediate or truncates C/S below fp32 will likely breach the RMS gate.**

## Performance baseline (device kernel ns, ln_layout, q/k/v preallocated L1, HiFi2, 1350 MHz)

`test_nomic_qkv_rotary_heads_perf.py` (in-process device profiler). Ablations use the `QKV_*` overrides; Δ is the
change against the production build.

| Build | 2x512 (M=32) | 4x512 (M=64) | 8x512 (M=128) |
|-------|--------------|--------------|---------------|
| **production** | **28,960** | **50,259** | **93,070** |
| accept / stretch (feature_spec) | 26,000 / 21,000 | 45,000 / 40,000 | 85,000 / 76,000 |
| no DRAM weight reads (`ABLATE_WREAD`) | 25,577 (−3.4 µs) | 46,907 (−3.4) | 89,772 (−3.3) |
| no rotary: C/S copy + SFPU (`ABLATE_ROTARY=1`) | 27,322 (−1.6) | 45,340 (−4.9) | 81,031 (−12.0) |
| no rotary SFPU, copy kept (`ABLATE_ROTARY=2`) | 27,502 (−1.5) | 45,698 (−4.6) | 83,693 (−9.4) |
| no q/k/v writes (`ABLATE_WRITES`) | 27,775 (−1.2) | 49,059 (−1.2) | 92,021 (−1.0) |
| no x gather (`ABLATE_GATHER`) | 28,821 (0) | 50,378 (0) | 93,184 (0) |
| rotary off + q/k 1x4 subblocks | 26,989 | 44,170 | 79,647 (−1.4 vs rotary off) |
| compute stubbed (data-movement floor) | 23,811 | 30,597 | 46,913 |
| rotary on math thread (`ROTARY_ON_PACK=0`) | 31,931 | 56,591 | 106,061 |

**All three focus shapes are above their acceptance ceilings** (by +11%, +12% and +9.5%).
- **8x512 is compute-bound.** The data-movement floor is 46.9 µs. The q/k rotary costs 12.0 µs (9.4 µs SFPU, 2.7 µs
  C/S copies), so the packer-thread SFPU is not hidden behind the next head's FMAs.
- **The matmul itself is below its ceiling.** With rotary off and 1x4 subblocks it runs at 79.6 µs. The HiFi2 ideal
  on 99 cores is 2400 tile products x 32 cycles ≈ 56.9 µs, so the matmul is at about 71%.
- **2x512 is data-movement-bound.** The floor is 23.8 µs against the 26 µs ceiling, the rotary only 1.6 µs. The
  M-block-0 weight stream (DRAM reads, plus 5 row multicasts of 80 KB per lane) dominates.

## Verifier CLI Summary

Results dir `/tmp/qkv_verifier_golden2`; the report is copied to
`ttnn/ttnn/operations/nomic_qkv_rotary_heads/verifier_report.json`.

- supported_pass: 68 (59 `test_golden.py` incl. the 3 LOOSE_CASES, 9 `test_regression.py`; all `ln_layout`)
- xfail_expected: 56 (all `placement=model_layout` → Refinement 1)
- supported_skipped: 56 (all `placement=op_layout`, skipped by the harness as identical to `ln_layout`)
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

Acceptance: `test_nomic_qkv_rotary_heads.py` 35/35. Precision baseline: 8/8.

## Recommendations

- **Queue** (`op_requirements.md`):
  - R1 adds `model_layout` (the only generality gap; host-side x source table for BLOCK_SHARDED COL_MAJOR).
  - R2–R4 are measured perf refinements on the flagged 8x512 profile:
    - R2: q/k rotary and last-K-block path, −12 µs available.
    - R3: M-block-0 weight stream and multicast fill, which also binds the 2x512 / 4x512 guards.
    - R4: K-loop efficiency and block co-tune toward stretch, with the `cb_heads_out` window folded in.
- **Precision headroom is 4.6% on q/k.** Perf work must keep one bf16 rounding, and must keep C/S and the reloaded
  partial at their current width.
- **L1 margin is 12 KiB a core at `block_m_tiles = 3`** (628 of 640 KiB). Any CB added by a perf refinement must come
  out of an existing one, for example the `cb_heads_out` window.
- **q/k vs v imbalance**: v cores (grid rows 6–8) carry no rotary. By the rotary ablation (inferred, not zoned) they
  finish up to about 12 µs earlier at 8x512. A lane
  table that moves work off the q/k rows is a knob-turn. It is bounded by the resident-weight L1: a 12-tile v lane
  needs 600 KiB of weights, so it does not fit as-is. Noted for the perf implementer; not a refinement by itself.
- **Delete or gate the `QKV_*` env scaffolding** after the perf queue.
