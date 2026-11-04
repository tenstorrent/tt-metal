# Verification Report: nomic_residual_layer_norm

Verifier pass, 2026-11-03, Blackhole p300c (11 x 10 compute grid, AICLK 1350 MHz per the profiler's `CHIP_FREQ`).
Golden suite results: `/tmp/nrln_golden2/` (a copy of `verifier_report.json` is in this directory).

**Bottom line**:
- Correct and registry-honest. Every loud verifier category is empty, and precision sits at the bf16 output-rounding
  floor.
- **The op misses its performance goal.** It is 1.45–1.60× faster than the incumbent; the required speedup is
  1.85×. Every focus shape is over its acceptance ceiling (table below). The perf refinements in
  `op_requirements.md` are the work that finishes this op; they are not polish.

| Perf case (LOOSE_CASES) | Measured device kernel | Incumbent | Acceptance ceiling (÷1.85) | Stretch | Speedup | Status |
|---|---|---|---|---|---|---|
| **8x512 residual new** (flagged) | **19.04 us** | 29.7 us | 16.1 us | 12.0 us | 1.56× | **misses by 2.9 us** |
| 4x512 residual new | 11.71 us | 18.7 us | 10.1 us | 7.0 us | 1.60× | misses by 1.6 us |
| 2x512 residual new | 8.26 us | 12.0 us | 6.5 us | 4.5 us | 1.45× | misses by 1.8 us |
| 8x512 no_residual new | 17.61 us | 27.2 us | 14.7 us | — | 1.54× | misses by 2.9 us |

## Code Review

### Fixed in place

1. **Half-exposed row-block knob (blocking-model fidelity).** `NUM_ROW_BLOCKS` was a global count guarded by
   `assert rows_per_shard % NUM_ROW_BLOCKS == 0`. Turning it to 2 (perf lamp L2) would crash every shape with 1 or 3
   rows a core. I replaced it with `MAX_BLOCK_ROWS`: `_plan()` takes `block_rows` = the largest divisor of
   `rows_per_shard` that is ≤ `MAX_BLOCK_ROWS`, and `num_row_blocks = rows_per_shard / block_rows`.
   - The default (`MAX_ROWS_PER_SHARD`) keeps one block = the whole shard, so behaviour is identical.
   - The knob can now be turned without breaking any shape. Verified with `MAX_BLOCK_ROWS = 2`: acceptance 20/20.
2. **DRY: the compute kernel restated host constants as literals.** It declared all 12 CB indices and the packed-stats
   slot count (`kStatSlots = 4`) as its own `constexpr` literals. The host has `CB_*` / `STAT_SLOTS`, and the reader
   and writer already received their CB ids as CT args.
   - The compute kernel now reads all of them from CT args 8–20 at namespace scope.
   - `stat_slots` is a `Plan` field, so the host's `stat_tiles` derivation and the kernel's slot packing come from one
     value.
   - The kernel `static_assert`s the mechanism cap (≤ 4 slots: columns {0, 1, 16, 17}).
3. **Operand-contract violations raised `ValueError`.** Affected: residual shape, dtype or memory config mismatch;
   gamma/beta dtype or layout; sharded gamma/beta. The prompt's Validation section requires `validate()` to raise
   `SupportRefusal` "unless all of these hold", so the model can fall back to its old path. All of them now raise
   `UnsupportedAxisValue`.

### Reviewed, no change needed

- **CB sync.** Push and wait counts balance for every CB:
  - x / residual: T credits per shard.
  - `cb_y`: closed ring of T pages, popped and pushed per DEST window.
  - Stats: 2P local, 2P·G gather, 2 final.
  - gamma/beta: pushed once, popped once at the end.
  - output: T.
- **`cb_stats_local` | `cb_stats_final` alias.** The writer pushes `cb_stats_gather` before popping `cb_stats_local`,
  and compute then writes `cb_stats_final` into the same memory. This is safe:
  - `ChainSenderPipeImpl::send` ends with a write barrier (local copy) or a `SourceL1Guard::Guard` trid flush.
  - `receive_and_forward` flushes under Guard.
  - So the source is never read after the gather credit is published.
- **Kernel hygiene.** `void kernel_main()`, `api/...` include paths and `TensorAccessor` are all correct.
- **Broadcasts.** The normalize stages use col-bcast for mean/rstd and row-bcast for gamma/beta. No redundant fills:
  gamma/beta fetch only the two 64-B face rows that carry row 0.
- **Helper usage.**
  - Element-wise phases (residual add, pairwise row sums, sum of squares, 4 normalize stages) use `eltwise_chain`.
  - The exchange uses `ttnn.Mcast` / `McastArgs`.
  - The raw-LLK parts (collapse+pack, finalize, extract) are justified in the kernel header and in op_design's "Helpers
    considered and rejected": no DEST→DEST shifted vector copy exists, and the chain elements have no vector-mode
    scoping. These are mechanism choices the implementer owns, and they produce no observable serialization.
- **`concurrent_unicast` uses raw NoC semaphore calls.** This is a non-default `EXCHANGE_MODE` alternative, kept as a
  measured knob (perf lamp L5). The Mcast helper has no handshake-free unicast form. Left as is; the default
  (`serialized_chain`) uses the helper.
- **Design deviation, accepted.**
  - The design asked for one rotating-sender channel with ChainUnicast. The Mcast helper refuses that combination: a
    rotating sender needs rectangular groups, and chain unicast needs a fixed sender. Row-major triples on an 11-wide
    grid wrap rows.
  - The implementer built G fixed-sender channels driven in member order instead. This reproduces the design's
    schedule: one rendezvous, G rounds, member order.
  - This is documented in the writer header and in a friction breadcrumb.
- **NoC assignment.** The reader is on NOC_1 and the writer on NOC_0, the reverse of the catalog default. This is
  harmless here: no activation crosses the NoC, and the gamma/beta reads are fully shadowed (measured below).

### Prompt rules (`eval/prompts/nomic_residual_layer_norm.txt` § Rules): all MUSTs hold

| Rule | Status |
|---|---|
| One native device-program dispatch per call; no ttnn ops on the path | ✓ a single `ttnn.generic_op` |
| Kernel file names / io order `[input, residual, gamma, beta, output]` | ✓ |
| x, residual, output stay in L1; no to_layout / tilize / pad / to_memory_config | ✓ zero-copy `cb_descriptor_from_sharded_tensor` |
| 1/768 never a bf16 constant; gamma not pre-scaled | ✓ `fl(1/768)` fp32 bits as a CT arg, multiplied on the SFPU in fp32; gamma fp32 as given |
| S % 32 ≠ 0 / uneven M: valid rows match; nothing written outside the op's tensors | ✓ padding rows stay in shard padding (golden `seq_padded` cells, 3x400 / 11x352 pass) |
| *prefer* fewer cross-core round trips (soft) | ✓ one exchange per block |
| *prefer* kernel-library helpers (soft) | ✓ see above; the raw-LLK parts are justified |
| CBs ≤ 256 KB a core at M = 128 | ✓ 240 KB (`circular_buffer_bytes`) |

### Design conformance

- **Algorithm.** Matches op_design: single fp32 `cb_y`, E[y²] − mean² clamped at 0, `fl(1/768)` in fp32, and a fixed
  contributor order.
- **Pipeline topology.** Matches, apart from the exchange-channel deviation above.
- **Work distribution.** Matches: 96 / 96 / 96 cores at the three focus shapes, up to 108 cores.
- **Grid fill.** The independent `rows` axis is spread over row groups, and the dependent `hidden` axis is cut 3
  ways with the planned combine. The machine is full.
- **Blocking model.** Knobs are parameters with a single source: `HIDDEN_SPLIT`, `MAX_BLOCK_ROWS`,
  `ELTWISE_BLOCK_TILES`, `STAT_SLOTS` and `EXCHANGE_MODE`. Every CB is sized from `Plan` (`block_rows`,
  `hidden_tiles`, `stat_tiles`, `hidden_split`); no CB is sized from a whole-op dimension.
- **Schedule expression.** Correct:
  - The reader pushes whole-shard credits once.
  - The writer waits once per block.
  - Compute stages run per block, in DEST windows of `ELTWISE_BLOCK_TILES` (fp32 half-sync DEST cap). The windows
    are not per-tile handshakes.

## Registry Conformance

- `INPUT_TAGGERS` (`tag_seq_alignment(inputs, axes)`), `SUPPORTED`, `EXCLUSIONS = []` and `validate()` are present
  and wired.
- `validate()` checks SUPPORTED per axis first, then EXCLUSIONS, then the shape gate (`_plan`), then the operand
  contract. It is the first statement of the entry point.
- The op file does **not** declare `INVALID`.
- No SUPPORTED auto-fixes were needed: no xpass drift.
- **INVALID audit** (`feature_spec.py`): `INVALID = []` is correct.
  - TARGET has one dtype (bf16) and one layout (TILE), so the canonical bf8b+ROW_MAJOR entry does not apply.
  - gamma/beta are mandatory, so there is no no-weight canonicalization.
  - Every TARGET combination is constructible.

## L1 Ledger Audit

The full audit is appended to `l1_ledger.md` § Verifier audit.

- **Currency.** All 10 declared CBs (3 io-backed, 7 allocated, with 2 alias pairs) have rows. Sizes match
  `cb_bytes_per_core()` and the descriptor. The symbol table now describes the `MAX_BLOCK_ROWS` knob.
- **Capacity vs live set.**
  - No unjustified buffers.
  - `cb_gamma` / `cb_beta` carry 1.5 KB of data in 48 KB of tile slots. This is required by the row-bcast unpacker's
    page-size indexing; accepted.
  - `cb_stats_gather` full-tile payload (1/8 useful lanes, lamp L4) is folded into **Refinement 1**: at G = 8 it is
    the largest scratch term.
- **Page format vs DEST.** No finding. fp32 DEST, every intermediate Float32; 16-bit pages are io or a dataflow-only
  payload.
- **Disjoint lifetimes.** Both aliases are justified. The `cb_stats_local` alias is race-free (see Code Review).
- **Bounds.** Closed form in `block_rows ≤ 4`, `hidden_tiles = 24/G`, `stat_tiles = ceil(block_rows/4)` and `G`, each
  with its establishing predicate.
- **Data-movement budget.** Present and consistent with the code: activations make 0 NoC/DRAM crossings; gamma/beta
  rows are read per core; statistics take 16 KB of link traffic a core.
  - The cheaper-traffic splits (`whole_rows`, `affine_mcast`) are `deferred` with positive reasons.
  - **The `affine_mcast` deferral reason was verified by measurement.** An ablation removed every gamma/beta read.
    NCRISC dropped from 7.1 to 0.4 µs, but device kernel time was unchanged (19.07 vs 19.04 µs at 8x512; 8.28 vs
    8.28 µs at 2x512). The reads are fully shadowed, so the op is **compute-bound**.
- **Block-size defaults.** Sharded → one block = the whole resident shard: held.
  - Measured with `MAX_BLOCK_ROWS = 2`: 22.5 µs at 8x512 (+3.5 µs), flat at M = 64 / 32. The compute loop runs block
    b's normalize before block b+1's phase A, so the second block buys a second exchange round and no overlap.
  - The default stays. The overlap needs a software-pipelined schedule (Refinement 3).
- **Per-core footprint**, beyond io:
  `4096·block_rows·hidden_tiles + 6144·hidden_tiles + 8192·block_rows + 8192·stat_tiles·(1 + G)`.
  This is 120 / 160 / 240 KB at M = 32 / 64 / 128.
  - `cb_y` scales with rows × hidden.
  - gamma/beta scale with hidden.
  - The stats terms scale with `stat_tiles × G`.

## Precision Baseline

`tests/ttnn/unit_tests/operations/nomic_residual_layer_norm/test_nomic_residual_layer_norm_precision_baseline.py`.
Float64 reference, seed 1234, gamma TF32-rounded fp32, beta bf16. ULP and ratio are taken over elements with
|ref| > 0.1·std.

| Shape (rows a core) | Residual | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP (bf16) | ≤1 ULP | got/true ratio p5 / median / p95 |
|---|---|---|---|---|---|---|---|---|
| 2x512 (1) | yes | 0.9999984 | 0.0299 | 0.00148 | 0.00224 | 4 | 99.97% | 0.9957 / 0.99867 / 1.0016 |
| 4x512 (2) | yes | 0.9999984 | 0.0301 | 0.00147 | 0.00224 | 2 | 99.98% | 0.9957 / 0.99867 / 1.0016 |
| 5x512 (3, partial shard) | yes | 0.9999984 | 0.0296 | 0.00148 | 0.00225 | 3 | 99.97% | 0.9957 / 0.99867 / 1.0016 |
| 8x512 (4) | yes | 0.9999984 | 0.0272 | 0.00147 | 0.00224 | 3 | 99.98% | 0.9957 / 0.99867 / 1.0016 |
| 11x352 (4, partial, wrapping groups) | yes | 0.9999984 | 0.0301 | 0.00148 | 0.00224 | 3 | 99.98% | 0.9957 / 0.99867 / 1.0016 |
| 2x512 | no | 0.9999984 | 0.0289 | 0.00154 | 0.00229 | 3 | 99.97% | 0.9956 / 0.99861 / 1.0016 |
| 4x512 | no | 0.9999984 | 0.0271 | 0.00150 | 0.00228 | 4 | 99.96% | 0.9956 / 0.99862 / 1.0016 |
| 5x512 | no | 0.9999984 | 0.0286 | 0.00150 | 0.00228 | 3 | 99.97% | 0.9956 / 0.99861 / 1.0016 |
| 8x512 | no | 0.9999984 | 0.0315 | 0.00150 | 0.00228 | 3 | 99.97% | 0.9956 / 0.99862 / 1.0016 |
| 11x352 | no | 0.9999984 | 0.0333 | 0.00151 | 0.00228 | 3 | 99.96% | 0.9956 / 0.99862 / 1.0016 |

**Assessment.** At the bf16 output-rounding floor:
- The golden helpers put output rounding alone at about 0.0023 relative RMS. The op measures 0.0022–0.0023.
- Max abs error ≈ 0.03 at |out| ≈ 4 is one bf16 ULP there.
- Results are identical across rows-per-shard 1–4 and partial shards. The regression tests (constant rows → beta;
  1σ / 3σ offset rows) pass.

**Scale-vs-precision triage.**
- The ratio median sits at 0.99867, a systematic shrink of about 0.13% of |out|.
- This is **not** the scale-bug signature. Relative RMS is 0.0022, not ≳ 0.1. The spread (p5–p95 ≈ 0.6%) is wider
  than the offset, and the offset is below half a bf16 ULP.
- It is design risk P3 as predicted: the 4 normalize stages each re-read fp32 `cb_y` through srcA with TF32
  truncation, about −2⁻¹¹ relative per read.
- No precision refinement is filed. Normalize-stage fusion (perf lamp L6, in Refinement 2) reduces the number of
  truncating re-reads, so it should pull the median toward 1. Refinement 2 records the ratio as a side metric.

**Recommended tolerances.** PCC ≥ 0.99999, relative RMS ≤ 0.005, max bf16 ULP ≤ 8 (on |ref| > 0.1·std),
|median ratio − 1| ≤ 2e-3. These match the golden suite plus the baseline test's gates.

## Verifier CLI Summary

`python3 -m eval.verify_supported /tmp/nrln_golden2 ttnn.operations.nomic_residual_layer_norm`, after the fixes. The
pre-fix run was identical.

- supported_pass: 46. That is 36 grid cells (18 shapes × 2 residual modes), 4 loose perf cases and 6 regression
  tests.
- xfail_expected: 108. 36 × (op_layout, in_place), 36 × (model_layout, in_place) and 36 × (model_layout, new). This is
  exactly TARGET − SUPPORTED, all of it queued in Refinement 1.
- invalid_skipped: 0 (INVALID is empty).
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

Acceptance test: 20 / 20 pass. Precision baseline: 10 / 10 pass.

## Perf measurements

Taken with `scripts/run_safe_pytest.sh --profile` on `test_nomic_residual_layer_norm_perf.py`, at
DEVICE KERNEL DURATION, 1350 MHz.

| Case | Kernel | NCRISC (reader) | TRISC1 (math) | BRISC (writer) |
|---|---|---|---|---|
| 8x512 residual | 19.04 us | 7.3 us | 18.70 us | 19.04 us |
| 4x512 residual | 11.71 us | 7.0 us | 11.42 us | 11.71 us |
| 2x512 residual | 8.26 us | 7.1 us | 7.86 us | 8.26 us |
| 8x512 no residual | 17.61 us | 7.1 us | 17.28 us | 17.61 us |

**What the numbers say.**
- **Compute-bound.** TRISC tracks kernel time, and the reader is fully shadowed (ablation above).
- A linear fit over tiles a core (8 / 16 / 32) gives **≈ 4.7 µs fixed** plus **≈ 0.45 µs (≈ 600 cycles) a tile**.
  The design's planning budget was ≈ 1 k cycles fixed + exchange + finalize ≈ 3.5k cycles (2.6 µs), and ≈ 400 cycles a
  tile.
- Both terms are over budget:
  - **The per-tile slope dominates at 8x512.** 32 tiles × 0.45 µs = 14.4 µs.
  - **The fixed intercept dominates at 2x512.** 4.7 of 8.3 µs.
- The residual add costs 1.4 µs at 8x512 (residual minus no-residual), about 60 cycles a tile, in line with the
  budget's 41.
- To meet every ceiling, both terms must come down, so the queue has two perf refinements (one per term).

## Recommendations

- **Perf is the priority debt.** The op is correct and registry-clean but fails its acceptance ceilings at every
  focus shape. Refinements 2 and 3 are the critical path to "done".
- **L2 needs a schedule change, not a knob turn.** Lamp L2 (row-block overlap) cannot be reached by turning
  `MAX_BLOCK_ROWS` alone (measured +3.5 µs). The compute kernel must issue phase A of block b+1 before finalize/normalize
  of block b, and `cb_stats_gather` must hold two rounds.
- **HIDDEN_SPLIT was never swept.** The prompt asks for the layout to be "chosen by measurement", but G = 3 comes from
  the planning model (lamp L1), and no sweep is recorded. The sweep is constrained:
  - `hidden_tiles` must be even for the pairwise row sum, so G = 8 needs the odd-width row sum from Refinement 1.
  - `MAX_ROWS_PER_SHARD = 4` and the 256 KB budget limit G = 2 at M = 128 (r = 3, T = 36 → `cb_y` alone is 144 KB).
  - It is listed as a lever in Refinement 2.
- **Mcast helper documentation gap** (implementer friction breadcrumb). Rotating sender ⇒ rectangular groups; chain
  unicast ⇒ fixed sender. Neither is documented in `mcast/README.md` / `host/mcast.hpp`. This matters again in
  Refinement 1: the model layout's column groups *are* rectangles, so a rotating sender becomes legal there.
- **Precision.** No lever is needed. The 0.13% TF32 shrink (P3) is inside the budget and shrinks naturally with
  normalize fusion.
